"""Encode an instance map into the vector targets of CentroidNetV2."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_centroidnet._geometry import instance_layout
from cuvis_ai_centroidnet._validation import positive_distance

_DEFAULT_MAX_PAIRWISE = 2**22


def _nearest_centroid(
    centre_y: Tensor, centre_x: Tensor, height: int, width: int, max_pairwise: int
) -> Tensor:
    """Return the index of the nearest centroid for every pixel, row-major ``[H * W]``.

    The centroids are compared a few at a time, so no ``[N, H * W]`` matrix is held at
    once. A pixel at the same distance from two centroids goes to the lower index.
    """
    device = centre_y.device
    pixels = height * width
    ys = torch.arange(height, dtype=torch.float32, device=device).repeat_interleave(width)
    xs = torch.arange(width, dtype=torch.float32, device=device).repeat(height)
    best = torch.zeros(pixels, dtype=torch.int64, device=device)
    best_distance = torch.full((pixels,), math.inf, dtype=torch.float32, device=device)

    step = max(1, max_pairwise // pixels)
    for start in range(0, centre_y.numel(), step):
        chunk_y = centre_y[start : start + step, None]
        chunk_x = centre_x[start : start + step, None]
        distance = (ys - chunk_y).square() + (xs - chunk_x).square()
        nearest = distance.min(dim=0).values
        # The lowest row among equal distances, spelled out so that it does not depend on
        # how a backend breaks ties in argmin.
        rows = torch.arange(distance.shape[0], device=device)[:, None]
        first = torch.where(distance == nearest, rows, distance.shape[0]).min(dim=0).values
        closer = nearest < best_distance
        best = torch.where(closer, first + start, best)
        best_distance = torch.where(closer, nearest, best_distance)
    return best


class InstanceVectorFieldEncoder(Node):
    """Turn an instance map into the training targets: two vector fields, mask, weights.

    This is the target encoding of CentroidNetV2 (Section 3.3.2 of the paper). For every
    pixel of a frame, background included:

    1. The nearest instance is the one whose centroid (the mean position of its pixels) is
       closest. At exactly the same distance the lower instance id wins.
    2. ``centroid_vectors`` holds the offset from the pixel to that centroid.
    3. ``border_vectors`` holds the offset from the pixel to the nearest border pixel of
       that instance. A border pixel is an instance pixel with a 4-neighbour that carries
       another id or lies outside the frame. Among border pixels at the same distance the
       first in row-major order is taken.
    4. Where the border vector is longer than ``max_distance``, both vectors are zero. A
       pixel far from every object therefore votes for itself, and so does the middle of
       an object whose radius exceeds ``max_distance``: choose ``max_distance`` larger than
       the largest object radius.

    Vectors are in pixels with the row offset in channel 0 and the column offset in
    channel 1, the order ``CentroidNetHeads`` produces. Ids only name the instances: they
    need not be consecutive, and 0 or a negative id is background.

    ``weights`` is 1 everywhere except on pixels that keep a vector (step 4) and whose
    nearest instance touches the frame edge, where it is 0. An object cut by the edge of a
    frame or of a training crop has its centroid in the wrong place, so those pixels are
    left out of ``VectorFieldLoss``. Pixels with a zero target keep weight 1. For this to
    work on crops, crop the cube and the instance map together and encode afterwards.

    ``object_mask`` is 1 on every instance pixel and 0 on background: the target of the
    class logits for the two classes of the paper (object and background), in the form a
    cross-entropy loss node takes.

    The node needs the labels, so it runs in training, validation and test only. It has no
    gradient, and it cannot be exported: the number of instances and of their border
    pixels depends on the data.

    Parameters
    ----------
    max_distance : float
        Vote range in pixels: the longest border vector that is kept. Must be positive and
        finite, and equal to the ``max_distance`` of ``CentroidNetHeads`` and
        ``VectorFieldLoss`` in the same pipeline.
    max_pairwise : int, optional
        Upper limit for the number of elements of one intermediate distance matrix
        (instances by pixels, and pixels by border pixels). Larger matrices are processed
        in chunks; a chunk is never smaller than one row, so one instance against all
        pixels of a frame always fits. The result does not depend on this value. Default
        ``2**22``.

    Raises
    ------
    ValueError
        If ``max_distance`` is not positive and finite, or ``max_pairwise`` is not positive.

    References
    ----------
    K. Dijkstra, J. van de Loosdrecht, W. A. Atsma, L. R. B. Schomaker, M. A. Wiering.
    "CentroidNetV2: A hybrid deep neural network for small-object segmentation and
    counting." Neurocomputing 423 (2021), pages 490 to 505, Section 3.3.2.
    https://doi.org/10.1016/j.neucom.2020.10.075

    The loss weights are not part of the paper. Its reference implementation encodes the
    full image and crops the targets afterwards.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {
            NodeTag.TORCH,
            NodeTag.PREPROCESSING,
            NodeTag.TRAINING,
            NodeTag.SEGMENTATION,
            NodeTag.DETECTION,
        }
    )

    # Needs the instance labels, so it does not run at inference.
    EXECUTION_STAGES = frozenset({ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST})

    INPUT_SPECS = {
        "instances": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Instance map [B, H, W]: one positive id per object, 0 = background",
        ),
    }
    OUTPUT_SPECS = {
        "centroid_vectors": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 2),
            description="Target vector to the nearest object centre in pixels [B, H, W, 2] (y, x)",
        ),
        "border_vectors": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 2),
            description="Target vector to the border of that object in pixels [B, H, W, 2] (y, x)",
        ),
        "weights": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Loss weight per pixel [B, H, W, 1]: 0 around objects cut by the edge",
        ),
        "object_mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Class target [B, H, W]: 1 on object pixels, 0 on background",
        ),
    }

    def __init__(
        self, max_distance: float, max_pairwise: int = _DEFAULT_MAX_PAIRWISE, **kwargs: Any
    ) -> None:
        self.max_distance = positive_distance(max_distance, type(self).__name__)
        if max_pairwise <= 0:
            raise ValueError(
                f"{type(self).__name__}: max_pairwise must be a positive number of elements; "
                f"got {max_pairwise!r}."
            )
        self.max_pairwise = int(max_pairwise)
        super().__init__(max_distance=self.max_distance, max_pairwise=self.max_pairwise, **kwargs)

    @torch.no_grad()
    def forward(self, instances: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the target vector fields, the loss weights and the object mask."""
        frames = [self._encode_frame(frame) for frame in instances]
        centre, border, weights = (torch.stack(parts) for parts in zip(*frames, strict=True))
        return {
            "centroid_vectors": centre,
            "border_vectors": border,
            "weights": weights,
            "object_mask": (instances > 0).to(torch.int32),
        }

    def _encode_frame(self, frame: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode one ``[H, W]`` instance map."""
        height, width = frame.shape
        device = frame.device
        centre = torch.zeros(height, width, 2, dtype=torch.float32, device=device)
        border = torch.zeros_like(centre)
        weights = torch.ones(height, width, 1, dtype=torch.float32, device=device)

        labels = frame.to(torch.int64)
        layout = instance_layout(labels)
        count = layout.ids.numel()
        if count == 0:
            return centre, border, weights
        centre_y, centre_x = layout.centroids.unbind(dim=1)

        nearest_instance = _nearest_centroid(
            centre_y, centre_x, height, width, self.max_pairwise
        ).reshape(height, width)

        # Border pixels: a 4-neighbour with another id. The zero padding makes the frame
        # edge count as outside.
        padded = F.pad(labels, (1, 1, 1, 1), value=0)
        differs = (
            (padded[:-2, 1:-1] != labels)
            | (padded[2:, 1:-1] != labels)
            | (padded[1:-1, :-2] != labels)
            | (padded[1:-1, 2:] != labels)
        )
        on_border = differs.reshape(-1)[layout.pixels]
        border_owner = layout.owner[on_border]
        # Grouped by instance; the stable sort keeps the row-major order inside a group.
        border_pixels = layout.pixels[on_border][torch.argsort(border_owner, stable=True)]
        group_ends = torch.cumsum(torch.bincount(border_owner, minlength=count), dim=0).tolist()

        frame_edge = torch.cat([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        touches_edge = torch.isin(layout.ids, frame_edge).tolist()

        reach = math.ceil(self.max_distance)
        group_start = 0
        for instance, group_end in enumerate(group_ends):
            group = border_pixels[group_start:group_end]
            group_start = group_end
            border_y, border_x = group // width, group % width

            # A pixel farther than the vote range from the bounding box is farther than
            # that from every border pixel, so only the grown box has to be searched.
            top = max(int(border_y.min()) - reach, 0)
            bottom = min(int(border_y.max()) + reach + 1, height)
            left = max(int(border_x.min()) - reach, 0)
            right = min(int(border_x.max()) + reach + 1, width)
            window = nearest_instance[top:bottom, left:right] == instance
            pixel_y, pixel_x = window.nonzero(as_tuple=True)
            pixel_y, pixel_x = pixel_y + top, pixel_x + left

            border_count = group.numel()
            position = torch.arange(border_count, device=device)
            step = max(1, self.max_pairwise // border_count)
            for start in range(0, pixel_y.numel(), step):
                y, x = pixel_y[start : start + step], pixel_x[start : start + step]
                squared = (border_y - y[:, None]).square() + (border_x - x[:, None]).square()
                # Exact integer distances. Adding the position makes the minimum unique:
                # the first border pixel in row-major order among the nearest ones.
                key = (squared * border_count + position).min(dim=1).values
                nearest = key % border_count
                in_range = (key // border_count).to(torch.float64).sqrt() <= self.max_distance
                y, x, nearest = y[in_range], x[in_range], nearest[in_range]

                centre[y, x, 0] = centre_y[instance] - y.to(torch.float32)
                centre[y, x, 1] = centre_x[instance] - x.to(torch.float32)
                border[y, x, 0] = (border_y[nearest] - y).to(torch.float32)
                border[y, x, 1] = (border_x[nearest] - x).to(torch.float32)
                if touches_edge[instance]:
                    weights[y, x, 0] = 0.0

        return centre, border, weights
