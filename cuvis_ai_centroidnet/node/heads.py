"""Split a backbone output into the three CentroidNetV2 outputs."""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_centroidnet._validation import positive_distance

_VECTOR_PLANES = 4


class CentroidNetHeads(Node):
    """Split a backbone output into centre vectors, border vectors and class logits.

    CentroidNetV2 asks one segmentation backbone for ``4 + K`` output planes per pixel
    (Eq. 1 and 2 of the paper): two planes with the vector to the nearest object centre,
    two planes with the vector to the border of that object, and ``K`` class logits. This
    node names those planes. Any backbone whose output has the size of its input can stand
    in front of it, for example ``DynUNet`` of cuvis-ai-unet with ``num_classes = 4 + K``.

    Plane order, as in the paper: centre y, centre x, border y, border x, then the logits.
    Both vectors are relative to the pixel and carry the row offset in channel 0 and the
    column offset in channel 1, so ``VectorFieldVoteAccumulator`` reads them with
    ``order="yx"``.

    The backbone predicts the vectors divided by ``max_distance`` (the vote range in
    pixels), which keeps its regression targets within ``[-1, 1]``; this node multiplies
    them back, so both vector outputs are in pixels. The logits pass through unchanged.
    No activation is applied to any plane: vectors have to be able to be negative.

    Parameters
    ----------
    max_distance : float
        Vote range in pixels, the longest vector the encoder keeps. Must be positive and
        finite, and equal to the ``max_distance`` of ``InstanceVectorFieldEncoder`` and
        ``VectorFieldLoss`` in the same pipeline.

    Raises
    ------
    ValueError
        If ``max_distance`` is not positive and finite, or if ``features`` has fewer than
        5 channels (four vector planes and at least one class plane).

    References
    ----------
    K. Dijkstra, J. van de Loosdrecht, W. A. Atsma, L. R. B. Schomaker, M. A. Wiering.
    "CentroidNetV2: A hybrid deep neural network for small-object segmentation and
    counting." Neurocomputing 423 (2021), pages 490 to 505, Section 3.1, Eq. 1 and 2.
    https://doi.org/10.1016/j.neucom.2020.10.075

    The paper states the output depth once as 6 and once as 5; ``4 + K`` with ``K = 2``
    (object and background) is what its reference implementation uses.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {
            NodeTag.ONNX,
            NodeTag.TORCH,
            NodeTag.DIFFERENTIABLE,
            NodeTag.SEGMENTATION,
            NodeTag.DETECTION,
        }
    )

    INPUT_SPECS = {
        "features": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Backbone output [B, H, W, 4 + K]: centre y, x; border y, x; K logits",
        ),
    }
    OUTPUT_SPECS = {
        "centroid_vectors": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 2),
            description="Vector to the nearest object centre in pixels [B, H, W, 2] as (y, x)",
        ),
        "border_vectors": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 2),
            description="Vector to the border of that object in pixels [B, H, W, 2] as (y, x)",
        ),
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Per-pixel class logits [B, H, W, K]",
        ),
    }

    def __init__(self, max_distance: float, **kwargs: Any) -> None:
        self.max_distance = positive_distance(max_distance, type(self).__name__)
        super().__init__(max_distance=self.max_distance, **kwargs)

    def forward(self, features: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the two vector fields in pixels and the class logits."""
        channels = features.shape[-1]
        if channels <= _VECTOR_PLANES:
            raise ValueError(
                f"{type(self).__name__}: features needs at least 5 channels "
                f"(4 vector planes and the class logits); got {channels}."
            )
        return {
            "centroid_vectors": features[..., 0:2] * self.max_distance,
            "border_vectors": features[..., 2:_VECTOR_PLANES] * self.max_distance,
            "logits": features[..., _VECTOR_PLANES:],
        }
