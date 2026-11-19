"""Derive an instance map from a class mask."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

_CONNECTIVITIES = (4, 8)


def _label_frame(mask: np.ndarray, connectivity: int) -> np.ndarray:
    """Label the connected components of every positive class id of one ``[H, W]`` mask.

    Ids run from 1 in row-major order of each component's first pixel, whatever its class.
    """
    labels = np.zeros(mask.shape, dtype=np.int32)
    found = 0
    for class_id in np.unique(mask[mask > 0]):
        count, components = cv2.connectedComponents(
            (mask == class_id).astype(np.uint8), connectivity=connectivity
        )
        labels += np.where(components > 0, components + found, 0).astype(np.int32)
        found += count - 1
    if found == 0:
        return labels

    ids, first_pixel = np.unique(labels.ravel(), return_index=True)
    is_object = ids > 0
    in_scan_order = ids[is_object][np.argsort(first_pixel[is_object], kind="stable")]
    renumber = np.zeros(found + 1, dtype=np.int32)
    renumber[in_scan_order] = np.arange(1, found + 1, dtype=np.int32)
    return renumber[labels]


class ConnectedComponentInstances(Node):
    """Turn a class mask into an instance map by connected components.

    Every connected region of one class id becomes one instance. Regions of different
    classes are separate instances even where they touch; two objects of the same class
    that touch become one instance, which is the limit of deriving instances from a class
    mask. Ids are 1 to N per frame, in row-major order of each instance's first pixel, and
    0 is background. A class id of 0 or below is background.

    Use this node when the annotations carry no object ids. Annotations with object ids
    should be turned into an instance map directly and fed to
    ``InstanceVectorFieldEncoder`` in its place.

    The labelling runs in OpenCV on the CPU, one frame at a time: the node has no gradient
    and cannot be exported. It is a label-preparation step, so it only runs where labels
    exist (train, validation and test).

    Parameters
    ----------
    connectivity : int, optional
        4 joins pixels that share an edge, 8 also joins diagonal neighbours (default 8).

    Raises
    ------
    ValueError
        If ``connectivity`` is not 4 or 8.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.PREPROCESSING, NodeTag.TRAINING, NodeTag.NUMPY})

    # Needs the label mask, so it does not run at inference.
    EXECUTION_STAGES = frozenset({ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST})

    INPUT_SPECS = {
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Class-id mask [B, H, W], 0 = background",
        ),
    }
    OUTPUT_SPECS = {
        "instances": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Instance map [B, H, W]: ids 1..N per frame, 0 = background",
        ),
    }

    def __init__(self, connectivity: int = 8, **kwargs: Any) -> None:
        if connectivity not in _CONNECTIVITIES:
            raise ValueError(
                f"{type(self).__name__}: connectivity must be 4 or 8; got {connectivity!r}."
            )
        self.connectivity = int(connectivity)
        super().__init__(connectivity=self.connectivity, **kwargs)

    @torch.no_grad()
    def forward(self, mask: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the instance map of every frame of ``mask``."""
        frames = mask.detach().cpu().numpy()
        labels = np.stack([_label_frame(frame, self.connectivity) for frame in frames])
        return {"instances": torch.from_numpy(labels).to(device=mask.device, dtype=torch.int32)}
