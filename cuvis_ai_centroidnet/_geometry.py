"""Instance geometry shared by the target encoder and the detection metric."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class InstanceLayout(NamedTuple):
    """The instances of one frame, in ascending id order."""

    ids: Tensor
    """Positive instance ids ``[N]``, ascending."""
    centroids: Tensor
    """Mean pixel position of every instance ``[N, 2]`` as (y, x), float32."""
    pixels: Tensor
    """Row-major flat position of every object pixel ``[M]``."""
    owner: Tensor
    """Position in ``ids`` of the instance each object pixel belongs to ``[M]``."""


def instance_layout(labels: Tensor) -> InstanceLayout:
    """Return ids, centroids and pixel membership of an int64 ``[H, W]`` instance map.

    An id of 0 or below is background. A frame without instances gives empty tensors.
    """
    width = labels.shape[1]
    flat = labels.reshape(-1)
    pixels = (flat > 0).nonzero().squeeze(1)
    ids = torch.unique(flat[pixels])
    owner = torch.searchsorted(ids, flat[pixels])

    # Sums in float64 stay exact for any frame size.
    count = ids.numel()
    sizes = torch.bincount(owner, minlength=count).to(torch.float64)
    totals = torch.zeros(2, count, dtype=torch.float64, device=labels.device)
    totals[0].index_add_(0, owner, (pixels // width).to(torch.float64))
    totals[1].index_add_(0, owner, (pixels % width).to(torch.float64))
    centroids = (totals / sizes).to(torch.float32).T
    return InstanceLayout(ids=ids, centroids=centroids, pixels=pixels, owner=owner)
