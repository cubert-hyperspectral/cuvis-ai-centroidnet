"""Synthetic scenes for the tests: instance maps of discs, pure tensors, no files."""

from __future__ import annotations

import torch


def disc_instances(
    height: int,
    width: int,
    discs: list[tuple[int, int, int]],
) -> torch.Tensor:
    """Instance map ``[H, W]`` (int32) of discs given as ``(centre_y, centre_x, radius)``.

    Disc ``i`` gets id ``i + 1``; a later disc overwrites an earlier one where they overlap,
    so overlapping discs touch along a shared boundary. A disc of radius ``r`` around an
    integer centre has an odd diameter and its centroid on that centre pixel.
    """
    instances = torch.zeros(height, width, dtype=torch.int32)
    ys = torch.arange(height).unsqueeze(1)
    xs = torch.arange(width).unsqueeze(0)
    for index, (centre_y, centre_x, radius) in enumerate(discs):
        inside = (ys - centre_y) ** 2 + (xs - centre_x) ** 2 <= radius**2
        instances[inside] = index + 1
    return instances
