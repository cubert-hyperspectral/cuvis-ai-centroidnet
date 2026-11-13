"""Shared fixtures."""

from __future__ import annotations

import pytest
import torch
from scenes import disc_instances


@pytest.fixture()
def scene() -> torch.Tensor:
    """Three separate discs with integer centres in a 24 x 30 frame, none at the edge."""
    return disc_instances(24, 30, [(6, 7, 3), (15, 20, 4), (17, 6, 2)])
