"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from cuvis_ai_core.utils.node_registry import NodeRegistry
from scenes import disc_instances

MANIFEST = Path(__file__).resolve().parents[1] / "plugins.yaml"


@pytest.fixture()
def scene() -> torch.Tensor:
    """Three separate discs with integer centres in a 24 x 30 frame, none at the edge."""
    return disc_instances(24, 30, [(6, 7, 3), (15, 20, 4), (17, 6, 2)])


@pytest.fixture()
def registry() -> NodeRegistry:
    """A node registry that knows this plugin through the shipped manifest."""
    nodes = NodeRegistry()
    nodes.register_plugin(str(MANIFEST))
    return nodes
