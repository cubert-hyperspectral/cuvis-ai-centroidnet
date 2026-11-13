"""The shipped manifest loads through NodeRegistry and resolves every node it lists."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

MANIFEST = Path(__file__).resolve().parents[1] / "plugins.yaml"


@pytest.mark.integration
def test_manifest_loads_and_registers_nodes() -> None:
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    listed = [
        entry["class_name"].rsplit(".", 1)[1]
        for entry in yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["capabilities"]
    ]
    assert listed, "the manifest lists no node"

    registry = NodeRegistry()
    registry.register_plugin(str(MANIFEST))
    assert "centroidnet" in registry.list_plugins()
    for class_name in listed:
        assert registry.get(class_name) is not None, f"{class_name} does not resolve"


@pytest.mark.integration
def test_manifest_lists_every_node_module() -> None:
    """A node module that is missing from the manifest cannot be used in a pipeline yaml."""
    node_dir = MANIFEST.parent / "cuvis_ai_centroidnet" / "node"
    modules = {path.stem for path in node_dir.glob("*.py") if path.stem != "__init__"}
    listed = {
        entry["class_name"].rsplit(".", 2)[1]
        for entry in yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["capabilities"]
    }
    assert listed == modules
