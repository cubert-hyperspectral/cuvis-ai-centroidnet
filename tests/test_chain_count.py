"""Chain: heads, the vote nodes of cuvis-ai and VotePeakCount, with and without DynUNet.

Needs cuvis-ai with VectorFieldVoteAccumulator and LocalMaximaSuppression, and cuvis-ai-unet.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from cuvis_ai_centroidnet.configs import PIPELINE_DIR

pytestmark = pytest.mark.chain

MANIFEST = Path(__file__).resolve().parents[1] / "plugins.yaml"
COUNT_YAML = PIPELINE_DIR / "centroidnet_count.yaml"


def _registry():
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    registry = NodeRegistry()
    registry.register_plugin(str(MANIFEST))
    return registry


def _disc_features(height: int, width: int, centre: tuple[int, int], radius: int, scale: float):
    """Backbone output whose centre planes point every disc pixel at ``centre``.

    Written out by hand: plane 0 is the row offset ``centre_y - y``, plane 1 the column
    offset ``centre_x - x``, both divided by ``scale`` as the backbone would predict them.
    Returns the features and the number of disc pixels.
    """
    features = torch.zeros(1, height, width, 6)
    pixels = 0
    for y in range(height):
        for x in range(width):
            if (y - centre[0]) ** 2 + (x - centre[1]) ** 2 <= radius**2:
                features[0, y, x, 0] = (centre[0] - y) / scale
                features[0, y, x, 1] = (centre[1] - x) / scale
                pixels += 1
    return features, pixels


def test_votes_land_on_the_centre_in_row_and_column_order() -> None:
    """An off-diagonal centre in a non-square frame: a swapped axis would miss it."""
    from cuvis_ai.node.spatial import LocalMaximaSuppression
    from cuvis_ai.node.vote_accumulation import VectorFieldVoteAccumulator

    from cuvis_ai_centroidnet.node.count import VotePeakCount
    from cuvis_ai_centroidnet.node.heads import CentroidNetHeads

    height, width, centre = 9, 15, (2, 11)
    features, pixels = _disc_features(height, width, centre, radius=2, scale=8.0)

    vectors = CentroidNetHeads(max_distance=8.0).forward(features=features)["centroid_vectors"]
    votes = VectorFieldVoteAccumulator(order="yx").forward(vectors=vectors)["scores"]
    peaks = LocalMaximaSuppression(window=7).forward(scores=votes)["scores"]
    count = VotePeakCount(min_votes=2).forward(peaks=peaks)["count"]

    assert votes[0, centre[0], centre[1], 0].item() == pixels
    assert int(peaks[0, :, :, 0].argmax()) == centre[0] * width + centre[1]
    assert count.tolist() == [1]


def test_count_pipeline_runs_at_inference_with_only_a_cube() -> None:
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_schemas.enums import ExecutionStage

    pipeline = CuvisPipeline.load_pipeline(str(COUNT_YAML), node_registry=_registry())
    generator = torch.Generator().manual_seed(0)
    batch = {
        "cube": torch.rand(2, 32, 40, 61, generator=generator),
        "wavelengths": torch.arange(450, 450 + 61 * 8, 8, dtype=torch.int32).repeat(2, 1),
    }
    out = pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)

    count = out[("Count", "count")]
    assert count.shape == (2,)
    assert count.dtype == torch.int32
    assert out[("Heads", "centroid_vectors")].shape == (2, 32, 40, 2)


def test_count_pipeline_survives_save_and_reload(tmp_path: Path) -> None:
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_schemas.enums import ExecutionStage

    pipeline = CuvisPipeline.load_pipeline(str(COUNT_YAML), node_registry=_registry())
    generator = torch.Generator().manual_seed(1)
    batch = {
        "cube": torch.rand(1, 32, 40, 61, generator=generator),
        "wavelengths": torch.arange(450, 450 + 61 * 8, 8, dtype=torch.int32).repeat(1, 1),
    }
    before = pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)

    pipeline.save_to_file(str(tmp_path / "p.yaml"), save_weights=True)
    restored = CuvisPipeline.load_pipeline(
        str(tmp_path / "p.yaml"),
        weights_path=str(tmp_path / "p.pt"),
        node_registry=_registry(),
    )
    after = restored.forward(batch=batch, stage=ExecutionStage.INFERENCE)

    torch.testing.assert_close(
        after[("Heads", "centroid_vectors")],
        before[("Heads", "centroid_vectors")],
        rtol=0.0,
        atol=1e-6,
    )
    assert torch.equal(after[("Count", "count")], before[("Count", "count")])


@pytest.mark.parametrize("config", sorted(PIPELINE_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_agree_on_vote_range_and_axis_order(config: Path) -> None:
    """max_distance and the (y, x) order live in several nodes; they must not drift apart."""
    nodes = yaml.safe_load(config.read_text(encoding="utf-8"))["nodes"]
    by_class: dict[str, list[dict]] = {}
    for node in nodes:
        by_class.setdefault(node["class_name"].rsplit(".", 1)[1], []).append(node["hparams"])

    distances = {
        hparams["max_distance"]
        for name in ("CentroidNetHeads", "InstanceVectorFieldEncoder", "VectorFieldLoss")
        for hparams in by_class.get(name, [])
    }
    assert len(distances) == 1, f"max_distance differs between nodes: {sorted(distances)}"
    for hparams in by_class["VectorFieldVoteAccumulator"]:
        assert hparams["order"] == "yx"
