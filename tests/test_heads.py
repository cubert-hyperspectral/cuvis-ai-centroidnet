"""CentroidNetHeads: plane order, scaling by the vote range, validation and export."""

from __future__ import annotations

import math

import pytest
import torch

from cuvis_ai_centroidnet.node.heads import CentroidNetHeads

pytestmark = pytest.mark.unit


def _features(batch: int = 2, height: int = 3, width: int = 4, classes: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return torch.randn(batch, height, width, 4 + classes, generator=generator)


def test_planes_follow_the_paper_order() -> None:
    """Planes 0, 1 are the centre vector (y, x), planes 2, 3 the border vector, the rest logits."""
    features = torch.zeros(1, 1, 1, 6)
    features[0, 0, 0] = torch.tensor([0.1, 0.2, 0.3, 0.4, 5.0, 6.0])
    out = CentroidNetHeads(max_distance=10.0).forward(features=features)

    torch.testing.assert_close(out["centroid_vectors"][0, 0, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(out["border_vectors"][0, 0, 0], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(out["logits"][0, 0, 0], torch.tensor([5.0, 6.0]))


@pytest.mark.parametrize("classes", [1, 2, 5])
@pytest.mark.parametrize("max_distance", [1.0, 12.5, 40.0])
def test_matches_a_plain_slice(classes: int, max_distance: float) -> None:
    features = _features(classes=classes)
    out = CentroidNetHeads(max_distance=max_distance).forward(features=features)

    torch.testing.assert_close(out["centroid_vectors"], features[..., 0:2] * max_distance)
    torch.testing.assert_close(out["border_vectors"], features[..., 2:4] * max_distance)
    assert torch.equal(out["logits"], features[..., 4:])


def test_port_contract() -> None:
    features = _features(batch=2, height=5, width=3, classes=3)
    out = CentroidNetHeads(max_distance=8.0).forward(features=features)

    assert set(out) == set(CentroidNetHeads.OUTPUT_SPECS)
    assert out["centroid_vectors"].shape == (2, 5, 3, 2)
    assert out["border_vectors"].shape == (2, 5, 3, 2)
    assert out["logits"].shape == (2, 5, 3, 3)
    for name, spec in CentroidNetHeads.OUTPUT_SPECS.items():
        assert out[name].dtype == spec.dtype


def test_gradient_reaches_every_plane() -> None:
    features = _features().requires_grad_(True)
    out = CentroidNetHeads(max_distance=4.0).forward(features=features)
    (out["centroid_vectors"].sum() + out["border_vectors"].sum() + out["logits"].sum()).backward()

    expected = torch.ones_like(features)
    expected[..., :4] = 4.0
    torch.testing.assert_close(features.grad, expected)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_rejects_a_vote_range_that_is_not_positive_and_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="max_distance"):
        CentroidNetHeads(max_distance=bad)


@pytest.mark.parametrize("channels", [1, 4])
def test_rejects_a_backbone_output_without_class_planes(channels: int) -> None:
    with pytest.raises(ValueError, match="at least 5"):
        CentroidNetHeads(max_distance=4.0).forward(features=torch.zeros(1, 2, 2, channels))


def test_hparams_keep_the_vote_range() -> None:
    assert CentroidNetHeads(max_distance=17.0).hparams["max_distance"] == 17.0


class _Probe(torch.nn.Module):
    def __init__(self, node: CentroidNetHeads) -> None:
        super().__init__()
        self.node = node

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.node(features=features)
        return out["centroid_vectors"], out["border_vectors"], out["logits"]


def test_export_trace_reproduces_forward() -> None:
    node = CentroidNetHeads(max_distance=9.0).eval()
    program = torch.export.export(_Probe(node), (_features(),))

    other = _features() * 3.0
    traced = program.module()(other)
    expected = node.forward(features=other)
    for got, name in zip(traced, ("centroid_vectors", "border_vectors", "logits"), strict=True):
        torch.testing.assert_close(got, expected[name])
