"""VectorFieldLoss: mean Euclidean distance of a vector field, in units of the vote range."""

from __future__ import annotations

import math

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory

from cuvis_ai_centroidnet.node.losses import VectorFieldLoss

pytestmark = pytest.mark.unit


def _fields(batch: int = 2, height: int = 4, width: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    predictions = torch.randn(batch, height, width, 2, generator=generator) * 6.0
    targets = torch.randn(batch, height, width, 2, generator=generator) * 6.0
    return predictions, targets


def _loop_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    max_distance: float,
    weights: torch.Tensor | None = None,
) -> float:
    """Eq. 6 written out pixel by pixel, in float64."""
    total, weight_sum = 0.0, 0.0
    batch, height, width, _ = predictions.shape
    for b in range(batch):
        for y in range(height):
            for x in range(width):
                dy = float(predictions[b, y, x, 0]) - float(targets[b, y, x, 0])
                dx = float(predictions[b, y, x, 1]) - float(targets[b, y, x, 1])
                w = 1.0 if weights is None else float(weights[b, y, x, 0])
                total += w * math.sqrt(dy * dy + dx * dx)
                weight_sum += w
    return total / weight_sum / max_distance


def test_hand_calculated_distance() -> None:
    """Two pixels: a 3-4-5 triangle and an exact hit; mean 2.5 pixels, 0.25 of a range of 10."""
    predictions = torch.tensor([[[[3.0, 4.0], [1.0, -2.0]]]])
    targets = torch.tensor([[[[0.0, 0.0], [1.0, -2.0]]]])
    loss = VectorFieldLoss(max_distance=10.0).forward(predictions=predictions, targets=targets)

    torch.testing.assert_close(loss["loss"], torch.tensor(0.25), rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("max_distance", [1.0, 7.5, 40.0])
def test_matches_a_plain_loop(max_distance: float) -> None:
    predictions, targets = _fields()
    loss = VectorFieldLoss(max_distance=max_distance).forward(
        predictions=predictions, targets=targets
    )["loss"]

    assert loss.item() == pytest.approx(_loop_loss(predictions, targets, max_distance), rel=1e-5)


def test_equals_the_loss_on_vectors_divided_by_the_range() -> None:
    """Dividing by max_distance gives the loss the paper computes on normalised vectors."""
    predictions, targets = _fields()
    in_pixels = VectorFieldLoss(max_distance=12.0).forward(
        predictions=predictions, targets=targets
    )["loss"]
    normalised = VectorFieldLoss(max_distance=1.0).forward(
        predictions=predictions / 12.0, targets=targets / 12.0
    )["loss"]

    torch.testing.assert_close(in_pixels, normalised, rtol=1e-5, atol=0.0)


def test_weight_multiplies_the_loss() -> None:
    predictions, targets = _fields()
    plain = VectorFieldLoss(max_distance=5.0).forward(predictions=predictions, targets=targets)
    scaled = VectorFieldLoss(max_distance=5.0, weight=0.25).forward(
        predictions=predictions, targets=targets
    )

    torch.testing.assert_close(scaled["loss"], plain["loss"] * 0.25)


def test_weights_give_a_weighted_mean() -> None:
    predictions, targets = _fields()
    generator = torch.Generator().manual_seed(2)
    weights = torch.rand(2, 4, 5, 1, generator=generator)
    loss = VectorFieldLoss(max_distance=3.0).forward(
        predictions=predictions, targets=targets, weights=weights
    )["loss"]

    assert loss.item() == pytest.approx(_loop_loss(predictions, targets, 3.0, weights), rel=1e-5)


def test_pixels_with_zero_weight_do_not_count_and_get_no_gradient() -> None:
    predictions, targets = _fields()
    predictions.requires_grad_(True)
    weights = torch.ones(2, 4, 5, 1)
    weights[:, :, 2:] = 0.0
    loss = VectorFieldLoss(max_distance=3.0).forward(
        predictions=predictions, targets=targets, weights=weights
    )["loss"]
    loss.backward()

    kept = _loop_loss(predictions.detach()[:, :, :2], targets[:, :, :2], 3.0)
    assert loss.item() == pytest.approx(kept, rel=1e-5)
    assert torch.count_nonzero(predictions.grad[:, :, 2:]) == 0
    assert torch.count_nonzero(predictions.grad[:, :, :2]) > 0


def test_all_weights_zero_gives_a_zero_loss_with_a_zero_gradient() -> None:
    predictions, targets = _fields()
    predictions.requires_grad_(True)
    loss = VectorFieldLoss(max_distance=3.0).forward(
        predictions=predictions, targets=targets, weights=torch.zeros(2, 4, 5, 1)
    )["loss"]
    loss.backward()

    assert loss.item() == 0.0
    assert torch.count_nonzero(predictions.grad) == 0
    assert torch.isfinite(predictions.grad).all()


def test_gradient_is_finite_where_prediction_equals_target() -> None:
    targets = torch.tensor([[[[2.0, -3.0], [0.0, 0.0]]]])
    predictions = targets.clone().requires_grad_(True)
    loss = VectorFieldLoss(max_distance=8.0).forward(predictions=predictions, targets=targets)
    loss["loss"].backward()

    assert torch.isfinite(predictions.grad).all()
    assert loss["loss"].item() == pytest.approx(0.0, abs=1e-6)


def test_gradient_points_from_target_to_prediction() -> None:
    """One pixel 5 pixels off: the gradient is the unit direction over the range."""
    predictions = torch.tensor([[[[3.0, 4.0]]]], requires_grad=True)
    VectorFieldLoss(max_distance=10.0).forward(
        predictions=predictions, targets=torch.zeros(1, 1, 1, 2)
    )["loss"].backward()

    torch.testing.assert_close(
        predictions.grad, torch.tensor([[[[0.06, 0.08]]]]), rtol=1e-5, atol=1e-7
    )


def test_port_contract_and_stages() -> None:
    predictions, targets = _fields()
    loss = VectorFieldLoss(max_distance=2.0).forward(predictions=predictions, targets=targets)

    assert set(loss) == {"loss"}
    assert loss["loss"].shape == ()
    assert loss["loss"].dtype == torch.float32
    assert VectorFieldLoss._category == NodeCategory.LOSS
    assert VectorFieldLoss.EXECUTION_STAGES == frozenset(
        {ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST}
    )


@pytest.mark.parametrize("bad", [0.0, -2.0, math.nan, math.inf])
def test_rejects_a_vote_range_that_is_not_positive_and_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="max_distance"):
        VectorFieldLoss(max_distance=bad)


@pytest.mark.parametrize("bad", [0.0, -1e-9, math.nan, math.inf])
def test_rejects_an_eps_that_is_not_positive_and_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="eps"):
        VectorFieldLoss(max_distance=4.0, eps=bad)


def test_hparams_round_trip() -> None:
    hparams = VectorFieldLoss(max_distance=9.0, weight=0.5, eps=1e-10).hparams

    assert (hparams["max_distance"], hparams["weight"], hparams["eps"]) == (9.0, 0.5, 1e-10)


class _Probe(torch.nn.Module):
    def __init__(self, node: VectorFieldLoss) -> None:
        super().__init__()
        self.node = node

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        return self.node(predictions=predictions, targets=targets, weights=weights)["loss"]


def test_export_trace_reproduces_forward() -> None:
    node = VectorFieldLoss(max_distance=6.0).eval()
    predictions, targets = _fields()
    weights = torch.ones(2, 4, 5, 1)
    program = torch.export.export(_Probe(node), (predictions, targets, weights))

    other_weights = torch.zeros(2, 4, 5, 1)
    other_weights[0] = 1.0
    traced = program.module()(targets, predictions * 0.5, other_weights)
    expected = node.forward(predictions=targets, targets=predictions * 0.5, weights=other_weights)
    torch.testing.assert_close(traced, expected["loss"])
