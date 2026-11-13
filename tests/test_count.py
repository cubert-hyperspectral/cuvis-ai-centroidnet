"""VotePeakCount: peaks at or above the vote threshold, per image."""

from __future__ import annotations

import math

import pytest
import torch

from cuvis_ai_centroidnet.node.count import VotePeakCount

pytestmark = pytest.mark.unit


def _peaks(values: list[list[list[float]]]) -> torch.Tensor:
    """Build ``[B, H, W, 1]`` from nested ``[B][H][W]`` lists."""
    return torch.tensor(values, dtype=torch.float32).unsqueeze(-1)


def test_counts_peaks_at_or_above_the_threshold_per_image() -> None:
    peaks = _peaks(
        [
            [[0.0, 30.0, 0.0], [29.0, 0.0, 31.0]],
            [[0.0, 0.0, 0.0], [0.0, 45.0, 0.0]],
        ]
    )
    count = VotePeakCount(min_votes=30).forward(peaks=peaks)["count"]

    assert count.tolist() == [2, 1]


def test_matches_a_plain_loop() -> None:
    generator = torch.Generator().manual_seed(3)
    peaks = torch.randint(0, 9, (4, 6, 5, 1), generator=generator).to(torch.float32)
    expected = [
        sum(1 for value in image.flatten().tolist() if value >= 4.0) for image in peaks.unbind(0)
    ]

    assert VotePeakCount(min_votes=4).forward(peaks=peaks)["count"].tolist() == expected


def test_port_contract() -> None:
    count = VotePeakCount(min_votes=2).forward(peaks=torch.zeros(3, 4, 5, 1))["count"]

    assert count.shape == (3,)
    assert count.dtype == VotePeakCount.OUTPUT_SPECS["count"].dtype == torch.int32


def test_a_frame_of_self_votes_counts_nothing() -> None:
    """A pixel with a zero vector votes for itself, so an empty frame is a map of ones."""
    assert VotePeakCount(min_votes=2).forward(peaks=torch.ones(1, 8, 8, 1))["count"].tolist() == [0]


def test_nan_and_minus_infinity_are_not_peaks() -> None:
    peaks = _peaks([[[math.nan, -math.inf, 5.0]]])

    assert VotePeakCount(min_votes=2).forward(peaks=peaks)["count"].tolist() == [1]


@pytest.mark.parametrize("bad", [1, 1.99, 0, -3, math.nan, math.inf])
def test_rejects_a_threshold_that_would_count_self_votes(bad: float) -> None:
    with pytest.raises(ValueError, match="min_votes"):
        VotePeakCount(min_votes=bad)


def test_hparams_keep_the_threshold() -> None:
    assert VotePeakCount(min_votes=12).hparams["min_votes"] == 12.0


class _Probe(torch.nn.Module):
    def __init__(self, node: VotePeakCount) -> None:
        super().__init__()
        self.node = node

    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        return self.node(peaks=peaks)["count"]


def test_export_trace_reproduces_forward() -> None:
    node = VotePeakCount(min_votes=3).eval()
    generator = torch.Generator().manual_seed(5)
    example = torch.randint(0, 7, (2, 5, 4, 1), generator=generator).to(torch.float32)
    program = torch.export.export(_Probe(node), (example,))

    other = torch.randint(0, 7, (2, 5, 4, 1), generator=generator).to(torch.float32)
    assert torch.equal(program.module()(other), node.forward(peaks=other)["count"])
