"""Choosing a vote threshold from the values CentroidDetectionMetrics reports."""

from __future__ import annotations

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

from cuvis_ai_centroidnet.node.metrics import CentroidDetectionMetrics
from cuvis_ai_centroidnet.selection import best_threshold, scores_by_threshold

pytestmark = pytest.mark.unit


def _scores(rows: dict[float, tuple[float, float]]) -> dict[float, dict[str, float]]:
    """Scores per threshold from ``{threshold: (f1, count_error)}``."""
    return {t: {"f1": f1, "count_error": error} for t, (f1, error) in rows.items()}


def test_the_threshold_with_the_highest_f1_wins() -> None:
    assert best_threshold(_scores({10.0: (0.5, 0.1), 20.0: (0.8, 0.9), 50.0: (0.7, 0.0)})) == 20.0


def test_equal_f1_goes_to_the_lower_count_error() -> None:
    assert best_threshold(_scores({10.0: (0.8, 0.4), 20.0: (0.8, 0.2), 50.0: (0.8, 0.3)})) == 20.0


def test_equal_f1_and_count_error_go_to_the_higher_threshold() -> None:
    assert best_threshold(_scores({10.0: (0.8, 0.2), 20.0: (0.8, 0.2), 5.0: (0.8, 0.2)})) == 20.0


def test_no_scores_is_an_error() -> None:
    with pytest.raises(ValueError, match="threshold"):
        best_threshold({})


def test_scores_are_read_from_the_metric_names_of_the_node() -> None:
    """The values come straight from a metric node, with and without a ``<node>/`` prefix."""
    node = CentroidDetectionMetrics(thresholds=[2.5, 30.0], match_radius=2.0)
    peaks = torch.zeros(1, 9, 9, 1)
    peaks[0, 4, 4, 0] = 40.0
    peaks[0, 1, 7, 0] = 3.0
    instances = torch.zeros(1, 9, 9, dtype=torch.int32)
    instances[0, 4, 4] = 1
    out = node.forward(peaks=peaks, instances=instances, context=Context(stage=ExecutionStage.VAL))
    values = {metric.name: metric.value for metric in out["metrics"]}

    plain = scores_by_threshold(values, node.thresholds)
    prefixed = scores_by_threshold(
        {f"Metrics/{name}": value for name, value in values.items()},
        node.thresholds,
        prefix="Metrics/",
    )

    assert plain == prefixed
    assert list(plain) == [2.5, 30.0]
    assert plain[30.0]["f1"] == 1.0
    assert plain[2.5]["false_positives"] == 1.0
    assert plain[2.5]["precision"] == 0.5
    assert set(plain[30.0]) == {
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
        "predicted_count",
        "count_error",
    }
    assert best_threshold(plain) == 30.0


def test_a_threshold_the_node_did_not_score_is_an_error() -> None:
    with pytest.raises(KeyError, match="_at_7"):
        scores_by_threshold({"f1_at_30": 1.0}, [7.0])
