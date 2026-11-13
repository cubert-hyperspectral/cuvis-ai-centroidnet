"""Choose a vote threshold from the values ``CentroidDetectionMetrics`` reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cuvis_ai_centroidnet.node.metrics import PER_THRESHOLD_METRICS, metric_name


def scores_by_threshold(
    metrics: Mapping[str, float], thresholds: Sequence[float], prefix: str = ""
) -> dict[float, dict[str, float]]:
    """Regroup the reported values as ``{threshold: {metric: value}}``.

    ``metrics`` maps reported names to values, for example the ``metrics`` of an
    evaluation summary of the framework; ``prefix`` is what precedes the names there
    (``"<node name>/"``). Raises ``KeyError`` for a threshold the node did not score.
    """
    return {
        threshold: {
            name: metrics[prefix + metric_name(name, threshold)] for name in PER_THRESHOLD_METRICS
        }
        for threshold in thresholds
    }


def best_threshold(scores: Mapping[float, Mapping[str, float]]) -> float:
    """Return the threshold with the highest F1.

    Among thresholds with the same F1 the one with the lower count error wins, then the
    higher threshold, which gives fewer false alarms on data it was not chosen on. Choose
    on a validation split and keep the value for the test split.
    """
    if not scores:
        raise ValueError("best_threshold needs the scores of at least one threshold.")
    return max(scores, key=lambda t: (scores[t]["f1"], -scores[t]["count_error"], t))
