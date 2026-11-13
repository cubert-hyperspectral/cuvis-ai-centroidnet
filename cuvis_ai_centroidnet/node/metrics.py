"""Score vote peaks against the centroids of labelled instances."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context, Metric
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor, nn
from torchmetrics import Metric as TorchMetric

from cuvis_ai_centroidnet._geometry import instance_layout
from cuvis_ai_centroidnet._validation import positive_pixels, vote_threshold

# Reported once per vote threshold, under the name ``metric_name`` gives.
PER_THRESHOLD_METRICS = (
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "predicted_count",
    "count_error",
)
# The names that are plain sums; the others are ratios of two sums.
_COUNTS = frozenset({"true_positives", "false_positives", "false_negatives", "predicted_count"})


def metric_name(name: str, threshold: float) -> str:
    """Return the reported name of a per-threshold metric, for example ``f1_at_30``."""
    return f"{name}_at_{threshold:g}"


class _EpochRatio(TorchMetric):
    """Ratio of two counts, each summed over the epoch before the division.

    Without a denominator the metric is the plain sum of the numerator. A ratio whose
    denominator is zero is reported as 0.
    """

    full_state_update = False
    numerator: Tensor
    denominator: Tensor

    def __init__(self, is_ratio: bool) -> None:
        super().__init__()
        self.is_ratio = is_ratio
        self.add_state("numerator", torch.zeros((), dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("denominator", torch.zeros((), dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, numerator: int, denominator: int = 0) -> None:
        """Add the counts of one batch."""
        self.numerator += numerator
        self.denominator += denominator

    def compute(self) -> Tensor:
        """Return the sum, or the ratio of the two sums."""
        total = self.numerator.to(torch.float32)
        if not self.is_ratio:
            return total
        divisor = self.denominator.to(torch.float32)
        return torch.where(divisor > 0, total / divisor.clamp_min(1.0), torch.zeros_like(total))


def _count_hits(distance: Tensor, match_radius: float) -> int:
    """Match peaks to centroids one to one and return the number of matched peaks.

    ``distance`` is ``[P, N]`` with the peaks in the order in which they choose. Each peak
    takes the nearest centroid that is still free and within ``match_radius``; among
    centroids at the same distance the first one (the lower id).
    """
    reachable = distance <= match_radius
    free = torch.ones(distance.shape[1], dtype=torch.bool, device=distance.device)
    hits = 0
    # Only peaks with a centroid in reach can match; the others never enter the loop.
    for peak in reachable.any(dim=1).nonzero().squeeze(1).tolist():
        candidates = torch.where(reachable[peak] & free, distance[peak], math.inf)
        nearest = candidates.min()
        if torch.isfinite(nearest):
            free[(candidates == nearest).nonzero()[0]] = False
            hits += 1
    return hits


class CentroidDetectionMetrics(Node):
    """Precision, recall, F1 and count error of vote peaks against labelled instances.

    ``peaks`` is a vote map after non-maximum suppression (the output of
    ``LocalMaximaSuppression``): the number of votes at every local maximum and zero
    elsewhere. Every pixel with at least ``threshold`` votes is a predicted object centre
    (Eq. 12 of the paper). The ground truth is the centroid of every instance of
    ``instances``.

    Matching, per frame and per threshold:

    1. The peaks choose in order of their votes, the strongest first. Peaks with the same
       number of votes choose in row-major order.
    2. A peak takes the nearest centroid within ``match_radius`` pixels that no earlier
       peak has taken. Of two centroids at the same distance it takes the lower id.
    3. A peak that found a centroid is a true positive, any other peak a false positive,
       and a centroid that no peak took a false negative.

    The paper matches fitted object shapes to annotated shapes by intersection over
    union (Section 4.4, Eq. 13 to 19). This node works on the centres alone, before any
    shape is fitted, which is why it matches by distance. A centroid can lie outside a
    concave object; it is still the point the network was trained to vote for.

    Reported for every vote threshold ``t`` in ``thresholds`` (names end in ``_at_<t>``):
    ``true_positives``, ``false_positives``, ``false_negatives``, ``precision``,
    ``recall``, ``f1``, ``predicted_count`` (the number of peaks) and ``count_error``
    (the mean over frames of the absolute difference between predicted and true count,
    so over- and under-counting in different frames do not cancel). ``true_count`` is
    reported once. Several thresholds in one pass make it possible to choose the
    threshold on a validation split.

    Every value is pooled over the epoch: counts are summed over all frames, and each
    ratio is formed once from the sums. The value on the ``metrics`` port is therefore the
    running value over the batches of the current epoch, and the last batch carries the
    epoch result, independent of the batch size. A ratio with a zero denominator is 0.
    The node runs in validation and test.

    Parameters
    ----------
    thresholds : Sequence[float]
        Vote thresholds to score. Each must be at least 2, because a pixel whose vector
        is zero votes for itself; they must be distinct.
    match_radius : float
        Largest distance in pixels between a peak and the centroid it may match.
        Positive and finite.

    Raises
    ------
    ValueError
        If ``thresholds`` is empty, holds a value below 2 or not finite, or holds the same
        value twice, or if ``match_radius`` is not positive and finite.

    References
    ----------
    K. Dijkstra, J. van de Loosdrecht, W. A. Atsma, L. R. B. Schomaker, M. A. Wiering.
    "CentroidNetV2: A hybrid deep neural network for small-object segmentation and
    counting." Neurocomputing 423 (2021), pages 490 to 505, Section 4.4, Eq. 12 to 19.
    https://doi.org/10.1016/j.neucom.2020.10.075
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.DETECTION})

    EXECUTION_STAGES = frozenset({ExecutionStage.VAL, ExecutionStage.TEST})

    INPUT_SPECS = {
        "peaks": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Suppressed vote map [B, H, W, 1]: votes at local maxima, 0 elsewhere",
        ),
        "instances": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Instance map [B, H, W]: one positive id per object, 0 = background",
        ),
    }
    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects")}

    # Every reported name accumulates over the epoch. The trainer logs the live objects
    # of pooled_metrics() once per epoch instead of averaging the per-batch values. The
    # names depend on the thresholds, so each instance sets its own.
    POOLED_METRIC_NAMES: frozenset[str] = frozenset()

    def __init__(self, thresholds: Sequence[float], match_radius: float, **kwargs: Any) -> None:
        owner = type(self).__name__
        self.thresholds = [vote_threshold(value, owner, "thresholds") for value in thresholds]
        # Two thresholds that print alike would report under the same names.
        labels = {metric_name("", value) for value in self.thresholds}
        if not labels or len(labels) != len(self.thresholds):
            raise ValueError(
                f"{owner}: thresholds must hold at least one vote threshold and no value "
                f"twice; got {list(thresholds)!r}."
            )
        self.match_radius = positive_pixels(match_radius, owner, "match_radius")
        super().__init__(thresholds=self.thresholds, match_radius=self.match_radius, **kwargs)

        objects = {"true_count": _EpochRatio(is_ratio=False)}
        for threshold in self.thresholds:
            for name in PER_THRESHOLD_METRICS:
                is_ratio = name not in _COUNTS
                objects[metric_name(name, threshold)] = _EpochRatio(is_ratio=is_ratio)
        # The list registers the objects as submodules, so they move with the pipeline;
        # a threshold such as 2.5 cannot be part of a submodule name, hence the plain dict.
        self._registered = nn.ModuleList(objects.values())
        self._objects = objects
        self.POOLED_METRIC_NAMES = frozenset(objects)
        self._last_key: tuple[ExecutionStage, int] | None = None

    def forward(
        self, peaks: Tensor, instances: Tensor, context: Context, **_: Any
    ) -> dict[str, Any]:
        """Add one batch to the epoch counts and return the running values."""
        objects = self._objects
        epoch_key = (context.stage, context.epoch)
        if self._last_key != epoch_key:
            for metric in objects.values():
                metric.reset()
            self._last_key = epoch_key

        frames = peaks.shape[0]
        true_count = 0
        # Per threshold: hits, predicted count, summed absolute count error.
        totals = [[0, 0, 0] for _ in self.thresholds]
        for votes, labels in zip(peaks[..., 0], instances, strict=True):
            centroids = instance_layout(labels.to(torch.int64)).centroids.to(torch.float64)
            true = centroids.shape[0]
            true_count += true
            for total, threshold in zip(totals, self.thresholds, strict=True):
                hits, predicted = self._score_frame(votes, centroids, threshold)
                total[0] += hits
                total[1] += predicted
                total[2] += abs(predicted - true)

        objects["true_count"].update(true_count)
        for (hits, predicted, error), threshold in zip(totals, self.thresholds, strict=True):
            false_positives = predicted - hits
            false_negatives = true_count - hits
            updates: dict[str, tuple[int, ...]] = {
                "true_positives": (hits,),
                "false_positives": (false_positives,),
                "false_negatives": (false_negatives,),
                "precision": (hits, predicted),
                "recall": (hits, true_count),
                "f1": (2 * hits, 2 * hits + false_positives + false_negatives),
                "predicted_count": (predicted,),
                "count_error": (error, frames),
            }
            for name, counts in updates.items():
                objects[metric_name(name, threshold)].update(*counts)

        metrics = [
            Metric(
                name=name,
                value=float(metric.compute()),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
            for name, metric in objects.items()
        ]
        return {"metrics": metrics}

    def _score_frame(self, votes: Tensor, centroids: Tensor, threshold: float) -> tuple[int, int]:
        """Return the matched and the total number of peaks of one ``[H, W]`` vote map."""
        flat = votes.reshape(-1)
        position = (flat >= threshold).nonzero().squeeze(1)
        predicted = position.numel()
        if predicted == 0 or centroids.shape[0] == 0:
            return 0, predicted

        # Strongest first; the stable sort keeps the row-major order among equal votes.
        position = position[torch.argsort(flat[position], descending=True, stable=True)]
        width = votes.shape[1]
        peak = torch.stack([position // width, position % width], dim=1).to(torch.float64)
        distance = (peak[:, None, :] - centroids[None, :, :]).square().sum(dim=-1).sqrt()
        return _count_hits(distance, self.match_radius), predicted

    def pooled_metrics(self) -> dict[str, TorchMetric]:
        """Return the live metric object of every reported name.

        The trainer and the evaluation pass of the framework reduce each object once per
        epoch. The mapping is empty until the node has seen a batch, so nothing is logged
        for a run in which it never ran.
        """
        if self._last_key is None:
            return {}
        return dict(self._objects)
