"""CentroidDetectionMetrics: vote peaks matched to instance centroids."""

from __future__ import annotations

import math

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory
from cuvis_ai_schemas.execution import Context, Metric
from scenes import disc_instances

from cuvis_ai_centroidnet.node.metrics import CentroidDetectionMetrics

pytestmark = pytest.mark.unit

VAL = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)


def _peaks(height: int, width: int, points: list[tuple[int, int, float]]) -> torch.Tensor:
    """Suppressed vote map ``[1, H, W, 1]``: the given ``(y, x, votes)``, zero elsewhere."""
    peaks = torch.zeros(1, height, width, 1)
    for y, x, votes in points:
        peaks[0, y, x, 0] = votes
    return peaks


def _points(height: int, width: int, points: list[tuple[int, int]]) -> torch.Tensor:
    """Instance map ``[1, H, W]`` of one-pixel instances; id ``i + 1`` for point ``i``."""
    instances = torch.zeros(1, height, width, dtype=torch.int32)
    for index, (y, x) in enumerate(points):
        instances[0, y, x] = index + 1
    return instances


def _values(out: dict) -> dict[str, float]:
    return {metric.name: metric.value for metric in out["metrics"]}


def _node(thresholds=(30.0,), match_radius: float = 3.0) -> CentroidDetectionMetrics:
    return CentroidDetectionMetrics(thresholds=list(thresholds), match_radius=match_radius)


def _reference(peaks: torch.Tensor, instances: torch.Tensor, threshold: float, radius: float):
    """The matching rule written out for one frame; returns (tp, fp, fn)."""
    height, width = instances.shape
    centroids = {}
    for i in sorted(int(i) for i in instances.unique().tolist() if i > 0):
        ys, xs = (instances == i).nonzero(as_tuple=True)
        centroids[i] = (float(ys.float().mean()), float(xs.float().mean()))
    found = [
        (float(peaks[y, x]), y, x)
        for y in range(height)
        for x in range(width)
        if peaks[y, x] >= threshold
    ]
    found.sort(key=lambda peak: (-peak[0], peak[1], peak[2]))
    free = dict(centroids)
    tp = 0
    for _, y, x in found:
        best, best_d = None, None
        for i, (cy, cx) in free.items():
            d = math.hypot(cy - y, cx - x)
            if d <= radius and (best_d is None or d < best_d):
                best, best_d = i, d
        if best is not None:
            del free[best]
            tp += 1
    return tp, len(found) - tp, len(free)


def test_hits_misses_and_false_alarms_on_a_hand_checked_frame() -> None:
    """Two discs; one peak on a centre, one a pixel off, one far from both."""
    instances = disc_instances(20, 20, [(5, 5, 2), (14, 13, 2)]).unsqueeze(0)
    peaks = _peaks(20, 20, [(5, 5, 40.0), (14, 14, 35.0), (2, 17, 31.0)])

    values = _values(_node().forward(peaks=peaks, instances=instances, context=VAL))

    assert values["true_positives_at_30"] == 2.0
    assert values["false_positives_at_30"] == 1.0
    assert values["false_negatives_at_30"] == 0.0
    assert values["precision_at_30"] == pytest.approx(2 / 3)
    assert values["recall_at_30"] == 1.0
    assert values["f1_at_30"] == pytest.approx(0.8)
    assert values["predicted_count_at_30"] == 3.0
    assert values["true_count"] == 2.0
    assert values["count_error_at_30"] == 1.0


def test_each_threshold_is_scored_on_its_own() -> None:
    instances = disc_instances(20, 20, [(5, 5, 2), (14, 13, 2)]).unsqueeze(0)
    peaks = _peaks(20, 20, [(5, 5, 40.0), (14, 14, 35.0), (2, 17, 31.0)])

    values = _values(
        _node(thresholds=(30.0, 32.5, 38.0)).forward(peaks=peaks, instances=instances, context=VAL)
    )

    assert (values["true_positives_at_32.5"], values["false_positives_at_32.5"]) == (2.0, 0.0)
    assert (values["true_positives_at_38"], values["false_negatives_at_38"]) == (1.0, 1.0)
    assert values["recall_at_38"] == 0.5
    assert values["predicted_count_at_38"] == 1.0


def test_a_peak_at_the_threshold_counts_and_one_just_below_does_not() -> None:
    instances = _points(9, 9, [(4, 4)])

    at = _values(
        _node().forward(peaks=_peaks(9, 9, [(4, 4, 30.0)]), instances=instances, context=VAL)
    )
    below = _values(
        _node().forward(peaks=_peaks(9, 9, [(4, 4, 29.5)]), instances=instances, context=VAL)
    )

    assert at["true_positives_at_30"] == 1.0
    assert (below["predicted_count_at_30"], below["false_negatives_at_30"]) == (0.0, 1.0)


def test_a_peak_exactly_at_the_match_radius_is_a_hit() -> None:
    instances = _points(9, 12, [(4, 2)])
    values = _values(
        _node(match_radius=5.0).forward(
            peaks=_peaks(9, 12, [(8, 5, 40.0)]), instances=instances, context=VAL
        )
    )
    beyond = _values(
        _node(match_radius=4.9).forward(
            peaks=_peaks(9, 12, [(8, 5, 40.0)]), instances=instances, context=VAL
        )
    )

    # (8, 5) is 4 rows and 3 columns from (4, 2): exactly 5 pixels.
    assert values["true_positives_at_30"] == 1.0
    assert (beyond["false_positives_at_30"], beyond["false_negatives_at_30"]) == (1.0, 1.0)


def test_two_peaks_on_one_instance_are_one_hit_and_one_false_alarm() -> None:
    instances = disc_instances(15, 15, [(7, 7, 4)]).unsqueeze(0)
    peaks = _peaks(15, 15, [(7, 6, 50.0), (7, 9, 45.0)])

    values = _values(_node().forward(peaks=peaks, instances=instances, context=VAL))

    assert values["true_positives_at_30"] == 1.0
    assert values["false_positives_at_30"] == 1.0
    assert values["false_negatives_at_30"] == 0.0


def test_the_strongest_peak_chooses_first_and_takes_the_nearest_centroid() -> None:
    """Centroids at columns 4 and 10; the strong peak sits between them, nearer to 4."""
    instances = _points(11, 15, [(5, 4), (5, 10)])
    # The weaker peak is next to centroid 1, which the stronger one has taken; centroid 2
    # is 5 pixels from it, beyond the radius of 4.
    peaks = _peaks(11, 15, [(5, 6, 50.0), (5, 5, 40.0)])

    values = _values(_node(match_radius=4.0).forward(peaks=peaks, instances=instances, context=VAL))

    assert values["true_positives_at_30"] == 1.0
    assert values["false_positives_at_30"] == 1.0
    assert values["false_negatives_at_30"] == 1.0


def test_a_peak_equally_far_from_two_centroids_takes_the_lower_id() -> None:
    """The peak at column 7 is 3 pixels from both centroids; id 1 is the right-hand one."""
    instances = _points(11, 15, [(5, 10), (5, 4)])
    # The second peak reaches only the left centroid (id 2). It is a hit only if the
    # first peak took id 1.
    peaks = _peaks(11, 15, [(5, 7, 50.0), (5, 2, 40.0)])

    values = _values(_node(match_radius=3.0).forward(peaks=peaks, instances=instances, context=VAL))

    assert values["true_positives_at_30"] == 2.0
    assert values["false_positives_at_30"] == 0.0


@pytest.mark.parametrize(
    ("first_votes", "second_votes", "hits"), [(40.0, 40.0, 1.0), (40.0, 41.0, 2.0)]
)
def test_peaks_with_equal_votes_choose_in_row_major_order(
    first_votes: float, second_votes: float, hits: float
) -> None:
    """Centroids at columns 2 and 10. The peak at column 7 reaches both, the one at 12 only 10.

    When the peak at column 7 chooses first it takes the nearer centroid at column 10 and
    leaves the other peak without a partner. When the peak at column 12 chooses first,
    both find one.
    """
    instances = _points(11, 15, [(5, 10), (5, 2)])
    peaks = _peaks(11, 15, [(5, 7, first_votes), (5, 12, second_votes)])

    values = _values(_node(match_radius=5.0).forward(peaks=peaks, instances=instances, context=VAL))

    assert values["true_positives_at_30"] == hits


def test_a_concave_instance_is_matched_at_its_centroid_outside_the_mask() -> None:
    """A ring: the centroid is the hole in the middle."""
    ring = disc_instances(21, 21, [(10, 10, 7)])
    ring[disc_instances(21, 21, [(10, 10, 4)]) > 0] = 0
    assert ring[10, 10] == 0

    values = _values(
        _node(match_radius=2.0).forward(
            peaks=_peaks(21, 21, [(10, 10, 60.0)]), instances=ring.unsqueeze(0), context=VAL
        )
    )

    assert (values["true_positives_at_30"], values["false_negatives_at_30"]) == (1.0, 0.0)


def test_single_votes_of_the_background_are_never_peaks() -> None:
    """A pixel with a zero vector votes for itself, so an untouched vote map is all ones."""
    instances = disc_instances(12, 12, [(6, 6, 2)]).unsqueeze(0)
    values = _values(
        _node(thresholds=(2.0,)).forward(
            peaks=torch.ones(1, 12, 12, 1), instances=instances, context=VAL
        )
    )

    assert values["predicted_count_at_2"] == 0.0
    assert values["false_negatives_at_2"] == 1.0


def test_an_empty_frame_reports_zero_for_every_ratio() -> None:
    values = _values(
        _node().forward(
            peaks=torch.zeros(1, 8, 8, 1),
            instances=torch.zeros(1, 8, 8, dtype=torch.int32),
            context=VAL,
        )
    )

    assert set(values.values()) == {0.0}
    assert all(math.isfinite(value) for value in values.values())


def test_matches_the_written_out_rule_on_random_frames() -> None:
    generator = torch.Generator().manual_seed(7)
    node = _node(thresholds=(30.0,), match_radius=3.5)
    totals = [0, 0, 0]
    for batch_idx in range(6):
        centres = torch.randint(3, 27, (5, 2), generator=generator).tolist()
        instances = disc_instances(30, 30, [(y, x, 2) for y, x in centres])
        peaks = torch.zeros(30, 30)
        spots = torch.randint(0, 30, (14, 2), generator=generator)
        # Votes repeat on purpose, so the order among equal peaks matters.
        peaks[spots[:, 0], spots[:, 1]] = torch.randint(28, 34, (14,), generator=generator).float()
        for index, part in enumerate(_reference(peaks, instances, 30.0, 3.5)):
            totals[index] += part
        out = node.forward(
            peaks=peaks[None, ..., None],
            instances=instances.unsqueeze(0),
            context=Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=batch_idx),
        )

    values = _values(out)
    assert totals[0] > 0 and totals[1] > 0 and totals[2] > 0
    assert values["true_positives_at_30"] == totals[0]
    assert values["false_positives_at_30"] == totals[1]
    assert values["false_negatives_at_30"] == totals[2]


def _four_frames() -> tuple[torch.Tensor, torch.Tensor]:
    """Frames with 1 of 1, 0 of 3, 2 of 2 and 0 of 0 objects found; one false alarm in the last."""
    layouts = [
        ([(4, 4)], [(4, 4, 40.0)]),
        ([(2, 2), (2, 8), (8, 5)], []),
        ([(3, 3), (8, 8)], [(3, 3, 50.0), (8, 9, 45.0)]),
        ([], [(5, 5, 33.0)]),
    ]
    instances = torch.cat([_points(11, 11, points) for points, _ in layouts])
    peaks = torch.cat([_peaks(11, 11, found) for _, found in layouts])
    return peaks, instances


def test_ratios_are_pooled_over_the_epoch_and_not_averaged_over_batches() -> None:
    peaks, instances = _four_frames()
    node = _node()
    for index in range(4):
        out = node.forward(
            peaks=peaks[index : index + 1],
            instances=instances[index : index + 1],
            context=Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=index),
        )

    values = _values(out)
    # 3 hits, 1 false alarm, 3 misses over the four frames.
    assert values["precision_at_30"] == pytest.approx(3 / 4)
    assert values["recall_at_30"] == pytest.approx(3 / 6)
    assert values["f1_at_30"] == pytest.approx(6 / 10)
    assert values["true_count"] == 6.0
    assert values["predicted_count_at_30"] == 4.0
    # |1 - 1| + |0 - 3| + |2 - 2| + |1 - 0| over four frames.
    assert values["count_error_at_30"] == pytest.approx(1.0)


@pytest.mark.parametrize("sizes", [(4,), (1, 3), (2, 2), (1, 1, 1, 1)])
def test_the_epoch_result_does_not_depend_on_the_batch_size(sizes: tuple[int, ...]) -> None:
    peaks, instances = _four_frames()
    node = _node(thresholds=(30.0, 42.0))
    start = 0
    for batch_idx, size in enumerate(sizes):
        node.forward(
            peaks=peaks[start : start + size],
            instances=instances[start : start + size],
            context=Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=batch_idx),
        )
        start += size

    pooled = {name: float(metric.compute()) for name, metric in node.pooled_metrics().items()}

    assert pooled["f1_at_30"] == pytest.approx(0.6)
    assert pooled["count_error_at_30"] == pytest.approx(1.0)
    assert pooled["true_positives_at_42"] == 2.0
    assert pooled["recall_at_42"] == pytest.approx(2 / 6)


def test_count_errors_of_opposite_sign_do_not_cancel() -> None:
    """Three peaks for one object, then none for two: the totals agree, the frames do not."""
    instances = torch.cat([_points(11, 11, [(5, 5)]), _points(11, 11, [(2, 2), (8, 8)])])
    peaks = torch.cat(
        [_peaks(11, 11, [(5, 5, 40.0), (1, 9, 40.0), (9, 1, 40.0)]), _peaks(11, 11, [])]
    )

    values = _values(_node().forward(peaks=peaks, instances=instances, context=VAL))

    assert values["predicted_count_at_30"] == values["true_count"] == 3.0
    assert values["count_error_at_30"] == 2.0


def test_a_new_epoch_and_a_new_stage_start_from_zero() -> None:
    node = _node()
    hit = (_peaks(9, 9, [(4, 4, 40.0)]), _points(9, 9, [(4, 4)]))
    miss = (_peaks(9, 9, []), _points(9, 9, [(4, 4)]))

    node.forward(peaks=hit[0], instances=hit[1], context=VAL)
    same_epoch = _values(
        node.forward(
            peaks=miss[0],
            instances=miss[1],
            context=Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=1),
        )
    )
    next_epoch = _values(
        node.forward(
            peaks=miss[0], instances=miss[1], context=Context(stage=ExecutionStage.VAL, epoch=1)
        )
    )
    test_stage = _values(
        node.forward(
            peaks=hit[0], instances=hit[1], context=Context(stage=ExecutionStage.TEST, epoch=1)
        )
    )

    assert same_epoch["recall_at_30"] == 0.5
    assert next_epoch["recall_at_30"] == 0.0
    assert (test_stage["recall_at_30"], test_stage["true_count"]) == (1.0, 1.0)


def test_every_reported_name_is_pooled_and_has_a_live_metric_object() -> None:
    node = _node(thresholds=(30.0, 45.0))
    assert node.pooled_metrics() == {}

    peaks, instances = _four_frames()
    out = node.forward(peaks=peaks, instances=instances, context=VAL)
    names = {metric.name for metric in out["metrics"]}

    assert names == set(node.POOLED_METRIC_NAMES) == set(node.pooled_metrics())
    assert len(names) == 2 * 8 + 1
    for metric in out["metrics"]:
        assert float(node.pooled_metrics()[metric.name].compute()) == pytest.approx(metric.value)


def test_pooled_objects_can_be_reset_from_outside() -> None:
    """The evaluation pass of the framework resets the objects before it starts."""
    node = _node()
    node.forward(peaks=_peaks(9, 9, [(4, 4, 40.0)]), instances=_points(9, 9, [(4, 4)]), context=VAL)
    for metric in node.pooled_metrics().values():
        metric.reset()

    values = _values(
        node.forward(
            peaks=_peaks(9, 9, []),
            instances=_points(9, 9, [(4, 4)]),
            context=Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=1),
        )
    )

    assert (values["true_count"], values["recall_at_30"]) == (1.0, 0.0)


def test_port_contract() -> None:
    peaks, instances = _four_frames()
    context = Context(stage=ExecutionStage.TEST, epoch=3, batch_idx=5)
    out = _node().forward(peaks=peaks, instances=instances, context=context)

    assert set(out) == set(CentroidDetectionMetrics.OUTPUT_SPECS) == {"metrics"}
    for metric in out["metrics"]:
        assert isinstance(metric, Metric)
        assert isinstance(metric.value, float)
        assert (metric.stage, metric.epoch, metric.batch_idx) == (ExecutionStage.TEST, 3, 5)


def test_runs_in_validation_and_test_only() -> None:
    assert CentroidDetectionMetrics.EXECUTION_STAGES == frozenset(
        {ExecutionStage.VAL, ExecutionStage.TEST}
    )
    assert CentroidDetectionMetrics._category == NodeCategory.METRIC


@pytest.mark.parametrize("bad", [[], [1.5], [30.0, 1.0], [math.nan], [30.0, 30.0]])
def test_rejects_thresholds_that_are_empty_below_two_or_repeated(bad: list[float]) -> None:
    with pytest.raises(ValueError, match="thresholds"):
        CentroidDetectionMetrics(thresholds=bad, match_radius=3.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_rejects_a_match_radius_that_is_not_positive_and_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="match_radius"):
        CentroidDetectionMetrics(thresholds=[30.0], match_radius=bad)


def test_hparams_round_trip() -> None:
    hparams = CentroidDetectionMetrics(thresholds=[45, 30.0], match_radius=4).hparams

    assert hparams["thresholds"] == [45.0, 30.0]
    assert hparams["match_radius"] == 4.0
    rebuilt = CentroidDetectionMetrics(**hparams)
    assert rebuilt.POOLED_METRIC_NAMES == _node(thresholds=(45.0, 30.0)).POOLED_METRIC_NAMES
