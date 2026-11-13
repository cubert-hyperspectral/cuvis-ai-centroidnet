"""Chain: encoder targets decoded by the vote nodes, a backbone trained on them, the training yaml.

Needs cuvis-ai with VectorFieldVoteAccumulator and LocalMaximaSuppression, cuvis-ai-unet
and cuvis-ai-augment.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from cuvis_ai_core.utils.node_registry import NodeRegistry
from scenes import disc_instances

from cuvis_ai_centroidnet.configs import PIPELINE_DIR

pytestmark = pytest.mark.chain

TRAIN_YAML = PIPELINE_DIR / "centroidnet_fod_lentils_train.yaml"


def _decode(instances: torch.Tensor, max_distance: float, window: int = 7):
    """Encode ``[B, H, W]`` instances and decode the centre targets; returns votes and peaks."""
    from cuvis_ai.node.spatial import LocalMaximaSuppression
    from cuvis_ai.node.vote_accumulation import VectorFieldVoteAccumulator

    from cuvis_ai_centroidnet.node.encoder import InstanceVectorFieldEncoder

    targets = InstanceVectorFieldEncoder(max_distance=max_distance).forward(instances=instances)
    votes = VectorFieldVoteAccumulator(order="yx").forward(vectors=targets["centroid_vectors"])
    peaks = LocalMaximaSuppression(window=window).forward(scores=votes["scores"])["scores"]
    return votes["scores"], peaks


def _expected_votes(instances: torch.Tensor, max_distance: float) -> torch.Tensor:
    """Vote map of ONE instance in an ``[H, W]`` frame, counted pixel by pixel.

    A pixel within ``max_distance`` of the border votes for the pixel it reaches when both
    components of its vector to the centroid are rounded half to even. Every other pixel
    votes for itself.
    """
    height, width = instances.shape
    ys, xs = (instances > 0).nonzero(as_tuple=True)
    centre_y, centre_x = float(ys.float().mean()), float(xs.float().mean())
    border = [
        (y, x)
        for y, x in zip(ys.tolist(), xs.tolist(), strict=True)
        if any(
            not (0 <= ny < height and 0 <= nx < width) or instances[ny, nx] == 0
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
        )
    ]
    votes = torch.zeros(height, width)
    for y in range(height):
        for x in range(width):
            nearest = min((by - y) ** 2 + (bx - x) ** 2 for by, bx in border)
            if nearest**0.5 <= max_distance:
                votes[y + round(centre_y - y), x + round(centre_x - x)] += 1
            else:
                votes[y, x] += 1
    return votes


def test_every_vote_of_an_odd_disc_lands_on_its_centre_pixel() -> None:
    from cuvis_ai_centroidnet.node.count import VotePeakCount

    instances = disc_instances(31, 33, [(14, 17, 5)])
    votes, peaks = _decode(instances.unsqueeze(0), max_distance=8.0)

    expected = _expected_votes(instances, 8.0)
    assert torch.equal(votes[0, ..., 0], expected)
    # Everything within 8 pixels of the rim votes for (14, 17); nothing else gets two votes.
    assert expected[14, 17] > 400
    assert int((expected > 1).sum()) == 1
    assert VotePeakCount(min_votes=2.0).forward(peaks=peaks)["count"].tolist() == [1]


def test_a_centre_between_four_pixels_splits_its_votes_over_them() -> None:
    """A 6 x 6 square has its centroid at (12.5, 12.5); the offsets round half to even."""
    instances = torch.zeros(26, 26, dtype=torch.int32)
    instances[10:16, 10:16] = 1
    votes, peaks = _decode(instances.unsqueeze(0), max_distance=6.0)

    expected = _expected_votes(instances, 6.0)
    assert torch.equal(votes[0, ..., 0], expected)
    shared = expected[12:14, 12:14]
    assert int((expected > 1).sum()) == 4
    assert shared.min() > 50
    # The four pixels are one plateau or near it; suppression keeps a single peak.
    assert int((peaks[0, ..., 0] >= 2).sum()) == 1


def test_two_neighbouring_objects_decode_to_two_peaks_that_match_their_centroids() -> None:
    from cuvis_ai_schemas.enums import ExecutionStage
    from cuvis_ai_schemas.execution import Context

    from cuvis_ai_centroidnet.node.metrics import CentroidDetectionMetrics

    # Rims two pixels apart, centres 12 pixels apart: more than the suppression window.
    instances = disc_instances(30, 40, [(14, 13, 5), (14, 25, 5)]).unsqueeze(0)
    votes, peaks = _decode(instances, max_distance=7.0, window=7)

    metrics = CentroidDetectionMetrics(thresholds=[20.0], match_radius=1.0).forward(
        peaks=peaks, instances=instances, context=Context(stage=ExecutionStage.VAL)
    )["metrics"]
    values = {metric.name: metric.value for metric in metrics}

    assert votes[0, 14, 13, 0] > 20 and votes[0, 14, 25, 0] > 20
    assert values["predicted_count_at_20"] == 2.0
    assert values["f1_at_20"] == 1.0


def test_a_small_backbone_fits_one_batch_and_its_votes_give_the_right_count() -> None:
    """All three losses fall, the vector planes get gradients, the decoded count is right."""
    from cuvis_ai.node.spatial import LocalMaximaSuppression
    from cuvis_ai.node.vote_accumulation import VectorFieldVoteAccumulator
    from cuvis_ai_unet.node.dynunet import DynUNet
    from cuvis_ai_unet.node.losses import CrossEntropyLoss

    from cuvis_ai_centroidnet.node.count import VotePeakCount
    from cuvis_ai_centroidnet.node.encoder import InstanceVectorFieldEncoder
    from cuvis_ai_centroidnet.node.heads import CentroidNetHeads
    from cuvis_ai_centroidnet.node.losses import VectorFieldLoss

    torch.manual_seed(0)
    max_distance = 10.0
    instances = torch.stack(
        [
            disc_instances(48, 48, [(14, 14, 5), (32, 34, 6)]),
            disc_instances(48, 48, [(24, 22, 7), (9, 38, 4), (40, 40, 4)]),
        ]
    )
    # Objects are brighter than the background in four bands, plus a little noise.
    bands = torch.tensor([0.25, 0.5, 0.75, 1.0])
    cube = (instances > 0).float().unsqueeze(-1) * bands + 0.05 * torch.randn(2, 48, 48, 4)

    net = DynUNet(mode="2d", in_channels=4, num_classes=6, features=[8, 16, 32])
    net.train()
    heads = CentroidNetHeads(max_distance=max_distance)
    targets = InstanceVectorFieldEncoder(max_distance=max_distance).forward(instances=instances)
    vector_loss = VectorFieldLoss(max_distance=max_distance)
    class_loss = CrossEntropyLoss()

    def losses() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = net.forward(data=cube)["logits"]
        features.retain_grad()
        out = heads.forward(features=features)
        centre = vector_loss.forward(
            predictions=out["centroid_vectors"],
            targets=targets["centroid_vectors"],
            weights=targets["weights"],
        )["loss"]
        border = vector_loss.forward(
            predictions=out["border_vectors"],
            targets=targets["border_vectors"],
            weights=targets["weights"],
        )["loss"]
        classes = class_loss.forward(logits=out["logits"], targets=targets["object_mask"])["loss"]
        return centre, border, classes, features

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    first: list[float] = []
    for step in range(100):
        centre, border, classes, features = losses()
        optimizer.zero_grad()
        (centre + border + classes).backward()
        if step == 0:
            first = [centre.item(), border.item(), classes.item()]
            # Every group of output planes receives a gradient: centre, border, logits.
            plane_gradient = features.grad.abs().sum(dim=(0, 1, 2))
            assert torch.isfinite(plane_gradient).all()
            assert (plane_gradient > 0).all()
        optimizer.step()

    last = [centre.item(), border.item(), classes.item()]
    for before, after in zip(first, last, strict=True):
        assert after < before / 3

    net.eval()
    with torch.no_grad():
        vectors = heads.forward(features=net.forward(data=cube)["logits"])["centroid_vectors"]
        votes = VectorFieldVoteAccumulator(order="yx").forward(vectors=vectors)["scores"]
        peaks = LocalMaximaSuppression(window=7).forward(scores=votes)["scores"]
    assert VotePeakCount(min_votes=10.0).forward(peaks=peaks)["count"].tolist() == [2, 3]


def _batch(seed: int, frames: int = 2) -> dict[str, torch.Tensor]:
    """A lentils-shaped batch, smaller: 61 bands, a class mask with ids 0 and 3."""
    generator = torch.Generator().manual_seed(seed)
    layouts = [[(60, 70, 14), (150, 180, 20)], [(120, 90, 18)]]
    mask = torch.stack([disc_instances(240, 260, layouts[i % 2]) for i in range(frames)])
    return {
        "cube": torch.rand(frames, 240, 260, 61, generator=generator),
        "mask": (mask.clamp(max=1) * 3).to(torch.int32),
        "wavelengths": torch.arange(450, 450 + 61 * 8, 8, dtype=torch.int32).repeat(frames, 1),
    }


def test_training_pipeline_crops_in_training_and_keeps_full_frames_in_validation(
    registry: NodeRegistry,
) -> None:
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_schemas.enums import ExecutionStage

    pipeline = CuvisPipeline.load_pipeline(str(TRAIN_YAML), node_registry=registry)
    batch = _batch(0)

    train = pipeline.forward(batch=batch, stage=ExecutionStage.TRAIN)
    with torch.no_grad():
        val = pipeline.forward(batch=batch, stage=ExecutionStage.VAL)

    # Two frames, four crops each.
    assert train[("Heads", "centroid_vectors")].shape == (8, 192, 192, 2)
    assert train[("Encoder", "centroid_vectors")].shape == (8, 192, 192, 2)
    assert ("Metrics", "metrics") not in train
    total = sum(train[(name, "loss")] for name in ("CentreLoss", "BorderLoss", "ClassLoss"))
    assert torch.isfinite(total) and total.requires_grad

    assert val[("Heads", "centroid_vectors")].shape == (2, 240, 260, 2)
    # Connected components of the class mask: the objects of the frame, whatever the class id.
    assert val[("Instances", "instances")].amax(dim=(1, 2)).tolist() == [2, 1]
    assert {metric.name for metric in val[("Metrics", "metrics")]} >= {"true_count", "f1_at_100"}


def test_training_pipeline_counts_at_inference_without_a_mask_after_a_reload(
    tmp_path: Path, registry: NodeRegistry
) -> None:
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_schemas.enums import ExecutionStage

    pipeline = CuvisPipeline.load_pipeline(str(TRAIN_YAML), node_registry=registry)
    batch = _batch(1)
    del batch["mask"]
    with torch.no_grad():
        before = pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)

    pipeline.save_to_file(str(tmp_path / "p.yaml"), save_weights=True)
    restored = CuvisPipeline.load_pipeline(
        str(tmp_path / "p.yaml"),
        weights_path=str(tmp_path / "p.pt"),
        node_registry=registry,
    )
    with torch.no_grad():
        after = restored.forward(batch=batch, stage=ExecutionStage.INFERENCE)

    assert before[("Count", "count")].shape == (2,)
    assert not any(name in {"Instances", "Encoder", "Metrics", "ClassLoss"} for name, _ in before)
    torch.testing.assert_close(
        after[("Heads", "centroid_vectors")],
        before[("Heads", "centroid_vectors")],
        rtol=0.0,
        atol=1e-5,
    )
    assert torch.equal(after[("Count", "count")], before[("Count", "count")])


def test_the_evaluation_pass_of_the_framework_reports_the_pooled_metrics(
    registry: NodeRegistry,
) -> None:
    """evaluate_split reduces the metric node once per pass, as the trainer does per epoch."""
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_core.training.evaluation import evaluate_split
    from cuvis_ai_schemas.enums import ExecutionStage

    pipeline = CuvisPipeline.load_pipeline(str(TRAIN_YAML), node_registry=registry)
    metrics_node = {node.name: node for node in pipeline.nodes}["Metrics"]
    loader = [_batch(2, frames=1), _batch(3, frames=2)]

    summary = evaluate_split(
        pipeline, loader, stage=ExecutionStage.VAL, metric_nodes=[metrics_node]
    )
    again = evaluate_split(pipeline, loader, stage=ExecutionStage.VAL, metric_nodes=[metrics_node])

    assert summary.frames == 3
    # Frames with 2, 2 and 1 objects.
    assert summary.metrics["Metrics/true_count"] == 5.0
    assert set(summary.metrics) == {f"Metrics/{name}" for name in metrics_node.POOLED_METRIC_NAMES}
    # A second pass starts from zero instead of adding to the first.
    assert again.metrics == summary.metrics
