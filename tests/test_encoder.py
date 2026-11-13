"""InstanceVectorFieldEncoder: centre and border vector targets from an instance map."""

from __future__ import annotations

import math

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage, NodeTag
from scenes import disc_instances

from cuvis_ai_centroidnet.node.encoder import InstanceVectorFieldEncoder

pytestmark = pytest.mark.unit


def _reference(instances: torch.Tensor, max_distance: float):
    """Section 3.3.2 of the paper for one ``[H, W]`` frame, written out pixel by pixel.

    Returns centre vectors, border vectors (both ``[H, W, 2]`` as y, x) and weights ``[H, W]``.
    """
    height, width = instances.shape
    ids = sorted(int(i) for i in instances.unique().tolist() if i > 0)
    centre = torch.zeros(height, width, 2)
    border = torch.zeros(height, width, 2)
    weights = torch.ones(height, width)
    if not ids:
        return centre, border, weights

    centroids, borders, cut = {}, {}, {}
    for i in ids:
        pixels = [(y, x) for y in range(height) for x in range(width) if instances[y, x] == i]
        centroids[i] = (
            torch.tensor([p[0] for p in pixels], dtype=torch.float32).mean(),
            torch.tensor([p[1] for p in pixels], dtype=torch.float32).mean(),
        )
        borders[i] = []
        cut[i] = False
        for y, x in pixels:
            at_edge = y in (0, height - 1) or x in (0, width - 1)
            cut[i] = cut[i] or at_edge
            neighbours = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
            other = any(
                not (0 <= ny < height and 0 <= nx < width) or instances[ny, nx] != i
                for ny, nx in neighbours
            )
            if other:
                borders[i].append((y, x))

    for y in range(height):
        for x in range(width):
            best, best_d = None, None
            for i in ids:
                d = (y - centroids[i][0]) ** 2 + (x - centroids[i][1]) ** 2
                if best_d is None or d < best_d:
                    best, best_d = i, d
            nearest, nearest_d = None, None
            for by, bx in borders[best]:
                d = (by - y) ** 2 + (bx - x) ** 2
                if nearest_d is None or d < nearest_d:
                    nearest, nearest_d = (by, bx), d
            if math.sqrt(nearest_d) > max_distance:
                continue
            centre[y, x, 0] = centroids[best][0] - y
            centre[y, x, 1] = centroids[best][1] - x
            border[y, x, 0] = nearest[0] - y
            border[y, x, 1] = nearest[1] - x
            if cut[best]:
                weights[y, x] = 0.0
    return centre, border, weights


def _encode(instances: torch.Tensor, max_distance: float):
    batch = instances if instances.ndim == 3 else instances.unsqueeze(0)
    return InstanceVectorFieldEncoder(max_distance=max_distance).forward(instances=batch)


def test_one_disc_points_at_its_centre_and_at_its_rim() -> None:
    """Hand-checked: a radius-2 disc around (4, 6) in a 9 x 13 frame."""
    out = _encode(disc_instances(9, 13, [(4, 6, 2)]), max_distance=20.0)
    centre, border = out["centroid_vectors"][0], out["border_vectors"][0]

    # A pixel up and to the left of the disc points down (+y) and right (+x).
    assert centre[1, 2].tolist() == [3.0, 4.0]
    # The centre pixel itself: no offset to the centre. Its nearest border pixels are the
    # four diagonal ones; the first in row-major order is up and to the left.
    assert centre[4, 6].tolist() == [0.0, 0.0]
    assert border[4, 6].tolist() == [-1.0, -1.0]
    # A rim pixel is its own nearest border pixel.
    assert border[4, 8].tolist() == [0.0, 0.0]
    # A pixel right of the disc: the nearest rim pixel is (4, 8), four columns to the left.
    assert border[4, 12].tolist() == [0.0, -4.0]


@pytest.mark.parametrize("max_distance", [3.0, 6.5, 50.0])
def test_matches_the_pixel_loop_on_the_scene(scene: torch.Tensor, max_distance: float) -> None:
    out = _encode(scene, max_distance)
    centre, border, weights = _reference(scene, max_distance)

    torch.testing.assert_close(out["centroid_vectors"][0], centre, rtol=0.0, atol=1e-5)
    torch.testing.assert_close(out["border_vectors"][0], border, rtol=0.0, atol=1e-5)
    torch.testing.assert_close(out["weights"][0, ..., 0], weights)


def test_matches_the_pixel_loop_on_touching_and_cut_objects() -> None:
    """Overlapping discs share a boundary; one disc runs out of the frame."""
    instances = disc_instances(20, 22, [(5, 5, 4), (8, 10, 4), (17, 20, 5), (14, 3, 1)])
    out = _encode(instances, max_distance=7.0)
    centre, border, weights = _reference(instances, 7.0)

    torch.testing.assert_close(out["centroid_vectors"][0], centre, rtol=0.0, atol=1e-5)
    torch.testing.assert_close(out["border_vectors"][0], border, rtol=0.0, atol=1e-5)
    torch.testing.assert_close(out["weights"][0, ..., 0], weights)


def test_frames_of_a_batch_are_encoded_independently(scene: torch.Tensor) -> None:
    other = disc_instances(24, 30, [(12, 15, 5)])
    batch = torch.stack([scene, other])
    out = _encode(batch, max_distance=9.0)

    for index, frame in enumerate((scene, other)):
        single = _encode(frame, max_distance=9.0)
        for name in ("centroid_vectors", "border_vectors", "weights"):
            assert torch.equal(out[name][index], single[name][0])


def test_a_frame_without_instances_is_all_zero_vectors_with_full_weight() -> None:
    out = _encode(torch.zeros(1, 6, 7, dtype=torch.int32), max_distance=5.0)

    assert torch.count_nonzero(out["centroid_vectors"]) == 0
    assert torch.count_nonzero(out["border_vectors"]) == 0
    assert torch.equal(out["weights"], torch.ones(1, 6, 7, 1))


def test_a_one_pixel_instance_is_its_own_centre_and_border() -> None:
    instances = torch.zeros(7, 7, dtype=torch.int32)
    instances[3, 3] = 1
    out = _encode(instances, max_distance=2.5)

    assert out["centroid_vectors"][0, 3, 3].tolist() == [0.0, 0.0]
    assert out["border_vectors"][0, 3, 3].tolist() == [0.0, 0.0]
    assert out["centroid_vectors"][0, 3, 1].tolist() == [0.0, 2.0]
    # Three pixels away is beyond the vote range: both vectors are zero.
    assert out["centroid_vectors"][0, 3, 0].tolist() == [0.0, 0.0]
    assert out["border_vectors"][0, 3, 0].tolist() == [0.0, 0.0]


def test_an_exact_tie_between_two_centroids_goes_to_the_lower_id() -> None:
    """Two single pixels at (2, 1) and (2, 5): column 3 is exactly halfway."""
    instances = torch.zeros(5, 7, dtype=torch.int32)
    instances[2, 5] = 1
    instances[2, 1] = 2
    out = _encode(instances, max_distance=10.0)

    assert out["centroid_vectors"][0, 2, 3].tolist() == [0.0, 2.0]
    assert out["centroid_vectors"][0, 0, 3].tolist() == [2.0, 2.0]


def test_the_interior_of_an_instance_larger_than_the_vote_range_is_zeroed() -> None:
    """A radius-6 disc with a range of 3: its middle is more than 3 pixels from the rim."""
    out = _encode(disc_instances(21, 21, [(10, 10, 6)]), max_distance=3.0)

    assert out["centroid_vectors"][0, 10, 12].tolist() == [0.0, 0.0]
    assert out["border_vectors"][0, 10, 12].tolist() == [0.0, 0.0]
    # Two pixels inside the rim is within range and still points at the centre.
    assert out["centroid_vectors"][0, 10, 14].tolist() == [0.0, -4.0]


def test_an_instance_cut_by_the_edge_gets_zero_weight_within_its_range() -> None:
    instances = disc_instances(16, 24, [(8, 0, 4), (8, 16, 3)])
    out = _encode(instances, max_distance=5.0)
    weights = out["weights"][0, ..., 0]

    # On the cut disc and on background next to it.
    assert weights[8, 2] == 0.0
    assert weights[8, 6] == 0.0
    # The complete disc and the background around it keep their weight.
    assert weights[8, 16] == 1.0
    assert weights[8, 21] == 1.0
    # Far from every object the target is a zero vector and counts.
    assert weights[0, 10] == 1.0
    assert out["centroid_vectors"][0, 0, 10].tolist() == [0.0, 0.0]


def test_ids_need_not_be_consecutive() -> None:
    instances = disc_instances(12, 14, [(3, 3, 2), (8, 9, 2)])
    relabelled = torch.where(instances == 1, 40, instances)
    relabelled = torch.where(instances == 2, 7, relabelled).to(torch.int32)
    plain, sparse = _encode(instances, 6.0), _encode(relabelled, 6.0)

    # Ids only name the instances; no pixel of this scene is an exact tie.
    for name in ("centroid_vectors", "border_vectors", "weights"):
        assert torch.equal(plain[name], sparse[name])


def test_port_contract(scene: torch.Tensor) -> None:
    out = _encode(torch.stack([scene, scene]), max_distance=8.0)

    assert set(out) == set(InstanceVectorFieldEncoder.OUTPUT_SPECS)
    assert out["centroid_vectors"].shape == (2, 24, 30, 2)
    assert out["border_vectors"].shape == (2, 24, 30, 2)
    assert out["weights"].shape == (2, 24, 30, 1)
    assert out["object_mask"].shape == (2, 24, 30)
    for name, spec in InstanceVectorFieldEncoder.OUTPUT_SPECS.items():
        expected = torch.int32 if name == "object_mask" else torch.float32
        assert out[name].dtype == spec.dtype == expected
        assert not out[name].requires_grad


def test_the_object_mask_is_one_on_every_instance_whatever_its_id() -> None:
    """The class target of a two-class loss: object against background."""
    instances = disc_instances(12, 14, [(3, 3, 2), (8, 9, 2)])
    instances[instances == 2] = 40
    instances[0, 0] = -3
    out = _encode(instances, max_distance=6.0)

    assert out["object_mask"].unique().tolist() == [0, 1]
    assert torch.equal(out["object_mask"][0] == 1, instances > 0)


def test_no_border_vector_is_longer_than_the_vote_range(scene: torch.Tensor) -> None:
    out = _encode(scene, max_distance=4.0)

    assert float(out["border_vectors"].norm(dim=-1).max()) <= 4.0


def test_a_small_pairwise_budget_gives_the_same_result(scene: torch.Tensor) -> None:
    """The distance matrices are cut into chunks; the chunk size must not change anything."""
    whole = InstanceVectorFieldEncoder(max_distance=9.0).forward(instances=scene.unsqueeze(0))
    chunked = InstanceVectorFieldEncoder(max_distance=9.0, max_pairwise=64).forward(
        instances=scene.unsqueeze(0)
    )

    for name in ("centroid_vectors", "border_vectors", "weights"):
        assert torch.equal(whole[name], chunked[name])


def test_runs_with_labels_only_and_is_not_marked_exportable() -> None:
    assert InstanceVectorFieldEncoder.EXECUTION_STAGES == frozenset(
        {ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST}
    )
    assert NodeTag.ONNX not in InstanceVectorFieldEncoder._tags


@pytest.mark.parametrize("bad", [0.0, -3.0, math.nan, math.inf])
def test_rejects_a_vote_range_that_is_not_positive_and_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="max_distance"):
        InstanceVectorFieldEncoder(max_distance=bad)


@pytest.mark.parametrize("bad", [0, -5])
def test_rejects_a_pairwise_budget_that_is_not_positive(bad: int) -> None:
    with pytest.raises(ValueError, match="max_pairwise"):
        InstanceVectorFieldEncoder(max_distance=4.0, max_pairwise=bad)


def test_hparams_round_trip() -> None:
    hparams = InstanceVectorFieldEncoder(max_distance=11.0, max_pairwise=1000).hparams

    assert (hparams["max_distance"], hparams["max_pairwise"]) == (11.0, 1000)
