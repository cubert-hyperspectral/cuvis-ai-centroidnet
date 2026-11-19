"""ConnectedComponentInstances: an instance map from a class mask."""

from __future__ import annotations

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage, NodeTag

from cuvis_ai_centroidnet.node.instances import ConnectedComponentInstances

pytestmark = pytest.mark.unit


def _mask(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor([rows], dtype=torch.int32)


def _flood_fill(mask: torch.Tensor, connectivity: int) -> torch.Tensor:
    """Reference labelling: raster scan, one flood fill per unlabelled object pixel."""
    height, width = mask.shape
    labels = torch.zeros(height, width, dtype=torch.int32)
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    next_id = 0
    for y in range(height):
        for x in range(width):
            if mask[y, x] <= 0 or labels[y, x] != 0:
                continue
            next_id += 1
            labels[y, x] = next_id
            stack = [(y, x)]
            while stack:
                cy, cx = stack.pop()
                for dy, dx in steps:
                    ny, nx = cy + dy, cx + dx
                    inside = 0 <= ny < height and 0 <= nx < width
                    if inside and labels[ny, nx] == 0 and mask[ny, nx] == mask[y, x]:
                        labels[ny, nx] = next_id
                        stack.append((ny, nx))
    return labels


def test_two_separate_objects_get_ids_in_row_major_order() -> None:
    mask = _mask(
        [
            [0, 0, 0, 3, 3],
            [3, 3, 0, 3, 3],
            [3, 3, 0, 0, 0],
        ]
    )
    instances = ConnectedComponentInstances().forward(mask=mask)["instances"]

    assert instances.tolist() == [
        [
            [0, 0, 0, 1, 1],
            [2, 2, 0, 1, 1],
            [2, 2, 0, 0, 0],
        ]
    ]


def test_touching_objects_of_two_classes_are_two_instances() -> None:
    mask = _mask([[1, 1, 2, 2], [1, 1, 2, 2]])
    instances = ConnectedComponentInstances().forward(mask=mask)["instances"]

    assert instances.tolist() == [[[1, 1, 2, 2], [1, 1, 2, 2]]]


def test_ids_follow_the_first_pixel_across_classes() -> None:
    """Class 5 starts before class 2 in the raster scan, so it gets the lower id."""
    mask = _mask([[0, 5, 0, 0], [2, 0, 0, 5], [2, 0, 0, 0]])
    instances = ConnectedComponentInstances().forward(mask=mask)["instances"]

    assert instances.tolist() == [[[0, 1, 0, 0], [2, 0, 0, 3], [2, 0, 0, 0]]]


def test_diagonal_neighbours_join_with_8_and_split_with_4() -> None:
    mask = _mask([[1, 0], [0, 1]])

    joined = ConnectedComponentInstances(connectivity=8).forward(mask=mask)["instances"]
    split = ConnectedComponentInstances(connectivity=4).forward(mask=mask)["instances"]
    assert joined.tolist() == [[[1, 0], [0, 1]]]
    assert split.tolist() == [[[1, 0], [0, 2]]]


@pytest.mark.parametrize("connectivity", [4, 8])
def test_matches_a_flood_fill_on_random_masks(connectivity: int) -> None:
    generator = torch.Generator().manual_seed(21)
    mask = torch.randint(0, 3, (3, 12, 14), generator=generator, dtype=torch.int32)
    instances = ConnectedComponentInstances(connectivity=connectivity).forward(mask=mask)

    for frame, got in zip(mask, instances["instances"], strict=True):
        assert torch.equal(got, _flood_fill(frame, connectivity))


def test_ids_restart_in_every_frame() -> None:
    mask = torch.zeros(2, 3, 3, dtype=torch.int32)
    mask[0, 0, 0] = 1
    mask[0, 2, 2] = 1
    mask[1, 1, 1] = 4
    instances = ConnectedComponentInstances().forward(mask=mask)["instances"]

    assert instances[0].unique().tolist() == [0, 1, 2]
    assert instances[1].unique().tolist() == [0, 1]


def test_empty_and_negative_masks_have_no_instance() -> None:
    mask = torch.zeros(1, 4, 4, dtype=torch.int32)
    mask[0, 1, 1] = -1
    instances = ConnectedComponentInstances().forward(mask=mask)["instances"]

    assert torch.count_nonzero(instances) == 0


def test_port_contract() -> None:
    mask = torch.zeros(2, 5, 6, dtype=torch.int32)
    mask[:, 1:3, 2:4] = 7
    instances = ConnectedComponentInstances().forward(mask=mask)["instances"]

    assert instances.shape == (2, 5, 6)
    assert instances.dtype == torch.int32
    assert instances.device == mask.device


def test_runs_with_labels_only_and_is_not_marked_exportable() -> None:
    assert ConnectedComponentInstances.EXECUTION_STAGES == frozenset(
        {ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST}
    )
    assert NodeTag.ONNX not in ConnectedComponentInstances._tags


@pytest.mark.parametrize("bad", [0, 6, 16, -4])
def test_rejects_an_unknown_connectivity(bad: int) -> None:
    with pytest.raises(ValueError, match="connectivity"):
        ConnectedComponentInstances(connectivity=bad)


def test_hparams_keep_the_connectivity() -> None:
    assert ConnectedComponentInstances(connectivity=4).hparams["connectivity"] == 4
