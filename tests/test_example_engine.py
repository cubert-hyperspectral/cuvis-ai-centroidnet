"""The shared code of examples/fod_lentils: the checks that guard a saved artifact.

The example scripts need the dataloader plugin and recordings to run end to end. These
tests cover what can go wrong before and after that: a missing weights file, a folder that
already holds a run, a threshold nobody scored, a git call that does not answer.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace

import click
import pytest
import torch
import yaml
from click.testing import CliRunner
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

from cuvis_ai_centroidnet.node.metrics import PER_THRESHOLD_METRICS, CentroidDetectionMetrics

pytestmark = pytest.mark.unit

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "fod_lentils"


@pytest.fixture(scope="module")
def example() -> Iterator[Callable[[str], ModuleType]]:
    """Import a module of the example folder by its bare name, as the scripts do."""
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        yield importlib.import_module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))


@pytest.fixture()
def engine(example: Callable[[str], ModuleType]) -> ModuleType:
    return example("_engine")


@pytest.fixture(autouse=True)
def commit_not_remembered(engine: ModuleType) -> Iterator[None]:
    """The engine reads the commit once per process; every test starts without one."""
    engine.git_commit.cache_clear()
    yield
    engine.git_commit.cache_clear()


def _row(f1: float, count_error: float) -> dict[str, float]:
    """The values of one vote threshold, as ``score_split`` returns them."""
    return dict.fromkeys(PER_THRESHOLD_METRICS, 0.0) | {"f1": f1, "count_error": count_error}


def _result(split: str, scores: dict[float, dict[str, float]]) -> dict:
    return {"split": split, "frames": 3, "true_count": 5.0, "scores": scores}


def _artifact(folder: Path) -> Path:
    """A saved pipeline config with a ``Count`` node, without weights."""
    artifact = folder / "pipeline.yaml"
    content = {"nodes": [{"name": "Count", "hparams": {"min_votes": 100.0}}]}
    artifact.write_text(yaml.safe_dump(content), encoding="utf-8")
    return artifact


def test_a_saved_artifact_without_its_weights_is_refused(
    engine: ModuleType, tmp_path: Path
) -> None:
    """Without the weights the pipeline would be scored with a random backbone."""
    artifact = _artifact(tmp_path)

    with pytest.raises(click.UsageError, match=r"pipeline\.pt"):
        engine.load_pipeline(artifact, trained=True)


def test_training_into_a_folder_with_a_trained_pipeline_stops_before_the_fit(
    engine: ModuleType, tmp_path: Path
) -> None:
    """The threshold and scores of the first run would stay next to the new weights."""
    _artifact(tmp_path)

    with pytest.raises(click.UsageError, match="already holds"):
        engine.train(
            None,
            None,
            out_dir=tmp_path,
            epochs=1,
            lr=1e-3,
            seed=42,
            accelerator="cpu",
            val_every=0,
            save_best_val=False,
            limit_train_batches=None,
            limit_val_batches=None,
            run_meta={},
        )


def test_reserving_a_folder_writes_the_run_record_at_once(
    engine: ModuleType, tmp_path: Path
) -> None:
    """The record marks the folder as taken before anything is trained."""
    out = tmp_path / "new" / "run"

    engine.reserve_run_folder(out, {"epochs": 3, "seed": 7})

    assert json.loads((out / "run.json").read_text(encoding="utf-8")) == {"epochs": 3, "seed": 7}
    with pytest.raises(click.UsageError, match="already holds a run"):
        engine.reserve_run_folder(out, {"epochs": 3, "seed": 7})


def test_a_folder_that_another_run_has_claimed_is_refused(
    engine: ModuleType, tmp_path: Path
) -> None:
    """A run that is still training has saved no pipeline yet; its record stops a second one."""
    (tmp_path / "run.json").write_text("{}", encoding="utf-8")

    with pytest.raises(click.UsageError, match="already holds a run"):
        engine.reserve_run_folder(tmp_path, {"epochs": 1})
    assert (tmp_path / "run.json").read_text(encoding="utf-8") == "{}"


def test_training_claims_its_folder_before_the_fit(
    engine: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two runs started into one folder must not both reach the fit and overwrite each other."""
    pipeline = SimpleNamespace(
        name="p",
        nodes=[SimpleNamespace(name=name) for name in engine.REQUIRED_NODES],
        unfreeze_nodes_by_name=lambda names: None,
    )
    seen: dict = {}

    class StopAtFit:
        def __init__(self, **options: object) -> None:
            seen["seed"] = options["training_config"].seed

        def fit(self) -> None:
            seen["record"] = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
            raise RuntimeError("fit started")

    monkeypatch.setattr(engine, "GradientTrainer", StopAtFit)
    monkeypatch.setattr(engine, "PostFitCallback", lambda **_: SimpleNamespace(result=None))
    arguments = {
        "out_dir": tmp_path,
        "epochs": 2,
        "lr": 1e-3,
        "seed": 7,
        "accelerator": "cpu",
        "val_every": 0,
        "save_best_val": False,
        "limit_train_batches": None,
        "limit_val_batches": None,
        "run_meta": {"batch": 4},
    }

    with pytest.raises(RuntimeError, match="fit started"):
        engine.train(pipeline, None, **arguments)

    assert seen["seed"] == 7
    record = seen["record"]
    assert (record["seed"], record["epochs"], record["batch"]) == (7, 2, 4)
    assert record["fit_and_scoring_seconds"] is None
    with pytest.raises(click.UsageError, match="already holds a run"):
        engine.train(pipeline, None, **arguments)


def test_the_same_seed_gives_the_same_initial_weights(engine: ModuleType) -> None:
    def weights(seed: int) -> torch.Tensor:
        engine.seed_run(seed)
        return torch.nn.Linear(4, 3).weight.detach().clone()

    assert torch.equal(weights(7), weights(7))
    assert not torch.equal(weights(7), weights(8))


@pytest.mark.parametrize(("flags", "seed"), [([], 42), (["--seed", "7"], 7)])
def test_train_seeds_before_it_builds_the_pipeline(
    example: Callable[[str], ModuleType],
    engine: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    flags: list[str],
    seed: int,
) -> None:
    """The backbone gets its weights when the config is loaded, so the seed has to come first."""
    files = [tmp_path / name for name in ("universe.csv", "splits.json")]
    for path in files:
        path.write_text("", encoding="utf-8")
    arguments = ["--universe-csv", files[0], "--splits-json", files[1], "--out", tmp_path / "run"]
    order: list[tuple[str, int | None]] = []

    def stop(*_: object, **__: object) -> None:
        raise click.UsageError("stopped before the recordings are opened")

    monkeypatch.setattr(engine, "seed_run", lambda value: order.append(("seed", value)))
    monkeypatch.setattr(engine, "load_pipeline", lambda *_, **__: order.append(("load", None)))
    monkeypatch.setattr(engine, "make_datamodule", stop)

    result = CliRunner().invoke(example("train").main, [str(item) for item in arguments] + flags)

    assert "stopped before the recordings are opened" in result.output
    assert order == [("seed", seed), ("load", None)]


def test_a_stored_threshold_that_the_count_node_does_not_use_is_refused(
    engine: ModuleType, tmp_path: Path
) -> None:
    """The threshold lives in two files; evaluation must not score at one and count at the other."""
    artifact = _artifact(tmp_path)
    engine.store_threshold(artifact, _result("val", {10.0: _row(0.4, 0.5), 20.0: _row(0.8, 0.1)}))
    # As if a later selection had stopped between its two writes.
    engine.set_min_votes(artifact, 50.0)

    with pytest.raises(click.UsageError, match="select_threshold"):
        engine.stored_threshold(artifact)


def test_a_config_without_a_count_node_cannot_take_a_threshold(
    engine: ModuleType, tmp_path: Path
) -> None:
    artifact = tmp_path / "pipeline.yaml"
    artifact.write_text(yaml.safe_dump({"nodes": [{"name": "Backbone"}]}), encoding="utf-8")

    with pytest.raises(click.UsageError, match="Count"):
        engine.set_min_votes(artifact, 20.0)


def test_a_pipeline_without_the_node_names_of_the_example_is_refused(engine: ModuleType) -> None:
    pipeline = SimpleNamespace(
        nodes=[SimpleNamespace(name="Backbone"), SimpleNamespace(name="Count")]
    )

    with pytest.raises(click.UsageError, match="CentreLoss, BorderLoss, ClassLoss, Metrics"):
        engine.nodes_by_name(pipeline)


@pytest.mark.parametrize(
    "error",
    [subprocess.TimeoutExpired(cmd="git", timeout=10), FileNotFoundError("git")],
    ids=["timeout", "no git"],
)
def test_the_commit_is_none_when_git_does_not_answer(
    engine: ModuleType, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """A slow or missing git must not end a run after the fit."""

    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(subprocess, "run", fail)

    assert engine.git_commit() is None


def test_the_commit_stays_the_one_read_first(
    engine: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A commit made while the fit runs does not belong in the files of that run."""
    answers = iter(["aaa", "bbb"])

    def answer(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout=next(answers))

    monkeypatch.setattr(subprocess, "run", answer)

    assert engine.git_commit() == "aaa"
    assert engine.git_commit() == "aaa"


def test_the_chosen_threshold_is_stored_with_the_artifact_and_read_back(
    engine: ModuleType, tmp_path: Path
) -> None:
    artifact = _artifact(tmp_path)
    result = _result("val", {10.0: _row(0.4, 0.5), 20.0: _row(0.8, 0.1), 50.0: _row(0.6, 0.2)})

    chosen = engine.store_threshold(artifact, result)

    assert chosen == 20.0
    assert engine.stored_threshold(artifact) == 20.0
    record = json.loads((tmp_path / "pipeline.threshold.json").read_text(encoding="utf-8"))
    assert record["chosen_on"] == "val"
    assert set(record["scores"]) == {"10", "20", "50"}
    count = yaml.safe_load(artifact.read_text(encoding="utf-8"))["nodes"][0]
    assert count["hparams"]["min_votes"] == 20.0
    # Files are replaced as a whole; nothing half-written stays behind.
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "pipeline.threshold.json",
        "pipeline.yaml",
    ]


def test_two_artifacts_in_one_folder_keep_their_own_threshold(
    engine: ModuleType, tmp_path: Path
) -> None:
    """The result files carry the name of their artifact, not only that of the folder."""
    first = _artifact(tmp_path)
    second = tmp_path / "candidate.yaml"
    second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")

    engine.store_threshold(first, _result("val", {10.0: _row(0.8, 0.1), 50.0: _row(0.2, 0.9)}))
    engine.store_threshold(second, _result("val", {10.0: _row(0.2, 0.9), 50.0: _row(0.8, 0.1)}))

    assert engine.stored_threshold(first) == 10.0
    assert engine.stored_threshold(second) == 50.0


def test_scores_at_a_threshold_the_node_did_not_score_are_refused(
    engine: ModuleType, tmp_path: Path
) -> None:
    artifact = _artifact(tmp_path)
    result = _result("test", {10.0: _row(0.4, 0.5)})

    with pytest.raises(click.UsageError, match="30"):
        engine.store_metrics(artifact, result, 30.0)
    assert not (tmp_path / "pipeline.test_metrics.json").exists()


def test_the_scores_of_a_split_are_written_at_the_stored_threshold(
    engine: ModuleType, tmp_path: Path
) -> None:
    artifact = _artifact(tmp_path)
    result = _result("test", {10.0: _row(0.4, 0.5), 20.0: _row(0.8, 0.1)})

    engine.store_metrics(artifact, result, 20.0)

    record = json.loads((tmp_path / "pipeline.test_metrics.json").read_text(encoding="utf-8"))
    assert record["min_votes"] == 20.0
    assert record["f1"] == 0.8
    assert record["true_count"] == 5.0


@pytest.mark.parametrize("script", ["select_threshold", "evaluate"])
def test_a_batch_limit_of_zero_is_a_usage_error(
    example: Callable[[str], ModuleType], tmp_path: Path, script: str
) -> None:
    """Zero batches would give an empty evaluation with no scores to read."""
    files = [tmp_path / name for name in ("universe.csv", "splits.json", "pipeline.yaml")]
    for path in files:
        path.write_text("", encoding="utf-8")
    arguments = ["--universe-csv", files[0], "--splits-json", files[1], "--artifact", files[2]]

    result = CliRunner().invoke(
        example(script).main, [str(item) for item in arguments] + ["--limit-batches", "0"]
    )

    assert result.exit_code == 2
    assert "--limit-batches" in result.output


@pytest.fixture()
def datamodule_options(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Stand in for the dataloader plugin and record how its datamodule is built."""
    recorded: list[dict] = []
    plugin = ModuleType("cuvis_ai_dataloader.data")
    plugin.MultiCu3sDataModule = lambda **options: recorded.append(options) or "datamodule"
    monkeypatch.setitem(sys.modules, "cuvis_ai_dataloader", ModuleType("cuvis_ai_dataloader"))
    monkeypatch.setitem(sys.modules, "cuvis_ai_dataloader.data", plugin)
    monkeypatch.setattr("cuvis_ai_core.data.splits_io.load_splits", lambda path: f"splits:{path}")
    return recorded


def test_the_frames_are_read_from_the_recordings_on_reader_threads(
    engine: ModuleType, datamodule_options: list[dict], tmp_path: Path
) -> None:
    """Reader threads work inside the training process, so there are no worker processes."""
    universe, splits = tmp_path / "universe.csv", tmp_path / "splits.json"

    module = engine.make_datamodule(
        universe,
        splits,
        batch=4,
        max_open_sessions=2,
        read_threads=6,
        read_ahead=3,
        coherent_batches=True,
        sdk_cuda=True,
        cuda_cubes=False,
    )

    assert module == "datamodule"
    assert datamodule_options == [
        {
            "universe_csv": str(universe),
            "splits": f"splits:{splits}",
            "batch_size": 4,
            "num_workers": 0,
            "max_open_sessions": 2,
            "read_threads": 6,
            "read_ahead": 3,
            "source_coherent_batches": True,
            "sdk_cuda": True,
            "cuda_cubes": False,
        }
    ]


READER_DEFAULTS = {
    "max_open_sessions": 2,
    "read_threads": 4,
    "read_ahead": 4,
    "coherent_batches": True,
}


@pytest.mark.parametrize("script", ["train", "select_threshold", "evaluate"])
@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], READER_DEFAULTS),
        (
            ["--read-threads", "0", "--read-ahead", "0", "--mixed-batches"]
            + ["--max-open-sessions", "5"],
            {"max_open_sessions": 5, "read_threads": 0, "read_ahead": 0, "coherent_batches": False},
        ),
    ],
)
def test_every_script_passes_the_reader_options_on(
    example: Callable[[str], ModuleType],
    engine: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script: str,
    flags: list[str],
    expected: dict,
) -> None:
    """A script that dropped one would read with the slow defaults of the datamodule."""
    files = [tmp_path / name for name in ("universe.csv", "splits.json", "pipeline.yaml")]
    for path in files:
        path.write_text("", encoding="utf-8")
    target = ["--out", tmp_path / "run"] if script == "train" else ["--artifact", files[2]]
    arguments = ["--universe-csv", files[0], "--splits-json", files[1], *target, *flags]
    seen: dict = {}

    def stop(*_: object, **options: object) -> None:
        seen.update(options)
        raise click.UsageError("stopped before the recordings are opened")

    monkeypatch.setattr(engine, "load_pipeline", lambda *_, **__: None)
    monkeypatch.setattr(engine, "stored_threshold", lambda _: 30.0)
    monkeypatch.setattr(engine, "make_datamodule", stop)

    result = CliRunner().invoke(example(script).main, [str(item) for item in arguments])

    assert "stopped before the recordings are opened" in result.output
    assert {name: seen[name] for name in expected} == expected


@pytest.mark.parametrize("script", ["train", "select_threshold", "evaluate"])
@pytest.mark.parametrize(("option", "value"), [("--read-ahead", "9"), ("--read-threads", "17")])
def test_a_reader_option_beyond_the_limit_of_the_datamodule_is_a_usage_error(
    example: Callable[[str], ModuleType], tmp_path: Path, script: str, option: str, value: str
) -> None:
    """The datamodule stops at 8 frames ahead and 16 threads, the threads only at the first read."""
    files = [tmp_path / name for name in ("universe.csv", "splits.json", "pipeline.yaml")]
    for path in files:
        path.write_text("", encoding="utf-8")
    target = ["--out", tmp_path / "run"] if script == "train" else ["--artifact", files[2]]
    arguments = ["--universe-csv", files[0], "--splits-json", files[1], *target]

    result = CliRunner().invoke(
        example(script).main, [str(item) for item in arguments] + [option, value]
    )

    assert result.exit_code == 2
    assert option in result.output


def test_the_run_record_holds_the_reader_options(
    example: Callable[[str], ModuleType],
    engine: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Coherent batches change which frames share a step, so a result names how it was read."""
    files = [tmp_path / name for name in ("universe.csv", "splits.json")]
    for path in files:
        path.write_text("", encoding="utf-8")
    arguments = ["--universe-csv", files[0], "--splits-json", files[1], "--out", tmp_path / "run"]
    seen: dict = {}

    def stop(*_: object, run_meta: dict, **__: object) -> None:
        seen.update(run_meta)
        raise click.UsageError("stopped before the fit")

    monkeypatch.setattr(engine, "load_pipeline", lambda *_, **__: None)
    monkeypatch.setattr(engine, "make_datamodule", lambda *_, **__: None)
    monkeypatch.setattr(engine, "train", stop)

    result = CliRunner().invoke(
        example("train").main, [str(item) for item in arguments] + ["--mixed-batches"]
    )

    assert "stopped before the fit" in result.output
    assert seen["batch"] == 4
    assert seen["reader"] == {
        "max_open_sessions": 2,
        "read_threads": 4,
        "read_ahead": 4,
        "coherent_batches": False,
        "sdk_cuda": True,
        "cuda_cubes": False,
    }


def _scored_node(votes: float) -> CentroidDetectionMetrics:
    """A metric node that saw one frame: one object, one peak with ``votes`` at its centre."""
    node = CentroidDetectionMetrics(thresholds=[10.0, 50.0], match_radius=3.0)
    instances = torch.zeros(1, 9, 9, dtype=torch.int32)
    instances[0, 3:6, 3:6] = 1
    peaks = torch.zeros(1, 9, 9, 1)
    peaks[0, 4, 4, 0] = votes
    node.forward(peaks=peaks, instances=instances, context=Context(stage=ExecutionStage.VAL))
    return node


@pytest.mark.parametrize(("votes", "best"), [(30.0, 1.0), (60.0, 1.0), (5.0, 0.0)])
def test_the_best_f1_over_the_thresholds_is_logged_after_validation(
    engine: ModuleType, votes: float, best: float
) -> None:
    """The epoch to keep is chosen by what the vote map finds, at whatever threshold."""
    logged: dict[str, float] = {}
    module = SimpleNamespace(log=lambda name, value: logged.update({name: value}))

    engine.BestValidationF1(_scored_node(votes)).on_validation_epoch_end(None, module)

    assert logged == {engine.BEST_F1: best}


def test_nothing_is_logged_for_a_validation_pass_without_frames(engine: ModuleType) -> None:
    """A node that saw no batch has no scores; a logged 0 would look like a bad epoch."""
    logged: dict[str, float] = {}
    module = SimpleNamespace(log=lambda name, value: logged.update({name: value}))
    node = CentroidDetectionMetrics(thresholds=[10.0], match_radius=3.0)

    engine.BestValidationF1(node).on_validation_epoch_end(None, module)

    assert logged == {}
