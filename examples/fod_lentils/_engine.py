"""Shared code of the FOD lentils example: data, training, scoring a split.

The graph is the packaged config ``centroidnet_fod_lentils_train.yaml``. Training uses the
framework's ``GradientTrainer``, which scores the validation and the test split once the
fit is done; a saved pipeline is scored again with the framework's ``evaluate_split``. The
three CLIs next to this file (``train.py``, ``select_threshold.py``, ``evaluate.py``) are
thin fronts.
"""

from __future__ import annotations

import functools
import itertools
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import torch
import yaml
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import GradientTrainer
from cuvis_ai_core.training.config import PipelineMetadata
from cuvis_ai_core.training.evaluation import evaluate_split
from cuvis_ai_core.training.post_fit import EvaluationStatus, PostFitCallback
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.training.config import TrainingConfig
from cuvis_ai_schemas.training.optimizer import OptimizerConfig
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from cuvis_ai_centroidnet.configs import PIPELINE_DIR
from cuvis_ai_centroidnet.selection import best_threshold, scores_by_threshold

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "plugins.yaml"
TRAIN_CONFIG = PIPELINE_DIR / "centroidnet_fod_lentils_train.yaml"
LOSS_NODES = ("CentreLoss", "BorderLoss", "ClassLoss")
THRESHOLD_FILE = "threshold.json"


def data_options(command: Callable[..., Any]) -> Callable[..., Any]:
    """Add the options every CLI of the example needs to find and read the frames."""
    options = [
        click.option(
            "--universe-csv",
            required=True,
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            help="universe.csv of the dataset download: one row per frame.",
        ),
        click.option(
            "--splits-json",
            required=True,
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            help="Split file of the dataset download, for example splits/adaclip.json.",
        ),
        click.option("--batch", default=4, show_default=True, help="Frames per batch."),
        click.option(
            "--max-open-sessions",
            default=3,
            show_default=True,
            help="cu3s recordings kept open per loader; each holds GPU memory with --sdk-cuda.",
        ),
        click.option(
            "--sdk-cuda/--no-sdk-cuda",
            default=True,
            show_default=True,
            help="Process the cubes with the Cuvis SDK on the GPU, or on the CPU (slower).",
        ),
        click.option("--cuda-cubes", is_flag=True, help="Keep the cubes on the GPU after reading."),
    ]
    return functools.reduce(lambda wrapped, option: option(wrapped), reversed(options), command)


def make_datamodule(
    universe_csv: Path,
    splits_json: Path,
    *,
    batch: int,
    max_open_sessions: int,
    sdk_cuda: bool,
    cuda_cubes: bool,
) -> Any:
    """Return the multi-file cu3s datamodule of the dataset.

    The frames are read straight from the recordings; the batch ``mask`` holds the class
    ids of the COCO annotations. ``num_workers`` stays 0: the Cuvis SDK reads in this
    process.
    """
    from cuvis_ai_core.data.splits_io import load_splits
    from cuvis_ai_dataloader.data import MultiCu3sDataModule

    return MultiCu3sDataModule(
        universe_csv=str(universe_csv),
        splits=load_splits(str(splits_json)),
        batch_size=batch,
        num_workers=0,
        max_open_sessions=max_open_sessions,
        sdk_cuda=sdk_cuda,
        cuda_cubes=cuda_cubes,
    )


def registry() -> NodeRegistry:
    """Return a registry that knows this plugin; the other node classes load by path."""
    nodes = NodeRegistry()
    nodes.register_plugin(str(MANIFEST))
    return nodes


def load_pipeline(config: Path, *, device: str | None = None) -> CuvisPipeline:
    """Build a pipeline from a config, with the weights next to it when they exist."""
    weights = config.with_suffix(".pt")
    return CuvisPipeline.load_pipeline(
        str(config),
        weights_path=str(weights) if weights.exists() else None,
        device=device,
        node_registry=registry(),
    )


def default_device() -> str:
    """Return ``cuda`` when a GPU is visible, else ``cpu``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def git_commit() -> str | None:
    """Return the commit of this checkout, or ``None`` outside a git checkout."""
    try:
        done = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError:
        return None
    return done.stdout.strip() or None


class EpochLog(Callback):
    """Print one line per training epoch: duration and the logged values."""

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        """Remember when the epoch started."""
        self._start = time.monotonic()

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Print the epoch's duration and metrics."""
        seconds = time.monotonic() - self._start
        values = {name: round(float(value), 4) for name, value in trainer.callback_metrics.items()}
        stamp = time.strftime("%H:%M:%S")
        click.echo(f"[{stamp}] epoch {trainer.current_epoch:>3} | {seconds:7.1f} s | {values}")


def train(
    pipeline: CuvisPipeline,
    datamodule: Any,
    *,
    out_dir: Path,
    epochs: int,
    lr: float,
    accelerator: str,
    val_every: int,
    save_best_val: bool,
    limit_train_batches: int | None,
    limit_val_batches: int | None,
    run_meta: dict[str, Any],
) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Train the backbone against the three losses, save the pipeline, return its scores.

    Returns the path of the saved ``pipeline.yaml`` (weights in ``pipeline.pt`` next to
    it, settings in ``run.json``) and the detection scores of the validation and the test
    split, in the form ``score_split`` returns. The framework scores both splits after
    every fit, on the weights that are saved: with ``save_best_val`` those of the epoch
    with the lowest validation loss, otherwise those of the last epoch.

    Validation during training runs on full frames in tiles, which is slow: it is off
    unless ``val_every`` is positive.
    """
    if save_best_val and val_every <= 0:
        raise click.UsageError("--save-best-val needs --val-every N, so that validation runs.")
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = {node.name: node for node in pipeline.nodes}
    pipeline.unfreeze_nodes_by_name(["Backbone"])

    post_fit = PostFitCallback(metric_nodes=[nodes["Metrics"]])
    callbacks: list[Callback] = [EpochLog(), post_fit]
    if save_best_val:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(out_dir / "checkpoints"), monitor="val_loss", mode="min", save_top_k=1
            )
        )
    trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=datamodule,
        loss_nodes=[nodes[name] for name in LOSS_NODES],
        metric_nodes=[nodes["Metrics"]] if val_every > 0 else None,
        training_config=TrainingConfig(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=1,
            enable_progress_bar=False,
            log_every_n_steps=1,
            enable_checkpointing=save_best_val,
            num_sanity_val_steps=0,
            # Without validation: point past the last epoch, so that it never runs.
            check_val_every_n_epoch=val_every if val_every > 0 else epochs + 1,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            optimizer=OptimizerConfig(name="adam", lr=lr),
        ),
        callbacks=callbacks,
    )
    start = time.monotonic()
    trainer.fit()
    seconds = time.monotonic() - start

    artifact = out_dir / "pipeline.yaml"
    pipeline.save_to_file(
        str(artifact),
        save_weights=True,
        metadata=PipelineMetadata(
            name=pipeline.name,
            description="CentroidNetV2 vector fields trained on Industrial FOD lentils",
        ),
    )
    run = {
        "artifact": str(artifact),
        "git_commit": git_commit(),
        "fit_and_scoring_seconds": round(seconds, 1),
        "epochs": epochs,
        "lr": lr,
        **run_meta,
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    click.echo(f"saved {artifact} and {artifact.with_suffix('.pt')}")

    results = {}
    evaluations = post_fit.result.evaluations if post_fit.result is not None else {}
    for split, outcome in evaluations.items():
        if outcome.status is EvaluationStatus.EVALUATED:
            results[split] = _split_result(split, outcome.frames, outcome.metrics, nodes["Metrics"])
    return artifact, results


def _split_result(split: str, frames: int, metrics: dict[str, float], node: Any) -> dict[str, Any]:
    """Regroup the pooled values of the ``Metrics`` node for one split."""
    prefix = f"{node.name}/"
    return {
        "split": split,
        "frames": frames,
        "true_count": metrics[f"{prefix}true_count"],
        "scores": scores_by_threshold(metrics, node.thresholds, prefix=prefix),
    }


def score_split(
    pipeline: CuvisPipeline, datamodule: Any, split: str, limit_batches: int | None = None
) -> dict[str, Any]:
    """Run one split through the pipeline and return the detection scores.

    The result holds ``frames``, ``true_count`` and ``scores``: per vote threshold of the
    ``Metrics`` node the values that node reports, pooled over the split.
    ``limit_batches`` stops after that many batches, for a smoke run.
    """
    datamodule.setup(stage=None)
    loader, stage = {
        "val": (datamodule.val_dataloader, ExecutionStage.VAL),
        "test": (datamodule.test_dataloader, ExecutionStage.TEST),
    }[split]
    node = {n.name: n for n in pipeline.nodes}["Metrics"]
    batches = itertools.islice(loader(), limit_batches)
    summary = evaluate_split(pipeline, batches, stage=stage, metric_nodes=[node])
    return _split_result(split, summary.frames, summary.metrics, node)


def print_scores(result: dict[str, Any], only: float | None = None) -> None:
    """Print the scores of a split as a table, one row per vote threshold."""
    click.echo(f"{result['split']}: {result['frames']} frames, {result['true_count']:.0f} objects")
    click.echo(
        f"{'votes':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'prec':>6} {'recall':>6} {'F1':>6} "
        f"{'count':>7} {'|err|/frame':>11}"
    )
    for threshold, row in result["scores"].items():
        if only is not None and threshold != only:
            continue
        click.echo(
            f"{threshold:>8g} {row['true_positives']:>6.0f} {row['false_positives']:>6.0f} "
            f"{row['false_negatives']:>6.0f} {row['precision']:>6.3f} {row['recall']:>6.3f} "
            f"{row['f1']:>6.3f} {row['predicted_count']:>7.0f} {row['count_error']:>11.3f}"
        )


def set_min_votes(config: Path, min_votes: float) -> None:
    """Write the chosen vote threshold into the ``Count`` node of a saved pipeline config."""
    content = yaml.safe_load(config.read_text(encoding="utf-8"))
    count = next(node for node in content["nodes"] if node["name"] == "Count")
    count.setdefault("hparams", {})["min_votes"] = float(min_votes)
    config.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")


def store_threshold(artifact: Path, result: dict[str, Any]) -> float:
    """Choose the vote threshold from the scores of a split and store it with the artifact.

    The threshold with the highest F1 goes into ``threshold.json`` next to the artifact
    and into the ``Count`` node of the pipeline config, so the artifact counts with it at
    inference. Returns the chosen threshold.
    """
    print_scores(result)
    chosen = best_threshold(result["scores"])
    if result["scores"][chosen]["f1"] == 0.0:
        click.echo("No threshold matched a single object; the choice below means nothing.")
    set_min_votes(artifact, chosen)
    record = {
        "min_votes": chosen,
        "chosen_on": result["split"],
        "frames": result["frames"],
        "git_commit": git_commit(),
        "scores": {f"{threshold:g}": row for threshold, row in result["scores"].items()},
    }
    target = artifact.with_name(THRESHOLD_FILE)
    target.write_text(json.dumps(record, indent=2), encoding="utf-8")
    click.echo(f"chosen: {chosen:g} votes; written to {target} and to Count.min_votes")
    return chosen


def stored_threshold(artifact: Path) -> float:
    """Return the vote threshold stored next to an artifact."""
    source = artifact.with_name(THRESHOLD_FILE)
    if not source.exists():
        raise click.UsageError(f"{source} is missing; run select_threshold.py first.")
    return float(json.loads(source.read_text(encoding="utf-8"))["min_votes"])


def store_metrics(artifact: Path, result: dict[str, Any], min_votes: float) -> None:
    """Print the scores of a split at one threshold and write them next to the artifact."""
    print_scores(result, only=min_votes)
    record = {
        "split": result["split"],
        "min_votes": min_votes,
        "frames": result["frames"],
        "true_count": result["true_count"],
        "git_commit": git_commit(),
        **result["scores"][min_votes],
    }
    target = artifact.with_name(f"{result['split']}_metrics.json")
    target.write_text(json.dumps(record, indent=2), encoding="utf-8")
    click.echo(f"written to {target}")
