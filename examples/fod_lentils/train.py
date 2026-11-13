"""Train the CentroidNetV2 vector fields on the Industrial FOD lentils set.

    python train.py --universe-csv <lentils>/universe.csv \
        --splits-json <lentils>/splits/adaclip.json --out runs/lentils

``<lentils>`` is the download of ``Industrial_FOD_Lentils`` (or of the 69-frame
``Demo_Industrial_FOD_Lentils``) from ``cuvis_ai_core.data.public_datasets``. The frames
are read from its cu3s recordings, which needs the Cuvis SDK. They are read on threads of
this process (``--read-threads``, ``--read-ahead``), and every batch comes from one
recording (``--coherent-batches``), so the loader does not reopen recordings all the time.

Every step reads ``--batch`` frames and cuts four foreground-biased 192 x 192 crops from
each (set in the pipeline config). The result is ``<out>/pipeline.yaml`` with its weights
in ``pipeline.pt``.

After the fit the framework scores the validation and the test split. The vote threshold
with the highest F1 on validation is stored with the artifact (``pipeline.threshold.json``
and the ``Count`` node), and the test scores at that threshold go to
``pipeline.test_metrics.json``. ``select_threshold.py`` and ``evaluate.py`` repeat the two
steps for a saved artifact.
"""

from __future__ import annotations

from pathlib import Path

import _engine as engine
import click


@click.command(help=__doc__, context_settings={"max_content_width": 100})
@engine.data_options
@click.option(
    "--out",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="New folder for pipeline.yaml, pipeline.pt and run.json; one folder per run.",
)
@click.option(
    "--pipeline",
    "config",
    default=engine.TRAIN_CONFIG,
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Pipeline config to train.",
)
@click.option("--epochs", default=20, show_default=True)
@click.option("--lr", default=1e-3, show_default=True, help="Adam learning rate.")
@click.option(
    "--val-every",
    default=0,
    show_default=True,
    help="Validate every N epochs on full frames; 0 turns validation off.",
)
@click.option(
    "--save-best-val",
    is_flag=True,
    help="Save the epoch with the lowest validation loss; needs --val-every.",
)
@click.option("--accelerator", default="auto", show_default=True)
@click.option(
    "--limit-train-batches",
    type=click.IntRange(min=1),
    help="Stop each epoch after N batches (smoke runs).",
)
@click.option(
    "--limit-val-batches",
    type=click.IntRange(min=1),
    help="Validate on N batches only (smoke runs).",
)
def main(
    universe_csv: Path,
    splits_json: Path,
    batch: int,
    max_open_sessions: int,
    read_threads: int,
    read_ahead: int,
    coherent_batches: bool,
    sdk_cuda: bool,
    cuda_cubes: bool,
    out: Path,
    config: Path,
    epochs: int,
    lr: float,
    val_every: int,
    save_best_val: bool,
    accelerator: str,
    limit_train_batches: int | None,
    limit_val_batches: int | None,
) -> None:
    """Build the pipeline, train it, save it and report its scores."""
    pipeline = engine.load_pipeline(config)
    datamodule = engine.make_datamodule(
        universe_csv,
        splits_json,
        batch=batch,
        max_open_sessions=max_open_sessions,
        read_threads=read_threads,
        read_ahead=read_ahead,
        coherent_batches=coherent_batches,
        sdk_cuda=sdk_cuda,
        cuda_cubes=cuda_cubes,
    )
    artifact, results = engine.train(
        pipeline,
        datamodule,
        out_dir=out,
        epochs=epochs,
        lr=lr,
        accelerator=accelerator,
        val_every=val_every,
        save_best_val=save_best_val,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        run_meta={
            "config": str(config),
            "universe_csv": str(universe_csv),
            "splits_json": str(splits_json),
            "batch": batch,
            "coherent_batches": coherent_batches,
        },
    )

    if "val" not in results:
        click.echo("No validation scores; choose a threshold with select_threshold.py.")
        return
    min_votes = engine.store_threshold(artifact, results["val"])
    if "test" in results:
        engine.store_metrics(artifact, results["test"], min_votes)


if __name__ == "__main__":
    main()
