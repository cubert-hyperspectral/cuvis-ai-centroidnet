"""Choose the vote threshold of a trained pipeline on the validation split.

    python select_threshold.py --artifact runs/lentils/pipeline.yaml \
        --universe-csv <lentils>/universe.csv --splits-json <lentils>/splits/adaclip.json

Every threshold of the ``Metrics`` node is scored in one pass. The one with the highest
F1 is written to ``pipeline.threshold.json`` next to the artifact and into the ``Count`` node of
``pipeline.yaml``, so the artifact counts with it at inference. ``evaluate.py`` reads the
same file: the test split never takes part in the choice.
"""

from __future__ import annotations

from pathlib import Path

import _engine as engine
import click


@click.command(help=__doc__, context_settings={"max_content_width": 100})
@engine.data_options
@click.option(
    "--artifact",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="pipeline.yaml written by train.py.",
)
@click.option("--device", default=None, help="cuda or cpu; default: cuda when available.")
@click.option(
    "--limit-batches",
    type=click.IntRange(min=1),
    help="Score the first N batches only (smoke runs).",
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
    artifact: Path,
    device: str | None,
    limit_batches: int | None,
) -> None:
    """Score the validation split and store the chosen threshold."""
    pipeline = engine.load_pipeline(
        artifact, device=device or engine.default_device(), trained=True
    )
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
    result = engine.score_split(pipeline, datamodule, "val", limit_batches)
    engine.store_threshold(artifact, result)


if __name__ == "__main__":
    main()
