"""Score a trained pipeline on the test split at the threshold chosen on validation.

    python evaluate.py --artifact runs/lentils/pipeline.yaml \
        --universe-csv <lentils>/universe.csv --splits-json <lentils>/splits/adaclip.json

Reads the vote threshold from ``pipeline.threshold.json`` next to the artifact (written by
``train.py`` or ``select_threshold.py``) and reports true and false positives, false negatives, precision,
recall, F1, the predicted and the true count and the mean absolute count error per frame.
The numbers are also written to ``pipeline.<split>_metrics.json``.
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
@click.option("--split", default="test", show_default=True, type=click.Choice(["val", "test"]))
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
    sdk_cuda: bool,
    cuda_cubes: bool,
    artifact: Path,
    split: str,
    device: str | None,
    limit_batches: int | None,
) -> None:
    """Score one split at the stored threshold."""
    min_votes = engine.stored_threshold(artifact)

    pipeline = engine.load_pipeline(
        artifact, device=device or engine.default_device(), trained=True
    )
    datamodule = engine.make_datamodule(
        universe_csv,
        splits_json,
        batch=batch,
        max_open_sessions=max_open_sessions,
        sdk_cuda=sdk_cuda,
        cuda_cubes=cuda_cubes,
    )
    result = engine.score_split(pipeline, datamodule, split, limit_batches)
    engine.store_metrics(artifact, result, min_votes)


if __name__ == "__main__":
    main()
