"""Argument checks shared by the nodes."""

from __future__ import annotations

import math


def positive_distance(max_distance: float, owner: str) -> float:
    """Return ``max_distance`` as a float, or raise when it is not positive and finite.

    The vote range scales the vector planes in ``CentroidNetHeads``, clips the targets in
    ``InstanceVectorFieldEncoder`` and normalises ``VectorFieldLoss``. Zero would switch the
    predictions off and divide the loss by zero, so there is no "no clipping" value.
    """
    value = float(max_distance)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{owner}: max_distance must be a positive, finite number of pixels; "
            f"got {max_distance!r}."
        )
    return value


# A pixel whose vector is zero votes for itself, so every background pixel of a vote map
# holds one vote. A threshold below two would count each of them as an object.
MIN_VOTE_THRESHOLD = 2.0


def vote_threshold(threshold: float, owner: str, name: str) -> float:
    """Return a vote threshold as a float, or raise when it is below two or not finite."""
    value = float(threshold)
    if not math.isfinite(value) or value < MIN_VOTE_THRESHOLD:
        raise ValueError(
            f"{owner}: {name} must be a finite number of at least 2 votes, because a pixel "
            f"with a zero vector votes for itself; got {threshold!r}."
        )
    return value
