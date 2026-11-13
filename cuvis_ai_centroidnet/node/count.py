"""Count the vote peaks of an image."""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_centroidnet._validation import vote_threshold


class VotePeakCount(Node):
    """Count the vote peaks that reach a threshold, one number per image.

    The input is a vote map after non-maximum suppression (``LocalMaximaSuppression`` of
    cuvis-ai behind ``VectorFieldVoteAccumulator``): zero everywhere except at the local
    maxima, which hold their vote count. A peak is an object centre when it holds at least
    ``min_votes`` votes (Eq. 12 of the paper); the node returns how many there are. It
    needs no labels, so it is the counting output of an inference pipeline.

    A NaN or minus-infinity entry is not a peak.

    Parameters
    ----------
    min_votes : float
        Fewest votes a peak needs to count as an object centre. At least 2: a pixel with a
        zero vector votes for itself, so a frame without objects is a map of ones. Choose
        the value on a validation split; the paper's reference code starts at 30.

    Raises
    ------
    ValueError
        If ``min_votes`` is below 2 or not finite.

    References
    ----------
    K. Dijkstra, J. van de Loosdrecht, W. A. Atsma, L. R. B. Schomaker, M. A. Wiering.
    "CentroidNetV2: A hybrid deep neural network for small-object segmentation and
    counting." Neurocomputing 423 (2021), pages 490 to 505, Section 3.3.1, Eq. 12.
    https://doi.org/10.1016/j.neucom.2020.10.075
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.ONNX, NodeTag.TORCH, NodeTag.POSTPROCESSING, NodeTag.DETECTION})

    INPUT_SPECS = {
        "peaks": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Suppressed vote map [B, H, W, 1]: votes at local maxima, 0 elsewhere",
        ),
    }
    OUTPUT_SPECS = {
        "count": PortSpec(
            dtype=torch.int32,
            shape=(-1,),
            description="Number of peaks with at least min_votes votes, per image [B]",
        ),
    }

    def __init__(self, min_votes: float, **kwargs: Any) -> None:
        self.min_votes = vote_threshold(min_votes, type(self).__name__, "min_votes")
        super().__init__(min_votes=self.min_votes, **kwargs)

    def forward(self, peaks: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the number of peaks at or above ``min_votes`` for every image."""
        is_centre = peaks >= self.min_votes
        return {"count": is_centre.sum(dim=(1, 2, 3)).to(torch.int32)}
