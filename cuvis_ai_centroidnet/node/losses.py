"""Loss on a predicted vector field."""

from __future__ import annotations

import math
from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_centroidnet._validation import positive_distance

_VECTORS_SPEC_SHAPE = (-1, -1, -1, 2)


class VectorFieldLoss(Node):
    """Mean Euclidean distance between a predicted and a target vector field.

    The vector loss of CentroidNetV2 (Eq. 6): for every pixel the length of the difference
    between the predicted and the target vector, averaged over the pixels. It is the
    distance itself, not its square, so a few badly wrong vectors do not dominate.

    Both fields are in pixels. The mean is divided by ``max_distance``, which gives the
    value the paper's network sees on vectors normalised to the vote range and keeps the
    loss on the scale of the cross-entropy term it is summed with.

    CentroidNetV2 uses this node twice, once for the centre vectors and once for the
    border vectors, and adds a cross-entropy loss on the class logits (Eq. 7 and 10). The
    trainer sums the loss nodes; no combined node is needed.

    With ``weights`` connected the mean is weighted: a pixel with weight 0 does not count
    and gets no gradient (``InstanceVectorFieldEncoder`` marks objects cut by the image
    edge that way). When every weight is 0 the loss is 0.

    Parameters
    ----------
    max_distance : float
        Vote range in pixels by which the mean distance is divided. Must be positive and
        finite, and equal to the ``max_distance`` of ``CentroidNetHeads`` and
        ``InstanceVectorFieldEncoder`` in the same pipeline.
    weight : float, optional
        Multiplier on the loss, for balancing it against other loss nodes (default 1.0).
    eps : float, optional
        Added to the squared distance under the root (default 1e-12), so the gradient is
        finite where prediction and target are equal. It adds at most ``sqrt(eps)``
        pixels to a distance.

    Raises
    ------
    ValueError
        If ``max_distance`` or ``eps`` is not positive and finite.

    References
    ----------
    K. Dijkstra, J. van de Loosdrecht, W. A. Atsma, L. R. B. Schomaker, M. A. Wiering.
    "CentroidNetV2: A hybrid deep neural network for small-object segmentation and
    counting." Neurocomputing 423 (2021), pages 490 to 505, Section 3.2.2, Eq. 6 and 7.
    https://doi.org/10.1016/j.neucom.2020.10.075
    """

    _category = NodeCategory.LOSS
    _tags = frozenset(
        {
            NodeTag.ONNX,
            NodeTag.TORCH,
            NodeTag.TRAINING,
            NodeTag.DIFFERENTIABLE,
            NodeTag.REGRESSION,
        }
    )

    # A loss runs in train, validation and test, never at inference.
    EXECUTION_STAGES = frozenset({ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST})

    INPUT_SPECS = {
        "predictions": PortSpec(
            dtype=torch.float32,
            shape=_VECTORS_SPEC_SHAPE,
            description="Predicted vectors in pixels [B, H, W, 2]",
        ),
        "targets": PortSpec(
            dtype=torch.float32,
            shape=_VECTORS_SPEC_SHAPE,
            description="Target vectors in pixels [B, H, W, 2]",
        ),
        "weights": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Per-pixel weight of the mean [B, H, W, 1]; 0 leaves a pixel out",
            optional=True,
        ),
    }
    OUTPUT_SPECS = {
        "loss": PortSpec(dtype=torch.float32, shape=(), description="Scalar loss value"),
    }

    def __init__(
        self,
        max_distance: float,
        weight: float = 1.0,
        eps: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        owner = type(self).__name__
        self.max_distance = positive_distance(max_distance, owner)
        self.weight = float(weight)
        self.eps = float(eps)
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError(f"{owner}: eps must be positive and finite; got {eps!r}.")
        super().__init__(max_distance=self.max_distance, weight=self.weight, eps=self.eps, **kwargs)

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        weights: Tensor | None = None,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Return the weighted mean vector distance in units of ``max_distance``."""
        squared = (predictions - targets).square().sum(dim=-1, keepdim=True)
        distance = torch.sqrt(squared + self.eps)
        if weights is None:
            mean = distance.mean()
        else:
            total = weights.sum()
            # All weights zero: the numerator is zero too, so any nonzero divisor gives 0.
            divisor = torch.where(total > 0, total, torch.ones_like(total))
            mean = (distance * weights).sum() / divisor
        return {"loss": mean * (self.weight / self.max_distance)}
