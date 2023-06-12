import torch
import torch.nn as nn

from mmrotate.models.builder import ROTATED_LOSSES
from mmdet.models.losses.utils import weighted_loss
from typing import Optional
from torch import Tensor

@weighted_loss
def spatial_smoothl1_loss(pred: Tensor, target: Tensor, beta: float = 1.0, beta_horizontal: float = 1.0) -> Tensor:
    """Smooth L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()

    # Horizontal part
    pred_horizontal = pred[:, 0:4]
    target_horizontal = target[:, 0:4]
    
    diff_horizonal = torch.abs(pred_horizontal - target_horizontal)
    loss_horizonal = torch.where(diff_horizonal < beta_horizontal, 0.5 * diff_horizonal * diff_horizonal / beta_horizontal,
                       diff_horizonal - 0.5 * beta_horizontal)
    
    # Oriented part
    pred_oriented = pred[:,4]
    target_oriented = target[:, 4]

    diff_oriented = torch.abs(pred_oriented - target_oriented)
    loss_oriented = torch.where(diff_oriented < beta, 0.5 * diff_oriented * diff_oriented / beta,
                       diff_oriented - 0.5 * beta).unsqueeze(1)
    
    # Concat
    loss = torch.concat((loss_horizonal, loss_oriented), 1)

    return loss

@ROTATED_LOSSES.register_module
class SpatialSmoothL1Loss(nn.Module):

    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 beta: float = 1.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.beta_horizontal = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * spatial_smoothl1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            beta_horizontal=self.beta_horizontal,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox