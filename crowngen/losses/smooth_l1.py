"""
Smooth L1 손실 함수.

Boundary Prediction Module에서 사용하는 손실 함수입니다.
논문 수식:
  smooth_l1(x) = 0.5 * x² if |x| < 1 else |x| - 0.5
  L_bound = (1/|X|) * Σ smooth_l1(B_pred_i - B_gt_i)
"""

import torch
import torch.nn.functional as F
from typing import Optional


def smooth_l1_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Smooth L1 손실 (타겟 치아에만 적용).

    Args:
        pred: (B, T, 5) 예측된 실린더 파라미터
        gt: (B, T, 5) 정답 실린더 파라미터
        mask: (B, T) 타겟 마스크 (1=타겟, 0=컨텍스트)
        reduction: 'mean' 또는 'sum'

    Returns:
        스칼라 손실
    """
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).expand_as(pred)  # (B, T, 5)
        loss = F.smooth_l1_loss(pred * mask_expanded, gt * mask_expanded, reduction='sum')
        n_targets = mask.sum().clamp(min=1)
        return loss / (n_targets * pred.shape[-1])
    else:
        return F.smooth_l1_loss(pred, gt, reduction=reduction)
