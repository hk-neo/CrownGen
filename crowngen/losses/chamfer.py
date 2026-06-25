"""
Chamfer Distance 손실 함수.

포인트 클라우드 간의 양방향 최근접 이웃 거리를 측정합니다.
CrownGen 평가에서 CD-L1을 기본 메트릭으로 사용합니다.

CD-L1 = (1/|S1|) Σ_{x∈S1} min_{y∈S2} ||x-y||_1
      + (1/|S2|) Σ_{y∈S2} min_{x∈S1} ||y-x||_1
"""

import torch
import torch.nn.functional as F
from typing import Optional


def chamfer_distance_l1(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L1 Chamfer Distance.

    Args:
        pred: (B, N, 3) 예측 포인트 클라우드
        gt: (B, M, 3) 정답 포인트 클라우드
        mask: (B,) 또는 (B, 1) 유효 샘플 마스크 (선택)

    Returns:
        스칼라 CD-L1 값
    """
    # (B, N, M) 거리 행렬
    dist_matrix = torch.cdist(pred, gt, p=1)  # L1 거리

    # 양방향 최솟값
    min_dist_pred = dist_matrix.min(dim=-1)[0]  # (B, N) — 각 pred 포인트의 최근접 gt 거리
    min_dist_gt = dist_matrix.min(dim=-2)[0]    # (B, M) — 각 gt 포인트의 최근접 pred 거리

    # 평균
    cd = min_dist_pred.mean(dim=-1) + min_dist_gt.mean(dim=-1)  # (B,)

    if mask is not None:
        cd = (cd * mask.float()).sum() / mask.float().sum().clamp(min=1)
    else:
        cd = cd.mean()

    return cd


def chamfer_distance_l2(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduce: str = 'mean',
) -> torch.Tensor:
    """L2 Chamfer Distance.

    Args:
        pred: (B, N, 3) 예측 포인트 클라우드
        gt: (B, M, 3) 정답 포인트 클라우드
        mask: (B,) 유효 샘플 마스크 (선택)
        reduce: 'mean' 또는 'sum'

    Returns:
        스칼라 CD-L2 값
    """
    # (B, N, M) 거리 제곱 행렬
    dist_matrix = torch.cdist(pred, gt, p=2) ** 2

    min_dist_pred = dist_matrix.min(dim=-1)[0]  # (B, N)
    min_dist_gt = dist_matrix.min(dim=-2)[0]    # (B, M)

    if reduce == 'mean':
        cd = min_dist_pred.mean(dim=-1) + min_dist_gt.mean(dim=-1)
    else:
        cd = min_dist_pred.sum(dim=-1) + min_dist_gt.sum(dim=-1)

    if mask is not None:
        cd = (cd * mask.float()).sum() / mask.float().sum().clamp(min=1)
    else:
        cd = cd.mean()

    return cd


def chamfer_distance_with_indices(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> tuple:
    """Chamfer Distance + 최근접 이웃 인덱스 반환 (F1 계산용).

    Args:
        pred: (B, N, 3)
        gt: (B, M, 3)

    Returns:
        cd_l1: 스칼라 CD-L1
        pred_to_gt_idx: (B, N) 각 pred 포인트의 최근접 gt 인덱스
        gt_to_pred_idx: (B, M) 각 gt 포인트의 최근접 pred 인덱스
    """
    dist_matrix = torch.cdist(pred, gt, p=1)

    min_dist_pred, pred_to_gt_idx = dist_matrix.min(dim=-1)  # (B, N)
    min_dist_gt, gt_to_pred_idx = dist_matrix.min(dim=-2)    # (B, M)

    cd = min_dist_pred.mean() + min_dist_gt.mean()

    return cd, pred_to_gt_idx, gt_to_pred_idx
