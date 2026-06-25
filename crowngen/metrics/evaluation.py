"""
CrownGen 평가 메트릭 모듈.

논문에서 사용하는 메트릭:
  1. CD-L1: Chamfer Distance L1 (포인트 클라우드 레벨)
  2. EMD: Earth Mover's Distance (포인트 클라우드 레벨)
  3. F1@τ: F1 Score at threshold τ (τ = 0.3, 0.5, 1.0mm)
  4. ASD: Average Surface Distance (메쉬 레벨)
  5. NC: Normal Consistency (메쉬 레벨)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


def chamfer_distance_l1(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> torch.Tensor:
    """CD-L1: 양방향 최근젠 이웃 L1 거리 평균.

    Args:
        pred: (B, N, 3) 또는 (N, 3)
        gt: (B, M, 3) 또는 (M, 3)

    Returns:
        스칼라 CD-L1
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

    dist = torch.cdist(pred, gt, p=1)  # (B, N, M)
    min_dist_pred = dist.min(dim=-1)[0].mean(dim=-1)  # (B,)
    min_dist_gt = dist.min(dim=-2)[0].mean(dim=-1)    # (B,)
    return (min_dist_pred + min_dist_gt).mean()


def chamfer_distance_l2(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> torch.Tensor:
    """CD-L2: 양방향 최근젠 이웃 L2 거리 평균."""
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

    dist = torch.cdist(pred, gt, p=2) ** 2
    min_dist_pred = dist.min(dim=-1)[0].mean(dim=-1)
    min_dist_gt = dist.min(dim=-2)[0].mean(dim=-1)
    return (min_dist_pred + min_dist_gt).mean()


def earth_mover_distance(
    pred: torch.Tensor,
    gt: torch.Tensor,
    max_iter: int = 100,
) -> torch.Tensor:
    """Earth Mover's Distance (근사).

    정확한 EMD는 O(n³)이므로, 근사 방법 사용.
    논문에서는 optimal transport 기반 EMD 사용.

    간이 구현: 양방향 최근젠 이웃의 평균 거리 (CD와 유사하지만
    Hungarian matching 기반은 아님). 정확한 EMD는
    pytorch3d 또는 external library 필요.

    Args:
        pred: (B, N, 3) 또는 (N, 3)
        gt: (B, M, 3) 또는 (M, 3)

    Returns:
        스칼라 EMD
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

    # 근사 EMD: greedy matching
    B, N, _ = pred.shape
    M = gt.shape[1]

    total_dist = 0.0
    for b in range(B):
        dist = torch.cdist(pred[b:b+1], gt[b:b+1], p=2).squeeze(0)  # (N, M)

        # Greedy assignment
        assigned_pred = set()
        assigned_gt = set()
        dist_sum = 0.0
        assignments = 0

        # 거리 기준 정렬
        flat_dist = dist.flatten()
        sorted_indices = flat_dist.argsort()

        for idx in sorted_indices:
            i = idx.item() // M
            j = idx.item() % M
            if i in assigned_pred or j in assigned_gt:
                continue
            dist_sum += dist[i, j].item()
            assigned_pred.add(i)
            assigned_gt.add(j)
            assignments += 1
            if assignments >= min(N, M):
                break

        total_dist += dist_sum / max(assignments, 1)

    return torch.tensor(total_dist / B)


def f1_score_at_threshold(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float,
) -> Dict[str, float]:
    """F1 Score at given threshold.

    각 pred 포인트에 대해 gt 내 최근접 이웃 거리가 threshold 이내이면
    True Positive로 간주합니다.

    Args:
        pred: (N, 3) 예측 포인트 클라우드
        gt: (M, 3) 정답 포인트 클라우드
        threshold: 거리 임계값 (mm)

    Returns:
        {'precision': float, 'recall': float, 'f1': float}
    """
    dist_matrix = torch.cdist(pred.unsqueeze(0), gt.unsqueeze(0), p=2).squeeze(0)  # (N, M)

    # Precision: pred 포인트 중 threshold 이내에 gt 포인트가 있는 비율
    min_dist_pred = dist_matrix.min(dim=-1)[0]  # (N,)
    precision = (min_dist_pred < threshold).float().mean().item()

    # Recall: gt 포인트 중 threshold 이내에 pred 포인트가 있는 비율
    min_dist_gt = dist_matrix.min(dim=-2)[0]  # (M,)
    recall = (min_dist_gt < threshold).float().mean().item()

    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1}


def average_surface_distance(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
) -> float:
    """Average Surface Distance (ASD).

    메쉬 표면에서 샘플링한 포인트 클라우드 간의 평균 표면 거리.
    논문에서는 메쉬 레벨 평가에 사용.

    Args:
        pred_points: (N, 3) 예측 메쉬에서 샘플링한 포인트
        gt_points: (M, 3) 정답 메쉬에서 샘플링한 포인트

    Returns:
        ASD 값 (mm)
    """
    dist = torch.cdist(pred_points.unsqueeze(0), gt_points.unsqueeze(0), p=2).squeeze(0)

    # 양방향
    dist_pred_to_gt = dist.min(dim=-1)[0]  # (N,)
    dist_gt_to_pred = dist.min(dim=-2)[0]  # (M,)

    asd = (dist_pred_to_gt.mean() + dist_gt_to_pred.mean()).item() / 2.0
    return asd


def normal_consistency(
    pred_points: torch.Tensor,
    pred_normals: torch.Tensor,
    gt_points: torch.Tensor,
    gt_normals: torch.Tensor,
) -> float:
    """Normal Consistency (NC).

    대응점 간 법선 벡터의 코사인 유사도 평균.
    논문에서는 메쉬 레벨 평가에 사용.

    Args:
        pred_points: (N, 3) 예측 포인트
        pred_normals: (N, 3) 예측 법선
        gt_points: (M, 3) 정답 포인트
        gt_normals: (M, 3) 정답 법선

    Returns:
        NC 값 (0~1, 높을수록 좋음)
    """
    dist = torch.cdist(pred_points.unsqueeze(0), gt_points.unsqueeze(0), p=2).squeeze(0)

    # 각 pred 포인트의 최근접 gt 인덱스
    nearest_gt_idx = dist.argmin(dim=-1)  # (N,)

    # 대응하는 법선 추출
    matched_normals = gt_normals[nearest_gt_idx]  # (N, 3)

    # 코사인 유사도
    cos_sim = F.cosine_similarity(pred_normals, matched_normals, dim=-1)  # (N,)

    # 절대값 (법선 방향이 반대일 수 있음)
    nc = cos_sim.abs().mean().item()
    return nc


def compute_all_point_cloud_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    f1_thresholds: List[float] = [0.3, 0.5, 1.0],
) -> Dict[str, float]:
    """포인트 클라우드 레벨 전체 메트릭 계산.

    Args:
        pred: (N, 3) 예측
        gt: (M, 3) 정답
        f1_thresholds: F1 임계값 리스트

    Returns:
        메트릭 딕셔너리
    """
    metrics = {}

    # CD-L1
    metrics['cd_l1'] = chamfer_distance_l1(pred, gt).item()

    # CD-L2
    metrics['cd_l2'] = chamfer_distance_l2(pred, gt).item()

    # EMD (근사)
    metrics['emd'] = earth_mover_distance(pred, gt).item()

    # F1 at thresholds
    for threshold in f1_thresholds:
        f1_result = f1_score_at_threshold(pred, gt, threshold)
        metrics[f'f1@{threshold}'] = f1_result['f1']
        metrics[f'precision@{threshold}'] = f1_result['precision']
        metrics[f'recall@{threshold}'] = f1_result['recall']

    return metrics


def compute_all_mesh_metrics(
    pred_points: torch.Tensor,
    pred_normals: torch.Tensor,
    gt_points: torch.Tensor,
    gt_normals: torch.Tensor,
) -> Dict[str, float]:
    """메쉬 레벨 전체 메트릭 계산.

    Args:
        pred_points: (N, 3) 예측 메쉬 샘플링 포인트
        pred_normals: (N, 3) 예측 법선
        gt_points: (M, 3) 정답 메쉬 샘플링 포인트
        gt_normals: (M, 3) 정답 법선

    Returns:
        메트릭 딕셔너리
    """
    metrics = {}

    # ASD
    metrics['asd'] = average_surface_distance(pred_points, gt_points)

    # NC
    metrics['nc'] = normal_consistency(pred_points, pred_normals, gt_points, gt_normals)

    # CD (참고용)
    metrics['cd_l1'] = chamfer_distance_l1(pred_points, gt_points).item()

    return metrics
