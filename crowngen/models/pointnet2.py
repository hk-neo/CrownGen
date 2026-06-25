"""
PointNet++ 기반 Set Abstraction (SA) 및 Feature Propagation (FP) 모듈.

CrownGen은 PointNet++의 U-Net 인코더-디코더를 PVC 연산자로 대체하여 사용.

최적화 (v3):
  - SA: PointNet(MLP)로 local grouping 처리 + max-pool 후 PVC 적용
    → PVC가 npoint(256)개에만 적용되어 32배 연산 감소
  - Random sampling (학습 시), FPS (추론 시)
  - Flat gather로 메모리 절감
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .pvc import PointVoxelConv


def farthest_point_sample(
    xyz: torch.Tensor,
    npoint: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """최원점 샘플링(FPS) — 추론용."""
    B, _, N = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, :, farthest].view(B, 3, 1)
        dist = torch.sum((xyz - centroid) ** 2, dim=1)
        distance = torch.min(distance, dist)
        farthest = torch.argmax(distance, dim=1)
    centroid_xyz = torch.gather(xyz, 2, centroids.unsqueeze(1).expand(-1, 3, -1))
    return centroids, centroid_xyz


def random_sample(
    xyz: torch.Tensor,
    npoint: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """랜덤 샘플링 — 학습용 (빠름)."""
    B, _, N = xyz.shape
    perm = torch.stack([
        torch.randperm(N, device=xyz.device)[:npoint] for _ in range(B)
    ])
    sampled_xyz = torch.gather(xyz, 2, perm.unsqueeze(1).expand(-1, 3, -1))
    return perm, sampled_xyz


def gather_neighbors(
    source: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """메모리 효율적 neighbor gathering (expand 대신 flat gather)."""
    B, C, N = source.shape
    npoint, nsample = idx.shape[1], idx.shape[2]
    flat_idx = idx.reshape(B, -1)
    gathered = torch.gather(source, 2, flat_idx.unsqueeze(1).expand(-1, C, -1))
    return gathered.reshape(B, C, npoint, nsample)


class SetAbstraction(nn.Module):
    """Set Abstraction 블록 (PointNet + PVC 분리 구조).

    효율적인 2단계 처리:
      1. PointNet(MLP): grouped features (npoint×nsample) → max-pool → (npoint)
      2. PVC: pooled features (npoint)에 적용 → 최종 특징

    Args:
        npoint: 출력 포인트 수
        radius: Ball query 반경
        nsample: 이웃 샘플링 수
        in_channel: 입력 채널 수
        out_channels: 출력 채널 수 리스트
        voxel_res: PVC 복셀 해상도
        dropout: PVC 드롭아웃
        use_fps: True=FPS(추론), False=랜덤(학습)
    """

    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        out_channels: list,
        voxel_res: int,
        dropout: float = 0.1,
        use_fps: bool = False,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.use_fps = use_fps

        ch_in = in_channel + 3  # xyz + features

        # Stage 1: Local PointNet (grouped features → max-pool)
        # Conv1d만 사용 (PVC보다 훨씬 빠름)
        local_layers = []
        for ch_out in out_channels:
            local_layers.append(nn.Conv1d(ch_in, ch_out, 1))
            local_layers.append(nn.BatchNorm1d(ch_out))
            local_layers.append(nn.ReLU(inplace=True))
            ch_in = ch_out
        self.local_net = nn.Sequential(*local_layers)

        # Stage 2: PVC on pooled features (npoint개에만 적용)
        self.pvc = PointVoxelConv(ch_in, ch_in, voxel_res, dropout)

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, 3, N) 포인트 좌표
            features: (B, C_in, N) 포인트 특징

        Returns:
            new_xyz: (B, 3, npoint) 다운샘플링된 좌표
            new_features: (B, C_out, npoint) 집계된 특징
        """
        B, _, N = xyz.shape

        # 대표 포인트 선택
        if self.use_fps:
            _, centroid_xyz = farthest_point_sample(xyz, self.npoint)
        else:
            _, centroid_xyz = random_sample(xyz, self.npoint)

        # kNN 이웃 검색
        dist = torch.cdist(
            centroid_xyz.permute(0, 2, 1),
            xyz.permute(0, 2, 1)
        )
        _, topk_idx = dist.topk(min(self.nsample, N), dim=-1, largest=False)

        # Neighbor gathering (메모리 효율)
        group_xyz = gather_neighbors(xyz, topk_idx)          # (B, 3, np, ns)
        group_xyz_norm = group_xyz - centroid_xyz.unsqueeze(3)

        if features is not None and features.shape[1] > 0:
            group_features = gather_neighbors(features, topk_idx)
            grouped = torch.cat([group_xyz_norm, group_features], dim=1)
        else:
            grouped = group_xyz_norm

        # Stage 1: Local PointNet → max-pool
        B_g, C, npoint, nsample = grouped.shape
        grouped_flat = grouped.reshape(B_g, C, npoint * nsample)
        local_feat = self.local_net(grouped_flat)                    # (B, C_out, np*ns)
        C_out = local_feat.shape[1]
        pooled = local_feat.reshape(B_g, C_out, npoint, nsample)
        pooled = pooled.max(dim=-1)[0]                               # (B, C_out, np)

        # Stage 2: PVC on pooled features
        _, new_features = self.pvc(centroid_xyz, pooled)

        return centroid_xyz, new_features


class FeaturePropagation(nn.Module):
    """Feature Propagation 블록 (PVC 기반).

    Args:
        in_channel: 업샘플링할 특징 채널 수
        skip_channel: 스킵 연결 특징 채널 수
        out_channels: 출력 채널 수 리스트
        voxel_res: PVC 복셀 해상도
        dropout: PVC 드롭아웃
    """

    def __init__(
        self,
        in_channel: int,
        skip_channel: int,
        out_channels: list,
        voxel_res: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        pvc_layers = []
        ch_in = in_channel + skip_channel
        for ch_out in out_channels:
            pvc_layers.append(PointVoxelConv(ch_in, ch_out, voxel_res, dropout))
            ch_in = ch_out
        self.pvc_layers = nn.ModuleList(pvc_layers)

    def forward(
        self,
        xyz: torch.Tensor,
        xyz_src: torch.Tensor,
        features: torch.Tensor,
        skip_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            xyz: (B, 3, N_target) 목표 포인트 좌표
            xyz_src: (B, 3, N_src) 소스 포인트 좌표
            features: (B, C_in, N_src) 업샘플링할 특징
            skip_features: (B, C_skip, N_target) 스킵 연결 특징

        Returns:
            (B, C_out, N_target) 융합된 특징
        """
        B, _, N_target = xyz.shape
        _, _, N_src = xyz_src.shape

        if N_src < N_target:
            dist = torch.cdist(
                xyz.permute(0, 2, 1),
                xyz_src.permute(0, 2, 1)
            )
            k = min(3, N_src)
            dist_topk, idx_topk = dist.topk(k, dim=-1, largest=False)
            weight = 1.0 / (dist_topk + 1e-8)
            weight = weight / weight.sum(dim=-1, keepdim=True)

            C = features.shape[1]
            flat_idx = idx_topk.reshape(B, -1)
            all_feats = torch.gather(
                features, 2, flat_idx.unsqueeze(1).expand(-1, C, -1)
            ).reshape(B, C, N_target, k)
            w = weight.unsqueeze(1)  # (B, 1, N_target, k)
            interpolated = (all_feats * w).sum(dim=-1)
        else:
            interpolated = features

        if skip_features is not None:
            combined = torch.cat([interpolated, skip_features], dim=1)
        else:
            combined = interpolated

        out = combined
        for pvc in self.pvc_layers:
            xyz, out = pvc(xyz, out)

        return out
