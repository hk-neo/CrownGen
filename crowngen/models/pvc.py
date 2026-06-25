"""
Point-Voxel Convolution (PVC) 연산자 (최적화 v2).

Vectorized Voxelization/Devoxelization — Python for 루프 제거:
  - 좌표 정규화: 벡터화된 min-max
  - Voxelization: scatter_add 배치 차원 처리
  - Devoxelization: 간단한 인덱스 조회

Reference: PVCNN (Liu et al., NeurIPS 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Voxelization(nn.Module):
    """포인트 클라우드를 복셀 그리드로 변환 (벡터화).

    Args:
        resolution: 복셀 그리드 해상도 (예: 32 → 32×32×32)
    """

    def __init__(self, resolution: int):
        super().__init__()
        self.resolution = resolution

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: (B, C, N) 포인트 특징
            coords: (B, 3, N) 정규화된 좌표 [0, 1]

        Returns:
            (B, C, resolution^3) 복셀 특징
        """
        B, C, N = features.shape
        res = self.resolution
        n_voxels = res ** 3

        # 벡터화된 좌표 정규화: (B, 3, N) → [0, 1]
        c_min = coords.amin(dim=-1, keepdim=True)  # (B, 3, 1)
        c_max = coords.amax(dim=-1, keepdim=True)  # (B, 3, 1)
        c_range = c_max - c_min
        # 0-division 방지: range가 작으면 0.5로
        safe_range = c_range.clamp(min=1e-6)
        coords_norm = torch.where(
            c_range > 1e-6,
            (coords - c_min) / safe_range,
            torch.full_like(coords, 0.5)
        )

        # 복셀 인덱스: (B, 3, N) → (B, N)
        voxel_idx = (coords_norm * (res - 1)).long().clamp(0, res - 1)
        linear_idx = voxel_idx[:, 0] * res * res + voxel_idx[:, 1] * res + voxel_idx[:, 2]

        # scatter_add로 특징 집계 (배치별 처리 — scatter_add는 1D 인덱스만 지원)
        output = torch.zeros(B, C, n_voxels, device=features.device, dtype=features.dtype)
        count = torch.zeros(B, n_voxels, device=features.device, dtype=features.dtype)

        # 배치별 scatter_add (C 루프는 벡터화)
        ones = torch.ones(N, device=features.device, dtype=features.dtype)
        for b in range(B):
            count[b].scatter_add_(0, linear_idx[b], ones)
            # C채널 한번에 scatter_add
            idx_expand = linear_idx[b].unsqueeze(0).expand(C, -1)  # (C, N)
            output[b].scatter_add_(1, idx_expand, features[b])

        count = count.clamp(min=1).unsqueeze(1)  # (B, 1, n_voxels)
        output = output / count

        return output  # (B, C, res^3)


class Devoxelization(nn.Module):
    """복셀 특징을 포인트 클라우드로 분배 (벡터화).

    Args:
        resolution: 복셀 그리드 해상도
    """

    def __init__(self, resolution: int):
        super().__init__()
        self.resolution = resolution

    def forward(
        self,
        voxel_features: torch.Tensor,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            voxel_features: (B, C, res^3) 복셀 특징
            coords: (B, 3, N) 정규화된 좌표

        Returns:
            (B, C, N) 포인트 특징
        """
        B, C, _ = voxel_features.shape
        res = self.resolution
        N = coords.shape[2]

        # 벡터화된 좌표 정규화
        c_min = coords.amin(dim=-1, keepdim=True)
        c_max = coords.amax(dim=-1, keepdim=True)
        c_range = c_max - c_min
        safe_range = c_range.clamp(min=1e-6)
        coords_norm = torch.where(
            c_range > 1e-6,
            (coords - c_min) / safe_range,
            torch.full_like(coords, 0.5)
        )

        # 복셀 인덱스 → 직접 조회
        voxel_idx = (coords_norm * (res - 1)).long().clamp(0, res - 1)
        linear_idx = (
            voxel_idx[:, 0] * res * res +
            voxel_idx[:, 1] * res +
            voxel_idx[:, 2]
        )  # (B, N)

        # 배치별 인덱싱
        output = torch.zeros(B, C, N, device=voxel_features.device, dtype=voxel_features.dtype)
        for b in range(B):
            output[b] = voxel_features[b][:, linear_idx[b]]

        return output  # (B, C, N)


class PointVoxelConv(nn.Module):
    """Point-Voxel Convolution 블록.

    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        resolution: 복셀 해상도
        dropout: 드롭아웃 비율
        fusion: 결합 방식 ('add' 또는 'concat')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution: int,
        dropout: float = 0.1,
        fusion: str = 'add'
    ):
        super().__init__()
        self.fusion = fusion

        # 포인트 분기: 포인트 단위 MLP
        self.point_mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 복셀 분기: 복셀 단위 MLP
        self.voxel_mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Voxelize / Devoxelize
        self.voxelize = Voxelization(resolution)
        self.devoxelize = Devoxelization(resolution)

        # Fusion
        if fusion == 'concat':
            self.fusion_layer = nn.Conv1d(out_channels * 2, out_channels, 1)
        elif fusion == 'add':
            self.fusion_layer = None

        self.dropout = nn.Dropout(dropout)

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
            xyz: (B, 3, N) 좌표 (변경 없음)
            out: (B, C_out, N) 출력 특징
        """
        # 포인트 분기
        point_feat = self.point_mlp(features)  # (B, C_out, N)

        # 복셀 분기
        voxel_feat = self.voxelize(features, xyz)   # (B, C_in, res^3)
        voxel_feat = self.voxel_mlp(voxel_feat)      # (B, C_out, res^3)
        voxel_feat = self.devoxelize(voxel_feat, xyz) # (B, C_out, N)

        # 결합
        if self.fusion == 'add':
            out = point_feat + voxel_feat
        elif self.fusion == 'concat':
            out = self.fusion_layer(torch.cat([point_feat, voxel_feat], dim=1))
        else:
            out = point_feat + voxel_feat

        out = self.dropout(out)

        return xyz, out
