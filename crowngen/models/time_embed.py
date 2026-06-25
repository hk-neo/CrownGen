"""
Sinusoidal Timestep 임베딩.

DDPM에서 확산 타임스텝 t를 고차원 벡터로 변환하여
각 SA/FP 블록에 주입합니다.

표준 sinusoidal positional encoding + MLP.
"""

import torch
import torch.nn as nn
import math


class TimestepEmbedding(nn.Module):
    """확산 타임스텝 t를 임베딩 벡터로 변환.

    Args:
        embed_dim: 출력 임베딩 차원 (기본 128)
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        # Sinusoidal → MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def _sinusoidal_encoding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal 위치 인코딩.

        Args:
            t: (B,) 정수 타임스텝 [0, T)

        Returns:
            (B, embed_dim) sinusoidal 인코딩
        """
        device = t.device
        half_dim = self.embed_dim // 2

        # 주파수 계산
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)

        # t를 float로 변환하고 주파수와 결합
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (B, embed_dim)

        # 홀수 차원인 경우 0 패딩
        if self.embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=device)], dim=1)

        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """타임스텝 임베딩.

        Args:
            t: (B,) 정수 타임스텝 또는 (B, 1) 실수 타임스텝

        Returns:
            (B, embed_dim) 임베딩 벡터
        """
        if t.dim() == 1:
            t = t.long()
        else:
            t = t.squeeze(-1).long()

        x = self._sinusoidal_encoding(t)
        return self.mlp(x)


class TimeConditioning(nn.Module):
    """타임스텝 임베딩을 특징 맵에 주입 (FiLM 스타일).

    time_emb → MLP → scale, shift → features * scale + shift

    Args:
        time_dim: 타임 임베딩 차원
        feature_dim: 특징 채널 수
    """

    def __init__(self, time_dim: int, feature_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, feature_dim * 2),
        )
        self.feature_dim = feature_dim

    def forward(
        self,
        features: torch.Tensor,
        time_emb: torch.Tensor
    ) -> torch.Tensor:
        """특징에 타임 조건부 변환 적용.

        Args:
            features: (B, C, N) 특징 맵
            time_emb: (B, time_dim) 타임 임베딩

        Returns:
            (B, C, N) 조건부 변환된 특징
        """
        params = self.time_mlp(time_emb)  # (B, 2*C)
        scale, shift = params.chunk(2, dim=1)  # 각각 (B, C)
        scale = scale.unsqueeze(-1)  # (B, C, 1)
        shift = shift.unsqueeze(-1)  # (B, C, 1)
        return features * (1 + scale) + shift
