"""
FDI (Fédération Dentaire Internationale) 치아 번호 체계 유틸리티.

CrownGen은 지그재그(zig-zag) FDI 순서를 사용하여 치아 간 거리를
상대적 위치 인코딩(RPE)으로 변환합니다.

지그재그 순서: 상악/하악, 좌/우를 번갈아 배치
  17, 47, 16, 46, 15, 45, 14, 44, 13, 43, 12, 42, 11, 41,
  21, 31, 22, 32, 23, 33, 24, 34, 25, 35, 26, 36, 27, 37
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple

# ──────────────────────────────────────────────────
# 지그재그 FDI 순서 (CrownGen 논문 Figure 7c)
# 상악/하악 교차, 후방→전방→후방 순서
# ──────────────────────────────────────────────────
ZIGZAG_FDI_ORDER = [
    17, 47, 16, 46, 15, 45, 14, 44, 13, 43, 12, 42, 11, 41,
    21, 31, 22, 32, 23, 33, 24, 34, 25, 35, 26, 36, 27, 37
]

# 28개 영구 치아 (제3대구치 제외)
ALL_28_FDI = list(range(11, 18)) + list(range(21, 28)) + \
              list(range(31, 38)) + list(range(41, 48))

# FDI → 지그재그 인덱스 매핑
FDI_TO_ZIGZAG = {fdi: idx for idx, fdi in enumerate(ZIGZAG_FDI_ORDER)}
ZIGZAG_TO_FDI = {idx: fdi for idx, fdi in enumerate(ZIGZAG_FDI_ORDER)}

# ──────────────────────────────────────────────────
# 좌우 반전(Mirror) FDI 매핑 — 데이터 증강용
# 좌측 치아 ↔ 우측 치아 (십자가 중심)
# 예: 11 ↔ 21, 16 ↔ 26, 31 ↔ 41, 36 ↔ 46
# ──────────────────────────────────────────────────
MIRROR_REMAP = {}
for fdi in ALL_28_FDI:
    quadrant = fdi // 10
    tooth = fdi % 10
    if quadrant == 1:
        MIRROR_REMAP[fdi] = 20 + tooth   # 1x → 2x
    elif quadrant == 2:
        MIRROR_REMAP[fdi] = 10 + tooth   # 2x → 1x
    elif quadrant == 3:
        MIRROR_REMAP[fdi] = 40 + tooth   # 3x → 4x
    elif quadrant == 4:
        MIRROR_REMAP[fdi] = 30 + tooth   # 4x → 3x

# 치아 기능 그룹
INCISORS = [11, 12, 21, 22, 31, 32, 41, 42]
CANINES = [13, 23, 33, 43]
PREMOLARS = [14, 15, 24, 25, 34, 35, 44, 45]
MOLARS = [16, 17, 26, 27, 36, 37, 46, 47]
UPPER_TEETH = list(range(11, 18)) + list(range(21, 28))
LOWER_TEETH = list(range(31, 38)) + list(range(41, 48))


def fdi_to_zigzag_index(fdi_labels: torch.Tensor) -> torch.Tensor:
    """FDI 라벨을 지그재그 인덱스로 변환.

    Args:
        fdi_labels: (..., ) 정수 텐서, FDI 치아 번호

    Returns:
        (..., ) 정수 텐서, 지그재그 인덱스 (0-27)
    """
    mapping = torch.zeros(48, dtype=torch.long)
    for fdi, idx in FDI_TO_ZIGZAG.items():
        mapping[fdi] = idx
    return mapping[fdi_labels.clamp(0, 47).long()]


def zigzag_index_to_fdi(indices: torch.Tensor) -> torch.Tensor:
    """지그재그 인덱스를 FDI 라벨로 변환."""
    mapping = torch.tensor(ZIGZAG_FDI_ORDER, dtype=torch.long)
    return mapping[indices.clamp(0, 27).long()]


def compute_rpe_matrix(max_teeth: int = 28) -> torch.Tensor:
    """모든 치아 쌍에 대한 상대적 위치 인코딩(RPE) 행렬을 사전 계산.

    CrownGen DITA의 핵심: 지그재그 인덱스 차이 Δij를 3차원 벡터로 변환.

    r_ij = [log(1 + max(Δij, 0)),
            log(1 + max(-Δij, 0)),
            1_{Δij=0}]

    Args:
        max_teeth: 최대 치아 수 (기본 28)

    Returns:
        (max_teeth, max_teeth, 3) RPE 벡터 행렬
    """
    indices = torch.arange(max_teeth)
    delta = indices[:, None] - indices[None, :]  # (28, 28)

    rpe = torch.stack([
        torch.log1p(delta.clamp(min=0).float()),      # 양수 거리
        torch.log1p((-delta).clamp(min=0).float()),    # 음수 거리
        (delta == 0).float()                            # 자기 자신
    ], dim=-1)

    return rpe  # (28, 28, 3)


class FDIEmbedding(nn.Module):
    """FDI 치아 번호를 고차원 임베딩으로 변환.

    CrownGen 논문: "8-dimensional embedding of the tooth's unique FDI identifier"

    Args:
        embed_dim: 임베딩 차원 (논문: 8)
        max_teeth: 최대 치아 수 (기본 28)
    """

    def __init__(self, embed_dim: int = 8, max_teeth: int = 28):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_teeth = max_teeth
        # 학습 가능한 임베딩 테이블 (28개 치아 × embed_dim)
        self.embedding = nn.Embedding(max_teeth, embed_dim)
        # FDI → 인덱스 매핑 (고정, 학습 불가)
        self.register_buffer(
            'fdi_to_idx',
            torch.zeros(48, dtype=torch.long)
        )
        for fdi, idx in FDI_TO_ZIGZAG.items():
            self.fdi_to_idx[fdi] = idx

    def forward(self, fdi_labels: torch.Tensor) -> torch.Tensor:
        """FDI 라벨을 임베딩 벡터로 변환.

        Args:
            fdi_labels: (..., ) 정수 텐서

        Returns:
            (..., embed_dim) 임베딩 벡터
        """
        indices = self.fdi_to_idx[fdi_labels.clamp(0, 47).long()]
        return self.embedding(indices)


def get_functional_group(fdi: int) -> str:
    """FDI 번호로부터 치아 기능 그룹 반환."""
    tooth = fdi % 10
    if tooth in [1, 2]:
        return 'incisor'
    elif tooth == 3:
        return 'canine'
    elif tooth in [4, 5]:
        return 'premolar'
    elif tooth in [6, 7]:
        return 'molar'
    return 'unknown'
