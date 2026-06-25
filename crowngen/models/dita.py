"""
Distance-weighted Inter-Tooth Attention (DITA) 레이어.

CrownGen의 핵심 혁신: 치아 간의 형태학적 관계를 거리 기반
상대적 위치 인코딩(RPE)으로 명시적으로 모델링합니다.

논문 수식 (Eq. 13-15):
  r_ij = [log(1+max(Δij,0)), log(1+max(-Δij,0)), 1_{Δij=0}]
  e_ij = (1/√F) · q_i^T · k_j + q_i^T · p^K_ij + p^Q_ij^T · k_j
  α_ij = softmax(e_ij)
  z_i^out = z_i^in + Σ_j α_ij · (v_j + p^V_ij)

DITA는 SA/FP 블록 이후, 그리고 보틀넥에 삽입됩니다 (총 9개 레이어).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..data.fdi import compute_rpe_matrix


class DITA(nn.Module):
    """Distance-weighted Inter-Tooth Attention 레이어.

    Args:
        dim: 치아별 특징 차원
        num_heads: 어텐션 헤드 수
        rpe_hidden: RPE MLP 은닉 차원
        max_teeth: 최대 치아 수
        dropout: 어텐션 드롭아웃
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        rpe_hidden: int = 64,
        max_teeth: int = 28,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/√F

        # Q, K, V 프로젝션
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # 출력 프로젝션
        self.out_proj = nn.Linear(dim, dim)

        # RPE (Relative Positional Encoding) MLP
        # 3차원 RPE 벡터 → Q/K/V 편향 (per head × per head_dim)
        self.rpe_mlp = nn.Sequential(
            nn.Linear(3, rpe_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(rpe_hidden, num_heads * self.head_dim * 3)  # Q, K, V 각각에 대해
        )

        # 사전 계산된 RPE 행렬 등록
        self.register_buffer('rpe_table', compute_rpe_matrix(max_teeth))

        self.dropout = nn.Dropout(dropout)

    def _compute_rpe_biases(
        self,
        tooth_indices: torch.Tensor
    ) -> tuple:
        """치아 인덱스 쌍에 대한 RPE 편향 벡터 계산.

        Args:
            tooth_indices: (B, T) 치아의 지그재그 인덱스

        Returns:
            (p_Q, p_K, p_V): 각각 (B, T, T, num_heads, head_dim)
        """
        B, T = tooth_indices.shape

        # RPE 테이블에서 해당 인덱스 쌍의 벡터 조회
        # rpe_table: (28, 28, 3)
        idx_i = tooth_indices.unsqueeze(2).expand(-1, -1, T)  # (B, T, T)
        idx_j = tooth_indices.unsqueeze(1).expand(-1, T, -1)  # (B, T, T)

        rpe_vectors = self.rpe_table[idx_i, idx_j]  # (B, T, T, 3)

        # MLP로 Q/K/V 편향 생성
        # (B*T*T, 3) → MLP → (B*T*T, num_heads * head_dim * 3)
        BTT = B * T * T
        rpe_input = rpe_vectors.reshape(BTT, 3)
        rpe_output = self.rpe_mlp(rpe_input)  # (BTT, num_heads * head_dim * 3)

        # Q, K, V 편향으로 분리: (B, T, T, num_heads, head_dim) each
        rpe_output = rpe_output.reshape(B, T, T, self.num_heads * 3, self.head_dim)
        p_Q = rpe_output[..., :self.num_heads, :].reshape(B, T, T, self.num_heads, self.head_dim)
        p_K = rpe_output[..., self.num_heads:2*self.num_heads, :].reshape(B, T, T, self.num_heads, self.head_dim)
        p_V = rpe_output[..., 2*self.num_heads:, :].reshape(B, T, T, self.num_heads, self.head_dim)

        return p_Q, p_K, p_V

    def forward(
        self,
        z: torch.Tensor,
        tooth_indices: Optional[torch.Tensor] = None,
        tooth_valid: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DITA 순방향.

        Args:
            z: (B, T, dim) 치아별 특징 벡터 (T=최대 28)
            tooth_indices: (B, T) 치아의 지그재그 인덱스 (없으면 0..T-1)
            tooth_valid: (B, T) 유효 치아 마스크 (1=유효, 0=무효). key 마스킹용.
            key_mask: (B, T) 어텐션 key 로 참여할 치아 마스크 (1=참여, 0=배제).
                None 이면 tooth_valid 사용. Boundary 모듈에서 타겟 치아를 key
                에서 배제해 "컨텍스트 non-zero 치아에만 attend" (논문 명세) 하려
                사용. 모든 치아가 query 로는 남는다 (타겟이 컨텍스트를 읽게 함).

        Returns:
            (B, T, dim) 어텐션 적용된 특징 (잔차 연결 포함)
        """
        B, T, D = z.shape

        if tooth_indices is None:
            tooth_indices = torch.arange(T, device=z.device).unsqueeze(0).expand(B, -1)

        # Q, K, V 프로젝션: (B, T, dim) → (B, T, H, head_dim) → (B, H, T, head_dim)
        Q = self.q_proj(z).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(z).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(z).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q, K, V: (B, H, T, head_dim)

        # 표준 어텐션 스코어: (B, H, T, T)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # RPE 편향 계산
        p_Q, p_K, p_V = self._compute_rpe_biases(tooth_indices)
        # p_Q, p_K, p_V: (B, T, T, H, D) → (B, H, T, T, head_dim) 으로 리셰이프
        p_Q = p_Q.reshape(B, T, T, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        p_K = p_K.reshape(B, T, T, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        p_V = p_V.reshape(B, T, T, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        # RPE가 포함된 어텐션 스코어:
        # e_ij = (1/√F)·q_i·k_j + q_i·p^K_ij + p^Q_ij·k_j
        # q_i·p^K_ij: (B, H, T, 1, head_dim) × (B, H, T, T, head_dim) → (B, H, T, T)
        rpe_k = (Q.unsqueeze(3) * p_K).sum(dim=-1)   # (B, H, T, T)
        rpe_q = (p_Q * K.unsqueeze(2)).sum(dim=-1)    # (B, H, T, T)

        attn_scores = attn_scores + rpe_k + rpe_q

        # 유효하지 않은 치아 마스킹 (key 차원).
        # key_mask 가 주어지면 그것을, 아니면 tooth_valid 를 사용.
        key_valid = key_mask if key_mask is not None else tooth_valid
        if key_valid is not None:
            # (B, T) → (B, 1, 1, T) — key 차원 마스킹
            mask = key_valid.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # 소프트맥스
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, T, T)
        attn_weights = self.dropout(attn_weights)

        # NaN 처리 (모든 값이 -inf인 경우)
        attn_weights = attn_weights.nan_to_num(0.0)

        # 어텐션 출력: (B, H, T, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # RPE가 포함된 값: Σ_j α_ij · (v_j + p^V_ij)
        # p^V: (B, H, T, T, head_dim)
        attn_output = attn_output + (attn_weights.unsqueeze(-1) * p_V).sum(dim=3)

        # (B, H, T, head_dim) → (B, T, dim)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, T, D)
        attn_output = self.out_proj(attn_output)

        # 잔차 연결
        return z + attn_output
