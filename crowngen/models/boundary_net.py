"""
Boundary Prediction Module.

컨텍스트 치아를 분석하여 각 결손치의 원통형 경계 파라미터를 예측합니다.
  B = (cx, cy, cz, r, h) — 5개 스칼라

아키텍처: 메인 디노이징 네트워크의 인코더(3 SA 블록) + 회귀 헤드
- 입력: 512 포인트/치아, 타겟 치아는 영벡터
- 손실: Smooth L1 (타겟 치아에만 적용)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .pointnet2 import SetAbstraction
from .dita import DITA
from .time_embed import TimestepEmbedding
from ..data.fdi import FDIEmbedding


class BoundaryPredictor(nn.Module):
    """Boundary Prediction Module.

    Args:
        config: 모델 설정 (boundary 섹션)
    """

    def __init__(self, config: dict):
        super().__init__()

        model_cfg = config.get('model', config)
        bnd_cfg = model_cfg.get('boundary', model_cfg)
        max_teeth = config.get('data', config).get('max_teeth', 28)

        sa_channels = bnd_cfg.get('sa_channels', [[64, 64], [128, 128], [256, 256]])
        sa_npoints = bnd_cfg.get('sa_npoints', [256, 64, 16])
        sa_voxel_res = bnd_cfg.get('sa_voxel_res', [16, 8, 4])
        sa_radius = bnd_cfg.get('sa_radius', [0.3, 0.4, 0.5])
        pvc_dropout = bnd_cfg.get('pvc_dropout', 0.3)
        dita_heads = config.get('model', config).get('dita_heads', 8)
        dita_rpe_hidden = config.get('model', config).get('dita_rpe_hidden', 64)
        output_dim = bnd_cfg.get('output_dim', 5)
        self.max_teeth = max_teeth

        # ── 입력 프로젝션 ──
        # 입력: xyz(3) + FDI 임베딩(8) = 11차원 (타겟은 영벡터)
        self.input_proj = nn.Conv1d(3 + 8, sa_channels[0][0], 1)

        # FDI 학습 가능 임베딩 (denoise_net과 동일 모듈 사용)
        self.fdi_embedding = FDIEmbedding(embed_dim=8, max_teeth=max_teeth)

        # ── SA 블록 + DITA ──
        self.sa_blocks = nn.ModuleList()
        self.dita_layers = nn.ModuleList()
        self.pos_proj = nn.ModuleList()

        in_ch = sa_channels[0][0]
        for i in range(len(sa_channels)):
            self.sa_blocks.append(
                SetAbstraction(
                    npoint=sa_npoints[i],
                    radius=sa_radius[i],
                    nsample=32,
                    in_channel=in_ch,
                    out_channels=sa_channels[i],
                    voxel_res=sa_voxel_res[i],
                    dropout=pvc_dropout,
                )
            )
            out_ch = sa_channels[i][-1]
            self.dita_layers.append(
                DITA(dim=out_ch, num_heads=dita_heads, rpe_hidden=dita_rpe_hidden)
            )
            # 치아별 centroid(절대 위치) → DITA 설명자 차원으로 주입.
            # PVC 복셀화가 치아별 min-max 정규화(pvc.py)로 절대 위치를 잃기 때문에,
            # 컨텍스트 치아의 아치 내 위치를 DITA가 직접 보게 한다. 그래야 타겟이
            # "빈 자리" 위치를 컨텍스트 기하에서 유추할 수 있다.
            self.pos_proj.append(nn.Linear(3, out_ch))
            in_ch = out_ch

        # ── 글로벌 풀링 ──
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # ── 회귀 헤드 ──
        last_ch = sa_channels[-1][-1]
        self.regressor = nn.Sequential(
            nn.Linear(last_ch, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
        )

    def forward(
        self,
        tooth_points: torch.Tensor,
        fdi_labels: torch.Tensor,
        tooth_valid: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Boundary 예측 순방향.

        Args:
            tooth_points: (B, 28, N, 3) 치아 포인트 클라우드
                          타겟 치아는 0으로 채워짐
            fdi_labels: (B, 28) FDI 치아 번호
            tooth_valid: (B, 28) 유효 치아 마스크
            target_mask: (B, 28) 타겟 마스크

        Returns:
            (B, 28, 5) 예측된 실린더 파라미터
        """
        B, T, N, C = tooth_points.shape
        device = tooth_points.device

        # ── FDI 학습 가능 임베딩 특징: (B, 28, 8) → (B, 28, N, 8) ──
        fdi_feat = self.fdi_embedding(fdi_labels).unsqueeze(2).expand(B, T, N, -1)

        # ── 입력 특징: xyz + FDI ──
        input_feat = torch.cat([tooth_points, fdi_feat], dim=-1)  # (B, 28, N, 11)

        # ── (B, 28, N, 11) → (B*28, 11, N) ──
        x = input_feat.permute(0, 1, 3, 2).reshape(B * T, 11, N)
        xyz = tooth_points.reshape(B * T, N, 3).permute(0, 2, 1)

        # 입력 프로젝션
        x = self.input_proj(x)

        # ── SA + DITA 블록 ──
        # 논문 명세: "DITA layers are thereby constrained to compute attention
        # scores only over the non-zero feature vectors of the context teeth Y".
        # 타겟 치아는 key/value 에서 배제(컨텍스트만 attend) → 타겟 쿼리가 환자별
        # 컨텍스트 기하에서 자기 위치를 읽어오도록 강제 (FDI 평균 prior 탈피).
        ctx_key = tooth_valid * (1 - target_mask)   # (B, 28) 컨텍스트만 1
        # 치아별 centroid (입력 해상도 그대로). 컨텍스트=실제 위치, 타겟=0.
        centroid = tooth_points.mean(dim=2)          # (B, 28, 3)
        for i in range(len(self.sa_blocks)):
            xyz, x = self.sa_blocks[i](xyz, x)

            # DITA: (B*28, C, npoint) → (B, 28, C) → DITA → reshape
            npoint = x.shape[2]
            ch = x.shape[1]
            x_pooled = x.reshape(B, T, ch, npoint).mean(dim=-1)
            # 절대 위치(centroid) 주입 → DITA가 컨텍스트 위치를 직접 인지
            x_pooled = x_pooled + self.pos_proj[i](centroid)
            tooth_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            x_pooled = self.dita_layers[i](
                x_pooled, tooth_indices, tooth_valid, key_mask=ctx_key
            )
            x = x + x_pooled.unsqueeze(-1).reshape(B * T, ch, 1).expand(-1, -1, npoint)

        # ── 글로벌 풀링: (B*28, C, npoint) → (B*28, C) ──
        x = self.global_pool(x).squeeze(-1)  # (B*28, C)

        # ── 회귀: (B*28, C) → (B*28, 5) ──
        boundary_pred = self.regressor(x)  # (B*28, 5)

        # ── (B, 28, 5)로 리셰이프 ──
        boundary_pred = boundary_pred.reshape(B, T, 5)

        return boundary_pred


def boundary_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Boundary 예측 손실 (Smooth L1, 타겟 치아에만 적용).

    Args:
        pred: (B, 28, 5) 예측값
        gt: (B, 28, 5) 정답
        target_mask: (B, 28) 타겟 마스크

    Returns:
        스칼라 손실
    """
    mask = target_mask.unsqueeze(-1).expand_as(pred)  # (B, 28, 5)
    loss = F.smooth_l1_loss(pred * mask, gt * mask, reduction='sum')
    n_targets = target_mask.sum().clamp(min=1)
    return loss / (n_targets * 5)
