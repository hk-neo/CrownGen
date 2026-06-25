"""
CrownGen Denoising Network (U-Net).

논문의 핵심 아키텍처:
  4 SA blocks (인코더) + 4 FP blocks (디코더)
  9 DITA layers (SA 후 4 + 보틀넥 1 + FP 후 4)
  PVC 연산자 + 타임 임베딩 주입

핵심 패턴:
  - SA/FP는 치아별 독립 처리: (B*28, N, C)를 PVCNN에 통과
  - DITA는 치아 간 상호작용: (B, 28, C)로 리셰이프 후 어텐션
  - 타겟 치아에만 노이즈 적용, 컨텍스트는 깨끗하게 유지
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple

from .pvc import PointVoxelConv
from .pointnet2 import SetAbstraction, FeaturePropagation
from .dita import DITA
from .time_embed import TimestepEmbedding, TimeConditioning
from ..data.fdi import FDIEmbedding


class DenoiseNetwork(nn.Module):
    """CrownGen Denoising U-Net Network.

    Args:
        config: 모델 설정 딕셔너리 (configs/default.yaml에서 로드)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # 설정 추출
        model_cfg = config.get('model', config)
        self.max_teeth = config.get('data', config).get('max_teeth', 28)
        self.input_dim = model_cfg.get('input_dim', 17)
        sa_channels = model_cfg.get('sa_channels', [[64, 64], [128, 128], [256, 256], [512, 512]])
        sa_npoints = model_cfg.get('sa_npoints', [512, 256, 64, 16])
        sa_voxel_res = model_cfg.get('sa_voxel_res', [32, 16, 8, 4])
        sa_radius = model_cfg.get('sa_radius', [0.2, 0.3, 0.4, 0.5])
        fp_channels = model_cfg.get('fp_channels', [[512, 512], [256, 256], [128, 128], [64, 64]])
        fp_voxel_res = model_cfg.get('fp_voxel_res', [4, 8, 16, 32])
        pvc_dropout = model_cfg.get('pvc_dropout', 0.1)
        dita_heads = model_cfg.get('dita_heads', 8)
        dita_rpe_hidden = model_cfg.get('dita_rpe_hidden', 64)
        time_embed_dim = model_cfg.get('time_embed_dim', 128)
        subbatch_size = model_cfg.get('tooth_subbatch_size', 7)

        self.subbatch_size = subbatch_size
        self.n_sa_levels = len(sa_channels)

        # ── 입력 프로젝션 ──
        first_ch = sa_channels[0][0]
        self.input_proj = nn.Conv1d(self.input_dim, first_ch, 1)

        # ── 타임 임베딩 ──
        self.time_embed = TimestepEmbedding(time_embed_dim)

        # ── FDI 학습 가능 임베딩 (논문: "8-dimensional embedding of the tooth's
        #    unique FDI identifier"). fdi.FDIEmbedding은 FDI 번호 → 지그재그
        #    인덱스 → nn.Embedding(28, embed_dim) 매핑을 수행한다.
        fdi_embed_dim = model_cfg.get('fdi_embed_dim', 8)
        self.fdi_embedding = FDIEmbedding(embed_dim=fdi_embed_dim, max_teeth=self.max_teeth)

        # ── 인코더: SA 블록 + DITA ──
        self.sa_blocks = nn.ModuleList()
        self.sa_dita = nn.ModuleList()
        self.sa_time_cond = nn.ModuleList()

        in_ch = first_ch
        for i in range(self.n_sa_levels):
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
            self.sa_dita.append(
                DITA(dim=out_ch, num_heads=dita_heads, rpe_hidden=dita_rpe_hidden)
            )
            self.sa_time_cond.append(
                TimeConditioning(time_embed_dim, out_ch)
            )
            in_ch = out_ch

        # ── 보틀넥 DITA ──
        bottleneck_dim = sa_channels[-1][-1]
        self.bottleneck_dita = DITA(
            dim=bottleneck_dim, num_heads=dita_heads, rpe_hidden=dita_rpe_hidden
        )

        # ── 디코더: FP 블록 + DITA ──
        self.fp_blocks = nn.ModuleList()
        self.fp_dita = nn.ModuleList()
        self.fp_time_cond = nn.ModuleList()

        for i in range(self.n_sa_levels):
            # Skip connection from encoder at the TARGET resolution level
            # FP[i] upsamples to level (n_sa-2-i), so skip comes from SA at that level
            if i < self.n_sa_levels - 1:
                skip_ch = sa_channels[self.n_sa_levels - 2 - i][-1]
            else:
                skip_ch = 0  # No skip at original input resolution

            if i == 0:
                in_ch_fp = bottleneck_dim
            else:
                in_ch_fp = fp_channels[i - 1][-1]

            self.fp_blocks.append(
                FeaturePropagation(
                    in_channel=in_ch_fp,
                    skip_channel=skip_ch,
                    out_channels=fp_channels[i],
                    voxel_res=fp_voxel_res[i],
                    dropout=pvc_dropout,
                )
            )
            out_ch_fp = fp_channels[i][-1]
            self.fp_dita.append(
                DITA(dim=out_ch_fp, num_heads=dita_heads, rpe_hidden=dita_rpe_hidden)
            )
            self.fp_time_cond.append(
                TimeConditioning(time_embed_dim, out_ch_fp)
            )

        # ── 출력 프로젝션 ──
        final_dim = fp_channels[-1][-1]
        self.output_proj = nn.Sequential(
            nn.Conv1d(final_dim, final_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(final_dim, 3, 1),  # xyz 노이즈 예측
        )

    def forward(
        self,
        x_t: torch.Tensor,
        target_mask: torch.Tensor,
        context_mask: torch.Tensor,
        fdi_labels: torch.Tensor,
        boundaries: torch.Tensor,
        t: torch.Tensor,
        tooth_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Denoising Network 순방향.

        Args:
            x_t: (B, 28, N, 3) 노이즈가 있는 포인트 클라우드
            target_mask: (B, 28) 타겟 마스크 (1=타겟)
            context_mask: (B, 28) 컨텍스트 마스크 (1=컨텍스트)
            fdi_labels: (B, 28) FDI 치아 번호
            boundaries: (B, 28, 5) 실린더 경계 파라미터
            t: (B,) 확산 타임스텝
            tooth_valid: (B, 28) 유효 치아 마스크

        Returns:
            (B, 28, N, 3) 예측된 노이즈
        """
        B, T, N, _ = x_t.shape
        device = x_t.device

        # ── 특징 구성: xyz + binary + FDI_emb + boundary ──
        binary_indicator = target_mask.unsqueeze(-1).unsqueeze(-1).expand(B, T, N, 1)
        boundary_feat = boundaries.unsqueeze(2).expand(B, T, N, -1)  # (B, 28, N, 5)

        # FDI 학습 가능 임베딩: (B, 28, 8) → (B, 28, N, 8)
        fdi_feat = self.fdi_embedding(fdi_labels).unsqueeze(2).expand(B, T, N, -1)

        # 입력 특징 결합: (B, 28, N, 17) = 3(xyz) + 1(binary) + 8(FDI) + 5(boundary)
        input_feat = torch.cat([x_t, binary_indicator, fdi_feat, boundary_feat], dim=-1)

        # ── (B, 28, N, C) → (B*28, C, N) 로 변환 ──
        x = input_feat.permute(0, 1, 3, 2).reshape(B * T, self.input_dim, N)
        xyz = x_t.reshape(B * T, N, 3).permute(0, 2, 1)  # (B*28, 3, N)

        # 타임 임베딩: (B,) → (B, time_dim) → (B*28, time_dim)
        t_emb = self.time_embed(t)  # (B, time_dim)
        t_emb_expanded = t_emb.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)

        # 입력 프로젝션
        x = self.input_proj(x)  # (B*28, first_ch, N)

        # ── 인코더 (SA + DITA) ──
        skip_features = []
        skip_xyzs = []

        for i in range(self.n_sa_levels):
            xyz, x = self.sa_blocks[i](xyz, x)  # 다운샘플링
            skip_features.append(x)
            skip_xyzs.append(xyz)

            # 타임 조건부 변환
            x = self.sa_time_cond[i](x, t_emb_expanded)

            # DITA: (B*28, C, npoint) → (B, 28, C) → DITA → (B*28, C, npoint)
            npoint = x.shape[2]
            ch = x.shape[1]
            x_for_dita = x.reshape(B, T, ch, npoint).permute(0, 1, 3, 2).reshape(B, T * npoint, ch)
            # DITA는 (B, T, dim) 형태 필요 → 단순화: 치아당 평균 풀링
            x_pooled = x.reshape(B, T, ch, npoint).mean(dim=-1)  # (B, 28, ch)
            tooth_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            x_pooled = self.sa_dita[i](x_pooled, tooth_indices, tooth_valid)  # (B, 28, ch)
            # 풀링된 특징을 다시 포인트에 더함 (브로드캐스팅)
            x = x + x_pooled.unsqueeze(-1).reshape(B * T, ch, 1).expand(-1, -1, npoint)

        # ── 보틀넥 DITA ──
        bottleneck_ch = x.shape[1]
        bottleneck_npoints = x.shape[2]
        x_bottleneck = x.reshape(B, T, bottleneck_ch, bottleneck_npoints).mean(dim=-1)
        tooth_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x_bottleneck = self.bottleneck_dita(x_bottleneck, tooth_indices, tooth_valid)
        x = x + x_bottleneck.unsqueeze(-1).reshape(B * T, bottleneck_ch, 1).expand(-1, -1, bottleneck_npoints)

        # ── 디코더 (FP + DITA) ──
        # 원래 해상도의 좌표 저장 (마지막 FP 업샘플링용)
        original_xyz = x_t.reshape(B * T, N, 3).permute(0, 2, 1)  # (B*28, 3, N)

        for i in range(self.n_sa_levels):
            # 타겟 해상도 결정: SA 레벨 역순으로 업샘플링
            # FP[0]: 16→64, FP[1]: 64→256, FP[2]: 256→512, FP[3]: 512→1024
            if i < self.n_sa_levels - 1:
                target_level = self.n_sa_levels - 2 - i
                target_xyz = skip_xyzs[target_level]
                skip_feat = skip_features[target_level]
            else:
                # 마지막 FP: 원래 N 포인트로 복원
                target_xyz = original_xyz
                skip_feat = None

            # FP: xyz(저해상도)에서 target_xyz(고해상도)로 업샘플링
            x = self.fp_blocks[i](target_xyz, xyz, x, skip_feat)
            xyz = target_xyz  # xyz를 새 해상도로 업데이트

            # 타임 조건부 변환
            x = self.fp_time_cond[i](x, t_emb_expanded)

            # DITA
            npoint = x.shape[2]
            ch = x.shape[1]
            x_pooled = x.reshape(B, T, ch, npoint).mean(dim=-1)
            x_pooled = self.fp_dita[i](x_pooled, tooth_indices, tooth_valid)
            x = x + x_pooled.unsqueeze(-1).reshape(B * T, ch, 1).expand(-1, -1, npoint)

        # ── 출력: 노이즈 예측 ──
        noise_pred = self.output_proj(x)  # (B*28, 3, N)
        noise_pred = noise_pred.reshape(B, T, 3, N).permute(0, 1, 3, 2)  # (B, 28, N, 3)

        # 컨텍스트 치아의 예측은 0으로 마스킹
        context_mask_3d = context_mask.unsqueeze(-1).unsqueeze(-1).expand_as(noise_pred)
        noise_pred = noise_pred * (1 - context_mask_3d)

        return noise_pred
