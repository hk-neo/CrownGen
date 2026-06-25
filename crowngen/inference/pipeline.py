"""
CrownGen End-to-End 추론 파이프라인.

워크플로우:
  1. 입력: 분할된 치아 포인트 클라우드 + FDI 라벨 + 결손치 정보
  2. Boundary Prediction: 각 결손치의 실린더 경계 예측
  3. Diffusion Sampling: DDPM 역확산으로 크라운 포인트 클라우드 생성
  4. (선택) DPSR: 포인트 클라우드 → Watertight 메쉬 변환
  5. 출력: 생성된 크라운 포인트 클라우드 또는 메쉬

사용법:
  python -m crowngen.inference.pipeline \
      --config crowngen/configs/default.yaml \
      --boundary_ckpt runs/boundary/checkpoints/best.pt \
      --diffusion_ckpt runs/diffusion_stage1/checkpoints/best.pt \
      --input_dir Data/processed \
      --patient_id SAMPLE001 \
      --missing_teeth 36 46 \
      --output_dir results/
"""

import argparse
import json
import yaml
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from crowngen.models.boundary_net import BoundaryPredictor
from crowngen.models.denoise_net import DenoiseNetwork
from crowngen.models.diffusion import GaussianDiffusion
from crowngen.models.ema import EMA
from crowngen.data.fdi import ZIGZAG_FDI_ORDER, FDI_TO_ZIGZAG, ALL_28_FDI


class CrownGenPipeline:
    """CrownGen End-to-End 추론 파이프라인.

    Args:
        config: 모델 설정
        boundary_ckpt: Boundary 모델 체크포인트 경로
        diffusion_ckpt: Diffusion 모델 체크포인트 경로
        device: 연산 디바이스
        n_points: 치아당 포인트 수
    """

    def __init__(
        self,
        config: dict,
        boundary_ckpt: str,
        diffusion_ckpt: str,
        device: str = 'cuda',
        n_points: int = 1024,
        use_ema: bool = True,
    ):
        self.config = config
        self.device = torch.device(device)
        self.n_points = n_points

        # Boundary Predictor
        self.boundary_model = BoundaryPredictor(config).to(self.device)
        ckpt = torch.load(boundary_ckpt, map_location=self.device)
        self.boundary_model.load_state_dict(ckpt['model_state_dict'])
        self.boundary_model.eval()

        # Denoising Network
        self.denoise_model = DenoiseNetwork(config).to(self.device)
        ckpt = torch.load(diffusion_ckpt, map_location=self.device)
        self.denoise_model.load_state_dict(ckpt['model_state_dict'])
        self.denoise_model.eval()

        # EMA: 학습 시 유지된 EMA shadow 가 있으면 로드. 추론(샘플링)은 EMA 가중치로.
        self.ema = EMA(self.denoise_model, decay=0.995)
        self.use_ema = use_ema and ('ema_state_dict' in ckpt)
        if self.use_ema:
            self.ema.load_state_dict(ckpt['ema_state_dict'])
            print(f"  EMA 가중치 로드 (샘플링에 사용)")
        elif use_ema:
            print(f"  ⚠️ 체크포인트에 ema_state_dict 없음 — 학습 가중치로 샘플링")

        # Diffusion 프로세스
        diff_cfg = config['diffusion']
        self.diffusion = GaussianDiffusion(
            timesteps=diff_cfg['timesteps'],
            beta_min=diff_cfg['beta_min'],
            beta_max=diff_cfg['beta_max'],
            schedule=diff_cfg['schedule'],
        )

        n_params_b = sum(p.numel() for p in self.boundary_model.parameters())
        n_params_d = sum(p.numel() for p in self.denoise_model.parameters())
        print(f"CrownGenPipeline 초기화 완료")
        print(f"  Boundary: {n_params_b:,} params")
        print(f"  Diffusion: {n_params_d:,} params")
        print(f"  Device: {self.device}")

    @torch.no_grad()
    def predict_boundaries(
        self,
        tooth_points: torch.Tensor,
        fdi_labels: torch.Tensor,
        tooth_valid: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Boundary 예측.

        Args:
            tooth_points: (B, 28, N, 3) 전체 치아 포인트
            fdi_labels: (B, 28) FDI 라벨
            tooth_valid: (B, 28) 유효 마스크
            target_mask: (B, 28) 타겟 마스크

        Returns:
            (B, 28, 5) 예측된 실린더 경계
        """
        # 타겟 치아 포인트를 영벡터로 마스킹
        masked_points = tooth_points.clone()
        mask_3d = target_mask.unsqueeze(-1).unsqueeze(-1).expand_as(tooth_points)
        masked_points = masked_points * (1 - mask_3d.float())

        boundaries = self.boundary_model(
            masked_points, fdi_labels, tooth_valid, target_mask
        )
        return boundaries

    @torch.no_grad()
    def generate_crowns(
        self,
        tooth_points: torch.Tensor,
        fdi_labels: torch.Tensor,
        tooth_valid: torch.Tensor,
        target_mask: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> torch.Tensor:
        """Diffusion 샘플링으로 크라운 생성.

        Args:
            tooth_points: (B, 28, N, 3) 컨텍스트 치아 포인트
            fdi_labels: (B, 28)
            tooth_valid: (B, 28)
            target_mask: (B, 28)
            boundaries: (B, 28, 5)

        Returns:
            (B, 28, N, 3) 생성된 전체 포인트 클라우드
        """
        B, T, N, _ = tooth_points.shape
        device = self.device

        # 초기 상태 X_T 샘플링.
        # 논문: "We start by sampling from the prior, X_T ~ N(0,I), within the
        # predicted boundaries B". 따라서 타겟 치아는 예측된 실린더 경계 내에서
        # N(0,I) 노이즈로 초기화하고, 컨텍스트 치아는 실제 환자 치아를 그대로
        # 유지(고정)한다.
        target_4d = target_mask.unsqueeze(-1).unsqueeze(-1).expand_as(tooth_points).float()

        noise_init = torch.randn_like(tooth_points)
        cx = boundaries[:, :, 0:1]   # (B, T, 1)
        cy = boundaries[:, :, 1:2]
        cz = boundaries[:, :, 2:3]
        r = boundaries[:, :, 3:4]
        h = boundaries[:, :, 4:5]
        # N(0,1) → 실린더 내부 (반경 r, 높이 h). diffusion.sample()과 동일 스케일.
        noise_init[..., 0] = noise_init[..., 0] * (r * 0.5) + cx
        noise_init[..., 1] = noise_init[..., 1] * (r * 0.5) + cy
        noise_init[..., 2] = noise_init[..., 2] * (h * 0.25) + cz

        # 타겟 = 노이즈 초기화, 컨텍스트 = 실제 치아
        x = noise_init * target_4d + tooth_points * (1.0 - target_4d)

        # 샘플링은 EMA 가중치로 수행 (품질 향상)
        if self.use_ema:
            self.ema.apply_to(self.denoise_model)
        try:
            return self._reverse_sample(
                x, tooth_points, target_4d, target_mask, fdi_labels,
                tooth_valid, boundaries, B
            )
        finally:
            if self.use_ema:
                self.ema.restore(self.denoise_model)

    @torch.no_grad()
    def _reverse_sample(self, x, tooth_points, target_4d, target_mask,
                        fdi_labels, tooth_valid, boundaries, B):
        """DDPM 역방향 샘플링 루프 (EMA 컨텍스트는 호출측에서 관리)."""
        device = self.device
        # DDPM 역방향 샘플링
        for t_step in reversed(range(self.diffusion.timesteps)):
            t_tensor = torch.full((B,), t_step, device=device, dtype=torch.long)

            noise_pred = self.denoise_model(
                x_t=x,
                target_mask=target_mask,
                context_mask=1 - target_mask,
                fdi_labels=fdi_labels,
                boundaries=boundaries,
                t=t_tensor,
                tooth_valid=tooth_valid,
            )

            # DDPM 샘플링 공식
            beta_t = self.diffusion.betas[t_step]
            alpha_t = self.diffusion.alphas[t_step]
            sqrt_one_minus_alpha_bar = self.diffusion.sqrt_one_minus_alphas_cumprod[t_step]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / sqrt_one_minus_alpha_bar) * noise_pred
            )

            if t_step > 0:
                noise = torch.randn_like(x)
                sqrt_var = torch.sqrt(self.diffusion.posterior_variance[t_step])
                x_new = mean + sqrt_var * noise
            else:
                x_new = mean

            # 타겟 치아만 업데이트, 컨텍스트는 원본 유지
            mask_4d = target_mask.unsqueeze(-1).unsqueeze(-1).expand_as(x)
            x = x_new * mask_4d.float() + tooth_points * (1 - mask_4d.float())

        return x

    @torch.no_grad()
    def __call__(
        self,
        tooth_points: torch.Tensor,
        fdi_labels: torch.Tensor,
        tooth_valid: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """End-to-End 크라운 생성.

        Args:
            tooth_points: (1, 28, N, 3) 입력 포인트 클라우드
            fdi_labels: (1, 28) FDI 라벨
            tooth_valid: (1, 28) 유효 마스크
            target_mask: (1, 28) 생성할 타겟 마스크

        Returns:
            {'generated': (1, 28, N, 3), 'boundaries': (1, 28, 5)}
        """
        t0 = time.time()

        # Step 1: Boundary 예측
        boundaries = self.predict_boundaries(
            tooth_points, fdi_labels, tooth_valid, target_mask
        )
        t_boundary = time.time()

        # Step 2: Diffusion 샘플링
        generated = self.generate_crowns(
            tooth_points, fdi_labels, tooth_valid, target_mask, boundaries
        )
        t_diffusion = time.time()

        print(f"  Boundary: {t_boundary - t0:.1f}s")
        print(f"  Diffusion: {t_diffusion - t_boundary:.1f}s ({self.diffusion.timesteps} steps)")
        print(f"  Total: {t_diffusion - t0:.1f}s")

        return {
            'generated': generated,
            'boundaries': boundaries,
        }


def load_patient_data(
    npz_path: str,
    missing_teeth: List[int],
    n_points: int = 1024,
) -> Dict[str, torch.Tensor]:
    """처리된 .npz 파일에서 단일 환자 데이터를 로드하고 배치 형식으로 변환.

    Args:
        npz_path: .npz 파일 경로
        missing_teeth: 결손치 FDI 번호 리스트
        n_points: 치아당 포인트 수

    Returns:
        배치 형식 텐서 딕셔너리
    """
    data = np.load(npz_path)

    tooth_points = torch.zeros(28, n_points, 3, dtype=torch.float32)
    fdi_labels = torch.zeros(28, dtype=torch.long)
    boundaries = torch.zeros(28, 5, dtype=torch.float32)
    tooth_valid = torch.zeros(28, dtype=torch.long)
    target_mask = torch.zeros(28, dtype=torch.long)

    missing_set = set(missing_teeth)

    for slot_idx, fdi in enumerate(ZIGZAG_FDI_ORDER):
        jaw = 'upper' if fdi // 10 in [1, 2] else 'lower'
        pc_key = f"{jaw}_{fdi}_pc"
        bound_key = f"{jaw}_{fdi}_bound"

        if pc_key in data:
            pc = torch.from_numpy(data[pc_key]).float()
            if pc.shape[0] >= n_points:
                idx = torch.randperm(pc.shape[0])[:n_points]
                pc = pc[idx]
            else:
                n_extra = n_points - pc.shape[0]
                extra_idx = torch.randint(0, pc.shape[0], (n_extra,))
                extra = pc[extra_idx] + torch.randn(n_extra, 3) * 0.01
                pc = torch.cat([pc, extra], dim=0)

            tooth_points[slot_idx] = pc
            fdi_labels[slot_idx] = fdi

            if fdi in missing_set:
                target_mask[slot_idx] = 1
            else:
                tooth_valid[slot_idx] = 1

            if bound_key in data:
                boundaries[slot_idx] = torch.from_numpy(data[bound_key]).float()
        else:
            # 데이터에 없는 치아 = 결손
            fdi_labels[slot_idx] = fdi
            if fdi in missing_set:
                target_mask[slot_idx] = 1

    # 배치 차원 추가
    return {
        'tooth_points': tooth_points.unsqueeze(0),
        'fdi_labels': fdi_labels.unsqueeze(0),
        'tooth_valid': tooth_valid.unsqueeze(0),
        'target_mask': target_mask.unsqueeze(0),
        'boundaries': boundaries.unsqueeze(0),
    }


def save_results(
    generated: torch.Tensor,
    boundaries: torch.Tensor,
    target_mask: torch.Tensor,
    fdi_labels: torch.Tensor,
    output_dir: str,
    patient_id: str,
):
    """생성 결과를 저장.

    각 타겟 치아별로 포인트 클라우드를 .npy로 저장.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask = target_mask[0].cpu()  # (28,)
    fdi = fdi_labels[0].cpu()    # (28,)
    gen = generated[0].cpu()     # (28, N, 3)
    bnd = boundaries[0].cpu()    # (28, 5)

    results = {}
    for i in range(28):
        if mask[i] == 1:
            fdi_num = fdi[i].item()
            points = gen[i].numpy()   # (N, 3)
            boundary = bnd[i].numpy()  # (5,)

            np.save(output_dir / f"{patient_id}_{fdi_num}_crown.npy", points)
            np.save(output_dir / f"{patient_id}_{fdi_num}_boundary.npy", boundary)

            results[str(fdi_num)] = {
                'n_points': points.shape[0],
                'boundary': boundary.tolist(),
            }

    with open(output_dir / f"{patient_id}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"결과 저장: {output_dir} ({len(results)}개 크라운)")


def parse_args():
    parser = argparse.ArgumentParser(description='CrownGen 추론')
    parser.add_argument('--config', type=str, default='crowngen/configs/default.yaml')
    parser.add_argument('--boundary_ckpt', type=str, required=True)
    parser.add_argument('--diffusion_ckpt', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True,
                        help='처리된 .npz 파일 경로')
    parser.add_argument('--missing_teeth', type=int, nargs='+', required=True,
                        help='결손치 FDI 번호 (예: 36 46)')
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 파이프라인 초기화
    pipeline = CrownGenPipeline(
        config=config,
        boundary_ckpt=args.boundary_ckpt,
        diffusion_ckpt=args.diffusion_ckpt,
        device=args.device,
        n_points=config['data']['n_points'],
    )

    # 데이터 로드
    patient_id = Path(args.input_path).stem
    print(f"\n환자: {patient_id}")
    print(f"결손치: {args.missing_teeth}")

    batch = load_patient_data(
        args.input_path,
        args.missing_teeth,
        n_points=config['data']['n_points'],
    )

    # 디바이스로 이동
    for key in batch:
        batch[key] = batch[key].to(args.device)

    # 추론
    print("\n크라운 생성 중...")
    result = pipeline(
        tooth_points=batch['tooth_points'],
        fdi_labels=batch['fdi_labels'],
        tooth_valid=batch['tooth_valid'],
        target_mask=batch['target_mask'],
    )

    # 결과 저장
    save_results(
        generated=result['generated'],
        boundaries=result['boundaries'],
        target_mask=batch['target_mask'],
        fdi_labels=batch['fdi_labels'],
        output_dir=args.output_dir,
        patient_id=patient_id,
    )

    print("\n✅ 추론 완료!")


if __name__ == '__main__':
    main()
