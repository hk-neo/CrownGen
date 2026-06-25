"""
Gaussian Diffusion (DDPM) 프로세스.

CrownGen의 핵심 생성 메커니즘:
- Forward process: 깨끗한 치아 포인트 클라우드에 점진적으로 가우시안 노이즈 추가
- Reverse process: 노이즈에서 시작하여 학습된 디노이징 네트워크로 치아 복원

논문 파라미터:
  T = 1000, β_min = 1e-4, β_max = 2e-2, 선형 스케줄
  Loss = MSE(ε - ε_θ(x_t, Y, B, t))  — 타겟 치아에만 적용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class GaussianDiffusion(nn.Module):
    """Gaussian Diffusion 프로세스 관리.

    Args:
        timesteps: 확산 스텝 수 (기본 1000)
        beta_min: 최소 노이즈 스케줄
        beta_max: 최대 노이즈 스케줄
        schedule: 'linear' 또는 'cosine'
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_min: float = 1e-4,
        beta_max: float = 2e-2,
        schedule: str = 'linear',
    ):
        super().__init__()
        self.timesteps = timesteps

        # 노이즈 스케줄
        if schedule == 'linear':
            betas = torch.linspace(beta_min, beta_max, timesteps, dtype=torch.float32)
        elif schedule == 'cosine':
            # Improved DDPM cosine schedule
            steps = timesteps + 1
            s = 0.008
            t_arr = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos((t_arr / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # 사전 계산된 값들 (버퍼로 등록)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # 역방향 분산
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def forward_process(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: x_0 → x_t.

        q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t) · I)

        Args:
            x_0: (B, 28, N, 3) 깨끗한 치아 포인트 클라우드
            t: (B,) 타임스텝 인덱스
            noise: (B, 28, N, 3) 가우시안 노이즈 (없으면 자동 생성)

        Returns:
            x_t: 노이즈가 추가된 포인트 클라우드
            noise: 사용된 노이즈
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # 브로드캐스팅을 위한 차원 확장 (버퍼는 CPU → 인덱싱 후 target device로)
        sqrt_alpha = self.sqrt_alphas_cumprod[t.cpu()].to(x_0.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t.cpu()].to(x_0.device)

        # (B,) → (B, 1, 1, 1) for broadcasting with (B, 28, N, 3)
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    def training_loss(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """학습 손실 계산.

        L = E[||ε - ε_θ(x_t, Y, B, t)||²]
        손실은 타겟(마스킹된) 치아에만 적용.

        Args:
            model: DenoiseNetwork
            batch: 데이터 배치 딕셔너리

        Returns:
            스칼라 손실 값
        """
        x_0 = batch['tooth_points'].to(device)      # (B, 28, N, 3)
        target_mask = batch['target_mask'].to(device)  # (B, 28)
        context_mask = batch['context_mask'].to(device)
        fdi_labels = batch['fdi_labels'].to(device)
        boundaries = batch['boundaries'].to(device)
        tooth_valid = batch['tooth_valid'].to(device)

        B, T, N, C = x_0.shape

        # 랜덤 타임스텝 샘플링
        t = torch.randint(0, self.timesteps, (B,), device=device, dtype=torch.long)

        # 랜덤 노이즈
        noise = torch.randn_like(x_0)

        # Forward diffusion
        x_t, _ = self.forward_process(x_0, t, noise)

        # 논문 명세: "context teeth Y ... remain noise-free and serve only as a
        # condition". forward_process 는 전체 치아에 노이즈를 더하므로, 컨텍스트
        # 치아는 다시 clean x_0 로 복원한다 (타겟 치아만 노이즈가 적용됨).
        # 추론(pipeline.generate_crowns)도 컨텍스트를 깨끗하게 고정하므로
        # train/test 분포를 일치시킨다.
        ctx_3d = context_mask.unsqueeze(-1).unsqueeze(-1).expand_as(x_0).float()
        x_t = x_t * (1.0 - ctx_3d) + x_0 * ctx_3d

        # 모델 예측
        noise_pred = model(
            x_t=x_t,
            target_mask=target_mask,
            context_mask=context_mask,
            fdi_labels=fdi_labels,
            boundaries=boundaries,
            t=t,
            tooth_valid=tooth_valid,
        )  # (B, 28, N, 3)

        # 타겟 치아에만 손실 적용
        target_mask_3d = target_mask.unsqueeze(-1).unsqueeze(-1).expand_as(x_0)  # (B, 28, N, 3)
        loss = F.mse_loss(noise_pred * target_mask_3d, noise * target_mask_3d)

        return loss

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: int,
        **kwargs
    ) -> torch.Tensor:
        """단일 역방향 샘플링 스텝: x_t → x_{t-1}.

        Args:
            model: DenoiseNetwork
            x_t: (B, 28, N, 3) 현재 노이즈 상태
            t: 현재 타임스텝 (정수)
            **kwargs: 모델에 전달할 추가 인자

        Returns:
            (B, 28, N, 3) 한 스텝 디노이징된 결과
        """
        B = x_t.shape[0]
        device = x_t.device
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        # 노이즈 예측
        noise_pred = model(x_t=x_t, t=t_tensor, **kwargs)

        # DDPM 샘플링 공식:
        # x_{t-1} = (1/√α_t)(x_t - β_t/√(1-ᾱ_t) · ε_θ) + √β̃_t · η
        dev = x_t.device
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t]

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * noise_pred
        )

        if t > 0:
            noise = torch.randn_like(x_t)
            sqrt_posterior_var = torch.sqrt(self.posterior_variance[t])
            x_prev = mean + sqrt_posterior_var * noise
        else:
            x_prev = mean

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        boundaries: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """완전한 역방향 샘플링: x_T ~ N(0,I) → x_0.

        초기 노이즈는 예측된 실린더 경계 내에서 샘플링됩니다.

        Args:
            model: DenoiseNetwork
            shape: (B, 28, N, 3) 출력 형태
            boundaries: (B, 28, 5) 실린더 경계 파라미터
            **kwargs: 모델에 전달할 추가 인자

        Returns:
            (B, 28, N, 3) 생성된 치아 포인트 클라우드
        """
        device = boundaries.device
        B, T, N, C = shape

        # 초기 노이즈: 실린더 경계 내에서 샘플링
        x = torch.randn(shape, device=device)
        # cx, cy, cz, r, h
        cx = boundaries[:, :, 0].unsqueeze(-1)  # (B, 28, 1)
        cy = boundaries[:, :, 1].unsqueeze(-1)
        cz = boundaries[:, :, 2].unsqueeze(-1)
        r = boundaries[:, :, 3].unsqueeze(-1)
        h = boundaries[:, :, 4].unsqueeze(-1)

        # 실린더 내부로 제약
        x[:, :, :, 0] = x[:, :, :, 0] * r * 0.5 + cx
        x[:, :, :, 1] = x[:, :, :, 1] * r * 0.5 + cy
        x[:, :, :, 2] = x[:, :, :, 2] * h * 0.25 + cz

        # 역방향 샘플링 루프
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, t, boundaries=boundaries, **kwargs)

        return x
