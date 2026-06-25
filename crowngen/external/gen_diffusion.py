"""PVD-style Gaussian Diffusion for CrownGen generation (공식 model/diffusion.py 포팅).

eps-prediction, fixedsmall variance. betas linear (β 1e-4→2e-2, T=1000) = 논문 설정.
손실은 타겟 치아(teethmask)에만. 샘플링은 p_sample_loop (x_T 노이즈 → x_0 크라운).
"""
import numpy as np
import torch
import torch.nn as nn

from .gen_autoencoder import PVCNN2

__all__ = ['GaussianDiffusion', 'GenModel', 'get_betas']


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        return np.linspace(b_start, b_end, time_num, dtype=np.float64)
    raise NotImplementedError(schedule_type)


class GaussianDiffusion:
    def __init__(self, betas, model_mean_type='eps', model_var_type='fixedsmall'):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        betas = betas.astype(np.float64)
        assert (betas > 0).all() and (betas <= 1).all()
        self.num_timesteps = int(betas.shape[0])

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = torch.from_numpy(alphas_cumprod).float()
        self.alphas_cumprod_prev = torch.from_numpy(alphas_cumprod_prev).float()
        self.sqrt_alphas_cumprod = torch.from_numpy(np.sqrt(alphas_cumprod)).float()
        self.sqrt_one_minus_alphas_cumprod = torch.from_numpy(np.sqrt(1. - alphas_cumprod)).float()

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance = torch.from_numpy(posterior_variance).float()
        self.posterior_log_variance_clipped = torch.from_numpy(np.log(np.clip(posterior_variance, 1e-20, None))).float()
        self.posterior_mean_coef1 = torch.from_numpy(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).float()
        self.posterior_mean_coef2 = torch.from_numpy((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)).float()

    @staticmethod
    def _extract(a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t.long())
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise)

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            x_t - self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        ) / self._extract(self.sqrt_alphas_cumprod.to(x_t.device), t, x_t.shape).clamp(min=1e-8)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        m1 = self._extract(self.posterior_mean_coef1.to(x_t.device), t, x_t.shape)
        m2 = self._extract(self.posterior_mean_coef2.to(x_t.device), t, x_t.shape)
        return m1 * x_start + m2 * x_t

    def p_mean_variance(self, denoise_fn, xt, t, model_kwargs):
        model_output = denoise_fn(xt, t, model_kwargs)
        pred_xstart = self._predict_xstart_from_eps(xt, t, model_output)
        # eps→x0 는 고 t에서 증폭돼 큰 값이 나올 수 있음. 데이터 범위로 clamp 안정화
        # (improved-DDPM 표준 기법). 데이터는 ~[-1,1] 정규화.
        pred_xstart = pred_xstart.clamp(-1.0, 1.0)
        model_mean = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=xt, t=t)
        model_var = self._extract(self.posterior_variance.to(xt.device), t, xt.shape)
        model_log_var = self._extract(self.posterior_log_variance_clipped.to(xt.device), t, xt.shape)
        return model_mean, model_var, model_log_var, pred_xstart

    @torch.no_grad()
    def p_sample(self, denoise_fn, xt, t, model_kwargs, noise_fn=torch.randn):
        model_mean, _, model_log_var, _ = self.p_mean_variance(denoise_fn, xt, t, model_kwargs)
        noise = noise_fn(size=xt.shape, dtype=xt.dtype, device=xt.device)
        nonzero = (t != 0).float().reshape(xt.shape[0], *([1] * (len(xt.shape) - 1)))
        return model_mean + nonzero * torch.exp(0.5 * model_log_var) * noise

    @torch.no_grad()
    def p_sample_loop(self, model_kwargs, denoise_fn, noise_fn=torch.randn):
        l_mask = model_kwargs['l_mask']
        x0 = model_kwargs['x0']
        lm4 = l_mask.unsqueeze(-1).unsqueeze(-1)            # (B,28,1,1)
        # 타겟은 순수 노이즈(고 t의 q_sample 분포와 일치), 컨텍스트는 clean x0
        xt = noise_fn(size=x0.shape, dtype=x0.dtype, device=x0.device) * lm4 + x0 * (1 - lm4)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((xt.shape[0],), i, device=xt.device, dtype=torch.long)
            xt = self.p_sample(denoise_fn, xt, t, model_kwargs, noise_fn)
            # 컨텍스트는 항상 clean(x0) 유지
            xt = xt * lm4 + x0 * (1 - lm4)
        return xt

    def p_losses(self, denoise_fn, x_start, t, noise, model_kwargs):
        x_t = self.q_sample(x_start, t, noise=noise)
        model_kwargs = dict(model_kwargs)
        model_kwargs['xt'] = x_t
        model_output = denoise_fn(x_t, t, model_kwargs)
        target = noise                                  # eps-prediction
        l_mask = model_kwargs['l_mask']
        mask = l_mask.unsqueeze(-1).unsqueeze(-1)        # (B,nT,1,1) → 타겟 치아에만 loss
        sq = ((target - model_output) ** 2) * mask
        return sq.sum() / mask.sum().clamp(min=1) / 3    # per-element MSE (3 coords)


class GenModel(nn.Module):
    """diffusion + denoiser(PVCNN2) 래퍼. boundary ckpt 를 로드해 bound 예측에 사용."""
    def __init__(self, betas, embed_dim=64, dropout=0.1, extra_feature_channels=9, mask_mode='official'):
        super().__init__()
        self.diffusion = GaussianDiffusion(betas)
        self.model = PVCNN2(num_classes=3, embed_dim=embed_dim, use_att=True, dropout=dropout,
                            extra_feature_channels=extra_feature_channels, mask_mode=mask_mode)

    def _denoise(self, xt, t, model_kwargs):
        return self.model(xt, t, False,
                          x0=model_kwargs['x0'], l_mask=model_kwargs['l_mask'],
                          o_mask=model_kwargs['o_mask'], bound=model_kwargs['bound'])[0]

    def loss(self, noise, model_kwargs):
        x_start = model_kwargs['x0']                     # GT clean teeth = 복원 목표
        B = x_start.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=x_start.device)
        return self.diffusion.p_losses(self._denoise, x_start, t, noise, model_kwargs)

    @torch.no_grad()
    def sample(self, model_kwargs, noise_fn=torch.randn):
        return self.diffusion.p_sample_loop(model_kwargs, self._denoise, noise_fn=noise_fn)

    def train(self, mode=True):
        self.model.train(mode); return self
    def eval(self):
        self.model.eval(); return self
