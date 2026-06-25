"""
Exponential Moving Average (EMA) of model parameters.

Point-diffusion 모델(PVD 등 논문이 인용한 백본 포함)은 샘플 품질 향상을 위해
학습 가중치의 EMA 를 유지하고 추론 시 EMA 가중치로 샘플링한다. EMA 가 없으면
val loss 가 낮아도 생성 품질이 떨어지는 경우가 흔하다 (1차 학습 결과 부진의
주요 원인 후보).

사용:
  ema = EMA(model, decay=0.995)
  # 매 optimizer step 후:
  ema.update(model)
  # 추론 시:
  ema.apply_to(model)   # model 의 파라미터를 EMA shadow 로 교체
  ... sample ...
  ema.restore(model)    # 원래 학습 가중치로 복구 (계속 학습할 때)

체크포인트에는 ema.shadow 를 함께 저장한다.
"""

import copy
from typing import Dict

import torch
import torch.nn as nn


class EMA:
    """파라미터 EMA 유지.

    Args:
        model: EMA 를 적용할 모델 (현재 가중치로 shadow 초기화)
        decay: EMA 감쇠율 (PVD 기본 0.995). 클수록 평균화 느림.
        warmup: 초기 N 스텝 동안 decay 대신 shadow=param 으로 빠르게 따라감.
    """

    def __init__(self, model: nn.Module, decay: float = 0.995, warmup: int = 0):
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {
            n: p.detach().clone().float()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        # 추론 apply_to / restore 용 백업
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        # warmup 동안은 shadow 를 현재 가중치로 덮어쓰기 (초기 분산 안정화)
        if self.num_updates <= self.warmup:
            for n, p in model.named_parameters():
                if n in self.shadow:
                    self.shadow[n].copy_(p.detach().to(self.shadow[n].dtype))
            return
        for n, p in model.named_parameters():
            if n in self.shadow:
                p_detach = p.detach().to(self.shadow[n].dtype)
                self.shadow[n].mul_(self.decay).add_(p_detach, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        """model 의 파라미터를 EMA shadow 로 교체 (현재 값은 _backup 에 보관)."""
        self._backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self._backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].to(p.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        """apply_to 로 교체했던 원래 학습 가중치로 복구."""
        for n, p in model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: Dict[str, torch.Tensor]):
        for n in self.shadow:
            if n in state:
                self.shadow[n].copy_(state[n].to(self.shadow[n].dtype))
