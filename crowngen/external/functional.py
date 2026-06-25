"""Functional ops — 공식 CUDA 커널 우선, 컴파일 불가 시 순수 torch fallback.

CUDA_HOME 가 세팅돼 있고 nvcc 가 있으면 cg 공식 CUDA ops(빠름) 사용.
아니면 functional_torch (느리지만 컴파일 불필요).
"""
import os
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')

try:
    from .cuda_func import (
        avg_voxelize, trilinear_devoxelize, ball_query, grouping,
        furthest_point_sample, nearest_neighbor_interpolate, gather,
        kl_loss, huber_loss,
    )
    _BACKEND = 'cuda'
except Exception as _e:                       # noqa
    from .functional_torch import (                           # type: ignore
        avg_voxelize, trilinear_devoxelize, ball_query, grouping,
        furthest_point_sample, nearest_neighbor_interpolate, gather,
        kl_loss, huber_loss,
    )
    _BACKEND = f'torch (CUDA unavailable: {type(_e).__name__})'

__all__ = ['gather', 'furthest_point_sample', 'grouping', 'ball_query',
           'avg_voxelize', 'trilinear_devoxelize', 'nearest_neighbor_interpolate',
           'kl_loss', 'huber_loss']
