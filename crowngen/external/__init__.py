"""CrownGen official-architecture port (pure-torch, no CUDA compile).

- functional: torch 구현 of PVCNN custom ops
- modules: SharedMLP / PVConv(Conv3d) / PointNetSA / IntertoothAttentionBlock (per-point DITA)
- pvcnn: PVCNN2 (boundary) + BoundEncoder
"""
from . import functional, modules, pvcnn
from .pvcnn import PVCNN2, BoundEncoder
