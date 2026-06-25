"""Generation 모듈용 PVCNN building blocks (공식 cg_generation_module 포팅).

boundary(modules.py)와의 차이: gen 은 튜플에 o_mask 없이 temb(time+bound 임베딩) 사용.
PVConv/SAModule/FPModule 은 temb 3-투플/5-투플. RPE/어텐션/functional 은 boundary 재사용.
"""
import math
import torch
import torch.nn as nn

from . import functional as F
from .modules import SharedMLP, Swish, SE3d, Voxelization, BallQuery

__all__ = ['GenPVConv', 'GenPointNetSAModule', 'GenPointNetFPModule', 'Transformer', 'LayerNorm']


class GenPVConv(nn.Module):
    """gen 용 PVConv: 입력 (features, coords, temb) → (fused, coords, temb). o_mask 없음."""
    def __init__(self, in_channels, out_channels, kernel_size, resolution, attention=False,
                 dropout=0.1, with_se=False, with_se_relu=False, normalize=True, eps=0):
        super().__init__()
        self.resolution = resolution
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        layers = [nn.Conv3d(in_channels, out_channels, kernel_size, 1, kernel_size // 2),
                  nn.GroupNorm(8, out_channels), Swish()]
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Conv3d(out_channels, out_channels, kernel_size, 1, kernel_size // 2),
                   nn.GroupNorm(8, out_channels), Swish()]   # voxel self-attention off (메모리)
        if with_se:
            layers.append(SE3d(out_channels, use_relu=with_se_relu))
        self.voxel_layers = nn.Sequential(*layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        features, coords, temb = inputs
        vf, vc = self.voxelization(features, coords)
        vf = self.voxel_layers(vf)
        vf = F.trilinear_devoxelize(vf, vc, self.resolution, self.training)
        return vf + self.point_features(features), coords, temb


class GenPointNetSAModule(nn.Module):
    def __init__(self, num_centers, radius, num_neighbors, in_channels, out_channels, include_coordinates=True):
        super().__init__()
        if not isinstance(radius, (list, tuple)):
            radius = [radius]
        if not isinstance(num_neighbors, (list, tuple)):
            num_neighbors = [num_neighbors] * len(radius)
        if not isinstance(out_channels[0], (list, tuple)):
            out_channels = [out_channels] * len(radius)
        groupers, mlps = [], []
        total = 0
        for _r, _oc, _n in zip(radius, out_channels, num_neighbors):
            groupers.append(BallQuery(_r, _n, include_coordinates))
            mlps.append(SharedMLP(in_channels + (3 if include_coordinates else 0), _oc, dim=2))
            total += _oc[-1]
            in_channels = _oc[-1]
        self.num_centers = num_centers
        self.out_channels = total
        self.groupers = nn.ModuleList(groupers)
        self.mlps = nn.ModuleList(mlps)

    def forward(self, inputs):
        features, coords, temb = inputs
        centers_coords = F.furthest_point_sample(coords, self.num_centers)
        feats_list = []
        for grouper, mlp in zip(self.groupers, self.mlps):
            nf, gtemb = mlp(grouper(coords, centers_coords, temb, features))
            feats_list.append(nf.max(dim=-1).values)
        out = feats_list[0] if len(feats_list) == 1 else torch.cat(feats_list, 1)
        gtemb_out = gtemb.max(dim=-1).values if gtemb.shape[1] > 0 else gtemb
        return out, centers_coords, gtemb_out


class GenPointNetFPModule(nn.Module):
    """feature propagation (decoder upsampling). nearest-neighbor 보간 + SharedMLP."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = SharedMLP(in_channels=in_channels, out_channels=out_channels, dim=1)

    def forward(self, inputs):
        # (points_coords, centers_coords, centers_features, [points_features], temb)
        if len(inputs) == 4:
            points_coords, centers_coords, centers_features, temb = inputs
            points_features = None
        else:
            points_coords, centers_coords, centers_features, points_features, temb = inputs
        interp = F.nearest_neighbor_interpolate(points_coords, centers_coords, centers_features)
        interp_temb = F.nearest_neighbor_interpolate(points_coords, centers_coords, temb)
        if points_features is not None:
            interp = torch.cat([interp, points_features], dim=1)
        return self.mlp(interp), points_coords, interp_temb


# ───────────────── Transformer (CLIP ViT) for boundary conditioning ─────────────────

class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)


class _QKVMultiheadAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.n_heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.n_heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        w = torch.einsum('bthc,bshc->bhts', q * scale, k * scale)
        w = torch.softmax(w.float(), dim=-1).type(qkv.dtype)
        return torch.einsum('bhts,bshc->bthc', w, v).reshape(bs, n_ctx, -1)


class _MultiheadAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.c_qkv = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)
        self.attention = _QKVMultiheadAttention(heads)

    def forward(self, x):
        return self.c_proj(self.attention(self.c_qkv(x)))


class _ResidualAttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn = _MultiheadAttention(width, heads)
        self.ln_1 = LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
        self.ln_2 = LayerNorm(width)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """boundary embedding 용 CLIP-ViT 스타일 transformer."""
    def __init__(self, width, layers, heads):
        super().__init__()
        self.resblocks = nn.ModuleList([_ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x):
        for b in self.resblocks:
            x = b(x)
        return x
