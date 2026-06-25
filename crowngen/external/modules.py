"""CrownGen boundary PVCNN2 building blocks (공식 cg_boundary_prediction_module 포팅).

CUDA 확장 대신 ../external/functional (순수 torch) 사용. 구조/시그니처는 공식에 일치.
"""
import functools
import torch
import torch.nn as nn

from . import functional as F

__all__ = [
    'Swish', 'SharedMLP', 'SE3d', 'BallQuery', 'Voxelization', 'PVConv',
    'PointNetSAModule', 'GroupNorm32', 'IntertoothAttentionBlock',
]


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# ───────────────── SharedMLP ─────────────────

class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1):
        super().__init__()
        conv = nn.Conv1d if dim == 1 else (nn.Conv2d if dim == 2 else None)
        if conv is None:
            raise ValueError(dim)
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for oc in out_channels:
            layers += [conv(in_channels, oc, 1), nn.GroupNorm(8, oc), Swish()]
            in_channels = oc
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return (self.layers(inputs[0]), *inputs[1:])
        return self.layers(inputs)


# ───────────────── SE3d ─────────────────

class SE3d(nn.Module):
    def __init__(self, channel, reduction=8, use_relu=False):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(True) if use_relu else Swish(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        m = inputs.mean(-1).mean(-1).mean(-1)
        return inputs * self.fc(m).view(inputs.shape[0], inputs.shape[1], 1, 1, 1)


# ───────────────── BallQuery ─────────────────

class BallQuery(nn.Module):
    def __init__(self, radius, num_neighbors, include_coordinates=True):
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.include_coordinates = include_coordinates

    def forward(self, points_coords, centers_coords, femb, points_features=None):
        neighbor_indices = F.ball_query(centers_coords, points_coords, self.radius, self.num_neighbors)
        neighbor_coordinates = F.grouping(points_coords, neighbor_indices) - centers_coords.unsqueeze(-1)
        if points_features is None:
            assert self.include_coordinates
            neighbor_features = neighbor_coordinates
        else:
            neighbor_features = F.grouping(points_features, neighbor_indices)
            if self.include_coordinates:
                neighbor_features = torch.cat([neighbor_coordinates, neighbor_features], dim=1)
        return neighbor_features, F.grouping(femb, neighbor_indices)


# ───────────────── Voxelization ─────────────────

class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            denom = norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps
            # 결손(포인트 0) 치아는 0/0 → nan 방지. 이 치아 복셀 값은 devoxelize 에서
            # tooth_mask 로 0 처리되므로 denom floor 값은 결과에 영향 없음.
            denom = denom.clamp(min=1e-6)
            norm_coords = norm_coords / denom + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).to(torch.int32)
        return F.avg_voxelize(features, vox_coords, self.r), norm_coords


# ───────────────── PVConv ─────────────────

class _Attention(nn.Module):
    def __init__(self, in_ch, num_groups, D=3):
        super().__init__()
        conv = nn.Conv3d if D == 3 else nn.Conv1d
        self.q, self.k, self.v, self.out = [conv(in_ch, in_ch, 1) for _ in range(4)]
        self.norm = nn.GroupNorm(num_groups, in_ch)
        self.nonlin = Swish()
        self.sm = nn.Softmax(-1)

    def forward(self, x):
        B, C = x.shape[:2]
        q = self.q(x).reshape(B, C, -1)
        k = self.k(x).reshape(B, C, -1)
        v = self.v(x).reshape(B, C, -1)
        w = self.sm(torch.matmul(q.permute(0, 2, 1), k))
        h = torch.matmul(v, w.permute(0, 2, 1)).reshape(B, C, *x.shape[2:])
        h = self.out(h)
        x = self.nonlin(self.norm(h + x))
        return x


class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, attention=False,
                 dropout=0.1, with_se=False, with_se_relu=False, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        layers = [nn.Conv3d(in_channels, out_channels, kernel_size, 1, kernel_size // 2),
                  nn.GroupNorm(8, out_channels), Swish()]
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Conv3d(out_channels, out_channels, kernel_size, 1, kernel_size // 2),
                   nn.GroupNorm(8, out_channels),
                   _Attention(out_channels, 8) if attention else Swish()]
        if with_se:
            layers.append(SE3d(out_channels, use_relu=with_se_relu))
        self.voxel_layers = nn.Sequential(*layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        features, coords, femb, o_mask = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = F.trilinear_devoxelize(
            voxel_features, voxel_coords, self.resolution, self.training
        )
        fused = voxel_features + self.point_features(features)
        return fused, coords, femb, o_mask


# ───────────────── PointNet SA ─────────────────

class PointNetSAModule(nn.Module):
    def __init__(self, num_centers, radius, num_neighbors, in_channels, out_channels,
                 include_coordinates=True):
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
        features, coords, femb, o_mask = inputs
        centers_coords = F.furthest_point_sample(coords, self.num_centers)
        feats_list = []
        for grouper, mlp in zip(self.groupers, self.mlps):
            nf, gf = mlp(grouper(coords, centers_coords, femb, features))
            feats_list.append(nf.max(dim=-1).values)
        out = feats_list[0] if len(feats_list) == 1 else torch.cat(feats_list, 1)
        gf_out = gf.max(dim=-1).values if gf.shape[1] > 0 else gf
        return out, centers_coords, gf_out, o_mask


# ───────────────── Intertooth attention (DITA) ─────────────────

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def _normalization(channels):
    g = 32 if channels % 32 == 0 else (8 if channels % 8 == 0 else channels)
    return GroupNorm32(g, channels)


def _zero_module(m):
    for p in m.parameters():
        p.detach().zero_()
    return m


class RPENet(nn.Module):
    """거리(Δij) + time/slot 임베딩 → per-head RPE bias."""
    def __init__(self, channels, num_heads, time_embed_dim):
        super().__init__()
        self.embed_distances = nn.Linear(3, channels)
        self.embed_diffusion_time = nn.Linear(time_embed_dim, channels)
        self.silu = nn.SiLU()
        self.out = _zero_module(nn.Linear(channels, channels))
        self.channels = channels
        self.num_heads = num_heads

    def forward(self, femb, relative_distances):
        distance_embs = torch.stack([
            torch.log(1 + relative_distances.clamp(min=0)),
            torch.log(1 + (-relative_distances).clamp(min=0)),
            (relative_distances == 0).float(),
        ], dim=-1)                                          # (B,T,T,3)
        B, T, _ = relative_distances.shape
        C = self.channels
        emb = self.embed_diffusion_time(femb).view(B, T, 1, C) + self.embed_distances(distance_embs)
        # emb broadcasts to (B,T,T,C); view → (B,T,T,H,F)
        return self.out(self.silu(emb)).view(B, T, T, self.num_heads, C // self.num_heads)


class _RPE(nn.Module):
    def __init__(self, channels, num_heads, time_embed_dim, use_rpe_net=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.use_rpe_net = use_rpe_net
        if use_rpe_net:
            self.rpe_net = RPENet(channels, num_heads, time_embed_dim)

    def get_R(self, pairwise_distances, femb):
        # → (B, T, T, H, F)
        return self.rpe_net(femb, pairwise_distances)

    def forward(self, x, pairwise_distances, femb, mode):
        R = self.get_R(pairwise_distances, femb)            # (B,T,T,H,F)
        if mode == "qk":
            return torch.einsum("bdhtf,btshf->bdhts", x, R)
        elif mode == "v":
            return torch.einsum("bdhts,btshf->bdhtf", x, R)


class RPEAttention(nn.Module):
    def __init__(self, channels, num_heads, time_embed_dim, use_rpe_net=True,
                 use_rpe_q=True, use_rpe_k=True, use_rpe_v=True, mask_mode='official'):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = _zero_module(nn.Linear(channels, channels))
        self.norm = _normalization(channels)
        self.mask_mode = mask_mode   # 'official'(atlas) | 'context'(missing→present 허용)
        self.rpe_q = _RPE(channels, num_heads, time_embed_dim, use_rpe_net) if use_rpe_q else None
        self.rpe_k = _RPE(channels, num_heads, time_embed_dim, use_rpe_net) if use_rpe_k else None
        self.rpe_v = _RPE(channels, num_heads, time_embed_dim, use_rpe_net) if use_rpe_v else None

    def forward(self, x, femb, dentition_fdi_indices, attn_mask=None):
        B, D, C, T = x.shape
        x = x.reshape(B * D, C, T)
        x = self.norm(x).view(B, D, C, T)
        x = x.permute(0, 1, 3, 2)                            # B,D,T,C
        qkv = self.qkv(x).reshape(B, D, T, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)                  # 3,B,D,H,T,F
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)                       # B,D,H,T,T
        pairwise = (dentition_fdi_indices.unsqueeze(-1) - dentition_fdi_indices.unsqueeze(-2))  # B,T,T
        if self.rpe_k is not None:
            attn = attn + self.rpe_k(q, pairwise, femb, "qk")
        if self.rpe_q is not None:
            attn = attn + self.rpe_q(k * self.scale, pairwise, femb, "qk").transpose(-1, -2)

        if attn_mask is not None:
            if self.mask_mode == 'context':
                # missing(query) 가 present(key=context) 를 보도록 허용.
                # present query 는 present key 만 (오염 방지).
                allowed = attn_mask.view(B, 1, T) + (1 - attn_mask.view(B, T, 1)) * (1 - attn_mask.view(B, 1, T))
            else:
                allowed = attn_mask.view(B, 1, T) * attn_mask.view(B, T, 1) \
                    + (1 - attn_mask.view(B, 1, T)) * (1 - attn_mask.view(B, T, 1))
            inf_mask = (1 - allowed) * 1e9
            attn = attn - inf_mask.view(B, 1, 1, T, T)
        attn = torch.softmax(attn.float(), dim=-1).type(q.dtype)

        out = attn @ v                                       # B,D,H,T,F
        if self.rpe_v is not None:
            out = out + self.rpe_v(attn, pairwise, femb, "v")
        out = out.permute(0, 1, 3, 2, 4).reshape(B, D, T, C)  # B,D,T,C
        out = self.proj_out(out)
        x = x + out
        x = x.permute(0, 1, 3, 2)                            # B,D,C,T
        return x


class IntertoothAttentionBlock(nn.Module):
    """치아 간 어텐션. (BT,C,P) → (B,P,C,T) 로 reshape 후 T 축 어텐션(포인트별)."""
    def __init__(self, channels, num_heads, use_rpe_net, time_embed_dim=None, mask_mode='official'):
        super().__init__()
        self.intertooth_attention = RPEAttention(
            channels=channels, num_heads=num_heads,
            time_embed_dim=time_embed_dim, use_rpe_net=use_rpe_net, mask_mode=mask_mode,
        )

    def forward(self, x, attn_mask, femb, T, dentition_fdi_indices=None):
        BT, C, P = x.shape
        B = BT // T
        x = x.view(B, T, C, P).permute(0, 3, 2, 1)          # B,P,C,T
        x = self.intertooth_attention(
            x, femb, dentition_fdi_indices,
            attn_mask=attn_mask.view(B, T),
        )
        return x.permute(0, 3, 2, 1).reshape(BT, C, P)
