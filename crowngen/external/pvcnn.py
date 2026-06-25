"""CrownGen boundary PVCNN2 모델 (공식 cg_boundary_prediction_module/model/pvcnn.py 포팅).

구조/블록 구성은 공식에 일치. CUDA ops → external.functional.
"""
import functools
import torch
import torch.nn as nn

from .modules import (
    SharedMLP, PVConv, PointNetSAModule, Swish, IntertoothAttentionBlock,
)

__all__ = ['PVCNN2', 'BoundEncoder']


def _linear_gn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.GroupNorm(8, out_channels), Swish())


def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    if dim == 1:
        block = _linear_gn_relu
    else:
        block = SharedMLP
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels
    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(width_multiplier * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        layers.append(nn.Linear(in_channels, out_channels[-1]) if classifier
                      else _linear_gn_relu(in_channels, int(width_multiplier * out_channels[-1])))
    else:
        layers.append(nn.Linear(in_channels, out_channels[-1]) if False else
                      (nn.Conv1d(in_channels, out_channels[-1], 1) if classifier
                       else SharedMLP(in_channels, int(width_multiplier * out_channels[-1]))))
    final = out_channels[-1] if classifier else int(width_multiplier * out_channels[-1])
    return layers, final, final


def create_pointnet_components(blocks, in_channels, embed_dim, with_se=False, normalize=True, eps=0,
                               width_multiplier=1, voxel_resolution_multiplier=1):
    width_mult = width_multiplier
    voxel_res_mult = voxel_resolution_multiplier
    layers = []
    concat_channels = 0
    block_idx = 0
    for stage_idx, (out_channels, num_blocks, voxel_resolution) in enumerate(blocks):
        out_channels = int(width_mult * out_channels)
        for block_in_stage in range(num_blocks):
            attention = stage_idx % 2 == 0 and stage_idx > 0 and block_in_stage == 0
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = functools.partial(PVConv, kernel_size=3,
                                          resolution=int(voxel_res_mult * voxel_resolution),
                                          attention=attention, with_se=with_se, normalize=normalize, eps=eps)
            if block_idx == 0:
                layers.append(block(in_channels, out_channels))
            else:
                layers.append(block(in_channels + embed_dim, out_channels))
            in_channels = out_channels
            concat_channels += out_channels
            block_idx += 1
    return layers, in_channels, concat_channels


def create_pointnet2_sa_components(sa_blocks, extra_feature_channels, embed_dim=64, use_att=False,
                                   dropout=0.1, with_se=False, normalize=True, eps=0,
                                   width_multiplier=1, voxel_resolution_multiplier=1, voxel_attention=False):
    width_mult = width_multiplier
    voxel_res_mult = voxel_resolution_multiplier
    in_channels = extra_feature_channels + 3
    sa_layers, sa_in_channels = [], []
    stage_idx = 0
    num_centers = None

    for conv_configs, sa_configs in sa_blocks:
        conv_block_idx = 0
        sa_in_channels.append(in_channels)
        stage_blocks = []

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(width_mult * out_channels)
            for block_in_stage in range(num_blocks):
                # 복셀 내 self-attention(_Attention)은 4096² 행렬로 메모리 과다.
                # 공유 GPU 환경에서는 voxel_attention=False 로 끈다 (DITA 와 별개).
                attention = (stage_idx + 1) % 2 == 0 and stage_idx > 0 and voxel_attention and block_in_stage == 0
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(PVConv, kernel_size=3,
                                              resolution=int(voxel_res_mult * voxel_resolution),
                                              attention=attention, dropout=dropout,
                                              with_se=with_se and not attention, with_se_relu=True,
                                              normalize=normalize, eps=eps)
                if stage_idx == 0:
                    stage_blocks.append(block(in_channels, out_channels))
                elif conv_block_idx == 0:
                    stage_blocks.append(block(in_channels + embed_dim, out_channels))
                in_channels = out_channels
                conv_block_idx += 1
            extra_feature_channels = in_channels

        num_centers, radius, num_neighbors, out_channels = sa_configs
        scaled = []
        for oc in out_channels:
            scaled.append([int(width_mult * x) for x in oc] if isinstance(oc, (list, tuple))
                          else int(width_mult * oc))
        out_channels = scaled

        sa_block = functools.partial(PointNetSAModule, num_centers=num_centers,
                                     radius=radius, num_neighbors=num_neighbors)
        sa_input_channels = extra_feature_channels + (embed_dim if conv_block_idx == 0 else 0)
        stage_blocks.append(sa_block(in_channels=sa_input_channels, out_channels=out_channels,
                                     include_coordinates=True))
        stage_idx += 1
        in_channels = extra_feature_channels = stage_blocks[-1].out_channels
        sa_layers.append(stage_blocks[0] if len(stage_blocks) == 1 else nn.Sequential(*stage_blocks))

    num_centers = 1 if num_centers is None else num_centers
    return sa_layers, sa_in_channels, in_channels, num_centers


class PVCNN2(nn.Module):
    """Boundary prediction network (공식 PVCNN2). 입력 (B,28,3,P), o_mask (B,28)."""
    def __init__(self, output_dim, sa_blocks, embed_dim, use_att, dropout=0.1,
                 extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1,
                 mask_mode='official', voxel_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.extra_feature_channels = extra_feature_channels
        self.in_channels = extra_feature_channels + 3
        self.output_dim = output_dim

        self.fdi_embedding = nn.Embedding(num_embeddings=28, embedding_dim=embed_dim)

        sa_layers, _, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True,
            embed_dim=embed_dim, use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            voxel_attention=voxel_attention,
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        self.sa_att_dict = nn.ModuleDict({
            idx: IntertoothAttentionBlock(int(channels_sa_features // d), num_heads=4,
                                          use_rpe_net=True, time_embed_dim=embed_dim, mask_mode=mask_mode)
            for idx, d in {'1': 4, '2': 2}.items()
        })
        self.global_att = (IntertoothAttentionBlock(channels_sa_features, num_heads=4,
                                                    use_rpe_net=True, time_embed_dim=embed_dim,
                                                    mask_mode=mask_mode)
                           if use_att else None)

        final_layers, _, _ = create_pointnet_components(
            [(channels_sa_features, 1, 8)], channels_sa_features, embed_dim, with_se=True,
            normalize=True, width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
        )
        self.final_pointnet_layer = nn.ModuleList(final_layers)
        self.bound_fc = nn.Sequential(
            nn.Linear(channels_sa_features, channels_sa_features // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(channels_sa_features // 2, output_dim),
        )

    def forward(self, dentition_points, o_mask):
        # dentition_points (B, nT=28, 3, P), o_mask (B, nT) 1=present 0=missing
        B, nT, nD, nP = dentition_points.shape
        device = dentition_points.device
        fdi_idx = torch.arange(28, device=device).unsqueeze(0).repeat(B, 1)
        attn_mask = o_mask
        fdi_emb = self.fdi_embedding(fdi_idx).reshape(B * nT, self.embed_dim)   # (B*nT, embed_dim)
        femb = fdi_emb[:, :, None].expand(-1, -1, nP)                           # (B*nT, embed_dim, P)

        obs_indicator = torch.ones_like(dentition_points[:, :, :1, :]) * o_mask.view(B, nT, 1, 1)
        x = torch.cat([dentition_points * o_mask.view(B, nT, 1, 1), obs_indicator], dim=2)  # (B,28,4,P)
        x = x.reshape(B * nT, nD + self.extra_feature_channels, nP)
        o_mask_flat = o_mask.reshape(B * nT)

        coords = x[:, :3, :].contiguous()
        features = x

        for sa_idx, layer in enumerate(self.sa_layers):
            s = str(sa_idx)
            if s in self.sa_att_dict:
                features = self.sa_att_dict[s](features, attn_mask, fdi_emb, nT, fdi_idx)
            inp = features if sa_idx == 0 else torch.cat([features, femb], dim=1)
            features, coords, femb_out, o_mask_flat = layer((inp, coords, femb, o_mask_flat))
            femb = femb_out

        if self.global_att is not None:
            features = self.global_att(features, attn_mask, fdi_emb, nT, fdi_idx)

        final_out, _, _, _ = self.final_pointnet_layer[0]((features, coords, femb, o_mask_flat))
        final_out = final_out.mean(dim=2)                          # (B*nT, C)
        out = self.bound_fc(final_out).view(B, nT, self.output_dim)
        return out


class BoundEncoder(nn.Module):
    """Boundary prediction wrapper (공식 bound_encoder.py). sa_blocks 는 논문/공식 기본값.

    mask_mode: 'official'(atlas, 공식) | 'context'(missing→present 허용, 논문 prose 부합).
    """
    def __init__(self, output_dim=5, dropout=0.3, max_missing_teeth=6, mask_mode='official', voxel_attention=False):
        super().__init__()
        self.max_missing_teeth = max_missing_teeth
        self.output_dim = output_dim
        sa_blocks = [
            ((16, 2, 32), (128, 0.1, 32, (32, 64))),
            ((32, 3, 16), (64, 0.2, 32, (64, 128))),
            ((64, 3, 8), (16, 0.4, 32, (128, 256))),
        ]
        self.model = PVCNN2(output_dim=output_dim, sa_blocks=sa_blocks, embed_dim=64,
                            use_att=True, extra_feature_channels=1, dropout=dropout,
                            mask_mode=mask_mode, voxel_attention=voxel_attention)

    def forward(self, dentition_points, exist_mask):
        return self.model(dentition_points, exist_mask)

    @staticmethod
    def loss(pred_bound, gt_bound, missing_mask):
        """Smooth L1, missing 치아에만. missing_mask (B,nT) 1=missing."""
        import torch.nn.functional as Fn
        l = Fn.smooth_l1_loss(pred_bound, gt_bound, reduction='none')      # (B,nT,5)
        l = l * missing_mask.unsqueeze(-1)
        return l.sum() / missing_mask.sum().clamp(min=1)
