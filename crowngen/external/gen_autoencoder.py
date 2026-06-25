"""Generation denoiser U-Net (공식 cg_generation_module/model/autoencoder.py 포팅).

PVD 스타일 PVCNN2 encoder(4 SA)-decoder(4 FP) + inter-tooth 어텐션.
조건: time emb + boundary emb(Transformer 가공) → temb 으로 각 레이어/어텐션에 주입.
boundary(modules.py)의 RPE/IntertoothAttention 을 FactorizedAttention 으로 재사용.
"""
import functools
import torch
import torch.nn as nn

from .modules import IntertoothAttentionBlock, SharedMLP, Swish
from .gen_modules import GenPVConv, GenPointNetSAModule, GenPointNetFPModule, Transformer, LayerNorm
from .pvcnn import create_pointnet_components, create_mlp_components

__all__ = ['PVCNN2', 'PVCNN2Base']


class FactorizedAttentionBlock(IntertoothAttentionBlock):
    """gen 용: temb(time+bound) 사용, attn_mask=ones(전부 참여). 부모(IntertoothAttention) 재사용."""
    def forward(self, x, attn_mask, temb, T, dentition_fdi_indices=None, **kwargs):
        return super().forward(x, attn_mask, temb, T, dentition_fdi_indices)


def create_pointnet2_sa_components(sa_blocks, extra_feature_channels, embed_dim=64, use_att=False,
                                   dropout=0.1, with_se=False, normalize=True, eps=0,
                                   width_multiplier=1, voxel_resolution_multiplier=1, voxel_attention=False):
    width_mult = width_multiplier
    voxel_res_mult = voxel_resolution_multiplier
    in_channels = extra_feature_channels + 3
    sa_layers, sa_in_channels = [], []
    stage_idx = 0
    for conv_configs, sa_configs in sa_blocks:
        conv_block_idx = 0
        sa_in_channels.append(in_channels)
        stage_blocks = []
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(width_mult * out_channels)
            for block_in_stage in range(num_blocks):
                attention = (stage_idx + 1) % 2 == 0 and stage_idx > 0 and voxel_attention and block_in_stage == 0
                block = (SharedMLP if voxel_resolution is None else functools.partial(
                    GenPVConv, kernel_size=3, resolution=int(voxel_res_mult * voxel_resolution),
                    attention=attention, dropout=dropout, with_se=with_se and not attention,
                    with_se_relu=True, normalize=normalize, eps=eps))
                if stage_idx == 0:
                    stage_blocks.append(block(in_channels, out_channels))
                elif conv_block_idx == 0:
                    stage_blocks.append(block(in_channels + embed_dim, out_channels))
                in_channels = out_channels
                conv_block_idx += 1
            extra_feature_channels = in_channels
        num_centers, radius, num_neighbors, out_channels = sa_configs
        scaled = [[int(width_mult * x) for x in oc] if isinstance(oc, (list, tuple))
                  else int(width_mult * oc) for oc in out_channels]
        out_channels = scaled
        sa_block = functools.partial(GenPointNetSAModule, num_centers=num_centers,
                                     radius=radius, num_neighbors=num_neighbors)
        sa_input_channels = extra_feature_channels + (embed_dim if conv_block_idx == 0 else 0)
        stage_blocks.append(sa_block(in_channels=sa_input_channels, out_channels=out_channels))
        stage_idx += 1
        in_channels = extra_feature_channels = stage_blocks[-1].out_channels
        sa_layers.append(stage_blocks[0] if len(stage_blocks) == 1 else nn.Sequential(*stage_blocks))
    return sa_layers, sa_in_channels, in_channels


def create_pointnet2_fp_modules(fp_blocks, in_channels, sa_in_channels, embed_dim=64, use_att=False,
                                dropout=0.1, with_se=False, normalize=True, eps=0,
                                width_multiplier=1, voxel_resolution_multiplier=1, voxel_attention=False):
    width_mult = width_multiplier
    voxel_res_mult = voxel_resolution_multiplier
    fp_layers = []
    stage_idx = 0
    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        blocks = []
        out_channels = tuple(int(width_mult * oc) for oc in fp_configs)
        blocks.append(GenPointNetFPModule(
            in_channels=in_channels + sa_in_channels[-1 - fp_idx] + embed_dim, out_channels=out_channels))
        in_channels = out_channels[-1]
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(width_mult * out_channels)
            for block_in_stage in range(num_blocks):
                attention = stage_idx % 2 == 0 and stage_idx < len(fp_blocks) - 1 and voxel_attention and block_in_stage == 0
                block = (SharedMLP if voxel_resolution is None else functools.partial(
                    GenPVConv, kernel_size=3, resolution=int(voxel_res_mult * voxel_resolution),
                    attention=attention, dropout=dropout, with_se=with_se and not attention,
                    with_se_relu=True, normalize=normalize, eps=eps))
                blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
        fp_layers.append(blocks[0] if len(blocks) == 1 else nn.Sequential(*blocks))
        stage_idx += 1
    return fp_layers, in_channels


class PVCNN2Base(nn.Module):
    """Generation denoiser U-Net 본체."""
    sa_blocks = []   # subclass 에서 정의
    fp_blocks = []

    def __init__(self, num_classes, embed_dim, use_att, dropout=0.1,
                 extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1, mask_mode='official'):
        super().__init__()
        self.embed_dim = embed_dim
        self.extra_feature_channels = extra_feature_channels
        self.in_channels = extra_feature_channels + 3
        self.mask_mode = mask_mode

        self.fdi_embedding = nn.Embedding(28, 8)
        sa_layers, sa_in_channels, channels_sa = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True,
            embed_dim=embed_dim, use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.sa_layers = nn.ModuleList(sa_layers)
        sa_in_channels[0] = extra_feature_channels

        # 모든 attention 을 mask_mode 적용(기본 official=전부 참여해도 무방; gen 은 attn_mask=ones)
        self.sa_att_dict = nn.ModuleDict({
            idx: FactorizedAttentionBlock(int(channels_sa // d), num_heads=4, use_rpe_net=True,
                                          time_embed_dim=embed_dim, mask_mode=mask_mode)
            for idx, d in {'1': 8, '2': 4, '3': 2}.items()})
        self.global_att = (None if not use_att else FactorizedAttentionBlock(
            channels_sa, num_heads=4, use_rpe_net=True, time_embed_dim=embed_dim, mask_mode=mask_mode))
        self.fp_att_dict = nn.ModuleDict({
            idx: FactorizedAttentionBlock(int(channels_sa // d), num_heads=4, use_rpe_net=True,
                                          time_embed_dim=embed_dim, mask_mode=mask_mode)
            for idx, d in {'0': 2, '1': 2, '2': 4}.items()})

        fp_layers, channels_fp = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa, sa_in_channels=sa_in_channels,
            with_se=True, embed_dim=embed_dim, use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.fp_layers = nn.ModuleList(fp_layers)

        layers, _, _ = create_mlp_components(in_channels=channels_fp,
                                             out_channels=[128, 0.5, num_classes], classifier=True, dim=2,
                                             width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

        self.embedf = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LeakyReLU(0.1, True),
                                    nn.Linear(embed_dim, embed_dim))
        self.bound_embedding = nn.Linear(5, embed_dim)
        self.bound_transformer = Transformer(width=embed_dim, layers=4, heads=8)
        self.bound_final_ln = LayerNorm(embed_dim)

    def get_timestep_embedding(self, timesteps, device):
        half = self.embed_dim // 2
        emb = torch.tensor([10000 ** (-(i / max(half - 1, 1))) for i in range(half)],
                           device=device, dtype=torch.float32)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), 'constant', 0)
        return emb

    def forward(self, xt, t, return_attn_weights, x0, l_mask, o_mask, bound):
        B, nT, nD, nP = xt.shape
        device = xt.device
        t = t.view(B, 1).expand(B, nT).reshape(B * nT)
        fdi_idx = torch.arange(28, device=device).unsqueeze(0).repeat(B, 1)
        attn_mask = torch.ones_like(o_mask)                       # gen: 전부 참여
        fdi_emb = self.fdi_embedding(fdi_idx)                     # (B,28,8)

        bound_emb = self.bound_embedding(bound)                   # (B,28,embed_dim)
        bound_emb = self.bound_transformer(bound_emb)
        bound_emb = self.bound_final_ln(bound_emb).reshape(B * nT, self.embed_dim)
        temb_raw = self.embedf(self.get_timestep_embedding(t, device)) + bound_emb   # (B*nT,embed_dim)

        temb = temb_raw[:, :, None].expand(-1, -1, nP)            # (B*nT,embed_dim,P)
        obs = torch.ones_like(xt[:, :, :1, :]) * o_mask.view(B, nT, 1, 1)
        fdi_pts = fdi_emb.unsqueeze(3).repeat(1, 1, 1, nP)        # (B,28,8,P)
        x = torch.cat([xt * l_mask.view(B, nT, 1, 1) + x0 * o_mask.view(B, nT, 1, 1), fdi_pts, obs], dim=2)
        x = x.reshape(B * nT, nD + self.extra_feature_channels, nP)
        coords, features = x[:, :3, :].contiguous(), x
        coords_list, in_features_list = [], []
        for sa_idx, layer in enumerate(self.sa_layers):
            in_features_list.append(features); coords_list.append(coords)
            if str(sa_idx) in self.sa_att_dict:
                features = self.sa_att_dict[str(sa_idx)](features, attn_mask, temb_raw, nT, fdi_idx)
            inp = features if sa_idx == 0 else torch.cat([features, temb], dim=1)
            features, coords, temb = layer((inp, coords, temb))
        in_features_list[0] = x[:, 3:, :].contiguous()
        if self.global_att is not None:
            features = self.global_att(features, attn_mask, temb_raw, nT, fdi_idx)
        for fp_idx, layer in enumerate(self.fp_layers):
            jump_coords = coords_list[-1 - fp_idx]; jump_feats = in_features_list[-1 - fp_idx]
            features, coords, temb = layer((jump_coords, coords, torch.cat([features, temb], dim=1), jump_feats, temb))
            if str(fp_idx) in self.fp_att_dict:
                features = self.fp_att_dict[str(fp_idx)](features, attn_mask, temb_raw, nT, fdi_idx)
        out = self.classifier(features).view(B, nT, nD, nP)
        return out, None


class PVCNN2(PVCNN2Base):
    """공식 gen 설정 (sa_blocks/fp_blocks)."""
    sa_blocks = [
        ((32, 2, 32), (512, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_classes, embed_dim, use_att, dropout, extra_feature_channels=9,
                 width_multiplier=1.0, voxel_resolution_multiplier=1.0, mask_mode='official'):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim, use_att=use_att, dropout=dropout,
                         extra_feature_channels=extra_feature_channels, mask_mode=mask_mode,
                         width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
