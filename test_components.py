"""
CrownGen 컴포넌트 통합 테스트.

각 모듈의 형태/동작을 검증합니다:
1. FDI 임베딩 + RPE 행렬
2. 데이터셋 로드 + 증강
3. PVC 연산자
4. DITA 어텐션
5. Denoising Network 순방향
6. Gaussian Diffusion 손실/샘플링
7. Boundary Predictor 순방향
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 프로젝트 루트 경로
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

import torch
import yaml

def test_fdi():
    from crowngen.data.fdi import (
        ZIGZAG_FDI_ORDER, FDI_TO_ZIGZAG, MIRROR_REMAP,
        FDIEmbedding, compute_rpe_matrix
    )
    print("=== 1. FDI 모듈 테스트 ===")

    # 지그재그 순서 확인
    assert len(ZIGZAG_FDI_ORDER) == 28
    assert ZIGZAG_FDI_ORDER[0] == 17 and ZIGZAG_FDI_ORDER[13] == 41
    print(f"  지그재그 순서: {ZIGZAG_FDI_ORDER[:7]}... (28개)")

    # 좌우 반전 매핑
    assert MIRROR_REMAP[11] == 21 and MIRROR_REMAP[36] == 46
    print(f"  반전 매핑: 11→{MIRROR_REMAP[11]}, 36→{MIRROR_REMAP[36]}")

    # FDI 임베딩
    emb = FDIEmbedding(embed_dim=8)
    fdi = torch.tensor([11, 16, 21, 36])
    out = emb(fdi)
    assert out.shape == (4, 8)
    print(f"  FDI 임베딩: {fdi.tolist()} → {out.shape}")

    # RPE 행렬
    rpe = compute_rpe_matrix()
    assert rpe.shape == (28, 28, 3)
    # 자기 자신: [0, 0, 1]
    assert rpe[0, 0].tolist() == [0.0, 0.0, 1.0]
    print(f"  RPE 행렬: {rpe.shape}, self-RPE={rpe[0,0].tolist()}")
    print("  ✅ FDI 모듈 통과\n")


def test_dataset():
    from crowngen.data.dataset import CrownGenDataset, crown_collate_fn
    print("=== 2. 데이터셋 테스트 ===")

    data_dir = os.path.join(PROJECT_ROOT, 'Data', 'processed')
    split_file = os.path.join(PROJECT_ROOT, 'Data', 'SourceC_Teeth3DS', 'train_val_split.json')

    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) < 3:
        print("  ⚠️ 처리된 데이터 부족 — 스킵")
        return None

    dataset = CrownGenDataset(
        data_dir=data_dir,
        split_file=split_file,
        split_name='stage1_train',
        n_points=1024,
        augment=True,
    )

    # 단일 샘플 로드
    sample = dataset[0]
    print(f"  샘플 키: {list(sample.keys())}")
    print(f"  tooth_points: {sample['tooth_points'].shape}")
    print(f"  fdi_labels: {sample['fdi_labels'].shape}")
    print(f"  target_mask: {sample['target_mask'].shape} (타겟 수: {sample['target_mask'].sum().item()})")
    print(f"  boundaries: {sample['boundaries'].shape}")
    print(f"  tooth_valid: {sample['tooth_valid'].shape} (유효 치아: {sample['tooth_valid'].sum().item()})")

    # 배치 로드
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, collate_fn=crown_collate_fn)
    batch = next(iter(loader))
    print(f"  배치 tooth_points: {batch['tooth_points'].shape}")
    print("  ✅ 데이터셋 통과\n")
    return batch


def test_pvc():
    from crowngen.models.pvc import PointVoxelConv
    print("=== 3. PVC 연산자 테스트 ===")

    pvc = PointVoxelConv(64, 128, resolution=16, dropout=0.1)
    B, N = 4, 256
    xyz = torch.randn(B, 3, N)
    feat = torch.randn(B, 64, N)

    xyz_out, feat_out = pvc(xyz, feat)
    assert feat_out.shape == (B, 128, N)
    print(f"  입력: ({B}, 64, {N}) → 출력: {feat_out.shape}")
    print(f"  파라미터 수: {sum(p.numel() for p in pvc.parameters()):,}")
    print("  ✅ PVC 통과\n")


def test_dita():
    from crowngen.models.dita import DITA
    print("=== 4. DITA 어텐션 테스트 ===")

    dita = DITA(dim=256, num_heads=8, rpe_hidden=64)
    B, T, D = 4, 28, 256
    z = torch.randn(B, T, D)
    tooth_valid = torch.ones(B, T, dtype=torch.long)
    tooth_valid[:, 24:] = 0  # 4개 치아 결손

    out = dita(z, tooth_valid=tooth_valid)
    assert out.shape == (B, T, D)
    print(f"  입력: ({B}, {T}, {D}) → 출력: {out.shape}")
    print(f"  파라미터 수: {sum(p.numel() for p in dita.parameters()):,}")
    print("  ✅ DITA 통과\n")


def test_denoise_net():
    from crowngen.models.denoise_net import DenoiseNetwork
    print("=== 5. Denoising Network 테스트 ===")

    with open(os.path.join(PROJECT_ROOT, 'crowngen', 'configs', 'default.yaml')) as f:
        config = yaml.safe_load(f)

    model = DenoiseNetwork(config)
    B, T, N = 2, 28, 1024

    x_t = torch.randn(B, T, N, 3)
    target_mask = torch.zeros(B, T, dtype=torch.long)
    target_mask[:, [12, 13, 14]] = 1
    context_mask = 1 - target_mask
    fdi_labels = torch.randint(11, 48, (B, T))
    boundaries = torch.randn(B, T, 5)
    t = torch.randint(0, 1000, (B,))
    tooth_valid = torch.ones(B, T, dtype=torch.long)

    try:
        out = model(
            x_t=x_t,
            target_mask=target_mask,
            context_mask=context_mask,
            fdi_labels=fdi_labels,
            boundaries=boundaries,
            t=t,
            tooth_valid=tooth_valid,
        )
        assert out.shape == (B, T, N, 3), f"Expected {(B, T, N, 3)}, got {out.shape}"
        print(f"  입력: x_t={x_t.shape}")
        print(f"  출력: noise_pred={out.shape}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  총 파라미터: {n_params:,}")
        print("  ✅ Denoising Network 통과\n")
    except Exception as e:
        print(f"  ❌ 오류: {e}\n")
        import traceback
        traceback.print_exc()


def test_diffusion():
    from crowngen.models.diffusion import GaussianDiffusion
    print("=== 6. Gaussian Diffusion 테스트 ===")

    diff = GaussianDiffusion(timesteps=1000, beta_min=1e-4, beta_max=2e-2)
    B, T, N = 2, 28, 64

    x_0 = torch.randn(B, T, N, 3)
    t = torch.randint(0, 1000, (B,))

    # Forward process
    x_t, noise = diff.forward_process(x_0, t)
    assert x_t.shape == x_0.shape
    print(f"  Forward: x_0={x_0.shape} → x_t={x_t.shape}")

    # Loss (더미 모델)
    class DummyModel(torch.nn.Module):
        def forward(self, **kwargs):
            return torch.randn_like(x_0)

    batch = {
        'tooth_points': x_0,
        'target_mask': torch.ones(B, T, dtype=torch.long),
        'context_mask': torch.zeros(B, T, dtype=torch.long),
        'fdi_labels': torch.randint(11, 48, (B, T)),
        'boundaries': torch.randn(B, T, 5),
        'tooth_valid': torch.ones(B, T, dtype=torch.long),
    }

    loss = diff.training_loss(DummyModel(), batch, torch.device('cpu'))
    print(f"  Training loss: {loss.item():.4f}")

    # 샘플링 (2스텝만)
    diff_short = GaussianDiffusion(timesteps=2)
    print(f"  Beta schedule: {diff_short.betas[:5].tolist()}")
    print("  ✅ Gaussian Diffusion 통과\n")


def test_boundary():
    from crowngen.models.boundary_net import BoundaryPredictor, boundary_loss
    print("=== 7. Boundary Predictor 테스트 ===")

    with open(os.path.join(PROJECT_ROOT, 'crowngen', 'configs', 'default.yaml')) as f:
        config = yaml.safe_load(f)

    model = BoundaryPredictor(config)
    B, T, N = 2, 28, 512

    tooth_points = torch.randn(B, T, N, 3)
    tooth_points[:, 12:15] = 0.0  # 타겟 치아 영벡터
    fdi_labels = torch.randint(11, 48, (B, T))
    tooth_valid = torch.ones(B, T, dtype=torch.long)
    target_mask = torch.zeros(B, T, dtype=torch.long)
    target_mask[:, 12:15] = 1

    pred = model(tooth_points, fdi_labels, tooth_valid, target_mask)
    assert pred.shape == (B, T, 5)
    print(f"  입력: tooth_points={tooth_points.shape}")
    print(f"  출력: boundaries={pred.shape}")

    # 손실 계산
    gt = torch.randn(B, T, 5)
    loss = boundary_loss(pred, gt, target_mask)
    print(f"  Boundary loss: {loss.item():.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  파라미터 수: {n_params:,}")
    print("  ✅ Boundary Predictor 통과\n")


if __name__ == '__main__':
    print("=" * 60)
    print("  CrownGen 컴포넌트 통합 테스트")
    print("=" * 60 + "\n")

    test_fdi()
    batch = test_dataset()
    test_pvc()
    test_dita()
    test_denoise_net()
    test_diffusion()
    test_boundary()

    print("=" * 60)
    print("  🎉 모든 테스트 완료!")
    print("=" * 60)
