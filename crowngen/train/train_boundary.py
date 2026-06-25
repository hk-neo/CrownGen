"""
Boundary Prediction Module 학습 스크립트.

논문 설정:
  - 에폭: 1000
  - 옵티마이저: Adam
  - 학습률: 3×10⁻⁴ → 3×10⁻⁶ (Cosine Annealing)
  - 드롭아웃: 0.3
  - 손실: Smooth L1 (타겟 치아에만)

사용법:
  python -m crowngen.train.train_boundary \
      --config crowngen/configs/default.yaml \
      --output_dir runs/boundary
"""

import argparse
import os
import sys
import time
import json
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from crowngen.data.dataset import CrownGenDataset, crown_collate_fn
from crowngen.models.boundary_net import BoundaryPredictor, boundary_loss
from crowngen.losses.smooth_l1 import smooth_l1_loss


def parse_args():
    parser = argparse.ArgumentParser(description='CrownGen Boundary Prediction 학습')
    parser.add_argument('--config', type=str, default='crowngen/configs/default.yaml')
    parser.add_argument('--output_dir', type=str, default='runs/boundary')
    parser.add_argument('--resume', type=str, default=None, help='체크포인트 경로')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb', action='store_true', help='Weights & Biases 로깅')
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, device, epoch):
    """한 에폭 학습."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        # 데이터를 디바이스로 이동
        tooth_points = batch['tooth_points'].to(device)  # (B, 28, N, 3)
        fdi_labels = batch['fdi_labels'].to(device)
        tooth_valid = batch['tooth_valid'].to(device)
        target_mask = batch['target_mask'].to(device)
        gt_boundaries = batch['boundaries'].to(device)

        # 타겟 치아의 포인트를 영벡터로 마스킹 (Boundary 모듈 입력 조건)
        masked_points = tooth_points.clone()
        for b in range(masked_points.shape[0]):
            for t in range(28):
                if target_mask[b, t] == 1:
                    masked_points[b, t] = 0.0

        # 순방향
        pred_boundaries = model(masked_points, fdi_labels, tooth_valid, target_mask)

        # 손실 계산
        loss = boundary_loss(pred_boundaries, gt_boundaries, target_mask)

        # 역방향
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device):
    """검증."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        tooth_points = batch['tooth_points'].to(device)
        fdi_labels = batch['fdi_labels'].to(device)
        tooth_valid = batch['tooth_valid'].to(device)
        target_mask = batch['target_mask'].to(device)
        gt_boundaries = batch['boundaries'].to(device)

        masked_points = tooth_points.clone()
        for b in range(masked_points.shape[0]):
            for t in range(28):
                if target_mask[b, t] == 1:
                    masked_points[b, t] = 0.0

        pred_boundaries = model(masked_points, fdi_labels, tooth_valid, target_mask)
        loss = boundary_loss(pred_boundaries, gt_boundaries, target_mask)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    args = parse_args()

    # 설정 로드
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 출력 디렉토리
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)

    # 디바이스
    device = torch.device(args.device)
    print(f"Device: {device}")

    # 데이터셋
    train_cfg = config['training']['boundary']
    data_cfg = config['data']

    print("데이터셋 로드 중...")
    train_dataset = CrownGenDataset(
        data_dir=data_cfg['processed_dir'],
        split_file=data_cfg['split_file'],
        split_name='stage1_train',
        n_points=data_cfg.get('n_points_boundary', 512),
        mask_range=tuple(data_cfg['mask_range']),
        augment=True,
        is_boundary=True,
    )

    val_dataset = CrownGenDataset(
        data_dir=data_cfg['processed_dir'],
        split_file=data_cfg['split_file'],
        split_name='stage1_val',
        n_points=data_cfg.get('n_points_boundary', 512),
        mask_range=tuple(data_cfg['mask_range']),
        augment=False,
        is_boundary=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        collate_fn=crown_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        collate_fn=crown_collate_fn,
        pin_memory=True,
    )

    # 모델
    _raw_model = BoundaryPredictor(config).to(device)
    model = _raw_model

    # Multi-GPU (DataParallel)
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel: {n_gpus} GPUs")
        # 배치 사이즈를 GPU 수에 맞게 스케일
        train_cfg['batch_size'] = train_cfg['batch_size'] * n_gpus

    n_params = sum(p.numel() for p in model.parameters())
    print(f"BoundaryPredictor 파라미터: {n_params:,}")
    print(f"배치 사이즈: {train_cfg['batch_size']}")

    # 옵티마이저 (경량 weight decay: 비정규화 타깃으로 인한 과적합 보험.
    # 정규화된 데이터가 주효하지만, 1차 Train 0.67/Val 1.5 격차 방어용)
    weight_decay = train_cfg.get('weight_decay', 1e-4)
    optimizer = Adam(model.parameters(), lr=train_cfg['lr'], weight_decay=weight_decay)

    # 스케줄러: Cosine Annealing (lr_min까지 감소)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg['epochs'],
        eta_min=train_cfg['lr_min'],
    )

    # WandB (선택)
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project='crowngen',
                name='boundary_prediction',
                config=config,
            )
        except ImportError:
            print("wandb가 설치되지 않음 — 로깅 없이 진행")

    # 체크포인트 복원
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state_dict = ckpt['model_state_dict']
        # DataParallel 저장 체크포인트의 module. prefix 제거
        if any(k.startswith('module.') for k in state_dict):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        _raw_model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"체크포인트 복원: epoch {start_epoch}, val_loss {best_val_loss:.4f}")

    # 학습 루프
    print(f"\n{'='*60}")
    print(f"  Boundary Prediction 학습 시작")
    print(f"  에폭: {train_cfg['epochs']}, LR: {train_cfg['lr']} → {train_cfg['lr_min']}")
    print(f"  배치: {train_cfg['batch_size']}, 학습: {len(train_dataset)}, 검증: {len(val_dataset)}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, train_cfg['epochs']):
        t0 = time.time()

        # 학습
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # 검증
        val_loss = validate(model, val_loader, device)

        # 스케줄러 업데이트
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        # 로깅
        print(f"Epoch {epoch+1}/{train_cfg['epochs']} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {lr:.2e} | Time: {elapsed:.1f}s")

        if wandb_run:
            import wandb
            wandb.log({
                'train/loss': train_loss,
                'val/loss': val_loss,
                'train/lr': lr,
                'train/epoch': epoch,
            })

        # 최적 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': _raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'config': config,
            }, output_dir / 'checkpoints' / 'best.pt')
            print(f"  → 최적 모델 저장 (val_loss: {val_loss:.4f})")

        # 주기적 체크포인트
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': _raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'config': config,
            }, output_dir / 'checkpoints' / f'epoch_{epoch+1}.pt')

    # 최종 모델 저장
    torch.save({
        'epoch': train_cfg['epochs'] - 1,
        'model_state_dict': _raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'config': config,
    }, output_dir / 'checkpoints' / 'final.pt')

    print(f"\n{'='*60}")
    print(f"  학습 완료! Best val_loss: {best_val_loss:.4f}")
    print(f"  체크포인트: {output_dir / 'checkpoints'}")
    print(f"{'='*60}")

    if wandb_run:
        wandb.finish()


if __name__ == '__main__':
    main()
