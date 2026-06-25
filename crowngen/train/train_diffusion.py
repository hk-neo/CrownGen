"""
Diffusion Generative Module 학습 스크립트 (Stage 1 & 2 공용) — DDP 지원.

Stage 1: 완전 치열 스캔으로 초기 모델 학습
  - 에폭: 3000, LR: 4e-5, 1500 에폭에서 0.4 감쇠
Stage 2: 확장 데이터셋으로 파인튜닝
  - 에폭: 2400 (또는 3000), LR: 2e-5, 800 에폭마다 0.45 감쇠

사용법 (단일 GPU):
  python -m crowngen.train.train_diffusion \
      --config crowngen/configs/default.yaml \
      --stage 1 \
      --output_dir runs/diffusion_stage1

사용법 (DDP, N개 GPU):
  torchrun --nproc_per_node=N -m crowngen.train.train_diffusion \
      --config crowngen/configs/default.yaml \
      --stage 1 \
      --boundary_ckpt runs/boundary/checkpoints/best.pt \
      --output_dir runs/diffusion_stage1
"""

import argparse
import os
import time
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from crowngen.data.dataset import CrownGenDataset, crown_collate_fn
from crowngen.models.denoise_net import DenoiseNetwork
from crowngen.models.boundary_net import BoundaryPredictor
from crowngen.models.diffusion import GaussianDiffusion
from crowngen.models.ema import EMA


def parse_args():
    parser = argparse.ArgumentParser(description='CrownGen Diffusion 학습')
    parser.add_argument('--config', type=str, default='crowngen/configs/default.yaml')
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--boundary_ckpt', type=str, default=None,
                        help='Boundary 모델 체크포인트 (필수)')
    parser.add_argument('--diffusion_ckpt', type=str, default=None,
                        help='Stage 1 체크포인트 (Stage 2에서 이어서 학습 시)')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size per GPU (config 파일 오버라이드)')
    return parser.parse_args()


def setup_ddp():
    """DDP 초기화. torchrun 환경에서 자동 감지."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    return 0, 0, 1, False


def cleanup_ddp(is_ddp):
    if is_ddp:
        dist.destroy_process_group()


class DiffusionTrainer:
    """Diffusion 모델 학습 매니저."""

    def __init__(self, config, stage, device, boundary_ckpt=None):
        self.config = config
        self.stage = stage
        self.device = device
        self._raw_model = None  # DDP 래핑 전 원본 모델 참조

        # 학습 설정 (Stage별 분기)
        train_key = f'stage{stage}'
        self.train_cfg = config['training'][train_key]
        self.diff_cfg = config['diffusion']

        # Denoising Network
        self.model = DenoiseNetwork(config).to(device)
        self._raw_model = self.model  # 체크포인트 저장용

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"DenoiseNetwork 파라미터: {n_params:,}")

        # Diffusion 프로세스
        self.diffusion = GaussianDiffusion(
            timesteps=self.diff_cfg['timesteps'],
            beta_min=self.diff_cfg['beta_min'],
            beta_max=self.diff_cfg['beta_max'],
            schedule=self.diff_cfg['schedule'],
        )

        # Boundary Predictor (고정, 추론 전용)
        self.boundary_model = None
        if boundary_ckpt:
            self.boundary_model = BoundaryPredictor(config).to(device)
            ckpt = torch.load(boundary_ckpt, map_location=device)
            # DataParallel 저장 체크포인트의 module. prefix 제거
            state_dict = ckpt['model_state_dict']
            if any(k.startswith('module.') for k in state_dict):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            self.boundary_model.load_state_dict(state_dict)
            self.boundary_model.eval()
            for p in self.boundary_model.parameters():
                p.requires_grad = False
            print(f"Boundary 모델 로드: {boundary_ckpt}")

        # Mixed Precision
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

        # EMA (Point-diffusion 품질 핵심 — PVD 기본 decay 0.995)
        ema_decay = config.get('training', {}).get('ema_decay', 0.995)
        ema_warmup = config.get('training', {}).get('ema_warmup', 0)
        self.ema = EMA(self.model, decay=ema_decay, warmup=ema_warmup)

    def wrap_ddp(self, local_rank):
        """DDP 래핑."""
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        print(f"DDP: rank {dist.get_rank()}, world_size {dist.get_world_size()}")

    def get_boundaries(self, batch):
        """Boundary 파라미터 획득."""
        if self.boundary_model is not None:
            with torch.no_grad():
                tooth_points = batch['tooth_points'].to(self.device)
                fdi_labels = batch['fdi_labels'].to(self.device)
                tooth_valid = batch['tooth_valid'].to(self.device)
                target_mask = batch['target_mask'].to(self.device)

                masked_points = tooth_points.clone()
                mask_3d = target_mask.unsqueeze(-1).unsqueeze(-1).expand_as(tooth_points)
                masked_points = masked_points * (1 - mask_3d.float())

                boundaries = self.boundary_model(
                    masked_points, fdi_labels, tooth_valid, target_mask
                )
                return boundaries
        else:
            return batch['boundaries'].to(self.device)

    def train_one_epoch(self, loader, optimizer, epoch):
        """한 에폭 학습."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            boundaries = self.get_boundaries(batch)
            batch['boundaries'] = boundaries

            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                loss = self.diffusion.training_loss(self.model, batch, self.device)

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()

            # DDP 래핑된 경우 내부 모델로 EMA 업데이트
            base = self.model.module if hasattr(self.model, 'module') else self.model
            self.ema.update(base)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, loader):
        """검증."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            boundaries = self.get_boundaries(batch)
            batch['boundaries'] = boundaries

            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                loss = self.diffusion.training_loss(self.model, batch, self.device)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)


def main():
    args = parse_args()

    # DDP 초기화
    rank, local_rank, world_size, is_ddp = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')

    if rank == 0:
        print(f"DDP: {is_ddp}, world_size: {world_size}")
        print(f"Stage: {args.stage}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'checkpoints').mkdir(exist_ok=True)
    if is_ddp:
        dist.barrier()

    # 데이터셋
    data_cfg = config['data']
    train_key = f'stage{args.stage}_train'
    val_key = f'stage{args.stage}_val'

    if rank == 0:
        print("데이터셋 로드 중...")
    train_dataset = CrownGenDataset(
        data_dir=data_cfg['processed_dir'],
        split_file=data_cfg['split_file'],
        split_name=train_key,
        n_points=data_cfg['n_points'],
        mask_range=tuple(data_cfg['mask_range']),
        augment=True,
    )

    val_dataset = CrownGenDataset(
        data_dir=data_cfg['processed_dir'],
        split_file=data_cfg['split_file'],
        split_name=val_key,
        n_points=data_cfg['n_points'],
        mask_range=tuple(data_cfg['mask_range']),
        augment=False,
    )

    # DataLoader with DistributedSampler
    bs_per_gpu = args.batch_size or config['training'][f'stage{args.stage}']['batch_size']
    num_workers = config['training'].get('num_workers', 4)

    if is_ddp:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs_per_gpu,
        shuffle=(not is_ddp),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=crown_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=bs_per_gpu,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=crown_collate_fn,
        pin_memory=True,
    )

    # 트레이너
    trainer = DiffusionTrainer(
        config, args.stage, device,
        boundary_ckpt=args.boundary_ckpt,
    )

    # DDP 래핑
    if is_ddp:
        trainer.wrap_ddp(local_rank)

    # 옵티마이저
    train_cfg = trainer.train_cfg
    optimizer = Adam(trainer.model.parameters(), lr=train_cfg['lr'])

    # 스케줄러 (Stage별 다름)
    if args.stage == 1:
        scheduler = MultiStepLR(
            optimizer,
            milestones=[train_cfg['lr_decay_epoch']],
            gamma=train_cfg['lr_decay_factor'],
        )
    else:
        scheduler = StepLR(
            optimizer,
            step_size=train_cfg['lr_step_size'],
            gamma=train_cfg['lr_decay_factor'],
        )

    # WandB (rank 0만)
    wandb_run = None
    if args.wandb and rank == 0:
        try:
            import wandb
            wandb_run = wandb.init(
                project='crowngen',
                name=f'diffusion_stage{args.stage}',
                config=config,
            )
        except ImportError:
            print("wandb 미설치 — 로깅 없이 진행")

    # 체크포인트 복원
    start_epoch = 0
    best_val_loss = float('inf')
    if args.diffusion_ckpt:
        ckpt = torch.load(args.diffusion_ckpt, map_location=device)
        state_dict = ckpt['model_state_dict']
        if any(k.startswith('module.') for k in state_dict):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        trainer._raw_model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        if rank == 0:
            print(f"체크포인트 복원: epoch {start_epoch}")

    # 학습 루프
    epochs = train_cfg['epochs']
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  Diffusion Stage {args.stage} 학습 시작")
        print(f"  DDP: {is_ddp} ({world_size} GPUs), BS/GPU: {bs_per_gpu}")
        print(f"  에폭: {epochs}, LR: {train_cfg['lr']}")
        print(f"  학습: {len(train_dataset)}, 검증: {len(val_dataset)}")
        print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # DDP: epoch마다 sampler 설정
        if is_ddp:
            train_sampler.set_epoch(epoch)

        train_loss = trainer.train_one_epoch(train_loader, optimizer, epoch)
        val_loss = trainer.validate(val_loader)

        # DDP: val_loss 평균 내기
        if is_ddp:
            val_tensor = torch.tensor([val_loss], device=device)
            dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_tensor.item()

        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"LR: {lr:.2e} | Time: {elapsed:.1f}s")

        if wandb_run:
            import wandb
            wandb.log({
                f'stage{args.stage}/train_loss': train_loss,
                f'stage{args.stage}/val_loss': val_loss,
                f'stage{args.stage}/lr': lr,
                f'stage{args.stage}/epoch': epoch,
            })

        # 최적 모델 저장 (rank 0만)
        if rank == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'stage': args.stage,
                    'model_state_dict': trainer._raw_model.state_dict(),
                    'ema_state_dict': trainer.ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }, output_dir / 'checkpoints' / 'best.pt')
                print(f"  → 최적 모델 저장 (val_loss: {val_loss:.4f})")

            # 주기적 체크포인트
            if (epoch + 1) % 200 == 0:
                torch.save({
                    'epoch': epoch,
                    'stage': args.stage,
                    'model_state_dict': trainer._raw_model.state_dict(),
                    'ema_state_dict': trainer.ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }, output_dir / 'checkpoints' / f'epoch_{epoch+1}.pt')

    # 최종 저장 (rank 0만)
    if rank == 0:
        torch.save({
            'epoch': epochs - 1,
            'stage': args.stage,
            'model_state_dict': trainer._raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'config': config,
        }, output_dir / 'checkpoints' / 'final.pt')

        print(f"\n{'='*60}")
        print(f"  Stage {args.stage} 학습 완료! Best val_loss: {best_val_loss:.4f}")
        print(f"{'='*60}")

    if wandb_run:
        wandb.finish()

    cleanup_ddp(is_ddp)


if __name__ == '__main__':
    main()
