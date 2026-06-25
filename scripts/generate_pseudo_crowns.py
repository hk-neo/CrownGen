"""
Stage 2 pseudo-crown 데이터 확장 파이프라인 (논문의 자가 부트스트래핑).

논문 Stage 2:
  1. Stage 1 모델로 부분 무치아(partially edentulous) 스캔에 추론해 빈 자리에
     해부학적으로 타당한 "pseudo-crown(가짜 크라운)"을 생성.
  2. 이를 채워 넣어 *완전 치열(fully dentate)* 확장 데이터셋을 구축.
  3. Stage 2 fine-tune 은 이 확장 세트 + 원래 완전 치열 세트에서 학습.
     학습 신호는 다수의 고품질 자연치가 지배하므로 pseudo-crown 의 부정확성에
     견고하다 (논문 ablation: 데이터 확장이 가장 중요한 구성요소).

기존 구현은 이 단계가 빠져 있어 stage2 가 stage1 과 동일한 데이터에서 단순
파인튜닝만 수행했고, best val_loss 가 stage1 과 동일하게 멈추는 결과를 냈다.

입력:
  --boundary_ckpt : 학습된 boundary 모델
  --diffusion_ckpt : 학습된 stage1 diffusion 모델
  --input_dir : 부분 무치아 처리 .npz 디렉토리 (Data/processed)
  --split_file / --split : 채울 대상 스캔 목록 (예: stage2_train)
  --output_dir : pseudo-crown 이 채워진 확장 .npz 저장 디렉토리

출력:
  결손치가 pseudo-crown 으로 채워진 fully-dentate .npz (stage2 학습에 바로 사용).

사용 예:
  python scripts/generate_pseudo_crowns.py \
      --config crowngen/configs/default.yaml \
      --boundary_ckpt runs/boundary/checkpoints/best.pt \
      --diffusion_ckpt runs/diffusion_stage1/checkpoints/best.pt \
      --input_dir Data/processed \
      --split_file Data/SourceC_Teeth3DS/train_val_split.json \
      --split stage2_train \
      --output_dir Data/processed_pseudo
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# 프로젝트 루트를 path 에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from crowngen.inference.pipeline import CrownGenPipeline
from crowngen.data.fdi import ZIGZAG_FDI_ORDER


def jaw_of(fdi: int) -> str:
    return 'upper' if fdi // 10 in (1, 2) else 'lower'


def load_scan(npz_path: str, n_points: int):
    """한 스캔 .npz 를 28슬롯 텐서로 로드.

    결손치(데이터에 포인트 클라우드가 없는 슬롯)는 target 으로 표시한다.
    Returns: dict with tooth_points(1,28,N,3), fdi_labels, tooth_valid, target_mask
    """
    data = np.load(npz_path)
    tooth_points = torch.zeros(28, n_points, 3, dtype=torch.float32)
    fdi_labels = torch.zeros(28, dtype=torch.long)
    tooth_valid = torch.zeros(28, dtype=torch.long)
    target_mask = torch.zeros(28, dtype=torch.long)

    for slot, fdi in enumerate(ZIGZAG_FDI_ORDER):
        key = f"{jaw_of(fdi)}_{fdi}_pc"
        fdi_labels[slot] = fdi
        if key in data:
            pc = torch.from_numpy(data[key]).float()
            if pc.shape[0] >= n_points:
                pc = pc[torch.randperm(pc.shape[0])[:n_points]]
            else:
                idx = torch.randint(0, pc.shape[0], (n_points,))
                pc = pc[idx] + torch.randn(n_points, 3) * 0.01
            tooth_points[slot] = pc
            tooth_valid[slot] = 1
        else:
            # 결손치 → pseudo-crown 생성 대상
            target_mask[slot] = 1

    return {
        'tooth_points': tooth_points.unsqueeze(0),
        'fdi_labels': fdi_labels.unsqueeze(0),
        'tooth_valid': tooth_valid.unsqueeze(0),
        'target_mask': target_mask.unsqueeze(0),
    }


def fill_and_save(data, generated, target_mask, fdi_labels, out_path: Path):
    """결손 슬롯에 생성된 pseudo-crown 을 채워 fully-dentate .npz 로 저장.

    기존 자연치 키는 그대로 보존하고, 결손 슬롯에 대해
    {jaw}_{fdi}_pc / {jaw}_{fdi}_bound 키를 새로 추가한다.
    """
    save = dict(data)
    gen = generated[0].cpu().numpy()        # (28, N, 3)
    bnd_present = 'bound' in data  # 경계 키 존재 여부 (나중에 boundary 도 같이 저장)
    mask = target_mask[0].cpu().numpy()
    fdi = fdi_labels[0].cpu().numpy()

    n_filled = 0
    for slot in range(28):
        if mask[slot] == 1:
            key_pc = f"{jaw_of(int(fdi[slot]))}_{int(fdi[slot])}_pc"
            save[key_pc] = gen[slot].astype(np.float32)
            n_filled += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **save)
    return n_filled


def main():
    ap = argparse.ArgumentParser(description='Stage 2 pseudo-crown 데이터 확장')
    ap.add_argument('--config', default='crowngen/configs/default.yaml')
    ap.add_argument('--boundary_ckpt', required=True)
    ap.add_argument('--diffusion_ckpt', required=True)
    ap.add_argument('--input_dir', default='Data/processed')
    ap.add_argument('--split_file', required=True)
    ap.add_argument('--split', default='stage2_train')
    ap.add_argument('--output_dir', default='Data/processed_pseudo')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max_patients', type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    n_points = config['data']['n_points']

    pipeline = CrownGenPipeline(
        config=config,
        boundary_ckpt=args.boundary_ckpt,
        diffusion_ckpt=args.diffusion_ckpt,
        device=args.device,
        n_points=n_points,
    )

    with open(args.split_file) as f:
        pids = json.load(f)[args.split]
    if args.max_patients:
        pids = pids[:args.max_patients]

    out_dir = Path(args.output_dir)
    print(f"=== pseudo-crown 생성: {len(pids)} 스캔 → {out_dir} ===")

    n_ok = n_skip = total_filled = 0
    for i, pid in enumerate(pids):
        in_path = Path(args.input_dir) / f"{pid}.npz"
        if not in_path.exists():
            n_skip += 1
            continue

        batch = load_scan(str(in_path), n_points)
        # 결손치가 없으면 건너뛴다 (이미 완전 치열)
        if int(batch['target_mask'].sum()) == 0:
            n_skip += 1
            continue

        for k in batch:
            batch[k] = batch[k].to(args.device)

        try:
            result = pipeline(
                tooth_points=batch['tooth_points'],
                fdi_labels=batch['fdi_labels'],
                tooth_valid=batch['tooth_valid'],
                target_mask=batch['target_mask'],
            )
            data = np.load(in_path)
            filled = fill_and_save(
                data, result['generated'], batch['target_mask'],
                batch['fdi_labels'], out_dir / f"{pid}.npz",
            )
            total_filled += filled
            n_ok += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(pids)}] {pid}: {filled}개 pseudo-crown 채움")
        except Exception as e:
            print(f"  [{i+1}/{len(pids)}] ❌ {pid}: {e}")
            n_skip += 1

    print(f"\n=== 완료: 확장 {n_ok}건, 스킵 {n_skip}건, 채운 크라운 {total_filled}개 ===")
    print(f"출력: {out_dir}")
    print("다음 단계: stage2_train split 의 input_dir 을 이 디렉토리로 학습.")


if __name__ == '__main__':
    main()
