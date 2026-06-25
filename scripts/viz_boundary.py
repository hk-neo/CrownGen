"""Boundary 예측 시각화 — GT(초록) vs 예측(빨강) 실린더 + 컨텍스트 치아(회색).

모델이 없으면 40ep 학습 후 저장. val 케이스 N개 → PNG (상단 XY 평면도 + 3D).
"""
import argparse, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from torch.utils.data import DataLoader

import importlib.util
spec = importlib.util.spec_from_file_location('tbo', 'scripts/train_boundary_official.py')
tbo = importlib.util.module_from_spec(spec); spec.loader.exec_module(tbo)
from crowngen.external import BoundEncoder

CKPT = 'runs2/boundary_official.pt'


def train_and_save(data_dir, split_file, epochs=40):
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    dev = torch.device('cuda')
    tr = tbo.BoundDataset(data_dir, split_file, 'stage1_train', 512, (1, 6), augment=True)
    tl = DataLoader(tr, 4, shuffle=True, num_workers=4, collate_fn=tbo.collate, drop_last=True)
    model = BoundEncoder(5, 0.3, 6, mask_mode='official').to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=3e-6)
    for ep in range(1, epochs + 1):
        model.train()
        for b in tl:
            exist, miss = tbo.make_exist_mask(b['valid'], 6)
            pred = model(b['points'].to(dev), exist.to(dev))
            loss = BoundEncoder.loss(pred, b['bound'].to(dev), miss.to(dev))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        if ep % 10 == 0:
            print(f'  train ep {ep}/{epochs}', flush=True)
    os.makedirs('runs2', exist_ok=True)
    torch.save(model.state_dict(), CKPT)
    print(f'  saved {CKPT}')
    return model


def cyl_xy_circle(cyl, n=40):
    cx, cy, cz, h, r = cyl
    t = np.linspace(0, 2 * np.pi, n)
    return cx + r * np.cos(t), cy + r * np.sin(t), cz, h


def render(model, sample, out_path, tag=''):
    dev = next(model.parameters()).device
    pts = sample['points'].unsqueeze(0).to(dev)         # (1,28,3,P)
    valid = sample['valid'].unsqueeze(0)
    exist, miss = tbo.make_exist_mask(valid, 6)
    model.eval()
    with torch.no_grad():
        pred = model(pts, exist.to(dev))[0].cpu().numpy()   # (28,5)
    gt = sample['bound'].numpy()
    mm = miss[0].numpy()
    pts_np = sample['points'].numpy()                       # (28,3,P)

    fig = plt.figure(figsize=(14, 6))
    # --- 상단: XY 평면도 (교합면에서 내려다봄) ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_aspect('equal'); ax1.set_title(f'XY (occlusal) — {tag}')
    # 컨텍스트 치아 점 (회색, 얕게)
    for s in range(28):
        if valid[0, s] > 0 and mm[s] == 0:
            ax1.scatter(pts_np[s, 0], pts_np[s, 1], s=1, c='lightgray', alpha=0.5)
    # missing 치아: GT(초록) vs 예측(빨강) 원
    for s in range(28):
        if mm[s] == 1:
            gx, gy, _, _ = cyl_xy_circle(gt[s])
            ax1.plot(gx, gy, 'g-', lw=2, label='GT' if s == int(np.where(mm == 1)[0][0]) else None)
            px, py, _, _ = cyl_xy_circle(pred[s])
            ax1.plot(px, py, 'r--', lw=2, label='pred' if s == int(np.where(mm == 1)[0][0]) else None)
            ax1.scatter(gt[s, 0], gt[s, 1], c='g', s=30, zorder=5)
            ax1.scatter(pred[s, 0], pred[s, 1], c='r', marker='x', s=40, zorder=5)
    ax1.legend(fontsize=8); ax1.set_xlabel('X'); ax1.set_ylabel('Y')

    # --- 하단: 3D ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title(f'3D — {tag}')
    for s in range(28):
        if valid[0, s] > 0 and mm[s] == 0:
            ax2.scatter(pts_np[s, 0], pts_np[s, 1], pts_np[s, 2], s=1, c='lightgray', alpha=0.4)
    for s in range(28):
        if mm[s] == 1:
            for cyl, col, ls in [(gt[s], 'g', '-'), (pred[s], 'r', '--')]:
                cx, cy, cz, h, r = cyl
                t = np.linspace(0, 2 * np.pi, 30)
                for zoff in (-h / 2, h / 2):
                    ax2.plot(cx + r * np.cos(t), cy + r * np.sin(t),
                             np.full_like(t, cz + zoff), color=col, linestyle=ls, lw=1.5)
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'  saved {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='Data/processed_norm2')
    ap.add_argument('--split_file', default='Data/SourceC_Teeth3DS/train_val_split.json')
    ap.add_argument('--n_cases', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=40)
    args = ap.parse_args()

    if os.path.exists(CKPT):
        print(f'loading {CKPT}')
        model = BoundEncoder(5, 0.3, 6, mask_mode='official').to('cuda')
        model.load_state_dict(torch.load(CKPT, map_location='cuda'))
    else:
        print(f'no ckpt — training {args.epochs}ep...')
        model = train_and_save(args.data_dir, args.split_file, args.epochs)

    va = tbo.BoundDataset(args.data_dir, args.split_file, 'stage1_val', 512, (1, 6), augment=False)
    os.makedirs('runs2/viz', exist_ok=True)
    for c in range(args.n_cases):
        render(model, va[c], f'runs2/viz/boundary_{c:02d}.png', tag=f'val case {c}')


if __name__ == '__main__':
    main()
