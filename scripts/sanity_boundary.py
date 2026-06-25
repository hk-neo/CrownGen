"""Boundary sanity 재학습 — 정규화 데이터로 val loss 가 떨어지는지 확인 (2차 디리스크).

train_boundary.py 의 셋업을 그대로 미러링하되 에폭을 줄여(기본 80ep) 빠르게 검증.
데이터 디렉토리를 인자로 받아 processed_norm vs processed 비교도 가능.
"""
import argparse, time, yaml, torch
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from crowngen.data.dataset import CrownGenDataset, crown_collate_fn
from crowngen.models.boundary_net import BoundaryPredictor, boundary_loss


def run_epoch(model, loader, opt, device, train):
    model.train() if train else model.eval()
    tot, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for b in loader:
            tp = b['tooth_points'].to(device)        # is_boundary=True → 타겟 이미 영벡터
            fdi = b['fdi_labels'].to(device)
            tv = b['tooth_valid'].to(device)
            tm = b['target_mask'].to(device)
            gtb = b['boundaries'].to(device)
            pred = model(tp, fdi, tv, tm)
            loss = boundary_loss(pred, gtb, tm)
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); n += 1
    return tot / max(n, 1)


def _cylinder_mask(cyl, grid_min, grid_max, res=32):
    """실린더 (cx,cy,cz,r,h) 를 3D 복셀 마스크로 래스터화. (scale-invariant)"""
    cx, cy, cz, r, h = cyl
    if r <= 0 or h <= 0:
        return torch.zeros(res, res, res)
    gx = torch.linspace(grid_min[0], grid_max[0], res)
    gy = torch.linspace(grid_min[1], grid_max[1], res)
    gz = torch.linspace(grid_min[2], grid_max[2], res)
    Z, Y, X = torch.meshgrid(gz, gy, gx, indexing='ij')
    in_xy = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    in_z = (Z >= cz - h / 2) & (Z <= cz + h / 2)
    return (in_xy & in_z).float()


def eval_dice_iou(model, loader, device, res=32):
    """논문 metric: 실린더 경계 Dice / IoU (scale-invariant, 타겟 치아만).

    pred/gt 실린더를 공통 bounding grid 에 래스터화해 교집합/합집합 복셀 수로
    Dice/IoU 를 계산한다. 좌표 단위(mm vs 정규화)에 무관하므로 논문 수치(0.883/0.796)와
    직접 비교 가능하다.
    """
    import numpy as np
    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for b in loader:
            tp = b['tooth_points'].to(device)
            pred = model(tp, b['fdi_labels'].to(device), b['tooth_valid'].to(device),
                         b['target_mask'].to(device)).cpu().numpy()
            gt = b['boundaries'].numpy()
            tm = b['target_mask'].numpy()
            for bi in range(pred.shape[0]):
                for ti in range(28):
                    if tm[bi, ti] == 1:
                        p = pred[bi, ti]; g = gt[bi, ti]
                        # 두 실린더를 모두 담는 bounding box
                        cmin = np.minimum(p[:3], g[:3]) - [max(p[3], g[3]), max(p[3], g[3]), max(p[4], g[4])]
                        cmax = np.maximum(p[:3], g[:3]) + [max(p[3], g[3]), max(p[3], g[3]), max(p[4], g[4])]
                        grid_min = (cmin - 0.05).tolist()
                        grid_max = (cmax + 0.05).tolist()
                        mp = _cylinder_mask(p, grid_min, grid_max, res)
                        mg = _cylinder_mask(g, grid_min, grid_max, res)
                        sp, sg, inter = mp.sum(), mg.sum(), (mp * mg).sum()
                        if sp + sg > 0:
                            dices.append((2 * inter / (sp + sg)).item())
                        union = sp + sg - inter
                        if union > 0:
                            ious.append((inter / union).item())
    return float(np.mean(dices)), float(np.mean(ious)), len(dices)


def main():
    import numpy as np  # eval 안에서 사용
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='Data/processed_norm')
    ap.add_argument('--split_file', default='Data/SourceC_Teeth3DS/train_val_split.json')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--tag', default='norm')
    args = ap.parse_args()

    cfg = yaml.safe_load(open('crowngen/configs/default.yaml'))
    dev = torch.device('cuda')
    n_pts = cfg['data'].get('n_points_boundary', 512)
    mask = tuple(cfg['data']['mask_range'])

    tr = CrownGenDataset(args.data_dir, args.split_file, 'stage1_train', n_pts, mask, augment=True, is_boundary=True)
    va = CrownGenDataset(args.data_dir, args.split_file, 'stage1_val', n_pts, mask, augment=False, is_boundary=True)
    print(f"[{args.tag}] train={len(tr)} val={len(va)} data={args.data_dir}")
    tl = DataLoader(tr, args.batch_size, shuffle=True, num_workers=4, collate_fn=crown_collate_fn)
    vl = DataLoader(va, args.batch_size, shuffle=False, num_workers=4, collate_fn=crown_collate_fn)

    m = BoundaryPredictor(cfg).to(dev)
    opt = Adam(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=3e-6)

    best = float('inf')
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        trl = run_epoch(m, tl, opt, dev, True)
        val = run_epoch(m, vl, None, dev, False)
        sch.step()
        best = min(best, val)
        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            print(f"[{args.tag}] ep {ep:3d}/{args.epochs} | train {trl:.4f} | val {val:.4f} | best {best:.4f} | lr {opt.param_groups[0]['lr']:.1e} | {time.time()-t0:.1f}s")
    print(f"[{args.tag}] DONE best_val={best:.4f}")

    # 논문 metric (scale-invariant) 으로 최종 평가
    dice, iou, n = eval_dice_iou(m, vl, dev, res=32)
    print(f"[{args.tag}] GEOMETRIC (n={n} targets): Dice={dice:.3f}  IoU={iou:.3f}  "
          f"(논문: Dice 0.883 / IoU 0.796)")


if __name__ == '__main__':
    main()
