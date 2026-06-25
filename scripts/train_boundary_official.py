"""Official-architecture boundary 학습 (PVCNN2) + Dice/IoU 평가.

데이터: Data/processed_norm2 (Procrustes 정렬). 28 슬롯(지그재그 FDI 순서).
GT boundary 순서는 공식과 동일하게 (cx,cy,cz,h,r) 로 맞춘다.
"""
import argparse, os, sys, json, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from crowngen.external import BoundEncoder
from crowngen.data.fdi import ZIGZAG_FDI_ORDER


def jaw_of(fdi):
    return 'upper' if fdi // 10 in (1, 2) else 'lower'


class BoundDataset(Dataset):
    def __init__(self, data_dir, split_file, split, n_points=512, mask_range=(1, 6), augment=True):
        self.dir = data_dir
        self.n_points = n_points
        self.mask_range = mask_range
        self.augment = augment
        pids = json.load(open(split_file))[split]
        self.files = [os.path.join(data_dir, f'{p}.npz') for p in pids
                      if os.path.exists(os.path.join(data_dir, f'{p}.npz'))]
        print(f'[{split}] {len(self.files)} patients')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        pts = np.zeros((28, 3, self.n_points), dtype=np.float32)   # 채널 우선 (PVCNN2 호환)
        bnd = np.zeros((28, 5), dtype=np.float32)   # (cx,cy,cz,h,r)
        valid = np.zeros(28, dtype=np.float32)
        for s, fdi in enumerate(ZIGZAG_FDI_ORDER):
            k = f'{jaw_of(fdi)}_{fdi}_pc'
            if k in d:
                pc = d[k]
                if pc.shape[0] >= self.n_points:
                    pc = pc[np.random.permutation(pc.shape[0])[:self.n_points]]
                else:
                    idx = np.random.choice(pc.shape[0], self.n_points, replace=True)
                    pc = pc[idx]
                pts[s] = pc.T.astype(np.float32)        # (3, P)
                bk = k.replace('_pc', '_bound')
                if bk in d:
                    b = d[bk]           # our order (cx,cy,cz,r,h)
                    bnd[s] = [b[0], b[1], b[2], b[4], b[3]]   # → (cx,cy,cz,h,r)
                valid[s] = 1.0
        # augmentation: isotropic scale + mirror(x→-x) — Procrustes 프레임이므로 x 미러 유효
        if self.augment:
            sc = np.random.uniform(0.95, 1.05)
            pts = pts * sc
            bnd = bnd * sc
            if np.random.rand() < 0.5:
                pts[:, 0, :] *= -1     # x 축 반전 (채널 0)
                bnd[:, 0] *= -1
        return {
            'points': torch.from_numpy(pts),        # (28,3,P)
            'bound': torch.from_numpy(bnd),          # (28,5)
            'valid': torch.from_numpy(valid),        # (28,)
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch], 0) for k in batch[0]}


def make_exist_mask(valid, max_missing):
    """present(valid=1) 중 1~max_missing 개를 missing 으로. exist_mask (B,28,1,1)."""
    B = valid.shape[0]
    exist = valid.clone()
    miss = torch.zeros_like(valid)
    for b in range(B):
        present = torch.where(valid[b] > 0)[0]
        if len(present) == 0:
            continue
        k = min(random.randint(*((1, max_missing))), len(present))
        idx = present[torch.randperm(len(present))[:k]]
        exist[b, idx] = 0
        miss[b, idx] = 1.0
    return exist.view(B, 28, 1, 1), miss


def _cyl_mask(cyl, gmin, gmax, res=32):
    cx, cy, cz, h, r = cyl
    if r <= 0 or h <= 0:
        return torch.zeros(res, res, res)
    import numpy as np
    gx = np.linspace(gmin[0], gmax[0], res)
    gy = np.linspace(gmin[1], gmax[1], res)
    gz = np.linspace(gmin[2], gmax[2], res)
    Z, Y, X = np.meshgrid(gz, gy, gx, indexing='ij')
    inside = ((X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2) & (Z >= cz - h / 2) & (Z <= cz + h / 2)
    return torch.from_numpy(inside.astype(np.float32))


@torch.no_grad()
def eval_dice_iou(model, loader, device, max_missing, res=32):
    model.eval()
    ds, ios = [], []
    for b in loader:
        pts = b['points'].to(device)
        exist, miss = make_exist_mask(b['valid'], max_missing)
        pred = model(pts, exist.to(device)).cpu().numpy()
        gt = b['bound'].numpy()
        mm = miss.numpy()
        for bi in range(pred.shape[0]):
            for ti in range(28):
                if mm[bi, ti] == 1:
                    p, g = pred[bi, ti], gt[bi, ti]
                    cmin = np.minimum(p[:3], g[:3]) - [max(p[4], g[4]), max(p[4], g[4]), max(p[3], g[3])]
                    cmax = np.maximum(p[:3], g[:3]) + [max(p[4], g[4]), max(p[4], g[4]), max(p[3], g[3])]
                    gmn = (cmin - 0.05).tolist(); gmx = (cmax + 0.05).tolist()
                    mp = _cyl_mask(p, gmn, gmx, res); mg = _cyl_mask(g, gmn, gmx, res)
                    sp, sg, inter = mp.sum(), mg.sum(), (mp * mg).sum()
                    if sp + sg > 0:
                        ds.append((2 * inter / (sp + sg)).item())
                    un = sp + sg - inter
                    if un > 0:
                        ios.append((inter / un).item())
    return float(np.mean(ds)), float(np.mean(ios)), len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='Data/processed_norm2')
    ap.add_argument('--split_file', default='Data/SourceC_Teeth3DS/train_val_split.json')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--end_lr', type=float, default=3e-6)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--tag', default='official')
    ap.add_argument('--mask_mode', default='official', choices=['official', 'context'])
    ap.add_argument('--voxel_attention', type=int, default=0)
    ap.add_argument('--eval_every', type=int, default=200)
    ap.add_argument('--out_ckpt', default=None)
    ap.add_argument('--resume', default=None, help='이전 체크포인트에서 이어서(가중치 로드)')
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    device = torch.device('cuda')

    tr = BoundDataset(args.data_dir, args.split_file, 'stage1_train', 512, (1, 6), augment=True)
    va = BoundDataset(args.data_dir, args.split_file, 'stage1_val', 512, (1, 6), augment=False)
    tl = DataLoader(tr, args.batch_size, shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
    vl = DataLoader(va, args.batch_size, shuffle=False, num_workers=2, collate_fn=collate)

    model = BoundEncoder(output_dim=5, dropout=0.3, max_missing_teeth=6, mask_mode=args.mask_mode,
                         voxel_attention=bool(args.voxel_attention)).to(device)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f'resumed from {args.resume}', flush=True)
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.end_lr)
    print(f'params {sum(p.numel() for p in model.parameters()):,}', flush=True)

    out_ckpt = args.out_ckpt or f'runs2/boundary_{args.tag}.pt'
    best_dice = -1.0
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tl_loss = 0; nb = 0
        for b in tl:
            pts = b['points'].to(device); gt = b['bound'].to(device)
            exist, miss = make_exist_mask(b['valid'], 6)
            pred = model(pts, exist.to(device))
            loss = BoundEncoder.loss(pred, gt, miss.to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl_loss += loss.item(); nb += 1
        sch.step()
        info = f'ep {ep:4d}/{args.epochs} | train {tl_loss/nb:.4f} | lr {opt.param_groups[0]["lr"]:.1e} | {time.time()-t0:.0f}s'
        # 주기적 Dice 평가 + best ckpt 저장
        if ep % args.eval_every == 0 or ep == args.epochs:
            dice, iou, n = eval_dice_iou(model, vl, device, 6)
            info += f' | Dice {dice:.3f} IoU {iou:.3f}'
            if dice > best_dice:
                best_dice = dice
                os.makedirs('runs2', exist_ok=True)
                torch.save(model.state_dict(), out_ckpt)
        print(f'[{args.tag}] {info}', flush=True)

    dice, iou, n = eval_dice_iou(model, vl, device, 6)
    print(f'[{args.tag}] FINAL GEOMETRIC (n={n}): Dice={dice:.3f} IoU={iou:.3f} | best={best_dice:.3f} (논문 0.883/0.796) | ckpt={out_ckpt}', flush=True)


if __name__ == '__main__':
    main()
