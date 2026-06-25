"""G1: partial-realistic boundary 재학습.

train_boundary_official.py 기반 + 2가지 변경 (C1/OOD 해소):
  1) 학습 데이터에 부분무치아 환자 포함 (stage1_train + stage2_train) → 진짜 sparse 아치 노출.
  2) 현실적/강도 마스킹 (clustered chain + heavy + light 혼합, max_missing=12).
평가: 기존 eval_dice_iou(light) + 신규 eval_heavy_overlap(heavy 마스킹 Dice + 인접 겹침 비율).
GT 없는 진짜 결손은 타겟 불가 → present 치아를 현실적으로 많이 마스킹해 sparse 시나리오를 만들고
그 마스킹 치아(GT 있음)로 학습.
"""
import argparse, os, sys, json, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from crowngen.external import BoundEncoder
from crowngen.data.fdi import ZIGZAG_FDI_ORDER


def jaw_of(fdi):
    return 'upper' if fdi // 10 in (1, 2) else 'lower'


class BoundDataset(Dataset):
    """pids 리스트로 로드. 완전/부분무치아 모두 허용 (valid 로 표식)."""
    def __init__(self, data_dir, pids, n_points=512, augment=True):
        self.dir = data_dir; self.n_points = n_points; self.augment = augment
        self.files = [os.path.join(data_dir, f'{p}.npz') for p in pids
                      if os.path.exists(os.path.join(data_dir, f'{p}.npz'))]
        print(f'  BoundDataset: {len(self.files)} patients')

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        pts = np.zeros((28, 3, self.n_points), dtype=np.float32)
        bnd = np.zeros((28, 5), dtype=np.float32)
        valid = np.zeros(28, dtype=np.float32)
        for s, fdi in enumerate(ZIGZAG_FDI_ORDER):
            k = f'{jaw_of(fdi)}_{fdi}_pc'
            if k in d:
                pc = d[k]
                if pc.shape[0] >= self.n_points:
                    pc = pc[np.random.permutation(pc.shape[0])[:self.n_points]]
                else:
                    pc = pc[np.random.choice(pc.shape[0], self.n_points, replace=True)]
                pts[s] = pc.T.astype(np.float32)
                bk = k.replace('_pc', '_bound')
                if bk in d:
                    b = d[bk]; bnd[s] = [b[0], b[1], b[2], b[4], b[3]]
                valid[s] = 1.0
        if self.augment:
            sc = np.random.uniform(0.95, 1.05); pts *= sc; bnd *= sc
            if np.random.rand() < 0.5: pts[:, 0, :] *= -1; bnd[:, 0] *= -1
        return {'points': torch.from_numpy(pts), 'bound': torch.from_numpy(bnd),
                'valid': torch.from_numpy(valid)}


def collate(batch):
    return {k: torch.stack([b[k] for b in batch], 0) for k in batch[0]}


def make_realistic_mask(valid, max_missing=12, light_prob=0.3):
    """현실적 결손 마스킹: clustered chain / heavy / light 혼합. present(GT 있음)를 타겟화."""
    B = valid.shape[0]
    exist = valid.clone(); miss = torch.zeros_like(valid)
    for b in range(B):
        present = torch.where(valid[b] > 0)[0]
        n = len(present)
        if n == 0: continue
        r = random.random()
        if r < 0.45:                       # clustered chain (arch 인접 다수)
            k = min(random.randint(3, max_missing), n)
            start = random.randint(0, n - 1)
            idx = torch.stack([present[(start + j) % n] for j in range(k)])
        elif r < 0.70:                     # heavy random
            k = min(random.randint(5, max_missing), n)
            idx = present[torch.randperm(n)[:k]]
        else:                              # light random (원래 분포 유지)
            k = min(random.randint(1, 6), n)
            idx = present[torch.randperm(n)[:k]]
        exist[b, idx] = 0; miss[b, idx] = 1.0
    return exist.view(B, 28, 1, 1), miss


def make_light_mask(valid, max_missing=6):
    """기존 eval용 light 마스킹 (비교 기준)."""
    B = valid.shape[0]; exist = valid.clone(); miss = torch.zeros_like(valid)
    for b in range(B):
        present = torch.where(valid[b] > 0)[0]
        if len(present) == 0: continue
        k = min(random.randint(1, max_missing), len(present))
        idx = present[torch.randperm(len(present))[:k]]
        exist[b, idx] = 0; miss[b, idx] = 1.0
    return exist.view(B, 28, 1, 1), miss


def _cyl_mask(cyl, gmin, gmax, res=32):
    cx, cy, cz, h, r = cyl
    if r <= 0 or h <= 0: return torch.zeros(res, res, res)
    gx = np.linspace(gmin[0], gmax[0], res); gy = np.linspace(gmin[1], gmax[1], res); gz = np.linspace(gmin[2], gmax[2], res)
    Z, Y, X = np.meshgrid(gz, gy, gx, indexing='ij')
    inside = ((X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2) & (Z >= cz - h / 2) & (Z <= cz + h / 2)
    return torch.from_numpy(inside.astype(np.float32))


def _dice_iou(pred, gt, miss, res=32):
    ds, ios = [], []
    for ti in range(28):
        if miss[ti] != 1: continue
        p, g = pred[ti], gt[ti]
        cmin = np.minimum(p[:3], g[:3]) - [max(p[4], g[4]), max(p[4], g[4]), max(p[3], g[3])]
        cmax = np.maximum(p[:3], g[:3]) + [max(p[4], g[4]), max(p[4], g[4]), max(p[3], g[3])]
        gmn = (cmin - 0.05).tolist(); gmx = (cmax + 0.05).tolist()
        mp = _cyl_mask(p, gmn, gmx, res); mg = _cyl_mask(g, gmn, gmx, res)
        sp, sg, inter = mp.sum(), mg.sum(), (mp * mg).sum()
        if sp + sg > 0: ds.append((2 * inter / (sp + sg)).item())
        un = sp + sg - inter
        if un > 0: ios.append((inter / un).item())
    return ds, ios


@torch.no_grad()
def eval_dice_iou(model, loader, device, mask_fn, max_missing, tag=''):
    """mask_fn 으로 마스킹 → Dice/IoU. mask_fn 에 max_missing 전달."""
    model.eval(); ds, ios = [], []
    for b in loader:
        pts = b['points'].to(device); gt = b['bound'].numpy()
        exist, miss = mask_fn(b['valid'], max_missing)
        pred = model(pts, exist.to(device)).cpu().numpy(); mm = miss.numpy()
        for bi in range(pred.shape[0]):
            d, io = _dice_iou(pred[bi], gt[bi], mm[bi])
            ds += d; ios += io
    md = float(np.mean(ds)) if ds else 0.0; mi = float(np.mean(ios)) if ios else 0.0
    print(f'  [{tag}] Dice {md:.3f} IoU {mi:.3f} (n={len(ds)})', flush=True)
    return md, mi


@torch.no_grad()
def eval_overlap(model, loader, device, max_missing=10, res=32, tag='overlap'):
    """heavy 마스킹 → Dice + 인접 예측 cylinder 겹침 비율 (xy 최소거리 < 0.08)."""
    model.eval(); ds = []; overlaps = 0; cases = 0
    for b in loader:
        pts = b['points'].to(device); gt = b['bound'].numpy()
        exist, miss = make_realistic_mask(b['valid'], max_missing)
        pred = model(pts, exist.to(device)).cpu().numpy(); mm = miss.numpy()
        for bi in range(pred.shape[0]):
            d, _ = _dice_iou(pred[bi], gt[bi], mm[bi]); ds += d
            slots = np.where(mm[bi] == 1)[0]
            if len(slots) >= 2:
                xy = pred[bi, slots, :2]
                mind = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
                mind[mind == 0] = 9
                if mind.min() < 0.08: overlaps += 1
                cases += 1
    md = float(np.mean(ds)) if ds else 0.0
    ov = overlaps / cases if cases else 0.0
    print(f'  [{tag}] Dice {md:.3f} · 겹침비율 {ov*100:.1f}% ({overlaps}/{cases})', flush=True)
    return md, ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='Data/processed_norm2')
    ap.add_argument('--split_file', default='Data/SourceC_Teeth3DS/train_val_split.json')
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--end_lr', type=float, default=3e-6)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--max_missing', type=int, default=12)
    ap.add_argument('--tag', default='bound_g1')
    ap.add_argument('--out_ckpt', default='runs2/boundary_g1.pt')
    ap.add_argument('--smoke', type=int, default=0, help='1이면 5ep만 타이밍 측정')
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device('cuda')

    sp = json.load(open(args.split_file))
    # 학습: 완전치열(stage1_train) + 부분무치아(stage2_train) → OOD 해소
    tr_pids = sp['stage1_train'] + sp['stage2_train']
    # 평가: 기존(stage1_val, light) + held-out 부분무치아(stage2_val 앞 30명, heavy/overlap)
    va_full = sp['stage1_val']
    va_partial = sp.get('stage2_val', [])[:30]
    tr = BoundDataset(args.data_dir, tr_pids, 512, augment=True)
    va_f = BoundDataset(args.data_dir, va_full, 512, augment=False)
    va_p = BoundDataset(args.data_dir, va_partial, 512, augment=False)
    tl = DataLoader(tr, args.batch_size, shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
    vlf = DataLoader(va_f, args.batch_size, shuffle=False, num_workers=2, collate_fn=collate)
    vlp = DataLoader(va_p, args.batch_size, shuffle=False, num_workers=2, collate_fn=collate)

    model = BoundEncoder(output_dim=5, dropout=0.3, max_missing_teeth=12, mask_mode='official').to(device)
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    epochs = 5 if args.smoke else args.epochs
    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=args.end_lr)
    print(f'params {sum(p.numel() for p in model.parameters()):,} | train {len(tr)} val_full {len(va_f)} val_partial {len(va_p)}', flush=True)

    if args.smoke:
        model.train(); t0 = time.time(); nb = 0
        for b in tl:
            pts = b['points'].to(device); gt = b['bound'].to(device)
            exist, miss = make_realistic_mask(b['valid'], args.max_missing)
            pred = model(pts, exist.to(device))
            loss = BoundEncoder.loss(pred, gt, miss.to(device))
            opt.zero_grad(); loss.backward(); opt.step(); nb += 1
            if nb >= 5: break
        print(f'SMOKE 5 batches: {(time.time()-t0)/nb:.2f}s/batch, {len(tl)} batches/ep → ~{(time.time()-t0)/nb*len(tl):.0f}s/ep', flush=True)
        print(f'GPU mem: {torch.cuda.max_memory_allocated()/1e9:.1f} GB', flush=True)
        return

    best_dice = -1.0
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); tl_loss = 0; nb = 0
        for b in tl:
            pts = b['points'].to(device); gt = b['bound'].to(device)
            exist, miss = make_realistic_mask(b['valid'], args.max_missing)
            pred = model(pts, exist.to(device))
            loss = BoundEncoder.loss(pred, gt, miss.to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl_loss += loss.item(); nb += 1
        sch.step()
        msg = f'ep {ep:4d}/{epochs} | train {tl_loss/nb:.4f} | lr {opt.param_groups[0]["lr"]:.1e} | {time.time()-t0:.0f}s'
        if ep % 100 == 0 or ep == epochs:
            print(msg, flush=True)
            dl, _ = eval_dice_iou(model, vlf, device, make_light_mask, 6, tag=f'light ep{ep}')
            eval_overlap(model, vlp, device, args.max_missing, tag=f'heavy/partial ep{ep}')
            os.makedirs('runs2', exist_ok=True)
            torch.save(model.state_dict(), f'runs2/boundary_g1_ep{ep}.pt')      # 매 eval 스냅샷
            if dl > best_dice:
                best_dice = dl
                torch.save(model.state_dict(), 'runs2/boundary_g1_best.pt')     # best(light Dice)
                print(f'  ★ new best light Dice {dl:.3f} → boundary_g1_best.pt', flush=True)
        else:
            print(msg, flush=True)
    print(f'[{args.tag}] DONE → {args.out_ckpt}', flush=True)


if __name__ == '__main__':
    main()
