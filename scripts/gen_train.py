"""Generation(diffusion) 학습 스크립트 — stage1 (완전 치열).

데이터: Data/processed_norm2 (Procrustes 정렬). 28개 치아 환자만 stage1에 사용.
model_kwargs: x0(clean GT 점), l_mask(타겟), o_mask(컨텍스트), bound(GT 실린더).
bound 순서 (cx,cy,cz,h,r) — boundary 모델 출력과 일치시킴 (추론 시 boundary 모델 예측 사용).
EMA + cosine LR(논문 stage1: 4e-5, 3000ep).
"""
import argparse, os, sys, random, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from crowngen.external.gen_diffusion import GenModel, get_betas
from crowngen.external.pvcnn import create_mlp_components  # noqa (import sanity)
from crowngen.models.ema import EMA
from crowngen.data.fdi import ZIGZAG_FDI_ORDER


def jaw_of(fdi):
    return 'upper' if fdi // 10 in (1, 2) else 'lower'


class GenDataset(Dataset):
    def __init__(self, data_dir, patients, n_points=1024, mask_range=(1, 6), augment=True,
                 require_full=True):
        self.dir = data_dir
        self.n_points = n_points
        self.mask_range = mask_range
        self.augment = augment
        self.files = []
        for p in patients:
            f = os.path.join(data_dir, f'{p}.npz')
            if not os.path.exists(f):
                continue
            if require_full:
                d = np.load(f)
                n = sum(1 for fdi in ZIGZAG_FDI_ORDER if f'{jaw_of(fdi)}_{fdi}_pc' in d)
                if n != 28:
                    continue
            self.files.append(f)
        print(f'  GenDataset: {len(self.files)} patients (require_full={require_full})')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        pts = np.zeros((28, 3, self.n_points), dtype=np.float32)
        bnd = np.zeros((28, 5), dtype=np.float32)
        for s, fdi in enumerate(ZIGZAG_FDI_ORDER):
            k = f'{jaw_of(fdi)}_{fdi}_pc'
            if k in d:
                pc = d[k]
                if pc.shape[0] >= self.n_points:
                    pc = pc[np.random.permutation(pc.shape[0])[:self.n_points]]
                else:
                    idx = np.random.choice(pc.shape[0], self.n_points, replace=True)
                    pc = pc[idx]
                pts[s] = pc.T.astype(np.float32)
                bk = k.replace('_pc', '_bound')
                if bk in d:
                    b = d[bk]                       # (cx,cy,cz,r,h)
                    bnd[s] = [b[0], b[1], b[2], b[4], b[3]]   # → (cx,cy,cz,h,r)
        if self.augment:
            sc = np.random.uniform(0.95, 1.05)
            pts *= sc; bnd *= sc
            if np.random.rand() < 0.5:             # 좌우 반전 (Procrustes 프레임)
                pts[:, 0, :] *= -1; bnd[:, 0] *= -1
        return {'points': torch.from_numpy(pts), 'bound': torch.from_numpy(bnd)}


def collate(batch):
    return {k: torch.stack([b[k] for b in batch], 0) for k in batch[0]}


def make_kwargs(batch, device, max_missing=6):
    B = batch['points'].shape[0]
    l_mask = torch.zeros(B, 28)
    for b in range(B):
        k = random.randint(*((1, max_missing)))
        idx = torch.randperm(28)[:k]
        l_mask[b, idx] = 1.0
    l_mask = l_mask.to(device)
    o_mask = 1.0 - l_mask
    bound = batch['bound'].to(device)
    if getattr(make_kwargs, 'zero_bound', 0):
        bound = torch.zeros_like(bound)             # ablation: boundary conditioning 끔
    return {
        'x0': batch['points'].to(device),          # (B,28,3,P) clean GT
        'l_mask': l_mask, 'o_mask': o_mask,
        'bound': bound,        # (B,28,5)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='Data/processed_norm2')
    ap.add_argument('--split_file', default='Data/SourceC_Teeth3DS/train_val_split.json')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--lr', type=float, default=4e-5)
    ap.add_argument('--n_points', type=int, default=1024)
    ap.add_argument('--eval_every', type=int, default=20)
    ap.add_argument('--sample_every', type=int, default=40)   # 샘플 생성(느림)
    ap.add_argument('--tag', default='gen_stage1')
    ap.add_argument('--zero_bound', type=int, default=0, help='1이면 boundary condition 끔(ablation)')
    ap.add_argument('--resume', default=None, help='이 ckpt에서 이어서(가중치+EMA 로드)')
    ap.add_argument('--stage2', type=int, default=0, help='1이면 processed_stage2 split 사용')
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    device = torch.device('cuda')
    make_kwargs.zero_bound = args.zero_bound

    splits = json.load(open(args.split_file))
    if args.stage2:
        s2 = json.load(open(f'{args.data_dir}/split.json'))
        tr_pids = s2['stage2_all_train']
        va_pids = s2['stage2_all_val']
    else:
        tr_pids = splits['stage1_train'] + splits['stage2_train']
        va_pids = splits['stage1_val']
    tr = GenDataset(args.data_dir, tr_pids, args.n_points, require_full=True, augment=True)
    va = GenDataset(args.data_dir, va_pids, args.n_points, require_full=True, augment=False)
    tl = DataLoader(tr, args.batch_size, shuffle=True, num_workers=3, collate_fn=collate, drop_last=True)
    vl = DataLoader(va, args.batch_size, shuffle=False, num_workers=2, collate_fn=collate)

    betas = get_betas('linear', 1e-4, 2e-2, 1000)
    model = GenModel(betas, embed_dim=64, dropout=0.1, extra_feature_channels=9).to(device)
    print(f'params {sum(p.numel() for p in model.parameters()):,}', flush=True)
    start_ep = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.model.load_state_dict(ck['model'])
        start_ep = ck.get('ep', 0)
        print(f'resumed from {args.resume} (ep {start_ep})', flush=True)
    ema = EMA(model.model, decay=0.995)
    if args.resume and os.path.exists(args.resume) and ck.get('ema'):
        try:
            ema.load_state_dict(ck['ema'])
        except Exception:
            pass
    # 다중 GPU: denoiser(PVCNN2) 를 DataParallel 로 감싸 batch 를 GPU별 분산 (메모리 + 속도).
    # bare 참조를 EMA/저장에 계속 사용 (DP 감싸도 underlying 파라미터는 동일).
    bare = model.model
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model.model = torch.nn.DataParallel(bare, device_ids=list(range(n_gpu)))
        print(f'[DP] denoiser 병렬화: {n_gpu} GPUs · batch {args.batch_size} → {args.batch_size//n_gpu}/GPU', flush=True)
    opt = Adam(model.parameters(), lr=args.lr)
    # 잔여 구간에 맞춰 cosine (이어서일 때 자연스러운 감소)
    sch = CosineAnnealingLR(opt, T_max=max(args.epochs - start_ep, 1), eta_min=args.lr * 0.01)

    for ep in range(start_ep + 1, args.epochs + 1):
        model.train(); t0 = time.time(); tloss = 0; nb = 0
        for b in tl:
            kw = make_kwargs(b, device)
            noise = torch.randn_like(kw['x0'])
            loss = model.loss(noise, kw)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema.update(bare)
            tloss += loss.item(); nb += 1
        sch.step()
        msg = f'ep {ep:4d}/{args.epochs} | train {tloss/nb:.4f} | lr {opt.param_groups[0]["lr"]:.1e} | {time.time()-t0:.0f}s'
        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            model.eval(); vloss = 0; vn = 0
            with torch.no_grad():
                for b in vl:
                    kw = make_kwargs(b, device); noise = torch.randn_like(kw['x0'])
                    vloss += model.loss(noise, kw).item(); vn += 1
            msg += f' | val {vloss/vn:.4f}'
        print(f'[{args.tag}] {msg}', flush=True)
        os.makedirs('runs2', exist_ok=True)
        _sd = model.model.module.state_dict() if isinstance(model.model, torch.nn.DataParallel) else model.model.state_dict()
        torch.save({'model': _sd, 'ema': ema.state_dict(), 'ep': ep},
                   f'runs2/{args.tag}_last.pt', )

    print(f'[{args.tag}] DONE. ckpt=runs2/{args.tag}_last.pt', flush=True)


if __name__ == '__main__':
    main()
