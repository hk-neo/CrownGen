"""gen2k(2000ep) vs gen3k(ep2846) 생성 품질 페어드 CD 비교.
동일 환자·동일 타겟 마스킹·동일 노이즈 시드 → 모델만 바꿔 샘플링 (둘 다 GT bound, EMA).
어떤 모델이 더 나은 크라운을 만드는지 CD×10³로 비교.
"""
import argparse, os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import importlib.util
_st = importlib.util.spec_from_file_location('tbo', 'scripts/gen_train.py')
tbo = importlib.util.module_from_spec(_st); _st.loader.exec_module(tbo)
from crowngen.external.gen_diffusion import GenModel, get_betas
from crowngen.losses.chamfer import chamfer_distance_l1

A = ('gen2k_2000ep', 'runs2/gen2k_last.pt', 'red')
B = ('gen3k_ep3000', 'runs2/gen3k_ep3000.pt', 'blue')
RUNS = [A, B]


def load_model(path, device):
    betas = get_betas('linear', 1e-4, 2e-2, 1000)
    m = GenModel(betas, embed_dim=64, dropout=0.1, extra_feature_channels=9).to(device)
    ck = torch.load(path, map_location=device)
    m.model.load_state_dict(ck['model']); m.eval()
    if ck.get('ema'):
        from crowngen.models.ema import EMA
        ema = EMA(m.model, 0.995); ema.load_state_dict(ck['ema']); ema.apply_to(m.model)
        print(f'  {os.path.basename(path)}: EMA 적용', flush=True)
    return m


def cd_one(model, x0, lm, om, bound, device, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        gen = model.sample(dict(x0=x0, l_mask=lm, o_mask=om, bound=bound))
    gen = gen[0].cpu().numpy(); gt = x0[0].cpu().numpy(); lm0 = lm[0].cpu().numpy()
    tidx = np.where(lm0 > 0)[0]
    cds = []
    for ti in tidx:
        p = torch.from_numpy(gen[ti].T).unsqueeze(0)
        g = torch.from_numpy(gt[ti].T).unsqueeze(0)
        cds.append(chamfer_distance_l1(p, g).item())
    return float(np.mean(cds)) if cds else float('nan'), len(tidx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_patients', type=int, default=8)
    ap.add_argument('--data_dir', default='Data/processed_norm2')
    ap.add_argument('--split_file', default='Data/SourceC_Teeth3DS/train_val_split.json')
    args = ap.parse_args()
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    splits = json.load(open(args.split_file))
    va = tbo.GenDataset(args.data_dir, splits['stage1_val'], 1024, require_full=True, augment=False)
    models = {name: load_model(path, device) for name, path, _ in RUNS}

    random.seed(0)
    rows = []
    for pi in range(min(args.n_patients, len(va))):
        s = va[pi]
        x0 = s['points'].unsqueeze(0).to(device)
        gt_bound = s['bound'].unsqueeze(0).to(device)
        lm = torch.zeros(1, 28)
        idx = torch.randperm(28)[:random.randint(1, 6)]; lm[0, idx] = 1
        lm = lm.to(device); om = 1 - lm
        seed = 1000 + pi
        r = {'pi': pi}
        for name, _, _ in RUNS:
            cd, nt = cd_one(models[name], x0, lm, om, gt_bound, device, seed)
            r[name] = cd * 1e3; r['nt'] = nt
        r['diff'] = r[A[0]] - r[B[0]]   # 음수 = gen3k(2846) 우위
        rows.append(r)
        print(f"patient {pi:2d} (t={r['nt']}): {A[0]} {r[A[0]]:5.1f} | {B[0]} {r[B[0]]:5.1f} | "
              f"Δ(2k−3k) {r['diff']:+5.1f}", flush=True)

    a = np.array([r[A[0]] for r in rows]); b = np.array([r[B[0]] for r in rows]); d = a - b
    print('\n===== SUMMARY (CD×10³, 낮을수록 좋음) =====', flush=True)
    print(f'  {A[0]:14s} mean {a.mean():.2f}  med {np.median(a):.2f}', flush=True)
    print(f'  {B[0]:14s} mean {b.mean():.2f}  med {np.median(b):.2f}', flush=True)
    print(f'  페어드 평균(2k − 3k) = {d.mean():+.2f}  (음=gen3k 우위)', flush=True)
    print(f'  gen3k가 더 좋은 환자: {(d > 0).sum()}/{len(d)}  |  gen2k 우위 {(d < 0).sum()}', flush=True)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else float('inf')
    print(f'  paired t = {t:+.2f} (|t|>2 대략 유의)', flush=True)


if __name__ == '__main__':
    main()
