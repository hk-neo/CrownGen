"""gen2k (boundary 조건 O) vs gen2k_nobound (조건 X) 페어드 CD 비교.

동일 환자 · 동일 타겟 마스킹 · 동일 노이즈 시드 → 모델만 바꿔 샘플링.
gen2k        : GT bound 사용 (학습 조건)
gen2k_nobound: zero bound   (학습 조건, zero_bound=1)
EMA 가중치로 샘플링. 결과: CD×10³, 페어드 차이(bound - nobound, 음=bound 우위).
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

A = ('gen2k (bound)',     'runs2/gen2k_last.pt',         True)    # GT bound
B = ('gen2k_nobound',     'runs2/gen2k_nobound_last.pt', False)   # zero bound
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
    torch.manual_seed(seed)                       # 샘플링 노이즈 고정 → 페어드 공정
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
    ap.add_argument('--n_patients', type=int, default=12)
    ap.add_argument('--data_dir', default='Data/processed_norm2')
    ap.add_argument('--split_file', default='Data/SourceC_Teeth3DS/train_val_split.json')
    args = ap.parse_args()

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    splits = json.load(open(args.split_file))
    va = tbo.GenDataset(args.data_dir, splits['stage1_val'], 1024, require_full=True, augment=False)
    models = {name: load_model(path, device) for name, path, _ in RUNS}
    use_gt = {name: gt for name, _, gt in RUNS}

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
            bound = gt_bound if use_gt[name] else torch.zeros_like(gt_bound)
            cd, nt = cd_one(models[name], x0, lm, om, bound, device, seed)
            r[name] = cd * 1e3
            r['nt'] = nt
        r['diff'] = r[A[0]] - r[B[0]]            # 음수 = bound 가 더 좋음
        rows.append(r)
        print(f"patient {pi:2d} (targets={r['nt']}): "
              f"{A[0]} {r[A[0]]:5.1f} | {B[0]} {r[B[0]]:5.1f} | Δ {r['diff']:+5.1f}", flush=True)

    a = np.array([r[A[0]] for r in rows]); b = np.array([r[B[0]] for r in rows])
    d = a - b
    print('\n========== SUMMARY (CD×10³, 낮을수록 좋음) ==========', flush=True)
    print(f'  {A[0]:16s}  mean {a.mean():.2f}  med {np.median(a):.2f}', flush=True)
    print(f'  {B[0]:16s}  mean {b.mean():.2f}  med {np.median(b):.2f}', flush=True)
    print(f'  페어드 평균 차이 (bound - nobound) = {d.mean():+.2f}  '
          f'(음=bound 우위, 양=nobound 우위)', flush=True)
    print(f'  bound 가 더 좋은 환자 수: {(d < 0).sum()}/{len(d)}  |  '
          f'동점 {(d == 0).sum()}  |  nobound 우위 {(d > 0).sum()}', flush=True)
    # paired t (근사)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else float('inf')
    print(f'  paired t = {t:+.2f}  (|t|>2 대략 유의, n={len(d)})', flush=True)
    out = 'runs2/compare_gen_bound.json'
    json.dump({'rows': rows, 'mean': {A[0]: float(a.mean()), B[0]: float(b.mean())},
               'paired_diff_mean': float(d.mean())}, open(out, 'w'), indent=2)
    print(f'WROTE {out}', flush=True)


if __name__ == '__main__':
    main()
