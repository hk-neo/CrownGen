"""gen_aligned 생성 결과 시각화 데이터.
aligned_norm val 환자에서 present 치아 1~4개 마스킹 → gen_aligned로 크라운 생성 →
GT 점(초록) vs 생성 점(주황) + present(회색). per-tooth CD. → runs2/viz/gen_aligned/data.js
"""
import sys, os, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
import json, numpy as np, torch
from crowngen.external.gen_diffusion import GenModel, get_betas
from crowngen.losses.chamfer import chamfer_distance_l1
from crowngen.data.fdi import ZIGZAG_FDI_ORDER
from gen_stage2_pseudo import load_patient, jaw_of

DATA = 'Data/aligned_norm'
SPLIT = 'Data/SourceC_Teeth3DS/train_val_split.json'


def dn(pts, n=400):
    if len(pts) > n:
        return pts[np.linspace(0, len(pts) - 1, n).astype(int)]
    return pts


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs2/gen_aligned_last.pt')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--mask_n', type=int, default=3)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--out', default='runs2/viz/gen_aligned/data.js')
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device('cuda')
    betas = get_betas('linear', 1e-4, 2e-2, 1000)
    model = GenModel(betas, embed_dim=64, dropout=0.1, extra_feature_channels=9).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.model.load_state_dict(ck['model'])
    if ck.get('ema'):
        from crowngen.models.ema import EMA
        ema = EMA(model.model, 0.995); ema.load_state_dict(ck['ema']); ema.apply_to(model.model)
    model.eval()
    print(f'ckpt {args.ckpt} (ep {ck.get("ep","?")})', flush=True)

    sp = json.load(open(SPLIT))
    pids = [p for p in sp['stage1_val'] if os.path.exists(f'{DATA}/{p}.npz')][:args.n]
    cases = []
    for pid in pids:
        pts, bnd, valid, _ = load_patient(f'{DATA}/{pid}.npz', 1024)
        present = np.where(valid > 0)[0]
        k = min(args.mask_n, len(present))
        idx = np.random.choice(present, k, replace=False)
        x0 = torch.from_numpy(pts).unsqueeze(0).to(dev)
        bound = torch.from_numpy(bnd).unsqueeze(0).to(dev)   # GT bound
        lm = np.zeros(28); lm[idx] = 1
        lm_t = torch.from_numpy(lm).float().unsqueeze(0).to(dev); om = 1 - lm_t
        with torch.no_grad():
            gen = model.sample(dict(x0=x0, l_mask=lm_t, o_mask=om, bound=bound))[0].cpu().numpy()
        real = []
        for s in present:
            real.extend(dn(pts[s].T, 120).tolist())
        teeth = []
        for s in idx:
            g_pts = dn(pts[s].T, 500)              # GT (P,3)
            p_pts = dn(gen[s].T, 500)              # generated
            cd = chamfer_distance_l1(torch.from_numpy(g_pts).unsqueeze(0),
                                     torch.from_numpy(p_pts).unsqueeze(0)).item()
            teeth.append({'slot': int(s), 'fdi': int(ZIGZAG_FDI_ORDER[s]),
                          'gt': g_pts.tolist(), 'gen': p_pts.tolist(), 'cd': round(cd * 1e3, 1)})
        cases.append({'patient': pid, 'real_pts': real, 'teeth': teeth})
        print(f'  {pid}: {k}치아, CD ' + ', '.join(f"{t['cd']}" for t in teeth), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('window.GA_DATA = ' + json.dumps({'cases': cases}) + ';\n')
    allc = [t['cd'] for c in cases for t in c['teeth']]
    print(f'WROTE {args.out}: {len(cases)} cases, mean CD {np.mean(allc):.1f}×10⁻³', flush=True)


if __name__ == '__main__':
    main()
