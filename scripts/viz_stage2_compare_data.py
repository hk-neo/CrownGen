"""gen_aligned vs gen_stage2_aligned(aligned) 2모델 비교 뷰어 데이터.
aligned_norm val 10환자에서 present 1~4개 마스킹 → 두 모델로 각각 생성 → GT vs gen + per-tooth CD.
→ runs2/viz/stage2_compare/data.js
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
MODELS = [
    ('gen_aligned', 'runs2/gen_aligned_last.pt'),
    ('stage2', 'runs2/gen_stage2_aligned_last.pt'),
]


def dn(pts, n=400):
    idx = np.linspace(0, len(pts) - 1, n).astype(int) if len(pts) > n else np.arange(len(pts))
    return pts[idx]


def load_model(path, dev):
    betas = get_betas('linear', 1e-4, 2e-2, 1000)
    m = GenModel(betas, embed_dim=64, dropout=0.1, extra_feature_channels=9).to(dev)
    ck = torch.load(path, map_location=dev)
    m.model.load_state_dict(ck['model'])
    if ck.get('ema'):
        from crowngen.models.ema import EMA
        ema = EMA(m.model, 0.995); ema.load_state_dict(ck['ema']); ema.apply_to(m.model)
    m.eval()
    return m


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--mask_n', type=int, default=3)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default='runs2/viz/stage2_compare/data.js')
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device('cuda')
    models = {tag: load_model(path, dev) for tag, path in MODELS}
    print(f'loaded: {[t for t,_ in MODELS]}', flush=True)

    sp = json.load(open(SPLIT))
    pids = [p for p in sp['stage1_val'] if os.path.exists(f'{DATA}/{p}.npz')][:args.n]
    cases = []
    for pid in pids:
        pts, bnd, valid, _ = load_patient(f'{DATA}/{pid}.npz', 1024)
        present = np.where(valid > 0)[0]
        k = min(args.mask_n, len(present))
        idx = np.random.choice(present, k, replace=False)
        x0 = torch.from_numpy(pts).unsqueeze(0).to(dev)
        bound = torch.from_numpy(bnd).unsqueeze(0).to(dev)
        lm = np.zeros(28); lm[idx] = 1
        lm_t = torch.from_numpy(lm).float().unsqueeze(0).to(dev); om = 1 - lm_t
        real = []
        for s in present:
            real.extend(dn(pts[s].T, 100).tolist())
        mout = {}
        for tag, model in models.items():
            with torch.no_grad():
                gen = model.sample(dict(x0=x0, l_mask=lm_t, o_mask=om, bound=bound))[0].cpu().numpy()
            teeth = []
            for s in idx:
                g = dn(pts[s].T, 400); p = dn(gen[s].T, 400)
                cd = chamfer_distance_l1(torch.from_numpy(g).unsqueeze(0),
                                         torch.from_numpy(p).unsqueeze(0)).item()
                teeth.append({'fdi': int(ZIGZAG_FDI_ORDER[s]), 'gt': g.tolist(), 'gen': p.tolist(),
                              'cd': round(cd * 1e3, 1)})
            mout[tag] = teeth
        cases.append({'patient': pid, 'real_pts': real, 'models': mout})
        cds_a = [t['cd'] for t in mout['gen_aligned']]
        cds_s = [t['cd'] for t in mout['stage2']]
        print(f'  {pid}: gen_aligned {np.mean(cds_a):.1f} vs stage2 {np.mean(cds_s):.1f}', flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('window.COMP_DATA = ' + json.dumps({'cases': cases}) + ';\n')
    print(f'WROTE {args.out}: {len(cases)} cases', flush=True)


if __name__ == '__main__':
    main()
