"""aligned boundary 예측 시각화용 데이터.
aligned_norm val 환자에서: present 치아 점 + (마스킹한 present 치아의) GT cylinder vs
aligned 모델 예측 cylinder. → runs2/viz/boundary_aligned/data.js
"""
import sys, os, random, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import numpy as np, torch
from crowngen.external import BoundEncoder
from crowngen.data.fdi import ZIGZAG_FDI_ORDER
from gen_stage2_pseudo import load_patient, jaw_of

DATA = 'Data/aligned_norm'
SPLIT = 'Data/SourceC_Teeth3DS/train_val_split.json'


def down(pts, n=200):
    if len(pts) > n:
        idx = np.linspace(0, len(pts) - 1, n).astype(int)
        return pts[idx]
    return pts


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs2/boundary_aligned.pt')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--out', default='runs2/viz/boundary_aligned/data.js')
    ap.add_argument('--mask_n', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    dev = torch.device('cuda')
    m = BoundEncoder(5, 0.3, 6, 'official').to(dev)
    m.load_state_dict(torch.load(args.ckpt, map_location=dev)); m.eval()
    sp = json.load(open(SPLIT))
    pids = [p for p in sp['stage1_val'] if os.path.exists(f'{DATA}/{p}.npz')][:args.n]

    cases = []
    for pid in pids:
        pts, bnd, valid, _ = load_patient(f'{DATA}/{pid}.npz', 1024)
        present = np.where(valid > 0)[0]
        if len(present) < 2:
            continue
        k = min(args.mask_n, len(present))
        idx = np.random.choice(present, k, replace=False)
        x0 = torch.from_numpy(pts).unsqueeze(0).to(dev)
        # ★ 평가할 치아를 missing(exist=0) 으로 마스킹해야 모델이 예측.
        om_np = valid.copy(); om_np[idx] = 0
        om = torch.from_numpy(om_np).unsqueeze(0).to(dev)
        with torch.no_grad():
            pred = m(x0, om.view(1, 28, 1, 1))[0].cpu().numpy()  # (28,5) cx,cy,cz,h,r
        # present 점 (회색 컨텍스트)
        real = []
        for s in present:
            real.extend(down(pts[s].T, 120).tolist())   # pts[s] is (3,P) → (P,3)
        teeth = []
        for s in idx:
            teeth.append({
                'slot': int(s), 'fdi': int(ZIGZAG_FDI_ORDER[s]),
                'gt': [float(x) for x in bnd[s]],         # cx,cy,cz,h,r
                'pred': [float(x) for x in pred[s]],
            })
        cases.append({'patient': pid, 'real_pts': real, 'teeth': teeth})
        print(f'  {pid}: {k}개 마스킹', flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('window.BAL_DATA = ' + json.dumps({'cases': cases}) + ';\n')
    print(f'WROTE {args.out}: {len(cases)} cases', flush=True)


if __name__ == '__main__':
    main()
