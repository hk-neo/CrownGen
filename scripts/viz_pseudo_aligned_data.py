"""aligned pseudo-crown 시각화 데이터.
processed_stage2_aligned(채운) vs aligned_norm(원본 partial) 비교 →
pseudo 슬롯(채워진 크라운) 식별. present(회색) + pseudo 크라운(주황).
→ runs2/viz/pseudo_aligned/data.js
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
import numpy as np
from crowngen.data.fdi import ZIGZAG_FDI_ORDER
from viz_pseudo_data import load_slots, downsample, jaw

NORM = 'Data/aligned_norm'                    # 원본 partial
S2 = 'Data/processed_stage2_aligned'          # 채워진


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--out', default='runs2/viz/pseudo_aligned/data.js')
    args = ap.parse_args()
    cases = []
    for path in sorted(glob.glob(f'{S2}/*.npz')):
        pid = os.path.basename(path).replace('.npz', '')
        norm_path = f'{NORM}/{pid}.npz'
        if not os.path.exists(norm_path):
            continue
        filled = load_slots(path)
        orig = load_slots(norm_path)
        if filled is None or orig is None:
            continue
        if len(filled) != 28 or len(orig) >= 28:
            continue                       # 완전치열이거나 덜 채워진
        pseudo_slots = sorted(set(filled) - set(orig))
        real_slots = sorted(set(orig) & set(filled))
        if not pseudo_slots:
            continue
        real_pts = []
        for s in real_slots:
            real_pts.extend(downsample(filled[s], 120).tolist())
        pseudo = [{'slot': int(s), 'fdi': int(ZIGZAG_FDI_ORDER[s]),
                   'pts': downsample(filled[s], 500).tolist()} for s in pseudo_slots]
        cases.append({'patient': pid, 'n_real': len(real_slots), 'n_pseudo': len(pseudo_slots),
                      'real_pts': real_pts, 'pseudo': pseudo})
        if len(cases) >= args.n:
            break
    payload = {'fdi_order': list(ZIGZAG_FDI_ORDER), 'cases': cases}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('window.PA_DATA = ' + json.dumps(payload) + ';\n')
    print(f'WROTE {args.out}: {len(cases)} cases', flush=True)


if __name__ == '__main__':
    main()
