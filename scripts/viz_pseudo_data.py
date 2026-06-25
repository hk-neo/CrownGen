"""생성된 pseudo-crown 시각화용 데이터 준비.

processed_stage2_3k 에서 부분무치아(채워진) 환자를 찾아 →
real 치아(원본) 점 + pseudo-crown(3000ep 모델이 채운) 점을 덤프.
pseudo 슬롯 = processed_norm2엔 없고 stage2_3k엔 있는 슬롯.
(CPU만 사용 — 학습/샤드와 GPU 안 겹침. 읽기 실패 파일은 스킵.)
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from crowngen.data.fdi import ZIGZAG_FDI_ORDER

NORM2 = 'Data/processed_norm2'
S2 = 'Data/processed_stage2_3k'


def jaw(f):
    return 'upper' if f // 10 in (1, 2) else 'lower'


def load_slots(path):
    try:
        d = np.load(path)
    except Exception:
        return None
    slots = {}
    for s, fdi in enumerate(ZIGZAG_FDI_ORDER):
        k = f'{jaw(fdi)}_{fdi}_pc'
        if k in d:
            slots[s] = np.asarray(d[k], dtype=np.float32)  # (N,3)
    return slots


def downsample(pts, per):
    if len(pts) > per:
        idx = np.linspace(0, len(pts) - 1, per).astype(int)
        return pts[idx]
    return pts


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--out', default='runs2/viz/pseudo/data.js')
    args = ap.parse_args()

    cases = []
    for path in sorted(glob.glob(f'{S2}/*.npz')):
        pid = os.path.basename(path).replace('.npz', '')
        norm_path = f'{NORM2}/{pid}.npz'
        if not os.path.exists(norm_path):
            continue
        filled = load_slots(path)
        orig = load_slots(norm_path)
        if filled is None or orig is None:
            continue
        if len(filled) != 28 or len(orig) >= 28:
            continue                       # 채워진 부분무치아가 아님(완전치열이거나 아직 덜 채워짐)
        pseudo_slots = sorted(set(filled) - set(orig))
        real_slots = sorted(set(orig) & set(filled))
        if not pseudo_slots:
            continue
        real_pts = []
        for s in real_slots:
            real_pts.extend(downsample(filled[s], 120).tolist())
        pseudo = []
        for s in pseudo_slots:
            pseudo.append({'slot': int(s), 'fdi': int(ZIGZAG_FDI_ORDER[s]),
                           'pts': downsample(filled[s], 500).tolist()})
        cases.append({'patient': pid, 'n_real': len(real_slots), 'n_pseudo': len(pseudo_slots),
                      'real_pts': real_pts, 'pseudo': pseudo})
        if len(cases) >= args.n:
            break

    payload = {'fdi_order': list(ZIGZAG_FDI_ORDER), 'cases': cases}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('window.PSEUDO_DATA = ' + json.dumps(payload) + ';\n')
    print(f'WROTE {args.out}: {len(cases)} cases', flush=True)
    for c in cases:
        print(f"  {c['patient']}: real {c['n_real']} + pseudo {c['n_pseudo']} "
              f"(FDI {[p['fdi'] for p in c['pseudo']]})", flush=True)


if __name__ == '__main__':
    main()
