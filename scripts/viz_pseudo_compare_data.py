"""old vs new(G1) boundary pseudo-crown 비교 뷰어용 데이터.

runs2/pseudo_compare/{old,new}/{pid}.npz 를 읽어:
  - real_pts: present 치아 점 (양쪽 공통, 회색)
  - old/new: 빈 슬롯에 생성된 crown 점 (각 boundary 결과)
→ runs2/viz/pseudo_compare/data.js
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from crowngen.data.fdi import ZIGZAG_FDI_ORDER

OLD = 'runs2/pseudo_compare/old'
NEW = 'runs2/pseudo_compare/new'
ARCH = 'runs2/pseudo_compare/arch'
VARIANTS = {'old': OLD, 'new': NEW, 'arch': ARCH}


def jaw(f):
    return 'upper' if f // 10 in (1, 2) else 'lower'


def load_slots(path):
    d = np.load(path)
    slots = {}
    valid = d['valid'] if 'valid' in d else None
    for s, fdi in enumerate(ZIGZAG_FDI_ORDER):
        k = f'{jaw(fdi)}_{fdi}_pc'
        if k in d:
            slots[s] = np.asarray(d[k], dtype=np.float32)
    return slots, valid


def downsample(pts, per):
    if len(pts) > per:
        idx = np.linspace(0, len(pts) - 1, per).astype(int)
        return pts[idx]
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pids', nargs='+', default=['0132CR0A', '013Z9SM2', '0IU0UV8E',
                                                   '38JDN4ZV', 'SZQ66Y5A', '0142CYK4'])
    ap.add_argument('--out', default='runs2/viz/pseudo_compare/data.js')
    args = ap.parse_args()

    cases = []
    for pid in args.pids:
        paths = {v: f'{VARIANTS[v]}/{pid}.npz' for v in VARIANTS}
        if not all(os.path.exists(p) for p in paths.values()):
            print(f'  skip {pid} (not all variants generated)'); continue
        vslots = {v: load_slots(paths[v]) for v in VARIANTS}
        valid = vslots['old'][1] if vslots['old'][1] is not None else vslots['new'][1]
        missing = [s for s in range(28) if valid[s] == 0]
        present = [s for s in range(28) if valid[s] != 0]
        if not missing:
            continue

        real_upper = []; real_lower = []
        for s in present:
            pts = downsample(vslots['new'][0].get(s, vslots['old'][0].get(s)), 120).tolist()
            if jaw(ZIGZAG_FDI_ORDER[s]) == 'upper':
                real_upper.extend(pts)
            else:
                real_lower.extend(pts)

        def pseudo(slots):
            return [{'slot': int(s), 'fdi': int(ZIGZAG_FDI_ORDER[s]),
                     'pts': downsample(slots[s], 500).tolist()} for s in missing if s in slots]

        # 겹침: 빈슬롯 crown 중심 vs 전체 arch 최근접거리 (임계 0.06 = 치아 반지름급, 진짜 붕괴)
        def overlap_stats(slots):
            centers = np.zeros((28, 2))
            for s in range(28):
                if s in slots:
                    centers[s] = slots[s].mean(0)[:2]
            nn, ov = [], 0
            for s in missing:
                d = np.linalg.norm(centers - centers[s], axis=1); d[s] = 9
                m = d.min(); nn.append(float(m))
                if m < 0.06:
                    ov += 1
            return {'ov': ov, 'n': len(missing), 'pct': round(ov / max(1, len(missing)) * 100),
                    'nn': round(float(np.mean(nn)), 3)}

        case = {'patient': pid, 'n_real': len(present), 'n_miss': len(missing),
                'real_upper_pts': real_upper, 'real_lower_pts': real_lower}
        for v in VARIANTS:
            case[v] = pseudo(vslots[v][0])
            case[f'{v}_stat'] = overlap_stats(vslots[v][0])
        cases.append(case)

    payload = {'fdi_order': list(ZIGZAG_FDI_ORDER), 'cases': cases}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('window.PCMP_DATA = ' + json.dumps(payload) + ';\n')
    print(f'WROTE {args.out}: {len(cases)} cases', flush=True)


if __name__ == '__main__':
    main()
