"""에폭별 크라운 진화 웹 시각화용 데이터 준비 (치아 1개 단위).

runs2/ply/patient{0,1,2}_{ep1100..2000,gt}.ply 는 gen_progression.py 가 TARGET_SLOTS=[12,13]
두 치아를 1024점씩 이어 붙인 점구름(2048점). 이를 치아별로 반 쪼개 → 치아당 CD(ep→GT) 계산.
개선이 가장 큰 환자 1명의 '치아별' 데이터를 runs2/viz/gen_progression/data.js 로 덤프.
(뷰어에서 두 치아 중 한 개만 선택해 본다.)
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from crowngen.losses.chamfer import chamfer_distance_l1
from crowngen.data.fdi import ZIGZAG_FDI_ORDER

PLY = 'runs2/ply'
PATIENTS = [0, 1, 2]
EPOCHS = ['1100', '1300', '1500', '1700', '1900', '2000']
TARGET_SLOTS = [12, 13]          # gen_progression.py 기준 (이어붙인 순서)
PER = 1024                       # 치아당 점 수 (gen_progression n_points)


def parse_ply(path):
    lines = open(path).read().splitlines()
    n = 0; start = 0
    for i, l in enumerate(lines):
        if l.startswith('element vertex'):
            n = int(l.split()[-1])
        if l.strip() == 'end_header':
            start = i + 1; break
    return np.array([[float(x) for x in lines[j].split()[:3]] for j in range(start, start + n)],
                    dtype=np.float32)


def cd_l1(a, b):
    with torch.no_grad():
        return chamfer_distance_l1(torch.from_numpy(a).unsqueeze(0),
                                   torch.from_numpy(b).unsqueeze(0)).item()


def main():
    allp = {}
    for p in PATIENTS:
        gt = parse_ply(f'{PLY}/patient{p}_gt.ply')
        gens = {e: parse_ply(f'{PLY}/patient{p}_ep{e}.ply') for e in EPOCHS}
        teeth = []
        improves = []
        for ti, slot in enumerate(TARGET_SLOTS):
            gtt = gt[ti * PER:(ti + 1) * PER]
            series = []
            for e in EPOCHS:
                gent = gens[e][ti * PER:(ti + 1) * PER]
                series.append({'ep': int(e), 'pts': gent.tolist(),
                               'cd': round(cd_l1(gent, gtt) * 1e3, 1)})
            teeth.append({'slot': int(slot), 'fdi': int(ZIGZAG_FDI_ORDER[slot]),
                          'gt_pts': gtt.tolist(), 'series': series})
            improves.append(series[0]['cd'] - series[-1]['cd'])
        allp[p] = {'teeth': teeth, 'improve': sum(improves)}
        print(f"patient{p}: ΔCD 치아별 {[round(x,1) for x in improves]}  합 {sum(improves):.1f}", flush=True)

    best = max(allp, key=lambda p: allp[p]['improve'])
    print(f'\n선택: patient{best} (치아들 합 기준 개선 최대)', flush=True)
    payload = {'patient': best, 'epochs': EPOCHS, 'teeth': allp[best]['teeth']}

    out = 'runs2/viz/gen_progression/data.js'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        f.write('window.GEN_PROGRESSION_DATA = ' + json.dumps(payload) + ';\n')
    print(f'WROTE {out} ({os.path.getsize(out)//1024} KB)', flush=True)
    for t in payload['teeth']:
        cds = [s['cd'] for s in t['series']]
        print(f"  치아 slot{t['slot']} FDI{t['fdi']}: {cds[0]:.1f}→{cds[-1]:.1f} (Δ{cds[0]-cds[-1]:+.1f})", flush=True)


if __name__ == '__main__':
    main()
