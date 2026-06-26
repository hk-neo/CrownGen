"""boundary 위치 진단: 평면(교합면 xy) 도면.
present 치아(파랑) + OLD 예측 빈슬롯(주황) + NEW 예측 빉슬롯(초록) 을
중심(cx,cy)+반지름(r) 원으로 그려, 빈 자리(틈)에 들어갔는지 치아 위에 겹쳤는지 확인.
upper/lower 분리. PNG 저장.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from crowngen.data.fdi import ZIGZAG_FDI_ORDER

OLD = 'runs2/pseudo_compare/old'
NEW = 'runs2/pseudo_compare/new'


def jaw_of(fdi):
    return 'upper' if fdi // 10 in (1, 2) else 'lower'


def load_bounds(pid, sub):
    d = np.load(f'runs2/pseudo_compare/{sub}/{pid}.npz')
    valid = d['valid']
    pb = d['pred_bound']          # (28,5) cx,cy,cz,h,r (new 슬롯은 예측, present 도 예측치 들어있음)
    # present 치아 GT bound 는 원본 norm2 에서
    orig = np.load(f'Data/processed_norm2/{pid}.npz')
    gt = np.zeros((28, 5))
    for s, fdi in enumerate(ZIGZAG_FDI_ORDER):
        k = f'{jaw_of(fdi)}_{fdi}_bound'
        if k in orig:
            b = orig[k]; gt[s] = [b[0], b[1], b[2], b[4], b[3]]   # cx,cy,cz,h,r
    return valid, gt, pb


def draw_circle(ax, cx, cy, r, **kw):
    ax.add_patch(plt.Circle((cx, cy), r, **kw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pids', nargs='+', default=['0132CR0A', '013Z9SM2', 'V5KEFD0N'])
    ap.add_argument('--out', default='runs2/diag_boundary')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for pid in args.pids:
        valid, gt, pb_new = load_bounds(pid, 'new')
        _, _, pb_old = load_bounds(pid, 'old')
        for jaw in ('upper', 'lower'):
            slots = [s for s, fdi in enumerate(ZIGZAG_FDI_ORDER) if jaw_of(fdi) == jaw]
            if not any(valid[s] == 0 for s in slots):
                continue
            fig, ax = plt.subplots(1, 1, figsize=(9, 7))
            # present (GT bound)
            for s in slots:
                if valid[s] != 0:
                    draw_circle(ax, gt[s, 0], gt[s, 1], gt[s, 4], fill=True,
                                fc='#9ec5fe', ec='#2a6fd6', alpha=0.55)
                    ax.text(gt[s, 0], gt[s, 1], str(ZIGZAG_FDI_ORDER[s]), fontsize=7,
                            ha='center', va='center', color='#10316b')
            # missing: OLD 예측 (주황 점선) / NEW 예측 (초록 실선)
            for s in slots:
                if valid[s] == 0:
                    fdi = ZIGZAG_FDI_ORDER[s]
                    draw_circle(ax, pb_old[s, 0], pb_old[s, 1], pb_old[s, 4], fill=False,
                                ec='#f39c12', lw=2.0, ls='--')
                    draw_circle(ax, pb_new[s, 0], pb_new[s, 1], pb_new[s, 4], fill=False,
                                ec='#2ecc71', lw=2.2)
                    ax.plot(pb_old[s, 0], pb_old[s, 1], 'o', color='#f39c12', ms=3)
                    ax.plot(pb_new[s, 0], pb_new[s, 1], 's', color='#2ecc71', ms=3)
                    ax.text(pb_new[s, 0], pb_new[s, 1] + 0.04, str(fdi), fontsize=7,
                            ha='center', color='#1a7d44')
            # 인접 present-present 간격 참고선
            pres = [s for s in slots if valid[s] != 0]
            spacings = []
            for i in range(len(pres) - 1):
                d = np.hypot(gt[pres[i], 0] - gt[pres[i + 1], 0], gt[pres[i], 1] - gt[pres[i + 1], 1])
                spacings.append(d)
            sp = np.mean(spacings) if spacings else float('nan')
            ax.set_aspect('equal'); ax.grid(alpha=0.3)
            ax.set_title(f'{pid} · {jaw} jaw · 파랑=present(GT) / 주황점선=OLD / 초록=NEW\n'
                         f'present 인접간격 평균 {sp:.3f} (빈슬롯 예측이 이보다 가까우면 치아에 겹침)')
            ax.legend(handles=[
                plt.Line2D([], [], marker='o', color='w', markerfacecolor='#9ec5fe', markersize=12, label='present (GT)'),
                plt.Line2D([], [], color='#f39c12', lw=2, ls='--', label='OLD 예측 빈슬롯'),
                plt.Line2D([], [], color='#2ecc71', lw=2, label='NEW(G1) 예측 빈슬롯'),
            ], loc='upper right')
            out = f'{args.out}/{pid}_{jaw}.png'
            plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
            print(f'  {out}  (present 간격 {sp:.3f})', flush=True)
    print('DONE')


if __name__ == '__main__':
    main()
