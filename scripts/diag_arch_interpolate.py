"""아치 보간(arch interpolation) 프로토타입: 빈 슬롯 위치를 present 치아로 보간해
"치아 위 겹침"이 구조적으로 사라지는지 검증.

방법: 턱(upper/lower)마다 present 치아를 arch index(입 안의 순서)로 정렬.
빈 슬롯 t는 양옆 present 치아 a<b 사이에 (t-a)/(b-a) 비율로 선형 보간.
끝(보간 불가)은 가장 가까운 present 2개로 외삽.
위치(cx,cy,cz)만 보간; h,r 은 present 중간값 사용(여기선 위치 검증만).
결과: 보간 위치의 최근접 present 거리 → 0.08 이하(치아 위) 비율이 0에 가까워야 함.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from crowngen.data.fdi import ZIGZAG_FDI_ORDER


def jaw_of(fdi):
    return 'upper' if fdi // 10 in (1, 2) else 'lower'


def arch_index(fdi):
    """턱 안에서의 아치 순서 0..15. upper: 18→0..11→7,21→8..28→15. lower 동일 구조."""
    q, x = fdi // 10, fdi % 10
    if q in (1, 4):
        return 8 - x      # 18→0, 11→7  /  48→0, 41→7
    else:
        return 8 + (x - 1)  # 21→8, 28→15 / 31→8, 38→15


def interpolate_positions(valid, gt, interior_only=False):
    """valid(28,), gt(28,5) → 빈 슬롯에 보간 위치 채운 (28,5).
    interior_only=True 면 끝자리 결손(한쪽 present 없음)은 NaN → 호출자가 boundary fallback."""
    out = gt.copy()
    for jaw in ('upper', 'lower'):
        slots = [s for s, fdi in enumerate(ZIGZAG_FDI_ORDER) if jaw_of(fdi) == jaw]
        pres = [(arch_index(ZIGZAG_FDI_ORDER[s]), s) for s in slots if valid[s] != 0]
        miss = [(arch_index(ZIGZAG_FDI_ORDER[s]), s) for s in slots if valid[s] == 0]
        if not miss:
            continue
        pres.sort()
        # 인접 present 간 평균 아치 간격 (외삽 클램프 기준)
        aidx = [ai for ai, _ in pres]
        spacing = float(np.median([aidx[i+1]-aidx[i] for i in range(len(aidx)-1)])) if len(aidx) > 1 else 1.0
        spacing = max(spacing, 1.0)
        for t, s in miss:
            # t를 기준으로 좌우 present 찾기
            left = [(ai, sl) for ai, sl in pres if ai < t]
            right = [(ai, sl) for ai, sl in pres if ai > t]
            if interior_only and not (left and right):
                out[s] = np.nan          # 끝자리 결손: 외삽 안 함 → fallback
                continue
            if left and right:
                a = left[-1]; b = right[0]; frac = (t - a[0]) / (b[0] - a[0])
                out[s, :3] = gt[a[1], :3] + frac * (gt[b[1], :3] - gt[a[1], :3])
            elif right and len(right) >= 2:   # 왼쪽 끝 외삽: 더 왼쪽으로
                b = right[0]; c = right[1]
                dirv = gt[b[1], :3] - gt[c[1], :3]
                nrm = np.linalg.norm(dirv)
                if nrm > 1e-6:
                    dist_per = nrm / max(1e-6, abs(b[0] - c[0]))
                    step_idx = min(abs(t - b[0]), 2.0)          # 최대 2칸 외삽
                    out[s, :3] = gt[b[1], :3] + step_idx * dist_per * (dirv / nrm)
                else:
                    out[s, :3] = gt[b[1], :3]
            elif left and len(left) >= 2:     # 오른쪽 끝 외삽: 더 오른쪽으로
                a = left[-1]; c = left[-2]
                dirv = gt[a[1], :3] - gt[c[1], :3]
                nrm = np.linalg.norm(dirv)
                if nrm > 1e-6:
                    dist_per = nrm / max(1e-6, abs(a[0] - c[0]))
                    step_idx = min(abs(t - a[0]), 2.0)
                    out[s, :3] = gt[a[1], :3] + step_idx * dist_per * (dirv / nrm)
                else:
                    out[s, :3] = gt[a[1], :3]
            elif right:
                out[s, :3] = gt[right[0][1], :3]   # present 1개뿐: 불가피하게 근처
            elif left:
                out[s, :3] = gt[left[-1][1], :3]
            # h,r 은 일단 present 중간값
            if pres:
                hs = [gt[sl, 3] for _, sl in pres]; rs = [gt[sl, 4] for _, sl in pres]
                out[s, 3] = float(np.median(hs)); out[s, 4] = float(np.median(rs))
    return out


def main():
    pids = ['0132CR0A', '013Z9SM2', '8MTEIYKY', '4O6ZE6F3', '4J24X0ES', '01MAVT6A',
            'XNNFHCFV', '12MM6PD7', 'LBY32W80', '8JYS0DD1', 'EUCFQ9KW', 'V5KEFD0N',
            '01JHAGK0', '0140E7V2', 'KS84R9LU', '0151RHMK', '38JDN4ZV', 'FJPU30RL']
    all_interp = []
    for pid in pids:
        orig = np.load(f'Data/processed_norm2/{pid}.npz')
        dn = np.load(f'runs2/pseudo_compare/new/{pid}.npz')
        valid = dn['valid']
        gt = np.full((28, 5), np.nan)
        for s, fdi in enumerate(ZIGZAG_FDI_ORDER):
            k = f'{jaw_of(fdi)}_{fdi}_bound'
            if k in orig:
                b = orig[k]; gt[s] = [b[0], b[1], b[2], b[4], b[3]]
        interp = interpolate_positions(valid, gt)
        for s in range(28):
            if valid[s] == 0:
                pres = np.where(valid != 0)[0]
                d = min(np.hypot(gt[p, 0] - interp[s, 0], gt[p, 1] - interp[s, 1]) for p in pres)
                all_interp.append(d)
    all_interp = np.array(all_interp)
    print(f'아치 보간 위치 → 최근접 present 거리 ({len(all_interp)} 빈슬롯)')
    print(f'  중간 {np.median(all_interp):.3f} / 평균 {all_interp.mean():.3f} / 최소 {all_interp.min():.3f}')
    print(f'  치아 위(<0.08): {(all_interp<0.08).mean()*100:.0f}%   심각(<0.06): {(all_interp<0.06).mean()*100:.0f}%')
    print(f'  (대조) NEW boundary: 치아위 42% / 심각 23%')


if __name__ == '__main__':
    main()
