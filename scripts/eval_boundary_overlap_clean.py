"""G1 boundary 개선을 apples-to-apples 로 검증.

동일 부분무치아 환자(stage2_val, 두 모델 모두 held-out)에서
old(boundary_official_long) vs new(boundary_g1_best) 의 위치 비교.

노이즈 원인 2가지를 제거:
  1) 겹침은 '진짜 빈 슬롯(valid==0)' 의 예측 위치로만 측정 → 마스킹 무작위성 배제.
     (학습 eval 은 present 치아를 무작위 마스킹 → 매번 케이스가 달라서 흔들렸음.)
  2) 두 모델에 동일 입력 → 비교가 공정.

측정:
  Dice      — present 치아 light 마스킹(GT 있음), 환자별 고정 시드로 동일 마스크.
  overlap%  — 빈 슬롯 예측위치 vs 전체 arch(present GT + 빈 슬롯 예측) 최소 xy 거리 < 임계.
              임계 0.08(인접 포함, 느슨) · 0.05(진짜 붕괴, 엄격) 둘 다.
  nn-dist   — 빈 슬롯→가장 가까운 다른 슬롯 평균 거리 (정상 인접 간격 참고용).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
import json, random
import numpy as np
import torch

from crowngen.external import BoundEncoder
from crowngen.data.fdi import ZIGZAG_FDI_ORDER
from train_boundary_g1 import _cyl_mask, _dice_iou, make_light_mask, collate  # noqa


def jaw_of(fdi):
    return 'upper' if fdi // 10 in (1, 2) else 'lower'


def load_patient(path, n_points=1024, rng=np.random):
    d = np.load(path)
    pts = np.zeros((28, 3, n_points), np.float32)
    bnd = np.zeros((28, 5), np.float32)
    valid = np.zeros(28, np.float32)
    for s, fdi in enumerate(ZIGZAG_FDI_ORDER):
        k = f'{jaw_of(fdi)}_{fdi}_pc'
        if k in d:
            pc = d[k]
            if pc.shape[0] >= n_points:
                pc = pc[rng.permutation(pc.shape[0])[:n_points]]
            else:
                pc = pc[rng.choice(pc.shape[0], n_points, replace=True)]
            pts[s] = pc.T.astype(np.float32)
            bk = k.replace('_pc', '_bound')
            if bk in d:
                b = d[bk]; bnd[s] = [b[0], b[1], b[2], b[4], b[3]]
            valid[s] = 1.0
    return pts, bnd, valid


@torch.no_grad()
def run_model(model, pts, valid, device, max_missing):
    """부분무치아 입력 → 전체 28슬롯 cylinder 예측(present 도 포함해 일관 비교)."""
    x = torch.from_numpy(pts).unsqueeze(0).to(device)
    v = torch.from_numpy(valid).unsqueeze(0).to(device)
    exist = v.view(1, 28, 1, 1)            # 진짜 present 만 context
    pred = model(x, exist)[0].cpu().numpy()  # (28,5) cx,cy,cz,h,r
    return pred


def overlap_stats(bnd_full, missing_idx, thrs=(0.08, 0.05)):
    """빈 슬롯 예측위치 가 전체 arch 와 얼마나 가까운지.
    bnd_full: (28,5) — present 는 GT, missing 은 예측 채운 전체 위치."""
    if len(missing_idx) == 0:
        return None
    centers = bnd_full[:, :2]                       # (28,2) xy
    n_ov = {t: 0 for t in thrs}; nn = []
    for s in missing_idx:
        d = np.linalg.norm(centers - centers[s], axis=1)
        d[s] = 9                                     # 자기 자신 제외
        m = d.min()
        nn.append(m)
        for t in thrs:
            if m < t:
                n_ov[t] += 1
    return dict(n=len(missing_idx),
               ov={t: n_ov[t] / len(missing_idx) for t in thrs},
               nn_mean=float(np.mean(nn)))


def present_spacing(bnd_full, valid):
    """정상 present-present 인접 간격 참고 (임계 의미 파악용)."""
    pres = np.where(valid > 0)[0]
    if len(pres) < 2:
        return None
    c = bnd_full[pres, :2]
    d = np.linalg.norm(c[:, None] - c[None, :], axis=-1)
    d[d == 0] = 9
    return float(d.min(axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='Data/processed_norm2')
    ap.add_argument('--split_file', default='Data/SourceC_Teeth3DS/train_val_split.json')
    ap.add_argument('--old_ckpt', default='runs2/boundary_official_long.pt')
    ap.add_argument('--new_ckpt', default='runs2/boundary_g1_best.pt')
    ap.add_argument('--n_points', type=int, default=1024)
    args = ap.parse_args()

    device = torch.device('cuda')
    sp = json.load(open(args.split_file))

    # 부분무치아 환자(stage2_val, 두 모델 모두 held-out)
    pids = []
    for p in sp.get('stage2_val', []):
        f = f'{args.data_dir}/{p}.npz'
        if not os.path.exists(f):
            continue
        d = np.load(f)
        n = sum(1 for fdi in ZIGZAG_FDI_ORDER if f'{jaw_of(fdi)}_{fdi}_pc' in d)
        if 14 <= n < 28:
            pids.append(p)
    print(f'부분무치아 평가 환자: {len(pids)}명', flush=True)

    old_m = BoundEncoder(5, 0.3, 6, mask_mode='official').to(device)
    old_m.load_state_dict(torch.load(args.old_ckpt, map_location=device)); old_m.eval()
    new_m = BoundEncoder(5, 0.3, 12, mask_mode='official').to(device)
    new_m.load_state_dict(torch.load(args.new_ckpt, map_location=device)); new_m.eval()
    print('모델 로드: old=%s new=%s' % (os.path.basename(args.old_ckpt), os.path.basename(args.new_ckpt)), flush=True)

    THRS = (0.08, 0.05)
    agg = {tag: {'dice': [], 'ov': {t: [] for t in THRS}, 'nn': [], 'spacing': []}
           for tag in ('old', 'new')}

    for i, pid in enumerate(pids):
        # 환자별 고정 시드 → 동일 입력/동일 light 마스크 (비교 공정)
        rng = np.random.RandomState(1000 + i)
        pts, bnd, valid = load_patient(f'{args.data_dir}/{pid}.npz', args.n_points, rng)
        miss = list(np.where(valid == 0)[0])

        # Dice: present 치아 light 마스킹(GT 있음), 동일 마스크 양쪽 적용
        v_t = torch.from_numpy(valid).unsqueeze(0)
        random.seed(2000 + i)  # make_light_mask 내부 random 고정
        exist, mmiss = make_light_mask(v_t, 6)
        exist = exist.to(device); mmiss = mmiss.squeeze(0).numpy()
        gt = bnd

        for tag, model, mmx in (('old', old_m, 6), ('new', new_m, 12)):
            pred_full = run_model(model, pts, valid, device, mmx)
            # Dice (마스킹된 present 치아)
            d, _ = _dice_iou(pred_full, gt, mmiss)
            if d:
                agg[tag]['dice'].append(float(np.mean(d)))
            # 겹침: 진짜 빈 슬롯 예측위치 vs 전체 arch
            bnd_full = bnd.copy()
            bnd_full[miss] = pred_full[miss]
            st = overlap_stats(bnd_full, miss, THRS)
            if st:
                for t in THRS:
                    agg[tag]['ov'][t].append(st['ov'][t])
                agg[tag]['nn'].append(st['nn_mean'])
            spc = present_spacing(bnd_full, valid)
            if spc:
                agg[tag]['spacing'].append(spc)
        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{len(pids)}...', flush=True)

    print('\n=== 결과 (평균, {len(pids)}명 부분무치아) ===' if False else '\n=== 결과 ===')
    for tag in ('old', 'new'):
        a = agg[tag]
        md = float(np.mean(a['dice'])) * 100 if a['dice'] else 0
        spc = float(np.mean(a['spacing'])) if a['spacing'] else 0
        print(f'\n[{tag}]')
        print(f'  Dice(light 마스킹)      : {md/100:.3f}')
        print(f'  겹침%(0.08, 느슨/인접)  : {np.mean(a["ov"][0.08])*100:.1f}%')
        print(f'  겹침%(0.05, 엄격/붕괴)  : {np.mean(a["ov"][0.05])*100:.1f}%')
        print(f'  빈슬롯→최근접 평균거리  : {np.mean(a["nn"]):.3f}')
        print(f'  정상 present-present 간격: {spc:.3f}  (참고: 임계 의미)')

    print('\n→ 겹침% (0.05 엄격) old→new 감소폭이 G1 효과의 깨끗한 척도.')


if __name__ == '__main__':
    main()
