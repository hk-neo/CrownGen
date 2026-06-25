"""FDI-correspondence generalized Procrustes 표준화.

발견: 현재(PCA) 표준화에서 슬롯별 centroid의 환자 간 표준편차가 0.37(정규화 단위)이고,
그중 Y(전후)축이 0.4로 압도적. 공식 boundary는 "슬롯 위치"로 예측(missing 치아가
컨텍스트를 못 봄)하므로, 이 Y 드리프트가 Dice 0의 직접 원인.

치료: FDI 라벨이 주는 치아별 대응(correspondence)을 이용해 각 환자의 28개 치아
centroid를 공통 아치 템플릿으로 rigid+scale 정렬(generalized Procrustes).
랜드마크 불필요. 정렬 변환(R,t,s)을 모든 포인트에 적용하고 boundary 를 재계산.

입력: Data/processed_norm (현재 normalized 데이터)
출력: Data/processed_norm2 (Procrustes 정렬된 데이터)
"""
import argparse, os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from crowngen.data.fdi import ZIGZAG_FDI_ORDER


def jaw_of(fdi):
    return 'upper' if fdi // 10 in (1, 2) else 'lower'


def load_centroids(npz_path):
    """한 환자의 (28,) centroid 배열. 결손 슬롯은 NaN."""
    d = np.load(npz_path)
    cents = np.full((28, 3), np.nan, dtype=np.float32)
    for i, fdi in enumerate(ZIGZAG_FDI_ORDER):
        k = f"{jaw_of(fdi)}_{fdi}_pc"
        if k in d:
            cents[i] = d[k].mean(0)
    return cents, d


def procrustes_align(X, Y):
    """X, Y: (P,3) 대응점(둘 다 centering 가정). X를 Y로 정렬하는 R, s 반환."""
    # optimal rotation via SVD
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ U.T * D if False else Vt.T @ U.T
    # 부호 보정 (반사 방지)
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    s = np.trace(np.diag(S) @ R) / (np.sum(X ** 2) + 1e-12) if False else \
        (S.sum() / (np.sum(X ** 2) + 1e-12))
    return R, float(s)


def generalized_procrustes(centroids_list, n_iters=7):
    """centroids_list: [(28,3) with NaN for missing]. 템플릿 + per-patient (R,t,s)."""
    # 1) 각 환자 centroid 를 present 치아 평균으로 centering
    centered = []
    for c in centroids_list:
        valid = ~np.isnan(c[:, 0])
        mu = c[valid].mean(0)
        cc = c - mu
        centered.append((cc, valid, mu))

    # 2) 템플릿 초기화: centered centroid 의 평균(NaN 무시)
    def nanmean(stack):
        with np.errstate(invalid='ignore'):
            m = np.nanmean(stack, axis=0)
        return np.nan_to_num(m, nan=0.0)

    stack = np.stack([c for c, _, _ in centered])  # (P,28,3)
    tmpl = nanmean(stack)
    tmpl -= tmpl.mean(0)
    tmpl /= (np.linalg.norm(tmpl, axis=1).max() + 1e-12)

    transforms = [None] * len(centroids_list)
    for it in range(n_iters):
        aligned = []
        for p, (cc, valid, mu) in enumerate(centered):
            idx = np.where(valid)[0]
            X = cc[idx] - cc[idx].mean(0)
            Y = tmpl[idx] - tmpl[idx].mean(0)
            R, s = procrustes_align(X, Y)
            # full transform: centered(cc) → R*s*cc, then +tmpl.mean - s*R*cc_mean? 간단히:
            # 점 p 원본 → mu + (p-mu) 에 R,s 적용 후 tmpl 프레임으로.
            transforms[p] = (R, s, mu)
            aligned_p = (cc[idx] - cc[idx].mean(0)) @ R.T * s  # 정렬된 (present)
            full = np.full((28, 3), np.nan)
            full[idx] = aligned_p
            aligned.append(full)
        tmpl_new = nanmean(np.stack(aligned))
        tmpl_new -= tmpl_new.mean(0)
        tmpl_new /= (np.linalg.norm(tmpl_new, axis=1).max() + 1e-12)
        if np.linalg.norm(tmpl_new - tmpl) < 1e-4:
            tmpl = tmpl_new
            break
        tmpl = tmpl_new
    return tmpl, transforms, centered


def apply_transform_to_points(d, R, s, mu):
    """npz 의 모든 치아 포인트 클라우드에 transform 적용. boundary 재계산."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('pre', 'scripts/preprocess_teeth3ds.py')
    pre = importlib.util.module_from_spec(spec); spec.loader.exec_module(pre)
    save = {}
    for k in d.files:
        if k.endswith('_pc'):
            pc = d[k]
            pc_new = ((pc - mu) @ R.T) * s
            save[k] = pc_new.astype(np.float32)
            # 대응 boundary 재계산 (좌표계가 바뀌었으므로)
            fdi = int(k.split('_')[1])
            bkey = k.replace('_pc', '_bound')
            save[bkey] = pre.compute_cylinder_boundary(pc_new)
        elif k.endswith('_bound'):
            continue  # 위에서 _pc 와 짝지어 재계산
        else:
            save[k] = d[k]
    # _pc 없는 치아의 _bound 만 남는 경우 정리: _pc 기준으로 이미 처리
    return save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', default='Data/processed_norm')
    ap.add_argument('--out_dir', default='Data/processed_norm2')
    ap.add_argument('--split_file', default='Data/SourceC_Teeth3DS/train_val_split.json')
    args = ap.parse_args()

    splits = json.load(open(args.split_file))
    # 템플릿은 모든 사용 가능 환자로 (더 많을수록 안정)
    files = sorted(glob.glob(f'{args.in_dir}/*.npz'))
    print(f'loading {len(files)} patients for Procrustes...')
    cents_list, data_list, pids = [], [], []
    for f in files:
        c, d = load_centroids(f)
        if (~np.isnan(c[:, 0])).sum() >= 14:  # 최소 14개 치아 있어야 정렬 의미
            cents_list.append(c); data_list.append(d); pids.append(os.path.basename(f)[:-4])
    print(f'  {len(cents_list)} patients with >=14 teeth')

    tmpl, transforms, centered = generalized_procrustes(cents_list)
    print('template (Procrustes mean) slot centroids computed.')

    os.makedirs(args.out_dir, exist_ok=True)
    # 정렬 후 환자 간 slot centroid std 측정
    aligned_centroids = []
    for p, (R, s, mu) in enumerate(transforms):
        c = cents_list[p]
        valid = ~np.isnan(c[:, 0])
        cc = c - mu
        ac = np.full((28, 3), np.nan)
        ac[valid] = cc[valid] @ R.T * s
        aligned_centroids.append(ac)
    stack = np.stack(aligned_centroids)
    std = np.nanstd(stack, axis=0)
    print(f'POST-PROCRUSTES slot centroid std (mean |std|) = {np.nanmean(np.linalg.norm(std,axis=1)):.3f} (was 0.37)')
    for fdi in [11, 16, 36]:
        i = ZIGZAG_FDI_ORDER.index(fdi)
        print(f'  FDI {fdi}: std_xyz={np.round(std[i],3)}')

    # 저장
    for p, (R, s, mu) in enumerate(transforms):
        save = apply_transform_to_points(data_list[p], R, s, mu)
        np.savez_compressed(f'{args.out_dir}/{pids[p]}.npz', **save)
    print(f'saved {len(pids)} → {args.out_dir}')


if __name__ == '__main__':
    main()
