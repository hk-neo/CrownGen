"""aligned_data(사용자 강체 정렬: rest position + FDI 배치) → 학습 npz.

preprocess_teeth3ds.py 와 다른 점: PCA standardize 를 안 함 (사용자 정렬 보존).
대신: 전체 치열(상하 함께) centroid 중심화 + 균일 스케일(평균 치아 간격→0.16,
processed_norm2 스케일에 맞춰 eval/뷰어/임계 그대로 재사용). rest position(상하 분리) 유지.

출력 포맷: {jaw}_{fdi}_pc / {jaw}_{fdi}_bound / {jaw}_labels  (processed_norm2 호환).
"""
import argparse, os, sys, json
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
import numpy as np
import importlib.util
# preprocess_teeth3ds 함수 재사용
_spec = importlib.util.spec_from_file_location('pre', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preprocess_teeth3ds.py'))
pre = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(pre)

TARGET_SPACING = 0.16   # processed_norm2 평균 치아 간격에 맞춤


def process_patient(pid, data_dir, n_points=1024):
    jaw_clouds = {}
    for jaw in ('upper', 'lower'):
        jd = Path(data_dir) / jaw / pid
        obj, js = jd / f'{pid}_{jaw}.obj', jd / f'{pid}_{jaw}.json'
        if not obj.exists() or not js.exists():
            continue
        verts, faces = pre.load_obj(obj)
        labels, instances, _ = pre.load_json_labels(js)
        for fdi in set(labels) - {0}:
            pc = pre.extract_tooth_point_cloud(verts, faces, labels, instances, int(fdi), n_points)
            if pc is not None:
                jaw_clouds.setdefault(jaw, {})[int(fdi)] = pc
    if not jaw_clouds:
        return None

    # 전체 치아 centroid (상하 함께) → 중심화 기준 + 스케일(평균 간격)
    all_cents = []
    for jaw, clouds in jaw_clouds.items():
        for fdi, pc in clouds.items():
            all_cents.append(pc.mean(0))
    all_cents = np.array(all_cents)
    center = all_cents.mean(0)
    # 평균 최근접 centroid 간격 (numpy)
    if len(all_cents) >= 2:
        D = np.linalg.norm(all_cents[:, None] - all_cents[None, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        spacing = float(D.min(axis=1).mean())
    else:
        spacing = 10.0
    scale = TARGET_SPACING / max(spacing, 1e-6)

    save = {}
    for jaw, clouds in jaw_clouds.items():
        labels_out = []
        for fdi, pc in clouds.items():
            pc_s = ((pc - center) * scale).astype(np.float32)
            bound = pre.compute_cylinder_boundary(pc_s)   # (cx,cy,cz,r,h) — scaled frame
            save[f'{jaw}_{fdi}_pc'] = pc_s
            save[f'{jaw}_{fdi}_bound'] = bound
            labels_out.append(fdi)
        save[f'{jaw}_labels'] = np.array(sorted(labels_out))
    n = sum(len(c) for c in jaw_clouds.values())
    return save, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='Data/SourceC_Teeth3DS/aligned_data')
    ap.add_argument('--split_file', default='Data/SourceC_Teeth3DS/train_val_split.json')
    ap.add_argument('--output_dir', default='Data/aligned_norm')
    ap.add_argument('--n_points', type=int, default=1024)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    sp = json.load(open(args.split_file))
    pids = []
    for k in ('stage1_train', 'stage1_val', 'stage2_train', 'stage2_val'):
        pids += sp.get(k, [])
    pids = sorted(set(pids))
    print(f'aligned preprocess: {len(pids)} pids → {args.output_dir}', flush=True)
    ok = 0
    for i, pid in enumerate(pids):
        r = process_patient(pid, args.data_dir, args.n_points)
        if r is None:
            print(f'  skip {pid} (no data)'); continue
        save, n = r
        np.savez_compressed(Path(args.output_dir) / f'{pid}.npz', **save)
        ok += 1
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(pids)} ({ok} ok)', flush=True)
    print(f'DONE: {ok}/{len(pids)} → {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
