"""
CrownGen 데이터 전처리 스크립트
Teeth3DS+ OBJ/JSON → CrownGen 학습용 포인트 클라우드 (.npy)

수행 작업:
1. OBJ 메쉬에서 치아별 포인트 클라우드 추출 (1024 포인트/치아, 균일 샘플링)
2. JSON에서 FDI 라벨 파싱 → 치아 인스턴스 분리
3. 상/하악 페어링 및 교합 정렬 (간이 버전)
4. 좌표계 표준화 (4전치 중심=원점)
5. CrownGen 학습 형식으로 저장

출력: processed/{patient_id}.pt
  - context_teeth: dict[fdi_label -> np.ndarray(1024, 3)]
  - target_teeth: dict[fdi_label -> np.ndarray(1024, 3)]  (훈련 시 랜덤 마스킹)
  - all_teeth: dict[fdi_label -> np.ndarray(1024, 3)]
  - boundary_params: dict[fdi_label -> (cx, cy, cz, r, h)]

사용법:
  python preprocess_teeth3ds.py --data_dir ../Data/SourceC_Teeth3DS/Teeth3DS_full \
                                --split_file ../Data/SourceC_Teeth3DS/train_val_split.json \
                                --output_dir ../Data/processed \
                                --split stage1_train
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# ──────────────────────────────────────────────────
# OBJ 파일 로더
# ──────────────────────────────────────────────────

def load_obj(obj_path):
    """OBJ 파일에서 vertices와 faces를 로드"""
    vertices = []
    faces = []
    with open(obj_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                # Handle f v or f v/vt or f v/vt/vn formats
                face_verts = []
                for p in parts[1:]:
                    idx = int(p.split('/')[0]) - 1  # OBJ is 1-indexed
                    face_verts.append(idx)
                faces.append(face_verts)
    return np.array(vertices, dtype=np.float32), faces


def load_json_labels(json_path):
    """JSON에서 FDI 라벨과 instance ID를 로드"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['labels'], data['instances'], data.get('jaw', 'unknown')


# ──────────────────────────────────────────────────
# 포인트 클라우드 추출
# ──────────────────────────────────────────────────

def extract_tooth_point_cloud(vertices, faces, labels, instances, target_label, n_points=1024):
    """
    특정 FDI 라벨의 치아 포인트 클라우드를 추출

    Args:
        vertices: (N, 3) 정점 배열
        faces: 면 리스트
        labels: 정점별 FDI 라벨
        instances: 정점별 instance ID
        target_label: 추출할 FDI 라벨
        n_points: 샘플링할 포인트 수

    Returns:
        point_cloud: (n_points, 3) 균일 샘플링된 포인트 클라우드
    """
    # 해당 라벨의 정점 인덱스 찾기
    mask = np.array(labels) == target_label
    if not np.any(mask):
        return None

    tooth_vertices = vertices[mask]

    # 해당 치아가 포함된 면 찾기 (fan 삼각화: 다각형 면을 삼각형으로 변환)
    tooth_faces = []
    for face in faces:
        if all(labels[vi] == target_label for vi in face if vi < len(labels)):
            # face: [v0, v1, ..., vn] → (v0, vi, vi+1) 삼각형들
            for k in range(1, len(face) - 1):
                tooth_faces.append([face[0], face[k], face[k + 1]])

    if len(tooth_faces) == 0:
        # 면 기반 샘플링이 불가하면 정점에서 직접 샘플링
        return uniform_sample_from_vertices(tooth_vertices, n_points)

    # 면 표면에서 면적 가중 균일 샘플링
    return uniform_sample_from_mesh(tooth_vertices, tooth_faces, vertices, n_points)


def uniform_sample_from_vertices(vertices, n_points):
    """정점에서 균일 샘플링 (fallback)."""
    if len(vertices) <= n_points:
        indices = np.random.choice(len(vertices), n_points, replace=True)
        points = vertices[indices]
        noise = np.random.normal(0, 0.01, points.shape).astype(np.float32)
        points = points + noise
    else:
        indices = np.random.choice(len(vertices), n_points, replace=False)
        points = vertices[indices]

    return points.astype(np.float32)


def _triangle_area(v0, v1, v2):
    """세 점으로 이루어진 삼각형 면적."""
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)


def uniform_sample_from_mesh(tooth_verts, tooth_faces, all_vertices, n_points):
    """메쉬 표면에서 면적 가중 균일 샘플링 (논문: "sampled uniformly from its mesh surface").

    각 면의 면적에 비례해 면을 선택한 뒤, 무작위 중심좌표(barycentric)로 면
    내부의 점을 샘플링한다. 정점 밀도 편향을 피해 진짜 표면 위의 균일 분포를
    얻는다. 면이 없거나 너무 적으면 정점 샘플링으로 폴백.
    """
    if tooth_faces is None or len(tooth_faces) == 0:
        return uniform_sample_from_vertices(tooth_verts, n_points)

    faces = np.asarray(tooth_faces, dtype=np.int64)
    # 인덱스 범위 확인 (all_vertices 기준)
    faces = faces[np.all(faces < len(all_vertices), axis=1)]
    if len(faces) == 0:
        return uniform_sample_from_vertices(tooth_verts, n_points)

    p0 = all_vertices[faces[:, 0]]
    p1 = all_vertices[faces[:, 1]]
    p2 = all_vertices[faces[:, 2]]
    areas = _triangle_area(p0, p1, p2)
    total = areas.sum()
    if total <= 0:
        return uniform_sample_from_vertices(tooth_verts, n_points)

    probs = areas / total
    face_idx = np.random.choice(len(faces), size=n_points, p=probs)

    # 무작위 중심좌표
    r1 = np.sqrt(np.random.rand(n_points, 1))
    r2 = np.random.rand(n_points, 1)
    a = 1.0 - r1
    b = r1 * (1.0 - r2)
    c = r1 * r2

    pts = (a * p0[face_idx] + b * p1[face_idx] + c * p2[face_idx]).astype(np.float32)
    return pts


def min_enclosing_circle_2d(points):
    """2D 점집합의 최소 외접원 (Welzl 알고리즘).

    논문 기준의 GT 실린더는 XY 투영의 "minimal enclosing circle"로 정의된다.
    기존 구현(centroid + max 거리)은 비대칭 형태에서 최대 ~30% 큰 원을 준다.

    Returns:
        (cx, cy, r)
    """
    pts = np.asarray(points, dtype=np.float64)
    # 무작위화된 Welzl 은 외접원의 기대 성능이 좋다.
    idx = np.random.permutation(len(pts))
    pts = pts[idx]

    cx = cy = 0.0
    r = 0.0
    for i in range(len(pts)):
        p = pts[i]
        if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 <= r * r:
            continue
        # p is on the boundary: recompute from scratch using first i+1 points
        cx, cy, r = float(p[0]), float(p[1]), 0.0
        for j in range(i):
            q = pts[j]
            if (q[0] - cx) ** 2 + (q[1] - cy) ** 2 <= r * r:
                continue
            # circle through p and q
            cx = (p[0] + q[0]) / 2.0
            cy = (p[1] + q[1]) / 2.0
            r = np.hypot(p[0] - cx, p[1] - cy)
            for k in range(j):
                s = pts[k]
                if (s[0] - cx) ** 2 + (s[1] - cy) ** 2 <= r * r:
                    continue
                # circumscribed circle of triangle p, q, s
                ax, ay = p[0] - s[0], p[1] - s[1]
                bx, by = q[0] - s[0], q[1] - s[1]
                d = 2.0 * (ax * by - ay * bx)
                if abs(d) < 1e-12:
                    continue
                ux = (by * (ax * ax + ay * ay) - ay * (bx * bx + by * by)) / d
                uy = (ax * (bx * bx + by * by) - bx * (ax * ax + ay * ay)) / d
                cx = s[0] + ux
                cy = s[1] + uy
                r = np.hypot(ux, uy)
    return np.array([cx, cy, r], dtype=np.float32)


# ──────────────────────────────────────────────────
# 좌표계 표준화
# ──────────────────────────────────────────────────

def standardize_coordinate_system(upper_clouds, lower_clouds):
    """상/하악을 하나의 공유 정준 좌표계로 표준화.

    기존 구현의 결함 (치명적):
      - 상/하악을 *각각* 중심화 → 두 악의 centroid 가 모두 원점이 되어, 28슬롯
        텐서에서 상/하악이 같은 공간에 완전히 겹침 (대합치 관계 붕괴).
      - PCA 고유벡터를 구하고 적용하지 않음 (정렬 no-op).

    수정:
      - 상+하악을 합쳐 *한 번에* 정렬 (공유 프레임).
      - PCA: 분산이 가장 큰 축 = X(좌우), 중간 = Y(전후), 가장 작은 = Z(교합면
        법선/치아 장축). 치열궁은 평면형이므로 수직(Z) 분산이 가장 작다.
      - 좌우 반전 증강(x→-x)이 진짜 시상면 미러가 되도록 X를 좌우축으로 고정.
      - 원본 Source C는 상/하악이 미정합(같은 공간에 겹침)이므로, 상악은 +z,
        하악은 -z 로 분리해 근사 교합(occlusal plane ≈ z=0)을 부여.
        ※ 완전한 MCM 교합 정합은 치과 랜드마크가 필요해 여기서는 휴리스틱 사용.
      - 전체 치열을 [-1,1]에 가깝게 스케일 정규화 (config 의 ball-query 반경
        [0.2~0.5] 및 논문 CD(x1e3) 스케일과 일치).

    Returns:
        upper_out, lower_out, info(dict: scale, offset, rotation) —
        info 는 역변환(생성 결과를 원본 좌표로 되돌릴 때)용.
    """
    def stack(clouds):
        if not clouds:
            return np.zeros((0, 3), dtype=np.float32)
        return np.vstack(list(clouds.values()))

    all_points = np.concatenate([stack(upper_clouds), stack(lower_clouds)], axis=0)
    if all_points.shape[0] < 4:
        return upper_clouds, lower_clouds, {'scale': 1.0, 'offset': np.zeros(3),
                                            'rotation': np.eye(3), 'z_gap': 0.0}

    # 1) 중심 이동
    centroid = all_points.mean(axis=0)

    # 2) PCA 정렬 (고유값 오름차순: [Z, Y, X])
    cov = np.cov((all_points - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)        # columns ascending
    # X = 최대 분산(좌우), Y = 중간(전후), Z = 최소(수직/교합법선)
    R = eigvecs[:, ::-1]                            # (3,3) 열이 [X, Y, Z]

    # PCA 부호 모호성 제거: 각 주축 부호를 결정론적으로 고정
    transformed = (all_points - centroid) @ R
    signs = np.sign(transformed.sum(axis=0))
    signs[signs == 0] = 1.0
    R = R * signs                                   # 열 단위 부호 반전

    # 해부학적 handedness 고정: 환자 우측(사분면 1,4 = 11-17, 41-47)이 항상 +X,
    # 좌측(사분면 2,3)이 -X가 되도록 정렬. 이래야 좌우 반전 증강(x→-x)이 진짜
    # 시상면 미러가 되고 환자 간에 프레임이 일관된다.
    right_fd = set(range(11, 18)) | set(range(41, 48))
    left_fd = set(range(21, 28)) | set(range(31, 38))

    def side_mean_x(clouds):
        """현재 R 프레임에서 좌/우 치아의 평균 X."""
        r, l = [], []
        for fdi, pc in clouds.items():
            xm = float(((pc - centroid) @ R)[:, 0].mean())
            (r if int(fdi) in right_fd else l).append(xm)
        return (np.mean(r) if r else None, np.mean(l) if l else None)

    # reference = 상악 (원본 스캔에서 상악이 표준 방향). 상악이 없으면 하악 사용.
    ref_clouds = upper_clouds if upper_clouds else lower_clouds
    rr, rl = side_mean_x(ref_clouds)
    if rr is not None and rl is not None and rr < rl:
        R[:, 0] *= -1.0                             # reference 우측을 +X로

    # 하악이 상악 대비 X 반전 저장된 경우 보정(un-mirror). 비정합 분리 스캔에서
    # 하악이 상악과 좌우 반전되어 저장되는 일이 흔하다(상악 우측=+X, 하악 우측=-X).
    lower_flip_x = False
    if upper_clouds and lower_clouds:
        lr, ll = side_mean_x(lower_clouds)
        if lr is not None and ll is not None and lr < ll:
            lower_flip_x = True                     # 하악 X 뒤집기 → 상악과 일치

    def apply(clouds, z_shift, flip_x=False):
        out = {}
        for fdi, pc in clouds.items():
            p = (pc - centroid) @ R
            if flip_x:
                p[:, 0] *= -1.0
            p[:, 2] += z_shift
            out[fdi] = p.astype(np.float32)
        return out

    # 3) 상/하악 z 분리 (근사 교합)
    #    치열 전체 높이의 절반을 각 악의 오프셋으로 사용 → 교합면 ≈ z=0
    extent = transformed.max(axis=0) - transformed.min(axis=0)
    z_gap = float(extent[2]) * 0.5

    upper_out = apply(upper_clouds, +z_gap / 2.0, flip_x=False)
    lower_out = apply(lower_clouds, -z_gap / 2.0, flip_x=lower_flip_x)

    # 4) 스케일 정규화: 최대 반경을 1 로 (좌표 ∈ 약 [-1,1])
    combined = np.concatenate([stack(upper_out), stack(lower_out)], axis=0)
    scale = float(np.max(np.linalg.norm(combined, axis=1)))
    scale = scale if scale > 1e-6 else 1.0
    for clouds in (upper_out, lower_out):
        for fdi in clouds:
            clouds[fdi] = (clouds[fdi] / scale).astype(np.float32)

    info = {
        'scale': scale,
        'offset': centroid.astype(np.float32),
        'rotation': R.astype(np.float32),
        'z_gap': (z_gap / 2.0 / scale),   # 정규화 공간에서의 상/하악 z 오프셋
    }
    return upper_out, lower_out, info


# ──────────────────────────────────────────────────
# 실린더 경계(Boundary) 파라미터 계산
# ──────────────────────────────────────────────────

def compute_cylinder_boundary(point_cloud):
    """
    포인트 클라우드로부터 원통형 경계 파라미터 계산
    B = (cx, cy, cz, r, h)

    논문 기준 (표준화된 좌표계 전제 — 교합면 = XY 평면, 치아 장축 ≈ Z):
    - XY 평면 투영 → 최소 외접원 (cx, cy, r)  [Welzl]
    - Z축 방향 → 높이 h = z_max - z_min, 중심 cz = (z_min + z_max)/2
    """
    xy = point_cloud[:, :2]
    cx, cy, r = min_enclosing_circle_2d(xy)

    z = point_cloud[:, 2]
    z_min, z_max = z.min(), z.max()
    h = z_max - z_min
    cz = (z_min + z_max) / 2.0

    return np.array([cx, cy, cz, r, h], dtype=np.float32)


# ──────────────────────────────────────────────────
# 메인 전처리 파이프라인
# ──────────────────────────────────────────────────

def preprocess_patient(patient_id, data_dir, n_points=1024):
    """
    한 환자의 상/하악 스캔을 전처리하여 CrownGen 학습 형식으로 변환.

    핵심: 상/하악을 먼저 모두 추출한 뒤 *함께* 표준화(공유 정준 프레임)하여,
    상/하악이 원점에서 겹치는 버그를 방지한다.

    Returns:
        dict with keys: 'patient_id', 'jaws' -> {jaw: {tooth_clouds, boundaries, ...}},
        'norm_info'
    """
    result = {'patient_id': patient_id, 'jaws': {}, 'norm_info': None}

    jaw_clouds = {'upper': {}, 'lower': {}}
    for jaw in ['upper', 'lower']:
        jaw_dir = Path(data_dir) / jaw / patient_id
        obj_path = jaw_dir / f"{patient_id}_{jaw}.obj"
        json_path = jaw_dir / f"{patient_id}_{jaw}.json"

        if not obj_path.exists() or not json_path.exists():
            continue

        vertices, faces = load_obj(obj_path)
        labels, instances, jaw_label = load_json_labels(json_path)

        unique_labels = set(labels) - {0}
        for fdi_label in unique_labels:
            pc = extract_tooth_point_cloud(vertices, faces, labels, instances, fdi_label, n_points)
            if pc is not None:
                jaw_clouds[jaw][fdi_label] = pc

    if not jaw_clouds['upper'] and not jaw_clouds['lower']:
        return result

    # 상/하악을 함께 표준화 (공유 정준 좌표계 + 스케일 정규화 + z 분리)
    upper_out, lower_out, info = standardize_coordinate_system(
        jaw_clouds['upper'], jaw_clouds['lower']
    )
    result['norm_info'] = info

    for jaw, clouds in (('upper', upper_out), ('lower', lower_out)):
        if not clouds:
            continue
        boundaries = {fdi: compute_cylinder_boundary(pc) for fdi, pc in clouds.items()}
        result['jaws'][jaw] = {
            'tooth_clouds': clouds,
            'boundaries': boundaries,
            'n_teeth': len(clouds),
            'labels': sorted(clouds.keys()),
        }

    return result


def save_processed(result, output_dir):
    """전처리 결과를 .npz 파일로 저장"""
    patient_id = result['patient_id']
    out_path = Path(output_dir) / f"{patient_id}.npz"

    save_dict = {}
    for jaw_name, jaw_data in result['jaws'].items():
        for fdi, pc in jaw_data['tooth_clouds'].items():
            save_dict[f"{jaw_name}_{fdi}_pc"] = pc
            save_dict[f"{jaw_name}_{fdi}_bound"] = jaw_data['boundaries'][fdi]
        save_dict[f"{jaw_name}_labels"] = np.array(jaw_data['labels'])

    # 정규화 정보 (생성 결과를 원본 좌표로 역변환할 때 사용)
    info = result.get('norm_info')
    if info is not None:
        save_dict['norm_scale'] = np.array(info['scale'], dtype=np.float32)
        save_dict['norm_offset'] = info['offset'].astype(np.float32)
        save_dict['norm_rotation'] = info['rotation'].astype(np.float32)
        save_dict['norm_z_gap'] = np.array(info['z_gap'], dtype=np.float32)

    np.savez_compressed(out_path, **save_dict)


def main():
    parser = argparse.ArgumentParser(description='CrownGen 데이터 전처리')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Teeth3DS_full 디렉토리 경로')
    parser.add_argument('--split_file', type=str, required=True,
                        help='train_val_split.json 경로')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='전처리된 데이터를 저장할 디렉토리')
    parser.add_argument('--split', type=str, default='stage1_train',
                        choices=['stage1_train', 'stage1_val', 'external_test',
                                 'stage2_train', 'stage2_val'],
                        help='처리할 데이터 분할')
    parser.add_argument('--n_points', type=int, default=1024,
                        help='치아당 샘플링 포인트 수')
    parser.add_argument('--max_patients', type=int, default=None,
                        help='처리할 최대 환자 수 (테스트용)')
    args = parser.parse_args()

    # Load split
    with open(args.split_file) as f:
        splits = json.load(f)

    patient_ids = splits[args.split]
    if args.max_patients:
        patient_ids = patient_ids[:args.max_patients]

    print(f"=== CrownGen 데이터 전처리 ===")
    print(f"분할: {args.split}")
    print(f"환자 수: {len(patient_ids)}")
    print(f"포인트/치아: {args.n_points}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    success, failed = 0, 0

    for i, pid in enumerate(patient_ids):
        try:
            result = preprocess_patient(pid, args.data_dir, args.n_points)
            if result['jaws']:
                save_processed(result, args.output_dir)
                n_teeth = sum(j['n_teeth'] for j in result['jaws'].values())
                print(f"  [{i+1}/{len(patient_ids)}] ✅ {pid}: {n_teeth}개 치아")
                success += 1
            else:
                print(f"  [{i+1}/{len(patient_ids)}] ⚠️  {pid}: 치아 없음")
                failed += 1
        except Exception as e:
            print(f"  [{i+1}/{len(patient_ids)}] ❌ {pid}: {e}")
            failed += 1

    print(f"\n=== 완료 ===")
    print(f"성공: {success}, 실패: {failed}")
    print(f"출력: {args.output_dir}")


if __name__ == '__main__':
    main()
