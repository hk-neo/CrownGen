"""Teeth3DS 원본 OBJ + JSON(치아 라벨) → 치아별 FDI 색 PLY.

MeshLab/Blender 로 열면 각 치아가 FDI별 색으로 보임 (raw 메시 형상 그대로,
위치/스케일 변환 없음 — 원본 그대로 확인용). 0(잇몸/배경)=짙은 회색.

사용:
  python scripts/teeth3ds_to_ply.py --patient 0132CR0A            # upper+lower 각각
  python scripts/teeth3ds_to_ply.py --patient 0132CR0A --combine  # 상하 합친 하나
"""
import argparse, os, sys, json
import numpy as np
import colorsys

ROOT = 'Data/SourceC_Teeth3DS/Teeth3DS_full'


def fdi_color(fdi):
    """FDI → (r,g,b) 0-255. 사분면별 색상 + 치아별 음영."""
    if fdi == 0:
        return (45, 45, 48)                      # 잇몸/배경
    q, x = fdi // 10, fdi % 10                    # 사분면, 치아번호
    hue = {1: 0.00, 2: 0.08, 3: 0.55, 4: 0.65}[q]   # UR 빨강 / UL 주황 / LL 청 / LR 보라
    hue = (hue + x * 0.012) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def parse_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for ln in f:
            if ln.startswith('v '):
                p = ln.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif ln.startswith('f '):
                idx = [int(t.split('/')[0]) for t in ln.split()[1:]]
                idx = [(i - 1 if i > 0 else len(verts) + i) for i in idx]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))   # fan triangulate
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def write_ply(path, verts, colors, faces):
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(verts)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write(f'element face {len(faces)}\n')
        f.write('property list uchar int vertex_indices\nend_header\n')
        for v, c in zip(verts, colors):
            f.write(f'{v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n')
        for fc in faces:
            f.write(f'3 {fc[0]} {fc[1]} {fc[2]}\n')


def convert(patient, jaw, out):
    obj = f'{ROOT}/{jaw}/{patient}/{patient}_{jaw}.obj'
    js = f'{ROOT}/{jaw}/{patient}/{patient}_{jaw}.json'
    if not os.path.exists(obj):
        print(f'  skip {patient} {jaw} (no obj)'); return False
    verts, faces = parse_obj(obj)
    labels = np.array(json.load(open(js))['labels'])
    cmap = {fdi: fdi_color(fdi) for fdi in np.unique(labels)}
    colors = np.array([cmap[l] for l in labels], dtype=np.uint8)
    write_ply(out, verts, colors, faces)
    print(f'  {patient} {jaw}: {len(verts)}v/{len(faces)}f, FDI {sorted([k for k in cmap if k])} → {out}')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patient', required=True)
    ap.add_argument('--out_dir', default='runs2/teeth3ds_ply')
    ap.add_argument('--combine', action='store_true', help='상하를 한 PLY로 합침')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.combine:
        allv, allc, allf = [], [], []
        off = 0
        for jaw in ('upper', 'lower'):
            obj = f'{ROOT}/{jaw}/{args.patient}/{args.patient}_{jaw}.obj'
            js = f'{ROOT}/{jaw}/{args.patient}/{args.patient}_{jaw}.json'
            if not os.path.exists(obj): continue
            v, fc = parse_obj(obj); lab = np.array(json.load(open(js))['labels'])
            cm = {fdi: fdi_color(fdi) for fdi in np.unique(lab)}
            col = np.array([cm[l] for l in lab], dtype=np.uint8)
            allv.append(v); allc.append(col); allf.append(fc + off); off += len(v)
            print(f'  {jaw}: {len(v)}v')
        v = np.concatenate(allv); c = np.concatenate(allc); ff = np.concatenate(allf)
        out = f'{args.out_dir}/{args.patient}_both.ply'
        write_ply(out, v, c, ff)
        print(f'  → {out} ({len(v)}v/{len(ff)}f)')
    else:
        for jaw in ('upper', 'lower'):
            convert(args.patient, jaw, f'{args.out_dir}/{args.patient}_{jaw}.ply')


if __name__ == '__main__':
    main()
