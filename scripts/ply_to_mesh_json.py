"""runs2/mesh_demo/*.ply → runs2/viz/mesh_demo/data.js
Open3D로 binary PLY 읽어서 vertex+face JSON으로. Three.js 뷰어용.
crown env (python3.11) 로 실행."""
import os, re, json, glob, sys
import numpy as np
import open3d as o3d
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crowngen.data.fdi import ZIGZAG_FDI_ORDER

SRC = 'runs2/mesh_dpsr'  # DPSR 메시 (논문 방식)
MAX_FACES = 15000  # 웹 뷰어용 decimation (DPSR는 100K faces → 15K로 축소)
OUT = 'runs2/viz/mesh_demo/data.js'
NORM = 'Data/aligned_norm'


def jaw_of(fdi):
    return 'upper' if fdi // 10 in (1, 2) else 'lower'


def get_real_pts(pid):
    """환자의 present 치아 점 (회색 컨텍스트)."""
    path = f'{NORM}/{pid}.npz'
    if not os.path.exists(path):
        return []
    d = np.load(path)
    pts = []
    for fdi in ZIGZAG_FDI_ORDER:
        k = f'{jaw_of(fdi)}_{fdi}_pc'
        if k in d:
            pc = d[k]  # (N,3)
            if len(pc) > 100:
                idx = np.linspace(0, len(pc) - 1, 100).astype(int)
                pc = pc[idx]
            pts.extend(pc.tolist())
    return pts


def parse_ply(path):
    mesh = o3d.io.read_triangle_mesh(path)
    # 웹 뷰어용 decimation (DPSR는 100K faces → MAX_FACES로 축소)
    if len(mesh.triangles) > MAX_FACES:
        mesh = mesh.simplify_quadric_decimation(MAX_FACES)
        mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.triangles, dtype=np.uint32)
    return v, f


def main():
    files = glob.glob(f'{SRC}/*.ply')
    # 그룹: patient_FDI_type → {gt, gen}
    groups = {}
    for f in files:
        base = os.path.basename(f).replace('.ply', '')
        m = re.match(r'(.+)_FDI(\d+)_(gt|gen)', base)
        if not m:
            continue
        pid, fdi, typ = m.group(1), int(m.group(2)), m.group(3)
        key = (pid, fdi)
        if key not in groups:
            groups[key] = {}
        v, fc = parse_ply(f)
        groups[key][typ] = {'v': v.tolist(), 'f': fc.tolist()}
        print(f'  {base}: {len(v)}v {len(fc)}f', flush=True)

    # 환자별로 그룹 (한 화면에 여러 치아 메시 + IOS 아치)
    by_patient = {}
    for (pid, fdi), meshes in sorted(groups.items()):
        if 'gt' in meshes and 'gen' in meshes:
            if pid not in by_patient:
                by_patient[pid] = {'patient': pid, 'teeth': [], 'real_pts': get_real_pts(pid)}
            by_patient[pid]['teeth'].append({'fdi': fdi, 'gt': meshes['gt'], 'gen': meshes['gen']})
    cases = list(by_patient.values())

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('window.MESH_DATA = ' + json.dumps({'cases': cases}) + ';\n')
    print(f'WROTE {OUT}: {len(cases)} cases (gt+gen pairs)', flush=True)


if __name__ == '__main__':
    main()
