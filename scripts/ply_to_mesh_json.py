"""runs2/mesh_demo/*.ply → runs2/viz/mesh_demo/data.js
Open3D로 binary PLY 읽어서 vertex+face JSON으로. Three.js 뷰어용.
crown env (python3.11) 로 실행."""
import os, re, json, glob, sys
import numpy as np
import open3d as o3d

SRC = 'runs2/mesh_demo'
OUT = 'runs2/viz/mesh_demo/data.js'


def parse_ply(path):
    mesh = o3d.io.read_triangle_mesh(path)
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

    cases = []
    for (pid, fdi), meshes in sorted(groups.items()):
        if 'gt' in meshes and 'gen' in meshes:
            cases.append({'patient': pid, 'fdi': fdi, 'gt': meshes['gt'], 'gen': meshes['gen']})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('window.MESH_DATA = ' + json.dumps({'cases': cases}) + ';\n')
    print(f'WROTE {OUT}: {len(cases)} cases (gt+gen pairs)', flush=True)


if __name__ == '__main__':
    main()
