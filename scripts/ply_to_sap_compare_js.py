"""runs2/mesh_sap_compare/*.ply → runs2/viz/mesh_sap_compare/data.js
3-panel three.js viewer용."""
import os, re, json, glob
import numpy as np, open3d as o3d

SRC = 'runs2/mesh_sap_compare'
OUT = 'runs2/viz/mesh_sap_compare/data.js'
NORM = 'Data/aligned_norm'
MAX_FACES = 15000


def jaw_of(fdi): return 'upper' if fdi // 10 in (1, 2) else 'lower'


def get_real_pts(pid):
    p = f'{NORM}/{pid}.npz'
    if not os.path.exists(p): return []
    d = np.load(p); pts = []
    for fdi_str in [k.split('_')[2].split('.')[0] for k in d.keys() if k.endswith('_pc')]:
        try: fdi = int(fdi_str)
        except: continue
        k = f'{jaw_of(fdi)}_{fdi}_pc'
        if k in d:
            pc = d[k]
            if len(pc) > 100:
                idx = np.linspace(0, len(pc)-1, 100).astype(int); pc = pc[idx]
            pts.extend(pc.tolist())
    return pts


def parse_ply(path):
    m = o3d.io.read_triangle_mesh(path)
    if len(m.triangles) > MAX_FACES:
        m = m.simplify_quadric_decimation(MAX_FACES); m.compute_vertex_normals()
    v = np.asarray(m.vertices, dtype=np.float32); f = np.asarray(m.triangles, dtype=np.uint32)
    return v, f


def main():
    files = glob.glob(f'{SRC}/*.ply')
    by_patient = {}
    pat = re.compile(r'(.+)_FDI(\d+)_(gt|gen)__(poisson|sap_pre|sap_fine)')
    for fpath in files:
        base = os.path.basename(fpath).replace('.ply','')
        m = pat.match(base)
        if not m: continue
        pid, fdi, lbl, method = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        key = (pid, fdi)
        if key not in by_patient:
            by_patient[key] = {'pid': pid, 'fdi': fdi, 'gt_label': lbl, 'methods': {}}
        v, f = parse_ply(fpath)
        by_patient[key]['methods'][method] = {'v': v.tolist(), 'f': f.tolist()}

    # 환자별 grouping (같은 환자/같은 FDI는 1 케이스, 2 method × 2 label)
    by_pid = {}
    for k, info in by_patient.items():
        pid = info['pid']
        if pid not in by_pid:
            by_pid[pid] = {'patient': pid, 'teeth': [], 'real_pts': get_real_pts(pid)}
        # best methods 채워진 키만 push
        if len(info['methods']) >= 1:
            by_pid[pid]['teeth'].append({'fdi': info['fdi'], 'label': info['gt_label'], 'methods': info['methods']})
    cases = list(by_pid.values())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fp:
        fp.write('window.MESH_DATA = ' + json.dumps({'cases': cases}) + ';\n')
    print(f'WROTE {OUT}: {len(cases)} patients', flush=True)

if __name__ == '__main__':
    main()