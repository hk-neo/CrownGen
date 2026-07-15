"""DPSR fine-tune 3-way 비교:
1) 표준 Poisson + Taubin (현재 viewer, runs2/mesh_demo/)
2) SAP pre-trained (ours_noise_005.pt)
3) SAP fine-tuned (sap_finetuned_best.pt)
메트릭: Chamfer-L2, Edge L2, Watertight.
환자 6명 × 1~3 teeth × {gt|gen}, seed=11.
"""
import os, sys, json, glob, random, argparse, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/crowngen/external'); sys.path.insert(0, ROOT + '/scripts')
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
import numpy as np, torch
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crowngen.external.gen_diffusion import GenModel, get_betas
from crowngen.data.fdi import ZIGZAG_FDI_ORDER
from gen_stage2_pseudo import load_patient, jaw_of
from mesh_recon.src.model import Encode2Points
from mesh_recon.src.utils import load_model_manual, load_config
from mesh_recon.src.dpsr import DPSR

CKPT_GEN = 'runs2/gen_stage2_aligned_last.pt'
SAP_PRE = 'runs2/dpsr_weights/ours_noise_005.pt'
SAP_FINE = 'runs2/sap_finetuned_best.pt'
DPSR_CFG = 'crowngen/external/mesh_recon/configs/learning_based/noise_small/tooth_1024.yaml'
DPSR_DEFAULT = 'crowngen/external/mesh_recon/configs/default.yaml'
DATA = 'Data/aligned_norm'
SPLIT = 'Data/SourceC_Teeth3DS/train_val_split.json'
POISSON_DIR = 'runs2/mesh_demo'  # 기존 PLY 재사용 (정답)
OUT = 'runs2/mesh_sap_compare'
CHART = f'{OUT}/charts/comparison.png'
JSON_OUT = 'runs2/sap_eval.json'


def load_gen(dev):
    betas = get_betas('linear', 1e-4, 2e-2, 1000)
    m = GenModel(betas, embed_dim=64, dropout=0.1, extra_feature_channels=9).to(dev)
    ck = torch.load(CKPT_GEN, map_location=dev); m.model.load_state_dict(ck['model'])
    if ck.get('ema'):
        from crowngen.models.ema import EMA
        ema = EMA(m.model, 0.995); ema.load_state_dict(ck['ema']); ema.apply_to(m.model)
    m.eval(); return m


def load_sap(ckpt, dev):
    cfg = load_config(DPSR_CFG, DPSR_DEFAULT)
    model = Encode2Points(cfg).to(dev)
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        # fine-tuned ckpt is raw state_dict; pre-trained has wrapper
        if isinstance(ck, dict) and 'state_dict' in ck:
            load_model_manual(ck['state_dict'], model)
        else:
            load_model_manual(ck, model)
        print(f'loaded {ckpt}', flush=True)
    else:
        print(f'WARN {ckpt} missing -- using skeleton weights', flush=True)
    model.eval(); return model, cfg


def sap_mesh(pc, sap_model, dpsr, cfg, dev):
    p_min = pc.min(0); p_max = pc.max(0); scale = (p_max - p_min) + 1e-8
    p_norm = (pc - p_min) / scale
    p_t = torch.from_numpy(p_norm.astype(np.float32)).unsqueeze(0).to(dev)
    with torch.no_grad():
        pred_pc, pred_n = sap_model(p_t)
    pred_pc = np.clip(pred_pc[0].cpu().numpy(), 0, 0.99) * scale + p_min
    if cfg['model']['normal_normalize']:
        pred_n = pred_n / (pred_n.norm(-1, keepdim=True)+1e-8)
    pred_n = pred_n[0].cpu().numpy()
    # DPSR -> mesh
    pc_t = torch.from_numpy(pred_pc.astype(np.float32)).unsqueeze(0).to(dev)
    n_t = torch.from_numpy(pred_n.astype(np.float32)).unsqueeze(0).to(dev)
    with torch.no_grad():
        psr = dpsr(pc_t, n_t)
        if cfg['model']['psr_tanh']: psr = torch.tanh(psr)
        from mesh_recon.src.utils import mc_from_psr
        v, f, _ = mc_from_psr(psr, pytorchify=True)
    v = v.cpu().numpy() * scale + p_min
    f = f.cpu().numpy().astype(np.int64)
    return v, f


def chamfer_l2(mesh, gt_pc, n_samples=30000):
    v = np.asarray(mesh.vertices); f = np.asarray(mesh.triangles)
    if len(f) == 0: return float('inf')
    # barycentric uniform sample on faces
    r1 = np.random.rand(n_samples); r2 = np.random.rand(n_samples)
    mask = r1 + r2 > 1; r1[mask] = 1 - r1[mask]; r2[mask] = 1 - r2[mask]
    r3 = 1 - r1 - r2
    face_idx = np.random.randint(0, len(f), n_samples)
    pts = r1[:,None]*v[f[face_idx,0]] + r2[:,None]*v[f[face_idx,1]] + r3[:,None]*v[f[face_idx,2]]
    d_mg = np.min(np.linalg.norm(pts[:,None] - gt_pc[None], axis=2), axis=1).mean()
    d_gm = np.min(np.linalg.norm(gt_pc[:,None] - pts[None], axis=2), axis=1).mean()
    return float(d_mg + d_gm)


def normal_consistency(mesh, gt_pc):
    v = np.asarray(mesh.vertices); n_v = np.asarray(mesh.vertex_normals)
    if len(n_v) == 0: return 0.0
    # nearest neighbor: gt_pc <-> v
    d = np.linalg.norm(gt_pc[:,None] - v[None], axis=2)
    idx = d.argmin(0)  # for each vertex, nearest gt
    gt_n = np.zeros_like(n_v)
    from sklearn.neighbors import NearestNeighbors
    # approximate gt_pc normals via nearest vertex normal in mesh
    gt_n[idx] = n_v
    cos = (n_v * gt_n).sum(-1)
    return float(1 - np.clip(np.abs(cos).mean(), 0, 1))


def edge_l2(mesh):
    v = np.asarray(mesh.vertices); f = np.asarray(mesh.triangles)
    if len(f) == 0: return 0.0
    e = []
    for tri in f:
        for a, b in [(0,1),(1,2),(2,0)]:
            e.append(np.linalg.norm(v[tri[a]] - v[tri[b]]))
    return float(np.mean(e))


def watertight(mesh):
    v = np.asarray(mesh.vertices); f = np.asarray(mesh.triangles)
    if len(f) == 0: return 0
    # Build Open3D mesh from numpy arrays
    m_o3d = o3d.geometry.TriangleMesh()
    m_o3d.vertices = o3d.utility.Vector3dVector(v)
    m_o3d.triangles = o3d.utility.Vector3iVector(f)
    num_comp = len(np.unique(m_o3d.cluster_connected_triangles()[0]))
    return 1 if num_comp == 1 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--no_fine', action='store_true', help='only Poisson vs SAP-pre (skip fine-tuned)')
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(OUT, exist_ok=True); os.makedirs(f'{OUT}/charts', exist_ok=True)
    dev = torch.device('cuda')
    gen = load_gen(dev)
    sap_pre, cfg = load_sap(SAP_PRE, dev)
    dpsr = DPSR(res=(64,64,64), sig=2).to(dev)
    sap_fine = None
    if not args.no_fine and os.path.exists(SAP_FINE):
        sap_fine, _ = load_sap(SAP_FINE, dev)
    print(f'eval: n={args.n} no_fine={args.no_fine}', flush=True)

    sp = json.load(open(SPLIT))
    pids = [p for p in sp['stage1_val'] if os.path.exists(f'{DATA}/{p}.npz')][:args.n]
    results = {'poisson': [], 'sap_pre': [], 'sap_fine': []}
    for pid in pids:
        pts, bnd, valid, _ = load_patient(f'{DATA}/{pid}.npz', 1024)
        present = np.where(valid > 0)[0]
        k = min(random.randint(1,3), len(present))
        idx = np.random.choice(present, k, replace=False)
        x0 = torch.from_numpy(pts).unsqueeze(0).to(dev)
        bound = torch.from_numpy(bnd).unsqueeze(0).to(dev)
        lm = np.zeros(28); lm[idx] = 1
        lm_t = torch.from_numpy(lm).float().unsqueeze(0).to(dev); om = 1 - lm_t
        with torch.no_grad():
            gen_pc = gen.sample(dict(x0=x0, l_mask=lm_t, o_mask=om, bound=bound))[0].cpu().numpy()
        for s in idx:
            fdi = ZIGZAG_FDI_ORDER[s]
            for label, cloud in [('gt', pts[s].T), ('gen', gen_pc[s].T)]:
                # 1. Poisson (re-use existing)
                poisson_ply = f'{POISSON_DIR}/{pid}_FDI{fdi}_{label}.ply'
                if os.path.exists(poisson_ply):
                    p_mesh = o3d.io.read_triangle_mesh(poisson_ply)
                else:
                    # fallback: compute now
                    from gen_mesh import poisson_mesh
                    p_mesh = poisson_mesh(cloud, smooth_iters=10, decimate_target=0)
                results['poisson'].append({
                    'pid': pid, 'fdi': fdi, 'label': label,
                    'chamfer': chamfer_l2(p_mesh, cloud), 'edge': edge_l2(p_mesh),
                    'watertight': watertight(p_mesh),
                })
                # save copy for viewer
                o3d.io.write_triangle_mesh(f'{OUT}/{pid}_FDI{fdi}_{label}__poisson.ply', p_mesh)
                # 2. SAP-pre (skip if PLY already exists)
                sap_pre_ply = f'{OUT}/{pid}_FDI{fdi}_{label}__sap_pre.ply'
                if os.path.exists(sap_pre_ply):
                    sm = o3d.io.read_triangle_mesh(sap_pre_ply)
                else:
                    v, f = sap_mesh(cloud, sap_pre, dpsr, cfg, dev)
                    sm = o3d.geometry.TriangleMesh()
                    sm.vertices = o3d.utility.Vector3dVector(v); sm.triangles = o3d.utility.Vector3iVector(f); sm.compute_vertex_normals()
                    o3d.io.write_triangle_mesh(sap_pre_ply, sm)
                results['sap_pre'].append({
                    'pid': pid, 'fdi': fdi, 'label': label,
                    'chamfer': chamfer_l2(sm, cloud), 'edge': edge_l2(sm),
                    'watertight': watertight(sm),
                })
                # 3. SAP-fine (skip if PLY already exists)
                if sap_fine is not None:
                    sap_fine_ply = f'{OUT}/{pid}_FDI{fdi}_{label}__sap_fine.ply'
                    if os.path.exists(sap_fine_ply):
                        sm2 = o3d.io.read_triangle_mesh(sap_fine_ply)
                    else:
                        v2, f2 = sap_mesh(cloud, sap_fine, dpsr, cfg, dev)
                        sm2 = o3d.geometry.TriangleMesh()
                        sm2.vertices = o3d.utility.Vector3dVector(v2); sm2.triangles = o3d.utility.Vector3iVector(f2); sm2.compute_vertex_normals()
                        o3d.io.write_triangle_mesh(sap_fine_ply, sm2)
                    results['sap_fine'].append({
                        'pid': pid, 'fdi': fdi, 'label': label,
                        'chamfer': chamfer_l2(sm2, cloud), 'edge': edge_l2(sm2),
                        'watertight': watertight(sm2),
                    })
                print(f'  {pid} FDI{fdi} {label} done', flush=True)

    # filter inf chamfer and compute averages
    summary = {}
    for k in results: results[k] = [r for r in results[k] if r['chamfer'] != float('inf')]
    for k in results:
        if results[k]:
            summary[k] = {
                'n': len(results[k]),
                'chamfer_mean': float(np.mean([r['chamfer'] for r in results[k]])),
                'edge_mean': float(np.mean([r['edge'] for r in results[k]])),
                'watertight_rate': float(np.mean([r['watertight'] for r in results[k]])),
            }
    json.dump({'summary': summary, 'raw': results}, open(JSON_OUT, 'w'), indent=2)
    print('SUMMARY:', summary, flush=True)

    # chart
    methods = [m for m in ['poisson', 'sap_pre', 'sap_fine'] if m in summary]
    metrics = ['chamfer_mean', 'edge_mean', 'watertight_rate']
    titles = ['Chamfer-L2 (mesh->GT)', 'Mean edge length (smoothness)', 'Watertight rate']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, (m, t) in enumerate(zip(metrics, titles)):
        vals = [summary[mt][m] for mt in methods]
        axes[i].bar(methods, vals, color=['#3fb950', '#f0883e', '#58a6ff'])
        axes[i].set_title(t); axes[i].tick_params(axis='x', rotation=20)
    fig.suptitle('3-way mesh comparison (Poisson / SAP-pre / SAP-fine)')
    fig.tight_layout(); fig.savefig(CHART, dpi=110); plt.close()
    print(f'CHART -> {CHART}', flush=True)


if __name__ == '__main__':
    main()
