"""DPSR(Shape as Points) 메시 재구성 — 논문 방식.
gen_stage2_aligned로 크라운 점구름 생성 → DPSR(학습된 Poisson) → Marching Cubes → 메시.
standard Poisson 대신 학습 기반 → 더 매끄러운 메시.
crown env (python3.11) 로 실행.
"""
import sys, os, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'crowngen', 'external'))  # mesh_recon 패키지용
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
import json, numpy as np, torch
import open3d as o3d
from crowngen.external.gen_diffusion import GenModel, get_betas
from crowngen.data.fdi import ZIGZAG_FDI_ORDER
from gen_stage2_pseudo import load_patient, jaw_of

# DPSR imports
from mesh_recon.src.model import Encode2Points
from mesh_recon.src.dpsr import DPSR
from mesh_recon.src.utils import load_model_manual, load_config, mc_from_psr

GEN_CKPT = 'runs2/gen_stage2_aligned_last.pt'
DPSR_CKPT = 'runs2/dpsr_weights/ours_noise_005.pt'
DPSR_CFG = 'crowngen/external/mesh_recon/configs/learning_based/noise_small/tooth_1024.yaml'
DPSR_DEFAULT = 'crowngen/external/mesh_recon/configs/default.yaml'
DATA = 'Data/aligned_norm'
SPLIT = 'Data/SourceC_Teeth3DS/train_val_split.json'
OUT = 'runs2/mesh_dpsr'


def load_gen_model(dev):
    betas = get_betas('linear', 1e-4, 2e-2, 1000)
    m = GenModel(betas, embed_dim=64, dropout=0.1, extra_feature_channels=9).to(dev)
    ck = torch.load(GEN_CKPT, map_location=dev)
    m.model.load_state_dict(ck['model'])
    if ck.get('ema'):
        from crowngen.models.ema import EMA
        ema = EMA(m.model, 0.995); ema.load_state_dict(ck['ema']); ema.apply_to(m.model)
    m.eval(); return m


def load_dpsr(dev):
    cfg = load_config(DPSR_CFG, DPSR_DEFAULT)
    model = Encode2Points(cfg).to(device=dev)
    ck = torch.load(DPSR_CKPT, map_location=dev, weights_only=False)
    load_model_manual(ck['state_dict'], model)
    model.eval()
    res = cfg['model']['grid_res']
    dpsr = DPSR(res=(res, res, res), sig=cfg['model']['psr_sigma']).to(dev)
    return model, dpsr


def dpsr_mesh(points, model, dpsr, dev):
    """점구름 (N,3) → DPSR → mesh (verts, faces). 점을 [0,1]로 정규화 후 DPSR."""
    p_min = points.min(0); p_max = points.max(0)
    scale = p_max - p_min + 1e-8
    p_norm = (points - p_min) / scale  # [0,1]
    p_t = torch.from_numpy(p_norm.astype(np.float32)).unsqueeze(0).to(dev)  # (1,N,3)
    with torch.no_grad():
        pred_points, pred_normals = model(p_t)
        psr_grid = dpsr(pred_points, pred_normals)  # (1,1,R,R,R)
        verts, faces, _ = mc_from_psr(psr_grid, pytorchify=True)
    verts = verts.cpu().numpy() * scale + p_min  # denormalize
    faces = faces.cpu().numpy()
    return verts, faces


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--mask_n', type=int, default=2)
    ap.add_argument('--seed', type=int, default=11)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device('cuda')
    gen_model = load_gen_model(dev)
    dpsr_model, dpsr = load_dpsr(dev)
    print(f'models loaded: gen={GEN_CKPT}, dpsr={DPSR_CKPT}', flush=True)

    sp = json.load(open(SPLIT))
    pids = [p for p in sp['stage1_val'] if os.path.exists(f'{DATA}/{p}.npz')][:args.n]
    for pid in pids:
        pts, bnd, valid, _ = load_patient(f'{DATA}/{pid}.npz', 1024)
        present = np.where(valid > 0)[0]
        k = min(random.randint(1, 3), len(present))
        idx = np.random.choice(present, k, replace=False)
        x0 = torch.from_numpy(pts).unsqueeze(0).to(dev)
        bound = torch.from_numpy(bnd).unsqueeze(0).to(dev)
        lm = np.zeros(28); lm[idx] = 1
        lm_t = torch.from_numpy(lm).float().unsqueeze(0).to(dev); om = 1 - lm_t
        with torch.no_grad():
            gen = gen_model.sample(dict(x0=x0, l_mask=lm_t, o_mask=om, bound=bound))[0].cpu().numpy()
        for s in idx:
            fdi = ZIGZAG_FDI_ORDER[s]
            for label, cloud in [('gt', pts[s].T), ('gen', gen[s].T)]:
                verts, faces = dpsr_mesh(cloud, dpsr_model, dpsr, dev)
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(verts)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh.compute_vertex_normals()
                path = f'{OUT}/{pid}_FDI{fdi}_{label}.ply'
                o3d.io.write_triangle_mesh(path, mesh)
                print(f'  {pid} FDI{fdi} {label}: {len(verts)}v/{len(faces)}f DPSR → {path}', flush=True)
    print(f'DONE → {OUT}/', flush=True)


if __name__ == '__main__':
    main()
