"""SAP PSR GT 캐시 빌더 — Teeth3DS 환자별 GT 점구름을 DPSR PSR volume으로 변환.

각 캐시 항목: runs2/sap_cache/{pid}_FDI{fdi}.npz
  - psr_vol: (64,64,64) float16  PSR occupancy field
  - pc:      (1024,3) float32   원본 GT 점구름
  - normals: (1024,3) float16   (unused, zeros)

 Workflow: GT PC → Open3D Poisson surface recon (depth=8) → DPSR(GT mesh normals) → PSR vol.
"""
import gc
import argparse
import os, sys, glob

# mesh_recon package (dpsr, model, utils)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'crowngen', 'external'))

import numpy as np
import torch
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

from crowngen.data.fdi import ZIGZAG_FDI_ORDER
from mesh_recon.src.dpsr import DPSR

# ── paths ──────────────────────────────────────────────────────────────────
NORM     = 'Data/aligned_norm'
CACHE    = 'runs2/sap_cache'
GRID_RES = 64

# DPSR (not strictly needed here – used implicitly via sig param)
DPSR_SIG = 2   # Gaussian smoothing width for DPSR

# ── helpers ─────────────────────────────────────────────────────────────────

def jaw_of(fdi: int) -> str:
    """FDI tooth numbering → jaw side."""
    return 'upper' if fdi // 10 in (1, 2) else 'lower'


def load_tooth_pc(pid: str, fdi: int) -> np.ndarray | None:
    """Load one tooth's GT point cloud from aligned_norm npz.

    Returns:
        (1024, 3) float32 array, or None if missing / degenerate.
    """
    path = f'{NORM}/{pid}.npz'
    if not os.path.exists(path):
        return None
    d = np.load(path)
    key = f'{jaw_of(fdi)}_{fdi}_pc'
    pc = d.get(key)
    if pc is None or len(pc) < 100:
        return None
    return pc.astype(np.float32)


def cache_path(pid: str, fdi: int) -> str:
    os.makedirs(CACHE, exist_ok=True)
    return f'{CACHE}/{pid}_FDI{fdi}.npz'


def need_build(path: str) -> bool:
    return not os.path.exists(path)


# ── PSR GT generation ───────────────────────────────────────────────────────

def make_psr_gt(pc: np.ndarray, dev: torch.device) -> np.ndarray:
    """(1024, 3) GT PC → (64,64,64) float16 PSR volume.

    Steps:
        1. Estimate + orient point normals via Open3D.
        2. Poisson surface reconstruction (depth=8, trim bottom 5% density).
        3. For each GT point: nearest-vertex on mesh → assign mesh vertex normal.
        4. DPSR with GT-normalised points + mesh-assigned normals → PSR vol.
    """
    # ── 1. point normals ────────────────────────────────────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.astype(np.float64))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(20)

    # ── 2. Poisson surface ──────────────────────────────────────────────────
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8
    )
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.percentile(densities, 5))

    # ── 3. Assign mesh vertex normals to GT points ──────────────────────────
    p_min = pc.min(0)
    p_max = pc.max(0)
    scale = (p_max - p_min) + 1e-8
    pc_norm = (pc - p_min) / scale  # [0, 1]
    # clip to (0, 1) exclusive upper bound — point_rasterize expects (0, 1)
    pc_norm = np.clip(pc_norm, 0.0, 1.0 - 1e-6)

    mesh_v = np.asarray(mesh.vertices, dtype=np.float64)
    mesh_n = np.asarray(mesh.vertex_normals, dtype=np.float64)
    mesh_v_norm = (mesh_v - p_min) / scale
    mesh_v_norm = np.clip(mesh_v_norm, 0.0, 1.0)

    nn = NearestNeighbors(n_neighbors=1).fit(mesh_v_norm)
    _, idx = nn.kneighbors(pc_norm)
    assigned_n = mesh_n[idx.squeeze(-1)]

    del pcd
    del mesh, densities

    # ── 4. DPSR ─────────────────────────────────────────────────────────────
    # point_rasterize expects (0, 1); pass pc_norm
    p_t = torch.from_numpy(pc_norm.astype(np.float32)).unsqueeze(0).to(dev)
    n_t = torch.from_numpy(assigned_n.astype(np.float32)).unsqueeze(0).to(dev)

    dpsr = DPSR(res=(GRID_RES, GRID_RES, GRID_RES), sig=DPSR_SIG).to(dev)
    with torch.no_grad():
        psr_vol = dpsr(p_t, n_t)  # (1, 64, 64, 64) float

    return psr_vol.squeeze(0).cpu().numpy().astype(np.float16)
    # explicit cleanup of intermediate Open3D objects to avoid memory leak


# ── per-patient builder ─────────────────────────────────────────────────────

def build_one(pid: str, idx_map: dict, dev: torch.device) -> int:
    """Build all tooth cache entries for one patient.

    Args:
        pid:     patient ID string (e.g. '00OMSZGW')
        idx_map: fdi → ZIGZAG index (unused here, kept for API compat)
        dev:     cuda device

    Returns:
        Number of cache entries saved.
    """
    saved = 0
    for fdi in ZIGZAG_FDI_ORDER:
        path = cache_path(pid, fdi)
        if not need_build(path):
            continue
        pc = load_tooth_pc(pid, fdi)
        if pc is None:
            continue
        try:
            vol = make_psr_gt(pc, dev)
        except Exception as e:
            print(f'  skip {pid} FDI{fdi}: {e}', flush=True)
            continue
        np.savez_compressed(
            path,
            psr_vol=vol,
            pc=pc,
            normals=np.zeros_like(pc)
        )
        saved += 1
        if saved % 50 == 0:
            print(f'  {pid}: {saved} saved', flush=True)
    print(f'{pid} done: {saved} new', flush=True)
    gc.collect()
    if str(dev).startswith('cuda'):
        import torch
        torch.cuda.empty_cache()
    return saved


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SAP PSR GT cache builder')
    parser.add_argument('--pid', type=str, default=None,
                        help='Process only this patient ID and exit')
    args = parser.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={dev} cache={CACHE} res={GRID_RES}^3 depth=8', flush=True)

    if args.pid is not None:
        # Single-patient mode: fresh subprocess
        idx_map = {fdi: i for i, fdi in enumerate(ZIGZAG_FDI_ORDER)}
        saved = build_one(args.pid, idx_map, dev)
        print(f'subprocess done: {args.pid} saved={saved}', flush=True)
        return

    # Batch mode: iterate over all patients (existing behavior)
    files = sorted(glob.glob(f'{NORM}/*.npz'))
    print(f'patients: {len(files)}', flush=True)

    for f in files:
        pid = os.path.basename(f).replace('.npz', '')
        idx_map = {fdi: i for i, fdi in enumerate(ZIGZAG_FDI_ORDER)}
        build_one(pid, idx_map, dev)


if __name__ == '__main__':
    main()
