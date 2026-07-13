"""최종 모델(gen_stage2_aligned)로 크라운 점구름 생성 → Open3D Poisson 메시.
GT vs 생성 메시 비교용 PLY 출력 (MeshLab/뷰어).
crown env (python3.11, Open3D+PyTorch) 로 실행.
"""
import sys, os, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
import json, numpy as np, torch
import open3d as o3d
from crowngen.external.gen_diffusion import GenModel, get_betas
from crowngen.data.fdi import ZIGZAG_FDI_ORDER
from gen_stage2_pseudo import load_patient, jaw_of

DATA = 'Data/aligned_norm'
SPLIT = 'Data/SourceC_Teeth3DS/train_val_split.json'
CKPT = 'runs2/gen_stage2_aligned_last.pt'
OUT = 'runs2/mesh_demo'


def load_model(dev):
    betas = get_betas('linear', 1e-4, 2e-2, 1000)
    m = GenModel(betas, embed_dim=64, dropout=0.1, extra_feature_channels=9).to(dev)
    ck = torch.load(CKPT, map_location=dev)
    m.model.load_state_dict(ck['model'])
    if ck.get('ema'):
        from crowngen.models.ema import EMA
        ema = EMA(m.model, 0.995); ema.load_state_dict(ck['ema']); ema.apply_to(m.model)
    m.eval(); return m


def poisson_mesh(points, depth=9, smooth_iters=10, decimate_target=0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(15)
    # 노이즈 제거 (생성 점은 강하게)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
    pcd, _ = pcd.remove_radius_outlier(nb_points=5, radius=0.03)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    threshold = np.percentile(densities, 5)
    mesh.remove_vertices_by_mask(densities < threshold)
    # decimation (스파이크 제거: 더 적은 삼각형으로 평균화)
    if decimate_target > 0 and len(mesh.triangles) > decimate_target:
        mesh = mesh.simplify_quadric_decimation(decimate_target)
    # 평활화
    if smooth_iters > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iters)
        mesh.compute_vertex_normals()
    return mesh


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
    model = load_model(dev)
    print(f'model: {CKPT}', flush=True)

    sp = json.load(open(SPLIT))
    pids = [p for p in sp['stage1_val'] if os.path.exists(f'{DATA}/{p}.npz')][:args.n]
    for pid in pids:
        pts, bnd, valid, _ = load_patient(f'{DATA}/{pid}.npz', 1024)
        present = np.where(valid > 0)[0]
        k = min(random.randint(1, 3), len(present))  # 환자마다 1~3개 랜덤
        idx = np.random.choice(present, k, replace=False)
        x0 = torch.from_numpy(pts).unsqueeze(0).to(dev)
        bound = torch.from_numpy(bnd).unsqueeze(0).to(dev)
        lm = np.zeros(28); lm[idx] = 1
        lm_t = torch.from_numpy(lm).float().unsqueeze(0).to(dev); om = 1 - lm_t
        with torch.no_grad():
            gen = model.sample(dict(x0=x0, l_mask=lm_t, o_mask=om, bound=bound))[0].cpu().numpy()
        for s in idx:
            fdi = ZIGZAG_FDI_ORDER[s]
            gt_pts = pts[s].T  # (P,3) GT
            gen_pts = gen[s].T  # (P,3) 생성
            # Poisson 메시
            gt_mesh = poisson_mesh(gt_pts, smooth_iters=10, decimate_target=0)
            gen_mesh = poisson_mesh(gen_pts, smooth_iters=100, decimate_target=2000)
            gt_mesh.compute_vertex_normals(); gen_mesh.compute_vertex_normals()
            gt_path = f'{OUT}/{pid}_FDI{fdi}_gt.ply'
            gen_path = f'{OUT}/{pid}_FDI{fdi}_gen.ply'
            o3d.io.write_triangle_mesh(gt_path, gt_mesh)
            o3d.io.write_triangle_mesh(gen_path, gen_mesh)
            gt_v, gt_f = len(gt_mesh.vertices), len(gt_mesh.triangles)
            gen_v, gen_f = len(gen_mesh.vertices), len(gen_mesh.triangles)
            print(f'  {pid} FDI{fdi}: GT mesh {gt_v}v/{gt_f}f, gen mesh {gen_v}v/{gen_f}f → {gen_path}', flush=True)
    print(f'DONE → {OUT}/', flush=True)


if __name__ == '__main__':
    main()
