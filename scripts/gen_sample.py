"""Generation 결과 샘플링 + CD 평가 + PNG.

학습된 gen ckpt 로드 → val 환자에서 타겟 치아 마스킹 → diffusion 샘플링으로 크라운 생성 →
GT 와 Chamfer Distance(CD-L1) 측정 + 시각화 PNG.
bound 는 기본 GT(학습과 동일), --bound_model 시 boundary 모델 예측 사용.
"""
import argparse, os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

import importlib.util
st = importlib.util.spec_from_file_location('tbo', 'scripts/gen_train.py'); tbo = importlib.util.module_from_spec(st); st.loader.exec_module(tbo)
from crowngen.external.gen_diffusion import GenModel, get_betas
from crowngen.external import BoundEncoder
from crowngen.losses.chamfer import chamfer_distance_l1


def load_boundary(path):
    m = BoundEncoder(5, 0.3, 6, mask_mode='official', voxel_attention=False)
    m.load_state_dict(torch.load(path, map_location='cpu')); m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data_dir', default='Data/processed_norm2')
    ap.add_argument('--split_file', default='Data/SourceC_Teeth3DS/train_val_split.json')
    ap.add_argument('--bound_model', default=None, help='boundary ckpt (안 주면 GT bound)')
    ap.add_argument('--n_patients', type=int, default=4)
    ap.add_argument('--out', default='runs2/viz/gen')
    ap.add_argument('--tag', default='gen')
    args = ap.parse_args()

    os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    betas = get_betas('linear', 1e-4, 2e-2, 1000)
    model = GenModel(betas, embed_dim=64, dropout=0.1, extra_feature_channels=9).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    model.model.load_state_dict(sd); model.eval()
    # EMA 가중치로 샘플링 (eps 보정 더 좋음). ema 있으면 적용.
    if isinstance(ck, dict) and 'ema' in ck and ck['ema']:
        from crowngen.models.ema import EMA
        ema = EMA(model.model, decay=0.995); ema.load_state_dict(ck['ema']); ema.apply_to(model.model)
        print('using EMA weights for sampling')
    print('loaded gen ckpt', args.ckpt)

    bnd_model = load_boundary(args.bound_model).to(device) if args.bound_model else None
    if bnd_model:
        print('using boundary model:', args.bound_model)

    splits = json.load(open(args.split_file))
    va = tbo.GenDataset(args.data_dir, splits['stage1_val'], 1024, require_full=True, augment=False)
    os.makedirs(args.out, exist_ok=True)
    random.seed(0)

    cds = []
    for pi in range(min(args.n_patients, len(va))):
        s = va[pi]
        x0 = s['points'].unsqueeze(0).to(device)              # (1,28,3,1024) GT
        gt_bound = s['bound'].unsqueeze(0).to(device)          # (1,28,5)
        # 타겟 1~6개 마스킹
        lm = torch.zeros(1, 28); idx = torch.randperm(28)[:random.randint(1, 6)]; lm[0, idx] = 1
        lm = lm.to(device); om = 1 - lm
        # bound: boundary 모델 예측 or GT
        if bnd_model is not None:
            exist = om
            with torch.no_grad():
                bound = bnd_model(x0, exist.view(1, 28, 1, 1))
        else:
            bound = gt_bound
        kw = dict(x0=x0, l_mask=lm, o_mask=om, bound=bound)
        with torch.no_grad():
            gen = model.sample(kw)                              # (1,28,3,1024)
        gen = gen[0].cpu().numpy(); gt = x0[0].cpu().numpy(); lm0 = lm[0].cpu().numpy()
        # 타겟 치아별 CD (점은 channel-first (3,P) → (P,3) 로 transpose)
        tidx = np.where(lm0 > 0)[0]
        pcds = []
        for ti in tidx:
            p = torch.from_numpy(gen[ti].T).unsqueeze(0)   # (1,P,3)
            g = torch.from_numpy(gt[ti].T).unsqueeze(0)
            cd = chamfer_distance_l1(p, g).item()
            pcds.append(cd)
        mean_cd = float(np.mean(pcds)) if pcds else float('nan')
        cds.append(mean_cd)
        print(f'patient {pi}: targets={len(tidx)} CD-L1={mean_cd:.4f} (×10³={mean_cd*1e3:.1f})', flush=True)
        # PNG: GT(회색) vs 생성(빨강) for 타겟 치아
        fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
        for ti in range(28):
            if lm0[ti] == 0:
                ax.scatter(gt[ti, 0], gt[ti, 1], gt[ti, 2], s=1, c='lightgray', alpha=0.3)
        for ti in tidx:
            ax.scatter(gt[ti, 0], gt[ti, 1], gt[ti, 2], s=4, c='green', alpha=0.6)
            ax.scatter(gen[ti, 0], gen[ti, 1], gen[ti, 2], s=4, c='red', alpha=0.8)
        ax.set_title(f'{args.tag} patient {pi}: CD×10³={mean_cd*1e3:.1f} (논문~34)\nGT=green/gray, gen=red')
        plt.tight_layout(); plt.savefig(f'{args.out}/{args.tag}_{pi:02d}.png', dpi=110); plt.close()

    print(f'\n[{args.tag}] MEAN CD-L1 = {np.mean(cds):.4f} (×10³ = {np.mean(cds)*1e3:.1f}) | 논문 1개복원 CD-L1≈34.5×10⁻³', flush=True)
    print('논문 수치(34.5)와 비교: 작을수록 좋음. 합리적 범위면 generation 성공.')


if __name__ == '__main__':
    main()
