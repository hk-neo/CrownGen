"""generation 모델을 부분무치아(stage2_val)에서 평가 — partial 복원 CD.
full-val이 아닌 진짜 partial 컨텍스트에서 present 치아 1~6개 마스킹 → 복원 → GT와 CD.
(gen_sample는 full 전용이라 partial 변형.) bound는 GT 사용(생성 모델 자체 품질 분리)."""
import sys, os, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
import json, numpy as np, torch
from crowngen.external.gen_diffusion import GenModel, get_betas
from crowngen.losses.chamfer import chamfer_distance_l1
from crowngen.data.fdi import ZIGZAG_FDI_ORDER
from gen_stage2_pseudo import load_patient, jaw_of


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data_dir', default='Data/processed_norm2')
    ap.add_argument('--split_file', default='Data/SourceC_Teeth3DS/train_val_split.json')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device('cuda')

    betas = get_betas('linear', 1e-4, 2e-2, 1000)
    model = GenModel(betas, embed_dim=64, dropout=0.1, extra_feature_channels=9).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.model.load_state_dict(ck['model'])
    if ck.get('ema'):
        from crowngen.models.ema import EMA
        ema = EMA(model.model, 0.995); ema.load_state_dict(ck['ema']); ema.apply_to(model.model)
    model.eval()
    print(f'ckpt {args.ckpt} (ep {ck.get("ep","?")})', flush=True)

    sp = json.load(open(args.split_file))
    pids = [p for p in sp.get('stage2_val', []) if os.path.exists(f'{args.data_dir}/{p}.npz')][:args.n]
    cds = []
    for pi, pid in enumerate(pids):
        pts, bnd, valid, _ = load_patient(f'{args.data_dir}/{pid}.npz', 1024)
        present = np.where(valid > 0)[0]
        if len(present) < 2:
            continue
        k = min(random.randint(1, 6), len(present))
        idx = np.random.choice(present, k, replace=False)
        x0 = torch.from_numpy(pts).unsqueeze(0).to(dev)
        gt = bnd.copy()
        lm = np.zeros(28); lm[idx] = 1
        lm = torch.from_numpy(lm).float().unsqueeze(0).to(dev); om = 1 - lm
        bound = torch.from_numpy(gt).unsqueeze(0).to(dev)   # GT bound (생성 품질 분리)
        with torch.no_grad():
            gen = model.sample(dict(x0=x0, l_mask=lm, o_mask=om, bound=bound))[0].cpu().numpy()
        gt_pts = pts  # (28,3,P)
        pc = []
        for ti in idx:
            p = torch.from_numpy(gen[ti].T).unsqueeze(0)
            g = torch.from_numpy(gt_pts[ti].T).unsqueeze(0)
            pc.append(chamfer_distance_l1(p, g).item())
        m = float(np.mean(pc)) if pc else float('nan')
        cds.append(m)
        print(f'  {pid}: mask {k} present → CD-L1 {m:.4f} (×10³ {m*1e3:.1f})', flush=True)
    print(f'>> {os.path.basename(args.ckpt)} partial(stage2_val) MEAN CD-L1 = {np.mean(cds)*1e3:.1f}×10⁻³ (n={len(cds)})', flush=True)


if __name__ == '__main__':
    main()
