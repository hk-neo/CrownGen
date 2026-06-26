"""비교용 pseudo-crown 생성: 지정 환자 리스트에 대해 주어진 boundary 체크포인트로
빈 슬롯을 채운 crown 점구름을 저장. (old vs new boundary 시각 A/B 용)

gen_stage2_pseudo.py 의 핵심 로직 재사용, but:
  - 환자 리스트를 인자로 (비교용 소수 환자만)
  - out: runs2/pseudo_compare/{suffix}/{pid}.npz  (28슬롯 점구름: present=GT, missing=생성)
"""
import argparse, os, sys
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
import json
import numpy as np
import torch

from crowngen.external.gen_diffusion import GenModel, get_betas
from crowngen.external import BoundEncoder
from crowngen.data.fdi import ZIGZAG_FDI_ORDER
from gen_stage2_pseudo import load_patient, jaw_of
from diag_arch_interpolate import interpolate_positions   # 아치 보간 위치


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_ckpt', default='runs2/gen3k_ep3000.pt')
    ap.add_argument('--bound_ckpt', required=True)
    ap.add_argument('--bound_max_missing', type=int, default=12)
    ap.add_argument('--data_dir', default='Data/processed_norm2')
    ap.add_argument('--pids', nargs='+', required=True)
    ap.add_argument('--suffix', required=True, help='출력 디렉토리명 (old/new 등)')
    ap.add_argument('--n_points', type=int, default=1024)
    ap.add_argument('--arch_pos', action='store_true',
                    help='빈 슬롯 위치(cx,cy,cz)를 아치 보간으로 덮어쓰고 h,r만 boundary 예측 사용')
    args = ap.parse_args()

    out_dir = f'runs2/pseudo_compare/{args.suffix}'
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    betas = get_betas('linear', 1e-4, 2e-2, 1000)
    gen_model = GenModel(betas, embed_dim=64, dropout=0.1, extra_feature_channels=9).to(device)
    ck = torch.load(args.gen_ckpt, map_location=device)
    gen_model.model.load_state_dict(ck['model'])
    if ck.get('ema'):
        from crowngen.models.ema import EMA
        ema = EMA(gen_model.model, 0.995); ema.load_state_dict(ck['ema']); ema.apply_to(gen_model.model)
    gen_model.eval()
    print(f'gen model loaded (ep {ck.get("ep","?")})', flush=True)

    bnd_model = BoundEncoder(5, 0.3, args.bound_max_missing, mask_mode='official').to(device)
    bnd_model.load_state_dict(torch.load(args.bound_ckpt, map_location=device))
    bnd_model.eval()
    print(f'boundary loaded: {os.path.basename(args.bound_ckpt)} (mm={args.bound_max_missing})', flush=True)

    for pid in args.pids:
        src = f'{args.data_dir}/{pid}.npz'
        if not os.path.exists(src):
            print(f'  skip {pid} (no npz)'); continue
        pts, bnd, valid, _ = load_patient(src, args.n_points)
        x0 = torch.from_numpy(pts).unsqueeze(0).to(device)
        valid_t = torch.from_numpy(valid).unsqueeze(0).to(device)
        l_mask = 1.0 - valid_t; o_mask = valid_t
        if l_mask.sum() == 0:
            print(f'  skip {pid} (no missing)'); continue

        with torch.no_grad():
            pred_bound = bnd_model(x0, o_mask.view(1, 28, 1, 1))   # (1,28,5) boundary 예측
        bound = torch.from_numpy(bnd).unsqueeze(0).to(device).clone()
        if args.arch_pos:
            # robust 하이브리드: 내부 결손(양옆 present) → 아치 보간 위치;
            # 끝자리 결손(한쪽 present 없음) → boundary 예측 위치 fallback. h,r 은 항상 boundary.
            arch = interpolate_positions(valid, bnd, interior_only=True)  # 끝자리는 NaN
            pb = pred_bound[0].cpu().numpy()
            eff = pb.copy()
            n_int = n_term = 0
            for s in range(28):
                if valid[s] == 0:
                    if not np.isnan(arch[s, 0]):        # 내부 결손: 보간 위치 + boundary h,r
                        eff[s, :3] = arch[s, :3]; eff[s, 3:5] = pb[s, 3:5]; n_int += 1
                    else:                               # 끝자리 결손: boundary 예측 그대로
                        n_term += 1
                    bound[0, s] = torch.from_numpy(eff[s]).to(device)
            eff_bound = eff
        else:
            for s in range(28):
                if valid[s] == 0:
                    bound[0, s] = pred_bound[0, s]
            eff_bound = pred_bound[0].cpu().numpy()
        with torch.no_grad():
            gen = gen_model.sample(dict(x0=x0, l_mask=l_mask, o_mask=o_mask, bound=bound))
        gen = gen[0].cpu().numpy()   # (28,3,P)

        # 저장: 28슬롯 점구름 + valid + pred_bound. present=원본 GT 점, missing=생성 점.
        slot_pcs = {}
        for s, fdi in enumerate(ZIGZAG_FDI_ORDER):
            k = f'{jaw_of(fdi)}_{fdi}'
            slot_pcs[f'{k}_pc'] = gen[s].T.astype(np.float32)   # (P,3)
        np.savez_compressed(f'{out_dir}/{pid}.npz',
                            valid=valid.astype(np.float32),
                            pred_bound=eff_bound.astype(np.float32),
                            **slot_pcs)
        n_miss = int(l_mask.sum().item())
        print(f'  [{pid}] missing {n_miss} 채움 → {out_dir}/{pid}.npz', flush=True)

    print(f'DONE → {out_dir}', flush=True)


if __name__ == '__main__':
    main()
