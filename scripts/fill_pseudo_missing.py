"""processed_stage2 누락 partial을 강제 재생성 (skip 없음, pid별 에러 캡처).

gen_stage2_pseudo 의 skip-existing/race 로 누락된 pid들을 확정 채운다.
ARCH 하이브리드 위치 (내부 보간 + 끝자리 boundary), h,r=boundary.
"""
import argparse, os, sys, json
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
import numpy as np
import torch
from crowngen.external.gen_diffusion import GenModel, get_betas
from crowngen.external import BoundEncoder
from crowngen.data.fdi import ZIGZAG_FDI_ORDER
from gen_stage2_pseudo import load_patient, jaw_of
from diag_arch_interpolate import interpolate_positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_ckpt', default='runs2/gen3k_ep3000.pt')
    ap.add_argument('--bound_ckpt', default='runs2/boundary_g1_best.pt')
    ap.add_argument('--bound_max_missing', type=int, default=12)
    ap.add_argument('--data_dir', default='Data/processed_norm2')
    ap.add_argument('--pids_file', default='/tmp/miss25.txt')
    ap.add_argument('--out_dir', default='Data/processed_stage2')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    betas = get_betas('linear', 1e-4, 2e-2, 1000)
    gen = GenModel(betas, embed_dim=64, dropout=0.1, extra_feature_channels=9).to(device)
    ck = torch.load(args.gen_ckpt, map_location=device)
    gen.model.load_state_dict(ck['model'])
    if ck.get('ema'):
        from crowngen.models.ema import EMA
        ema = EMA(gen.model, 0.995); ema.load_state_dict(ck['ema']); ema.apply_to(gen.model)
    gen.eval()
    bnd = BoundEncoder(5, 0.3, args.bound_max_missing, mask_mode='official').to(device)
    bnd.load_state_dict(torch.load(args.bound_ckpt, map_location=device)); bnd.eval()
    print('models loaded', flush=True)

    pids = [p for p in open(args.pids_file).read().split() if p]
    ok = fail = 0
    for pid in pids:
        try:
            src = f'{args.data_dir}/{pid}.npz'
            if not os.path.exists(src):
                print(f'  [{pid}] NO norm2'); fail += 1; continue
            pts, b, valid, orig = load_patient(src, 1024)
            x0 = torch.from_numpy(pts).unsqueeze(0).to(device)
            vt = torch.from_numpy(valid).unsqueeze(0).to(device)
            l_mask = 1.0 - vt; o_mask = vt
            if l_mask.sum() == 0:
                print(f'  [{pid}] no missing'); fail += 1; continue
            with torch.no_grad():
                pb = bnd(x0, o_mask.view(1, 28, 1, 1))[0].cpu().numpy()
            eff = pb.copy()
            arch = interpolate_positions(valid, b, interior_only=True)
            for s in range(28):
                if valid[s] == 0 and not np.isnan(arch[s, 0]):
                    eff[s, :3] = arch[s, :3]
            bound = torch.from_numpy(b).unsqueeze(0).to(device).clone()
            for s in range(28):
                if valid[s] == 0:
                    bound[0, s] = torch.from_numpy(eff[s]).to(device)
            with torch.no_grad():
                g = gen.sample(dict(x0=x0, l_mask=l_mask, o_mask=o_mask, bound=bound))[0].cpu().numpy()
            save = dict(orig)
            for s, fdi in enumerate(ZIGZAG_FDI_ORDER):
                if valid[s] == 0:
                    k = f'{jaw_of(fdi)}_{fdi}_pc'
                    save[k] = g[s].T.astype(np.float32)
                    bb = eff[s]
                    save[k.replace('_pc', '_bound')] = np.array([bb[0], bb[1], bb[2], bb[4], bb[3]], dtype=np.float32)
            dst = f'{args.out_dir}/{pid}.npz'
            np.savez_compressed(dst, **save)              # 강제 덮어쓰기 (skip 없음)
            assert os.path.exists(dst) and os.path.getsize(dst) > 1000, 'write size check'
            print(f'  [{pid}] OK {int(l_mask.sum())}개 채움 ({os.path.getsize(dst)//1024}KB)'); ok += 1
        except Exception as e:
            import traceback
            print(f'  [{pid}] FAIL: {type(e).__name__}: {e}'); traceback.print_exc(); fail += 1
    print(f'DONE ok={ok} fail={fail}', flush=True)


if __name__ == '__main__':
    main()
