"""Stage 2 pseudo-crown 데이터 확장.

gen2k(2000ep) 모델로 부분무치아 스캔의 빈 치아 자리에 크라운 생성 → 채워 넣어
완전 치열 확장 데이터셋 구축. boundary 모델로 빈 자리 경계 예측.
출력: processed_stage2/ (완전치열 원본 + pseudo-crown 채운 스캔).
"""
import sys, os, json, time
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
import numpy as np
import torch

from crowngen.external.gen_diffusion import GenModel, get_betas
from crowngen.external import BoundEncoder
from crowngen.data.fdi import ZIGZAG_FDI_ORDER
from diag_arch_interpolate import interpolate_positions   # ARCH 하이브리드 위치 보간


def jaw_of(fdi):
    return 'upper' if fdi // 10 in (1, 2) else 'lower'


def load_patient(npz_path, n_points=1024):
    """28슬롯 (3,P) 점 + bound(cx,cy,cz,h,r) + valid 로드."""
    d = np.load(npz_path)
    pts = np.zeros((28, 3, n_points), dtype=np.float32)
    bnd = np.zeros((28, 5), dtype=np.float32)
    valid = np.zeros(28, dtype=np.float32)
    for s, fdi in enumerate(ZIGZAG_FDI_ORDER):
        k = f'{jaw_of(fdi)}_{fdi}_pc'
        if k in d:
            pc = d[k]
            if pc.shape[0] >= n_points:
                pc = pc[np.random.permutation(pc.shape[0])[:n_points]]
            else:
                idx = np.random.choice(pc.shape[0], n_points, replace=True)
                pc = pc[idx]
            pts[s] = pc.T.astype(np.float32)
            bk = k.replace('_pc', '_bound')
            if bk in d:
                b = d[bk]
                bnd[s] = [b[0], b[1], b[2], b[4], b[3]]  # →(cx,cy,cz,h,r)
            valid[s] = 1.0
    return pts, bnd, valid, d


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_ckpt', default='runs2/gen2k_last.pt')
    ap.add_argument('--bound_ckpt', default='runs2/boundary_official_long.pt')
    ap.add_argument('--data_dir', default='Data/processed_norm2')
    ap.add_argument('--split_file', default='Data/SourceC_Teeth3DS/train_val_split.json')
    ap.add_argument('--out_dir', default='Data/processed_stage2')
    ap.add_argument('--n_points', type=int, default=1024)
    ap.add_argument('--shard', type=int, default=0, help='병렬 샤딩 인덱스')
    ap.add_argument('--nshards', type=int, default=1, help='전체 샤드 수')
    ap.add_argument('--arch_pos', action='store_true',
                    help='빈 슬롯 위치를 ARCH 하이브리드로: 내부 결손=아치 보간, 끝자리 결손=boundary. h,r은 boundary.')
    ap.add_argument('--bound_max_missing', type=int, default=12, help='boundary 모델 max_missing_teeth (G1=12)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    # 모델 로드
    betas = get_betas('linear', 1e-4, 2e-2, 1000)
    gen_model = GenModel(betas, embed_dim=64, dropout=0.1, extra_feature_channels=9).to(device)
    ck = torch.load(args.gen_ckpt, map_location=device)
    gen_model.model.load_state_dict(ck['model'])
    if ck.get('ema'):
        from crowngen.models.ema import EMA
        ema = EMA(gen_model.model, 0.995); ema.load_state_dict(ck['ema']); ema.apply_to(gen_model.model)
    gen_model.eval()
    print('gen model loaded (ep', ck.get('ep', '?'), ')', flush=True)

    bnd_model = BoundEncoder(5, 0.3, args.bound_max_missing, mask_mode='official').to(device)
    bnd_model.load_state_dict(torch.load(args.bound_ckpt, map_location=device))
    bnd_model.eval()
    print(f'boundary model loaded (max_missing={args.bound_max_missing}'
          f'{", ARCH 하이브리드 위치" if args.arch_pos else ""})', flush=True)

    splits = json.load(open(args.split_file))
    # stage2_train split에서 부분무치아(28 미만) 식별 + 완전치열(28) 복사
    all_pids = []
    for split in ['stage2_train', 'stage2_val', 'stage1_train', 'stage1_val']:
        all_pids += splits.get(split, [])
    all_pids = list(set(all_pids))

    full_pids = []
    partial_pids = []
    for pid in all_pids:
        npz = f'{args.data_dir}/{pid}.npz'
        if not os.path.exists(npz):
            continue
        d = np.load(npz)
        n = sum(1 for fdi in ZIGZAG_FDI_ORDER if f'{jaw_of(fdi)}_{fdi}_pc' in d)
        if n == 28:
            full_pids.append(pid)
        elif n >= 14:
            partial_pids.append(pid)

    # 샤딩: 부분무치아를 nshards 로 분할 (병렬 실행용). 완전치열 복사는 shard 0 만.
    partial_pids = partial_pids[args.shard::args.nshards]
    print(f'[shard {args.shard}/{args.nshards}] 완전치열 {len(full_pids)}명 + '
          f'이 샤드 부분무치아 {len(partial_pids)}명', flush=True)

    import shutil
    if args.shard == 0:
        for pid in full_pids:
            dst = f'{args.out_dir}/{pid}.npz'
            if not os.path.exists(dst):
                shutil.copy(f'{args.data_dir}/{pid}.npz', dst)
        print(f'완전치열 {len(full_pids)}명 복사 완료 (shard0)', flush=True)

    # 부분무치아 pseudo-crown 채우기 (1명씩)
    for i, pid in enumerate(partial_pids):
        dst = f'{args.out_dir}/{pid}.npz'
        if os.path.exists(dst):
            continue  # 이미 처리됨

        pts, bnd, valid, orig_data = load_patient(f'{args.data_dir}/{pid}.npz', args.n_points)
        x0 = torch.from_numpy(pts).unsqueeze(0).to(device)     # (1,28,3,P)
        valid_t = torch.from_numpy(valid).unsqueeze(0).to(device)
        l_mask = 1.0 - valid_t   # missing = target
        o_mask = valid_t          # present = context

        if l_mask.sum() == 0:
            shutil.copy(f'{args.data_dir}/{pid}.npz', dst)
            continue

        # boundary 예측 (missing 치아의 경계)
        with torch.no_grad():
            exist_mask = o_mask.view(1, 28, 1, 1)
            pred_bound = bnd_model(x0, exist_mask)  # (1,28,5) cx,cy,cz,h,r

        # effective bound 구성: present=GT, missing=ARCH 하이브리드(arch_pos) 또는 boundary 예측
        pb = pred_bound[0].cpu().numpy()           # (28,5)
        eff = pb.copy()
        if args.arch_pos:
            arch = interpolate_positions(valid, bnd, interior_only=True)  # 내부 결손만, 끝자리 NaN
            for s in range(28):
                if valid[s] == 0 and not np.isnan(arch[s, 0]):
                    eff[s, :3] = arch[s, :3]        # 내부 결손: 아치 보간 위치(cx,cy,cz); h,r은 boundary 유지
        # 끝자리 결손(NaN)은 eff=boundary 그대로
        bound = torch.from_numpy(bnd).unsqueeze(0).to(device).clone()
        for s in range(28):
            if valid[s] == 0:
                bound[0, s] = torch.from_numpy(eff[s]).to(device)

        # 크라운 샘플링 (missing 치아)
        with torch.no_grad():
            gen = gen_model.sample(dict(x0=x0, l_mask=l_mask, o_mask=o_mask, bound=bound))
        gen = gen[0].cpu().numpy()  # (28,3,P)

        # 채워 넣기: missing 슬롯에 생성 점 저장
        save_dict = dict(orig_data)
        for s, fdi in enumerate(ZIGZAG_FDI_ORDER):
            if valid[s] == 0:  # missing
                k = f'{jaw_of(fdi)}_{fdi}_pc'
                save_dict[k] = gen[s].T.astype(np.float32)  # (P,3)
                # boundary도 저장 (effective 값: ARCH 위치 또는 boundary 예측)
                bk = k.replace('_pc', '_bound')
                b = eff[s]
                save_dict[bk] = np.array([b[0], b[1], b[2], b[4], b[3]], dtype=np.float32)  # (cx,cy,cz,r,h)

        np.savez_compressed(dst, **save_dict)
        n_missing = int(l_mask.sum().item())
        print(f'  [{i+1}/{len(partial_pids)}] {pid}: {n_missing}개 채움', flush=True)

    print(f'Stage 2 데이터 구축 완료 → {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
