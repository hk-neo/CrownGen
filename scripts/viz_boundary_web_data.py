"""Boundary 예측 side-by-side 웹 시각화용 데이터 준비.

각 체크포인트(official/voxatt/context)를 로드 → val 케이스별로 케이스 인덱스를
시드로 한 고정 마스킹을 적용(모든 런이 같은 결손 치아를 예측) → per-case
Dice/IoU 계산 → runs2/viz/web/data.js 로 덤프.
"""
import argparse, os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import importlib.util
_spec = importlib.util.spec_from_file_location('tbo', 'scripts/train_boundary_official.py')
tbo = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(tbo)
from crowngen.external import BoundEncoder
from crowngen.data.fdi import ZIGZAG_FDI_ORDER

RUNS = [
    {"id": "official_long", "label": "official (long)", "ckpt": "runs2/boundary_official_long.pt",
     "mask_mode": "official", "voxel_attention": False, "color": "#e74c3c", "logged_dice": 0.770},
    {"id": "official_cont", "label": "official (cont077)", "ckpt": "runs2/boundary_official_cont.pt",
     "mask_mode": "official", "voxel_attention": False, "color": "#f39c12", "logged_dice": 0.756},
    {"id": "context",  "label": "context",  "ckpt": "runs2/boundary_context.pt",
     "mask_mode": "context",  "voxel_attention": False, "color": "#3498db", "logged_dice": 0.767},
    {"id": "voxatt",   "label": "voxatt",   "ckpt": "runs2/boundary_voxatt.pt",
     "mask_mode": "official", "voxel_attention": True,  "color": "#9b59b6", "logged_dice": 0.507},
]
CONTEXT_PTS_PER_SLOT = 200
RES = 32


def fixed_exist_mask(valid, max_missing, seed):
    """케이스 시드로 결정론적 마스킹 → (exist (28,1,1), miss (28,)). 모든 런에 동일."""
    rng = random.Random(seed)
    present = np.where(valid > 0)[0]
    if len(present) == 0:
        return valid.reshape(28, 1, 1).astype(np.float32), np.zeros(28, np.float32)
    k = min(rng.randint(1, max_missing), len(present))
    pick = rng.sample(range(len(present)), k)
    idx = present[pick]
    exist = valid.copy(); miss = np.zeros_like(valid)
    exist[idx] = 0; miss[idx] = 1.0
    return exist.reshape(28, 1, 1).astype(np.float32), miss.astype(np.float32)


def load_model(run, device):
    m = BoundEncoder(output_dim=5, dropout=0.3, max_missing_teeth=6,
                     mask_mode=run["mask_mode"], voxel_attention=run["voxel_attention"]).to(device)
    m.load_state_dict(torch.load(run["ckpt"], map_location=device))
    m.eval()
    return m


def case_dice_iou(pred, gt, miss):
    """pred, gt: (28,5) np; miss: (28,) np. → (dice, iou) 결손 슬롯 평균."""
    ds, ios = [], []
    for ti in range(28):
        if miss[ti] != 1:
            continue
        p, g = pred[ti], gt[ti]
        cmin = np.minimum(p[:3], g[:3]) - [max(p[4], g[4]), max(p[4], g[4]), max(p[3], g[3])]
        cmax = np.maximum(p[:3], g[:3]) + [max(p[4], g[4]), max(p[4], g[4]), max(p[3], g[3])]
        gmn = (cmin - 0.05).tolist(); gmx = (cmax + 0.05).tolist()
        mp = tbo._cyl_mask(p, gmn, gmx, RES); mg = tbo._cyl_mask(g, gmn, gmx, RES)
        sp, sg, inter = mp.sum(), mg.sum(), (mp * mg).sum()
        if sp + sg > 0:
            ds.append((2 * inter / (sp + sg)).item())
        un = sp + sg - inter
        if un > 0:
            ios.append((inter / un).item())
    d = float(np.mean(ds)) if ds else 0.0
    i = float(np.mean(ios)) if ios else 0.0
    return d, i


def context_points(sample, miss, per_slot):
    """present & non-missing 슬롯 점을 슬롯당 per_slot 개로 균일 서브샘플 → (N,3) 리스트."""
    pts = sample['points'].numpy()      # (28,3,P)
    valid = sample['valid'].numpy()     # (28,)
    out = []
    for s in range(28):
        if valid[s] > 0 and miss[s] == 0:
            p = pts[s].T                # (P,3)
            if len(p) > per_slot:
                idx = np.linspace(0, len(p) - 1, per_slot).astype(int)
                p = p[idx]
            out.extend(p.tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='Data/processed_norm2')
    ap.add_argument('--split_file', default='Data/SourceC_Teeth3DS/train_val_split.json')
    ap.add_argument('--n_cases', type=int, default=40)
    ap.add_argument('--out', default='runs2/viz/web/data.js')
    args = ap.parse_args()

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    va = tbo.BoundDataset(args.data_dir, args.split_file, 'stage1_val', 512, (1, 6), augment=False)
    n = min(args.n_cases, len(va))

    models = {}
    for r in RUNS:
        if not os.path.exists(r["ckpt"]):
            print(f'WARN skip {r["id"]}: {r["ckpt"]} 없음', flush=True)
            continue
        try:
            models[r["id"]] = load_model(r, device)
            print(f'loaded {r["id"]} <- {r["ckpt"]}', flush=True)
        except Exception as e:
            print(f'WARN skip {r["id"]}: 로드 실패 {e}', flush=True)

    runs_meta = [r for r in RUNS if r["id"] in models]
    cases = []
    with torch.no_grad():
        for ci in range(n):
            s = va[ci]
            exist, miss = fixed_exist_mask(s['valid'].numpy(), 6, seed=1000 + ci)
            exist_t = torch.from_numpy(exist).unsqueeze(0).to(device)   # (1,28,1,1)
            pts = s['points'].unsqueeze(0).to(device)                   # (1,28,3,P)
            gt = s['bound'].numpy()                                     # (28,5)
            miss_np = miss
            missing_slots = [int(t) for t in range(28) if miss_np[t] == 1]

            pred = {"context_pts": context_points(s, miss_np, CONTEXT_PTS_PER_SLOT),
                    "gt": {str(t): [float(v) for v in gt[t]] for t in missing_slots},
                    "missing": missing_slots, "predictions": {}}
            for r in runs_meta:
                p = models[r["id"]](pts, exist_t)[0].cpu().numpy()      # (28,5)
                d, i = case_dice_iou(p, gt, miss_np)
                pred["predictions"][r["id"]] = {
                    "cyl": {str(t): [float(v) for v in p[t]] for t in missing_slots},
                    "dice": round(d, 4), "iou": round(i, 4)}
            cases.append({"idx": ci, **pred})
            print(f'case {ci+1}/{n}: missing={missing_slots}', flush=True)

    payload = {"runs": [{k: v for k, v in r.items() if k != "ckpt"} for r in runs_meta],
               "fdi_order": list(ZIGZAG_FDI_ORDER), "cases": cases}

    # --- 검증 assertion ---
    assert len(cases) == n, f"case 수 불일치 {len(cases)} vs {n}"
    for c in cases:
        for r in runs_meta:
            pr = c["predictions"][r["id"]]
            assert 0.0 <= pr["dice"] <= 1.0, f"case {c['idx']} {r['id']} Dice 범위 이탈 {pr['dice']}"
    print(f'ASSERT OK: {n} cases × {len(runs_meta)} runs', flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write("window.BOUNDARY_DATA = " + json.dumps(payload) + ";\n")
    print(f'WROTE {args.out} ({os.path.getsize(args.out)//1024} KB)', flush=True)


if __name__ == '__main__':
    main()
