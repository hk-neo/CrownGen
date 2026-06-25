"""gen2k vs gen2k_nobound 생성 크라운 비교용 웹 시각화 데이터 준비.

동일 환자 · 동일 타겟 마스킹 · 동일 노이즈 시드로 두 모델을 샘플링 →
생성 점구름 + per-tooth CD + GT + 컨텍스트 점 → runs2/viz/gen_compare/data.js.
gen2k     : GT bound (학습 조건)
nobound   : zero bound (학습 조건)
EMA 가중치로 샘플링.
"""
import argparse, os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import importlib.util
_st = importlib.util.spec_from_file_location('tbo', 'scripts/gen_train.py')
tbo = importlib.util.module_from_spec(_st); _st.loader.exec_module(tbo)
from crowngen.external.gen_diffusion import GenModel, get_betas
from crowngen.losses.chamfer import chamfer_distance_l1

MODELS = [
    {"id": "gen2k",   "label": "gen2k (boundary O)",  "ckpt": "runs2/gen2k_last.pt",
     "use_gt_bound": True,  "color": "#e74c3c"},
    {"id": "nobound", "label": "gen2k_nobound (boundary X)", "ckpt": "runs2/gen2k_nobound_last.pt",
     "use_gt_bound": False, "color": "#9b59b6"},
]
CTX_PER_SLOT = 150
N_POINTS = 1024


def load_model(path, device):
    betas = get_betas('linear', 1e-4, 2e-2, 1000)
    m = GenModel(betas, embed_dim=64, dropout=0.1, extra_feature_channels=9).to(device)
    ck = torch.load(path, map_location=device)
    m.model.load_state_dict(ck['model']); m.eval()
    if ck.get('ema'):
        from crowngen.models.ema import EMA
        ema = EMA(m.model, 0.995); ema.load_state_dict(ck['ema']); ema.apply_to(m.model)
    return m


def downsample(points, per):
    if len(points) > per:
        idx = np.linspace(0, len(points) - 1, per).astype(int)
        return points[idx]
    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_patients', type=int, default=8)
    ap.add_argument('--data_dir', default='Data/processed_norm2')
    ap.add_argument('--split_file', default='Data/SourceC_Teeth3DS/train_val_split.json')
    ap.add_argument('--out', default='runs2/viz/gen_compare/data.js')
    args = ap.parse_args()

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    splits = json.load(open(args.split_file))
    va = tbo.GenDataset(args.data_dir, splits['stage1_val'], N_POINTS, require_full=True, augment=False)
    models = {m["id"]: load_model(m["ckpt"], device) for m in MODELS}

    random.seed(0)
    cases = []
    for pi in range(min(args.n_patients, len(va))):
        s = va[pi]
        x0 = s['points'].unsqueeze(0).to(device)          # (1,28,3,1024)
        gt_bound = s['bound'].unsqueeze(0).to(device)
        lm = torch.zeros(1, 28)
        idx = torch.randperm(28)[:random.randint(1, 6)]; lm[0, idx] = 1
        lm = lm.to(device); om = 1 - lm
        targets = [int(t) for t in np.where(lm[0].cpu().numpy() > 0)[0]]
        seed = 1000 + pi
        x0_np = x0[0].cpu().numpy()                        # (28,3,1024)

        # 컨텍스트 점(present & non-target) 다운샘플
        ctx = []
        for t in range(28):
            if t in targets:
                continue
            ctx.extend(downsample(x0_np[t].T, CTX_PER_SLOT).tolist())
        # GT 타겟 점 (전체 1024)
        gt_pts = {str(t): x0_np[t].T.tolist() for t in targets}

        gen = {}
        for m in MODELS:
            bound = gt_bound if m["use_gt_bound"] else torch.zeros_like(gt_bound)
            torch.manual_seed(seed)                        # 동일 노이즈 → 페어드 공정
            with torch.no_grad():
                g = models[m["id"]].sample(dict(x0=x0, l_mask=lm, o_mask=om, bound=bound))
            g_np = g[0].cpu().numpy()                      # (28,3,1024)
            tooth = {}
            for t in targets:
                p = torch.from_numpy(g_np[t].T).unsqueeze(0)
                q = torch.from_numpy(x0_np[t].T).unsqueeze(0)
                cd = chamfer_distance_l1(p, q).item()
                tooth[str(t)] = {"pts": g_np[t].T.tolist(), "cd": round(cd * 1e3, 1)}
            gen[m["id"]] = tooth

        cases.append({"idx": pi, "targets": targets, "context_pts": ctx, "gt": gt_pts, "gen": gen})
        cds = {m["id"]: round(np.mean([gen[m["id"]][str(t)]["cd"] for t in targets]), 1) for m in MODELS}
        print(f"case {pi+1}/{args.n_patients}: targets FDI-ish {targets} | "
              f"gen2k CD×10³ {cds['gen2k']:.1f} | nobound {cds['nobound']:.1f}", flush=True)

    payload = {"models": [{k: v for k, v in m.items() if k != "ckpt"} for m in MODELS],
               "fdi_order": list(tbo.ZIGZAG_FDI_ORDER), "cases": cases}
    assert len(cases) == min(args.n_patients, len(va))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write("window.GEN_COMPARE_DATA = " + json.dumps(payload) + ";\n")
    print(f'WROTE {args.out} ({os.path.getsize(args.out)//1024} KB)', flush=True)


if __name__ == '__main__':
    main()
