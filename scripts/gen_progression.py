"""에폭별 snapshot 으로 같은 환자 크라운 샘플링 → .ply 저장 (학습 추이 시각화).

같은 환자/마스킹/노이즈시드 고정 → 가중치(ep)만 바꿔가며 샘플링.
출력: runs2/ply/patient{P}_ep{E}.ply (생성 타겟 치아) + patient{P}_gt.ply (GT)
맥북 MeshLab/CloudCompare 에서 같은 환자의 ep1100→2000 을 순서대로 열면 진화 확인.
"""
import sys, os, json
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from crowngen.external.gen_diffusion import GenModel, get_betas
import importlib.util
st = importlib.util.spec_from_file_location('tbo', 'scripts/gen_train.py')
tbo = importlib.util.module_from_spec(st); st.loader.exec_module(tbo)

OUT = 'runs2/ply'
CKPTS = ['ep1100', 'ep1300', 'ep1500', 'ep1700', 'ep1900', 'last']  # last = ep2000
PATIENTS = [0, 1, 2]
SEED = 0
TARGET_SLOTS = [12, 13]  # 같은 2개 치아를 항상 타겟으로


def save_ply(points, path, color=(220, 30, 30)):
    points = np.asarray(points)
    n = len(points)
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {n}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p in points:
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {color[0]} {color[1]} {color[2]}\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    betas = get_betas('linear', 1e-4, 2e-2, 1000)
    sp = json.load(open('Data/SourceC_Teeth3DS/train_val_split.json'))
    va = tbo.GenDataset('Data/processed_norm2', sp['stage1_val'], 1024, require_full=True, augment=False)

    # 환자별 고정 데이터/마스킹 준비 + GT 저장
    pdata = {}
    for pi in PATIENTS:
        s = va[pi]
        x0 = s['points'].unsqueeze(0).to(device)
        lm = torch.zeros(1, 28, device=device)
        lm[0, TARGET_SLOTS] = 1
        om = 1 - lm
        bound = s['bound'].unsqueeze(0).to(device)
        pdata[pi] = (x0, lm, om, bound)
        # GT 타겟 치아 점(초록) 저장
        gt_pts = np.concatenate([x0[0, t].T.cpu().numpy() for t in TARGET_SLOTS])
        save_ply(gt_pts, f'{OUT}/patient{pi}_gt.ply', color=(30, 180, 30))
        print(f'patient{pi}: GT saved ({len(gt_pts)} pts)', flush=True)

    # 에폭별 가중치로 샘플링
    for ep in CKPTS:
        ck = f'runs2/gen2k_{ep}.pt'
        m = GenModel(betas, embed_dim=64, dropout=0.1, extra_feature_channels=9).to(device)
        c = torch.load(ck, map_location=device)
        m.model.load_state_dict(c['model'])
        if c.get('ema'):
            from crowngen.models.ema import EMA
            e = EMA(m.model, 0.995); e.load_state_dict(c['ema']); e.apply_to(m.model)
        m.eval()
        for pi in PATIENTS:
            x0, lm, om, bound = pdata[pi]
            torch.manual_seed(SEED)  # 노이즈 고정 → 가중치만 영향
            with torch.no_grad():
                gen = m.sample(dict(x0=x0, l_mask=lm, o_mask=om, bound=bound)).cpu().numpy()
            gpts = np.concatenate([gen[0, t].T for t in TARGET_SLOTS])
            tag = '2000' if ep == 'last' else ep.replace('ep', '')
            save_ply(gpts, f'{OUT}/patient{pi}_ep{tag}.ply')
            print(f'  ep{tag} patient{pi}: {len(gpts)} pts saved', flush=True)
        del m; torch.cuda.empty_cache()
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
