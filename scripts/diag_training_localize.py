"""학습 자체 문제 진단: 학습 분포 안(완전치열, GT 있음)에서 heavy 마스킹 시
모델이 빈(숨긴) 치아 위치를 잘 잡는가?

OOD가 원인이면 → full arch에서 heavy 마스킹해도(분포 내) 위치 정확해야 함.
학습/설계가 원인이면 → full arch heavy 마스킹에서도 위치 오차 커야 함.

측정(GT 있음): masked 치아의 ① Dice ② 위치오차 ||pred(cx,cy,cz)-GT||.
light(1-6) vs heavy(clustered/heavy, max12) 비교. old vs new(G1) 비교.
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
import numpy as np
import torch
from crowngen.external import BoundEncoder
from crowngen.data.fdi import ZIGZAG_FDI_ORDER
from train_boundary_g1 import BoundDataset, make_realistic_mask, make_light_mask, _dice_iou


def jaw_of(fdi):
    return 'upper' if fdi // 10 in (1, 2) else 'lower'


@torch.no_grad()
def eval_on(model, loader, device, mask_fn, max_missing):
    """GT 있는 masked 치아 기준 Dice + 위치오차."""
    model.eval(); dices = []; poserr = []
    for b in loader:
        pts = b['points'].to(device); gt = b['bound'].numpy()
        exist, miss = mask_fn(b['valid'], max_missing)
        pred = model(pts, exist.to(device)).cpu().numpy(); mm = miss.numpy()
        for bi in range(pred.shape[0]):
            d, _ = _dice_iou(pred[bi], gt[bi], mm[bi]); dices += d
            for ti in range(28):
                if mm[bi, ti] == 1:
                    poserr.append(float(np.linalg.norm(pred[bi, ti, :3] - gt[bi, ti, :3])))
    return (float(np.mean(dices)) if dices else 0), (float(np.mean(poserr)) if poserr else 0), len(dices)


def main():
    device = torch.device('cuda')
    sp = json.load(open('Data/SourceC_Teeth3DS/train_val_split.json'))
    va = BoundDataset('Data/processed_norm2', sp['stage1_val'], 512, augment=False)  # 40 full, GT 있음
    from torch.utils.data import DataLoader
    loader = DataLoader(va, 4, shuffle=False, num_workers=2,
                        collate_fn=lambda bt: {k: torch.stack([b[k] for b in bt], 0) for k in bt[0]})

    cks = [('OLD', 'runs2/boundary_official_long.pt', 6),
           ('NEW(G1)', 'runs2/boundary_g1_best.pt', 12)]
    print(f'평가: stage1_val 완전치열 {len(va)}명 (GT 있음, 학습 분포 내)', flush=True)
    print(f'정상 인접 간격 ~0.16 → 위치오차 0.08≈틈 중간, 0.16≈치아 1개 폭', flush=True)
    for tag, ck, mm in cks:
        m = BoundEncoder(5, 0.3, mm, mask_mode='official').to(device)
        m.load_state_dict(torch.load(ck, map_location=device))
        for mname, mfn, mxm in [('light(1-6)', make_light_mask, 6),
                                ('heavy/clustered(≤12)', make_realistic_mask, 12)]:
            dl, pe, n = eval_on(m, loader, device, mfn, mxm)
            print(f'  [{tag:8} | {mname:18}] Dice {dl:.3f} · 위치오차 {pe:.3f} (n={n})', flush=True)


if __name__ == '__main__':
    main()
