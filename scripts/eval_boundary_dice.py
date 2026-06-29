"""boundary Dice 2x2 cross-eval: (현 official / aligned 모델) × (norm2 / aligned_norm 데이터).
Dice 차이가 data/scale 탓인지 model 탓인지 가린다."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import torch
from torch.utils.data import DataLoader
from crowngen.external import BoundEncoder
from train_boundary_official import BoundDataset, eval_dice_iou

SPLIT = 'Data/SourceC_Teeth3DS/train_val_split.json'


def collate(batch):
    return {k: torch.stack([b[k] for b in batch], 0) for k in batch[0]}


def ev(ckpt, data_dir, tag):
    dev = torch.device('cuda')
    m = BoundEncoder(5, 0.3, 6, 'official').to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev)); m.eval()
    va = BoundDataset(data_dir, SPLIT, 'stage1_val', 512, (1, 6), augment=False)
    loader = DataLoader(va, 4, shuffle=False, num_workers=2, collate_fn=collate)
    d, io, n = eval_dice_iou(m, loader, dev, 6)
    print(f'  [{tag:28}] Dice {d:.3f}  IoU {io:.3f}', flush=True)
    return d


if __name__ == '__main__':
    print('=== 2x2 cross-eval (stage1_val) ===')
    ev('runs2/boundary_official_long.pt', 'Data/processed_norm2', 'official × norm2 (native)')
    ev('runs2/boundary_official_long.pt', 'Data/aligned_norm',    'official × aligned')
    ev('runs2/boundary_aligned.pt',       'Data/processed_norm2', 'aligned  × norm2')
    ev('runs2/boundary_aligned.pt',       'Data/aligned_norm',    'aligned  × aligned (native)')
