"""PSR GT cache (.npz) -> torch training item."""
import os
import glob
import json

import numpy as np
import torch
from torch.utils import data

CACHE_DIR = 'runs2/sap_cache'


def get_split_pids(split: str):
    """Return list of patient IDs for the given split.

    split in {'train', 'val', 'eval'}:
      - train  : stage1_train excluding those in stage1_val
      - val    : first 16 of stage1_val
      - eval   : first 6 of stage1_val (mesh_demo compatibility)
    """
    sp = json.load(open('Data/SourceC_Teeth3DS/train_val_split.json'))
    val_set = set(sp['stage1_val'])
    if split == 'train':
        return [p for p in sp['stage1_train'] if p not in val_set]
    if split == 'val':
        return sp['stage1_val'][:16]
    if split == 'eval':
        return sp['stage1_val'][:6]
    raise ValueError(split)


class ToothDataset(data.Dataset):
    """Wraps the PSR GT cache as a torch Dataset.

    Cache files are named ``<pid>_FDI<tooth>.npz`` under ``runs2/sap_cache/``.
    Only files that currently exist on disk are included (tolerant of a
    partially-built cache). Each ``__getitem__`` returns a dict with the
    fields expected by the SAP Trainer::

        {
            'inputs':        (1, 1024, 3)  float32  – point cloud, for Encoder
            'gt_psr':        (1, 64, 64, 64) float32 – PSR volume, for Decoder
            'gt_points':     (1024, 3) float32        – raw point cloud (clone)
            'gt_points.normals': (1024, 3) float32    – zeros (placeholder)
        }

    After DataLoader collate with batch=B the shapes become
    ``(B, 1, 1024, 3)`` and ``(B, 1, 64, 64, 64)`` respectively.
    """

    def __init__(self, split: str = 'train', max_items: int = None):
        self.split = split
        pids = get_split_pids(split)

        # Build set of cached file basenames (without .npz) that exist on disk
        cache_set = {
            os.path.basename(p).replace('.npz', '')
            for p in glob.glob(f'{CACHE_DIR}/*.npz')
        }

        self.items = []
        for pid in pids:
            for cache_id in cache_set:
                if cache_id.startswith(pid + '_FDI'):
                    self.items.append(f'{CACHE_DIR}/{cache_id}.npz')

        if max_items:
            self.items = self.items[:max_items]

        print(f'ToothDataset[{split}]: {len(self.items)} items', flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d = np.load(self.items[i])
        pc = torch.from_numpy(np.asarray(d['pc']).astype(np.float32))          # (1024, 3)
        psr = torch.from_numpy(np.asarray(d['psr_vol']).astype(np.float32))     # (64, 64, 64)
        return {
            'inputs': pc.unsqueeze(0),                    # (1, 1024, 3)
            'gt_psr': psr.unsqueeze(0),                   # (1, 64, 64, 64)
            'gt_points': pc.clone(),                     # (1024, 3) raw copy
            'gt_points.normals': torch.zeros_like(pc),  # (1024, 3) placeholder
        }
