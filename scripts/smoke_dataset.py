"""Smoke test for ToothDataset.

Bypasses the crowngen.external.mesh_recon package __init__ chain
(which requires trimesh) by loading tooth_dataset.py directly.
"""
import os, sys

# Ensure project root and crowngen/external are on the path so the
# tooth_dataset module can resolve its own imports (glob, json, torch).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.join(ROOT, 'crowngen', 'external'))

import importlib.util
spec = importlib.util.spec_from_file_location(
    'tooth_dataset',
    os.path.join(ROOT, 'crowngen', 'external', 'mesh_recon', 'src', 'data', 'tooth_dataset.py')
)
mod = importlib.util.module_from_spec(spec)
sys.modules['tooth_dataset'] = mod
spec.loader.exec_module(mod)

ToothDataset = mod.ToothDataset

print('=== ToothDataset smoke test ===')

ds = ToothDataset('train')
print('len:', len(ds))
it = ds[0]
print('keys:', list(it.keys()))
print('inputs:', it['inputs'].shape)
print('gt_psr:', it['gt_psr'].shape)
print('gt_points:', it['gt_points'].shape)
print('gt_points.normals:', it['gt_points.normals'].shape)

print()
ds_val = ToothDataset('val')
print('val len:', len(ds_val))

print()
ds_eval = ToothDataset('eval')
print('eval len:', len(ds_eval))

print()
print('=== OK ===')
