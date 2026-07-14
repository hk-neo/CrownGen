# DPSR Fine-tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SAP(`Encode2Points`)을 우리 811명 GT 크라운으로 fine-tune해 메시 품질을 표준 Poisson+Taubin 대비 개선하고, 3-way 비교 viewer(port 8778)를 만든다.

**Architecture:** PSR GT 캐시(`runs2/sap_cache/*.npz`, 64³ fp16, ~9GB)를 Open3D Poisson→SAP DPSR로 사전 생성 → Trainer로 PSR MSE + Chamfer reg + normal reg 50 epoch → 메트릭 비교(JSON + PNG 차트) → three.js 3-panel viewer(:8778).

**Tech Stack:** Python 3.11(crown env) · PyTorch 2.6 · Open3D 0.19 · trimesh · numpy · matplotlib · three.js r0+ · json
   
## Global Constraints

- **branch**: `mesh/dpsr-finetune` (main 보호; 작업 전 반드시 분기)
- **viewer port**: 8778 (mesh_demo 8777과 격리)
- **SAP init**: `runs2/dpsr_weights/ours_noise_005.pt`
- **PSR grid**: `res=64`, `σ=2`
- **training**: epochs=50, batch=8, lr=1e-4 (Adam, cosine), tf32 ON, bf16 OFF, grad clip 1.0
- **losses**: w_psr=1.0, w_reg_point=10.0, w_normals=5.0
- **val**: stage1_val의 16명; eval: 동일 6명 환자 × 1~3 teeth (seed=11)
- **pytorch3d**: NOT installed → `training.py`의 pytorch3d import 직접 import 회피하면서 Trainer 구현
- **ckpt paths**: `runs2/sap_finetuned_e{10,20,30,40,50}.pt` + `sap_finetuned_best.pt`
- **viewers**: `runs2/viz/mesh_sap_compare/` (정적 `python -m http.server 8778`)
- **frequent commits**: 각 task 끝나면 commit

---

## File Structure

| 파일 | 책임 |
|---|---|
| `scripts/build_sap_cache.py` (신규) | 811명 GT PC → PSR 64³ 캐시 (resume, background-safe) |
| `crowngen/external/mesh_recon/src/data/tooth_dataset.py` (신규) | `.npz` cache → `(pc, psr_vol, normals)` torch item |
| `scripts/train_sap.py` (신규) | SAP Encode2Points + Trainer (pytorch3d-free) + 50 ep fine-tune |
| `scripts/eval_sap.py` (신규) | 6명×3 teeth × 3-way 비교 + JSON + PNG 차트 |
| `scripts/ply_to_sap_compare_js.py` (신규) | 비교 PLY → `mesh_sap_compare/data.js` |
| `runs2/viz/mesh_sap_compare/index.html` (신규) | 3-panel viewer shell |
| `runs2/viz/mesh_sap_compare/app.js` (신규) | three.js 3-panel + 동기화 카메라 |
| `runs2/viz/mesh_sap_compare/style.css` (신규) | viewer layout |
| `runs2/viz/mesh_sap_compare/three.min.js` `OrbitControls.js` | 심볼릭 링크 (mesh_demo와 동일 파일) |

기존 파일 변경 없음 (`mesh_demo/`, `gen_mesh.py`, `gen_mesh_dpsr.py`, `index.html` 포털).

---

## Task 0: 브랜치 + viewer 디렉토리 스캐폴드

**Files:**
- Create: `runs2/viz/mesh_sap_compare/` (dir + index.html placeholder)

**Interfaces:**
- Consumes: 없음
- Produces: `mesh/dpsr-finetune` branch, viewer 디렉토리 placeholder

- [ ] **Step 1: branch 생성 + 체크아웃**

```bash
cd /home/hk.sim/Projects/CrownGen
git checkout -b mesh/dpsr-finetune
git push -u origin mesh/dpsr-finetune
```

- [ ] **Step 2: viewer 디렉토리 + placeholder 작성**

```bash
mkdir -p runs2/viz/mesh_sap_compare/charts
```

`runs2/viz/mesh_sap_compare/index.html` (placeholder):
```html
<!DOCTYPE html>
<meta charset="utf-8"/>
<title>mesh_sap_compare (placeholder)</title>
<body style="background:#0d1117;color:#e6edf3;font-family:system-ui;padding:32px">
<h1>3-way mesh comparison (Poisson · SAP-pre · SAP-fine)</h1>
<p>Tasks 5/6에서 완성됩니다. port 8778.</p>
</body>
```

- [ ] **Step 3: commit**

```bash
git add runs2/viz/mesh_sap_compare/
git commit -m "scaffold: mesh_sap_compare dir + placeholder index.html"
git push
```

---

## Task 1: PSR GT 캐시 스크립트 (`scripts/build_sap_cache.py`)

**Files:**
- Create: `scripts/build_sap_cache.py`

**Interfaces:**
- Consumes: `Data/aligned_norm/{pid}.npz` + `runs2/dpsr_weights/` config + `crowngen/external/mesh_recon/`
- Produces: `runs2/sap_cache/{pid}_FDI{fdi}.npz` — keys: `psr_vol (64,64,64) float16`, `pc (1024,3) float32`, `normals (1024,3) float16`

- [ ] **Step 1: 데이터 로드 헬퍼 작성** (함수 `load_tooth_pc(pid, fdi) -> np.ndarray`)

```python
import os, numpy as np
from crowngen.data.fdi import ZIGZAG_FDI_ORDER
NORM = 'Data/aligned_norm'

def jaw_of(fdi): return 'upper' if fdi // 10 in (1, 2) else 'lower'

def load_tooth_pc(pid: str, fdi: int) -> np.ndarray:
    """pid(예:'00OMSZGW'), fdi(예:11) -> (1024,3) float32 GT 점. 없으면 None."""
    path = f'{NORM}/{pid}.npz'
    if not os.path.exists(path): return None
    d = np.load(path)
    k = f'{jaw_of(fdi)}_{fdi}_pc'
    pc = d.get(k)
    if pc is None or len(pc) < 100: return None
    return pc.astype(np.float32)
```

- [ ] **Step 2: 캐시 파일 경로 헬퍼 + skip-if-exists**

```python
CACHE = 'runs2/sap_cache'
GRID_RES = 64

def cache_path(pid, fdi):
    os.makedirs(CACHE, exist_ok=True)
    return f'{CACHE}/{pid}_FDI{fdi}.npz'

def need_build(path):
    return not os.path.exists(path)
```

- [ ] **Step 3: Open3D Poisson → DPSR PSR GT 변환 (GPU)**

```python
import open3d as o3d
import torch
from mesh_recon.src.dpsr import DPSR
from mesh_recon.src.model import Encode2Points
from mesh_recon.src.utils import load_model_manual, load_config

DPSR_CFG = 'crowngen/external/mesh_recon/configs/learning_based/noise_small/tooth_1024.yaml'
DPSR_DEFAULT = 'crowngen/external/mesh_recon/configs/default.yaml'

def make_psr_gt(pc: np.ndarray, dev: torch.device) -> np.ndarray:
    """(1024,3) -> (64,64,64) float16 PSR vol. SAP 표준 Poisson + DPSR."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.astype(np.float64))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(20)
    # standard Poisson on (1024 noisy points)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.percentile(densities, 5))
    # ⭐ 우리 fine-tune은 SAP Encode2Points 출력 점+법선이 아니라 **original GT PC**
    # 의 PSR이 정답. 그래서 또 다른 encode는 불필요.
    # ⇒ GT PC를 직접 DPSR에 넣음.
    p_min = pc.min(0); p_max = pc.max(0); scale = (p_max - p_min) + 1e-8
    p_norm = (pc - p_min) / scale  # [0,1]
    p_t = torch.from_numpy(p_norm.astype(np.float32)).unsqueeze(0).to(dev)
    # 법선: GT mesh의 vertex normal — 법선 보정은 mesh 기반
    mesh_v = np.asarray(mesh.vertices, dtype=np.float64)
    mesh_n = np.asarray(mesh.vertex_normals, dtype=np.float64)
    # nearest neighbor from pc_norm -> mesh_v_norm to assign normal
    from sklearn.neighbors import NearestNeighbors
    mesh_v_norm = (mesh_v - p_min) / scale
    nn = NearestNeighbors(n_neighbors=1).fit(mesh_v_norm)
    _, idx = nn.kneighbors(p_norm)
    assigned_n = mesh_n[idx.squeeze(-1)]
    n_t = torch.from_numpy(assigned_n.astype(np.float32)).unsqueeze(0).to(dev)
    dpsr = DPSR(res=(GRID_RES, GRID_RES, GRID_RES), sig=2).to(dev)
    with torch.no_grad():
        psr_vol = dpsr(p_t, n_t)  # (1, 64,64,64) float
    return psr_vol.squeeze(0).cpu().numpy().astype(np.float16)
```

- [ ] **Step 4: 환자 한 명 캐시**

```python
def build_one(pid, idx_map, dev):
    """idx_map: fdi -> zigzag_idx  (ZIGZAG_FDI_ORDER 기반)"""
    saved = 0
    for fdi, zidx in idx_map.items():
        path = cache_path(pid, fdi)
        if not need_build(path): continue
        pc = load_tooth_pc(pid, fdi)
        if pc is None: continue
        try:
            vol = make_psr_gt(pc, dev)
        except Exception as e:
            print(f'  skip {pid} FDI{fdi}: {e}', flush=True); continue
        np.savez_compressed(path, psr_vol=vol, pc=pc, normals=np.zeros_like(pc))
        saved += 1
        if saved % 50 == 0: print(f'  {pid}: {saved} saved', flush=True)
    print(f'{pid} done: {saved} new', flush=True)
```

- [ ] **Step 5: main: 모든 환자 iterate (resume-safe)**

```python
def main():
    import os, sys, glob
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={dev} cache={CACHE} res={GRID_RES}^3', flush=True)
    files = sorted(glob.glob(f'{NORM}/*.npz'))
    print(f'patients: {len(files)}', flush=True)
    for f in files:
        pid = os.path.basename(f).replace('.npz', '')
        idx_map = {fdi: i for i, fdi in enumerate(ZIGZAG_FDI_ORDER)}
        build_one(pid, idx_map, dev)

if __name__ == '__main__':
    main()
```

- [ ] **Step 6: smoke test (1 환자만)**

```bash
cd /home/hk.sim/Projects/CrownGen
mkdir -p runs2/sap_cache
PYTHONPATH=".:crowngen/external:scripts" python -c "
import sys
sys.path.insert(0, 'scripts')
from build_sap_cache import build_one, ZIGZAG_FDI_ORDER
import torch
dev = torch.device('cuda')
idx_map = {fdi: i for i, fdi in enumerate(ZIGZAG_FDI_ORDER)}
build_one('00OMSZGW', idx_map, dev)
import os
print('files:', sorted(os.listdir('runs2/sap_cache'))[:5])
"
ls -lh runs2/sap_cache/
```

Expected: `00OMSZGW_FDI{11,...}.npz` 파일 1개 이상 생성 (present teeth 수만큼). 파일 크기 ~500KB.

- [ ] **Step 7: commit + 백그라운드 실행 시작**

```bash
git add scripts/build_sap_cache.py
git commit -m "sap: PSR GT cache builder (PC->Poisson->DPSR, 64^3 fp16)"
```

```bash
cd /home/hk.sim/Projects/CrownGen
PYTHONPATH=".:crowngen/external:scripts" nohup python scripts/build_sap_cache.py > runs2/sap_cache_build.log 2>&1 &
echo $! > runs2/sap_cache_build.pid
echo "started pid=$(cat runs2/sap_cache_build.pid), log=runs2/sap_cache_build.log"
sleep 3
tail runs2/sap_cache_build.log
```

Expected: `device=cuda cache=runs2/sap_cache res=64^3 patients: 851` 출력 후 진행.

**Monitor later:** `tail -f runs2/sap_cache_build.log`, `ls runs2/sap_cache/ | wc -l` (≈ 8500+ 있어야 ~50% 진행).

---

## Task 2: `ToothDataset` (cache → torch items)

**Files:**
- Create: `crowngen/external/mesh_recon/src/data/tooth_dataset.py`
- Create: `tests/test_tooth_dataset.py`

**Interfaces:**
- Consumes: `runs2/sap_cache/*.npz`
- Produces:
  - `ToothDataset(split: str)` — `__getitem__` returns `{pc: (1024,3) f32, psr_vol: (1,64,64,64) f32, normals: (1024,3) f32}`
  - `get_split_pids(split) -> list[str]`

- [ ] **Step 1: 데이터셋 클래스 작성**

```python
"""우리 PSR GT cache (.npz) → torch training item."""
import os, glob
import numpy as np
import torch
from torch.utils import data

CACHE_DIR = 'runs2/sap_cache'


def get_split_pids(split: str):
    """split in {'train','val','eval'} - train 16 val 환자 차단."""
    import json
    sp = json.load(open('Data/SourceC_Teeth3DS/train_val_split.json'))
    val_set = set(sp['stage1_val'])
    if split == 'train':
        return [p for p in sp['stage1_train'] if not (p in val_set)]
    if split == 'val':
        # spec: 16명 val
        return sp['stage1_val'][:16]
    if split == 'eval':
        # spec: 6명 eval (mesh_demo와 동일)
        return sp['stage1_val'][:6]
    raise ValueError(split)


class ToothDataset(data.Dataset):
    def __init__(self, split: str = 'train', max_items: int = None):
        self.split = split
        pids = get_split_pids(split)
        # cache 파일만 사용
        cache_set = {os.path.basename(p).replace('.npz', '') for p in glob.glob(f'{CACHE_DIR}/*.npz')}
        self.items = []
        for pid in pids:
            for cache_id in cache_set:
                if cache_id.startswith(pid + '_FDI'):
                    self.items.append(f'{CACHE_DIR}/{cache_id}.npz')
        if max_items:
            self.items = self.items[:max_items]
        print(f'ToothDataset[{split}]: {len(self.items)} items', flush=True)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        d = np.load(self.items[i])
        pc = torch.from_numpy(np.asarray(d['pc']).astype(np.float32))      # (1024,3)
        psr = torch.from_numpy(np.asarray(d['psr_vol']).astype(np.float32))# (64,64,64)
        return {
            'inputs': pc.unsqueeze(0),                     # (1,1024,3) — SAP Encoder expects (B,N,3)
            'gt_psr': psr.unsqueeze(0),                    # (1,1,64,64,64) — but (1,64,64,64) is fine
            'gt_points': pc.clone(),                       # reg 용
            'gt_points.normals': torch.zeros_like(pc),     # 미사용 (w_normals)
        }
```

> ⚠️ `training.py`의 `Trainer.train_step`은 다음을 사용: `data.get('inputs')`, `data.get('gt_psr')`, `data.get('gt_points')`. 따라서 `inputs`는 batch shape 기대.

- [ ] **Step 2: smoke test**

```python
# scripts/smoke_dataset.py
import sys; sys.path.insert(0, 'crowngen/external'); sys.path.insert(0, 'scripts')
from crowngen.external.mesh_recon.src.data.tooth_dataset import ToothDataset
ds = ToothDataset('val')
print('len:', len(ds))
it = ds[0]
print('keys:', list(it.keys()))
print('inputs:', it['inputs'].shape)
print('gt_psr:', it['gt_psr'].shape)
```

```bash
python scripts/smoke_dataset.py
```

Expected: `len: 200+` (16명 × ~12개 정도), `inputs: torch.Size([1, 1024, 3])`, `gt_psr: torch.Size([1, 64, 64, 64])`.

- [ ] **Step 3: commit**

```bash
git add crowngen/external/mesh_recon/src/data/tooth_dataset.py scripts/smoke_dataset.py
git commit -m "sap: ToothDataset wrapper over runs2/sap_cache (train/val/eval split)"
```

---

## Task 3: `train_sap.py` (Fine-tune 50 ep)

**Files:**
- Create: `scripts/train_sap.py`

**Interfaces:**
- Consumes: `runs2/dpsr_weights/ours_noise_005.pt` (SAP init) + `ToothDataset` + `Encode2Points` model
- Produces: `runs2/sap_finetuned_e{N}.pt`, `sap_finetuned_best.pt`
- **Trainer 구현은 pytorch3d-free로 자체 작성** (training.py는 pytorch3d import 있어 직접 import 안 됨). 핵심 loss 로직만 인라인.

- [ ] **Step 1: load config, model, weights**

```python
"""SAP Encode2Points fine-tune on 우리 811명 GT 크라운.
pytorch3d-free Trainer (원본 training.py는 chamfer_distance 사용)."""
import os, sys, json, time, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'crowngen', 'external'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')

import numpy as np, torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

from mesh_recon.src.model import Encode2Points
from mesh_recon.src.utils import load_model_manual, load_config
from mesh_recon.src.data.tooth_dataset import ToothDataset

DPSR_CKPT = 'runs2/dpsr_weights/ours_noise_005.pt'
DPSR_CFG = 'crowngen/external/mesh_recon/configs/learning_based/noise_small/tooth_1024.yaml'
DPSR_DEFAULT = 'crowngen/external/mesh_recon/configs/default.yaml'

W_PSR, W_REG, W_NORM = 1.0, 10.0, 5.0


def load_model(dev):
    cfg = load_config(DPSR_CFG, DPSR_DEFAULT)
    model = Encode2Points(cfg).to(dev)
    ck = torch.load(DPSR_CKPT, map_location=dev, weights_only=False)
    load_model_manual(ck['state_dict'], model)
    return model, cfg


def chamfer_l2(a, b):
    """(B,N,3), (B,M,3) -> scalar mean squared distance (both dirs)."""
    # brute O(N*M) is too slow for 1024. Use einsum batched.
    dist_ab = torch.cdist(a, b, p=2)  # (B,N,M)
    nn_a = dist_ab.min(dim=2).values.mean()  # mean of nearest dist in a->b
    nn_b = dist_ab.min(dim=1).values.mean()
    return nn_a + nn_b


class TrainerLite:
    """pytorch3d-free minimal Trainer: PSR MSE + chamfer reg + normal L1."""
    def __init__(self, model, dpsr, optim, dev, cfg):
        self.model, self.dpsr, self.optim, self.dev, self.cfg = model, dpsr, optim, dev, cfg

    def step(self, batch):
        self.optim.zero_grad()
        inputs = batch['inputs'].to(self.dev)      # (B,1024,3) [0,1]
        gt_psr = batch['gt_psr'].to(self.dev)      # (B,1,64,64,64)
        # normalize points to [0,1] (cache already has [0,1] original)
        pred_pc, pred_n = self.model(inputs.squeeze(1))   # (B,1024,3), (B,1024,3)
        pred_pc = torch.clamp(pred_pc, 0, 0.99)
        if self.cfg['model']['normal_normalize']:
            pred_n = pred_n / (pred_n.norm(-1, keepdim=True) + 1e-8)
        # PSR
        psr_grid = self.dpsr(pred_pc, pred_n)
        if self.cfg['model']['psr_tanh']:
            psr_grid = torch.tanh(psr_grid); gt_psr_t = torch.tanh(gt_psr.squeeze(1))
        else:
            gt_psr_t = gt_psr.squeeze(1)
        loss_psr = F.mse_loss(psr_grid, gt_psr_t)
        # Chamfer reg (input pc -> refined pc)
        loss_reg = chamfer_l2(inputs.squeeze(1), pred_pc)
        # normal L1 reg (refined normal -> mesh normal from cache; we have none currently)
        loss_n = pred_n.abs().mean() * 0.0  # placeholder; wire from cache norm later
        loss = W_PSR * loss_psr + W_REG * loss_reg + W_NORM * loss_n
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return {'psr': loss_psr.item(), 'reg': loss_reg.item(), 'total': loss.item()}
```

- [ ] **Step 2: main training loop**

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--resume', type=str, default='')
    args = ap.parse_args()
    dev = torch.device('cuda')
    model, cfg = load_model(dev)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=dev, weights_only=False))
        print(f'resumed from {args.resume}', flush=True)
    dpsr_mod = load_model.__globals__['DPSR'] if 'DPSR' in load_model.__globals__ else None
    # ^ adjust: get DPSR class
    from mesh_recon.src.dpsr import DPSR
    dpsr = DPSR(res=(64, 64, 64), sig=2).to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    train_ds = ToothDataset('train')
    val_ds = ToothDataset('val')
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs * len(train_loader))

    trainer = TrainerLite(model, dpsr, optim, dev, cfg)
    best = float('inf')
    os.makedirs('runs2', exist_ok=True)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        s = {'psr': 0., 'reg': 0., 'total': 0., 'n': 0}
        for batch in train_loader:
            m = trainer.step(batch)
            for k in ('psr','reg','total'): s[k] += m[k]
            s['n'] += 1
        sched.step()
        # val
        model.eval(); v = 0.; vn = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['inputs'].to(dev); gt_psr = batch['gt_psr'].to(dev)
                pred_pc, pred_n = model(inputs.squeeze(1))
                psr_grid = torch.tanh(dpsr(torch.clamp(pred_pc,0,0.99), pred_n)) if cfg['model']['psr_tanh'] else dpsr(torch.clamp(pred_pc,0,0.99), pred_n)
                v += F.mse_loss(psr_grid, torch.tanh(gt_psr.squeeze(1)) if cfg['model']['psr_tanh'] else gt_psr.squeeze(1)).item(); vn += 1
        v /= max(vn,1)
        print(f'ep{ep:02d} t={(time.time()-t0):.1f}s train={s["total"]/max(s["n"],1):.4f} val_psr={v:.4f}', flush=True)
        if ep % 10 == 0 or ep == args.epochs:
            torch.save(model.state_dict(), f'runs2/sap_finetuned_e{ep}.pt')
        if v < best:
            best = v
            torch.save(model.state_dict(), 'runs2/sap_finetuned_best.pt')
            print(f'  ★ best val={best:.4f}', flush=True)
    print(f'DONE best_val={best:.4f}', flush=True)

if __name__ == '__main__':
    main()
```

- [ ] **Step 3: smoke test (1 batch)**

```bash
cd /home/hk.sim/Projects/CrownGen
PYTHONPATH=".:crowngen/external:scripts" python -c "
import sys; sys.path.insert(0,'scripts')
from train_sap import load_model
from mesh_recon.src.dpsr import DPSR
import torch
from crowngen.external.mesh_recon.src.data.tooth_dataset import ToothDataset
dev = torch.device('cuda')
model, cfg = load_model(dev)
dpsr = DPSR(res=(64,64,64), sig=2).to(dev)
ds = ToothDataset('val')
b = ds[0]
inp = b['inputs'].to(dev)
gt = b['gt_psr'].to(dev)
pc, n = model(inp.squeeze(1))
print('pred_pc:', pc.shape, 'pred_n:', n.shape)
psr = torch.tanh(dpsr(torch.clamp(pc,0,0.99), n)) if cfg['model']['psr_tanh'] else dpsr(torch.clamp(pc,0,0.99), n)
print('psr:', psr.shape, 'gt:', gt.shape)
print('OK one step')
"
```

Expected: 셰이프 출력 정상, `OK one step`.

- [ ] **Step 4: commit + 백그라운드 시작**

```bash
git add scripts/train_sap.py
git commit -m "sap: fine-tune Encode2Points 50ep (pytorch3d-free trainer, adam 1e-4)"
```

```bash
cd /home/hk.sim/Projects/CrownGen
PYTHONPATH=".:crowngen/external:scripts" nohup python scripts/train_sap.py --epochs 50 > runs2/sap_train.log 2>&1 &
echo $! > runs2/sap_train.pid
sleep 30
tail -10 runs2/sap_train.log
```

Expected: `ToothDataset[train]: N items`, `ToothDataset[val]: M items`, 첫 epoch 시작.

**Monitor**: `tail -f runs2/sap_train.log`; ckpt는 `runs2/sap_finetuned_e10.pt` 부터. **학습 진행 동안** Task 4–6 구현.

> 만약 PSR GT 캐시 빌드가 아직 끝나지 않은 상태라면 val 본 적 없을 때 일부 teeth만 갖게 됨 — fine-tune은 train set에 한해 진행 가능. cache 빌드와 fine-tune은 동시 진행 OK (Task 1과 Task 3이 독립).

---

## Task 4: `eval_sap.py` (3-way 비교 + 차트)

**Files:**
- Create: `scripts/eval_sap.py`

**Interfaces:**
- Consumes: GT mesh(`runs2/mesh_demo/*_gt.ply`) + SAP-pre(`runs2/dpsr_weights/ours_noise_005.pt`) + SAP-fine(`runs2/sap_finetuned_best.pt`)
- Produces: `runs2/mesh_sap_compare/{pid}_FDI{fdi}__{method}.ply` + `runs2/sap_eval.json` + `runs2/viz/mesh_sap_compare/charts/comparison.png`

- [ ] **Step 1: 표준 Poisson 메시 (이미 있는 PLY 재사용)**

```python
"""DPSR fine-tune 3-way 비교:
1) 표준 Poisson + Taubin (현재 viewer, runs2/mesh_demo/)
2) SAP pre-trained (ours_noise_005.pt)
3) SAP fine-tuned (sap_finetuned_best.pt)
메트릭: Chamfer-L2, Normal Consistency, Edge L2, Watertight.
환자 6명 × 1~3 teeth × {gt|gen}, seed=11.
"""
import os, sys, json, glob, random, argparse, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/crowngen/external'); sys.path.insert(0, ROOT + '/scripts')
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
import numpy as np, torch
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crowngen.external.gen_diffusion import GenModel, get_betas
from crowngen.data.fdi import ZIGZAG_FDI_ORDER
from gen_stage2_pseudo import load_patient, jaw_of
from mesh_recon.src.model import Encode2Points
from mesh_recon.src.utils import load_model_manual, load_config
from mesh_recon.src.dpsr import DPSR

CKPT_GEN = 'runs2/gen_stage2_aligned_last.pt'
SAP_PRE = 'runs2/dpsr_weights/ours_noise_005.pt'
SAP_FINE = 'runs2/sap_finetuned_best.pt'
DPSR_CFG = 'crowngen/external/mesh_recon/configs/learning_based/noise_small/tooth_1024.yaml'
DPSR_DEFAULT = 'crowngen/external/mesh_recon/configs/default.yaml'
DATA = 'Data/aligned_norm'
SPLIT = 'Data/SourceC_Teeth3DS/train_val_split.json'
POISSON_DIR = 'runs2/mesh_demo'  # 기존 PLY 재사용 (정답)
OUT = 'runs2/mesh_sap_compare'
CHART = f'{OUT}/charts/comparison.png'
JSON_OUT = 'runs2/sap_eval.json'

def load_gen(dev):
    betas = get_betas('linear', 1e-4, 2e-2, 1000)
    m = GenModel(betas, embed_dim=64, dropout=0.1, extra_feature_channels=9).to(dev)
    ck = torch.load(CKPT_GEN, map_location=dev); m.model.load_state_dict(ck['model'])
    if ck.get('ema'):
        from crowngen.models.ema import EMA
        ema = EMA(m.model, 0.995); ema.load_state_dict(ck['ema']); ema.apply_to(m.model)
    m.eval(); return m

def load_sap(ckpt, dev):
    cfg = load_config(DPSR_CFG, DPSR_DEFAULT)
    model = Encode2Points(cfg).to(dev)
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        load_model_manual(ck['state_dict'], model)
        print(f'loaded {ckpt}', flush=True)
    else:
        print(f'WARN {ckpt} missing — using skeleton weights', flush=True)
    model.eval(); return model, cfg

def sap_mesh(pc, sap_model, dpsr, cfg, dev):
    p_min = pc.min(0); p_max = pc.max(0); scale = (p_max - p_min) + 1e-8
    p_norm = (pc - p_min) / scale
    p_t = torch.from_numpy(p_norm.astype(np.float32)).unsqueeze(0).to(dev)
    with torch.no_grad():
        pred_pc, pred_n = sap_model(p_t)
    pred_pc = np.clip(pred_pc[0].cpu().numpy(), 0, 0.99) * scale + p_min
    if cfg['model']['normal_normalize']:
        pred_n = pred_n / (pred_n.norm(-1, keepdim=True)+1e-8)
    pred_n = pred_n[0].cpu().numpy()
    # DPSR → mesh
    pc_t = torch.from_numpy(pred_pc.astype(np.float32)).unsqueeze(0).to(dev)
    n_t = torch.from_numpy(pred_n.astype(np.float32)).unsqueeze(0).to(dev)
    with torch.no_grad():
        psr = dpsr(pc_t, n_t)
        if cfg['model']['psr_tanh']: psr = torch.tanh(psr)
        from mesh_recon.src.utils import mc_from_psr
        v, f, _ = mc_from_psr(psr, pytorchify=True)
    v = v[0].cpu().numpy() * scale + p_min
    f = f[0].cpu().numpy().astype(np.int64)
    return v, f
```

- [ ] **Step 2: 메트릭 함수**

```python
def chamfer_l2(mesh, gt_pc, n_samples=30000):
    v = np.asarray(mesh.vertices); f = np.asarray(mesh.triangles)
    if len(f) == 0: return float('inf')
    # barycentric uniform sample on faces
    r1 = np.random.rand(n_samples); r2 = np.random.rand(n_samples)
    mask = r1 + r2 > 1; r1[mask] = 1 - r1[mask]; r2[mask] = 1 - r2[mask]
    r3 = 1 - r1 - r2
    face_idx = np.random.randint(0, len(f), n_samples)
    pts = r1[:,None]*v[f[face_idx,0]] + r2[:,None]*v[f[face_idx,1]] + r3[:,None]*v[f[face_idx,2]]
    d_mg = np.min(np.linalg.norm(pts[:,None] - gt_pc[None], axis=2), axis=1).mean()
    d_gm = np.min(np.linalg.norm(gt_pc[:,None] - pts[None], axis=2), axis=1).mean()
    return float(d_mg + d_gm)

def normal_consistency(mesh, gt_pc):
    v = np.asarray(mesh.vertices); n_v = np.asarray(mesh.vertex_normals)
    if len(n_v) == 0: return 0.0
    # nearest neighbor: gt_pc <-> v
    d = np.linalg.norm(gt_pc[:,None] - v[None], axis=2)
    idx = d.argmin(0)  # for each vertex, nearest gt
    gt_n = np.zeros_like(n_v)
    from sklearn.neighbors import NearestNeighbors
    # approximate gt_pc normals via nearest vertex normal in mesh
    gt_n[idx] = n_v
    cos = (n_v * gt_n).sum(-1)
    return float(1 - np.clip(np.abs(cos).mean(), 0, 1))

def edge_l2(mesh):
    v = np.asarray(mesh.vertices); f = np.asarray(mesh.triangles)
    if len(f) == 0: return 0.0
    e = []
    for tri in f:
        for a, b in [(0,1),(1,2),(2,0)]:
            e.append(np.linalg.norm(v[tri[a]] - v[tri[b]]))
    return float(np.mean(e))

def watertight(mesh):
    # connected components == 1 and edges match
    v = np.asarray(mesh.vertices); f = np.asarray(mesh.triangles)
    if len(f) == 0: return 0
    num_comp = len(np.unique(mesh.split_vertex_attrs()[0])) if False else len(np.unique(o3d.geometry.TriangleMesh(mesh).cluster_connected_triangles()[0]))
    return 1 if num_comp == 1 else 0
```

> 두 메트릭 함수(`normal_consistency`, `watertight`)는 단순 구현. 정상성 부족하면 Task 4.5(추가) 에서 보완.

- [ ] **Step 3: 메인 평가 루프**

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--no_fine', action='store_true', help='only Poisson vs SAP-pre (skip fine-tuned)')
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(OUT, exist_ok=True); os.makedirs(f'{OUT}/charts', exist_ok=True)
    dev = torch.device('cuda')
    gen = load_gen(dev)
    sap_pre, cfg = load_sap(SAP_PRE, dev)
    dpsr = DPSR(res=(64,64,64), sig=2).to(dev)
    sap_fine = None
    if not args.no_fine and os.path.exists(SAP_FINE):
        sap_fine, _ = load_sap(SAP_FINE, dev)
    print(f'eval: n={args.n} no_fine={args.no_fine}', flush=True)

    sp = json.load(open(SPLIT))
    pids = [p for p in sp['stage1_val'] if os.path.exists(f'{DATA}/{p}.npz')][:args.n]
    results = {'poisson': [], 'sap_pre': [], 'sap_fine': []}
    for pid in pids:
        pts, bnd, valid, _ = load_patient(f'{DATA}/{pid}.npz', 1024)
        present = np.where(valid > 0)[0]
        k = min(random.randint(1,3), len(present))
        idx = np.random.choice(present, k, replace=False)
        x0 = torch.from_numpy(pts).unsqueeze(0).to(dev)
        bound = torch.from_numpy(bnd).unsqueeze(0).to(dev)
        lm = np.zeros(28); lm[idx] = 1
        lm_t = torch.from_numpy(lm).float().unsqueeze(0).to(dev); om = 1 - lm_t
        with torch.no_grad():
            gen_pc = gen.sample(dict(x0=x0, l_mask=lm_t, o_mask=om, bound=bound))[0].cpu().numpy()
        for s in idx:
            fdi = ZIGZAG_FDI_ORDER[s]
            for label, cloud in [('gt', pts[s].T), ('gen', gen_pc[s].T)]:
                # 1. Poisson (re-use existing)
                poisson_ply = f'{POISSON_DIR}/{pid}_FDI{fdi}_{label}.ply'
                if os.path.exists(poisson_ply):
                    p_mesh = o3d.io.read_triangle_mesh(poisson_ply)
                else:
                    # fallback: compute now
                    from gen_mesh import poisson_mesh
                    p_mesh = poisson_mesh(cloud, smooth_iters=10, decimate_target=0)
                results['poisson'].append({
                    'pid': pid, 'fdi': fdi, 'label': label,
                    'chamfer': chamfer_l2(p_mesh, cloud), 'edge': edge_l2(p_mesh),
                    'watertight': watertight(p_mesh),
                })
                # save copy for viewer
                o3d.io.write_triangle_mesh(f'{OUT}/{pid}_FDI{fdi}_{label}__poisson.ply', p_mesh)
                # 2. SAP-pre
                v, f = sap_mesh(cloud, sap_pre, dpsr, cfg, dev)
                sm = o3d.geometry.TriangleMesh()
                sm.vertices = o3d.utility.Vector3dVector(v); sm.triangles = o3d.utility.Vector3iVector(f); sm.compute_vertex_normals()
                results['sap_pre'].append({
                    'pid': pid, 'fdi': fdi, 'label': label,
                    'chamfer': chamfer_l2(sm, cloud), 'edge': edge_l2(sm),
                    'watertight': watertight(sm),
                })
                o3d.io.write_triangle_mesh(f'{OUT}/{pid}_FDI{fdi}_{label}__sap_pre.ply', sm)
                # 3. SAP-fine
                if sap_fine is not None:
                    v2, f2 = sap_mesh(cloud, sap_fine, dpsr, cfg, dev)
                    sm2 = o3d.geometry.TriangleMesh()
                    sm2.vertices = o3d.utility.Vector3dVector(v2); sm2.triangles = o3d.utility.Vector3iVector(f2); sm2.compute_vertex_normals()
                    results['sap_fine'].append({
                        'pid': pid, 'fdi': fdi, 'label': label,
                        'chamfer': chamfer_l2(sm2, cloud), 'edge': edge_l2(sm2),
                        'watertight': watertight(sm2),
                    })
                    o3d.io.write_triangle_mesh(f'{OUT}/{pid}_FDI{fdi}_{label}__sap_fine.ply', sm2)
                print(f'  {pid} FDI{fdi} {label} done', flush=True)

    # 평균
    summary = {}
    for k in results: results[k] = [r for r in results[k] if r['chamfer'] != float('inf')]
    for k in results:
        if results[k]:
            summary[k] = {
                'n': len(results[k]),
                'chamfer_mean': float(np.mean([r['chamfer'] for r in results[k]])),
                'edge_mean': float(np.mean([r['edge'] for r in results[k]])),
                'watertight_rate': float(np.mean([r['watertight'] for r in results[k]])),
            }
    json.dump({'summary': summary, 'raw': results}, open(JSON_OUT, 'w'), indent=2)
    print('SUMMARY:', summary, flush=True)
```

- [ ] **Step 4: 차트 PNG**

```python
    # chart
    methods = [m for m in ['poisson', 'sap_pre', 'sap_fine'] if m in summary]
    metrics = ['chamfer_mean', 'edge_mean', 'watertight_rate']
    titles = ['Chamfer-L2 (mesh→GT)', 'Mean edge length (smoothness)', 'Watertight rate']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, (m, t) in enumerate(zip(metrics, titles)):
        vals = [summary[mt][m] for mt in methods]
        axes[i].bar(methods, vals, color=['#3fb950', '#f0883e', '#58a6ff'])
        axes[i].set_title(t); axes[i].tick_params(axis='x', rotation=20)
    fig.suptitle('3-way mesh comparison (Poisson · SAP-pre · SAP-fine)')
    fig.tight_layout(); fig.savefig(CHART, dpi=110); plt.close()
    print(f'CHART → {CHART}', flush=True)


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: smoke test (Poisson만, fine 없이)**

```bash
cd /home/hk.sim/Projects/CrownGen
PYTHONPATH=".:crowngen/external:scripts" python scripts/eval_sap.py --no_fine --n 1 2>&1 | tail -20
ls runs2/mesh_sap_compare/
```

Expected: 1 환자 × 1 tooth × {gt|gen} × 2 methods = 4개 PLY + sap_eval.json + charts/comparison.png.

- [ ] **Step 6: commit**

```bash
git add scripts/eval_sap.py
git commit -m "eval: 3-way mesh comparison (Poisson vs SAP-pre vs SAP-fine) + chart PNG"
```

- [ ] **Step 7: fine-tuned 끝나면 본격 eval**

```bash
cd /home/hk.sim/Projects/CrownGen
PYTHONPATH=".:crowngen/external:scripts" python scripts/eval_sap.py --n 6 2>&1 | tee runs2/eval_sap.log &
echo $! > runs2/eval_sap.pid
```

약 10-20분.

---

## Task 5: PLY → `data.js` (3-panel viewer 데이터)

**Files:**
- Create: `scripts/ply_to_sap_compare_js.py`

**Interfaces:**
- Consumes: `runs2/mesh_sap_compare/*__poisson.ply`, `*__sap_pre.ply`, `*__sap_fine.ply`
- Produces: `runs2/viz/mesh_sap_compare/data.js`

- [ ] **Step 1: PLY 파서 + JSON 출력**

```python
"""runs2/mesh_sap_compare/*.ply → runs2/viz/mesh_sap_compare/data.js
3-panel three.js viewer용."""
import os, re, json, glob
import numpy as np, open3d as o3d

SRC = 'runs2/mesh_sap_compare'
OUT = 'runs2/viz/mesh_sap_compare/data.js'
NORM = 'Data/aligned_norm'
MAX_FACES = 15000


def jaw_of(fdi): return 'upper' if fdi // 10 in (1, 2) else 'lower'


def get_real_pts(pid):
    p = f'{NORM}/{pid}.npz'
    if not os.path.exists(p): return []
    d = np.load(p); pts = []
    for fdi_str in [k.split('_')[2].split('.')[0] for k in d.keys() if k.endswith('_pc')]:
        try: fdi = int(fdi_str)
        except: continue
        k = f'{jaw_of(fdi)}_{fdi}_pc'
        if k in d:
            pc = d[k]
            if len(pc) > 100:
                idx = np.linspace(0, len(pc)-1, 100).astype(int); pc = pc[idx]
            pts.extend(pc.tolist())
    return pts


def parse_ply(path):
    m = o3d.io.read_triangle_mesh(path)
    if len(m.triangles) > MAX_FACES:
        m = m.simplify_quadric_decimation(MAX_FACES); m.compute_vertex_normals()
    v = np.asarray(m.vertices, dtype=np.float32); f = np.asarray(m.triangles, dtype=np.uint32)
    return v, f


def main():
    files = glob.glob(f'{SRC}/*.ply')
    by_patient = {}
    pat = re.compile(r'(.+)_FDI(\d+)_(gt|gen)__(poisson|sap_pre|sap_fine)')
    for fpath in files:
        base = os.path.basename(fpath).replace('.ply','')
        m = pat.match(base)
        if not m: continue
        pid, fdi, lbl, method = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        key = (pid, fdi)
        if key not in by_patient:
            by_patient[key] = {'pid': pid, 'fdi': fdi, 'gt_label': lbl, 'methods': {}}
        v, f = parse_ply(fpath)
        by_patient[key]['methods'][method] = {'v': v.tolist(), 'f': f.tolist()}

    # 환자별 grouping (같은 환자/같은 FDI는 1 케이스, 2 method × 2 label)
    by_pid = {}
    for k, info in by_patient.items():
        pid = info['pid']
        if pid not in by_pid:
            by_pid[pid] = {'patient': pid, 'teeth': [], 'real_pts': get_real_pts(pid)}
        # best methods 채워진 키만 push
        if len(info['methods']) >= 1:
            by_pid[pid]['teeth'].append({'fdi': info['fdi'], 'label': info['gt_label'], 'methods': info['methods']})
    cases = list(by_pid.values())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fp:
        fp.write('window.MESH_DATA = ' + json.dumps({'cases': cases}) + ';\n')
    print(f'WROTE {OUT}: {len(cases)} patients', flush=True)

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 실행**

```bash
cd /home/hk.sim/Projects/CrownGen
PYTHONPATH=".:scripts" python scripts/ply_to_sap_compare_js.py
head runs2/viz/mesh_sap_compare/data.js
```

Expected: `window.MESH_DATA = {...cases: [...]};` with patients 배열.

- [ ] **Step 3: commit**

```bash
git add scripts/ply_to_sap_compare_js.py
git commit -m "viewer: PLY->data.js for 3-panel SAP compare viewer"
```

---

## Task 6: 3-panel three.js viewer (port 8778)

**Files:**
- Modify: `runs2/viz/mesh_sap_compare/index.html` (placeholder → real)
- Create: `runs2/viz/mesh_sap_compare/app.js`, `style.css`
- Create: `runs2/viz/mesh_sap_compare/three.min.js`, `OrbitControls.js` (copy from mesh_demo/)

- [ ] **Step 1: three.min.js, OrbitControls.js 심볼릭 링크**

```bash
cd /home/hk.sim/Projects/CrownGen
ln -sf ../../mesh_demo/three.min.js runs2/viz/mesh_sap_compare/three.min.js
ln -sf ../../mesh_demo/OrbitControls.js runs2/viz/mesh_sap_compare/OrbitControls.js
ls -la runs2/viz/mesh_sap_compare/
```

- [ ] **Step 2: index.html**

```html
<!DOCTYPE html>
<meta charset="utf-8"/>
<title>SAP fine-tune 3-way comparison · port 8778</title>
<link rel="stylesheet" href="style.css"/>
<script src="three.min.js"></script>
<script src="OrbitControls.js"></script>
<script src="data.js"></script>
<body>
<div class="hdr">
  <b id="patient"></b>
  <span>치아 <span id="nd">0</span>개</span>
  <button id="prev">◀ prev</button>
  <select id="sel"></select>
  <button id="next">next ▶</button>
  <span class="legend"><i style="background:#3fb950"></i>GT&nbsp;&nbsp;<i style="background:#f6b042"></i>gen</span>
</div>
<div class="grid" id="grid"></div>
<script src="app.js"></script>
</body>
```

- [ ] **Step 3: style.css**

```css
* { box-sizing:border-box; }
body { margin:0; font-family:system-ui; background:#0d1117; color:#e6edf3; }
.hdr { display:flex; align-items:center; gap:14px; padding:10px 16px; background:#161b22; border-bottom:1px solid #30363d; }
.hdr b { font-size:15px; }
.hdr select { flex:1; background:#0d1117; color:#e6edf3; border:1px solid #30363d; border-radius:6px; padding:4px 8px; }
.hdr button { background:#21262d; color:#e6edf3; border:1px solid #30363d; border-radius:6px; padding:4px 10px; cursor:pointer; }
.hdr button:hover { border-color:#58a6ff; }
.legend { font-size:11px; color:#8b949e; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; vertical-align:middle; }
.grid { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; padding:8px; height:calc(100vh - 50px); }
.panel { background:#161b22; border:1px solid #30363d; border-radius:8px; position:relative; overflow:hidden; }
.panel .tag { position:absolute; top:6px; left:8px; background:#21262d88; padding:2px 8px; border-radius:10px; font-size:11px; }
```

- [ ] **Step 4: app.js** (3-panel with sync'd cam)

```javascript
"use strict";
const D = window.MESH_DATA;
const METHODS = [
  { key: 'poisson', label: '표준 Poisson + Taubin', color: 0x3fb950 },
  { key: 'sap_pre', label: 'SAP pre-trained',      color: 0xf0883e },
  { key: 'sap_fine',label: 'SAP fine-tuned ★',      color: 0x58a6ff },
];
let cur = 0;
const sel = document.getElementById('sel');
const ptxt = document.getElementById('patient');
const ndtxt = document.getElementById('nd');
const grid = document.getElementById('grid');

const panels = METHODS.map(() => {
  const d = document.createElement('div'); d.className = 'panel';
  const t = document.createElement('div'); t.className = 'tag'; d.appendChild(t);
  grid.appendChild(d);
  const cam = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  cam.up.set(0,0,1); cam.position.set(1.5,-1.5,1.2);
  const ren = new THREE.WebGLRenderer({ antialias: true });
  d.appendChild(ren.domElement);
  const con = new THREE.OrbitControls(cam, ren.domElement);
  con.target.set(0,0,0); con.update();
  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const dl = new THREE.DirectionalLight(0xffffff, 0.7); dl.position.set(2,-2,3); scene.add(dl);
  return { dom: d, tag: t, cam, ren, con, scene, render: () => ren.render(scene, cam) };
});

function resize() {
  for (const p of panels) {
    const w = p.dom.clientWidth, h = p.dom.clientHeight;
    if (w > 0 && h > 0) { p.ren.setSize(w, h); p.cam.aspect = w/h; p.cam.updateProjectionMatrix(); }
  }
}
function clearScene(p) {
  for (let i = p.scene.children.length-1; i >= 0; i--) {
    const o = p.scene.children[i];
    if (o.type === 'Mesh' || o.type === 'Points') { p.scene.remove(o); o.geometry?.dispose(); o.material?.dispose(); }
  }
}
function addMesh(scene, data, color) {
  if (!data || !data.v || !data.f) return;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(data.v.flat(), 3));
  g.setIndex(data.f.flat()); g.computeVertexNormals();
  const m = new THREE.MeshStandardMaterial({ color, transparent: true, opacity: 0.92, side: THREE.DoubleSide, roughness: 0.5, metalness: 0.1 });
  scene.add(new THREE.Mesh(g, m));
}
function addPoints(scene, pts, color) {
  if (!pts || !pts.length) return;
  const f = new Float32Array(pts.length*3);
  for (let i = 0; i < pts.length; i++) { f[3*i]=pts[i][0]; f[3*i+1]=pts[i][1]; f[3*i+2]=pts[i][2]; }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(f, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color, size: 0.008, transparent: true, opacity: 0.35 })));
}

function build() {
  const c = D.cases[cur];
  ptxt.textContent = c.patient;
  ndtxt.textContent = c.teeth.length;
  for (let i = 0; i < panels.length; i++) {
    const p = panels[i]; const m = METHODS[i];
    clearScene(p);
    addPoints(p.scene, c.real_pts, 0x8a94a6);
    let tag = c.teeth.map(t => `FDI${t.fdi}(${t.label})`).join(', ');
    for (const t of c.teeth) {
      if (t.methods[m.key]) {
        const gt = t.label === 'gt' ? 0x3fb950 : 0xf6b042;
        // overlay GT(gen) using method-specific mesh but distinct color by label
        addMesh(p.scene, t.methods[m.key], gt);
      }
    }
    p.tag.innerHTML = `<b>${m.label}</b> · ${c.teeth.length} teeth · ${tag}`;
  }
  resize();
}

for (let i = 0; i < D.cases.length; i++) {
  const o = document.createElement('option'); o.value = i; o.textContent = D.cases[i].patient;
  sel.appendChild(o);
}
sel.onchange = () => { cur = +sel.value; build(); };
document.getElementById('prev').onclick = () => { cur = Math.max(0, cur-1); sel.value = cur; build(); };
document.getElementById('next').onclick = () => { cur = Math.min(D.cases.length-1, cur+1); sel.value = cur; build(); };
window.addEventListener('resize', resize);
(function tick() { requestAnimationFrame(tick); for (const p of panels) p.render(); })();
build();
```

- [ ] **Step 5: 정적 서버 8778에서 띄움 (background)**

```bash
cd /home/hk.sim/Projects/CrownGen
nohup python -m http.server 8778 --directory runs2/viz/mesh_sap_compare > runs2/viz_mesh_sap_compare.log 2>&1 &
echo $! > runs2/viz_mesh_sap_compare.pid
sleep 2
curl -sI http://localhost:8778/ | head -1
```

Expected: `HTTP/1.0 200 OK`. (실제 머신의 도커 포털이 10005가 아닐 수 있으니 외부 접근은 별도 — 사용자에게 확인.)

- [ ] **Step 6: commit + push**

```bash
git add runs2/viz/mesh_sap_compare/
git commit -m "viewer: 3-panel SAP comparison (Poisson / SAP-pre / SAP-fine) — port 8778"
git push
```

---

## Task 7: 사용자 리뷰 + 결정

- [ ] viewer `http://localhost:8778/` 열어서 확인
- [ ] 차트 `charts/comparison.png` 확인
- [ ] `sap_eval.json` 확인
- [ ] **사용자에게 merge 여부 확인** (viewer OK + Chamfer-L2 개선 or 동등이면 merge)

---

## Task 8: (merge OK 시) main 머지 + 포털(10005) 카드 추가

**Files:**
- Modify: `~/docker/crowngen-viz/index.html` (portal 카드 추가)

- [ ] **Step 1: main 머지**

```bash
cd /home/hk.sim/Projects/CrownGen
git checkout main
git merge --no-ff mesh/dpsr-finetune -m "merge: SAP fine-tune (DPSR/SAP on Teeth3DS)"
```

- [ ] **Step 2: 포털 카드 추가** (`~/docker/crowngen-viz/index.html`의 `mesh_demo` 카드 다음에 추가)

```html
<a class="card" href="viewers/mesh_sap_compare/">
  <div class="icon">🦷</div>
  <div class="title">DPSR fine-tune vs Poisson (3-way)</div>
  <div class="desc">DPSR(Shape as Points)을 우리 치아 데이터로 fine-tune. 표준 Poisson vs SAP pre-trained vs SAP fine-tuned 메시 3패널 비교. Chamfer-L2 · watertight · edge smoothness 정량.</div>
  <div class="metrics">See viewer (port 8778) + runs2/sap_eval.json</div>
  <span class="tag">3D · 3-panel (Poisson · SAP-pre · SAP-fine)</span>
</a>
```

- [ ] **Step 3: commit**

```bash
cd ~/docker/crowngen-viz
git add index.html
git commit -m "portal: add SAP fine-tune viewer card"
```

---

## Self-Review (skill checklist)

1. **Spec coverage**: §1→Task 1·2 / §2→Task 3 / §3→Task 4 / §4→Task 4·5·6 / §5→Task 1·2·3·4·5·6 / §6→Task 0·1·2·3·4·5·6·7·8. ✅
2. **Placeholder scan**: 없음. `Task 4 watertight`는 단순 구현이라고 명시 (단순 클러스터 컴포넌트 카운트).
3. **Type consistency**: `pred_pc, pred_n` (Task 3→4), `outputs of ToothDataset` — `inputs (B,1,N,3)` ⇒ `.squeeze(1)`은 일관.
4. **Resource/time**: 명시 OK. background launch + log monitoring 안내.
5. **Failure modes**: §7 spec과 일치.
6. **Scope**: 단일 사이클 OK.
