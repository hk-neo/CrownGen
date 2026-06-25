# Boundary 3D Side-by-Side 뷰어 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Boundary 예측 모델(official/voxatt/context)의 val 예측을 side-by-side 동기화 3D로 보여주는 정적 웹 페이지를 만든다.

**Architecture:** (1) Python 스크립트가 각 체크포인트를 로드해 고정 마스킹으로 val 케이스별 예측 + per-case Dice/IoU를 계산해 `data.js`로 덤프. (2) Three.js 정적 페이지가 `data.js`를 로드해 런별 패널을 side-by-side로 렌더. 단일 카메라 + 단일 OrbitControls(컨테이너에 바인딩)로 모든 패널을 동기화 회전.

**Tech Stack:** Python 3.10 / PyTorch + CUDA(데이터 준비, 1회), Three.js r128(전역 `THREE` + `THREE.OrbitControls`, 로컬 벤더링), 바닐라 JS/HTML/CSS.

**참고 — git:** 이 프로젝트는 git 저장소가 아님. 따라서 commit 스텝은 생략하고 각 태스크 끝에 검증 스텝으로 대체.

**선행 확인(완료):** 인터넷 OK(cdnjs 200), `crowngen.data.fdi.ZIGZAG_FDI_ORDER` 존재, `BoundEncoder(output_dim=5, dropout=0.3, max_missing_teeth=6, mask_mode='official', voxel_attention=False)`, 데이터 키 `upper_11_pc`/`upper_11_bound` 형식.

---

## 파일 구조

- **Create** `scripts/viz_boundary_web_data.py` — 체크포인트 로드 → 고정 마스킹 예측 → per-case Dice/IoU → `data.js` 덤프. 단일 책임: 데이터 생성.
- **Create** `runs2/viz/web/three.min.js` — Three.js r128 UMD 빌드(로컬 벤더링, 오프라인 보장).
- **Create** `runs2/viz/web/OrbitControls.js` — r128용 OrbitControls(전역 `THREE.OrbitControls`).
- **Create** `runs2/viz/web/data.js` — 생성 산출물(`window.BOUNDARY_DATA = {...}`).
- **Create** `runs2/viz/web/index.html` — 페이지 골격 + 상단/하단 바.
- **Create** `runs2/viz/web/style.css` — 레이아웃.
- **Create** `runs2/viz/web/app.js` — Three.js 씬/카메라/패널/네비게이션/메트릭.

---

## Task 1: Three.js + OrbitControls 로컬 벤더링

**Files:**
- Create: `runs2/viz/web/three.min.js`
- Create: `runs2/viz/web/OrbitControls.js`

- [ ] **Step 1: 디렉토리 생성 + r128 다운로드**

```bash
mkdir -p runs2/viz/web
curl -sL https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js -o runs2/viz/web/three.min.js
curl -sL https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js -o runs2/viz/web/OrbitControls.js
```

- [ ] **Step 2: 다운로드 검증**

```bash
ls -la runs2/viz/web/three.min.js runs2/viz/web/OrbitControls.js
head -c 80 runs2/viz/web/three.min.js; echo
grep -c "OrbitControls" runs2/viz/web/OrbitControls.js
```
Expected: 두 파일 모두 수십~수백 KB, `three.min.js` 첫 토큰에 `three`/`!function`, OrbitControls.js grep 결과 ≥1.

---

## Task 2: 데이터 준비 스크립트 작성

**Files:**
- Create: `scripts/viz_boundary_web_data.py`

- [ ] **Step 1: 스크립트 작성** (아래 전체 코드 그대로 생성)

```python
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
    {"id": "official", "label": "official", "ckpt": "runs2/boundary_official_long.pt",
     "mask_mode": "official", "voxel_attention": False, "color": "#e74c3c", "final_dice": 0.770},
    {"id": "voxatt",   "label": "voxatt",   "ckpt": "runs2/boundary_voxatt.pt",
     "mask_mode": "official", "voxel_attention": True,  "color": "#9b59b6", "final_dice": 0.507},
    {"id": "context",  "label": "context",  "ckpt": "runs2/boundary_context.pt",
     "mask_mode": "context",  "voxel_attention": False, "color": "#3498db", "final_dice": 0.756},
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
```

- [ ] **Step 2: 문법 검증**

```bash
python3 -c "import ast; ast.parse(open('scripts/viz_boundary_web_data.py').read()); print('syntax OK')"
```
Expected: `syntax OK`

---

## Task 3: 데이터 준비 실행 → data.js 생성 + 검증

**Files:**
- Create: `runs2/viz/web/data.js` (스크립트가 생성)

- [ ] **Step 1: 스크립트 실행 (GPU, 수십 초)**

```bash
python3 scripts/viz_boundary_web_data.py --n_cases 40 2>&1 | tee runs2/viz/web/_data_prep.log
```
Expected: `loaded official/voxatt/context` 각각 출력 → `case 1/40 ... case 40/40` → `ASSERT OK: 40 cases × 3 runs` → `WROTE runs2/viz/web/data.js (XX KB)`.

- [ ] **Step 2: 평균 Dice가 로그 FINAL 값과 대략 일치 확인**

```bash
python3 -c "
import json
d=json.load(open('runs2/viz/web/data.js').replace('window.BOUNDARY_DATA = ','').replace(';\n',''))
import numpy as np
for r in d['runs']:
    ds=[c['predictions'][r['id']]['dice'] for c in d['cases']]
    print(f\"{r['id']:9s} mean Dice {np.mean(ds):.3f} (final_dice {r['final_dice']:.3f})\")
"
```
Expected: official ≈0.7~0.8, voxatt ≈0.4~0.55, context ≈0.7~0.8 (FINAL 0.770/0.507/0.756 근방; 고정 마스킹이라 약간 달라도 추세 일치).

---

## Task 4: HTML + CSS 골격

**Files:**
- Create: `runs2/viz/web/index.html`
- Create: `runs2/viz/web/style.css`

- [ ] **Step 1: index.html 작성**

```html
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<title>CrownGen · Boundary 3D Viewer</title>
<link rel="stylesheet" href="style.css"/>
</head>
<body>
<header>
  <h1>CrownGen · Boundary 예측 뷰어</h1>
  <div class="nav">
    <button id="prevCase">◀</button>
    <select id="caseSelect"></select>
    <span id="caseInfo"></span>
    <button id="nextCase">▶</button>
  </div>
  <div id="runToggles" class="toggles"></div>
  <div id="toothToggles" class="toggles"></div>
</header>
<main id="panels"></main>
<footer>
  <div id="metrics"></div>
  <div id="paramTable"></div>
</footer>
<script src="three.min.js"></script>
<script src="OrbitControls.js"></script>
<script src="data.js"></script>
<script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: style.css 작성**

```css
* { box-sizing: border-box; }
body { margin:0; font-family: system-ui, sans-serif; background:#1b1f23; color:#e6edf3; }
header { padding:10px 14px; background:#0d1117; border-bottom:1px solid #30363d; }
header h1 { font-size:16px; margin:0 0 8px; }
.nav { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
.toggles { display:flex; flex-wrap:wrap; gap:6px; margin:4px 0; }
.toggles button { font-size:12px; padding:3px 8px; border:1px solid #30363d; border-radius:4px;
  background:#21262d; color:#e6edf3; cursor:pointer; }
.toggles button.on { background:#2f6f43; border-color:#3fb950; }
main#panels { display:flex; height:58vh; }
.panel { flex:1; position:relative; border-right:1px solid #30363d; min-width:0; }
.panel:last-child { border-right:none; }
.panel canvas { display:block; width:100%; height:100%; }
.panel .ptitle { position:absolute; top:6px; left:8px; font-size:13px; font-weight:600;
  background:rgba(0,0,0,.5); padding:2px 6px; border-radius:4px; }
footer { padding:10px 14px; background:#0d1117; border-top:1px solid #30363d; font-size:12px; }
#metrics { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:8px; }
.metric { padding:4px 8px; border-radius:4px; background:#21262d; }
table { border-collapse:collapse; width:100%; }
th,td { border:1px solid #30363d; padding:3px 6px; text-align:right; font-family:ui-monospace,monospace; }
th { background:#161b22; }
```

- [ ] **Step 3: 파일 존재 확인**

```bash
ls -la runs2/viz/web/index.html runs2/viz/web/style.css
```
Expected: 두 파일 존재.

---

## Task 5: app.js 핵심 — 데이터 로드 + 단일 패널 렌더

**Files:**
- Create: `runs2/viz/web/app.js`

- [ ] **Step 1: app.js 작성 (아래 전체 코드)**

```javascript
"use strict";
const DATA = window.BOUNDARY_DATA;
const GT_COLOR = 0x2ecc71;
const CTX_COLOR = 0xbbbbbb;

const panelsEl = document.getElementById('panels');
const caseSelect = document.getElementById('caseSelect');
const caseInfo = document.getElementById('caseInfo');
const metricsEl = document.getElementById('metrics');
const paramTableEl = document.getElementById('paramTable');

let activeRuns = DATA.runs.map(r => r.id);
let currentCase = 0;
let visibleTeeth = null; // null = 모든 결손 치아 표시

// --- 공유 카메라 + 단일 OrbitControls(컨테이너 바인딩 → 어디서 드래그해도 동기화) ---
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
camera.up.set(0, 0, 1);
camera.position.set(2.5, -2.5, 2.0);
camera.lookAt(0, 0, 0);
const controls = new THREE.OrbitControls(camera, panelsEl);
controls.addEventListener('change', renderAll);

let panels = []; // [{renderer, scene, run}]

function defaultView() {
  camera.position.set(2.5, -2.5, 2.0);
  controls.target.set(0, 0, 0);
  controls.update();
}

function clearScene(scene) {
  while (scene.children.length > 0) {
    const o = scene.children[0];
    scene.remove(o);
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
  }
}

function addCylinder(scene, cyl, color, opacity, wire) {
  if (!cyl) return;
  const [cx, cy, cz, h, r] = cyl;
  if (![cx, cy, cz, h, r].every(Number.isFinite) || r <= 0 || h <= 0) return;
  const geo = new THREE.CylinderGeometry(Math.max(r, 1e-4), Math.max(r, 1e-4), Math.max(h, 1e-4), 36);
  geo.rotateX(Math.PI / 2); // 축을 y → z 로 정렬(데이터 좌표계)
  const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity, wireframe: !!wire });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(cx, cy, cz);
  scene.add(mesh);
}

function addPoints(scene, points, color) {
  if (!points || !points.length) return;
  const flat = new Float32Array(points.length * 3);
  for (let i = 0; i < points.length; i++) { flat[3*i] = points[i][0]; flat[3*i+1] = points[i][1]; flat[3*i+2] = points[i][2]; }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(flat, 3));
  const m = new THREE.PointsMaterial({ color, size: 0.012, sizeAttenuation: true, transparent: true, opacity: 0.6 });
  scene.add(new THREE.Points(g, m));
}

function toothVisible(slot) { return visibleTeeth === null || visibleTeeth.has(slot); }

function buildPanelScene(p, c) {
  clearScene(p.scene);
  addPoints(p.scene, c.context_pts, CTX_COLOR);
  for (const slot of c.missing) {
    if (!toothVisible(slot)) continue;
    addCylinder(p.scene, c.gt[slot], GT_COLOR, 0.30, false);          // GT: 초록 반투명
    const pred = c.predictions[p.run.id].cyl[slot];
    addCylinder(p.scene, pred, new THREE.Color(p.run.color).getHex(), 0.80, false); // pred: 불투명
    addCylinder(p.scene, pred, new THREE.Color(p.run.color).getHex(), 1.0, true);   // + 와이어
  }
}

function rebuildPanels() {
  for (const p of panels) { p.renderer.dispose(); if (p.renderer.domElement.parentNode) p.renderer.domElement.parentNode.remove(); }
  panels = [];
  panelsEl.innerHTML = '';
  const runs = DATA.runs.filter(r => activeRuns.includes(r.id));
  for (const run of runs) {
    const div = document.createElement('div');
    div.className = 'panel';
    const title = document.createElement('div');
    title.className = 'ptitle';
    title.textContent = run.label;
    title.style.color = run.color;
    div.appendChild(title);
    panelsEl.appendChild(div);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(div.clientWidth, div.clientHeight);
    div.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    panels.push({ renderer, scene, run, div });
  }
  const c = DATA.cases[currentCase];
  for (const p of panels) buildPanelScene(p, c);
  renderAll();
}

function renderAll() {
  for (const p of panels) {
    const w = p.div.clientWidth, h = p.div.clientHeight;
    if (w > 0 && h > 0) {
      p.renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      p.renderer.render(p.scene, camera);
    }
  }
}

function updateMetrics() {
  const c = DATA.cases[currentCase];
  metricsEl.innerHTML = '';
  for (const r of DATA.runs.filter(r => activeRuns.includes(r.id))) {
    const pr = c.predictions[r.id];
    const el = document.createElement('div');
    el.className = 'metric';
    el.innerHTML = `<b style="color:${r.color}">${r.label}</b> · Dice ${pr.dice.toFixed(3)} · IoU ${pr.iou.toFixed(3)}`;
    metricsEl.appendChild(el);
  }
  // 치아별 파라미터 표
  const rows = [['tooth', ...DATA.runs.filter(r=>activeRuns.includes(r.id)).map(r=>r.label+' pred'), 'GT']];
  for (const slot of c.missing) {
    if (!toothVisible(slot)) continue;
    const fdi = DATA.fdi_order[slot];
    const row = ['FDI '+fdi];
    for (const r of DATA.runs.filter(r => activeRuns.includes(r.id))) {
      const v = c.predictions[r.id].cyl[slot];
      row.push(v ? v.map(x=>x.toFixed(2)).join(',') : 'N/A');
    }
    row.push(c.gt[slot] ? c.gt[slot].map(x=>x.toFixed(2)).join(',') : 'N/A');
    rows.push(row);
  }
  paramTableEl.innerHTML = '<table>' + rows.map((rr,i) =>
    '<tr>' + rr.map(cc => (i===0 ? '<th>' : '<td>') + cc + (i===0 ? '</th>' : '</td>')).join('') + '</tr>').join('') + '</table>';
}

function loadCase() {
  const c = DATA.cases[currentCase];
  caseInfo.textContent = `(${currentCase+1}/${DATA.cases.length}) missing: ${c.missing.map(s=>'FDI '+DATA.fdi_order[s]).join(', ')}`;
  caseSelect.value = String(currentCase);
  rebuildPanels();
  updateMetrics();
  defaultView();
}

// --- 상단 바 구성 ---
function buildControls() {
  for (let i = 0; i < DATA.cases.length; i++) {
    const o = document.createElement('option');
    o.value = String(i); o.textContent = `case ${i+1}`;
    caseSelect.appendChild(o);
  }
  caseSelect.onchange = () => { currentCase = +caseSelect.value; loadCase(); };
  document.getElementById('prevCase').onclick = () => { currentCase = Math.max(0, currentCase-1); loadCase(); };
  document.getElementById('nextCase').onclick = () => { currentCase = Math.min(DATA.cases.length-1, currentCase+1); loadCase(); };

  const rt = document.getElementById('runToggles');
  for (const r of DATA.runs) {
    const b = document.createElement('button');
    b.textContent = r.label; b.className = 'on'; b.dataset.id = r.id;
    b.style.borderColor = r.color;
    b.onclick = () => {
      if (activeRuns.includes(r.id)) activeRuns = activeRuns.filter(x => x !== r.id);
      else { activeRuns.push(r.id); }
      b.classList.toggle('on');
      loadCase();
    };
    rt.appendChild(b);
  }
}

function buildToothToggles() {
  const tt = document.getElementById('toothToggles');
  tt.innerHTML = '';
  const c = DATA.cases[currentCase];
  const all = document.createElement('button');
  all.textContent = '전체'; all.className = visibleTeeth === null ? 'on' : '';
  all.onclick = () => { visibleTeeth = null; loadCase(); };
  tt.appendChild(all);
  for (const slot of c.missing) {
    const b = document.createElement('button');
    b.textContent = 'FDI ' + DATA.fdi_order[slot];
    b.className = (visibleTeeth === null || visibleTeeth.has(slot)) ? 'on' : '';
    b.onclick = () => {
      if (visibleTeeth === null) visibleTeeth = new Set(c.missing);
      if (visibleTeeth.has(slot)) visibleTeeth.delete(slot); else visibleTeeth.add(slot);
      loadCase();
    };
    tt.appendChild(b);
  }
}

const _origLoadCase = loadCase;
loadCase = function() { _origLoadCase(); buildToothToggles(); };

window.addEventListener('resize', renderAll);
buildControls();
loadCase();
```

- [ ] **Step 2: 문법 검증 (node)**

```bash
node --check runs2/viz/web/app.js && echo "syntax OK"
```
Expected: `syntax OK` (node 없으면 스킵, 브라우저에서 검증).

---

## Task 6: 브라우저 수동 검증

- [ ] **Step 1: 페이지 열기 (로컬 서버 — file:// 도 동작하지만 안정적인 http 서버 권장)**

```bash
cd runs2/viz/web && python3 -m http.server 8765
```
브라우저에서 `http://localhost:8765/` 열기.

- [ ] **Step 2: 검증 체크리스트**
  - [ ] 3개 패널(official/voxatt/context)이 side-by-side로 뜬다. 각 패널에 회색 컨텍스트 점 + 초록 GT 실린더 + 런 색 예측 실린더.
  - [ ] 한 패널을 드래그하면 3개 패널이 같이 회전(동기화).
  - [ ] ◀ ▶ / 드롭다운으로 케이스 전환 시 패널·메트릭·테이블 갱신.
  - [ ] 상단 런 토글 끄면 해당 패널 사라지고 재배치.
  - [ ] 하단 Dice가 official≈0.7~0.8, voxatt≈0.4~0.55, context≈0.7~0.8 추세.
  - [ ] 치아 토글로 특정 슬롯만 격리 가능.

- [ ] **Step 3: (수정 필요 시) app.js 편집 후 브라우저 새로고침.**

---

## Self-Review (작성자 점검 — 완료)

1. **Spec coverage**: 스펙의 모든 섹션 대응 — 데이터 준비(Task 2-3), 정적 웹 구조(Task 4-5), 동기화 카메라(Task 5 `controls` on `panelsEl`), 런 토글/케이스 네비/치아 토글(Task 5), per-case Dice/IoU + 파라미터 표(Task 5 `updateMetrics`), 에러처리(Task 2 ckpt 스킵 + `addCylinder` finite 체크), 검증(Task 3 assertion + Task 6 수동). ✅
2. **Placeholder scan**: TBD/TODO/"appropriate handling" 없음. 모든 코드 블록 완결. ✅
3. **Type consistency**: `c.predictions[id].cyl[slot]` / `.dice` / `.iou`, `c.gt[slot]`, `c.context_pts`, `c.missing`, `r.color`, `DATA.fdi_order[slot]` 가 Task 2 덤프 스키마와 일치. ✅
