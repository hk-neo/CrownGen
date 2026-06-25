"use strict";
// Boundary 예측 side-by-side 3D 뷰어. 단일 카메라 + 단일 OrbitControls(panelsEl) → 모든 패널 동기화 회전.
const DATA = window.BOUNDARY_DATA;
const GT_COLOR = 0x2ecc71;   // GT 실린더 (초록, 반투명)
const CTX_COLOR = 0xbbbbbb;  // 컨텍스트 치아 점 (회색)

const panelsEl = document.getElementById('panels');
const caseSelect = document.getElementById('caseSelect');
const caseInfo = document.getElementById('caseInfo');
const metricsEl = document.getElementById('metrics');
const paramTableEl = document.getElementById('paramTable');

let activeRuns = DATA.runs.map(r => r.id);           // 기본: 전체 런
let currentCase = 0;
let visibleTeeth = null;                              // null = 결손 치아 전부 표시

// 평균 Dice(전 케이스) — metrics 요약용
const MEAN_DICE = {};
for (const r of DATA.runs) {
  const ds = DATA.cases.map(c => c.predictions[r.id].dice);
  MEAN_DICE[r.id] = ds.reduce((a, b) => a + b, 0) / ds.length;
}

// --- 공유 카메라 + 단일 OrbitControls ---
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
camera.up.set(0, 0, 1);                               // 데이터 프레임: z=교합면 법선(위쪽)
camera.position.set(2.6, -2.6, 2.1);
const controls = new THREE.OrbitControls(camera, panelsEl);
controls.target.set(0, 0, 0);
controls.update();

let panels = []; // [{renderer, scene, run, div}]

function clearScene(scene) {
  for (let i = scene.children.length - 1; i >= 0; i--) {
    const o = scene.children[i];
    scene.remove(o);
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
  }
}

function addCylinder(scene, cyl, color, opacity, wire) {
  if (!cyl) return;
  const [cx, cy, cz, h, r] = cyl;
  if (![cx, cy, cz, h, r].every(Number.isFinite) || r <= 0 || h <= 0) return;
  const geo = new THREE.CylinderGeometry(Math.max(r, 1e-4), Math.max(r, 1e-4), Math.max(h, 1e-4), 36, 1, true);
  geo.rotateX(Math.PI / 2);                            // 축 y → z 정렬(데이터 좌표계)
  const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity, wireframe: !!wire, side: THREE.DoubleSide });
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
  const m = new THREE.PointsMaterial({ color, size: 0.013, sizeAttenuation: true, transparent: true, opacity: 0.55 });
  scene.add(new THREE.Points(g, m));
}

function toothVisible(slot) { return visibleTeeth === null || visibleTeeth.has(slot); }

function buildPanelScene(p, c) {
  clearScene(p.scene);
  addPoints(p.scene, c.context_pts, CTX_COLOR);
  const predColor = new THREE.Color(p.run.color).getHex();
  for (const slot of c.missing) {
    if (!toothVisible(slot)) continue;
    addCylinder(p.scene, c.gt[slot], GT_COLOR, 0.28, false);                 // GT: 초록 반투명
    const pred = c.predictions[p.run.id].cyl[slot];
    addCylinder(p.scene, pred, predColor, 0.55, false);                       // 예측: 런 색, 반투명
    addCylinder(p.scene, pred, predColor, 1.0, true);                         // 예측: 와이어프레임 윤곽
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
    title.style.color = run.color;
    title.id = 'ptitle-' + run.id;
    div.appendChild(title);
    panelsEl.appendChild(div);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(div.clientWidth || 200, div.clientHeight || 200);
    div.appendChild(renderer.domElement);
    panels.push({ renderer, scene: new THREE.Scene(), run, div, title });
  }
  const c = DATA.cases[currentCase];
  for (const p of panels) buildPanelScene(p, c);
  updatePanelTitles(c);
  renderAll();
}

function updatePanelTitles(c) {
  for (const p of panels) {
    const pr = c.predictions[p.run.id];
    p.title.innerHTML = `<span style="color:${p.run.color}">●</span> ${p.run.label}<br>
      <span class="cur">Dice ${pr.dice.toFixed(3)}</span> <span class="logged">(로그 ${p.run.logged_dice?.toFixed(3) ?? '-'})</span>`;
  }
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
  for (const r of DATA.runs) {
    const on = activeRuns.includes(r.id);
    const pr = c.predictions[r.id];
    const el = document.createElement('div');
    el.className = 'metric';
    el.style.opacity = on ? '1' : '0.4';
    el.innerHTML = `<b style="color:${r.color}">${r.label}</b>
      <span class="cur"> · 이 케이스 Dice ${pr.dice.toFixed(3)} / IoU ${pr.iou.toFixed(3)}</span><br>
      <span class="logged">전체 평균 ${MEAN_DICE[r.id].toFixed(3)} · 로그 best ${r.logged_dice?.toFixed(3) ?? '-'}</span>`;
    metricsEl.appendChild(el);
  }
}

function updateParamTable() {
  const c = DATA.cases[currentCase];
  const runs = DATA.runs.filter(r => activeRuns.includes(r.id));
  const head = ['tooth(FDI)', ...runs.map(r => r.label + ' 예측'), 'GT'];
  const rows = [head];
  for (const slot of c.missing) {
    if (!toothVisible(slot)) continue;
    const row = ['FDI ' + DATA.fdi_order[slot]];
    for (const r of runs) {
      const v = c.predictions[r.id].cyl[slot];
      row.push(v ? v.map(x => x.toFixed(2)).join(',') : 'N/A');
    }
    row.push(c.gt[slot] ? c.gt[slot].map(x => x.toFixed(2)).join(',') : 'N/A');
    rows.push(row);
  }
  paramTableEl.innerHTML = '<table>' + rows.map((rr, i) =>
    '<tr>' + rr.map(cc => (i === 0 ? '<th>' : '<td>') + cc + (i === 0 ? '</th>' : '</td>')).join('') + '</tr>').join('') + '</table>';
}

function buildToothToggles() {
  const tt = document.getElementById('toothToggles');
  tt.innerHTML = '';
  const c = DATA.cases[currentCase];
  const all = document.createElement('button');
  all.textContent = '전체';
  all.className = visibleTeeth === null ? 'on' : '';
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

function loadCase() {
  const c = DATA.cases[currentCase];
  caseInfo.textContent = `(${currentCase + 1}/${DATA.cases.length}) · 결손: ${c.missing.map(s => 'FDI ' + DATA.fdi_order[s]).join(', ')}`;
  caseSelect.value = String(currentCase);
  buildToothToggles();
  for (const p of panels) buildPanelScene(p, c);
  updatePanelTitles(c);
  updateMetrics();
  updateParamTable();
  renderAll();
}

function buildControls() {
  for (let i = 0; i < DATA.cases.length; i++) {
    const o = document.createElement('option');
    o.value = String(i); o.textContent = 'case ' + (i + 1);
    caseSelect.appendChild(o);
  }
  caseSelect.onchange = () => { currentCase = +caseSelect.value; loadCase(); };
  document.getElementById('prevCase').onclick = () => { currentCase = Math.max(0, currentCase - 1); loadCase(); };
  document.getElementById('nextCase').onclick = () => { currentCase = Math.min(DATA.cases.length - 1, currentCase + 1); loadCase(); };

  const rt = document.getElementById('runToggles');
  for (const r of DATA.runs) {
    const b = document.createElement('button');
    b.textContent = r.label;
    b.className = 'on';
    b.style.borderColor = r.color;
    b.onclick = () => {
      if (activeRuns.includes(r.id)) {
        if (activeRuns.length === 1) return;            // 최소 1개는 유지
        activeRuns = activeRuns.filter(x => x !== r.id);
        b.classList.remove('on');
      } else {
        activeRuns.push(r.id);
        b.classList.add('on');
      }
      rebuildPanels();
      loadCase();
    };
    rt.appendChild(b);
  }
}

controls.addEventListener('change', renderAll);
window.addEventListener('resize', renderAll);
buildControls();
rebuildPanels();
loadCase();
