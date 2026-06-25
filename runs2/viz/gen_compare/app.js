"use strict";
// gen2k vs gen2k_nobound 생성 크라운 비교. 단일 카메라 + OrbitControls(panelsEl) → 양 패널 동기화 회전.
const DATA = window.GEN_COMPARE_DATA;
const GT_COLOR = 0x2ecc71;    // GT 크라운 (초록)
const CTX_COLOR = 0xf0f0f0;    // 컨텍스트 치아 (흰색)

const panelsEl = document.getElementById('panels');
const caseSelect = document.getElementById('caseSelect');
const caseInfo = document.getElementById('caseInfo');
const metricsEl = document.getElementById('metrics');
const paramTableEl = document.getElementById('paramTable');

let activeModels = DATA.models.map(m => m.id);
let currentCase = 0;
let visibleTeeth = null;
let zFlip = false;                       // 전체 z축 반전(위아래) 토글

const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
camera.up.set(0, 0, 1);
camera.position.set(3.0, -3.0, 2.4);
const controls = new THREE.OrbitControls(camera, panelsEl);
controls.target.set(0, 0, 0);
controls.update();

let panels = [];

function clearScene(scene) {
  for (let i = scene.children.length - 1; i >= 0; i--) {
    const o = scene.children[i];
    scene.remove(o);
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
  }
}

function addPoints(scene, points, color, size, opacity) {
  if (!points || !points.length) return;
  const flat = new Float32Array(points.length * 3);
  for (let i = 0; i < points.length; i++) { flat[3*i]=points[i][0]; flat[3*i+1]=points[i][1]; flat[3*i+2]=points[i][2]; }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(flat, 3));
  const m = new THREE.PointsMaterial({ color, size, sizeAttenuation: true, transparent: true, opacity });
  scene.add(new THREE.Points(g, m));
}

function toothVisible(t) { return visibleTeeth === null || visibleTeeth.has(t); }

function buildPanelScene(p, c) {
  clearScene(p.scene);
  p.scene.scale.z = zFlip ? -1 : 1;      // 위아래 반전 적용
  addPoints(p.scene, c.context_pts, CTX_COLOR, 0.010, 0.6);
  const mcolor = new THREE.Color(p.model.color).getHex();
  for (const t of c.targets) {
    if (!toothVisible(t)) continue;
    addPoints(p.scene, c.gt[t], GT_COLOR, 0.013, 0.9);                  // GT
    addPoints(p.scene, c.gen[p.model.id][t].pts, mcolor, 0.013, 0.95);  // 생성
  }
}

function caseMeanCd(c, modelId) {
  const cds = c.targets.map(t => c.gen[modelId][t].cd);
  return cds.reduce((a, b) => a + b, 0) / cds.length;
}

function updatePanelTitles(c) {
  for (const p of panels) {
    const mean = caseMeanCd(c, p.model.id);
    p.title.innerHTML = `<span style="color:${p.model.color}">●</span> ${p.model.label}<br>CD×10³ <b>${mean.toFixed(1)}</b>`;
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

function rebuildPanels() {
  for (const p of panels) { p.renderer.dispose(); if (p.renderer.domElement.parentNode) p.renderer.domElement.parentNode.remove(); }
  panels = [];
  panelsEl.innerHTML = '';
  const models = DATA.models.filter(m => activeModels.includes(m.id));
  for (const model of models) {
    const div = document.createElement('div');
    div.className = 'panel';
    const title = document.createElement('div');
    title.className = 'ptitle';
    div.appendChild(title);
    panelsEl.appendChild(div);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(div.clientWidth || 300, div.clientHeight || 300);
    div.appendChild(renderer.domElement);
    panels.push({ renderer, scene: new THREE.Scene(), model, div, title });
  }
  const c = DATA.cases[currentCase];
  for (const p of panels) buildPanelScene(p, c);
  updatePanelTitles(c);
  renderAll();
}

function updateMetrics() {
  const c = DATA.cases[currentCase];
  metricsEl.innerHTML = '';
  // 이 케이스에서 누가 이겼는지
  const means = DATA.models.map(m => ({ id: m.id, label: m.label, color: m.color, cd: caseMeanCd(c, m.id) }));
  const best = Math.min(...means.map(m => m.cd));
  for (const m of means) {
    const el = document.createElement('div');
    el.className = 'metric' + (m.cd === best && means.length > 1 ? ' win' : '');
    el.innerHTML = `<b style="color:${m.color}">${m.label}</b><br><span class="cur">이 케이스 CD×10³ <b>${m.cd.toFixed(1)}</b></span>`;
    metricsEl.appendChild(el);
  }
  const diff = means[0].cd - means[1].cd;
  const el = document.createElement('div');
  el.className = 'metric';
  el.innerHTML = `차이(gen2k − nobound)<br><b>${diff >= 0 ? '+' : ''}${diff.toFixed(1)}</b> (음=gen2k 우위)`;
  metricsEl.appendChild(el);
}

function updateParamTable() {
  const c = DATA.cases[currentCase];
  const models = DATA.models.filter(m => activeModels.includes(m.id));
  const head = ['타겟(FDI)', ...models.map(m => m.label)];
  const rows = [head];
  for (const t of c.targets) {
    if (!toothVisible(t)) continue;
    const cds = models.map(m => c.gen[m.id][t].cd);
    const best = Math.min(...cds);
    const row = ['FDI ' + DATA.fdi_order[t]];
    cds.forEach(cd => row.push(cd.toFixed(1)));
    rows.push(row);
  }
  paramTableEl.innerHTML = '<table>' + rows.map((rr, i) => {
    const isHead = i === 0;
    return '<tr>' + rr.map((cc, j) => {
      const tag = isHead ? 'th' : 'td';
      // 데이터 행에서 최소 CD 셀 강조 (j>=1)
      const cls = (!isHead && j >= 1) ? ` class="${(+cc === Math.min(...rows[i].slice(1).map(Number)) && models.length > 1) ? 'win-cell' : ''}"` : '';
      return `<${tag}${cls}>${cc}</${tag}>`;
    }).join('') + '</tr>';
  }).join('') + '</table>';
}

function buildToothToggles() {
  const tt = document.getElementById('toothToggles');
  tt.innerHTML = '';
  const c = DATA.cases[currentCase];
  const all = document.createElement('button');
  all.textContent = '전체'; all.className = visibleTeeth === null ? 'on' : '';
  all.onclick = () => { visibleTeeth = null; loadCase(); };
  tt.appendChild(all);
  for (const t of c.targets) {
    const b = document.createElement('button');
    b.textContent = 'FDI ' + DATA.fdi_order[t];
    b.className = (visibleTeeth === null || visibleTeeth.has(t)) ? 'on' : '';
    b.onclick = () => {
      if (visibleTeeth === null) visibleTeeth = new Set(c.targets);
      if (visibleTeeth.has(t)) visibleTeeth.delete(t); else visibleTeeth.add(t);
      loadCase();
    };
    tt.appendChild(b);
  }
}

function loadCase() {
  const c = DATA.cases[currentCase];
  caseInfo.textContent = `(${currentCase + 1}/${DATA.cases.length}) · 타겟 ${c.targets.length}개: ${c.targets.map(t => 'FDI ' + DATA.fdi_order[t]).join(', ')}`;
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
  const flipBtn = document.getElementById('flipBtn');
  flipBtn.onclick = () => { zFlip = !zFlip; flipBtn.classList.toggle('on'); loadCase(); };

  const mt = document.getElementById('modelToggles');
  for (const m of DATA.models) {
    const b = document.createElement('button');
    b.textContent = m.label; b.className = 'on'; b.style.borderColor = m.color;
    b.onclick = () => {
      if (activeModels.includes(m.id)) {
        if (activeModels.length === 1) return;
        activeModels = activeModels.filter(x => x !== m.id); b.classList.remove('on');
      } else { activeModels.push(m.id); b.classList.add('on'); }
      rebuildPanels(); loadCase();
    };
    mt.appendChild(b);
  }
}

controls.addEventListener('change', renderAll);
window.addEventListener('resize', renderAll);
buildControls();
rebuildPanels();
loadCase();
