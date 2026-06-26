"use strict";
// boundary 위치 비교 3-way: OLD / NEW(G1) / ARCH(아치보간 위치).
// present=회색(공통). crown 색 = OLD 주황 / NEW 초록 / ARCH 시안.
// 3패널 회전 동기화. 환자별 겹침% 표시.
const DATA = window.PCMP_DATA;
const REAL_COLOR = 0xc8cdd6;
const VARIANTS = [
  {key: 'old',  color: 0xf39c12, name: 'OLD boundary (official_long)'},
  {key: 'new',  color: 0x2ecc71, name: 'NEW boundary (G1)'},
  {key: 'arch', color: 0x36d6e7, name: 'ARCH 하이브리드 (내부 결손=아치보간 / 끝자리 결손=boundary)'},
];

const caseSelect = document.getElementById('caseSelect');
const caseInfo = document.getElementById('caseInfo');
const infoEl = document.getElementById('info');
let currentCase = 0;

// 범용 패널 생성
function makePanel(id) {
  const el = document.getElementById(id);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(3.0, -3.0, 2.4);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  el.appendChild(renderer.domElement);
  const controls = new THREE.OrbitControls(camera, el);
  controls.target.set(0, 0, 0); controls.update();
  const scene = new THREE.Scene();
  return { el, camera, renderer, controls, scene };
}
const panels = VARIANTS.map((v, i) => Object.assign(makePanel('panel' + i), { variant: v }));

function resize() {
  for (const p of panels) {
    const w = p.el.clientWidth, h = p.el.clientHeight;
    if (w > 0 && h > 0) { p.renderer.setSize(w, h); p.camera.aspect = w / h; p.camera.updateProjectionMatrix(); }
  }
}
function clearScene(scene) {
  for (let i = scene.children.length - 1; i >= 0; i--) {
    const o = scene.children[i]; scene.remove(o);
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
  }
}
function addPoints(scene, pts, color, size, opacity) {
  if (!pts || !pts.length) return;
  const flat = new Float32Array(pts.length * 3);
  for (let i = 0; i < pts.length; i++) { flat[3*i]=pts[i][0]; flat[3*i+1]=pts[i][1]; flat[3*i+2]=pts[i][2]; }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(flat, 3));
  const m = new THREE.PointsMaterial({ color, size, sizeAttenuation: true, transparent: true, opacity });
  scene.add(new THREE.Points(g, m));
}
function buildSide(p) {
  clearScene(p.scene);
  const c = DATA.cases[currentCase];
  addPoints(p.scene, c.real_pts, REAL_COLOR, 0.010, 0.5);
  for (const ps of c[p.variant.key]) addPoints(p.scene, ps.pts, p.variant.color, 0.016, 0.95);
}
function buildScene() {
  const c = DATA.cases[currentCase];
  for (const p of panels) buildSide(p);
  const fdi = c.new.map(p => `<span class="fdi-pill">FDI ${p.fdi}</span>`).join('');
  const statLine = VARIANTS.map(v => {
    const s = c[v.key + '_stat'];
    const col = v.key === 'old' ? '#f6b042' : v.key === 'new' ? '#3ddc7e' : '#36d6e7';
    return `<span style="color:${col}">${v.key.toUpperCase()} 겹침 ${s.ov}/${s.n} (${s.pct}%, nn${s.nn})</span>`;
  }).join(' · ');
  infoEl.innerHTML = `<b>${c.patient}</b> · real ${c.n_real} + 결손 ${c.n_miss} · ${statLine}<br>결손 FDI: ${fdi}`;
  const archS = c.arch_stat, oldS = c.old_stat;
  caseInfo.textContent = `(${currentCase + 1}/${DATA.cases.length}) · 결손 ${c.n_miss} · 겹침 OLD ${oldS.pct}% → NEW ${c.new_stat.pct}% → ARCH ${archS.pct}%`;
  caseSelect.value = String(currentCase);
  resize();
}
function animate() {
  requestAnimationFrame(animate);
  for (const p of panels) { p.controls.update(); p.renderer.render(p.scene, p.camera); }
}

// 회전 동기화 (어느 패널을 돌려도 나머지 따라감)
let syncing = false;
function syncFrom(src) {
  if (syncing) return; syncing = true;
  for (const dst of panels) {
    if (dst === src) continue;
    dst.camera.position.copy(src.camera.position);
    dst.camera.quaternion.copy(src.camera.quaternion);
    dst.controls.target.copy(src.controls.target);
    dst.camera.up.copy(src.camera.up);
    dst.controls.update();
  }
  syncing = false;
}
for (const p of panels) p.controls.addEventListener('change', () => syncFrom(p));

for (let i = 0; i < DATA.cases.length; i++) {
  const o = document.createElement('option'); o.value = String(i); o.textContent = 'case ' + (i + 1);
  caseSelect.appendChild(o);
}
caseSelect.onchange = () => { currentCase = +caseSelect.value; buildScene(); };
document.getElementById('prevCase').onclick = () => { currentCase = Math.max(0, currentCase - 1); buildScene(); };
document.getElementById('nextCase').onclick = () => { currentCase = Math.min(DATA.cases.length - 1, currentCase + 1); buildScene(); };

window.addEventListener('resize', resize);
buildScene();
animate();
