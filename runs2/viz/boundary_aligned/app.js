"use strict";
// aligned boundary 예측 시각화. present 치아(회색 점) + GT cylinder(초록) vs 예측(주황 와이어).
const DATA = window.BAL_DATA;
const REAL_COLOR = 0xb8c0cc;
const GT_COLOR = 0x3ddc7e;     // 초록 = GT
const PRED_COLOR = 0xf6b042;   // 주황 = aligned 예측

const caseSelect = document.getElementById('caseSelect');
const caseInfo = document.getElementById('caseInfo');
const infoEl = document.getElementById('info');
let currentCase = 0, showGT = true, showPred = true;

const panel = document.getElementById('panel');
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
camera.up.set(0, 0, 1); camera.position.set(3.0, -3.0, 2.4);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
panel.appendChild(renderer.domElement);
const controls = new THREE.OrbitControls(camera, panel);
controls.target.set(0, 0, 0); controls.update();
const scene = new THREE.Scene();

function resize() {
  const w = panel.clientWidth, h = panel.clientHeight;
  if (w > 0 && h > 0) { renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
}
function clearScene() {
  for (let i = scene.children.length - 1; i >= 0; i--) {
    const o = scene.children[i]; scene.remove(o);
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
  }
}
function addPoints(pts, color, size, op) {
  if (!pts || !pts.length) return;
  const flat = new Float32Array(pts.length * 3);
  for (let i = 0; i < pts.length; i++) { flat[3*i]=pts[i][0]; flat[3*i+1]=pts[i][1]; flat[3*i+2]=pts[i][2]; }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(flat, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color, size, sizeAttenuation: true, transparent: true, opacity: op })));
}
function addCyl(cyl, color, wire) {
  const cx = cyl[0], cy = cyl[1], cz = cyl[2], h = cyl[3], r = cyl[4];
  if (!(r > 0) || !(h > 0)) return;
  const geo = new THREE.CylinderGeometry(r, r, h, 28, 1, true);
  const mat = new THREE.MeshBasicMaterial({ color, wireframe: wire, transparent: true, opacity: wire ? 0.95 : 0.18, side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(cx, cy, cz);
  mesh.rotation.x = Math.PI / 2;   // 기본 Y축 → Z축(우리 좌표계)
  scene.add(mesh);
}
function buildScene() {
  clearScene();
  const c = DATA.cases[currentCase];
  addPoints(c.real_pts, REAL_COLOR, 0.010, 0.5);
  const rows = c.teeth.map(t => {
    if (showGT) addCyl(t.gt, GT_COLOR, false);
    if (showPred) addCyl(t.pred, PRED_COLOR, true);
    const drift = Math.hypot(t.gt[0]-t.pred[0], t.gt[1]-t.pred[1], t.gt[2]-t.pred[2]);
    return `<span class="fdi-pill">FDI ${t.fdi} drift ${drift.toFixed(3)}</span>`;
  });
  infoEl.innerHTML = `<b>${c.patient}</b> · ${c.teeth.length}개 치아 예측 · ${rows.join('')}`;
  caseInfo.textContent = `(${currentCase + 1}/${DATA.cases.length}) · 초록=GT / 주황=예측`;
  caseSelect.value = String(currentCase);
  resize();
}
function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }

for (let i = 0; i < DATA.cases.length; i++) {
  const o = document.createElement('option'); o.value = String(i); o.textContent = c_name(i); caseSelect.appendChild(o);
}
function c_name(i) { return DATA.cases[i].patient; }
caseSelect.onchange = () => { currentCase = +caseSelect.value; buildScene(); };
document.getElementById('prevCase').onclick = () => { currentCase = Math.max(0, currentCase - 1); buildScene(); };
document.getElementById('nextCase').onclick = () => { currentCase = Math.min(DATA.cases.length - 1, currentCase + 1); buildScene(); };
document.getElementById('gtToggle').onclick = function () { showGT = !showGT; this.classList.toggle('on', showGT); buildScene(); };
document.getElementById('predToggle').onclick = function () { showPred = !showPred; this.classList.toggle('on', showPred); buildScene(); };
window.addEventListener('resize', resize);
buildScene();
animate();
