"use strict";
// gen_aligned 생성 결과. present(회색) + GT(초록) vs 생성 크라운(주황). per-tooth CD.
const DATA = window.GA_DATA;
const REAL_COLOR = 0x9aa3b0;
const GT_COLOR = 0x3ddc7e;     // 초록 = GT
const GEN_COLOR = 0xf6b042;    // 주황 = 생성

const caseSelect = document.getElementById('caseSelect');
const caseInfo = document.getElementById('caseInfo');
const infoEl = document.getElementById('info');
let currentCase = 0, showGT = true, showGen = true;

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
function buildScene() {
  clearScene();
  const c = DATA.cases[currentCase];
  addPoints(c.real_pts, REAL_COLOR, 0.010, 0.45);
  const pills = c.teeth.map(t => {
    if (showGT) addPoints(t.gt, GT_COLOR, 0.014, 0.9);
    if (showGen) addPoints(t.gen, GEN_COLOR, 0.014, 0.95);
    return `<span class="fdi-pill">FDI ${t.fdi} · CD ${t.cd}</span>`;
  });
  infoEl.innerHTML = `<b>${c.patient}</b> · 생성 크라운(주황) vs GT(초록) · ${pills.join('')}`;
  caseInfo.textContent = `(${currentCase + 1}/${DATA.cases.length}) · 초록=GT / 주황=생성 · CD 단위 ×10⁻³`;
  caseSelect.value = String(currentCase);
  resize();
}
function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }

for (let i = 0; i < DATA.cases.length; i++) {
  const o = document.createElement('option'); o.value = String(i); o.textContent = DATA.cases[i].patient; caseSelect.appendChild(o);
}
caseSelect.onchange = () => { currentCase = +caseSelect.value; buildScene(); };
document.getElementById('prevCase').onclick = () => { currentCase = Math.max(0, currentCase - 1); buildScene(); };
document.getElementById('nextCase').onclick = () => { currentCase = Math.min(DATA.cases.length - 1, currentCase + 1); buildScene(); };
document.getElementById('gtToggle').onclick = function () { showGT = !showGT; this.classList.toggle('on', showGT); buildScene(); };
document.getElementById('genToggle').onclick = function () { showGen = !showGen; this.classList.toggle('on', showGen); buildScene(); };
window.addEventListener('resize', resize);
buildScene();
animate();
