"use strict";
// Pseudo-crown 시각화. 단일 환자: real 치아(회색) + pseudo 크라운(주황, 3000ep 모델 생성).
const DATA = window.PSEUDO_DATA;
const REAL_COLOR = 0xb0b8c4;
const PSEUDO_COLOR = 0xf39c12;

const panel = document.getElementById('panel');
const caseSelect = document.getElementById('caseSelect');
const caseInfo = document.getElementById('caseInfo');
const infoEl = document.getElementById('info');

let currentCase = 0;
let showReal = true;
let showPseudo = true;

const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
camera.up.set(0, 0, 1);
camera.position.set(3.0, -3.0, 2.4);
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
function addPoints(pts, color, size, opacity) {
  if (!pts || !pts.length) return;
  const flat = new Float32Array(pts.length * 3);
  for (let i = 0; i < pts.length; i++) { flat[3*i]=pts[i][0]; flat[3*i+1]=pts[i][1]; flat[3*i+2]=pts[i][2]; }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(flat, 3));
  const m = new THREE.PointsMaterial({ color, size, sizeAttenuation: true, transparent: true, opacity });
  scene.add(new THREE.Points(g, m));
}

function buildScene() {
  clearScene();
  const c = DATA.cases[currentCase];
  if (showReal) addPoints(c.real_pts, REAL_COLOR, 0.010, 0.55);
  if (showPseudo) for (const p of c.pseudo) addPoints(p.pts, PSEUDO_COLOR, 0.016, 0.95);
  // info
  const fdi = c.pseudo.map(p => `<span class="fdi-pill">FDI ${p.fdi}</span>`).join('');
  infoEl.innerHTML = `<b>${c.patient}</b> · real ${c.n_real}개 + <b>pseudo ${c.n_pseudo}개</b> 채움 · pseudo 치아: ${fdi}`;
  caseInfo.textContent = `(${currentCase + 1}/${DATA.cases.length}) · pseudo ${c.n_pseudo}개`;
  caseSelect.value = String(currentCase);
  resize();
}

function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }

// controls
for (let i = 0; i < DATA.cases.length; i++) {
  const o = document.createElement('option'); o.value = String(i); o.textContent = 'case ' + (i + 1);
  caseSelect.appendChild(o);
}
caseSelect.onchange = () => { currentCase = +caseSelect.value; buildScene(); };
document.getElementById('prevCase').onclick = () => { currentCase = Math.max(0, currentCase - 1); buildScene(); };
document.getElementById('nextCase').onclick = () => { currentCase = Math.min(DATA.cases.length - 1, currentCase + 1); buildScene(); };
document.getElementById('realToggle').onclick = function () { showReal = !showReal; this.classList.toggle('on', showReal); buildScene(); };
document.getElementById('pseudoToggle').onclick = function () { showPseudo = !showPseudo; this.classList.toggle('on', showPseudo); buildScene(); };

window.addEventListener('resize', resize);
buildScene();
animate();
