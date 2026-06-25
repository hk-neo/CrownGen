"use strict";
// 에폭별 크라운 진화 — '치아 1개' 단위. 선택한 치아의 ep1100→2000 패널 6개 + GT 오버레이.
// 단일 카메라 + OrbitControls(panelsEl) → 전체 패널 동기화 회전.
const DATA = window.GEN_PROGRESSION_DATA;
const GT_COLOR = 0x2ecc71;

function lerpColor(a, b, t) {
  const ar = (a >> 16) & 255, ag = (a >> 8) & 255, ab = a & 255;
  const br = (b >> 16) & 255, bg = (b >> 8) & 255, bb = b & 255;
  const r = Math.round(ar + (br - ar) * t), g = Math.round(ag + (bg - ag) * t), c = Math.round(ab + (bb - ab) * t);
  return (r << 16) | (g << 8) | c;
}
const C0 = 0x4a90d9, CN = 0xe74c3c;  // 에폭 색: 초기(파랑) → 말기(빨강)

const panelsEl = document.getElementById('panels');
let selTooth = 0;
let showGT = true;
let zFlip = false;
let panels = [];

const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
camera.up.set(0, 0, 1);
camera.position.set(0.9, -0.9, 0.7);
const controls = new THREE.OrbitControls(camera, panelsEl);
controls.target.set(0, 0, 0); controls.update();

function clearScene(s) {
  for (let i = s.children.length - 1; i >= 0; i--) {
    const o = s.children[i]; s.remove(o);
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

function buildScenes() {
  for (const p of panels) { p.renderer.dispose(); if (p.renderer.domElement.parentNode) p.renderer.domElement.parentNode.remove(); }
  panels = []; panelsEl.innerHTML = '';
  const tooth = DATA.teeth[selTooth];
  const N = tooth.series.length;
  tooth.series.forEach((s, i) => {
    const div = document.createElement('div'); div.className = 'panel';
    const title = document.createElement('div'); title.className = 'ptitle';
    div.appendChild(title); panelsEl.appendChild(div);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(div.clientWidth || 300, div.clientHeight || 300);
    div.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.scale.z = zFlip ? -1 : 1;
    const ecol = lerpColor(C0, CN, i / Math.max(1, N - 1));
    addPoints(scene, s.pts, ecol, 0.016, 0.95);              // 생성 크라운 (에폭 색)
    if (showGT) addPoints(scene, tooth.gt_pts, GT_COLOR, 0.014, 0.5);  // GT
    title.innerHTML = `ep ${s.ep} · <span style="color:#3fb950">CD×10³ ${s.cd.toFixed(1)}</span>`;
    panels.push({ renderer, scene, div });
  });
  updateInfo();
  renderAll();
}

function renderAll() {
  for (const p of panels) {
    const w = p.div.clientWidth, h = p.div.clientHeight;
    if (w > 0 && h > 0) {
      p.renderer.setSize(w, h);
      camera.aspect = w / h; camera.updateProjectionMatrix();
      p.renderer.render(p.scene, camera);
    }
  }
}

function updateInfo() {
  const tooth = DATA.teeth[selTooth];
  const s0 = tooth.series[0], sN = tooth.series[tooth.series.length - 1];
  document.getElementById('subtitle').textContent =
    `— patient ${DATA.patient} · 치아 FDI ${tooth.fdi} (1개) · ${s0.ep}→${sN.ep}ep`;
  document.getElementById('trend').innerHTML =
    `FDI ${tooth.fdi} CD×10³: <b>${s0.cd.toFixed(1)}</b> (ep${s0.ep}) → <b>${sN.cd.toFixed(1)}</b> (ep${sN.ep}) ` +
    `<b>${(sN.cd - s0.cd).toFixed(1)}</b> (음=개선). 학습 진행하며 GT(초록)로 수렴.`;
}

// 치아 선택자
const toothSel = document.getElementById('toothSel');
DATA.teeth.forEach((t, i) => {
  const b = document.createElement('button');
  b.textContent = 'FDI ' + t.fdi;
  b.className = i === selTooth ? 'on' : '';
  b.onclick = () => { selTooth = i;
    [...toothSel.children].forEach((x, j) => x.classList.toggle('on', j === i));
    buildScenes();
  };
  toothSel.appendChild(b);
});

controls.addEventListener('change', renderAll);
window.addEventListener('resize', renderAll);
document.getElementById('gtToggle').onclick = function () { showGT = !showGT; this.classList.toggle('on', showGT); buildScenes(); };
document.getElementById('flipBtn').onclick = function () { zFlip = !zFlip; this.classList.toggle('on', zFlip); buildScenes(); };

buildScenes();
