"use strict";
const D = window.MESH_DATA;
const sel = document.getElementById('sel');
const infoEl = document.getElementById('info');
let cur = 0, showGT = true, showGen = true;

const panel = document.getElementById('panel');
const cam = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
cam.up.set(0, 0, 1); cam.position.set(1.5, -1.5, 1.2);
const ren = new THREE.WebGLRenderer({ antialias: true });
ren.setPixelRatio(devicePixelRatio); panel.appendChild(ren.domElement);
const con = new THREE.OrbitControls(cam, panel); con.target.set(0, 0, 0); con.update();
const scene = new THREE.Scene();
scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dl = new THREE.DirectionalLight(0xffffff, 0.7); dl.position.set(2, -2, 3); scene.add(dl);
const dl2 = new THREE.DirectionalLight(0x88aaff, 0.3); dl2.position.set(-2, 1, 1); scene.add(dl2);

function resize() { const w = panel.clientWidth, h = panel.clientHeight; if (w > 0 && h > 0) { ren.setSize(w, h); cam.aspect = w / h; cam.updateProjectionMatrix(); } }
function clr() { for (let i = scene.children.length - 1; i >= 0; i--) { const o = scene.children[i]; if (o.type === 'Mesh' || o.type === 'Group' || o.type === 'Points') { scene.remove(o); if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); } } }
function addPoints(pts, color, size, op) {
  if (!pts || !pts.length) return;
  const f = new Float32Array(pts.length * 3);
  for (let i = 0; i < pts.length; i++) { f[3*i]=pts[i][0]; f[3*i+1]=pts[i][1]; f[3*i+2]=pts[i][2]; }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(f, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color, size, sizeAttenuation: true, transparent: true, opacity: op })));
}
function mkMesh(data, color) {
  if (!data || !data.v || !data.f) return null;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(data.v.flat(), 3));
  g.setIndex(data.f.flat());
  g.computeVertexNormals();
  const m = new THREE.MeshStandardMaterial({ color, flatShading: false, side: THREE.DoubleSide,
    transparent: true, opacity: 0.92, roughness: 0.5, metalness: 0.1 });
  const mesh = new THREE.Mesh(g, m);
  // 원래 좌표 그대로 (아치 위치 유지, 중앙정렬 안 함)
  return mesh;
}
function build() {
  clr();
  const c = D.cases[cur];
  if (c.real_pts) addPoints(c.real_pts, 0x8a94a6, 0.008, 0.35);  // 회색 = present 치아 (IOS 아치 컨텍스트)
  if (showGT) { const m = mkMesh(c.gt, 0x3ddc7e); if (m) scene.add(m); }
  if (showGen) { const m = mkMesh(c.gen, 0xf6b042); if (m) scene.add(m); }
  infoEl.innerHTML = `<b>${c.patient}</b> · FDI ${c.fdi} · 회색=IOS 치아(컨텍스트) · GT(초록) ${c.gt.v.length/3}v vs 생성(주황) ${c.gen.v.length/3}v`;
  sel.value = String(cur); resize();
}
function animate() { requestAnimationFrame(animate); con.update(); ren.render(scene, cam); }

for (let i = 0; i < D.cases.length; i++) { const o = document.createElement('option'); o.value = i; o.textContent = `${D.cases[i].patient} FDI${D.cases[i].fdi}`; sel.appendChild(o); }
sel.onchange = () => { cur = +sel.value; build(); };
document.getElementById('prev').onclick = () => { cur = Math.max(0, cur - 1); build(); };
document.getElementById('next').onclick = () => { cur = Math.min(D.cases.length - 1, cur + 1); build(); };
document.getElementById('togGT').onclick = function () { showGT = !showGT; this.classList.toggle('on', showGT); build(); };
document.getElementById('togGen').onclick = function () { showGen = !showGen; this.classList.toggle('on', showGen); build(); };
window.addEventListener('resize', resize); build(); animate();
