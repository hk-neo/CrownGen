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