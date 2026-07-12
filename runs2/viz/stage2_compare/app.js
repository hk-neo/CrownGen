"use strict";
// gen_aligned vs stage2 2패널 비교. GT(초록) vs 생성(주황). 회전 동기화.
const DATA = window.COMP_DATA;
const REAL_COLOR = 0x9aa3b0, GT_COLOR = 0x3ddc7e, GEN_COLOR = 0xf6b042;
const KEYS = ['gen_aligned', 'stage2'];

const caseSelect = document.getElementById('caseSelect');
const caseInfo = document.getElementById('caseInfo');
const infoEl = document.getElementById('info');
let cur = 0;

function mkPanel(id) {
  const el = document.getElementById(id);
  const cam = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
  cam.up.set(0,0,1); cam.position.set(3,-3,2.4);
  const ren = new THREE.WebGLRenderer({antialias:true}); ren.setPixelRatio(devicePixelRatio); el.appendChild(ren.domElement);
  const con = new THREE.OrbitControls(cam, el); con.target.set(0,0,0); con.update();
  return {el, cam, ren, con, scene:new THREE.Scene()};
}
const P = [mkPanel('panel0'), mkPanel('panel1')];

function resize() { P.forEach(p=>{const w=p.el.clientWidth,h=p.el.clientHeight;if(w>0&&h>0){p.ren.setSize(w,h);p.cam.aspect=w/h;p.cam.updateProjectionMatrix();}}); }
function clr(s) { for(let i=s.children.length-1;i>=0;i--){const o=s.children[i];s.remove(o);if(o.geometry)o.geometry.dispose();if(o.material)o.material.dispose();} }
function ap(s,pts,c,sz,op) { if(!pts||!pts.length)return;const f=new Float32Array(pts.length*3);for(let i=0;i<pts.length;i++){f[3*i]=pts[i][0];f[3*i+1]=pts[i][1];f[3*i+2]=pts[i][2];}const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(f,3));s.add(new THREE.Points(g,new THREE.PointsMaterial({color:c,size:sz,sizeAttenuation:true,transparent:true,opacity:op}))); }

function build() {
  const c = DATA.cases[cur];
  P.forEach((p,pi) => {
    clr(p.scene); ap(p.scene, c.real_pts, REAL_COLOR, 0.010, 0.45);
    const teeth = c.models[KEYS[pi]];
    teeth.forEach(t => { ap(p.scene, t.gt, GT_COLOR, 0.014, 0.85); ap(p.scene, t.gen, GEN_COLOR, 0.014, 0.95); });
  });
  const cds = KEYS.map((k,i) => {
    const arr = c.models[k]; const m = arr.reduce((s,t)=>s+t.cd,0)/arr.length;
    return `${KEYS[i]}: ${arr.map(t=>t.cd).join('/')} (평균 ${m.toFixed(1)})`;
  });
  infoEl.innerHTML = `<b>${c.patient}</b> · ${cds.join(' · ')}`;
  const m0 = c.models['gen_aligned'].reduce((s,t)=>s+t.cd,0)/c.models['gen_aligned'].length;
  const m1 = c.models['stage2'].reduce((s,t)=>s+t.cd,0)/c.models['stage2'].length;
  caseInfo.textContent = `(${cur+1}/${DATA.cases.length}) · gen_aligned ${m0.toFixed(1)} vs stage2 ${m1.toFixed(1)}`;
  caseSelect.value = String(cur); resize();
}
function animate() { requestAnimationFrame(animate); P.forEach(p=>{p.con.update();p.ren.render(p.scene,p.cam);}); }

let syncing=false;
function sync(src){if(syncing)return;syncing=true;P.forEach(d=>{if(d===src)return;d.cam.position.copy(src.cam.position);d.cam.quaternion.copy(src.cam.quaternion);d.con.target.copy(src.con.target);d.cam.up.copy(src.cam.up);d.con.update();});syncing=false;}
P.forEach(p=>p.con.addEventListener('change',()=>sync(p)));

for(let i=0;i<DATA.cases.length;i++){const o=document.createElement('option');o.value=String(i);o.textContent=DATA.cases[i].patient;caseSelect.appendChild(o);}
caseSelect.onchange=()=>{cur=+caseSelect.value;build();};
document.getElementById('prevCase').onclick=()=>{cur=Math.max(0,cur-1);build();};
document.getElementById('nextCase').onclick=()=>{cur=Math.min(DATA.cases.length-1,cur+1);build();};
window.addEventListener('resize',resize); build(); animate();
