// Renders boards/previews/ from <build>/boards/manifest.json (written by make_boards.py) with three.js r160
// in headless Chrome, and runs the Khronos glTF Validator on every board GLB.
// node tools/boards/preview.mjs [name filters]
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { ROOT, BUILD as BOOKS_BUILD, PROFILE, chrome, chromeArgs, bodyAttrs } from '../books/lib.mjs';

const PREVIEWS = path.join(ROOT, 'boards', 'previews');
const BUILD = path.join(BOOKS_BUILD, 'boards');
const THREE = 'https://cdn.jsdelivr.net/npm/three@0.160.0';
const VALIDATOR = 'https://cdn.jsdelivr.net/npm/gltf-validator@2.0.0-dev.3.10/+esm';
const WORKERS = 2;

const PAGE_JS = `
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
const body = document.body;
const fail = (e) => { body.dataset.error = String((e && (e.stack || e.message)) || e); };
window.addEventListener('error', (e) => fail(e.error || e.message));
window.addEventListener('unhandledrejection', (e) => fail(e.reason));
window.createImageBitmap = undefined;
const ITEM = window.__ITEM;
const V = (a) => new THREE.Vector3(...a);
const LIGHTS = {
  studio: { bg: '#7f8287', env: 0.3, key: [0xfff4e8, 1.25], fill: [0xe8f0ff, 0.5], hemi: [0xffffff, 0x6b6e73, 0.2], shadows: false },
  frame: { bg: '#2b2522', env: 0, key: [0xfff1e0, 2.0], fill: [0xe8eeff, 0.5], hemi: [0xfff8f0, 0x5a5048, 1.8], shadows: false },
};
function corners(box) {
  const out = [];
  for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) out.push(new THREE.Vector3(x, y, z));
  return out;
}
function fitCamera(box, dir, fov, W, H, margin) {
  const centre = box.getCenter(new THREE.Vector3());
  const d = V(dir).normalize();
  const cam = new THREE.PerspectiveCamera(fov, W / H, 0.01, 200);
  let dist = box.getSize(new THREE.Vector3()).length() / (2 * Math.tan(fov * Math.PI / 360));
  let mid = [0, 0];
  for (let it = 0; it < 16; it++) {
    cam.position.copy(centre).addScaledVector(d, dist);
    cam.lookAt(centre); cam.updateMatrixWorld(); cam.updateProjectionMatrix();
    let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    for (const p of corners(box)) { p.project(cam); x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y); }
    mid = [(x0 + x1) / 2, (y0 + y1) / 2];
    dist *= Math.pow(Math.max((x1 - x0) / 2, (y1 - y0) / 2) / margin, 0.8);
  }
  cam.setViewOffset(W, H, mid[0] * W / 2, -mid[1] * H / 2, W, H);
  return cam;
}
function lightScene(scene, mode, centre, span, env, model) {
  const L = LIGHTS[mode];
  scene.background = new THREE.Color(L.bg);
  scene.environment = L.env ? env : null;
  model.traverse((o) => { if (o.isMesh && 'envMapIntensity' in o.material) o.material.envMapIntensity = L.env; });
  const key = new THREE.DirectionalLight(L.key[0], L.key[1]);
  key.position.copy(centre).add(new THREE.Vector3(-0.45, 1.3, 0.9).normalize().multiplyScalar(span * 3));
  key.target.position.copy(centre);
  if (L.shadows) {
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.radius = 4;
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.002;
    const sc = key.shadow.camera;
    sc.left = sc.bottom = -span * 1.3; sc.right = sc.top = span * 1.3; sc.near = 0.01; sc.far = span * 8;
  }
  const fill = new THREE.DirectionalLight(L.fill[0], L.fill[1]);
  fill.position.copy(centre).add(new THREE.Vector3(1.2, 0.5, 0.7).multiplyScalar(span * 3));
  fill.target.position.copy(centre);
  scene.add(key, key.target, fill, fill.target, new THREE.HemisphereLight(L.hemi[0], L.hemi[1], L.hemi[2]));
  return L;
}
try {
  const bin = Uint8Array.from(atob(ITEM.b64), (c) => c.charCodeAt(0));
  const gltf = await new Promise((res, rej) => new GLTFLoader().parse(bin.buffer.slice(0), '', res, rej));
  try {
    const { validateBytes, version } = await import(${JSON.stringify(VALIDATOR)});
    const r = await validateBytes(bin, { maxIssues: 50 });
    const s = r.issues;
    body.dataset.v = JSON.stringify({ validator: version(), errors: s.numErrors, warnings: s.numWarnings, infos: s.numInfos, hints: s.numHints,
      messages: s.messages.map((m) => [m.severity, m.code, m.pointer || '', m.message]).slice(0, 30),
      triangles: r.info && r.info.totalTriangleCount, vertices: r.info && r.info.totalVertexCount, drawCalls: r.info && r.info.drawCallCount });
  } catch (e) { body.dataset.validatorError = String(e && e.message || e); }
  const model = gltf.scene;
  model.traverse((o) => {
    if (!o.isMesh) return;
    const m = o.material;
    if (m.map) m.map.anisotropy = 4;
    o.castShadow = !m.transparent;
    // SwiftShader's 4x MSAA lets single samples of grazing faces hidden behind the board show through;
    // a slope-scaled depth offset on everything but the artwork avoids that (preview only).
    if (!m.transparent && !m.name.endsWith(' artwork')) Object.assign(m, { polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
  });
  const holder = new THREE.Group();
  holder.add(model);
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const env = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(renderer), 0.04).texture;
  ITEM.views.forEach((view, i) => {
    const [W, H] = view.size;
    renderer.setSize(W, H, false);
    const scene = new THREE.Scene();
    holder.position.set(...(view.offset || [0, 0, 0]));
    scene.add(holder);
    scene.updateMatrixWorld(true);
    const c = view.camera;
    let cam;
    if (view.mode === 'check') {
      model.traverse((o) => { if (o.isMesh) o.visible = o.material.name.endsWith(' artwork'); });
      scene.background = new THREE.Color('#ff00ff');
      cam = new THREE.OrthographicCamera(-c.w / 2, c.w / 2, c.h / 2, -c.h / 2, 0.5, 1.5);
      cam.position.copy(V(c.centre)).addScaledVector(V(c.normal), 1);
      cam.up.copy(V(c.up));
      cam.lookAt(V(c.centre));
      cam.updateProjectionMatrix();
    } else {
      model.traverse((o) => { if (o.isMesh) o.visible = true; });
      const box = new THREE.Box3();
      model.traverse((o) => { if (o.isMesh && !o.material.transparent) box.expandByObject(o); });
      const size = box.getSize(new THREE.Vector3());
      const span = Math.max(size.x, size.y, size.z);
      const L = lightScene(scene, view.mode, box.getCenter(new THREE.Vector3()), Math.max(span, 1), env, model);
      if (view.mode === 'studio') {
        if (view.floor) {
          const floor = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.MeshStandardMaterial({ color: '#8d8f93', roughness: 0.92, envMapIntensity: L.env }));
          floor.rotation.x = -Math.PI / 2;
          floor.position.y = -0.0005;
          scene.add(floor);
        }
        if (view.wall) {
          const wall = new THREE.Mesh(new THREE.PlaneGeometry(16, 8), new THREE.MeshStandardMaterial({ color: '#a4a6a9', roughness: 0.95, envMapIntensity: L.env }));
          wall.position.set(0, 4, -0.0004);
          scene.add(wall);
        }
        cam = fitCamera(box, c.dir, c.fov, W, H, c.margin);
      } else {
        const floor = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.MeshStandardMaterial({ color: '#6b6259', roughness: 0.9 }));
        floor.rotation.x = -Math.PI / 2;
        scene.add(floor);
        if (view.wall) {
          const wallMat = view.wallUnlit ? new THREE.MeshBasicMaterial({ color: view.wallColor })
            : new THREE.MeshStandardMaterial({ color: view.wallColor || '#a39c92', roughness: 0.95 });
          const wall = new THREE.Mesh(new THREE.PlaneGeometry(16, 8), wallMat);
          wall.position.set(0, 4, -0.0004);
          scene.add(wall);
        }
        const f = c.ppd * 180 / Math.PI;
        cam = new THREE.PerspectiveCamera(2 * Math.atan(H / 2 / f) * 180 / Math.PI, W / H, 0.02, 100);
        cam.position.copy(V(c.eye));
        cam.lookAt(V(c.target));
        cam.updateProjectionMatrix();
      }
      document.body.style.background = L.bg;
    }
    renderer.render(scene, cam);
    body.dataset['shot' + i] = renderer.domElement.toDataURL('image/png');
  });
  body.dataset.ready = '1';
} catch (e) { fail(e); }
`;

function page(item) {
  return `<!doctype html>
<html><head><meta charset="utf-8"><title>board preview</title>
<style>html,body{margin:0;padding:0;background:#222;overflow:hidden}</style>
<script type="importmap">{"imports":{"three":"${THREE}/build/three.module.js","three/addons/":"${THREE}/examples/jsm/"}}</script>
</head><body>
<script>window.__ITEM = ${JSON.stringify(item).replace(/</g, '\\u003c')};</script>
<script type="module">${PAGE_JS}</script>
</body></html>`;
}

function outPath(item, view) {
  if (view.mode === 'check') return path.join(BUILD, 'check', `${item.name}.png`);
  if (view.save === false) return path.join(BUILD, 'views', `${item.name}-${view.name}.png`);
  return path.join(PREVIEWS, `${item.name}-${view.name}.png`);
}

async function renderOnce(item, profile) {
  const glb = fs.readFileSync(item.file);
  const file = path.join(BUILD, 'pages', `${item.name}.html`);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, page({ name: item.name, views: item.views, b64: glb.toString('base64') }));
  const args = chromeArgs({ w: 800, h: 600, webgl: true, dumpDom: true, budget: 60000 })
    .map((a) => (a.startsWith('--user-data-dir=') ? `--user-data-dir=${profile}` : a));
  const { out } = await chrome([...args, pathToFileURL(file).href], 400_000);
  const a = bodyAttrs(out);
  const problems = [];
  if (a['data-error']) problems.push(a['data-error'].split('\n')[0]);
  if (a['data-ready'] !== '1') problems.push('page did not finish');
  const shots = [];
  item.views.forEach((view, i) => {
    const url = a[`data-shot${i}`];
    if (!url || !url.startsWith('data:image/png;base64,')) { problems.push(`no ${view.name} image`); return; }
    shots.push([outPath(item, view), Buffer.from(url.slice(url.indexOf(',') + 1), 'base64')]);
  });
  let v = null;
  if (a['data-validator-error']) problems.push('validator: ' + a['data-validator-error']);
  else if (!a['data-v']) problems.push('validator did not run');
  else v = JSON.parse(a['data-v']);
  if (v && v.errors + v.warnings > 0) problems.push(`${v.errors} validator errors, ${v.warnings} warnings`);
  return { problems, shots, v, bytes: glb.length };
}

async function renderItem(item, profile) {
  let r;
  for (let attempt = 0; attempt < 3; attempt++) {
    r = await renderOnce(item, profile);
    if (!r.problems.length || (r.v && r.v.errors + r.v.warnings > 0 && r.shots.length === item.views.length)) break;
  }
  for (const [file, data] of r.shots) {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, data);
  }
  return r;
}

async function main() {
  const filters = process.argv.slice(2);
  const manifest = JSON.parse(fs.readFileSync(path.join(BUILD, 'manifest.json'), 'utf8'));
  const items = manifest.boards.filter((b) => !filters.length || filters.some((f) => b.name.includes(f)));
  fs.mkdirSync(PREVIEWS, { recursive: true });
  const report = [];
  let next = 0;
  let bad = 0;
  async function worker(k) {
    const profile = `${PROFILE}-${k}`;
    fs.mkdirSync(profile, { recursive: true });
    while (next < items.length) {
      const item = items[next++];
      const r = await renderItem(item, profile);
      const v = r.v || {};
      report.push({ name: item.name, bytes: r.bytes, problems: r.problems, ...v });
      if (r.problems.length) bad++;
      console.log(`${r.problems.length ? 'FAIL' : 'ok  '} ${item.name.padEnd(28)} Khronos ${v.validator || '?'}: E${v.errors} W${v.warnings} I${v.infos} H${v.hints}  tris ${v.triangles}  draws ${v.drawCalls}  ${r.problems.join('; ')}`);
      for (const m of v.messages || []) console.log('       ', m.join(' | '));
    }
  }
  await Promise.all(Array.from({ length: Math.min(WORKERS, items.length) }, (_, k) => worker(k + 1)));
  fs.writeFileSync(path.join(BUILD, 'preview-report.json'), JSON.stringify(report, null, 1));
  console.log(bad ? `${bad} files with problems.` : `All ${items.length} files valid; previews in ${PREVIEWS}`);
  process.exitCode = bad ? 1 : 0;
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
