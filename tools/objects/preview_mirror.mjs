#!/usr/bin/env node
/* Renders objects/mirror.glb (three.js r160, headless Chrome) into objects/previews/: front, 3/4 and back under studio and plain
   light, the visitor view also in a dim and a bright hall; runs the glTF Validator on every objects/mirror*.glb.
   node tools/objects/preview_mirror.mjs [--qa]   (--qa adds close-ups and unlit-fallback views to <BOOKS_BUILD_DIR>/mirror-qa) */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { BUILD, PROFILE, bodyAttrs, chrome, chromeArgs, pngSize } from '../books/lib.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');
const OBJECTS = path.join(ROOT, 'objects');
const PREVIEWS = path.join(OBJECTS, 'previews');
const QA_DIR = path.join(BUILD, 'mirror-qa');
const THREE = 'https://cdn.jsdelivr.net/npm/three@0.160.0';
const VALIDATOR = 'https://cdn.jsdelivr.net/npm/gltf-validator@2.0.0-dev.3.10/+esm';
const VISITOR = { size: [1920, 1300], eye: [0, 1.6, 2.0], target: [0, 0.9, 0], fov: 52, room: true };

const VIEWS = [
  { name: 'front', size: [1080, 1440], dir: [0, 0, 1], fov: 24, fill: 0.9, lights: ['studio', 'plain'] },
  { name: 'angle', size: [1080, 1440], dir: [-0.62, 0.3, 0.72], fov: 24, fill: 0.9, lights: ['studio', 'plain'] },
  { name: 'visitor', ...VISITOR, lights: ['studio', 'plain', 'dim', 'bright'] },
  { name: 'back', size: [1080, 1440], dir: [0.4, 0.22, -0.89], fov: 24, fill: 0.9, lights: ['studio', 'plain'] },
];
const QA = [
  { name: 'side', size: [1080, 1440], dir: [1, 0.06, 0.02], fov: 24, fill: 0.9 },
  { name: 'pivot', size: [1200, 900], eye: [0.62, 1.13, 0.5], target: [0.32, 1.02, 0.0], fov: 22 },
  { name: 'corner', size: [1200, 900], eye: [-0.12, 0.56, 0.72], target: [-0.25, 0.41, 0.04], fov: 22 },
  { name: 'arch', size: [1200, 900], eye: [0.12, 1.78, 0.85], target: [0, 1.62, -0.04], fov: 24 },
  { name: 'grain', size: [1200, 900], eye: [0.42, 1.3, 0.62], target: [0.29, 1.2, 0.0], fov: 18 },
  { name: 'post', size: [1200, 900], eye: [0.8, 0.8, 0.55], target: [0.36, 0.7, 0.0], fov: 22 },
  { name: 'post-back', size: [1200, 900], eye: [0.05, 0.75, -0.65], target: [0.34, 0.65, 0.0], fov: 24 },
  { name: 'finial', size: [1200, 900], eye: [0.6, 1.28, 0.42], target: [0.352, 1.14, 0.0], fov: 16 },
  { name: 'foot', size: [1200, 900], eye: [0.78, 0.28, 0.62], target: [0.343, 0.04, 0.04], fov: 26 },
  { name: 'near', size: [1920, 1080], eye: [0.05, 1.6, 0.75], target: [0, 1.2, 0], fov: 45.8, room: true },
];

function page(glbs, jobs) {
  return `<!doctype html>
<html><head><meta charset="utf-8"><title>mirror preview</title>
<style>html,body{margin:0;padding:0;background:#7f8287;overflow:hidden}canvas,pre{display:none}</style>
<script type="importmap">{"imports":{"three":"${THREE}/build/three.module.js","three/addons/":"${THREE}/examples/jsm/"}}</script>
</head><body>
<script>window.__GLBS = ${JSON.stringify(glbs)}; window.__JOBS = ${JSON.stringify(jobs)};</script>
<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
const body = document.body;
const fail = (e) => { body.dataset.error = String(e && (e.stack || e.message) || e); };
window.addEventListener('error', (e) => fail(e.error || e.message));
window.createImageBitmap = undefined;
const SETUPS = {
  studio: { env: 0.3, shadows: true, dirs: [[0xfff4e8, 1.25, [-0.5, 1.4, 0.9]], [0xe8f0ff, 0.5, [1.2, 0.6, 0.6]]], hemi: [0xffffff, 0x6b6e73, 0.2],
    room: ['#8c847a', '#cbc5bb'] },
  plain: { env: 0, dirs: [[0xffffff, 1.9, [-0.6, 1.5, 1.0]]], hemi: [0xffffff, 0x4d4a46, 1.7], room: ['#8c847a', '#cbc5bb'] },
  dim: { env: 0.1, dirs: [[0xffc690, 0.6, [-0.5, 1.4, 0.9]], [0x9a8470, 0.1, [1.2, 0.6, 0.6]]], hemi: [0xffe0b8, 0x2a2018, 0.14],
    room: ['#8c847a', '#cbc5bb'] },
  bright: { env: 0, dirs: [[0xffffff, 2.6, [-0.6, 1.5, 1.0]]], hemi: [0xffffff, 0x8a857d, 2.8], room: ['#bdb4a7', '#ece7de'] },
};
try {
  const models = [];
  for (const [i, b64] of window.__GLBS.entries()) {
    const bin = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    try {
      const { validateBytes, version } = await import('${VALIDATOR}');
      const r = await validateBytes(bin, { maxIssues: 100 });
      const s = r.issues;
      body.dataset['v' + i] = JSON.stringify({
        validator: version(), errors: s.numErrors, warnings: s.numWarnings, infos: s.numInfos, hints: s.numHints,
        messages: s.messages.map((m) => [m.severity, m.code, m.pointer || '', m.message]).slice(0, 30),
        triangles: r.info && r.info.totalTriangleCount, vertices: r.info && r.info.totalVertexCount, drawCalls: r.info && r.info.drawCallCount,
      });
    } catch (e) { body.dataset['v' + i] = JSON.stringify({ error: String(e && e.message || e) }); }
    const gltf = await new Promise((res, rej) => new GLTFLoader().parse(bin.buffer, '', res, rej));
    const model = gltf.scene;
    const mats = [];
    model.traverse((o) => { if (o.isMesh) { o.castShadow = o.receiveShadow = !o.material.transparent; mats.push(o.material); } });
    model.updateMatrixWorld(true);
    const box = new THREE.Box3();
    model.traverse((o) => { if (o.isMesh && !o.material.transparent) box.expandByObject(o); });
    if (i === 0) body.dataset.bbox = JSON.stringify([box.min.toArray(), box.max.toArray()].map((v) => v.map((x) => +x.toFixed(4))));
    models.push({ model, mats, box, centre: box.getCenter(new THREE.Vector3()) });
  }
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const env = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(renderer), 0.04).texture;
  const plane = (w, h, color, rough) => new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshStandardMaterial({ color, roughness: rough, metalness: 0 }));
  function lights(scene, S, centre, mats) {
    scene.background = new THREE.Color('#7f8287');
    scene.environment = S.env ? env : null;
    for (const m of mats) if ('envMapIntensity' in m) m.envMapIntensity = S.env;
    S.dirs.forEach(([color, intensity, from], k) => {
      const d = new THREE.DirectionalLight(color, intensity);
      d.position.copy(centre).add(new THREE.Vector3(...from).multiplyScalar(4));
      d.target.position.copy(centre);
      if (S.shadows && k === 0) {
        d.castShadow = true;
        d.shadow.mapSize.set(2048, 2048);
        d.shadow.radius = 6;
        d.shadow.bias = -0.0004;
        d.shadow.normalBias = 0.002;
        Object.assign(d.shadow.camera, { left: -1.6, right: 1.6, top: 1.6, bottom: -1.6, near: 0.1, far: 14 });
      }
      scene.add(d, d.target);
    });
    scene.add(new THREE.HemisphereLight(...S.hemi));
  }
  function fit(view, w, h, box, centre) {
    const cam = new THREE.PerspectiveCamera(view.fov, w / h, 0.01, 60);
    const dir = new THREE.Vector3(...view.dir).normalize();
    const corners = [];
    for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) corners.push(new THREE.Vector3(x, y, z));
    let lo = 0.3, hi = 30;
    for (let k = 0; k < 40; k++) {
      const d = (lo + hi) / 2;
      cam.position.copy(centre).addScaledVector(dir, d);
      cam.lookAt(centre);
      cam.updateMatrixWorld(true);
      cam.updateProjectionMatrix();
      const m = Math.max(...corners.map((c) => { const p = c.clone().project(cam); return Math.max(Math.abs(p.x), Math.abs(p.y)); }));
      if (m > view.fill) lo = d; else hi = d;
    }
    return cam;
  }
  let shadowsOn = null;
  for (const job of window.__JOBS) {
    const [w, h] = job.size;
    const { model, mats, box, centre } = models[job.glb];
    const S = SETUPS[job.light];
    const scene = new THREE.Scene();
    scene.add(model);
    lights(scene, S, centre, mats);
    const shadows = !!S.shadows;
    if (shadows !== shadowsOn) { renderer.shadowMap.enabled = shadows; for (const m of models.flatMap((x) => x.mats)) m.needsUpdate = true; shadowsOn = shadows; }
    if (job.room) {
      const floor = plane(40, 40, S.room[0], 0.92);
      floor.rotation.x = -Math.PI / 2;
      floor.position.y = -0.0001;
      floor.receiveShadow = true;
      const wall = plane(14, 6, S.room[1], 0.95);
      wall.position.set(0, 3, -1.6);
      wall.receiveShadow = true;
      floor.material.envMapIntensity = wall.material.envMapIntensity = S.env;
      scene.add(floor, wall);
    } else if (shadows) {
      const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.ShadowMaterial({ opacity: 0.32 }));
      floor.rotation.x = -Math.PI / 2;
      floor.position.y = -0.0002;
      floor.receiveShadow = true;
      scene.add(floor);
    }
    let cam;
    if (job.eye) {
      cam = new THREE.PerspectiveCamera(job.fov, w / h, 0.01, 60);
      cam.position.set(...job.eye);
      cam.lookAt(new THREE.Vector3(...job.target));
      cam.updateProjectionMatrix();
    } else cam = fit(job, w, h, box, centre);
    renderer.setSize(w, h);
    renderer.render(scene, cam);
    renderer.render(scene, cam);
    const pre = document.createElement('pre');
    pre.id = 'shot-' + job.out;
    pre.textContent = renderer.domElement.toDataURL('image/png');
    body.appendChild(pre);
  }
  body.dataset.ready = '1';
} catch (e) { fail(e); }
</script></body></html>`;
}

async function run(glbFiles, jobs, label) {
  const file = path.join(BUILD, `mirror-preview-${label}.html`);
  fs.mkdirSync(BUILD, { recursive: true });
  fs.mkdirSync(PROFILE, { recursive: true });
  fs.writeFileSync(file, page(glbFiles.map((f) => fs.readFileSync(f).toString('base64')), jobs));
  const { out } = await chrome([...chromeArgs({ w: 800, h: 600, webgl: true, budget: 180000 }), pathToFileURL(file).href], 900_000);
  const a = bodyAttrs(out);
  const problems = [];
  if (a['data-error']) problems.push(a['data-error']);
  if (a['data-ready'] !== '1') problems.push('page did not finish');
  for (const job of jobs) {
    const m = out.match(new RegExp(`<pre id="shot-${job.out.replace(/[^\w-]/g, '')}">data:image/png;base64,([A-Za-z0-9+/=]+)</pre>`));
    if (!m) { problems.push(`no image for ${job.out}`); continue; }
    const buf = Buffer.from(m[1], 'base64');
    const [w, h] = pngSize(buf);
    if (w !== job.size[0] || h !== job.size[1]) problems.push(`${job.out} is ${w}x${h}`);
    fs.mkdirSync(path.dirname(job.file), { recursive: true });
    fs.writeFileSync(job.file, buf);
  }
  return { a, problems };
}

async function main() {
  const qa = process.argv.includes('--qa');
  const glbs = fs.readdirSync(OBJECTS).filter((f) => /^mirror.*\.glb$/.test(f)).sort((x, y) => (x === 'mirror.glb' ? -1 : y === 'mirror.glb' ? 1 : x.localeCompare(y)));
  if (glbs[0] !== 'mirror.glb') throw new Error('objects/mirror.glb not found');
  const files = glbs.map((f) => path.join(OBJECTS, f));
  const jobs = [];
  for (const v of VIEWS) for (const light of v.lights) jobs.push({ ...v, glb: 0, light, out: `mirror-${v.name}-${light}`, file: path.join(PREVIEWS, `mirror-${v.name}-${light}.png`) });
  if (qa) {
    for (const v of QA) for (const light of ['studio', 'plain']) jobs.push({ ...v, glb: 0, light, out: `qa-${v.name}-${light}`, file: path.join(QA_DIR, `${v.name}-${light}.png`) });
    glbs.slice(1).forEach((g, k) => {
      for (const light of VIEWS[2].lights) {
        const out = `${g.replace(/\.glb$/, '')}-visitor-${light}`;
        jobs.push({ ...VISITOR, glb: k + 1, light, out, file: path.join(QA_DIR, `${out}.png`) });
      }
    });
  }
  const { a, problems } = await run(files, jobs, qa ? 'qa' : 'main');
  glbs.forEach((g, i) => {
    const v = a[`data-v${i}`] ? JSON.parse(a[`data-v${i}`]) : { error: 'validator did not run' };
    const bytes = fs.statSync(files[i]).size;
    if (v.error) problems.push(`${g} validator: ${v.error}`);
    else if (v.errors > 0 || v.warnings > 0) problems.push(`${g}: ${v.errors} validation errors, ${v.warnings} warnings`);
    console.log(`objects/${g}  ${bytes} B  validator ${v.validator}: E${v.errors} W${v.warnings} I${v.infos} H${v.hints}  `
      + `triangles ${v.triangles}  vertices ${v.vertices}  draw calls ${v.drawCalls}`);
    for (const m of v.messages || []) console.log('   ', m.join(' | '));
  });
  console.log(`bbox ${a['data-bbox']}`);
  console.log(problems.length ? `FAIL\n  ${problems.join('\n  ')}` : `ok: previews in ${path.relative(ROOT, PREVIEWS)}${qa ? ` and ${QA_DIR}` : ''}`);
  process.exitCode = problems.length ? 1 : 0;
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
