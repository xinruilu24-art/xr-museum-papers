// Renders scene/backstage-set.glb + objects/mirror-gilded.glb the way FRAME shows them (Babylon.js, unlit materials,
// textures uploaded without sRGB buffers) in headless Chrome, and runs the Khronos glTF Validator on both GLBs.
//   node tools/scene/preview.mjs [view names]      (env SCENE_PREVIEW_OUT overrides the output folder)
// Cameras are glTF coordinates (Y up, audience +Z); Babylon's glTF root mirrors X, handled below.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { ROOT, BUILD, PROFILE, chrome, chromeArgs, bodyAttrs, pngSize } from '../books/lib.mjs';

const OUT = process.env.SCENE_PREVIEW_OUT || path.join(ROOT, 'scene', 'previews');
const BJS = 'https://cdn.jsdelivr.net/npm/babylonjs@8.0.0';
const VALIDATOR = 'https://cdn.jsdelivr.net/npm/gltf-validator@2.0.0-dev.3.10/+esm';
const GLBS = {
  set: path.join(ROOT, 'scene', 'backstage-set.glb'),
  mirror: path.join(ROOT, 'objects', 'mirror-gilded.glb'),
  panel: path.join(ROOT, 'boards', 'zhang-2024-stand.glb'),
  plaque: path.join(ROOT, 'boards', 'plaque-zhang-2024-stand.glb'),
};
const vfov = (mm) => 2 * Math.atan(18 / mm * 9 / 16);
// Recommended board placement (glTF/FRAME-UI coordinates, yaw about +Y in degrees, 0 = facing the audience).
export const PLACEMENT = {
  panel: { pos: [-1.15, 0.02, 0.15], yaw: 30 },
  plaque: { pos: [-2.0, 0.0, 3.2], yaw: 15 },
};
const VIEWS = [
  { name: 'audience', eye: [0, 1.6, 5.5], target: [0, 1.3, -0.5], fov: vfov(26) },
  { name: 'three', eye: [-1.7, 1.45, 2.85], target: [0.1, 1.18, -0.62], fov: vfov(28) },
  { name: 'inside', eye: [0.3, 1.6, 1.4], target: [0, 1.05, -0.62], fov: vfov(18) },
  { name: 'approach', eye: [3.5, 1.6, 6.0], target: [0, 1.25, -0.45], fov: vfov(26), gallery: true },
  { name: 'outside-side', eye: [9.0, 2.2, 3.2], target: [0, 1.9, 0], fov: vfov(30), gallery: true },
  { name: 'outside-back', eye: [-5.5, 2.4, -9.0], target: [0, 1.9, 0], fov: vfov(30), gallery: true },
  { name: 'boards', eye: [0.4, 1.6, 6.2], target: [-0.35, 1.15, -0.3], fov: vfov(24), boards: true, gallery: true },
  { name: 'boards-inside', eye: [0.55, 1.6, 2.2], target: [-0.6, 1.2, -0.4], fov: vfov(20), boards: true },
  { name: 'frame-fov', eye: [0, 1.6, 4.2], target: [0, 1.4, -0.5], fov: 1.2, gallery: true },
  { name: 'backstage', eye: [-1.85, 1.6, -1.9], target: [-2.4, 1.3, 0.6], fov: vfov(18) },
  { name: 'turn-around', eye: [0.25, 1.55, -0.2], target: [-0.3, 2.75, 3.0], fov: vfov(18), gallery: true },
];

const PAGE_JS = `
const body = document.body;
const fail = (e) => { body.dataset.error = String((e && (e.stack || e.message)) || e); };
window.addEventListener('error', (e) => fail(e.error || e.message));
window.addEventListener('unhandledrejection', (e) => fail(e.reason));
const CFG = window.__CFG;
(async () => {
  try {
    body.dataset.version = BABYLON.Engine.Version;
    const bytes = {};
    for (const [k, url] of Object.entries(CFG.glbs)) bytes[k] = new Uint8Array(await (await fetch(url)).arrayBuffer());
    try {
      const { validateBytes, version } = await import(CFG.validator);
      const v = {};
      for (const k of ['set', 'mirror']) {
        const r = await validateBytes(bytes[k], { maxIssues: 60 });
        const s = r.issues;
        v[k] = { validator: version(), errors: s.numErrors, warnings: s.numWarnings, infos: s.numInfos, hints: s.numHints,
          messages: s.messages.map((m) => [m.severity, m.code, m.pointer || '', m.message]).slice(0, 40),
          triangles: r.info && r.info.totalTriangleCount, vertices: r.info && r.info.totalVertexCount,
          drawCalls: r.info && r.info.drawCallCount, materials: r.info && r.info.materialCount,
          extensions: r.info && r.info.extensionsUsed, resources: r.info && (r.info.resources || []).map((x) => [x.pointer, x.mimeType, x.image && x.image.width, x.image && x.image.height]) };
      }
      body.dataset.v = JSON.stringify(v);
    } catch (e) { body.dataset.validatorError = String(e && e.message || e); }
    BABYLON.SceneLoader.OnPluginActivatedObservable.add((l) => { if (l.name === 'gltf') l.useSRGBBuffers = false; });
    const canvas = document.getElementById('c');
    const engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true });
    const stats = {};
    for (const view of CFG.views) {
      const scene = new BABYLON.Scene(engine);
      scene.clearColor = view.gallery ? new BABYLON.Color4(0.93, 0.93, 0.92, 1) : new BABYLON.Color4(0, 0, 0, 1);
      const hemi = new BABYLON.HemisphericLight('hemiLight', new BABYLON.Vector3(0, 1, 0), scene);
      hemi.groundColor = BABYLON.Color3.FromHexString('#858585');
      const dir = new BABYLON.DirectionalLight('dirLight', new BABYLON.Vector3(25, -37, 30).normalize(), scene);
      const load = async (k) => {
        const blob = new Blob([bytes[k]], { type: 'model/gltf-binary' });
        const url = URL.createObjectURL(blob);
        const r = await BABYLON.SceneLoader.ImportMeshAsync('', '', url, scene, null, '.glb');
        URL.revokeObjectURL(url);
        return r;
      };
      const set = await load('set');
      const mir = await load('mirror');
      if (view.boards) {
        for (const k of ['panel', 'plaque']) {
          const r = await load(k);
          const p = CFG.placement[k];
          const root = r.meshes[0];
          const holder = new BABYLON.TransformNode('holder-' + k, scene);
          root.parent = holder;
          holder.position = new BABYLON.Vector3(-p.pos[0], p.pos[1], p.pos[2]);
          holder.rotation.y = -p.yaw * Math.PI / 180;
        }
      }
      if (view.gallery) {
        const fm = new BABYLON.StandardMaterial('gfloor', scene);
        fm.diffuseColor = BABYLON.Color3.FromHexString('#d9d8d5'); fm.specularColor = new BABYLON.Color3(0, 0, 0);
        const g = BABYLON.MeshBuilder.CreateGround('g', { width: 60, height: 60 }, scene);
        g.material = fm; g.position.y = -0.002;
      }
      const e = view.eye, t = view.target;
      const cam = new BABYLON.UniversalCamera('cam', new BABYLON.Vector3(-e[0], e[1], e[2]), scene);
      cam.setTarget(new BABYLON.Vector3(-t[0], t[1], t[2]));
      cam.fov = view.fov; cam.minZ = 0.01; cam.maxZ = 500;
      await scene.whenReadyAsync();
      const inst = new BABYLON.SceneInstrumentation(scene);
      inst.captureFrameTime = true;
      for (let i = 0; i < 3; i++) scene.render();
      stats[view.name] = { drawCalls: inst.drawCallsCounter.current, activeMeshes: scene.getActiveMeshes().length,
        meshes: [set, mir].map((r) => r.meshes.filter((m) => m.getTotalVertices() > 0).map((m) => [m.name, m.material && m.material.name,
          m.material && m.material.unlit, m.material && m.material.transparencyMode, m.material && m.material.backFaceCulling])) };
      body.dataset['shot_' + view.name.replace(/-/g, '_')] = canvas.toDataURL('image/png');
      scene.dispose();
    }
    body.dataset.stats = JSON.stringify(stats);
    body.dataset.ready = '1';
  } catch (e) { fail(e); }
})();
`;

function page(cfg, w, h) {
  return `<!doctype html><html><head><meta charset="utf-8"><title>backstage preview</title>
<style>html,body{margin:0;background:#000;overflow:hidden}</style>
<script src="${BJS}/babylon.js"></script>
<script src="${BJS.replace('babylonjs@', 'babylonjs-loaders@')}/babylonjs.loaders.min.js"></script>
</head><body><canvas id="c" width="${w}" height="${h}" style="width:${w}px;height:${h}px"></canvas>
<script>window.__CFG = ${JSON.stringify(cfg).replace(/</g, '\\u003c')};</script>
<script>${PAGE_JS}</script></body></html>`;
}

async function main() {
  const want = process.argv.slice(2);
  const views = VIEWS.filter((v) => !want.length || want.includes(v.name));
  const [W, H] = [1920, 1080];
  const glbs = Object.fromEntries(Object.entries(GLBS).map(([k, f]) => [k, pathToFileURL(f).href]));
  const file = path.join(BUILD, 'scene-preview', 'preview.html');
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.mkdirSync(OUT, { recursive: true });
  fs.mkdirSync(PROFILE, { recursive: true });
  fs.writeFileSync(file, page({ glbs, views, validator: VALIDATOR, placement: PLACEMENT }, W, H));
  const args = [...chromeArgs({ w: W, h: H, webgl: true, budget: 900000 }), '--allow-file-access-from-files',
    pathToFileURL(file).href];
  const { out } = await chrome(args, 1_500_000);
  const a = bodyAttrs(out);
  console.log('babylon', a['data-version'] || '?', a['data-ready'] === '1' ? 'ready' : 'NOT READY', a['data-error'] || '');
  for (const v of views) {
    const url = a['data-shot_' + v.name.replace(/-/g, '_')];
    if (!url) { console.log('no image for', v.name); continue; }
    const buf = Buffer.from(url.slice(url.indexOf(',') + 1), 'base64');
    const f = path.join(OUT, `glb-${v.name}.png`);
    fs.writeFileSync(f, buf);
    console.log('wrote', path.relative(ROOT, f), pngSize(buf).join('x'));
  }
  const report = { validator: a['data-v'] ? JSON.parse(a['data-v']) : { error: a['data-validator-error'] || 'not run' },
    stats: a['data-stats'] ? JSON.parse(a['data-stats']) : null, placement: PLACEMENT };
  fs.writeFileSync(path.join(BUILD, 'scene-preview', 'report.json'), JSON.stringify(report, null, 1));
  for (const [k, v] of Object.entries(report.validator)) {
    if (!v || typeof v !== 'object') { console.log('validator', k, v); continue; }
    console.log(`validator ${k}: E${v.errors} W${v.warnings} I${v.infos} H${v.hints}  tris ${v.triangles}  draws ${v.drawCalls}`);
    for (const m of v.messages || []) console.log('   ', m.join(' | '));
  }
  if (report.stats) for (const [k, s] of Object.entries(report.stats)) console.log('view', k, 'drawCalls', s.drawCalls, 'activeMeshes', s.activeMeshes);
  process.exitCode = a['data-ready'] === '1' ? 0 : 1;
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
