"""Builds the zhang-2024 backstage booth for FRAME: assembles the stage set (set.py) and the gilded mirror (mirror.py),
lights it in Cycles, renders hero look-dev images, bakes the lighting into unlit texture atlases and exports two GLBs.
    blender -b --factory-startup -noaudio --python-exit-code 1 -P tools/scene/build.py -- [--work DIR] [--sources DIR]
        [--stages assemble,hero,bake,export] [--rebuild-set] [--rebuild-mirror] [--hero-views a,b] [--hero-res 1920]
        [--hero-samples 384] [--bake-samples 1024] [--quick]
--work (or env SCENE_WORK) holds the intermediate .blend files and bake images; --sources holds set.blend and
mirror.blend (default: the work dir); set.py and mirror.py are run there when a file is missing or a rebuild is asked.
Stage 'compose' (instead of 'bake') rebuilds the atlases from the saved raw bakes; hero renders composite the
alpha-blended beam and glows in display space, the way FRAME blends them. Full run: about 40 min on an RTX 3070 Ti.
Outputs: scene/backstage-set.glb, objects/mirror-gilded.glb, scene/previews/hero-*.png, tools/scene/backstage.blend.
GLB previews and the glTF Validator: node tools/scene/preview.mjs.
Coordinates: glTF metres (Y up, audience +Z, origin = stage centre on the floor); Blender = (x, -z, y)."""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import bmesh
import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORK = None
T0 = time.time()

FLOOR_Y = 0.02
MIRROR_AT = (0.0, FLOOR_Y, -0.65)
LEAN = math.radians(5.0)
MIRROR_FRONT_Z = 0.0706
VIEW = {'transform': 'AgX', 'look': 'AgX - Medium High Contrast', 'exposure': 0.0}
CAMERAS = {
    'audience': ((0.0, 1.6, 5.5), (0.0, 1.3, -0.5), 26.0),
    'three': ((-1.7, 1.45, 2.85), (0.1, 1.18, -0.62), 28.0),
    'inside': ((0.3, 1.6, 1.4), (0.0, 1.05, -0.62), 18.0),
    'approach': ((3.5, 1.6, 6.0), (0.0, 1.25, -0.45), 26.0),
}


def log(*a):
    print('[build %6.1fs]' % (time.time() - T0), *a, flush=True)


def options():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default=os.environ.get('SCENE_WORK') or os.path.join(tempfile.gettempdir(), 'xr-museum-scene-work'))
    ap.add_argument('--sources', default=None)
    ap.add_argument('--stages', default='assemble,hero,bake,export')
    ap.add_argument('--rebuild-set', action='store_true')
    ap.add_argument('--rebuild-mirror', action='store_true')
    ap.add_argument('--hero-views', default='audience,three,inside,approach')
    ap.add_argument('--hero-res', type=int, default=1920)
    ap.add_argument('--hero-samples', type=int, default=384)
    ap.add_argument('--hero-tag', default='')
    ap.add_argument('--hero-out', default=None)
    ap.add_argument('--bake-samples', type=int, default=1024)
    ap.add_argument('--quick', action='store_true')
    o = ap.parse_args(argv)
    o.work = os.path.abspath(o.work)
    o.sources = os.path.abspath(o.sources or o.work)
    return o


def B(p):
    p = np.asarray(p, float)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)


def bvec(p):
    return Vector(B(p).tolist())


def G(p):
    p = np.asarray(p, float)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def hexc(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


# ---------------------------------------------------------------- Cycles
def use_gpu(sc):
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for kind in ('OPTIX', 'CUDA'):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
            if any(d.type == kind for d in prefs.devices):
                for d in prefs.devices:
                    d.use = d.type == kind
                sc.cycles.device = 'GPU'
                return kind
        except Exception:
            pass
    sc.cycles.device = 'CPU'
    return 'CPU'


def set_view(sc, raw=False):
    vs = sc.view_settings
    if raw:
        vs.view_transform = 'Standard'
        vs.look = 'None'
        vs.exposure = 0.0
    else:
        vs.view_transform = VIEW['transform']
        vs.look = VIEW['look']
        vs.exposure = VIEW['exposure']
    vs.gamma = 1.0


def render_settings(sc, samples, denoise=True):
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.01
    sc.cycles.use_denoising = denoise
    try:
        sc.cycles.denoiser = 'OPTIX'
    except Exception:
        pass
    sc.cycles.max_bounces = 10
    sc.cycles.diffuse_bounces = 4
    sc.cycles.glossy_bounces = 4
    sc.cycles.transmission_bounces = 4
    sc.cycles.transparent_max_bounces = 32
    sc.cycles.volume_bounces = 0
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.cycles.sample_clamp_direct = 0.0
    sc.cycles.sample_clamp_indirect = 10.0
    sc.cycles.blur_glossy = 0.5
    sc.render.film_transparent = False
    sc.render.resolution_percentage = 100


# ---------------------------------------------------------------- sources
def run_blender(script, args):
    cmd = [bpy.app.binary_path, '-b', '--factory-startup', '-noaudio', '--python-exit-code', '1', '-P', script, '--'] + args
    log('running', os.path.basename(script), ' '.join(args))
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError('%s failed (%d)' % (script, r.returncode))


def ensure_sources(o):
    os.makedirs(o.sources, exist_ok=True)
    set_blend = os.path.join(o.sources, 'set.blend')
    mirror_blend = os.path.join(o.sources, 'mirror.blend')
    if o.rebuild_set or not os.path.exists(set_blend):
        run_blender(os.path.join(HERE, 'set.py'), ['--out', set_blend])
    if o.rebuild_mirror or not os.path.exists(mirror_blend):
        run_blender(os.path.join(HERE, 'mirror.py'), ['--work', o.sources, '--views', ''])
    return set_blend, mirror_blend


# ---------------------------------------------------------------- assemble
def collection(name, parent=None):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(c)
    return c


def append_mirror(mirror_blend):
    with bpy.data.libraries.load(mirror_blend, link=False) as (src, dst):
        dst.collections = [c for c in src.collections if c in ('Mirror', 'Mirror HP')]
        dst.texts = [t for t in src.texts if t == 'Mirror README']
    for c in dst.collections:
        bpy.context.scene.collection.children.link(c)
    root = bpy.data.objects['Mirror_Root']
    root.location = bvec(MIRROR_AT)
    return root


def look_at(ob, target_b):
    d = Vector(target_b) - ob.location
    ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


def add_light(coll, name, kind, pos, aim_dir, energy, kelvin, **kw):
    ld = bpy.data.lights.new(name, kind)
    ld.energy = energy
    ld.use_temperature = True
    ld.temperature = kelvin
    ld.color = (1, 1, 1)
    for k, v in kw.items():
        setattr(ld, k, v)
    ob = bpy.data.objects.new(name, ld)
    coll.objects.link(ob)
    ob.location = bvec(pos)
    look_at(ob, bvec(np.asarray(pos) + np.asarray(aim_dir)))
    return ob


LIGHTS = {}
LIGHT_POWER = {'key': 1700.0, 'studio': 80.0, 'foh': 35.0, 'leg': 1000.0, 'bk': 150.0, 'frc': 50.0}
GOLD_CARD = 2.4
VELVET_LIFT = (0.0, 0.011, 0.026)


def build_lights():
    for ob in list(bpy.data.objects):
        if ob.type == 'LIGHT':
            bpy.data.objects.remove(ob, do_unlink=True)
    coll = collection('Lighting')
    key = bpy.data.objects['Lantern_Key']
    lens, aim = np.array(key['lens_center_gltf']), unit(key['aim_gltf'])
    LIGHTS['key'] = (lens, aim, float(key['lens_radius']))
    add_light(coll, 'Key', 'SPOT', lens + 0.03 * aim, aim, LIGHT_POWER['key'], 3300, spot_size=math.radians(28), spot_blend=0.32,
              shadow_soft_size=0.085)
    st = bpy.data.objects['Studio_Light']
    face, fwd = np.array(st['diffuser_center_gltf']), unit(st['aim_gltf'])
    add_light(coll, 'Studio', 'AREA', face + 0.012 * fwd, fwd, LIGHT_POWER['studio'], 4300, shape='DISK', size=0.42, spread=math.radians(100))
    add_light(coll, 'FOH', 'AREA', (0.0, 3.75, 3.9), unit(np.array([0.0, 1.35, -0.9]) - np.array([0.0, 3.75, 3.9])),
              LIGHT_POWER['foh'], 3600, shape='RECTANGLE', size=5.2, size_y=1.2)
    for side, s in (('L', -1), ('R', 1)):
        pos, tgt = np.array([0.95 * s, 3.85, 4.7]), np.array([2.2 * s, 1.55, -0.22])
        add_light(coll, 'FOH_' + side, 'SPOT', pos, unit(tgt - pos), LIGHT_POWER['leg'], 4000,
                  spot_size=math.radians(34), spot_blend=0.85, shadow_soft_size=0.25)
    for nm, tgt, power, half, blend in (('Lantern_BK', (0.0, 1.85, -2.25), LIGHT_POWER['bk'], 26, 0.75),
                                         ('Lantern_FRC', (2.35, 1.0, -0.2), LIGHT_POWER['frc'], 24, 0.7)):
        L = bpy.data.objects[nm]
        lc, la = np.array(L['lens_center_gltf']), unit(L['aim_gltf'])
        add_light(coll, 'Wash_' + nm.split('_')[1], 'SPOT', lc + 0.03 * la, unit(np.asarray(tgt) - lc), power, 3000,
                  spot_size=math.radians(2 * half), spot_blend=blend, shadow_soft_size=0.08)
    w = bpy.context.scene.world
    bg = w.node_tree.nodes.get('Background')
    bg.inputs['Color'].default_value = (0.0016, 0.0014, 0.0013, 1)
    bg.inputs['Strength'].default_value = 1.0
    return coll


def velvet_lift():
    """Lifts the crimson velvet's green/blue a touch, so lit ridges read crimson rather than pure red."""
    m = bpy.data.materials['Velvet_Crimson']
    nt = m.node_tree
    b = next(n for n in nt.nodes if n.bl_idname == 'ShaderNodeBsdfPrincipled')
    if nt.nodes.get('VelvetLift'):
        return
    bc = b.inputs['Base Color']
    src = bc.links[0].from_socket
    mix = nt.nodes.new('ShaderNodeMix')
    mix.name = 'VelvetLift'
    mix.data_type = 'RGBA'
    mix.blend_type = 'ADD'
    sk = {x.identifier: x for x in mix.inputs}
    sk['Factor_Float'].default_value = 1.0
    sk['B_Color'].default_value = VELVET_LIFT + (1.0,)
    nt.links.new(src, sk['A_Color'])
    nt.links.new([x for x in mix.outputs if x.identifier == 'Result_Color'][0], bc)


def gold_card():
    """A dim warm card in front of the stage that only the gilding sees in its reflections (light linking): from
    the audience the frame's front faces would otherwise mirror the dark house and read brown, not gold."""
    coll = collection('Lighting')
    ob = bpy.data.objects.get('Gold_Card')
    if ob is None:
        me = bpy.data.meshes.new('Gold_Card')
        w, h = 3.6, 2.2
        me.from_pydata([(-w / 2, 0, -h / 2), (w / 2, 0, -h / 2), (w / 2, 0, h / 2), (-w / 2, 0, h / 2)], [], [(0, 1, 2, 3)])
        ob = bpy.data.objects.new('Gold_Card', me)
        coll.objects.link(ob)
        m = bpy.data.materials.new('Gold_Card')
        nt = m.node_tree
        nt.nodes.clear()
        out = nt.nodes.new('ShaderNodeOutputMaterial')
        em = nt.nodes.new('ShaderNodeEmission')
        em.name = 'Emit'
        bb = nt.nodes.new('ShaderNodeBlackbody')
        bb.inputs['Temperature'].default_value = 4200
        nt.links.new(bb.outputs[0], em.inputs['Color'])
        nt.links.new(em.outputs[0], out.inputs['Surface'])
        me.materials.append(m)
    ob.location = bvec((0.0, 1.9, 4.2))
    ob.rotation_euler = (0.0, 0.0, 0.0)
    ob.data.materials[0].node_tree.nodes['Emit'].inputs['Strength'].default_value = GOLD_CARD
    for a in ('visible_camera', 'visible_diffuse', 'visible_shadow', 'visible_transmission', 'visible_volume_scatter'):
        setattr(ob, a, False)
    ob.visible_glossy = True
    rc = bpy.data.collections.get('Gold Card Receivers') or bpy.data.collections.new('Gold Card Receivers')
    for nm in ('Mirror_Frame_HP', 'Mirror_Frame'):
        o = bpy.data.objects[nm]
        if o.name not in rc.objects:
            rc.objects.link(o)
    ob.light_linking.receiver_collection = rc
    return ob


def build_cameras():
    coll = collection('Cameras')
    for name, (pos, tgt, lens) in CAMERAS.items():
        cd = bpy.data.cameras.get('Cam_' + name) or bpy.data.cameras.new('Cam_' + name)
        cd.lens = lens
        cd.sensor_width = 36.0
        cd.sensor_fit = 'HORIZONTAL'
        cd.clip_start = 0.02
        cd.clip_end = 100.0
        ob = bpy.data.objects.get('Cam_' + name) or bpy.data.objects.new('Cam_' + name, cd)
        if ob.name not in coll.objects:
            coll.objects.link(ob)
        ob.location = bvec(pos)
        look_at(ob, bvec(tgt))
    return coll


# ---------------------------------------------------------------- beam (fins + dust motes) and its texture
BLEND_TEX = 512
FIN_PX = (64, 128)
FIN_N = 6
MOTE_BOX = (384, 384, 64)
BEAM_HALF = math.radians(13.0)
FIN_HALF = math.radians(17.5)
BEAM_R0 = 0.085
BEAM_ALPHA = 0.036
BEAM_RGB = hexc('#f6e4c8')
GLASS_SILVER = 0.035
MOTE_RGB = hexc('#fff4e2')


def fin_frame(k):
    lens, aim, _ = LIGHTS['key']
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = unit(np.cross(aim, e1))
    phi = math.pi * k / FIN_N
    return lens, aim, math.cos(phi) * e1 + math.sin(phi) * e2


def fin_point(k, s, v):
    """Point on fin k: s in [-1, 1] across (the fin is a little wider than the lit cone, for a faint halo), v in [0, 1]
    from the lens to the floor along each generator."""
    lens, aim, d = fin_frame(k)
    tn = math.tan(FIN_HALF)
    s = np.asarray(s, float)
    v = np.asarray(v, float)
    gen = aim + (s[..., None] * tn) * d
    base = lens + (s[..., None] * BEAM_R0) * d
    t_floor = (FLOOR_Y + 0.002 - base[..., 1]) / gen[..., 1]
    return base + (v * t_floor)[..., None] * gen, t_floor


def mirror_geometry():
    n = np.array([0.0, math.sin(LEAN), math.cos(LEAN)])
    q0 = np.array([0.0, FLOOR_Y, MIRROR_AT[2] + MIRROR_FRONT_Z])
    up = np.array([0.0, math.cos(LEAN), -math.sin(LEAN)])
    return q0, n, up


def haze_shadow(P):
    """Light reaching points P from the key lens past the mirror (soft, lens-sized penumbra), 1 = unshadowed."""
    lens, aim, rl = LIGHTS['key']
    q0, n, up = mirror_geometry()
    dpl = (lens - q0) @ n
    dpp = (P - q0) @ n
    behind = dpp < 0
    t = dpl / np.maximum(dpl - dpp, 1e-6)
    M = lens + (P - lens) * t[..., None]
    mx = M[..., 0]
    my = (M - q0) @ up
    pen = rl * np.clip(-dpp, 0, None) / max(dpl, 1e-3) + 0.01
    ex = smoothstep(-pen, pen, 0.45 - np.abs(mx))
    ey = smoothstep(-pen, pen, my) * smoothstep(-pen, pen, 2.05 - my)
    return 1.0 - behind * ex * ey, dpp


def fin_alpha(k, s, v):
    P, t_floor = fin_point(k, s, v)
    lens = LIGHTS['key'][0]
    dist = np.linalg.norm(P - lens, axis=-1)
    aim = LIGHTS['key'][1]
    ax = (P - lens) @ aim
    rad = np.linalg.norm(P - lens - ax[..., None] * aim, axis=-1)
    sc = rad / (BEAM_R0 + np.maximum(ax, 0) * math.tan(BEAM_HALF))
    prof = 0.9 * np.exp(-(sc / 0.8) ** 4) + 0.1 * np.exp(-(sc / 1.25) ** 2)
    prof = prof * np.clip(1 - np.abs(s) ** 8, 0, 1)
    along = (0.62 + 0.38 * np.exp(-dist / 2.4)) * smoothstep(1.0, 0.8, v) * smoothstep(0.0, 0.08, v)
    sh, dpp = haze_shadow(P)
    q0, n, up = mirror_geometry()
    my = (P - q0) @ up
    inside = smoothstep(0.62, 0.42, np.abs(P[..., 0])) * smoothstep(2.3, 2.02, my)
    near = 1.0 - inside * (1.0 - smoothstep(0.0, 0.45, dpp))
    return BEAM_ALPHA * prof * along * sh * near


def beam_texture(shadow_rgba=None):
    """512 px BLEND atlas: rows 0-383 contact shadow, rows 384-511 four fin profiles (64 x 128 each) and a mote sprite."""
    T = BLEND_TEX
    tex = np.zeros((T, T, 4), np.float32)
    if shadow_rgba is not None:
        tex[:384, :T] = shadow_rgba
    fw, fh = FIN_PX
    for k in range(FIN_N):
        cols = (np.arange(fw) + 0.5) / fw
        rows = (np.arange(fh) + 0.5) / fh
        S, V = np.meshgrid(cols * 2 - 1, rows)
        a = fin_alpha(k, S, V)
        blk = np.zeros((fh, fw, 4), np.float32)
        blk[..., :3] = BEAM_RGB
        blk[..., 3] = a
        tex[384:384 + fh, k * fw:(k + 1) * fw] = blk
    x0, y0, ms = MOTE_BOX
    yy, xx = np.mgrid[0:ms, 0:ms]
    r = np.hypot(xx + 0.5 - ms / 2, yy + 0.5 - ms / 2) / (ms / 2)
    tex[y0:y0 + ms, x0:x0 + ms, :3] = MOTE_RGB
    tex[y0:y0 + ms, x0:x0 + ms, 3] = np.clip(1 - r, 0, 1) ** 2.2 * 0.38
    return tex


def tex_uv(col, row):
    """Texel-space (x right, y down from the top) -> Blender UV."""
    return np.stack([np.asarray(col, float) / BLEND_TEX, 1.0 - np.asarray(row, float) / BLEND_TEX], -1)


def beam_mesh(rng):
    V, F, UV = [], [], []
    fw, fh = FIN_PX
    ns, nv = 9, 16
    for k in range(FIN_N):
        s = np.linspace(-1, 1, ns)
        v = np.linspace(0, 1, nv) ** 1.25
        S, Vv = np.meshgrid(s, v)
        P, _ = fin_point(k, S, Vv)
        base = sum(len(x) for x in V)
        V.append(P.reshape(-1, 3))
        cols = k * fw + 0.5 + (S + 1) / 2 * (fw - 1)
        rows = 384 + 0.5 + Vv * (fh - 1)
        UV.append(tex_uv(cols, rows).reshape(-1, 2))
        for i in range(nv - 1):
            for j in range(ns - 1):
                a, b, c, d = i * ns + j, i * ns + j + 1, (i + 1) * ns + j + 1, (i + 1) * ns + j
                F += [[base + a, base + b, base + c], [base + a, base + c, base + d]]
    x0, y0, ms = MOTE_BOX
    quad_uv = tex_uv([x0 + 1, x0 + ms - 1, x0 + ms - 1, x0 + 1], [y0 + ms - 1, y0 + ms - 1, y0 + 1, y0 + 1])
    motes, tries = [], 0
    while len(motes) < 18 and tries < 5000:
        tries += 1
        k = rng.integers(FIN_N)
        s, v = rng.uniform(-0.55, 0.55), rng.uniform(0.06, 0.7) ** 1.15
        lens, aim, d = fin_frame(k)
        e = unit(np.cross(aim, d))
        P, _ = fin_point(k, s, v)
        P = P + e * rng.uniform(-0.5, 0.5) * (BEAM_R0 + np.linalg.norm(P - lens) * math.tan(BEAM_HALF)) * 0.7
        sh, dpp = haze_shadow(P[None])
        q0, n, up = mirror_geometry()
        if sh[0] < 0.9 or (abs(P[0]) < 0.7 and 0 < dpp[0] < 0.3 and P[1] < 2.3):
            continue
        motes.append((P, rng.uniform(0.003, 0.008)))
    for P, r in motes:
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        for ax in range(3):
            a1, a2 = R[:, ax], R[:, (ax + 1) % 3]
            base = sum(len(x) for x in V)
            V.append(np.array([P - a1 * r - a2 * r, P + a1 * r - a2 * r, P + a1 * r + a2 * r, P - a1 * r + a2 * r]))
            UV.append(quad_uv)
            F += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return np.concatenate(V), np.array(F), np.concatenate(UV)


def image_from_array(name, arr, path, colorspace='sRGB', pack=False):
    """Writes arr (rows top-down, values 0..1, RGB or RGBA) to an 8-bit PNG at path and returns it as a file image."""
    h, w = arr.shape[:2]
    a = np.clip(np.asarray(arr, np.float32), 0, 1)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], 2)
    tmp = bpy.data.images.new('_tmp_' + name, w, h, alpha=True)
    tmp.colorspace_settings.name = 'Non-Color'
    tmp.alpha_mode = 'STRAIGHT'
    tmp.pixels.foreach_set(a[::-1].ravel())
    tmp.filepath_raw = path
    tmp.file_format = 'PNG'
    tmp.save()
    bpy.data.images.remove(tmp)
    old = bpy.data.images.get(name)
    if old is not None:
        old.name = '_old_' + name
    img = bpy.data.images.load(path, check_existing=False)
    img.name = name
    img.colorspace_settings.name = colorspace
    img.alpha_mode = 'STRAIGHT'
    if old is not None:
        old.user_remap(img)
        bpy.data.images.remove(old)
    if pack:
        img.pack()
    return img


def mat_blend_unlit(name, img):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    tx = nt.nodes.new('ShaderNodeTexImage')
    tx.image = img
    tx.interpolation = 'Linear'
    uv = nt.nodes.new('ShaderNodeUVMap')
    uv.uv_map = 'UVMap'
    nt.links.new(uv.outputs['UV'], tx.inputs['Vector'])
    mix = nt.nodes.new('ShaderNodeMixShader')
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    em = nt.nodes.new('ShaderNodeEmission')
    nt.links.new(tx.outputs['Color'], em.inputs['Color'])
    nt.links.new(tx.outputs['Alpha'], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs['Surface'])
    m.use_backface_culling = False
    try:
        m.surface_render_method = 'BLENDED'
    except Exception:
        pass
    return m


def mesh_object(name, coll, V_gltf, F, UV, mats, parent=None):
    me = bpy.data.meshes.get(name)
    if me is not None:
        for ob in list(bpy.data.objects):
            if ob.data == me:
                bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.meshes.remove(me)
    me = bpy.data.meshes.new(name)
    me.from_pydata(B(V_gltf).tolist(), [], np.asarray(F).tolist())
    lay = me.uv_layers.new(name='UVMap')
    lay.data.foreach_set('uv', np.asarray(UV, np.float32)[np.asarray(F).ravel()].ravel())
    for m in mats:
        me.materials.append(m)
    me.update()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    if parent is not None:
        ob.parent = parent
        ob.matrix_parent_inverse = parent.matrix_world.inverted()
    return ob


def build_beam():
    tex = beam_texture()
    img = image_from_array('mirror_blend', tex, os.path.join(WORK, 'mirror_blend.png'), 'Non-Color', pack=True)
    mat = mat_blend_unlit('Mirror Blend', img)
    V, F, UV = beam_mesh(np.random.default_rng(5))
    ob = mesh_object('Mirror_Beam', bpy.data.collections['Mirror'], V, F, UV, [mat])
    for a in ('visible_diffuse', 'visible_glossy', 'visible_shadow', 'visible_transmission', 'visible_volume_scatter'):
        setattr(ob, a, False)
    log('beam: %d tris' % len(F))
    return ob


GLOW_E = 1 / 2.4
GLOW_RGB = hexc('#ffedd2')


def glow_texture(T=128):
    """Radial glow sprite: a faint veil over the emitter, a soft halo just outside its rim, nothing at the edge."""
    yy, xx = np.mgrid[0:T, 0:T]
    r = np.hypot(xx + 0.5 - T / 2, yy + 0.5 - T / 2) / (T / 2)
    halo = 0.17 * np.exp(-((r - GLOW_E) / 0.16) ** 2) + 0.07 * np.exp(-((r - GLOW_E) / 0.42) ** 2)
    a = np.where(r < GLOW_E, 0.06 + 0.18 * (r / GLOW_E) ** 6, halo)
    a = a * np.clip((1 - r) / 0.25, 0, 1) ** 1.5
    tex = np.zeros((T, T, 4), np.float32)
    tex[..., :3] = GLOW_RGB
    tex[..., 3] = a
    return tex


def disk_mesh(centre, normal, radius, seg=40):
    n = unit(normal)
    a = unit(np.cross(n, [0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.cross(n, [1.0, 0.0, 0.0]))
    b = np.cross(n, a)
    ang = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    ring = centre + radius * (np.cos(ang)[:, None] * a + np.sin(ang)[:, None] * b)
    V = np.concatenate([centre[None], ring])
    F = [[0, 1 + i, 1 + (i + 1) % seg] for i in range(seg)]
    uv = np.concatenate([[[0.5, 0.5]], 0.5 + 0.5 * np.stack([np.cos(ang), np.sin(ang)], 1)])
    return V, np.array(F), uv


def build_glows():
    """Glow cards for the two lights that face visitors (FRAME has no bloom): the studio softlight and the key lens."""
    img = image_from_array('set_glow', glow_texture(), os.path.join(WORK, 'set_glow.png'), 'Non-Color', pack=True)
    mat = mat_blend_unlit('Set Glow', img)
    mat.use_backface_culling = True
    st = bpy.data.objects['Studio_Light']
    face, fwd = np.array(st['diffuser_center_gltf']), unit(st['aim_gltf'])
    lens, aim, rl = LIGHTS['key']
    Vs, Fs, UVs, n = [], [], [], 0
    for c, nrm, rad in ((face + 0.006 * fwd, fwd, 0.21 / GLOW_E), (lens + 0.004 * aim, aim, rl / GLOW_E)):
        V, F, uv = disk_mesh(np.asarray(c), np.asarray(nrm), rad)
        Vs.append(V)
        Fs.append(F + n)
        UVs.append(uv)
        n += len(V)
    ob = mesh_object('Set_Glows', bpy.data.collections['Set'], np.concatenate(Vs), np.concatenate(Fs),
                     np.concatenate(UVs), [mat])
    for a in ('visible_diffuse', 'visible_glossy', 'visible_shadow', 'visible_transmission', 'visible_volume_scatter'):
        setattr(ob, a, False)
    return ob


BLEND_OBJECTS = ('Mirror_Beam', 'Set_Glows')


def assemble(o):
    global WORK
    WORK = o.work
    set_blend, mirror_blend = ensure_sources(o)
    bpy.ops.wm.open_mainfile(filepath=set_blend)
    sc = bpy.context.scene
    append_mirror(mirror_blend)
    build_lights()
    velvet_lift()
    gold_card()
    build_cameras()
    for nm in list(bpy.data.collections.keys()):
        if nm == 'LookDev':
            c = bpy.data.collections[nm]
            for ob in list(c.objects):
                if ob.type == 'CAMERA':
                    c.objects.unlink(ob)
                    collection('Cameras').objects.link(ob)
    build_beam()
    build_glows()
    hero_glass()
    bpy.data.objects['Mirror_Frame'].hide_render = True
    bpy.data.objects['Mirror_Frame_HP'].hide_render = False
    sc.camera = bpy.data.objects['Cam_audience']
    render_settings(sc, o.hero_samples)
    set_view(sc)
    path = os.path.join(o.work, 'assembled.blend')
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)
    log('saved', path)


def load_lights_info():
    key = bpy.data.objects['Lantern_Key']
    LIGHTS['key'] = (np.array(key['lens_center_gltf']), unit(key['aim_gltf']), float(key['lens_radius']))


# ---------------------------------------------------------------- view transform helpers
_AGX = {}


def agx_apply(lin, path=None):
    """Scene-linear RGB (rows top-down) -> display values through the scene's view transform (16-bit PNG round trip)."""
    sc = bpy.context.scene
    set_view(sc)
    h, w = lin.shape[:2]
    path = path or os.path.join(WORK, '_agx.png')
    img = bpy.data.images.new('_agx', w, h, alpha=False, float_buffer=True)
    a = np.concatenate([np.asarray(lin, np.float32)[..., :3], np.ones((h, w, 1), np.float32)], 2)
    img.pixels.foreach_set(a[::-1].ravel())
    st = sc.render.image_settings
    old = (st.file_format, st.color_depth, st.color_mode)
    st.file_format, st.color_depth, st.color_mode = 'PNG', '16', 'RGB'
    img.save_render(path, scene=sc)
    st.file_format, st.color_depth, st.color_mode = old
    bpy.data.images.remove(img)
    return read_png(path)[..., :3]


def agx_curve():
    if 'x' not in _AGX:
        x = np.geomspace(1e-6, 256.0, 4096)
        y = agx_apply(np.repeat(x[None, :, None], 3, 2))[0, :, 0]
        _AGX['x'], _AGX['y'] = x, np.maximum.accumulate(y + np.arange(len(y)) * 1e-9)
    return _AGX['x'], _AGX['y']


def agx_inverse(disp, iters=4):
    """Display RGB -> scene-linear RGB whose view transform gives disp (per-channel start, refined in log space)."""
    cx, cy = agx_curve()
    lx = np.log(cx)
    slope = np.gradient(cy, lx)
    d = np.clip(np.asarray(disp, float), cy[0], cy[-1])
    x = np.exp(np.interp(d, cy, lx))
    for _ in range(iters):
        y = agx_apply(x)
        k = np.maximum(np.interp(np.log(x), lx, slope), 0.02)
        x = np.exp(np.clip(np.log(x) + (d - y) / k, lx[0], lx[-1]))
    return x


def save_float(name, arr, path):
    h, w = arr.shape[:2]
    img = bpy.data.images.new(name + '_tmp', w, h, alpha=True, float_buffer=True)
    a = np.asarray(arr, np.float32)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.ones((h, w, 1), np.float32)], 2)
    img.pixels.foreach_set(a[::-1].ravel())
    sc = bpy.context.scene
    st = sc.render.image_settings
    old_fmt = (st.file_format, st.color_depth, st.color_mode)
    st.file_format, st.color_depth, st.color_mode = 'OPEN_EXR', '32', 'RGBA'
    img.save_render(path, scene=sc)
    st.file_format, st.color_depth, st.color_mode = old_fmt
    bpy.data.images.remove(img)
    old = bpy.data.images.get(name)
    if old is not None:
        old.name = '_old_' + name
    img = bpy.data.images.load(path, check_existing=False)
    img.name = name
    if old is not None:
        old.user_remap(img)
        bpy.data.images.remove(old)
    return img


def glass_display_map():
    img = bpy.data.images['mirror_glass_unlit']
    w, h = img.size
    a = np.array(img.pixels[:], np.float32).reshape(h, w, 4)[::-1, :, :3]
    if img.is_float:
        a = np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(np.maximum(a, 0), 1 / 2.4) - 0.055)
    return a


def hero_glass():
    """Look-dev glass for the hero renders: the finished glass map as emission, pre-compensated so that the view
    transform shows it exactly as authored (what the unlit GLB shows), under a dim foxed silver and a glass coat."""
    disp = glass_display_map()
    emit = agx_inverse(disp)
    eimg = save_float('mirror_glass_emit', emit, os.path.join(WORK, 'mirror_glass_emit.exr'))
    eimg.pack()
    masks = bpy.data.images['mirror_glass_masks']
    m = bpy.data.materials.get('Mirror Glass Hero') or bpy.data.materials.new('Mirror Glass Hero')
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    uv = nt.nodes.new('ShaderNodeUVMap')
    uv.uv_map = 'UVMap'
    te = nt.nodes.new('ShaderNodeTexImage')
    te.image = eimg
    te.interpolation = 'Cubic'
    tm = nt.nodes.new('ShaderNodeTexImage')
    tm.image = masks
    tm.interpolation = 'Cubic'
    for t in (te, tm):
        nt.links.new(uv.outputs['UV'], t.inputs['Vector'])
    sep = nt.nodes.new('ShaderNodeSeparateColor')
    nt.links.new(tm.outputs['Color'], sep.inputs['Color'])
    mix = nt.nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    sk = {x.identifier: x for x in mix.inputs}
    nt.links.new(sep.outputs['Green'], sk['Factor_Float'])
    sk['A_Color'].default_value = (GLASS_SILVER, GLASS_SILVER, GLASS_SILVER * 0.94, 1)
    sk['B_Color'].default_value = (GLASS_SILVER * 0.35, GLASS_SILVER * 0.32, GLASS_SILVER * 0.25, 1)
    rough = nt.nodes.new('ShaderNodeMath')
    rough.operation = 'MULTIPLY_ADD'
    nt.links.new(sep.outputs['Green'], rough.inputs[0])
    rough.inputs[1].default_value = 0.25
    rough.inputs[2].default_value = 0.16
    bs = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new([x for x in mix.outputs if x.identifier == 'Result_Color'][0], bs.inputs['Base Color'])
    bs.inputs['Metallic'].default_value = 1.0
    nt.links.new(rough.outputs[0], bs.inputs['Roughness'])
    bs.inputs['Coat Weight'].default_value = 1.0
    bs.inputs['Coat Roughness'].default_value = 0.14
    bs.inputs['Coat IOR'].default_value = 1.52
    em = nt.nodes.new('ShaderNodeEmission')
    nt.links.new(te.outputs['Color'], em.inputs['Color'])
    em.inputs['Strength'].default_value = 1.0
    add = nt.nodes.new('ShaderNodeAddShader')
    nt.links.new(bs.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs['Surface'])
    g = bpy.data.objects['Mirror_Glass']
    g.data.materials[0] = m
    return m


# ---------------------------------------------------------------- hero renders
def read_exr(path):
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    a = np.array(img.pixels[:], np.float32).reshape(h, w, 4)[::-1]
    bpy.data.images.remove(img)
    return a


def read_png(path):
    img = bpy.data.images.load(path, check_existing=False)
    img.colorspace_settings.name = 'Non-Color'
    w, h = img.size
    a = np.array(img.pixels[:], np.float32).reshape(h, w, 4)[::-1]
    bpy.data.images.remove(img)
    return a


def write_png(path, arr):
    h, w = arr.shape[:2]
    img = bpy.data.images.new('_out', w, h, alpha=False)
    img.colorspace_settings.name = 'Non-Color'
    a = np.clip(arr, 0, 1).astype(np.float32)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], 2)
    img.pixels.foreach_set(a[::-1].ravel())
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)


def beam_layer(sc, cam, path_exr, samples=48):
    """Renders only the beam (everything else held out) with raw, display-referred colours, for compositing in
    display space exactly as an alpha-blended unlit mesh is drawn by FRAME."""
    saved = {}
    for ob in sc.objects:
        if ob.type == 'MESH' and ob.name not in BLEND_OBJECTS:
            saved[ob.name] = ob.is_holdout
            ob.is_holdout = True
    for nm in BLEND_OBJECTS:
        if nm in bpy.data.objects:
            bpy.data.objects[nm].hide_render = False
    st = (sc.render.film_transparent, sc.cycles.samples, sc.cycles.use_denoising, sc.render.image_settings.file_format,
          sc.render.image_settings.color_depth, sc.render.image_settings.color_mode)
    sc.render.film_transparent = True
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False
    set_view(sc, raw=True)
    sc.render.image_settings.file_format = 'OPEN_EXR'
    sc.render.image_settings.color_depth = '32'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.camera = cam
    sc.render.filepath = path_exr
    bpy.ops.render.render(write_still=True)
    for nm, v in saved.items():
        sc.objects[nm].is_holdout = v
    (sc.render.film_transparent, sc.cycles.samples, sc.cycles.use_denoising, sc.render.image_settings.file_format,
     sc.render.image_settings.color_depth, sc.render.image_settings.color_mode) = st
    set_view(sc)
    return read_exr(path_exr)


def hero(o, out_dir=None, tag=''):
    bpy.ops.wm.open_mainfile(filepath=os.path.join(o.work, 'assembled.blend'))
    load_lights_info()
    sc = bpy.context.scene
    kind = use_gpu(sc)
    render_settings(sc, o.hero_samples)
    set_view(sc)
    out_dir = out_dir or os.path.join(ROOT, 'scene', 'previews')
    tmp = os.path.join(o.work, 'hero-tmp')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp, exist_ok=True)
    res = o.hero_res
    sc.render.resolution_x, sc.render.resolution_y = res, int(round(res * 9 / 16))
    beam = bpy.data.objects['Mirror_Beam']
    bpy.data.objects['Mirror_Frame'].hide_render = True
    for v in [v for v in o.hero_views.split(',') if v]:
        cam = bpy.data.objects['Cam_' + v]
        for nm in BLEND_OBJECTS:
            if nm in bpy.data.objects:
                bpy.data.objects[nm].hide_render = True
        sc.camera = cam
        sc.render.image_settings.file_format = 'PNG'
        sc.render.image_settings.color_depth = '8'
        sc.render.image_settings.color_mode = 'RGB'
        base = os.path.join(tmp, 'base-%s%s.png' % (v, tag))
        sc.render.filepath = base
        t = time.time()
        bpy.ops.render.render(write_still=True)
        log('hero %s %.1fs (%s)' % (v, time.time() - t, kind))
        bg = read_png(base)[..., :3]
        lay = beam_layer(sc, cam, os.path.join(tmp, 'beam-%s%s.exr' % (v, tag)))
        out = lay[..., :3] + bg * (1.0 - lay[..., 3:4])
        log('beam layer max alpha %.3f' % float(lay[..., 3].max()))
        write_png(os.path.join(out_dir, 'hero-%s%s.png' % (v, tag)), out)
    beam.hide_render = False


# ---------------------------------------------------------------- bake: groups, UV atlases
BACKED = ('Curtain_Leg_L', 'Curtain_Leg_R', 'Curtain_Valance')
DECIMATE = {'Curtain_Back': 0.62, 'Drape_Wall_L': 0.7, 'Drape_Wall_R': 0.7}
SPEC_LIFT = {'Floor_Planks': (0.032, 0.03, 0.028), 'Velour_Black': (0.011, 0.010, 0.009), 'Velvet_Crimson': (0.010, 0.0015, 0.0025), 'Box_Inside': (0.004, 0.0035, 0.003)}
ATLAS = {'floor': (2048, 1536), 'stage': (2048, 2048), 'props': (2048, 2048), 'mirror': (2048, 2048), 'glass': (1024, 2048)}
FINAL = {'floor': (2048, 1536), 'stage': (2048, 2048), 'props': (2048, 2048), 'mirror': (1448, 1448), 'glass': (512, 1024)}
WEIGHT = {'Curtain_Leg_L_Rear': 0.6, 'Curtain_Leg_R_Rear': 0.6, 'Curtain_Valance_Rear': 0.5, 'Box_In': 0.22, 'Box_Out': 0.16, 'Drape_Wall_L': 0.38, 'Drape_Wall_R': 0.38, 'Drape_Front_L': 0.45,
          'Drape_Front_R': 0.45, 'Truss': 1.25, 'Curtain_Back': 0.9}
METAL_TINT = 0.72


def group_objects():
    """Bake/export groups (all export objects are meshes with identity or yaw transforms)."""
    stage = ['Box_In', 'Box_Out', 'Curtain_Leg_L', 'Curtain_Leg_R', 'Curtain_Back', 'Curtain_Valance', 'Drape_Wall_L',
             'Drape_Wall_R', 'Drape_Front_L', 'Drape_Front_R', 'Truss'] + sorted(o.name for o in bpy.data.objects
                                                                               if o.name.startswith('Lantern_'))
    stage += [n + '_Rear' for n in BACKED if n + '_Rear' in bpy.data.objects]
    props = sorted(o.name for o in bpy.data.collections['Props'].all_objects if o.type == 'MESH')
    return {'floor': ['Floor'], 'stage': stage, 'props': props, 'mirror': ['Mirror_Frame'], 'glass': ['Mirror_Glass']}


def curtain_backs():
    """Single-sided copies of the curtains facing the other way, so each side gets its own bake (the front side's
    light no longer shows on the back). Both copies are single-sided, so the coincident surfaces never fight."""
    for nm in BACKED:
        if nm + '_Rear' in bpy.data.objects:
            continue
        src = bpy.data.objects[nm]
        me = src.data.copy()
        me.name = nm + '_Rear'
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:], flip_multires=False)
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new(nm + '_Rear', me)
        ob.matrix_world = src.matrix_world
        ob.hide_render = True
        for c in src.users_collection:
            c.objects.link(ob)


def decimate_heavy():
    for nm, ratio in DECIMATE.items():
        ob = bpy.data.objects[nm]
        if ob.get('decimated'):
            continue
        m = ob.modifiers.new('Decimate', 'DECIMATE')
        m.decimate_type = 'COLLAPSE'
        m.ratio = ratio
        m.use_collapse_triangulate = True
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        ob.modifiers.clear()
        old = ob.data
        ob.data = me
        me.name = old.name
        bpy.data.meshes.remove(old)
        ob['decimated'] = ratio


def spec_lift(on):
    """View-independent stand-in for the sheen and satin gloss the bake loses: a little neutral (or tinted) albedo."""
    for nm, lift in SPEC_LIFT.items():
        m = bpy.data.materials.get(nm)
        if m is None:
            continue
        nt = m.node_tree
        b = next(n for n in nt.nodes if n.bl_idname == 'ShaderNodeBsdfPrincipled')
        bc = b.inputs['Base Color']
        node = nt.nodes.get('BakeSpecLift')
        if on and node is None:
            node = nt.nodes.new('ShaderNodeMix')
            node.name = 'BakeSpecLift'
            node.data_type = 'RGBA'
            node.blend_type = 'ADD'
            sk = {x.identifier: x for x in node.inputs}
            sk['Factor_Float'].default_value = 1.0
            sk['B_Color'].default_value = tuple(lift) + (1.0,)
            if bc.is_linked:
                nt.links.new(bc.links[0].from_socket, sk['A_Color'])
            else:
                sk['A_Color'].default_value = bc.default_value
            nt.links.new([x for x in node.outputs if x.identifier == 'Result_Color'][0], bc)
        elif not on and node is not None:
            sk = {x.identifier: x for x in node.inputs}
            if sk['A_Color'].is_linked:
                nt.links.new(sk['A_Color'].links[0].from_socket, bc)
            else:
                bc.default_value = sk['A_Color'].default_value
            nt.nodes.remove(node)


def split_box():
    if 'Box_Out' in bpy.data.objects:
        return
    ob = bpy.data.objects['Box_Shell']
    for nm, mat in (('Box_Out', 'Box_Outside'), ('Box_In', 'Box_Inside')):
        me = ob.data.copy()
        bm = bmesh.new()
        bm.from_mesh(me)
        idx = [i for i, m in enumerate(me.materials) if m.name == mat][0]
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.material_index != idx], context='FACES')
        bm.to_mesh(me)
        bm.free()
        me.name = nm
        o2 = bpy.data.objects.new(nm, me)
        o2.matrix_world = ob.matrix_world
        for c in ob.users_collection:
            c.objects.link(o2)
    ob.hide_render = True
    ob.hide_set(True)


def mesh_area(me):
    a = np.zeros(len(me.polygons))
    me.polygons.foreach_get('area', a)
    return a


def uv_tri_area(me, layer):
    uv = np.zeros(len(me.loops) * 2)
    me.uv_layers[layer].data.foreach_get('uv', uv)
    uv = uv.reshape(-1, 2)
    st = np.zeros(len(me.polygons), np.int64)
    tot = np.zeros(len(me.polygons), np.int64)
    me.polygons.foreach_get('loop_start', st)
    me.polygons.foreach_get('loop_total', tot)
    area = np.zeros(len(me.polygons))
    for k in range(1, int(tot.max()) - 1):
        ok = tot > k + 1
        a, b, c = uv[st[ok]], uv[st[ok] + k], uv[st[ok] + k + 1]
        area[ok] += 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return area, uv


def bake_layer(ob):
    me = ob.data
    lay = me.uv_layers.get('Bake') or me.uv_layers.new(name='Bake')
    src = me.uv_layers.get('UVMap')
    if src is not None:
        src.active_render = True
    me.uv_layers.active = lay
    return lay


def copy_uv(ob, src='UVMap', dst='Bake'):
    me = ob.data
    uv = np.zeros(len(me.loops) * 2)
    me.uv_layers[src].data.foreach_get('uv', uv)
    me.uv_layers[dst].data.foreach_set('uv', uv)


def edit_mode(objs):
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.hide_set(False)
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.select_all(action='SELECT')


def object_mode():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


def normalise_density(ob, weight):
    """Scales the object's Bake UVs so UV area / surface area = weight^2 (metres), i.e. equal texel density."""
    me = ob.data
    a3 = mesh_area(me) * float(np.linalg.det(np.asarray(ob.matrix_world)[:3, :3])) ** (2 / 3)
    a2, uv = uv_tri_area(me, 'Bake')
    k = weight * math.sqrt(a3.sum() / max(a2.sum(), 1e-12))
    uv = (uv - uv.mean(0)) * k
    me.uv_layers['Bake'].data.foreach_set('uv', uv.ravel())


def pack(objs, margin_px, size, rotate=True):
    edit_mode(objs)
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.uv.pack_islands(udim_source='CLOSEST_UDIM', rotate=rotate, rotate_method='ANY', scale=True,
                            merge_overlap=False, margin_method='FRACTION', margin=margin_px / max(size),
                            shape_method='CONCAVE')
    object_mode()


def unwrap_group(names, size, smart, margin_px=6):
    objs = [bpy.data.objects[n] for n in names]
    for ob in objs:
        bake_layer(ob)
    keep = [o for o in objs if o.name not in smart]
    for o in keep:
        copy_uv(o)
    sp = [o for o in objs if o.name in smart]
    if sp:
        edit_mode(sp)
        bpy.ops.uv.smart_project(angle_limit=math.radians(62), margin_method='FRACTION', island_margin=0.002,
                                 area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
        object_mode()
    for ob in objs:
        normalise_density(ob, WEIGHT.get(ob.name, 1.0))
    t = time.time()
    pack(objs, margin_px, size)
    log('packed %d objects in %.1fs' % (len(objs), time.time() - t))


def floor_uv(size):
    """Floor top: exact metric map over rows 32..1535 of the 2048 x 1536 atlas; the four 2 cm edge strips get
    their own 6 px rows in rows 0..31."""
    W, H = size
    ob = bpy.data.objects['Floor']
    me = ob.data
    bake_layer(ob)
    co = np.zeros(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    co = G(co.reshape(-1, 3))
    lv = np.zeros(len(me.loops), np.int64)
    me.loops.foreach_get('vertex_index', lv)
    st = np.zeros(len(me.polygons), np.int64)
    tot = np.zeros(len(me.polygons), np.int64)
    me.polygons.foreach_get('loop_start', st)
    me.polygons.foreach_get('loop_total', tot)
    nrm = np.zeros(len(me.polygons) * 3)
    me.polygons.foreach_get('normal', nrm)
    nrm = G(nrm.reshape(-1, 3))
    uv = np.zeros((len(me.loops), 2))
    x0, x1, z0, z1 = co[:, 0].min(), co[:, 0].max(), co[:, 2].min(), co[:, 2].max()
    top_rows = (32, H)
    rows_edge = {(0, 0, 1): 1, (0, 0, -1): 9, (-1, 0, 0): 17, (1, 0, 0): 25}
    for f in range(len(me.polygons)):
        L = np.arange(st[f], st[f] + tot[f])
        P = co[lv[L]]
        n = np.round(nrm[f]).astype(int)
        if n[1] > 0:
            u = (P[:, 0] - x0) / (x1 - x0)
            r = top_rows[0] + (z1 - P[:, 2]) / (z1 - z0) * (top_rows[1] - top_rows[0])
            uv[L] = np.stack([u, 1 - r / H], 1)
        else:
            row = rows_edge.get(tuple(n), 1)
            along = P[:, 0] if n[0] == 0 else P[:, 2]
            span = (x0, x1) if n[0] == 0 else (z0, z1)
            u = 2 / W + (along - span[0]) / (span[1] - span[0]) * (1 - 4 / W)
            r = row + (P[:, 1] / FLOOR_Y) * 5.0
            uv[L] = np.stack([u, 1 - r / H], 1)
    me.uv_layers['Bake'].data.foreach_set('uv', uv.ravel())
    return (x0, x1, z0, z1), top_rows


# ---------------------------------------------------------------- bake: materials and passes
def bake_target_node(m):
    nt = m.node_tree
    n = nt.nodes.get('BakeTarget')
    if n is None:
        n = nt.nodes.new('ShaderNodeTexImage')
        n.name = n.label = 'BakeTarget'
        n.location = (600, 400)
    return n


def set_targets(objs, img):
    for ob in objs:
        for s in ob.material_slots:
            if s.material is None:
                continue
            n = bake_target_node(s.material)
            n.image = img
            for o in s.material.node_tree.nodes:
                o.select = False
            n.select = True
            s.material.node_tree.nodes.active = n


def demetal(mats):
    """Bake-time view-independent stand-in for metals: a matte surface in the metal's own tint."""
    saved = []
    for m in mats:
        b = next((n for n in m.node_tree.nodes if n.bl_idname == 'ShaderNodeBsdfPrincipled'), None)
        if b is None:
            continue
        met = b.inputs['Metallic']
        if met.is_linked or met.default_value < 0.5:
            continue
        saved.append((m, met.default_value))
        met.default_value = 0.0
        tint = m.node_tree.nodes.new('ShaderNodeMix')
        tint.name = 'BakeTint'
        tint.data_type = 'RGBA'
        tint.blend_type = 'MULTIPLY'
        sk = {x.identifier: x for x in tint.inputs}
        sk['Factor_Float'].default_value = 1.0
        sk['B_Color'].default_value = (METAL_TINT, METAL_TINT, METAL_TINT, 1)
        bc = b.inputs['Base Color']
        if bc.is_linked:
            m.node_tree.links.new(bc.links[0].from_socket, sk['A_Color'])
        else:
            sk['A_Color'].default_value = bc.default_value
        m.node_tree.links.new([x for x in tint.outputs if x.identifier == 'Result_Color'][0], bc)
    return saved


def new_float(name, size, alpha=True):
    img = bpy.data.images.get(name)
    if img is not None:
        bpy.data.images.remove(img)
    img = bpy.data.images.new(name, size[0], size[1], alpha=alpha, float_buffer=True)
    img.colorspace_settings.name = 'Linear Rec.709'
    return img


def pixels(img):
    w, h = img.size
    return np.array(img.pixels[:], np.float32).reshape(h, w, 4)[::-1]


def bake(objs, active, img, kind, filt, samples, s2a=False, cage=0.004, ray=0.008, view_from='ABOVE_SURFACE', clear=True,
         **kw):
    sc = bpy.context.scene
    object_mode()
    sc.cycles.samples = samples
    sc.render.bake.view_from = view_from
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.hide_set(False)
        o.select_set(True)
    bpy.context.view_layer.objects.active = active
    t = time.time()
    bpy.ops.object.bake(type=kind, pass_filter=set(filt) or {'NONE'}, margin=0, use_clear=clear, target='IMAGE_TEXTURES',
                        use_selected_to_active=s2a, cage_extrusion=cage, max_ray_distance=ray, uv_layer='Bake', **kw)
    log('bake %-8s %-22s %s %4d spp %.1fs' % (kind, ','.join(sorted(filt)), img.name, samples, time.time() - t))
    return pixels(img)


def denoise(rgb, albedo=None, normal=None):
    """OpenImageDenoise through the compositor (HDR, optional albedo / normal guides), on an empty scene."""
    h, w = rgb.shape[:2]
    srcs = []

    def fimg(nm, arr):
        im = bpy.data.images.new(nm, w, h, alpha=True, float_buffer=True)
        a = np.concatenate([np.asarray(arr, np.float32)[..., :3], np.ones((h, w, 1), np.float32)], 2)
        im.pixels.foreach_set(a[::-1].ravel())
        srcs.append(im)
        return im
    sc = bpy.data.scenes.get('_denoise') or bpy.data.scenes.new('_denoise')
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = w, h, 100
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = 1
    sc.view_settings.view_transform = 'Standard'
    key = '_denoise_tree' + ('_a' if albedo is not None else '') + ('_n' if normal is not None else '')
    ng = bpy.data.node_groups.get(key)
    if ng is None:
        ng = bpy.data.node_groups.new(key, 'CompositorNodeTree')
        ng.interface.new_socket('Image', in_out='OUTPUT', socket_type='NodeSocketColor')
        dn = ng.nodes.new('CompositorNodeDenoise')
        dn.inputs['HDR'].default_value = True
        for nm, sock in (('in', 'Image'), ('alb', 'Albedo'), ('nrm', 'Normal')):
            if (nm == 'alb' and albedo is None) or (nm == 'nrm' and normal is None):
                continue
            ni = ng.nodes.new('CompositorNodeImage')
            ni.name = nm
            ng.links.new(ni.outputs['Image'], dn.inputs[sock])
        go = ng.nodes.new('NodeGroupOutput')
        ng.links.new(dn.outputs['Image'], go.inputs[0])
    ng.nodes['in'].image = fimg('_dn_in', rgb)
    if albedo is not None:
        ng.nodes['alb'].image = fimg('_dn_alb', albedo)
    if normal is not None:
        ng.nodes['nrm'].image = fimg('_dn_nrm', normal)
    sc.compositing_node_group = ng
    sc.render.use_compositing = True
    st = sc.render.image_settings
    st.file_format, st.color_depth, st.color_mode = 'OPEN_EXR', '32', 'RGBA'
    bpy.ops.render.render(scene=sc.name)
    p = os.path.join(WORK, '_dn.exr')
    bpy.data.images['Render Result'].save_render(p, scene=sc)
    out = read_exr(p)[..., :3]
    for im in srcs:
        bpy.data.images.remove(im)
    return out


def push_pull(rgb, mask):
    """Fills uncovered texels from coarser levels of a masked image pyramid (keeps mip levels free of black)."""
    levels = []
    c = rgb * mask[..., None]
    m = mask.astype(np.float32)
    while True:
        levels.append((c, m))
        h, w = m.shape
        if h <= 1 and w <= 1:
            break
        ph, pw = h + (h & 1), w + (w & 1)
        cp = np.zeros((ph, pw, c.shape[2]), np.float32)
        mp = np.zeros((ph, pw), np.float32)
        cp[:h, :w], mp[:h, :w] = c, m
        c = cp.reshape(ph // 2, 2, pw // 2, 2, -1).sum((1, 3))
        m = mp.reshape(ph // 2, 2, pw // 2, 2).sum((1, 3))
    fill = levels[-1][0] / np.maximum(levels[-1][1], 1e-8)[..., None]
    for c, m in reversed(levels[:-1]):
        h, w = m.shape
        up = np.repeat(np.repeat(fill, 2, 0), 2, 1)[:h, :w]
        own = c / np.maximum(m, 1e-8)[..., None]
        wgt = np.clip(m, 0, 1)[..., None]
        fill = own * wgt + up * (1 - wgt)
    return np.where(mask[..., None] > 0.5, rgb, fill)


def resize(rgb, size):
    """Box/area resample with Blender's image scaler (used only for downsizing bakes)."""
    h, w = rgb.shape[:2]
    img = bpy.data.images.new('_rs', w, h, alpha=True, float_buffer=True)
    a = np.concatenate([rgb[..., :3], np.ones((h, w, 1), np.float32)], 2)
    img.pixels.foreach_set(a[::-1].ravel())
    img.scale(size[0], size[1])
    out = pixels(img)[..., :3]
    bpy.data.images.remove(img)
    return out


def save_display(name, disp, ext='png', quality=95):
    """8-bit sRGB-encoded (display) image file in WORK/bake, loaded back as an sRGB file image."""
    path = os.path.join(WORK, 'bake', name + '.' + ext)
    h, w = disp.shape[:2]
    a = np.clip(disp, 0, 1).astype(np.float32)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.ones((h, w, 1), np.float32)], 2)
    img = bpy.data.images.new('_sd', w, h, alpha=True)
    img.colorspace_settings.name = 'Non-Color'
    img.pixels.foreach_set(a[::-1].ravel())
    img.filepath_raw = path
    img.file_format = 'JPEG' if ext == 'jpg' else 'PNG'
    if ext == 'jpg':
        img.save(quality=quality)
    else:
        img.save()
    bpy.data.images.remove(img)
    old = bpy.data.images.get(name)
    if old is not None:
        old.name = '_old_' + name
    out = bpy.data.images.load(path, check_existing=False)
    out.name = name
    out.colorspace_settings.name = 'sRGB'
    out.use_fake_user = True
    if old is not None:
        old.user_remap(out)
        bpy.data.images.remove(old)
    return out


def save_raw(name, arr):
    return save_float(name, arr, os.path.join(WORK, 'bake', name + '.exr'))


def hide_mirror(hidden):
    if 'Set_Glows' in bpy.data.objects:
        bpy.data.objects['Set_Glows'].hide_render = True
    for nm in ('Mirror_Frame_HP', 'Mirror_Glass', 'Mirror_Beam', 'Mirror_Frame', 'Mirror_Shadow'):
        ob = bpy.data.objects.get(nm)
        if ob is not None:
            ob.hide_render = hidden or nm in ('Mirror_Beam', 'Mirror_Shadow', 'Mirror_Frame')


def set_bake_lighting(sc, gallery):
    """gallery=True: the white FRAME gallery around the booth (for the outside of the box), stage lights off."""
    for ob in bpy.data.collections['Lighting'].objects:
        ob.hide_render = gallery
    bg = sc.world.node_tree.nodes.get('Background')
    bg.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1) if gallery else (0.0016, 0.0014, 0.0013, 1)
    g = bpy.data.objects.get('_gallery_floor')
    if gallery and g is None:
        me = bpy.data.meshes.new('_gallery_floor')
        me.from_pydata([(-30, -30, 0), (30, -30, 0), (30, 30, 0), (-30, 30, 0)], [], [(0, 1, 2, 3)])
        g = bpy.data.objects.new('_gallery_floor', me)
        sc.collection.objects.link(g)
        m = bpy.data.materials.new('_gallery_white')
        b = m.node_tree.nodes.get('Principled BSDF')
        b.inputs['Base Color'].default_value = (0.75, 0.75, 0.74, 1)
        me.materials.append(m)
    if g is not None:
        g.hide_render = not gallery


def bake_group(name, names, samples):
    """Bakes light (diffuse direct + indirect, denoised), albedo and emission for a group; returns linear arrays."""
    sc = bpy.context.scene
    objs = [bpy.data.objects[n] for n in names]
    size = ATLAS[name]
    out = {}
    for tag, kind, filt, spp in (('light', 'DIFFUSE', ('DIRECT', 'INDIRECT'), samples),
                                 ('albedo', 'DIFFUSE', ('COLOR',), 16),
                                 ('emit', 'EMIT', (), 16)):
        img = new_float('bk_%s_%s' % (name, tag), size)
        set_targets(objs, img)
        spec_lift(tag == 'albedo')
        fronts = [o for o in objs if not o.name.endswith('_Rear')]
        backs = [o for o in objs if o.name.endswith('_Rear')]
        for o in backs:
            o.hide_render = True
        out[tag] = bake(fronts, fronts[0], img, kind, filt, spp)
        if backs:
            srcs = [bpy.data.objects[o.name[:-5]] for o in backs]
            for o, src in zip(backs, srcs):
                o.hide_render, src.hide_render = False, True
            out[tag] = bake(backs, backs[0], img, kind, filt, spp, clear=False)
            for o, src in zip(backs, srcs):
                o.hide_render, src.hide_render = True, False
    spec_lift(False)
    out['mask'] = (out['albedo'][..., 3] > 0.5).astype(np.float32)
    return out


def compose(name, parts, light_key='light'):
    lit = parts[light_key][..., :3]
    dn = denoise(push_pull(lit, parts['mask']))
    lin = parts['albedo'][..., :3] * dn + parts['emit'][..., :3]
    return lin, dn


def write_atlas(name, lin, mask, ext='jpg', override=None):
    disp = agx_apply(lin)
    if override is not None:
        m, col = override
        disp = np.where(m[..., None], col, disp)
    size = FINAL[name]
    if (disp.shape[1], disp.shape[0]) != tuple(size):
        filled = push_pull(disp, mask)
        disp = resize(filled, size)
        mask = resize(np.repeat(mask[..., None], 3, 2), size)[..., 0] > 0.5
    disp = push_pull(disp, mask.astype(np.float32))
    img = save_display('atlas_' + name, disp, ext)
    log('atlas', name, img.size[:], ext)
    return img, disp


MIRROR_VIEW = (0.0, 1.62, 2.3)


def view_camera(name, pos, target):
    cam = bpy.data.objects.get(name)
    if cam is None:
        cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
        bpy.data.collections['Cameras'].objects.link(cam)
    cam.location = bvec(pos)
    look_at(cam, bvec(target))
    return cam


def mirror_bake(samples):
    """Carved high-poly onto the low-poly atlas with the look-dev materials (PBR gilding, dark wood). Gold is all
    reflection, so it is baked as seen from a central visitor position; texels that face away from that view use a
    head-on (above-surface) bake instead. Position, normal and albedo passes drive the blend and the denoiser."""
    sc = bpy.context.scene
    lp, hp = bpy.data.objects['Mirror_Frame'], bpy.data.objects['Mirror_Frame_HP']
    lp.hide_render = False
    vis = {a: getattr(lp, a) for a in ('visible_diffuse', 'visible_glossy', 'visible_shadow', 'visible_transmission',
                                        'visible_volume_scatter')}
    for a in vis:
        setattr(lp, a, False)
    hp.hide_render = False
    old_cam = sc.camera
    sc.camera = view_camera('Cam_mirrorbake', MIRROR_VIEW, (0.0, 1.05, -0.62))
    size = ATLAS['mirror']
    out = {}
    comb = ('DIRECT', 'INDIRECT', 'DIFFUSE', 'GLOSSY', 'EMIT')
    for tag, kind, filt, spp, vf, kw in (
            ('view', 'COMBINED', comb, samples, 'ACTIVE_CAMERA', {}),
            ('above', 'COMBINED', comb, max(64, samples // 2), 'ABOVE_SURFACE', {}),
            ('dcol', 'DIFFUSE', ('COLOR',), 32, 'ABOVE_SURFACE', {}),
            ('gcol', 'GLOSSY', ('COLOR',), 32, 'ABOVE_SURFACE', {}),
            ('normal', 'NORMAL', (), 32, 'ABOVE_SURFACE', {'normal_space': 'OBJECT'}),
            ('pos', 'POSITION', (), 8, 'ABOVE_SURFACE', {})):
        img = new_float('bk_mirror_' + tag, size)
        set_targets([lp], img)
        out[tag] = bake([hp, lp], lp, img, kind, filt, spp, s2a=True, cage=0.004, ray=0.009, view_from=vf, **kw)
    for a, v in vis.items():
        setattr(lp, a, v)
    lp.hide_render = True
    sc.camera = old_cam
    out['mask'] = (out['dcol'][..., 3] > 0.5).astype(np.float32)
    return out


def mirror_compose(r):
    m = r['mask']
    n = r['normal'][..., :3] * 2 - 1
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    p = r['pos'][..., :3]
    eye = B(np.asarray(MIRROR_VIEW))
    v = eye[None, None] - p
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)
    w = smoothstep(0.06, 0.32, (n * v).sum(-1))[..., None]
    mix = r['view'][..., :3] * w + r['above'][..., :3] * (1 - w)
    alb = np.clip(r['dcol'][..., :3] + r['gcol'][..., :3], 0, 1)
    fill = lambda a: push_pull(a, m)
    lin = denoise(fill(mix), fill(alb), fill(n))
    return lin * MIRROR_GAIN


MIRROR_GAIN = 1.0


def glass_bake(samples):
    """The glass seen from a central visitor position: its map (emission) plus the dim foxed reflection."""
    g = bpy.data.objects['Mirror_Glass']
    sc = bpy.context.scene
    cam = bpy.data.objects.get('Cam_glassbake')
    if cam is None:
        cd = bpy.data.cameras.new('Cam_glassbake')
        cam = bpy.data.objects.new('Cam_glassbake', cd)
        bpy.data.collections['Cameras'].objects.link(cam)
    cam.location = bvec((0.0, 1.6, 2.3))
    look_at(cam, bvec((0.0, 1.05, -0.62)))
    old_cam = sc.camera
    sc.camera = cam
    bake_layer(g)
    copy_uv(g)
    me = g.data
    nrm = np.zeros(len(me.polygons) * 3)
    me.polygons.foreach_get('normal', nrm)
    uv = np.zeros(len(me.loops) * 2)
    me.uv_layers['Bake'].data.foreach_get('uv', uv)
    uv = uv.reshape(-1, 2)
    for f, n in zip(me.polygons, nrm.reshape(-1, 3)):
        if n[1] > 0.5:
            uv[list(f.loop_indices)] = (0.5 / ATLAS['glass'][0], 0.5 / ATLAS['glass'][1])
    me.uv_layers['Bake'].data.foreach_set('uv', uv.ravel())
    size = ATLAS['glass']
    img = new_float('bk_glass_combined', size)
    set_targets([g], img)
    px = bake([g], g, img, 'COMBINED', ('EMIT', 'GLOSSY', 'DIFFUSE', 'DIRECT', 'INDIRECT'), samples,
              view_from='ACTIVE_CAMERA')
    sc.camera = old_cam
    return px


def shadow_texture(fl_with, fl_without, floor_map):
    """Contact + cast shadow of the mirror on the floor, as black with alpha (display space ratio of the two bakes)."""
    (x0, x1, z0, z1), (r0, r1) = floor_map
    H, W = fl_with.shape[:2]
    rows, cols = SHADOW_PX
    X = SHADOW_RECT[0] + (np.arange(cols) + 0.5) / cols * (SHADOW_RECT[1] - SHADOW_RECT[0])
    Z = SHADOW_RECT[3] - (np.arange(rows) + 0.5) / rows * (SHADOW_RECT[3] - SHADOW_RECT[2])
    XX, ZZ = np.meshgrid(X, Z)
    fx = (XX - x0) / (x1 - x0) * W - 0.5
    fy = r0 + (z1 - ZZ) / (z1 - z0) * (r1 - r0) - 0.5

    def sample(img):
        x0i = np.clip(np.floor(fx).astype(int), 0, W - 2)
        y0i = np.clip(np.floor(fy).astype(int), 0, H - 2)
        tx, ty = (fx - x0i)[..., None], (fy - y0i)[..., None]
        a = img[y0i, x0i] * (1 - tx) + img[y0i, x0i + 1] * tx
        b = img[y0i + 1, x0i] * (1 - tx) + img[y0i + 1, x0i + 1] * tx
        return a * (1 - ty) + b * ty

    lw = sample(fl_with) @ np.array([0.2126, 0.7152, 0.0722])
    lo = sample(fl_without) @ np.array([0.2126, 0.7152, 0.0722])
    alpha = np.clip(1 - lw / np.maximum(lo, 1e-4), 0, 0.97)
    ex = np.minimum.reduce([smoothstep(0, 0.08, XX - SHADOW_RECT[0]), smoothstep(0, 0.08, SHADOW_RECT[1] - XX),
                            smoothstep(0, 0.08, ZZ - SHADOW_RECT[2]), smoothstep(0, 0.08, SHADOW_RECT[3] - ZZ)])
    alpha = alpha * ex
    rgba = np.zeros((rows, cols, 4), np.float32)
    rgba[..., :3] = SHADOW_RGB
    rgba[..., 3] = alpha
    log('shadow alpha max %.3f, mean %.4f' % (alpha.max(), alpha.mean()))
    return rgba


SHADOW_PX = (384, 512)
SHADOW_RECT = (-2.2, 1.3, -2.24, -0.32)
SHADOW_RGB = hexc('#070403')


def shadow_mesh():
    x0, x1, z0, z1 = SHADOW_RECT
    y = FLOOR_Y + 0.004
    V = np.array([(x0, y, z1), (x1, y, z1), (x1, y, z0), (x0, y, z0)])
    rows, cols = SHADOW_PX
    UV = tex_uv([0.5, cols - 0.5, cols - 0.5, 0.5], [0.5, 0.5, rows - 0.5, rows - 0.5])
    return V, np.array([[0, 1, 2], [0, 2, 3]]), UV


def bake_all(o):
    global WORK
    WORK = o.work
    os.makedirs(os.path.join(WORK, 'bake'), exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=os.path.join(o.work, 'assembled.blend'))
    load_lights_info()
    sc = bpy.context.scene
    kind = use_gpu(sc)
    render_settings(sc, o.bake_samples, denoise=False)
    sc.cycles.use_adaptive_sampling = False
    set_view(sc)
    log('bake device', kind)
    split_box()
    decimate_heavy()
    curtain_backs()
    groups = group_objects()
    for nm in groups['stage'] + groups['props'] + ['Floor']:
        bpy.data.objects[nm].hide_render = nm.endswith('_Rear')
    keep_uv = {'Curtain_Leg_L', 'Curtain_Leg_R', 'Curtain_Back', 'Curtain_Valance', 'Drape_Wall_L', 'Drape_Wall_R',
               'Drape_Front_L', 'Drape_Front_R', 'Box_In', 'Box_Out'} | {n + '_Rear' for n in BACKED}
    floor_map = floor_uv(ATLAS['floor'])
    unwrap_group(groups['stage'], ATLAS['stage'], smart=set(groups['stage']) - keep_uv)
    unwrap_group(groups['props'], ATLAS['props'], smart=set(groups['props']))
    for ob in (bpy.data.objects['Mirror_Frame'],):
        lay = bake_layer(ob)
        copy_uv(ob)
    mats = {s.material for g in ('floor', 'stage', 'props') for n in groups[g] for s in bpy.data.objects[n].material_slots
            if s.material}
    demetal(mats)
    spp = 96 if o.quick else o.bake_samples
    results = {}
    hide_mirror(True)
    set_bake_lighting(sc, gallery=False)
    for g in ('floor', 'stage', 'props'):
        results[g] = bake_group(g, groups[g], spp)
    fl_light_m = None
    hide_mirror(False)
    img = new_float('bk_floor_light_mirror', ATLAS['floor'])
    set_targets([bpy.data.objects['Floor']], img)
    fl_light_m = bake([bpy.data.objects['Floor']], bpy.data.objects['Floor'], img, 'DIFFUSE', ('DIRECT', 'INDIRECT'), spp)
    results['mirror'] = mirror_bake(spp)
    glass = glass_bake(max(64, spp // 4))
    hide_mirror(True)
    set_bake_lighting(sc, gallery=True)
    bo = bpy.data.objects['Box_Out']
    img = new_float('bk_boxout_gallery', ATLAS['stage'])
    set_targets([bo], img)
    gal = bake([bo], bo, img, 'DIFFUSE', ('DIRECT', 'INDIRECT'), max(64, spp // 4))
    set_bake_lighting(sc, gallery=False)
    hide_mirror(False)
    for g, r in results.items():
        for k in [k for k in r if k != 'mask']:
            save_raw('bk_%s_%s' % (g, k), r[k])
    save_raw('bk_floor_light_mirror', fl_light_m)
    save_raw('bk_glass_combined', glass)
    save_raw('bk_boxout_gallery', gal)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(WORK, 'baked-raw.blend'), compress=True)
    finish(o, results, fl_light_m, glass, gal, floor_map)


def floor_bounds():
    me = bpy.data.objects['Floor'].data
    co = np.zeros(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    co = G(co.reshape(-1, 3))
    return (co[:, 0].min(), co[:, 0].max(), co[:, 2].min(), co[:, 2].max()), (32, ATLAS['floor'][1])


def compose_only(o):
    """Re-runs the composition from the saved raw bakes (no re-baking)."""
    global WORK
    WORK = o.work
    bpy.ops.wm.open_mainfile(filepath=os.path.join(o.work, 'baked-raw.blend'))
    load_lights_info()
    sc = bpy.context.scene
    use_gpu(sc)
    set_view(sc)
    rd = lambda n: read_exr(os.path.join(WORK, 'bake', 'bk_' + n + '.exr'))
    results = {}
    for g in ('floor', 'stage', 'props', 'mirror'):
        keys = ('view', 'above', 'dcol', 'gcol', 'normal', 'pos') if g == 'mirror' else ('light', 'albedo', 'emit')
        r = {k: rd('%s_%s' % (g, k)) for k in keys}
        r['mask'] = (r['dcol' if g == 'mirror' else 'albedo'][..., 3] > 0.5).astype(np.float32)
        results[g] = r
    finish(o, results, rd('floor_light_mirror'), rd('glass_combined'), rd('boxout_gallery'), floor_bounds())



def finish(o, results, fl_light_m, glass, gal, floor_map):
    """Composes the bakes into display-referred atlases (view transform applied), plus the glass and shadow maps."""
    atlases = {}
    for g in ('floor', 'stage', 'props', 'mirror'):
        r = results[g]
        if g == 'mirror':
            lin = mirror_compose(r)
            img, disp = write_atlas(g, lin, r['mask'], 'jpg')
            atlases[g] = img
            continue
        lin, dn = compose(g, r)
        override = None
        if g == 'stage':
            bo = gal[..., 3] > 0.5
            e = denoise(push_pull(gal[..., :3], bo.astype(np.float32)))[..., 1]
            e = np.clip(e / max(float(np.percentile(e[bo], 90)), 1e-4), 0, 1.15)
            alb = r['albedo'][..., 1]
            var = np.clip(alb / max(float(np.median(alb[bo])), 1e-5), 0.6, 1.4) ** 0.5
            col = hexc(BOX_OUT_RGB)[None, None] * ((0.62 + 0.38 * e) * var)[..., None]
            override = (bo, col)
        if g == 'floor':
            dn_m = denoise(push_pull(fl_light_m[..., :3], r['mask']))
            a_ref = np.median(r['albedo'][..., :3][r['mask'] > 0.5], axis=0)
            floor_with = agx_apply(a_ref * dn_m)
            floor_without = agx_apply(a_ref * dn)
        img, disp = write_atlas(g, lin, r['mask'], 'jpg', override)
        atlases[g] = img
    gl = agx_apply(glass[..., :3])
    gl = resize(gl, FINAL['glass'])
    atlases['glass'] = save_display('atlas_glass', gl, 'jpg')
    sh = shadow_texture(floor_with, floor_without, floor_map)
    build_beam()
    tex = beam_texture(sh)
    img = image_from_array('atlas_blend', tex, os.path.join(WORK, 'bake', 'atlas_blend.png'), 'sRGB')
    img.use_fake_user = True
    mat = bpy.data.materials['Mirror Blend']
    for n in mat.node_tree.nodes:
        if n.bl_idname == 'ShaderNodeTexImage':
            n.image = img
    V, F, UV = shadow_mesh()
    ob = mesh_object('Mirror_Shadow', bpy.data.collections['Mirror'], V, F, UV, [mat])
    ob.hide_render = True
    path = os.path.join(WORK, 'baked.blend')
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)
    log('saved', path)


BOX_OUT_RGB = '#2a2420'


# ---------------------------------------------------------------- export
def unlit_material(name, img, blend=False, double=False):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    tx = nt.nodes.new('ShaderNodeTexImage')
    tx.image = img
    tx.interpolation = 'Linear'
    if blend:
        mix = nt.nodes.new('ShaderNodeMixShader')
        tr = nt.nodes.new('ShaderNodeBsdfTransparent')
        nt.links.new(tx.outputs['Alpha'], mix.inputs[0])
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(tx.outputs['Color'], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs['Surface'])
    else:
        nt.links.new(tx.outputs['Color'], out.inputs['Surface'])
    m.use_backface_culling = not double
    try:
        m.surface_render_method = 'BLENDED' if blend else 'DITHERED'
    except Exception:
        pass
    return m


def export_copy(sources, name, mat, coll, uv='Bake'):
    dg = bpy.context.evaluated_depsgraph_get()
    copies = []
    for ob in sources:
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        me.transform(ob.matrix_world)
        for nm in [l.name for l in me.uv_layers if l.name != uv]:
            if nm in me.uv_layers:
                me.uv_layers.remove(me.uv_layers[nm])
        me.uv_layers[uv].name = 'UVMap'
        for a in [a.name for a in me.attributes if a.domain in ('POINT', 'CORNER', 'FACE') and a.name.startswith(('Col', 'Attribute', 'uvpart'))]:
            me.attributes.remove(me.attributes[a])
        for ca in list(me.color_attributes):
            me.color_attributes.remove(ca)
        me.materials.clear()
        me.materials.append(mat)
        me.polygons.foreach_set('material_index', np.zeros(len(me.polygons), np.int32))
        c = bpy.data.objects.new(ob.name + '_x', me)
        coll.objects.link(c)
        copies.append(c)
    object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    for c in copies:
        c.select_set(True)
    bpy.context.view_layer.objects.active = copies[0]
    if len(copies) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.name = name
    ob.data.name = name
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
    bm.to_mesh(me)
    bm.free()
    return ob


def write_glb(objs, path):
    object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    for ob in objs:
        ob.hide_set(False)
        ob.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=True, export_apply=True,
                              export_yup=True, export_texcoords=True, export_normals=False, export_tangents=False,
                              export_materials='EXPORT', export_image_format='AUTO', export_jpeg_quality=92,
                              export_vertex_color='NONE', export_attributes=False, export_cameras=False,
                              export_lights=False, export_animations=False, export_extras=False, export_skins=False,
                              export_morph=False, use_active_scene=True)
    return patch_glb(path)


def read_glb(path):
    import struct
    with open(path, 'rb') as f:
        data = f.read()
    magic, ver, length = struct.unpack_from('<III', data, 0)
    jl, jt = struct.unpack_from('<II', data, 12)
    js = json.loads(data[20:20 + jl].decode('utf-8'))
    bl, bt = struct.unpack_from('<II', data, 20 + jl)
    binc = data[28 + jl:28 + jl + bl]
    return js, binc


def write_glb_file(path, js, binc):
    import struct
    j = json.dumps(js, separators=(',', ':')).encode('utf-8')
    j += b' ' * ((4 - len(j) % 4) % 4)
    b = binc + b'\0' * ((4 - len(binc) % 4) % 4)
    total = 12 + 8 + len(j) + 8 + len(b)
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, total))
        f.write(struct.pack('<II', len(j), 0x4E4F534A))
        f.write(j)
        f.write(struct.pack('<II', len(b), 0x004E4942))
        f.write(b)


def image_dims(blob):
    import struct
    if blob[:8] == b'\x89PNG\r\n\x1a\n':
        return struct.unpack('>II', blob[16:24])
    i = 2
    while i < len(blob):
        if blob[i] != 0xFF:
            i += 1
            continue
        mk = blob[i + 1]
        if mk in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack('>HH', blob[i + 5:i + 9])
            return w, h
        i += 2 + struct.unpack('>H', blob[i + 2:i + 4])[0]
    return 0, 0


MATERIAL_SPEC = {'backstage_floor': ('OPAQUE', False), 'backstage_stage': ('OPAQUE', False),
                 'backstage_props': ('OPAQUE', False),
                 'backstage_props_ds': ('OPAQUE', True), 'backstage_glow': ('BLEND', False),
                 'mirror_frame': ('OPAQUE', False), 'mirror_glass': ('OPAQUE', False), 'mirror_blend': ('BLEND', True)}


def patch_glb(path):
    """Makes sure every material is KHR_materials_unlit with its atlas as baseColor, and reports the budgets."""
    js, binc = read_glb(path)
    for m in js.get('materials', []):
        m.setdefault('extensions', {})['KHR_materials_unlit'] = {}
        pbr = m.setdefault('pbrMetallicRoughness', {})
        pbr['metallicFactor'] = 0.0
        pbr['roughnessFactor'] = 1.0
        for k in ('normalTexture', 'occlusionTexture', 'emissiveTexture', 'emissiveFactor', 'extras'):
            m.pop(k, None)
        for k in list(m['extensions']):
            if k != 'KHR_materials_unlit':
                m['extensions'].pop(k)
        mode, double = MATERIAL_SPEC.get(m.get('name', ''), ('OPAQUE', False))
        m['doubleSided'] = double
        if mode == 'BLEND':
            m['alphaMode'] = 'BLEND'
        else:
            m.pop('alphaMode', None)
            m.pop('alphaCutoff', None)
    used = set(js.get('extensionsUsed', []))
    used.add('KHR_materials_unlit')
    js['extensionsUsed'] = sorted(used)
    for k in ('cameras', 'animations', 'skins'):
        js.pop(k, None)
    keep = [i for i, sc in enumerate(js.get('scenes', [])) if sc.get('nodes')]
    if keep and len(keep) != len(js['scenes']):
        js['scene'] = keep.index(js.get('scene', keep[0])) if js.get('scene') in keep else 0
        js['scenes'] = [js['scenes'][i] for i in keep]
    js.get('extensions', {}).pop('KHR_lights_punctual', None)
    if 'extensions' in js and not js['extensions']:
        js.pop('extensions')
    for n in js.get('nodes', []):
        n.pop('extras', None)
    js['asset']['generator'] = 'tools/scene/build.py (Blender glTF exporter, post-processed)'
    write_glb_file(path, js, binc)
    tris = 0
    for mesh in js['meshes']:
        for p in mesh['primitives']:
            tris += js['accessors'][p['indices']]['count'] // 3
    gpu = 0
    dims = []
    for im in js.get('images', []):
        bv = js['bufferViews'][im['bufferView']]
        blob = binc[bv.get('byteOffset', 0):bv.get('byteOffset', 0) + bv['byteLength']]
        w, h = image_dims(blob)
        dims.append((im.get('name'), im.get('mimeType'), w, h, bv['byteLength']))
        gpu += w * h * 4 * 4 / 3
    draws = sum(len(m['primitives']) for m in js['meshes'])
    rep = dict(path=path, bytes=os.path.getsize(path), triangles=tris, draw_calls=draws, gpu_mb=gpu / 2 ** 20,
               images=dims, materials=[(m.get('name'), m.get('alphaMode', 'OPAQUE'), m.get('doubleSided', False))
                                       for m in js.get('materials', [])])
    log('glb', json.dumps(rep))
    return rep


def export_all(o):
    global WORK
    WORK = o.work
    bpy.ops.wm.open_mainfile(filepath=os.path.join(o.work, 'baked.blend'))
    load_lights_info()
    groups = group_objects()
    A = {}
    for k in ('floor', 'stage', 'props', 'mirror', 'glass', 'blend'):
        A[k] = bpy.data.images.get('atlas_' + k)
        if A[k] is None:
            f = os.path.join(WORK, 'bake', 'atlas_%s.%s' % (k, 'png' if k == 'blend' else 'jpg'))
            A[k] = bpy.data.images.load(f, check_existing=False)
            A[k].name = 'atlas_' + k
    for img in A.values():
        img.colorspace_settings.name = 'sRGB'
    xs = collection('Export_Set')
    xm = collection('Export_Mirror')
    M = {
        'floor': unlit_material('backstage_floor', A['floor']),
        'stage': unlit_material('backstage_stage', A['stage']),
        'props': unlit_material('backstage_props', A['props']),
        'props_ds': unlit_material('backstage_props_ds', A['props'], double=True),
        'mirror': unlit_material('mirror_frame', A['mirror']),
        'glass': unlit_material('mirror_glass', A['glass']),
        'blend': unlit_material('mirror_blend', A['blend'], blend=True, double=True),
        'glow': unlit_material('backstage_glow', bpy.data.images['set_glow'], blend=True, double=False),
    }
    garments = [n for n in groups['props'] if n.startswith('Garment_')]
    parts = [
        ('backstage_floor', groups['floor'], M['floor'], xs),
        ('backstage_stage', groups['stage'], M['stage'], xs),
        ('backstage_props', [n for n in groups['props'] if n not in garments], M['props'], xs),
        ('backstage_garments', garments, M['props_ds'], xs),
        ('mirror_frame', ['Mirror_Frame'], M['mirror'], xm),
        ('mirror_glass', ['Mirror_Glass'], M['glass'], xm),
    ]
    set_objs, mir_objs = [], []
    for name, src, mat, coll in parts:
        ob = export_copy([bpy.data.objects[n] for n in src], name, mat, coll)
        (set_objs if coll is xs else mir_objs).append(ob)
    for nm in ('Mirror_Beam', 'Mirror_Shadow', 'Set_Glows'):
        src = bpy.data.objects[nm]
        src.data.uv_layers.active = src.data.uv_layers['UVMap']
        lay = src.data.uv_layers.get('Bake') or src.data.uv_layers.new(name='Bake')
        copy_uv(src)
    blend_ob = export_copy([bpy.data.objects['Mirror_Shadow'], bpy.data.objects['Mirror_Beam']], 'mirror_blend',
                           M['blend'], xm)
    mir_objs.append(blend_ob)
    set_objs.append(export_copy([bpy.data.objects['Set_Glows']], 'backstage_glow', M['glow'], xs))
    os.makedirs(os.path.join(ROOT, 'scene'), exist_ok=True)
    reps = [write_glb(set_objs, os.path.join(ROOT, 'scene', 'backstage-set.glb')),
            write_glb(mir_objs, os.path.join(ROOT, 'objects', 'mirror-gilded.glb'))]
    with open(os.path.join(WORK, 'glb-report.json'), 'w', encoding='utf-8') as f:
        json.dump(reps, f, indent=1)
    for img in list(bpy.data.images):
        if img.name.startswith(('bk_', '_')) or img.name == 'Render Result':
            bpy.data.images.remove(img)
    for img in bpy.data.images:
        if img.source == 'FILE' and img.packed_file is None and os.path.exists(bpy.path.abspath(img.filepath)):
            try:
                img.pack()
            except Exception:
                pass
    path = os.path.join(HERE, 'backstage.blend')
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)
    log('saved', path)


def main():
    global WORK
    o = options()
    bpy.context.preferences.filepaths.save_version = 0
    os.makedirs(o.work, exist_ok=True)
    WORK = o.work
    stages = [s for s in o.stages.split(',') if s]
    if 'assemble' in stages:
        assemble(o)
    if 'hero' in stages:
        hero(o, o.hero_out, o.hero_tag)
    if 'bake' in stages:
        bake_all(o)
    elif 'compose' in stages:
        compose_only(o)
    if 'export' in stages:
        export_all(o)
    log('done')


if __name__ == '__main__':
    main()
