"""Cycles look-dev previews of the stage set built by set.py.
    blender -b <dir>/set.blend --factory-startup -noaudio --python-exit-code 1 -P tools/scene/set_preview.py -- --out <dir>
        [--views Cam_Audience,Cam_Visitor] [--res 960] [--samples 160] [--haze 0.012] [--no-placeholder] [--exposure 0]
        [--cam name:x,y,z:tx,ty,tz:lens ...]  (extra cameras in glTF coordinates, ';'-separated) [--gallery]
--gallery adds a white floor and a bright white world as a stand-in for the FRAME gallery around the booth.
A stand-in mirror (gilded frame, smoky glass) is added at render time only, never saved; the haze volume likewise."""
import math
import os
import sys
import time

import bpy
from mathutils import Vector

T0 = time.time()


def get_args():
    a = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    out, i = {}, 0
    while i < len(a):
        if a[i].startswith('--'):
            if i + 1 < len(a) and not a[i + 1].startswith('--'):
                out[a[i][2:]] = a[i + 1]
                i += 2
                continue
            out[a[i][2:]] = True
        i += 1
    return out


def B(x, y, z):
    return Vector((x, -z, y))


def use_gpu():
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for t in ('OPTIX', 'CUDA'):
        try:
            prefs.compute_device_type = t
            prefs.get_devices()
            if any(d.type == t for d in prefs.devices):
                for d in prefs.devices:
                    d.use = d.type == t
                bpy.context.scene.cycles.device = 'GPU'
                return t
        except Exception:
            pass
    return 'CPU'


def material(name, base, metal=0.0, rough=0.5, emit=None):
    m = bpy.data.materials.new(name)
    if m.node_tree is None:
        m.use_nodes = True
    b = m.node_tree.nodes.get('Principled BSDF')
    b.inputs['Base Color'].default_value = base + (1,)
    b.inputs['Metallic'].default_value = metal
    b.inputs['Roughness'].default_value = rough
    return m


def cube(name, coll, center, size, mat, rot=None):
    me = bpy.data.meshes.new(name)
    hx, hy, hz = (s / 2 for s in size)
    V = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz), (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    me.from_pydata(V, [], F)
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    ob.location = center
    if rot:
        ob.rotation_euler = rot
    coll.objects.link(ob)
    return ob


def bevel(ob, width, segments=2):
    m = ob.modifiers.new('Bevel', 'BEVEL')
    m.width = width
    m.segments = segments
    m.limit_method = 'NONE'
    return ob


def placeholder(coll):
    """Plain stand-in for the gilded mirror (built by another task): bevelled frame, arched crest, smoky glass."""
    gold = material('LD_Gold', (0.62, 0.43, 0.17), 1.0, 0.34)
    glass = material('LD_SmokyGlass', (0.045, 0.055, 0.05), 1.0, 0.07)
    root = bpy.data.objects.new('LD_MirrorPlaceholder', None)
    root.location = B(0, 0.02, -0.65)
    root.rotation_euler = (math.radians(-4), 0, 0)
    coll.objects.link(root)
    W, H, F, D = 0.84, 1.8, 0.085, 0.055
    parts = [((0, -0.008, H / 2), (W - 2 * F + 0.02, 0.01, H - 2 * F + 0.02), glass, 0.0),
             ((0, 0, H - F / 2), (W, D, F), gold, 0.012), ((0, 0, F / 2), (W, D, F), gold, 0.012),
             ((-(W - F) / 2, 0, H / 2), (F, D, H), gold, 0.012), (((W - F) / 2, 0, H / 2), (F, D, H), gold, 0.012),
             ((0, 0.016, H / 2), (W - 2 * F + 0.03, 0.012, H - 2 * F + 0.03), gold, 0.004)]
    for i, (c, s, m, bw) in enumerate(parts):
        ob = cube('LD_Mirror_%d' % i, coll, Vector(c), s, m)
        if bw:
            bevel(ob, bw)
        ob.parent = root
    me = bpy.data.meshes.new('LD_Crest')
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=True, radius=1.0, segments=40)
    for v in bm.verts:
        v.co.y = max(v.co.y, 0.0)
    ext = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
    for v in [e for e in ext['geom'] if isinstance(e, bmesh.types.BMVert)]:
        v.co.z += 1.0
    bm.to_mesh(me)
    bm.free()
    me.materials.append(gold)
    crest = bpy.data.objects.new('LD_Crest', me)
    crest.scale = (0.24, 0.13, D * 0.8)
    crest.rotation_euler = (math.radians(90), 0, 0)
    crest.location = (0, D * 0.4, H - 0.02)
    coll.objects.link(crest)
    crest.parent = root
    bevel(crest, 0.05, 2)
    return root


def haze(coll, density):
    m = bpy.data.materials.new('LD_Haze')
    if m.node_tree is None:
        m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    v = nt.nodes.new('ShaderNodeVolumePrincipled')
    v.inputs['Density'].default_value = density
    v.inputs['Anisotropy'].default_value = 0.55
    v.inputs['Color'].default_value = (0.9, 0.88, 0.85, 1)
    nt.links.new(v.outputs['Volume'], out.inputs['Volume'])
    cube('LD_Haze', coll, B(0, 2.29, 0), (6.28, 4.68, 4.52), m)


def main():
    args = get_args()
    out = os.path.abspath(args.get('out', os.path.join(os.path.dirname(bpy.data.filepath), 'set-previews')))
    os.makedirs(out, exist_ok=True)
    sc = bpy.context.scene
    dev = use_gpu()
    res = int(args.get('res', 960))
    sc.render.engine = 'CYCLES'
    sc.render.resolution_x = res
    sc.render.resolution_y = int(round(res * 9 / 16))
    sc.render.resolution_percentage = 100
    sc.cycles.samples = int(args.get('samples', 160))
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = 'OPTIX' if dev == 'OPTIX' else 'OPENIMAGEDENOISE'
    except Exception:
        pass
    sc.cycles.max_bounces = 8
    sc.cycles.diffuse_bounces = 4
    sc.cycles.glossy_bounces = 4
    sc.cycles.volume_bounces = 1
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.cycles.sample_clamp_indirect = 8.0
    try:
        sc.view_settings.view_transform = 'AgX'
        for look in ('AgX - Medium High Contrast', 'Medium High Contrast'):
            try:
                sc.view_settings.look = look
                break
            except Exception:
                pass
    except Exception:
        pass
    sc.view_settings.exposure = float(args.get('exposure', 0.0))
    sc.render.image_settings.file_format = 'PNG'
    tmp = bpy.data.collections.new('LD_RenderOnly')
    sc.collection.children.link(tmp)
    if not args.get('no-placeholder'):
        placeholder(tmp)
    if args.get('haze'):
        haze(tmp, float(args.get('haze')))
    if args.get('gallery'):
        white = material('LD_GalleryFloor', (0.8, 0.79, 0.77), 0.0, 0.6)
        cube('LD_Gallery', tmp, Vector((0, 0, -0.005)), (40, 40, 0.01), white)
        w = sc.world.node_tree.nodes.get('Background')
        w.inputs['Color'].default_value = (0.85, 0.85, 0.85, 1)
        w.inputs['Strength'].default_value = 1.0
    views = str(args.get('views', 'Cam_Audience,Cam_Visitor,Cam_Left,Cam_Right,Cam_Truss')).split(',')
    if args.get('cam'):
        extra = []
        for spec in str(args['cam']).split(';'):
            name, pos, tgt, lens = spec.split(':')
            pos = [float(v) for v in pos.split(',')]
            tgt = [float(v) for v in tgt.split(',')]
            cd = bpy.data.cameras.new(name)
            cd.lens = float(lens)
            cd.clip_start = 0.01
            ob = bpy.data.objects.new(name, cd)
            ob.location = B(*pos)
            ob.rotation_euler = (B(*tgt) - B(*pos)).to_track_quat('-Z', 'Y').to_euler()
            tmp.objects.link(ob)
            extra.append(name)
        views = extra if not args.get('views') else views + extra
    tag = args.get('tag', '')
    for v in views:
        cam = bpy.data.objects.get(v)
        if cam is None:
            print('missing camera', v)
            continue
        sc.camera = cam
        sc.render.filepath = os.path.join(out, (tag + '-' if tag else '') + v.replace('Cam_', '').lower() + '.png')
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print('[preview] %s %.1fs (%s)' % (sc.render.filepath, time.time() - t, dev), flush=True)


if __name__ == '__main__':
    main()
