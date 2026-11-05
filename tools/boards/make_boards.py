# Builds boards/<name>.glb from every panels/*.png and boards/<name>-stand.glb for the abstract panels and
# plaques, checks them with tools/books/validate_glb.py, then renders boards/previews/ with preview.mjs.
# python -I -X utf8 tools/boards/make_boards.py [--no-previews] [name filters]
import io
import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, 'tools', 'books'))
import make_books as mb
import validate_glb

PANELS = os.path.join(ROOT, 'panels')
OUT = os.path.join(ROOT, 'boards')
BUILD = os.path.join(mb.BUILD, 'boards')
GENERATOR = 'xr-museum-papers tools/boards/make_boards.py'

EDGE = '#efece6'
STEEL = '#2f2d2b'
EYE = 1.6
PPD = 24.0

KINDS = {
    'panel': dict(px=(1500, 2000), size=(0.90, 1.20), tex=(1024, 2048), t=0.012, r=0.002, rb=0.001,
                  read=(1.0, 1.25, 1.5), wall=1.25),
    'plaque': dict(px=(1600, 900), size=(0.64, 0.36), tex=(1024, 512), t=0.006, r=0.0015, rb=0.0008,
                   read=(0.8, 1.0, 1.2), wall=1.25),
    'sign': dict(px=(2400, 1000), size=(2.88, 1.20), tex=(2048, 512), t=0.018, r=0.003, rb=0.0015,
                 read=(3.0, 4.5, 6.0), wall=2.3),
}
PANEL_CENTRE = 1.25
PANEL_BASE = (0.25, 0.17, 0.03)
PLAQUE_TILT = 40.0
PLAQUE_TOP = 1.05
PLAQUE_BASE = (0.16, 0.13, 0.028)
BASE_T = 0.016
POST = (0.02, 0.007)
GAP = 0.012
SHADOW_MARGIN = 0.06
WALL_SHADOW = dict(margin=0.08, drop=0.010, lift=0.005, tex=(512, 256), near=(0.32, 0.009), far=(0.12, 0.04))

X, Y, Z = np.eye(3)


def uv_xy(p):
    return (float(p[0]), float(p[1]))


def linear(hexstr):
    c = [int(hexstr[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return [round(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4, 6) for v in c]


def angles(n):
    return [(math.pi / 2) * j / n for j in range(n + 1)]


def rrect(hu, hv, R, m):
    corners = []
    for su, sv, a0 in ((1, 1, 0.0), (-1, 1, 90.0), (-1, -1, 180.0), (1, -1, 270.0)):
        for k in range(m + 1):
            a = math.radians(a0 + 90.0 * k / m)
            corners.append((su * (hu - R), sv * (hv - R), math.cos(a), math.sin(a)))
    return corners, R


def face(part, verts, uvf):
    out = []
    for p, n in verts:
        if not out or np.linalg.norm(p - out[-1][0]) > 1e-9:
            out.append((p, n))
    if len(out) > 1 and np.linalg.norm(out[0][0] - out[-1][0]) <= 1e-9:
        out.pop()
    if len(out) < 3:
        return
    pts, nrm = [p for p, _ in out], [n for _, n in out]
    part.poly(pts, [uvf(p) for p in pts], np.sum(nrm, axis=0), normals=nrm)


def sweep(part, shape, prof, frame, uvf=uv_xy):
    corners, R = shape
    U, V, N, O = frame
    rings = []
    for o, h, no, nh in prof:
        ring = []
        for cu, cv, du, dv in corners:
            u, v = cu + (R + o) * du, cv + (R + o) * dv
            hh = h(u, v) if callable(h) else h
            ring.append((O + U * u + V * v + N * hh, mb.unit(U * (no * du) + V * (no * dv) + N * nh)))
        rings.append(ring)
    for A, B in zip(rings, rings[1:]):
        for i in range(len(A)):
            j = (i + 1) % len(A)
            face(part, [A[i], A[j], B[j], B[i]], uvf)
    return rings


def cap(part, ring, centre, normal, uvf=uv_xy):
    for i in range(len(ring)):
        a, b = ring[i][0], ring[(i + 1) % len(ring)][0]
        if np.linalg.norm(a - b) > 1e-9:
            part.poly([centre, a, b], [uvf(centre), uvf(a), uvf(b)], normal)


def slab(part, hu, hv, R, T, rt, frame, m=6, bottom=True):
    U, V, N, O = frame
    prof = [(0.0, 0.0, 1.0, 0.0), (0.0, T - rt, 1.0, 0.0)]
    prof += [(-rt + rt * math.cos(a), T - rt + rt * math.sin(a), math.cos(a), math.sin(a)) for a in angles(3)[1:]]
    rings = sweep(part, rrect(hu, hv, R, m), prof, frame)
    cap(part, rings[-1], O + N * T, N)
    if bottom:
        cap(part, rings[0], O, -N)


def build_board(spec):
    W, H = spec['size']
    t, r, rb = spec['t'], spec['r'], spec['rb']
    hw, hh = W / 2, H / 2
    cy = r + hh
    O = np.array([0.0, cy, 0.0])
    prof = [(r * math.sin(a), t - r * (1 - math.cos(a)), math.sin(a), math.cos(a)) for a in angles(3)]
    prof += [(r - rb + rb * math.cos(a), rb * (1 - math.sin(a)), math.cos(a), -math.sin(a)) for a in angles(2)]
    edge = mb.Part()
    rings = sweep(edge, rrect(hw, hh, 0.0, 3), prof, (X, Y, Z, O))
    cap(edge, rings[-1], O, -Z)
    art = mb.Part()
    art.poly([(-hw, cy + hh, t), (hw, cy + hh, t), (hw, cy - hh, t), (-hw, cy - hh, t)],
             [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], Z)
    return art, edge, cy


def prism(part, shape, frame, h0, h1):
    rings = sweep(part, shape, [(0.0, h0, 1.0, 0.0), (0.0, h1, 1.0, 0.0)], frame)
    U, V, N, O = frame
    cap(part, rings[-1], O + N * h1, N)


def base_and_post(steel, base, post_top):
    bw, bd, br = base
    slab(steel, bw, bd, br, BASE_T, 0.005, (X, Z, Y, np.zeros(3)))
    slab(steel, 0.032, 0.032, 0.01, 0.006 + 0.001, 0.002, (X, Z, Y, np.array([0.0, BASE_T - 0.001, 0.0])),
         m=3, bottom=False)
    ph, pr = POST
    origin = np.array([0.0, BASE_T - 0.002, 0.0])
    if callable(post_top):
        sweep(steel, rrect(ph, ph, pr, 3), [(0.0, 0.0, 1.0, 0.0), (0.0, post_top, 1.0, 0.0)], (X, Z, Y, origin))
    else:
        slab(steel, ph, ph, pr, post_top - origin[1], 0.004, (X, Z, Y, origin), m=3, bottom=False)


def panel_stand(spec):
    W, H = spec['size']
    r = spec['r']
    steel = mb.Part()
    ph = POST[0]
    top = PANEL_CENTRE + H / 2 - 0.22
    base_and_post(steel, PANEL_BASE, top)
    for y in (PANEL_CENTRE - H / 2 + 0.16, top - 0.07):
        prism(steel, rrect(0.009, 0.009, 0.009, 3), (X, Y, Z, np.array([0.0, y, ph - 0.002])), 0.0, GAP + 0.0025)
    cy = r + H / 2
    return steel, [0.0, PANEL_CENTRE - cy, ph + GAP], None, np.eye(3), PANEL_BASE


def plaque_stand(spec):
    W, H = spec['size']
    t, r = spec['t'], spec['r']
    back = 90.0 - PLAQUE_TILT
    Rm = mb.rot_x(-back)
    cy = r + H / 2
    ty = PLAQUE_TOP - (Rm @ np.array([0.0, H + 2 * r, t]))[1]
    tz = -(Rm @ np.array([0.0, cy, 0.0]))[2]
    T = np.array([0.0, ty, tz])
    steel = mb.Part()
    plate_t = 0.006
    plate = mb.Part()
    slab(plate, 0.17, 0.08, 0.012, plate_t + 0.0003, 0.002, (X, Y, -Z, np.array([0.0, cy, 0.0003])), bottom=False)
    steel.extend(plate.transformed(Rm, T))
    n = Rm @ -Z
    p0 = T + Rm @ np.array([0.0, cy, -plate_t])

    def under(u, v):
        return p0[1] - (v - p0[2]) * n[2] / n[1] + 0.003 - (BASE_T - 0.002)

    base_and_post(steel, PLAQUE_BASE, under)
    return steel, T.tolist(), mb.quat_x(-back), Rm, PLAQUE_BASE


def unlit(g, name, texture):
    g.doc['materials'].append({
        'name': name,
        'pbrMetallicRoughness': {'baseColorTexture': {'index': texture}, 'metallicFactor': 0.0, 'roughnessFactor': 1.0},
        'extensions': {'KHR_materials_unlit': {}},
    })
    g._extensions.add('KHR_materials_unlit')
    return len(g.doc['materials']) - 1


def plain(g, name, colour, roughness):
    g.doc['materials'].append({
        'name': name,
        'pbrMetallicRoughness': {'baseColorFactor': linear(colour) + [1.0], 'metallicFactor': 0.0,
                                 'roughnessFactor': roughness},
    })
    return len(g.doc['materials']) - 1


def gpu_bytes(w, h):
    total = 0
    while True:
        total += w * h * 4
        if w == 1 and h == 1:
            return total
        w, h = max(1, w // 2), max(1, h // 2)


def kind_of(size):
    for k, spec in KINDS.items():
        if size[0] * spec['px'][1] == size[1] * spec['px'][0]:
            return k
    raise ValueError(f'unexpected artwork size {size}')


def texture_png(src, size):
    img = Image.open(src).convert('RGB')
    a = np.asarray(img).reshape(-1, 3)
    out = np.asarray(img.resize(size, Image.LANCZOS))
    img = Image.fromarray(np.clip(out, a.min(0), a.max(0)).astype(np.uint8), 'RGB')
    buf = io.BytesIO()
    img.save(buf, 'PNG', optimize=True)
    return img, buf.getvalue()


def new_glb():
    g = mb.Glb()
    g.doc['asset']['generator'] = GENERATOR
    return g


def wall_shadow(g, spec, cy):
    s = WALL_SHADOW
    r, m = spec['r'], s['margin']
    hw, hh = spec['size'][0] / 2 + r, spec['size'][1] / 2 + r
    img = mb.soft_shadow(s['tex'][0], hw, hh, m, *s['near'], *s['far'], r).resize(s['tex'], Image.BOX)
    x, y0, y1 = hw + m, cy - s['drop'] - hh - m, cy - s['drop'] + hh + m
    quad = mb.Part()
    quad.poly([(-x, y1, s['lift']), (x, y1, s['lift']), (x, y0, s['lift']), (-x, y0, s['lift'])],
              [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], Z)
    return quad, g.shadow_material('wall shadow', g.texture(mb.png(img), 'wall-shadow', mime='image/png'))


def board_mesh(g, name, spec, png, wall=False):
    art, edge, cy = build_board(spec)
    m_art = unlit(g, f'{name} artwork', g.texture(png, name, mime='image/png'))
    m_edge = plain(g, 'board edge', EDGE, 0.7)
    parts = [(art, m_art), (edge, m_edge)] + ([wall_shadow(g, spec, cy)] if wall else [])
    return g.mesh(name, parts), cy


def art_frame(spec, cy, T, Rm):
    centre = np.asarray(T, float) + Rm @ np.array([0.0, cy, spec['t']])
    return {'centre': centre.tolist(), 'normal': (Rm @ Z).tolist(), 'up': (Rm @ Y).tolist(),
            'size': list(spec['size'])}


def visitor_eye(centre, dist):
    c = np.asarray(centre, float)
    dz = math.sqrt(max(0.01, dist ** 2 - (EYE - c[1]) ** 2))
    return [float(c[0]), EYE, float(c[2] + dz)]


def views(kind, stand, art, offset, texsize):
    spec = KINDS[kind]
    W, H = spec['size']
    near, typical, far = spec['read']
    centre = (np.asarray(art['centre']) + np.asarray(offset)).tolist()
    front = {'panel': [900, 1200], 'plaque': [1280, 760], 'sign': [1600, 760]}[kind]
    if stand:
        front = {'panel': [900, 1300], 'plaque': [1100, 1100]}[kind]
    vis = {'panel': [1600, 1440], 'plaque': [1920, 1440], 'sign': [1920, 1200]}[kind]
    hang = [0.0, 0.0, 0.0] if stand else offset
    sign = kind == 'sign'
    out = [
        {'name': 'front', 'mode': 'studio', 'size': front, 'floor': stand, 'wall': sign, 'offset': hang,
         'camera': {'type': 'fit', 'dir': art['normal'], 'fov': 18, 'margin': 0.92}},
        {'name': 'angle', 'mode': 'studio', 'size': [1200, 900], 'floor': stand, 'wall': not stand, 'offset': hang,
         'camera': {'type': 'fit', 'dir': [-0.62, 0.34, 0.7], 'fov': 26, 'margin': 0.9}},
        {'name': 'side', 'mode': 'studio', 'size': [900, 900], 'save': False, 'floor': stand, 'wall': sign,
         'offset': hang, 'camera': {'type': 'fit', 'dir': [-1.0, 0.12, -0.25], 'fov': 26, 'margin': 0.9}},
    ]
    for label, d in (('visitor', typical), ('near', near), ('far', far)):
        out.append({'name': label, 'mode': 'frame', 'size': vis, 'offset': offset, 'wall': not stand,
                    'save': label == 'visitor',
                    'camera': {'type': 'eye', 'eye': visitor_eye(centre, d), 'target': centre, 'ppd': PPD,
                               'distance': d}})
    if not stand:
        out.append({'name': 'bright', 'mode': 'frame', 'size': vis, 'offset': offset, 'wall': True,
                    'wallColor': '#f4f1ea', 'wallUnlit': True, 'save': False,
                    'camera': {'type': 'eye', 'eye': visitor_eye(centre, typical), 'target': centre, 'ppd': PPD,
                               'distance': typical}})
    out.append({'name': 'check', 'mode': 'check', 'size': list(texsize), 'save': False,
                'camera': {'type': 'ortho', 'centre': art['centre'], 'normal': art['normal'], 'up': art['up'],
                           'w': W, 'h': H}})
    return out


def build(name, src, filters):
    with Image.open(src) as im:
        size = im.size
    kind = kind_of(size)
    spec = KINDS[kind]
    img, png = texture_png(src, spec['tex'])
    os.makedirs(os.path.join(BUILD, 'tex'), exist_ok=True)
    img.save(os.path.join(BUILD, 'tex', f'{name}.png'))
    gpu = gpu_bytes(*spec['tex'])
    rows = []

    g = new_glb()
    wall = kind == 'sign'
    mesh, cy = board_mesh(g, name, spec, png, wall=wall)
    g.node(name, mesh=mesh, root=True)
    path = os.path.join(OUT, f'{name}.glb')
    g.write(path)
    art = art_frame(spec, cy, [0.0, 0.0, 0.0], np.eye(3))
    offset = [0.0, spec['wall'] - cy, 0.0]
    rows.append({'name': name, 'file': path, 'kind': kind, 'stand': False, 'art': art,
                 'gpu': gpu + (gpu_bytes(*WALL_SHADOW['tex']) if wall else 0),
                 'tex': spec['tex'], 'src': src, 'views': views(kind, False, art, offset, spec['tex'])})

    if kind in ('panel', 'plaque'):
        g = new_glb()
        steel, T, q, Rm, base = (panel_stand if kind == 'panel' else plaque_stand)(spec)
        mesh, cy = board_mesh(g, name, spec, png)
        img_sh = mb.soft_shadow(256, base[0], base[1], SHADOW_MARGIN, 0.45, 0.008, 0.14, 0.03, base[2])
        quad = mb.decal_quad(base[0] + SHADOW_MARGIN, base[1] + SHADOW_MARGIN, mb.FLOOR_LIFT)
        m_steel = plain(g, 'stand', STEEL, 0.45)
        m_sh = g.shadow_material('stand contact shadow', g.texture(mb.png(img_sh), 'stand-contact-shadow',
                                                                    mime='image/png'))
        stand = g.node('stand', mesh=g.mesh('stand', [(steel, m_steel), (quad, m_sh)]))
        board = g.node('board', mesh=mesh, translation=T, rotation=q)
        g.node(f'{name}-stand', children=[stand, board], root=True)
        path = os.path.join(OUT, f'{name}-stand.glb')
        g.write(path)
        art = art_frame(spec, cy, T, Rm)
        rows.append({'name': f'{name}-stand', 'file': path, 'kind': kind, 'stand': True, 'art': art,
                     'gpu': gpu + gpu_bytes(256, 256), 'tex': spec['tex'], 'src': src,
                     'views': views(kind, True, art, [0.0, 0.0, 0.0], spec['tex'])})
    return [r for r in rows if not filters or any(f in r['name'] for f in filters)]


def compare_checks(rows):
    bad = 0
    for r in rows:
        shot = os.path.join(BUILD, 'check', f'{r["name"]}.png')
        if not os.path.exists(shot):
            print(f'FAIL {r["name"]}: no check render')
            bad += 1
            continue
        a = np.asarray(Image.open(shot).convert('RGB'), np.int16)
        b = np.asarray(Image.open(os.path.join(BUILD, 'tex', f'{os.path.basename(r["src"])[:-4]}.png')), np.int16)
        if a.shape != b.shape:
            print(f'FAIL {r["name"]}: check render {a.shape} vs texture {b.shape}')
            bad += 1
            continue
        d = np.abs(a - b)
        r['check'] = {'max': int(d.max()), 'mean': float(d.mean())}
        ok = d.max() <= 3
        bad += not ok
        print(f'{"ok  " if ok else "FAIL"} {r["name"]:28} front face vs texture: max diff {d.max()}, mean {d.mean():.3f}')
    return bad


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)
    rows = []
    for f in sorted(os.listdir(PANELS)):
        name = f[:-4]
        if f.endswith('.png') and '.' not in name and (not args or any(a in f'{name}-stand' for a in args)):
            rows += build(name, os.path.join(PANELS, f), args)
    bad = 0
    for r in rows:
        c = validate_glb.check(r['file'])
        r['bytes'], r['triangles'] = c['bytes'], c['triangles']
        r['bbox'] = (c['max'] - c['min']).tolist()
        bad += bool(c['problems'])
        print(f'{"FAIL" if c["problems"] else "ok  "} {os.path.basename(r["file"]):32} validate_glb.py  '
              f'{"; ".join(c["problems"])}')
    with open(os.path.join(BUILD, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({'boards': rows}, f, indent=1)
    if '--no-previews' not in sys.argv:
        sys.stdout.flush()
        res = subprocess.run(['node', os.path.join(HERE, 'preview.mjs')] + args, cwd=ROOT)
        bad += res.returncode != 0
        bad += compare_checks(rows)
    print(f'\n{"file":34} {"KB":>6} {"tris":>5} {"GPU MiB":>8}  size (m)')
    for r in rows:
        W, H = KINDS[r['kind']]['size']
        b = r['bbox']
        print(f'{os.path.basename(r["file"]):34} {r["bytes"] / 1024:6.0f} {r["triangles"]:5d} {r["gpu"] / 2 ** 20:8.2f}'
              f'  artwork {W:.2f} x {H:.2f}, overall {b[0]:.3f} x {b[1]:.3f} x {b[2]:.3f}')
    print(f'\n{len(rows)} files, {bad} problems.')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
