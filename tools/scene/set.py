"""Builds the Avatars & Identity stage set for zhang-2024 (black box, plank floor, red velvet curtains, lighting truss, props).
    blender -b --factory-startup -noaudio --python-exit-code 1 -P tools/scene/set.py -- --out <dir>/set.blend [--parts set,props] [--no-sim] [--reuse-tex]
--reuse-tex loads <dir>/set-tex/floor_*.png from an earlier run instead of regenerating them (about 40 s); --no-sim skips the heap cloth simulation.
Collections: "Set" (box, floor, curtains, truss, lanterns), "Props" (left and right side dressing), "LookDev" (provisional lights, cameras).
Layout values are glTF metres (Y up, audience +Z, visitor's right +X); meshes are converted to Blender's Z-up frame as (x, -z, y).
The mirror's spot (0, 0, -0.65) stays empty. Procedural shaders read Object coordinates, or the metre-scaled UVMap
(floor, trunk slats, lenses, diffuser), so bake them before joining objects or applying rotations."""
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

T0 = time.time()

# ---------------------------------------------------------------- layout (glTF metres)
FLOOR_Y = 0.02
BOX_W, BOX_D, BOX_H, WALL = 6.4, 4.8, 4.6, 0.05
XO, ZO = BOX_W / 2, BOX_D / 2
XI, ZI, YI = XO - WALL, ZO - WALL, BOX_H - WALL
OPEN_HW, OPEN_H = 2.5, 4.0
MIRROR = (0.0, 0.0, -0.65)
KEY_APEX = (0.0, 4.1, 0.6)
KEY_TARGET = (0.0, 0.85, -0.5)
LEG_X, LEG_Z, CURTAIN_TOP = 1.65, -0.25, 4.30
BACK_Z = -2.25
VALANCE_Z, VALANCE_TOP, VALANCE_BOTTOM = 2.10, YI - 0.01, 3.60
BATTEN_Y, BATTEN_ZS = 4.15, (0.6, -1.3)
GRID_Y, GRID_XS = 4.40, (-2.55, -0.85, 0.85, 2.55)
CURTAIN_PIPE_Y = CURTAIN_TOP + 0.019


def log(*a):
    print('[set %6.1fs]' % (time.time() - T0), *a, flush=True)


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


def B(P):
    P = np.asarray(P, dtype=np.float64)
    return np.stack([P[..., 0], -P[..., 2], P[..., 1]], axis=-1)


def bvec(p):
    return Vector(B(p).tolist())


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def aim_angles(d):
    """Pan about +Y and tilt below horizontal that turn local +Z onto direction d."""
    d = unit(d)
    return math.atan2(d[0], d[2]), math.asin(float(np.clip(-d[1], -1, 1)))


def aim_matrix(pan, tilt):
    return rot_y(pan) @ rot_x(tilt)


# ---------------------------------------------------------------- mesh parts (glTF space)
class Part:
    def __init__(self, V, F, L):
        self.V = np.asarray(V, float).reshape(-1, 3)
        self.F = np.asarray(F, np.int64)
        self.L = np.asarray(L, float).reshape(self.F.shape + (2,))

    def moved(self, R=None, t=None):
        V = self.V if R is None else self.V @ np.asarray(R, float).T
        return Part(V if t is None else V + np.asarray(t, float), self.F, self.L)

    def flipped(self):
        return Part(self.V, self.F[:, ::-1], self.L[:, ::-1])

    def normal_sum(self):
        P = self.V[self.F]
        if self.F.shape[1] == 3:
            return np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]).sum(0)
        return np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1]).sum(0)

    def facing(self, d):
        return self if float(np.dot(self.normal_sum(), d)) >= 0 else self.flipped()


def merge_parts(parts):
    V, F, L, n = [], [], [], 0
    for p in parts:
        V.append(p.V)
        F.append(p.F + n)
        L.append(p.L)
        n += len(p.V)
    if len({f.shape[1] for f in F}) != 1:
        raise ValueError('merge_parts needs one face size')
    return Part(np.concatenate(V), np.concatenate(F), np.concatenate(L))


def grid_part(G, UV, wrap=False, u_period=1.0):
    nr, nc = G.shape[:2]
    cc = nc if wrap else nc - 1
    r, c = np.meshgrid(np.arange(nr - 1), np.arange(cc), indexing='ij')
    r, c = r.ravel(), c.ravel()
    c1 = (c + 1) % nc
    F = np.stack([r * nc + c, r * nc + c1, (r + 1) * nc + c1, (r + 1) * nc + c], 1)
    L = np.asarray(UV, float).reshape(-1, 2)[F].copy()
    if wrap:
        seam = c1 == 0
        L[seam, 1, 0] += u_period
        L[seam, 2, 0] += u_period
    return Part(G.reshape(-1, 3), F, L)


def tris_part(V, F, L):
    return Part(V, F, L)


def quads_to_tris(p):
    if p.F.shape[1] == 3:
        return p
    F = np.concatenate([p.F[:, [0, 1, 2]], p.F[:, [0, 2, 3]]])
    L = np.concatenate([p.L[:, [0, 1, 2]], p.L[:, [0, 2, 3]]])
    return Part(p.V, F, L)


def weld(V, F, decimals=7):
    key = np.round(V, decimals)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return V[first], inv.ravel()[F]


def rbox(half, r=0.0, n=2, nflat=1, center=(0, 0, 0)):
    """Box with rounded edges (radius r) centred on `center`; quads with outward normals and per-face UVs in metres."""
    h = np.asarray(half, float)
    r = float(min(r, *h))

    def axis_pts(hh):
        if r <= 1e-9:
            return np.linspace(-hh, hh, nflat + 1)
        band = r * np.tan(np.linspace(0, math.pi / 4, n + 1))
        pos = (hh - r) + band
        flat = np.linspace(-(hh - r), hh - r, nflat + 1)[1:-1] if hh - r > 1e-9 else np.zeros(0)
        neg = -pos[::-1]
        if hh - r <= 1e-9:
            pos = pos[1:]
        return np.concatenate([neg, flat, pos])

    ax = [axis_pts(h[i]) for i in range(3)]
    Vs, Fs, Ls, off = [], [], [], 0
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        for s in (1, -1):
            u_ax, v_ax = (b, c) if s > 0 else (c, b)
            U, W = np.meshgrid(ax[u_ax], ax[v_ax], indexing='ij')
            Q = np.zeros(U.shape + (3,))
            Q[..., a] = s * h[a]
            Q[..., u_ax] = U
            Q[..., v_ax] = W
            Cc = np.clip(Q, -(h - r), h - r)
            D = Q - Cc
            nd = np.linalg.norm(D, axis=-1, keepdims=True)
            P = np.where(nd > 1e-12, Cc + r * D / np.maximum(nd, 1e-12), Q)
            nu, nv = U.shape
            i, j = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1), indexing='ij')
            i, j = i.ravel(), j.ravel()
            F = np.stack([i * nv + j, (i + 1) * nv + j, (i + 1) * nv + j + 1, i * nv + j + 1], 1)
            Pf = P.reshape(-1, 3)
            uv = np.stack([Pf[:, u_ax], Pf[:, v_ax]], 1)
            Vs.append(Pf)
            Fs.append(F + off)
            Ls.append(uv[F])
            off += len(Pf)
    V = np.concatenate(Vs)
    F = np.concatenate(Fs)
    L = np.concatenate(Ls)
    V, F = weld(V, F)
    return Part(V + np.asarray(center, float), F, L)


def box(lo, hi, r=0.0, n=1):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return rbox((hi - lo) / 2, r, n, 1, (hi + lo) / 2)


def fan(center, ring, outward):
    """Triangle fan closing a ring of points; `outward` picks the winding."""
    ring = np.asarray(ring, float)
    k = len(ring)
    V = np.concatenate([np.asarray(center, float)[None], ring])
    j = np.arange(k)
    F = np.stack([np.zeros(k, np.int64), 1 + (j + 1) % k, 1 + j], 1)
    p = Part(V, F, np.zeros((k, 3, 2)))
    uvv = (V - V[0])[:, :2]
    p.L = uvv[F]
    return p.facing(outward)


def lathe(prof, seg=24, phase=0.0):
    """Surface of revolution about local +Z from (r, z) points ordered back-axis -> rim -> front-axis; outward normals."""
    prof = np.asarray(prof, float)
    th = phase + np.linspace(0, 2 * math.pi, seg, endpoint=False)
    k = len(prof)
    G = np.stack([prof[:, :1] * np.cos(th), prof[:, :1] * np.sin(th), np.repeat(prof[:, 1:2], seg, 1)], -1)
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))])
    UV = np.stack(np.meshgrid(arc, np.arange(seg) / seg, indexing='ij')[::-1], -1)
    p = grid_part(G, UV, wrap=True, u_period=1.0)
    tri = quads_to_tris(p)
    P = tri.V[tri.F]
    area = np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    keep = area > 1e-12
    return Part(tri.V, tri.F[keep], tri.L[keep])


def frames(P, up=(0, 1, 0)):
    P = np.asarray(P, float)
    T = np.gradient(P, axis=0)
    T = unit(T)
    a = np.asarray(up, float)
    if abs(float(np.dot(T[0], unit(a)))) > 0.95:
        a = np.array([1.0, 0, 0]) if abs(T[0][0]) < 0.9 else np.array([0, 0, 1.0])
    N = np.zeros_like(P)
    N[0] = unit(np.cross(np.cross(T[0], a), T[0]))
    for i in range(1, len(P)):
        v = N[i - 1] - np.dot(N[i - 1], T[i]) * T[i]
        N[i] = unit(v)
    return T, N, np.cross(T, N)


def sweep(path, radius, seg=12, caps=(True, True), rb=None, up=(0, 1, 0), twist=0.0):
    """Tube along a path (glTF points); radius may vary per point; rb gives an elliptic section."""
    P = np.asarray(path, float)
    n = len(P)
    T, N, Bn = frames(P, up)
    r = np.broadcast_to(np.asarray(radius, float), (n,))
    rb = r if rb is None else np.broadcast_to(np.asarray(rb, float), (n,))
    th = twist + np.linspace(0, 2 * math.pi, seg, endpoint=False)
    G = P[:, None, :] + r[:, None, None] * np.cos(th)[None, :, None] * N[:, None, :] + rb[:, None, None] * np.sin(th)[None, :, None] * Bn[:, None, :]
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    UV = np.stack([np.broadcast_to(np.arange(seg) / seg, (n, seg)), np.broadcast_to(arc[:, None], (n, seg))], -1)
    parts = [quads_to_tris(grid_part(G, UV, wrap=True))]
    if caps[0]:
        parts.append(fan(P[0], G[0], -T[0]))
    if caps[1]:
        parts.append(fan(P[-1], G[-1], T[-1]))
    return merge_parts(parts)


def cyl(p0, p1, r, seg=12, caps=(True, True)):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    return sweep(np.stack([p0, p1]), r, seg, caps)


def disk(center, normal, r, seg=24, axis_u=None):
    normal = unit(normal)
    a = np.array([0, 1.0, 0]) if abs(normal[1]) < 0.9 else np.array([1.0, 0, 0])
    u = unit(np.cross(a, normal)) if axis_u is None else unit(axis_u)
    v = np.cross(normal, u)
    th = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    ring = np.asarray(center, float) + r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)
    p = fan(center, ring, normal)
    rel = p.V - np.asarray(center, float)
    uvv = np.stack([rel @ u, rel @ v], 1) / (2 * r) + 0.5
    p.L = uvv[p.F]
    return p


def arc_pts(c, r, a0, a1, n, plane=('x', 'y')):
    t = np.linspace(a0, a1, n)
    out = np.zeros((n, 3)) + np.asarray(c, float)
    idx = {'x': 0, 'y': 1, 'z': 2}
    out[:, idx[plane[0]]] += r * np.cos(t)
    out[:, idx[plane[1]]] += r * np.sin(t)
    return out


# ---------------------------------------------------------------- smooth noise
class Smooth2:
    """Sum of random plane waves: a cheap smooth 2D noise with unit standard deviation."""

    def __init__(self, rng, lx, ly, n=7):
        self.kx = 2 * math.pi / (lx * rng.uniform(0.7, 1.5, n)) * rng.choice([-1, 1], n)
        self.ky = 2 * math.pi / (ly * rng.uniform(0.7, 1.5, n)) * rng.choice([-1, 1], n)
        self.ph = rng.uniform(0, 2 * math.pi, n)
        a = rng.uniform(0.5, 1.0, n)
        self.a = a / math.sqrt((a ** 2).sum() / 2)

    def __call__(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        out = np.zeros(np.broadcast(x, y).shape)
        for kx, ky, ph, a in zip(self.kx, self.ky, self.ph, self.a):
            out += a * np.sin(kx * x + ky * y + ph)
        return out

    def dx(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        out = np.zeros(np.broadcast(x, y).shape)
        for kx, ky, ph, a in zip(self.kx, self.ky, self.ph, self.a):
            out += a * kx * np.cos(kx * x + ky * y + ph)
        return out


# ---------------------------------------------------------------- Blender objects
COLLS = {}


def collection(name, parent=None):
    if name in COLLS:
        return COLLS[name]
    c = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(c)
    COLLS[name] = c
    return c


def make_object(name, parts_mats, coll, origin=(0, 0, 0), yaw=0.0, smooth=True, sharp_angle=None, props=None):
    """parts_mats: list of (Part, material). Vertices are glTF coordinates relative to `origin`, unrotated;
    the object gets location `origin` and a rotation `yaw` about the vertical."""
    mats, Vs, Fs, Ls, Ms, n = [], [], [], [], [], 0
    for p, m in parts_mats:
        if m not in mats:
            mats.append(m)
        mi = mats.index(m)
        tri = quads_to_tris(p) if p.F.shape[1] != 4 else p
        Vs.append(tri.V)
        Fs.extend((tri.F + n).tolist())
        Ls.append(tri.L.reshape(-1, 2))
        Ms.append(np.full(len(tri.F), mi))
        n += len(tri.V)
    V = B(np.concatenate(Vs))
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], Fs)
    lay = me.uv_layers.new(name='UVMap')
    lay.data.foreach_set('uv', np.concatenate(Ls).astype(np.float32).ravel())
    for m in mats:
        me.materials.append(m)
    me.polygons.foreach_set('material_index', np.concatenate(Ms).astype(np.int32))
    if smooth:
        me.shade_smooth()
        if sharp_angle is not None:
            me.set_sharp_from_angle(angle=math.radians(sharp_angle))
    else:
        me.shade_flat()
    me.update()
    ob = bpy.data.objects.new(name, me)
    ob.location = bvec(origin)
    ob.rotation_euler = (0, 0, yaw)
    coll.objects.link(ob)
    for k, v in (props or {}).items():
        ob[k] = v
    return ob


def apply_modifiers(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    old = ob.data
    name = old.name
    ob.modifiers.clear()
    ob.data = me
    bpy.data.meshes.remove(old)
    me.name = name


def tri_count(ob):
    me = ob.data
    return sum(len(p.vertices) - 2 for p in me.polygons)


# ---------------------------------------------------------------- materials
def lin(c):
    c = np.asarray(c, float)
    return tuple(np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).tolist())


def hexc(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def rgba(c):
    c = tuple(c)
    return c + (1.0,) if len(c) == 3 else c


class NT:
    def __init__(self, mat):
        self.nt = mat.node_tree
        self.x = 0

    def node(self, kind, **inputs):
        n = self.nt.nodes.new(kind)
        n.location = (self.x, 0)
        self.x -= 200
        for k, v in inputs.items():
            self.set(n, k, v)
        return n

    def set(self, n, key, v):
        s = sock_in(n, key)
        if hasattr(v, 'is_output') or isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, s)
        else:
            s.default_value = v

    def link(self, a, b):
        self.nt.links.new(a, b)


def sock_in(n, key):
    for s in n.inputs:
        if s.identifier == key and s.enabled:
            return s
    for s in n.inputs:
        if s.name == key and s.enabled:
            return s
    for s in n.inputs:
        if s.identifier == key or s.name == key:
            return s
    raise KeyError(key)


def sock_out(n, key):
    for s in n.outputs:
        if (s.identifier == key or s.name == key) and s.enabled:
            return s
    for s in n.outputs:
        if s.identifier == key or s.name == key:
            return s
    raise KeyError(key)


def new_material(name):
    m = bpy.data.materials.new(name)
    if m.node_tree is None:
        m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (300, 0)
    b = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new(b.outputs['BSDF'], out.inputs['Surface'])
    return m, NT(m), b


def coord(t, kind='Object', scale=(1, 1, 1), offset=(0, 0, 0)):
    tc = t.node('ShaderNodeTexCoord')
    mp = t.node('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = scale
    mp.inputs['Location'].default_value = offset
    t.link(sock_out(tc, kind), mp.inputs['Vector'])
    return mp.outputs['Vector']


def noise(t, vec, scale, detail=3.0, rough=0.55, distortion=0.0, lo=0.3, hi=0.7):
    n = t.node('ShaderNodeTexNoise')
    n.inputs['Scale'].default_value = scale
    n.inputs['Detail'].default_value = detail
    n.inputs['Roughness'].default_value = rough
    n.inputs['Distortion'].default_value = distortion
    t.link(vec, n.inputs['Vector'])
    mr = t.node('ShaderNodeMapRange')
    mr.inputs['From Min'].default_value = lo
    mr.inputs['From Max'].default_value = hi
    t.link(n.outputs['Fac'], mr.inputs['Value'])
    return mr.outputs['Result']


def mixc(t, fac, a, b):
    m = t.node('ShaderNodeMix')
    m.data_type = 'RGBA'
    m.clamp_factor = True
    t.set(m, 'Factor_Float', fac)
    t.set(m, 'A_Color', rgba(a) if isinstance(a, tuple) else a)
    t.set(m, 'B_Color', rgba(b) if isinstance(b, tuple) else b)
    return sock_out(m, 'Result_Color')


def mixf(t, fac, a, b):
    m = t.node('ShaderNodeMix')
    m.data_type = 'FLOAT'
    m.clamp_factor = True
    t.set(m, 'Factor_Float', fac)
    t.set(m, 'A_Float', a)
    t.set(m, 'B_Float', b)
    return sock_out(m, 'Result_Float')


def math_node(t, op, a, b=0.0, clamp=False):
    m = t.node('ShaderNodeMath')
    m.operation = op
    m.use_clamp = clamp
    t.set(m, 0 if False else 'Value', a)
    ins = [s for s in m.inputs]
    if hasattr(b, 'is_output') or isinstance(b, bpy.types.NodeSocket):
        t.link(b, ins[1])
    else:
        ins[1].default_value = b
    return m.outputs[0]


def bump(t, height, strength, distance=0.002, normal=None):
    b = t.node('ShaderNodeBump')
    b.inputs['Strength'].default_value = strength
    b.inputs['Distance'].default_value = distance
    t.link(height, b.inputs['Height'])
    if normal is not None:
        t.link(normal, b.inputs['Normal'])
    return b.outputs['Normal']


def tag(m, **kw):
    for k, v in kw.items():
        m[k] = v


MATS = {}


def mat_basic(name, color, rough=0.5, metal=0.0, spec=0.5, sheen=0.0, sheen_tint=None, sheen_rough=0.5, var=None,
              bump_scale=None, bump_strength=0.0, coat=0.0, emission=None, emit=0.0, double=False, coord_scale=(1, 1, 1)):
    """Principled material; `var` = (color2, noise scale, amount) varies the base colour with object-space noise."""
    if name in MATS:
        return MATS[name]
    m, t, b = new_material(name)
    base = lin(color)
    vec = coord(t, 'Object', coord_scale) if (var or bump_scale) else None
    if var:
        c2, sc, amt = var
        f = noise(t, vec, sc, 4.0, 0.6)
        f = math_node(t, 'MULTIPLY', f, amt, clamp=True)
        t.link(mixc(t, f, base, lin(c2)), b.inputs['Base Color'])
    else:
        b.inputs['Base Color'].default_value = rgba(base)
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metal
    b.inputs['Specular IOR Level'].default_value = spec
    if sheen:
        b.inputs['Sheen Weight'].default_value = sheen
        b.inputs['Sheen Roughness'].default_value = sheen_rough
        b.inputs['Sheen Tint'].default_value = rgba(lin(sheen_tint or color))
    if coat:
        b.inputs['Coat Weight'].default_value = coat
        b.inputs['Coat Roughness'].default_value = 0.25
    if bump_scale:
        n = t.node('ShaderNodeTexNoise')
        n.inputs['Scale'].default_value = bump_scale
        n.inputs['Detail'].default_value = 6.0
        t.link(vec, n.inputs['Vector'])
        t.link(bump(t, n.outputs['Fac'], bump_strength, 0.002), b.inputs['Normal'])
    if emission is not None:
        b.inputs['Emission Color'].default_value = rgba(lin(emission))
        b.inputs['Emission Strength'].default_value = emit
    tag(m, bake_base=list(color), double_sided=bool(double))
    m.use_backface_culling = not double
    MATS[name] = m
    return m


def mat_velvet(name, lo, hi, sheen_tint, scale=1.0):
    if name in MATS:
        return MATS[name]
    m, t, b = new_material(name)
    vec = coord(t, 'Object', (scale, scale, scale * 0.22))
    f1 = noise(t, vec, 2.2, 3.0, 0.5, 0.0, 0.32, 0.68)
    col = mixc(t, f1, lin(lo), lin(hi))
    vec2 = coord(t, 'Object', (1, 1, 0.08))
    f2 = noise(t, vec2, 9.0, 2.0, 0.5, 0.0, 0.25, 0.75)
    col = mixc(t, math_node(t, 'MULTIPLY', f2, 0.35, True), col, tuple(np.array(lin(lo)) * 0.75))
    t.link(col, b.inputs['Base Color'])
    b.inputs['Roughness'].default_value = 0.92
    b.inputs['Specular IOR Level'].default_value = 0.18
    b.inputs['Sheen Weight'].default_value = 1.0
    b.inputs['Sheen Roughness'].default_value = 0.38
    b.inputs['Sheen Tint'].default_value = rgba(lin(sheen_tint))
    pile = t.node('ShaderNodeTexNoise')
    pile.inputs['Scale'].default_value = 420.0
    pile.inputs['Detail'].default_value = 2.0
    t.link(coord(t, 'Object'), pile.inputs['Vector'])
    t.link(bump(t, pile.outputs['Fac'], 0.06, 0.001), b.inputs['Normal'])
    tag(m, bake_base=list(hi), double_sided=True)
    m.use_backface_culling = False
    MATS[name] = m
    return m


def mat_image(name, albedo, rough_img=None, height_img=None, rough=0.5, bump_strength=0.4, spec=0.5, coat=0.0):
    if name in MATS:
        return MATS[name]
    m, t, b = new_material(name)
    uvn = t.node('ShaderNodeUVMap')
    uvn.uv_map = 'UVMap'
    ti = t.node('ShaderNodeTexImage')
    ti.image = albedo
    ti.interpolation = 'Cubic'
    t.link(uvn.outputs['UV'], ti.inputs['Vector'])
    t.link(ti.outputs['Color'], b.inputs['Base Color'])
    if rough_img is not None:
        tr = t.node('ShaderNodeTexImage')
        tr.image = rough_img
        t.link(uvn.outputs['UV'], tr.inputs['Vector'])
        t.link(tr.outputs['Color'], b.inputs['Roughness'])
    else:
        b.inputs['Roughness'].default_value = rough
    if height_img is not None:
        th = t.node('ShaderNodeTexImage')
        th.image = height_img
        th.interpolation = 'Cubic'
        t.link(uvn.outputs['UV'], th.inputs['Vector'])
        t.link(bump(t, th.outputs['Color'], bump_strength, 0.0025), b.inputs['Normal'])
    b.inputs['Specular IOR Level'].default_value = spec
    if coat:
        b.inputs['Coat Weight'].default_value = coat
        b.inputs['Coat Roughness'].default_value = 0.3
    tag(m, bake_base='texture:' + albedo.name, double_sided=False)
    MATS[name] = m
    return m


# ---------------------------------------------------------------- textures (numpy)
def cr_mat(n_out, n_in):
    t = (np.arange(n_out) + 0.5) / n_out * (n_in - 1)
    i0 = np.floor(t).astype(np.int64)
    f = t - i0
    w = np.stack([(-f ** 3 + 2 * f ** 2 - f) / 2, (3 * f ** 3 - 5 * f ** 2 + 2) / 2,
                  (-3 * f ** 3 + 4 * f ** 2 + f) / 2, (f ** 3 - f ** 2) / 2], 1)
    M = np.zeros((n_out, n_in), np.float32)
    rows = np.arange(n_out)
    for k in range(4):
        np.add.at(M, (rows, np.clip(i0 - 1 + k, 0, n_in - 1)), w[:, k].astype(np.float32))
    return M


def vnoise(rng, H, W, gh, gw):
    G = rng.standard_normal((max(gh, 2), max(gw, 2))).astype(np.float32)
    out = (cr_mat(H, G.shape[0]) @ G) @ cr_mat(W, G.shape[1]).T
    return out / (out.std() + 1e-9)


def save_image(name, arr, path, colorspace='sRGB'):
    arr = np.clip(np.asarray(arr, np.float32), 0, 1)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, 2)
    H, W = arr.shape[:2]
    rgba_ = np.concatenate([arr, np.ones((H, W, 1), np.float32)], 2)
    img = bpy.data.images.new(name, W, H, alpha=False)
    img.pixels.foreach_set(rgba_.ravel())
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    img.colorspace_settings.name = colorspace
    return img


def stroke(canvas, rng, p0, p1, width_px, amount, bend=0.0, px=640.0, origin=(0, 0)):
    """Adds a soft stroke (glTF floor metres, x/z) into canvas (rows=z from the front, cols=x)."""
    H, W = canvas.shape
    pts = []
    for s in np.linspace(0, 1, 6):
        q = (1 - s) * np.asarray(p0) + s * np.asarray(p1)
        d = np.asarray(p1) - np.asarray(p0)
        nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        pts.append(q + nrm * bend * 4 * s * (1 - s))
    pts = np.array(pts)
    cols = (pts[:, 0] - origin[0]) * px
    rows = (origin[1] - pts[:, 1]) * px
    pad = width_px * 3 + 2
    c0, c1 = int(max(0, cols.min() - pad)), int(min(W, cols.max() + pad + 1))
    r0, r1 = int(max(0, rows.min() - pad)), int(min(H, rows.max() + pad + 1))
    if c1 <= c0 or r1 <= r0:
        return
    yy, xx = np.mgrid[r0:r1, c0:c1].astype(np.float32)
    best = np.full(xx.shape, 1e9, np.float32)
    along = np.zeros(xx.shape, np.float32)
    for i in range(len(pts) - 1):
        ax, ay, bx, by = cols[i], rows[i], cols[i + 1], rows[i + 1]
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy + 1e-9
        t = np.clip(((xx - ax) * vx + (yy - ay) * vy) / L2, 0, 1)
        dd = np.hypot(xx - (ax + t * vx), yy - (ay + t * vy))
        m = dd < best
        best = np.where(m, dd, best)
        along = np.where(m, (i + t) / (len(pts) - 1), along)
    taper = np.maximum(np.sin(np.pi * np.clip(along, 0, 1)), 0) ** 0.6
    canvas[r0:r1, c0:c1] = np.maximum(canvas[r0:r1, c0:c1], amount * taper * np.exp(-(best / max(width_px, 0.3)) ** 2))


def floor_textures(W=4096, H=3072, seed=11):
    """Albedo (sRGB), roughness and height maps for the plank floor top; rows run from the front edge (z=+2.4) back."""
    rng = np.random.default_rng(seed)
    PX = W / BOX_W
    x = -XO + (np.arange(W) + 0.5) / PX
    z = ZO - (np.arange(H) + 0.5) / PX
    PW = 0.14
    k = np.floor((x + PW / 2) / PW).astype(np.int64)
    kmin = k.min()
    kc = k - kmin
    ncol = kc.max() + 1
    u = (x + PW / 2) - k * PW
    joints = []
    for ci in range(ncol):
        for _ in range(200):
            zz, js = -ZO - rng.uniform(0, 2.2), []
            while zz < ZO:
                zz += rng.uniform(0.95, 2.6)
                if zz < ZO - 0.05:
                    js.append(zz)
            ok = all(abs(a - b) > 0.24 for a in js for b in (joints[-1] if joints else []))
            ok = ok and all(abs(a - b) > 0.24 for a in js for b in (joints[-2] if len(joints) > 1 else []))
            if ok:
                break
        joints.append(js)
    seg_ids = np.zeros((H, ncol), np.int64)
    dj = np.full((H, ncol), 9.0)
    for ci, js in enumerate(joints):
        js = np.array(sorted(js))
        if len(js):
            seg_ids[:, ci] = np.searchsorted(js, z)
            dj[:, ci] = np.min(np.abs(z[:, None] - js[None, :]), axis=1)
    seg = seg_ids[:, kc]
    djp = dj[:, kc] * PX
    pid = kc[None, :] * 16 + seg
    npl = ncol * 16
    pr = rng.random((npl, 12))
    honey, brown, pale, grey = np.array([0.56, 0.40, 0.26]), np.array([0.42, 0.29, 0.18]), np.array([0.62, 0.49, 0.35]), np.array([0.45, 0.40, 0.34])
    pc = honey[None] * (1 - pr[:, :1]) + brown[None] * pr[:, :1]
    pc = pc * (0.82 + 0.3 * pr[:, 1:2])
    pc = pc * (1 - 0.35 * pr[:, 2:3] ** 2) + pale[None] * 0.35 * pr[:, 2:3] ** 2
    weathered = pr[:, 3] > 0.86
    pc[weathered] = pc[weathered] * 0.45 + grey[None] * 0.55
    dark = pr[:, 11] > 0.9
    pc[dark] *= 0.72
    col = pc[pid]
    lo = vnoise(rng, H, W, 14, 46)
    col *= (1 + 0.05 * lo)[..., None]
    warp = 1.3 * vnoise(rng, H, W, 12, 183) + 0.7 * vnoise(rng, H, W, 5, 44)
    freq = 90 + 70 * pr[:, 4]
    phase = pr[:, 5] * 10
    g = 0.5 + 0.5 * np.sin(2 * np.pi * (u[None, :] * freq[pid] + warp + phase[pid]))
    sharp = (7 + 9 * pr[:, 6])[pid]
    lines = g ** sharp
    contrast = (0.06 + 0.08 * pr[:, 7])[pid]
    soft = 0.5 + 0.5 * np.sin(2 * np.pi * (u[None, :] * (18 + 20 * pr[:, 9])[pid] + 0.6 * warp + phase[pid] * 0.7))
    col *= (1 - contrast * lines - 0.05 * soft)[..., None]
    col[..., 2] *= (1 - 0.06 * lines)
    fib = vnoise(rng, H, W, 80, 1100)
    col *= (1 + 0.03 * fib)[..., None]
    patina = vnoise(rng, H, W, 60, 80)
    col *= (1 + 0.05 * patina)[..., None]
    del g, warp
    dc = np.minimum(u, PW - u)[None, :] * PX
    gw = (0.75 + 0.7 * pr[:, 8])[pid]
    gap_c = np.clip(1.2 - (dc - gw), 0, 1)
    gap_j = np.clip(1.0 - (djp - 0.55), 0, 1)
    gap = np.maximum(gap_c, gap_j)
    dmin = np.minimum(dc, djp)
    ao = 1 - 0.42 * np.exp(-dmin / 2.4) - 0.12 * np.exp(-dmin / 9.0)
    X, Z = np.meshgrid(x, z)
    traffic = np.exp(-(X / 1.25) ** 4) * smoothstep(-1.4, -0.4, Z) + np.exp(-((X / 1.3) ** 2 + ((Z + 0.65) / 1.0) ** 2))
    traffic = np.clip(traffic, 0, 1)
    patchy = np.clip(0.5 + 0.5 * vnoise(rng, H, W, 30, 40), 0, 1)
    wear = traffic * patchy
    lum = col.mean(-1, keepdims=True)
    worn = col * 1.07 * 0.8 + lum * 1.07 * 0.2
    col = col * (1 - 0.32 * wear[..., None]) + worn * 0.32 * wear[..., None]
    edge = np.minimum(np.minimum(XI - np.abs(X), Z + ZI), ZO - Z)
    grime = np.clip(1 - edge / 0.45, 0, 1) ** 1.5 * 0.28
    grime += 0.06 * np.clip(0.5 + 0.5 * vnoise(rng, H, W, 24, 32), 0, 1)
    col *= (1 - grime)[..., None]
    scuff = np.zeros((H, W), np.float32)
    scratch = np.zeros((H, W), np.float32)
    origin = (-XO, ZO)
    n_s = 0
    while n_s < 170:
        px, pz = rng.uniform(-XI, XI), rng.uniform(-ZI, ZO)
        tw = np.exp(-(px / 1.25) ** 4) * (1 if pz > -1.4 else 0.25) + 0.25
        if rng.random() > min(tw, 1):
            continue
        ang = rng.normal(math.pi / 2, 0.9) if rng.random() < 0.6 else rng.uniform(0, math.pi)
        L = rng.uniform(0.015, 0.11)
        d = np.array([math.cos(ang), math.sin(ang)]) * L
        stroke(scuff, rng, (px, pz), (px + d[0], pz + d[1]), rng.uniform(1.0, 3.0), rng.uniform(0.05, 0.2), rng.normal(0, 0.01), PX, origin)
        n_s += 1
    for _ in range(420):
        px, pz = rng.uniform(-XI, XI), rng.uniform(-ZI, ZO)
        ang = rng.uniform(0, math.pi)
        L = rng.uniform(0.03, 0.26)
        d = np.array([math.cos(ang), math.sin(ang)]) * L
        stroke(scratch, rng, (px, pz), (px + d[0], pz + d[1]), rng.uniform(0.45, 0.9), rng.uniform(0.04, 0.13), rng.normal(0, 0.02), PX, origin)
    col = col * (1 - scuff[..., None]) + np.array([0.09, 0.075, 0.065]) * scuff[..., None]
    col = col * (1 - scratch[..., None]) + np.array([0.86, 0.76, 0.62]) * scratch[..., None]
    holes = np.zeros((H, W), np.float32)
    jz = np.arange(-ZO + 0.2, ZO, 0.4)
    yy, xx = np.mgrid[-3:4, -3:4]
    blob = np.exp(-(xx ** 2 + yy ** 2) / 1.1).astype(np.float32)
    for ci in range(ncol):
        for zz in list(jz) + [j + s for j in joints[ci] for s in (-0.025, 0.025)]:
            for uu in (0.022, PW - 0.022):
                cx = int(((ci + kmin) * PW - PW / 2 + uu + XO) * PX + rng.normal(0, 1.5))
                cy = int((ZO - zz) * PX + rng.normal(0, 1.5))
                if 3 <= cx < W - 4 and 3 <= cy < H - 4:
                    holes[cy - 3:cy + 4, cx - 3:cx + 4] = np.maximum(holes[cy - 3:cy + 4, cx - 3:cx + 4], blob * rng.uniform(0.18, 0.34))
    col = col * (1 - holes[..., None]) + np.array([0.12, 0.08, 0.05]) * holes[..., None]
    col *= ao[..., None]
    col = col * (1 - gap[..., None]) + np.array([0.05, 0.035, 0.025]) * gap[..., None]
    rough = 0.52 + 0.10 * (pr[:, 9][pid] - 0.5) + 0.16 * wear + 0.25 * gap + 0.10 * scuff + 0.12 * grime + 0.03 * fib
    bevel = smoothstep(0.0, 4.0, dmin)
    cup = (pr[:, 10][pid] - 0.5) * 0.08
    height = 0.55 + 0.4 * bevel + cup - 0.5 * gap + 0.035 * lines * 0 - 0.025 * lines - 0.15 * scratch - 0.3 * holes + 0.01 * fib
    return np.clip(col, 0, 1), np.clip(rough, 0.2, 0.95), np.clip(height, 0, 1)


# ---------------------------------------------------------------- curtains
def drape(p0, p1, front, top, *, pitch, amp, seed, seg=12, troch=0.5, pool=0.0, bend=0.07, bottom=None,
          flare_end=None, flare=0.0, wander=0.12, pvar=0.3, avar=0.3, h2=0.22, heading=0.3, top_amp=0.7,
          foot_amp=1.25, rows_mid=0.14, spread=0.0, gather=0.0, rows_top=(7, 5), rows_foot=9):
    """Velvet-like drape hung along a horizontal track p0 -> p1 ((x, z) pairs) with its face toward `front`.
    Folds: noise-modulated phase (irregular pitch, wandering), trochoid lateral term (round ridges, pinched valleys),
    regular pleats under the heading, flare and pooling on the floor when `bottom` is None."""
    rng = np.random.default_rng(seed)
    p0, p1, front = np.asarray(p0, float), np.asarray(p1, float), unit(np.asarray(front, float))
    ut = p1 - p0
    width = float(np.linalg.norm(ut))
    ut /= width
    nc = int(math.ceil(width / pitch * seg)) + 1
    X = np.linspace(0, width, nc)
    n1 = Smooth2(rng, 0.8, 7.0, 6)
    n2 = Smooth2(rng, 2.6, 3.8, 5)
    n3 = Smooth2(rng, 0.6, 5.0, 6)
    n4 = Smooth2(rng, 0.9, 4.5, 5)
    n5 = Smooth2(rng, 0.35, 0.6, 5)
    floor = bottom is None
    yb = FLOOR_Y if floor else bottom
    drop = top - yb - (bend if floor else 0.0)
    tv = [np.linspace(0, 0.16, rows_top[0]), np.linspace(0.22, 0.52, rows_top[1])]
    if drop > 1.4:
        k = max(2, int(round((drop - 1.25) / rows_mid)))
        tv.append(np.linspace(0.64, drop - 0.62, k))
        tv.append(np.linspace(drop - 0.5, drop, rows_foot))
    else:
        tv = [np.linspace(0, drop, max(5, int(drop / rows_mid)))]
    tvert = np.unique(np.round(np.concatenate(tv), 6))
    tvert = tvert[tvert <= drop + 1e-9]
    rows = [('v', t, 0.0) for t in tvert]
    if floor:
        for a in np.linspace(0, math.pi / 2, 6)[1:]:
            rows.append(('b', drop + a * bend, a))
        poolc = max(pool - bend, 0.0) * np.clip(0.7 + 0.45 * n5(X, 0.0), 0.25, 1.6)
        if pool > bend:
            for tau in np.linspace(0, 1, 4)[1:]:
                rows.append(('p', tau, math.pi / 2))
    nr = len(rows)
    G = np.zeros((nr, nc, 3))
    T = np.zeros((nr, nc))
    for ri, (kind, tval, alpha) in enumerate(rows):
        if kind == 'v':
            y_base = top - tval
            fwd = np.zeros(nc)
            t_arc = np.full(nc, tval)
        elif kind == 'b':
            y_base = yb + bend - bend * math.sin(alpha)
            fwd = np.full(nc, bend - bend * math.cos(alpha))
            t_arc = np.full(nc, tval)
        else:
            y_base = yb
            fwd = bend + tval * poolc
            t_arc = drop + bend * math.pi / 2 + tval * poolc
        ye = max(y_base, yb)
        hb = float(smoothstep(top - heading - 0.4, top - 0.04, ye))
        depth = float(np.clip((top - ye) / max(top - yb, 1e-6), 0, 1))
        foot = float(smoothstep(0.95, 0.0, ye - yb)) if floor else float(smoothstep(0.45, 0.0, ye - yb)) * 0.5
        wand = wander * (0.35 + 0.9 * depth) * (1 - hb)
        Ap = pvar * 0.75 / (2 * math.pi * pitch) * (1 - hb)
        phi = X / pitch + Ap * n1(X, ye) + wand * n2(X, ye) + gather * (1 - hb) * depth * np.sin(X / width * math.pi)
        prof = top_amp + (1 - top_amp) * float(smoothstep(top, top - 0.55, ye))
        prof *= 1 + (foot_amp - 1) * foot
        A = amp * prof * np.clip(1 + avar * (1 - 0.7 * hb) * n3(X, ye), 0.35, 1.9)
        th = 2 * math.pi * phi
        harm = h2 * (1 - hb) * n4(X, ye)
        D = A * (np.cos(th) + harm * np.cos(2 * th + 0.7))
        lat = troch * (1 - 0.5 * foot) * pitch / (2 * math.pi) * (A / amp) * np.sin(th)
        Xf = X.copy()
        if flare_end is not None and flare:
            s = X / width if flare_end == 1 else 1 - X / width
            g = foot + (0.6 * tval if kind == 'p' else 0.0)
            Xf = Xf + (1 if flare_end == 1 else -1) * flare * s ** 2.2 * g
        if spread and kind == 'p':
            Xf = Xf + spread * tval * (X / width - 0.5) * 2
        if kind == 'v':
            Deff, vert = D, 0.0
        else:
            beta = math.sin(alpha) ** 2
            roll = 0.5 * (D + A) * 0.6
            Deff = D * (1 - beta) + roll * beta
            vert = math.sin(alpha)
        xz = p0[None] + (Xf + lat)[:, None] * ut[None] + (fwd + Deff * math.cos(alpha))[:, None] * front[None]
        y = y_base + Deff * vert
        if floor:
            y = np.maximum(y, yb + 0.003)
        G[ri, :, 0] = xz[:, 0]
        G[ri, :, 1] = y
        G[ri, :, 2] = xz[:, 1]
        T[ri] = t_arc
    du = np.linalg.norm(np.diff(G, axis=1), axis=2)
    U = np.concatenate([np.zeros((nr, 1)), np.cumsum(du, axis=1)], 1)
    part = grid_part(G, np.stack([U, T], -1))
    return part.facing(np.array([front[0], 0.0, front[1]]))


# ---------------------------------------------------------------- set builders
def build_box(coll, mats):
    out_m, in_m = mats['box_out'], mats['box_in']
    q = []

    def quad(p0, p1, p2, p3, m, normal):
        V = np.array([p0, p1, p2, p3], float)
        e1, e2 = unit(V[1] - V[0]), None
        n = unit(np.asarray(normal, float))
        e2 = np.cross(n, e1)
        uv = np.stack([V @ e1, V @ e2], 1)
        p = Part(V, [[0, 1, 2, 3]], uv[[[0, 1, 2, 3]]])
        q.append((p.facing(n), m))

    def rect(axis, val, a0, a1, b0, b1, normal, m):
        pts = []
        for a, b in ((a0, b0), (a1, b0), (a1, b1), (a0, b1)):
            p = [0, 0, 0]
            p[axis] = val
            o = [i for i in range(3) if i != axis]
            p[o[0]], p[o[1]] = a, b
            pts.append(p)
        quad(*pts, m, normal)

    rect(0, -XO, 0, BOX_H, -ZO, ZO, (-1, 0, 0), out_m)
    rect(0, XO, 0, BOX_H, -ZO, ZO, (1, 0, 0), out_m)
    rect(2, -ZO, -XO, XO, 0, BOX_H, (0, 0, -1), out_m)
    rect(1, BOX_H, -XO, XO, -ZO, ZO, (0, 1, 0), out_m)
    rect(2, ZO, -XO, -OPEN_HW, 0, BOX_H, (0, 0, 1), out_m)
    rect(2, ZO, OPEN_HW, XO, 0, BOX_H, (0, 0, 1), out_m)
    rect(2, ZO, -OPEN_HW, OPEN_HW, OPEN_H, BOX_H, (0, 0, 1), out_m)
    rect(0, -XI, 0, YI, -ZI, ZI, (1, 0, 0), in_m)
    rect(0, XI, 0, YI, -ZI, ZI, (-1, 0, 0), in_m)
    rect(2, -ZI, -XI, XI, 0, YI, (0, 0, 1), in_m)
    rect(1, YI, -XI, XI, -ZI, ZI, (0, -1, 0), in_m)
    rect(2, ZI, -XI, -OPEN_HW, 0, YI, (0, 0, -1), in_m)
    rect(2, ZI, OPEN_HW, XI, 0, YI, (0, 0, -1), in_m)
    rect(2, ZI, -OPEN_HW, OPEN_HW, OPEN_H, YI, (0, 0, -1), in_m)
    rect(0, -OPEN_HW, 0, OPEN_H, ZI, ZO, (1, 0, 0), out_m)
    rect(0, OPEN_HW, 0, OPEN_H, ZI, ZO, (-1, 0, 0), out_m)
    rect(1, OPEN_H, -OPEN_HW, OPEN_HW, ZI, ZO, (0, -1, 0), out_m)
    return make_object('Box_Shell', q, coll, smooth=False)


def build_floor(coll, mats, tex_dir, seed=11, reuse=False):
    names = ('floor_albedo', 'floor_rough', 'floor_height')
    paths = [os.path.join(tex_dir, n + '.png') for n in names]
    if reuse and all(os.path.exists(q) for q in paths):
        log('floor textures (reused)')
        ims = [bpy.data.images.load(q) for q in paths]
        for im, n, cs in zip(ims, names, ('sRGB', 'Non-Color', 'Non-Color')):
            im.name = n
            im.colorspace_settings.name = cs
        ia, ir, ih = ims
    else:
        log('floor textures')
        alb, rough, height = floor_textures(seed=seed)
        ia = save_image('floor_albedo', alb, paths[0], 'sRGB')
        r2 = rough.reshape(rough.shape[0] // 2, 2, rough.shape[1] // 2, 2).mean((1, 3))
        ir = save_image('floor_rough', r2, paths[1], 'Non-Color')
        ih = save_image('floor_height', height, paths[2], 'Non-Color')
    for im in (ia, ir, ih):
        im.pack()
    m = mat_image('Floor_Planks', ia, ir, ih, bump_strength=0.35, spec=0.45, coat=0.0)
    mats['floor'] = m
    nx, nz = 32, 24
    xs, zs = np.linspace(-XO, XO, nx + 1), np.linspace(ZO, -ZO, nz + 1)
    Xg, Zg = np.meshgrid(xs, zs)
    G = np.stack([Xg, np.full_like(Xg, FLOOR_Y), Zg], -1)
    UV = np.stack([(Xg + XO) / BOX_W, (ZO - Zg) / BOX_D], -1)
    top = grid_part(G, UV).facing((0, 1, 0))
    parts = [(top, m)]
    for (a, b, fixed, axis, nrm) in ((-XO, XO, ZO, 2, (0, 0, 1)), (-XO, XO, -ZO, 2, (0, 0, -1)), (-ZO, ZO, -XO, 0, (-1, 0, 0)), (-ZO, ZO, XO, 0, (1, 0, 0))):
        if axis == 2:
            V = np.array([[a, 0, fixed], [b, 0, fixed], [b, FLOOR_Y, fixed], [a, FLOOR_Y, fixed]])
            uv = np.array([[(V[i, 0] + XO) / BOX_W, (ZO - fixed) / BOX_D + 0.003 * (V[i, 1] / FLOOR_Y)] for i in range(4)])
        else:
            V = np.array([[fixed, 0, a], [fixed, 0, b], [fixed, FLOOR_Y, b], [fixed, FLOOR_Y, a]])
            uv = np.array([[(fixed + XO) / BOX_W + 0.002 * (V[i, 1] / FLOOR_Y), (ZO - V[i, 2]) / BOX_D] for i in range(4)])
        parts.append((Part(V, [[0, 1, 2, 3]], uv[[[0, 1, 2, 3]]]).facing(nrm), m))
    return make_object('Floor', parts, coll, smooth=False)


def build_curtains(coll, mats):
    vel, blk = mats['velvet'], mats['velour_black']
    obs = []
    legL = drape((-XI + 0.12, LEG_Z), (-LEG_X, LEG_Z), (0, 1), CURTAIN_TOP, pitch=0.185, amp=0.078, seed=21, seg=12,
                 troch=0.55, pool=0.14, bend=0.1, flare_end=1, flare=0.11, spread=0.03)
    legR = drape((LEG_X, LEG_Z), (XI - 0.12, LEG_Z), (0, 1), CURTAIN_TOP, pitch=0.19, amp=0.08, seed=37, seg=12,
                 troch=0.55, pool=0.15, bend=0.1, flare_end=0, flare=0.11, spread=0.03)
    obs.append(make_object('Curtain_Leg_L', [(legL, vel)], coll))
    obs.append(make_object('Curtain_Leg_R', [(legR, vel)], coll))
    back = drape((-XI + 0.05, BACK_Z), (XI - 0.05, BACK_Z), (0, 1), CURTAIN_TOP, pitch=0.21, amp=0.062, seed=53, seg=8,
                 troch=0.5, pool=0.05, bend=0.04, avar=0.28, wander=0.1, rows_mid=0.25, foot_amp=1.1, rows_top=(5, 3), rows_foot=6)
    obs.append(make_object('Curtain_Back', [(back, vel)], coll))
    val = drape((-XI + 0.04, VALANCE_Z), (XI - 0.04, VALANCE_Z), (0, 1), VALANCE_TOP, pitch=0.21, amp=0.055, seed=71, seg=9,
                troch=0.55, bottom=VALANCE_BOTTOM, heading=0.1, top_amp=0.75, foot_amp=1.3, wander=0.1, pvar=0.32, avar=0.32, rows_mid=0.1)
    obs.append(make_object('Curtain_Valance', [(val, vel)], coll))
    for side, s in (('L', -1), ('R', 1)):
        dx = s * (XI - 0.06)
        d = drape((dx, ZI - 0.02), (dx, -ZI + 0.02), (-s, 0), YI - 0.005, pitch=0.30, amp=0.05, seed=90 + s, seg=7,
                  troch=0.4, pool=0.025, bend=0.02, wander=0.1, avar=0.25, heading=0.2, top_amp=0.85, rows_mid=0.5, foot_amp=1.05,
                  rows_top=(3, 2), rows_foot=4)
        obs.append(make_object('Drape_Wall_' + side, [(d, blk)], coll))
    for side, s in (('L', -1), ('R', 1)):
        x0, x1 = s * (XI - 0.04), s * (OPEN_HW + 0.03)
        d = drape((x0, ZI - 0.07), (x1, ZI - 0.07), (0, -1), YI - 0.005, pitch=0.28, amp=0.045, seed=95 + s, seg=7,
                  troch=0.4, pool=0.02, bend=0.02, wander=0.08, heading=0.2, top_amp=0.85, rows_mid=0.5, foot_amp=1.0,
                  rows_top=(3, 2), rows_foot=4)
        obs.append(make_object('Drape_Front_' + side, [(d, blk)], coll))
    return obs


# ---------------------------------------------------------------- truss and lanterns
PIPE_R = 0.024


def build_truss(coll, mats):
    pm = mats['pipe']
    parts = []
    for z in BATTEN_ZS:
        parts.append(cyl((-XI + 0.07, BATTEN_Y, z), (XI - 0.07, BATTEN_Y, z), PIPE_R, 16))
    for x in GRID_XS:
        parts.append(cyl((x, GRID_Y, -1.5), (x, GRID_Y, 0.8), PIPE_R, 16))
        for z in BATTEN_ZS:
            parts.append(cyl((x, BATTEN_Y + PIPE_R, z), (x, GRID_Y - PIPE_R, z), 0.016, 10))
            for yy in (BATTEN_Y, GRID_Y):
                parts.append(rbox((0.035, 0.034, 0.034), 0.006, 1, 1, (x, yy, z)))
        for z in (-1.42, 0.72):
            parts.append(cyl((x, GRID_Y + PIPE_R, z), (x, YI, z), 0.008, 8))
            parts.append(cyl((x, YI - 0.006, z), (x, YI, z), 0.03, 12))
    for s in (-1, 1):
        parts.append(cyl((s * (LEG_X - 0.06), CURTAIN_PIPE_Y, LEG_Z), (s * (XI - 0.05), CURTAIN_PIPE_Y, LEG_Z), 0.019, 12))
        parts.append(cyl((s * 2.55, CURTAIN_PIPE_Y + 0.019, LEG_Z), (s * 2.55, GRID_Y - PIPE_R, LEG_Z), 0.008, 8))
        parts.append(cyl((s * (LEG_X + 0.05), CURTAIN_PIPE_Y + 0.019, LEG_Z), (s * (LEG_X + 0.05), YI, LEG_Z), 0.008, 8))
    parts.append(cyl((-XI + 0.05, CURTAIN_PIPE_Y, BACK_Z), (XI - 0.05, CURTAIN_PIPE_Y, BACK_Z), 0.019, 12))
    for x in (-2.4, -0.8, 0.8, 2.4):
        parts.append(cyl((x, CURTAIN_PIPE_Y + 0.019, BACK_Z), (x, YI, BACK_Z), 0.008, 8))
    tri = [quads_to_tris(p) for p in parts]
    return make_object('Truss', [(merge_parts(tri), pm)], coll, sharp_angle=50)


def trapezoid_plate(w0, w1, length, thick):
    """Flat plate in local XY: hinge edge (width w0) on y=0, far edge (width w1) at y=length; thickness along z."""
    p = np.array([[-w0 / 2, 0], [w0 / 2, 0], [w1 / 2, length], [-w1 / 2, length]])
    V = np.concatenate([np.c_[p, np.full(4, -thick / 2)], np.c_[p, np.full(4, thick / 2)]])
    F = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    return Part(V, F, V[:, :2][F])


def lantern_body(kind):
    """Profile (r, z) about the optical axis (+Z forward), lens radius, lens z, colour-frame z and width."""
    if kind == 'fresnel':
        prof = [(0, -0.165), (0.07, -0.165), (0.092, -0.158), (0.104, -0.142), (0.11, -0.125), (0.114, -0.09), (0.12, -0.088),
                (0.12, -0.078), (0.115, -0.076), (0.117, -0.03), (0.123, -0.028), (0.123, -0.018), (0.118, -0.016), (0.12, 0.05),
                (0.125, 0.052), (0.125, 0.062), (0.12, 0.064), (0.121, 0.112), (0.128, 0.116), (0.128, 0.14), (0.109, 0.141), (0.104, 0.132)]
        return prof, 0.104, 0.132, 0.146, 0.27
    if kind == 'par':
        prof = [(0, -0.205), (0.06, -0.207), (0.09, -0.198), (0.104, -0.183), (0.109, -0.16), (0.109, -0.1), (0.114, -0.098),
                (0.114, -0.088), (0.109, -0.086), (0.109, 0.06), (0.114, 0.062), (0.114, 0.072), (0.109, 0.074), (0.11, 0.17),
                (0.118, 0.173), (0.118, 0.192), (0.1, 0.193), (0.096, 0.182)]
        return prof, 0.096, 0.182, 0.197, 0.25
    prof = [(0, -0.245), (0.045, -0.245), (0.055, -0.232), (0.06, -0.215), (0.09, -0.205), (0.118, -0.17), (0.13, -0.12),
            (0.127, -0.07), (0.112, -0.035), (0.09, -0.012), (0.086, 0.0), (0.086, 0.012), (0.081, 0.018), (0.081, 0.25),
            (0.089, 0.253), (0.089, 0.272), (0.075, 0.273), (0.071, 0.264)]
    return prof, 0.071, 0.264, 0.277, 0.205


def build_lantern(name, kind, x, batten_z, target, state, coll, mats, barn=None):
    body_m = mats['lantern']
    lens_m = {'dark': mats['lens_dark'], 'glow': mats['lens_glow'], 'key': mats['lens_key']}[state]
    prof, lens_r, lens_z, frame_z, frame_w = lantern_body(kind)
    P = np.asarray(prof)
    pipe_c = np.array([x, BATTEN_Y, batten_z])
    rmax = float(P[:, 0].max())
    pivot = pipe_c - np.array([0, 0.25, 0])
    for _ in range(3):
        pan, tilt = aim_angles(np.asarray(target, float) - pivot)
        reach = float(np.max(P[:, 0] * math.cos(tilt) - P[:, 1] * math.sin(tilt)))
        hy = max(reach, rmax) + 0.02
        pivot = pipe_c - np.array([0, hy + 0.03 + PIPE_R, 0])
    Rb = aim_matrix(pan, tilt)
    Ry = rot_y(pan)
    seg = 22
    parts = [(lathe(P, seg).moved(Rb, pivot), body_m), (disk((0, 0, lens_z), (0, 0, 1), lens_r, seg).moved(Rb, pivot), lens_m)]
    fw, ft = frame_w, 0.008
    for sx, sy, w, h in ((0, 1, fw, 0.022), (0, -1, fw, 0.022), (1, 0, 0.022, fw - 0.044), (-1, 0, 0.022, fw - 0.044)):
        c = np.array([sx * (fw / 2 - 0.011), sy * (fw / 2 - 0.011), frame_z])
        parts.append((rbox((w / 2, h / 2, ft / 2), 0.002, 1, 1, c).moved(Rb, pivot), body_m))
    if barn:
        leaves = ((barn[0], fw, fw * 1.05, 0.17), (barn[1], fw, fw * 1.05, 0.17), (barn[2], fw * 0.9, fw * 0.62, 0.12), (barn[3], fw * 0.9, fw * 0.62, 0.12))
        for i, (ang, w0, w1, ln) in enumerate(leaves):
            roll = [0, math.pi, -math.pi / 2, math.pi / 2][i]
            plate = trapezoid_plate(w0, w1, ln, 0.003).moved(rot_x(math.radians(90 - ang)))
            plate = plate.moved(rot_z(roll), np.array([0, 0, frame_z + 0.01]) + rot_z(roll) @ np.array([0, fw / 2, 0]))
            parts.append((plate.moved(Rb, pivot), body_m))
        ring = lathe([(fw * 0.5, frame_z + 0.004), (fw * 0.55, frame_z + 0.004), (fw * 0.55, frame_z + 0.014), (fw * 0.5, frame_z + 0.014)], 24)
        parts.append((ring.moved(Rb, pivot), body_m))
    if kind == 'ers':
        for a in (45, 135, 225, 315):
            tab = rbox((0.024, 0.007, 0.003), 0.002, 1, 1, (0.1, 0, 0.004)).moved(rot_z(math.radians(a)))
            parts.append((tab.moved(Rb, pivot), body_m))
    arm_x = rmax + 0.014
    for s in (-1, 1):
        arm = rbox((0.004, (hy + 0.004) / 2, 0.018), 0.002, 1, 1, (s * arm_x, (hy + 0.004) / 2 - 0.004, 0))
        parts.append((arm.moved(Ry, pivot), body_m))
        parts.append((cyl((s * (rmax - 0.01), 0, 0), (s * (arm_x + 0.022), 0, 0), 0.012, 12).moved(Ry, pivot), body_m))
        parts.append((cyl((s * (arm_x + 0.022), 0, 0), (s * (arm_x + 0.032), 0, 0), 0.02, 12).moved(Ry, pivot), body_m))
    parts.append((rbox((arm_x + 0.004, 0.004, 0.018), 0.002, 1, 1, (0, hy, 0)).moved(Ry, pivot), body_m))
    parts.append((cyl((0, hy, 0), (0, hy + 0.034, 0), 0.008, 10).moved(Ry, pivot), body_m))
    cp = arc_pts((0, 0, 0), PIPE_R + 0.009, -math.pi / 2 - 2.4, -math.pi / 2 + 2.4, 18, ('z', 'y'))
    parts.append((sweep(cp, 0.007, 8).moved(None, pipe_c), body_m))
    parts.append((rbox((0.012, 0.014, 0.02), 0.003, 1, 1, pipe_c + np.array([0, PIPE_R + 0.012, 0])), body_m))
    lens_c = pivot + Rb @ np.array([0, 0, lens_z])
    fwd = Rb @ np.array([0, 0, 1.0])
    ob = make_object(name, [(quads_to_tris(p), m) for p, m in parts], coll, sharp_angle=40,
                     props={'lens_center_gltf': [round(float(v), 4) for v in lens_c], 'aim_gltf': [round(float(v), 4) for v in fwd],
                            'lens_radius': lens_r, 'state': state})
    return ob, lens_c, fwd, lens_r


LANTERNS = [
    ('Lantern_Key', 'fresnel', 0.6, 0.0, KEY_TARGET, 'key', None),
    ('Lantern_FL', 'ers', 0.6, -2.3, (-0.7, 0.0, -0.3), 'dark', None),
    ('Lantern_FLC', 'fresnel', 0.6, -1.2, (-1.5, 0.0, 0.9), 'dark', (38, 55, 70, 62)),
    ('Lantern_FRC', 'par', 0.6, 1.25, (2.35, 1.0, -0.2), 'glow', None),
    ('Lantern_FR', 'fresnel', 0.6, 2.35, (1.0, 0.0, 0.4), 'dark', None),
    ('Lantern_BK', 'fresnel', -1.3, 0.8, (0.3, 1.7, BACK_Z), 'glow', (25, 48, 72, 66)),
]


def build_lanterns(coll, mats):
    info = {}
    for name, kind, bz, x, target, state, barn in LANTERNS:
        ob, lc, fwd, lr = build_lantern(name, kind, x, bz, target, state, coll, mats, barn)
        info[name] = dict(lens=lc, fwd=fwd, radius=lr, state=state, target=np.asarray(target, float))
    return info


# ---------------------------------------------------------------- look-dev rig
def look_at_rot(pos_b, target_b):
    d = Vector(target_b) - Vector(pos_b)
    return d.to_track_quat('-Z', 'Y').to_euler()


def add_light(coll, name, kind, pos_g, target_g, energy, color, **kw):
    ld = bpy.data.lights.new(name, kind)
    ld.energy = energy
    ld.color = color
    for k, v in kw.items():
        setattr(ld, k, v)
    ob = bpy.data.objects.new(name, ld)
    ob.location = bvec(pos_g)
    ob.rotation_euler = look_at_rot(bvec(pos_g), bvec(target_g))
    coll.objects.link(ob)
    return ob


CAMERAS = {
    'Cam_Audience': ((0.0, 1.6, 5.5), (0.0, 1.3, -0.5), 26),
    'Cam_Visitor': ((0.45, 1.62, 1.85), (-0.15, 1.35, -0.65), 24),
    'Cam_Left': ((1.35, 1.55, 1.95), (-2.15, 0.95, -0.45), 24),
    'Cam_Right': ((-1.25, 1.55, 2.0), (2.3, 0.95, 0.5), 24),
    'Cam_Truss': ((0.35, 1.5, 2.15), (0.0, 3.75, -0.7), 20),
}


def build_lookdev(coll, info):
    key = info['lanterns']['Lantern_Key']
    add_light(coll, 'LD_Key', 'SPOT', key['lens'] + 0.02 * key['fwd'], KEY_TARGET, 900.0, (1.0, 0.86, 0.68),
              spot_size=math.radians(40), spot_blend=0.28, shadow_soft_size=0.06)
    for nm, power, size in (('Lantern_FRC', 70.0, 46), ('Lantern_BK', 95.0, 50)):
        L = info['lanterns'][nm]
        add_light(coll, 'LD_Wash_' + nm.split('_')[1], 'SPOT', L['lens'] + 0.02 * L['fwd'], L['target'], power, (1.0, 0.72, 0.48),
                  spot_size=math.radians(size), spot_blend=0.7, shadow_soft_size=0.08)
    if 'studio' in info:
        s = info['studio']
        add_light(coll, 'LD_Studio', 'AREA', s['face'] + 0.01 * s['fwd'], s['face'] + s['fwd'], 60.0, (1.0, 0.9, 0.78),
                  shape='DISK', size=0.42, spread=math.radians(80))
    add_light(coll, 'LD_Fill', 'AREA', (0.0, 3.7, ZO + 1.2), (0.0, 2.3, -1.6), 55.0, (1.0, 0.9, 0.82),
              shape='RECTANGLE', size=5.0, size_y=1.6)
    for nm, (pos, tgt, lens) in CAMERAS.items():
        cd = bpy.data.cameras.new(nm)
        cd.lens = lens
        cd.sensor_width = 36
        cd.clip_start = 0.02
        cd.clip_end = 60
        ob = bpy.data.objects.new(nm, cd)
        ob.location = bvec(pos)
        ob.rotation_euler = look_at_rot(bvec(pos), bvec(tgt))
        coll.objects.link(ob)
    world = bpy.data.worlds.new('LD_World')
    if world.node_tree is None:
        world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    bg.inputs['Color'].default_value = (0.0016, 0.0014, 0.0013, 1)
    bg.inputs['Strength'].default_value = 1.0
    bpy.context.scene.world = world
    bpy.context.scene.camera = bpy.data.objects['Cam_Audience']


def mat_lens(name, base, rings=18.0, emission=None, emit=0.0, hot=1.0):
    """Lantern lens: concentric fresnel steps from the disk UVs (centred on 0.5, 0.5)."""
    if name in MATS:
        return MATS[name]
    m, t, b = new_material(name)
    uvn = t.node('ShaderNodeUVMap')
    uvn.uv_map = 'UVMap'
    vm = t.node('ShaderNodeVectorMath')
    vm.operation = 'DISTANCE'
    t.link(uvn.outputs['UV'], vm.inputs[0])
    vm.inputs[1].default_value = (0.5, 0.5, 0.0)
    r = sock_out(vm, 'Value')
    ring = math_node(t, 'SINE', math_node(t, 'MULTIPLY', r, rings * 2 * math.pi))
    ring = math_node(t, 'MULTIPLY', math_node(t, 'ADD', ring, 1.0), 0.5)
    col = mixc(t, math_node(t, 'MULTIPLY', ring, 0.5, True), lin(base), tuple(np.clip(np.array(lin(base)) * 2.2 + 0.004, 0, 1)))
    t.link(col, b.inputs['Base Color'])
    b.inputs['Roughness'].default_value = 0.12
    b.inputs['Specular IOR Level'].default_value = 0.8
    t.link(bump(t, ring, 0.25, 0.0005), b.inputs['Normal'])
    if emission is not None:
        b.inputs['Emission Color'].default_value = rgba(lin(emission))
        fall = t.node('ShaderNodeMapRange')
        fall.inputs['From Min'].default_value = 0.0
        fall.inputs['From Max'].default_value = 0.5
        fall.inputs['To Min'].default_value = emit * hot
        fall.inputs['To Max'].default_value = emit * 0.55
        t.link(r, fall.inputs['Value'])
        t.link(math_node(t, 'MULTIPLY', fall.outputs['Result'], math_node(t, 'ADD', math_node(t, 'MULTIPLY', ring, 0.25), 0.85)), b.inputs['Emission Strength'])
    tag(m, bake_base=list(base), double_sided=False, emissive=emission is not None)
    MATS[name] = m
    return m


# ---------------------------------------------------------------- materials table
def build_materials():
    m = {}
    m['velvet'] = mat_velvet('Velvet_Crimson', hexc('#5a0710'), hexc('#7a0c16'), hexc('#d4404c'))
    m['velour_black'] = mat_basic('Velour_Black', hexc('#141211'), rough=0.95, spec=0.15, sheen=0.7, sheen_tint=hexc('#3a3532'),
                                  sheen_rough=0.4, var=(hexc('#0e0d0c'), 3.0, 0.8), double=True)
    m['box_out'] = mat_basic('Box_Outside', hexc('#1b1612'), rough=0.82, spec=0.3, var=(hexc('#14100d'), 1.5, 0.7))
    m['box_in'] = mat_basic('Box_Inside', hexc('#0c0b0a'), rough=0.95, spec=0.15)
    m['pipe'] = mat_basic('Pipe_Black', hexc('#1c1c1c'), rough=0.34, spec=0.5, var=(hexc('#2a2826'), 8.0, 0.6))
    m['lantern'] = mat_basic('Lantern_Black', hexc('#1d1d1d'), rough=0.33, spec=0.55, var=(hexc('#302e2b'), 14.0, 0.55))
    m['lens_dark'] = mat_lens('Lens_Dark', hexc('#0c0e0f'), 16.0)
    m['lens_glow'] = mat_lens('Lens_Glow', hexc('#3a2614'), 16.0, emission=hexc('#ff9a4a'), emit=1.8, hot=1.3)
    m['lens_key'] = mat_lens('Lens_Key', hexc('#f2e4cc'), 16.0, emission=hexc('#ffe7c4'), emit=22.0, hot=1.5)
    return m


def reset_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    for blocks in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.images):
        for b in list(blocks):
            blocks.remove(b)


def write_report(out, info):
    import json
    report = {'objects': {}, 'totals': {}}
    for cname in ('Set', 'Props'):
        tot = 0
        for ob in COLLS[cname].all_objects:
            if ob.type == 'MESH':
                n = tri_count(ob)
                report['objects'][ob.name] = {'collection': cname, 'tris': n, 'materials': [s.material.name for s in ob.material_slots if s.material]}
                tot += n
        report['totals'][cname] = tot
    report['totals']['all'] = report['totals']['Set'] + report['totals']['Props']
    report['lanterns'] = {k: {'lens': [round(float(x), 4) for x in v['lens']], 'aim': [round(float(x), 4) for x in v['fwd']],
                              'radius': v['radius'], 'state': v['state']} for k, v in info['lanterns'].items()}
    if 'studio' in info:
        report['studio'] = {k: [round(float(x), 4) for x in v] for k, v in info['studio'].items()}
    with open(os.path.splitext(out)[0] + '-report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    return report


def main():
    args = get_args()
    out = args.get('out')
    if not out:
        out = os.path.join(os.environ.get('SCENE_WORK') or os.path.join(os.getcwd(), 'scene-work'), 'set.blend')
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tex_dir = os.path.join(os.path.dirname(out), 'set-tex')
    os.makedirs(tex_dir, exist_ok=True)
    parts = set(str(args.get('parts', 'set,props')).split(','))
    reset_scene()
    mats = build_materials()
    set_c, props_c, look_c = collection('Set'), collection('Props'), collection('LookDev')
    info = {}
    log('box/floor/curtains')
    build_box(set_c, mats)
    build_floor(set_c, mats, tex_dir, reuse=bool(args.get('reuse-tex')))
    build_curtains(set_c, mats)
    log('truss/lanterns')
    build_truss(set_c, mats)
    info['lanterns'] = build_lanterns(set_c, mats)
    if 'props' in parts:
        log('props')
        info.update(build_props(props_c, mats, sim=not args.get('no-sim')))
    build_lookdev(look_c, info)
    report = write_report(out, info)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
    log('saved', out, 'tris', report['totals'])


# ---------------------------------------------------------------- props: materials
FABRIC = {
    'charcoal': '#3a3938', 'white': '#e4e0d6', 'cream': '#dacdb3', 'black': '#1d1c1b', 'beige': '#c6b291',
    'paleblue': '#aebcc6', 'red': '#681319', 'ivory': '#e9e1cf', 'navy': '#1f293b', 'camel': '#98734e',
    'brown': '#5b4232', 'grey': '#78746e', 'taupe': '#8a7b6a',
}


def fabric(mats, key):
    k = 'fabric_' + key
    if k not in mats:
        c = np.array(hexc(FABRIC[key]))
        mats[k] = mat_basic('Fabric_' + key.capitalize(), tuple(c), rough=0.88, spec=0.22, sheen=0.45,
                            sheen_tint=tuple(np.clip(c * 1.25 + 0.05, 0, 1)), sheen_rough=0.45,
                            var=(tuple(c * 0.84), 16.0, 0.6), bump_scale=300.0, bump_strength=0.035, double=True)
    return mats[k]


def mat_wood_uv(name, base, dark, scale=34.0):
    if name in MATS:
        return MATS[name]
    m, t, b = new_material(name)
    uvn = t.node('ShaderNodeUVMap')
    uvn.uv_map = 'UVMap'
    w = t.node('ShaderNodeTexWave')
    w.wave_type = 'BANDS'
    w.bands_direction = 'Y'
    w.inputs['Scale'].default_value = scale
    w.inputs['Distortion'].default_value = 3.5
    w.inputs['Detail'].default_value = 4.0
    w.inputs['Detail Scale'].default_value = 2.5
    w.inputs['Detail Roughness'].default_value = 0.6
    t.link(uvn.outputs['UV'], w.inputs['Vector'])
    f = t.node('ShaderNodeMapRange')
    f.inputs['From Min'].default_value = 0.2
    f.inputs['From Max'].default_value = 1.0
    f.inputs['To Max'].default_value = 0.55
    t.link(w.outputs['Fac'], f.inputs['Value'])
    n = noise(t, coord(t, 'Object'), 7.0, 3.0, 0.6)
    col = mixc(t, f.outputs['Result'], lin(base), lin(dark))
    col = mixc(t, math_node(t, 'MULTIPLY', n, 0.45, True), col, tuple(np.array(lin(dark)) * 0.8))
    t.link(col, b.inputs['Base Color'])
    b.inputs['Roughness'].default_value = 0.55
    b.inputs['Specular IOR Level'].default_value = 0.45
    t.link(bump(t, w.outputs['Fac'], 0.04, 0.0006), b.inputs['Normal'])
    tag(m, bake_base=list(base), double_sided=False)
    MATS[name] = m
    return m


def mat_diffuser(name, color, strength):
    if name in MATS:
        return MATS[name]
    m, t, b = new_material(name)
    uvn = t.node('ShaderNodeUVMap')
    uvn.uv_map = 'UVMap'
    vm = t.node('ShaderNodeVectorMath')
    vm.operation = 'DISTANCE'
    t.link(uvn.outputs['UV'], vm.inputs[0])
    vm.inputs[1].default_value = (0.5, 0.5, 0.0)
    mr = t.node('ShaderNodeMapRange')
    mr.inputs['From Min'].default_value = 0.12
    mr.inputs['From Max'].default_value = 0.5
    mr.inputs['To Min'].default_value = strength
    mr.inputs['To Max'].default_value = strength * 0.62
    t.link(sock_out(vm, 'Value'), mr.inputs['Value'])
    b.inputs['Base Color'].default_value = rgba(lin(color))
    b.inputs['Roughness'].default_value = 0.6
    b.inputs['Emission Color'].default_value = rgba(lin(color))
    t.link(mr.outputs['Result'], b.inputs['Emission Strength'])
    tag(m, bake_base=list(color), double_sided=False, emissive=True)
    MATS[name] = m
    return m


def prop_materials(m):
    m['leather'] = mat_basic('Trunk_Leather', hexc('#2f1d12'), rough=0.62, spec=0.4, var=(hexc('#4d3523'), 5.0, 0.95),
                             bump_scale=110.0, bump_strength=0.1)
    m['trunk_wood'] = mat_wood_uv('Trunk_Wood', hexc('#694a2f'), hexc('#43301e'), 70.0)
    m['trim'] = mat_basic('Trunk_Trim', hexc('#5c4b31'), rough=0.42, metal=0.9, var=(hexc('#2c2419'), 9.0, 0.85))
    m['brass'] = mat_basic('Brass_Aged', hexc('#b48b47'), rough=0.36, metal=1.0, var=(hexc('#5a4322'), 11.0, 0.75))
    m['strap'] = mat_basic('Leather_Strap', hexc('#55361f'), rough=0.5, spec=0.4, var=(hexc('#352113'), 9.0, 0.7))
    m['suitcase'] = mat_basic('Suitcase_PaleBlue', hexc('#a3c3c8'), rough=0.36, spec=0.5, var=(hexc('#86a7ad'), 6.0, 0.75),
                              bump_scale=520.0, bump_strength=0.045)
    m['suit_trim'] = mat_basic('Suitcase_Cream', hexc('#dacfb5'), rough=0.33, spec=0.5, var=(hexc('#c1b293'), 12.0, 0.6))
    m['chrome'] = mat_basic('Chrome_Worn', hexc('#c8c7c2'), rough=0.18, metal=1.0, var=(hexc('#8a8882'), 18.0, 0.55))
    m['stand'] = mat_basic('Stand_Black', hexc('#151515'), rough=0.4, spec=0.5, var=(hexc('#252423'), 16.0, 0.5))
    m['rubber'] = mat_basic('Rubber_Black', hexc('#121212'), rough=0.75, spec=0.3)
    m['housing'] = mat_basic('Studio_Housing', hexc('#181818'), rough=0.38, spec=0.5, var=(hexc('#2a2827'), 12.0, 0.5))
    m['diffuser'] = mat_diffuser('Studio_Diffuser', hexc('#fff1dc'), 9.0)
    m['rail'] = mat_basic('Rail_Black', hexc('#171717'), rough=0.32, spec=0.5, var=(hexc('#262625'), 20.0, 0.4))
    m['hanger'] = mat_basic('Hanger_Wood', hexc('#a5703b'), rough=0.36, spec=0.5, var=(hexc('#7a4e27'), 14.0, 0.7))
    m['case'] = mat_basic('Case_Black', hexc('#1b1b1b'), rough=0.55, spec=0.4, var=(hexc('#121212'), 6.0, 0.7), bump_scale=600.0, bump_strength=0.03)
    m['alu'] = mat_basic('Case_Aluminium', hexc('#9d9c98'), rough=0.3, metal=1.0, var=(hexc('#6e6d69'), 14.0, 0.6))
    m['sandbag'] = mat_basic('Sandbag_Canvas', hexc('#2c2a26'), rough=0.9, spec=0.2, var=(hexc('#3a3731'), 10.0, 0.6), bump_scale=240.0, bump_strength=0.06)
    return m


# ---------------------------------------------------------------- props: helpers
def slat(lo, hi, long_axis, r=0.0025):
    p = box(lo, hi, r, 1)
    P = p.V[p.F]
    n = unit(np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1]))
    L = np.zeros(3)
    L[long_axis] = 1.0
    other = np.cross(n, L)
    bad = np.linalg.norm(other, axis=1) < 0.5
    other[bad] = np.array([0, 1.0, 0]) if long_axis != 1 else np.array([1.0, 0, 0])
    other = unit(other)
    p.L = np.stack([P @ L, np.einsum('mkj,mj->mk', P, other)], -1)
    return p


def rrect_path(hx, hy, r, n=8, z=0.0):
    pts = []
    for cx, cy, a0 in ((hx - r, hy - r, 0.0), (-hx + r, hy - r, math.pi / 2), (-hx + r, -hy + r, math.pi), (hx - r, -hy + r, 1.5 * math.pi)):
        for a in np.linspace(a0, a0 + math.pi / 2, n):
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a), z))
    pts.append(pts[0])
    return np.array(pts)


def bezier(p0, p1, p2, n=8):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = (np.asarray(p, float) for p in (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def catmull(pts, n_per=6):
    P = np.asarray(pts, float)
    P = np.concatenate([P[:1] * 2 - P[1:2], P, P[-1:] * 2 - P[-2:-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n_per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.array(out)


# ---------------------------------------------------------------- trunk
TRUNK_POS, TRUNK_YAW = (-2.35, 0.0, 1.15), math.radians(20)


def build_trunk(coll, mats):
    lea, wood, trim, brass, strap = mats['leather'], mats['trunk_wood'], mats['trim'], mats['brass'], mats['strap']
    hx, hz, Hh = 0.45, 0.275, 0.55
    y0, ys, yl = 0.015, 0.372, 0.379
    P = [(box((-hx, y0, -hz), (hx, ys, hz), 0.008, 1), lea), (box((-hx, yl, -hz), (hx, Hh, hz), 0.012, 2), lea)]
    pr, sl = 0.011, 0.024
    for ya, yb in ((0.045, 0.09), (0.298, 0.343), (0.405, 0.45)):
        P.append((slat((-hx + 0.05, ya, hz - 0.002), (hx - 0.05, yb, hz + pr), 0), wood))
        P.append((slat((-hx + 0.05, ya, -hz - pr), (hx - 0.05, yb, -hz + 0.002), 0), wood))
        P.append((slat((-hx - pr, ya, -hz + 0.05), (-hx + 0.002, yb, hz - 0.05), 2), wood))
        P.append((slat((hx - 0.002, ya, -hz + 0.05), (hx + pr, yb, hz - 0.05), 2), wood))
    for xs in (-0.27, 0.27):
        for ya, yb in ((0.09, 0.298), (0.45, Hh - 0.002)):
            P.append((slat((xs - sl, ya, hz - 0.002), (xs + sl, yb, hz + pr), 1), wood))
            P.append((slat((xs - sl, ya, -hz - pr), (xs + sl, yb, -hz + 0.002), 1), wood))
        P.append((slat((xs - sl, Hh - 0.002, -hz - pr), (xs + sl, Hh + pr, hz + pr), 2), wood))
    t, e = 0.021, 0.004
    for sx in (-1, 1):
        for sz in (-1, 1):
            xa, xb = (hx - t, hx + e) if sx > 0 else (-hx - e, -hx + t)
            za, zb = (hz - t, hz + e) if sz > 0 else (-hz - e, -hz + t)
            for ya, yb in ((y0, ys), (yl, Hh)):
                P.append((box((xa, ya, za), (xb, yb, zb)), trim))
    for sz in (-1, 1):
        za, zb = (hz - t, hz + e) if sz > 0 else (-hz - e, -hz + t)
        for ya, yb in ((y0 - e, y0 + t), (Hh - t, Hh + e)):
            P.append((box((-hx + 0.03, ya, za), (hx - 0.03, yb, zb)), trim))
        for ya, yb in ((ys - 0.017, ys + 0.001), (yl - 0.001, yl + 0.017)):
            P.append((box((-hx + 0.03, ya, hz - 0.003 if sz > 0 else -hz - e), (hx - 0.03, yb, hz + e if sz > 0 else -hz + 0.003)), trim))
    for sx in (-1, 1):
        xa, xb = (hx - t, hx + e) if sx > 0 else (-hx - e, -hx + t)
        for ya, yb in ((y0 - e, y0 + t), (Hh - t, Hh + e)):
            P.append((box((xa, ya, -hz + 0.03), (xb, yb, hz - 0.03)), trim))
        for ya, yb in ((ys - 0.017, ys + 0.001), (yl - 0.001, yl + 0.017)):
            P.append((box((hx - 0.003 if sx > 0 else -hx - e, ya, -hz + 0.03), (hx + e if sx > 0 else -hx + 0.003, yb, hz - 0.03)), trim))
    for sx in (-1, 1):
        for sz in (-1, 1):
            for yc in (y0 + 0.024, Hh - 0.024):
                P.append((rbox((0.034, 0.034, 0.034), 0.012, 1, 1, (sx * (hx - 0.024), yc, sz * (hz - 0.024))), brass))
            for yc in (ys - 0.012, yl + 0.012):
                P.append((rbox((0.026, 0.014, 0.026), 0.005, 1, 1, (sx * (hx - 0.018), yc, sz * (hz - 0.018))), brass))
    P.append((box((-0.042, ys - 0.075, hz + 0.003), (0.042, ys + 0.003, hz + 0.013), 0.004, 1), brass))
    P.append((box((-0.024, yl - 0.008, hz + 0.004), (0.024, yl + 0.062, hz + 0.0115), 0.003, 1), brass))
    P.append((box((-0.004, ys - 0.05, hz + 0.0125), (0.004, ys - 0.032, hz + 0.0135), 0.0), trim))
    for xc in (-0.36, 0.36):
        P.append((box((xc - 0.018, ys - 0.052, hz + 0.003), (xc + 0.018, ys + 0.002, hz + 0.0115), 0.003, 1), brass))
        P.append((box((xc - 0.012, yl - 0.004, hz + 0.003), (xc + 0.012, yl + 0.04, hz + 0.0105), 0.003, 1), brass))
    for sx in (-1, 1):
        s = np.linspace(0, 1, 13)
        path = np.stack([sx * (hx + 0.014 + 0.024 * np.sin(np.pi * s)), 0.262 - 0.012 * np.sin(np.pi * s), -0.09 + 0.18 * s], 1)
        P.append((sweep(path, 0.0145, 10, rb=0.0048), strap))
        for zc in (-0.095, 0.095):
            P.append((rbox((0.008, 0.022, 0.012), 0.004, 1, 1, (sx * (hx + 0.006), 0.262, zc)), brass))
    for zc in (-hz + 0.08, hz - 0.08):
        P.append((slat((-hx + 0.03, 0.0, zc - 0.025), (hx - 0.03, y0 + 0.002, zc + 0.025), 0), wood))
    return make_object('Trunk', P, coll, origin=(TRUNK_POS[0], FLOOR_Y, TRUNK_POS[2]), yaw=TRUNK_YAW, sharp_angle=50)


# ---------------------------------------------------------------- suitcase
SUIT_POS, SUIT_YAW = (1.9, 0.0, 1.5), math.radians(-32)


def build_suitcase(coll, mats):
    blue, cream, chrome, rubber = mats['suitcase'], mats['suit_trim'], mats['chrome'], mats['rubber']
    hx, hy, hz, r = 0.275, 0.19, 0.09, 0.034
    lift = 0.007
    P = [(rbox((hx, hy, hz), r, 4, 3, (0, hy + lift, 0)), blue)]
    band = rrect_path(hx + 0.0015, hy + 0.0015, r + 0.0015, 9) + np.array([0, hy + lift, 0])
    P.append((sweep(band, 0.0042, 10, caps=(False, False), rb=0.0085), cream))
    top = 2 * hy + lift
    s = np.linspace(0, 1, 15)
    grip = np.stack([-0.062 + 0.124 * s, top + 0.012 + 0.03 * np.sin(np.pi * s) ** 0.8, np.zeros_like(s)], 1)
    P.append((sweep(grip, 0.0115, 12, rb=0.0095), cream))
    for xc in (-0.068, 0.068):
        P.append((rbox((0.014, 0.007, 0.013), 0.004, 1, 1, (xc, top + 0.005, 0)), chrome))
    for xc in (-0.17, 0.17):
        P.append((rbox((0.022, 0.007, 0.017), 0.004, 1, 1, (xc, top + 0.003, 0)), chrome))
        P.append((rbox((0.008, 0.0025, 0.004), 0.001, 1, 1, (xc, top + 0.0105, 0.006)), rubber))
    for xc in (-0.2, 0.2):
        for zc in (-0.05, 0.05):
            P.append((cyl((xc, 0.0, zc), (xc, lift + 0.004, zc), 0.009, 10), rubber))
    return make_object('Suitcase', P, coll, origin=(SUIT_POS[0], FLOOR_Y, SUIT_POS[2]), yaw=SUIT_YAW, sharp_angle=60)


# ---------------------------------------------------------------- tripod light stands
def tripod_parts(mats, head_y, spread=0.55, hub_y=0.72, legs=(25, 145, 265), scale=1.0):
    stand, rubber = mats['stand'], mats['rubber']
    P = []
    for a in legs:
        d = np.array([math.cos(math.radians(a)), 0, math.sin(math.radians(a))])
        top = np.array([0, hub_y, 0]) + d * 0.035 * scale
        foot = d * spread + np.array([0, 0.016, 0])
        P.append((cyl(top, foot, 0.0125 * scale, 10), stand))
        P.append((cyl(foot - d * 0.01 + np.array([0, -0.016, 0]), foot + d * 0.012 + np.array([0, 0.004, 0]), 0.017 * scale, 10), rubber))
        P.append((cyl(top + (foot - top) * 0.47, np.array([0, hub_y * 0.56, 0]) + d * 0.027 * scale, 0.0065 * scale, 8), stand))
    P.append((cyl((0, hub_y - 0.045 * scale, 0), (0, hub_y + 0.045 * scale, 0), 0.034 * scale, 16), stand))
    P.append((cyl((0, hub_y * 0.56 - 0.02, 0), (0, hub_y * 0.56 + 0.02, 0), 0.027 * scale, 16), stand))
    s1 = hub_y + 0.5 * (head_y - hub_y) * 0.85
    s2 = hub_y + 0.85 * (head_y - hub_y) * 0.85
    top_y = head_y
    P.append((cyl((0, hub_y * 0.5, 0), (0, s1 + 0.03, 0), 0.019 * scale, 14), stand))
    P.append((cyl((0, s1 - 0.025, 0), (0, s1 + 0.035, 0), 0.025 * scale, 14), stand))
    P.append((cyl((0, s1, 0), (0, s2 + 0.03, 0), 0.0155 * scale, 14), stand))
    P.append((cyl((0, s2 - 0.02, 0), (0, s2 + 0.03, 0), 0.021 * scale, 14), stand))
    P.append((cyl((0, s2, 0), (0, top_y, 0), 0.0125 * scale, 12), stand))
    for yk, rk in ((hub_y, 0.034), (s1 + 0.005, 0.025), (s2 + 0.005, 0.021)):
        P.append((cyl((rk * scale - 0.004, yk, 0), (rk * scale + 0.04, yk, 0), 0.005, 8), stand))
        P.append((cyl((rk * scale + 0.04, yk - 0.017, 0), (rk * scale + 0.04, yk + 0.017, 0), 0.008, 8), stand))
    return P


STUDIO_POS, STUDIO_HEAD_Y, STUDIO_TARGET = (2.2, 0.0, 0.5), 2.1, (0.0, 1.3, 0.9)


def build_studio_light(coll, mats):
    org = np.array([STUDIO_POS[0], FLOOR_Y, STUDIO_POS[2]])
    Hc = np.array([0.0, STUDIO_HEAD_Y - FLOOR_Y, 0.0])
    housing, diff, chrome, stand = mats['housing'], mats['diffuser'], mats['chrome'], mats['stand']
    yoke_drop = 0.30
    P = tripod_parts(mats, Hc[1] - yoke_drop - 0.03, legs=(-15, 105, 225))
    pan, tilt = aim_angles(np.asarray(STUDIO_TARGET, float) - (org + Hc))
    Ry, Rb = rot_y(pan), aim_matrix(pan, tilt)
    R = 0.236
    arm_x = R + 0.022
    yk = [rbox((arm_x + 0.007, 0.007, 0.021), 0.003, 1, 1, (0, -yoke_drop, 0)),
          rbox((0.007, yoke_drop / 2 + 0.007, 0.021), 0.003, 1, 1, (arm_x, -yoke_drop / 2, 0)),
          rbox((0.007, yoke_drop / 2 + 0.007, 0.021), 0.003, 1, 1, (-arm_x, -yoke_drop / 2, 0)),
          cyl((0, -yoke_drop - 0.035, 0), (0, -yoke_drop, 0), 0.016, 12)]
    for s in (-1, 1):
        yk.append(cyl((s * (arm_x - 0.004), 0, 0), (s * (arm_x + 0.03), 0, 0), 0.02, 14))
        yk.append(cyl((s * (arm_x + 0.03), 0, 0), (s * (arm_x + 0.042), 0, 0), 0.03, 14))
    for p in yk:
        P.append((p.moved(Ry, Hc), stand))
    prof = [(0, -0.132), (0.12, -0.132), (0.16, -0.126), (0.19, -0.112), (0.207, -0.092), (0.218, -0.055), (0.224, -0.012),
            (0.227, 0.026), (0.236, 0.03), (0.236, 0.05), (0.215, 0.051), (0.21, 0.043)]
    head = [(lathe(prof, 44), housing), (disk((0, 0, 0.043), (0, 0, 1), 0.21, 44), diff),
            (rbox((0.085, 0.055, 0.024), 0.008, 2, 1, (0, 0.0, -0.152)), housing)]
    for s in (-1, 1):
        head.append((cyl((s * 0.2, 0, 0), (s * (arm_x - 0.004), 0, 0), 0.017, 12), housing))
    s_ = np.linspace(0, 1, 11)
    hp = np.stack([-0.08 + 0.16 * s_, 0.212 + 0.042 * np.sin(np.pi * s_) ** 0.7, np.full_like(s_, -0.045)], 1)
    head.append((sweep(hp, 0.008, 8), housing))
    for p, m in head:
        P.append((p.moved(Rb, Hc), m))
    back = Hc + Rb @ np.array([0.0, -0.04, -0.176])
    leg = np.array([math.cos(math.radians(-15)), 0, math.sin(math.radians(-15))])
    cable = [back, back + np.array([0.02, -0.12, 0.03]), np.array([0.05, Hc[1] - 0.55, 0.04]), np.array([0.035, 1.2, 0.03]),
             np.array([0.03, 0.8, 0.03]), leg * 0.2 + np.array([0, 0.5, 0.0]), leg * 0.42 + np.array([0, 0.2, 0]),
             leg * 0.5 + np.array([0, 0.008, 0]), leg * 0.62 + np.array([0.05, 0.006, -0.05]),
             np.array([XI - 0.15 - org[0], 0.006, -0.25]), np.array([XI - 0.04 - org[0], 0.006, -0.55])]
    P.append((sweep(catmull(cable, 7), 0.0045, 7), mats['rubber']))
    sb = leg * 0.47
    Rl = rot_y(math.radians(15))
    for s in (-1, 1):
        side = np.cross(leg, [0, 1, 0]) * s
        P.append((rbox((0.075, 0.04, 0.05), 0.032, 2, 1).moved(Rl, sb + side * 0.062 + np.array([0, 0.04, 0])), mats['sandbag']))
    sp = np.array([sb + np.cross(leg, [0, 1, 0]) * t + np.array([0, 0.07 + 0.085 * (1 - (t / 0.065) ** 2), 0]) for t in np.linspace(-0.065, 0.065, 9)])
    P.append((sweep(sp, 0.022, 8, rb=0.004, up=leg), mats['sandbag']))
    ob = make_object('Studio_Light', P, coll, origin=org, yaw=0.0, sharp_angle=45)
    face = org + Hc + Rb @ np.array([0, 0, 0.043])
    fwd = Rb @ np.array([0, 0, 1.0])
    ob['diffuser_center_gltf'] = [round(float(v), 4) for v in face]
    ob['aim_gltf'] = [round(float(v), 4) for v in fwd]
    return {'face': face, 'fwd': fwd}


def build_small_lamp(coll, mats, pos=(-2.72, 0.0, 0.38), head_y=1.42, target=(-0.3, 0.9, -0.4)):
    org = np.array([pos[0], FLOOR_Y, pos[2]])
    Hc = np.array([0.0, head_y - FLOOR_Y, 0.0])
    P = tripod_parts(mats, Hc[1] - 0.2, spread=0.4, hub_y=0.55, legs=(0, 120, 240), scale=0.85)
    prof, lens_r, lens_z, frame_z, frame_w = lantern_body('fresnel')
    prof = [(r * 0.72, z * 0.72) for r, z in prof]
    pan, tilt = aim_angles(np.asarray(target, float) - (org + Hc))
    Ry, Rb = rot_y(pan), aim_matrix(pan, tilt)
    for p, m in ((lathe(prof, 24), mats['lantern']), (disk((0, 0, lens_z * 0.72), (0, 0, 1), lens_r * 0.72, 24), mats['lens_dark'])):
        P.append((p.moved(Rb, Hc), m))
    arm_x = 0.128 * 0.72 + 0.012
    for s in (-1, 1):
        P.append((rbox((0.004, 0.1, 0.014), 0.002, 1, 1, (s * arm_x, -0.1, 0)).moved(Ry, Hc), mats['stand']))
        P.append((cyl((s * (arm_x - 0.012), 0, 0), (s * (arm_x + 0.025), 0, 0), 0.012, 10).moved(Ry, Hc), mats['stand']))
    P.append((rbox((arm_x + 0.004, 0.004, 0.014), 0.002, 1, 1, (0, -0.2, 0)).moved(Ry, Hc), mats['stand']))
    return make_object('Lamp_Small_Off', P, coll, origin=org, sharp_angle=45)


# ---------------------------------------------------------------- road case
def build_road_case(coll, mats, pos=(2.62, 0.0, -1.05), yaw=math.radians(-12)):
    case, alu, rubber, chrome = mats['case'], mats['alu'], mats['rubber'], mats['chrome']
    hx, hz, H = 0.31, 0.26, 1.06
    y0 = 0.085
    P = [(box((-hx, y0, -hz), (hx, H, hz), 0.004, 1), case)]
    t, e = 0.026, 0.004
    for sx in (-1, 1):
        for sz in (-1, 1):
            xa, xb = (hx - t, hx + e) if sx > 0 else (-hx - e, -hx + t)
            za, zb = (hz - t, hz + e) if sz > 0 else (-hz - e, -hz + t)
            P.append((box((xa, y0, za), (xb, H, zb), 0.002), alu))
            for yc in (y0 + 0.03, H - 0.03):
                P.append((rbox((0.034, 0.034, 0.034), 0.02, 1, 1, (sx * (hx - 0.02), yc, sz * (hz - 0.02))), chrome))
    for yy in (y0 + 0.012, H - 0.012):
        for sz in (-1, 1):
            za, zb = (hz - 0.022, hz + 0.004) if sz > 0 else (-hz - 0.004, -hz + 0.022)
            P.append((box((-hx + 0.03, yy - 0.012, za), (hx - 0.03, yy + 0.012, zb), 0.002), alu))
        for sx in (-1, 1):
            xa, xb = (hx - 0.022, hx + 0.004) if sx > 0 else (-hx - 0.004, -hx + 0.022)
            P.append((box((xa, yy - 0.012, -hz + 0.03), (xb, yy + 0.012, hz - 0.03), 0.002), alu))
    P.append((box((-hx - 0.004, 0.55, -hz - 0.004), (hx + 0.004, 0.575, hz + 0.004), 0.002), alu))
    for sx in (-1, 1):
        P.append((rbox((0.008, 0.045, 0.075), 0.004, 1, 1, (sx * (hx + 0.006), 0.82, 0)), chrome))
    for yc in (0.53, 0.6):
        for xc in (-0.15, 0.15):
            P.append((rbox((0.03, 0.02, 0.008), 0.004, 1, 1, (xc, yc, hz + 0.006)), chrome))
    for sx in (-1, 1):
        for sz in (-1, 1):
            c = (sx * (hx - 0.06), 0, sz * (hz - 0.06))
            P.append((cyl((c[0], 0.07, c[2]), (c[0], y0, c[2]), 0.03, 12), chrome))
            P.append((cyl((c[0] - 0.014, 0.035, c[2]), (c[0] + 0.014, 0.035, c[2]), 0.035, 14), rubber))
    return make_object('Road_Case', P, coll, origin=(pos[0], FLOOR_Y, pos[2]), yaw=yaw, sharp_angle=50)


# ---------------------------------------------------------------- clothing rail, hangers, garments
RAIL_POS, RAIL_YAW, RAIL_HALF, BAR_TOP = (-2.28, 0.0, -1.08), math.radians(15), 0.58, 1.60
HANG_W, HANG_DROP = 0.215, 0.055
GARMENTS = [
    dict(kind='jacket', color='charcoal', L=0.77, W=0.236, Wc=0.25, Wh=0.255, T=0.032, Ls=0.6, n=3.0, folds=0.007),
    dict(kind='shirt', color='white', L=0.8, W=0.226, Wc=0.236, Wh=0.24, T=0.012, Ls=0.6, n=2.6, folds=0.009),
    dict(kind='blouse', color='cream', L=0.66, W=0.206, Wc=0.214, Wh=0.23, T=0.011, Ls=0.56, n=2.4, folds=0.012),
    dict(kind='shirt', color='black', L=0.79, W=0.226, Wc=0.235, Wh=0.238, T=0.012, Ls=0.6, n=2.6, folds=0.009),
    dict(kind='jacket', color='beige', L=0.73, W=0.232, Wc=0.246, Wh=0.25, T=0.03, Ls=0.59, n=3.0, folds=0.007),
    dict(kind='shirt', color='paleblue', L=0.8, W=0.226, Wc=0.236, Wh=0.24, T=0.012, Ls=0.6, n=2.6, folds=0.009),
    dict(kind='dress', color='red', L=1.08, W=0.185, Wc=0.2, Wh=0.31, T=0.02, Ls=0.0, n=2.4, folds=0.02),
    dict(kind='blouse', color='ivory', L=0.65, W=0.206, Wc=0.216, Wh=0.232, T=0.011, Ls=0.55, n=2.4, folds=0.012),
    dict(kind='jacket', color='navy', L=0.76, W=0.236, Wc=0.25, Wh=0.255, T=0.032, Ls=0.6, n=3.0, folds=0.007),
    dict(kind='shirt', color='grey', L=0.79, W=0.226, Wc=0.235, Wh=0.24, T=0.012, Ls=0.6, n=2.6, folds=0.009),
    dict(kind='knit', color='brown', L=0.68, W=0.22, Wc=0.242, Wh=0.232, T=0.022, Ls=0.58, n=2.2, folds=0.01),
    dict(kind='coat', color='camel', L=1.13, W=0.24, Wc=0.254, Wh=0.275, T=0.04, Ls=0.63, n=3.0, folds=0.014),
]


def hanger_y(x):
    return -HANG_DROP * np.minimum(np.abs(x) / HANG_W, 1.25) ** 1.7


def hanger_rn(x):
    return 0.017 - 0.008 * np.minimum(np.abs(x) / HANG_W, 1.0) ** 1.5


def hanger_parts(mats):
    wood, chrome = mats['hanger'], mats['chrome']
    x = np.linspace(-HANG_W, HANG_W, 13)
    path = np.stack([x, hanger_y(x), np.zeros_like(x)], 1)
    arm = sweep(path, hanger_rn(x), 8, rb=0.0062 - 0.0012 * (np.abs(x) / HANG_W))
    bar_dy = 0.118
    rh = 0.0135 + 0.0028 + 0.0006
    stem = bezier((0, 0.012, 0), (-0.004, 0.07, 0), (-rh, bar_dy - 0.004, 0), 9)
    a = np.linspace(math.pi, -0.55, 14)
    hook = np.stack([rh * np.cos(a), bar_dy + rh * np.sin(a), np.zeros_like(a)], 1)
    wire = np.concatenate([stem, hook[1:]])
    return [(arm, wood), (sweep(wire, 0.0028, 6), chrome), (cyl((0, -0.006, 0), (0, 0.022, 0), 0.0048, 8), chrome)]


def garment_parts(g, rng, mats):
    W, Wc, Wh, T, L, Ls, ne = g['W'], g['Wc'], g['Wh'], g['T'], g['L'], g['Ls'], g['n']
    kind = g['kind']
    mat = fabric(mats, g['color'])
    nseg = 28
    th = np.linspace(0, 2 * math.pi, nseg, endpoint=False) + math.pi / nseg
    c, s = np.cos(th), np.sin(th)
    cx, cz = np.sign(c) * np.abs(c) ** (2 / ne), np.sign(s) * np.abs(s) ** (2 / ne)
    v = np.concatenate([[-0.035, 0.0], np.linspace(0.03, 1.0, 22) ** 1.12])
    if kind == 'dress':
        a = np.interp(v, [-0.035, 0, 0.12, 0.36, 0.52, 1.0], [W - 0.01, W, Wc, Wc - 0.035, Wc, Wh])
    else:
        a = np.interp(v, [-0.035, 0, 0.08, 0.3, 1.0], [W - 0.008, W, W + 0.008, Wc, Wh])
    hem_t = {'coat': 1.35, 'dress': 1.6, 'jacket': 1.15, 'knit': 1.3}.get(kind, 1.7)
    b = np.interp(v, [-0.035, 0, 0.08, 0.22, 0.7, 1.0], [0.0015, 0.0088, 0.6 * T, T, T * (1 + 0.4 * (hem_t - 1)), T * hem_t])
    collar_h, collar_w, collar_t = {'shirt': (0.016, 0.068, 0.012), 'blouse': (0.009, 0.075, 0.007), 'knit': (0.007, 0.08, 0.01),
                                    'jacket': (0.026, 0.1, 0.018), 'coat': (0.034, 0.11, 0.024), 'dress': (0.0, 0.07, 0.0)}[kind]
    X = a[:, None] * cx[None, :]
    xs = np.clip(X, -HANG_W, HANG_W)
    bump_ = np.clip(1 - (X / collar_w) ** 2, 0, 1) * (v[:, None] <= 0.02)
    ys = hanger_y(xs) + hanger_rn(xs) + 0.004 + collar_h * bump_
    pad = 0.008 if kind in ('jacket', 'coat') else 0.0
    ys = ys + pad * np.clip(np.abs(X) / W, 0, 1) ** 3
    y_hem = 0.017 + 0.004 - L
    rel = np.clip(np.abs(X) / np.maximum(a[:, None], 1e-6), 0, 1)
    if kind == 'shirt':
        y_hem = y_hem + 0.075 * rel ** 2.2
    elif kind == 'blouse':
        y_hem = y_hem + 0.03 * rel ** 2
    vv = np.clip(v, 0, 1)[:, None]
    Y = ys * (1 - vv) + y_hem * vv
    Y = np.where(v[:, None] < 0, Y + 0.003, Y)
    Z = (b[:, None] + collar_t * bump_) * cz[None, :]
    env = smoothstep(0.03, 0.55, vv) * (0.45 + 0.55 * vv)
    F = np.zeros_like(X)
    amp = g['folds'] * 1.9
    for k in range(4):
        lam = rng.uniform(0.11, 0.3)
        F += amp * rng.uniform(0.4, 1.0) * np.sin(2 * np.pi * X / lam + rng.uniform(0, 6.3) + 1.1 * np.sin(2 * np.pi * Y / 0.9 + rng.uniform(0, 6.3)))
    front = (cz[None, :] > 0)
    F2 = 0.5 * amp * np.sin(2 * np.pi * X / rng.uniform(0.09, 0.16) + rng.uniform(0, 6.3)) * smoothstep(0.45, 1.0, vv)
    Z = Z + F * env + np.where(front, F2, -F2 * 0.6)
    side = 0.016 if kind not in ('jacket', 'coat') else 0.009
    X = X + side * np.sin(2 * np.pi * Y / rng.uniform(0.28, 0.42) + rng.uniform(0, 6.3)) * env * cx[None, :] ** 3
    X = X * (1 + 0.05 * np.sin(2 * np.pi * Y / 0.5 + rng.uniform(0, 6.3)) * env)
    if kind in ('dress', 'coat'):
        Z = Z + 0.014 * np.sin(2 * np.pi * X / 0.19 + rng.uniform(0, 6.3)) * smoothstep(0.5, 1.0, vv)
    if kind in ('jacket', 'coat'):
        yl = Y - float(ys[1].max())
        lap = np.clip(-yl / 0.34, 0, 1)
        xv = (0.035 + 0.075 * (1 - lap)) * (1 - 0.15 * lap)
        ridge = np.exp(-((np.abs(X) - xv) / 0.016) ** 2) * (lap > 0) * (lap < 1) * np.sin(np.pi * lap) ** 0.5
        Z = Z + 0.011 * ridge * front
        Z = Z - 0.006 * np.exp(-(X / 0.012) ** 2) * smoothstep(0.3, 0.45, vv) * front
    G = np.stack([X, Y, Z], -1)
    UV = np.stack([np.broadcast_to(np.arange(nseg) / nseg, X.shape), Y], -1)
    parts = [(quads_to_tris(grid_part(G, UV, wrap=True)), mat)]
    if Ls > 0:
        ysh = float(hanger_y(np.array(HANG_W))) + 0.012 + pad
        thick = kind in ('jacket', 'coat', 'knit')
        for sd in (-1, 1):
            out = rng.uniform(0.0, 0.03)
            fw = rng.normal(0.012, 0.01)
            pts = [(sd * (W - 0.012), ysh - 0.008, 0.004), (sd * (W + 0.03), ysh - 0.08, 0.006),
                   (sd * (Wc + 0.03 + 0.4 * out), ysh - 0.28, 0.008 + 0.5 * fw),
                   (sd * (Wc + 0.026 + out), ysh - Ls * 0.72, 0.01 + fw),
                   (sd * (Wc + 0.02 + out), ysh - Ls, 0.012 + 1.3 * fw)]
            path = catmull(pts, 4)
            u = np.linspace(0, 1, len(path))
            rn = np.interp(u, [0, 0.15, 0.85, 0.9, 1], [0.072, 0.064, 0.05, 0.054, 0.054]) * (1.05 if thick else 1.0)
            rb = np.interp(u, [0, 0.2, 1], [0.03, 0.024, 0.018]) * (1.8 if thick else 1.0)
            rn = rn + 0.005 * np.sin(u * 15 + rng.uniform(0, 6)) * u
            rb = rb * (1 + 0.25 * np.sin(u * 11 + rng.uniform(0, 6)) * u)
            parts.append((sweep(path, rn, 12, rb=rb), mat))
    return parts


def build_rail(coll, mats):
    rail, rubber, chrome = mats['rail'], mats['rubber'], mats['chrome']
    org = (RAIL_POS[0], FLOOR_Y, RAIL_POS[2])
    r_t, rr = 0.0135, 0.06
    bar_c = BAR_TOP - r_t
    h = RAIL_HALF
    a1 = np.linspace(math.pi, math.pi / 2, 7)
    a2 = np.linspace(math.pi / 2, 0, 7)
    path = [np.array([[0, 0.1, -h], [0, bar_c - rr - 0.4, -h]]),
            np.stack([np.zeros(7), (bar_c - rr) + rr * np.sin(a1), (-h + rr) + rr * np.cos(a1)], 1),
            np.stack([np.zeros(7), (bar_c - rr) + rr * np.sin(a2), (h - rr) + rr * np.cos(a2)], 1),
            np.array([[0, bar_c - rr - 0.4, h], [0, 0.1, h]])]
    frame = np.concatenate(path)
    P = [(sweep(frame, r_t, 14), rail)]
    for zc in (-h, h):
        P.append((cyl((-0.27, 0.085, zc), (0.27, 0.085, zc), 0.016, 12), rail))
        P.append((rbox((0.03, 0.03, 0.022), 0.006, 1, 1, (0, 0.095, zc)), rail))
        for xc in (-0.25, 0.25):
            P.append((cyl((xc, 0.06, zc), (xc, 0.069, zc), 0.019, 12), chrome))
            for sx in (-1, 1):
                P.append((rbox((0.0018, 0.017, 0.012), 0.001, 1, 1, (xc + sx * 0.0145, 0.044, zc + 0.006)), chrome))
            P.append((cyl((xc - 0.012, 0.031, zc + 0.01), (xc + 0.012, 0.031, zc + 0.01), 0.031, 14), rubber))
    P.append((cyl((0, 0.16, -h), (0, 0.16, h), 0.011, 10), rail))
    obs = [make_object('Clothes_Rail', P, coll, origin=org, yaw=RAIL_YAW, sharp_angle=50)]
    rng = np.random.default_rng(404)
    slots = []
    zc = h - 0.075
    for i, g in enumerate(GARMENTS):
        if i:
            zc -= GARMENTS[i - 1]['T'] + g['T'] + rng.uniform(0.012, 0.04)
        slots.append(zc)
    span = slots[0] - slots[-1]
    lim = 2 * (h - 0.075)
    if span > lim:
        slots = [slots[0] - (slots[0] - z) * lim / span for z in slots]
    neck_y = bar_c - 0.118
    hp = hanger_parts(mats)
    hangers = []
    for i, (g, zc) in enumerate(zip(GARMENTS, slots)):
        R = rot_y(math.radians(6) + rng.normal(0, math.radians(3.0))) @ rot_x(rng.normal(0, math.radians(1.6))) @ rot_z(rng.normal(0, math.radians(1.0)))
        t = np.array([rng.normal(0, 0.004), neck_y, zc])
        for p, m in hp:
            hangers.append((p.moved(R, t), m))
        gp = [(p.moved(R, t), m) for p, m in garment_parts(g, rng, mats)]
        name = 'Garment_%02d_%s_%s' % (i + 1, g['kind'].capitalize(), g['color'].capitalize())
        obs.append(make_object(name, gp, coll, origin=org, yaw=RAIL_YAW))
    obs.append(make_object('Hangers', [(quads_to_tris(p), m) for p, m in hangers], coll, origin=org, yaw=RAIL_YAW, sharp_angle=60))
    return obs


# ---------------------------------------------------------------- heap of clothes (cloth simulation)
HEAP_POS = (2.55, 0.0, 1.1)
HEAP_PIECES = [
    dict(shape='tee', w=1.0, h=0.95, body=0.52, top=0.24, color='camel', yaw=15, off=(0.02, -0.02)),
    dict(shape='tee', w=0.92, h=0.62, body=0.47, top=0.19, color='taupe', yaw=80, off=(0.08, -0.07)),
    dict(shape='rect', w=0.62, h=0.72, color='grey', yaw=35, off=(0.09, 0.05)),
    dict(shape='tee', w=0.88, h=0.64, body=0.45, top=0.17, color='cream', yaw=-20, off=(-0.06, 0.09)),
    dict(shape='rect', w=0.48, h=1.15, color='red', yaw=-50, off=(-0.04, 0.02)),
    dict(shape='tee', w=0.82, h=0.56, body=0.42, top=0.16, color='beige', yaw=120, off=(0.06, -0.08)),
]


def cloth_piece_mesh(spec, res=0.03):
    w, h = spec['w'], spec['h']
    nu, nv = int(round(w / res)) + 1, int(round(h / res)) + 1
    us, vs = np.linspace(-w / 2, w / 2, nu), np.linspace(-h / 2, h / 2, nv)
    U, Vv = np.meshgrid(us, vs)
    keep = np.ones((nv, nu), bool)
    if spec['shape'] == 'tee':
        keep = (np.abs(U) <= spec['body'] / 2 + 1e-9) | (Vv >= h / 2 - spec['top'] - 1e-9)
    idx = -np.ones((nv, nu), np.int64)
    idx[keep] = np.arange(keep.sum())
    faces = []
    for j in range(nv - 1):
        for i in range(nu - 1):
            q = [idx[j, i], idx[j, i + 1], idx[j + 1, i + 1], idx[j + 1, i]]
            if min(q) >= 0:
                faces.append(q)
    return np.stack([U[keep], Vv[keep]], 1), np.array(faces)


def taubin(V, F, iters=6, lam=0.5, mu=-0.53):
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 3]], F[:, [3, 0]]])
    E = np.unique(np.sort(E, 1), axis=0)
    deg = np.zeros(len(V))
    np.add.at(deg, E[:, 0], 1)
    np.add.at(deg, E[:, 1], 1)
    for k in range(iters * 2):
        S = np.zeros_like(V)
        np.add.at(S, E[:, 0], V[E[:, 1]])
        np.add.at(S, E[:, 1], V[E[:, 0]])
        V = V + (lam if k % 2 == 0 else mu) * (S / np.maximum(deg, 1)[:, None] - V)
    return V


def build_heap(coll, mats, sim=True, frames=130):
    """Garments tossed onto the floor: tented sheets stacked above the spot, one cloth object with self-collision."""
    rng = np.random.default_rng(77)
    org = np.array([HEAP_POS[0], FLOOR_Y, HEAP_POS[2]])
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, frames
    Vs, Fs, UVs, Ms, n = [], [], [], [], 0
    for k, spec in enumerate(HEAP_PIECES):
        pts, F = cloth_piece_mesh(spec)
        u, v = pts[:, 0], pts[:, 1]
        nz = Smooth2(rng, 0.3, 0.3, 5)
        r2 = (u / (spec['w'] * 0.45)) ** 2 + (v / (spec['h'] * 0.45)) ** 2
        lift = 0.16 * np.exp(-r2 * 1.6) + 0.025 * np.sin(2 * np.pi * u / 0.23 + rng.uniform(0, 6.3)) + 0.02 * nz(u, v)
        local = np.stack([u, 0.07 + 0.11 * k + lift, v], 1)
        R = rot_y(math.radians(spec['yaw'])) @ rot_x(math.radians(rng.normal(0, 4)))
        Vg = local @ R.T + np.array([spec['off'][0], 0.0, spec['off'][1]])
        Vs.append(Vg)
        Fs.append(F + n)
        UVs.append(pts)
        Ms.append(np.full(len(F), k))
        spec['_range'] = (n, n + len(Vg))
        n += len(Vg)
    V0, F0, UV0, M0 = np.concatenate(Vs), np.concatenate(Fs), np.concatenate(UVs), np.concatenate(Ms)
    me = bpy.data.meshes.new('heap_sim')
    me.from_pydata(B(V0 + org).tolist(), [], F0.tolist())
    ob = bpy.data.objects.new('heap_sim', me)
    sc.collection.objects.link(ob)
    plane_me = bpy.data.meshes.new('heap_floor')
    plane_me.from_pydata([(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)], [], [(0, 1, 2, 3)])
    plane = bpy.data.objects.new('heap_floor', plane_me)
    plane.location = (org[0], -org[2], FLOOR_Y)
    sc.collection.objects.link(plane)
    plane.modifiers.new('Collision', 'COLLISION')
    plane.collision.thickness_outer = 0.004
    plane.collision.cloth_friction = 40.0
    co = V0 + org
    wx = XI - 0.15
    wall_me = bpy.data.meshes.new('heap_wall')
    wall_me.from_pydata([(wx, -ZO, 0), (wx, ZO, 0), (wx, ZO, 1.5), (wx, -ZO, 1.5)], [], [(0, 3, 2, 1)])
    wall = bpy.data.objects.new('heap_wall', wall_me)
    sc.collection.objects.link(wall)
    wall.modifiers.new('Collision', 'COLLISION')
    wall.collision.thickness_outer = 0.01
    temp = [wall]
    suit = bpy.data.objects.get('Suitcase')
    if suit is not None:
        suit.modifiers.new('HeapCollision', 'COLLISION')
        suit.collision.thickness_outer = 0.008
    if sim:
        cm = ob.modifiers.new('Cloth', 'CLOTH')
        st = cm.settings
        st.quality = 10
        st.mass = 0.3
        st.air_damping = 1.0
        st.tension_stiffness = st.compression_stiffness = 15.0
        st.shear_stiffness = 8.0
        st.bending_stiffness = 3.0
        st.tension_damping = st.compression_damping = st.shear_damping = 5.0
        st.bending_damping = 0.5
        cs = cm.collision_settings
        cs.collision_quality = 5
        cs.distance_min = 0.005
        cs.use_self_collision = True
        cs.self_distance_min = 0.006
        cs.self_friction = 10.0
        cm.point_cache.frame_start = 1
        cm.point_cache.frame_end = frames
        for f in range(1, frames + 1):
            sc.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        ev = ob.evaluated_get(dg)
        flat = np.zeros(len(ev.data.vertices) * 3)
        ev.data.vertices.foreach_get('co', flat)
        Vb = taubin(flat.reshape(-1, 3), F0, 5)
        co = np.stack([Vb[:, 0], np.maximum(Vb[:, 2], FLOOR_Y + 0.002), -Vb[:, 1]], 1)
        log('heap simulated')
    bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.objects.remove(plane, do_unlink=True)
    for t in temp:
        bpy.data.objects.remove(t, do_unlink=True)
    if suit is not None:
        suit.modifiers.remove(suit.modifiers['HeapCollision'])
    sc.frame_set(1)
    parts = []
    for k, spec in enumerate(HEAP_PIECES):
        a0, a1 = spec['_range']
        Fk = F0[M0 == k]
        parts.append((Part(co[a0:a1] - org, Fk - a0, UV0[Fk]), fabric(mats, spec['color'])))
    ob = make_object('Clothes_Heap', parts, coll, origin=org)
    sol = ob.modifiers.new('Solidify', 'SOLIDIFY')
    sol.thickness = 0.003
    sol.offset = 0.0
    apply_modifiers(ob)
    return ob


def build_props(coll, mats, sim=True):
    prop_materials(mats)
    left, right = collection('Props_Left', coll), collection('Props_Right', coll)
    build_trunk(left, mats)
    build_rail(left, mats)
    build_small_lamp(left, mats)
    info = {'studio': build_studio_light(right, mats)}
    build_suitcase(right, mats)
    build_road_case(right, mats)
    build_heap(right, mats, sim=sim)
    return info



if __name__ == '__main__':
    main()
