"""Builds the gilded baroque floor mirror for the zhang-2024 stage booth (Avatars & Identity), with Cycles look-dev.
    blender -b --factory-startup -noaudio --python-exit-code 1 -P tools/scene/mirror.py -- --work DIR [options]
Writes DIR/mirror.blend and DIR/mirror-previews/*.png, plus the antique-glass maps DIR/mirror_glass_*.png.
Collections: "Mirror" is the low-poly export set (Mirror_Frame: gold + dark wood sharing one UV atlas, Mirror_Glass);
"Mirror HP" holds the carved high-poly frame for look-dev and as the bake source; "Mirror Studio" is the preview rig.
Blender axes: Z up, the mirror faces -Y (glTF +Z), origin at its floor-contact centre; it leans back 5 degrees."""
import argparse
import math
import os
import sys
import time

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import mirror_relief as R  # noqa: E402

LEAN = math.radians(5.0)
KEY_POS = Vector((0.0, -1.25, 4.1))
EYE_POS = Vector((0.0, -3.0, 1.6))
SIDE_POS = Vector((2.2, -1.15, 2.1))
TARGET = Vector((0.0, 0.08, 1.0))
LP_TRIS = {'crest': 20000, 'corner': 6500, 'apron': 3000}
AO_SAMPLES = 6


def options():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default=os.environ.get('MIRROR_WORK', os.path.join(os.getcwd(), 'scene-work')))
    ap.add_argument('--views', default='front,three,crest')
    ap.add_argument('--looks', default='pbr')
    ap.add_argument('--samples', type=int, default=160)
    ap.add_argument('--res', type=int, default=960)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--no-blend', action='store_true')
    ap.add_argument('--tag', default='')
    return ap.parse_args(argv)


def log(*a):
    print('[mirror %6.1fs]' % (time.time() - T0), *a, flush=True)


# ---------------------------------------------------------------- mesh helpers

def to_b(u, v, d):
    return np.stack([np.asarray(u, float), -np.asarray(d, float), np.asarray(v, float)], -1)


def make_mesh(name, verts, faces, mat_index=None, smooth=True, sharp=None, uv=None):
    verts = np.ascontiguousarray(verts, np.float32).reshape(-1, 3)
    groups = faces if isinstance(faces, (list, tuple)) else [faces]
    groups = [np.asarray(g, np.int64) for g in groups if len(g)]
    sizes = np.concatenate([np.full(len(g), g.shape[1], np.int32) for g in groups])
    loops = np.concatenate([g.ravel() for g in groups]).astype(np.int32)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set('co', verts.ravel())
    me.loops.add(len(loops))
    me.loops.foreach_set('vertex_index', loops)
    me.polygons.add(len(sizes))
    starts = np.zeros(len(sizes), np.int32)
    starts[1:] = np.cumsum(sizes)[:-1]
    me.polygons.foreach_set('loop_start', starts)
    me.update(calc_edges=True)
    if mat_index is not None:
        me.polygons.foreach_set('material_index', np.asarray(mat_index, np.int32))
    if smooth:
        me.shade_smooth()
    if sharp is not None:
        me.set_sharp_from_angle(angle=math.radians(sharp))
    if uv is not None:
        layer = me.uv_layers.new(name='UVMap')
        layer.data.foreach_set('uv', np.asarray(uv, np.float32).ravel())
    return me


def add_object(name, me, coll, mats=()):
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    for m in mats:
        me.materials.append(m)
    return ob


def collection(name, parent=None):
    c = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(c)
    return c


# ---------------------------------------------------------------- frame moulding

A, B, VC = R.SIGHT_W / 2, R.SIGHT_H / 2, R.GLASS_CENTER


def side_map(k, s, f):
    """Point on side k (0 bottom, 1 right, 2 top, 3 left; counter-clockwise seen from the front) of the mitred
    frame at moulding offset s from the sight edge and fraction f along the side."""
    s, f = np.asarray(s, float), np.asarray(f, float)
    if k == 0:
        return -(A + s) + f * 2 * (A + s), VC - (B + s) + 0 * f
    if k == 1:
        return (A + s) + 0 * f, VC - (B + s) + f * 2 * (B + s)
    if k == 2:
        return (A + s) - f * 2 * (A + s), VC + (B + s) + 0 * f
    return -(A + s) + 0 * f, VC + (B + s) - f * 2 * (B + s)


def side_len(k, s):
    return 2 * (A + s) if k in (0, 2) else 2 * (B + s)


def sweep(prof, with_uv=False):
    """Mitred sweep of the closed section round the sight rectangle; one strip per side so the mitres stay sharp.
    Faces on the back and in the rebate get material 1 (bare wood). With with_uv, each half of a long side (and each
    short side) is unrolled into its own island in metres: u along the section, v along the side."""
    P = np.array([(s, d) for s, d, _ in prof])
    n = len(P)
    S, D = P[:, 0], P[:, 1]
    i = np.arange(n)
    i2 = (i + 1) % n
    on_back = (D[i] < 0.0005) & (D[i2] < 0.0005)
    in_rebate = (np.maximum(S[i], S[i2]) <= R.REBATE + 1e-6) & (np.maximum(D[i], D[i2]) < R.GLASS_FRONT + 0.001)
    seg_wood = on_back | in_rebate
    seg_len = np.hypot(S[i2] - S, D[i2] - D) * np.where(seg_wood, 0.3, 1.0)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    verts, faces, mats, uvs = [], [], [], []
    for k in range(4):
        nseg = 2 if k in (1, 3) else 1
        fs = np.linspace(0, 1, nseg + 1)
        base = sum(len(v) for v in verts)
        rows = []
        for f in fs:
            u_, v_ = side_map(k, S, f)
            rows.append(to_b(u_, v_, D))
        verts.append(np.concatenate(rows))
        for j in range(nseg):
            r0, r1 = base + j * n, base + (j + 1) * n
            faces.append(np.stack([r0 + i, r0 + i2, r1 + i2, r1 + i], 1))
            mats.append(seg_wood.astype(np.int32))
            if with_uv:
                L = side_len(k, S)
                ua, ub, La, Lb = arc[i], arc[i + 1], L[i], L[i2]
                dv = fs[j + 1] - fs[j]
                q = np.stack([np.stack([ua, 0 * ua], -1), np.stack([ub, 0 * ub], -1),
                              np.stack([ub, dv * Lb], -1), np.stack([ua, dv * La], -1)], 1)
                q[..., 0] += 3.0 * (2 * k + j)
                uvs.append(q.reshape(-1, 2))
    out = (np.concatenate(verts), np.concatenate(faces), np.concatenate(mats))
    return out + (np.concatenate(uvs),) if with_uv else out


def strip_faces(nj, ni, base=0):
    j, i = np.meshgrid(np.arange(nj - 1), np.arange(ni - 1), indexing='ij')
    idx = lambda jj, ii: base + jj * ni + ii
    return np.stack([idx(j, i), idx(j, i + 1), idx(j + 1, i + 1), idx(j + 1, i)], -1).reshape(-1, 4)


def bead_strip(step, nphi=13):
    """Bead-and-reel run round the inner bead (carved geometry for the high-poly frame)."""
    sb, db, rb = 0.0142, 0.0292, 0.0039
    verts, faces = [], []
    for k in range(4):
        L = side_len(k, sb)
        N = int(round(L / 0.0086))
        N += 1 - N % 2
        p = L / N
        nl = int(L / step) + 1
        f = np.linspace(0, 1, nl)
        x = ((f - 0.5) * L / p + 0.5) % 1.0 - 0.5
        bead = rb * np.sqrt(np.clip(1 - (x / 0.36) ** 2, 0, 1))
        reel = 0.80 * rb * np.sqrt(np.clip(1 - ((np.abs(x) - 0.462) / 0.032) ** 2, 0, 1))
        r = np.maximum(np.maximum(bead, reel), 0.36 * rb)
        phi = np.linspace(0, np.pi, nphi)
        s = sb - r[:, None] * np.cos(phi)[None, :]
        d = db + r[:, None] * np.sin(phi)[None, :]
        u, v = side_map(k, s, f[:, None] + 0 * s)
        base = sum(len(vv) for vv in verts)
        verts.append(to_b(u.ravel(), v.ravel(), d.ravel()))
        faces.append(strip_faces(nl, nphi, base))
    return np.concatenate(verts), np.concatenate(faces)


def leaf_band(step, ny=40):
    """Leaf-and-dart carved down the outer ogee (high-poly), riding on the low-poly ogee surface."""
    y = np.linspace(0.03, 0.97, ny)
    s0 = 0.074 + 0.0255 * y
    d0 = 0.054 - 0.0255 * (3 * y * y - 2 * y ** 3)
    ds = np.full_like(y, 0.0255)
    dd = -0.0255 * (6 * y - 6 * y * y)
    nn = np.hypot(ds, dd)
    ns, nd = -dd / nn, ds / nn
    verts, faces = [], []
    for k in range(4):
        L = side_len(k, 0.087)
        N = int(round(L / 0.0205))
        N += 1 - N % 2
        p = L / N
        nl = int(L / step) + 1
        f = np.linspace(0, 1, nl)
        x = (((f - 0.5) * L / p + 0.5) % 1.0 - 0.5)[:, None]
        Y = y[None, :]
        yl = np.clip((Y - 0.06) / 0.86, 0, 1)
        w = 0.40 * np.sin(np.pi * np.clip(yl, 1e-6, 1) ** 0.62) ** 0.85 + 1e-6
        q = np.abs(x) / w
        leaf = 0.0017 * np.sqrt(np.clip(1 - q * q, 0, 1)) * R.smoothstep(0.0, 0.14, yl)
        leaf -= 0.0006 * np.exp(-(x / 0.035) ** 2) * ((yl > 0.12) & (yl < 0.86)) * (q < 1)
        leaf = np.where(q < 1, np.maximum(leaf, 0.0), 0.0)
        xd = 0.5 - np.abs(x)
        wd = 0.075 * np.clip(1 - Y / 0.74, 0, 1) + 1e-6
        dart = 0.0011 * np.sqrt(np.clip(1 - (xd / wd) ** 2, 0, 1))
        rel = np.maximum(leaf, dart)
        s = s0[None, :] + ns[None, :] * rel
        d = d0[None, :] + nd[None, :] * rel
        u, v = side_map(k, s, f[:, None] + 0 * s)
        base = sum(len(vv) for vv in verts)
        verts.append(to_b(u.ravel(), v.ravel(), d.ravel()))
        faces.append(strip_faces(nl, ny, base))
    return np.concatenate(verts), np.concatenate(faces)


# ---------------------------------------------------------------- carved ornaments

def relief_mesh(F, h, place, mirrored, smooth_iters=3):
    """Closed solid from a height field: front surface, vertical walls round the (smoothed) outline and a flat back
    at ORN_BACK. Parts sunk below the moulding surface are trimmed. Back faces get material 1."""
    U, V = np.meshgrid(F.u, F.v)
    Ua, Va = place(U, V)
    live = (h > R.EMPTY) & (h >= R.mould_front(Ua, Va) - 0.0015)
    q = live[:-1, :-1] & live[1:, :-1] & live[:-1, 1:] & live[1:, 1:]
    used = np.zeros_like(live)
    used[:-1, :-1] |= q
    used[1:, :-1] |= q
    used[:-1, 1:] |= q
    used[1:, 1:] |= q
    vid = np.full(live.shape, -1, np.int64)
    nv = int(used.sum())
    vid[used] = np.arange(nv)
    jj, ii = np.nonzero(used)
    pu, pv, pd = Ua[jj, ii].copy(), Va[jj, ii].copy(), h[jj, ii].copy()
    qj, qi = np.nonzero(q)
    front = np.stack([vid[qj, qi], vid[qj, qi + 1], vid[qj + 1, qi + 1], vid[qj + 1, qi]], 1)
    ny, nx = live.shape
    qa = np.zeros((ny, nx - 1), bool)
    qa[:-1] = q
    qb = np.zeros((ny, nx - 1), bool)
    qb[1:] = q
    qr = np.zeros((ny - 1, nx), bool)
    qr[:, :-1] = q
    ql = np.zeros((ny - 1, nx), bool)
    ql[:, 1:] = q
    e = []
    j, i = np.nonzero(qa & ~qb)
    e.append(np.stack([vid[j, i + 1], vid[j, i]], 1))
    j, i = np.nonzero(qb & ~qa)
    e.append(np.stack([vid[j, i], vid[j, i + 1]], 1))
    j, i = np.nonzero(qr & ~ql)
    e.append(np.stack([vid[j, i], vid[j + 1, i]], 1))
    j, i = np.nonzero(ql & ~qr)
    e.append(np.stack([vid[j + 1, i], vid[j, i]], 1))
    edges = np.concatenate(e)
    for _ in range(smooth_iters):
        acc = np.zeros((nv, 2))
        cnt = np.zeros(nv)
        for a_, b_ in ((edges[:, 0], edges[:, 1]), (edges[:, 1], edges[:, 0])):
            np.add.at(acc, a_, np.stack([pu[b_], pv[b_]], 1))
            np.add.at(cnt, a_, 1)
        m = cnt > 0
        pu[m] = 0.5 * pu[m] + 0.5 * acc[m, 0] / cnt[m]
        pv[m] = 0.5 * pv[m] + 0.5 * acc[m, 1] / cnt[m]
    walls = np.stack([edges[:, 0], edges[:, 1], edges[:, 1] + nv, edges[:, 0] + nv], 1)
    back = front[:, ::-1] + nv
    P = np.concatenate([to_b(pu, pv, pd), to_b(pu, pv, np.full(nv, R.ORN_BACK))])
    faces = np.concatenate([front, walls, back])
    if mirrored:
        faces = faces[:, ::-1]
    mats = np.concatenate([np.zeros(len(front) + len(walls), np.int32), np.ones(len(back), np.int32)])
    return P, faces, mats


def ornament_parts(scale=1.0):
    """(name, field, height, placement, mirrored) for crest, four corners and apron; scale multiplies pixel size."""
    out = []
    F, h = R.crest(0.0006 * scale)
    out.append(('crest', F, h, lambda U, V: (U, R.FRAME_TOP + V), False))
    F, h = R.corner(0.0007 * scale)
    hw = R.OUT_W / 2
    out.append(('corner_tr', F, h, lambda U, V: (hw + U, R.FRAME_TOP + V), False))
    out.append(('corner_tl', F, h, lambda U, V: (-hw - U, R.FRAME_TOP + V), True))
    out.append(('corner_br', F, h, lambda U, V: (hw + U, R.FRAME_BOTTOM - V), True))
    out.append(('corner_bl', F, h, lambda U, V: (-hw - U, R.FRAME_BOTTOM - V), False))
    F, h = R.apron(0.0006 * scale)
    out.append(('apron', F, h, lambda U, V: (U, R.FRAME_BOTTOM + V), False))
    return out


# ---------------------------------------------------------------- glass, back, easel

def glass_mesh():
    W, H = R.SIGHT_W + 2 * R.REBATE, R.SIGHT_H + 2 * R.REBATE
    u0, v0 = -W / 2, VC - H / 2
    bev, d_edge = 0.022, 0.0125
    outer = np.array([(u0, v0), (u0 + W, v0), (u0 + W, v0 + H), (u0, v0 + H)])
    inner = outer + np.array([(bev, bev), (-bev, bev), (-bev, -bev), (bev, -bev)])
    P = np.concatenate([to_b(outer[:, 0], outer[:, 1], np.full(4, d_edge)),
                        to_b(inner[:, 0], inner[:, 1], np.full(4, R.GLASS_FRONT)),
                        to_b(outer[:, 0], outer[:, 1], np.full(4, R.GLASS_BACK))])
    faces = [[4, 5, 6, 7]]
    for k in range(4):
        k2 = (k + 1) % 4
        faces.append([k, k2, 4 + k2, 4 + k])
        faces.append([8 + k, 8 + k2, k2, k])
    faces.append([11, 10, 9, 8])
    faces = np.array(faces)
    uv_of = lambda idx: np.stack([(P[idx, 0] - u0) / W, (P[idx, 2] - v0) / H], 1)
    uv = np.concatenate([uv_of(f) for f in faces])
    return P, faces, uv


def box(c, size, rot=None):
    """Axis box (Blender axes) as 8 verts + 6 quads; optional 3x3 rotation about its centre."""
    hx, hy, hz = np.asarray(size) / 2
    p = np.array([(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
                  (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)])
    if rot is not None:
        p = p @ np.asarray(rot).T
    p = p + np.asarray(c)
    f = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    return p, f


def easel(M):
    """Backboard, hinge block and a slim splayed strut down to the floor behind (dark wood), in final coordinates."""
    parts = []
    W, H = R.SIGHT_W + 2 * R.REBATE - 0.002, R.SIGHT_H + 2 * R.REBATE - 0.002
    p, f = box((0, -(R.BOARD_BACK + R.GLASS_BACK) / 2, VC), (W, R.GLASS_BACK - R.BOARD_BACK, H))
    parts.append((p @ np.asarray(M)[:3, :3].T + np.asarray(M)[:3, 3], f))
    hinge_up = np.array([0.0, 0.016, 1.30])
    hinge = np.asarray(M)[:3, :3] @ hinge_up + np.asarray(M)[:3, 3]
    p, f = box(hinge_up, (0.11, 0.032, 0.07))
    parts.append((p @ np.asarray(M)[:3, :3].T + np.asarray(M)[:3, 3], f))
    top = hinge + np.array([0.0, 0.020, -0.02])
    foot = np.array([0.0, top[1] + 0.50, 0.012])
    axis = foot - top
    L = np.linalg.norm(axis)
    zc = axis / L
    xc = np.array([1.0, 0.0, 0.0])
    yc = np.cross(zc, xc)
    rot = np.stack([xc, yc, zc], 1)
    p, f = box((top + foot) / 2, (0.042, 0.022, L), rot)
    parts.append((p, f))
    p, f = box((0.0, foot[1] + 0.004, 0.012), (0.30, 0.050, 0.024))
    parts.append((p, f))
    verts, faces = [], []
    for p, f in parts:
        faces.append(f + sum(len(v) for v in verts))
        verts.append(p)
    return np.concatenate(verts), np.concatenate(faces)


# ---------------------------------------------------------------- antique glass maps

def fbm(shape, cells, octaves, rng, gain=0.5):
    h, w = shape
    out = np.zeros(shape)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        cy, cx = max(2, int(cells[0] * 2 ** o)), max(2, int(cells[1] * 2 ** o))
        g = rng.random((cy + 1, cx + 1))
        yy = np.linspace(0, cy, h)
        xx = np.linspace(0, cx, w)
        y0 = np.minimum(yy.astype(int), cy - 1)
        x0 = np.minimum(xx.astype(int), cx - 1)
        ty = R.smoothstep(0, 1, yy - y0)[:, None]
        tx = R.smoothstep(0, 1, xx - x0)[None, :]
        a = g[y0][:, x0]
        b = g[y0][:, x0 + 1]
        c = g[y0 + 1][:, x0]
        d = g[y0 + 1][:, x0 + 1]
        out += amp * ((a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty)
        tot += amp
        amp *= gain
    return out / tot


def glass_maps(w=1024, h=2048, seed=11):
    """Masks (speckle, foxing, scratches, bevel) and the finished unlit look of the smoky antique glass, both laid
    out over the whole glass panel (UV 0..1, the outer 8 mm hide under the frame)."""
    rng = np.random.default_rng(seed)
    Wm, Hm = R.SIGHT_W + 2 * R.REBATE, R.SIGHT_H + 2 * R.REBATE
    X, Y = np.meshgrid((np.arange(w) + 0.5) / w * Wm, (np.arange(h) + 0.5) / h * Hm)
    inset = np.minimum(np.minimum(X, Wm - X), np.minimum(Y, Hm - Y)) - R.REBATE
    n1 = fbm((h, w), (10, 4), 5, rng)
    n2 = fbm((h, w), (40, 16), 4, rng)
    n3 = fbm((h, w), (3, 1), 3, rng)
    fox = R.smoothstep(0.070, 0.004, inset + 0.05 * (n1 - 0.5)) * R.smoothstep(0.35, 0.75, n2 * 0.6 + n1 * 0.6)
    fox = np.clip(fox, 0, 1)
    speck = np.zeros((h, w))
    pxu, pxv = Wm / w, Hm / h
    def dots(n, rmin, rmax, amp, weight=None):
        if weight is None:
            pts = rng.random((n, 2)) * [Wm, Hm]
        else:
            cand = rng.random((n * 6, 2)) * [Wm, Hm]
            rows = np.clip((cand[:, 1] / Hm * h).astype(int), 0, h - 1)
            cols = np.clip((cand[:, 0] / Wm * w).astype(int), 0, w - 1)
            keep = rng.random(len(cand)) < weight[rows, cols]
            pts = cand[keep][:n]
        rs = rmin + (rmax - rmin) * rng.random(len(pts)) ** 3
        amps = amp * (0.35 + 0.65 * rng.random(len(pts)))
        for (x, y), r, a in zip(pts, rs, amps):
            i, j = int(x / pxu), int(y / pxv)
            ki, kj = int(r / pxu) + 2, int(r / pxv) + 2
            j0, j1, i0, i1 = max(0, j - kj), min(h, j + kj + 1), max(0, i - ki), min(w, i + ki + 1)
            if j1 <= j0 or i1 <= i0:
                continue
            dd = np.hypot(X[j0:j1, i0:i1] - x, Y[j0:j1, i0:i1] - y)
            v = a * np.clip(1.25 - dd / r, 0, 1) ** 1.5
            np.maximum(speck[j0:j1, i0:i1], v, out=speck[j0:j1, i0:i1])
    dots(750, 0.00035, 0.0011, 1.0)
    dots(900, 0.0004, 0.0016, 0.95, weight=0.15 + 0.85 * fox)
    dots(140, 0.0012, 0.0032, 0.55, weight=0.25 + 0.75 * fox)
    scratch = np.zeros((h, w))
    lines = [((0.17, 1.06), (0.46, 1.135), 0.00028, 0.55), ((0.20, 0.52), (0.31, 0.40), 0.0002, 0.32),
             ((0.42, 0.70), (0.50, 0.95), 0.00018, 0.25), ((0.09, 1.38), (0.16, 1.41), 0.0002, 0.3),
             ((0.52, 0.28), (0.57, 0.18), 0.0002, 0.28)]
    for (x0, y0), (x1, y1), wd, a in lines:
        p0, p1 = np.array([x0 + R.REBATE, y0 + R.REBATE]), np.array([x1 + R.REBATE, y1 + R.REBATE])
        dvec = p1 - p0
        L = np.linalg.norm(dvec)
        t = np.clip(((X - p0[0]) * dvec[0] + (Y - p0[1]) * dvec[1]) / (L * L), 0, 1)
        dist = np.hypot(X - (p0[0] + t * dvec[0]), Y - (p0[1] + t * dvec[1]))
        fade = np.sin(np.pi * t) ** 0.4 * (0.6 + 0.4 * n2)
        scratch = np.maximum(scratch, a * fade * np.clip(1 - dist / (wd + pxu * 0.7), 0, 1))
    bevel = np.clip(1 - (inset + R.REBATE) / 0.022, 0, 1)
    side = np.select([(Y - R.REBATE) < 0.022 + 0 * X, (Hm - R.REBATE - Y) < 0.022 + 0 * X], [1.0, 2.0], 0.0)
    halo = np.exp(-(((X - Wm / 2) / 0.36) ** 2 + ((Y - Hm * 0.80) / 0.62) ** 2))
    c0 = np.array([0x15, 0x1a, 0x17]) / 255.0
    c1 = np.array([0x2c, 0x33, 0x2e]) / 255.0
    lift = np.clip(1.0 * halo + 0.12 * (Y / Hm) + 0.12 * (n3 - 0.5), 0, 1)[..., None]
    col = c0 + (c1 - c0) * lift
    col *= (0.92 + 0.16 * n2[..., None] * 0.5)
    fox_col = np.array([0x22, 0x20, 0x18]) / 255.0
    pale = np.array([0x4c, 0x4d, 0x44]) / 255.0
    col = col * (1 - 0.55 * fox[..., None]) + fox_col * 0.55 * fox[..., None]
    pale_m = (fox * R.smoothstep(0.62, 0.8, n2))[..., None]
    col = col * (1 - 0.6 * pale_m) + pale * 0.6 * pale_m
    sil = np.array([0xc4, 0xc6, 0xbb]) / 255.0
    sp = np.clip(speck * (0.85 + 0.5 * halo + 0.3 * fox), 0, 1)[..., None]
    col = col * (1 - sp) + sil * sp
    sc = np.clip(scratch * (0.6 + 0.8 * halo), 0, 1)[..., None]
    col = col * (1 - sc) + np.array([0.62, 0.63, 0.58]) * sc
    bv = bevel[..., None]
    top_c, bot_c, side_c = (np.array(c) / 255 for c in ((0x6a, 0x5c, 0x42), (0x72, 0x70, 0x63), (0x4e, 0x3c, 0x36)))
    bc = np.where((side == 2)[..., None], top_c, np.where((side == 1)[..., None], bot_c, side_c))
    bc = bc * (0.75 + 0.5 * R.smoothstep(0.0, 0.6, bevel)[..., None])
    col = col * (1 - 0.75 * bv) + bc * 0.75 * bv
    line = np.exp(-((inset + R.REBATE - 0.0215) / 0.0007) ** 2)[..., None]
    col = col + (np.array([0.78, 0.74, 0.62]) - col) * 0.55 * line
    col = np.clip(col, 0, 1)
    masks = np.stack([np.clip(speck, 0, 1), fox, np.clip(scratch, 0, 1), bevel], -1)
    return masks, col


def save_image(name, arr, path, colorspace):
    h, w = arr.shape[:2]
    img = bpy.data.images.get(name) or bpy.data.images.new(name, width=w, height=h, alpha=True)
    if tuple(img.size) != (w, h):
        img.scale(w, h)
    img.colorspace_settings.name = colorspace
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :arr.shape[2]] = arr
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    img.pack()
    return img


# ---------------------------------------------------------------- materials

class Nodes:
    def __init__(self, mat):
        self.nt = mat.node_tree
        self.nt.nodes.clear()

    def put(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, sock)
        elif v is not None:
            sock.default_value = v

    def add(self, kind, ins=None, **props):
        n = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        for k, v in (ins or {}).items():
            self.put(n.inputs[k], v)
        return n

    def math(self, op, a, b=0.0, clamp=False):
        n = self.add('ShaderNodeMath', operation=op, use_clamp=clamp)
        self.put(n.inputs[0], a)
        self.put(n.inputs[1], b)
        return n.outputs[0]

    def vmath(self, op, a, b=None):
        n = self.add('ShaderNodeVectorMath', operation=op)
        self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        return n.outputs['Value'] if op in ('DOT_PRODUCT', 'LENGTH', 'DISTANCE') else n.outputs['Vector']

    def vscale(self, v, k):
        n = self.add('ShaderNodeVectorMath', operation='SCALE')
        self.put(n.inputs[0], v)
        self.put(n.inputs[3], k)
        return n.outputs['Vector']

    def mix(self, fac, a, b, blend='MIX'):
        n = self.add('ShaderNodeMix', data_type='RGBA', blend_type=blend, clamp_factor=True)
        self.put(n.inputs[0], fac)
        self.put(n.inputs[6], a)
        self.put(n.inputs[7], b)
        return n.outputs[2]

    def remap(self, v, a, b, c=0.0, d=1.0, smooth=False):
        n = self.add('ShaderNodeMapRange', interpolation_type='SMOOTHSTEP' if smooth else 'LINEAR', clamp=True)
        self.put(n.inputs[0], v)
        for k, x in zip((1, 2, 3, 4), (a, b, c, d)):
            n.inputs[k].default_value = x
        return n.outputs[0]

    def out(self, shader):
        o = self.add('ShaderNodeOutputMaterial', target='ALL')
        self.put(o.inputs['Surface'], shader)


def lin(hexstr):
    c = np.array([int(hexstr[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return (float(c[0]), float(c[1]), float(c[2]), 1.0)


def new_material(name):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    return m, Nodes(m)


def gold_masks(nb):
    """Shared masks: edge (convex, burnished), cav (deep recess), tone (mild hollows), bole (worn-through edges),
    grime (large blotches) and a fine noise."""
    tc = nb.add('ShaderNodeTexCoord')
    geo = nb.add('ShaderNodeNewGeometry')
    ao_s = nb.add('ShaderNodeAmbientOcclusion', {'Distance': 0.006}, samples=AO_SAMPLES, only_local=True)
    ao_l = nb.add('ShaderNodeAmbientOcclusion', {'Distance': 0.030}, samples=AO_SAMPLES, only_local=True)
    edge = nb.remap(geo.outputs['Pointiness'], 0.512, 0.575, smooth=True)
    cav = nb.remap(ao_s.outputs['AO'], 0.40, 0.92, 1.0, 0.0, smooth=True)
    tone = nb.remap(ao_l.outputs['AO'], 0.35, 0.95, 1.0, 0.0, smooth=True)
    big = nb.add('ShaderNodeTexNoise', {'Vector': tc.outputs['Object'], 'Scale': 4.0, 'Detail': 4.0, 'Roughness': 0.55})
    mid = nb.add('ShaderNodeTexNoise', {'Vector': tc.outputs['Object'], 'Scale': 22.0, 'Detail': 5.0, 'Roughness': 0.6})
    fine = nb.add('ShaderNodeTexNoise', {'Vector': tc.outputs['Object'], 'Scale': 120.0, 'Detail': 6.0, 'Roughness': 0.6})
    wear = nb.remap(fine.outputs['Fac'], 0.60, 0.70)
    bole = nb.math('MULTIPLY', nb.math('MULTIPLY', edge, wear), nb.remap(big.outputs['Fac'], 0.48, 0.62))
    grime = nb.math('MULTIPLY', nb.remap(big.outputs['Fac'], 0.30, 0.72, 0.70, 1.0),
                    nb.remap(mid.outputs['Fac'], 0.35, 0.65, 0.86, 1.0))
    return tc, edge, cav, tone, bole, grime, fine


def gold_pbr():
    """Look-dev water gilding: burnished leaf on the high points, matte leaf elsewhere, a darker toned gold in the
    hollows, umber dirt in the deep recesses, a little red bole where edges have worn through."""
    m, nb = new_material('Mirror Gold')
    tc, edge, cav, tone, bole, grime, fine = gold_masks(nb)
    base = nb.mix(edge, (0.62, 0.43, 0.17, 1), (0.94, 0.80, 0.50, 1))
    base = nb.mix(nb.math('MULTIPLY', tone, 0.75), base, (0.34, 0.21, 0.07, 1))
    base = nb.mix(nb.math('MULTIPLY', cav, 0.95), base, (0.030, 0.016, 0.006, 1))
    base = nb.mix(bole, base, (0.28, 0.060, 0.024, 1))
    base = nb.mix(1.0, base, nb.add('ShaderNodeCombineColor', {'Red': grime, 'Green': grime, 'Blue': grime}).outputs[0], 'MULTIPLY')
    metal = nb.math('SUBTRACT', 1.0, nb.math('MAXIMUM', nb.math('MULTIPLY', cav, 0.85), bole), clamp=True)
    rough = nb.math('ADD', nb.math('MULTIPLY', edge, -0.24), 0.36)
    rough = nb.math('ADD', rough, nb.math('MULTIPLY', nb.math('MAXIMUM', cav, nb.math('MULTIPLY', tone, 0.5)), 0.32))
    rough = nb.math('ADD', rough, nb.math('MULTIPLY', fine.outputs['Fac'], 0.08), clamp=True)
    gesso = nb.add('ShaderNodeTexNoise', {'Vector': tc.outputs['Object'], 'Scale': 420.0, 'Detail': 3.0})
    bump = nb.add('ShaderNodeBump', {'Strength': 0.12, 'Distance': 0.0003, 'Height': gesso.outputs['Fac']})
    bsdf = nb.add('ShaderNodeBsdfPrincipled', {'Base Color': base, 'Metallic': metal, 'Roughness': rough,
                                               'Normal': bump.outputs['Normal']})
    nb.out(bsdf.outputs[0])
    return m


def gold_bake():
    """Gold for baking to an unlit texture (bake Combined, or Emit alone for the reflection part). FRAME has no
    reflections, so the emission carries them: the visitor's view (EYE_POS) is mirrored about each normal and lit by
    a stage-like surround - the key lantern (KEY_POS), the warm floor pool below, dim red curtains to the sides -
    tinted by the leaf colour and dimmed in the toned hollows and umber recesses. A small diffuse term lets the
    real stage lights add soft shading and shadows during the bake. Positions are in the mirror's object space;
    the 'Bake Gain' value node scales the whole result."""
    m, nb = new_material('Mirror Gold Bake')
    tc, edge, cav, tone, bole, grime, fine = gold_masks(nb)
    p, n = tc.outputs['Object'], tc.outputs['Normal']
    L = nb.vmath('NORMALIZE', nb.vmath('SUBTRACT', tuple(KEY_POS), p))
    V = nb.vmath('NORMALIZE', nb.vmath('SUBTRACT', p, tuple(EYE_POS)))
    Rv = nb.vmath('REFLECT', V, n)
    rl = nb.math('MAXIMUM', nb.vmath('DOT_PRODUCT', Rv, L), 0.0)
    spec = nb.math('ADD', nb.math('MULTIPLY', nb.math('POWER', rl, 40.0), 1.8),
                   nb.math('ADD', nb.math('MULTIPLY', nb.math('POWER', rl, 8.0), 0.6),
                           nb.math('MULTIPLY', nb.math('POWER', rl, 2.0), 0.45)))
    rxyz = nb.add('ShaderNodeSeparateXYZ', {'Vector': Rv})
    floor = nb.math('MULTIPLY', nb.remap(nb.math('MULTIPLY', rxyz.outputs['Z'], -1.0), -0.10, 0.60, smooth=True), 0.55)
    side = nb.math('MULTIPLY', nb.remap(nb.math('ABSOLUTE', rxyz.outputs['X']), 0.30, 0.90, smooth=True), 0.18)
    env = nb.vmath('ADD', (0.09, 0.075, 0.055), nb.vscale((1.0, 0.86, 0.62), nb.math('ADD', spec, nb.math('MULTIPLY', edge, 0.25))))
    env = nb.vmath('ADD', env, nb.vscale((1.0, 0.76, 0.48), floor))
    env = nb.vmath('ADD', env, nb.vscale((0.9, 0.25, 0.20), side))
    f0 = nb.mix(edge, (0.88, 0.72, 0.38, 1), (1.0, 0.91, 0.64, 1))
    dim = nb.math('MULTIPLY', nb.math('SUBTRACT', 1.0, nb.math('MULTIPLY', cav, 0.92)),
                  nb.math('SUBTRACT', 1.0, nb.math('MULTIPLY', tone, 0.55)))
    dim = nb.math('MULTIPLY', dim, nb.math('MULTIPLY', grime, nb.math('SUBTRACT', 1.0, bole)))
    gain = nb.add('ShaderNodeValue', label='Bake Gain', name='Bake Gain')
    gain.outputs[0].default_value = 1.2
    dim = nb.math('MULTIPLY', dim, gain.outputs[0])
    refl = nb.vscale(nb.vmath('MULTIPLY', env, f0), dim)
    albedo = nb.mix(nb.math('MULTIPLY', cav, 0.85), lin('#5a4420'), lin('#1c1308'))
    albedo = nb.mix(bole, albedo, lin('#6e2a16'))
    albedo = nb.vscale(albedo, gain.outputs[0])
    diff = nb.add('ShaderNodeBsdfDiffuse', {'Color': albedo, 'Roughness': 0.5})
    emis = nb.add('ShaderNodeEmission', {'Color': refl, 'Strength': 1.0})
    add = nb.add('ShaderNodeAddShader')
    nb.put(add.inputs[0], diff.outputs[0])
    nb.put(add.inputs[1], emis.outputs[0])
    nb.out(add.outputs[0])
    return m


def gold_debug():
    m, nb = new_material('Mirror Gold Masks')
    tc, edge, cav, tone, bole, grime, fine = gold_masks(nb)
    col = nb.add('ShaderNodeCombineColor', {'Red': edge, 'Green': cav, 'Blue': tone})
    nb.out(nb.add('ShaderNodeEmission', {'Color': col.outputs[0], 'Strength': 1.0}).outputs[0])
    return m


def wood(bake=False):
    m, nb = new_material('Mirror Wood Bake' if bake else 'Mirror Wood')
    tc = nb.add('ShaderNodeTexCoord')
    mp = nb.add('ShaderNodeMapping', {'Vector': tc.outputs['Object'], 'Scale': (1.0, 1.0, 0.12)})
    wave = nb.add('ShaderNodeTexWave', {'Vector': mp.outputs['Vector'], 'Scale': 18.0, 'Distortion': 6.0, 'Detail': 3.0},
                  wave_type='BANDS', bands_direction='X')
    g = nb.remap(wave.outputs['Fac'], 0.0, 1.0, 0.75, 1.15)
    col = nb.mix(1.0, lin('#2a1a10'), nb.add('ShaderNodeCombineColor', {'Red': g, 'Green': g, 'Blue': g}).outputs[0], 'MULTIPLY')
    if bake:
        bsdf = nb.add('ShaderNodeBsdfDiffuse', {'Color': col})
    else:
        bsdf = nb.add('ShaderNodeBsdfPrincipled', {'Base Color': col, 'Roughness': 0.55})
    nb.out(bsdf.outputs[0])
    return m


def glass_pbr(look_img, mask_img):
    """Look-dev antique mirror: the finished glass map (haze, specks, foxing) as the light it scatters, under a
    dim, slightly foxed silver reflection with a clear glass coat."""
    m, nb = new_material('Mirror Glass')
    uvn = nb.add('ShaderNodeUVMap', uv_map='UVMap')
    look = nb.add('ShaderNodeTexImage', {'Vector': uvn.outputs['UV']}, image=look_img, interpolation='Cubic')
    tex = nb.add('ShaderNodeTexImage', {'Vector': uvn.outputs['UV']}, image=mask_img, interpolation='Cubic')
    sep = nb.add('ShaderNodeSeparateColor', {'Color': tex.outputs['Color']})
    fox = sep.outputs['Green']
    silver = nb.mix(fox, (0.16, 0.165, 0.15, 1), (0.05, 0.045, 0.035, 1))
    rough = nb.math('ADD', 0.03, nb.math('MULTIPLY', fox, 0.25))
    mirror = nb.add('ShaderNodeBsdfPrincipled', {'Base Color': silver, 'Metallic': 1.0, 'Roughness': rough,
                                                 'Coat Weight': 1.0, 'Coat Roughness': 0.012, 'Coat IOR': 1.52})
    em = nb.add('ShaderNodeEmission', {'Color': look.outputs['Color'], 'Strength': 0.9})
    add = nb.add('ShaderNodeAddShader')
    nb.put(add.inputs[0], mirror.outputs[0])
    nb.put(add.inputs[1], em.outputs[0])
    nb.out(add.outputs[0])
    return m


def glass_unlit(look_img):
    """What FRAME shows: the finished glass map as pure emission (the KHR_materials_unlit equivalent)."""
    m, nb = new_material('Mirror Glass Unlit')
    uvn = nb.add('ShaderNodeUVMap', uv_map='UVMap')
    tex = nb.add('ShaderNodeTexImage', {'Vector': uvn.outputs['UV']}, image=look_img, interpolation='Cubic')
    em = nb.add('ShaderNodeEmission', {'Color': tex.outputs['Color'], 'Strength': 1.0})
    nb.out(em.outputs[0])
    return m


def studio_materials():
    floor, nb = new_material('Studio Floor')
    tc = nb.add('ShaderNodeTexCoord')
    mp = nb.add('ShaderNodeMapping', {'Vector': tc.outputs['Object'], 'Scale': (6.0, 0.4, 1.0)})
    wave = nb.add('ShaderNodeTexWave', {'Vector': mp.outputs['Vector'], 'Scale': 2.2, 'Distortion': 1.5}, bands_direction='X')
    g = nb.remap(wave.outputs['Fac'], 0.0, 1.0, 0.8, 1.1)
    col = nb.mix(1.0, lin('#3a2616'), nb.add('ShaderNodeCombineColor', {'Red': g, 'Green': g, 'Blue': g}).outputs[0], 'MULTIPLY')
    nb.out(nb.add('ShaderNodeBsdfPrincipled', {'Base Color': col, 'Roughness': 0.42}).outputs[0])
    back, nb = new_material('Studio Backdrop')
    nb.out(nb.add('ShaderNodeBsdfPrincipled', {'Base Color': lin('#2a0508'), 'Roughness': 0.9,
                                               'Sheen Weight': 0.6, 'Sheen Tint': lin('#7a1c22')}).outputs[0])
    return floor, back


# ---------------------------------------------------------------- scene

def clear_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    for blocks in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.images):
        for b in list(blocks):
            blocks.remove(b)


def setup_cycles(sc, samples):
    sc.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    kind = 'NONE'
    for k in ('OPTIX', 'CUDA'):
        try:
            prefs.compute_device_type = k
            prefs.get_devices()
            if any(d.type == k for d in prefs.devices):
                for d in prefs.devices:
                    d.use = d.type == k
                kind = k
                break
        except Exception:
            pass
    sc.cycles.device = 'GPU' if kind != 'NONE' else 'CPU'
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.015
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = 'OPTIX' if kind == 'OPTIX' else 'OPENIMAGEDENOISE'
    except Exception:
        pass
    sc.cycles.max_bounces = 8
    sc.cycles.diffuse_bounces = 3
    sc.cycles.glossy_bounces = 5
    sc.cycles.blur_glossy = 0.5
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGB'
    return kind


def aim(ob, target):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


def quad(name, corners):
    return make_mesh(name, np.array(corners, float), np.array([[0, 1, 2, 3]]))


def studio(coll, floor_mat, back_mat):
    """Dark studio: plank-toned floor, deep red backdrop, the warm key lantern and the stage-right studio light,
    plus dim emissive cards (curtains, house) that only show in reflections."""
    add_object('Studio Floor', quad('Studio Floor', [(-5, -6, 0), (5, -6, 0), (5, 4, 0), (-5, 4, 0)]), coll, [floor_mat])
    add_object('Studio Backdrop', quad('Studio Backdrop', [(-5, 3.4, 0), (5, 3.4, 0), (5, 3.4, 5), (-5, 3.4, 5)]), coll,
               [back_mat])
    key = bpy.data.lights.new('Key Spot', 'SPOT')
    key.energy = 1100.0
    key.color = (1.0, 0.80, 0.58)
    key.spot_size = math.radians(40)
    key.spot_blend = 0.55
    key.shadow_soft_size = 0.05
    ko = bpy.data.objects.new('Key Spot', key)
    coll.objects.link(ko)
    ko.location = KEY_POS
    aim(ko, TARGET)
    side = bpy.data.lights.new('Studio Light', 'AREA')
    side.shape = 'DISK'
    side.size = 0.45
    side.energy = 90.0
    side.color = (1.0, 0.86, 0.68)
    so = bpy.data.objects.new('Studio Light', side)
    coll.objects.link(so)
    so.location = SIDE_POS
    aim(so, TARGET)
    cards = (('Curtain Card L', (-2.6, -0.4, 2.0), (90, 0, -90), (4.5, 4.0), (0.030, 0.0035, 0.004)),
             ('Curtain Card R', (2.6, -0.4, 2.0), (90, 0, 90), (4.5, 4.0), (0.030, 0.0035, 0.004)),
             ('House Card', (0.0, -6.0, 2.2), (90, 0, 0), (9.0, 4.4), (0.006, 0.0045, 0.0035)))
    for name, loc, rot, (w, hgt), col in cards:
        cm, nb = new_material('Studio ' + name)
        nb.out(nb.add('ShaderNodeEmission', {'Color': col + (1.0,), 'Strength': 1.0}).outputs[0])
        me = quad(name, [(-w / 2, -hgt / 2, 0), (w / 2, -hgt / 2, 0), (w / 2, hgt / 2, 0), (-w / 2, hgt / 2, 0)])
        ob = add_object(name, me, coll, [cm])
        ob.location = loc
        ob.rotation_euler = [math.radians(a) for a in rot]
        ob.visible_camera = False
        ob.visible_shadow = False
    world = bpy.data.worlds.new('Studio World')
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    bg.inputs['Color'].default_value = (0.0040, 0.0032, 0.0026, 1.0)
    bg.inputs['Strength'].default_value = 1.0
    bpy.context.scene.world = world


VIEWS = {
    'front': ((0.0, -3.3, 1.55), (0.0, 0.08, 1.02), 50, (720, 960)),
    'three': ((-1.95, -2.55, 1.58), (0.0, 0.10, 1.02), 50, (720, 960)),
    'crest': ((0.30, -0.98, 1.76), (0.0, 0.14, 1.90), 50, (960, 640)),
    'corner': ((-0.62, -0.95, 1.48), (-0.40, 0.14, 1.76), 50, (960, 720)),
    'foot': ((0.75, -0.85, 0.42), (0.40, 0.0, 0.10), 45, (960, 720)),
    'side': ((2.9, 0.15, 1.15), (0.0, 0.18, 1.0), 50, (720, 960)),
    'back': ((1.35, 2.75, 1.45), (0.0, 0.25, 0.95), 50, (720, 960), 3.0),
    'glass': ((0.15, -1.35, 1.25), (0.0, 0.06, 1.05), 50, (720, 960)),
}


def cameras(coll):
    out = {}
    for name, (loc, tgt, lens, *_) in VIEWS.items():
        cam = bpy.data.cameras.new('Cam ' + name)
        cam.lens = lens
        cam.sensor_fit = 'AUTO'
        cam.clip_start = 0.02
        ob = bpy.data.objects.new('Cam ' + name, cam)
        coll.objects.link(ob)
        ob.location = loc
        aim(ob, tgt)
        out[name] = ob
    return out


# ---------------------------------------------------------------- build

def apply_matrix(me, M):
    me.transform(M)
    me.update()


def decimate(ob, tris):
    me = ob.data
    cur = sum(len(p.vertices) - 2 for p in me.polygons)
    if cur > tris:
        mod = ob.modifiers.new('decimate', 'DECIMATE')
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = tris / cur
        mod.use_collapse_triangulate = True
        dg = bpy.context.evaluated_depsgraph_get()
        new = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
        ob.modifiers.remove(mod)
        old = ob.data
        ob.data = new
        new.name = old.name
        bpy.data.meshes.remove(old)
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
    bm.to_mesh(ob.data)
    bm.free()
    return sum(len(p.vertices) - 2 for p in ob.data.polygons)


def triangulate(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3], quad_method='BEAUTY', ngon_method='BEAUTY')
    bm.to_mesh(me)
    bm.free()


def join(objs, name):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.name = name
    ob.data.name = name
    return ob


def _uv_areas(me):
    uv = np.zeros(len(me.loops) * 2)
    me.uv_layers.active.data.foreach_get('uv', uv)
    uv = uv.reshape(-1, 2)
    starts = np.zeros(len(me.polygons), np.int64)
    me.polygons.foreach_get('loop_start', starts)
    a, b, c = uv[starts], uv[starts + 1], uv[starts + 2]
    return uv, starts, 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def unwrap_atlas(ob, size=2048, margin_px=3):
    """One atlas for the low-poly frame (planned 2048 px): the moulding keeps its unrolled strips, ornament fronts
    are projected straight onto the mirror plane (one island per ornament), walls and bare wood are smart-projected
    at 60 % and 30 % texel density; everything is packed together with a small margin."""
    me = ob.data
    nf = len(me.polygons)
    nrm = np.zeros(nf * 3)
    me.polygons.foreach_get('normal', nrm)
    nrm = nrm.reshape(-1, 3)
    mat = np.zeros(nf, np.int32)
    me.polygons.foreach_get('material_index', mat)
    part = np.zeros(nf, np.int32)
    me.attributes['uvpart'].data.foreach_get('value', part)
    front = np.array([0.0, -math.cos(LEAN), math.sin(LEAN)])
    up = np.array([0.0, math.sin(LEAN), math.cos(LEAN)])
    planar = (part == 1) & (mat == 0) & (nrm @ front > 0.25)
    walls = (part == 1) & (mat == 0) & ~planar
    hidden = (part == 1) & (mat == 1) & (nrm @ front > 0.5)
    bare = (part == 1) & (mat == 1) & ~hidden
    area3 = np.zeros(nf)
    me.polygons.foreach_get('area', area3)
    starts = np.zeros(nf, np.int64)
    me.polygons.foreach_get('loop_start', starts)
    bpy.ops.object.select_all(action='DESELECT')
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    for sel in (walls, bare, hidden):
        if not sel.any():
            continue
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='FACE')
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()
        for f in bm.faces:
            f.select_set(bool(sel[f.index]))
        bm.select_flush_mode()
        bmesh.update_edit_mesh(me)
        bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.0, area_weight=0.0,
                                 scale_to_bounds=False)
        bpy.ops.object.mode_set(mode='OBJECT')
    uv, starts, area_uv = _uv_areas(me)
    for g, (sel, dens) in enumerate(((walls, 0.6), (bare, 0.22), (hidden, 0.03))):
        if sel.any():
            k = math.sqrt(area3[sel].sum() / max(area_uv[sel].sum(), 1e-12)) * dens
            loops = (starts[sel][:, None] + np.arange(3)[None, :]).ravel()
            uv[loops] = uv[loops] * k + [30.0 + 3.0 * g, 0.0]
    co = np.zeros(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    lv = np.zeros(len(me.loops), np.int64)
    me.loops.foreach_get('vertex_index', lv)
    loops = (starts[planar][:, None] + np.arange(3)[None, :]).ravel()
    pc = co[lv[loops]]
    uv[loops] = np.stack([pc[:, 0] + 40.0, pc @ up], 1)
    me.uv_layers.active.data.foreach_set('uv', uv.ravel())
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.uv.pack_islands(rotate=True, scale=True, margin_method='FRACTION', margin=margin_px / size,
                            shape_method='CONCAVE')
    bpy.ops.object.mode_set(mode='OBJECT')
    uv, starts, area_uv = _uv_areas(me)
    inside = bool((uv >= -1e-6).all() and (uv <= 1 + 1e-6).all())
    gold_front = (mat == 0) & ((part == 0) | planar)
    px_per_m = math.sqrt(area_uv[gold_front].sum() / area3[gold_front].sum()) * size
    return float(area_uv.sum()), px_per_m, inside


def build(opt):
    work = os.path.abspath(opt.work)
    os.makedirs(work, exist_ok=True)
    scale = 2.0 if opt.quick else 1.0
    c_lp = collection('Mirror')
    c_hp = collection('Mirror HP')
    c_st = collection('Mirror Studio')
    gold = gold_pbr()
    gold_b = gold_bake()
    gold_d = gold_debug()
    gold_d.use_fake_user = True
    wood_m = wood()
    wood_b = wood(True)
    for m in (gold_b, wood_b):
        m.use_fake_user = True

    masks, look = glass_maps()
    mask_img = save_image('mirror_glass_masks', masks, os.path.join(work, 'mirror_glass_masks.png'), 'Non-Color')
    look_img = save_image('mirror_glass_unlit', look, os.path.join(work, 'mirror_glass_unlit.png'), 'sRGB')
    glass_m = glass_pbr(look_img, mask_img)
    glass_u = glass_unlit(look_img)
    glass_u.use_fake_user = True
    log('materials + glass maps')

    M = Matrix.Rotation(-LEAN, 4, 'X')
    lp_parts, hp_parts = [], []
    P, Fc, Mi, UV = sweep(R.profile(False), with_uv=True)
    me = make_mesh('moulding_lp', P, Fc, Mi, sharp=32, uv=UV)
    lp_parts.append(add_object('moulding_lp', me, c_lp, [gold, wood_m]))
    P, Fc, Mi = sweep(R.profile(True))
    me = make_mesh('moulding_hp', P, Fc, Mi, sharp=32)
    hp_parts.append(add_object('moulding_hp', me, c_hp, [gold, wood_m]))
    P, Fc = bead_strip(0.00025 * scale)
    hp_parts.append(add_object('beads_hp', make_mesh('beads_hp', P, Fc, sharp=60), c_hp, [gold, wood_m]))
    P, Fc = leaf_band(0.0005 * scale)
    hp_parts.append(add_object('leaves_hp', make_mesh('leaves_hp', P, Fc, sharp=60), c_hp, [gold, wood_m]))
    log('moulding')

    for name, F, h, place, mir in ornament_parts(scale):
        P, Fc, Mi = relief_mesh(F, h, place, mir)
        me = make_mesh(name + '_hp', P, Fc, Mi, sharp=55)
        hp_parts.append(add_object(name + '_hp', me, c_hp, [gold, wood_m]))
    log('hp ornaments')
    for name, F, h, place, mir in ornament_parts(2.0):
        P, Fc, Mi = relief_mesh(F, h, place, mir, smooth_iters=2)
        me = make_mesh(name + '_lp', P, Fc, Mi, sharp=55)
        ob = add_object(name + '_lp', me, c_lp, [gold, wood_m])
        decimate(ob, LP_TRIS[name.split('_')[0]])
        ob.data.set_sharp_from_angle(angle=math.radians(55))
        lp_parts.append(ob)
    log('lp ornaments')

    for ob in lp_parts + hp_parts:
        apply_matrix(ob.data, M)
    co = np.concatenate([coords(ob.data) for ob in lp_parts])
    zmin = float(co[:, 2].min())
    y_contact = float(np.median(co[co[:, 2] < zmin + 0.0015, 1]))
    T = Matrix.Translation((0.0, -y_contact, -zmin))
    for ob in lp_parts + hp_parts:
        apply_matrix(ob.data, T)
    MT = T @ M
    P, Fc = easel(MT)
    me = make_mesh('easel', P, Fc, np.ones(len(Fc), np.int32), smooth=False)
    easel_lp = add_object('easel_lp', me, c_lp, [gold, wood_m])
    easel_hp = add_object('easel_hp', me.copy(), c_hp, [gold, wood_m])
    lp_parts.append(easel_lp)
    hp_parts.append(easel_hp)
    P, Fc, uv = glass_mesh()
    P = (np.asarray(MT)[:3, :3] @ P.T).T + np.asarray(MT)[:3, 3]
    me = make_mesh('Mirror_Glass', P, Fc, smooth=False, uv=uv)
    glass_ob = add_object('Mirror_Glass', me, c_lp, [glass_m, glass_u])
    log('lean + easel + glass; contact y %.4f zmin %.4f' % (y_contact, zmin))

    for ob in lp_parts:
        me = ob.data
        if not me.uv_layers:
            me.uv_layers.new(name='UVMap')
        part = me.attributes.new('uvpart', 'INT', 'FACE')
        part.data.foreach_set('value', np.full(len(me.polygons), 0 if ob.name == 'moulding_lp' else 1, np.int32))
        triangulate(me)
    frame = join(lp_parts, 'Mirror_Frame')
    frame_hp = join(hp_parts, 'Mirror_Frame_HP')
    used, dens, inside = unwrap_atlas(frame)
    log('atlas: uv area used %.3f, gold texel density %.0f px/m at 2048, inside 0..1: %s' % (used, dens, inside))

    root = bpy.data.objects.new('Mirror_Root', None)
    root.empty_display_type = 'PLAIN_AXES'
    root.empty_display_size = 0.3
    c_lp.objects.link(root)
    for ob in (frame, glass_ob):
        ob.parent = root
    frame_hp.parent = root
    floor_m, back_m = studio_materials()
    studio(c_st, floor_m, back_m)
    cams = cameras(c_st)
    return dict(frame=frame, frame_hp=frame_hp, glass=glass_ob, cams=cams, gold=gold, gold_b=gold_b, gold_d=gold_d, wood=wood_m,
                wood_b=wood_b, glass_m=glass_m, glass_u=glass_u, work=work, root=root)


def coords(me):
    co = np.zeros(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    return co.reshape(-1, 3)


def stats(objs):
    out = {}
    for ob in objs:
        co = coords(ob.data)
        out[ob.name] = (sum(len(p.vertices) - 2 for p in ob.data.polygons), co.min(0), co.max(0))
    return out


README = '''Gilded stage mirror (zhang-2024, Avatars & Identity) - built by tools/scene/mirror.py.
Axes: Blender Z up, mirror faces -Y (glTF +Z). Origin = floor-contact centre; leans back 5 deg on a dark wood strut.
Place it by moving Mirror_Root (stage: glTF (0, 0, -0.65) = Blender (0, 0.65, 0)); do not apply the children's
transforms - Mirror Gold Bake reads key-light and visitor positions in the mirror's object space.

"Mirror" = export set. Mirror_Frame: all triangles, materials [Mirror Gold, Mirror Wood], one UV map "UVMap"
packed for a single 2048x2048 atlas (gold and wood share it; 3 px margins; ~1300 px/m on the gilding).
Mirror_Glass: 20 tris, UV 0..1 over the whole panel for a 1024x2048 map. mirror_glass_unlit.png (packed here and
saved next to the .blend) is already the finished unlit glass texture; material "Mirror Glass Unlit" shows it.

"Mirror HP" (excluded from the view layer) = carved high-poly bake source and look-dev geometry, same placement.
To bake the atlas: enable Mirror HP, give its slots Mirror Gold Bake / Mirror Wood Bake, select HP then Mirror_Frame
(active), Cycles bake Combined (or Emit + Diffuse) selected-to-active, cage extrusion ~4 mm, max ray ~8 mm,
into a 2048 image on Mirror_Frame's UVMap. The 'Bake Gain' value node in Mirror Gold Bake scales the result.
Mirror Gold (PBR) is for Cycles look-dev only.'''


def set_look(S, look):
    sc = bpy.context.scene
    if look == 'masks':
        S['frame_hp'].data.materials[0] = S['gold_d']
        sc.view_settings.view_transform = 'Standard'
        sc.view_settings.look = 'None'
    elif look == 'bake':
        S['frame_hp'].data.materials[0] = S['gold_b']
        S['frame_hp'].data.materials[1] = S['wood_b']
        S['glass'].active_material_index = 0
        S['glass'].data.materials[0] = S['glass_u']
        sc.view_settings.view_transform = 'Standard'
        sc.view_settings.look = 'None'
    else:
        S['frame_hp'].data.materials[0] = S['gold']
        S['frame_hp'].data.materials[1] = S['wood']
        S['glass'].data.materials[0] = S['glass_m']
        sc.view_settings.view_transform = 'AgX'
        try:
            sc.view_settings.look = 'AgX - Medium High Contrast'
        except Exception:
            sc.view_settings.look = 'None'
    sc.view_settings.exposure = 0.0


def render(S, opt):
    sc = bpy.context.scene
    out = os.path.join(S['work'], 'mirror-previews')
    os.makedirs(out, exist_ok=True)
    files = []
    for look in [l for l in opt.looks.split(',') if l]:
        S['frame'].hide_render = look != 'lp'
        S['frame_hp'].hide_render = look == 'lp'
        set_look(S, 'pbr' if look == 'lp' else look)
        for v in [v for v in opt.views.split(',') if v]:
            loc, tgt, lens, size, *ev = VIEWS[v]
            k = opt.res / max(size)
            sc.render.resolution_x, sc.render.resolution_y = int(size[0] * k), int(size[1] * k)
            sc.render.resolution_percentage = 100
            sc.camera = S['cams'][v]
            sc.view_settings.exposure = ev[0] if ev else 0.0
            path = os.path.join(out, '%s-%s%s.png' % (look, v, opt.tag))
            sc.render.filepath = path
            t = time.time()
            bpy.ops.render.render(write_still=True)
            log('render %s %s %.1fs' % (look, v, time.time() - t))
            files.append(path)
    set_look(S, 'pbr')
    S['frame'].hide_render = False
    S['frame_hp'].hide_render = True
    return files


def main():
    global T0
    T0 = time.time()
    opt = options()
    clear_scene()
    sc = bpy.context.scene
    kind = setup_cycles(sc, opt.samples)
    log('device', kind)
    S = build(opt)
    st = stats([S['frame'], S['glass'], S['frame_hp']])
    for k, (tris, lo, hi) in st.items():
        log('%-16s tris %8d  min %s  max %s  size %s' % (k, tris, np.round(lo, 4), np.round(hi, 4), np.round(hi - lo, 4)))
    files = render(S, opt) if opt.views else []
    for lc in bpy.context.view_layer.layer_collection.children:
        if lc.name == 'Mirror HP':
            lc.exclude = True
    txt = bpy.data.texts.get('Mirror README') or bpy.data.texts.new('Mirror README')
    txt.from_string(README)
    sc.camera = S['cams']['front']
    if not opt.no_blend:
        path = os.path.join(S['work'], 'mirror.blend')
        bpy.ops.wm.save_as_mainfile(filepath=path, compress=True, check_existing=False)
        log('saved', path)
    log('done', files)


if __name__ == '__main__':
    main()
