"""Builds objects/mirror.glb, the walnut cheval mirror for zhang-2024 (Avatars & Identity), and mirror-unlit.glb (unlit glass):
    python -I -X utf8 tools/objects/make_mirror.py
Metres, Y up, glass facing +Z, origin at the floor centre; the walnut is the lectern's (tools/books)."""
import math
import os
import sys

sys.dont_write_bytecode = True

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'books'))
from make_books import Glb, Part, jpeg, png, rot_x, unit, wood_texture

OBJECTS = os.path.join(ROOT, 'objects')
MM = 0.001

SIGHT_W, SIGHT_H = 0.55, 1.34
ARCH_SEGMENTS = 64
STEP = 0.1
FRAME_W = 48 * MM
FRAME_BOTTOM = 0.33
PIVOT_Y = 1.015
PIVOT_D = -5.5 * MM
TILT = 4.0
BEVEL_IN, BEVEL_OUT = 18 * MM, 4 * MM
BACK_N, BACK_D = 10 * MM, -10 * MM
GAP = 12 * MM
POST_HALF, POST_TOP_HALF, POST_TOP = 19 * MM, 16 * MM, 1.125
FOOT_LEN, FOOT_HALF_W, FOOT_H_MID, FOOT_H_END = 0.46, 26 * MM, 54 * MM, 28 * MM
FOOT_PAD, FOOT_RELIEF, FOOT_EASE = 0.15, 8 * MM, 4 * MM
FOOT_TOP = FOOT_H_MID - 3 * MM
STRETCHER_Y, STRETCHER_HALF = 0.12, (14 * MM, 10 * MM)
WOOD_SCALE, WOOD_ROUGHNESS = 0.6, 0.6
FRAME_UV = {'rail': (0.63, 0.03), 'side': (0.21, 0.99)}
FOOT_UV = {-1: (0.33, 0.04, 0.06), 1: (0.71, 0.55, 0.57)}
CLOTH_RGB, CLOTH_ROUGHNESS, CLOTH_TILE = (0x38, 0x2F, 0x5E), 0.8, 0.12
GLASS_KD, GLASS_KE, GLASS_ROUGHNESS = 0.75, 0.3, 0.3
DECAL_LIFT = 1 * MM
DECAL_HALF = (0.07, 0.29)


def post_half(y):
    return POST_HALF - (POST_HALF - POST_TOP_HALF) * (y - FOOT_TOP) / (POST_TOP - 2 * MM - FOOT_TOP)


POST_X = SIGHT_W / 2 + FRAME_W + GAP + post_half(PIVOT_Y)


class MirrorGlb(Glb):
    def __init__(self, unlit):
        super().__init__()
        self.unlit = unlit
        self.doc['asset']['generator'] = 'xr-museum-papers tools/objects/make_mirror.py'

    def glass(self, name, texture):
        if self.unlit:
            mat = {'name': name, 'extensions': {'KHR_materials_unlit': {}},
                   'pbrMetallicRoughness': {'baseColorTexture': {'index': texture}, 'metallicFactor': 0.0, 'roughnessFactor': 1.0}}
            self._extensions.add('KHR_materials_unlit')
        else:
            mat = {'name': name, 'emissiveTexture': {'index': texture}, 'emissiveFactor': [GLASS_KE] * 3,
                   'pbrMetallicRoughness': {'baseColorTexture': {'index': texture}, 'baseColorFactor': [GLASS_KD] * 3 + [1.0],
                                            'metallicFactor': 0.0, 'roughnessFactor': GLASS_ROUGHNESS}}
        self.doc['materials'].append(mat)
        return len(self.doc['materials']) - 1


class Welded:
    def __init__(self, part):
        P, N, UV, I = part.arrays()
        rows = np.ascontiguousarray(np.hstack([P, N, UV]) + np.float32(0.0))
        key = rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()
        _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
        order = np.argsort(first)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        keep = first[order]
        self.data = (P[keep], N[keep], UV[keep], rank[inverse.ravel()][I])
        self.P = part.P

    def arrays(self):
        return self.data


def signed_area(P):
    return 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))


def subdivide(path, members):
    P = np.asarray(path, float)
    pts, mem = [], []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        k = max(1, math.ceil(float(np.linalg.norm(b - a)) / STEP - 1e-9))
        pts += [a] + [a + (b - a) * s / k for s in range(1, k)]
        mem += [members[i]] * k
    return np.array(pts), mem


def path_frame(path):
    P = np.asarray(path, float)
    D = np.roll(P, -1, 0) - P
    T = D / np.linalg.norm(D, axis=1, keepdims=True)
    out = math.copysign(1.0, signed_area(P)) * np.stack([T[:, 1], -T[:, 0]], 1)
    prev = np.roll(out, 1, 0)
    bis = prev + out
    bis /= np.linalg.norm(bis, axis=1, keepdims=True)
    k = 1.0 / np.einsum('ij,ij->i', bis, out)
    turn = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', prev, out), -1, 1)))
    L = np.concatenate([[0.0], np.cumsum(np.linalg.norm(D, axis=1))])
    return P, out, bis, k, turn, L


def offset_path(path, n):
    P, _, bis, k, _, _ = path_frame(path)
    return P + n * k[:, None] * bis


def profile_normals(Q, crease_deg=35.0):
    dQ = np.diff(Q, axis=0)
    tq = dQ / np.linalg.norm(dQ, axis=1, keepdims=True)
    pn = np.stack([-tq[:, 1], tq[:, 0]], 1)
    lo, hi = pn.copy(), pn.copy()
    c = math.cos(math.radians(crease_deg))
    for j in range(len(pn)):
        if j > 0 and pn[j - 1] @ pn[j] > c:
            lo[j] = unit(pn[j - 1] + pn[j])
        if j < len(pn) - 1 and pn[j + 1] @ pn[j] > c:
            hi[j] = unit(pn[j + 1] + pn[j])
    S = np.concatenate([[0.0], np.cumsum(np.linalg.norm(dQ, axis=1))])
    return lo, hi, S


def sweep(parts, path, profile, tags, basis, uv, sharp_deg=30.0):
    o, e1, e2, e3 = (np.asarray(v, float) for v in basis)
    P, out, bis, k, turn, L = path_frame(path)
    m = len(P)
    Q = np.asarray(profile, float)
    lo, hi, S = profile_normals(Q)

    def corner(vi, seg, nv, Lv, qj, pq):
        n, d = Q[qj]
        p2 = P[vi] + n * k[vi] * bis[vi]
        pos = o + p2[0] * e1 + p2[1] * e2 + d * e3
        normal = pq[0] * (nv[0] * e1 + nv[1] * e2) + pq[1] * e3
        return pos, normal, uv(vi, seg, Lv, S[qj], n, d, nv, P[vi])

    for i in range(m):
        i1 = (i + 1) % m
        n0 = bis[i] if turn[i] <= sharp_deg else out[i]
        n1 = bis[i1] if turn[i1] <= sharp_deg else out[i]
        for j, tag in enumerate(tags):
            if tag is None:
                continue
            cs = [corner(i, i, n0, L[i], j, lo[j]), corner(i1, i, n1, L[i + 1], j, lo[j]),
                  corner(i1, i, n1, L[i + 1], j + 1, hi[j]), corner(i, i, n0, L[i], j + 1, hi[j])]
            nrm = [c[1] for c in cs]
            parts[tag].poly([c[0] for c in cs], [c[2] for c in cs], np.sum(nrm, axis=0), normals=nrm)


def triangulate(poly):
    P = np.asarray(poly, float)
    idx = list(range(len(P)))
    if signed_area(P) < 0:
        idx.reverse()

    def cross(a, b, c):
        return (P[b][0] - P[a][0]) * (P[c][1] - P[a][1]) - (P[b][1] - P[a][1]) * (P[c][0] - P[a][0])

    def shape(a, b, c):
        return cross(a, b, c) / max(float(np.sum((P[u] - P[v]) ** 2)) for u, v in ((a, b), (b, c), (c, a)))

    tris = []
    while len(idx) > 3:
        n = len(idx)
        best = None
        for e in range(n):
            a, b, c = idx[e - 1], idx[e], idx[(e + 1) % n]
            if cross(a, b, c) <= 1e-14:
                continue
            if any(cross(a, b, q) >= 0 and cross(b, c, q) >= 0 and cross(c, a, q) >= 0 for q in idx if q not in (a, b, c)):
                continue
            if best is None or shape(a, b, c) > best[0]:
                best = (shape(a, b, c), e, (a, b, c))
        if best is None:
            raise RuntimeError('polygon could not be triangulated')
        tris.append(best[2])
        idx.pop(best[1])
    tris.append(tuple(idx))
    return tris


def flat_cap(part, poly2d, basis, d, facing, uvf):
    o, e1, e2, e3 = (np.asarray(v, float) for v in basis)
    pts = [o + p[0] * e1 + p[1] * e2 + d * e3 for p in poly2d]
    for a, b, c in triangulate(poly2d):
        part.poly([pts[a], pts[b], pts[c]], [uvf(pts[a]), uvf(pts[b]), uvf(pts[c])], facing)


def centre_fan(part, pts, facing, uvf):
    c = np.mean(pts, axis=0)
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        part.poly([c, a, b], [uvf(c), uvf(a), uvf(b)], facing)


def rrect(ha, hb, r, segs=4):
    pts, nrm = [], []
    for ca, cb, a0 in ((ha - r, hb - r, 0), (r - ha, hb - r, 90), (r - ha, r - hb, 180), (ha - r, r - hb, 270)):
        for s in range(segs + 1):
            a = math.radians(a0 + 90 * s / segs)
            nrm.append((math.cos(a), math.sin(a)))
            pts.append((ca + r * math.cos(a), cb + r * math.sin(a)))
    return np.array(pts), np.array(nrm)


def loft(part, origin, axis, ea, eb, levels, cap_top=False, uv0=(0.0, 0.0), segs=4, start=0):
    o, ax, ea, eb = (np.asarray(v, float) for v in (origin, axis, ea, eb))
    fine = [levels[0]]
    for a, b in zip(levels, levels[1:]):
        k = max(1, math.ceil(abs(b[0] - a[0]) / STEP - 1e-9))
        fine += [tuple(x + (y - x) * s / k for x, y in zip(a, b)) for s in range(1, k)] + [b]
    rings = []
    for t, ha, hb, r in fine:
        p2, n2 = (np.roll(a, -start, 0) for a in rrect(ha, hb, r, segs))
        rings.append((t, o + t * ax + p2[:, :1] * ea + p2[:, 1:] * eb, n2[:, :1] * ea + n2[:, 1:] * eb))
    m = len(rings[0][1])
    per = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(np.vstack([rings[0][1], rings[0][1][:1]]), axis=0), axis=1))])
    for (t0, R0, N0), (t1, R1, _) in zip(rings, rings[1:]):
        for j in range(m):
            j1 = (j + 1) % m
            nrm = []
            for jj in (j, j1):
                tang = np.cross(ax, N0[jj])
                n = unit(np.cross(tang, R1[jj] - R0[jj]))
                nrm.append(n if n @ N0[jj] > 0 else -n)
            uvs = [(t0 / WOOD_SCALE + uv0[0], per[j] / WOOD_SCALE + uv0[1]), (t0 / WOOD_SCALE + uv0[0], per[j + 1] / WOOD_SCALE + uv0[1]),
                   (t1 / WOOD_SCALE + uv0[0], per[j + 1] / WOOD_SCALE + uv0[1]), (t1 / WOOD_SCALE + uv0[0], per[j] / WOOD_SCALE + uv0[1])]
            part.poly([R0[j], R0[j1], R1[j1], R1[j]], uvs, nrm[0] + nrm[1], normals=[nrm[0], nrm[1], nrm[1], nrm[0]])
    if cap_top:
        t, R, _ = rings[-1]
        centre_fan(part, R, ax, lambda p: ((p - o) @ ea / WOOD_SCALE + uv0[0], (p - o) @ eb / WOOD_SCALE + uv0[1]))


def lathe(part, profile, origin, axis, ref, grain, segs=24, uv0=(0.0, 0.0)):
    o, ax = np.asarray(origin, float), unit(axis)
    ref = unit(np.asarray(ref, float) - (np.asarray(ref, float) @ ax) * ax)
    side = np.cross(ax, ref)
    gu, gv = (np.asarray(g, float) for g in grain)
    Q = np.asarray(profile, float)
    lo, hi, _ = profile_normals(Q)
    lo, hi = -lo, -hi
    for j in range(len(Q) - 1):
        (r0, t0), (r1, t1) = Q[j], Q[j + 1]
        for s in range(segs):
            pts, nrm, uvs = [], [], []
            for r, t, pq, a in ((r0, t0, lo[j], s), (r0, t0, lo[j], s + 1), (r1, t1, hi[j], s + 1), (r1, t1, hi[j], s)):
                ang = 2 * math.pi * a / segs
                radial = math.cos(ang) * ref + math.sin(ang) * side
                if r < 1e-7 and any(np.allclose(p, o + t * ax) for p in pts):
                    continue
                pts.append(o + t * ax + r * radial)
                nrm.append(pq[0] * radial + pq[1] * ax)
                q = pts[-1] - o
                uvs.append((q @ gu / WOOD_SCALE + uv0[0], q @ gv / WOOD_SCALE + uv0[1]))
            part.poly(pts, uvs, np.sum(nrm, axis=0), normals=nrm)


def sight_path():
    r = SIGHT_W / 2
    hs = SIGHT_H - r
    pts = [(-r, 0.0), (r, 0.0), (r, hs)]
    pts += [(r * math.cos(math.pi * k / ARCH_SEGMENTS), hs + r * math.sin(math.pi * k / ARCH_SEGMENTS)) for k in range(1, ARCH_SEGMENTS)]
    pts.append((-r, hs))
    return subdivide(pts, ['rail'] + ['side'] * (len(pts) - 1))


def frame_profile():
    pts, tags = [(0.0, -1.0)], []

    def add(p, tag='wood'):
        pts.append(p)
        tags.append(tag)

    def arc(cx, cy, r, a0, a1, steps):
        for s in range(1, steps + 1):
            a = a0 + (a1 - a0) * s / steps
            add((cx + r * math.cos(a), cy + r * math.sin(a)))

    add((0.0, 5.0))
    arc(3.5, 5.0, 3.5, math.pi, math.pi / 2, 4)
    add((5.0, 8.5))
    arc(5.0, 13.0, 4.5, -math.pi / 2, 0.0, 5)
    for n in (13.5, 17.5, 21.5, 24.0, 28.0, 32.0, 36.0):
        add((n, 13.0 + 1.5 * math.sin(math.pi / 2 * (n - 9.5) / 26.5)))
    arc(36.0, 2.5, 12.0, math.pi / 2, 0.0, 8)
    add((48.0, -13.5))
    arc(45.5, -13.5, 2.5, 0.0, -math.pi / 2, 2)
    add((10.0, -16.0))
    add((10.0, -10.0))
    return [(n * MM, d * MM) for n, d in pts], tags


def foot_outline():
    half = FOOT_LEN / 2
    rn = FOOT_H_END / 2
    zj = half - rn
    pts = [(zj + rn * math.cos(-math.pi / 2 + math.pi * s / 8), rn + rn * math.sin(-math.pi / 2 + math.pi * s / 8)) for s in range(8)]
    for s in range(20):
        z = zj - 2 * zj * s / 20
        pts.append((z, FOOT_H_END + (FOOT_H_MID - FOOT_H_END) * 0.5 * (1 - math.cos(math.pi * (1 - abs(z) / zj)))))
    pts += [(-zj + rn * math.cos(math.pi / 2 + math.pi * s / 8), rn + rn * math.sin(math.pi / 2 + math.pi * s / 8)) for s in range(8)]
    pts += [(-zj, 0.0), (-FOOT_PAD, 0.0)]
    pts += [(z, FOOT_RELIEF * (1 - (z / FOOT_PAD) ** 2)) for z in np.linspace(-FOOT_PAD, FOOT_PAD, 14)[1:-1]]
    pts.append((FOOT_PAD, 0.0))
    return pts


def eased_band(half, r, steps=3):
    pts = [(-r, half)]
    pts += [(-r + r * math.sin(math.pi / 2 * s / steps), half - r + r * math.cos(math.pi / 2 * s / steps)) for s in range(1, steps + 1)]
    pts += [(-r + r * math.cos(math.pi / 2 * s / steps), -half + r - r * math.sin(math.pi / 2 * s / steps)) for s in range(steps + 1)]
    return pts


def build_frame(wood, cloth, field, bevel):
    path, members = sight_path()
    basis = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    profile, tags = frame_profile()

    def wood_uv(vi, seg, Lv, S, n, d, nv, p):
        du, dv = FRAME_UV[members[seg]]
        return (Lv / WOOD_SCALE + du, S / WOOD_SCALE + dv)

    sweep({'wood': wood}, path, profile, tags, basis, wood_uv)

    def bevel_uv(vi, seg, Lv, S, n, d, nv, p):
        return ((n + BEVEL_IN) / (BEVEL_IN + BEVEL_OUT), bevel_key(nv, p))

    sweep({'glass': bevel}, path, [(-BEVEL_IN, 0.0), (BEVEL_OUT, 0.0)], ['glass'], basis, bevel_uv)

    inner = offset_path(path, -BEVEL_IN)
    x0, x1 = inner[:, 0].min(), inner[:, 0].max()
    y0, y1 = inner[:, 1].min(), inner[:, 1].max()
    centre_fan(field, np.c_[inner, np.zeros(len(inner))], (0, 0, 1), lambda p: ((p[0] - x0) / (x1 - x0), (y1 - p[1]) / (y1 - y0)))

    back = offset_path(path, BACK_N)
    centre_fan(cloth, np.c_[back, np.full(len(back), BACK_D)], (0, 0, -1), lambda p: (p[0] / CLOTH_TILE, p[1] / CLOTH_TILE))
    return (x0, x1, y0, y1)


def build_stand(wood):
    X, Y, Z = np.eye(3)
    profile = eased_band(FOOT_HALF_W, FOOT_EASE)
    outline = foot_outline()
    inset = offset_path(outline, -FOOT_EASE)
    for side in (-1, 1):
        du, dv, dcap = FOOT_UV[side]
        x = side * POST_X
        basis = ((x, 0, 0), (0, 0, 1), (0, 1, 0), tuple(np.cross((0, 0, 1), (0, 1, 0))))
        sweep({'wood': wood}, outline, profile, ['wood'] * (len(profile) - 1), basis,
              lambda vi, seg, Lv, S, n, d, nv, p: (Lv / WOOD_SCALE + du, S / WOOD_SCALE + dv))
        e3 = np.asarray(basis[3], float)
        for d in (FOOT_HALF_W, -FOOT_HALF_W):
            flat_cap(wood, inset, basis, d, d * e3, lambda p: (p[2] / WOOD_SCALE + du, p[1] / WOOD_SCALE + dcap))
        loft(wood, (x, 0, 0), Y, X, Z,
             [(FOOT_TOP, POST_HALF, POST_HALF, 4 * MM), (POST_TOP - 2 * MM, POST_TOP_HALF, POST_TOP_HALF, 4 * MM),
              (POST_TOP, POST_TOP_HALF - 2 * MM, POST_TOP_HALF - 2 * MM, 2 * MM)], cap_top=True, uv0=(0.12 + 0.35 * side, 0.96),
             start=12 if side > 0 else 17)
        finial = [(13.5, -1.0), (13.5, 2.5), (12.5, 3.5), (9.5, 4.2), (7.5, 5.5), (6.5, 7.5), (6.0, 12.8)]
        a0 = -math.acos(6.0 / 11.0)
        for s in range(1, 15):
            a = a0 + (math.pi / 2 - a0) * s / 14
            finial.append((11.0 * math.cos(a), 22.0 + (15.0 if a > 0 else 11.0) * math.sin(a)))
        finial[-1] = (0.0, finial[-1][1])
        lathe(wood, [(r * MM, t * MM) for r, t in finial], (x, POST_TOP, 0), Y, X, (Y, X), uv0=(0.4, 0.08 if side < 0 else 0.6))
        post_out, post_in = x + side * post_half(PIVOT_Y), x - side * post_half(PIVOT_Y)
        knob = [(10.5, -2.0), (10.5, 2.0), (9.5, 3.0), (7.0, 4.5), (6.5, 7.5), (10.0, 10.5), (14.0, 14.0), (15.5, 17.5),
                (15.0, 20.5), (12.5, 23.5), (8.0, 25.5), (3.0, 26.3), (0.0, 26.5)]
        lathe(wood, [(r * MM, t * MM) for r, t in knob], (post_out, PIVOT_Y, 0), side * X, Y, (Y, Z), uv0=(0.3, 0.6))
        frame_side = side * (SIGHT_W / 2 + FRAME_W)
        length = abs(post_in - frame_side) + 4 * MM
        spacer = [(7.5, 0.0), (8.5, 1.0), (8.5, length / MM - 1.0), (7.5, length / MM)]
        lathe(wood, [(r * MM, t * MM) for r, t in spacer], (frame_side - side * 2 * MM, PIVOT_Y, 0), side * X, Y, (X, Y), segs=20,
              uv0=(0.15, 0.06))
    rail_x = POST_X - POST_HALF + 5 * MM
    loft(wood, (0, STRETCHER_Y, 0), X, Y, Z,
         [(-rail_x, STRETCHER_HALF[0], STRETCHER_HALF[1], 3 * MM), (rail_x, STRETCHER_HALF[0], STRETCHER_HALF[1], 3 * MM)],
         uv0=(0.58, 0.02), start=12)


def oklab_srgb(L, a, b):
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = np.stack([4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
                    -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
                    -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_], -1)
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def to_image(srgb, seed):
    rng = np.random.default_rng(seed)
    noise = (rng.random(srgb.shape) - rng.random(srgb.shape)) / 255
    return Image.fromarray(np.clip(np.round((srgb + noise) * 255), 0, 255).astype(np.uint8), 'RGB')


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def shoulder(L, knee=0.86, top=0.96):
    w = top - knee
    return np.where(L > knee, knee + w * np.tanh((L - knee) / w), L)


SILVER_TINT = (-0.005, -0.010)


def silver_l(yn):
    return 0.64 + 0.19 * smoothstep(0.02, 0.70, yn) - 0.02 * smoothstep(0.85, 1.0, yn)


def glass_field(bounds, size=(256, 512)):
    x0, x1, y0, y1 = bounds
    w, h = size
    X, Y = np.meshgrid(x0 + (np.arange(w) + 0.5) / w * (x1 - x0), y1 - (np.arange(h) + 0.5) / h * (y1 - y0))
    r = (x1 - x0) / 2
    ys = SIGHT_H - SIGHT_W / 2
    sdf = np.where(Y > ys, r - np.hypot(X, Y - ys), np.minimum(np.minimum(X - x0, x1 - X), Y - y0))
    yn = (Y - y0) / (y1 - y0)
    L = silver_l(yn) - 0.012 * X / r
    ang = math.radians(24)
    dist = (X + 0.10) * math.cos(ang) + (Y - 0.95) * math.sin(ang)
    L += 0.024 * np.exp(-(dist / 0.075) ** 2) * smoothstep(0.2, 0.6, yn)
    L -= 0.035 * np.exp(-np.maximum(sdf, 0) / 0.03)
    a = SILVER_TINT[0] + 0.002 * (1 - yn)
    b = SILVER_TINT[1] + 0.006 * (1 - yn) - 0.002 * yn
    return to_image(oklab_srgb(shoulder(L), a, b), 11)


def bevel_key(nv, p):
    yn = p[1] / SIGHT_H
    return float(np.clip(0.45 + 0.36 * nv[1] + 0.14 * (yn - 0.5) - 0.14 * nv[0], 0.03, 0.97))


def glass_bevel(size=(128, 32)):
    w, h = size
    N, K = np.meshgrid(-BEVEL_IN + (np.arange(w) + 0.5) / w * (BEVEL_IN + BEVEL_OUT), (np.arange(h) + 0.5) / h)
    L = 0.50 + 0.40 * K
    L += 0.02 - 0.07 * np.clip((N + 0.0165) / 0.012, 0, 1)
    L += (0.07 + 0.15 * K) * np.exp(-((N + 0.0175) / 0.0004) ** 2)
    t_out = np.clip((N + 0.010) / 0.010, 0, 1)
    a = SILVER_TINT[0] - 0.008 * t_out
    b = SILVER_TINT[1] + 0.002 * t_out
    f = smoothstep(-0.006, 0.0, N)
    L = L * (1 - 0.55 * f)
    a = a + 0.010 * f
    b = b + 0.016 * f
    return to_image(oklab_srgb(shoulder(L), a, b), 12)


def tile_noise(n, fx, fy, seed):
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.standard_normal((n, n)))
    kx, ky = np.fft.fftfreq(n)[None, :] * n, np.fft.fftfreq(n)[:, None] * n
    out = np.real(np.fft.ifft2(F * np.exp(-(kx / fx) ** 2 - (ky / fy) ** 2)))
    return (out - out.mean()) / out.std()


def book_cloth(n=256):
    lin = np.array([((c / 255) / 12.92) if c / 255 <= 0.04045 else ((c / 255 + 0.055) / 1.055) ** 2.4 for c in CLOTH_RGB])
    lms = np.cbrt(np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                            [0.0883024619, 0.2817188376, 0.6299787005]]) @ lin)
    L0, a0, b0 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                           [0.0259040371, 0.7827717662, -0.8086757660]]) @ lms
    L = L0 * (1 + 0.016 * tile_noise(n, 5, 5, 21) + 0.022 * tile_noise(n, 20, n / 2, 22)
              + 0.022 * tile_noise(n, n / 2, 20, 23) + 0.012 * tile_noise(n, n / 3, n / 3, 24))
    return to_image(oklab_srgb(L, np.full_like(L, a0), np.full_like(L, b0)), 25)


def foot_shadow(size=(64, 256)):
    w, h = size
    hx, hz = DECAL_HALF
    X, Z = np.meshgrid((np.arange(w) + 0.5) / w * 2 * hx - hx, (np.arange(h) + 0.5) / h * 2 * hz - hz)
    rr = 6 * MM
    qx, qz = np.abs(X) - (FOOT_HALF_W - rr), np.abs(Z) - (FOOT_LEN / 2 - FOOT_H_END / 2 - rr)
    d = np.maximum(np.hypot(np.maximum(qx, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, qz), 0) - rr, 0)
    contact = np.where(np.abs(Z) > FOOT_PAD, 1.0, 0.45 + 0.55 * (Z / FOOT_PAD) ** 2)
    alpha = 0.55 * contact * np.exp(-d / 0.007) + 0.2 * np.exp(-(d / 0.035) ** 2)
    alpha *= np.clip(np.minimum(hx - np.abs(X), hz - np.abs(Z)) / 0.02, 0, 1) ** 2
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 3] = np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, 'RGBA')


def decals(part):
    hx, hz = DECAL_HALF
    for x in (-POST_X, POST_X):
        part.poly([(x - hx, DECAL_LIFT, -hz), (x + hx, DECAL_LIFT, -hz), (x + hx, DECAL_LIFT, hz), (x - hx, DECAL_LIFT, hz)],
                  [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], (0, 1, 0))


def gpu_bytes(img):
    w, h = img.size
    return int(w * h * 4 * 4 / 3)


def main():
    wood_stand, wood_frame, cloth, field, bevel, shadow = Part(), Part(), Part(), Part(), Part(), Part()
    bounds = build_frame(wood_frame, cloth, field, bevel)
    build_stand(wood_stand)
    decals(shadow)
    R = rot_x(-TILT)
    pivot_local = np.array([0.0, PIVOT_Y - FRAME_BOTTOM - FRAME_W, PIVOT_D])
    t = np.array([0.0, PIVOT_Y, 0.0]) - R @ pivot_local
    wood = wood_stand.extend(wood_frame.transformed(R, t))
    cloth, field, bevel = (p.transformed(R, t) for p in (cloth, field, bevel))

    images = {'walnut': wood_texture(), 'cloth': book_cloth(), 'glass': glass_field(bounds), 'bevel': glass_bevel(), 'shadow': foot_shadow()}
    files = {'walnut': jpeg(images['walnut'], 88, 2), 'cloth': jpeg(images['cloth'], 90, 0), 'glass': png(images['glass']),
             'bevel': png(images['bevel']), 'shadow': png(images['shadow'])}
    meshes = [Welded(p) for p in (wood, cloth, field, bevel, shadow)]
    for name, unlit in (('mirror.glb', False), ('mirror-unlit.glb', True)):
        g = MirrorGlb(unlit)
        mats = [g.material('walnut', g.texture(files['walnut'], 'walnut', wrap='repeat'), WOOD_ROUGHNESS),
                g.material('book cloth', g.texture(files['cloth'], 'book-cloth', wrap='repeat'), CLOTH_ROUGHNESS),
                g.glass('silvered glass', g.texture(files['glass'], 'silvered-glass', mime='image/png')),
                g.glass('glass bevel', g.texture(files['bevel'], 'glass-bevel', mime='image/png')),
                g.shadow_material('contact shadow', g.texture(files['shadow'], 'contact-shadow', mime='image/png'))]
        g.node('mirror', mesh=g.mesh('Cheval mirror', list(zip(meshes, mats))), root=True)
        size = g.write(os.path.join(OBJECTS, name))
        verts = sum(len(m.data[0]) for m in meshes)
        print(f'objects/{name}  {size} bytes  {g.triangles} triangles  {verts} vertices  glass {"unlit" if unlit else "lit"}')

    allp = np.concatenate([np.array(m.P) for m in meshes[:-1]])
    lo, hi = allp.min(0), allp.max(0)
    tex = sum(gpu_bytes(im) for im in images.values())
    geo = sum(len(m.data[0]) * 32 + len(m.data[3]) * (2 if len(m.data[0]) < 65535 else 4) for m in meshes)
    print(f'GPU {(tex + geo) / 2 ** 20:.2f} MiB each = textures {tex / 2 ** 20:.2f} MiB (RGBA8 + mipmaps: '
          f'{", ".join(f"{k} {im.size[0]}x{im.size[1]}" for k, im in images.items())}) + geometry {geo / 2 ** 20:.2f} MiB')
    print(f'size {hi[0] - lo[0]:.3f} x {hi[1] - lo[1]:.3f} x {hi[2] - lo[2]:.3f} m, min {np.round(lo, 4)}, max {np.round(hi, 4)}')


if __name__ == '__main__':
    main()
