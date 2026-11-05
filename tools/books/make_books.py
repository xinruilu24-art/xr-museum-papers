#!/usr/bin/env python3
"""Step 2 of 3: builds the library GLBs from the rendered textures.

    python -I -X utf8 tools/books/make_books.py

Reads <build>/manifest.json (written by render_textures.mjs) and writes
books/<id>.glb, books/lectern.glb and books/<id>-on-lectern.glb.

Units are metres, Y up, glTF 2.0 binary with embedded JPEG textures.
A book lies flat, front cover up, head toward -Z, spine toward -X, origin
at the bottom centre of its footprint. The lectern faces +Z (the visitor
stands on the +Z side) with its origin at the bottom centre. The lectern
carries a soft floor-shadow decal and a book on the lectern a contact-shadow
decal (unlit, alpha-blended PNG), so they look grounded without real-time
shadows.
"""
import io
import json
import math
import os
import struct
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BOOKS = os.path.join(ROOT, 'books')


def node_tmpdir():
    # same folder as Node's os.tmpdir(), where render_textures.mjs writes by default
    if os.name == 'nt':
        return os.environ.get('TEMP') or os.environ.get('TMP') or tempfile.gettempdir()
    return os.environ.get('TMPDIR') or os.environ.get('TMP') or os.environ.get('TEMP') or '/tmp'


BUILD = os.environ.get('BOOKS_BUILD_DIR') or os.path.join(node_tmpdir(), 'xr-museum-books-build')

CLOTH_ROUGHNESS = 0.75
PAGE_ROUGHNESS = 0.8
WOOD_ROUGHNESS = 0.6
LECTERN_ANGLE = 20.0      # degrees
LIFT = 0.0005             # gap between book and lectern top
DECAL_LIFT = 0.0003       # contact shadow above the reading surface (under the book)
FLOOR_LIFT = 0.001        # floor shadow above the lectern's base plane
BACK_ON_LECTERN = (256, 256)   # the back cover lies face-down on the lectern
BOARD_ROUND = 0.0012      # radius of the rounded outer board edges
PAGE_V = 236 / 256        # page lines use the top of the edge texture,
BAND_V = (242 / 256, 255 / 256)   # headband stripes the bottom rows


# ---------------------------------------------------------------- glTF writer

class Glb:
    FLOAT, U16, U32 = 5126, 5123, 5125
    ARRAY, ELEMENTS = 34962, 34963

    def __init__(self):
        self.doc = {
            'asset': {'version': '2.0', 'generator': 'xr-museum-papers tools/books/make_books.py'},
            'scene': 0, 'scenes': [{'nodes': []}],
            'nodes': [], 'meshes': [], 'materials': [], 'textures': [], 'images': [],
            'samplers': [], 'accessors': [], 'bufferViews': [], 'buffers': [],
        }
        self.bin = bytearray()
        self._samplers = {}
        self._extensions = set()
        self.triangles = 0

    def _view(self, data, target=None):
        self.bin += b'\x00' * (-len(self.bin) % 4)
        view = {'buffer': 0, 'byteOffset': len(self.bin), 'byteLength': len(data)}
        if target is not None:
            view['target'] = target
        self.bin += data
        self.doc['bufferViews'].append(view)
        return len(self.doc['bufferViews']) - 1

    def _accessor(self, arr, ctype, atype, target, minmax=False):
        arr = np.ascontiguousarray(arr)
        acc = {'bufferView': self._view(arr.tobytes(), target), 'componentType': ctype,
               'count': int(arr.shape[0]), 'type': atype}
        if minmax:
            acc['min'] = [float(v) for v in arr.min(axis=0)]
            acc['max'] = [float(v) for v in arr.max(axis=0)]
        self.doc['accessors'].append(acc)
        return len(self.doc['accessors']) - 1

    def _sampler(self, wrap):
        if wrap not in self._samplers:
            mode = {'clamp': 33071, 'repeat': 10497}[wrap]
            self.doc['samplers'].append({'magFilter': 9729, 'minFilter': 9987, 'wrapS': mode, 'wrapT': mode})
            self._samplers[wrap] = len(self.doc['samplers']) - 1
        return self._samplers[wrap]

    def texture(self, data, name, wrap='clamp', mime='image/jpeg'):
        self.doc['images'].append({'name': name, 'mimeType': mime, 'bufferView': self._view(data)})
        self.doc['textures'].append({'name': name, 'sampler': self._sampler(wrap), 'source': len(self.doc['images']) - 1})
        return len(self.doc['textures']) - 1

    def shadow_material(self, name, texture):
        """Unlit black with the texture's alpha, blended: a soft shadow decal.
        Viewers without KHR_materials_unlit still show a black, blended,
        fully rough surface."""
        self.doc['materials'].append({
            'name': name,
            'pbrMetallicRoughness': {
                'baseColorTexture': {'index': texture},
                'baseColorFactor': [0.0, 0.0, 0.0, 1.0],
                'metallicFactor': 0.0,
                'roughnessFactor': 1.0,
            },
            'alphaMode': 'BLEND',
            'extensions': {'KHR_materials_unlit': {}},
        })
        self._extensions.add('KHR_materials_unlit')
        return len(self.doc['materials']) - 1

    def material(self, name, texture, roughness, metallic=0.0):
        self.doc['materials'].append({
            'name': name,
            'pbrMetallicRoughness': {
                'baseColorTexture': {'index': texture},
                'metallicFactor': metallic,
                'roughnessFactor': roughness,
            },
        })
        return len(self.doc['materials']) - 1

    def mesh(self, name, parts):
        prims = []
        for part, material in parts:
            P, N, UV, I = part.arrays()
            itype = self.U16 if I.max() < 65535 else self.U32
            prims.append({
                'attributes': {
                    'POSITION': self._accessor(P, self.FLOAT, 'VEC3', self.ARRAY, minmax=True),
                    'NORMAL': self._accessor(N, self.FLOAT, 'VEC3', self.ARRAY),
                    'TEXCOORD_0': self._accessor(UV, self.FLOAT, 'VEC2', self.ARRAY),
                },
                'indices': self._accessor(I.astype(np.uint16 if itype == self.U16 else np.uint32), itype, 'SCALAR', self.ELEMENTS),
                'material': material,
            })
            self.triangles += len(I) // 3
        self.doc['meshes'].append({'name': name, 'primitives': prims})
        return len(self.doc['meshes']) - 1

    def node(self, name, mesh=None, children=None, translation=None, rotation=None, root=False):
        node = {'name': name}
        if mesh is not None:
            node['mesh'] = mesh
        if children:
            node['children'] = list(children)
        if translation is not None:
            node['translation'] = [float(v) for v in translation]
        if rotation is not None:
            node['rotation'] = [float(v) for v in rotation]
        self.doc['nodes'].append(node)
        index = len(self.doc['nodes']) - 1
        if root:
            self.doc['scenes'][0]['nodes'].append(index)
        return index

    def write(self, path):
        self.bin += b'\x00' * (-len(self.bin) % 4)
        self.doc['buffers'] = [{'byteLength': len(self.bin)}]
        if self._extensions:
            self.doc['extensionsUsed'] = sorted(self._extensions)
        doc = {k: v for k, v in self.doc.items() if not (isinstance(v, list) and not v)}
        js = json.dumps(doc, separators=(',', ':')).encode('utf-8')
        js += b' ' * (-len(js) % 4)
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(struct.pack('<III', 0x46546C67, 2, total))
            f.write(struct.pack('<II', len(js), 0x4E4F534A))
            f.write(js)
            f.write(struct.pack('<II', len(self.bin), 0x004E4942))
            f.write(self.bin)
        os.replace(tmp, path)
        return total


# ---------------------------------------------------------------- mesh parts

def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def newell(pts):
    n = np.zeros(3)
    for i, p in enumerate(pts):
        q = pts[(i + 1) % len(pts)]
        n += np.array([(p[1] - q[1]) * (p[2] + q[2]), (p[2] - q[2]) * (p[0] + q[0]), (p[0] - q[0]) * (p[1] + q[1])])
    return unit(n)


class Part:
    """Triangles for one material. Polygons are given with the direction they
    should face; the winding is made counter-clockwise seen from that side."""

    def __init__(self):
        self.P, self.N, self.UV, self.I = [], [], [], []

    def poly(self, pts, uvs, facing, normals=None):
        pts = [np.asarray(p, dtype=float) for p in pts]
        facing = unit(facing)
        if np.dot(newell(pts), facing) < 0:
            pts, uvs = pts[::-1], uvs[::-1]
            normals = normals[::-1] if normals is not None else None
        base = len(self.P)
        for i, p in enumerate(pts):
            self.P.append(p)
            self.N.append(unit(normals[i]) if normals is not None else facing)
            self.UV.append(uvs[i])
        for k in range(1, len(pts) - 1):
            self.I += [base, base + k, base + k + 1]

    def transformed(self, R=None, t=None):
        out = Part()
        R = np.eye(3) if R is None else R
        t = np.zeros(3) if t is None else np.asarray(t, float)
        out.P = [R @ p + t for p in self.P]
        out.N = [unit(R @ n) for n in self.N]
        out.UV, out.I = list(self.UV), list(self.I)
        return out

    def extend(self, other):
        base = len(self.P)
        self.P += other.P
        self.N += other.N
        self.UV += other.UV
        self.I += [i + base for i in other.I]
        return self

    def arrays(self):
        return (np.array(self.P, dtype=np.float32), np.array(self.N, dtype=np.float32),
                np.array(self.UV, dtype=np.float32), np.array(self.I, dtype=np.int64))


def rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def quat_x(deg):
    a = math.radians(deg) / 2
    return [math.sin(a), 0.0, 0.0, math.cos(a)]


# ---------------------------------------------------------------- book

def arc_points(x_apex, chord_lo, chord_hi, sagitta, n):
    """Points of a circular arc from (x_joint, chord_lo) to (x_joint, chord_hi)
    bulging toward -X with its apex at x_apex. Returns (x, y, outward normal)."""
    yc = (chord_lo + chord_hi) / 2
    half = (chord_hi - chord_lo) / 2
    out = []
    if sagitta < 1e-4:
        for i in range(n + 1):
            t = i / n
            out.append((x_apex, chord_lo + t * (chord_hi - chord_lo), np.array([-1.0, 0.0, 0.0])))
        return out
    R = (half * half + sagitta * sagitta) / (2 * sagitta)
    cx = x_apex + R
    a = math.asin(min(1.0, half / R))
    for i in range(n + 1):
        phi = -a + 2 * a * i / n
        out.append((cx - R * math.cos(phi), yc + R * math.sin(phi), np.array([-math.cos(phi), math.sin(phi), 0.0])))
    return out


def build_book(d, segments=10):
    W, H, T = d['W'], d['H'], d['T']
    bt, sq, st, b = d['board'], d['square'], d['spineShell'], d['bulge']
    xs = -W / 2                     # spine apex
    xj = xs + b                     # joint: spine edge of the boards
    xe = W / 2                      # fore-edge of the boards
    cw = xe - xj                    # cover width
    z0, z1 = -H / 2, H / 2
    zp0, zp1 = z0 + sq, z1 - sq     # page block head / tail
    front, back, spine, pages = Part(), Part(), Part(), Part()

    def vz(z):
        return (z - z0) / H

    def uf(x):
        return (x - xj) / cw

    def ub(x):
        return (xe - x) / cw

    def box_face(part, pts, facing, uvf):
        part.poly(pts, [uvf(*p) for p in pts], facing)

    # Boards: flat outer face whose head, tail and fore-edge edges are rounded.
    # The cloth of the cover border runs over the rounding and is mirrored
    # onto the edge faces, so it seems to wrap around the board.
    r, k = BOARD_ROUND, 2

    def board(part, s, yo, yi, proj, edge_u, edge_v):
        """s = +1 for the front board (outer face up), -1 for the back.
        proj(x, z) gives the outer-face UV; edge_u(d) / edge_v(d, tail) map the
        distance d below the rounding onto the fore-edge / head and tail faces."""
        prof = []
        for j in range(k + 1):
            th = (math.pi / 2) * j / k
            prof.append((r * math.sin(th), yo - s * r * (1 - math.cos(th)), math.sin(th), s * math.cos(th)))

        def face(pts, facing, normals=None, uvf=None):
            uvf = uvf or (lambda p: proj(p[0], p[2]))
            part.poly(pts, [uvf(p) for p in pts], facing, normals=normals)

        face([(xj, yo, z0 + r), (xe - r, yo, z0 + r), (xe - r, yo, z1 - r), (xj, yo, z1 - r)], (0, s, 0))
        face([(xj, yi, z0), (xe, yi, z0), (xe, yi, z1), (xj, yi, z1)], (0, -s, 0))
        for j in range(k):
            (o0, y0, h0, v0), (o1, y1, h1, v1) = prof[j], prof[j + 1]
            # fore-edge rounding, mitred into the head and tail roundings
            face([(xe - r + o0, y0, z0 + r - o0), (xe - r + o1, y1, z0 + r - o1),
                  (xe - r + o1, y1, z1 - r + o1), (xe - r + o0, y0, z1 - r + o0)],
                 (h0 + h1, v0 + v1, 0), normals=[(h0, v0, 0), (h1, v1, 0), (h1, v1, 0), (h0, v0, 0)])
            for zs, zc in ((-1, z0 + r), (1, z1 - r)):
                face([(xj, y0, zc + zs * o0), (xe - r + o0, y0, zc + zs * o0),
                      (xe - r + o1, y1, zc + zs * o1), (xj, y1, zc + zs * o1)],
                     (0, v0 + v1, zs * (h0 + h1)),
                     normals=[(0, v0, zs * h0), (0, v0, zs * h0), (0, v1, zs * h1), (0, v1, zs * h1)])
        yk = prof[-1][1]
        face([(xe, yk, z0), (xe, yi, z0), (xe, yi, z1), (xe, yk, z1)], (1, 0, 0),
             uvf=lambda p: (edge_u(abs(p[1] - yk)), vz(p[2])))
        face([(xj, yk, z0), (xe, yk, z0), (xe, yi, z0), (xj, yi, z0)], (0, 0, -1),
             uvf=lambda p: (proj(p[0], z0)[0], edge_v(abs(p[1] - yk), False)))
        face([(xj, yk, z1), (xe, yk, z1), (xe, yi, z1), (xj, yi, z1)], (0, 0, 1),
             uvf=lambda p: (proj(p[0], z1)[0], edge_v(abs(p[1] - yk), True)))

    def edge_v(dist, tail):
        return 1 - dist / H if tail else dist / H

    # front: image left = spine; back (seen from behind): image left = fore-edge
    board(front, 1, T, T - bt, lambda x, z: (uf(x), vz(z)), lambda dist: 1 - dist / cw, edge_v)
    board(back, -1, 0.0, bt, lambda x, z: (ub(x), vz(z)), lambda dist: dist / cw, edge_v)

    # Spine shell: outer arc over the full thickness, inner arc between the
    # inner faces of the boards. t runs from the back board to the front.
    outer = arc_points(xs, 0.0, T, b, segments)
    inner = arc_points(xs + st, bt, T - bt, b - st, segments)
    for i in range(segments):
        t0, t1 = i / segments, (i + 1) / segments
        (ox0, oy0, on0), (ox1, oy1, on1) = outer[i], outer[i + 1]
        (ix0, iy0, in0), (ix1, iy1, in1) = inner[i], inner[i + 1]
        # outer surface (spine texture)
        spine.poly([(ox0, oy0, z0), (ox1, oy1, z0), (ox1, oy1, z1), (ox0, oy0, z1)],
                   [(t0, 0.0), (t1, 0.0), (t1, 1.0), (t0, 1.0)], (on0 + on1) / 2, normals=[on0, on1, on1, on0])
        # inner surface, seen only through the head and tail squares
        spine.poly([(ix0, iy0, z0), (ix1, iy1, z0), (ix1, iy1, z1), (ix0, iy0, z1)],
                   [(t0, 0.0), (t1, 0.0), (t1, 1.0), (t0, 1.0)], -(in0 + in1) / 2, normals=[-in0, -in1, -in1, -in0])
        # end faces of the shell, cloth mirrored from the spine's head/tail
        d0 = math.hypot(ox0 - ix0, oy0 - iy0) / H
        d1 = math.hypot(ox1 - ix1, oy1 - iy1) / H
        spine.poly([(ox0, oy0, z0), (ox1, oy1, z0), (ix1, iy1, z0), (ix0, iy0, z0)],
                   [(t0, 0.0), (t1, 0.0), (t1, d1), (t0, d0)], (0, 0, -1))
        spine.poly([(ox0, oy0, z1), (ox1, oy1, z1), (ix1, iy1, z1), (ix0, iy0, z1)],
                   [(t0, 1.0), (t1, 1.0), (t1, 1 - d1), (t0, 1 - d0)], (0, 0, 1))
    # Joint strips: close the shell where it meets the boards (x = xj). The
    # boards' head and tail edges are rounded right up to the joint, but the
    # shell's ends are square, so without these the small notch at each of
    # the four corners would open into the shell.
    for (ax, ay, _), (bx, by, _), ua, ub in ((outer[0], inner[0], 0.0, bt / d['spineArc']),
                                             (inner[-1], outer[-1], 1 - bt / d['spineArc'], 1.0)):
        spine.poly([(ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z1)],
                   [(ua, 0.0), (ub, 0.0), (ub, 1.0), (ua, 1.0)], (1, 0, 0))

    # Page block: spine side follows the inner arc, fore-edge slightly concave
    # (a rounded-and-backed book), recessed by the squares.
    xf_edge = xe - sq
    hh = (T - 2 * bt) / 2
    dip = 0.5 * b

    def xf(y):
        return xf_edge - dip * (1 - ((y - T / 2) / hh) ** 2)

    def fore_normal(y):
        slope = 2 * dip * (y - T / 2) / (hh * hh)      # dx/dy
        return unit([1.0, -slope, 0.0])

    xmin = xs + st

    def vp(y):
        return PAGE_V * (y - bt) / (T - 2 * bt)

    def upx(x):
        return (x - xmin) / (xf_edge - xmin)

    for i in range(segments):
        (ax, ay, _), (bx, by, _) = inner[i], inner[i + 1]
        fa, fb = xf(ay), xf(by)
        na, nb = fore_normal(ay), fore_normal(by)
        pages.poly([(fa, ay, zp0), (fb, by, zp0), (fb, by, zp1), (fa, ay, zp1)],
                   [(0.0, vp(ay)), (0.0, vp(by)), (1.0, vp(by)), (1.0, vp(ay))], (na + nb) / 2, normals=[na, nb, nb, na])
        for z, facing in ((zp0, (0, 0, -1)), (zp1, (0, 0, 1))):
            pages.poly([(ax, ay, z), (fa, ay, z), (fb, by, z), (bx, by, z)],
                       [(upx(ax), vp(ay)), (upx(fa), vp(ay)), (upx(fb), vp(by)), (upx(bx), vp(by))], facing)

    # Headbands: a small striped bead at the head and tail of the page block,
    # hugging the spine; seen through the squares. Its first and last inner
    # points sit on the boards' inner faces (y = bt and y = T - bt), so the
    # bead ends flush against the boards instead of leaving open ends.
    hb, wb = 0.0014, 0.0026
    v0, v1 = BAND_V
    vm = (v0 + v1) / 2

    def band_point(i):
        x, y, n = inner[i]
        if i == 0:
            return (x + wb, bt)
        if i == segments:
            return (x + wb, T - bt)
        return (x - n[0] * wb, y - n[1] * wb)

    for zs, zp in ((-1, zp0), (1, zp1)):
        ztop = zp + zs * hb
        along = 0.0
        for i in range(segments):
            (ax, ay, an), (bx, by, bn) = inner[i], inner[i + 1]
            ia, ib = -an, -bn                            # toward the page block
            seg = math.hypot(bx - ax, by - ay)
            u0, u1 = along / 0.064, (along + seg) / 0.064   # 1 mm stripes
            along += seg
            pa = band_point(i)
            pb = band_point(i + 1)
            tilt_a = ia * 0.6 + np.array([0, 0, zs * 0.8])
            tilt_b = ib * 0.6 + np.array([0, 0, zs * 0.8])
            pages.poly([(ax, ay, ztop), (bx, by, ztop), (pb[0], pb[1], ztop), (pa[0], pa[1], ztop)],
                       [(u0, v0), (u1, v0), (u1, vm), (u0, vm)], (0, 0, zs),
                       normals=[(0, 0, zs), (0, 0, zs), tilt_b, tilt_a])
            pages.poly([(pa[0], pa[1], ztop), (pb[0], pb[1], ztop), (pb[0], pb[1], zp), (pa[0], pa[1], zp)],
                       [(u0, vm), (u1, vm), (u1, v1), (u0, v1)], (ia + ib) / 2,
                       normals=[tilt_a, tilt_b, ib, ia])
    return {'front': front, 'back': back, 'spine': spine, 'pages': pages}


# ---------------------------------------------------------------- lectern

def wood_texture(size=1024, seed=7, out_size=(256, 1024)):
    """Tileable dark walnut. The grain runs along u (image x). It is drawn at
    size x size and stored at out_size: full resolution across the grain,
    where the rings are, and a quarter of it along the grain, where wood
    changes slowly (a quarter of the GPU memory of a square texture)."""
    rng = np.random.default_rng(seed)
    n = size

    def noise(cy, cx, octaves=4, persistence=0.5):
        out = np.zeros((n, n))
        amp, total = 1.0, 0.0
        for o in range(octaves):
            gy, gx = cy * 2 ** o, cx * 2 ** o
            g = rng.random((gy, gx))
            ys = np.arange(n) * gy / n
            xs = np.arange(n) * gx / n
            y0 = np.floor(ys).astype(int)
            x0 = np.floor(xs).astype(int)
            fy = ys - y0
            fx = xs - x0
            fy = fy * fy * (3 - 2 * fy)
            fx = fx * fx * (3 - 2 * fx)
            y1, x1 = (y0 + 1) % gy, (x0 + 1) % gx
            a, b_ = g[np.ix_(y0, x0)], g[np.ix_(y0, x1)]
            c, d = g[np.ix_(y1, x0)], g[np.ix_(y1, x1)]
            v = (a * (1 - fx) + b_ * fx) * (1 - fy)[:, None] + (c * (1 - fx) + d * fx) * fy[:, None]
            out += amp * v
            total += amp
            amp *= persistence
        return out / total

    u = (np.arange(n)[None, :] / n) * np.ones((n, 1))
    v = (np.arange(n)[:, None] / n) * np.ones((1, n))
    rings = 88                                  # growth rings across one tile
    widths = 1 + 0.4 * (2 * rng.random(rings) - 1)               # +-40% ring widths
    bounds = np.concatenate([[0.0], np.cumsum(widths)]) / widths.sum()
    warp = (noise(2, 1, 3) - 0.5) * 7.0 + (noise(6, 2, 3) - 0.5) * 3.0 + (noise(24, 3, 2) - 0.5) * 0.7

    def arch(u0, v0, p, q):
        # Tileable ridge in the ring field, long along the grain and narrow
        # across it: the rings around it close into long nested arches, the
        # cathedral figure of flat-sawn wood.
        return ((1 + np.cos(2 * np.pi * (u - u0))) / 2) ** p * ((1 + np.cos(2 * np.pi * (v - v0))) / 2) ** q

    warp += 13.5 * arch(0.30, 0.34, 0.9, 5) + 11.0 * arch(0.80, 0.80, 1.4, 7)
    s = np.mod(v + warp / rings, 1.0)
    ring_id = np.clip(np.searchsorted(bounds, s, side='right') - 1, 0, rings - 1)
    f = (s - bounds[ring_id]) / (bounds[ring_id + 1] - bounds[ring_id])
    ring_dark = rng.random(rings)[ring_id] * 0.6 + 0.4
    line = np.exp(-((f - 0.82) / 0.09) ** 2) * ring_dark       # soft latewood line
    late = np.clip((f - 0.35) / 0.5, 0, 1) ** 2 * 0.3           # darkening toward it
    pores = np.clip((noise(700, 5, 2) - 0.58) * 6, 0, 1)         # sparse pore streaks
    fine = noise(380, 3, 3) - 0.5
    figure = noise(3, 2, 4) - 0.5
    bands = noise(16, 1, 3) - 0.5                                # long colour streaks
    base = np.array([82, 50, 31], float)
    dark = np.array([40, 23, 13], float)
    light = np.array([112, 72, 44], float)
    k = np.clip(0.30 * line + late + 0.3 * pores + 0.45 * fine, 0, 1)[..., None]
    col = base * (1 - k) + dark * k
    col += (light - base) * np.clip(figure * 1.2 + bands * 1.1, -0.5, 0.55)[..., None]
    col *= (1 + 0.05 * (noise(48, 12, 2) - 0.5))[..., None]
    img = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8), 'RGB')
    return img.resize(out_size, Image.LANCZOS) if out_size and tuple(out_size) != img.size else img


class WoodMapper:
    """Planar UVs in metres / scale, the grain (texture u) along an axis.
    Faces that do not contain that axis get horizontal grain."""

    def __init__(self, grain_axis=0, scale=0.6, offset=(0.0, 0.0)):
        self.g, self.s, self.o = grain_axis, scale, offset

    def uv(self, p, facing):
        k = int(np.argmax(np.abs(facing)))
        plane = [a for a in range(3) if a != k]
        ua = self.g if self.g in plane else (plane[1] if plane[0] == 1 else plane[0])
        va = plane[0] if plane[1] == ua else plane[1]
        return (p[ua] / self.s + self.o[0], p[va] / self.s + self.o[1])


def bevel_box(part, size, r, mapper, R=None, t=None):
    """Box with chamfered edges and corners. The chamfers use the normals of
    the faces they join, so the edges shade like rounded ones."""
    R = np.eye(3) if R is None else R
    t = np.zeros(3) if t is None else np.asarray(t, float)
    h = np.array(size, float) / 2
    i = h - r
    local = Part()

    def add(pts, normals):
        facing = unit(np.sum(normals, axis=0))
        uvs = [mapper.uv(R @ p + t, R @ facing) for p in pts]
        local.poly(pts, uvs, facing, normals=normals)

    axes = np.eye(3)
    for a in range(3):
        for s in (-1, 1):
            b_, c = [x for x in range(3) if x != a]
            pts = []
            for sb, sc in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[a], p[b_], p[c] = s * h[a], sb * i[b_], sc * i[c]
                pts.append(p)
            add(pts, [s * axes[a]] * 4)
    for a in range(3):
        for b_ in range(a + 1, 3):
            c = 3 - a - b_
            for sa in (-1, 1):
                for sb in (-1, 1):
                    pts, nrm = [], []
                    for sc, side in ((-1, 'a'), (1, 'a'), (1, 'b'), (-1, 'b')):
                        p = np.zeros(3)
                        p[c] = sc * i[c]
                        if side == 'a':
                            p[a], p[b_] = sa * h[a], sb * i[b_]
                            nrm.append(sa * axes[a])
                        else:
                            p[a], p[b_] = sa * i[a], sb * h[b_]
                            nrm.append(sb * axes[b_])
                        pts.append(p)
                    add(pts, nrm)
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                s = np.array([sx, sy, sz], float)
                pts, nrm = [], []
                for a in range(3):
                    p = s * i
                    p[a] = s[a] * h[a]
                    pts.append(p)
                    nrm.append(s[a] * axes[a])
                add(pts, nrm)
    part.extend(local.transformed(R, t))


def lathe(part, profile, segments, mapper_scale=0.6, crease_deg=35.0, offset=(0.0, 0.0)):
    """Surface of revolution about Y. profile: [(radius, y), ...] bottom to top.
    Grain runs up the column (texture u = height)."""
    seg_n = []
    for (r0, y0), (r1, y1) in zip(profile, profile[1:]):
        dr, dy = r1 - r0, y1 - y0
        seg_n.append(unit([dy, -dr]))                    # (radial, y), outward
    # normals at the ends of every profile segment, smoothed over gentle bends
    ends = []
    for k, nk in enumerate(seg_n):
        lo, hi = nk, nk
        if k > 0 and math.degrees(math.acos(np.clip(np.dot(seg_n[k - 1], nk), -1, 1))) < crease_deg:
            lo = unit(seg_n[k - 1] + nk)
        if k < len(seg_n) - 1 and math.degrees(math.acos(np.clip(np.dot(seg_n[k + 1], nk), -1, 1))) < crease_deg:
            hi = unit(seg_n[k + 1] + nk)
        ends.append((lo, hi))
    ref = 0.05
    for k, ((r0, y0), (r1, y1)) in enumerate(zip(profile, profile[1:])):
        (n0, n1) = ends[k]
        for j in range(segments):
            a0, a1 = 2 * math.pi * j / segments, 2 * math.pi * (j + 1) / segments
            pts, nrm, uvs = [], [], []
            for (r, y, nn), a in (((r0, y0, n0), a0), ((r0, y0, n0), a1), ((r1, y1, n1), a1), ((r1, y1, n1), a0)):
                ca, sa = math.cos(a), math.sin(a)
                pts.append((r * ca, y, r * sa))
                nrm.append(np.array([nn[0] * ca, nn[1], nn[0] * sa]))
                uvs.append((y / mapper_scale + offset[0], a * ref / mapper_scale + offset[1]))
            part.poly(pts, uvs, unit(np.sum(nrm, axis=0)), normals=nrm)


def quarter(o0, y0, r, kind, n=4):
    """Profile points of a quarter-round moulding that starts at (o0, y0) and
    steps in by r while rising r: 'ovolo' bulges out, 'cove' is hollow."""
    pts = []
    for j in range(1, n + 1):
        th = (math.pi / 2) * j / n
        if kind == 'ovolo':
            pts.append((o0 + r * (1 - math.cos(th)), y0 + r * math.sin(th)))
        else:
            pts.append((o0 + r * math.sin(th), y0 + r * (1 - math.cos(th))))
    return pts


def rect_sweep(part, hx, hz, profile, mapper, crease_deg=35.0):
    """Moulded block on the Y axis: a rectangle with half sizes (hx, hz),
    inset by the profile [(inset, y), ...] (bottom to top), with mitred
    corners, closed by flat bottom and top faces."""
    seg_n = [unit([y1 - y0, o1 - o0]) for (o0, y0), (o1, y1) in zip(profile, profile[1:])]  # (out, up)
    ends = []
    for k, nk in enumerate(seg_n):
        lo, hi = nk, nk
        if k > 0 and math.degrees(math.acos(np.clip(np.dot(seg_n[k - 1], nk), -1, 1))) < crease_deg:
            lo = unit(seg_n[k - 1] + nk)
        if k < len(seg_n) - 1 and math.degrees(math.acos(np.clip(np.dot(seg_n[k + 1], nk), -1, 1))) < crease_deg:
            hi = unit(seg_n[k + 1] + nk)
        ends.append((lo, hi))
    for k, ((o0, y0), (o1, y1)) in enumerate(zip(profile, profile[1:])):
        n0, n1 = ends[k]
        for ax, sg in ((0, 1), (0, -1), (2, 1), (2, -1)):
            al = 2 - ax                                  # the other horizontal axis
            ha, hb = (hx, hz) if ax == 0 else (hz, hx)
            pts, nrm = [], []
            for (o, y, nn), s2 in (((o0, y0, n0), -1), ((o0, y0, n0), 1), ((o1, y1, n1), 1), ((o1, y1, n1), -1)):
                p = np.zeros(3)
                p[1], p[ax], p[al] = y, sg * (ha - o), s2 * (hb - o)
                q = np.zeros(3)
                q[1], q[ax] = nn[1], sg * nn[0]
                pts.append(p)
                nrm.append(q)
            facing = unit(np.sum(nrm, axis=0))
            part.poly(pts, [mapper.uv(p, facing) for p in pts], facing, normals=nrm)
    for (o, y), fy in ((profile[0], -1.0), (profile[-1], 1.0)):
        pts = [np.array([sx * (hx - o), y, sz * (hz - o)]) for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        facing = np.array([0.0, fy, 0.0])
        part.poly(pts, [mapper.uv(p, facing) for p in pts], facing)


PLINTH = (0.25, 0.22)     # half sizes of the lectern's footprint on the floor


def build_lectern():
    """Classic library lectern: moulded two-step plinth, turned baluster
    column with a collar and a cove capital, moulded skirt, desk box and a
    20 degree reading top with a book ledge. Returns (part, info)."""
    wood = Part()
    angle = LECTERN_ANGLE
    top_w, top_d, top_t = 0.55, 0.42, 0.022
    height = 1.05
    Rt = rot_x(angle)
    sa, ca = math.sin(math.radians(angle)), math.cos(math.radians(angle))
    cy = height - (top_d / 2) * sa             # centre of the reading surface
    centre = np.array([0.0, cy, 0.0])

    # plinth: a wide base with a rounded (ovolo) top edge, and on it a
    # smaller block whose top edge is a cove with a fillet
    base = [(0.0, 0.0), (0.0, 0.024)] + quarter(0.0, 0.024, 0.010, 'ovolo')
    rect_sweep(wood, PLINTH[0], PLINTH[1], base, WoodMapper(0, offset=(0.13, 0.41)))
    upper = [(0.0, 0.032), (0.0, 0.046)] + quarter(0.0, 0.046, 0.010, 'cove', 3) + [(0.014, 0.056), (0.014, 0.060)]
    rect_sweep(wood, 0.185, 0.16, upper, WoodMapper(0, offset=(0.57, 0.08)))
    # turned baluster column: base torus and scotia, a shaft that swells to
    # its widest about a third of the way up, a collar, and a cove capital
    col_top = 0.83
    profile = [
        (0.084, 0.058), (0.084, 0.066),                                         # fillet
        (0.088, 0.070), (0.089, 0.075), (0.087, 0.080), (0.081, 0.084),         # torus
        (0.075, 0.086), (0.070, 0.090), (0.066, 0.097), (0.064, 0.106),         # scotia
        (0.063, 0.120), (0.066, 0.190), (0.068, 0.290), (0.068, 0.370),         # shaft, swell
        (0.065, 0.450), (0.060, 0.540), (0.055, 0.630), (0.051, 0.700), (0.050, 0.724),
        (0.053, 0.728), (0.057, 0.733), (0.058, 0.740), (0.056, 0.747), (0.051, 0.751),  # collar
        (0.048, 0.755), (0.048, 0.762),                                         # neck
        (0.052, 0.775), (0.060, 0.788), (0.072, 0.799), (0.085, 0.806), (0.094, 0.810),  # cove
        (0.096, 0.814), (0.096, 0.826), (0.093, 0.830), (0.090, col_top + 0.003),        # abacus
    ]
    lathe(wood, profile, 16, offset=(0.21, 0.66))
    # moulded skirt and desk box under the reading top
    skirt_y = col_top + 0.011
    bevel_box(wood, (0.50, 0.022, 0.32), 0.006, WoodMapper(0, offset=(0.37, 0.29)), t=(0, skirt_y, 0))
    bx, bz, yb = 0.235, 0.14, skirt_y + 0.011

    def under(z):                              # underside of the top board (+2 mm overlap)
        return cy - top_t * ca - (z + top_t * sa) * (sa / ca) + 0.002

    m = WoodMapper(0, offset=(0.77, 0.52))
    faces = [
        ([(-bx, yb, bz), (bx, yb, bz), (bx, under(bz), bz), (-bx, under(bz), bz)], (0, 0, 1)),
        ([(-bx, yb, -bz), (bx, yb, -bz), (bx, under(-bz), -bz), (-bx, under(-bz), -bz)], (0, 0, -1)),
        ([(bx, yb, -bz), (bx, yb, bz), (bx, under(bz), bz), (bx, under(-bz), -bz)], (1, 0, 0)),
        ([(-bx, yb, -bz), (-bx, yb, bz), (-bx, under(bz), bz), (-bx, under(-bz), -bz)], (-1, 0, 0)),
    ]
    for pts, facing in faces:
        wood.poly(pts, [m.uv(np.array(p), np.array(facing, float)) for p in pts], facing)
    # reading top, its upper surface through the centre point
    bevel_box(wood, (top_w, top_t, top_d), 0.004, WoodMapper(0, offset=(0.31, 0.12)),
              R=Rt, t=centre + Rt @ np.array([0, -top_t / 2, 0]))
    # book ledge along the lower (visitor) edge
    lip_h, lip_d = 0.02, 0.016
    lip_inner = top_d / 2 - lip_d              # local z of the face the book rests against
    bevel_box(wood, (top_w - 0.02, lip_h + 0.004, lip_d + 0.002), 0.004, WoodMapper(0, offset=(0.44, 0.71)),
              R=Rt, t=centre + Rt @ np.array([0, lip_h / 2 - 0.002, top_d / 2 - lip_d / 2 + 0.001]))
    info = {'centre': centre, 'R': Rt, 'lip_inner': lip_inner, 'angle': angle, 'top': (top_w, top_d)}
    return wood, info


# ---------------------------------------------------------------- output

def jpeg(img, quality=90, subsampling=0):
    if isinstance(img, str):
        img = Image.open(img)
    img = img.convert('RGB')
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=quality, subsampling=subsampling, optimize=True)
    return buf.getvalue()


def add_book_mesh(g, book, parts, tex):
    mats = {
        'front': g.material(f'{book["id"]} cover', g.texture(tex['front'], f'{book["id"]}-front'), CLOTH_ROUGHNESS),
        'spine': g.material(f'{book["id"]} spine', g.texture(tex['spine'], f'{book["id"]}-spine'), CLOTH_ROUGHNESS),
        'back': g.material(f'{book["id"]} back', g.texture(tex['back'], f'{book["id"]}-back'), CLOTH_ROUGHNESS),
        'pages': g.material(f'{book["id"]} pages', g.texture(tex['edges'], f'{book["theme"]}-page-edges'), PAGE_ROUGHNESS),
    }
    return g.mesh(f'Book {book["id"]}', [(parts[k], mats[k]) for k in ('front', 'spine', 'back', 'pages')])


def png(img):
    buf = io.BytesIO()
    img.save(buf, 'PNG', optimize=True)
    return buf.getvalue()


def soft_shadow(size, half_w, half_h, margin, near, near_fall, far, far_fall, radius):
    """Black RGBA image for a shadow decal: alpha falls off around a rounded
    rectangle (half sizes in metres) centred in a quad that is margin larger
    on every side, and reaches zero at the quad's border. Rows run along z
    (v), columns along x (u)."""
    W2, H2 = half_w + margin, half_h + margin
    X, Z = np.meshgrid((np.arange(size) + 0.5) / size * 2 * W2 - W2, (np.arange(size) + 0.5) / size * 2 * H2 - H2)
    qx, qz = np.abs(X) - (half_w - radius), np.abs(Z) - (half_h - radius)
    d = np.hypot(np.maximum(qx, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, qz), 0) - radius
    dp = np.maximum(d, 0)
    a = near * np.exp(-dp / near_fall) + far * np.exp(-(dp / far_fall) ** 2)
    a *= np.clip(np.minimum(W2 - np.abs(X), H2 - np.abs(Z)) / (0.3 * margin), 0, 1) ** 2
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., 3] = np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, 'RGBA')


def decal_quad(half_w, half_h, y):
    part = Part()
    part.poly([(-half_w, y, -half_h), (half_w, y, -half_h), (half_w, y, half_h), (-half_w, y, half_h)],
              [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], (0, 1, 0))
    return part


FLOOR_MARGIN = 0.12


def floor_shadow():
    """Soft contact shadow on the floor around the plinth (it also helps where
    FRAME has no real-time shadows)."""
    img = soft_shadow(256, PLINTH[0], PLINTH[1], FLOOR_MARGIN, 0.42, 0.022, 0.2, 0.07, 0.01)
    return decal_quad(PLINTH[0] + FLOOR_MARGIN, PLINTH[1] + FLOOR_MARGIN, FLOOR_LIFT), png(img)


BOOK_MARGIN = 0.016


def book_shadow(d):
    """Contact shadow under a book on the reading top, in the book's frame:
    darkest at the book's edges and fading within about a centimetre."""
    img = soft_shadow(256, d['W'] / 2, d['H'] / 2, BOOK_MARGIN, 0.6, 0.0045, 0.16, 0.010, 0.002)
    return decal_quad(d['W'] / 2 + BOOK_MARGIN, d['H'] / 2 + BOOK_MARGIN, DECAL_LIFT - LIFT), png(img)


def add_lectern_mesh(g, wood_part, wood_jpeg, floor):
    mat = g.material('lectern wood', g.texture(wood_jpeg, 'walnut', wrap='repeat'), WOOD_ROUGHNESS)
    quad, shadow_png = floor
    shadow = g.shadow_material('lectern floor shadow', g.texture(shadow_png, 'lectern-floor-shadow', mime='image/png'))
    return g.mesh('Lectern', [(wood_part, mat), (quad, shadow)])


def load_texture(path, size=None):
    """A rendered texture, downsampled to its stored size if it was rendered
    larger (see RENDER_SCALE in lib.mjs)."""
    img = Image.open(path).convert('RGB')
    if size and tuple(img.size) != tuple(size):
        img = img.resize(tuple(size), Image.LANCZOS)
    return img


def main():
    with open(os.path.join(BUILD, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    os.makedirs(BOOKS, exist_ok=True)
    edges = {}
    wood_img = wood_texture()
    wood_img.save(os.path.join(BUILD, 'tex', 'walnut.png'))
    wood_jpg = jpeg(wood_img, 88, 2)
    lectern_part, L = build_lectern()
    floor = floor_shadow()
    rows = []

    g = Glb()
    g.node('lectern', mesh=add_lectern_mesh(g, lectern_part, wood_jpg, floor), root=True)
    path = os.path.join(BOOKS, 'lectern.glb')
    rows.append(('lectern.glb', g.write(path), g.triangles))

    for book in manifest['books']:
        d = book['dims']
        parts = build_book(d)
        size = book.get('textureSizes', {})
        tex = {
            'front': jpeg(load_texture(book['textures']['front'], size.get('front')), 90, 0),
            'spine': jpeg(load_texture(book['textures']['spine'], size.get('spine')), 90, 0),
            'back': jpeg(load_texture(book['textures']['back'], size.get('back')), 88, 2),
            'edges': edges.setdefault(book['textures']['edges'],
                                      jpeg(load_texture(book['textures']['edges'], size.get('edges')), 90, 0)),
        }
        g = Glb()
        g.node(book['id'], mesh=add_book_mesh(g, book, parts, tex), root=True)
        name = f'{book["id"]}.glb'
        rows.append((name, g.write(os.path.join(BOOKS, name)), g.triangles))

        # book on the lectern: tail against the ledge, centred left-right,
        # with a soft contact shadow on the reading top under it. Its back
        # cover faces the reading top, so a small copy of that texture will do
        # (only the board edges show it).
        g = Glb()
        lectern = g.node('lectern', mesh=add_lectern_mesh(g, lectern_part, wood_jpg, floor))
        z_off = L['lip_inner'] - 0.001 - d['H'] / 2
        t = L['centre'] + L['R'] @ np.array([0.0, LIFT, z_off])
        q = quat_x(L['angle'])
        tex_l = dict(tex, back=jpeg(load_texture(book['textures']['back'], BACK_ON_LECTERN), 88, 2))
        bk = g.node(book['id'], mesh=add_book_mesh(g, book, parts, tex_l), translation=t, rotation=q)
        quad, shadow_png = book_shadow(d)
        mat = g.shadow_material(f'{book["id"]} contact shadow',
                                g.texture(shadow_png, f'{book["id"]}-contact-shadow', mime='image/png'))
        sh = g.node(f'{book["id"]}-contact-shadow', mesh=g.mesh(f'Contact shadow {book["id"]}', [(quad, mat)]),
                    translation=t, rotation=q)
        g.node(f'{book["id"]}-on-lectern', children=[lectern, bk, sh], root=True)
        name = f'{book["id"]}-on-lectern.glb'
        rows.append((name, g.write(os.path.join(BOOKS, name)), g.triangles))

    w = max(len(r[0]) for r in rows)
    print(f'{"file".ljust(w)}  {"bytes":>9}  {"triangles":>9}')
    for name, size, tris in rows:
        print(f'{name.ljust(w)}  {size:>9}  {tris:>9}')


if __name__ == '__main__':
    main()
