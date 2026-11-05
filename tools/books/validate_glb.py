#!/usr/bin/env python3
"""Checks the library GLBs without glTF packages (preview.mjs also runs the
Khronos glTF Validator):

    python -I -X utf8 tools/books/validate_glb.py            every books/*.glb
    python -I -X utf8 tools/books/validate_glb.py a.glb ...

Checks the GLB container and chunk padding, JSON references, buffer view and
accessor bounds, POSITION min/max, unit normals, index range, triangle
winding against the normals (counter-clockwise seen from outside),
degenerate triangles, UVs inside [0, 1] for clamped textures, that every
embedded image decodes at a power-of-two size, and the 5 MB size limit.
Prints size, triangles and the bounding box (node transforms applied)."""
import glob
import io
import json
import os
import struct
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
BOOKS = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'books')
MAX_BYTES = 5 * 1024 * 1024
COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def quat_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def check(path):
    problems = []
    data = open(path, 'rb').read()
    magic, version, length = struct.unpack_from('<III', data, 0)
    if magic != 0x46546C67 or version != 2:
        return {'problems': ['not a glTF 2.0 binary']}
    if length != len(data):
        problems.append(f'header length {length} != file size {len(data)}')
    jlen, jtype = struct.unpack_from('<II', data, 12)
    if jtype != 0x4E4F534A or jlen % 4:
        problems.append('bad JSON chunk')
    doc = json.loads(data[20:20 + jlen].decode('utf-8'))
    off = 20 + jlen
    blen, btype = struct.unpack_from('<II', data, off)
    if btype != 0x004E4942 or blen % 4:
        problems.append('bad BIN chunk')
    binary = data[off + 8:off + 8 + blen]
    if off + 8 + blen != len(data):
        problems.append('trailing bytes after BIN chunk')
    if doc.get('asset', {}).get('version') != '2.0':
        problems.append('asset.version is not 2.0')
    if doc['buffers'][0]['byteLength'] > blen:
        problems.append('buffer longer than BIN chunk')

    views = doc.get('bufferViews', [])
    for i, v in enumerate(views):
        if v['byteOffset'] + v['byteLength'] > blen:
            problems.append(f'bufferView {i} out of range')

    def read(ai):
        a = doc['accessors'][ai]
        v = views[a['bufferView']]
        dt = COMP[a['componentType']]
        n = NCOMP[a['type']]
        start = v['byteOffset'] + a.get('byteOffset', 0)
        if start % np.dtype(dt).itemsize:
            problems.append(f'accessor {ai} misaligned')
        size = a['count'] * n * np.dtype(dt).itemsize
        if start + size > v['byteOffset'] + v['byteLength']:
            problems.append(f'accessor {ai} exceeds its bufferView')
        arr = np.frombuffer(binary, dtype=dt, count=a['count'] * n, offset=start)
        return a, (arr.reshape(-1, n) if n > 1 else arr)

    clamp_tex = set()
    for ti, t in enumerate(doc.get('textures', [])):
        s = doc['samplers'][t['sampler']]
        if s.get('wrapS') == 33071 and s.get('wrapT') == 33071:
            clamp_tex.add(ti)
    for ii, img in enumerate(doc.get('images', [])):
        v = views[img['bufferView']]
        if 'target' in v:
            problems.append(f'image {ii} bufferView has a target')
        im = Image.open(io.BytesIO(binary[v['byteOffset']:v['byteOffset'] + v['byteLength']]))
        im.load()
        w, h = im.size
        if w & (w - 1) or h & (h - 1):
            problems.append(f'image {ii} is {w}x{h}, not power-of-two')
        fmt = {'JPEG': 'image/jpeg', 'PNG': 'image/png'}.get(im.format)
        if fmt != img.get('mimeType'):
            problems.append(f'image {ii} is {im.format} but mimeType is {img.get("mimeType")}')
        if im.mode not in ('RGB', 'L', 'RGBA', 'LA'):
            problems.append(f'image {ii} mode {im.mode}')
        if im.mode in ('RGBA', 'LA'):
            for m in doc.get('materials', []):
                ti = m.get('pbrMetallicRoughness', {}).get('baseColorTexture', {}).get('index')
                if ti is not None and doc['textures'][ti]['source'] == ii and m.get('alphaMode') != 'BLEND':
                    problems.append(f'image {ii} has alpha but material {m.get("name")} is not BLEND')

    mesh_tris = {}
    mesh_points = {}
    for mi, mesh in enumerate(doc['meshes']):
        tris, pts = 0, []
        for pi, prim in enumerate(mesh['primitives']):
            tag = f'mesh {mi} prim {pi}'
            if prim.get('mode', 4) != 4:
                problems.append(f'{tag} is not TRIANGLES')
            pa, P = read(prim['attributes']['POSITION'])
            na, N = read(prim['attributes']['NORMAL'])
            ua, UV = read(prim['attributes']['TEXCOORD_0'])
            ia, I = read(prim['indices'])
            I = I.astype(np.int64)
            if not (len(P) == len(N) == len(UV)):
                problems.append(f'{tag} attribute counts differ')
            if not np.allclose(pa['min'], P.min(0), atol=0) or not np.allclose(pa['max'], P.max(0), atol=0):
                problems.append(f'{tag} POSITION min/max do not match the data')
            if I.max() >= len(P) or len(I) % 3:
                problems.append(f'{tag} bad indices')
            ln = np.linalg.norm(N, axis=1)
            if np.abs(ln - 1).max() > 1e-4:
                problems.append(f'{tag} normals not unit length (max error {np.abs(ln - 1).max():.2e})')
            t = I.reshape(-1, 3)
            a, b, c = P[t[:, 0]].astype(float), P[t[:, 1]].astype(float), P[t[:, 2]].astype(float)
            g = np.cross(b - a, c - a)
            area = np.linalg.norm(g, axis=1) / 2
            nbar = N[t[:, 0]] + N[t[:, 1]] + N[t[:, 2]]
            wrong = int(np.sum(np.einsum('ij,ij->i', g, nbar) <= 0))
            if wrong:
                problems.append(f'{tag} {wrong} triangles wound against their normals')
            degenerate = int(np.sum(area < 1e-10))
            if degenerate:
                problems.append(f'{tag} {degenerate} degenerate triangles')
            tex = doc['materials'][prim['material']]['pbrMetallicRoughness'].get('baseColorTexture', {}).get('index')
            if tex in clamp_tex and (UV.min() < -1e-4 or UV.max() > 1 + 1e-4):
                problems.append(f'{tag} UVs outside [0,1] on a clamped texture ({UV.min():.3f}..{UV.max():.3f})')
            if not np.isfinite(P).all() or not np.isfinite(UV).all():
                problems.append(f'{tag} non-finite values')
            tris += len(t)
            pts.append(P.astype(float))
        mesh_tris[mi] = tris
        mesh_points[mi] = np.concatenate(pts)

    # walk the scene with node transforms
    total_tris, world = 0, []

    def walk(ni, M, t):
        nonlocal total_tris
        node = doc['nodes'][ni]
        R = quat_matrix(node.get('rotation', [0, 0, 0, 1]))
        S = np.diag(node.get('scale', [1, 1, 1]))
        M2 = M @ R @ S
        t2 = t + M @ np.array(node.get('translation', [0, 0, 0]), float)
        if 'mesh' in node:
            total_tris += mesh_tris[node['mesh']]
            world.append(mesh_points[node['mesh']] @ M2.T + t2)
        for ch in node.get('children', []):
            walk(ch, M2, t2)

    for ni in doc['scenes'][doc.get('scene', 0)]['nodes']:
        walk(ni, np.eye(3), np.zeros(3))
    allp = np.concatenate(world)
    if len(data) > MAX_BYTES:
        problems.append('larger than 5 MB')
    return {'bytes': len(data), 'triangles': total_tris, 'min': allp.min(0), 'max': allp.max(0), 'problems': problems,
            'materials': len(doc.get('materials', [])), 'images': [Image.open(io.BytesIO(binary[views[i['bufferView']]['byteOffset']:views[i['bufferView']]['byteOffset'] + views[i['bufferView']]['byteLength']])).size for i in doc.get('images', [])]}


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(BOOKS, '*.glb')))
    bad = 0
    for f in files:
        r = check(f)
        name = os.path.basename(f)
        if 'bytes' not in r:
            print(f'FAIL {name}: {"; ".join(r["problems"])}')
            bad += 1
            continue
        size = r['max'] - r['min']
        status = 'FAIL' if r['problems'] else 'ok  '
        bad += bool(r['problems'])
        print(f'{status} {name:30} {r["bytes"]:>8} B  {r["triangles"]:>5} tris  '
              f'size {size[0]:.4f} x {size[1]:.4f} x {size[2]:.4f} m  min y {r["min"][1]:+.4f}  '
              f'images {" ".join(f"{w}x{h}" for w, h in r["images"])}')
        for p in r['problems']:
            print('      ', p)
    print(f'\n{len(files) - bad} of {len(files)} files passed.')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
