"""Moulding section and carved-ornament height fields for the gilded stage mirror (pure numpy, imported by mirror.py).
Units are metres in the mirror's own plane: u runs right, v runs up, d is the height above the frame's back plane."""
import numpy as np

SIGHT_W, SIGHT_H = 0.62, 1.55
MOULD_W = 0.11
FOOT = 0.028
OUT_W, OUT_H = SIGHT_W + 2 * MOULD_W, SIGHT_H + 2 * MOULD_W
FRAME_BOTTOM, FRAME_TOP = FOOT, FOOT + OUT_H
SIGHT_BOTTOM = FRAME_BOTTOM + MOULD_W
GLASS_CENTER = SIGHT_BOTTOM + SIGHT_H / 2
GLASS_FRONT, GLASS_BACK, BOARD_BACK = 0.0158, 0.010, 0.004
REBATE = 0.008
ORN_BACK = 0.005
EMPTY = -1.0


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _arc(cx, cy, rx, ry, a0, a1, n):
    a = np.radians(np.linspace(a0, a1, n))
    return [(cx + rx * np.cos(x), cy + ry * np.sin(x)) for x in a]


def profile(hp=False):
    """Closed moulding section from the sight edge outward and round the back: list of (s, d, sharp)."""
    pts = []

    def add(seq, sharp_first=False, sharp_last=False):
        for i, (s, d) in enumerate(seq):
            if pts and abs(pts[-1][0] - s) < 1e-7 and abs(pts[-1][1] - d) < 1e-7:
                pts[-1] = (s, d, pts[-1][2] or (sharp_first and i == 0))
                continue
            sharp = (i == 0 and sharp_first) or (i == len(seq) - 1 and sharp_last)
            pts.append((s, d, sharp))

    add([(0.0, GLASS_FRONT), (0.0, 0.026)], True, False)
    add(_arc(0.004, 0.026, 0.004, 0.004, 180, 90, 9))
    add([(0.0105, 0.0305)], False, True)
    if hp:
        add([(0.0105, 0.0292), (0.0179, 0.0292)], True, True)
    else:
        add(_arc(0.0142, 0.0305, 0.0037, 0.0037, 180, 0, 13), True, True)
    add([(0.0179, 0.0305), (0.0185, 0.0335)], True, True)
    add(_arc(0.0185, 0.057, 0.022, 0.0235, 270, 360, 13), True, True)
    for k in range(3):
        add(_arc(0.0418 + 0.0025 * k, 0.0570, 0.00125, 0.00125, 180, 0, 7), True, True)
    add([(0.04805, 0.0575)], True, True)
    add(_arc(0.0612, 0.0575, 0.0128, 0.0125, 180, 0, 21), True, False)
    add([(0.074, 0.054)], False, True)
    drop = 0.0012 if hp else 0.0
    t = np.linspace(0, 1, 15)
    og = [(0.074 + 0.0255 * x, 0.054 - drop * np.sin(np.pi * x) - 0.0255 * (3 * x * x - 2 * x ** 3)) for x in t]
    add(og, True, True)
    add([(0.1010, 0.0285), (0.1010, 0.0270)], True, True)
    add(_arc(0.1048, 0.0270, 0.0038, 0.0038, 180, 0, 11), True, True)
    add([(0.1100, 0.0250), (0.1100, 0.0), (REBATE, 0.0), (REBATE, GLASS_FRONT + 0.0002)], True, True)
    return pts


def front_table(step=0.0001):
    """Front height of the low-poly section as a function of s (max over the section)."""
    pts = np.array([p[:2] for p in profile(False)])
    front = pts[: int(np.argmax(pts[:, 0] >= 0.1100)) + 2]
    s = np.arange(0.0, MOULD_W + step / 2, step)
    d = np.zeros_like(s)
    for (s0, d0), (s1, d1) in zip(front[:-1], front[1:]):
        lo, hi = min(s0, s1), max(s0, s1)
        m = (s >= lo - 1e-9) & (s <= hi + 1e-9)
        if hi - lo < 1e-9:
            d[m] = np.maximum(d[m], max(d0, d1))
        else:
            d[m] = np.maximum(d[m], d0 + (d1 - d0) * (s[m] - s0) / (s1 - s0))
    return s, d


_FRONT = None


def mould_front(u, v):
    """Front height of the moulding at frame-plane points (0 where there is no moulding)."""
    global _FRONT
    if _FRONT is None:
        _FRONT = front_table()
    s = np.maximum(np.abs(u) - SIGHT_W / 2, np.abs(v - GLASS_CENTER) - SIGHT_H / 2)
    d = np.interp(s, _FRONT[0], _FRONT[1])
    return np.where((s >= 0) & (s <= MOULD_W), d, 0.0)


class Field:
    def __init__(self, u0, v0, u1, v1, px):
        self.px = float(px)
        self.u0, self.v0 = float(u0), float(v0)
        self.nx = int(round((u1 - u0) / px)) + 1
        self.ny = int(round((v1 - v0) / px)) + 1
        self.u = self.u0 + np.arange(self.nx) * self.px
        self.v = self.v0 + np.arange(self.ny) * self.px

    def empty(self):
        return np.full((self.ny, self.nx), EMPTY, np.float64)

    def box(self, umin, vmin, umax, vmax):
        i0 = max(0, int(np.floor((umin - self.u0) / self.px)))
        i1 = min(self.nx, int(np.ceil((umax - self.u0) / self.px)) + 1)
        j0 = max(0, int(np.floor((vmin - self.v0) / self.px)))
        j1 = min(self.ny, int(np.ceil((vmax - self.v0) / self.px)) + 1)
        return j0, j1, i0, i1

    def grid(self, j0=0, j1=None, i0=0, i1=None):
        return np.meshgrid(self.u[i0:i1], self.v[j0:j1])


def resample(pts, step):
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    acc = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(np.ceil(acc[-1] / step)) + 1)
    si = np.linspace(0.0, acc[-1], n)
    out = np.stack([np.interp(si, acc, pts[:, 0]), np.interp(si, acc, pts[:, 1])], 1)
    return out, si / max(acc[-1], 1e-12)


def _fn(x, t):
    t = np.asarray(t, float)
    return np.broadcast_to(np.asarray(x(t) if callable(x) else x, float), t.shape).copy()


def bez(p0, p1, p2, p3, n=160):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


def chain(*parts):
    out = [np.asarray(parts[0], float)]
    for p in parts[1:]:
        p = np.asarray(p, float)
        out.append(p[1:] if np.linalg.norm(p[0] - out[-1][-1]) < 1e-9 else p)
    return np.concatenate(out)


def volute(c, r0, r1, a0, turns, cw=True, n=300):
    t = np.linspace(0, 1, n)
    r = r0 * (r1 / r0) ** t
    a = np.radians(a0) + (-1 if cw else 1) * 2 * np.pi * turns * t
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1)


def scroll(p0, d0, c, r0, r1, a0, turns, cw=True, lead=(0.35, 0.35)):
    """A lead-in curve from p0 (heading d0) that runs tangentially into a volute round centre c."""
    v = volute(c, r0, r1, a0, turns, cw)
    q, tq = v[0], v[1] - v[0]
    tq = tq / np.linalg.norm(tq)
    d0 = np.asarray(d0, float) / np.linalg.norm(d0)
    L = np.linalg.norm(q - np.asarray(p0, float))
    return chain(bez(p0, np.asarray(p0) + d0 * L * lead[0], q - tq * L * lead[1], q), v)


def mirror_u(pts):
    p = np.array(pts, float)
    p[:, 0] *= -1
    return p


def tube(F, pts, r, z, k=1.0, out=None, sym=False):
    """Sphere-swept tube: union of z + k*sqrt(r^2 - d^2) along the path (r and z may be functions of t)."""
    out = F.empty() if out is None else out
    for path in ([pts, mirror_u(pts)] if sym else [pts]):
        P, t = resample(path, F.px * 0.4)
        R, Z = _fn(r, t), _fn(z, t)
        for (pu, pv), rr, zz in zip(P, R, Z):
            j0, j1, i0, i1 = F.box(pu - rr, pv - rr, pu + rr, pv + rr)
            if j1 <= j0 or i1 <= i0:
                continue
            d2 = (F.v[j0:j1, None] - pv) ** 2 + (F.u[None, i0:i1] - pu) ** 2
            h = np.where(d2 < rr * rr, zz + k * np.sqrt(np.maximum(rr * rr - d2, 0.0)), EMPTY)
            sub = out[j0:j1, i0:i1]
            np.maximum(sub, h, out=sub)
    return out


def cut_field(F, pts, w, depth, sym=False, sharp=1.4, shape=None):
    """Depth of a tapered V/U groove along a path, on F's grid (or on a sub-window shape at F's origin)."""
    cut = np.zeros((F.ny, F.nx) if shape is None else shape)
    for path in ([pts, mirror_u(pts)] if sym else [pts]):
        P, t = resample(path, F.px * 0.4)
        W, D = _fn(w, t), _fn(depth, t)
        for (pu, pv), ww, dd in zip(P, W, D):
            if ww <= 0 or dd <= 0:
                continue
            j0, j1, i0, i1 = F.box(pu - ww, pv - ww, pu + ww, pv + ww)
            j1, i1 = min(j1, cut.shape[0]), min(i1, cut.shape[1])
            if j1 <= j0 or i1 <= i0:
                continue
            d = np.sqrt((F.v[j0:j1, None] - pv) ** 2 + (F.u[None, i0:i1] - pu) ** 2)
            c = dd * np.clip(1 - d / ww, 0, 1) ** sharp
            sub = cut[j0:j1, i0:i1]
            np.maximum(sub, c, out=sub)
    return cut


def groove(F, h, pts, w, depth, sym=False, sharp=1.4):
    """Carve a groove into h along a path (only where h exists)."""
    cut = cut_field(F, pts, w, depth, sym, sharp)
    live = h > EMPTY
    h[live] -= cut[live]
    return h


def beads(F, centers, r, z, k=1.0, out=None, sym=False):
    out = F.empty() if out is None else out
    cs = np.asarray(centers, float)
    rs = np.broadcast_to(np.asarray(r, float), (len(cs),))
    zs = np.broadcast_to(np.asarray(z, float), (len(cs),))
    if sym:
        cs, rs, zs = np.concatenate([cs, mirror_u(cs)]), np.tile(rs, 2), np.tile(zs, 2)
    for (pu, pv), rr, zz in zip(cs, rs, zs):
        j0, j1, i0, i1 = F.box(pu - rr, pv - rr, pu + rr, pv + rr)
        if j1 <= j0 or i1 <= i0:
            continue
        d2 = (F.v[j0:j1, None] - pv) ** 2 + (F.u[None, i0:i1] - pu) ** 2
        hh = np.where(d2 < rr * rr, zz + k * np.sqrt(np.maximum(rr * rr - d2, 0.0)), EMPTY)
        sub = out[j0:j1, i0:i1]
        np.maximum(sub, hh, out=sub)
    return out


def _spine_map(F, spine, reach):
    """Nearest-point map to a spine curve within reach: distance, curve parameter t and side sign."""
    P, t = resample(spine, F.px * 0.5)
    tg = np.gradient(P, axis=0)
    tg /= np.linalg.norm(tg, axis=1, keepdims=True)
    nrm = np.stack([-tg[:, 1], tg[:, 0]], 1)
    lo, hi = P.min(0) - reach, P.max(0) + reach
    j0, j1, i0, i1 = F.box(lo[0], lo[1], hi[0], hi[1])
    D = np.full((j1 - j0, i1 - i0), np.inf)
    T = np.zeros_like(D)
    S = np.zeros_like(D)
    for (pu, pv), tk, nk in zip(P, t, nrm):
        a0, a1, b0, b1 = F.box(pu - reach, pv - reach, pu + reach, pv + reach)
        du = F.u[None, b0:b1] - pu
        dv = F.v[a0:a1, None] - pv
        d2 = du * du + dv * dv
        sl = (slice(a0 - j0, a1 - j0), slice(b0 - i0, b1 - i0))
        better = d2 < D[sl]
        D[sl] = np.where(better, d2, D[sl])
        T[sl] = np.where(better, tk, T[sl])
        S[sl] = np.where(better, du * nk[0] + dv * nk[1], S[sl])
    return (j0, j1, i0, i1), np.sqrt(D), T, np.sign(S) + (S == 0)


def leaf(F, spine, width, z, height, lobes=5, start=0.14, notch=0.34, teeth=3, tooth=0.12, rib=0.0022, rib_w=0.0035,
         veins=0.0016, eyes=0.7, eye_depth=0.006, curl=0.0, edge=0.0012, out=None, sym=False):
    """Acanthus leaf along a spine: forward-pointing serrated lobes, a raised midrib, veins running into each lobe
    and drilled eyes at the notches. width(t) is the half width, z(t) the base and height(t) the dome."""
    out = F.empty() if out is None else out
    bnd = start + (1 - start) * (np.arange(lobes + 1) / lobes) ** 0.85
    for path in ([spine, mirror_u(spine)] if sym else [spine]):
        tt = np.linspace(0, 1, 400)
        wmax = float(np.max(_fn(width, tt)))
        (j0, j1, i0, i1), dist, T, sgn = _spine_map(F, path, wmax * 1.02)
        W = _fn(width, T.ravel()).reshape(T.shape)
        k = np.clip(np.searchsorted(bnd, T, side='right') - 1, 0, lobes - 1)
        x = np.clip((T - bnd[k]) / np.maximum(bnd[k + 1] - bnd[k], 1e-9), 0, 1)
        rise = np.sin(0.5 * np.pi * np.clip(x / 0.78, 0, 1)) ** 0.8
        fall = np.cos(0.5 * np.pi * np.clip((x - 0.78) / 0.22, 0, 1)) ** 0.7
        lobe = np.where(x < 0.78, rise, fall)
        lobe = np.where(T < start, 1.0, lobe)
        toothm = 1 - tooth * (1 - np.abs(np.cos(np.pi * teeth * np.clip(x / 0.78, 0, 1)))) ** 2 * (x < 0.78)
        Ws = np.maximum(W * (1 - notch * (1 - lobe)) * toothm, 1e-6)
        q = dist / Ws
        inside = q < 1
        Z = _fn(z, T.ravel()).reshape(T.shape)
        H = _fn(height, T.ravel()).reshape(T.shape)
        hh = Z + edge + H * np.clip(1 - q * q, 0, 1) ** 0.6 * (0.86 + 0.14 * lobe)
        hh += rib * np.exp(-(dist / rib_w) ** 2) * (1 - 0.6 * T)
        if curl:
            hh += curl * smoothstep(0.7, 1.0, T) * (1 - q * q)
        Pp, _ = resample(path, F.px * 0.5)
        tg = np.gradient(Pp, axis=0)
        tg /= np.linalg.norm(tg, axis=1, keepdims=True)
        nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
        n = len(Pp) - 1
        local = Field(F.u[i0], F.v[j0], F.u[i1 - 1], F.v[j1 - 1], F.px)
        cut = np.zeros_like(hh)
        wf = lambda tv: float(_fn(width, np.array([tv]))[0])
        for kk in range(lobes):
            a_, b_ = bnd[kk], bnd[kk + 1]
            tb, te = a_ + 0.05 * (b_ - a_), a_ + 0.74 * (b_ - a_)
            ib, ie = int(tb * n), int(te * n)
            we = wf(te) * (1 - notch * (1 - np.sin(0.5 * np.pi * 0.74 / 0.78) ** 0.8))
            for sd in (-1, 1):
                pb = Pp[ib] + nr[ib] * sd * rib_w * 0.6
                pe = Pp[ie] + nr[ie] * sd * we * 0.82
                pm = 0.5 * (pb + pe) + nr[(ib + ie) // 2] * sd * we * 0.15
                if veins > 0:
                    c = cut_field(local, bez(pb, pm, pm, pe, 40), lambda t: rib_w * 0.5 * (1 - 0.6 * t),
                                  lambda t: veins * (1 - 0.6 * t), shape=hh.shape)
                    np.maximum(cut, c, out=cut)
                if eyes and kk > 0:
                    ia = int(min(a_ + 0.02, 1.0) * n)
                    wa = wf(a_) * (1 - notch) * 0.78
                    e0 = Pp[ia] + nr[ia] * sd * wa
                    e1 = Pp[max(ia - 6, 0)] + nr[max(ia - 6, 0)] * sd * wa * 0.9
                    c = cut_field(local, np.stack([e1, e0]), wf(a_) * 0.16 * eyes, eye_depth, sharp=0.6,
                                  shape=hh.shape)
                    np.maximum(cut, c, out=cut)
        hh = np.where(inside, np.maximum(hh - cut, ORN_BACK + 0.004), EMPTY)
        sub = out[j0:j1, i0:i1]
        np.maximum(sub, hh, out=sub)
    return out


def shell(F, apex, R0, theta_max, ribs, z0, dome, rib_h, lip=0.004, squash=0.18, out=None, flat=0.22):
    """Scallop shell fanning upward from apex: flat-topped ribs split by V-grooves, scalloped rim rolled forward."""
    out = F.empty() if out is None else out
    j0, j1, i0, i1 = F.box(apex[0] - R0 * 1.1, apex[1] - 0.01, apex[0] + R0 * 1.1, apex[1] + R0 * 1.1)
    U, V = F.grid(j0, j1, i0, i1)
    du, dv = U - apex[0], V - apex[1]
    rho = np.hypot(du, dv)
    th = np.arctan2(du, dv)
    tm = np.radians(theta_max)
    q = (th + tm) / (2 * tm)
    ph = q * ribs
    f = ph - np.floor(ph)
    c = smoothstep(0, flat, f) * smoothstep(0, flat, 1 - f)
    R = R0 * (1 - squash * (th / tm) ** 2) * (0.94 + 0.06 * np.sin(np.pi * f) ** 0.5)
    x = rho / R
    inside = (np.abs(th) < tm) & (x < 1) & (q > 0) & (q < 1)
    g = 0.30 + 0.70 * np.sin(0.5 * np.pi * np.clip(x / 0.55, 0, 1)) ** 0.8 - 0.12 * smoothstep(0.55, 0.9, x)
    rim = lip * smoothstep(0.82, 0.96, x)
    edge_round = np.sqrt(np.clip((1 - x) / 0.05, 0, 1))
    side_fall = 1 - 0.45 * (th / tm) ** 4
    h = z0 + (dome * g * side_fall + rib_h * smoothstep(0.05, 0.5, x) * (c - 1) + rim) * (0.55 + 0.45 * edge_round)
    h = np.where(inside, h, EMPTY)
    sub = out[j0:j1, i0:i1]
    np.maximum(sub, h, out=sub)
    return out


def fan(F, apex, axis, R0, half, rays, z0, dome, ray_h, lip=0.0, squash=0.2, flat=0.0, tip=0.12, bend=0.0,
        pointed=False, base=0.30, out=None):
    """A fan of rounded rays from apex toward the axis direction (degrees, 90 = up): a shell, or one acanthus lobe."""
    out = F.empty() if out is None else out
    j0, j1, i0, i1 = F.box(apex[0] - R0 * 1.05, apex[1] - R0 * 1.05, apex[0] + R0 * 1.05, apex[1] + R0 * 1.05)
    if j1 <= j0 or i1 <= i0:
        return out
    U, V = F.grid(j0, j1, i0, i1)
    ax = np.radians(axis)
    du, dv = U - apex[0], V - apex[1]
    along = du * np.cos(ax) + dv * np.sin(ax)
    across = -du * np.sin(ax) + dv * np.cos(ax)
    rho = np.hypot(du, dv)
    th = np.arctan2(across, along) + np.radians(bend) * np.clip(rho / R0, 0, 1) ** 1.3
    tm = np.radians(half)
    q = (th + tm) / (2 * tm)
    ph = q * rays
    f = ph - np.floor(ph)
    c = smoothstep(0, flat, f) * smoothstep(0, flat, 1 - f) if flat > 0 else np.sin(np.pi * f) ** 0.6
    prof = (1 - np.abs(2 * f - 1)) ** 0.9 if pointed else np.sin(np.pi * f) ** 0.5
    R = R0 * (1 - squash * (th / tm) ** 2) * (1 - tip + tip * prof)
    x = rho / R
    inside = (np.abs(th) < tm) & (x < 1) & (q > 0) & (q < 1)
    g = base + (1 - base) * np.sin(0.5 * np.pi * np.clip(x / 0.55, 0, 1)) ** 0.8 - 0.12 * smoothstep(0.55, 0.9, x)
    rim = lip * smoothstep(0.80, 0.96, x)
    edge_round = np.sqrt(np.clip((1 - x) / 0.06, 0, 1))
    side_fall = 1 - 0.45 * (th / tm) ** 4
    h = z0 + (dome * g * side_fall + ray_h * smoothstep(0.05, 0.5, x) * (c - 1) + rim) * (0.55 + 0.45 * edge_round)
    h = np.where(inside, h, EMPTY)
    sub = out[j0:j1, i0:i1]
    np.maximum(sub, h, out=sub)
    return out


def acanthus(F, spine, size, z, lobes=4, spread=(55, 40), half=(52, 44), rays=(3, 3), tip_rays=4, rise=0.0012,
             body=0.35, bend=45, out=None, sym=False, sides=(-1, 1), t0=0.03, t1=0.62, pointed=False, stagger=0.5):
    """Acanthus leaf blocked in the way carvers do it: a slim blade with a raised midrib carrying pairs of
    overlapping fan-shaped lobes whose fingers sweep toward the tip, ending in a curled tip lobe.
    size(t) is the lobe length along the spine."""
    out = F.empty() if out is None else out
    for path in ([spine, mirror_u(spine)] if sym else [spine]):
        P, t = resample(path, F.px * 0.5)
        tg = np.gradient(P, axis=0)
        tg /= np.linalg.norm(tg, axis=1, keepdims=True)
        nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
        n = len(P) - 1
        mirrored = path is not spine
        S = _fn(size, t)
        Z = _fn(z, t)
        def blade_w(tt):
            w = body * _fn(size, tt) * np.sin(np.pi * np.clip(tt, 0, 1) ** 0.7) ** 0.5 + 0.0015
            return w * (1 - smoothstep(0.80, 0.92, tt))
        leaf(F, P, blade_w, z, lambda tt: 0.16 * _fn(size, tt), lobes=1, start=1.0, notch=0.0, tooth=0.0,
             rib=0.06 * float(S.mean()), rib_w=0.07 * float(S.mean()), veins=0.0, eyes=0.0, out=out)
        for k in range(lobes):
            f = k / max(lobes - 1, 1)
            tk = t0 + (t1 - t0) * f
            i = int(tk * n)
            ang = np.degrees(np.arctan2(tg[i, 1], tg[i, 0]))
            sp = spread[0] + (spread[1] - spread[0]) * f
            hf = half[0] + (half[1] - half[0]) * f
            rr = int(round(rays[0] + (rays[1] - rays[0]) * f))
            for sd in sides:
                sdd = -sd if mirrored else sd
                i = int(min(tk + (stagger * (t1 - t0) / max(lobes - 1, 1) if sd > 0 else 0.0), 0.95) * n)
                apex = P[i] - nr[i] * sdd * 0.10 * S[i]
                fan(F, apex, ang + sdd * sp, S[i], hf, rr, Z[i] + rise * k, S[i] * 0.28, S[i] * 0.10,
                    lip=S[i] * 0.05, squash=0.18, tip=0.22, bend=-sdd * bend, pointed=pointed, base=0.5, out=out)
        it = int(min(t1 + 0.12, 0.93) * n)
        ang = np.degrees(np.arctan2(tg[it, 1], tg[it, 0]))
        fan(F, P[it], ang, S[it] * 1.1, 46, tip_rays, Z[it] + rise * lobes, S[it] * 0.32, S[it] * 0.10,
            lip=S[it] * 0.06, squash=0.25, tip=0.24, pointed=pointed, base=0.5, out=out)
    return out


def cscroll(c1, r1, a1, t1, cw1, c2, r2, a2, t2, cw2, eye=0.22, lead=(0.4, 0.4)):
    """Double-ended scroll: a volute round c1 unwinding into a curve that winds into a volute round c2.
    Returns the path and the curve parameters where the two volutes end and begin."""
    v1 = volute(c1, r1, r1 * eye, a1, t1, cw1)[::-1]
    v2 = volute(c2, r2, r2 * eye, a2, t2, cw2)
    q1, q2 = v1[-1], v2[0]
    d1 = (v1[-1] - v1[-2]) / np.linalg.norm(v1[-1] - v1[-2])
    d2 = (v2[1] - v2[0]) / np.linalg.norm(v2[1] - v2[0])
    L = np.linalg.norm(q2 - q1)
    path = chain(v1, bez(q1, q1 + d1 * L * lead[0], q2 - d2 * L * lead[1], q2), v2)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    acc = np.concatenate([[0], np.cumsum(seg)]) / seg.sum()
    return path, acc[len(v1) - 1], acc[len(v1) - 1 + 160 - 1]


def union(*hs):
    out = hs[0].copy()
    for h in hs[1:]:
        np.maximum(out, h, out=out)
    return out


def soften(h, passes=1):
    """Masked [1 2 1] blur that never grows or shrinks the outline."""
    live = h > EMPTY
    for _ in range(passes):
        a = np.where(live, h, 0.0)
        w = live.astype(float)
        for ax in (0, 1):
            a = 0.25 * np.roll(a, 1, ax) + 0.5 * a + 0.25 * np.roll(a, -1, ax)
            w = 0.25 * np.roll(w, 1, ax) + 0.5 * w + 0.25 * np.roll(w, -1, ax)
        h = np.where(live, a / np.maximum(w, 1e-9), EMPTY)
    return h


# ---------------------------------------------------------------- ornaments

def taper(rb, ta, tb, e1=0.45, e2=0.5, extra=0.0):
    """Radius along a double scroll: full in the body, thinning into both volute eyes."""
    def r(t):
        t = np.asarray(t, float)
        a = e1 + (1 - e1) * smoothstep(0, ta, t)
        b = 1 - (1 - e2) * smoothstep(tb, 1, t)
        return rb * np.minimum(a, b) + extra
    return r


def crest(px=0.0006):
    """Top-centre crest. Local v = 0 is the frame's outer top edge; returns (Field, h)."""
    F = Field(-0.30, -0.105, 0.30, 0.262, px)
    parts = []
    parts.append(fan(F, (0.0, -0.052), 90, 0.182, 60, 15, 0.044, 0.027, 0.006, lip=0.004, squash=0.25, flat=0.22,
                     tip=0.06))
    parts.append(beads(F, [(0.0, -0.048)], 0.0085, 0.066, 0.8))
    path, ta, tb = cscroll((0.076, 0.106), 0.017, 90, 1.05, False, (0.236, 0.024), 0.027, 90, 1.2, True,
                           lead=(0.45, 0.45))
    sc = tube(F, path, taper(0.0145, ta, tb, 0.42, 0.48), 0.039, 0.78, sym=True)
    body = path[int(len(path) * ta * 0.8):int(len(path) * (tb + (1 - tb) * 0.3))]
    groove(F, sc, body, 0.0050, 0.0024, sym=True)
    parts.append(sc)
    parts.append(beads(F, [(0.076, 0.106), (0.236, 0.024)], [0.0060, 0.0074], [0.041, 0.042], 0.9, sym=True))
    parts.append(acanthus(F, bez((0.228, 0.062), (0.196, 0.094), (0.152, 0.128), (0.100, 0.150), 140),
                          lambda t: 0.050 - 0.020 * t, lambda t: 0.044 + 0.004 * t, sym=True))
    parts.append(acanthus(F, bez((0.216, 0.000), (0.190, 0.001), (0.158, 0.005), (0.124, 0.012), 120),
                          lambda t: 0.032 - 0.010 * t, lambda t: 0.034 + 0.003 * t, lobes=3, sym=True))
    parts.append(acanthus(F, bez((0.0, 0.104), (0.0, 0.146), (0.001, 0.190), (0.0, 0.230), 120),
                          lambda t: 0.044 - 0.018 * t, lambda t: 0.048 + 0.004 * t))
    parts.append(acanthus(F, bez((0.008, 0.104), (0.026, 0.124), (0.046, 0.140), (0.072, 0.156), 120),
                          lambda t: 0.034 - 0.012 * t, lambda t: 0.046 + 0.003 * t, lobes=3, sym=True))
    h = soften(union(*parts), 1)
    return F, h


def corner(px=0.0007):
    """Top-right corner ornament. Local origin at the frame's outer corner; the frame lies at u<0, v<0.
    Symmetric about the diagonal; the other corners are mirror copies."""
    F = Field(-0.175, -0.175, 0.045, 0.045, px)
    parts = []
    parts.append(fan(F, (-0.030, -0.030), 45, 0.060, 80, 11, 0.044, 0.020, 0.005, lip=0.004, squash=0.22,
                     flat=0.22, tip=0.07))
    for flip in (False, True):
        def fl(p):
            p = np.asarray(p, float)
            return p[:, ::-1].copy() if flip else p
        path = fl(scroll((-0.060, -0.040), (-1, 0.02), (-0.128, -0.044), 0.012, 0.0030, 80, 1.05, cw=False,
                         lead=(0.4, 0.5)))
        tb = tube(F, path, lambda t: 0.0075 * (1 - 0.55 * smoothstep(0.45, 1.0, t)) + 0.0012,
                  lambda t: 0.066 - 0.004 * t, 0.8)
        groove(F, tb, path[:int(len(path) * 0.55)], 0.0030, 0.0014)
        parts.append(tb)
        parts.append(beads(F, fl([(-0.128, -0.044)]), 0.0046, 0.066, 0.9))
        lf = fl(bez((-0.034, -0.016), (-0.070, -0.012), (-0.120, -0.010), (-0.172, -0.012), 120))
        parts.append(acanthus(F, lf, lambda t: 0.036 - 0.014 * t, lambda t: 0.050 - 0.006 * t))
    dg = bez((-0.042, -0.042), (-0.066, -0.066), (-0.092, -0.092), (-0.124, -0.124), 120)
    parts.append(acanthus(F, dg, lambda t: 0.036 - 0.014 * t, lambda t: 0.070 - 0.004 * t, lobes=3))
    parts.append(beads(F, [(-0.040, -0.040)], 0.011, 0.068, 0.8))
    h = soften(union(*parts), 1)
    return F, h


def apron(px=0.0006):
    """Bottom-centre apron. Local v = 0 is the frame's outer bottom edge; it hangs a little below."""
    F = Field(-0.135, -0.024, 0.135, 0.085, px)
    parts = []
    parts.append(fan(F, (0.0, 0.060), 270, 0.080, 62, 11, 0.046, 0.020, 0.005, lip=0.0035, squash=0.3, flat=0.22,
                     tip=0.07))
    parts.append(acanthus(F, bez((0.020, 0.058), (0.050, 0.058), (0.085, 0.058), (0.125, 0.058), 100),
                          lambda t: 0.030 - 0.010 * t, lambda t: 0.068 - 0.006 * t, lobes=3, sym=True))
    parts.append(beads(F, [(0.0, 0.060)], 0.009, 0.068, 0.8))
    h = soften(union(*parts), 1)
    return F, h
