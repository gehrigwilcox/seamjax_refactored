"""
Minimap maths (pure numpy, no drawing): a 2-D chart of the rooms from the tiling's own geometry, and the turn that makes the chart
match what the player sees.

  project(L, centre)   chart of all rooms: plane as is; hyperbolic plane in the Poincare disk; sphere in an azimuthal-equidistant chart.
                       Centred on room `centre` (or the middle of the dungeon if None); centred on a room, directions out of it are exact.
  chart(L, centre)     adds spacing / room radii / extent (cached in L)
  orient(...)          angle and mirror flag that turn the chart so the doors of your room point where you see them on screen
  draw(...)            the minimap of the debug overlay, on a canvas (canvas.py)

L is a dict: sig (floor id), m, C (room centre vectors), geo (0 flat 1 hyperbolic 2 sphere 3 wrapped), doors, dseam, ns, flip, ckey.
"""
import math, colorsys
import numpy as np
from seamjax.core.geometry.view import view_np


def project(L, centre):
    """2-D chart of all rooms from the tiling's own geometry, centred on room `centre` (or the middle of the dungeon if None): plane as is;
    hyperbolic plane in the Poincare disk, sphere in an azimuthal-equidistant chart.  Centred on a room, directions out of that room are exact."""
    C = L['C']; geo = L['geo']
    if geo == 1:
        mc = C[centre] if centre is not None else C.sum(0); mc = mc/np.sqrt(max(mc[2]**2-mc[0]**2-mc[1]**2, 1e-9)); phi = math.atan2(mc[1], mc[0]); eta = math.atanh(min(.999999, math.hypot(mc[0], mc[1])/mc[2]))
        cp, sp_ = math.cos(-phi), math.sin(-phi); x = cp*C[:, 0]-sp_*C[:, 1]; y = sp_*C[:, 0]+cp*C[:, 1]; z = C[:, 2]
        x, z = math.cosh(eta)*x-math.sinh(eta)*z, -math.sinh(eta)*x+math.cosh(eta)*z; return np.stack([x, y], 1)/(1+z[:, None])
    if geo == 2:
        mc = C[centre] if centre is not None else C.sum(0)
        if np.linalg.norm(mc) < 1e-6: mc = C[0]
        mc = mc/np.linalg.norm(mc); ax = np.cross(mc, [0., 0., 1.]); sn_ = np.linalg.norm(ax); ang = math.atan2(sn_, mc[2])
        if sn_ <= 1e-9 and mc[2] < 0: C = C*np.array([1., -1., -1.])                 # exactly at the south pole: turn the sphere over
        if sn_ > 1e-9:
            ax /= sn_; K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]); C = C@(np.eye(3)+math.sin(ang)*K+(1-math.cos(ang))*K@K).T
        th = np.arccos(np.clip(C[:, 2], -1, 1)); ph = np.arctan2(C[:, 1], C[:, 0]); return np.stack([th*np.cos(ph), th*np.sin(ph)], 1)
    return C[:, :2]-(C[centre, :2] if centre is not None else C[:, :2].mean(0))

def chart(L, centre):
    key = (L['sig'], centre)
    if L['ckey'] != key:
        P = project(L, centre); m = L['m']
        if centre is None: P = P-(P.min(0)+P.max(0))/2
        dist = np.full(m, 1e9); allv = []
        for i in range(m):
            for k in range(int(L['ns'][i])):
                j = L['doors'][i, k]
                if j >= 0 and L['dseam'][i, k] == 0: dd = float(np.hypot(*(P[i]-P[j]))); dist[i] = min(dist[i], dd); allv.append(dd)
        sp = float(np.median(allv)) if allv else 1.
        if centre is not None:                                                  # zoom so the neighbours of your room are a fixed distance away
            near = [float(np.hypot(*(P[j]-P[centre]))) for k in range(int(L['ns'][centre])) for j in [L['doors'][centre, k]] if j >= 0 and L['dseam'][centre, k] == 0]
            if near: sp = float(np.median(near))
        dist = np.where(dist > 1e8, sp, dist)
        L.update(ckey=key, P=P, sp=sp, rr=dist*.36, ext=max(float(np.hypot(P[:, 0], P[:, 1]).max()), 1e-6))
    return L

def orient(L, P, r, n, doors, sides, va, mir):
    """angle (and whether to mirror) that turns the chart so that the doors of room r (n sides, doors, sides) point where you see them"""
    ds, ms = [], []                                  # ds: outward side normals as you see them (where the neighbour is); ms: the same neighbours in the chart
    for k in range(n):
        j = doors[k]
        if j >= 0 and j < L['m'] and L['dseam'][r, k] == 0 and np.hypot(*(P[j]-P[r])) > 1e-9:
            ds.append(complex(*view_np(va, mir, *sides.normal[k]))); ms.append(complex(*(P[j]-P[r])))
    if not ds: return 0., L['flip']
    ds = np.array(ds)/np.abs(ds); ms = np.array(ms)/np.abs(ms); fits = []
    for flip in (False, True):
        b = np.conj(ms) if flip else ms; th = float(np.angle(np.sum(ds*np.conj(b)))); fits.append((float(np.sum(np.abs(ds-np.exp(1j*th)*b)**2)), th, flip))
    best = min(fits)
    if len(ds) < 2 or abs(fits[0][0]-fits[1][0]) < 1e-3: best = next(f for f in fits if f[2] == L['flip'])      # cannot tell: keep the last answer
    L['flip'] = best[2]; return best[1], best[2]


# ---------------------------------------------------------------- drawing
MODES = ['off', 'whole', 'explored']                 # 0 off, 1 the whole dungeon, 2 only rooms you have seen (and their doors)

def hsl(h, s, l):
    """(r, g, b) 0..255 from hue in degrees, saturation and lightness in 0..1"""
    return tuple(int(round(255*v)) for v in colorsys.hls_to_rgb((h % 360)/360., l, s))

def draw(cv, L, sn, room, mode, follow, show_hunter, VW, VH, small=19):
    """the minimap in the bottom right corner, turned to match your view; room numbers match the door labels"""
    if mode == 0: return
    r = int(sn.room); m = L['m']; box = 112.; cx, cy = VW-box-14, VH-box-14
    ch = chart(L, r if follow else None); P = ch['P']; va = float(sn.view_angle); mir = float(sn.mirror)
    sc = 42./ch['sp'] if follow else (box-16)/ch['ext']                              # pixels per chart unit
    th, flip = orient(L, P, r, room.n, room.doors, room.sides, va, mir)
    Z = (np.conj(P[:, 0]+1j*P[:, 1]) if flip else P[:, 0]+1j*P[:, 1])*np.exp(1j*th)*sc; X, Y = cx+Z.real, cy+Z.imag
    inside = np.hypot(X-cx, Y-cy) < box-8; pt = lambda i: (X[i], Y[i])
    vis = np.asarray(sn.visited)[:m]
    cv.rect(cx-box-4, cy-box-4, 2*box+8, 2*box+8, (8, 10, 14, 170))
    show = np.ones(m, bool) if mode == 1 else vis.copy()
    if mode == 2:
        for i in range(m):
            if vis[i]:
                for k in range(int(L['ns'][i])):
                    if L['doors'][i, k] >= 0: show[L['doors'][i, k]] = True
    show &= inside; nbr = {int(j) for j in L['doors'][r][:int(L['ns'][r])] if j >= 0}
    for i in range(m):                                                               # door links (not drawn across a seam of a wrapped world)
        if not show[i]: continue
        for k in range(int(L['ns'][i])):
            j = L['doors'][i, k]
            if j > i and show[j] and L['dseam'][i, k] == 0: cv.line(pt(i), pt(j), (255, 240, 160) if r in (i, j) else (90, 110, 120), 2 if r in (i, j) else 1)
            elif j >= 0 and L['dseam'][i, k] > 0 and show[j]: cv.disk(*pt(i), 2, (255, 190, 110))       # orange dot: a seam leaves here
    for i in range(m):
        if not show[i]: continue
        rad = max(3., min(16., L['rr'][i]*sc)); x, y = pt(i); nn = int(L['ns'][i]); hue = (i*137.508) % 360
        if vis[i]: cv.ngon(x, y, rad, nn, hsl(hue, .45, .35))
        cv.ngon(x, y, rad, nn, hsl(hue, .5, .6) if vis[i] else (120, 130, 135), 1)
        if i == L['start']: cv.disk(x, y+rad+2, 2, (150, 235, 255))
        if i == L['exit'] and (mode == 1 or vis[i]): cv.ring(x, y, rad+3, (140, 255, 215), 2)
        cv.text(str(i), x, y-5, 13, (255, 255, 255) if i == r else (255, 240, 160) if i in nbr else (190, 200, 205), 'c')
    R = sn.roamers
    for i in np.where(np.asarray(R.alive))[0]:                                       # followers on the way: small orange dots
        rr = int(R.room[i])
        if rr < m and show[rr]: cv.disk(pt(rr)[0]+4, pt(rr)[1]+4, 3, (255, 160, 60))
    hr = int(sn.hunter.room)
    if show_hunter and float(sn.hunter.dead) <= 0 and hr < m and show[hr]: cv.disk(*pt(hr), 5, (255, 70, 90))
    cv.ring(*pt(r), max(4., min(18., L['rr'][r]*sc))+3, (255, 255, 255), 2)
    cv.text('map: %s, %s  (M, C)' % (MODES[mode], 'follows you' if follow else 'fixed'), cx-box-4, cy-box-22, small, (170, 190, 195))
