"""
Minimap maths (pure numpy, no drawing): a 2-D chart of the rooms from the tiling's own geometry, and the turn that makes the chart
match what the player sees.

  project(L, centre)   chart of all rooms: plane as is; hyperbolic plane in the Poincare disk; sphere in an azimuthal-equidistant chart.
                       Centred on room `centre` (or the middle of the dungeon if None); centred on a room, directions out of it are exact.
  chart(L, centre)     adds spacing / room radii / extent (cached in L)
  orient(...)          angle and mirror flag that turn the chart so the doors of your room point where you see them on screen

L is a dict: sig (floor id), m, C (room centre vectors), geo (0 flat 1 hyperbolic 2 sphere 3 wrapped), doors, dseam, ns, flip, ckey.
"""
import math
import numpy as np


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

def orient(L, P, r, c, va, mir, view):
    """angle (and whether to mirror) that turns the chart so that the doors of your room point where you see them on the screen"""
    ds, ms = [], []                                  # ds: outward side normals as you see them (where the neighbour is); ms: the same neighbours in the chart
    for k in range(c['n']):
        j = c['doors'][k]
        if j >= 0 and j < L['m'] and L['dseam'][r, k] == 0 and np.hypot(*(P[j]-P[r])) > 1e-9:
            ds.append(complex(*view(va, mir, *c['side'].normal[k]))); ms.append(complex(*(P[j]-P[r])))
    if not ds: return 0., L['flip']
    ds = np.array(ds)/np.abs(ds); ms = np.array(ms)/np.abs(ms); fits = []
    for flip in (False, True):
        b = np.conj(ms) if flip else ms; th = float(np.angle(np.sum(ds*np.conj(b)))); fits.append((float(np.sum(np.abs(ds-np.exp(1j*th)*b)**2)), th, flip))
    best = min(fits)
    if len(ds) < 2 or abs(fits[0][0]-fits[1][0]) < 1e-3: best = next(f for f in fits if f[2] == L['flip'])      # cannot tell: keep the last answer
    L['flip'] = best[2]; return best[1], best[2]
