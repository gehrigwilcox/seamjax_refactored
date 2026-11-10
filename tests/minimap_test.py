"""Minimap orientation: the chart is turned so the doors of your room point where you see them on screen.
Centred on your room the match is within the room's own wobble (rooms are not perfect regular polygons, up to ~8 degrees); the fixed whole-dungeon chart is only approximate (checked loosely)."""
import _setup, math
import numpy as np, jax, jax.numpy as jnp
from seamjax.gen import load_specs, gen_dungeon
from seamjax.minimap import chart, orient
SP = load_specs(); CX, CY = 480., 300.; rng = np.random.default_rng(3)
def view(va, mir, x, y): u = mir*x; c, s = math.cos(va), math.sin(va); return c*u-s*y, s*u+c*y
gd = jax.jit(lambda k, f: gen_dungeon(SP, k, f))
def err(L, ch, r, d, va, mir, th, flip):
    P = ch['P']; out = []
    for k in range(int(d['n'][r])):
        j = int(d['doors'][r, k])
        if j < 0 or L['dseam'][r, k] != 0: continue
        S = d['side'][r][k]; a = complex(*view(va, mir, S[2], S[3])); z = complex(*(P[j]-P[r])); z = (np.conj(z) if flip else z)*np.exp(1j*th)
        out.append(abs(math.degrees(np.angle(a/z))))
    return out
cats = {}; worst_follow = 0.; fixed = []
for f in (1, 2, 3, 4, 5, 6, 7, 8, 9, 12):
    for seed in range(3):
        d = jax.tree_util.tree_map(np.asarray, gd(jax.random.PRNGKey(seed*31+f), jnp.int32(f))); m = int(d['n_rooms']); geo = int(d['geo'])
        L = dict(sig=(f, seed), m=m, C=d['mpos'][:m].astype(float), geo=geo, doors=d['doors'][:m], dseam=d['dseam'][:m], ns=d['n'][:m], flip=False, ckey=None)
        for r in range(m):
            va, mir = rng.uniform(-math.pi, math.pi), float(rng.choice([-1., 1.]))
            c = dict(n=int(d['n'][r]), doors=d['doors'][r], side=d['side'][r])
            ch = chart(L, r); th, flip = orient(L, ch['P'], r, c, va, mir, view); e = err(L, ch, r, d, va, mir, th, flip)
            if e: worst_follow = max(worst_follow, max(e)); cats[geo] = max(cats.get(geo, 0.), max(e))
            ch = chart(L, None); th, flip = orient(L, ch['P'], r, c, va, mir, view); e = err(L, ch, r, d, va, mir, th, flip)
            if e: fixed.append(np.mean(e))
print('worst door-direction error when centred on your room: %.2f deg  (by world kind %s)' % (worst_follow, {k: round(v, 2) for k, v in cats.items()}))
print('fixed whole-dungeon chart: mean error %.1f deg, 90th percentile %.1f deg' % (np.mean(fixed), np.percentile(fixed, 90)))
assert worst_follow < 10., 'centred chart not aligned with the view'
assert np.mean(fixed) < 25., 'fixed chart badly misaligned'
print('PASS')
