"""
Software renderer for Seam in pure JAX: state -> RGB image, per-pixel, vmap/jit friendly.
It draws exactly what the game draws: the room turned/mirrored by the view, polygon walls with door gaps,
obstacles, exit pad, enemies and the Hunter (kept upright), pickups, bullets, the red warning door.
Use it for pixel-observation agents or for looking at what the GPU simulation is doing.
"""
import numpy as np, jax, jax.numpy as jnp
from seamjax.env import *

def _hsl(h, s, l):
    c = (1-jnp.abs(2*l-1))*s; hp = (h % 360.)/60.; x = c*(1-jnp.abs(hp % 2-1)); m = l-c/2
    z = jnp.zeros_like(h)
    r, g, b = [(c, x, z), (x, c, z), (z, c, x), (z, x, c), (x, z, c), (c, z, x)][0]  # placeholder, replaced below
    i = jnp.floor(hp).astype(jnp.int32) % 6
    R = jnp.stack([c, x, z, z, x, c])[i]; G = jnp.stack([x, c, c, x, z, z])[i]; B = jnp.stack([z, z, x, c, c, x])[i]
    return jnp.stack([R+m, G+m, B+m])

def render(s, H=100, W=160):
    dun, room = s['dun'], s['room']
    n = dun['n'][room]; side = dun['side'][room]; doors = dun['doors'][room]
    obs, ot = room_obs(s, room)
    va, mir = s['va'], s['mir']
    hue = (room*137.508) % 360.
    floor_c = _hsl(hue, .35, .09); floor_hi = _hsl(hue, .5, .14); wall_c = _hsl(hue, .55, .58); obs_c = _hsl(hue, .35, .2)
    obs_edge = _hsl(hue, .5, .5)
    # sprites live in screen space (upright), so move their centers there once
    toscr = lambda x, y: tuple(jnp.array(view_vec(va, mir, x-CX, y-CY))+jnp.array([CX, CY]))
    en = s['en'][room]; pk = s['pk'][room]
    ex_, ey_ = jax.vmap(toscr)(en[:, 0], en[:, 1]); er = jnp.where(en[:, 3] > .5, 14., 12.)
    bx_, by_ = jax.vmap(toscr)(s['pb'][:, 0], s['pb'][:, 1]); ebx, eby = jax.vmap(toscr)(s['eb'][:, 0], s['eb'][:, 1])
    pkx, pky = jax.vmap(toscr)(pk[:, 0], pk[:, 1]); plx, ply = toscr(s['x'], s['y']); hx, hy = toscr(s['hx'], s['hy'])
    padx, pady = toscr(dun['pad'][room, 0], dun['pad'][room, 1])
    hv = (s['hroom'] == room) & s['hhere'] & (s['hdead'] <= 0)
    # door the Hunter will come through
    hr = s['hroom']; hd = dun['doors'][hr]; arrk = dun['arr'][hr]
    pn = hnode(dun, room, sec_point(dun, room, s['x'], s['y'])); hn = hnode(dun, hr, s['hsec'])
    okd = (hd == room) & (hnode(dun, hr, dun['dsec'][hr]) == hn) & (hnode(dun, room, dun['dsec'][room, arrk]) == pn)
    kk = jnp.argmax(okd)
    warn_side = jnp.where((hn != pn) & (s['hdead'] <= 0) & jnp.any(okd), arrk[kk], -1)
    dsec2 = (dun['dsec'][room] > 0) & (dun['wkind'][room] > 0)               # doors on the second side of an inner wall are orange
    sc = jnp.array([VW/W, VH/H]); ks = jnp.arange(NS)
    def pixel(yx):
        px, py = (yx[1]+.5)*sc[0], (yx[0]+.5)*sc[1]
        lx, ly = rot_inv(px-CX, py-CY, va, mir); lx, ly = lx+CX, ly+CY          # screen -> room frame
        d = (lx-side[:, 0])*side[:, 2] + (ly-side[:, 1])*side[:, 3]
        valid = ks < n; dm = jnp.where(valid, d, -1e9); dmax = jnp.max(dm); kmax = jnp.argmax(dm)
        lat = (lx-side[:, 6])*side[:, 4] + (ly-side[:, 7])*side[:, 5]
        gap = (doors >= 0) & (jnp.abs(lat) < GAP)
        inside = dmax < 0
        band = (dmax > -3) & (dmax < 3)
        wall = band & ~gap[kmax]
        doorline = band & gap[kmax]
        col = jnp.where(inside, floor_c + .5*(floor_hi-floor_c)*(((lx//44).astype(jnp.int32)+(ly//44).astype(jnp.int32)) % 2 == 0), 0.)
        col = jnp.where(wall, wall_c, col)
        col = jnp.where(doorline, jnp.where(kmax == warn_side, jnp.array([1., .2, .3]), jnp.where(dsec2[kmax], jnp.array([1., .75, .43]), jnp.array([.6, .85, 1.]))), col)
        # obstacles
        def ob(o, t):
            circ = jnp.hypot(lx-o[0], ly-o[1]) < o[2]
            rect = (lx > o[0]) & (lx < o[0]+o[2]) & (ly > o[1]) & (ly < o[1]+o[3])
            return jnp.where(t == 0, circ, jnp.where(t == 1, rect, False))
        hit = jax.vmap(ob)(obs, ot); col = jnp.where(jnp.any(hit & (obs[:, 4] <= 0)), obs_c + .25, col)
        wallhit = jnp.any(hit & (obs[:, 4] > 0))                              # inner wall: pink if solid, blue if it can open
        col = jnp.where(wallhit, jnp.where(dun['wkind'][room] == 2, jnp.array([.35, .6, 1.]), jnp.array([1., .4, .62])), col)
        # exit pad
        rpad = jnp.hypot(px-padx, py-pady); isx = room == dun['exit']
        cleared = ~jnp.any(en[:, 7] > .5)
        padc = jnp.where(cleared, jnp.array([.55, 1., .85]), jnp.array([.55, .55, .55]))
        ring = isx & (jnp.abs(rpad-18) < 2.5); col = jnp.where(ring, padc, col)
        # sprites (screen space)
        circ = lambda cx, cy, r: jnp.hypot(px-cx, py-cy) < r
        e_hit = jax.vmap(circ)(ex_, ey_, er) & (en[:, 7] > .5)
        col = jnp.where(jnp.any(e_hit & (en[:, 3] < .5)), jnp.array([1., .48, .27]), col)
        col = jnp.where(jnp.any(e_hit & (en[:, 3] > .5)), jnp.array([.71, .49, 1.]), col)
        col = jnp.where(jnp.any(jax.vmap(circ)(pkx, pky, jnp.full(NPK, 7.)) & (pk[:, 2] > .5)), jnp.array([1., .36, .45]), col)
        col = jnp.where(jnp.any(jax.vmap(circ)(bx_, by_, jnp.full(NB, 4.)) & (s['pb'][:, 5] > .5)), jnp.array([1., .89, .48]), col)
        col = jnp.where(jnp.any(jax.vmap(circ)(ebx, eby, jnp.full(NEB, 5.)) & (s['eb'][:, 5] > .5)), jnp.array([1., .54, .82]), col)
        col = jnp.where(hv & circ(hx, hy, 24.), jnp.array([.64, .07, .17]), col)
        blink = (s['inv'] > 0) & (jnp.floor(s['t']*20) % 2 == 1)
        col = jnp.where(circ(plx, ply, PR) & ~blink, jnp.array([.91, .96, 1.]), col)
        return jnp.clip(col, 0., 1.)
    ys, xs = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
    return jax.vmap(jax.vmap(pixel))(jnp.stack([ys, xs], -1))

def render_batch(S, H=100, W=160): return jax.vmap(lambda s: render(s, H, W))(S)

def save_png(img, path):
    from PIL import Image
    Image.fromarray((np.array(img)*255).astype(np.uint8)).save(path)
