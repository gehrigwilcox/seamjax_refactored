"""
Software renderer for Seam in pure JAX: state -> RGB image, per-pixel, vmap/jit friendly.
It draws exactly what the game draws: the room turned/mirrored by the view, polygon walls with door gaps,
obstacles, exit pad, enemies and the Hunter (kept upright), pickups, bullets, the red warning door.
Use it for pixel-observation agents or for looking at what the GPU simulation is doing.
"""
import numpy as np, jax, jax.numpy as jnp
from seamjax.sim.queries import room_obs, hunter_door, hunter_visible
from seamjax.core.geometry.view import view_vec, rot_inv
from seamjax.core.config import CX, CY, VIEW_W as VW, VIEW_H as VH, DOOR_GAP as GAP, PLAYER_R as PR, MAX_SIDES as NS, MAX_PICKUPS as NPK, \
    MAX_PLAYER_BULLETS as NB, MAX_ENEMY_BULLETS as NEB
from seamjax.core.types import OBST_CIRCLE, OBST_RECT, WALL_GATED
from seamjax.sim.systems.enemies import radius, of

def _hsl(h, s, l):
    c = (1-jnp.abs(2*l-1))*s; hp = (h % 360.)/60.; x = c*(1-jnp.abs(hp % 2-1)); m = l-c/2
    z = jnp.zeros_like(h)
    i = jnp.floor(hp).astype(jnp.int32) % 6
    R = jnp.stack([c, x, z, z, x, c])[i]; G = jnp.stack([x, c, c, x, z, z])[i]; B = jnp.stack([z, z, x, c, c, x])[i]
    return jnp.stack([R+m, G+m, B+m])

def render(s, H=100, W=160):
    dun, room = s.dun, s.room
    n = dun.n[room]; sd = dun.sides[room]; doors = dun.doors[room]
    ob = room_obs(s, room)
    va, mir = s.view_angle, s.mirror
    hue = (room*137.508) % 360.
    floor_c = _hsl(hue, .35, .09); floor_hi = _hsl(hue, .5, .14); wall_c = _hsl(hue, .55, .58); obs_c = _hsl(hue, .35, .2)
    # sprites live in screen space (upright), so move their centers there once
    toscr = lambda x, y: tuple(jnp.array(view_vec(va, mir, x-CX, y-CY))+jnp.array([CX, CY]))
    en = s.enemies[room]; pk = s.pickups[room]
    ex_, ey_ = jax.vmap(toscr)(en.x, en.y); er = radius(en); ecol = of('colour', en.kind)
    bx_, by_ = jax.vmap(toscr)(s.pb.x, s.pb.y); ebx, eby = jax.vmap(toscr)(s.eb.x, s.eb.y)
    pkx, pky = jax.vmap(toscr)(pk.x, pk.y); plx, ply = toscr(s.player.x, s.player.y); hx, hy = toscr(s.hunter.x, s.hunter.y)
    padx, pady = toscr(dun.pad[room, 0], dun.pad[room, 1])
    hv = hunter_visible(s)
    warn_side = hunter_door(s)                                                # door the Hunter will come through
    dsec2 = (dun.door_sec[room] > 0) & (dun.wall_kind[room] > 0)              # doors on the second side of an inner wall are orange
    sc = jnp.array([VW/W, VH/H]); ks = jnp.arange(NS)
    def pixel(yx):
        px, py = (yx[1]+.5)*sc[0], (yx[0]+.5)*sc[1]
        lx, ly = rot_inv(px-CX, py-CY, va, mir); lx, ly = lx+CX, ly+CY          # screen -> room frame
        d = (lx-sd.mid[:, 0])*sd.normal[:, 0] + (ly-sd.mid[:, 1])*sd.normal[:, 1]
        valid = ks < n; dm = jnp.where(valid, d, -1e9); dmax = jnp.max(dm); kmax = jnp.argmax(dm)
        lat = (lx-sd.door[:, 0])*sd.tangent[:, 0] + (ly-sd.door[:, 1])*sd.tangent[:, 1]
        gap = (doors >= 0) & (jnp.abs(lat) < GAP)
        inside = dmax < 0
        band = (dmax > -3) & (dmax < 3)
        wall = band & ~gap[kmax]
        doorline = band & gap[kmax]
        col = jnp.where(inside, floor_c + .5*(floor_hi-floor_c)*(((lx//44).astype(jnp.int32)+(ly//44).astype(jnp.int32)) % 2 == 0), 0.)
        col = jnp.where(wall, wall_c, col)
        col = jnp.where(doorline, jnp.where(kmax == warn_side, jnp.array([1., .2, .3]), jnp.where(dsec2[kmax], jnp.array([1., .75, .43]), jnp.array([.6, .85, 1.]))), col)
        # obstacles
        def one(kind, p, z):
            circ = jnp.hypot(lx-p[0], ly-p[1]) < z[0]
            rect = (lx > p[0]) & (lx < p[0]+z[0]) & (ly > p[1]) & (ly < p[1]+z[1])
            return jnp.where(kind == OBST_CIRCLE, circ, jnp.where(kind == OBST_RECT, rect, False))
        hit = jax.vmap(one)(ob.kind, ob.pos, ob.size); col = jnp.where(jnp.any(hit & ~ob.wall), obs_c + .25, col)
        wallhit = jnp.any(hit & ob.wall)                                      # inner wall: pink if solid, blue if it can open
        col = jnp.where(wallhit, jnp.where(dun.wall_kind[room] == WALL_GATED, jnp.array([.35, .6, 1.]), jnp.array([1., .4, .62])), col)
        # exit pad
        rpad = jnp.hypot(px-padx, py-pady); isx = room == dun.exit
        cleared = ~jnp.any(en.alive)
        padc = jnp.where(cleared, jnp.array([.55, 1., .85]), jnp.array([.55, .55, .55]))
        ring = isx & (jnp.abs(rpad-18) < 2.5); col = jnp.where(ring, padc, col)
        # sprites (screen space)
        circ = lambda cx, cy, r: jnp.hypot(px-cx, py-cy) < r
        e_hit = jax.vmap(circ)(ex_, ey_, er) & en.alive
        col = jnp.where(jnp.any(e_hit), ecol[jnp.argmax(e_hit)], col)                     # an enemy in its kind's colour
        col = jnp.where(jnp.any(jax.vmap(circ)(pkx, pky, jnp.full(NPK, 7.)) & pk.alive), jnp.array([1., .36, .45]), col)
        col = jnp.where(jnp.any(jax.vmap(circ)(bx_, by_, jnp.full(NB, 4.)) & s.pb.alive), jnp.array([1., .89, .48]), col)
        col = jnp.where(jnp.any(jax.vmap(circ)(ebx, eby, jnp.full(NEB, 5.)) & s.eb.alive), jnp.array([1., .54, .82]), col)
        col = jnp.where(hv & circ(hx, hy, 24.), jnp.array([.64, .07, .17]), col)
        blink = (s.player.inv > 0) & (jnp.floor(s.t*20) % 2 == 1)
        col = jnp.where(circ(plx, ply, PR) & ~blink, jnp.array([.91, .96, 1.]), col)
        return jnp.clip(col, 0., 1.)
    ys, xs = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
    return jax.vmap(jax.vmap(pixel))(jnp.stack([ys, xs], -1))

def render_batch(S, H=100, W=160): return jax.vmap(lambda s: render(s, H, W))(S)

def save_png(img, path):
    from PIL import Image
    Image.fromarray((np.array(img)*255).astype(np.uint8)).save(path)
