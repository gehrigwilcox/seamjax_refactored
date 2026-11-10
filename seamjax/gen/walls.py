"""
The inner wall of a room: a straight wall right across the room that splits it into two sections, each holding some of
the doors. It is kept only if both sections are walkable from their own doors and not from each other. Solid walls never
open; gated ones open when your side is cleared (dungeon-level gating may turn solid walls into gated ones, see layout.py).
"""
import numpy as np, jax, jax.numpy as jnp
from seamjax.core.config import CX, CY, MAX_OBSTACLES as NO, ROOM_RC as RC, WALL_GATED_P, WALL_THICK, WALL_CLEAR, WALL_MINLEN
from seamjax.core.struct import pytree
from seamjax.core.types import OBST_CIRCLE, OBST_RECT, AXIS_NONE, AXIS_H, AXIS_V, WALL_NONE, WALL_SOLID, WALL_GATED
from seamjax.core.geometry.collide import inside_poly
from seamjax.gen.grid import free_grid, flood, door_entries
from seamjax.gen.obstacles import drop

OFFSETS = np.arange(-110., 111., 10., dtype=np.float32)             # candidate wall positions around the centre (numpy: no JAX at import)


@pytree
class Wall:
    axis: object        # AXIS_*
    pos: object
    kind: object        # WALL_*
    door_sec: object    # [NS] section of each door


def _crowds(ob, row_pos, row_size):
    """obstacles within 40 px of the wall rectangle"""
    rx, ry, rw, rh = row_pos[0], row_pos[1], row_size[0], row_size[1]
    def one(kind, p, z):
        qx, qy = jnp.clip(p[0], rx, rx+rw), jnp.clip(p[1], ry, ry+rh)
        circ = jnp.hypot(p[0]-qx, p[1]-qy) < z[0]+40.
        rect = (p[0] < rx+rw+40) & (p[0]+z[0] > rx-40) & (p[1] < ry+rh+40) & (p[1]+z[1] > ry-40)
        return jnp.where(kind == OBST_CIRCLE, circ, jnp.where(kind == OBST_RECT, rect, False))
    return jax.vmap(one)(ob.kind, ob.pos, ob.size)


def inner_wall(key, sh, ob, reach, open_mask, want):
    """try to add an inner wall (if `want`). Returns (obstacles, reach, reach of the first door's section, Wall)"""
    kw = jax.random.split(jax.random.fold_in(key, 55), 3)
    px_d, py_d = door_entries(sh.sides)
    Pw = jnp.stack([CY+OFFSETS, CX+OFFSETS])                         # [2 axes (h, v), 23] candidate positions
    cdoor = jnp.stack([py_d, px_d])                                  # door coordinate across each axis
    na = jnp.sum(open_mask)
    A_ = jnp.sum(open_mask[None, None, :] & (cdoor[:, None, :] < Pw[:, :, None]-WALL_CLEAR), -1)
    B_ = jnp.sum(open_mask[None, None, :] & (cdoor[:, None, :] > Pw[:, :, None]+WALL_CLEAR), -1)
    okc = (A_ >= 1) & (B_ >= 1) & (A_+B_ == na)                      # doors on both sides, none too close to the wall
    ci = jnp.argmax(jnp.where(okc, jax.random.gumbel(kw[0], okc.shape), -1e9).reshape(-1)); wa, wo = ci//OFFSETS.shape[0], ci % OFFSETS.shape[0]
    P = Pw[wa, wo]; horiz = wa == 0
    # its extent: as far as the polygon goes along that line
    tt = jnp.arange(-RC*1.1, RC*1.1, 3.)
    wx = jnp.where(horiz, CX+tt, P); wy = jnp.where(horiz, P, CY+tt)
    insd = jax.vmap(lambda x, y: inside_poly(sh.sides, sh.n, x, y, 2.))(wx, wy); along = jnp.where(horiz, wx, wy)
    lo = jnp.min(jnp.where(insd, along, 1e9)); hi = jnp.max(jnp.where(insd, along, -1e9))
    gated = jax.random.uniform(kw[1]) < WALL_GATED_P
    pos = jnp.where(horiz, jnp.stack([lo-3, P-WALL_THICK/2]), jnp.stack([P-WALL_THICK/2, lo-3]))
    size = jnp.where(horiz, jnp.stack([hi-lo+6, WALL_THICK]), jnp.stack([WALL_THICK, hi-lo+6]))
    # drop obstacles that crowd the wall, then put the wall in the last slot
    obw = drop(ob, _crowds(ob, pos, size))
    obw = obw.replace(kind=obw.kind.at[NO-1].set(OBST_RECT), pos=obw.pos.at[NO-1].set(pos), size=obw.size.at[NO-1].set(size), wall=obw.wall.at[NO-1].set(True))
    dsec = jnp.where(open_mask, (cdoor[wa] >= P).astype(jnp.int32), 0)
    freew = free_grid(sh.sides, sh.n, obw)
    r0, c0 = flood(freew, sh.sides, open_mask & (dsec == 0)); r1, c1 = flood(freew, sh.sides, open_mask & (dsec == 1))
    ok = want & jnp.any(okc) & (hi-lo >= WALL_MINLEN) & c0 & c1 & ~jnp.any(r0 & r1)
    ob = jax.tree_util.tree_map(lambda a, b: jnp.where(ok, a, b), obw, ob)
    dsec = jnp.where(ok, dsec, 0)
    k0 = jnp.argmax(open_mask)
    reach_first = jnp.where(ok, jnp.where(dsec[k0] == 0, r0, r1), reach)
    reach = jnp.where(ok, r0 | r1, reach)
    wall = Wall(axis=jnp.where(ok, jnp.where(horiz, AXIS_H, AXIS_V), AXIS_NONE).astype(jnp.int32), pos=jnp.where(ok, P, 0.),
                kind=jnp.where(ok, jnp.where(gated, WALL_GATED, WALL_SOLID), WALL_NONE).astype(jnp.int32), door_sec=dsec)
    return ob, reach, reach_first, wall
