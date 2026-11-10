"""
Obstacles inside a room. A room gets one style at random (if it is big enough): nothing, 2-4 pillars, 1-3 blocks, one big
block, a central pillar, or a wall across the middle with a gap. A layout that would cut the doors apart is dropped.

Each style is a function (keys, shape, margin) -> Obstacles, so a new style is one more function in STYLES.
"""
import jax, jax.numpy as jnp
from seamjax.core import struct
from seamjax.core.config import CX, CY, MAX_OBSTACLES as NO, DOOR_GAP as GAP, ROOM_RC as RC
from seamjax.core.types import Obstacles, OBST_NONE, OBST_CIRCLE, OBST_RECT
from seamjax.core.geometry.collide import inside_poly
from seamjax.gen.shape import uniform
from seamjax.gen.grid import reach_grid

N_STYLE_KEYS = 12


def empty():
    return Obstacles(kind=jnp.full((NO,), OBST_NONE, jnp.int32), pos=jnp.zeros((NO, 2)), size=jnp.zeros((NO, 2)), wall=jnp.zeros(NO, bool))

def put(ob, i, kind, pos, size, on, wall=False):
    """set slot i if `on`"""
    new = ob.replace(kind=ob.kind.at[i].set(kind), pos=ob.pos.at[i].set(jnp.stack(pos)), size=ob.size.at[i].set(jnp.stack(size)),
                     wall=ob.wall.at[i].set(wall))
    return struct.where(on, new, ob)

def drop(ob, gone):
    """remove the obstacles flagged in `gone`"""
    return Obstacles(kind=jnp.where(gone, OBST_NONE, ob.kind), pos=jnp.where(gone[:, None], 0., ob.pos), size=jnp.where(gone[:, None], 0., ob.size),
                     wall=jnp.where(gone, False, ob.wall))

def pick_spot(key, sh, margin, size, K=40):
    """a random spot where something of `size` stays `margin` away from the walls (first of K tries); (x, y, found)"""
    a, b = jax.random.split(key); x = uniform(a, (K,), CX-RC, CX+RC); y = uniform(b, (K,), CY-RC, CY+RC)
    ok = jax.vmap(lambda x, y, s: inside_poly(sh.sides, sh.n, x, y, margin+s))(x, y, jnp.broadcast_to(size, (K,)))
    i = jnp.argmax(ok); return x[i], y[i], jnp.any(ok)


# ---------------------------------------------------------------- styles. ks: this room's style keys; mo: margin to the walls
def none(ks, sh, mo):
    return empty()

def pillars(ks, sh, mo):
    """2-4 round pillars"""
    m = 2+jax.random.randint(ks[0], (), 0, 3); ob = empty()
    for i in range(4):
        rr = uniform(jax.random.fold_in(ks[1], i), (), 16., 30.); x, y, ok = pick_spot(jax.random.fold_in(ks[2], i), sh, mo, rr)
        ob = put(ob, i, OBST_CIRCLE, (x, y), (rr, 0.), ok & (i < m))
    return ob

def blocks(ks, sh, mo):
    """1-3 rectangular blocks"""
    m = 1+jax.random.randint(ks[3], (), 0, 3); ob = empty()
    for i in range(3):
        w = uniform(jax.random.fold_in(ks[4], i), (), 40., 110.); h = uniform(jax.random.fold_in(ks[5], i), (), 30., 80.)
        x, y, ok = pick_spot(jax.random.fold_in(ks[6], i), sh, mo, jnp.maximum(w, h)*.6)
        ob = put(ob, i, OBST_RECT, (x-w/2, y-h/2), (w, h), ok & (i < m))
    return ob

def big_block(ks, sh, mo):
    w = uniform(ks[7], (), 90., 180.); h = uniform(ks[8], (), 60., 120.); x, y, ok = pick_spot(ks[9], sh, mo, jnp.maximum(w, h)*.6)
    return put(empty(), 0, OBST_RECT, (x-w/2, y-h/2), (w, h), ok)

def centre_pillar(ks, sh, mo):
    rr = uniform(ks[10], (), 34., 56.)
    return put(empty(), 0, OBST_CIRCLE, (CX, CY), (rr, 0.), inside_poly(sh.sides, sh.n, CX, CY, mo+rr))

def split_wall(ks, sh, mo):
    """a thin wall through the centre, horizontal or vertical, as long as the room allows, with a gap to walk through"""
    k5 = jax.random.split(ks[11], 3); horiz = jax.random.uniform(k5[0]) < .5
    inside = lambda x, y: inside_poly(sh.sides, sh.n, x, y, mo)
    ls = jnp.arange(40., 171., 10.)
    okl = jax.vmap(lambda l: jnp.where(horiz, inside(CX+l, CY) & inside(CX-l, CY), inside(CX, CY+l) & inside(CX, CY-l)))(ls)
    good = jnp.cumprod(okl.astype(jnp.int32)) > 0
    L = jnp.max(jnp.where(good, ls, 0.)); gc = uniform(k5[1], (), -L+50, L-50); on = L >= 80
    a = lambda h, v: jnp.where(horiz, h, v)
    ob = put(empty(), 0, OBST_RECT, (a(CX-L, CX-8), a(CY-8, CY-L)), (a(L+gc-GAP, 16.), a(16., L+gc-GAP)), on)
    return put(ob, 1, OBST_RECT, (a(CX+gc+GAP, CX-8), a(CY-8, CY+gc+GAP)), (a(L-gc-GAP, 16.), a(16., L-gc-GAP)), on)

STYLES = (none, pillars, blocks, big_block, centre_pillar, split_wall)


def layout(k_style, ks, sh, open_mask):
    """the room's obstacles and the reachability grid that goes with them. ks: N_STYLE_KEYS keys"""
    mo = jnp.minimum(110., sh.inradius*.5)                               # keep obstacles this far from the walls
    style = jnp.where(mo >= 60., jax.random.randint(k_style, (), 0, len(STYLES)), 0)        # small rooms stay empty
    every = [f(ks, sh, mo) for f in STYLES]
    ob = jax.tree_util.tree_map(lambda *a: jnp.stack(a)[style], *every)
    reach, conn = reach_grid(sh.sides, sh.n, ob, open_mask)
    ob = drop(ob, jnp.broadcast_to(~conn, (NO,)))                        # a layout that cuts doors apart is dropped
    reach, _ = reach_grid(sh.sides, sh.n, ob, open_mask)
    return ob, reach
