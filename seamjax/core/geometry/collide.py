"""
Collision of circles against a room: its polygon walls (Sides, n of them) and its obstacles (Obstacles).

One copy of this maths serves the generator (is a spot free?), the simulation (push the player, enemies and the Hunter
out of walls; kill bullets that leave the room) and anything else that needs it. Everything is in the room's own frame.
"""
import jax, jax.numpy as jnp
from jax import lax
from seamjax.core.config import MAX_SIDES as NS, MAX_OBSTACLES as NO
from seamjax.core.types import OBST_CIRCLE, OBST_RECT


# ---------------------------------------------------------------- walls
def side_dist(sides, x, y):
    """signed distance of (x, y) from every side line (negative inside). [NS]"""
    return (x-sides.mid[:, 0])*sides.normal[:, 0] + (y-sides.mid[:, 1])*sides.normal[:, 1]

def inside_poly(sides, n, x, y, m):
    """(x, y) lies inside the polygon at least m from every side"""
    valid = jnp.arange(NS) < n
    return jnp.all(jnp.where(valid, side_dist(sides, x, y) <= -m, True))

def push_inside(x, y, r, sides, n):
    """move a circle of radius r back inside the polygon (two passes over the sides, for corners)"""
    def body(i, c):
        x, y = c; k = i % NS; mid, nrm = sides.mid[k], sides.normal[k]
        d = (x-mid[0])*nrm[0] + (y-mid[1])*nrm[1]
        hit = (k < n) & (d > -r)
        return jnp.where(hit, x-nrm[0]*(d+r), x), jnp.where(hit, y-nrm[1]*(d+r), y)
    return lax.fori_loop(0, 2*NS, body, (x, y))

def clamp_through_doors(x, y, r, gap, sides, n, doors):
    """like push_inside, but a side with a door lets the circle through where it is within `gap` of the door centre.
    Returns (x, y, side being walked out through or -1)."""
    def body(i, c):
        x, y, ex = c; k = i % NS; mid, nrm, tan, door = sides.mid[k], sides.normal[k], sides.tangent[k], sides.door[k]
        d = (x-mid[0])*nrm[0] + (y-mid[1])*nrm[1]
        active = (k < n) & (d > -r)
        lat = (x-door[0])*tan[0] + (y-door[1])*tan[1]
        chan = (doors[k] >= 0) & (jnp.abs(lat) < gap)
        ex = jnp.where(active & chan & (d > 6), k, ex)
        push = active & ~chan
        return jnp.where(push, x-nrm[0]*(d+r), x), jnp.where(push, y-nrm[1]*(d+r), y), ex
    return lax.fori_loop(0, 2*NS, body, (x, y, jnp.int32(-1)))


# ---------------------------------------------------------------- obstacles
def _push_obs(cx, cy, r, kind, pos, size):
    """push a circle of radius r at (cx, cy) out of one obstacle"""
    # circle obstacle
    dx, dy = cx-pos[0], cy-pos[1]; d = jnp.hypot(dx, dy); m = r+size[0]
    k = jnp.where(d < 1e-4, 0., (m-d)/jnp.maximum(d, 1e-9))
    hc = (kind == OBST_CIRCLE) & (d < m)
    cx_c = jnp.where(hc, jnp.where(d < 1e-4, cx+m, cx+dx*k), cx)
    cy_c = jnp.where(hc, jnp.where(d < 1e-4, cy, cy+dy*k), cy)
    # rect obstacle
    qx, qy = jnp.clip(cx, pos[0], pos[0]+size[0]), jnp.clip(cy, pos[1], pos[1]+size[1])
    rx, ry = cx-qx, cy-qy; d2 = rx*rx+ry*ry
    hr = (kind == OBST_RECT) & (d2 < r*r)
    d_ = jnp.sqrt(jnp.maximum(d2, 1e-12)); kr = (r-d_)/d_
    l, rr, t, b = cx-pos[0], pos[0]+size[0]-cx, cy-pos[1], pos[1]+size[1]-cy
    mn = jnp.minimum(jnp.minimum(l, rr), jnp.minimum(t, b))
    ix = jnp.where(mn == l, pos[0]-r, jnp.where(mn == rr, pos[0]+size[0]+r, cx))
    iy = jnp.where((mn != l) & (mn != rr), jnp.where(mn == t, pos[1]-r, pos[1]+size[1]+r), cy)
    inside = d2 <= 1e-4
    cx_r = jnp.where(hr, jnp.where(inside, ix, cx+rx*kr), cx)
    cy_r = jnp.where(hr, jnp.where(inside, iy, cy+ry*kr), cy)
    isr = kind == OBST_RECT
    return jnp.where(isr, cx_r, cx_c), jnp.where(isr, cy_r, cy_c)

def push_out_obstacles(x, y, r, ob):
    """push a circle of radius r out of every obstacle, one after the other"""
    for i in range(NO):
        x, y = _push_obs(x, y, r, ob.kind[i], ob.pos[i], ob.size[i])
    return x, y

def hits_obstacles(x, y, r, ob):
    """a circle of radius r at (x, y) touches some obstacle"""
    def one(kind, pos, size):
        circ = jnp.hypot(x-pos[0], y-pos[1]) < r+size[0]
        qx, qy = jnp.clip(x, pos[0], pos[0]+size[0]), jnp.clip(y, pos[1], pos[1]+size[1])
        rect = (x-qx)**2+(y-qy)**2 < r*r
        return jnp.where(kind == OBST_CIRCLE, circ, jnp.where(kind == OBST_RECT, rect, False))
    return jnp.any(jax.vmap(one)(ob.kind, ob.pos, ob.size))


# ---------------------------------------------------------------- whole room
def push_out(x, y, r, sides, n, ob):
    """keep a circle of radius r in the room: out of the obstacles, then inside the walls (enemies, the Hunter)"""
    x, y = push_out_obstacles(x, y, r, ob)
    return push_inside(x, y, r, sides, n)

def blocked(x, y, r, sides, n, ob):
    """a circle of radius r at (x, y) has left the room or touches an obstacle (bullets)"""
    return ~inside_poly(sides, n, x, y, 0.) | hits_obstacles(x, y, r, ob)

def free_spot(x, y, r, sides, n, ob):
    """a circle of radius r fits at (x, y): r away from every wall and not touching an obstacle (generator)"""
    return inside_poly(sides, n, x, y, r) & ~hits_obstacles(x, y, r, ob)
