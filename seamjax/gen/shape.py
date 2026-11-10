"""The polygon of a room: a regular n-gon squashed along x and y, with one door position per side (used or not)."""
import jax, jax.numpy as jnp
from seamjax.core.config import TAU, CX, CY, MAX_SIDES as NS, DOOR_GAP, DOOR_INSET, ROOM_RC as RC, ROOM_SQUASH_MIN as SMIN
from seamjax.core.struct import pytree
from seamjax.core.types import Sides


def uniform(key, shape=(), lo=0., hi=1.):
    return lo + jax.random.uniform(key, shape)*(hi-lo)


@pytree
class Shape:
    sides: Sides
    n: object
    inradius: object    # centre to the nearest side
    radius: object      # bounding radius
    area: object


def room_shape(k_sx, k_sy, k_door, n):
    """side 0 is at the top, sides go clockwise; each door sits at a random offset along its side, clear of the corners"""
    sx, sy = uniform(k_sx, (), SMIN, 1.), uniform(k_sy, (), SMIN, 1.)
    k = jnp.arange(NS); valid = k < n; nf = n.astype(jnp.float32)
    aA = TAU*(jnp.mod(k-1, n)+.5)/nf; aB = TAU*(k+.5)/nf                # the two corners of each side
    Ax, Ay = CX+RC*sx*jnp.sin(aA), CY-RC*sy*jnp.cos(aA)
    Bx, By = CX+RC*sx*jnp.sin(aB), CY-RC*sy*jnp.cos(aB)
    ln = jnp.hypot(Bx-Ax, By-Ay)+1e-9; tx, ty = (Bx-Ax)/ln, (By-Ay)/ln
    mx, my = (Ax+Bx)/2, (Ay+By)/2; nx, ny = ty, -tx; half = ln/2
    off = uniform(k_door, (NS,), -1., 1.)*jnp.maximum(0., half-DOOR_GAP-DOOR_INSET)
    dx, dy = mx+tx*off, my+ty*off
    v2 = lambda a, b: jnp.stack([a, b], 1)*valid[:, None]
    side = Sides(mid=v2(mx, my), normal=v2(nx, ny), tangent=v2(tx, ty), door=v2(dx, dy))
    rin = jnp.min(jnp.where(valid, (mx-CX)*nx+(my-CY)*ny, 1e9))
    area = jnp.sum(jnp.where(valid, half*((mx-CX)*nx+(my-CY)*ny), 0.))
    return Shape(sides=side, n=n, inradius=rin, radius=RC*jnp.maximum(sx, sy), area=area)
