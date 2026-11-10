"""
How you see a room: a rotation by the view angle `va` after an optional mirror `mir` (+1 / -1).

Rooms are stored in their own frame. Walking through a door can turn your view (curvature, seams) and mirror it (a
glide-reflection seam); `through_door` is that whole non-Euclidean step. Controls stay in screen space, so input is
mapped back with `rot_inv`.

`view_vec` / `rot_inv` are JAX; `view_np` is the same map for the viewer, in plain floats (no device calls while drawing).
"""
import math
import jax.numpy as jnp
from seamjax.core.config import TAU


def view_vec(va, mir, x, y):
    """room-frame vector -> screen"""
    u = mir*x; c, s = jnp.cos(va), jnp.sin(va)
    return c*u - s*y, s*u + c*y

def rot_inv(x, y, va, mir):
    """screen direction -> room frame (the inverse of view_vec)"""
    c, s = jnp.cos(va), jnp.sin(va)
    return mir*(x*c + y*s), -x*s + y*c

def view_np(va, mir, x, y):
    """view_vec for the viewer: python / numpy floats"""
    u = mir*x; c, s = math.cos(va), math.sin(va)
    return c*u - s*y, s*u + c*y

def through_door(va, mir, side, n, arrive, n2, flip):
    """your view after walking out through `side` of an n-gon and in through side `arrive` of an n2-gon.
    The door you leave by and the door you enter by point the same way on screen; a flipped door mirrors the view.
    Returns (view angle, mirror)."""
    phi = TAU*side/n
    hx, hy = view_vec(va, mir, jnp.sin(phi), -jnp.cos(phi))
    m2 = jnp.where(flip == 1, -mir, mir); psi = TAU*arrive/n2
    va2 = jnp.mod(jnp.arctan2(-hy, -hx) - jnp.arctan2(-jnp.cos(psi), m2*jnp.sin(psi)), TAU)
    return va2, m2
