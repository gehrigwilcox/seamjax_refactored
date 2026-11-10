"""
Difficulty as one smooth function of the floor. Everything that gets harder with depth reads it from here,
so the curve can be tuned (or replaced) in one place. All values are jnp scalars, safe inside jit/vmap.

floor 1   flat rooms only, small dungeon, no inner walls, few and mostly melee enemies, a late Hunter
floor 2-3 wrapped (seamed) worlds appear, then curved ones
floor 6+  full mix of geometries, full-size dungeons, inner walls in ~30% of rooms
Enemy count, enemy speed and Hunter speed keep climbing with the floor until FLOOR_CAP, then stay put.
"""
import numpy as np
import jax.numpy as jnp

FLOOR_CAP = 24                        # difficulty stops growing here; the run itself is endless

# tiling category mix (flat, hyperbolic, spherical, wrapped grid) at anchor floors; linear in between
_AF = np.array([1., 2., 3., 5.], np.float32)
_CAT = np.array([[1., 0., 0., 0.],
                  [.5, 0., 0., .5],
                  [.25, .2, .2, .35],
                  [0., .4225, .2275, .35]], np.float32)

def difficulty(floor):
    f = jnp.minimum(jnp.asarray(floor, jnp.float32), FLOOR_CAP)
    cat_p = jnp.stack([jnp.interp(f, _AF, _CAT[:, c]) for c in range(4)])
    return dict(
        f=f,
        cat_p=cat_p/cat_p.sum(),
        size=jnp.clip(.4+.12*(f-1), .4, 1.),                       # share of the full dungeon size
        wall_p=jnp.clip(.06*(f-1), 0., .3),                        # chance a room gets an inner wall
        shooter_p=jnp.clip(.12+.03*(f-1), .12, .45),               # share of ranged enemies
        drop_p=jnp.clip(.14+.008*f, .14, .24),                     # health drop rate rises a little to offset the pressure
        hunter_first=jnp.clip(26.-3.*f, 10., 22.),                 # seconds before the Hunter first sets off
    )
