"""
Fixed-capacity entity pools. A pool is any @pytree whose fields all have the slot axis first and which has a boolean
`alive` field; a free slot is one with alive == False. Bullets, enemies and pickups are all pools.

Capacity never changes (jit needs fixed shapes): spawning into a full pool does nothing.
"""
import jax, jax.numpy as jnp
from seamjax.core import struct


def first_free(pool):
    """index of the first free slot (0 if there is none; check pool.alive at that index)"""
    return jnp.argmin(pool.alive)

def put(pool, i, new):
    """write one entity (`new`: same type, one value per field) into slot i"""
    return jax.tree_util.tree_map(lambda a, v: a.at[i].set(v), pool, new)

def spawn(pool, new, cond=True):
    """put `new` into the first free slot if `cond` holds and there is room"""
    f = first_free(pool)
    return struct.where(cond & ~pool.alive[f], put(pool, f, new), pool)

def fill(pool, new):
    """put the rows of `new` (a pool of the same capacity) into the free slots, in order: the i-th free slot gets row i.
    Occupied slots are kept; rows beyond the number of free slots are dropped."""
    free = ~pool.alive
    slot = jnp.argsort(~free)                                        # free slots first, in slot order
    return jax.tree_util.tree_map(lambda o, v: o.at[slot].set(jnp.where(free[slot], v, o[slot])), pool, new)

def set_row(pools, r, rows):
    """pools with a leading per-room axis: replace room r's pool by `rows`"""
    return jax.tree_util.tree_map(lambda a, v: a.at[r].set(v), pools, rows)
