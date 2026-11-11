"""
Fixed-capacity entity pools. A pool is any @pytree whose fields all have the slot axis first and which has a boolean
`alive` field; a free slot is one with alive == False. Bullets, enemies and pickups are all pools.

Capacity never changes (jit needs fixed shapes): spawning into a full pool does nothing.
"""
import jax, jax.numpy as jnp
from seamjax.core import struct
from seamjax.core.ops import argsort_small


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
    slot = argsort_small(~free)                                      # free slots first, in slot order
    return jax.tree_util.tree_map(lambda o, v: o.at[slot].set(jnp.where(free[slot], v, o[slot])), pool, new)

def set_row(pools, r, rows):
    """pools with a leading per-room axis: replace room r's pool by `rows`"""
    return jax.tree_util.tree_map(lambda a, v: a.at[r].set(v), pools, rows)


def place(pool, rows):
    """put the alive rows of `rows` (a pool-shaped batch of any length) into the free slots of `pool`, in order: exactly what
    spawning them one after another would do (the k-th alive row goes into the k-th free slot; rows beyond the free slots are
    dropped), in one step instead of a loop. Returns (pool, which rows were placed)"""
    n = pool.alive.shape[0]; m = rows.alive.shape[0]; k = min(n, m)
    ro = argsort_small(~rows.alive)                                   # alive rows first, in their order
    srt = jax.tree_util.tree_map(lambda a: a[ro], rows)
    sl = argsort_small(pool.alive)[:k]                                # free slots first, in slot order
    ok = ~pool.alive[sl] & srt.alive[:k]
    out = jax.tree_util.tree_map(lambda o, v: o.at[sl].set(jnp.where(ok.reshape((-1,)+(1,)*(o.ndim-1)), v[:k], o[sl])), pool, srt)
    placed = jnp.zeros(m, bool).at[ro[:k]].set(ok)
    return out, placed
