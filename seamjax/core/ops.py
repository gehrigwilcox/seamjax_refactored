"""
Small array operations that are much faster than the jnp ones for the tiny per-game arrays of this game.

jnp.argsort of a 10-16 element row, vmapped over a batch of games, launches a sort kernel that costs ~1.8 ms per call on a
GTX 1660 for 2048 games. Counting comparisons does the same job in ~0.01 ms. Use argsort_small for arrays of up to a few
dozen elements; keep jnp.argsort for long ones (the work here grows with n squared).
"""
import jax.numpy as jnp


def argsort_small(x):
    """stable argsort of a short 1-D array (same result as jnp.argsort: ties keep their order)"""
    n = x.shape[0]; i = jnp.arange(n)
    before = (x[None, :] < x[:, None]) | ((x[None, :] == x[:, None]) & (i[None, :] < i[:, None]))     # [i, j]: j sorts before i
    rank = jnp.sum(before, axis=1)
    return jnp.zeros(n, jnp.int32).at[rank].set(i.astype(jnp.int32))
