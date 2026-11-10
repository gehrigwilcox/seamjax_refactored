"""Running many games at once with bounded memory: chunked vmap, and rebuilding only the games that need it."""
import jax, jax.numpy as jnp
from jax import lax
from seamjax.core.config import GEN_CHUNK


def cvmap(f, chunk=GEN_CHUNK):
    """vmap(f) that runs `chunk` rows at a time (lax.map over chunks): same result, bounded scratch memory"""
    def g(*args):
        B = args[0].shape[0]; nc = -(-B//chunk); pad = nc*chunk-B
        pd = lambda a: jnp.concatenate([a, jnp.repeat(a[:1], pad, 0)], 0) if pad else a
        args = jax.tree_util.tree_map(lambda a: pd(a).reshape((nc, chunk)+a.shape[1:]), args)
        out = lax.map(lambda xs: jax.vmap(f)(*xs), args)
        return jax.tree_util.tree_map(lambda a: a.reshape((nc*chunk,)+a.shape[2:])[:B], out)
    return g

def regen_where(S, need, fn, chunk=GEN_CHUNK):
    """replace the rows of the batched state S flagged in `need` by fn(rows) (fn is vmapped over `chunk` rows at a time).
    Only the flagged rows are rebuilt, in chunks, so cost and memory follow how many games finished, not the batch size."""
    B = need.shape[0]
    def body(c):
        S, need = c
        idx = jnp.argsort(~need)[:chunk] if B >= chunk else jnp.argsort(~need)           # flagged rows first
        valid = need[idx]
        rows = jax.tree_util.tree_map(lambda a: a[idx], S)
        new = jax.vmap(fn)(rows, idx)
        def put(a, b):
            m = valid.reshape((-1,)+(1,)*(a.ndim-1)); return a.at[idx].set(jnp.where(m, b, a[idx]))
        return jax.tree_util.tree_map(put, S, new), need.at[idx].set(False)
    S, _ = lax.while_loop(lambda c: jnp.any(c[1]), body, (S, need))
    return S
