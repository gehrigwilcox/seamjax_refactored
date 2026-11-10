"""Dungeons exported from the original JS game (tests/reference/export_pool.py), used only to check that the JAX
simulation matches the JS game frame for frame. Not needed to play or train."""
import os, numpy as np, jax, jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))

def load_pool(path=None):
    z = np.load(path or os.path.join(HERE, 'pool.npz'))
    return {k: jnp.asarray(z[k]) for k in z.files}

def pool_source(P):
    N = P['n'].shape[0]
    def src(key, floor=1, idx=None):
        d = jax.random.randint(key, (), 0, N) if idx is None else jnp.int32(idx)
        return {k: v[d] for k, v in P.items()}
    return src
