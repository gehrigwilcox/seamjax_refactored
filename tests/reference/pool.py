"""Dungeons exported from the original JS game (tests/reference/export_pool.py), used only to check that the JAX
simulation matches the JS game frame for frame. Not needed to play or train."""
import os, numpy as np, jax, jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))

def load_pool(path=None):
    z = np.load(path or os.path.join(HERE, 'pool.npz'))
    return {k: jnp.asarray(z[k]) for k in z.files}

def to_dungeon(P):
    """the export format (flat arrays, packed sides / obstacles) -> a batched Dungeon"""
    from seamjax.core.types import Dungeon, Sides, Obstacles
    sd, ob = P['side'], P['obs']; N, R = P['n'].shape; z = lambda *s: jnp.zeros(s, jnp.int32)
    return Dungeon(n=P['n'], doors=P['doors'], arrive=P['arr'], flip=P['flip'],
                   sides=Sides(mid=sd[..., 0:2], normal=sd[..., 2:4], tangent=sd[..., 4:6], door=sd[..., 6:8]),
                   obstacles=Obstacles(kind=P['obs_t'], pos=ob[..., 0:2], size=ob[..., 2:4], wall=ob[..., 4] > 0),
                   pad=P['pad'], radius=P['rb'], inradius=P['rin'], area=P['area'], spawn=P['spawn'], spawn_ok=P['spawn_ok'], spawn_sec=P['spawn_sec'],
                   wall_axis=P['wax'], wall_pos=P['wpos'], wall_kind=P['wkind'], door_sec=P['dsec'], depth=P['depth'],
                   map_pos=jnp.zeros((N, R, 3), sd.dtype), door_seam=z(N, R, sd.shape[2]), n_rooms=P['n_rooms'], start=P['start'], startpos=P['startpos'],
                   exit=P['exit'], floor=P['floor'], category=P['geo'], spec=z(N), holonomy=jnp.zeros(N, sd.dtype), dist=P['dist'], node_dist=P['ndist'])

def pool_source(P):
    D = to_dungeon(P); N = D.n.shape[0]
    def src(key, floor=1, idx=None):
        d = jax.random.randint(key, (), 0, N) if idx is None else jnp.int32(idx)
        return D[d]
    return src
