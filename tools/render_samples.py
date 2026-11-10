"""Save a picture of generated rooms (floor given as argv[1], default 8) with the pure-JAX renderer. usage: python tools/render_samples.py [floor]"""
import _path, sys
import numpy as np, jax, jax.numpy as jnp
from seamjax import Game
from seamjax.rl.render import render, save_png
f = int(sys.argv[1]) if len(sys.argv) > 1 else 8
G = Game.generated(); S = jax.jit(jax.vmap(G.reset))(jax.random.split(jax.random.PRNGKey(5), 16), jnp.full(16, f))
imgs = []
for i in range(4):
    s = jax.tree_util.tree_map(lambda a: a[i], S); imgs.append(np.array(render(s, 200, 320)))
save_png(np.concatenate([np.concatenate(imgs[:2], 1), np.concatenate(imgs[2:], 1)], 0), 'samples.png'); print('wrote samples.png')
