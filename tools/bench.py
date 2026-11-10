"""Simulation speed: batched steps (with auto-reset and the observation) per second.
usage: python tools/bench.py [batch sizes...]      (default 512; e.g. `python tools/bench.py 512 2048 8192`)"""
import _path, sys
import time, numpy as np, jax, jax.numpy as jnp
from jax import lax
from seamjax.sim.game import Game
from seamjax.sim.observe import observe
from seamjax.backend import describe
print('JAX backend:', describe())
G = Game.generated(); T = 300
obs_f = jax.vmap(observe)

def bench(B):
    state = G.reset_batch(jax.random.split(jax.random.PRNGKey(1), B), jnp.full(B, 6))
    def body(c, t):
        s, k = c; k, ka, kr = jax.random.split(k, 3)
        a = jax.random.uniform(ka, (B, 4), minval=-1, maxval=1)
        s, r, d, _ = G.step_auto_batch(s, a, kr)
        o = obs_f(s)
        return (s, k), (r, d, jnp.isfinite(o).all(), s.player.hp.min(), s.player.x.min(), s.player.x.max())
    run = jax.jit(lambda s, k: lax.scan(body, (s, k), jnp.arange(T)))
    t0 = time.time(); (s, k), out = run(state, jax.random.PRNGKey(2)); jax.block_until_ready(out); t1 = time.time()
    _, out = run(state, jax.random.PRNGKey(2)); jax.block_until_ready(out); t2 = time.time()
    r, d, fin, hpmin, xmin, xmax = [np.array(o) for o in out]
    print(f'B={B:6d}: compile+run {t1-t0:5.1f}s, second run {t2-t1:6.2f}s -> {B*T/(t2-t1):>12,.0f} env-steps/s on {jax.devices()[0]}')
    print(f'          all obs finite {bool(fin.all())}, episodes ended {int(d.sum())}, reward mean/step {float(r.mean()):.2e}, '
          f'x range {float(xmin.min()):.0f}..{float(xmax.max()):.0f}, kills {int(np.array(s.kills).sum())}, min hp {int(hpmin.min())}')

for B in [int(a) for a in sys.argv[1:]] or [512]:
    bench(B)
