"""Time each part of a frame, batched on the device. usage: python profile_systems.py [B]"""
import _path, sys, time
import jax, jax.numpy as jnp
from jax import lax
from seamjax.sim.game import Game
from seamjax.sim.step import step, combat
from seamjax.sim.state import no_events
from seamjax.sim.observe import observe
from seamjax.sim.systems import hunter, player, transit, enemies, projectiles, pickups, rooms

B = int(sys.argv[1]) if len(sys.argv) > 1 else 2048; T = 200; dt = 1/60.
G = Game.generated()
S = jax.jit(G.reset_batch)(jax.random.split(jax.random.PRNGKey(1), B), jnp.full(B, 6))
A = jax.random.uniform(jax.random.PRNGKey(3), (B, 4), minval=-1, maxval=1)
ev0 = no_events()

parts = {
    'full step': lambda s, a: step(s, a)[0],
    'hunter.tick': lambda s, a: hunter.tick(s, ev0, dt)[0],
    'player.move': lambda s, a: player.move(s, a, dt)[0],
    'transit (+spawn)': lambda s, a: transit.transit(s, ev0, jnp.int32(0))[0],
    'enemies.spawn': lambda s, a: enemies.spawn(s, s.key, 0),
    'combat (all)': lambda s, a: combat(s, ev0, a, dt)[0],
    'player.shoot': lambda s, a: player.shoot(s, a, dt),
    'player_bullet_hits': lambda s, a: projectiles.player_bullet_hits(s, ev0, dt)[0],
    'enemies.shoot': lambda s, a: enemies.shoot(s, s.key, dt),
    'enemies.move': lambda s, a: enemies.move(s, dt)[0],
    'enemy_bullets': lambda s, a: projectiles.enemy_bullets(s, dt)[0],
    'pickups.drop+take': lambda s, a: pickups.take(pickups.drop(s, s.enemies.alive[s.room], s.key), ev0)[0],
    'rooms': lambda s, a: rooms.stairs(*rooms.open_gates(s, ev0))[0],
}

def timeit(f):
    run = jax.jit(lambda S: lax.scan(lambda S, _: (jax.vmap(f)(S, A), None), S, None, length=T)[0])
    jax.block_until_ready(run(S)); t = time.time(); jax.block_until_ready(run(S)); return (time.time()-t)/T*1e3

obs = jax.jit(lambda S: lax.scan(lambda c, _: (c + jax.vmap(observe)(S).sum(), None), 0., None, length=T)[0])
jax.block_until_ready(obs(S)); t = time.time(); jax.block_until_ready(obs(S)); t_obs = (time.time()-t)/T*1e3

print('B=%d on %s, ms per batched frame:' % (B, jax.devices()[0]))
for name, f in parts.items():
    print('  %-20s %7.3f' % (name, timeit(f)))
print('  %-20s %7.3f' % ('observe', t_obs))
