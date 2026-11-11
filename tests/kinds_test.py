import _setup
"""Enemy kinds are data: adding a row to core/kinds.ENEMY_KINDS adds a kind of enemy, with nothing else to change.
Here a third kind (a big, slow, tough brute that makes up a third of the spawns) is added, and the simulation spawns it with
its own hp and radius, keeps the other kinds' shares, moves it, and it shows up in the observation as not ranged."""
import numpy as np, jax, jax.numpy as jnp
from seamjax.core import kinds
BRUTE = kinds.EnemyKind('brute', radius=20., hp=8., behaviour=kinds.CHASE, speed=40., share=(1/3, 0., 1/3, 1/3))
kinds.ENEMY_KINDS = kinds.ENEMY_KINDS + (BRUTE,)                       # before anything is traced
from seamjax.difficulty import difficulty
from seamjax.sim.game import Game
from seamjax.sim.state import no_events
from seamjax.sim.systems.transit import transit
from seamjax.sim.systems.enemies import radius
from seamjax.sim.observe import observe

mix = np.array(difficulty(8)['enemy_mix']); print('enemy mix on floor 8 (chaser, shooter, brute):', mix.round(3))
assert mix.shape == (3,) and abs(mix.sum()-1) < 1e-6 and abs(mix[2]-1/3) < 1e-6 and abs(mix[1]-float(np.clip(.12+.03*7, .12, .45))) < 1e-6

G = Game.generated(); B = 64
S = jax.jit(jax.vmap(G.reset))(jax.random.split(jax.random.PRNGKey(5), B), jnp.full(B, 8))

@jax.jit
@jax.vmap
def enter(s):
    """walk into the first neighbour of the start room (its section spawns)"""
    k = jnp.argmax(s.dun.doors[s.room] >= 0)
    s2, _ = transit(s.replace(spawned=jnp.zeros_like(s.spawned)), no_events(), k)
    return s2, s2.enemies[s2.room], radius(s2.enemies[s2.room]), observe(s2)
S2, en, rad, obs = enter(S)
alive = np.array(en.alive); kind = np.array(en.kind)[alive]; hp = np.array(en.hp)[alive]; r = np.array(rad)[alive]
counts = np.bincount(kind, minlength=3); print('spawned by kind:', counts)
assert counts[2] > 0 and counts[0] > 0, 'every kind spawns'
table = kinds.table()
assert (hp == table.hp[kind]).all() and (r == table.radius[kind]).all(), 'hp and radius come from the table'
assert abs(counts[2]/counts.sum() - 1/3) < .12, 'the brute share is about a third'

# the brute moves (CHASE) and is solid; and in the observation it counts as not ranged
@jax.jit
def run(S):
    def body(S, _):
        S = S.replace(player=S.player.replace(inv=jnp.full_like(S.player.inv, 9.)))
        return jax.vmap(lambda s: G.step(s, jnp.zeros(4))[0])(S), None
    return jax.lax.scan(body, S, None, length=120)[0]
S3 = run(S2)
moved = np.array(S3.enemies.x[np.arange(B), S3.room] != S2.enemies.x[np.arange(B), S2.room]) & np.array(S3.enemies.kind[np.arange(B), S3.room] == 2)
assert moved.any(), 'brutes move'
assert np.isfinite(np.array(obs)).all()
ef = np.array(obs)[:, 8+12*8:8+12*8+40].reshape(B, 8, 5)                # nearest enemies: present, x, y, ranged, hp/4
assert set(np.unique(ef[ef[..., 0] > .5][:, 3])) <= {0., 1.}
print('PASS')
