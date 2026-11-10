import _setup
import numpy as np, jax, jax.numpy as jnp
from seamjax.env import *; from seamjax.gen import *
from seamjax.core.config import *
"""Batched regeneration touches only finished games, and memory stays bounded however big the batch is."""
SP = load_specs(); E = make_env(gen_source(SP)); B = 150                       # not a multiple of the chunk size on purpose
S = jax.jit(E['reset_batch'])(jax.random.split(jax.random.PRNGKey(0), B), jnp.full(B, 3))
assert S.dun.floor.shape == (B,) and bool((S.dun.floor == 3).all())
# kill every 7th game, send every 11th down the stairs
dead = (np.arange(B) % 7 == 0); down = (np.arange(B) % 11 == 0) & ~dead
S2 = S.replace(player=S.player.replace(hp=jnp.where(jnp.asarray(dead), 0, S.player.hp)))
# force "descended" by teleporting those onto the exit pad of a cleared exit room
d = S.dun; ex = np.array(d.exit); pad = np.array(d.pad)[np.arange(B), ex]
m = jnp.asarray(down)
P = S2.player
S2 = S2.replace(room=jnp.where(m, jnp.asarray(ex), S2.room),
                player=P.replace(x=jnp.where(m, jnp.asarray(pad[:, 0]), P.x), y=jnp.where(m, jnp.asarray(pad[:, 1]), P.y), inv=jnp.full(B, 9.)),
                spawned=jnp.where(m[:, None, None], True, S2.spawned), wall_open=jnp.where(m[:, None], True, S2.wall_open),
                hunter=S2.hunter.replace(dead=jnp.full(B, 1e9)), kills=jnp.full(B, 5, jnp.int32))
# 'wopen' on solid walls is not what the game does, but only the exit rule matters here
S3, r, done, info = jax.jit(E['step_auto_batch'])(S2, jnp.zeros((B, 4)), jax.random.PRNGKey(1))
fl = np.array(S3.dun.floor); done = np.array(done); desc = np.array(info['descended'])
assert (done == dead).all(), 'only the dead ones are done'
assert (fl[dead] == 1).all() and (fl[desc] == 4).all() and (fl[~dead & ~desc] == 3).all(), (fl[dead][:5], fl[desc][:5])
assert desc.sum() > 5 and (np.array(S3.kills)[desc] == 5).all() and (np.array(S3.kills)[dead] == 0).all()
same = ~dead & ~desc
assert np.allclose(np.array(S3.dun.sides.mid)[same], np.array(S.dun.sides.mid)[same]), 'untouched games keep their dungeon'
assert (np.array(S3.player.hp)[dead] == MAX_HP).all()
# memory: scratch for a big batch is bounded by the chunk size, not the batch size
for BB in (64, 1024, 4096):
    f = jax.jit(E['reset_batch']).lower(jax.random.split(jax.random.PRNGKey(0), BB), jnp.full(BB, 6)).compile()
    tmp = f.memory_analysis().temp_size_in_bytes/1e9; print('reset of %5d games: scratch %.2f GB' % (BB, tmp)); assert tmp < 1.0
print('PASS')
