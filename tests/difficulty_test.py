import _setup
import numpy as np, jax, jax.numpy as jnp
from seamjax.env import *; from seamjax.gen import *
"""Difficulty rises smoothly with the floor, and taking the stairs continues the run (hp+1, kills kept, floor+1)."""
SP = load_specs(); E = make_env(gen_source(SP)); B = 96
gen = jax.jit(jax.vmap(lambda k, f: gen_dungeon(SP, k, f)))
rows = {}
for f in (1, 2, 3, 5, 8, 14):
    D = {k: np.array(v) for k, v in gen(jax.random.split(jax.random.PRNGKey(f), B), jnp.full(B, f)).items()}
    m = D['n_rooms']; walls = (D['wax'] > 0).sum()/m.sum()
    cat = np.bincount(D['geo'], minlength=4)/B
    rows[f] = dict(rooms=m.mean(), walls=walls, flat=cat[0], hyper=cat[1], sphere=cat[2], grid=cat[3])
    print(f'floor {f:2d}: rooms {m.mean():5.1f}  walls/room {walls:.2f}  flat {cat[0]:.2f} hyper {cat[1]:.2f} sphere {cat[2]:.2f} grid {cat[3]:.2f}')
assert rows[1]['flat'] == 1 and rows[1]['walls'] == 0, 'floor 1 must be flat and wall-free'
assert rows[1]['rooms'] < rows[3]['rooms'] < rows[8]['rooms'] + 1e-9 and rows[1]['rooms'] < 14
assert rows[1]['walls'] == 0 and rows[3]['walls'] < rows[8]['walls']
assert rows[8]['hyper'] > .15 and rows[8]['sphere'] > .05 and rows[2]['hyper'] == 0 and rows[2]['grid'] > .3
# enemies per entry, hunter delay, shooter share go up with depth
from seamjax.difficulty import difficulty
d = [difficulty(f) for f in (1, 4, 10, 40)]
assert float(d[0]['shooter_p']) < float(d[1]['shooter_p']) < float(d[2]['shooter_p']) <= float(d[3]['shooter_p']) <= .45
assert float(d[0]['hunter_first']) > float(d[2]['hunter_first']) >= float(d[3]['hunter_first'])
# the stairs: put the player on the exit pad of a cleared exit room
s = jax.jit(E['reset'])(jax.random.PRNGKey(4), 2)
dun = s['dun']; ex = int(dun['exit']); pad = np.array(dun['pad'][ex])
s = {**s, 'room': jnp.int32(ex), 'x': jnp.float32(pad[0]), 'y': jnp.float32(pad[1]), 'hp': jnp.int32(3), 'kills': jnp.int32(7), 'inv': jnp.float32(9.),
     'hdead': jnp.float32(1e9), 'spawned': jnp.ones_like(s['spawned']), 'wopen': jnp.ones_like(s['wopen']) | (dun['wkind'] > 0)}
s2, r, done, info = jax.jit(E['step_auto'])(s, jnp.zeros(4), jax.random.PRNGKey(9))
print('after stairs: floor', int(s2['dun']['floor']), 'hp', int(s2['hp']), 'kills', int(s2['kills']), 'done', bool(done), 'reward', float(r))
assert not bool(done) and bool(info['descended']) and int(s2['dun']['floor']) == 3 and int(s2['hp']) == 4 and int(s2['kills']) == 7 and float(r) >= R_DESCEND-.01
# death starts a new run at floor 1
s = {**s, 'hp': jnp.int32(0)}
s3, r, done, info = jax.jit(E['step_auto'])(s, jnp.zeros(4), jax.random.PRNGKey(9))
assert bool(done) and int(s3['dun']['floor']) == 1 and int(s3['hp']) == MAXHP and int(s3['kills']) == 0
print('PASS')
