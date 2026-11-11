import _setup
"""Bosses (sim/systems/bosses.py): a test boss with three phases is added (on every 2nd floor). It must spawn once in the exit
room on its floors and only there, change phase as its hp falls (and never back), move as each phase says, fire each phase's
volley (an aimed fan, a ring, a spinning ring), and hold the stairs until it is down."""
import numpy as np, jax, jax.numpy as jnp
from seamjax.core import kinds
TAU = 2*np.pi
TB = kinds.EnemyKind('t_boss', radius=30., hp=30., behaviour=kinds.STAND, speed=0., hp_per_floor=2., boss_every=2, share=(0., 0., 0., 0.), phases=(
    kinds.Phase(1., kinds.STAND, count=5, spread=.6, aimed=True, interval=.5),
    kinds.Phase(.5, kinds.CHASE, speed=60., count=12, spread=TAU, aimed=False, interval=.7),
    kinds.Phase(.2, kinds.SKIRMISH, speed=50., count=3, spread=TAU, aimed=False, spin=2., interval=.2)))
kinds.ENEMY_KINDS = kinds.ENEMY_KINDS + (TB,); IB = len(kinds.ENEMY_KINDS)-1
from seamjax.core.config import CX, CY, BOSS_TELEGRAPH, PHASE_PAUSE
from seamjax.sim.game import Game
from seamjax.sim.systems import enemies
from seamjax.sim.queries import sec_point

G = Game.generated(); step = jax.jit(G.step)

def exit_room(floor, seed=1):
    """stand in the exit room and let its pad's section spawn"""
    s = G.reset(jax.random.PRNGKey(seed), floor); d = s.dun; ex = d.exit; pad = d.pad[ex]
    s = s.replace(room=ex, player=s.player.replace(x=pad[0], y=pad[1], inv=jnp.float32(1e9)), hunter=s.hunter.replace(dead=jnp.float32(1e9)))
    return jax.jit(enemies.spawn)(s, jax.random.PRNGKey(2), sec_point(d, ex, pad[0], pad[1]))

def boss_slot(s):
    en = s.enemies[s.room]; m = np.array(en.alive & (en.kind == IB)); return m, int(np.argmax(m))

# ---- only on boss floors, once, in the exit room
for floor, want in ((2, 1), (3, 0), (4, 1)):
    s = exit_room(floor); m, b = boss_slot(s); assert m.sum() == want, (floor, m.sum())
    if want:
        assert float(s.enemies.hp[s.room, b]) == 30.+2.*floor and float(s.enemies.telegraph[s.room, b]) == BOSS_TELEGRAPH
assert sum(int(np.array(s.enemies.alive[r] & (s.enemies.kind[r] == IB)).sum()) for r in range(s.enemies.alive.shape[0]) if r != int(s.room)) == 0
print('spawns on floors 2 and 4 in the exit room, not on floor 3; hp grows with the floor')

def run(s, frames):
    for _ in range(frames): s = step(s, jnp.zeros(4))[0]
    return s

def volley_size(s, frames):
    """the largest number of enemy bullets that appeared in one frame"""
    best = 0
    for _ in range(frames):
        n0 = int(np.array(s.eb.alive).sum()); s = step(s, jnp.zeros(4))[0]; best = max(best, int(np.array(s.eb.alive).sum())-n0)
    return s, best

# ---- phase 0: stands still, fires an aimed fan of 5 (after the telegraph and the opening pause)
s = exit_room(2); m, b = boss_slot(s); r = int(s.room)
s = run(s, int(60*(BOSS_TELEGRAPH+PHASE_PAUSE))+2); x0 = float(s.enemies.x[r, b])
s, v0 = volley_size(s, 60); assert v0 == 5, v0
assert abs(float(s.enemies.x[r, b])-x0) < 1e-3 and int(s.enemies.phase[r, b]) == 0
print('phase 0: stands, fan of', v0)
# ---- phase 1 at half hp: chases, a ring of 12
s = s.replace(enemies=s.enemies.replace(hp=s.enemies.hp.at[r, b].set(.45*34.)))
s = run(s, 2); assert int(s.enemies.phase[r, b]) == 1
p0 = np.array([s.enemies.x[r, b], s.enemies.y[r, b]]); s, v1 = volley_size(s, 100); p1 = np.array([s.enemies.x[r, b], s.enemies.y[r, b]])
assert v1 == 12 and np.hypot(*(p1-p0)) > 20, (v1, p1-p0)
print('phase 1: chases (moved %.0f px), ring of %d' % (np.hypot(*(p1-p0)), v1))
# ---- phase 2 below a fifth: a ring of 3 that turns between volleys
s = s.replace(enemies=s.enemies.replace(hp=s.enemies.hp.at[r, b].set(.15*34.))); s = run(s, 2); assert int(s.enemies.phase[r, b]) == 2
s = run(s, int(60*PHASE_PAUSE)+2); angs = []
for _ in range(40):
    before = np.array(s.eb.alive); s = step(s, jnp.zeros(4))[0]; new = np.array(s.eb.alive) & ~before
    if new.sum() == 3: angs.append(np.sort(np.mod(np.arctan2(np.array(s.eb.vy)[new], np.array(s.eb.vx)[new]), TAU)))
assert len(angs) >= 2 and not np.allclose(angs[0], angs[1], atol=1e-3), 'the spiral turns'
print('phase 2: rings of 3, turning between volleys (%d volleys seen)' % len(angs))
# ---- never back to an earlier phase
s = s.replace(enemies=s.enemies.replace(hp=s.enemies.hp.at[r, b].set(34.))); s = run(s, 2); assert int(s.enemies.phase[r, b]) == 2
# ---- the stairs wait for the boss
d = s.dun; pad = d.pad[r]; s = s.replace(player=s.player.replace(x=pad[0], y=pad[1]), wall_open=s.wall_open | (d.wall_kind > 0))
s = s.replace(enemies=s.enemies.replace(alive=s.enemies.alive.at[r].set(jnp.arange(s.enemies.alive.shape[1]) == b)))      # only the boss left
_, _, _, ev = step(s, jnp.zeros(4)); assert not bool(ev.descended), 'no stairs with the boss alive'
s = s.replace(enemies=s.enemies.replace(alive=s.enemies.alive.at[r, b].set(False)))
_, _, _, ev = step(s, jnp.zeros(4)); assert bool(ev.descended), 'stairs once it is down'
print('never returns to an earlier phase; the stairs wait for it')
print('PASS')
