import _setup
"""Player classes and their weapons, in the real simulation: the scatter fans its pellets, the lance pierces enemies and passes
obstacles, the blade hits in its arc and only there, max hp follows the class, and the class survives the stairs and death.
Class 0 (the gunner) is the game as it always was: the golden trace checks that."""
import numpy as np, jax, jax.numpy as jnp
from seamjax.core import kinds
from seamjax.core.config import CX, CY
from seamjax.core.types import OBST_NONE, OBST_CIRCLE
from seamjax.sim.game import Game
from seamjax.sim.observe import observe

G = Game.generated(); step = jax.jit(G.step); reset = jax.jit(G.reset)
idx = {c.name: i for i, c in enumerate(kinds.PLAYER_CLASSES)}

def room_for(cls, enemies=(), obstacle=None):
    """the start room emptied (no obstacles, no Hunter), the player in the middle, `enemies` = [(x, y, hp)] as still chasers"""
    s = reset(jax.random.PRNGKey(4), 3, None, cls); r = int(s.room); d = s.dun
    ob = d.obstacles; kind = ob.kind.at[r].set(OBST_NONE)
    if obstacle is not None:
        kind = kind.at[r, 0].set(OBST_CIRCLE); ob = ob.replace(pos=ob.pos.at[r, 0].set(jnp.array(obstacle[:2])), size=ob.size.at[r, 0].set(jnp.array([obstacle[2], 0.])))
    d = d.replace(obstacles=ob.replace(kind=kind, wall=ob.wall.at[r].set(False)), wall_axis=d.wall_axis.at[r].set(0), wall_kind=d.wall_kind.at[r].set(0))
    en = s.enemies; NE = en.alive.shape[1]
    for i in range(NE):
        on = i < len(enemies); x, y, hp = enemies[i] if on else (0., 0., 0.)
        en = en.replace(x=en.x.at[r, i].set(x), y=en.y.at[r, i].set(y), hp=en.hp.at[r, i].set(hp), kind=en.kind.at[r, i].set(0),
                        telegraph=en.telegraph.at[r, i].set(0.), flash=en.flash.at[r, i].set(0.), alive=en.alive.at[r, i].set(on))
    return s.replace(dun=d, enemies=en, player=s.player.replace(x=jnp.float32(CX), y=jnp.float32(CY), inv=jnp.float32(99.)),
                     hunter=s.hunter.replace(dead=jnp.float32(1e9))), r

def run(s, frames, fire_frames=1):
    """aim right; hold the aim for fire_frames frames, then let go"""
    for f in range(frames): s = step(s, jnp.array([0., 0., 1., 0.]) if f < fire_frames else jnp.zeros(4))[0]
    return s

# ---- pellets: one shot of the scatter is a fan of 5 over its spread; the gunner fires 1
for name, n in (('gunner', 1), ('scatter', 5)):
    s, r = room_for(idx[name]); s = run(s, 1)
    pb = jax.tree_util.tree_map(np.array, s.pb); ang = np.arctan2(pb.vy[pb.alive], pb.vx[pb.alive])
    assert pb.alive.sum() == n, (name, pb.alive.sum())
    if n > 1: assert abs((ang.max()-ang.min())-kinds.SCATTER.spread) < 1e-3, ang
print('pellets: gunner 1, scatter 5 over %.2f rad' % kinds.SCATTER.spread)

# ---- pierce: three enemies in a line; one lance shot kills all three, one gunner shot only the first
line = [(CX+70, CY, 1.), (CX+110, CY, 1.), (CX+150, CY, 1.)]
for name, want in (('lancer', 3), ('gunner', 1)):
    s, r = room_for(idx[name], line); s = run(s, 20)
    killed = 3-int(np.array(s.enemies.alive[r]).sum()); assert killed == want, (name, killed)
print('pierce: lance kills 3 in a line, gunner 1')

# ---- ghost: an obstacle between you and an enemy: the lance shoots through it, the gunner's bullet stops
for name, survives in (('lancer', False), ('gunner', True)):
    s, r = room_for(idx[name], [(CX+130, CY, 1.)], obstacle=(CX+60, CY, 18.)); s = run(s, 20)
    assert bool(s.enemies.alive[r, 0]) == survives, name
print('ghost: lance passes the obstacle, gunner does not')

# ---- melee: the blade hits in front within reach, not behind, not beyond reach; it spawns no bullets
s, r = room_for(idx['blade'], [(CX+40, CY, 2.), (CX-40, CY, 2.), (CX+120, CY, 2.)]); s = run(s, 1)
alive = np.array(s.enemies.alive[r, :3]); assert alive.tolist() == [False, True, True], alive
assert int(np.array(s.pb.alive).sum()) == 0 and float(s.player.swing) > 0
print('melee: the swing kills in front within reach only')

# ---- max hp, the observation, and the class through the stairs and death
for name in idx:
    s, r = room_for(idx[name]); c = kinds.PLAYER_CLASSES[idx[name]]
    assert int(s.player.hp) == c.max_hp and abs(float(observe(s)[2])-1.) < 1e-6
    n1 = G.next_state(s, jax.random.PRNGKey(1), jnp.bool_(False), jnp.bool_(True)); n2 = G.next_state(s, jax.random.PRNGKey(1), jnp.bool_(True), jnp.bool_(False))
    assert int(n1.player.cls) == idx[name] == int(n2.player.cls) and int(n2.player.hp) == c.max_hp
print('max hp per class:', {c.name: c.max_hp for c in kinds.PLAYER_CLASSES}, '- the class stays on the stairs and after death')

# ---- the batched path: classes per game
S = jax.jit(G.reset_batch)(jax.random.split(jax.random.PRNGKey(0), 4), jnp.full(4, 1), jnp.arange(4))
assert np.array(S.player.cls).tolist() == [0, 1, 2, 3]
print('PASS')
