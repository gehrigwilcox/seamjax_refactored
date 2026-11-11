import _setup
"""Followers come after you through the dungeon (sim/systems/roamers.py). Three follower kinds are added for the test (one per
route, never spawned on their own), put in the start room, and you walk out through a door:
  - they leave the room with you, one after another
  - DIRECT ones come in through the door you used, warned one hop ahead, and become ordinary enemies of their kind
  - FLANK ones come in through another door than yours (when the dungeon has one)
  - DETOUR ones take more doors than the direct route"""
import numpy as np, jax, jax.numpy as jnp
from seamjax.core import kinds
F = dict(follows=True, follow_delay=.2, hop=.3, share=(0., 0., 0., 0.))
kinds.ENEMY_KINDS = kinds.ENEMY_KINDS + (kinds.EnemyKind('t_direct', radius=12., hp=3., behaviour=kinds.CHASE, speed=0., route=kinds.DIRECT, **F),
                                         kinds.EnemyKind('t_flank', radius=12., hp=3., behaviour=kinds.CHASE, speed=0., route=kinds.FLANK, **F),
                                         kinds.EnemyKind('t_detour', radius=12., hp=3., behaviour=kinds.CHASE, speed=0., route=kinds.DETOUR, **F))
DIRECT, FLANK, DETOUR = len(kinds.ENEMY_KINDS)-3, len(kinds.ENEMY_KINDS)-2, len(kinds.ENEMY_KINDS)-1
from seamjax.core.config import CX, CY, MAX_ROAMERS
from seamjax.sim.game import Game
from seamjax.sim.state import no_events
from seamjax.sim.systems.transit import transit
from seamjax.sim.systems.roamers import warn_sides

G = Game.generated(); B = 48
S0 = jax.jit(jax.vmap(G.reset))(jax.random.split(jax.random.PRNGKey(2), B), jnp.full(B, 2))      # floor 2: few inner walls

def setup(s, kind, n=2):
    """n followers of `kind` in the start room, everything calm, then walk out through the first door"""
    r = s.room; en = s.enemies
    for i in range(en.alive.shape[1]):
        on = i < n
        en = en.replace(x=en.x.at[r, i].set(CX+30*i), y=en.y.at[r, i].set(CY), hp=en.hp.at[r, i].set(3.), kind=en.kind.at[r, i].set(kind),
                        telegraph=en.telegraph.at[r, i].set(0.), alive=en.alive.at[r, i].set(on))
    s = s.replace(enemies=en, player=s.player.replace(inv=jnp.float32(1e9)), hunter=s.hunter.replace(dead=jnp.float32(1e9)),
                  spawned=jnp.ones_like(s.spawned))                                   # no fresh spawns: only the followers
    side = jnp.argmax(s.dun.doors[r] >= 0)
    return transit(s, no_events(), side)[0], r, side

@jax.jit
@jax.vmap
def follow(s, kind):
    s, r0, side = setup(s, kind)
    left = jnp.sum(s.roamers.alive); stayed = jnp.sum(s.enemies.alive[r0])
    entry = s.dun.arrive[r0, side]                                                     # the side you came in through
    def body(c, _):
        s, warned, hops, first_side = c
        w = warn_sides(s); before = s.roamers.room
        s2 = G.step(s, jnp.zeros(4))[0]
        hops = hops + jnp.sum(s2.roamers.alive & (s2.roamers.room != before))
        en = s2.enemies[s2.room]; arrived = en.alive & (en.kind == kind)
        new = arrived & ~(s.enemies.alive[s.room] & (s.enemies.kind[s.room] == kind))
        # the side nearest a newly arrived enemy
        d = jnp.hypot(s2.dun.sides.door[s2.room, :, 0][None, :]-en.x[:, None], s2.dun.sides.door[s2.room, :, 1][None, :]-en.y[:, None])
        nearest = jnp.argmin(jnp.where(jnp.arange(d.shape[1]) < s2.dun.n[s2.room], d, 1e9), axis=1)
        k = jnp.argmax(new); first_side = jnp.where((first_side < 0) & jnp.any(new), nearest[k], first_side)
        warned = warned | jnp.any(w & (jnp.arange(w.shape[0]) == nearest[k]) & jnp.any(new))
        return (s2, warned, hops, first_side), None
    (s, warned, hops, first_side), _ = jax.lax.scan(body, (s, False, jnp.int32(0), jnp.int32(-1)), None, length=60*20)
    arrived = jnp.sum(s.enemies.alive[s.room] & (s.enemies.kind[s.room] == kind))
    return left, stayed, arrived, jnp.sum(s.roamers.alive), warned, hops, first_side, entry, s.room == s.dun.doors[r0, side]

res = {}
for name, kind in (('direct', DIRECT), ('flank', FLANK), ('detour', DETOUR)):
    res[name] = [np.array(x) for x in follow(S0, jnp.full(B, kind))]
    left, stayed, arrived, still, warned, hops, first_side, entry, same_room = res[name]
    assert (left == 2).all() and (stayed == 0).all(), (name, 'both followers leave with you')
    assert same_room.all()
    print('%-6s: arrived %d of %d, still travelling %d, warned before arriving %d of %d games, mean doors taken %.1f' % (
        name, arrived.sum(), 2*B, still.sum(), warned.sum(), B, hops.mean()))
    assert (arrived+still == 2).all()

d = res['direct']; assert (d[2] == 2).all(), 'direct followers all arrive'
def plain(i):
    """the room you left joins yours by one door and has no solid inner wall (else another door can be just as short, or the
    follower stands on the far side of the wall and must go round)"""
    dd = np.array(S0.dun.doors[i]); r0 = int(S0.dun.start[i]); p = dd[r0, int(np.argmax(dd[r0] >= 0))]
    return int((dd[r0] == p).sum()) == 1 and int(S0.dun.wall_kind[i, r0]) != 1
pl = np.array([plain(i) for i in range(B)])
print('direct: through the door you used in %d of %d plain games (%d other games)' % ((d[6] == d[7])[pl].sum(), pl.sum(), (~pl).sum()))
assert pl.sum() > B//2 and (d[6] == d[7])[pl].all(), 'direct followers come in through the door you used'
assert d[4].all(), 'their door is warned before they come in'
# where a flank is possible at all: from the room you left, another room next to yours is reachable without going through yours
from collections import deque
def flank_possible(i):
    d = jax.tree_util.tree_map(lambda a: np.array(a[i]), S0.dun); r0 = int(d.start); k = int(np.argmax(d.doors[r0] >= 0)); p = int(d.doors[r0, k])
    if (d.wall_kind > 0).any(): return None                                            # sections complicate it: leave those out
    others = {int(j) for j in d.doors[p][:d.n[p]] if j >= 0 and j != r0}
    seen = {r0}; q = deque([r0])
    while q:
        a = q.popleft()
        for j in d.doors[a][:d.n[a]]:
            j = int(j)
            if j >= 0 and j != p and j not in seen: seen.add(j); q.append(j)
    return bool(others & seen)
f = res['flank']; other = (f[6] >= 0) & (f[6] != f[7]); pos = [flank_possible(i) for i in range(B)]
can = np.array([x is True for x in pos]); cannot = np.array([x is False for x in pos])
print('flank: another door than yours in %d of %d games where a flank is possible; %d of %d where it is not' % (
    (other & can).sum(), can.sum(), (other & cannot).sum(), cannot.sum()))
assert can.sum() >= 3 and (other & can).sum() >= .9*can.sum() and (other & cannot).sum() == 0
t = res['detour']; print('detour: %.1f doors on average, direct %.1f' % (t[5].mean(), d[5].mean())); assert t[5].mean() > d[5].mean()+1
print('PASS')
