"""
Roaming enemies: followers come after you through the dungeon.

When you leave a room, its living enemies whose kind `follows` (core/kinds.py) leave the room's enemy pool and join the roamer
pool, one after another (FOLLOW_STAGGER apart). Roaming, an enemy is only a place on the room graph, like the Hunter away
from you; it hops one door every `hop` seconds along a route of its kind:

    DIRECT   the true shortest route to you
    FLANK    first to a room next to yours other than the one you came from, never through yours, so it comes in by
             another door (if the dungeon has no such way, it comes straight in)
    DETOUR   first via a random room at most 4 doors out of its way, never through yours

When it reaches your section it comes in through the door it used (or a door of your section, if you walked into its room)
and becomes an ordinary enemy again, briefly telegraphed. If the room's enemy slots are all taken it waits.
A roamer one door away warns at that door, as the Hunter does.

Randomness is folded from the step key without advancing it, so a game without followers plays exactly as before.
"""
import jax, jax.numpy as jnp
from jax import lax
from seamjax.core import kinds, pool
from seamjax.core.config import MAX_SIDES as NS, MAX_ENEMIES as NE, INF, ARRIVE_TELEGRAPH, FOLLOW_STAGGER, HUNTER_DOOR_INSET
from seamjax.core.types import Roamers, Enemies
from seamjax.sim.queries import hnode, sec_point, player_node
from seamjax.sim.systems.routing import next_hop, entry_side, real_nodes, neighbours
from seamjax.sim.systems.enemies import of


def _waypoint(dun, route, room, sec, p, key):
    """the node a FLANK / DETOUR roamer in (room, sec) heads for first, on its way to node p; -1 = none (go straight)"""
    D = dun.node_dist; n = hnode(dun, room, sec); ok = real_nodes(dun) & (D[n] < INF); g = jax.random.gumbel(key, ok.shape)
    nbn, stepok = neighbours(dun, room, sec)
    first_step = jnp.any((stepok & (nbn != p))[:, None] & (D[nbn] == D[n][None, :]-1), axis=0)    # a closer first step that is not into p
    flank = ok & first_step & (D[:, p] == 1) & (jnp.arange(ok.shape[0]) != n)                # next to you, not where it is
    detour = ok & first_step & (D[n] >= 1) & (D[:, p] >= 2) & (D[n]+D[:, p] <= D[n, p]+4)
    wf = jnp.where(jnp.any(flank), jnp.argmin(jnp.where(flank, D[n]+.1*g, 1e9)), -1)          # the nearest such room
    wd = jnp.where(jnp.any(detour), jnp.argmax(jnp.where(detour, g, -1e9)), -1)               # any such room
    return jnp.where(route == kinds.FLANK, wf, jnp.where(route == kinds.DETOUR, wd, -1)).astype(jnp.int32)


def depart(s, room, target):
    """you are leaving `room` for node `target`: its followers set off after you"""
    dun = s.dun; en = s.enemies[room]; follows = of('follows', en.kind) & en.alive; route = of('route', en.kind)
    def one(i, c):
        alive, R, order = c
        f = pool.first_free(R); go = follows[i] & ~R.alive[f]
        sec = sec_point(dun, room, en.x[i], en.y[i])
        way = _waypoint(dun, route[i], room, sec, target, jax.random.fold_in(s.key, 101+i))
        new = Roamers(alive=True, kind=en.kind[i], hp=en.hp[i], room=room, sec=sec, timer=of('follow_delay', en.kind[i])+FOLLOW_STAGGER*order, way=way)
        R = jax.tree_util.tree_map(lambda a, b: jnp.where(go, a, b), pool.put(R, f, new), R)
        return alive.at[i].set(alive[i] & ~go), R, order+go.astype(jnp.int32)
    alive, R, _ = lax.fori_loop(0, NE, one, (en.alive, s.roamers, jnp.int32(0)))
    return s.replace(roamers=R, enemies=pool.set_row(s.enemies, room, en.replace(alive=alive)))


def _hop(dun, room, sec, way, pn):
    """toward the waypoint without stepping into your node, else (no waypoint, or no such step) straight to you"""
    w = next_hop(dun, room, sec, jnp.maximum(way, 0), pn); d = next_hop(dun, room, sec, pn)
    use = (way >= 0) & w[0]
    return tuple(jnp.where(use, a, b) for a, b in zip(w, d))


def _move(dun, pn, dt, R):
    """one roamer: count down, hop one door toward its waypoint or you. Returns (roamer, came in by a door, through which side)"""
    n = hnode(dun, R.room, R.sec)
    way = jnp.where((R.way >= 0) & (dun.node_dist[n, jnp.maximum(R.way, 0)] == 0), -1, R.way)          # waypoint reached
    tm = R.timer-dt
    ok, room, sec, side, nn = _hop(dun, R.room, R.sec, way, pn)
    can = R.alive & (n != pn) & (tm <= 0) & ok
    R = R.replace(room=jnp.where(can, room, R.room), sec=jnp.where(can, sec, R.sec), timer=jnp.where(can, of('hop', R.kind), tm),
                  way=jnp.where(can & (nn == way), -1, way))
    return R, can & (nn == pn), side


def tick(s, ev, dt):
    dun = s.dun; pn = player_node(s); room = s.room
    R, by_door, side = jax.vmap(lambda r: _move(dun, pn, dt, r))(s.roamers)
    here = R.alive & (hnode(dun, R.room, R.sec) == pn)
    side = jnp.where(by_door, side, entry_side(dun, room, pn))                                   # walked into its room: a door of your section
    def come_in(i, c):
        R, en = c
        f = pool.first_free(en); go = here[i] & ~en.alive[f]
        door, nrm = dun.sides.door[room, side[i]], dun.sides.normal[room, side[i]]
        new = Enemies(x=door[0]-nrm[0]*HUNTER_DOOR_INSET, y=door[1]-nrm[1]*HUNTER_DOOR_INSET, hp=R.hp[i], kind=R.kind[i],
                      fire=of('first_shot', R.kind[i]), telegraph=ARRIVE_TELEGRAPH, flash=0., alive=True, phase=0, ptimer=0.)
        en = jax.tree_util.tree_map(lambda a, b: jnp.where(go, a, b), pool.put(en, f, new), en)
        return R.replace(alive=R.alive.at[i].set(R.alive[i] & ~go)), en
    R, en = lax.fori_loop(0, R.alive.shape[0], come_in, (R, s.enemies[room]))
    return s.replace(roamers=R, enemies=pool.set_row(s.enemies, room, en)), ev


def warn_sides(s):
    """[NS] sides of your room a roamer will come in through at its next hop"""
    dun = s.dun; pn = player_node(s); R = s.roamers
    def one(R):
        n = hnode(dun, R.room, R.sec)
        way = jnp.where((R.way >= 0) & (dun.node_dist[n, jnp.maximum(R.way, 0)] == 0), -1, R.way)
        ok, room, sec, side, nn = _hop(dun, R.room, R.sec, way, pn)
        return R.alive & (n != pn) & ok & (nn == pn), side
    coming, side = jax.vmap(one)(R)
    return jnp.any(coming[:, None] & (side[:, None] == jnp.arange(NS)[None, :]), axis=0)
