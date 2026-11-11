"""
Moving on the room graph. Things that travel between rooms without being simulated (the Hunter, roaming enemies) live on
"nodes": room-sections where a solid inner wall separates and a gated one does not (queries.hnode). The dungeon carries the
door distance between every pair of nodes (Dungeon.node_dist), so the next step of a true shortest route is one lookup.
"""
import jax.numpy as jnp
from seamjax.core.config import MAX_SIDES as NS, INF
from seamjax.core.types import WALL_SOLID
from seamjax.sim.queries import hnode


def neighbours(dun, r, sec):
    """[NS] the node behind each side of room r, and whether that side is a door out of section sec"""
    hn = hnode(dun, r, sec)
    doors = dun.doors[r]; arrk = dun.arrive[r]; nb = jnp.maximum(doors, 0)
    nbnode = hnode(dun, nb, dun.door_sec[nb, arrk])
    return nbnode, (doors >= 0) & (jnp.arange(NS) < dun.n[r]) & (hnode(dun, r, dun.door_sec[r]) == hn)


def next_hop(dun, r, sec, target, avoid=-1):
    """the door out of room r, section sec, that starts a shortest route to node `target`.
    With `avoid` (a node), never step into it, and only take a step that brings you closer (so a detour cannot loop).
    Returns (ok, next room, its section, the side you arrive through, the next node); ok is False if there is no such step"""
    doors = dun.doors[r]; arrk = dun.arrive[r]
    nbnode, stepok = neighbours(dun, r, sec)
    dd = jnp.where(stepok & (nbnode != avoid), dun.node_dist[nbnode, target], INF)
    bk = jnp.argmin(dd); best = doors[bk]
    arrside = arrk[bk]
    ok = (dd[bk] < INF) & ((avoid < 0) | (dd[bk] < dun.node_dist[hnode(dun, r, sec), target]))
    return ok, best, dun.door_sec[best, arrside], arrside, nbnode[bk]


def entry_side(dun, room, node):
    """a side of `room` with a door into node `node` (where something already in your section is placed); 0 if none"""
    rd = dun.doors[room]
    okk = (rd >= 0) & (jnp.arange(NS) < dun.n[room]) & (hnode(dun, room, dun.door_sec[room]) == node)
    return jnp.where(jnp.any(okk), jnp.argmax(okk), 0)


def real_nodes(dun):
    """[2R] which node indices exist: section 0 of every room, section 1 only where a solid wall splits the room"""
    i = jnp.arange(dun.node_dist.shape[0]); r = i // 2
    return (r < dun.n_rooms) & ((i % 2 == 0) | (dun.wall_kind[r] == WALL_SOLID))
