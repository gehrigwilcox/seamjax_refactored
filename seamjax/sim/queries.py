"""
Read-only questions about a game state, used by the systems, the observation, the renderers and the HUD.

Sections: an inner wall splits a room into section 0 and section 1 (by y for a horizontal wall, by x for a vertical one).
The Hunter moves on "nodes": a node is a room-section, except that a gated wall does not separate (the Hunter ignores gates).
"""
import jax.numpy as jnp
from seamjax.core.config import INF
from seamjax.core.types import OBST_NONE, AXIS_NONE, AXIS_H, WALL_SOLID


def room_obs(s, room):
    """the obstacles of a room as they block things now (an open inner wall no longer counts)"""
    ob = s.dun.obstacles[room]
    return ob.replace(kind=jnp.where(ob.wall & s.wall_open[room], OBST_NONE, ob.kind))

def room_geom(s, room):
    """(sides, n, obstacles) of a room, the arguments of the collision functions"""
    return s.dun.sides[room], s.dun.n[room], room_obs(s, room)

def sec_point(dun, room, x, y):
    """section (0 / 1) of the room that (x, y) lies in"""
    ax, pos = dun.wall_axis[room], dun.wall_pos[room]
    return jnp.where(ax == AXIS_NONE, 0, jnp.where(ax == AXIS_H, y >= pos, x >= pos)).astype(jnp.int32)

def eff_sec(s, room, sec):
    """an opened wall merges the two sections"""
    return jnp.where(s.wall_open[room], 0, sec)

def hnode(dun, r, sec):
    """the Hunter's node for section `sec` of room r"""
    return 2*r + jnp.where(dun.wall_kind[r] == WALL_SOLID, sec, 0)

def player_node(s):
    return hnode(s.dun, s.room, sec_point(s.dun, s.room, s.player.x, s.player.y))

def hunter_dist(s):
    """doors between you and the Hunter in section space: 0 same section, -1 down, -2 cut off by a wall"""
    d = s.dun.node_dist[hnode(s.dun, s.hunter.room, s.hunter.sec), player_node(s)]
    return jnp.where(s.hunter.dead > 0, -1, jnp.where(d >= INF, -2, d)).astype(jnp.int32)

def hunter_door(s):
    """side of your room through which the Hunter would come (it is one door-step away from your section), else -1"""
    room = s.room; dun = s.dun; hr = s.hunter.room
    pn = player_node(s); hn = hnode(dun, hr, s.hunter.sec)
    doors = dun.doors[hr]; arrk = dun.arrive[hr]
    ok = (doors == room) & (hnode(dun, hr, dun.door_sec[hr]) == hn) & (hnode(dun, room, dun.door_sec[room, arrk]) == pn)
    k = jnp.argmax(ok)
    return jnp.where((hn != pn) & (s.hunter.dead <= 0) & jnp.any(ok), arrk[k], -1)

def hunter_visible(s):
    return (s.hunter.room == s.room) & s.hunter.here & (s.hunter.dead <= 0)
