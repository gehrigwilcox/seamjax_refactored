"""Walking through a door into the next room: the whole non-Euclidean part of a step."""
import jax, jax.numpy as jnp
from seamjax.core.config import MAX_PLAYER_BULLETS as NB, MAX_ENEMY_BULLETS as NEB, ENTRY_INSET, HUNTER_GRACE
from seamjax.core.geometry.view import through_door
from seamjax.sim.queries import sec_point, eff_sec
from seamjax.sim.state import cond, empty_bullets
from seamjax.sim.systems import enemies


def transit(s, ev, side):
    """leave the current room through `side`: arrive just inside the matching door of the neighbour, with the view turned
    (and maybe mirrored) so the door you walked through stays where you saw it. Bullets do not follow you. A section you
    enter for the first time spawns its enemies."""
    room = s.room; dun = s.dun
    nr = dun.doors[room, side]; arrive = dun.arrive[room, side]
    va, mir = through_door(s.view_angle, s.mirror, side, dun.n[room], arrive, dun.n[nr], dun.flip[room, side])
    door, nrm = dun.sides.door[nr, arrive], dun.sides.normal[nr, arrive]
    x, y = door[0]-nrm[0]*ENTRY_INSET, door[1]-nrm[1]*ENTRY_INSET
    left = s.hunter.room == room                                       # leaving the Hunter's room: it follows soon
    hunter = s.hunter.replace(timer=jnp.where(left, jnp.minimum(s.hunter.timer, HUNTER_GRACE), s.hunter.timer),
                              here=jnp.where(left, False, s.hunter.here))
    first = ~s.visited[nr]
    s = s.replace(room=nr, view_angle=va, mirror=mir, player=s.player.replace(x=x, y=y), pb=empty_bullets(NB), eb=empty_bullets(NEB), hunter=hunter,
                  visited=s.visited.at[nr].set(True))
    ev = ev.replace(newroom=first.astype(jnp.int32))
    sec = eff_sec(s, nr, sec_point(dun, nr, x, y))
    fresh = ~s.spawned[nr, sec]
    k, key = jax.random.split(s.key)
    s = s.replace(spawned=s.spawned.at[nr, sec].set(True), key=key)
    return cond(fresh, lambda s, ev: (enemies.spawn(s, k, sec), ev), s, ev)
