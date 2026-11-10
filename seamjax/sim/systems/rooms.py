"""Room progress: a gated inner wall opens once your side is cleared; the stairs work from a cleared exit room."""
import jax.numpy as jnp
from seamjax.core.config import PAD_R
from seamjax.core.types import WALL_GATED
from seamjax.sim.queries import sec_point


def cleared(s):
    return ~jnp.any(s.enemies.alive[s.room])


def open_gates(s, ev):
    """a gated wall opens when your section has spawned and has no enemies left; the other section counts as spawned then"""
    room = s.room; dun = s.dun
    psec = sec_point(dun, room, s.player.x, s.player.y)
    gate = (dun.wall_kind[room] == WALL_GATED) & ~s.wall_open[room] & cleared(s) & s.spawned[room, psec]
    wopen = s.wall_open.at[room].set(s.wall_open[room] | gate)
    spawned = jnp.where(gate, s.spawned.at[room].set(True), s.spawned)
    return s.replace(wall_open=wopen, spawned=spawned), ev.replace(opened=gate.astype(jnp.int32))


def stairs(s, ev):
    """standing on the exit pad of the cleared exit room, in the pad's section (or the wall is open): descend"""
    room = s.room; dun = s.dun; pad = dun.pad[room]; x, y = s.player.x, s.player.y
    same = (sec_point(dun, room, pad[0], pad[1]) == sec_point(dun, room, x, y)) | s.wall_open[room]
    return s, ev.replace(descended=(room == dun.exit) & cleared(s) & (jnp.hypot(x-pad[0], y-pad[1]) < PAD_R) & same)
