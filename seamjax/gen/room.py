"""
Designing one room, in stages: shape -> obstacles -> inner wall -> exit pad, spawn points, start position.
Everything stays reachable from the room's open doors.
"""
import jax
from seamjax.core.struct import pytree
from seamjax.core.types import Sides, Obstacles
from seamjax.gen.shape import room_shape
from seamjax.gen.obstacles import layout, N_STYLE_KEYS
from seamjax.gen.walls import inner_wall, Wall
from seamjax.gen.placement import exit_pad, spawn_points, start_position


@pytree
class RoomDesign:
    sides: Sides
    obstacles: Obstacles
    wall: Wall
    pad: object
    radius: object
    inradius: object
    area: object
    spawn: object
    spawn_ok: object
    spawn_sec: object
    startpos: object


def design_room(key, n, is_exit, open_mask, want_wall):
    """an n-sided room whose doors are open_mask [NS]; the exit room gets a pad; want_wall: try an inner wall"""
    ks = jax.random.split(key, 4 + N_STYLE_KEYS)
    sh = room_shape(ks[0], ks[1], ks[2], n)
    ob, reach = layout(ks[3], ks[4:], sh, open_mask)
    ob, reach, reach_first, wall = inner_wall(key, sh, ob, reach, open_mask, want_wall)
    pad = exit_pad(key, sh, ob, reach, is_exit)
    spawn, spawn_ok, spawn_sec = spawn_points(key, sh, ob, reach, wall, pad, is_exit)
    return RoomDesign(sides=sh.sides, obstacles=ob, wall=wall, pad=pad, radius=sh.radius, inradius=sh.inradius, area=sh.area,
                      spawn=spawn, spawn_ok=spawn_ok, spawn_sec=spawn_sec, startpos=start_position(reach_first))
