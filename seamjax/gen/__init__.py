"""
The dungeon generator. Pure JAX, fixed-size arrays, jit / vmap friendly; runs on the GPU.

Input: the tiling tables (specs.py), built once. Everything random per dungeon is here, in stages:

    layout.py       which tiling, which rooms, which doors, start / exit, gating of solid walls
    room.py         one room: shape.py (polygon) -> obstacles.py (one style of five) -> walls.py (inner wall)
                    -> placement.py (exit pad, spawn points, start position); grid.py keeps it all reachable
    dungeon.py      gen_dungeon(SP, key, floor) -> Dungeon, and gen_source(SP) for sim.game.Game
"""
from seamjax.specs import load_specs
from seamjax.gen.dungeon import gen_dungeon, gen_source
