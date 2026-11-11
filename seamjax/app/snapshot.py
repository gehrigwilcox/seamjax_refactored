"""
What the viewer reads from the device: a Snapshot of the game each frame (one device read), the drawing tables of the
current room (read once per room) and the dungeon layout for the minimap (read once per floor).

Everything here is numpy once read: drawing never touches the device.
"""
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from seamjax.core.struct import pytree
from seamjax.core.types import Player, Enemies, Pickups, Bullets, Hunter, Sides, Obstacles
from seamjax.sim.queries import hunter_door, hunter_dist
from seamjax.sim.systems.roamers import warn_sides


@pytree
class Flags:
    """things that happened since the last read (several frames may pass between two reads)"""
    opened: object      # a gated wall opened
    died: object
    descended: object   # took the stairs


@pytree
class Snapshot:
    room: object
    floor: object
    spec: object        # index of the world in the catalogue (specs.json)
    start: object
    exit: object
    n_rooms: object
    kills: object
    view_angle: object
    mirror: object
    player: Player
    enemies: Enemies    # of the current room
    pickups: Pickups    # of the current room
    pb: Bullets
    eb: Bullets
    hunter: Hunter
    hunter_door: object # side of this room the Hunter will come through, -1 none
    hunter_dist: object # doors to the Hunter: 0 same section, -1 down, -2 cut off by a wall
    roamer_doors: object  # [NS] sides of this room a follower will come through
    roamers: object     # Roamers: enemies following you (for the debug overlay and the minimap)
    wall_open: object   # of the current room
    spawned: object     # [2] sections of the current room that have spawned
    visited: object     # [R]
    flags: Flags


def no_flags():
    return Flags(opened=jnp.bool_(False), died=jnp.bool_(False), descended=jnp.bool_(False))


def take(s, flags):
    """the Snapshot of state s (jit this; device_get the result)"""
    r = s.room; d = s.dun
    return Snapshot(room=r, floor=d.floor, spec=d.spec, start=d.start, exit=d.exit, n_rooms=d.n_rooms, kills=s.kills,
                    view_angle=s.view_angle, mirror=s.mirror, player=s.player, enemies=s.enemies[r], pickups=s.pickups[r], pb=s.pb, eb=s.eb,
                    hunter=s.hunter, hunter_door=hunter_door(s), hunter_dist=hunter_dist(s), roamer_doors=warn_sides(s), roamers=s.roamers, wall_open=s.wall_open[r], spawned=s.spawned[r],
                    visited=s.visited, flags=flags)


def hunter_shown(sn):
    """the Hunter is standing in your room"""
    return bool(sn.hunter.here) and int(sn.hunter.room) == int(sn.room) and float(sn.hunter.dead) <= 0


# ---------------------------------------------------------------- the current room, for drawing
def room_tables(s):
    """device side: what drawing a room needs (jit this)"""
    d = s.dun; r = s.room
    return d.room(r), d.spawn[r], d.spawn_ok[r], d.spawn_sec[r]


@dataclass
class RoomDraw:
    n: int
    sides: Sides            # [n]
    doors: np.ndarray       # [n]
    corners: np.ndarray     # [n, 2]: corner k is between side k and side k+1
    obstacles: Obstacles
    wall_axis: int
    wall_kind: int
    wall_pos: float
    door_sec: np.ndarray    # [n]
    pad: np.ndarray
    spawn: np.ndarray
    spawn_ok: np.ndarray
    spawn_sec: np.ndarray
    hue: float              # the room's colour

    @classmethod
    def build(cls, tables, room):
        rm, spawn, spawn_ok, spawn_sec = tables
        n = int(rm.n); side = rm.sides[:n]; V = []
        for k in range(n):                                            # polygon corners: intersect neighbouring side lines
            j = (k+1) % n; Mx = np.array([side.normal[k], side.normal[j]])
            bb = np.array([side.normal[k] @ side.mid[k], side.normal[j] @ side.mid[j]])
            V.append(np.linalg.solve(Mx, bb) if abs(np.linalg.det(Mx)) > 1e-9 else side.mid[k])
        return cls(n=n, sides=side, doors=np.asarray(rm.doors)[:n], corners=np.array(V), obstacles=rm.obstacles, wall_axis=int(rm.wall_axis),
                   wall_kind=int(rm.wall_kind), wall_pos=float(rm.wall_pos), door_sec=np.asarray(rm.door_sec)[:n], pad=np.asarray(rm.pad),
                   spawn=np.asarray(spawn), spawn_ok=np.asarray(spawn_ok), spawn_sec=np.asarray(spawn_sec), hue=(room*137.508) % 360)


# ---------------------------------------------------------------- the dungeon, for the minimap
def map_tables(s):
    """device side: the per-floor minimap data (jit this)"""
    d = s.dun
    return dict(mpos=d.map_pos, dseam=d.door_seam, doors=d.doors, n=d.n, n_rooms=d.n_rooms, geo=d.category, start=d.start, exit=d.exit)

def map_layout(T, gen):
    """the minimap's layout dict (see minimap.py)"""
    m = int(T['n_rooms'])
    return dict(sig=gen, m=m, C=np.asarray(T['mpos'])[:m].astype(float), geo=int(T['geo']), doors=np.asarray(T['doors'])[:m],
                dseam=np.asarray(T['dseam'])[:m], ns=np.asarray(T['n'])[:m], start=int(T['start']), exit=int(T['exit']), flip=False, ckey=None)
