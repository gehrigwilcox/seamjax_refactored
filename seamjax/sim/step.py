"""
One frame of the game: the systems, in order.

    step(s, action) -> (s, ev)      action = [move_x, move_y, aim_x, aim_y] in screen space; ev = StepEvents of this frame

A frame either walks you through a door (transit) or plays the room (combat); both are computed and one is kept, which is
what vmap does anyway. The order of the systems, and of the random keys they draw, is part of the game's behaviour: the
golden trace (tests/golden_test.py) pins it down.
"""
import jax, jax.numpy as jnp
from seamjax.core import struct
from seamjax.sim.state import cond, hurt, no_events
from seamjax.sim.systems import player, hunter, transit, enemies, projectiles, pickups, rooms

DT = 1/60.


def step(s, action, dt=DT):
    ev = no_events()
    s = s.replace(player=s.player.replace(inv=s.player.inv-dt))
    s, ev = hunter.tick(s, ev, dt)
    s, side = player.move(s, action, dt)
    s_door, ev_door = transit.transit(s, ev, jnp.maximum(side, 0))
    s_room, ev_room = combat(s, ev, action, dt)
    s, ev = struct.where(side >= 0, (s_door.replace(dun=None), ev_door), (s_room.replace(dun=None), ev_room))
    return s.replace(dun=s_door.dun, t=s.t+dt, steps=s.steps+1), ev


def combat(s, ev, action, dt):
    """everything that happens in a room during a frame you stay in it"""
    k_fire, key = jax.random.split(s.key); key, k_drop = jax.random.split(key); s = s.replace(key=key)
    s, swung = player.shoot(s, action, dt)
    s, ev, killed = projectiles.player_bullet_hits(s, ev, dt, swung)
    s = pickups.drop(s, killed, k_drop)
    s = enemies.shoot(s, k_fire, dt)
    s, contact = enemies.move(s, dt)
    s, shot = projectiles.enemy_bullets(s, dt)
    s, ev = cond(contact | shot, hurt, s, ev)
    s, ev = pickups.take(s, ev)
    s, ev = rooms.open_gates(s, ev)
    return rooms.stairs(s, ev)
