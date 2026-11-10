"""The player: walking (controls are in screen space, the room may be turned or mirrored) and the gun."""
import jax.numpy as jnp
from seamjax.core import pool
from seamjax.core.config import (DOOR_GAP, PLAYER_R, PLAYER_SPEED, AIM_DEADZONE, FIRE_DELAY, BULLET_SPEED, BULLET_LIFE, BULLET_MUZZLE, BULLET_R)
from seamjax.core.types import Bullets
from seamjax.core.geometry.view import rot_inv
from seamjax.core.geometry.collide import push_out_obstacles, clamp_through_doors
from seamjax.sim.queries import room_geom
from seamjax.sim.systems.projectiles import fly

DOOR_PASS = DOOR_GAP - PLAYER_R*.2 - .01            # how far off a door's centre you can still walk through it


def move(s, action, dt):
    """walk by the move stick. Returns (s, side being walked out through or -1)"""
    sides, n, ob = room_geom(s, s.room)
    mv = jnp.clip(action[:2], -1, 1); ml = jnp.hypot(mv[0], mv[1]); mv = jnp.where(ml > 1, mv/ml, mv)
    mx, my = rot_inv(mv[0], mv[1], s.view_angle, s.mirror)
    x, y = s.player.x+mx*PLAYER_SPEED*dt, s.player.y+my*PLAYER_SPEED*dt
    x, y = push_out_obstacles(x, y, PLAYER_R, ob)
    x, y, ex = clamp_through_doors(x, y, PLAYER_R, DOOR_PASS, sides, n, s.dun.doors[s.room])
    return s.replace(player=s.player.replace(x=x, y=y)), ex


def shoot(s, action, dt):
    """aim with the aim stick; holding it fires (one bullet per FIRE_DELAY). Moves the player's bullets."""
    P = s.player
    ax, ay = rot_inv(action[2], action[3], s.view_angle, s.mirror)
    firing = jnp.hypot(action[2], action[3]) > AIM_DEADZONE
    aim = jnp.where(firing, jnp.arctan2(ay, ax), P.aim)
    fire = P.fire-dt
    shoot = firing & (fire <= 0)
    fire = jnp.where(shoot, FIRE_DELAY, fire)
    new = Bullets(x=P.x+jnp.cos(aim)*BULLET_MUZZLE, y=P.y+jnp.sin(aim)*BULLET_MUZZLE, vx=jnp.cos(aim)*BULLET_SPEED, vy=jnp.sin(aim)*BULLET_SPEED,
                  life=BULLET_LIFE, alive=True)
    pb = fly(pool.spawn(s.pb, new, shoot), dt, BULLET_R, *room_geom(s, s.room))
    return s.replace(player=P.replace(aim=aim, fire=fire), pb=pb)
