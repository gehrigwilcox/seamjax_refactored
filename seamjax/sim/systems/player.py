"""The player: walking (controls are in screen space, the room may be turned or mirrored) and the weapon of their class
(core/kinds.py: a gun with pellets, spread, pierce and pass-through, or a melee swing)."""
import jax.numpy as jnp
from seamjax.core import pool
from seamjax.core import kinds
from seamjax.core.config import DOOR_GAP, PLAYER_R, AIM_DEADZONE, SWING_SHOW
from seamjax.core.geometry.view import rot_inv
from seamjax.core.geometry.collide import push_out_obstacles, clamp_through_doors
from seamjax.sim.queries import room_geom
from seamjax.sim.state import bullet, class_of
from seamjax.sim.systems.projectiles import fly

DOOR_PASS = DOOR_GAP - PLAYER_R*.2 - .01            # how far off a door's centre you can still walk through it


def move(s, action, dt):
    """walk by the move stick. Returns (s, side being walked out through or -1)"""
    sides, n, ob = room_geom(s, s.room)
    mv = jnp.clip(action[:2], -1, 1); ml = jnp.hypot(mv[0], mv[1]); mv = jnp.where(ml > 1, mv/ml, mv)
    mx, my = rot_inv(mv[0], mv[1], s.view_angle, s.mirror)
    speed = class_of(s, 'speed')
    x, y = s.player.x+mx*speed*dt, s.player.y+my*speed*dt
    x, y = push_out_obstacles(x, y, PLAYER_R, ob)
    x, y, ex = clamp_through_doors(x, y, PLAYER_R, DOOR_PASS, sides, n, s.dun.doors[s.room])
    return s.replace(player=s.player.replace(x=x, y=y)), ex


def shoot(s, action, dt):
    """aim with the aim stick; holding it fires (one shot per w_delay): a gun spawns its pellets fanned over its spread, a melee
    weapon swings instead. Moves the player's bullets. Returns (s, swung this frame)"""
    P = s.player; w = lambda name: class_of(s, 'w_'+name)
    ax, ay = rot_inv(action[2], action[3], s.view_angle, s.mirror)
    firing = jnp.hypot(action[2], action[3]) > AIM_DEADZONE
    aim = jnp.where(firing, jnp.arctan2(ay, ax), P.aim)
    fire = P.fire-dt
    shoot = firing & (fire <= 0)
    fire = jnp.where(shoot, w('delay'), fire)
    melee = w('melee'); n = w('pellets'); spread = w('spread'); muzzle = w('muzzle'); speed = w('speed')
    p = jnp.arange(max(c.weapon.pellets for c in kinds.PLAYER_CLASSES)); one = jnp.ones(p.shape[0])
    a = aim + jnp.where(n > 1, spread*(p/jnp.maximum(n-1, 1) - .5), 0.)                # pellet p of n, evenly over the fan
    new = bullet(P.x+jnp.cos(a)*muzzle, P.y+jnp.sin(a)*muzzle, jnp.cos(a)*speed, jnp.sin(a)*speed, one*w('life'), one*w('damage'),
                 (one*w('pierce')).astype(jnp.int32), one*w('ghost') > 0).replace(alive=shoot & ~melee & (p < n), hits=jnp.zeros(p.shape[0], jnp.int32))
    pb, _ = pool.place(s.pb, new)
    pb = fly(pb, dt, w('radius'), *room_geom(s, s.room))
    swung = shoot & melee
    swing = jnp.where(swung, SWING_SHOW, jnp.maximum(P.swing-dt, 0.))
    return s.replace(player=P.replace(aim=aim, fire=fire, swing=swing), pb=pb), swung
