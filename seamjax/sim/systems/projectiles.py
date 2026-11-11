"""Bullets: flying, and what they hit. The player's bullets (and melee swings) hit enemies and the Hunter; enemy bullets hit the
player. A bullet carries its weapon's damage, how many more enemies it may pierce, and whether it passes obstacles (ghost)."""
import jax, jax.numpy as jnp
from seamjax.core import pool
from seamjax.core.config import (MAX_ENEMIES as NE, MAX_PLAYER_BULLETS as NB, PLAYER_R, ENEMY_BULLET_R, HIT_FLASH, HUNTER_R, HUNTER_HP,
                                 HUNTER_RESPAWN, HUNTER_KILL_BONUS)
from seamjax.sim.systems.enemies import radius
from seamjax.core.geometry.collide import inside_poly, hits_obstacles
from seamjax.sim.queries import room_geom
from seamjax.sim.state import class_of


def fly(b, dt, r, sides, n, ob):
    """move a bullet pool one step; bullets that run out of life, leave the room or hit an obstacle (unless ghost) die"""
    x, y, life = b.x+b.vx*dt, b.y+b.vy*dt, b.life-dt
    hit = jax.vmap(lambda a, c, g: ~inside_poly(sides, n, a, c, 0.) | (hits_obstacles(a, c, r, ob) & ~g))(x, y, b.ghost)
    return b.replace(x=x, y=y, life=life, alive=b.alive & (life > 0) & ~hit)


def _within_arc(dx, dy, aim, arc):
    return jnp.abs(jnp.mod(jnp.arctan2(dy, dx)-aim+jnp.pi, 2*jnp.pi)-jnp.pi) <= arc/2


def player_bullet_hits(s, ev, dt, swung=False):
    """the player's bullets against the enemies of the room and the Hunter: each bullet hits at most one target a frame (the
    first in slot order) and never the same one twice; it stops unless it can still pierce. A melee swing (swung) hits every
    target within reach and arc of the aim. Returns (s, ev, mask of enemies killed this step)."""
    room = s.room; pb = s.pb; H = s.hunter; P = s.player
    bx, by, alive = pb.x, pb.y, pb.alive; R = class_of(s, 'w_radius')
    reach, arc, mdmg = class_of(s, 'w_reach'), class_of(s, 'w_arc'), class_of(s, 'w_damage')
    # ---- vs enemies
    en = s.enemies[room]
    ealive = en.alive; er = radius(en); vuln = ealive & (en.telegraph <= 0)
    dist = jnp.hypot(bx[:, None]-en.x[None, :], by[:, None]-en.y[None, :])
    already = ((pb.hits[:, None] >> jnp.arange(NE)[None, :]) & 1) > 0
    hit = (dist < er[None, :]+R) & vuln[None, :] & alive[:, None] & ~already
    first = jnp.argmax(hit, axis=1); anyh = jnp.any(hit, axis=1)
    onehot = jax.nn.one_hot(first, NE)*anyh[:, None]
    dmg = (onehot*pb.damage[:, None]).sum(0)
    alive = alive & ~(anyh & (pb.pierce <= 0))
    pierce = pb.pierce-anyh.astype(jnp.int32); hits = pb.hits | jnp.where(anyh, 1 << first, 0)
    mx, my = en.x-P.x, en.y-P.y
    mhit = swung & vuln & (jnp.hypot(mx, my) < reach+er) & _within_arc(mx, my, P.aim, arc)
    dmg = dmg + jnp.where(mhit, mdmg, 0.)
    # ---- vs hunter
    hbit = 1 << NE
    hdist = jnp.hypot(bx-H.x, by-H.y)
    hvuln = H.here & (H.room == room) & (H.dead <= 0)
    hh = (hdist < HUNTER_R+R) & alive & hvuln & ((hits & hbit) == 0)
    hfirst = jnp.argmax(hh); hany = jnp.any(hh)
    hh1 = jax.nn.one_hot(hfirst, NB)*hany
    hdmg = jnp.sum(hh1*pb.damage)
    alive = alive & ~((hh1 > .5) & (pierce <= 0))
    pierce = pierce-(hh1 > .5).astype(jnp.int32); hits = hits | jnp.where(hh1 > .5, hbit, 0)
    hmx, hmy = H.x-P.x, H.y-P.y
    hm = swung & hvuln & (jnp.hypot(hmx, hmy) < reach+HUNTER_R) & _within_arc(hmx, hmy, P.aim, arc)
    hdmg = hdmg + jnp.where(hm, mdmg, 0.)
    hhp = H.hp-hdmg; hkill = hvuln & (hhp <= 0) & (hdmg > 0)
    # ---- apply
    ehp = en.hp-dmg; ekill = ealive & (ehp <= 0) & (dmg > 0)
    flash = jnp.where(dmg > 0, HIT_FLASH, jnp.maximum(en.flash-dt, 0.))
    nkill = jnp.sum(ekill).astype(jnp.int32)
    s = s.replace(pb=pb.replace(alive=alive, pierce=pierce, hits=hits),
                  enemies=pool.set_row(s.enemies, room, en.replace(hp=ehp, flash=flash, alive=ealive & ~ekill)),
                  hunter=H.replace(hp=jnp.where(hkill, HUNTER_HP, hhp), dead=jnp.where(hkill, HUNTER_RESPAWN, H.dead), here=jnp.where(hkill, False, H.here)),
                  kills=s.kills+nkill+HUNTER_KILL_BONUS*hkill.astype(jnp.int32))
    return s, ev.replace(killed=nkill, hit_hunter=(hany | hm).astype(jnp.int32)), ekill


def enemy_bullets(s, dt):
    """move the enemies' bullets; returns (s, a bullet hit the player)"""
    eb = fly(s.eb, dt, ENEMY_BULLET_R, *room_geom(s, s.room))
    ehit = eb.alive & (jnp.hypot(eb.x-s.player.x, eb.y-s.player.y) < PLAYER_R+ENEMY_BULLET_R)
    return s.replace(eb=eb.replace(alive=eb.alive & ~ehit)), jnp.any(ehit)
