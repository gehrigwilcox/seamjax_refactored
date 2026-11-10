"""Bullets: flying, and what they hit. The player's bullets hit enemies and the Hunter; enemy bullets hit the player."""
import jax, jax.numpy as jnp
from seamjax.core import pool
from seamjax.core.config import (MAX_ENEMIES as NE, MAX_PLAYER_BULLETS as NB, BULLET_R, PLAYER_R, ENEMY_BULLET_R, CHASER_R, SHOOTER_R,
                                 HIT_FLASH, HUNTER_R, HUNTER_HP, HUNTER_RESPAWN, HUNTER_KILL_BONUS)
from seamjax.core.geometry.collide import blocked
from seamjax.sim.queries import room_geom


def fly(b, dt, r, sides, n, ob):
    """move a bullet pool one step; bullets that run out of life, leave the room or hit an obstacle die"""
    x, y, life = b.x+b.vx*dt, b.y+b.vy*dt, b.life-dt
    hit = jax.vmap(lambda a, c: blocked(a, c, r, sides, n, ob))(x, y)
    return b.replace(x=x, y=y, life=life, alive=b.alive & (life > 0) & ~hit)


def player_bullet_hits(s, ev, dt):
    """the player's bullets against the enemies of the room and the Hunter: one hit per bullet (the first target in slot order).
    Returns (s, ev, mask of enemies killed this step)."""
    room = s.room; pb = s.pb; H = s.hunter
    bx, by, alive = pb.x, pb.y, pb.alive
    # ---- vs enemies
    en = s.enemies[room]
    ealive = en.alive; er = jnp.where(en.shooter, SHOOTER_R, CHASER_R); vuln = ealive & (en.telegraph <= 0)
    dist = jnp.hypot(bx[:, None]-en.x[None, :], by[:, None]-en.y[None, :])
    hit = (dist < er[None, :]+BULLET_R) & vuln[None, :] & alive[:, None]
    first = jnp.argmax(hit, axis=1); anyh = jnp.any(hit, axis=1)
    onehot = jax.nn.one_hot(first, NE)*anyh[:, None]
    dmg = onehot.sum(0)
    alive = alive & ~anyh
    # ---- vs hunter
    hdist = jnp.hypot(bx-H.x, by-H.y)
    hvuln = H.here & (H.room == room) & (H.dead <= 0)
    hh = (hdist < HUNTER_R+BULLET_R) & alive & hvuln
    hfirst = jnp.argmax(hh); hany = jnp.any(hh)
    hh1 = jax.nn.one_hot(hfirst, NB)*hany
    hdmg = jnp.sum(hh1)
    alive = alive & ~(hh1 > .5)
    hhp = H.hp-hdmg; hkill = hvuln & (hhp <= 0) & (hdmg > 0)
    # ---- apply
    ehp = en.hp-dmg; ekill = ealive & (ehp <= 0) & (dmg > 0)
    flash = jnp.where(dmg > 0, HIT_FLASH, jnp.maximum(en.flash-dt, 0.))
    nkill = jnp.sum(ekill).astype(jnp.int32)
    s = s.replace(pb=pb.replace(alive=alive),
                  enemies=pool.set_row(s.enemies, room, en.replace(hp=ehp, flash=flash, alive=ealive & ~ekill)),
                  hunter=H.replace(hp=jnp.where(hkill, HUNTER_HP, hhp), dead=jnp.where(hkill, HUNTER_RESPAWN, H.dead), here=jnp.where(hkill, False, H.here)),
                  kills=s.kills+nkill+HUNTER_KILL_BONUS*hkill.astype(jnp.int32))
    return s, ev.replace(killed=nkill, hit_hunter=hany.astype(jnp.int32)), ekill


def enemy_bullets(s, dt):
    """move the enemies' bullets; returns (s, a bullet hit the player)"""
    eb = fly(s.eb, dt, ENEMY_BULLET_R, *room_geom(s, s.room))
    ehit = eb.alive & (jnp.hypot(eb.x-s.player.x, eb.y-s.player.y) < PLAYER_R+ENEMY_BULLET_R)
    return s.replace(eb=eb.replace(alive=eb.alive & ~ehit)), jnp.any(ehit)
