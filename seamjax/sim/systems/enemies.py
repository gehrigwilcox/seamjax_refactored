"""
Enemies: spawning when you first enter a room-section, shooting, and moving. Only the current room's enemies act.

Two kinds: chasers run straight at you; shooters keep their distance (back off, close in, or strafe) and fire.
An enemy is telegraphed for TELEGRAPH seconds after it spawns: shown, but it neither moves, shoots nor takes damage.
"""
import jax, jax.numpy as jnp
from jax import lax
from seamjax.difficulty import difficulty, FLOOR_CAP
from seamjax.core import pool
from seamjax.core.ops import argsort_small
from seamjax.core.config import (MAX_ENEMIES as NE, N_SPAWN_POINTS, PLAYER_R, CHASER_R, SHOOTER_R, CHASER_HP, SHOOTER_HP, CHASER_SPEED,
                                 CHASER_SPEED_PER_FLOOR, SHOOTER_SPEED, SHOOTER_RETREAT, SHOOTER_APPROACH, SHOOTER_STRAFE, SHOOTER_FIRST_SHOT,
                                 SHOOTER_FIRST_SHOT_RAND, SHOOTER_RELOAD, SHOOTER_RELOAD_RAND, ENEMY_BULLET_SPEED, ENEMY_BULLET_LIFE,
                                 TELEGRAPH, MAX_SPAWN, AREA_PER_ENEMY, SPAWN_MIN, SPAWN_MAX)
from seamjax.core.types import Enemies, Bullets
from seamjax.core.geometry.collide import push_out
from seamjax.sim.queries import room_geom, eff_sec


def radius(en):
    return jnp.where(en.shooter, SHOOTER_R, CHASER_R)

def active(en):
    """alive and done being telegraphed"""
    return en.alive & (en.telegraph <= 0)


def spawn(s, key, sec):
    """fill section `sec` of the current room with enemies: as many as the floor wants and the section's area allows,
    at pre-validated spawn points not too close to you (fewer rather than an ambush)"""
    room = s.room; dun = s.dun
    k0, k1, k2, k3, k4 = jax.random.split(key, 5)
    floor = jnp.minimum(dun.floor, FLOOR_CAP)
    isx = room == dun.exit
    want = 2 + jnp.floor(floor*.8).astype(jnp.int32) + jax.random.randint(k0, (), 0, 3) + jnp.where(isx, 2, 0)
    pts = dun.spawn[room]
    ok0 = dun.spawn_ok[room] > 0
    mine = ok0 & (eff_sec(s, room, dun.spawn_sec[room]) == sec)
    frac = jnp.where(jnp.any(ok0), jnp.sum(mine)/jnp.maximum(jnp.sum(ok0), 1), 1.)          # a section only holds its share of the room
    cap = jnp.maximum(1, jnp.round(dun.area[room]*frac/AREA_PER_ENEMY).astype(jnp.int32)) + jnp.where(isx, 1, 0)
    cnt = jnp.minimum(jnp.minimum(MAX_SPAWN, want), cap)
    minD = jnp.clip(dun.radius[room]*.55, SPAWN_MIN, SPAWN_MAX)
    dpl = jnp.hypot(pts[:, 0]-s.player.x, pts[:, 1]-s.player.y)
    okp = mine & (dpl >= minD*.6)
    score = jax.random.uniform(k1, (N_SPAWN_POINTS,)) + jnp.where(dpl >= minD, 0., -5.) + jnp.where(okp, 0., -10.)
    order = argsort_small(-score)[:NE]
    shooter = jax.random.uniform(k3, (NE,)) < difficulty(floor)['shooter_p']
    new = Enemies(x=pts[order, 0], y=pts[order, 1], hp=jnp.where(shooter, SHOOTER_HP, CHASER_HP), shooter=shooter,
                  fire=SHOOTER_FIRST_SHOT + jax.random.uniform(k4, (NE,))*SHOOTER_FIRST_SHOT_RAND, telegraph=jnp.full(NE, TELEGRAPH),
                  flash=jnp.zeros(NE), alive=okp[order] & (jnp.arange(NE) < cnt))
    rows = pool.fill(s.enemies[room], new)                           # into free slots: the other section's enemies stay
    return s.replace(enemies=pool.set_row(s.enemies, room, rows))


def shoot(s, key, dt):
    """shooters count down and fire at you"""
    room = s.room; en = s.enemies[room]
    act = active(en)
    edx, edy = s.player.x-en.x, s.player.y-en.y; ed = jnp.hypot(edx, edy)+1e-9
    efire = en.fire-jnp.where(act & en.shooter, dt, 0.)
    wantf = act & en.shooter & (efire <= 0)
    efire = jnp.where(wantf, SHOOTER_RELOAD+jax.random.uniform(key, (NE,))*SHOOTER_RELOAD_RAND, efire)
    def place(i, eb):
        new = Bullets(x=en.x[i], y=en.y[i], vx=edx[i]/ed[i]*ENEMY_BULLET_SPEED, vy=edy[i]/ed[i]*ENEMY_BULLET_SPEED, life=ENEMY_BULLET_LIFE, alive=True)
        return pool.spawn(eb, new, wantf[i])
    eb = lax.fori_loop(0, NE, place, s.eb)
    return s.replace(eb=eb, enemies=pool.set_row(s.enemies, room, en.replace(fire=efire)))


def move(s, dt):
    """steer, collide with the room, keep apart; telegraphs count down. Returns (s, an enemy touches you)"""
    room = s.room; en = s.enemies[room]; floor = jnp.minimum(s.dun.floor, FLOOR_CAP).astype(jnp.float32)
    sides, n, ob = room_geom(s, room)
    act = active(en); spawning = en.alive & (en.telegraph > 0); er = radius(en)
    edx, edy = s.player.x-en.x, s.player.y-en.y; ed = jnp.hypot(edx, edy)+1e-9
    spc = CHASER_SPEED+floor*CHASER_SPEED_PER_FLOOR
    cdx, cdy = edx/ed*spc, edy/ed*spc
    sp = SHOOTER_SPEED
    retreat = ed < SHOOTER_RETREAT; approach = ed > SHOOTER_APPROACH
    sdx = jnp.where(retreat, -edx/ed*sp, jnp.where(approach, edx/ed*sp, -edy/ed*sp*SHOOTER_STRAFE))
    sdy = jnp.where(retreat, -edy/ed*sp, jnp.where(approach, edy/ed*sp, edx/ed*sp*SHOOTER_STRAFE))
    vx_ = jnp.where(en.shooter, sdx, cdx); vy_ = jnp.where(en.shooter, sdy, cdy)
    nx = en.x+jnp.where(act, vx_*dt, 0.); ny = en.y+jnp.where(act, vy_*dt, 0.)
    nx2, ny2 = jax.vmap(lambda a, b, r: push_out(a, b, r, sides, n, ob))(nx, ny, er)
    nx = jnp.where(act, nx2, nx); ny = jnp.where(act, ny2, ny)
    contact = jnp.any(act & (ed < er+PLAYER_R))
    # separation
    ddx, ddy = nx[None, :]-nx[:, None], ny[None, :]-ny[:, None]; dd = jnp.hypot(ddx, ddy)
    m = er[:, None]+er[None, :]
    pair = (dd > 0) & (dd < m) & act[:, None] & act[None, :] & ~jnp.eye(NE, dtype=bool)
    kk = jnp.where(pair, (m-dd)/2/jnp.maximum(dd, 1e-9), 0.)
    nx = nx - (ddx*kk).sum(1); ny = ny - (ddy*kk).sum(1)
    tele = jnp.maximum(jnp.where(spawning, en.telegraph-dt, en.telegraph), 0.)
    return s.replace(enemies=pool.set_row(s.enemies, room, en.replace(x=nx, y=ny, telegraph=tele))), contact
