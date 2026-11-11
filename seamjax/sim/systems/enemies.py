"""
Enemies: spawning when you first enter a room-section, shooting, and moving. Only the current room's enemies act.

What an enemy is comes from its kind (core/kinds.py): radius, hp, speed, behaviour (CHASE runs straight at you, SKIRMISH keeps
its distance: backs off, closes in, or strafes), and whether and how it shoots.
An enemy is telegraphed for TELEGRAPH seconds after it spawns: shown, but it neither moves, shoots nor takes damage.
"""
import jax, jax.numpy as jnp
from jax import lax
from seamjax.difficulty import difficulty, FLOOR_CAP
from seamjax.core import pool
from seamjax.core.ops import argsort_small
from seamjax.core import kinds
from seamjax.core.config import (MAX_ENEMIES as NE, N_SPAWN_POINTS, PLAYER_R, TELEGRAPH, MAX_SPAWN, AREA_PER_ENEMY, SPAWN_MIN, SPAWN_MAX, CX, CY,
                                 BOSS_TELEGRAPH, PHASE_PAUSE)
from seamjax.core.types import Enemies
from seamjax.sim.state import bullet
from seamjax.core.geometry.collide import push_out
from seamjax.sim.queries import room_geom, eff_sec, sec_point


def of(name, kind):
    """a column of the kind table (kinds.ENEMY_KINDS, read when the step is traced) for an array of kinds"""
    return jnp.asarray(getattr(kinds.table(), name))[kind]

def radius(en):
    return of('radius', en.kind)

def ranged(en):
    return of('ranged', en.kind)

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
    pad = dun.pad[room]; pad_sec = eff_sec(s, room, sec_point(dun, room, pad[0], pad[1]))
    boss_on = [isx & (dun.floor % bk.boss_every == 0) & (sec == pad_sec) for _, bk in kinds.bosses()]     # a boss floor's exit section
    if boss_on:                                                      # the boss takes a place of the room's capacity; fewer others with it
        on = jnp.any(jnp.stack(boss_on)); cnt = jnp.where(on, jnp.minimum(jnp.minimum(cnt, cap-len(boss_on)), MAX_SPAWN//2), cnt)
    minD = jnp.clip(dun.radius[room]*.55, SPAWN_MIN, SPAWN_MAX)
    dpl = jnp.hypot(pts[:, 0]-s.player.x, pts[:, 1]-s.player.y)
    okp = mine & (dpl >= minD*.6)
    score = jax.random.uniform(k1, (N_SPAWN_POINTS,)) + jnp.where(dpl >= minD, 0., -5.) + jnp.where(okp, 0., -10.)
    order = argsort_small(-score)[:NE]
    # kinds by the floor's mix: kind K-1 if u < p[K-1], else K-2 if u < p[K-1]+p[K-2], ... (one uniform draw per enemy)
    mix = difficulty(floor)['enemy_mix']; thr = jnp.cumsum(mix[::-1])[:-1]
    kind = (mix.shape[0]-1 - jnp.sum(jax.random.uniform(k3, (NE,))[:, None] >= thr[None, :], axis=1)).astype(jnp.int32)
    new = Enemies(x=pts[order, 0], y=pts[order, 1], hp=of('hp', kind), kind=kind,
                  fire=of('first_shot', kind) + jax.random.uniform(k4, (NE,))*of('first_shot_rand', kind), telegraph=jnp.full(NE, TELEGRAPH),
                  flash=jnp.zeros(NE), alive=okp[order] & (jnp.arange(NE) < cnt), phase=jnp.zeros(NE, jnp.int32), ptimer=jnp.zeros(NE))
    rows = pool.fill(s.enemies[room], new)                           # into free slots: the other section's enemies stay
    for (bi, bk), on in zip(kinds.bosses(), boss_on):                 # a boss floor: the boss waits in the exit pad's section
        safe = mine & (dpl >= minD*.6); dc = jnp.hypot(pts[:, 0]-CX, pts[:, 1]-CY)       # nearest the middle (its rings fill the room), not on you
        at = jnp.where(jnp.any(safe), jnp.argmin(jnp.where(safe, dc, 1e9)), jnp.argmax(jnp.where(mine, dpl, -1.)))
        x = jnp.where(jnp.any(mine), pts[at, 0], CX); y = jnp.where(jnp.any(mine), pts[at, 1], CY)
        rows = pool.spawn(rows, Enemies(x=x, y=y, hp=bk.hp+floor*bk.hp_per_floor, kind=bi, fire=PHASE_PAUSE, telegraph=BOSS_TELEGRAPH, flash=0.,
                                        alive=True, phase=0, ptimer=0.), on)
    return s.replace(enemies=pool.set_row(s.enemies, room, rows))


def shoot(s, key, dt):
    """ranged kinds count down and fire at you"""
    room = s.room; en = s.enemies[room]
    act = active(en); shoots = ranged(en)
    edx, edy = s.player.x-en.x, s.player.y-en.y; ed = jnp.hypot(edx, edy)+1e-9
    efire = en.fire-jnp.where(act & shoots, dt, 0.)
    wantf = act & shoots & (efire <= 0)
    efire = jnp.where(wantf, of('reload', en.kind)+jax.random.uniform(key, (NE,))*of('reload_rand', en.kind), efire)
    bspeed, blife = of('bullet_speed', en.kind), of('bullet_life', en.kind)
    one = jnp.ones(NE)
    eb, _ = pool.place(s.eb, bullet(en.x, en.y, edx/ed*bspeed, edy/ed*bspeed, blife, one, one.astype(jnp.int32)*0, one < 0).replace(
        alive=wantf, hits=jnp.zeros(NE, jnp.int32)))
    return s.replace(eb=eb, enemies=pool.set_row(s.enemies, room, en.replace(fire=efire)))


def move(s, dt):
    """steer, collide with the room, keep apart; telegraphs count down. Returns (s, an enemy touches you)"""
    room = s.room; en = s.enemies[room]; floor = jnp.minimum(s.dun.floor, FLOOR_CAP).astype(jnp.float32)
    sides, n, ob = room_geom(s, room)
    act = active(en); spawning = en.alive & (en.telegraph > 0); er = radius(en)
    edx, edy = s.player.x-en.x, s.player.y-en.y; ed = jnp.hypot(edx, edy)+1e-9
    sp = of('speed', en.kind)+floor*of('speed_per_floor', en.kind)
    # CHASE: straight at you
    cdx, cdy = edx/ed*sp, edy/ed*sp
    # SKIRMISH: back off when close, close in when far, else strafe round you
    retreat = ed < of('retreat', en.kind); approach = ed > of('approach', en.kind); strafe = of('strafe', en.kind)
    sdx = jnp.where(retreat, -edx/ed*sp, jnp.where(approach, edx/ed*sp, -edy/ed*sp*strafe))
    sdy = jnp.where(retreat, -edy/ed*sp, jnp.where(approach, edy/ed*sp, edx/ed*sp*strafe))
    beh = of('behaviour', en.kind)
    if kinds.bosses():                                               # a boss moves as its current phase says
        P = kinds.phase_table(); boss = of('boss_every', en.kind) > 0; ph = jnp.minimum(en.phase, P.below.shape[1]-1)
        bsp = jnp.asarray(P.speed)[en.kind, ph]
        cdx, cdy = jnp.where(boss, edx/ed*bsp, cdx), jnp.where(boss, edy/ed*bsp, cdy)
        sdx = jnp.where(boss, jnp.where(retreat, -edx/ed*bsp, jnp.where(approach, edx/ed*bsp, -edy/ed*bsp*strafe)), sdx)
        sdy = jnp.where(boss, jnp.where(retreat, -edy/ed*bsp, jnp.where(approach, edy/ed*bsp, edx/ed*bsp*strafe)), sdy)
        beh = jnp.where(boss, jnp.asarray(P.behaviour)[en.kind, ph], beh)
    skirmish = beh == kinds.SKIRMISH; stand = beh == kinds.STAND
    vx_ = jnp.where(stand, 0., jnp.where(skirmish, sdx, cdx)); vy_ = jnp.where(stand, 0., jnp.where(skirmish, sdy, cdy))
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
