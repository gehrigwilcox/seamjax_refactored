"""
Bosses: enemy kinds with phases (core/kinds.py: boss_every, phases). A boss spawns in the exit room on its floors
(enemies.spawn), so the stairs wait until it is down. Its phase follows its hp: phase j starts once hp falls to phases[j].below
of the full amount (never back). Entering a phase restarts the phase timer and holds fire for PHASE_PAUSE. Movement follows
the phase (enemies.move); here the boss fires the phase's volley every `interval`: `count` bullets over `spread` radians
(2*pi = a ring), centred on you if `aimed`, else on an angle turning by `spin` per second (a spiral).

One boss per room fires at a time (the first in slot order). With no boss kinds this system is skipped entirely.
"""
import jax.numpy as jnp
from seamjax.difficulty import FLOOR_CAP
from seamjax.core import kinds, pool, struct
from seamjax.core.config import MAX_ENEMY_BULLETS as NEB, PHASE_PAUSE
from seamjax.core.types import Bullets
from seamjax.sim.systems.enemies import of, active, radius


def full_hp(s, en):
    return of('hp', en.kind)+jnp.minimum(s.dun.floor, FLOOR_CAP)*of('hp_per_floor', en.kind)


def tick(s, dt):
    if not kinds.bosses(): return s
    room = s.room; en = s.enemies[room]; P = kinds.phase_table(); NP = P.below.shape[1]
    boss = of('boss_every', en.kind) > 0
    # ---- phase: the number of thresholds (after the first phase) the hp has fallen to
    below = jnp.asarray(P.below)[en.kind]
    reached = jnp.sum((jnp.arange(NP)[None, :] > 0) & (below >= 0) & ((en.hp/full_hp(s, en))[:, None] <= below), axis=1)
    phase = jnp.where(boss, jnp.maximum(en.phase, reached), en.phase).astype(jnp.int32)
    changed = boss & (phase > en.phase)
    col = lambda name: jnp.asarray(getattr(P, name))[en.kind, phase]
    act = active(en) & boss
    ptimer = jnp.where(changed, 0., jnp.where(act, en.ptimer+dt, en.ptimer))
    fire = jnp.where(changed, PHASE_PAUSE, jnp.where(act, en.fire-dt, en.fire))
    want = act & ~changed & (fire <= 0)
    fire = jnp.where(want, col('interval'), fire)
    s = s.replace(enemies=pool.set_row(s.enemies, room, en.replace(phase=phase, ptimer=ptimer, fire=fire)))
    # ---- the volley of the first boss that fires, into free enemy-bullet slots
    b = jnp.argmax(want); go = jnp.any(want)
    count = col('count')[b]; spread = col('spread')[b]; bx, by = en.x[b], en.y[b]
    base = jnp.where(col('aimed')[b], jnp.arctan2(s.player.y-by, s.player.x-bx), ptimer[b]*col('spin')[b])
    j = jnp.arange(NEB)
    ring = spread >= 2*jnp.pi-1e-3
    a = base + jnp.where(ring, 2*jnp.pi*j/jnp.maximum(count, 1), jnp.where(count > 1, spread*(j/jnp.maximum(count-1, 1)-.5), 0.))
    v, r = col('bullet_speed')[b], radius(en)[b]
    z = jnp.zeros(NEB); i0 = jnp.zeros(NEB, jnp.int32)
    volley = Bullets(x=bx+jnp.cos(a)*r, y=by+jnp.sin(a)*r, vx=jnp.cos(a)*v, vy=jnp.sin(a)*v, life=jnp.full(NEB, col('bullet_life')[b]),
                     alive=go & (j < count), damage=z+1., pierce=i0, ghost=jnp.zeros(NEB, bool), hits=i0)
    return s.replace(eb=struct.where(go, pool.fill(s.eb, volley), s.eb))
