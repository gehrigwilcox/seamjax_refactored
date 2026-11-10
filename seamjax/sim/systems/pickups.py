"""Health pickups: killed enemies sometimes drop one where they died; walk over one to heal (not above MAX_HP)."""
import jax, jax.numpy as jnp
from jax import lax
from seamjax.difficulty import difficulty
from seamjax.core import pool
from seamjax.core.config import MAX_ENEMIES as NE, MAX_PICKUPS as NPK, MAX_HP, PICKUP_R
from seamjax.core.types import Pickups


def drop(s, killed, key):
    """each enemy in `killed` (mask over the room's enemy slots) drops a pickup with the floor's drop chance"""
    room = s.room; en = s.enemies[room]
    d = killed & (jax.random.uniform(key, (NE,)) < difficulty(s.dun.floor)['drop_p'])
    def put(i, pk): return pool.spawn(pk, Pickups(x=en.x[i], y=en.y[i], alive=True), d[i])
    return s.replace(pickups=pool.set_row(s.pickups, room, lax.fori_loop(0, NE, put, s.pickups[room])))


def take(s, ev):
    room = s.room; x, y = s.player.x, s.player.y
    def one(i, c):
        pk, hp, got = c
        near = pk.alive[i] & (jnp.hypot(pk.x[i]-x, pk.y[i]-y) < PICKUP_R) & (hp < MAX_HP)
        return pk.replace(alive=jnp.where(near, pk.alive.at[i].set(False), pk.alive)), hp+near.astype(jnp.int32), got+near.astype(jnp.int32)
    pk, hp, got = lax.fori_loop(0, NPK, one, (s.pickups[room], s.player.hp, jnp.int32(0)))
    return s.replace(pickups=pool.set_row(s.pickups, room, pk), player=s.player.replace(hp=hp)), ev.replace(healed=got)
