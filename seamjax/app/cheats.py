"""Debug-mode changes to the game state (keys N H K O T G in the viewer). Pure functions of a GameState."""
import jax.numpy as jnp
from seamjax.sim.state import class_of
from seamjax.core.types import WALL_GATED


def to_stairs(s):
    """stand on the exit pad, with the exit room cleared"""
    d = s.dun; ex = d.exit; pd = d.pad[ex]
    return s.replace(room=ex, player=s.player.replace(x=pd[0], y=pd[1], inv=jnp.float32(9.)), spawned=jnp.ones_like(s.spawned),
                     enemies=s.enemies.replace(alive=s.enemies.alive.at[ex].set(False)))

def heal(s):
    return s.replace(player=s.player.replace(hp=class_of(s, 'max_hp')))

def kill_room(s):
    return s.replace(enemies=s.enemies.replace(alive=s.enemies.alive.at[s.room].set(False)))

def open_gate(s):
    return s.replace(wall_open=s.wall_open.at[s.room].set(s.dun.wall_kind[s.room] == WALL_GATED), spawned=s.spawned.at[s.room].set(True))

def toggle_hunter(s):
    """take the Hunter out of the game (dead for a very long time), or bring it back"""
    return s.replace(hunter=s.hunter.replace(dead=jnp.where(s.hunter.dead > 1e8, 0., 1e9).astype(jnp.float32)))

def god(s):
    return s.replace(player=s.player.replace(hp=class_of(s, 'max_hp'), inv=jnp.float32(9.)))

KEYS = dict(stairs=to_stairs, heal=heal, kill=kill_room, open=open_gate, hunter=toggle_hunter, god=god)
