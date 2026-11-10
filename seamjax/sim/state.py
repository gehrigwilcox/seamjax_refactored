"""
Building blocks the systems share: empty pools, empty step events, a conditional update of the state, and taking a hit.

Every system has the signature  (s, ev, ...) -> (s, ev): it reads and returns the GameState `s` and adds what happened
to the StepEvents `ev` of this step.
"""
import jax.numpy as jnp
from jax import lax
from seamjax.core.config import MAX_ENEMIES as NE, INVULN
from seamjax.core.types import Bullets, Enemies, StepEvents


def empty_bullets(n):
    z = jnp.zeros(n); return Bullets(x=z, y=z, vx=z, vy=z, life=z, alive=jnp.zeros(n, bool))

def empty_enemies(R):
    z = jnp.zeros((R, NE)); b = jnp.zeros((R, NE), bool)
    return Enemies(x=z, y=z, hp=z, shooter=b, fire=z, telegraph=z, flash=z, alive=b)

def no_events():
    i0 = jnp.int32(0)
    return StepEvents(took=i0, newroom=i0, killed=i0, hit_hunter=i0, descended=False, shake=False, healed=i0, opened=i0)

def cond(pred, f, s, ev):
    """(s, ev) = f(s, ev) if pred else (s, ev), as a lax.cond that leaves the dungeon tables out of the branches (they never change
    during a step, and under vmap both branches are computed and selected, so carrying them would be costly)"""
    dun = s.dun
    def run(c):
        s2, ev2 = f(c[0].replace(dun=dun), c[1]); return s2.replace(dun=None), ev2
    s2, ev2 = lax.cond(pred, run, lambda c: c, (s.replace(dun=None), ev))
    return s2.replace(dun=dun), ev2

def hurt(s, ev):
    """take one hit unless still invulnerable from the last one"""
    ok = s.player.inv <= 0
    return (s.replace(player=s.player.replace(hp=s.player.hp-ok.astype(jnp.int32), inv=jnp.where(ok, INVULN, s.player.inv))),
            ev.replace(took=ev.took+ok.astype(jnp.int32)))
