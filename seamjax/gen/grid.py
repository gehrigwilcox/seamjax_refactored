"""
The coarse walkability grid of a room (GRID_W x GRID_H cells of GRID_CELL px): where the player fits, and which cells can
be reached from which doors. Used to keep every door, exit pad and spawn point reachable.
"""
import numpy as np, jax, jax.numpy as jnp
from jax import lax
from seamjax.core.config import CX, CY, PLAYER_R, MAX_SIDES as NS, GRID_W as GW, GRID_H as GH, GRID_CELL as CS, ENTRY_INSET
from seamjax.core.geometry.collide import free_spot

_XS = (np.arange(GW)+.5)*CS; _YS = (np.arange(GH)+.5)*CS              # numpy: nothing may start the JAX backend at import time
X, Y = np.meshgrid(_XS, _YS)                                         # cell centres, [GH, GW]


def free_grid(side, n, ob):
    """[GH, GW] cells where the player (plus a margin) fits"""
    return jax.vmap(jax.vmap(lambda x, y: free_spot(x, y, PLAYER_R+4, side, n, ob)))(X, Y)

def door_entries(side):
    """where you stand just after coming in through each side's door"""
    return side.door[:, 0]-side.normal[:, 0]*ENTRY_INSET, side.door[:, 1]-side.normal[:, 1]*ENTRY_INSET

def flood(free, side, open_mask):
    """cells reachable from the doors in open_mask (flood fill by repeated dilation); connected = all those doors in one piece.
    With no door at all it floods from the room centre."""
    ex, ey = door_entries(side)
    has = jnp.any(open_mask)
    ex = jnp.where(open_mask, ex, jnp.where(has, 1e9, CX)); ey = jnp.where(open_mask, ey, jnp.where(has, 1e9, CY))
    d = jnp.where(free[None], jnp.hypot(X[None]-ex[:, None, None], Y[None]-ey[:, None, None]), 1e9)       # [NS, GH, GW]
    snap = jnp.argmin(d.reshape(NS, -1), axis=1)                                                         # nearest free cell per door
    live = open_mask | ~has
    first = jnp.argmax(live)
    reach = jnp.zeros(GH*GW, bool).at[snap[first]].set(True).reshape(GH, GW)
    def dil(i, r):
        p = jnp.pad(r, 1); nb = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        return free & (r | nb)
    reach = lax.fori_loop(0, GW+GH, dil, reach)
    connected = jnp.all(jnp.where(live, reach.reshape(-1)[snap], True))
    return reach, connected

def reach_grid(side, n, ob, open_mask):
    return flood(free_grid(side, n, ob), side, open_mask)

def at(reach, x, y):
    """the grid value of the cell containing (x, y)"""
    return reach[jnp.clip((y//CS).astype(jnp.int32), 0, GH-1), jnp.clip((x//CS).astype(jnp.int32), 0, GW-1)]
