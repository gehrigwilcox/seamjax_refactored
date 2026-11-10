"""Where things go in a finished room: the exit pad, the enemy spawn points and the start position. All reachable."""
import jax, jax.numpy as jnp
from seamjax.core.config import CX, CY, N_SPAWN_POINTS as NSP, GRID_W as GW, GRID_H as GH, GRID_CELL as CS, ROOM_RC as RC
from seamjax.core.types import AXIS_NONE, AXIS_H
from seamjax.core.geometry.collide import inside_poly, hits_obstacles, side_dist
from seamjax.gen.shape import uniform
from seamjax.gen.grid import at


def _cells():
    xs_ = (jnp.arange(GW)+.5)*CS; ys_ = (jnp.arange(GH)+.5)*CS
    return jnp.meshgrid(xs_, ys_)


def exit_pad(key, sh, ob, reach, is_exit):
    """a random reachable spot well inside the room and clear of obstacles; else the deepest such cell. (0, 0) outside the exit room"""
    kp = jax.random.split(jax.random.fold_in(key, 77), 3)
    px = uniform(kp[0], (80,), CX-RC, CX+RC); py = uniform(kp[1], (80,), CY-RC, CY+RC)
    okp = jax.vmap(lambda x, y: inside_poly(sh.sides, sh.n, x, y, 90.) & ~hits_obstacles(x, y, 44., ob) & at(reach, x, y))(px, py)
    XX, YY = _cells()
    valid = jnp.arange(sh.sides.mid.shape[0]) < sh.n
    depth = jax.vmap(jax.vmap(lambda x, y: jnp.min(jnp.where(valid, -side_dist(sh.sides, x, y), 1e9))))(XX, YY)
    clr = jax.vmap(jax.vmap(lambda x, y: ~hits_obstacles(x, y, 44., ob)))(XX, YY)
    fc = jnp.argmax(jnp.where(reach & clr, depth, -1e9))
    fb = jnp.stack([XX.reshape(-1)[fc], YY.reshape(-1)[fc]])
    j = jnp.argmax(okp); pad = jnp.where(jnp.any(okp), jnp.stack([px[j], py[j]]), fb)
    return jnp.where(is_exit, pad, 0.)


def section_of(wall, x, y):
    return jnp.where(wall.axis == AXIS_NONE, 0, jnp.where(wall.axis == AXIS_H, y >= wall.pos, x >= wall.pos)).astype(jnp.int32)


def spawn_points(key, sh, ob, reach, wall, pad, is_exit):
    """NSP reachable spawn points away from walls, obstacles and the exit pad, shared between the two sections.
    Returns (points [NSP, 2], valid, section)"""
    ksp = jax.random.split(jax.random.fold_in(key, 99), 2)
    sx_ = CX+(uniform(ksp[0], (600,))*2-1)*sh.radius; sy_ = CY+(uniform(ksp[1], (600,))*2-1)*sh.radius
    oks = jax.vmap(lambda x, y: inside_poly(sh.sides, sh.n, x, y, 40.) & ~hits_obstacles(x, y, 24., ob) & at(reach, x, y))(sx_, sy_)
    oks = oks & ((jnp.hypot(sx_-pad[0], sy_-pad[1]) > 70) | ~is_exit)
    sec_all = section_of(wall, sx_, sy_)
    rank = jnp.where(sec_all == 0, jnp.cumsum(oks & (sec_all == 0)), jnp.cumsum(oks & (sec_all == 1)))
    order = jnp.argsort(jnp.where(oks, rank*2+sec_all, 100000))[:NSP]          # alternate the sections
    spawn = jnp.stack([sx_[order], sy_[order]], 1)
    return spawn, oks[order].astype(jnp.int32), section_of(wall, spawn[:, 0], spawn[:, 1])


def start_position(reach_first):
    """the reachable cell nearest the centre, in the section of the first door"""
    XX, YY = _cells()
    ci = jnp.argmin(jnp.where(reach_first, jnp.hypot(XX-CX, YY-CY), 1e9))
    return jnp.stack([XX.reshape(-1)[ci], YY.reshape(-1)[ci]])
