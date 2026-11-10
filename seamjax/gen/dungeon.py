"""One dungeon: the layout (layout.py), then every room (room.py), then the gating that keeps every section reachable."""
import jax, jax.numpy as jnp
from seamjax.difficulty import difficulty
from seamjax.core.config import MAX_ROOMS as R
from seamjax.core.types import Dungeon
from seamjax.gen import layout
from seamjax.gen.room import design_room


def gen_dungeon(SP, key, floor):
    """a random dungeon for `floor` from the tiling tables SP (specs.load_specs). Pure JAX: jit / vmap it."""
    k = jax.random.split(key, 12)
    dif = difficulty(floor)
    sid = layout.choose_spec(SP, k[0], dif['cat_p'])
    t = layout.tiling(SP, sid)
    sel = layout.pick_rooms(t, k[1:6], dif['size'])
    c = layout.compact(t, sel, layout.open_doors(t, sel, k[6:10]))
    dist, start, depth, exit_ = layout.start_and_exit(c, t['hub'], k[10], k[11])
    # every room (padding slots get a triangle that is zeroed below)
    want_wall = jax.random.uniform(jax.random.fold_in(key, 6), (R,)) < dif['wall_p']
    rooms = jax.vmap(design_room)(jax.random.split(jax.random.fold_in(key, 5), R), jnp.maximum(c.n, 3), jnp.arange(R) == exit_, c.doors >= 0, want_wall)
    rooms = jax.tree_util.tree_map(lambda a: jnp.where(c.live.reshape((R,)+(1,)*(a.ndim-1)), a, jnp.zeros_like(a)), rooms)
    wall = rooms.wall
    wkind, ndist = layout.fix_sections(key, c, start, wall.kind, wall.door_sec)
    ob = rooms.obstacles
    return Dungeon(n=c.n, doors=c.doors.astype(jnp.int32), arrive=c.arrive.astype(jnp.int32), flip=c.flip.astype(jnp.int32),
                   sides=rooms.sides, obstacles=ob.replace(kind=jnp.where(c.live[:, None], ob.kind, -1)), pad=rooms.pad, radius=rooms.radius,
                   inradius=rooms.inradius, area=rooms.area, spawn=rooms.spawn, spawn_ok=rooms.spawn_ok, spawn_sec=rooms.spawn_sec,
                   wall_axis=wall.axis, wall_pos=wall.pos, wall_kind=wkind, door_sec=wall.door_sec.astype(jnp.int32), depth=depth.astype(jnp.int32),
                   map_pos=jnp.where(c.live[:, None], SP['cvec'][sid][c.order], 0),
                   door_seam=jnp.where(c.doors >= 0, SP['seam'][sid][c.order], 0).astype(jnp.int32),
                   n_rooms=c.m.astype(jnp.int32), start=start, startpos=rooms.startpos[start], exit=exit_, floor=jnp.asarray(floor, jnp.int32),
                   category=SP['cat'][sid]+0*sid, spec=sid, holonomy=SP['hol'][sid], dist=dist.astype(jnp.int32), node_dist=ndist.astype(jnp.int32))


def gen_source(SP):
    """source(key, floor, idx=None) for sim.game.Game: a fresh GPU-generated dungeon for that floor"""
    def src(key, floor, idx=None):
        return gen_dungeon(SP, key, floor)
    return src
