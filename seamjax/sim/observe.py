"""
The RL observation: egocentric, in screen space (what you see, so a mirrored room looks mirrored), no map. 194 floats:

    head (8)        your position, hp, invulnerable, sides of the room, view mirror, doors to the Hunter, floor
    sides (12 x 8)  valid, outward normal, has a door, your distance, door offset, Hunter arrives here, door on the wall's 2nd side
    enemies (8 x 5) nearest first: present, position, ranged, hp
    bullets (8 x 4) nearest enemy bullets: position, velocity
    tail (8)        Hunter visible / position / hp, exit room / pad position, room cleared
    pickup (3)      nearest health pickup: present, position
    wall (7)        inner wall: present, gated, open, normal, your distance, your side
"""
import jax, jax.numpy as jnp
from seamjax.difficulty import FLOOR_CAP
from seamjax.core.config import CX, CY, MAX_SIDES as NS, HUNTER_HP, INF
from seamjax.sim.state import class_of
from seamjax.sim.systems.roamers import warn_sides
from seamjax.core.types import AXIS_H, WALL_GATED
from seamjax.core.geometry.view import view_vec
from seamjax.core.geometry.collide import side_dist
from seamjax.core.ops import argsort_small
from seamjax.sim.systems.enemies import ranged
from seamjax.sim.queries import hnode, player_node, hunter_door, hunter_visible, sec_point


OBS_EN, OBS_EB = 8, 8
OBS_DIM = 8 + NS*8 + OBS_EN*5 + OBS_EB*4 + 8 + 3 + 7

def observe(s):
    dun = s.dun; room = s.room; n = dun.n[room]; sides = dun.sides[room]; doors = dun.doors[room]; P = s.player
    va, mir = s.view_angle, s.mirror; sc = 1/300.
    x, y = P.x-CX, P.y-CY
    vx, vy = view_vec(va, mir, x, y)
    hd_ = dun.node_dist[hnode(dun, s.hunter.room, s.hunter.sec), player_node(s)]
    hdist = jnp.where(s.hunter.dead > 0, -1., jnp.where(hd_ >= INF, -2., hd_.astype(jnp.float32)))     # -1 down, -2 cut off by a wall
    head = jnp.stack([vx*sc, vy*sc, P.hp/class_of(s, 'max_hp'), (P.inv > 0).astype(jnp.float32), n/12., mir, jnp.clip(hdist, -2, 12)/12., jnp.minimum(dun.floor, FLOOR_CAP)/FLOOR_CAP])
    # sides: valid, outward normal (screen), door, signed dist of player, door offset along the side, hunter-arrival door, wall side
    nxs, nys = jax.vmap(lambda a, b: view_vec(va, mir, a, b))(sides.normal[:, 0], sides.normal[:, 1])
    dpl = side_dist(sides, P.x, P.y)*sc
    lat = ((sides.door[:, 0]-sides.mid[:, 0])*sides.tangent[:, 0] + (sides.door[:, 1]-sides.mid[:, 1])*sides.tangent[:, 1])*sc
    valid = (jnp.arange(NS) < n).astype(jnp.float32)
    warn = ((jnp.arange(NS) == hunter_door(s)) | warn_sides(s)).astype(jnp.float32)          # the Hunter or a follower comes in here
    sf = jnp.stack([valid, nxs*valid, nys*valid, (doors >= 0)*valid, dpl*valid, lat*valid, warn*valid, (dun.door_sec[room] > 0)*valid*(dun.wall_kind[room] > 0)], 1)
    en = s.enemies[room]; alive = en.alive
    rx, ry = en.x-P.x, en.y-P.y
    ex_, ey_ = jax.vmap(lambda a, b: view_vec(va, mir, a, b))(rx, ry)
    dist = jnp.where(alive, jnp.hypot(rx, ry), 1e9); order = argsort_small(dist)[:OBS_EN]
    ef = jnp.stack([alive[order], ex_[order]*sc*alive[order], ey_[order]*sc*alive[order], ranged(en)[order], en.hp[order]/4.*alive[order]], 1)
    eb = s.eb; balive = eb.alive
    brx, bry = eb.x-P.x, eb.y-P.y
    bx_, by_ = jax.vmap(lambda a, b: view_vec(va, mir, a, b))(brx, bry)
    bvx, bvy = jax.vmap(lambda a, b: view_vec(va, mir, a, b))(eb.vx, eb.vy)
    bd = jnp.where(balive, jnp.hypot(brx, bry), 1e9); bo = argsort_small(bd)[:OBS_EB]
    bf = jnp.stack([bx_[bo]*sc*balive[bo], by_[bo]*sc*balive[bo], bvx[bo]/190.*balive[bo], bvy[bo]/190.*balive[bo]], 1)
    # hunter + exit
    H = s.hunter
    hrx, hry = view_vec(va, mir, H.x-P.x, H.y-P.y)
    hv = hunter_visible(s)
    pad = dun.pad[room]; prx, pry = view_vec(va, mir, pad[0]-P.x, pad[1]-P.y)
    isx = room == dun.exit
    tail = jnp.stack([hv*1., hv*hrx*sc, hv*hry*sc, hv*H.hp/HUNTER_HP, isx*1., isx*prx*sc, isx*pry*sc, (~jnp.any(alive))*1.])
    pk = s.pickups[room]; pa = pk.alive
    pdist = jnp.where(pa, jnp.hypot(pk.x-P.x, pk.y-P.y), 1e9); pi_ = jnp.argmin(pdist)
    prx2, pry2 = view_vec(va, mir, pk.x[pi_]-P.x, pk.y[pi_]-P.y); pany = jnp.any(pa)
    ptail = jnp.stack([pany*1., pany*prx2*sc, pany*pry2*sc])
    # inner wall: present, gated, open, its normal on screen, your signed distance to it, your side
    ax = dun.wall_axis[room]; hasw = ax > 0; wnx, wny = view_vec(va, mir, (ax == 2)*1., (ax == 1)*1.)
    wd = jnp.where(ax == AXIS_H, P.y, P.x)-dun.wall_pos[room]
    psec = sec_point(dun, room, P.x, P.y).astype(jnp.float32)
    wtail = jnp.stack([hasw*1., (dun.wall_kind[room] == WALL_GATED)*1., s.wall_open[room]*1., hasw*wnx, hasw*wny, hasw*wd*sc, hasw*psec])
    return jnp.concatenate([head, sf.reshape(-1), ef.reshape(-1), bf.reshape(-1), tail, ptail, wtail])
