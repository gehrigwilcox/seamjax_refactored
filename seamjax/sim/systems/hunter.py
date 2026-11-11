"""
The Hunter. Away from you it waits, then hops door by door along the true shortest route to your section (solid walls
separate sections, gated ones do not). In your section it chases you. It can be shot down; then it respawns far away.
"""
import jax, jax.numpy as jnp
from seamjax.difficulty import FLOOR_CAP
from seamjax.core.config import (INF, PLAYER_R, HUNTER_HP, HUNTER_R, HUNTER_SPEED, HUNTER_SPEED_PER_FLOOR, HUNTER_HOP,
                                 HUNTER_HOP_PER_FLOOR, HUNTER_HOP_MIN, HUNTER_RESPAWN_DELAY, HUNTER_DOOR_INSET)
from seamjax.core.types import Hunter
from seamjax.core.geometry.collide import push_out
from seamjax.sim.queries import room_geom, hnode, player_node
from seamjax.sim.state import cond, hurt
from seamjax.sim.systems.routing import next_hop, entry_side


def tick(s, ev, dt):
    room = s.room; dun = s.dun; floor = jnp.minimum(dun.floor, FLOOR_CAP); H = s.hunter; P = s.player; R = dun.n.shape[0]
    sides, n, ob = room_geom(s, room)
    pn = player_node(s)                                                      # where you are, in the Hunter's world
    hf = jnp.maximum(H.flash-dt, 0.)
    # --- dead / respawn (somewhere at least 3 doors from you)
    dead = H.dead > 0
    hd2 = jnp.where(dead, H.dead-dt, 0.)
    respawn = dead & (hd2 <= 0)
    k, key = jax.random.split(s.key); s = s.replace(key=key)
    k, ksec = jax.random.split(k)
    dfar = dun.node_dist[2*jnp.arange(R), pn]
    far = (dfar >= 3) & (dfar < INF) & (jnp.arange(R) < dun.n_rooms)
    gum = jax.random.gumbel(k, (R,))
    newr = jnp.where(jnp.any(far), jnp.argmax(jnp.where(far, gum, -1e9)), dun.exit).astype(jnp.int32)
    newsec = jnp.where(dun.wall_kind[newr] > 0, jax.random.bernoulli(ksec), False).astype(jnp.int32)
    # --- away: count down, hop along the true shortest route (through section nodes)
    hr = H.room; hn = hnode(dun, hr, H.sec)
    away = ~dead & (hn != pn)
    tm = jnp.where(away, H.timer-dt, H.timer)
    hop = away & (tm <= 0)
    ok, best, bsec, arrside, nnode = next_hop(dun, hr, H.sec, pn); can = hop & ok
    hroom = jnp.where(can, best, hr); hsec = jnp.where(can, bsec, H.sec)
    tm = jnp.where(can, jnp.maximum(HUNTER_HOP_MIN, HUNTER_HOP-floor*HUNTER_HOP_PER_FLOOR), tm)
    arrive_here = can & (nnode == pn)
    door, nrm = sides.door[arrside], sides.normal[arrside]
    hx = jnp.where(arrive_here, door[0]-nrm[0]*HUNTER_DOOR_INSET, H.x); hy = jnp.where(arrive_here, door[1]-nrm[1]*HUNTER_DOOR_INSET, H.y)
    here = jnp.where(arrive_here, True, jnp.where(away, False, H.here))
    # --- in the player's section: chase
    inroom = ~dead & (hnode(dun, hroom, hsec) == pn)
    k0 = entry_side(dun, room, pn); door0, nrm0 = sides.door[k0], sides.normal[k0]
    place0 = inroom & ~here                                                  # already in your section (e.g. you walked in): put it at a door
    hx = jnp.where(place0, door0[0]-nrm0[0]*HUNTER_DOOR_INSET, hx); hy = jnp.where(place0, door0[1]-nrm0[1]*HUNTER_DOOR_INSET, hy)
    here = here | place0
    chase = inroom & here & ~arrive_here                                    # on the arrival frame it only appears (as the JS game)
    dx, dy = P.x-hx, P.y-hy; dist = jnp.hypot(dx, dy)+1e-9; sp = HUNTER_SPEED + floor*HUNTER_SPEED_PER_FLOOR
    nx_, ny_ = push_out(hx+dx/dist*sp*dt, hy+dy/dist*sp*dt, HUNTER_R, sides, n, ob)
    hx = jnp.where(chase, nx_, hx); hy = jnp.where(chase, ny_, hy)
    touch = chase & (dist < HUNTER_R+PLAYER_R)
    s = s.replace(hunter=Hunter(room=jnp.where(respawn, newr, hroom), sec=jnp.where(respawn, newsec, hsec),
                                timer=jnp.where(respawn, HUNTER_RESPAWN_DELAY, tm), here=jnp.where(respawn, False, here), x=hx, y=hy,
                                hp=jnp.where(respawn, HUNTER_HP, H.hp), dead=hd2, flash=hf))
    return cond(touch, hurt, s, ev.replace(shake=arrive_here))
