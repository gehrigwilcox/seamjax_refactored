"""
Seam, headless and batched in JAX.

The dungeon (rooms, doors, geometry, obstacles, shortest-path table) is a set of padded arrays.
Everything that happens per step is fixed-size array maths, so `jax.vmap(step)` runs thousands of
games at once and `jit` puts them on a GPU unchanged.

Geometry note: all the non-Euclidean content (curved tilings, glued edges, mirrored seams) is already
baked into the door tables (neighbor room, arrival side, flip flag). The only geometry done per step is
the view rotation/mirror on room change: one atan2 and a sign.

The state is a GameState (core/types.py); every constant comes from core/config.py.
"""
import jax, jax.numpy as jnp
from jax import lax
from seamjax.difficulty import difficulty, FLOOR_CAP
from seamjax.core import struct, pool
from seamjax.core.geometry.view import view_vec, rot_inv, through_door
from seamjax.core.geometry.collide import side_dist, push_out, push_out_obstacles, clamp_through_doors, blocked
from seamjax.core.config import (
    CX, CY, MAX_SIDES as NS, MAX_ENEMIES as NE, MAX_PLAYER_BULLETS as NB, MAX_ENEMY_BULLETS as NEB,
    MAX_PICKUPS as NPK, N_SPAWN_POINTS, GEN_CHUNK, DOOR_GAP, ENTRY_INSET, PLAYER_R, PLAYER_SPEED, MAX_HP, INVULN, AIM_DEADZONE,
    FIRE_DELAY, BULLET_SPEED, BULLET_LIFE, BULLET_MUZZLE, BULLET_R, PICKUP_R, PAD_R, CHASER_R, SHOOTER_R, CHASER_HP, SHOOTER_HP,
    CHASER_SPEED, CHASER_SPEED_PER_FLOOR, SHOOTER_SPEED, SHOOTER_RETREAT, SHOOTER_APPROACH, SHOOTER_STRAFE, SHOOTER_FIRST_SHOT,
    SHOOTER_FIRST_SHOT_RAND, SHOOTER_RELOAD, SHOOTER_RELOAD_RAND, ENEMY_BULLET_SPEED, ENEMY_BULLET_LIFE, ENEMY_BULLET_R, HIT_FLASH,
    TELEGRAPH, MAX_SPAWN, AREA_PER_ENEMY, SPAWN_MIN, SPAWN_MAX, HUNTER_HP, HUNTER_R, HUNTER_SPEED, HUNTER_SPEED_PER_FLOOR, HUNTER_HOP,
    HUNTER_HOP_PER_FLOOR, HUNTER_HOP_MIN, HUNTER_RESPAWN, HUNTER_RESPAWN_DELAY, HUNTER_GRACE, HUNTER_DOOR_INSET, HUNTER_KILL_BONUS,
    INF, R_KILL, R_HIT, R_ROOM, R_DESCEND, R_DEATH, R_HEAL, R_GATE)
from seamjax.core.types import Player, Bullets, Enemies, Pickups, Hunter, StepEvents, GameState, OBST_NONE, AXIS_NONE, AXIS_H, WALL_SOLID, WALL_GATED


# ---------------------------------------------------------------- sections: an inner wall splits a room in two
def room_obs(s, room):
    """the obstacles of a room as they block things now (an open inner wall no longer counts)"""
    ob = s.dun.obstacles[room]
    return ob.replace(kind=jnp.where(ob.wall & s.wall_open[room], OBST_NONE, ob.kind))

def sec_point(dun, room, x, y):
    """section (0 / 1) of the room that (x, y) lies in"""
    ax, pos = dun.wall_axis[room], dun.wall_pos[room]
    return jnp.where(ax == AXIS_NONE, 0, jnp.where(ax == AXIS_H, y >= pos, x >= pos)).astype(jnp.int32)

def eff_sec(s, room, sec):                                    # an opened wall merges the two sections
    return jnp.where(s.wall_open[room], 0, sec)

def hnode(dun, r, sec):
    """node of the Hunter's world: a room-section, except that a gated wall does not separate (the Hunter ignores gates)"""
    return 2*r + jnp.where(dun.wall_kind[r] == WALL_SOLID, sec, 0)

def player_node(s):
    return hnode(s.dun, s.room, sec_point(s.dun, s.room, s.player.x, s.player.y))

def cnd(pred, f, s):
    """lax.cond over the game state without dragging the dungeon tables through the select (they never change in a step)"""
    dun = s.dun
    out = lax.cond(pred, lambda r: f(r.replace(dun=dun)).replace(dun=None), lambda r: r, s.replace(dun=None))
    return out.replace(dun=dun)

def fly(b, dt, r, sides, n, ob):
    """move a bullet pool one step; bullets that run out of life, leave the room or hit an obstacle die"""
    x, y, life = b.x+b.vx*dt, b.y+b.vy*dt, b.life-dt
    hit = jax.vmap(lambda a, c: blocked(a, c, r, sides, n, ob))(x, y)
    return b.replace(x=x, y=y, life=life, alive=b.alive & (life > 0) & ~hit)


# ---------------------------------------------------------------- batching helpers
def cvmap(f, chunk=GEN_CHUNK):
    """vmap(f) that runs `chunk` rows at a time (lax.map over chunks): same result, bounded scratch memory"""
    def g(*args):
        B = args[0].shape[0]; nc = -(-B//chunk); pad = nc*chunk-B
        pd = lambda a: jnp.concatenate([a, jnp.repeat(a[:1], pad, 0)], 0) if pad else a
        args = jax.tree_util.tree_map(lambda a: pd(a).reshape((nc, chunk)+a.shape[1:]), args)
        out = lax.map(lambda xs: jax.vmap(f)(*xs), args)
        return jax.tree_util.tree_map(lambda a: a.reshape((nc*chunk,)+a.shape[2:])[:B], out)
    return g

def regen_where(S, need, fn, chunk=GEN_CHUNK):
    """replace the rows of the batched state S flagged in `need` by fn(rows) (fn is vmapped over `chunk` rows at a time).
    Only the flagged rows are rebuilt, in chunks, so cost and memory follow how many games finished, not the batch size."""
    B = need.shape[0]
    def body(c):
        S, need = c
        idx = jnp.argsort(~need)[:chunk] if B >= chunk else jnp.argsort(~need)           # flagged rows first
        valid = need[idx]
        rows = jax.tree_util.tree_map(lambda a: a[idx], S)
        new = jax.vmap(fn)(rows, idx)
        def put(a, b):
            m = valid.reshape((-1,)+(1,)*(a.ndim-1)); return a.at[idx].set(jnp.where(m, b, a[idx]))
        return jax.tree_util.tree_map(put, S, new), need.at[idx].set(False)
    S, _ = lax.while_loop(lambda c: jnp.any(c[1]), body, (S, need))
    return S


# ---------------------------------------------------------------- queries used by the observation, the renderers and the HUD
def hunter_dist(s):
    """doors between you and the Hunter in section space: 0 same section, -1 down, -2 cut off by a wall"""
    d = s.dun.node_dist[hnode(s.dun, s.hunter.room, s.hunter.sec), player_node(s)]
    return jnp.where(s.hunter.dead > 0, -1, jnp.where(d >= INF, -2, d)).astype(jnp.int32)

def hunter_door(s):
    """side of your room through which the Hunter would come (it is one door-step away from your section), else -1"""
    room = s.room; dun = s.dun; hr = s.hunter.room
    pn = player_node(s); hn = hnode(dun, hr, s.hunter.sec)
    doors = dun.doors[hr]; arrk = dun.arrive[hr]
    ok = (doors == room) & (hnode(dun, hr, dun.door_sec[hr]) == hn) & (hnode(dun, room, dun.door_sec[room, arrk]) == pn)
    k = jnp.argmax(ok)
    return jnp.where((hn != pn) & (s.hunter.dead <= 0) & jnp.any(ok), arrk[k], -1)

def hunter_visible(s):
    return (s.hunter.room == s.room) & s.hunter.here & (s.hunter.dead <= 0)


# ---------------------------------------------------------------- empty pools
def _bullets(n):
    z = jnp.zeros(n); return Bullets(x=z, y=z, vx=z, vy=z, life=z, alive=jnp.zeros(n, bool))

def _enemies(R):
    z = jnp.zeros((R, NE)); b = jnp.zeros((R, NE), bool)
    return Enemies(x=z, y=z, hp=z, shooter=b, fire=z, telegraph=z, flash=z, alive=b)

def _no_events(shake=False):
    i0 = jnp.int32(0)
    return StepEvents(took=i0, newroom=i0, killed=i0, hit_hunter=i0, descended=False, shake=shake, healed=i0, opened=i0)


# ---------------------------------------------------------------- observation (egocentric, screen frame, no map)
OBS_EN, OBS_EB = 8, 8
OBS_DIM = 8 + NS*8 + OBS_EN*5 + OBS_EB*4 + 8 + 3 + 7

def observe(s):
    dun = s.dun; room = s.room; n = dun.n[room]; sides = dun.sides[room]; doors = dun.doors[room]; P = s.player
    va, mir = s.view_angle, s.mirror; sc = 1/300.
    x, y = P.x-CX, P.y-CY
    vx, vy = view_vec(va, mir, x, y)
    hd_ = dun.node_dist[hnode(dun, s.hunter.room, s.hunter.sec), player_node(s)]
    hdist = jnp.where(s.hunter.dead > 0, -1., jnp.where(hd_ >= INF, -2., hd_.astype(jnp.float32)))     # -1 down, -2 cut off by a wall
    head = jnp.stack([vx*sc, vy*sc, P.hp/MAX_HP, (P.inv > 0).astype(jnp.float32), n/12., mir, jnp.clip(hdist, -2, 12)/12., jnp.minimum(dun.floor, FLOOR_CAP)/FLOOR_CAP])
    # sides: valid, outward normal (screen), door, signed dist of player, door offset along the side, hunter-arrival door, wall side
    nxs, nys = jax.vmap(lambda a, b: view_vec(va, mir, a, b))(sides.normal[:, 0], sides.normal[:, 1])
    dpl = side_dist(sides, P.x, P.y)*sc
    lat = ((sides.door[:, 0]-sides.mid[:, 0])*sides.tangent[:, 0] + (sides.door[:, 1]-sides.mid[:, 1])*sides.tangent[:, 1])*sc
    valid = (jnp.arange(NS) < n).astype(jnp.float32)
    warn = (jnp.arange(NS) == hunter_door(s)).astype(jnp.float32)
    sf = jnp.stack([valid, nxs*valid, nys*valid, (doors >= 0)*valid, dpl*valid, lat*valid, warn*valid, (dun.door_sec[room] > 0)*valid*(dun.wall_kind[room] > 0)], 1)
    en = s.enemies[room]; alive = en.alive
    rx, ry = en.x-P.x, en.y-P.y
    ex_, ey_ = jax.vmap(lambda a, b: view_vec(va, mir, a, b))(rx, ry)
    dist = jnp.where(alive, jnp.hypot(rx, ry), 1e9); order = jnp.argsort(dist)[:OBS_EN]
    ef = jnp.stack([alive[order], ex_[order]*sc*alive[order], ey_[order]*sc*alive[order], en.shooter[order], en.hp[order]/4.*alive[order]], 1)
    eb = s.eb; balive = eb.alive
    brx, bry = eb.x-P.x, eb.y-P.y
    bx_, by_ = jax.vmap(lambda a, b: view_vec(va, mir, a, b))(brx, bry)
    bvx, bvy = jax.vmap(lambda a, b: view_vec(va, mir, a, b))(eb.vx, eb.vy)
    bd = jnp.where(balive, jnp.hypot(brx, bry), 1e9); bo = jnp.argsort(bd)[:OBS_EB]
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


# ---------------------------------------------------------------- the game
def make_env(source):
    """source(key, floor, idx=None) -> Dungeon for ONE dungeon. Returns the env functions.
    Every game carries its own dungeon in its state, so a GPU generator can replace the pool unchanged."""
    R = source(jax.random.PRNGKey(0), jnp.int32(1), 0).n.shape[0]

    # ---------------- state ----------------
    def reset(key, floor=1, d=None):
        """a fresh run at `floor` (floor 1 is the start of a run)"""
        k0, k2, k3 = jax.random.split(key, 3)
        dun = source(k0, jnp.asarray(floor, jnp.int32), d)
        start = dun.start; ex = dun.exit
        depth = dun.depth; thr = jnp.maximum(2, jnp.floor(depth[ex]*.5).astype(jnp.int32))
        cand = (depth >= thr) & (jnp.arange(R) < dun.n_rooms)
        gum = jax.random.gumbel(k2, (R,))
        hroom = jnp.where(jnp.any(cand), jnp.argmax(jnp.where(cand, gum, -1e9)), ex)
        x, y = dun.startpos[0], dun.startpos[1]                     # a reachable spot near the middle of the start room
        hsec = jnp.where(dun.wall_kind[hroom] > 0, jax.random.bernoulli(jax.random.fold_in(k2, 3)), False).astype(jnp.int32)
        z = jnp.zeros((R, NPK))
        s = GameState(dun=dun, room=start, view_angle=0., mirror=1.,
                      player=Player(x=x, y=y, hp=jnp.int32(MAX_HP), inv=0., fire=0., aim=0.),
                      pb=_bullets(NB), eb=_bullets(NEB), enemies=_enemies(R), pickups=Pickups(x=z, y=z, alive=jnp.zeros((R, NPK), bool)),
                      hunter=Hunter(room=hroom.astype(jnp.int32), sec=hsec, timer=difficulty(floor)['hunter_first'], here=False, x=0., y=0.,
                                    hp=HUNTER_HP, dead=0., flash=0.),
                      visited=jnp.zeros(R, bool).at[start].set(True), wall_open=jnp.zeros(R, bool),
                      spawned=jnp.zeros((R, 2), bool).at[start, sec_point(dun, start, x, y)].set(True),       # the start section stays empty
                      t=0., steps=jnp.int32(0), kills=jnp.int32(0), done=False, key=k3, ev=_no_events())
        return jax.tree_util.tree_map(jnp.asarray, s)

    def hurt(s):
        ok = s.player.inv <= 0
        return s.replace(player=s.player.replace(hp=s.player.hp-ok.astype(jnp.int32), inv=jnp.where(ok, INVULN, s.player.inv)),
                         ev=s.ev.replace(took=s.ev.took+ok.astype(jnp.int32)))

    # ---------------- room change (the whole non-Euclidean part of a step) ----------------
    def transit(s, ex):
        room = s.room; dun = s.dun
        nr = dun.doors[room, ex]; arrive = dun.arrive[room, ex]
        va, m2 = through_door(s.view_angle, s.mirror, ex, dun.n[room], arrive, dun.n[nr], dun.flip[room, ex])
        door, nrm = dun.sides.door[nr, arrive], dun.sides.normal[nr, arrive]
        x, y = door[0]-nrm[0]*ENTRY_INSET, door[1]-nrm[1]*ENTRY_INSET
        left = s.hunter.room == room                                       # hunter grace when you leave its room
        hunter = s.hunter.replace(timer=jnp.where(left, jnp.minimum(s.hunter.timer, HUNTER_GRACE), s.hunter.timer),
                                  here=jnp.where(left, False, s.hunter.here))
        first = ~s.visited[nr]
        s = s.replace(room=nr, view_angle=va, mirror=m2, player=s.player.replace(x=x, y=y), pb=_bullets(NB), eb=_bullets(NEB), hunter=hunter,
                      visited=s.visited.at[nr].set(True), ev=s.ev.replace(newroom=first.astype(jnp.int32)))
        e = eff_sec(s, nr, sec_point(dun, nr, x, y))                      # each section spawns its enemies on first entry
        spawn = ~s.spawned[nr, e]
        k, key = jax.random.split(s.key)
        s = s.replace(spawned=s.spawned.at[nr, e].set(True), key=key)
        return cnd(spawn, lambda s: spawn_enemies(s, k, e), s)

    def spawn_enemies(s, key, e):
        room = s.room; dun = s.dun
        k0, k1, k2, k3, k4 = jax.random.split(key, 5)
        floor = jnp.minimum(dun.floor, FLOOR_CAP)
        isx = room == dun.exit
        want = 2 + jnp.floor(floor*.8).astype(jnp.int32) + jax.random.randint(k0, (), 0, 3) + jnp.where(isx, 2, 0)
        pts = dun.spawn[room]                                        # pre-validated, reachable spawn points
        ok0 = dun.spawn_ok[room] > 0
        mine = ok0 & (eff_sec(s, room, dun.spawn_sec[room]) == e)
        frac = jnp.where(jnp.any(ok0), jnp.sum(mine)/jnp.maximum(jnp.sum(ok0), 1), 1.)          # a section only holds its share of the room
        cap = jnp.maximum(1, jnp.round(dun.area[room]*frac/AREA_PER_ENEMY).astype(jnp.int32)) + jnp.where(isx, 1, 0)
        cnt = jnp.minimum(jnp.minimum(MAX_SPAWN, want), cap)         # never more than the room can hold
        minD = jnp.clip(dun.radius[room]*.55, SPAWN_MIN, SPAWN_MAX)
        dpl = jnp.hypot(pts[:, 0]-s.player.x, pts[:, 1]-s.player.y)
        okp = mine & (dpl >= minD*.6)                                # better to spawn fewer than to ambush
        score = jax.random.uniform(k1, (N_SPAWN_POINTS,)) + jnp.where(dpl >= minD, 0., -5.) + jnp.where(okp, 0., -10.)
        order = jnp.argsort(-score)[:NE]
        shooter = jax.random.uniform(k3, (NE,)) < difficulty(floor)['shooter_p']
        new = Enemies(x=pts[order, 0], y=pts[order, 1], hp=jnp.where(shooter, SHOOTER_HP, CHASER_HP), shooter=shooter,
                      fire=SHOOTER_FIRST_SHOT + jax.random.uniform(k4, (NE,))*SHOOTER_FIRST_SHOT_RAND, telegraph=jnp.full(NE, TELEGRAPH),
                      flash=jnp.zeros(NE), alive=okp[order] & (jnp.arange(NE) < cnt))
        rows = pool.fill(s.enemies[room], new)                       # into free slots: the other section's enemies stay
        return s.replace(enemies=pool.set_row(s.enemies, room, rows))

    # ---------------- hunter ----------------
    def hunter_tick(s, dt):
        room = s.room; dun = s.dun; floor = jnp.minimum(dun.floor, FLOOR_CAP); H = s.hunter; P = s.player
        sides, n = dun.sides[room], dun.n[room]
        ob = room_obs(s, room)
        pn = player_node(s)                                                      # where you are, in the Hunter's world
        hf = jnp.maximum(H.flash-dt, 0.)
        # --- dead / respawn
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
        doors = dun.doors[hr]; arrk = dun.arrive[hr]; nb = jnp.maximum(doors, 0)
        nbnode = hnode(dun, nb, dun.door_sec[nb, arrk])
        stepok = (doors >= 0) & (jnp.arange(NS) < dun.n[hr]) & (hnode(dun, hr, dun.door_sec[hr]) == hn)
        dd = jnp.where(stepok, dun.node_dist[nbnode, pn], INF)
        bk = jnp.argmin(dd); best = doors[bk]; can = hop & (dd[bk] < INF)
        arrside = arrk[bk]
        hroom = jnp.where(can, best, hr); hsec = jnp.where(can, dun.door_sec[best, arrside], H.sec)
        tm = jnp.where(can, jnp.maximum(HUNTER_HOP_MIN, HUNTER_HOP-floor*HUNTER_HOP_PER_FLOOR), tm)
        arrive_here = can & (nbnode[bk] == pn)
        door, nrm = sides.door[arrside], sides.normal[arrside]
        hx = jnp.where(arrive_here, door[0]-nrm[0]*HUNTER_DOOR_INSET, H.x); hy = jnp.where(arrive_here, door[1]-nrm[1]*HUNTER_DOOR_INSET, H.y)
        here = jnp.where(arrive_here, True, jnp.where(away, False, H.here))
        # --- in the player's section: chase
        inroom = ~dead & (hnode(dun, hroom, hsec) == pn)
        rd = dun.doors[room]
        okk = (rd >= 0) & (jnp.arange(NS) < n) & (hnode(dun, room, dun.door_sec[room]) == pn)
        k0 = jnp.where(jnp.any(okk), jnp.argmax(okk), 0); door0, nrm0 = sides.door[k0], sides.normal[k0]
        place0 = inroom & ~here
        hx = jnp.where(place0, door0[0]-nrm0[0]*HUNTER_DOOR_INSET, hx); hy = jnp.where(place0, door0[1]-nrm0[1]*HUNTER_DOOR_INSET, hy)
        here = here | place0
        chase = inroom & here & ~arrive_here
        # JS: a hunter that just arrived also chases in following frames; on the arrival frame it only appears
        dx, dy = P.x-hx, P.y-hy; dist = jnp.hypot(dx, dy)+1e-9; sp = HUNTER_SPEED + floor*HUNTER_SPEED_PER_FLOOR
        nx_, ny_ = hx+dx/dist*sp*dt, hy+dy/dist*sp*dt
        nx_, ny_ = push_out(nx_, ny_, HUNTER_R, sides, n, ob)
        hx = jnp.where(chase, nx_, hx); hy = jnp.where(chase, ny_, hy)
        touch = chase & (dist < HUNTER_R+PLAYER_R)
        s = s.replace(hunter=Hunter(room=jnp.where(respawn, newr, hroom), sec=jnp.where(respawn, newsec, hsec),
                                    timer=jnp.where(respawn, HUNTER_RESPAWN_DELAY, tm), here=jnp.where(respawn, False, here), x=hx, y=hy,
                                    hp=jnp.where(respawn, HUNTER_HP, H.hp), dead=hd2, flash=hf),
                      ev=s.ev.replace(shake=arrive_here))
        return cnd(touch, hurt, s)

    # ---------------- the step ----------------
    def step(s, action, dt=1/60.):
        room = s.room
        s = s.replace(ev=_no_events(s.ev.shake))
        n = s.dun.n[room]; sides = s.dun.sides[room]; doors = s.dun.doors[room]
        ob = room_obs(s, room)
        s = s.replace(player=s.player.replace(inv=s.player.inv-dt))
        s = hunter_tick(s, dt)
        # ---- movement
        mv = jnp.clip(action[:2], -1, 1); ml = jnp.hypot(mv[0], mv[1]); mv = jnp.where(ml > 1, mv/ml, mv)
        mx, my = rot_inv(mv[0], mv[1], s.view_angle, s.mirror)
        x, y = s.player.x+mx*PLAYER_SPEED*dt, s.player.y+my*PLAYER_SPEED*dt
        x, y = push_out_obstacles(x, y, PLAYER_R, ob)
        x, y, ex = clamp_through_doors(x, y, PLAYER_R, DOOR_GAP - PLAYER_R*.2 - .01, sides, n, doors)    # doors let you through, walls do not
        s_move = s.replace(player=s.player.replace(x=x, y=y))
        s_tr = transit(s_move, jnp.maximum(ex, 0))
        # ---- rest of the frame (only if we did not just change room)
        s2 = combat(s_move, action, dt)
        s = struct.where(ex >= 0, s_tr.replace(dun=None), s2.replace(dun=None)).replace(dun=s_tr.dun)
        s = s.replace(t=s.t+dt, steps=s.steps+1)
        # ---- reward / done
        ev = s.ev
        reward = (R_KILL*ev.killed + R_HIT*ev.took + R_ROOM*ev.newroom
                  + R_HEAL*ev.healed + R_GATE*ev.opened + R_DESCEND*ev.descended + jnp.where(s.player.hp <= 0, R_DEATH, 0.))
        done = s.player.hp <= 0                      # the run ends only on death; stairs are reported as `descended`
        return s.replace(done=done), reward, done

    def combat(s, action, dt):
        room = s.room; dun = s.dun; floor = jnp.minimum(dun.floor, FLOOR_CAP).astype(jnp.float32)
        n = dun.n[room]; sides = dun.sides[room]
        ob = room_obs(s, room)
        P = s.player; H = s.hunter; x, y = P.x, P.y
        # ---- aim / fire
        ax, ay = rot_inv(action[2], action[3], s.view_angle, s.mirror)
        firing = jnp.hypot(action[2], action[3]) > AIM_DEADZONE
        aim = jnp.where(firing, jnp.arctan2(ay, ax), P.aim)
        fire = P.fire-dt
        shoot = firing & (fire <= 0)
        fire = jnp.where(shoot, FIRE_DELAY, fire)
        newb = Bullets(x=x+jnp.cos(aim)*BULLET_MUZZLE, y=y+jnp.sin(aim)*BULLET_MUZZLE, vx=jnp.cos(aim)*BULLET_SPEED, vy=jnp.sin(aim)*BULLET_SPEED,
                       life=BULLET_LIFE, alive=True)
        pb = fly(pool.spawn(s.pb, newb, shoot), dt, BULLET_R, sides, n, ob)       # player bullets: move, die on walls/obstacles
        bx, by, alive = pb.x, pb.y, pb.alive
        # ---- vs enemies
        en = s.enemies[room]
        ealive = en.alive; er = jnp.where(en.shooter, SHOOTER_R, CHASER_R); vuln = ealive & (en.telegraph <= 0)
        dist = jnp.hypot(bx[:, None]-en.x[None, :], by[:, None]-en.y[None, :])
        hit = (dist < er[None, :]+BULLET_R) & vuln[None, :] & alive[:, None]
        first = jnp.argmax(hit, axis=1); anyh = jnp.any(hit, axis=1)
        onehot = jax.nn.one_hot(first, NE)*anyh[:, None]
        dmg = onehot.sum(0)
        alive = alive & ~anyh
        # ---- vs hunter
        hdist = jnp.hypot(bx-H.x, by-H.y)
        hvuln = H.here & (H.room == room) & (H.dead <= 0)
        hh = (hdist < HUNTER_R+BULLET_R) & alive & hvuln
        hfirst = jnp.argmax(hh); hany = jnp.any(hh)
        hh1 = jax.nn.one_hot(hfirst, NB)*hany
        hdmg = jnp.sum(hh1)
        alive = alive & ~(hh1 > .5)
        hhp = H.hp-hdmg; hkill = hvuln & (hhp <= 0) & (hdmg > 0)
        pb = pb.replace(alive=alive)
        ehp = en.hp-dmg; ekill = ealive & (ehp <= 0) & (dmg > 0)
        flash = jnp.where(dmg > 0, HIT_FLASH, jnp.maximum(en.flash-dt, 0.))
        # ---- enemies move
        spawning = ealive & (en.telegraph > 0)
        edx, edy = x-en.x, y-en.y; ed = jnp.hypot(edx, edy)+1e-9
        shooter = en.shooter
        spc = CHASER_SPEED+floor*CHASER_SPEED_PER_FLOOR
        cdx, cdy = edx/ed*spc, edy/ed*spc
        sp = SHOOTER_SPEED
        retreat = ed < SHOOTER_RETREAT; approach = ed > SHOOTER_APPROACH
        sdx = jnp.where(retreat, -edx/ed*sp, jnp.where(approach, edx/ed*sp, -edy/ed*sp*SHOOTER_STRAFE))
        sdy = jnp.where(retreat, -edy/ed*sp, jnp.where(approach, edy/ed*sp, edx/ed*sp*SHOOTER_STRAFE))
        vx_ = jnp.where(shooter, sdx, cdx); vy_ = jnp.where(shooter, sdy, cdy)
        act = ealive & ~ekill & ~spawning
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
        # shooters fire
        efire = en.fire-jnp.where(act & shooter, dt, 0.)
        wantf = act & shooter & (efire <= 0)
        k, key = jax.random.split(s.key); key, kd = jax.random.split(key)
        efire = jnp.where(wantf, SHOOTER_RELOAD+jax.random.uniform(k, (NE,))*SHOOTER_RELOAD_RAND, efire)
        def place(i, eb):
            new = Bullets(x=en.x[i], y=en.y[i], vx=edx[i]/ed[i]*ENEMY_BULLET_SPEED, vy=edy[i]/ed[i]*ENEMY_BULLET_SPEED, life=ENEMY_BULLET_LIFE, alive=True)
            return pool.spawn(eb, new, wantf[i])
        eb = lax.fori_loop(0, NE, place, s.eb)
        espawn = jnp.where(spawning, en.telegraph-dt, en.telegraph)
        en_new = Enemies(x=nx, y=ny, hp=ehp, shooter=en.shooter, fire=efire, telegraph=jnp.maximum(espawn, 0.), flash=flash, alive=ealive & ~ekill)
        # ---- enemy bullets
        eb = fly(eb, dt, ENEMY_BULLET_R, sides, n, ob)
        ehit = eb.alive & (jnp.hypot(eb.x-x, eb.y-y) < PLAYER_R+ENEMY_BULLET_R)
        eb = eb.replace(alive=eb.alive & ~ehit)
        # ---- apply
        nkill = jnp.sum(ekill).astype(jnp.int32)
        s = s.replace(player=P.replace(aim=aim, fire=fire), pb=pb, eb=eb, key=key, enemies=pool.set_row(s.enemies, room, en_new),
                      kills=s.kills+nkill+HUNTER_KILL_BONUS*hkill.astype(jnp.int32),
                      hunter=H.replace(hp=jnp.where(hkill, HUNTER_HP, hhp), dead=jnp.where(hkill, HUNTER_RESPAWN, H.dead), here=jnp.where(hkill, False, H.here)),
                      ev=s.ev.replace(killed=nkill, hit_hunter=hany.astype(jnp.int32)))
        s = cnd(contact | jnp.any(ehit), hurt, s)
        # ---- health pickups: enemies sometimes drop one where they die; walk over one to heal
        drop = ekill & (jax.random.uniform(kd, (NE,)) < difficulty(dun.floor)['drop_p'])
        def put_pk(i, pk): return pool.spawn(pk, Pickups(x=en.x[i], y=en.y[i], alive=True), drop[i])
        pk = lax.fori_loop(0, NE, put_pk, s.pickups[room])
        def take(i, c):
            pk, hp, got = c
            near = pk.alive[i] & (jnp.hypot(pk.x[i]-x, pk.y[i]-y) < PICKUP_R) & (hp < MAX_HP)
            return pk.replace(alive=jnp.where(near, pk.alive.at[i].set(False), pk.alive)), hp+near.astype(jnp.int32), got+near.astype(jnp.int32)
        pk, hp, got = lax.fori_loop(0, NPK, take, (pk, s.player.hp, jnp.int32(0)))
        s = s.replace(pickups=pool.set_row(s.pickups, room, pk), player=s.player.replace(hp=hp), ev=s.ev.replace(healed=got))
        # ---- exit pad
        pad = dun.pad[room]
        cleared = ~jnp.any(en_new.alive)
        # ---- a gated wall opens once your side is clear
        psec = sec_point(dun, room, x, y)
        gate = (dun.wall_kind[room] == WALL_GATED) & ~s.wall_open[room] & cleared & s.spawned[room, psec]
        wopen = s.wall_open.at[room].set(s.wall_open[room] | gate)
        spawned = jnp.where(gate, s.spawned.at[room].set(True), s.spawned)
        same = (sec_point(dun, room, pad[0], pad[1]) == psec) | wopen[room]
        descended = (room == dun.exit) & cleared & (jnp.hypot(x-pad[0], y-pad[1]) < PAD_R) & same
        return s.replace(wall_open=wopen, spawned=spawned, ev=s.ev.replace(opened=gate.astype(jnp.int32), descended=descended))

    def next_state(s, key, done, desc):
        """what follows a finished step: death starts a new run at floor 1; stairs continue the run one floor deeper
        (hp +1, kills / time carried over)."""
        floor = jnp.where(done, 1, s.dun.floor+1)
        n = reset(key, floor)
        keep = desc & ~done
        return n.replace(player=n.player.replace(hp=jnp.where(keep, jnp.minimum(MAX_HP, s.player.hp+1), n.player.hp)),
                         kills=jnp.where(keep, s.kills, 0), steps=jnp.where(keep, s.steps, 0), t=jnp.where(keep, s.t, 0.))

    def _info(s, desc):
        return dict(descended=desc, floor=s.dun.floor, kills=s.kills, hp=s.player.hp, t=s.t)

    def step_auto(s, action, key, dt=1/60.):
        """one game, never ends: dies -> new run, reaches the stairs -> next floor. Returns (state, reward, done, info)"""
        s2, r, done = step(s, action, dt)
        desc = s2.ev.descended & ~done
        s3 = lax.cond(done | desc, lambda s_: next_state(s_, key, done, desc), lambda s_: s_, s2)
        return s3, r, done, _info(s2, desc)

    def step_auto_batch(S, actions, key, dt=1/60.):
        """Batched step with auto-reset/advance. New dungeons are generated only when some game finished or
        took the stairs (one real lax.cond, outside vmap), then merged into those games."""
        S2, r, done = jax.vmap(lambda s, a: step(s, a, dt))(S, actions)
        B = done.shape[0]
        desc = S2.ev.descended & ~done
        keys = jax.random.split(key, B)
        regen = lambda S2: regen_where(S2, done | desc, lambda row, i: next_state(row, keys[i], done[i], desc[i]))
        return lax.cond(jnp.any(done | desc), regen, lambda S2: S2, S2), r, done, _info(S2, desc)

    return dict(reset_batch=lambda keys, floors: cvmap(reset)(keys, floors), reset=reset, step=step, hunter_door=hunter_door,
                hunter_dist=hunter_dist, next_state=next_state, step_auto=step_auto, step_auto_batch=step_auto_batch, observe=observe,
                obs_dim=OBS_DIM, transit=transit)
