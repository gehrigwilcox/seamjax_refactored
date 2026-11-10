"""
Seam, headless and batched in JAX.

The dungeon (rooms, doors, geometry, obstacles, shortest-path table) is a set of padded arrays.
Everything that happens per step is fixed-size array maths, so `jax.vmap(step)` runs thousands of
games at once and `jit` puts them on a GPU unchanged.

Geometry note: all the non-Euclidean content (curved tilings, glued edges, mirrored seams) is already
baked into the door tables (neighbor room, arrival side, flip flag). The only geometry done per step is
the view rotation/mirror on room change: one atan2 and a sign.
"""
import os, numpy as np, jax, jax.numpy as jnp
from jax import lax
from seamjax.difficulty import difficulty, FLOOR_CAP

VW, VH, CX, CY = 960., 600., 480., 300.
TAU = 2*np.pi
PR, GAP, PSPEED, MAXHP = 11., 34., 215., 5
NS, NO, NE, NB, NEB, NPK = 12, 6, 10, 24, 24, 6           # max sides, obstacles, enemies per room, player bullets, enemy bullets
FIRE_DELAY, BSPEED, BLIFE, INVULN = .14, 520., .9, 1.
H_HP, H_R, H_SPEED, H_SPF, H_HOP0, H_HOPF, H_HOPMIN, H_RESPAWN, H_GRACE = 18., 19., 88., 5., 5.2, .25, 2.8, 22., 1.6

# reward weights (tunable)
R_KILL, R_HIT, R_ROOM, R_DESCEND, R_DEATH, R_HUNTER, R_HEAL, R_GATE = .1, -.2, .05, 5., -5., .5, .05, .05
PICKUP_R = 20.
AREA_PER_ENEMY, SPAWN_MIN, SPAWN_MAX, TELEGRAPH = 30000., 160., 260., 1.0

def rot_inv(x, y, va, mir):
    c, s = jnp.cos(va), jnp.sin(va)
    return mir*(x*c + y*s), -x*s + y*c
def view_vec(va, m, x, y):
    u = m*x; c, s = jnp.cos(va), jnp.sin(va)
    return c*u - s*y, s*u + c*y

def _push_obs(cx, cy, r, ot, o):
    """push a circle out of one obstacle (ot: 0 circle, 1 rect, -1 none). o = [x,y,w|r,h,_]"""
    # circle obstacle
    dx, dy = cx-o[0], cy-o[1]; d = jnp.hypot(dx, dy); m = r+o[2]
    k = jnp.where(d < 1e-4, 0., (m-d)/jnp.maximum(d, 1e-9))
    hc = (ot == 0) & (d < m)
    cx_c = jnp.where(hc, jnp.where(d < 1e-4, cx+m, cx+dx*k), cx)
    cy_c = jnp.where(hc, jnp.where(d < 1e-4, cy, cy+dy*k), cy)
    # rect obstacle
    qx, qy = jnp.clip(cx, o[0], o[0]+o[2]), jnp.clip(cy, o[1], o[1]+o[3])
    rx, ry = cx-qx, cy-qy; d2 = rx*rx+ry*ry
    hr = (ot == 1) & (d2 < r*r)
    d_ = jnp.sqrt(jnp.maximum(d2, 1e-12)); kr = (r-d_)/d_
    l, rr, t, b = cx-o[0], o[0]+o[2]-cx, cy-o[1], o[1]+o[3]-cy
    mn = jnp.minimum(jnp.minimum(l, rr), jnp.minimum(t, b))
    ix = jnp.where(mn == l, o[0]-r, jnp.where(mn == rr, o[0]+o[2]+r, cx))
    iy = jnp.where((mn != l) & (mn != rr), jnp.where(mn == t, o[1]-r, o[1]+o[3]+r), cy)
    inside = d2 <= 1e-4
    cx_r = jnp.where(hr, jnp.where(inside, ix, cx+rx*kr), cx)
    cy_r = jnp.where(hr, jnp.where(inside, iy, cy+ry*kr), cy)
    isr = ot == 1
    return jnp.where(isr, cx_r, cx_c), jnp.where(isr, cy_r, cy_c)

def push_all_obs(x, y, r, obs, ot):
    for i in range(NO):
        x, y = _push_obs(x, y, r, ot[i], obs[i])
    return x, y

def hits_obs(x, y, r, obs, ot):
    def one(o, t):
        circ = jnp.hypot(x-o[0], y-o[1]) < r+o[2]
        qx, qy = jnp.clip(x, o[0], o[0]+o[2]), jnp.clip(y, o[1], o[1]+o[3])
        rect = (x-qx)**2+(y-qy)**2 < r*r
        return jnp.where(t == 0, circ, jnp.where(t == 1, rect, False))
    return jnp.any(jax.vmap(one)(obs, ot))

def inside_poly(side, n, x, y, m):
    valid = jnp.arange(NS) < n
    d = (x-side[:, 0])*side[:, 2] + (y-side[:, 1])*side[:, 3]
    return jnp.all(jnp.where(valid, d <= -m, True))

def push_inside(x, y, r, side, n):
    def body(i, c):
        x, y = c; k = i % NS; S = side[k]
        d = (x-S[0])*S[2] + (y-S[1])*S[3]
        hit = (k < n) & (d > -r)
        return jnp.where(hit, x-S[2]*(d+r), x), jnp.where(hit, y-S[3]*(d+r), y)
    return lax.fori_loop(0, 2*NS, body, (x, y))

def clamp_player(x, y, side, n, doors):
    gh = GAP - PR*.2 - .01
    def body(i, c):
        x, y, ex = c; k = i % NS; S = side[k]
        d = (x-S[0])*S[2] + (y-S[1])*S[3]
        active = (k < n) & (d > -PR)
        lat = (x-S[6])*S[4] + (y-S[7])*S[5]
        chan = (doors[k] >= 0) & (jnp.abs(lat) < gh)
        ex = jnp.where(active & chan & (d > 6), k, ex)
        push = active & ~chan
        return jnp.where(push, x-S[2]*(d+PR), x), jnp.where(push, y-S[3]*(d+PR), y), ex
    return lax.fori_loop(0, 2*NS, body, (x, y, jnp.int32(-1)))

# ---------------- sections: an inner wall splits a room in two ----------------
# dun['wax'][r]: 0 no wall, 1 horizontal (splits by y), 2 vertical (splits by x); dun['wpos'][r]: its coordinate
# dun['wkind'][r]: 0 none, 1 solid (never opens), 2 gated (opens when your side is cleared)
# the wall itself is an ordinary rect obstacle flagged by obs[...,4] > 0; it vanishes once state['wopen'][r]
def room_obs(s, room):
    obs = s['dun']['obs'][room]; ot = s['dun']['obs_t'][room]
    return obs, jnp.where((obs[:, 4] > 0) & s['wopen'][room], -1, ot)

def sec_point(dun, room, x, y):
    ax, pos = dun['wax'][room], dun['wpos'][room]
    return jnp.where(ax == 0, 0, jnp.where(ax == 1, y >= pos, x >= pos)).astype(jnp.int32)

def eff_sec(s, room, sec):                                    # an opened wall merges the two sections
    return jnp.where(s['wopen'][room], 0, sec)

def hnode(dun, r, sec):
    """node of the Hunter's world: a room-section, except that a gated wall does not separate (the Hunter ignores gates)"""
    return 2*r + jnp.where(dun['wkind'][r] == 1, sec, 0)

def cnd(pred, f, s):
    """lax.cond over the game state without dragging the dungeon tables through the select (they never change in a step)"""
    dun = s['dun']; rest = {k: v for k, v in s.items() if k != 'dun'}
    out = lax.cond(pred, lambda r: {k: v for k, v in f({**r, 'dun': dun}).items() if k != 'dun'}, lambda r: r, rest)
    return {**out, 'dun': dun}

GEN_CHUNK = int(os.environ.get('SEAM_GEN_CHUNK', 64))            # dungeons generated at once; building a dungeon needs ~4 MB of scratch, so this bounds GPU memory

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

def make_env(source):
    """source(key, floor, idx=None) -> dict of padded dungeon tables for ONE dungeon. Returns the env functions.
    Every game carries its own dungeon in its state, so a GPU generator can replace the pool unchanged."""
    R = source(jax.random.PRNGKey(0), jnp.int32(1), 0)['n'].shape[0]

    # ---------------- state ----------------
    def reset(key, floor=1, d=None):
        """a fresh run at `floor` (floor 1 is the start of a run)"""
        k0, k2, k3 = jax.random.split(key, 3)
        dun = source(k0, jnp.asarray(floor, jnp.int32), d)
        start = dun['start']; ex = dun['exit']
        depth = dun['depth']; thr = jnp.maximum(2, jnp.floor(depth[ex]*.5).astype(jnp.int32))
        cand = (depth >= thr) & (jnp.arange(R) < dun['n_rooms'])
        gum = jax.random.gumbel(k2, (R,))
        hroom = jnp.where(jnp.any(cand), jnp.argmax(jnp.where(cand, gum, -1e9)), ex)
        s = dict(dun=dun, room=start, x=jnp.float32(CX), y=jnp.float32(CY), inv=0., fire=0., aim=0., va=0., mir=1., hp=jnp.int32(MAXHP),
                 t=0., steps=jnp.int32(0), kills=jnp.int32(0), visited=jnp.zeros(R, bool).at[start].set(True),
                 pb=jnp.zeros((NB, 6)), eb=jnp.zeros((NEB, 6)), en=jnp.zeros((R, NE, 8)), pk=jnp.zeros((R, NPK, 3)),
                 hroom=hroom.astype(jnp.int32), hsec=jnp.int32(0), htimer=difficulty(floor)['hunter_first'], hhere=False, hx=0., hy=0., hhp=H_HP, hdead=0.,
                 hflash=0., done=False, key=k3, took=jnp.int32(0), newroom=jnp.int32(0), killed=jnp.int32(0),
                 hit_hunter=jnp.int32(0), descended=False, shake=False, healed=jnp.int32(0), opened=jnp.int32(0),
                 wopen=jnp.zeros(R, bool), spawned=jnp.zeros((R, 2), bool))
        # same fix-up as the JS game: don't start inside an obstacle
        side = dun['side'][start]; obs, ot = dun['obs'][start], dun['obs_t'][start]
        s['x'] = dun['startpos'][0]; s['y'] = dun['startpos'][1]      # a reachable spot near the middle of the start room
        s['spawned'] = s['spawned'].at[start, sec_point(dun, start, s['x'], s['y'])].set(True)    # the start section stays empty
        s['hsec'] = jnp.where(dun['wkind'][hroom] > 0, jax.random.bernoulli(jax.random.fold_in(k2, 3)), False).astype(jnp.int32)
        return jax.tree_util.tree_map(jnp.asarray, s)

    def hurt(s):
        ok = s['inv'] <= 0
        return {**s, 'hp': s['hp']-ok.astype(jnp.int32), 'inv': jnp.where(ok, INVULN, s['inv']), 'took': s.get('took', 0)+ok.astype(jnp.int32)}

    # ---------------- room change (the whole non-Euclidean part of a step) ----------------
    def transit(s, ex):
        room = s['room']
        n = s['dun']['n'][room]; nr = s['dun']['doors'][room, ex]; arrive = s['dun']['arr'][room, ex]; fl = s['dun']['flip'][room, ex]
        phi = TAU*ex/n
        hx, hy = view_vec(s['va'], s['mir'], jnp.sin(phi), -jnp.cos(phi))
        m2 = jnp.where(fl == 1, -s['mir'], s['mir']); psi = TAU*arrive/s['dun']['n'][nr]
        va = jnp.mod(jnp.arctan2(-hy, -hx) - jnp.arctan2(-jnp.cos(psi), m2*jnp.sin(psi)), TAU)
        S = s['dun']['side'][nr, arrive]
        s = {**s, 'room': nr, 'va': va, 'mir': m2, 'x': S[6]-S[2]*30, 'y': S[7]-S[3]*30,
             'pb': jnp.zeros((NB, 6)), 'eb': jnp.zeros((NEB, 6))}
        # hunter grace when you leave its room
        left = s['hroom'] == room
        s['htimer'] = jnp.where(left, jnp.minimum(s['htimer'], H_GRACE), s['htimer']); s['hhere'] = jnp.where(left, False, s['hhere'])
        first = ~s['visited'][nr]
        newly = first.astype(jnp.int32)
        s['visited'] = s['visited'].at[nr].set(True)
        s['newroom'] = newly
        e = eff_sec(s, nr, sec_point(s['dun'], nr, s['x'], s['y']))        # each section spawns its enemies on first entry
        spawn = ~s['spawned'][nr, e]
        s['spawned'] = s['spawned'].at[nr, e].set(True)
        k, s['key'] = jax.random.split(s['key'])
        s = cnd(spawn, lambda s: spawn_enemies(s, k, e), s)
        return s

    def spawn_enemies(s, key, e):
        room = s['room']
        k0, k1, k2, k3, k4 = jax.random.split(key, 5)
        floor = jnp.minimum(s['dun']['floor'], FLOOR_CAP)
        isx = room == s['dun']['exit']
        want = 2 + jnp.floor(floor*.8).astype(jnp.int32) + jax.random.randint(k0, (), 0, 3) + jnp.where(isx, 2, 0)
        pts = s['dun']['spawn'][room]                                # 16 pre-validated, reachable spawn points per room
        ok0 = s['dun']['spawn_ok'][room] > 0
        mine = ok0 & (eff_sec(s, room, s['dun']['spawn_sec'][room]) == e)
        frac = jnp.where(jnp.any(ok0), jnp.sum(mine)/jnp.maximum(jnp.sum(ok0), 1), 1.)          # a section only holds its share of the room
        cap = jnp.maximum(1, jnp.round(s['dun']['area'][room]*frac/AREA_PER_ENEMY).astype(jnp.int32)) + jnp.where(isx, 1, 0)
        cnt = jnp.minimum(jnp.minimum(8, want), cap)                 # never more than the room can hold
        minD = jnp.clip(s['dun']['rb'][room]*.55, SPAWN_MIN, SPAWN_MAX)
        dpl = jnp.hypot(pts[:, 0]-s['x'], pts[:, 1]-s['y'])
        okp = mine & (dpl >= minD*.6)                                # better to spawn fewer than to ambush
        score = jax.random.uniform(k1, (16,)) + jnp.where(dpl >= minD, 0., -5.) + jnp.where(okp, 0., -10.)
        order = jnp.argsort(-score)[:NE]
        ex, ey = pts[order, 0], pts[order, 1]
        alive = okp[order] & (jnp.arange(NE) < cnt)
        shooter = jax.random.uniform(k3, (NE,)) < difficulty(floor)['shooter_p']
        fire0 = 1 + jax.random.uniform(k4, (NE,))*1.5
        en = jnp.stack([ex, ey, jnp.where(shooter, 4., 3.), shooter.astype(jnp.float32), fire0, jnp.full(NE, TELEGRAPH), jnp.zeros(NE), alive.astype(jnp.float32)], 1)
        old = s['en'][room]; free = old[:, 7] < .5                   # new enemies go into free slots (the other section may be occupied)
        slot = jnp.argsort(~free)
        en = old.at[slot].set(jnp.where(free[slot][:, None], en, old[slot]))
        return {**s, 'en': s['en'].at[room].set(en)}

    # ---------------- hunter ----------------
    def hunter_tick(s, dt):
        room = s['room']; dun = s['dun']; floor = jnp.minimum(dun['floor'], FLOOR_CAP)
        side, n = dun['side'][room], dun['n'][room]
        obs, ot = room_obs(s, room)
        pn = hnode(dun, room, sec_point(dun, room, s['x'], s['y']))          # where you are, in the Hunter's world
        hf = jnp.maximum(s['hflash']-dt, 0.)
        # --- dead / respawn
        dead = s['hdead'] > 0
        hd2 = jnp.where(dead, s['hdead']-dt, 0.)
        respawn = dead & (hd2 <= 0)
        k, key = jax.random.split(s['key']); s = {**s, 'key': key}
        k, ksec = jax.random.split(k)
        dfar = dun['ndist'][2*jnp.arange(R), pn]
        far = (dfar >= 3) & (dfar < 999) & (jnp.arange(R) < dun['n_rooms'])
        gum = jax.random.gumbel(k, (R,))
        newr = jnp.where(jnp.any(far), jnp.argmax(jnp.where(far, gum, -1e9)), dun['exit']).astype(jnp.int32)
        newsec = jnp.where(dun['wkind'][newr] > 0, jax.random.bernoulli(ksec), False).astype(jnp.int32)
        # --- away: count down, hop along the true shortest route (through section nodes)
        hr = s['hroom']; hn = hnode(dun, hr, s['hsec'])
        away = ~dead & (hn != pn)
        tm = jnp.where(away, s['htimer']-dt, s['htimer'])
        hop = away & (tm <= 0)
        doors = dun['doors'][hr]; arrk = dun['arr'][hr]; nb = jnp.maximum(doors, 0)
        nbnode = hnode(dun, nb, dun['dsec'][nb, arrk])
        stepok = (doors >= 0) & (jnp.arange(NS) < dun['n'][hr]) & (hnode(dun, hr, dun['dsec'][hr]) == hn)
        dd = jnp.where(stepok, dun['ndist'][nbnode, pn], 999)
        bk = jnp.argmin(dd); best = doors[bk]; can = hop & (dd[bk] < 999)
        arrside = arrk[bk]
        hroom = jnp.where(can, best, hr); hsec = jnp.where(can, dun['dsec'][best, arrside], s['hsec'])
        tm = jnp.where(can, jnp.maximum(H_HOPMIN, H_HOP0-floor*H_HOPF), tm)
        arrive_here = can & (nbnode[bk] == pn)
        S = dun['side'][room, arrside]
        hx = jnp.where(arrive_here, S[6]-S[2]*34, s['hx']); hy = jnp.where(arrive_here, S[7]-S[3]*34, s['hy'])
        here = jnp.where(arrive_here, True, jnp.where(away, False, s['hhere']))
        # --- in the player's section: chase
        inroom = ~dead & (hnode(dun, hroom, hsec) == pn)
        rd = dun['doors'][room]
        okk = (rd >= 0) & (jnp.arange(NS) < n) & (hnode(dun, room, dun['dsec'][room]) == pn)
        S0 = side[jnp.where(jnp.any(okk), jnp.argmax(okk), 0)]
        place0 = inroom & ~here
        hx = jnp.where(place0, S0[6]-S0[2]*34, hx); hy = jnp.where(place0, S0[7]-S0[3]*34, hy)
        here = here | place0
        chase = inroom & here & ~arrive_here
        # JS: a hunter that just arrived also chases in following frames; on the arrival frame it only appears
        dx, dy = s['x']-hx, s['y']-hy; dist = jnp.hypot(dx, dy)+1e-9; sp = H_SPEED + floor*H_SPF
        nx_, ny_ = hx+dx/dist*sp*dt, hy+dy/dist*sp*dt
        nx_, ny_ = push_all_obs(nx_, ny_, H_R, obs, ot); nx_, ny_ = push_inside(nx_, ny_, H_R, side, n)
        hx = jnp.where(chase, nx_, hx); hy = jnp.where(chase, ny_, hy)
        touch = chase & (dist < H_R+PR)
        s = {**s, 'hdead': hd2, 'hflash': hf, 'htimer': jnp.where(respawn, 8., tm),
             'hroom': jnp.where(respawn, newr, hroom), 'hsec': jnp.where(respawn, newsec, hsec),
             'hhere': jnp.where(respawn, False, here), 'hx': hx, 'hy': hy,
             'hhp': jnp.where(respawn, H_HP, s['hhp'])}
        s['shake'] = arrive_here
        return cnd(touch, hurt, s)

    # ---------------- the step ----------------
    def step(s, action, dt=1/60.):
        room = s['room']
        floor = s['dun']['floor']
        s = {**s, 'took': jnp.int32(0), 'newroom': jnp.int32(0), 'killed': jnp.int32(0), 'hit_hunter': jnp.int32(0), 'descended': False, 'healed': jnp.int32(0), 'opened': jnp.int32(0)}
        n = s['dun']['n'][room]; side = s['dun']['side'][room]; doors = s['dun']['doors'][room]
        obs, ot = room_obs(s, room)
        s['inv'] = s['inv']-dt
        s = hunter_tick(s, dt)
        # ---- movement
        mv = jnp.clip(action[:2], -1, 1); ml = jnp.hypot(mv[0], mv[1]); mv = jnp.where(ml > 1, mv/ml, mv)
        mx, my = rot_inv(mv[0], mv[1], s['va'], s['mir'])
        x, y = s['x']+mx*PSPEED*dt, s['y']+my*PSPEED*dt
        x, y = push_all_obs(x, y, PR, obs, ot)
        x, y, ex = clamp_player(x, y, side, n, doors)
        s_move = {**s, 'x': x, 'y': y}
        s_tr = transit(s_move, jnp.maximum(ex, 0))
        # ---- rest of the frame (only if we did not just change room)
        s2 = combat(s_move, action, dt)
        sel = lambda a, b: jnp.where(ex >= 0, a, b)
        s = {k: (s_tr[k] if k == 'dun' else jax.tree_util.tree_map(sel, s_tr[k], s2[k])) for k in s_tr}
        s['t'] = s['t']+dt; s['steps'] = s['steps']+1
        # ---- reward / done
        reward = (R_KILL*s['killed'] + R_HIT*s['took'] + R_ROOM*s['newroom'] + R_HUNTER*s['hit_hunter']*0
                  + R_HEAL*s['healed'] + R_GATE*s['opened'] + R_DESCEND*s['descended'] + jnp.where(s['hp'] <= 0, R_DEATH, 0.))
        done = s['hp'] <= 0                          # the run ends only on death; stairs are reported as `descended`
        s['done'] = done
        return s, reward, done

    def combat(s, action, dt):
        room = s['room']; floor = jnp.minimum(s['dun']['floor'], FLOOR_CAP).astype(jnp.float32)
        n = s['dun']['n'][room]; side = s['dun']['side'][room]
        obs, ot = room_obs(s, room)
        x, y = s['x'], s['y']
        # ---- aim / fire
        ax, ay = rot_inv(action[2], action[3], s['va'], s['mir'])
        firing = jnp.hypot(action[2], action[3]) > .25
        aim = jnp.where(firing, jnp.arctan2(ay, ax), s['aim'])
        fire = s['fire']-dt
        shoot = firing & (fire <= 0)
        fire = jnp.where(shoot, FIRE_DELAY, fire)
        pb = s['pb']; free = jnp.argmin(pb[:, 5]); can = shoot & (pb[free, 5] < .5)
        newb = jnp.array([x+jnp.cos(aim)*14, y+jnp.sin(aim)*14, jnp.cos(aim)*BSPEED, jnp.sin(aim)*BSPEED, BLIFE, 1.])
        newb = jnp.stack([x+jnp.cos(aim)*14, y+jnp.sin(aim)*14, jnp.cos(aim)*BSPEED, jnp.sin(aim)*BSPEED, jnp.full_like(x, BLIFE), jnp.ones_like(x)])
        pb = jnp.where(can, pb.at[free].set(newb), pb)
        # ---- player bullets: move, die on walls/obstacles
        alive = pb[:, 5] > .5
        bx, by = pb[:, 0]+pb[:, 2]*dt, pb[:, 1]+pb[:, 3]*dt; life = pb[:, 4]-dt
        wall = ~jax.vmap(lambda a, b: inside_poly(side, n, a, b, 0.))(bx, by)
        ob = jax.vmap(lambda a, b: hits_obs(a, b, 3., obs, ot))(bx, by)
        alive = alive & (life > 0) & ~wall & ~ob
        # ---- vs enemies
        en = s['en'][room]
        ealive = en[:, 7] > .5; er = jnp.where(en[:, 3] > .5, 14., 12.); vuln = ealive & (en[:, 5] <= 0)
        dist = jnp.hypot(bx[:, None]-en[None, :, 0], by[:, None]-en[None, :, 1])
        hit = (dist < er[None, :]+3) & vuln[None, :] & alive[:, None]
        first = jnp.argmax(hit, axis=1); anyh = jnp.any(hit, axis=1)
        onehot = jax.nn.one_hot(first, NE)*anyh[:, None]
        dmg = onehot.sum(0)
        alive = alive & ~anyh
        # ---- vs hunter
        hdist = jnp.hypot(bx-s['hx'], by-s['hy'])
        hvuln = s['hhere'] & (s['hroom'] == room) & (s['hdead'] <= 0)
        hh = (hdist < H_R+3) & alive & hvuln
        hfirst = jnp.argmax(hh) ; hany = jnp.any(hh)
        hh1 = jax.nn.one_hot(hfirst, NB)*hany
        hdmg = jnp.sum(hh1)
        alive = alive & ~(hh1 > .5)
        hhp = s['hhp']-hdmg; hkill = hvuln & (hhp <= 0) & (hdmg > 0)
        pb = jnp.stack([bx, by, pb[:, 2], pb[:, 3], life, alive.astype(jnp.float32)], 1)
        ehp = en[:, 2]-dmg; ekill = ealive & (ehp <= 0) & (dmg > 0)
        flash = jnp.where(dmg > 0, .1, jnp.maximum(en[:, 6]-dt, 0.))
        # ---- enemies move
        spawning = ealive & (en[:, 5] > 0)
        px_, py_ = x, y
        edx, edy = px_-en[:, 0], py_-en[:, 1]; ed = jnp.hypot(edx, edy)+1e-9
        shooter = en[:, 3] > .5
        spc = 70+floor*7
        cdx, cdy = edx/ed*spc, edy/ed*spc
        sp = 55.
        retreat = ed < 230; approach = ed > 340
        sdx = jnp.where(retreat, -edx/ed*sp, jnp.where(approach, edx/ed*sp, -edy/ed*sp*.6))
        sdy = jnp.where(retreat, -edy/ed*sp, jnp.where(approach, edy/ed*sp, edx/ed*sp*.6))
        vx_ = jnp.where(shooter, sdx, cdx); vy_ = jnp.where(shooter, sdy, cdy)
        act = ealive & ~ekill & ~spawning
        nx = en[:, 0]+jnp.where(act, vx_*dt, 0.); ny = en[:, 1]+jnp.where(act, vy_*dt, 0.)
        def fix(a, b, r):
            a, b = push_all_obs(a, b, r, obs, ot); return push_inside(a, b, r, side, n)
        nx2, ny2 = jax.vmap(fix)(nx, ny, er)
        nx = jnp.where(act, nx2, nx); ny = jnp.where(act, ny2, ny)
        contact = jnp.any(act & (ed < er+PR))
        # separation
        ddx, ddy = nx[None, :]-nx[:, None], ny[None, :]-ny[:, None]; dd = jnp.hypot(ddx, ddy)
        m = er[:, None]+er[None, :]
        pair = (dd > 0) & (dd < m) & act[:, None] & act[None, :] & ~jnp.eye(NE, dtype=bool)
        kk = jnp.where(pair, (m-dd)/2/jnp.maximum(dd, 1e-9), 0.)
        nx = nx - (ddx*kk).sum(1) + 0.0; ny = ny - (ddy*kk).sum(1) + 0.0
        # shooters fire
        efire = en[:, 4]-jnp.where(act & shooter, dt, 0.)
        wantf = act & shooter & (efire <= 0)
        k, key = jax.random.split(s['key']); key, kd = jax.random.split(key)
        efire = jnp.where(wantf, 1.6+jax.random.uniform(k, (NE,))*.9, efire)
        eb = s['eb']
        def place(i, eb):
            f = jnp.argmin(eb[:, 5]); ok = wantf[i] & (eb[f, 5] < .5)
            nb_ = jnp.stack([en[i, 0], en[i, 1], edx[i]/ed[i]*190, edy[i]/ed[i]*190, jnp.float32(3.), jnp.float32(1.)])
            return jnp.where(ok, eb.at[f].set(nb_), eb)
        eb = lax.fori_loop(0, NE, place, eb)
        espawn = jnp.where(spawning, en[:, 5]-dt, en[:, 5])
        newalive = ealive & ~ekill
        en_new = jnp.stack([nx, ny, ehp, en[:, 3], efire, jnp.maximum(espawn, 0.), flash, newalive.astype(jnp.float32)], 1)
        # ---- enemy bullets
        ealive_b = eb[:, 5] > .5
        ex_, ey_ = eb[:, 0]+eb[:, 2]*dt, eb[:, 1]+eb[:, 3]*dt; elife = eb[:, 4]-dt
        ewall = ~jax.vmap(lambda a, b: inside_poly(side, n, a, b, 0.))(ex_, ey_)
        eob = jax.vmap(lambda a, b: hits_obs(a, b, 4., obs, ot))(ex_, ey_)
        ealive_b = ealive_b & (elife > 0) & ~ewall & ~eob
        ehit = ealive_b & (jnp.hypot(ex_-px_, ey_-py_) < PR+4)
        ealive_b = ealive_b & ~ehit
        eb = jnp.stack([ex_, ey_, eb[:, 2], eb[:, 3], elife, ealive_b.astype(jnp.float32)], 1)
        # ---- apply
        s = {**s, 'aim': aim, 'fire': fire, 'pb': pb, 'eb': eb, 'key': key, 'en': s['en'].at[room].set(en_new),
             'kills': s['kills']+jnp.sum(ekill).astype(jnp.int32)+5*hkill.astype(jnp.int32), 'killed': jnp.sum(ekill).astype(jnp.int32),
             'hit_hunter': hany.astype(jnp.int32), 'hhp': jnp.where(hkill, H_HP, hhp),
             'hdead': jnp.where(hkill, H_RESPAWN, s['hdead']), 'hhere': jnp.where(hkill, False, s['hhere'])}
        s = cnd(contact | jnp.any(ehit), hurt, s)
        # ---- health pickups: enemies sometimes drop one where they die; walk over one to heal
        drop = ekill & (jax.random.uniform(kd, (NE,)) < difficulty(s['dun']['floor'])['drop_p'])
        def put_pk(i, pk):
            f = jnp.argmin(pk[:, 2]); ok = drop[i] & (pk[f, 2] < .5)
            return jnp.where(ok, pk.at[f].set(jnp.stack([en[i, 0], en[i, 1], jnp.float32(1.)])), pk)
        pk = lax.fori_loop(0, NE, put_pk, s['pk'][room])
        def take(i, c):
            pk, hp, got = c
            near = (pk[i, 2] > .5) & (jnp.hypot(pk[i, 0]-px_, pk[i, 1]-py_) < PICKUP_R) & (hp < MAXHP)
            return jnp.where(near, pk.at[i, 2].set(0.), pk), hp+near.astype(jnp.int32), got+near.astype(jnp.int32)
        pk, hp, got = lax.fori_loop(0, NPK, take, (pk, s['hp'], jnp.int32(0)))
        s = {**s, 'pk': s['pk'].at[room].set(pk), 'hp': hp, 'healed': got}
        # ---- exit pad
        pad = s['dun']['pad'][room]
        cleared = ~jnp.any(en_new[:, 7] > .5)
        # ---- a gated wall opens once your side is clear
        psec = sec_point(s['dun'], room, px_, py_)
        gate = (s['dun']['wkind'][room] == 2) & ~s['wopen'][room] & cleared & s['spawned'][room, psec]
        wopen = s['wopen'].at[room].set(s['wopen'][room] | gate)
        spawned = jnp.where(gate, s['spawned'].at[room].set(True), s['spawned'])
        same = (sec_point(s['dun'], room, pad[0], pad[1]) == psec) | wopen[room]
        s = {**s, 'wopen': wopen, 'spawned': spawned, 'opened': gate.astype(jnp.int32)}
        s['descended'] = (room == s['dun']['exit']) & cleared & (jnp.hypot(px_-pad[0], py_-pad[1]) < 26) & same
        return s

    # ---------------- observation (egocentric, screen frame, no map) ----------------
    OBS_SIDES, OBS_EN, OBS_EB = NS, 8, 8
    obs_dim = 8 + NS*8 + OBS_EN*5 + OBS_EB*4 + 8 + 3 + 7
    def observe(s):
        room = s['room']; n = s['dun']['n'][room]; side = s['dun']['side'][room]; doors = s['dun']['doors'][room]
        va, mir = s['va'], s['mir']; sc = 1/300.
        x, y = s['x']-CX, s['y']-CY
        vx, vy = view_vec(va, mir, x, y)
        dun = s['dun']; pn = hnode(dun, room, sec_point(dun, room, s['x'], s['y'])); hn = hnode(dun, s['hroom'], s['hsec'])
        hd_ = dun['ndist'][hn, pn]
        hdist = jnp.where(s['hdead'] > 0, -1., jnp.where(hd_ >= 999, -2., hd_.astype(jnp.float32)))     # -1 down, -2 cut off by a wall
        head = jnp.stack([vx*sc, vy*sc, s['hp']/MAXHP, (s['inv'] > 0).astype(jnp.float32), n/12., mir, jnp.clip(hdist, -2, 12)/12., jnp.minimum(s['dun']['floor'], FLOOR_CAP)/FLOOR_CAP])
        # sides: valid, outward normal (screen), door, signed dist of player, hunter-arrival door
        nxs, nys = jax.vmap(lambda a, b: view_vec(va, mir, a, b))(side[:, 2], side[:, 3])
        dpl = ((s['x']-side[:, 0])*side[:, 2] + (s['y']-side[:, 1])*side[:, 3])*sc
        lat = ((side[:, 6]-side[:, 0])*side[:, 4] + (side[:, 7]-side[:, 1])*side[:, 5])*sc
        valid = (jnp.arange(NS) < n).astype(jnp.float32)
        warn = (jnp.arange(NS) == _hunter_door(s)).astype(jnp.float32)
        sf = jnp.stack([valid, nxs*valid, nys*valid, (doors >= 0)*valid, dpl*valid, lat*valid, warn*valid, (dun['dsec'][room] > 0)*valid*(dun['wkind'][room] > 0)], 1)
        en = s['en'][room]; alive = en[:, 7] > .5
        rx, ry = en[:, 0]-s['x'], en[:, 1]-s['y']
        ex_, ey_ = jax.vmap(lambda a, b: view_vec(va, mir, a, b))(rx, ry)
        dist = jnp.where(alive, jnp.hypot(rx, ry), 1e9); order = jnp.argsort(dist)[:OBS_EN]
        ef = jnp.stack([alive[order], ex_[order]*sc*alive[order], ey_[order]*sc*alive[order], en[order, 3], en[order, 2]/4.*alive[order]], 1)
        eb = s['eb']; balive = eb[:, 5] > .5
        brx, bry = eb[:, 0]-s['x'], eb[:, 1]-s['y']
        bx_, by_ = jax.vmap(lambda a, b: view_vec(va, mir, a, b))(brx, bry)
        bvx, bvy = jax.vmap(lambda a, b: view_vec(va, mir, a, b))(eb[:, 2], eb[:, 3])
        bd = jnp.where(balive, jnp.hypot(brx, bry), 1e9); bo = jnp.argsort(bd)[:OBS_EB]
        bf = jnp.stack([bx_[bo]*sc*balive[bo], by_[bo]*sc*balive[bo], bvx[bo]/190.*balive[bo], bvy[bo]/190.*balive[bo]], 1)
        # hunter + exit
        hrx, hry = view_vec(va, mir, s['hx']-s['x'], s['hy']-s['y'])
        hv = (s['hroom'] == room) & s['hhere'] & (s['hdead'] <= 0)
        pad = s['dun']['pad'][room]; prx, pry = view_vec(va, mir, pad[0]-s['x'], pad[1]-s['y'])
        isx = room == s['dun']['exit']
        tail = jnp.stack([hv*1., hv*hrx*sc, hv*hry*sc, hv*s['hhp']/H_HP, isx*1., isx*prx*sc, isx*pry*sc, (~jnp.any(alive))*1.])
        pk = s['pk'][room]; pa = pk[:, 2] > .5
        pdist = jnp.where(pa, jnp.hypot(pk[:, 0]-s['x'], pk[:, 1]-s['y']), 1e9); pi_ = jnp.argmin(pdist)
        prx2, pry2 = view_vec(va, mir, pk[pi_, 0]-s['x'], pk[pi_, 1]-s['y']); pany = jnp.any(pa)
        ptail = jnp.stack([pany*1., pany*prx2*sc, pany*pry2*sc])
        # inner wall: present, gated, open, its normal on screen, your signed distance to it, your side
        ax = dun['wax'][room]; hasw = ax > 0; wnx, wny = view_vec(va, mir, (ax == 2)*1., (ax == 1)*1.)
        wd = jnp.where(ax == 1, s['y'], s['x'])-dun['wpos'][room]
        psec = sec_point(dun, room, s['x'], s['y']).astype(jnp.float32)
        wtail = jnp.stack([hasw*1., (dun['wkind'][room] == 2)*1., s['wopen'][room]*1., hasw*wnx, hasw*wny, hasw*wd*sc, hasw*psec])
        return jnp.concatenate([head, sf.reshape(-1), ef.reshape(-1), bf.reshape(-1), tail, ptail, wtail])

    def hunter_dist(s):
        """doors between you and the Hunter in section space: 0 same section, -1 down, -2 cut off by a wall"""
        dun = s['dun']; room = s['room']
        pn = hnode(dun, room, sec_point(dun, room, s['x'], s['y'])); hn = hnode(dun, s['hroom'], s['hsec']); d = dun['ndist'][hn, pn]
        return jnp.where(s['hdead'] > 0, -1, jnp.where(d >= 999, -2, d)).astype(jnp.int32)

    def _hunter_door(s):
        """side of your room through which the Hunter would come (it is one door-step away from your section), else -1"""
        room = s['room']; dun = s['dun']; hr = s['hroom']
        pn = hnode(dun, room, sec_point(dun, room, s['x'], s['y'])); hn = hnode(dun, hr, s['hsec'])
        doors = dun['doors'][hr]; arrk = dun['arr'][hr]
        ok = (doors == room) & (hnode(dun, hr, dun['dsec'][hr]) == hn) & (hnode(dun, room, dun['dsec'][room, arrk]) == pn)
        k = jnp.argmax(ok)
        return jnp.where((hn != pn) & (s['hdead'] <= 0) & jnp.any(ok), arrk[k], -1)

    def next_state(s, key, done, desc):
        """what follows a finished step: death starts a new run at floor 1; stairs continue the run one floor deeper
        (hp +1, kills / time carried over)."""
        floor = jnp.where(done, 1, s['dun']['floor']+1)
        n = reset(key, floor)
        keep = desc & ~done
        return {**n, 'hp': jnp.where(keep, jnp.minimum(MAXHP, s['hp']+1), n['hp']), 'kills': jnp.where(keep, s['kills'], 0),
                'steps': jnp.where(keep, s['steps'], 0), 't': jnp.where(keep, s['t'], 0.)}

    def step_auto(s, action, key, dt=1/60.):
        """one game, never ends: dies -> new run, reaches the stairs -> next floor. Returns (state, reward, done, info)"""
        s2, r, done = step(s, action, dt)
        desc = s2['descended'] & ~done
        info = dict(descended=desc, floor=s2['dun']['floor'], kills=s2['kills'], hp=s2['hp'], t=s2['t'])
        s3 = lax.cond(done | desc, lambda s_: next_state(s_, key, done, desc), lambda s_: s_, s2)
        return s3, r, done, info

    def step_auto_batch(S, actions, key, dt=1/60.):
        """Batched step with auto-reset/advance. New dungeons are generated only when some game finished or
        took the stairs (one real lax.cond, outside vmap), then merged into those games."""
        S2, r, done = jax.vmap(lambda s, a: step(s, a, dt))(S, actions)
        B = done.shape[0]
        desc = S2['descended'] & ~done
        info = dict(descended=desc, floor=S2['dun']['floor'], kills=S2['kills'], hp=S2['hp'], t=S2['t'])
        keys = jax.random.split(key, B)
        regen = lambda S2: regen_where(S2, done | desc, lambda row, i: next_state(row, keys[i], done[i], desc[i]))
        return lax.cond(jnp.any(done | desc), regen, lambda S2: S2, S2), r, done, info

    return dict(reset_batch=lambda keys, floors: cvmap(reset)(keys, floors), reset=reset, step=step, hunter_door=_hunter_door, hunter_dist=hunter_dist, next_state=next_state, step_auto=step_auto, step_auto_batch=step_auto_batch, observe=observe, obs_dim=obs_dim, transit=transit)
