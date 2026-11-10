"""
GPU dungeon generator for Seam. Pure JAX, fixed-size arrays, vmap/jit friendly.

Constant inputs (specs.npz, exported once from the JS tilings): for each tiling, a patch of rooms with
neighbor/arrival-side/flip/seam tables and the list of corner cycles. Everything random per dungeon is here:
which tiling, which connected set of rooms, which doors (random spanning tree + extra loops + forced corner
loops), start/exit, and every room's shape, obstacles, exit pad and spawn points.
"""
import numpy as np, jax, jax.numpy as jnp
from jax import lax
from seamjax.difficulty import difficulty
from seamjax.specs import load_specs
from seamjax.core.geometry.collide import inside_poly, hits_obstacles, free_spot, side_dist
from seamjax.core.config import (TAU, CX, CY, DOOR_GAP as GAP, MAX_ROOMS as R, MAX_SIDES as NS_, MAX_OBSTACLES as NO, N_SPAWN_POINTS as NSP,
                                 PLAYER_R as PR_, GRID_W as GW, GRID_H as GH, GRID_CELL as CS, ROOM_RC as RC, ROOM_SQUASH_MIN as SMIN,
                                 DOOR_INSET, INF, WALL_GATED_P as GATED_P, WALL_THICK as WALL_TH, WALL_CLEAR, WALL_MINLEN)
from seamjax.core.types import Dungeon, Sides, Obstacles

def _ob(obs, ot):
    """packed obstacle rows [x, y, w|r, h, is_wall] + kinds -> Obstacles (the generator still works on packed rows internally)"""
    return Obstacles(kind=ot, pos=obs[..., 0:2], size=obs[..., 2:4], wall=obs[..., 4] > 0)

def _gum(key, shape): return jax.random.gumbel(key, shape)

def choose_spec(SP, key, floor):
    cat = SP['cat']; cnt = jnp.array([jnp.sum(cat == c) for c in range(4)], jnp.float32)
    pc = difficulty(floor)['cat_p']                                  # which kind of world gets likelier with depth
    w = pc[cat]/cnt[cat]
    return jnp.argmax(jnp.where(w > 0, jnp.log(jnp.maximum(w, 1e-30)) + _gum(key, cat.shape), -1e9)).astype(jnp.int32)

# ---------------------------------------------------------------- room design

_XS = (np.arange(GW)+.5)*CS; _YS = (np.arange(GH)+.5)*CS                  # numpy: nothing may start the JAX backend at import time
_X, _Y = np.meshgrid(_XS, _YS)                                       # [GH, GW]

def free_grid(side, n, obs, ot):
    """[GH, GW] cells where the player (plus a margin) fits"""
    return jax.vmap(jax.vmap(lambda x, y: free_spot(x, y, PR_+4, side, n, _ob(obs, ot))))(_X, _Y)

def flood(free, side, open_mask):
    """cells reachable from the doors in open_mask (flood fill by repeated dilation); connected = all those doors in one piece"""
    X, Y = _X, _Y
    ex = side.door[:, 0]-side.normal[:, 0]*30; ey = side.door[:, 1]-side.normal[:, 1]*30
    has = jnp.any(open_mask)
    ex = jnp.where(open_mask, ex, jnp.where(has, 1e9, CX)); ey = jnp.where(open_mask, ey, jnp.where(has, 1e9, CY))
    d = jnp.where(free[None], jnp.hypot(X[None]-ex[:, None, None], Y[None]-ey[:, None, None]), 1e9)       # [12, GH, GW]
    snap = jnp.argmin(d.reshape(NS_, -1), axis=1)                                                         # nearest free cell per door
    live = open_mask | ~has
    first = jnp.argmax(live)
    reach = jnp.zeros(GH*GW, bool).at[snap[first]].set(True).reshape(GH, GW)
    def dil(i, r):
        p = jnp.pad(r, 1); nb = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        return free & (r | nb)
    reach = lax.fori_loop(0, GW+GH, dil, reach)
    connected = jnp.all(jnp.where(live, reach.reshape(-1)[snap], True))
    return reach, connected

def reach_grid(side, n, obs, ot, open_mask):
    return flood(free_grid(side, n, obs, ot), side, open_mask)

def design_room(key, n, is_exit, open_mask, want_wall):
    ks = jax.random.split(key, 16)
    U = lambda k, shape=(), lo=0., hi=1.: lo + jax.random.uniform(k, shape)*(hi-lo)
    sx, sy = U(ks[0], (), SMIN, 1.), U(ks[1], (), SMIN, 1.)
    k = jnp.arange(NS_); valid = k < n; nf = n.astype(jnp.float32)
    aA = TAU*(jnp.mod(k-1, n)+.5)/nf; aB = TAU*(k+.5)/nf
    Ax, Ay = CX+RC*sx*jnp.sin(aA), CY-RC*sy*jnp.cos(aA)
    Bx, By = CX+RC*sx*jnp.sin(aB), CY-RC*sy*jnp.cos(aB)
    ln = jnp.hypot(Bx-Ax, By-Ay)+1e-9; tx, ty = (Bx-Ax)/ln, (By-Ay)/ln
    mx, my = (Ax+Bx)/2, (Ay+By)/2; nx, ny = ty, -tx; half = ln/2
    off = U(ks[2], (NS_,), -1., 1.)*jnp.maximum(0., half-GAP-DOOR_INSET)
    dx, dy = mx+tx*off, my+ty*off
    v2 = lambda a, b: jnp.stack([a, b], 1)*valid[:, None]
    side = Sides(mid=v2(mx, my), normal=v2(nx, ny), tangent=v2(tx, ty), door=v2(dx, dy))
    rin = jnp.min(jnp.where(valid, (mx-CX)*nx+(my-CY)*ny, 1e9)); rb = RC*jnp.maximum(sx, sy)
    mo = jnp.minimum(110., rin*.5); enable = mo >= 60.
    style = jax.random.randint(ks[3], (), 0, 6)

    def pick(key, size, K=40):
        a, b = jax.random.split(key); x = U(a, (K,), CX-RC, CX+RC); y = U(b, (K,), CY-RC, CY+RC)
        ok = jax.vmap(lambda x, y, s: inside_poly(side, n, x, y, mo+s))(x, y, jnp.broadcast_to(size, (K,)))
        i = jnp.argmax(ok); return x[i], y[i], jnp.any(ok)
    obs = jnp.zeros((NO, 5)); ot = jnp.full((NO,), -1, jnp.int32)
    def put(obs, ot, i, row, typ, on):
        return jnp.where(on, obs.at[i].set(row), obs), jnp.where(on, ot.at[i].set(typ), ot)
    # style 1: 2-4 circles
    m1 = 2+jax.random.randint(ks[4], (), 0, 3); o1, t1 = obs, ot
    for i in range(4):
        rr = U(jax.random.fold_in(ks[5], i), (), 16., 30.); x, y, ok = pick(jax.random.fold_in(ks[6], i), rr)
        o1, t1 = put(o1, t1, i, jnp.array([x, y, rr, 0., 0.]), 0, ok & (i < m1))
    # style 2: 1-3 rects
    m2 = 1+jax.random.randint(ks[7], (), 0, 3); o2, t2 = obs, ot
    for i in range(3):
        w = U(jax.random.fold_in(ks[8], i), (), 40., 110.); h = U(jax.random.fold_in(ks[9], i), (), 30., 80.)
        x, y, ok = pick(jax.random.fold_in(ks[10], i), jnp.maximum(w, h)*.6)
        o2, t2 = put(o2, t2, i, jnp.array([x-w/2, y-h/2, w, h, 0.]), 1, ok & (i < m2))
    # style 3: one big rect
    w = U(ks[11], (), 90., 180.); h = U(ks[12], (), 60., 120.); x, y, ok = pick(ks[13], jnp.maximum(w, h)*.6)
    o3, t3 = put(obs, ot, 0, jnp.array([x-w/2, y-h/2, w, h, 0.]), 1, ok)
    # style 4: central pillar
    rr = U(ks[14], (), 34., 56.); ok4 = inside_poly(side, n, CX, CY, mo+rr)
    o4, t4 = put(obs, ot, 0, jnp.array([CX, CY, rr, 0., 0.]), 0, ok4)
    # style 5: wall with a gap
    k5 = jax.random.split(ks[15], 3); horiz = jax.random.uniform(k5[0]) < .5
    ls = jnp.arange(40., 171., 10.)
    okl = jax.vmap(lambda l: jnp.where(horiz, inside_poly(side, n, CX+l, CY, mo) & inside_poly(side, n, CX-l, CY, mo),
                                       inside_poly(side, n, CX, CY+l, mo) & inside_poly(side, n, CX, CY-l, mo)))(ls)
    good = jnp.cumprod(okl.astype(jnp.int32)) > 0
    L = jnp.max(jnp.where(good, ls, 0.)); gc = U(k5[1], (), -L+50, L-50); on5 = L >= 80
    a1h = jnp.array([CX-L, CY-8, L+gc-GAP, 16., 0.]); a2h = jnp.array([CX+gc+GAP, CY-8, L-gc-GAP, 16., 0.])
    a1v = jnp.array([CX-8, CY-L, 16., L+gc-GAP, 0.]); a2v = jnp.array([CX-8, CY+gc+GAP, 16., L-gc-GAP, 0.])
    o5, t5 = put(obs, ot, 0, jnp.where(horiz, a1h, a1v), 1, on5)
    o5, t5 = put(o5, t5, 1, jnp.where(horiz, a2h, a2v), 1, on5)
    O = jnp.stack([obs, o1, o2, o3, o4, o5]); T = jnp.stack([ot, t1, t2, t3, t4, t5])
    sel = jnp.where(enable, style, 0)
    obs, ot = O[sel], T[sel]
    reach, conn = reach_grid(side, n, obs, ot, open_mask)
    obs = jnp.where(conn, obs, 0.); ot = jnp.where(conn, ot, -1)          # an obstacle layout that cuts doors apart is dropped
    reach, _ = reach_grid(side, n, obs, ot, open_mask)
    # ---- inner wall: splits the room into two sections, each holding some of the doors
    kw = jax.random.split(jax.random.fold_in(key, 55), 3)
    px_d = side.door[:, 0]-side.normal[:, 0]*30; py_d = side.door[:, 1]-side.normal[:, 1]*30
    offs = jnp.arange(-110., 111., 10.)
    Pw = jnp.stack([CY+offs, CX+offs])                                                      # [2 axes (h, v), 23]
    cdoor = jnp.stack([py_d, px_d])                                                         # door coordinate across each axis
    na = jnp.sum(open_mask)
    A_ = jnp.sum(open_mask[None, None, :] & (cdoor[:, None, :] < Pw[:, :, None]-WALL_CLEAR), -1)
    B_ = jnp.sum(open_mask[None, None, :] & (cdoor[:, None, :] > Pw[:, :, None]+WALL_CLEAR), -1)
    okc = (A_ >= 1) & (B_ >= 1) & (A_+B_ == na)
    ci = jnp.argmax(jnp.where(okc, jax.random.gumbel(kw[0], okc.shape), -1e9).reshape(-1)); wa, wo = ci//23, ci % 23
    Pw_ = Pw[wa, wo]; horizw = wa == 0
    tt = jnp.arange(-RC*1.1, RC*1.1, 3.)
    wx = jnp.where(horizw, CX+tt, Pw_); wy = jnp.where(horizw, Pw_, CY+tt)
    insd = jax.vmap(lambda x, y: inside_poly(side, n, x, y, 2.))(wx, wy); along = jnp.where(horizw, wx, wy)
    lo = jnp.min(jnp.where(insd, along, 1e9)); hi = jnp.max(jnp.where(insd, along, -1e9))
    gated = jax.random.uniform(kw[1]) < GATED_P
    row = jnp.where(horizw, jnp.stack([lo-3, Pw_-WALL_TH/2, hi-lo+6, WALL_TH, 1.]), jnp.stack([Pw_-WALL_TH/2, lo-3, WALL_TH, hi-lo+6, 1.]))
    # drop obstacles that crowd the wall (40 px margin), then put the wall in the last slot
    def crowd(o, t):
        qx, qy = jnp.clip(o[0], row[0], row[0]+row[2]), jnp.clip(o[1], row[1], row[1]+row[3])
        circ = jnp.hypot(o[0]-qx, o[1]-qy) < o[2]+40.
        rect = (o[0] < row[0]+row[2]+40) & (o[0]+o[2] > row[0]-40) & (o[1] < row[1]+row[3]+40) & (o[1]+o[3] > row[1]-40)
        return jnp.where(t == 0, circ, jnp.where(t == 1, rect, False))
    gone = jax.vmap(crowd)(obs, ot)
    obsw = jnp.where(gone[:, None], 0., obs).at[NO-1].set(row); otw = jnp.where(gone, -1, ot).at[NO-1].set(1)
    dsec_ = jnp.where(open_mask, (cdoor[wa] >= Pw_).astype(jnp.int32), 0)
    freew = free_grid(side, n, obsw, otw)
    r0, c0 = flood(freew, side, open_mask & (dsec_ == 0)); r1, c1 = flood(freew, side, open_mask & (dsec_ == 1))
    okw = want_wall & jnp.any(okc) & (hi-lo >= WALL_MINLEN) & c0 & c1 & ~jnp.any(r0 & r1)
    obs = jnp.where(okw, obsw, obs); ot = jnp.where(okw, otw, ot)
    dsec_ = jnp.where(okw, dsec_, 0)
    k0 = jnp.argmax(open_mask)
    reach_first = jnp.where(okw, jnp.where(dsec_[k0] == 0, r0, r1), reach)
    reach = jnp.where(okw, r0 | r1, reach)
    wax = jnp.where(okw, jnp.where(horizw, 1, 2), 0).astype(jnp.int32); wpos = jnp.where(okw, Pw_, 0.)
    wkind = jnp.where(okw, jnp.where(gated, 2, 1), 0).astype(jnp.int32)
    at = lambda x, y: reach[jnp.clip((y//CS).astype(jnp.int32), 0, GH-1), jnp.clip((x//CS).astype(jnp.int32), 0, GW-1)]
    # exit pad
    kp = jax.random.split(jax.random.fold_in(key, 77), 3)
    px = U(kp[0], (80,), CX-RC, CX+RC); py = U(kp[1], (80,), CY-RC, CY+RC)
    okp = jax.vmap(lambda x, y: inside_poly(side, n, x, y, 90.) & ~hits_obstacles(x, y, 44., _ob(obs, ot)) & at(x, y))(px, py)
    xs_ = (jnp.arange(GW)+.5)*CS; ys_ = (jnp.arange(GH)+.5)*CS; XX, YY = jnp.meshgrid(xs_, ys_)
    depth = jax.vmap(jax.vmap(lambda x, y: jnp.min(jnp.where(valid, -side_dist(side, x, y), 1e9))))(XX, YY)
    clr = jax.vmap(jax.vmap(lambda x, y: ~hits_obstacles(x, y, 44., _ob(obs, ot))))(XX, YY)
    fc = jnp.argmax(jnp.where(reach & clr, depth, -1e9))                                                   # fallback: deepest reachable cell
    fb = jnp.stack([XX.reshape(-1)[fc], YY.reshape(-1)[fc]])
    j = jnp.argmax(okp); pad = jnp.where(jnp.any(okp), jnp.stack([px[j], py[j]]), fb)
    pad = jnp.where(is_exit, pad, 0.)
    # spawn points (16 pre-validated)
    ksp = jax.random.split(jax.random.fold_in(key, 99), 2)
    sx_ = CX+(U(ksp[0], (600,))*2-1)*rb; sy_ = CY+(U(ksp[1], (600,))*2-1)*rb
    oks = jax.vmap(lambda x, y: inside_poly(side, n, x, y, 40.) & ~hits_obstacles(x, y, 24., _ob(obs, ot)) & at(x, y))(sx_, sy_)
    oks = oks & ((jnp.hypot(sx_-pad[0], sy_-pad[1]) > 70) | ~is_exit)
    sec_all = jnp.where(wax == 0, 0, jnp.where(wax == 1, sy_ >= wpos, sx_ >= wpos)).astype(jnp.int32)
    rank = jnp.where(sec_all == 0, jnp.cumsum(oks & (sec_all == 0)), jnp.cumsum(oks & (sec_all == 1)))
    order = jnp.argsort(jnp.where(oks, rank*2+sec_all, 100000))[:NSP]          # both sections get their share of the 16 points
    spawn = jnp.stack([sx_[order], sy_[order]], 1); spawn_ok = oks[order].astype(jnp.int32)
    spawn_sec = jnp.where(wax == 0, 0, jnp.where(wax == 1, spawn[:, 1] >= wpos, spawn[:, 0] >= wpos)).astype(jnp.int32)
    area = jnp.sum(jnp.where(valid, half*((mx-CX)*nx+(my-CY)*ny), 0.))
    dcs = jnp.where(reach_first, jnp.hypot(XX-CX, YY-CY), 1e9); ci = jnp.argmin(dcs)             # start in the section of the first door
    startpos = jnp.stack([XX.reshape(-1)[ci], YY.reshape(-1)[ci]])
    return side, obs, ot, pad, rb, rin, spawn, spawn_ok, area, startpos, wax, wpos, wkind, dsec_, spawn_sec

# ---------------------------------------------------------------- dungeon
def gen_dungeon(SP, key, floor):
    RP = SP['n'].shape[1]
    k = jax.random.split(key, 12)
    sid = choose_spec(SP, k[0], floor)
    n = SP['n'][sid]; nbr = SP['nbr'][sid]; nbs = SP['nbs'][sid]; flp = SP['flip'][sid]; seam = SP['seam'][sid]
    hub = SP['hub'][sid] > 0; nrooms = SP['nrooms'][sid]; whole = SP['whole'][sid] > 0
    sub = SP['sub'][sid] > 0; wrapped = SP['wrapped'][sid] > 0; vcn = SP['vcn'][sid]
    validr = jnp.arange(RP) < nrooms
    sides = jnp.arange(NS_)
    exists = (nbr >= 0) & (sides[None, :] < n[:, None]) & validr[:, None]

    # ---- 1. which rooms (a connected blob)
    u = jax.random.uniform(k[1])
    dif = difficulty(floor)
    target = jnp.where(sub, jnp.round(nrooms*(.6+u*.14)).astype(jnp.int32),
                       jnp.minimum(nrooms, 24+jnp.floor(jax.random.uniform(k[2])*8).astype(jnp.int32)+vcn))
    target = jnp.maximum(6, jnp.round(target*dif['size']).astype(jnp.int32))                    # early floors are small
    seam_slots = exists & (seam > 0)
    sg = jnp.where(seam_slots, _gum(k[3], seam_slots.shape), -1e9); si = jnp.argmax(sg.reshape(-1)); sa, sk_ = si//NS_, si % NS_
    seed_seam = jnp.zeros(RP, bool).at[sa].set(True).at[jnp.maximum(nbr[sa, sk_], 0)].set(True)
    seed_rand = jnp.zeros(RP, bool).at[jnp.argmax(jnp.where(validr, _gum(k[4], (RP,)), -1e9))].set(True)
    seed = jnp.where(jnp.any(hub), hub, jnp.where(wrapped & jnp.any(seam_slots), seed_seam, seed_rand))
    def grow(i, c):
        sel, key = c; key, kk = jax.random.split(key)
        B = jnp.maximum(nbr, 0)
        front = exists & sel[:, None] & ~sel[B]
        g = jnp.where(front, _gum(kk, front.shape), -1e9).reshape(-1); j = jnp.argmax(g)
        ok = jnp.any(front) & (jnp.sum(sel) < target)
        return jnp.where(ok, sel.at[B.reshape(-1)[j]].set(True), sel), key
    sel, _ = lax.fori_loop(0, 48, grow, (seed, k[5]))
    sel = jnp.where(whole & (dif['size'] >= .99), validr, sel)
    # ---- 2. doors: spanning tree + loops + forced corner loops
    B = jnp.maximum(nbr, 0)
    A_idx = jnp.broadcast_to(jnp.arange(RP)[:, None], (RP, NS_))
    edge = exists & sel[:, None] & sel[B] & (A_idx < B)
    w = jnp.where(edge, jax.random.uniform(k[6], edge.shape), 2.).reshape(-1)
    order = jnp.argsort(w)[:256]
    ea, ek = order//NS_, order % NS_; eb = B.reshape(-1)[order]; evalid = edge.reshape(-1)[order]
    def kruskal(i, c):
        comp, chosen = c; a, b = ea[i], eb[i]
        join = evalid[i] & (comp[a] != comp[b])
        comp = jnp.where(join & (comp == comp[b]), comp[a], comp)
        return comp, chosen.at[i].set(join)
    _, tree = lax.fori_loop(0, 256, kruskal, (jnp.arange(RP), jnp.zeros(256, bool)))
    openm = jnp.zeros(RP*NS_, bool).at[order].set(tree).reshape(RP, NS_)
    extraP = jnp.where(wrapped, .24, .16)
    openm = openm | (edge & (jax.random.uniform(k[7], edge.shape) < extraP))
    # make sure a seam can be walked through (two of them)
    sedge = edge & (seam > 0)
    have = jnp.sum(openm & sedge)
    sc_ = jnp.where(sedge & ~openm, _gum(k[8], sedge.shape), -1e9).reshape(-1)
    rank = jnp.argsort(jnp.argsort(-sc_)).reshape(sedge.shape)
    openm = openm | (wrapped & sedge & ~openm & (rank < jnp.maximum(0, 2-have)))
    # forced loops around corners lying fully inside the dungeon
    cr, ck, cl = SP['cyc_room'][sid], SP['cyc_k'][sid], SP['cyc_len'][sid]
    L = cr.shape[1]; lm = jnp.arange(L)[None, :] < cl[:, None]
    full = (cl > 0) & jnp.all(jnp.where(lm, sel[cr], True), axis=1)
    allhub = jnp.all(jnp.where(lm, hub[cr], True), axis=1)
    force = full & (allhub | (jax.random.uniform(k[9], cl.shape) < .45))
    fm = lm & force[:, None]
    fr, fk = cr.reshape(-1), ck.reshape(-1)
    forced = jnp.zeros((RP, NS_), bool).at[jnp.where(fm.reshape(-1), fr, 0), jnp.where(fm.reshape(-1), fk, 0)].max(fm.reshape(-1))
    openm = openm | (forced & exists & sel[:, None] & sel[B])
    # symmetrise (an edge is open from both ends)
    sym = jnp.zeros((RP, NS_), bool).at[B.reshape(-1), nbs.reshape(-1)].max((openm & exists).reshape(-1))
    openm = openm | sym
    # ---- 3. compact to R rooms
    m = jnp.sum(sel)
    ordr = jnp.argsort(~sel)[:R]                       # selected rooms first, in patch order
    newid = jnp.cumsum(sel)-1
    live = jnp.arange(R) < m
    pn = jnp.where(live, n[ordr], 0)
    dn = jnp.where(live[:, None] & openm[ordr], newid[B[ordr]], -1)
    arr = jnp.where(dn >= 0, nbs[ordr], 0); flipc = jnp.where(dn >= 0, flp[ordr], 0)
    # ---- 4. distances, start, exit
    adj = jnp.zeros((R, R), bool).at[jnp.broadcast_to(jnp.arange(R)[:, None], (R, NS_)).reshape(-1), jnp.maximum(dn, 0).reshape(-1)].max((dn >= 0).reshape(-1))
    dist = jnp.where(adj, 1, INF); dist = jnp.where(jnp.eye(R, dtype=bool), 0, dist)
    dist = lax.fori_loop(0, R, lambda kk, D: jnp.minimum(D, D[:, kk, None]+D[None, kk, :]), dist)
    hubc = hub[ordr] & live
    cand = live & ~hubc; cand = jnp.where(jnp.any(cand), cand, live)
    start = jnp.argmax(jnp.where(cand, _gum(k[10], (R,)), -1e9)).astype(jnp.int32)
    depth = jnp.where(live & (dist[start] < INF), dist[start], -1)
    exit_ = jnp.argmax(jnp.where(depth >= 0, depth+jax.random.uniform(k[11], (R,))*.5, -1e9)).astype(jnp.int32)
    # ---- 5. rooms
    dk = jax.random.split(jax.random.fold_in(key, 5), R)
    want = jax.random.uniform(jax.random.fold_in(key, 6), (R,)) < dif['wall_p']
    side, obs, ot, pad, rb, rin, spawn, sok, area, spos, wax, wpos, wkind, dsec, ssec = jax.vmap(design_room)(dk, jnp.maximum(pn, 3), jnp.arange(R) == exit_, dn >= 0, want)
    z = lambda a: jnp.where(live.reshape((R,)+(1,)*(a.ndim-1)), a, 0)
    wax, wpos, wkind, dsec, ssec = z(wax), z(wpos), z(wkind), z(dsec), z(ssec)
    # ---- 6. gating: solid walls split the world, so open gates until every section can be reached (as fixSections in JS)
    nb_ = jnp.maximum(dn, 0); ridx = jnp.arange(R)
    def node_adj(wk):
        hn = lambda r, sec: 2*r + jnp.where(wk[r] == 1, sec, 0)
        fr_ = hn(ridx[:, None], dsec); to_ = hn(nb_, dsec[nb_, arr])
        return jnp.zeros((2*R, 2*R), bool).at[fr_.reshape(-1), to_.reshape(-1)].max((dn >= 0).reshape(-1))
    k0s = jnp.argmax(dn[start] >= 0); start_node = 2*start + jnp.where(wkind[start] == 1, dsec[start, k0s], 0)
    def fix(i, c):
        wk, kk = c; kk, kg = jax.random.split(kk)
        A = node_adj(wk)
        seen = lax.fori_loop(0, 2*R, lambda j, sn: sn | jnp.any(A & sn[:, None], axis=0), jnp.zeros(2*R, bool).at[start_node].set(True))
        bad = (wk == 1) & live & (seen[0::2] != seen[1::2])
        pick = jnp.argmax(jnp.where(bad, _gum(kg, (R,)), -1e9))
        return jnp.where(jnp.any(bad) & (ridx == pick), 2, wk), kk
    wkind, _ = lax.fori_loop(0, R, fix, (wkind, jax.random.fold_in(key, 8)))
    A = node_adj(wkind)
    nd0 = jnp.where(A, 1, INF); nd0 = jnp.where(jnp.eye(2*R, dtype=bool), 0, nd0)
    ndist = jnp.minimum(lax.fori_loop(0, 2*R, lambda kk, D: jnp.minimum(D, D[:, kk, None]+D[None, kk, :]), nd0), INF)
    side = jax.tree_util.tree_map(z, side)
    return Dungeon(n=pn, doors=dn.astype(jnp.int32), arrive=arr.astype(jnp.int32), flip=flipc.astype(jnp.int32),
                   sides=side,
                   obstacles=_ob(z(obs), jnp.where(live[:, None], ot, -1)), pad=z(pad), radius=z(rb), inradius=z(rin), area=z(area),
                   spawn=z(spawn), spawn_ok=z(sok), spawn_sec=ssec, wall_axis=wax, wall_pos=wpos, wall_kind=wkind, door_sec=dsec.astype(jnp.int32),
                   depth=depth.astype(jnp.int32), map_pos=z(SP['cvec'][sid][ordr]), door_seam=jnp.where(dn >= 0, SP['seam'][sid][ordr], 0).astype(jnp.int32),
                   n_rooms=m.astype(jnp.int32), start=start, startpos=spos[start], exit=exit_, floor=jnp.asarray(floor, jnp.int32),
                   category=SP['cat'][sid]+0*sid, spec=sid, holonomy=SP['hol'][sid], dist=dist.astype(jnp.int32), node_dist=ndist.astype(jnp.int32))

def gen_source(SP):
    """source(key, floor, idx=None) for sim.game.Game: a fresh GPU-generated dungeon for that floor"""
    def src(key, floor, idx=None):
        return gen_dungeon(SP, key, floor)
    return src
