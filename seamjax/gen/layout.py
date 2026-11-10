"""
The layout of a dungeon on a tiling patch (specs.py), in stages:

  choose_spec     which world (tiling), by the floor's mix of world kinds
  pick_rooms      a connected blob of rooms of the floor's size
  open_doors      a random spanning tree, extra loops, walkable seams, forced loops round corners (so curvature can be felt)
  compact         renumber the chosen rooms 0..m-1 and pad to MAX_ROOMS
  start_and_exit  door distances; start somewhere, exit far away
  fix_sections    solid inner walls split the world; turn some into gated ones until every section is reachable
"""
import jax, jax.numpy as jnp
from jax import lax
from seamjax.core.config import MAX_ROOMS as R, MAX_SIDES as NS, INF
from seamjax.core.struct import pytree
from seamjax.core.types import WALL_SOLID, WALL_GATED


def _gum(key, shape): return jax.random.gumbel(key, shape)

def _all_pairs(adj, n):
    """door distances from a boolean adjacency matrix (Floyd-Warshall); INF where unreachable"""
    d = jnp.where(adj, 1, INF); d = jnp.where(jnp.eye(n, dtype=bool), 0, d)
    return lax.fori_loop(0, n, lambda kk, D: jnp.minimum(D, D[:, kk, None]+D[None, kk, :]), d)


def choose_spec(SP, key, floor_mix):
    """a tiling, so that each kind of world (flat, hyperbolic, spherical, wrapped) comes up with probability floor_mix[kind]"""
    cat = SP['cat']; cnt = jnp.array([jnp.sum(cat == c) for c in range(4)], jnp.float32)
    w = floor_mix[cat]/cnt[cat]
    return jnp.argmax(jnp.where(w > 0, jnp.log(jnp.maximum(w, 1e-30)) + _gum(key, cat.shape), -1e9)).astype(jnp.int32)


def tiling(SP, sid):
    """the tables of one tiling patch; rooms are patch rooms (RP of them), sides are 0..NS-1"""
    t = {k: SP[k][sid] for k in ('n', 'nbr', 'nbs', 'flip', 'seam', 'nrooms', 'vcn', 'cyc_room', 'cyc_k', 'cyc_len', 'cvec')}
    t.update(hub=SP['hub'][sid] > 0, whole=SP['whole'][sid] > 0, sub=SP['sub'][sid] > 0, wrapped=SP['wrapped'][sid] > 0)
    RP = t['n'].shape[0]
    t['valid'] = jnp.arange(RP) < t['nrooms']
    t['exists'] = (t['nbr'] >= 0) & (jnp.arange(NS)[None, :] < t['n'][:, None]) & t['valid'][:, None]      # side has a neighbour
    t['B'] = jnp.maximum(t['nbr'], 0)
    return t


def pick_rooms(t, keys, size):
    """a connected set of rooms grown from a seed: the hub (spheres), a seam (wrapped worlds) or a random room"""
    k1, k2, k3, k4, k5 = keys
    nrooms, exists, nbr, B, RP = t['nrooms'], t['exists'], t['nbr'], t['B'], t['n'].shape[0]
    u = jax.random.uniform(k1)
    target = jnp.where(t['sub'], jnp.round(nrooms*(.6+u*.14)).astype(jnp.int32),
                       jnp.minimum(nrooms, 24+jnp.floor(jax.random.uniform(k2)*8).astype(jnp.int32)+t['vcn']))
    target = jnp.maximum(6, jnp.round(target*size).astype(jnp.int32))                        # early floors are small
    seam_slots = exists & (t['seam'] > 0)
    sg = jnp.where(seam_slots, _gum(k3, seam_slots.shape), -1e9); si = jnp.argmax(sg.reshape(-1)); sa, sk_ = si//NS, si % NS
    seed_seam = jnp.zeros(RP, bool).at[sa].set(True).at[jnp.maximum(nbr[sa, sk_], 0)].set(True)
    seed_rand = jnp.zeros(RP, bool).at[jnp.argmax(jnp.where(t['valid'], _gum(k4, (RP,)), -1e9))].set(True)
    seed = jnp.where(jnp.any(t['hub']), t['hub'], jnp.where(t['wrapped'] & jnp.any(seam_slots), seed_seam, seed_rand))
    def grow(i, c):
        sel, key = c; key, kk = jax.random.split(key)
        front = exists & sel[:, None] & ~sel[B]
        g = jnp.where(front, _gum(kk, front.shape), -1e9).reshape(-1); j = jnp.argmax(g)
        ok = jnp.any(front) & (jnp.sum(sel) < target)
        return jnp.where(ok, sel.at[B.reshape(-1)[j]].set(True), sel), key
    sel, _ = lax.fori_loop(0, 48, grow, (seed, k5))
    return jnp.where(t['whole'] & (size >= .99), t['valid'], sel)


def open_doors(t, sel, keys):
    """which sides of the chosen rooms get a door. [RP, NS] bool, open from both ends"""
    k6, k7, k8, k9 = keys
    exists, B, seam, wrapped, RP = t['exists'], t['B'], t['seam'], t['wrapped'], t['n'].shape[0]
    A_idx = jnp.broadcast_to(jnp.arange(RP)[:, None], (RP, NS))
    edge = exists & sel[:, None] & sel[B] & (A_idx < B)                 # each connection once
    # a random spanning tree (Kruskal on random weights)
    w = jnp.where(edge, jax.random.uniform(k6, edge.shape), 2.).reshape(-1)
    order = jnp.argsort(w)[:256]
    ea = order//NS; eb = B.reshape(-1)[order]; evalid = edge.reshape(-1)[order]
    def kruskal(i, c):
        comp, chosen = c; a, b = ea[i], eb[i]
        join = evalid[i] & (comp[a] != comp[b])
        comp = jnp.where(join & (comp == comp[b]), comp[a], comp)
        return comp, chosen.at[i].set(join)
    _, tree = lax.fori_loop(0, 256, kruskal, (jnp.arange(RP), jnp.zeros(256, bool)))
    openm = jnp.zeros(RP*NS, bool).at[order].set(tree).reshape(RP, NS)
    # extra loops
    openm = openm | (edge & (jax.random.uniform(k7, edge.shape) < jnp.where(wrapped, .24, .16)))
    # a wrapped world gets at least two walkable seams
    sedge = edge & (seam > 0)
    have = jnp.sum(openm & sedge)
    sc_ = jnp.where(sedge & ~openm, _gum(k8, sedge.shape), -1e9).reshape(-1)
    rank = jnp.argsort(jnp.argsort(-sc_)).reshape(sedge.shape)
    openm = openm | (wrapped & sedge & ~openm & (rank < jnp.maximum(0, 2-have)))
    # forced loops round corners that lie fully inside the dungeon (all of them on a hub, else 45%)
    cr, ck, cl = t['cyc_room'], t['cyc_k'], t['cyc_len']
    L = cr.shape[1]; lm = jnp.arange(L)[None, :] < cl[:, None]
    full = (cl > 0) & jnp.all(jnp.where(lm, sel[cr], True), axis=1)
    allhub = jnp.all(jnp.where(lm, t['hub'][cr], True), axis=1)
    force = full & (allhub | (jax.random.uniform(k9, cl.shape) < .45))
    fm = (lm & force[:, None]).reshape(-1)
    forced = jnp.zeros((RP, NS), bool).at[jnp.where(fm, cr.reshape(-1), 0), jnp.where(fm, ck.reshape(-1), 0)].max(fm)
    openm = openm | (forced & exists & sel[:, None] & sel[B])
    # symmetrise
    sym = jnp.zeros((RP, NS), bool).at[B.reshape(-1), t['nbs'].reshape(-1)].max((openm & exists).reshape(-1))
    return openm | sym


@pytree
class Compact:
    """the chosen rooms renumbered 0..m-1 and padded to R"""
    m: object           # number of rooms
    live: object        # [R] slot is a room
    order: object       # [R] patch room of each slot
    n: object           # [R]
    doors: object       # [R, NS] -1 = no door
    arrive: object
    flip: object


def compact(t, sel, openm):
    m = jnp.sum(sel)
    ordr = jnp.argsort(~sel)[:R]                       # selected rooms first, in patch order
    newid = jnp.cumsum(sel)-1
    live = jnp.arange(R) < m
    dn = jnp.where(live[:, None] & openm[ordr], newid[t['B'][ordr]], -1)
    return Compact(m=m, live=live, order=ordr, n=jnp.where(live, t['n'][ordr], 0), doors=dn,
                   arrive=jnp.where(dn >= 0, t['nbs'][ordr], 0), flip=jnp.where(dn >= 0, t['flip'][ordr], 0))


def start_and_exit(c, hub, k10, k11):
    """door distances; the start is a random non-hub room, the exit one of the farthest. Returns (dist, start, depth, exit)"""
    dn = c.doors
    adj = jnp.zeros((R, R), bool).at[jnp.broadcast_to(jnp.arange(R)[:, None], (R, NS)).reshape(-1), jnp.maximum(dn, 0).reshape(-1)].max((dn >= 0).reshape(-1))
    dist = _all_pairs(adj, R)
    hubc = hub[c.order] & c.live
    cand = c.live & ~hubc; cand = jnp.where(jnp.any(cand), cand, c.live)
    start = jnp.argmax(jnp.where(cand, _gum(k10, (R,)), -1e9)).astype(jnp.int32)
    depth = jnp.where(c.live & (dist[start] < INF), dist[start], -1)
    exit_ = jnp.argmax(jnp.where(depth >= 0, depth+jax.random.uniform(k11, (R,))*.5, -1e9)).astype(jnp.int32)
    return dist, start, depth, exit_


def fix_sections(key, c, start, wkind, dsec):
    """turn solid walls into gated ones, at random, until every room-section is reachable from the start (as fixSections in the
    JS game). Returns (wall kinds, node distances [2R, 2R]); a node is a room-section as the Hunter sees it (sim/queries.hnode)."""
    dn, arr = c.doors, c.arrive
    nb_ = jnp.maximum(dn, 0); ridx = jnp.arange(R)
    def node_adj(wk):
        hn = lambda r, sec: 2*r + jnp.where(wk[r] == WALL_SOLID, sec, 0)
        fr_ = hn(ridx[:, None], dsec); to_ = hn(nb_, dsec[nb_, arr])
        return jnp.zeros((2*R, 2*R), bool).at[fr_.reshape(-1), to_.reshape(-1)].max((dn >= 0).reshape(-1))
    k0s = jnp.argmax(dn[start] >= 0); start_node = 2*start + jnp.where(wkind[start] == WALL_SOLID, dsec[start, k0s], 0)
    def fix(i, cc):
        wk, kk = cc; kk, kg = jax.random.split(kk)
        A = node_adj(wk)
        seen = lax.fori_loop(0, 2*R, lambda j, sn: sn | jnp.any(A & sn[:, None], axis=0), jnp.zeros(2*R, bool).at[start_node].set(True))
        bad = (wk == WALL_SOLID) & c.live & (seen[0::2] != seen[1::2])
        pick = jnp.argmax(jnp.where(bad, _gum(kg, (R,)), -1e9))
        return jnp.where(jnp.any(bad) & (ridx == pick), WALL_GATED, wk), kk
    wkind, _ = lax.fori_loop(0, R, fix, (wkind, jax.random.fold_in(key, 8)))
    return wkind, jnp.minimum(_all_pairs(node_adj(wkind), 2*R), INF)
