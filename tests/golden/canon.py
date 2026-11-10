"""
Golden-trace driver: runs the simulation and generator through their public API and returns a dict of named arrays
whose meaning does not depend on how the state is laid out internally.

This file is the ONLY place the golden test knows the state layout. When the layout changes (a refactor), update the
adapters below so they return the same canonical arrays; the recorded trace (trace.npz) must then still match.
"""
import numpy as np, jax, jax.numpy as jnp
from jax import lax

B, T, EVERY = 16, 2400, 30                       # games, steps (1/60 s each), observation sampling period
FLOORS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 30], np.int32)
DUNGEON_FLOORS = np.arange(1, 17, dtype=np.int32)


# ---------------------------------------------------------------- adapters (the only layout-specific code)
def api():
    from seamjax.specs import load_specs
    from seamjax.gen import gen_source, gen_dungeon
    from seamjax.env import make_env
    SP = load_specs()
    E = make_env(gen_source(SP))
    return dict(SP=SP, gen=lambda k, f: gen_dungeon(SP, k, f), reset_batch=E['reset_batch'], step_batch=E['step_auto_batch'],
                observe=E['observe'], hunter_door=E['hunter_door'], hunter_dist=E['hunter_dist'])


def canon_dungeon(d):
    """dungeon tables -> canonical dict"""
    sd, ob = d.sides, d.obstacles
    return dict(n_rooms=d.n_rooms, n=d.n, doors=d.doors, arr=d.arrive, flip=d.flip,
                side_mid=sd.mid, side_normal=sd.normal, side_tangent=sd.tangent, side_door=sd.door,
                obs_kind=ob.kind, obs_pos=ob.pos, obs_size=ob.size, obs_is_wall=ob.wall,
                pad=d.pad, rb=d.radius, rin=d.inradius, start=d.start, startpos=d.startpos, exit=d.exit, floor=d.floor,
                geo=d.category, depth=d.depth, dist=d.dist, wax=d.wall_axis, wpos=d.wall_pos, wkind=d.wall_kind, dsec=d.door_sec,
                spawn_sec=d.spawn_sec, ndist=d.node_dist, spawn=d.spawn, spawn_ok=d.spawn_ok, area=d.area, spec=d.spec,
                hol=d.holonomy, mpos=d.map_pos, dseam=d.door_seam)


def canon_step(s):
    """per-step scalars of one game"""
    P, H = s.player, s.hunter
    return dict(room=s.room, x=P.x, y=P.y, va=s.view_angle, mir=s.mirror, hp=P.hp, kills=s.kills, inv=P.inv,
                fire=P.fire, aim=P.aim, t=s.t, steps=s.steps, floor=s.dun.floor,
                hroom=H.room, hsec=H.sec, hx=H.x, hy=H.y, hhp=H.hp, htimer=H.timer, hhere=H.here,
                hdead=H.dead, hflash=H.flash,
                n_alive=jnp.sum(s.enemies.alive[s.room]), n_pb=jnp.sum(s.pb.alive), n_eb=jnp.sum(s.eb.alive))


def canon_state(s):
    """the whole state of one game (minus the dungeon, checked separately)"""
    en, pb, eb, pk = s.enemies, s.pb, s.eb, s.pickups
    xy = lambda p: jnp.stack([p.x, p.y], -1)
    out = canon_step(s)
    out.update(visited=s.visited, wopen=s.wall_open, spawned=s.spawned, key=s.key,
               en_pos=xy(en), en_hp=en.hp, en_shooter=en.shooter, en_fire=en.fire, en_telegraph=en.telegraph,
               en_flash=en.flash, en_alive=en.alive,
               pb_pos=xy(pb), pb_vel=jnp.stack([pb.vx, pb.vy], -1), pb_life=pb.life, pb_alive=pb.alive,
               eb_pos=xy(eb), eb_vel=jnp.stack([eb.vx, eb.vy], -1), eb_life=eb.life, eb_alive=eb.alive,
               pk_pos=xy(pk), pk_alive=pk.alive)
    return out


def dungeon_of(s): return s.dun


def keep_alive(s, on):
    """top up hp (so these games live long enough to explore)"""
    return s.replace(player=s.player.replace(hp=jnp.where(on, jnp.int32(5), s.player.hp)))


def to_stairs(s):
    """stand on the exit pad of a cleared exit room (as the viewer's debug key N)"""
    d = s.dun; ex = d.exit; pd = d.pad[ex]
    return s.replace(room=ex, player=s.player.replace(x=pd[0], y=pd[1], inv=jnp.float32(9.)), spawned=jnp.ones_like(s.spawned),
                     enemies=s.enemies.replace(alive=s.enemies.alive.at[ex].set(False)), wall_open=s.wall_open | (d.wall_kind > 0))


# ---------------------------------------------------------------- driver (layout independent)
def _where(m, a, b):
    return jax.tree_util.tree_map(lambda x, y: jnp.where(m.reshape((-1,)+(1,)*(x.ndim-1)), x, y), a, b)


def policy(o, t, i):
    """a deterministic player reading only the observation vector: aim at the nearest enemy and strafe, else wander"""
    ef = o[8+12*8:8+12*8+5]
    v = ef[1:3]/(jnp.hypot(ef[1], ef[2])+1e-9)
    side = jnp.where((t//40) % 2 == 0, 1., -1.)
    fight = jnp.concatenate([jnp.stack([v[1], -v[0]])*side*.7 - v*(jnp.hypot(ef[1], ef[2]) < .5)*.5, v])
    ph = t/30. + i*.7
    wander = jnp.stack([jnp.cos(ph*.9), jnp.sin(ph*.6), jnp.cos(ph*2.), jnp.sin(ph*2.)])
    return jnp.where(ef[0] > .5, fight, wander).astype(jnp.float32)


def run():
    A = api()
    out = {}
    D = jax.jit(jax.vmap(A['gen']))(jax.random.split(jax.random.PRNGKey(123), len(DUNGEON_FLOORS)), jnp.asarray(DUNGEON_FLOORS))
    out.update({'dun/'+k: v for k, v in canon_dungeon(D).items()})
    obs_b = jax.vmap(A['observe'])

    @jax.jit
    def sim(S, key):
        def body(c, t):
            S, key = c; key, k = jax.random.split(key)
            S = jax.vmap(keep_alive)(S, jnp.arange(B) % 2 == 1)                       # odd games cannot die: they explore further
            stairs = (t % 600 == 300) & (jnp.arange(B) % 4 == 0)                      # every 4th game takes the stairs every 10 s
            S = _where(stairs, jax.vmap(to_stairs)(S), S)
            o = obs_b(S)
            a = jax.vmap(policy, in_axes=(0, None, 0))(o, t, jnp.arange(B))
            S, r, done, info = A['step_batch'](S, a, k)
            rec = jax.vmap(canon_step)(S)
            rec.update(reward=r, done=done, descended=info['descended'], hdoor=jax.vmap(A['hunter_door'])(S), hdist=jax.vmap(A['hunter_dist'])(S))
            return (S, key), (rec, o)
        (S, _), (rec, obs) = lax.scan(body, (S, key), jnp.arange(T))
        return S, rec, obs[::EVERY]
    S0 = jax.jit(A['reset_batch'])(jax.random.split(jax.random.PRNGKey(7), B), jnp.asarray(FLOORS))
    out.update({'start/'+k: v for k, v in jax.vmap(canon_state)(S0).items()})
    S, rec, obs = sim(S0, jax.random.PRNGKey(8))
    out.update({'step/'+k: v for k, v in rec.items()})
    out['obs'] = obs
    out.update({'end/'+k: v for k, v in jax.vmap(canon_state)(S).items()})
    out.update({'enddun/'+k: v for k, v in canon_dungeon(jax.vmap(dungeon_of)(S)).items()})
    return {k: np.asarray(v) for k, v in out.items()}
