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
    """dungeon tables -> canonical dict (obstacles split into named parts, sides into named columns)"""
    side = d['side']; obs = d['obs']; ot = d['obs_t']
    return dict(n_rooms=d['n_rooms'], n=d['n'], doors=d['doors'], arr=d['arr'], flip=d['flip'],
                side_mid=side[..., 0:2], side_normal=side[..., 2:4], side_tangent=side[..., 4:6], side_door=side[..., 6:8],
                obs_kind=ot, obs_pos=obs[..., 0:2], obs_size=obs[..., 2:4], obs_is_wall=obs[..., 4] > 0,
                pad=d['pad'], rb=d['rb'], rin=d['rin'], start=d['start'], startpos=d['startpos'], exit=d['exit'], floor=d['floor'],
                geo=d['geo'], depth=d['depth'], dist=d['dist'], wax=d['wax'], wpos=d['wpos'], wkind=d['wkind'], dsec=d['dsec'],
                spawn_sec=d['spawn_sec'], ndist=d['ndist'], spawn=d['spawn'], spawn_ok=d['spawn_ok'], area=d['area'], spec=d['spec'],
                hol=d['hol'], mpos=d['mpos'], dseam=d['dseam'])


def canon_step(s):
    """per-step scalars of one game"""
    return dict(room=s['room'], x=s['x'], y=s['y'], va=s['va'], mir=s['mir'], hp=s['hp'], kills=s['kills'], inv=s['inv'],
                fire=s['fire'], aim=s['aim'], t=s['t'], steps=s['steps'], floor=s['dun']['floor'],
                hroom=s['hroom'], hsec=s['hsec'], hx=s['hx'], hy=s['hy'], hhp=s['hhp'], htimer=s['htimer'], hhere=s['hhere'],
                hdead=s['hdead'], hflash=s['hflash'],
                n_alive=jnp.sum(s['en'][s['room'], :, 7] > .5), n_pb=jnp.sum(s['pb'][:, 5] > .5), n_eb=jnp.sum(s['eb'][:, 5] > .5))


def canon_state(s):
    """the whole state of one game (minus the dungeon, checked separately)"""
    en, pb, eb, pk = s['en'], s['pb'], s['eb'], s['pk']
    out = canon_step(s)
    out.update(visited=s['visited'], wopen=s['wopen'], spawned=s['spawned'], key=jax.random.key_data(s['key']) if jnp.issubdtype(s['key'].dtype, jax.dtypes.prng_key) else s['key'],
               en_pos=en[..., 0:2], en_hp=en[..., 2], en_shooter=en[..., 3] > .5, en_fire=en[..., 4], en_telegraph=en[..., 5],
               en_flash=en[..., 6], en_alive=en[..., 7] > .5,
               pb_pos=pb[:, 0:2], pb_vel=pb[:, 2:4], pb_life=pb[:, 4], pb_alive=pb[:, 5] > .5,
               eb_pos=eb[:, 0:2], eb_vel=eb[:, 2:4], eb_life=eb[:, 4], eb_alive=eb[:, 5] > .5,
               pk_pos=pk[..., 0:2], pk_alive=pk[..., 2] > .5)
    return out


def dungeon_of(s): return s['dun']


def keep_alive(s, on):
    """top up hp (so these games live long enough to explore)"""
    return {**s, 'hp': jnp.where(on, jnp.int32(5), s['hp'])}


def to_stairs(s):
    """stand on the exit pad of a cleared exit room (as the viewer's debug key N)"""
    d = s['dun']; ex = d['exit']; pd = d['pad'][ex]
    return {**s, 'room': ex, 'x': pd[0], 'y': pd[1], 'spawned': jnp.ones_like(s['spawned']), 'inv': jnp.float32(9.),
            'en': s['en'].at[ex, :, 7].set(0.), 'wopen': s['wopen'] | (d['wkind'] > 0)}


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
