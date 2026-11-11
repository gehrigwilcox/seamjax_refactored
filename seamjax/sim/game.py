"""
A game run: starting a run, the next floor, and steps that carry on by themselves (death -> new run, stairs -> next floor).

    G = Game.generated()                     # dungeons from the GPU generator (seamjax/gen)
    s = G.reset(key, floor)                  # one game;  G.reset_batch(keys, floors) for many
    s, reward, done, ev = G.step(s, action)  # one frame (pure: seamjax.sim.step.step)
    S, reward, done, info = G.step_auto_batch(S, actions, key)     # many games, finished ones replaced

Everything here is jit / vmap friendly. A Game only holds the dungeon source: source(key, floor, idx=None) -> Dungeon.
"""
import jax, jax.numpy as jnp
from jax import lax
from seamjax.difficulty import difficulty
from seamjax.core import struct
from seamjax.core import kinds
from seamjax.core.config import MAX_PLAYER_BULLETS as NB, MAX_ENEMY_BULLETS as NEB, MAX_PICKUPS as NPK, HUNTER_HP
from seamjax.core.types import Player, Pickups, Hunter, GameState
from seamjax.sim.state import empty_bullets, empty_enemies, class_of
from seamjax.sim.queries import sec_point
from seamjax.sim.step import step as sim_step, DT
from seamjax.sim.reward import reward, done as is_done
from seamjax.sim.batch import cvmap, regen_where


class Game:
    def __init__(self, source):
        self.source = source
        self.R = source(jax.random.PRNGKey(0), jnp.int32(1), 0).n.shape[0]

    @classmethod
    def generated(cls, SP=None):
        """dungeons from the on-device generator, from the tiling tables SP (loaded if not given)"""
        from seamjax.gen import gen_source
        from seamjax.specs import load_specs
        return cls(gen_source(load_specs() if SP is None else SP))

    # ---------------------------------------------------------------- starting
    def reset(self, key, floor=1, d=None, cls=0):
        """a fresh game at `floor` (floor 1 is the start of a run), playing class `cls` (index into kinds.PLAYER_CLASSES)"""
        R = self.R
        k0, k2, k3 = jax.random.split(key, 3)
        dun = self.source(k0, jnp.asarray(floor, jnp.int32), d)
        start = dun.start; ex = dun.exit
        # the Hunter starts at least half way to the exit
        depth = dun.depth; thr = jnp.maximum(2, jnp.floor(depth[ex]*.5).astype(jnp.int32))
        cand = (depth >= thr) & (jnp.arange(R) < dun.n_rooms)
        gum = jax.random.gumbel(k2, (R,))
        hroom = jnp.where(jnp.any(cand), jnp.argmax(jnp.where(cand, gum, -1e9)), ex)
        x, y = dun.startpos[0], dun.startpos[1]                     # a reachable spot near the middle of the start room
        hsec = jnp.where(dun.wall_kind[hroom] > 0, jax.random.bernoulli(jax.random.fold_in(k2, 3)), False).astype(jnp.int32)
        z = jnp.zeros((R, NPK))
        s = GameState(dun=dun, room=start, view_angle=0., mirror=1.,
                      player=Player(x=x, y=y, hp=jnp.asarray(kinds.class_table().max_hp)[cls], inv=0., fire=0., aim=0., cls=jnp.asarray(cls, jnp.int32), swing=0.),
                      pb=empty_bullets(NB), eb=empty_bullets(NEB), enemies=empty_enemies(R), pickups=Pickups(x=z, y=z, alive=jnp.zeros((R, NPK), bool)),
                      hunter=Hunter(room=hroom.astype(jnp.int32), sec=hsec, timer=difficulty(floor)['hunter_first'], here=False, x=0., y=0.,
                                    hp=HUNTER_HP, dead=0., flash=0.),
                      visited=jnp.zeros(R, bool).at[start].set(True), wall_open=jnp.zeros(R, bool),
                      spawned=jnp.zeros((R, 2), bool).at[start, sec_point(dun, start, x, y)].set(True),       # the start section stays empty
                      t=0., steps=jnp.int32(0), kills=jnp.int32(0), key=k3)
        return jax.tree_util.tree_map(jnp.asarray, s)

    def reset_batch(self, keys, floors, classes=None):
        if classes is None: return cvmap(self.reset)(keys, floors)
        return cvmap(lambda k, f, c: self.reset(k, f, cls=c))(keys, floors, classes)

    def next_state(self, s, key, done, desc):
        """what follows a finished step: death starts a new run at floor 1; stairs continue the run one floor deeper
        (hp +1, kills / time carried over). The class stays."""
        floor = jnp.where(done, 1, s.dun.floor+1)
        n = self.reset(key, floor, cls=s.player.cls)
        keep = desc & ~done
        return n.replace(player=n.player.replace(hp=jnp.where(keep, jnp.minimum(class_of(s, 'max_hp'), s.player.hp+1), n.player.hp)),
                         kills=jnp.where(keep, s.kills, 0), steps=jnp.where(keep, s.steps, 0), t=jnp.where(keep, s.t, 0.))

    # ---------------------------------------------------------------- stepping
    @staticmethod
    def step(s, action, dt=DT):
        """one frame. Returns (state, reward, done, StepEvents)"""
        s, ev = sim_step(s, action, dt)
        return s, reward(s, ev), is_done(s), ev

    @staticmethod
    def step_frames(s, action, n):
        """n frames with the same action, frozen from the frame where you die or take the stairs (so nothing leaks past it).
        Returns (state, summed reward, done, descended)"""
        def body(c, _):
            s, rew, stop, desc = c
            s2, r, d, ev = Game.step(s, action)
            keep = struct.where(stop, s.replace(dun=None), s2.replace(dun=None)).replace(dun=s.dun)      # (the dungeon never changes in a step)
            return (keep, rew + jnp.where(stop, 0., r), stop | d | ev.descended, jnp.where(stop, desc, ev.descended)), None
        (s, rew, _, desc), _ = lax.scan(body, (s, jnp.float32(0.), jnp.bool_(False), jnp.bool_(False)), None, length=n)
        return s, rew, is_done(s), desc

    def _advance(self, S, actions, frame_skip):
        """batched frames: (S, reward, done, descended)"""
        if frame_skip == 1:
            S, r, done, ev = jax.vmap(self.step)(S, actions); return S, r, done, ev.descended
        return jax.vmap(lambda s, a: self.step_frames(s, a, frame_skip))(S, actions)

    def step_auto(self, s, action, key, dt=DT):
        """one game that never ends: death -> new run, stairs -> next floor. Returns (state, reward, done, info)"""
        s2, r, done, ev = self.step(s, action, dt)
        desc = ev.descended & ~done
        s3 = lax.cond(done | desc, lambda s_: self.next_state(s_, key, done, desc), lambda s_: s_, s2)
        return s3, r, done, info(s2, desc)

    def step_auto_batch(self, S, actions, key, frame_skip=1):
        """many games, `frame_skip` frames each. Games that died or took the stairs get their next dungeon, generated only for
        them (one real lax.cond outside the vmap). Returns (states, reward, done, info)"""
        S2, r, done, desc = self._advance(S, actions, frame_skip)
        desc = desc & ~done
        keys = jax.random.split(key, done.shape[0])
        regen = lambda S2: regen_where(S2, done | desc, lambda row, i: self.next_state(row, keys[i], done[i], desc[i]))
        return lax.cond(jnp.any(done | desc), regen, lambda S2: S2, S2), r, done, info(S2, desc)


def info(s, desc):
    return dict(descended=desc, floor=s.dun.floor, kills=s.kills, hp=s.player.hp, t=s.t)
