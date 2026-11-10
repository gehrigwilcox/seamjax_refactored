"""
Gymnasium interfaces for Seam. Dungeons are generated on the device by seamjax/gen; nothing comes from files or JS.

    import gymnasium as gym, seamjax                 # registers "Seam-v0"
    env = gym.make("Seam-v0")                        # one game (jit-compiled, runs on CPU or GPU)
    obs, info = env.reset(seed=0)
    obs, r, terminated, truncated, info = env.step(env.action_space.sample())

    from seamjax import SeamVectorEnv
    venv = SeamVectorEnv(num_envs=2048)              # thousands of games in one jitted call
    obs, info = venv.reset(seed=0)
    obs, r, term, trunc, info = venv.step(actions)   # actions: (num_envs, 4)

Action (Box, -1..1, shape 4): [move_x, move_y, aim_x, aim_y] in SCREEN space (what you see, never flipped).
The gun fires while the aim stick is held (|aim| > .25). One env step = `frame_skip` frames of 1/60 s.
Observation (default 'vector', 194 floats, egocentric, no map) or 'pixels' (H, W, 3 uint8 from render.py).
Reward: kills +.1, hit -.2, new room +.05, health +.05, gate opened +.05, stairs +5, death -5.
Episode: terminated on death; truncated after `max_steps` env steps. Taking the stairs does NOT end the episode:
the run goes on one floor deeper (harder, see difficulty.py); `info['floor']` tells you where you are.
"""
import numpy as np, jax, jax.numpy as jnp
from jax import lax
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import VectorEnv, AutoresetMode
from seamjax.core import struct
from seamjax.sim.game import Game, info as step_info
from seamjax.sim.observe import observe, OBS_DIM
from seamjax.rl.render import render

_CACHE = {}
def _core(frame_skip):
    """one shared set of jitted functions per frame_skip (compiling takes a while, do it once)"""
    if frame_skip in _CACHE: return _CACHE[frame_skip]
    G = Game.generated()

    def one(s, a, key):
        """frame_skip frames; the stairs continue the run, death ends the episode (reset() starts the next)"""
        s, rew, done, desc = G.step_frames(s, a, frame_skip)
        desc = desc & ~done
        info = step_info(s, desc)
        s = lax.cond(desc, lambda s_: G.next_state(s_, key, jnp.bool_(False), jnp.bool_(True)), lambda s_: s_, s)
        return s, rew, done, info

    C = dict(reset=jax.jit(G.reset), one=jax.jit(one), obs=jax.jit(observe), batch=jax.jit(lambda S, A, k: G.step_auto_batch(S, A, k, frame_skip)),
             reset_b=jax.jit(G.reset_batch), obs_b=jax.jit(jax.vmap(observe)),
             render=jax.jit(render, static_argnums=(1, 2)), render_b=jax.jit(jax.vmap(render, in_axes=(0, None, None)), static_argnums=(1, 2)))
    _CACHE[frame_skip] = C; return C

def _obs_space(mode, pixels):
    if mode == 'pixels': return spaces.Box(0, 255, (pixels[0], pixels[1], 3), np.uint8)
    return None

class SeamEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 15}
    def __init__(self, frame_skip=4, max_steps=5000, start_floor=1, obs_mode='vector', pixels=(100, 160), render_mode=None):
        self.C = _core(frame_skip); self.max_steps = max_steps; self.start_floor = start_floor; self.obs_mode = obs_mode
        self.pixels = pixels; self.render_mode = render_mode
        self.action_space = spaces.Box(-1., 1., (4,), np.float32)
        self.observation_space = _obs_space(obs_mode, pixels) or spaces.Box(-10., 10., (OBS_DIM,), np.float32)
        self._key = jax.random.PRNGKey(0); self.s = None; self.n = 0

    def _obs(self):
        if self.obs_mode == 'pixels': return (np.array(self.C['render'](self.s, *self.pixels))*255).astype(np.uint8)
        return np.array(self.C['obs'](self.s), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None or self.s is None: self._key = jax.random.PRNGKey(0 if seed is None else seed)
        self._key, k = jax.random.split(self._key)
        self.s = self.C['reset'](k, (options or {}).get('floor', self.start_floor)); self.n = 0
        return self._obs(), dict(floor=int(self.s.dun.floor))

    def step(self, action):
        self._key, k = jax.random.split(self._key)
        self.s, r, done, info = self.C['one'](self.s, jnp.asarray(action, jnp.float32), k); self.n += 1
        return self._obs(), float(r), bool(done), self.n >= self.max_steps, {k_: v.item() for k_, v in info.items()}

    def render(self):
        return (np.array(self.C['render'](self.s, 200, 320))*255).astype(np.uint8)

class SeamVectorEnv(VectorEnv):
    """thousands of games stepped in one jitted call (put JAX on a GPU and this is the fast path).
    Autoreset is SAME_STEP: when a game ends, the returned obs is already the first obs of the next run and
    the last one of the finished run is in info['final_obs'] (valid where info['_final_obs'])."""
    metadata = {'autoreset_mode': AutoresetMode.SAME_STEP, 'render_modes': []}
    def __init__(self, num_envs=256, frame_skip=4, max_steps=5000, start_floor=1):
        self.C = _core(frame_skip); self.num_envs = num_envs; self.max_steps = max_steps; self.start_floor = start_floor
        od = OBS_DIM
        self.single_action_space = spaces.Box(-1., 1., (4,), np.float32); self.single_observation_space = spaces.Box(-10., 10., (od,), np.float32)
        self.action_space = spaces.Box(-1., 1., (num_envs, 4), np.float32); self.observation_space = spaces.Box(-10., 10., (num_envs, od), np.float32)
        self._key = jax.random.PRNGKey(0); self.S = None; self.n = np.zeros(num_envs, np.int32)

    def reset(self, *, seed=None, options=None):
        if seed is not None or self.S is None: self._key = jax.random.PRNGKey(0 if seed is None else seed)
        self._key, k = jax.random.split(self._key)
        self.S = self.C['reset_b'](jax.random.split(k, self.num_envs), jnp.full(self.num_envs, self.start_floor)); self.n[:] = 0
        return np.array(self.C['obs_b'](self.S)), {}

    def step(self, actions):
        self._key, k = jax.random.split(self._key)
        S, r, done, info = self.C['batch'](self.S, jnp.asarray(actions, jnp.float32), k)
        self.n += 1; trunc = self.n >= self.max_steps
        done = np.array(done); info = {k_: np.array(v) for k_, v in info.items()}
        over = done | trunc
        obs = np.array(self.C['obs_b'](S))
        if trunc.any():       # time limit: start a new run for those too
            fresh = self.C['reset_b'](jax.random.split(k, self.num_envs), jnp.full(self.num_envs, self.start_floor))
            S = struct.where(jnp.asarray(trunc & ~done), fresh, S)
        if over.any():
            info['final_obs'] = obs.copy(); info['_final_obs'] = over     # observation before the reset (valid where over)
        self.S = S; self.n[over] = 0
        obs_after = np.array(self.C['obs_b'](S)) if over.any() else obs
        return obs_after, np.array(r, np.float32), done, trunc & ~done, info

gym.register('Seam-v0', entry_point='seamjax.rl.gym_env:SeamEnv')
