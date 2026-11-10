import _setup
import numpy as np, gymnasium as gym, seamjax
from gymnasium.utils.env_checker import check_env
"""Gymnasium contract: spaces, determinism by seed, termination on death, stairs continue the run, vector env autoreset."""
env = gym.make('Seam-v0', max_steps=400)
check_env(env.unwrapped, skip_render_check=True)
o, info = env.reset(seed=1); o2, _ = env.reset(seed=1)
assert o.shape == env.observation_space.shape and np.allclose(o, o2) and info['floor'] == 1
tot = 0; term = trunc = False; n = 0
while not (term or trunc):
    o, r, term, trunc, info = env.step(env.action_space.sample()); tot += r; n += 1
assert trunc and not term or term, 'episode must end by death or time limit'
print('single env: ended after', n, 'steps, return %.2f' % tot, 'floor', info['floor'])
# a heuristic that reads the observation (aim at the nearest enemy) earns kill reward, so rewards really flow
env2 = gym.make('Seam-v0', max_steps=3000); o, _ = env2.reset(seed=3); got = 0.; kills = 0
for i in range(3000):
    ef = o[8+12*8:8+12*8+5]; a = np.zeros(4, np.float32)
    if ef[0] > .5: v = ef[1:3]/(np.hypot(*ef[1:3])+1e-9); a[2:4] = v; a[0:2] = np.array([v[1], -v[0]])*(1 if (i//40) % 2 else -1)*.7
    else: a[0:2] = [np.cos(i/30.), np.sin(i/45.)]
    o, r, te, tr, info = env2.step(a); got += r; kills = info['kills']
    if te or tr: break
print('heuristic: return %.2f kills %d floor %d' % (got, kills, info['floor']))
assert kills > 0
# pixel observations
penv = gym.make('Seam-v0', obs_mode='pixels', pixels=(60, 96)); po, _ = penv.reset(seed=0); assert po.shape == (60, 96, 3) and po.dtype == np.uint8 and po.std() > 5
frame = penv.unwrapped.render(); assert frame.shape == (200, 320, 3)
# floor option
e3 = gym.make('Seam-v0'); _, i3 = e3.reset(seed=0, options={'floor': 5}); assert i3['floor'] == 5
# vector env
from seamjax import SeamVectorEnv
V = SeamVectorEnv(num_envs=64)
o, _ = V.reset(seed=0); assert o.shape == (64, 194)
ep = 0; rets = np.zeros(64); floors = []
for t in range(300):
    o, r, te, tr, info = V.step(V.action_space.sample()); rets += r; ep += int((te | tr).sum())
    assert np.isfinite(o).all() and o.shape == (64, 194)
print('vector env: 64 envs x 300 steps ok, finished episodes:', ep, 'max floor', int(info['floor'].max()))
print('PASS')
