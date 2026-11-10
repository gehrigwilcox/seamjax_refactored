"""Seam: a non-Euclidean roguelike twin-stick shooter, simulated and generated entirely in JAX."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")     # do not reserve ~75% of the GPU memory up front
from seamjax import backend; backend.apply_env()                   # choose CUDA / ROCm / Metal / CPU before JAX starts (see backend.py)
from seamjax.specs import load_specs
from seamjax.gen import gen_dungeon, gen_source
from seamjax.env import make_env
from seamjax.core.types import GameState, Dungeon
from seamjax.difficulty import difficulty, FLOOR_CAP
try:                                   # registers Seam-v0 / SeamVec-v0 when gymnasium is installed
    from seamjax.gym_env import SeamEnv, SeamVectorEnv
except ImportError:
    pass
