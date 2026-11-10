"""The RL reward of a frame, from what happened in it (StepEvents) and whether you died."""
import jax.numpy as jnp
from seamjax.core.config import R_KILL, R_HIT, R_ROOM, R_DESCEND, R_DEATH, R_HEAL, R_GATE


def done(s):
    """the run ends only on death; the stairs are reported as ev.descended"""
    return s.player.hp <= 0

def reward(s, ev):
    return (R_KILL*ev.killed + R_HIT*ev.took + R_ROOM*ev.newroom
            + R_HEAL*ev.healed + R_GATE*ev.opened + R_DESCEND*ev.descended + jnp.where(done(s), R_DEATH, 0.))
