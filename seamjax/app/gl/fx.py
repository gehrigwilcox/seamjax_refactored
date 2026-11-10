"""Game events (events.py) -> particles, screen shake and flashes, drawn by the OpenGL view."""
import math
import numpy as np
from seamjax.app.gl.shaders import NPART
from seamjax.core import kinds


class Fx:
    """Game events -> particles, shake and flashes."""
    def __init__(self):
        self.part = np.zeros((NPART, 11), 'f4'); self.ring = 0; self.rng = np.random.default_rng(1)
        self.shake = 0.; self.hit = 0.; self.flash = 0.; self.fade = 0.

    def burst(self, p, n, col, speed, life, size, now, spread=1.):
        a = self.rng.uniform(0, 2*math.pi, n); sp = speed*self.rng.uniform(.25, 1., n)**.7
        rows = np.zeros((n, 11), 'f4'); rows[:, 0:2] = p; rows[:, 2] = np.cos(a)*sp*spread; rows[:, 3] = np.sin(a)*sp*spread
        rows[:, 4] = now; rows[:, 5] = life*self.rng.uniform(.6, 1.2, n); rows[:, 6:9] = col; rows[:, 9] = 1.; rows[:, 10] = size*self.rng.uniform(.6, 1.3, n)
        i = (self.ring+np.arange(n)) % NPART; self.part[i] = rows; self.ring = (self.ring+n) % NPART

    def apply(self, events, M, now):
        """turn events (see events.py) into particles, shake and flashes; M maps a world point to the screen"""
        for e in events:
            n, p = e['name'], np.array(M(e['x'], e['y']))
            if n == 'hit': self.shake = max(self.shake, .9); self.hit = 1.; self.burst(p, 18, (1., .35, .4), 190, .55, 6, now)
            elif n == 'heal': self.burst(p, 14, (1., .45, .6), 120, .7, 5, now)
            elif n == 'death': self.shake = 1.4; self.burst(p, 70, (1., .7, .75), 330, 1.1, 8, now)
            elif n == 'kill':
                self.burst(p, 26, tuple(c/255. for c in kinds.ENEMY_KINDS[e['kind']].colour), 250, .65, 6, now); self.burst(p, 8, (1., 1., 1.), 160, .35, 4, now); self.shake = max(self.shake, .28)
            elif n == 'hunter_hit': self.burst(p, 8, (1., .36, .45), 170, .4, 4, now)
            elif n == 'hunter_die': self.burst(p, 60, (1., .36, .45), 300, 1., 7, now); self.shake = max(self.shake, .8); self.flash = .6
            elif n == 'shoot': self.burst(p, 4, (1., .89, .48), 130, .22, 3, now, .6)
            elif n in ('descend', 'newrun'): self.fade = 1.

    def decay(self, dt):
        self.shake *= math.exp(-dt*7.); self.hit *= math.exp(-dt*5.); self.flash *= math.exp(-dt*6.); self.fade = max(0., self.fade-dt*1.8)
