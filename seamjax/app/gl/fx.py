"""Game events (events.py) -> particles, screen shake and flashes, drawn by the OpenGL view; what each event does is in seamjax/theme.py."""
import math
import numpy as np
from seamjax.app.gl.shaders import NPART
from seamjax import theme as themes


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

    def apply(self, events, M, now, theme=None):
        """turn events (see events.py) into particles, shake and flashes, as the theme (seamjax/theme.py) says;
        M maps a world point to the screen"""
        theme = theme or themes.THEME
        for e in events:
            fx = theme.effect(e); p = np.array(M(e['x'], e['y']))
            for b in fx.bursts: self.burst(p, b.count, b.colour, b.speed, b.life, b.size, now, b.spread)
            self.shake = max(self.shake, fx.shake); self.hit = max(self.hit, fx.hit); self.flash = max(self.flash, fx.flash); self.fade = max(self.fade, fx.fade)

    def decay(self, dt):
        self.shake *= math.exp(-dt*7.); self.hit *= math.exp(-dt*5.); self.flash *= math.exp(-dt*6.); self.fade = max(0., self.fade-dt*1.8)
