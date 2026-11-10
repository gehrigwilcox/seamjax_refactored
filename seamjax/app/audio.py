"""
Generative sound for Seam.  Every sound is synthesised by JAX (on the GPU when there is one) from a few formulas: sweeps,
noise, FM bells, detuned oscillators and a one-pole filter.  Nothing is recorded.

  bank()          -> {name: float32 [variants, samples]} one-shot effects, each with a few random variants so repeats differ
  loops()         -> {name: float32 [samples]}  seamless 4 s drones, one per world type (flat / hyper / sphere / grid), and a heartbeat
  Audio           plays them through pygame.mixer: events (events.py) trigger effects, the world type picks the drone and the
                  Hunter's distance sets the heartbeat volume.  Synthesis runs in a background thread so the window opens at once.
"""
import threading
import numpy as np
import jax, jax.numpy as jnp
from jax import lax

SR = 22050; TAU = 2*np.pi; LOOP = 4.0; NVAR = 3


def _t(sec): return jnp.arange(int(sec*SR))/SR
def _phase(f): return TAU*jnp.cumsum(f)/SR                                # integrate a frequency curve (Hz per sample)
def _lp(x, a):                                                            # one-pole low-pass, a in (0,1] per sample (1 = open)
    a = jnp.broadcast_to(a, x.shape)
    return lax.scan(lambda y, xa: ((y := y+xa[1]*(xa[0]-y)), y), 0., (x, a))[1]
def _noise(key, n): return jax.random.normal(key, (n,))
def _norm(x, peak=.8): return x/(jnp.max(jnp.abs(x))+1e-6)*peak
def _fade(x, ms=6):                                                       # tiny fade in/out: no clicks
    k = int(ms*SR/1000); w = jnp.ones_like(x).at[:k].set(jnp.linspace(0, 1, k)).at[-k:].set(jnp.linspace(1, 0, k)); return x*w
def _jit(key, amt=.08): return 1+amt*jax.random.normal(key, ())          # random pitch variation per variant


def shoot(key):
    k1, k2 = jax.random.split(key); t = _t(.11); f = (1100*jnp.exp(-20*t)+220)*_jit(k1)
    return _norm(_fade(jnp.tanh(3*jnp.sin(_phase(f)))*jnp.exp(-28*t)+.3*_noise(k2, t.size)*jnp.exp(-220*t)), .5)

def kill(key, shooter):
    k1, k2 = jax.random.split(key); t = _t(.45); j = _jit(k1)
    if shooter:                                                           # bright FM bell: a purple enemy rings as it breaks
        fc = 520*j; bell = jnp.sin(_phase(jnp.full_like(t, fc))+3.*jnp.exp(-9*t)*jnp.sin(TAU*fc*2.3*t))*jnp.exp(-7*t)
        return _norm(_fade(bell+.5*_lp(_noise(k2, t.size), .5)*jnp.exp(-30*t)), .7)
    body = jnp.tanh(2*jnp.sin(_phase((260*jnp.exp(-9*t)+60)*j)))*jnp.exp(-9*t)        # orange: a duller thud with a crunch
    return _norm(_fade(body+.8*_lp(_noise(k2, t.size), .25)*jnp.exp(-14*t)), .75)

def hit(key):
    k1, k2 = jax.random.split(key); t = _t(.4)
    thump = jnp.sin(_phase((50+110*jnp.exp(-16*t))*_jit(k1, .04)))*jnp.exp(-7*t)
    return _norm(_fade(jnp.tanh(2.5*(thump+.6*_lp(_noise(k2, t.size), .35)*jnp.exp(-12*t)))), .85)

def heal(key):
    t = _t(.6); j = _jit(key, .02); out = 0.
    for i, f in enumerate((523.25, 659.25, 783.99, 1046.5)):               # a rising C major arpeggio
        d = t-.09*i; out += jnp.where(d >= 0, jnp.sin(TAU*f*j*d)*(1-jnp.exp(-d*400))*jnp.exp(-6*jnp.maximum(d, 0)), 0.)
    return _norm(_fade(out), .55)

def death(key):
    k1, k2 = jax.random.split(key); t = _t(1.5); f = 40+190*jnp.exp(-2.4*t); out = 0.
    for d in (.0, .013, -.011): out += 2*(jnp.mod(_phase(f*(1+d))/TAU, 1.)-.5)
    out = _lp(out/3, .02+.5*jnp.exp(-3*t))*(1-t/1.5)**1.5*(.75+.25*jnp.sin(TAU*7*t))
    return _norm(_fade(out+.15*_lp(_noise(k2, t.size), .1)*jnp.exp(-5*t), 20), .85)

def hunter_hit(key):
    t = _t(.28); out = 0.
    for f, a in ((340., 1.), (521., .7), (877., .5)): out += a*jnp.sin(TAU*f*_jit(key, .02)*t)*jnp.exp(-14*t)
    return _norm(_fade(jnp.tanh(1.5*out)), .6)

def hunter_die(key):
    k1, k2 = jax.random.split(key); t = _t(1.3); out = _lp(_noise(k1, t.size), .08+.5*jnp.exp(-9*t))*jnp.exp(-4*t)*2
    for f in (110., 165., 220., 277.):
        out += .5*jnp.sin(_phase(f*(1-.25*(1-jnp.exp(-2*t)))))*jnp.exp(-3.2*t)
    return _norm(_fade(out+.8*jnp.sin(_phase(55*jnp.exp(-1.5*t)+25))*jnp.exp(-3*t), 15), .85)

def wall_open(key):
    t = _t(.75); n = _lp(_noise(key, t.size), .03+.35*(t/.75)**2)*jnp.sin(jnp.pi*t/.75)**1.5*3
    return _norm(_fade(n+.5*jnp.sin(TAU*58*t)*jnp.sin(jnp.pi*t/.75)), .6)

def descend(key):
    t = _t(1.1); f = (110+770*jnp.exp(-3.2*t))*_jit(key, .05); tri = 2*jnp.abs(2*(jnp.mod(_phase(f)/TAU, 1.)-.5))-1
    out = (.7*tri+.45*jnp.sin(_phase(f*1.5)))*(1-t/1.1)**1.2*(.8+.2*jnp.sin(TAU*9*t))
    return _norm(_fade(_lp(out, .05+.6*jnp.exp(-2.5*t)), 12), .65)

ONE_SHOTS = dict(shoot=shoot, kill_shooter=lambda k: kill(k, True), kill_charger=lambda k: kill(k, False), hit=hit, heal=heal,
                 death=death, hunter_hit=hunter_hit, hunter_die=hunter_die, open=wall_open, descend=descend)


def bank(seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), NVAR)
    return {n: np.asarray(jax.jit(jax.vmap(f))(keys)) for n, f in ONE_SHOTS.items()}


# ---- loops: every frequency is a whole number of cycles per LOOP seconds, so the loop has no seam
def _q(f): return np.round(f*LOOP)/LOOP

def drone(parts, lfo=1, gate=0):
    t = _t(LOOP); out = 0.
    for f, a, ph in parts: out += a*jnp.sin(TAU*_q(f)*t+ph)
    out = out*(.75+.25*jnp.sin(TAU*lfo*t/LOOP))
    if gate: out = out*(.55+.45*jnp.maximum(0., jnp.sin(TAU*gate*t/LOOP))**2)
    return _norm(out, .5)

WORLD_DRONES = {                                                           # in the order of cat: flat, hyper, sphere, grid
    'flat': lambda: drone([(55, .5, 0), (82.5, .3, 1), (110, .15, 2), (165, .06, 0)], lfo=1),
    'hyper': lambda: drone([(41.25, .5, 0), (41.75, .5, 0), (62, .25, 1), (124.25, .1, 2)], lfo=2),          # beating pair: restless
    'sphere': lambda: drone([(196, .3, 0), (294, .25, 1), (392, .15, 2), (98, .35, 0)], lfo=2),               # airy and high
    'grid': lambda: drone([(73.5, .5, 0), (147, .25, 1), (220.5, .1, 2)], lfo=1, gate=4),                      # pulsing
}

def heartbeat():
    t = _t(1.0); lub = jnp.sin(TAU*58*t)*jnp.exp(-30*t); d = t-.27; dub = jnp.where(d >= 0, jnp.sin(TAU*48*d)*jnp.exp(-34*jnp.maximum(d, 0)), 0.)
    return _norm(lub+.8*dub, .9)

def loops():
    out = {k: np.asarray(jax.jit(f)()) for k, f in WORLD_DRONES.items()}; out['heartbeat'] = np.asarray(jax.jit(heartbeat)()); return out


class Audio:
    CATS = ['flat', 'hyper', 'sphere', 'grid']
    MAP = dict(shoot='shoot', heal='heal', hit='hit', death='death', hunter_hit='hunter_hit', hunter_die='hunter_die', open='open', descend='descend')

    def __init__(self, volume=.7):
        import pygame
        self.pg = pygame; self.ready = False; self.volume = volume; self.sounds = {}; self.cat = None; self.beat = None; self.err = None
        pygame.mixer.pre_init(SR, -16, 2, 512)
        try: pygame.mixer.init(frequency=SR, size=-16, channels=2, buffer=512, allowedchanges=0); pygame.mixer.set_num_channels(24); self.ok = True
        except Exception as ex: self.ok = False; self.err = str(ex); return
        self.ready_evt = threading.Event(); threading.Thread(target=self._build, daemon=True).start()

    def _snd(self, x):
        a = np.clip(x, -1, 1)*32000; return self.pg.sndarray.make_sound(np.ascontiguousarray(np.stack([a, a], 1).astype(np.int16)))

    def _build(self):
        try:
            B = bank(); L = loops(); self.sounds = {k: [self._snd(v) for v in vs] for k, vs in B.items()}; self.loops = {k: self._snd(v) for k, v in L.items()}
            self.beat = self.loops['heartbeat'].play(-1); self.beat.set_volume(0.); self.ready = True; self._world()
        except Exception as ex: self.err = repr(ex)

    def _world(self):
        if self.ready and self.cat is not None:
            if getattr(self, 'ch', None): self.ch.fadeout(500)
            self.ch = self.loops[self.cat].play(-1, fade_ms=800); self.ch.set_volume(.32*self.volume)

    def set_world(self, cat):
        if cat != self.cat: self.cat = cat; self._world()

    def set_hunter(self, hdist):
        """heartbeat gets louder as the Hunter gets closer (hdist = doors away; <0 means none)"""
        if self.ready and self.beat is not None: self.beat.set_volume(self.volume*(0. if hdist < 0 else max(0., 1-hdist/6.)**1.6*.8))

    def play(self, name, pan=0., vol=1., rng=np.random):
        if not (self.ok and self.ready): return
        s = self.sounds[name][rng.randint(NVAR)]; ch = s.play()
        if ch: l, r = (1-max(0., pan)*.7, 1+min(0., pan)*.7); ch.set_volume(vol*self.volume*l, vol*self.volume*r)

    def events(self, evs, pan_of):
        """evs from events.Events; pan_of(x, y) -> -1..1 (left..right on the screen)"""
        for e in evs:
            n = e['name']; name = ('kill_shooter' if e['shooter'] else 'kill_charger') if n == 'kill' else self.MAP.get(n)
            if name: self.play(name, pan_of(e['x'], e['y']) if n not in ('hit', 'death', 'descend', 'open') else 0., .6 if n == 'shoot' else 1.)
