"""
The viewer's handle on the game on the device. Everything the viewer does with JAX goes through here; the rest of the app
works on numpy snapshots.

    S = Session(floor, seed)      # compiles on first use; S.new_run() starts a run
    S.step(action)                # one 1/60 s frame (several per drawn frame if the display lags)
    sn = S.read()                 # the one device read of a drawn frame -> Snapshot (numpy)
    S.descend()                   # the next floor after the stairs
    S.cheat('stairs')             # debug keys (cheats.py)
    S.room(sn), S.map()           # drawing tables of the current room / the minimap layout, cached
"""
import os, json, threading
import jax, jax.numpy as jnp
from seamjax.specs import load_specs
from seamjax.gen import gen_source
from seamjax.sim.game import Game
from seamjax.app import cheats
from seamjax.app.snapshot import Flags, no_flags, take, room_tables, RoomDraw, map_tables, map_layout

_META = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'specs.json')


class Session:
    def __init__(self, floor=1, seed=0):
        self.floor0 = floor
        self.G = G = Game(gen_source(load_specs()))
        self.meta = json.load(open(_META))                  # per world: name, category, corner turn
        self._reset = jax.jit(G.reset)
        self._next = jax.jit(G.next_state)
        self._take = jax.jit(take)
        self._room = jax.jit(room_tables)
        self._map = jax.jit(map_tables)
        self._cheat = {k: jax.jit(f) for k, f in cheats.KEYS.items()}

        @jax.jit
        def advance(s, action, flags):
            s, _, done, ev = G.step(s, action)
            return s, Flags(opened=flags.opened | (ev.opened > 0), died=flags.died | done, descended=flags.descended | ev.descended)
        self._advance = advance
        self.key = jax.random.PRNGKey(seed)
        self.gen = 0                                        # bumps whenever a new floor is generated
        self._room_cache = self._map_cache = None
        self.new_run()

    # ---------------------------------------------------------------- playing
    def new_run(self):
        self.key, k = jax.random.split(self.key); self.gen += 1
        self.s = self._reset(k, self.floor0); self.flags = no_flags()

    def step(self, action, god=False):
        if god: self.s = self._cheat['god'](self.s)
        self.s, self.flags = self._advance(self.s, jnp.asarray(action), self.flags)

    def read(self):
        """the Snapshot (numpy); the flags start over"""
        sn = jax.device_get(self._take(self.s, self.flags)); self.flags = no_flags(); return sn

    def descend(self):
        self.key, k = jax.random.split(self.key); self.gen += 1
        self.s = self._next(self.s, k, jnp.bool_(False), jnp.bool_(True))

    def cheat(self, name):
        self.s = self._cheat[name](self.s)

    def world(self, sn):
        """catalogue entry of the current world: name, cat (flat / hyper / sphere / grid), hol (corner turn in degrees)"""
        return self.meta[int(sn.spec)]

    # ---------------------------------------------------------------- tables for drawing (read once per room / per floor)
    def room(self, sn):
        sig = (self.gen, int(sn.room))
        if self._room_cache is None or self._room_cache[0] != sig:
            self._room_cache = (sig, RoomDraw.build(jax.device_get(self._room(self.s)), sig[1]))
        return self._room_cache[1]

    def map(self):
        if self._map_cache is None or self._map_cache['sig'] != self.gen:
            self._map_cache = map_layout(jax.device_get(self._map(self.s)), self.gen)
        return self._map_cache

    def warm_up(self):
        """compile, in the background, what the first descent and the debug keys need, while you already play"""
        s0 = self.s
        def run():
            try:
                jax.block_until_ready(self._next(s0, jax.random.PRNGKey(0), jnp.bool_(False), jnp.bool_(True)))
                for f in list(self._cheat.values()) + [self._map]: jax.block_until_ready(f(s0))
            except Exception: pass
        self._warm = threading.Thread(target=run, daemon=True); self._warm.start()

    def busy(self):
        return getattr(self, '_warm', None) is not None and self._warm.is_alive()

    def close(self):
        """wait for the background compile: exiting while XLA is compiling in another thread crashes the process"""
        if self.busy(): self._warm.join()
