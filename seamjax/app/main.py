"""
The viewer: a window, the game loop, input, drawing and sound. Run it with `uv run seam` or `python -m seamjax.play`
(see play.py for the controls and flags).

The simulation runs on the JAX device (session.py). To keep the frame rate smooth the viewer makes ONE device read per drawn
frame (a Snapshot) and does all drawing maths in numpy. The room is drawn by OpenGL (gl/), or by pygame on the CPU
(software.py) with --software or when OpenGL cannot start; the HUD is the same code on both (hud.py, via a canvas).
"""
import os, sys, argparse, time
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')       # don't grab most of the GPU memory for a game
import numpy as np
from seamjax.core.config import VIEW_W as VW, VIEW_H as VH, CX, CY

DT = 1/60.


def parse(argv):
    ap = argparse.ArgumentParser(prog='seam', description='Seam: a non-Euclidean roguelike twin-stick shooter')
    ap.add_argument('--floor', type=int, default=1); ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--bot', action='store_true', help='let a simple scripted bot play (demo / smoke test)')
    ap.add_argument('--debug', action='store_true', help='debug overlay (F1) and debug keys (N H G K O T M C B)')
    ap.add_argument('--debug-stairs', action='store_true', help='teleport onto the stairs at frame 30 (tests the descent path)')
    ap.add_argument('--frames', type=int, default=0, help='quit after this many frames (smoke test)')
    ap.add_argument('--shot', default='', help='save a screenshot to this path when quitting')
    ap.add_argument('--map-follow', action='store_true', help='debug minimap: centre on your room instead of showing the whole dungeon fixed')
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'rocm', 'metal'], default=None, help='JAX backend (default: automatic; see backend.py)')
    ap.add_argument('--mute', action='store_true', help='no sound')
    ap.add_argument('--software', action='store_true', help='use the pygame CPU renderer instead of OpenGL')
    ap.add_argument('--timing', action='store_true', help='print where the frame time goes')
    return ap.parse_args(argv)


class Display:
    """the window: OpenGL (gl, fx, canvas) if it can start, else a pygame surface (screen, canvas)"""
    def __init__(self, software):
        import pygame
        self.pg = pygame; pygame.init()
        self.gl = self.fx = self.screen = None; headless = os.environ.get('SDL_VIDEODRIVER') == 'dummy'
        if not software and not headless:
            try:
                pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3); pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
                pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
                pygame.display.set_mode((int(VW), int(VH)), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE, vsync=1)
                from seamjax.app.gl.view import GLView
                from seamjax.app.gl.canvas import GLCanvas
                from seamjax.app.gl.fx import Fx
                self.gl = GLView(VW, VH); self.fx = Fx(); self.canvas = GLCanvas(self.gl)
            except Exception as ex:
                print('OpenGL view unavailable (%s: %s); using the software renderer. Use --software to silence this.' % (type(ex).__name__, ex))
                self.gl = None; pygame.display.quit(); pygame.display.init()
        if not self.gl:
            from seamjax.app.canvas import PygameCanvas
            self.screen = pygame.display.set_mode((int(VW), int(VH)), pygame.RESIZABLE | pygame.SCALED if not headless else 0)
            self.canvas = PygameCanvas(self.screen)
        pygame.display.set_caption('Seam')

    def message(self, msg, title=True):
        from seamjax.app import hud
        hud.loading(self.canvas, msg, title); self.pg.display.flip(); self.pg.event.pump()

    def to_logical(self, pos):
        return self.gl.to_logical(pos) if self.gl else pos

    def screenshot(self, path):
        self.pg.image.save(self.gl.screenshot() if self.gl else self.screen, path)


def start_jax(display):
    """start the JAX backend with the on-disk compilation cache; a problem is reported with advice, not a traceback"""
    import jax
    try:
        jax.config.update('jax_compilation_cache_dir', os.path.join(os.path.expanduser('~'), '.cache', 'seamjax-jit'))
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.)
    except Exception: pass
    from seamjax.backend import ensure, describe, BackendError
    try: info = ensure()
    except BackendError as ex: display.pg.quit(); sys.exit('seam: '+str(ex))
    print('JAX backend: %s%s' % (describe(), ('  (%s)' % info['note']) if info['note'] else ''))
    return info, describe()


class App:
    def __init__(self, a):
        self.a = a
        self.d = Display(a.software); self.d.message('starting JAX...')
        info, backend = start_jax(self.d)
        from seamjax.app import hud, input as controls
        from seamjax.app.events import Events
        from seamjax.app.session import Session
        self.hud, self.events = hud, Events()
        self.audio = None
        if not a.mute:                                                 # generative sound: JAX synthesises it in a background thread
            from seamjax.app.audio import Audio
            self.audio = Audio()
            if not self.audio.ok: print('no sound device (%s); continuing without sound. --mute silences this.' % self.audio.err); self.audio = None
        self.d.message('loading the tiling tables and compiling the game on %s (the first launch takes a while)...' % info['platform'])
        self.S = Session(a.floor, a.seed if a.seed is not None else int(time.time()) % 100000)
        self.controls = controls.Controls(self.d.to_logical); self.bot = controls.bot
        self.dbg = hud.Debug(on=a.debug, map_mode=1 if a.debug else 0, map_follow=a.map_follow, seed=a.seed, backend=backend)
        self.toast = hud.Toast(); self.toast.show(['Floor %d' % a.floor], 2.5)
        self.dead = False; self.sn = self.S.read()
        self.S.warm_up()

    # ---------------------------------------------------------------- input
    def on_key(self, key):
        pg = self.d.pg; S = self.S; dbg = self.dbg
        if key == pg.K_F1: dbg.on = not dbg.on
        elif key == pg.K_r: S.new_run(); self.dead = False; self.toast.show(['Floor %d' % self.a.floor], 2.5); self.sn = S.read()
        elif dbg.on:
            cheat = {pg.K_n: ('stairs', 'jumped to the stairs'), pg.K_h: ('heal', 'healed'), pg.K_k: ('kill', 'enemies cleared'),
                     pg.K_o: ('open', 'gate opened (if this room has one)'), pg.K_t: ('hunter', 'Hunter toggled')}.get(key)
            if cheat: S.cheat(cheat[0]); self.toast.show(['debug: '+cheat[1]], 1.5)
            elif key == pg.K_g: dbg.god = not dbg.god; self.toast.show(['debug: god mode ' + ('on' if dbg.god else 'off')], 1.5)
            elif key == pg.K_m: dbg.map_mode = (dbg.map_mode+1) % 3
            elif key == pg.K_c: dbg.map_follow = not dbg.map_follow
            elif key == pg.K_b:
                w = S.world(self.sn); print('floor %d  world %s [%s]  corner turn %.0f deg  room %d of %d' % (int(self.sn.floor), w['name'], w['cat'], w['hol'], int(self.sn.room), int(self.sn.n_rooms)))

    # ---------------------------------------------------------------- one drawn frame
    def simulate(self, frame):
        """up to 2 simulation steps per drawn frame, so a slow frame cannot spiral"""
        steps = 0
        while self.accum >= DT and not self.dead and steps < 2:
            self.accum -= DT; self.t += DT; self.toast.tick(DT); steps += 1
            if self.a.debug_stairs and frame == 30 and steps == 1: self.S.cheat('stairs')
            self.S.step(self.bot(self.sn) if self.a.bot else self.controls.action(self.sn), self.dbg.god)
        if self.accum > .2: self.accum = 0.

    def after_read(self):
        """react to what happened: toasts, death, the stairs. Returns the events for effects and sound"""
        sn = self.sn; fl = sn.flags; S = self.S
        if fl.opened: self.toast.show(['A wall slides open.'], 2.5)
        if fl.died: self.dead = True
        elif fl.descended:
            self.d.message('descending...', False)
            self.toast.show(self.hud.floor_summary(S.world(sn), int(sn.floor)), 5.)
            S.descend(); self.accum = 0.; self.sn = sn = S.read()
        if self.dead: self.accum = 0.
        return self.events.update(sn, (S.gen, int(sn.room)), S.gen, self.dead, bool(fl.opened))

    def draw(self, evs):
        from seamjax.core.geometry.view import view_np
        sn = self.sn; S = self.S; d = self.d; room = S.room(sn); va = float(sn.view_angle); mir = float(sn.mirror); world = S.world(sn)
        M = lambda x, y: tuple(np.array(view_np(va, mir, x-CX, y-CY))+(CX, CY))            # room frame -> screen
        if self.audio:
            self.audio.events(evs, lambda x, y: float(np.clip(view_np(va, mir, x-CX, y-CY)[0]/CX, -1, 1)))
            self.audio.set_world(world['cat']); self.audio.set_hunter(-1 if self.dead else int(sn.hunter_dist))
        if d.gl:
            now = time.time()-self.t_start; d.fx.apply(evs, M, now); d.fx.decay(now-self.fx_t); self.fx_t = now
            from seamjax import theme
            d.gl.scene(sn, room, M, va, mir, self.t, theme.THEME.worlds[world['cat']], d.fx, now)
        else:
            from seamjax.app.software import draw_room
            draw_room(d.screen, sn, room, M, self.t)
        d.canvas.begin(); self.hud.draw(d.canvas, sn, room, M, world, self.toast, self.dead, self.dbg, S.map, self.a.bot); d.canvas.end()

    def run(self):
        pg = self.d.pg; a = self.a; clock = pg.time.Clock()
        self.accum = 0.; self.t = 0.; self.t_start = time.time(); self.fx_t = 0.
        frames = 0; running = True; last = time.time(); tm = dict(sim=0., read=0., draw=0.)
        while running:
            for ev in pg.event.get():
                if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key == pg.K_ESCAPE): running = False
                elif ev.type == pg.KEYDOWN: self.on_key(ev.key)
            now = time.time(); self.accum += min(.05, now-last); last = now
            t0 = time.time(); self.simulate(frames)
            t1 = time.time(); self.sn = self.S.read(); t2 = time.time()                        # the one device read of this frame
            evs = self.after_read()
            t3 = time.time(); self.draw(evs); pg.display.flip(); t4 = time.time(); clock.tick(60); frames += 1
            tm['sim'] += t1-t0; tm['read'] += t2-t1; tm['draw'] += t4-t3
            ms = self.dbg.ms; self.dbg.ms = [.9*ms[0]+.1*1000*(t1-t0), .9*ms[1]+.1*1000*(t2-t1), .9*ms[2]+.1*1000*(t4-t3)]
            self.dbg.fps = .95*self.dbg.fps+.05*clock.get_fps()
            if a.timing and frames % 120 == 0:
                print('ms/frame  sim %.2f  device read %.2f  draw+flip %.2f' % tuple(1000*tm[k]/120 for k in ('sim', 'read', 'draw'))); tm = dict(sim=0., read=0., draw=0.)
            if a.frames and frames >= a.frames: running = False
        if a.shot: self.d.screenshot(a.shot)
        if self.S.busy(): self.d.message('finishing the background compile...', False)
        self.S.close(); pg.quit()


def main(argv=None):
    a = parse(argv)
    if a.device:
        os.environ['SEAM_BACKEND'] = a.device
        from seamjax import backend; backend.apply_env()                  # the flag wins over the environment (JAX has not started yet)
    App(a).run()
