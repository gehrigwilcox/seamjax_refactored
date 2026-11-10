"""
Play Seam on the JAX simulation (the same one the RL environments use).

    python -m seamjax.play [--floor N] [--seed S] [--bot]

Move: WASD or arrow keys   Aim + fire: mouse (hold left button), or I/J/K/L, or a gamepad   R: new run   Esc: quit.
Controls are always in screen space: up is up, whatever the room does to the view.
Pink wall = solid (never opens), blue wall = gated (opens when the room is clear). Orange doors are on the second
side of a wall. A red flashing door is where the Hunter will come through.

Debug mode (python -m seamjax.play --debug): F1 toggles the overlay (floor, world type, room, section, Hunter, enemies, timings,
door labels "room id / side of wall", spawn points) and these keys work:
  N  jump to the stairs of this floor (clears the exit room first)      H  heal to full            G  toggle god mode
  K  kill every enemy in the room       O  open this room's gated wall      T  remove / restore the Hunter
  M  minimap: off / explored / whole dungeon (turned to match your view; room numbers match the door labels)
  C  minimap follows you (centred on your room) or stays fixed        (start that way with --map-follow)
  B  print this floor's tiling name and corner-turn angle to the console
Sound is synthesised by JAX (audio.py); --mute turns it off.
Other flags: --timing (frame-time breakdown in the console), --floor N, --seed S, --bot.

Drawing: the room is drawn by the GPU through OpenGL (glview.py); --software uses the old pygame CPU renderer, and is also the automatic fallback.
Performance note: the simulation runs on the JAX device. To keep the frame rate
smooth the viewer makes exactly ONE device read per frame (a compact snapshot) and does all drawing maths in numpy.
"""
import os, sys, math, argparse, time, json
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')       # don't grab most of the GPU memory for a game
import numpy as np

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--floor', type=int, default=1); ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--bot', action='store_true', help='let a simple scripted bot play (demo / smoke test)')
    ap.add_argument('--debug', action='store_true', help='debug overlay (F1) and debug keys (N H G K O T B)')
    ap.add_argument('--debug-stairs', action='store_true', help='teleport onto the stairs at frame 30 (tests the descent path)')
    ap.add_argument('--frames', type=int, default=0, help='quit after this many frames (smoke test)')
    ap.add_argument('--shot', default='', help='save a screenshot to this path when quitting')
    ap.add_argument('--map-follow', action='store_true', help='debug minimap: centre on your room instead of showing the whole dungeon fixed')
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'rocm', 'metal'], default=None, help='JAX backend (default: automatic; see backend.py)')
    ap.add_argument('--mute', action='store_true', help='no sound')
    ap.add_argument('--software', action='store_true', help='use the old pygame CPU renderer instead of OpenGL')
    ap.add_argument('--timing', action='store_true', help='print where the frame time goes')
    a = ap.parse_args(argv)
    if a.device:
        os.environ['SEAM_BACKEND'] = a.device
        from seamjax import backend; backend.apply_env()                  # the flag wins over the environment (JAX has not started yet)
    import pygame
    pygame.init(); VW, VH = 960., 600.
    gl = fx = cv = None; headless = os.environ.get('SDL_VIDEODRIVER') == 'dummy'
    if not a.software and not headless:                            # OpenGL view: the room is drawn by the GPU (see glview.py)
        try:
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3); pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
            pygame.display.set_mode((int(VW), int(VH)), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE, vsync=1)
            from seamjax.glview import GLView, GLCanvas, Fx
            gl = GLView(VW, VH); fx = Fx(); cv = GLCanvas(gl)
        except Exception as ex:
            print('OpenGL view unavailable (%s: %s); using the software renderer. Use --software to silence this.' % (type(ex).__name__, ex))
            gl = None; pygame.display.quit(); pygame.display.init()
    screen = None
    if not gl:
        from seamjax.canvas import PygameCanvas
        screen = pygame.display.set_mode((int(VW), int(VH)), pygame.RESIZABLE | pygame.SCALED if not headless else 0); cv = PygameCanvas(screen)
    pygame.display.set_caption('Seam'); clock = pygame.time.Clock(); FONT, BIG, SMALL = 22, 56, 19            # text sizes
    def say(txt, y=VH/2, size=FONT, col=(210, 235, 230)): cv.text(txt, VW/2, y, size, col, 'c')
    def loading(msg, title=True):                                  # the window is up before anything slow starts
        cv.begin((8, 10, 14))
        if title: say('SEAM', VH/2-60, BIG, (150, 235, 255))
        say(msg, VH/2+10); cv.end(); pygame.display.flip(); pygame.event.pump()
    loading('starting JAX...')
    import jax, jax.numpy as jnp
    try:                                                           # remember compiled programs between launches (next start is much faster)
        jax.config.update('jax_compilation_cache_dir', os.path.join(os.path.expanduser('~'), '.cache', 'seamjax-jit'))
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.)
    except Exception: pass
    loading('loading the tiling tables...')
    from seamjax.backend import ensure, describe, BackendError
    try: info = ensure()                                           # start the JAX backend now; a problem is reported with advice, not a traceback
    except BackendError as ex: pygame.quit(); sys.exit('seam: '+str(ex))
    print('JAX backend: %s%s' % (describe(), ('  (%s)' % info['note']) if info['note'] else ''))
    from seamjax.specs import load_specs
    from seamjax.gen import gen_source
    from seamjax.env import make_env
    from seamjax.core.config import CX, CY, DOOR_GAP as GAP, PLAYER_R as PR, MAX_HP as MAXHP, HUNTER_R as H_R, HUNTER_HP, CHASER_R, SHOOTER_R
    from seamjax.core.types import OBST_NONE, OBST_CIRCLE, WALL_GATED
    from seamjax.events import Events
    events = Events(); audio = None
    if not a.mute:                                                 # generative sound: JAX synthesises it in a background thread
        from seamjax.audio import Audio
        audio = Audio()
        if not audio.ok: print('no sound device (%s); continuing without sound. --mute silences this.' % audio.err); audio = None
    SP = load_specs(); meta = json.load(open(os.path.join(os.path.dirname(__file__), 'data', 'specs.json')))
    E = make_env(gen_source(SP))
    reset = jax.jit(E['reset']); nxt = jax.jit(E['next_state'])
    hdoor_f, hdist_f = E['hunter_door'], E['hunter_dist']

    @jax.jit
    def advance(s, act, acc):
        """one 1/60 s step; acc = flags seen since the last snapshot (opened, died, stairs)"""
        s, rew, done = E['step'](s, act)
        return s, acc | jnp.stack([s.ev.opened > 0, done, s.ev.descended])

    @jax.jit
    def snapshot(s, acc):
        r = s.room; d = s.dun; P = s.player; H = s.hunter
        return dict(room=r, floor=d.floor, spec=d.spec, exit=d.exit, kills=s.kills, hp=P.hp, x=P.x, y=P.y, va=s.view_angle, mir=s.mirror,
                    aim=P.aim, inv=P.inv, en=s.enemies[r], pk=s.pickups[r], pb=s.pb, eb=s.eb, hx=H.x, hy=H.y, hhere=H.here, hroom=H.room,
                    hsec=H.sec, hdead=H.dead, hflash=H.flash, hhp=H.hp, wopen=s.wall_open[r], hdoor=hdoor_f(s), hdist=hdist_f(s), acc=acc,
                    spawned=s.spawned[r], n_rooms=d.n_rooms, start=d.start, visited=s.visited)

    @jax.jit
    def tables(s):
        d = s.dun; r = s.room
        return dict(room=d.room(r), spawn=d.spawn[r], spawn_ok=d.spawn_ok[r], spawn_sec=d.spawn_sec[r])

    @jax.jit
    def layout(s):
        d = s.dun; return dict(mpos=d.map_pos, dseam=d.door_seam, doors=d.doors, n=d.n, n_rooms=d.n_rooms, geo=d.category, start=d.start, exit=d.exit)

    # ---- debug helpers (device side)
    @jax.jit
    def dbg_stairs(s):
        d = s.dun; ex = d.exit; pd = d.pad[ex]
        return s.replace(room=ex, player=s.player.replace(x=pd[0], y=pd[1], inv=jnp.float32(9.)), spawned=jnp.ones_like(s.spawned),
                         enemies=s.enemies.replace(alive=s.enemies.alive.at[ex].set(False)))
    @jax.jit
    def dbg_heal(s): return s.replace(player=s.player.replace(hp=jnp.int32(MAXHP)))
    @jax.jit
    def dbg_kill(s): return s.replace(enemies=s.enemies.replace(alive=s.enemies.alive.at[s.room].set(False)))
    @jax.jit
    def dbg_open(s): return s.replace(wall_open=s.wall_open.at[s.room].set(s.dun.wall_kind[s.room] == WALL_GATED), spawned=s.spawned.at[s.room].set(True))
    @jax.jit
    def dbg_hunter(s): return s.replace(hunter=s.hunter.replace(dead=jnp.where(s.hunter.dead > 1e8, 0., 1e9).astype(jnp.float32)))
    @jax.jit
    def god(s): return s.replace(player=s.player.replace(hp=jnp.int32(MAXHP), inv=jnp.float32(9.)))
    dbg = dict(on=a.debug, god=False, fps=60., ms=[0., 0., 0.], mapmode=1 if a.debug else 0, follow=a.map_follow)      # mapmode: 0 off, 1 whole dungeon, 2 explored only

    loading('compiling the world generator and simulation on %s (the first launch takes a while)...' % info['platform'])
    key = jax.random.PRNGKey(a.seed if a.seed is not None else int(time.time()) % 100000)
    gen_id = [0]                                                                  # bumps whenever a new floor is generated
    def new_run(key):
        key, k = jax.random.split(key); gen_id[0] += 1; return key, reset(k, a.floor)
    key, s = new_run(key)
    joy = pygame.joystick.Joystick(0) if pygame.joystick.get_count() else None
    toast = [['Floor %d' % a.floor], 2.5]
    cache = {'sig': None}

    # ---- everything below is numpy: no device calls while drawing
    def view(va, mir, x, y):
        u = mir*x; c, s_ = math.cos(va), math.sin(va); return c*u-s_*y, s_*u+c*y
    def room_tables(sn):
        sig = (gen_id[0], int(sn['room']))
        if cache['sig'] != sig:
            T = jax.device_get(tables(s)); rm = T['room']; n = int(rm.n); side = rm.sides[:n]; V = []
            for k in range(n):                                                    # polygon corners: intersect neighbouring side lines
                (p0, p1), (q0, q1) = side.normal[k], side.normal[(k+1) % n]; Mx = np.array([[p0, p1], [q0, q1]])
                bb = np.array([side.normal[k] @ side.mid[k], side.normal[(k+1) % n] @ side.mid[(k+1) % n]])
                V.append(np.linalg.solve(Mx, bb) if abs(np.linalg.det(Mx)) > 1e-9 else side.mid[k])
            cache.update(sig=sig, n=n, side=side, doors=np.asarray(rm.doors)[:n], V=np.array(V), ob=rm.obstacles,
                         wax=int(rm.wall_axis), wkind=int(rm.wall_kind), wpos=float(rm.wall_pos), dsec=np.asarray(rm.door_sec)[:n], pad=np.asarray(rm.pad),
                         spawn=np.asarray(T['spawn']), spawn_ok=np.asarray(T['spawn_ok']), spawn_sec=np.asarray(T['spawn_sec']), hue=(sig[1]*137.508) % 360)
        return cache
    def hsl(h, sat, l):
        c = pygame.Color(0); c.hsla = (h % 360, sat*100, l*100, 100); return c

    mp = {'sig': None}
    from seamjax.minimap import chart, orient
    from seamjax.worldname import explain as explain_world
    def map_layout():
        """per-floor minimap data (room centre vectors, doors, which doors cross a wrap seam); fetched once per floor"""
        if mp['sig'] == gen_id[0]: return mp
        Ld = jax.device_get(layout(s)); m = int(Ld['n_rooms'])
        mp.clear(); mp.update(sig=gen_id[0], m=m, C=np.asarray(Ld['mpos'])[:m].astype(float), geo=int(Ld['geo']), doors=np.asarray(Ld['doors'])[:m], dseam=np.asarray(Ld['dseam'])[:m],
                              ns=np.asarray(Ld['n'])[:m], start=int(Ld['start']), exit=int(Ld['exit']), flip=False, ckey=None)       # geo: 0 flat 1 hyper 2 sphere 3 wrapped
        return mp

    def draw_minimap(sn, r, c):
        if dbg['mapmode'] == 0: return
        L = map_layout(); L['sig'] = gen_id[0]; m = L['m']; box = 112.; cx, cy = VW-box-14, VH-box-14; follow = dbg['follow']
        ch = chart(L, r if follow else None); P = ch['P']; va = float(sn['va']); mir = float(sn['mir'])
        sc = 42./ch['sp'] if follow else (box-16)/ch['ext']                              # pixels per chart unit
        th, flip = orient(L, P, r, c, va, mir, view)                                           # the map is turned to match what you see
        Z = (np.conj(P[:, 0]+1j*P[:, 1]) if flip else P[:, 0]+1j*P[:, 1])*np.exp(1j*th)*sc; X, Y = cx+Z.real, cy+Z.imag
        inside = np.hypot(X-cx, Y-cy) < box-8; pt = lambda i: (X[i], Y[i])
        vis = np.asarray(sn['visited'])[:m]
        cv.rect(cx-box-4, cy-box-4, 2*box+8, 2*box+8, (8, 10, 14, 170))
        show = np.ones(m, bool) if dbg['mapmode'] == 1 else vis.copy()
        if dbg['mapmode'] == 2:
            for i in range(m):
                if vis[i]:
                    for k in range(int(L['ns'][i])):
                        if L['doors'][i, k] >= 0: show[L['doors'][i, k]] = True
        show &= inside; nbr = {int(j) for j in L['doors'][r][:int(L['ns'][r])] if j >= 0}
        for i in range(m):                                                           # door links (not drawn across a seam of a wrapped world)
            if not show[i]: continue
            for k in range(int(L['ns'][i])):
                j = L['doors'][i, k]
                if j > i and show[j] and L['dseam'][i, k] == 0: cv.line(pt(i), pt(j), (255, 240, 160) if r in (i, j) else (90, 110, 120), 2 if r in (i, j) else 1)
                elif j >= 0 and L['dseam'][i, k] > 0 and show[j]: cv.disk(*pt(i), 2, (255, 190, 110))       # orange dot: a seam leaves here
        for i in range(m):
            if not show[i]: continue
            rad = max(3., min(16., L['rr'][i]*sc)); x, y = pt(i); nn = int(L['ns'][i]); hue = (i*137.508) % 360
            if vis[i]: cv.ngon(x, y, rad, nn, tuple(hsl(hue, .45, .35)))
            cv.ngon(x, y, rad, nn, tuple(hsl(hue, .5, .6)) if vis[i] else (120, 130, 135), 1)
            if i == L['start']: cv.disk(x, y+rad+2, 2, (150, 235, 255))
            if i == L['exit'] and (dbg['mapmode'] == 1 or vis[i]): cv.ring(x, y, rad+3, (140, 255, 215), 2)
            cv.text(str(i), x, y-5, 13, (255, 255, 255) if i == r else (255, 240, 160) if i in nbr else (190, 200, 205), 'c')       # room numbers match the door labels
        if int(sn['hdead']) <= 0 and dbg['on'] and int(sn['hroom']) < m and show[int(sn['hroom'])]: cv.disk(*pt(int(sn['hroom'])), 5, (255, 70, 90))
        cv.ring(*pt(r), max(4., min(18., L['rr'][r]*sc))+3, (255, 255, 255), 2)
        cv.text('map: %s, %s  (M, C)' % (['off', 'whole', 'explored'][dbg['mapmode']], 'follows you' if follow else 'fixed'), cx-box-4, cy-box-22, SMALL, (170, 190, 195))

    def draw_debug(sn, c, M, r):
        n = c['n']; wax = c['wax']
        for q, ok, sc in zip(c['spawn'], c['spawn_ok'], c['spawn_sec']):                      # where enemies may appear (green = section 0, orange = section 1)
            if ok: cv.disk(*M(q[0], q[1]), 2, (110, 255, 140) if sc == 0 else (255, 190, 90))
        if wax:
            w = c['wpos']; a0, a1 = ((0, w), (VW, w)) if wax == 1 else ((w, 0), (w, VH)); cv.line(M(*a0), M(*a1), (255, 255, 255), 1)
        for k in range(n):                                                                    # door labels: where it leads, and which side of a wall
            if c['doors'][k] >= 0:
                (dx, dy), (nx, ny) = c['side'].door[k], c['side'].normal[k]; x, y = M(dx-nx*26, dy-ny*26)
                cv.text('->%d%s' % (c['doors'][k], ('  side %d' % c['dsec'][k]) if c['wkind'] else ''), x, y-8, SMALL, (255, 240, 160), 'c')
        sec = 0 if not wax else int((float(sn['y']) if wax == 1 else float(sn['x'])) >= c['wpos'])
        sp = meta[int(sn['spec'])]; alive = int(np.asarray(sn['en'].alive).sum())
        lines = ['DEBUG (F1 hides)  N stairs  H heal  G god %s  K kill all  O open gate  T hunter  M map  C map follows you  B info' % ('ON' if dbg['god'] else 'off'),
                 'world %s [%s]  corner turn %.0f deg  rooms %d' % (sp['name'], sp['cat'], sp['hol'], int(sn['n_rooms'])),
                 '      '+explain_world(sp['name'], sp['cat']),
                 'room %d (start %d, exit %d)  view mirror %+d  pos (%.0f, %.0f)  section %d  wall %s%s' % (r, int(sn['start']), int(sn['exit']), int(round(float(sn['mir']))), float(sn['x']), float(sn['y']), sec,
                     ['none', 'solid', 'gated'][c['wkind']], ' (open)' if bool(sn['wopen']) else ''),
                 'hunter room %d section %d  doors away %d  here %s   enemies %d   sections spawned %s' % (int(sn['hroom']), int(sn['hsec']), int(sn['hdist']), bool(sn['hhere']), alive, np.asarray(sn['spawned']).astype(int).tolist()),
                 'green dot = enemy spawn point (orange: other side of a wall)   ->N = door to room N',
                 '%.0f fps   sim %.1f ms   device read %.1f ms   draw %.1f ms   seed %s   jax: %s' % (dbg['fps'], dbg['ms'][0], dbg['ms'][1], dbg['ms'][2], a.seed, describe())]
        for i, ln in enumerate(lines): cv.text(ln, 12, 58+i*17, SMALL, (255, 240, 160))

    def draw_scene(sn, t, c, M):                                     # software renderer only
        va = float(sn['va']); mir = float(sn['mir']); n = c['n']; r = int(sn['room'])
        screen.fill((6, 8, 12)); poly = [M(*v) for v in c['V']]
        layer = pygame.Surface((int(VW), int(VH))); layer.fill(hsl(c['hue'], .35, .1))
        for gx in range(0, int(VW), 44): pygame.draw.line(layer, hsl(c['hue'], .3, .14), M(gx, 0), M(gx, VH), 1)
        for gy in range(0, int(VH), 44): pygame.draw.line(layer, hsl(c['hue'], .3, .14), M(0, gy), M(VW, gy), 1)
        pm = pygame.Surface((int(VW), int(VH)), pygame.SRCALPHA); pygame.draw.polygon(pm, (255, 255, 255, 255), poly)
        pygame.mask.from_surface(pm).to_surface(screen, setsurface=layer, unsetcolor=None)        # grid only inside the room
        wopen = bool(sn['wopen']); hd = int(sn['hdoor']); wallcol = hsl(c['hue'], .55, .58)
        for k in range(n):
            A, B = c['V'][(k-1) % n], c['V'][k]
            if c['doors'][k] >= 0:
                tg, dr = c['side'].tangent[k], c['side'].door[k]; g0 = dr-tg*GAP; g1 = dr+tg*GAP
                near = A if np.dot(A-g0, A-g0) < np.dot(B-g0, B-g0) else B; far = B if near is A else A
                pygame.draw.line(screen, wallcol, M(*near), M(*g0), 4); pygame.draw.line(screen, wallcol, M(*g1), M(*far), 4)
                col = (255, 60, 90) if k == hd else (255, 190, 110) if (c['wkind'] > 0 and c['dsec'][k] > 0) else (150, 235, 255)
                pygame.draw.line(screen, col, M(*g0), M(*g1), 8 if k == hd else 5)
            else: pygame.draw.line(screen, wallcol, M(*A), M(*B), 4)
        ob = c['ob']
        for ty, (ox, oy), (w, h), wall in zip(ob.kind, ob.pos, ob.size, ob.wall):
            if ty == OBST_NONE: continue
            if wall:                                                                  # inner wall
                if wopen: continue
                col = (90, 150, 255) if c['wkind'] == WALL_GATED else (255, 100, 158)
            else: col = hsl(c['hue'], .35, .26)
            if ty == OBST_CIRCLE: pygame.draw.circle(screen, col, M(ox, oy), int(w))
            else: pygame.draw.polygon(screen, col, [M(ox, oy), M(ox+w, oy), M(ox+w, oy+h), M(ox, oy+h)])
        en = sn['en']; alive = np.asarray(en.alive)
        if r == int(sn['exit']):
            px, py = M(*c['pad']); pygame.draw.circle(screen, (140, 255, 215) if not alive.any() else (130, 130, 130), (int(px), int(py)), 18, 3)
        pk = sn['pk']
        for i in np.where(pk.alive)[0]:
            x, y = M(pk.x[i], pk.y[i]); pygame.draw.circle(screen, (255, 92, 115), (int(x-3), int(y)), 5); pygame.draw.circle(screen, (255, 92, 115), (int(x+3), int(y)), 5)
            pygame.draw.polygon(screen, (255, 92, 115), [(x-8, y+1), (x+8, y+1), (x, y+11)])
        for i in np.where(alive)[0]:
            x, y = M(en.x[i], en.y[i]); sh = bool(en.shooter[i]); rad = int(SHOOTER_R if sh else CHASER_R); tg = float(en.telegraph[i])
            if tg > 0: pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), int(rad+4*tg), 1)
            else: pygame.draw.circle(screen, (255, 255, 255) if en.flash[i] > 0 else ((180, 126, 255) if sh else (255, 122, 69)), (int(x), int(y)), rad)
        for bl, col, rad in ((sn['pb'], (255, 226, 122), 4), (sn['eb'], (255, 138, 210), 5)):
            for i in np.where(bl.alive)[0]: pygame.draw.circle(screen, col, tuple(map(int, M(bl.x[i], bl.y[i]))), rad)
        if bool(sn['hhere']) and int(sn['hroom']) == r and float(sn['hdead']) <= 0:
            x, y = M(float(sn['hx']), float(sn['hy']))
            pts = [(x+H_R*1.2*math.sin(2*math.pi*i/5), y-H_R*1.2*math.cos(2*math.pi*i/5)) for i in range(5)]
            pygame.draw.polygon(screen, (255, 255, 255) if float(sn['hflash']) > 0 else (163, 18, 43), pts); pygame.draw.polygon(screen, (255, 93, 115), pts, 3)
        px, py = M(float(sn['x']), float(sn['y']))
        if not (float(sn['inv']) > 0 and int(t*20) % 2): pygame.draw.circle(screen, (232, 245, 255), (int(px), int(py)), int(PR))
        ang = float(sn['aim']); ax, ay = view(va, mir, math.cos(ang), math.sin(ang)); pygame.draw.line(screen, (255, 226, 122), (px, py), (px+ax*20, py+ay*20), 3)

    def draw_hud(sn, t, dead, c, M):
        r = int(sn['room'])
        if bool(sn['hhere']) and int(sn['hroom']) == r and float(sn['hdead']) <= 0:                  # Hunter health bar
            x, y = M(float(sn['hx']), float(sn['hy'])); cv.rect(x-18, y-34, 36, 4, (60, 10, 20)); cv.rect(x-18, y-34, 36*float(sn['hhp'])/HUNTER_HP, 4, (255, 93, 115))
        for i in range(MAXHP):
            cv.rect(12+i*22, 10, 16, 16, (255, 92, 115) if i < int(sn['hp']) else (60, 40, 46), 4)
        cv.text('Floor %d   kills %d   %s' % (int(sn['floor']), int(sn['kills']), 'in a wall-split room' if c['wax'] else ''), 12, 34, FONT, (180, 215, 210))
        hdd = int(sn['hdist'])
        ht = 'Hunter down' if hdd == -1 else 'Hunter cut off by a wall' if hdd == -2 else 'Hunter in the room' if hdd == 0 else 'Hunter %d doors away' % hdd
        cv.text(ht, VW-12, 12, FONT, (201, 138, 149), 'r')
        if dbg['on']: draw_debug(sn, c, M, r); draw_minimap(sn, r, c)
        if toast[1] > 0:
            for i, ln in enumerate(toast[0]): say(ln, 70+i*24, FONT, (230, 240, 225))
        if dead: say('YOU DIED  -  press R', VH/2-30, BIG, (255, 120, 140)); say('reached floor %d with %d kills' % (int(sn['floor']), int(sn['kills'])), VH/2+20)
        if a.bot: say('bot', VH-24, FONT, (120, 140, 150))


    t_start = time.time(); fxt = [0.]
    def draw(sn, t, dead, evs):
        c = room_tables(sn); va = float(sn['va']); mir = float(sn['mir'])
        M = lambda x, y: tuple(np.array(view(va, mir, x-CX, y-CY))+(CX, CY))
        if gl:
            now = time.time()-t_start; fx.apply(evs, M, now); fx.decay(now-fxt[0]); fxt[0] = now
            gl.scene(sn, c, M, va, mir, t, {'flat': 0, 'hyper': 1, 'sphere': 2, 'grid': 3}[meta[int(sn['spec'])]['cat']], fx, now)
        else: draw_scene(sn, t, c, M)
        cv.begin(); draw_hud(sn, t, dead, c, M); cv.end()

    def inputs(sn):
        k = pygame.key.get_pressed(); mv = np.array([float(k[pygame.K_d] or k[pygame.K_RIGHT])-float(k[pygame.K_a] or k[pygame.K_LEFT]),
                                                   float(k[pygame.K_s] or k[pygame.K_DOWN])-float(k[pygame.K_w] or k[pygame.K_UP])])
        aim = np.array([float(k[pygame.K_l])-float(k[pygame.K_j]), float(k[pygame.K_k])-float(k[pygame.K_i])])
        if pygame.mouse.get_pressed()[0]:
            p = np.array(view(float(sn['va']), float(sn['mir']), float(sn['x'])-CX, float(sn['y'])-CY))+(CX, CY)
            d = np.array(gl.to_logical(pygame.mouse.get_pos()) if gl else pygame.mouse.get_pos(), float)-p; aim = d/(np.hypot(*d)+1e-9)
        if joy:
            ax = [joy.get_axis(i) for i in range(min(joy.get_numaxes(), 4))]
            if len(ax) >= 4:
                if np.hypot(ax[0], ax[1]) > .15: mv = np.array(ax[:2])
                if np.hypot(ax[2], ax[3]) > .25: aim = np.array(ax[2:4])
        return np.concatenate([mv, aim]).astype(np.float32)

    def bot(sn):                                                     # stand off, shoot the nearest enemy; just a demo
        en = sn['en']; al = np.asarray(en.alive)
        if not al.any(): return np.zeros(4, np.float32)
        d = np.hypot(en.x-float(sn['x']), en.y-float(sn['y'])); d[~al] = 1e9; j = int(d.argmin())
        v = np.array(view(float(sn['va']), float(sn['mir']), en.x[j]-float(sn['x']), en.y[j]-float(sn['y']))); v /= np.hypot(*v)+1e-9
        return np.array([-v[0]*(d[j] < 160)+v[1]*.6, -v[1]*(d[j] < 160)-v[0]*.6, v[0], v[1]], np.float32)

    zero = jnp.zeros(3, bool); acc = zero; sn = jax.device_get(snapshot(s, acc)); dead = False
    import threading
    def warm(s0):                                                  # compile what the first descent needs while you are already playing
        try:
            jax.block_until_ready(nxt(s0, jax.random.PRNGKey(0), jnp.bool_(False), jnp.bool_(True)))
            for f in (dbg_stairs, dbg_heal, dbg_kill, dbg_open, dbg_hunter, god, layout): jax.block_until_ready(f(s0))
        except Exception: pass
    threading.Thread(target=warm, args=(s,), daemon=True).start()
    accum = 0.; t = 0.; frames = 0; running = True; last = time.time(); tm = dict(sim=0., read=0., draw=0.)
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE): running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_F1: dbg['on'] = not dbg['on']
            if ev.type == pygame.KEYDOWN and dbg['on']:
                kk = ev.key
                if kk == pygame.K_n: s = dbg_stairs(s); toast[:] = [['debug: jumped to the stairs'], 1.5]
                elif kk == pygame.K_h: s = dbg_heal(s); toast[:] = [['debug: healed'], 1.]
                elif kk == pygame.K_g: dbg['god'] = not dbg['god']; toast[:] = [['debug: god mode ' + ('on' if dbg['god'] else 'off')], 1.5]
                elif kk == pygame.K_k: s = dbg_kill(s); toast[:] = [['debug: enemies cleared'], 1.]
                elif kk == pygame.K_o: s = dbg_open(s); toast[:] = [['debug: gate opened (if this room has one)'], 1.5]
                elif kk == pygame.K_t: s = dbg_hunter(s); toast[:] = [['debug: Hunter toggled'], 1.5]
                elif kk == pygame.K_m: dbg['mapmode'] = (dbg['mapmode']+1) % 3
                elif kk == pygame.K_c: dbg['follow'] = not dbg['follow']
                elif kk == pygame.K_b:
                    sp = meta[int(sn['spec'])]; print('floor %d  world %s [%s]  corner turn %.0f deg  room %d of %d' % (int(sn['floor']), sp['name'], sp['cat'], sp['hol'], int(sn['room']), int(sn['n_rooms'])))
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                key, s = new_run(key); dead = False; toast[:] = [['Floor %d' % a.floor], 2.5]; acc = zero; sn = jax.device_get(snapshot(s, acc))
        now = time.time(); accum += min(.05, now-last); last = now; t0 = time.time(); steps = 0
        while accum >= 1/60. and not dead and steps < 2:                 # at most 2 steps per frame so a slow frame cannot spiral
            accum -= 1/60.; t += 1/60.; toast[1] -= 1/60.; steps += 1
            if a.debug_stairs and frames == 30 and steps == 1: s = dbg_stairs(s)
            act = bot(sn) if a.bot else inputs(sn)
            if dbg['god']: s = god(s)
            s, acc = advance(s, jnp.asarray(act), acc)
        if accum > .2: accum = 0.
        t1 = time.time(); sn = jax.device_get(snapshot(s, acc)); t2 = time.time()                        # the one device read of this frame
        opened, died, stairs = (bool(v) for v in sn['acc']); acc = zero
        if opened: toast[:] = [['A wall slides open.'], 2.5]
        if died: dead = True
        elif stairs:
            prev = int(sn['floor']); sp = meta[int(sn['spec'])]
            loading('descending...', False)
            key, k = jax.random.split(key); gen_id[0] += 1; s = nxt(s, k, jnp.bool_(False), jnp.bool_(True)); accum = 0.
            kind = {'flat': 'flat', 'hyper': 'hyperbolic', 'sphere': 'spherical', 'grid': 'wrapped'}[sp['cat']]
            toast[:] = [['Floor %d' % (prev+1), 'Floor %d was %s (%s): once round a corner you turned %.0f deg' % (prev, sp['name'], kind, sp['hol'])], 5.]
            sn = jax.device_get(snapshot(s, zero))
        if dead: accum = 0.
        evs = events.update(sn, (gen_id[0], int(sn['room'])), gen_id[0], dead, opened)
        if audio:
            audio.events(evs, lambda x, y: float(np.clip(view(float(sn['va']), float(sn['mir']), x-CX, y-CY)[0]/CX, -1, 1)))
            audio.set_world(meta[int(sn['spec'])]['cat']); audio.set_hunter(-1 if dead else int(sn['hdist']))
        t3 = time.time(); draw(sn, t, dead, evs); pygame.display.flip(); t4 = time.time(); clock.tick(60); frames += 1
        tm['sim'] += t1-t0; tm['read'] += t2-t1; tm['draw'] += t4-t3
        dbg['ms'] = [.9*dbg['ms'][0]+.1*1000*(t1-t0), .9*dbg['ms'][1]+.1*1000*(t2-t1), .9*dbg['ms'][2]+.1*1000*(t4-t3)]; dbg['fps'] = .95*dbg['fps']+.05*clock.get_fps()
        if a.timing and frames % 120 == 0:
            print('ms/frame  sim %.2f  device read %.2f  draw+flip %.2f' % tuple(1000*tm[k]/120 for k in ('sim', 'read', 'draw'))); tm = dict(sim=0., read=0., draw=0.)
        if a.frames and frames >= a.frames: running = False
    if a.shot: pygame.image.save(gl.screenshot() if gl else screen, a.shot)
    pygame.quit()

if __name__ == '__main__':
    main()
