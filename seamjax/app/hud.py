"""
Everything drawn over the room: hearts, floor and kills, the Hunter's distance, messages ("toasts"), the death screen, and in
debug mode the overlay text, door labels, spawn points and the minimap. Written against the canvas interface (canvas.py),
so the same code draws with OpenGL or pygame.
"""
from dataclasses import dataclass, field
import numpy as np
from seamjax.core.config import VIEW_W as VW, VIEW_H as VH, HUNTER_HP
from seamjax.core import kinds
from seamjax.difficulty import FLOOR_CAP
from seamjax.app import minimap
from seamjax.app.snapshot import hunter_shown
from seamjax.app.worldname import explain as explain_world

FONT, BIG, SMALL = 22, 56, 19                         # text sizes
KIND = {'flat': 'flat', 'hyper': 'hyperbolic', 'sphere': 'spherical', 'grid': 'wrapped'}


@dataclass
class Toast:
    """a few lines of text in the middle of the screen for a while"""
    lines: list = field(default_factory=list)
    left: float = 0.
    def show(self, lines, secs): self.lines, self.left = list(lines), secs
    def tick(self, dt): self.left -= dt


@dataclass
class Debug:
    on: bool = False
    god: bool = False
    map_mode: int = 0                                 # minimap.MODES
    map_follow: bool = False
    fps: float = 60.
    ms: list = field(default_factory=lambda: [0., 0., 0.])     # smoothed sim / device read / draw times
    seed: object = None
    backend: str = ''


def say(cv, txt, y=VH/2, size=FONT, col=(210, 235, 230)):
    cv.text(txt, VW/2, y, size, col, 'c')


def loading(cv, msg, title=True):
    """a full-screen message (the window is up before anything slow starts)"""
    cv.begin((8, 10, 14))
    if title: say(cv, 'SEAM', VH/2-60, BIG, (150, 235, 255))
    say(cv, msg, VH/2+10); cv.end()


def draw(cv, sn, room, M, world, toast, dead, dbg, map_layout=None, bot=False):
    """M maps a room-frame point to the screen; world: the catalogue entry; map_layout: a callable (fetched only when shown)"""
    if hunter_shown(sn):                                                     # Hunter health bar
        x, y = M(float(sn.hunter.x), float(sn.hunter.y)); cv.rect(x-18, y-34, 36, 4, (60, 10, 20)); cv.rect(x-18, y-34, 36*float(sn.hunter.hp)/HUNTER_HP, 4, (255, 93, 115))
    en = sn.enemies                                                          # a boss in the room: its name and health across the top
    for i in np.where(np.asarray(en.alive))[0]:
        k = kinds.ENEMY_KINDS[int(en.kind[i])]
        if k.boss_every:
            full = k.hp+min(int(sn.floor), FLOOR_CAP)*k.hp_per_floor; w = 360.
            cv.text(k.name.upper(), VW/2, 46, FONT, (255, 170, 220), 'c'); cv.rect(VW/2-w/2, 72, w, 8, (60, 20, 50), 3)
            cv.rect(VW/2-w/2, 72, w*max(0., float(en.hp[i]))/full, 8, (255, 110, 200), 3); break
    pc = kinds.PLAYER_CLASSES[int(sn.player.cls)]
    for i in range(pc.max_hp):
        cv.rect(12+i*22, 10, 16, 16, (255, 92, 115) if i < int(sn.player.hp) else (60, 40, 46), 4)
    cv.text('Floor %d   kills %d   %s%s' % (int(sn.floor), int(sn.kills), '' if pc is kinds.GUNNER else pc.name+'   ', 'in a wall-split room' if room.wall_axis else ''),
            12, 34, FONT, (180, 215, 210))
    hd = int(sn.hunter_dist)
    cv.text('Hunter down' if hd == -1 else 'Hunter cut off by a wall' if hd == -2 else 'Hunter in the room' if hd == 0 else 'Hunter %d doors away' % hd,
            VW-12, 12, FONT, (201, 138, 149), 'r')
    if dbg.on:
        draw_debug(cv, sn, room, M, world, dbg)
        if dbg.map_mode and map_layout: minimap.draw(cv, map_layout(), sn, room, dbg.map_mode, dbg.map_follow, True, VW, VH, SMALL)
    if toast.left > 0:
        for i, ln in enumerate(toast.lines): say(cv, ln, 70+i*24, FONT, (230, 240, 225))
    if dead: say(cv, 'YOU DIED  -  press R', VH/2-30, BIG, (255, 120, 140)); say(cv, 'reached floor %d with %d kills' % (int(sn.floor), int(sn.kills)), VH/2+20)
    if bot: say(cv, 'bot', VH-24, FONT, (120, 140, 150))


def draw_debug(cv, sn, room, M, world, dbg):
    for q, ok, sc in zip(room.spawn, room.spawn_ok, room.spawn_sec):         # where enemies may appear (green = section 0, orange = section 1)
        if ok: cv.disk(*M(q[0], q[1]), 2, (110, 255, 140) if sc == 0 else (255, 190, 90))
    if room.wall_axis:
        w = room.wall_pos; a0, a1 = ((0, w), (VW, w)) if room.wall_axis == 1 else ((w, 0), (w, VH)); cv.line(M(*a0), M(*a1), (255, 255, 255), 1)
    for k in range(room.n):                                                  # door labels: where it leads, and which side of a wall
        if room.doors[k] >= 0:
            (dx, dy), (nx, ny) = room.sides.door[k], room.sides.normal[k]; x, y = M(dx-nx*26, dy-ny*26)
            cv.text('->%d%s' % (room.doors[k], ('  side %d' % room.door_sec[k]) if room.wall_kind else ''), x, y-8, SMALL, (255, 240, 160), 'c')
    px, py = float(sn.player.x), float(sn.player.y)
    sec = 0 if not room.wall_axis else int((py if room.wall_axis == 1 else px) >= room.wall_pos)
    H = sn.hunter
    lines = ['DEBUG (F1 hides)  N stairs  H heal  G god %s  K kill all  O open gate  T hunter  M map  C map follows you  B info' % ('ON' if dbg.god else 'off'),
             'world %s [%s]  corner turn %.0f deg  rooms %d' % (world['name'], world['cat'], world['hol'], int(sn.n_rooms)),
             '      '+explain_world(world['name'], world['cat']),
             'room %d (start %d, exit %d)  view mirror %+d  pos (%.0f, %.0f)  section %d  wall %s%s' % (
                 int(sn.room), int(sn.start), int(sn.exit), int(round(float(sn.mirror))), px, py, sec, ['none', 'solid', 'gated'][room.wall_kind],
                 ' (open)' if bool(sn.wall_open) else ''),
             'hunter room %d section %d  doors away %d  here %s   enemies %d   sections spawned %s' % (
                 int(H.room), int(H.sec), int(sn.hunter_dist), bool(H.here), int(np.asarray(sn.enemies.alive).sum()), np.asarray(sn.spawned).astype(int).tolist()),
             'followers on the way %d   green dot = enemy spawn point (orange: other side of a wall)   ->N = door to room N' % int(np.asarray(sn.roamers.alive).sum()),
             '%.0f fps   sim %.1f ms   device read %.1f ms   draw %.1f ms   seed %s   jax: %s' % (dbg.fps, *dbg.ms, dbg.seed, dbg.backend)]
    for i, ln in enumerate(lines): cv.text(ln, 12, 58+i*17, SMALL, (255, 240, 160))


def floor_summary(world, floor):
    """the toast after taking the stairs: what the floor you left was"""
    return ['Floor %d' % (floor+1), 'Floor %d was %s (%s): once round a corner you turned %.0f deg' % (floor, world['name'], KIND[world['cat']], world['hol'])]
