"""The room drawn with plain pygame on the CPU (--software, and the fallback when OpenGL cannot start).
Colours, shapes and textures come from seamjax/theme.py, as in the OpenGL view (no glow, bloom or floor textures here)."""
import math
from dataclasses import replace
import numpy as np
import pygame
from seamjax.core.config import VIEW_W as VW, VIEW_H as VH, DOOR_GAP, PLAYER_R, HUNTER_R, SWING_SHOW
from seamjax.core import kinds
from seamjax.core.types import OBST_NONE, OBST_CIRCLE, WALL_GATED
from seamjax.core.geometry.view import view_np
from seamjax import theme as themes
from seamjax.app.minimap import hsl
from seamjax.app.snapshot import hunter_shown

_IMAGES = {}


def c255(col):
    return tuple(int(round(255*min(1., v))) for v in col)


def _image(path, rad, ang=0.):
    if path not in _IMAGES: _IMAGES[path] = pygame.image.load(themes.asset(path))
    im = pygame.transform.smoothscale(_IMAGES[path], (max(1, int(2*rad)), max(1, int(2*rad))))
    return pygame.transform.rotate(im, -math.degrees(ang)) if ang else im


def sprite(screen, look, p, rad, colour=None, ang=0., t=0.):
    """one thing as the theme draws it: its texture, or its shape in its colour"""
    x, y = p; col = c255(colour or look.colour); ang = ang+t*look.spin
    if look.texture:
        im = _image(look.texture, rad, ang); screen.blit(im, im.get_rect(center=(int(x), int(y)))); return
    if look.shape == 'ring': pygame.draw.circle(screen, col, (int(x), int(y)), int(rad), max(1, int(look.width)))
    elif look.shape == 'pentagon':
        pts = [(x+rad*math.sin(ang+2*math.pi*i/5), y-rad*math.cos(ang+2*math.pi*i/5)) for i in range(5)]
        pygame.draw.polygon(screen, col, pts); pygame.draw.polygon(screen, c255([min(1., v*1.4+.2) for v in look.colour]), pts, 3)
    elif look.shape == 'heart':
        s = rad/11; pygame.draw.circle(screen, col, (int(x-3*s), int(y)), int(5*s)); pygame.draw.circle(screen, col, (int(x+3*s), int(y)), int(5*s))
        pygame.draw.polygon(screen, col, [(x-8*s, y+1*s), (x+8*s, y+1*s), (x, y+11*s)])
    else: pygame.draw.circle(screen, col, (int(x), int(y)), max(1, int(rad)))


def draw_room(screen, sn, room, M, t, theme=None):
    """M maps a room-frame point to the screen; t: time in seconds (blinking)"""
    T = theme or themes.THEME; L = T.looks; W = T.walls
    va = float(sn.view_angle); mir = float(sn.mirror); n = room.n; V = room.corners
    screen.fill((6, 8, 12)); poly = [M(*v) for v in V]
    layer = pygame.Surface((int(VW), int(VH))); layer.fill(hsl(room.hue, .35, .1))
    for gx in range(0, int(VW), 44): pygame.draw.line(layer, hsl(room.hue, .3, .14), M(gx, 0), M(gx, VH), 1)
    for gy in range(0, int(VH), 44): pygame.draw.line(layer, hsl(room.hue, .3, .14), M(0, gy), M(VW, gy), 1)
    pm = pygame.Surface((int(VW), int(VH)), pygame.SRCALPHA); pygame.draw.polygon(pm, (255, 255, 255, 255), poly)
    pygame.mask.from_surface(pm).to_surface(screen, setsurface=layer, unsetcolor=None)        # grid only inside the room
    wopen = bool(sn.wall_open); hd = int(sn.hunter_door); wallcol = hsl(room.hue, *W.wall)
    # walls and doors
    for k in range(n):
        A, B = V[(k-1) % n], V[k]
        if room.doors[k] >= 0:
            tg, dr = room.sides.tangent[k], room.sides.door[k]; g0 = dr-tg*DOOR_GAP; g1 = dr+tg*DOOR_GAP
            near = A if np.dot(A-g0, A-g0) < np.dot(B-g0, B-g0) else B; far = B if near is A else A
            pygame.draw.line(screen, wallcol, M(*near), M(*g0), 4); pygame.draw.line(screen, wallcol, M(*g1), M(*far), 4)
            rw = bool(sn.roamer_doors[k])
            col = W.hunter_door if k == hd else W.roamer_door if rw else W.door_other_side if (room.wall_kind > 0 and room.door_sec[k] > 0) else W.door
            pygame.draw.line(screen, c255(col), M(*g0), M(*g1), 8 if k == hd or rw else 5)
        else: pygame.draw.line(screen, wallcol, M(*A), M(*B), 4)
    # obstacles and the inner wall
    ob = room.obstacles
    for ty, (ox, oy), (w, h), wall in zip(ob.kind, ob.pos, ob.size, ob.wall):
        if ty == OBST_NONE or (wall and wopen): continue
        col = c255(W.gated_wall if room.wall_kind == WALL_GATED else W.solid_wall) if wall else hsl(room.hue, *W.obstacle)
        if ty == OBST_CIRCLE: pygame.draw.circle(screen, col, M(ox, oy), int(w))
        else: pygame.draw.polygon(screen, col, [M(ox, oy), M(ox+w, oy), M(ox+w, oy+h), M(ox, oy+h)])
    # exit pad, pickups, enemies, bullets
    en = sn.enemies; alive = np.asarray(en.alive)
    if int(sn.room) == int(sn.exit):
        lk = L['pad'] if not alive.any() else L['pad_closed']; sprite(screen, lk, M(*room.pad), lk.size)
    pk = sn.pickups
    for i in np.where(pk.alive)[0]: sprite(screen, L['pickup'], M(pk.x[i], pk.y[i]), L['pickup'].size)
    for i in np.where(alive)[0]:
        kind = kinds.ENEMY_KINDS[int(en.kind[i])]; look = T.enemy(kind.name).look; rad = look.size or kind.radius; tg = float(en.telegraph[i])
        if tg > 0: sprite(screen, L['telegraph'], M(en.x[i], en.y[i]), rad+4*tg)
        else: sprite(screen, look, M(en.x[i], en.y[i]), rad, T.flash_colour if en.flash[i] > 0 else None)
    pc = kinds.PLAYER_CLASSES[int(sn.player.cls)]; wpn = pc.weapon
    for bl, lk in ((sn.pb, T.weapon(wpn.name)), (sn.eb, L['enemy_bullet'])):
        if lk.shape == 'streak': lk = replace(lk, shape='disk')                  # no streaks here: a dot
        for i in np.where(bl.alive)[0]: sprite(screen, lk, M(bl.x[i], bl.y[i]), lk.size+1)
    # the Hunter, and you
    if hunter_shown(sn):
        lk = L['hunter']; sprite(screen, lk, M(float(sn.hunter.x), float(sn.hunter.y)), lk.size or HUNTER_R*1.2,
                                 T.flash_colour if float(sn.hunter.flash) > 0 else None, t=t)
    px, py = M(float(sn.player.x), float(sn.player.y))
    if not (float(sn.player.inv) > 0 and int(t*20) % 2): lk = T.player(pc.name); sprite(screen, lk, (px, py), lk.size or PLAYER_R)
    if wpn.melee and float(sn.player.swing) > 0:                                  # a melee swing: a fan of lines over the arc
        col = c255([v*min(1., float(sn.player.swing)/SWING_SHOW+.3) for v in T.weapon(wpn.name).colour])
        for j in range(7):
            a = float(sn.player.aim) + wpn.arc*(j/6-.5); dx, dy = view_np(va, mir, math.cos(a), math.sin(a))
            pygame.draw.line(screen, col, (px+dx*PLAYER_R, py+dy*PLAYER_R), (px+dx*wpn.reach, py+dy*wpn.reach), 2)
    ang = float(sn.player.aim); ax, ay = view_np(va, mir, math.cos(ang), math.sin(ang))
    pygame.draw.line(screen, c255(L['aim'].colour), (px, py), (px+ax*L['aim'].width, py+ay*L['aim'].width), 3)
