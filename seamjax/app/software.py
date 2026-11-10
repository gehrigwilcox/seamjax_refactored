"""The room drawn with plain pygame on the CPU (--software, and the fallback when OpenGL cannot start)."""
import math
import numpy as np
import pygame
from seamjax.core.config import VIEW_W as VW, VIEW_H as VH, DOOR_GAP, PLAYER_R, HUNTER_R, CHASER_R, SHOOTER_R
from seamjax.core.types import OBST_NONE, OBST_CIRCLE, WALL_GATED
from seamjax.core.geometry.view import view_np
from seamjax.app.minimap import hsl
from seamjax.app.snapshot import hunter_shown


def draw_room(screen, sn, room, M, t):
    """M maps a room-frame point to the screen; t: time in seconds (blinking)"""
    va = float(sn.view_angle); mir = float(sn.mirror); n = room.n; V = room.corners
    screen.fill((6, 8, 12)); poly = [M(*v) for v in V]
    layer = pygame.Surface((int(VW), int(VH))); layer.fill(hsl(room.hue, .35, .1))
    for gx in range(0, int(VW), 44): pygame.draw.line(layer, hsl(room.hue, .3, .14), M(gx, 0), M(gx, VH), 1)
    for gy in range(0, int(VH), 44): pygame.draw.line(layer, hsl(room.hue, .3, .14), M(0, gy), M(VW, gy), 1)
    pm = pygame.Surface((int(VW), int(VH)), pygame.SRCALPHA); pygame.draw.polygon(pm, (255, 255, 255, 255), poly)
    pygame.mask.from_surface(pm).to_surface(screen, setsurface=layer, unsetcolor=None)        # grid only inside the room
    wopen = bool(sn.wall_open); hd = int(sn.hunter_door); wallcol = hsl(room.hue, .55, .58)
    # walls and doors
    for k in range(n):
        A, B = V[(k-1) % n], V[k]
        if room.doors[k] >= 0:
            tg, dr = room.sides.tangent[k], room.sides.door[k]; g0 = dr-tg*DOOR_GAP; g1 = dr+tg*DOOR_GAP
            near = A if np.dot(A-g0, A-g0) < np.dot(B-g0, B-g0) else B; far = B if near is A else A
            pygame.draw.line(screen, wallcol, M(*near), M(*g0), 4); pygame.draw.line(screen, wallcol, M(*g1), M(*far), 4)
            col = (255, 60, 90) if k == hd else (255, 190, 110) if (room.wall_kind > 0 and room.door_sec[k] > 0) else (150, 235, 255)
            pygame.draw.line(screen, col, M(*g0), M(*g1), 8 if k == hd else 5)
        else: pygame.draw.line(screen, wallcol, M(*A), M(*B), 4)
    # obstacles and the inner wall
    ob = room.obstacles
    for ty, (ox, oy), (w, h), wall in zip(ob.kind, ob.pos, ob.size, ob.wall):
        if ty == OBST_NONE or (wall and wopen): continue
        col = ((90, 150, 255) if room.wall_kind == WALL_GATED else (255, 100, 158)) if wall else hsl(room.hue, .35, .26)
        if ty == OBST_CIRCLE: pygame.draw.circle(screen, col, M(ox, oy), int(w))
        else: pygame.draw.polygon(screen, col, [M(ox, oy), M(ox+w, oy), M(ox+w, oy+h), M(ox, oy+h)])
    # exit pad, pickups, enemies, bullets
    en = sn.enemies; alive = np.asarray(en.alive)
    if int(sn.room) == int(sn.exit):
        px, py = M(*room.pad); pygame.draw.circle(screen, (140, 255, 215) if not alive.any() else (130, 130, 130), (int(px), int(py)), 18, 3)
    pk = sn.pickups
    for i in np.where(pk.alive)[0]:
        x, y = M(pk.x[i], pk.y[i]); pygame.draw.circle(screen, (255, 92, 115), (int(x-3), int(y)), 5); pygame.draw.circle(screen, (255, 92, 115), (int(x+3), int(y)), 5)
        pygame.draw.polygon(screen, (255, 92, 115), [(x-8, y+1), (x+8, y+1), (x, y+11)])
    for i in np.where(alive)[0]:
        x, y = M(en.x[i], en.y[i]); sh = bool(en.shooter[i]); rad = int(SHOOTER_R if sh else CHASER_R); tg = float(en.telegraph[i])
        if tg > 0: pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), int(rad+4*tg), 1)
        else: pygame.draw.circle(screen, (255, 255, 255) if en.flash[i] > 0 else ((180, 126, 255) if sh else (255, 122, 69)), (int(x), int(y)), rad)
    for bl, col, rad in ((sn.pb, (255, 226, 122), 4), (sn.eb, (255, 138, 210), 5)):
        for i in np.where(bl.alive)[0]: pygame.draw.circle(screen, col, tuple(map(int, M(bl.x[i], bl.y[i]))), rad)
    # the Hunter, and you
    if hunter_shown(sn):
        x, y = M(float(sn.hunter.x), float(sn.hunter.y))
        pts = [(x+HUNTER_R*1.2*math.sin(2*math.pi*i/5), y-HUNTER_R*1.2*math.cos(2*math.pi*i/5)) for i in range(5)]
        pygame.draw.polygon(screen, (255, 255, 255) if float(sn.hunter.flash) > 0 else (163, 18, 43), pts); pygame.draw.polygon(screen, (255, 93, 115), pts, 3)
    px, py = M(float(sn.player.x), float(sn.player.y))
    if not (float(sn.player.inv) > 0 and int(t*20) % 2): pygame.draw.circle(screen, (232, 245, 255), (int(px), int(py)), int(PLAYER_R))
    ang = float(sn.player.aim); ax, ay = view_np(va, mir, math.cos(ang), math.sin(ang)); pygame.draw.line(screen, (255, 226, 122), (px, py), (px+ax*20, py+ay*20), 3)
