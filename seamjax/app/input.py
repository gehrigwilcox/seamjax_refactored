"""Turning the keyboard, mouse or a gamepad into an action [move_x, move_y, aim_x, aim_y] in screen space; and a demo bot."""
import numpy as np
import pygame
from seamjax.core.config import CX, CY
from seamjax.core.geometry.view import view_np


def player_on_screen(sn):
    return np.array(view_np(float(sn.view_angle), float(sn.mirror), float(sn.player.x)-CX, float(sn.player.y)-CY))+(CX, CY)


class Controls:
    """WASD / arrows move; I J K L or the mouse (hold the left button) aim and fire; a gamepad's two sticks do both"""
    def __init__(self, to_logical=None):
        self.to_logical = to_logical or (lambda p: p)       # window pixels -> logical 960 x 600 pixels
        self.joy = pygame.joystick.Joystick(0) if pygame.joystick.get_count() else None

    def action(self, sn):
        k = pygame.key.get_pressed()
        mv = np.array([float(k[pygame.K_d] or k[pygame.K_RIGHT])-float(k[pygame.K_a] or k[pygame.K_LEFT]),
                       float(k[pygame.K_s] or k[pygame.K_DOWN])-float(k[pygame.K_w] or k[pygame.K_UP])])
        aim = np.array([float(k[pygame.K_l])-float(k[pygame.K_j]), float(k[pygame.K_k])-float(k[pygame.K_i])])
        if pygame.mouse.get_pressed()[0]:
            d = np.array(self.to_logical(pygame.mouse.get_pos()), float)-player_on_screen(sn); aim = d/(np.hypot(*d)+1e-9)
        if self.joy:
            ax = [self.joy.get_axis(i) for i in range(min(self.joy.get_numaxes(), 4))]
            if len(ax) >= 4:
                if np.hypot(ax[0], ax[1]) > .15: mv = np.array(ax[:2])
                if np.hypot(ax[2], ax[3]) > .25: aim = np.array(ax[2:4])
        return np.concatenate([mv, aim]).astype(np.float32)


def bot(sn):
    """keep some distance from the nearest enemy, circle it and shoot it; a demo / smoke test, not a good player"""
    en = sn.enemies; al = np.asarray(en.alive); x, y = float(sn.player.x), float(sn.player.y)
    if not al.any(): return np.zeros(4, np.float32)
    d = np.hypot(en.x-x, en.y-y); d[~al] = 1e9; j = int(d.argmin())
    v = np.array(view_np(float(sn.view_angle), float(sn.mirror), en.x[j]-x, en.y[j]-y)); v /= np.hypot(*v)+1e-9
    return np.array([-v[0]*(d[j] < 160)+v[1]*.6, -v[1]*(d[j] < 160)-v[0]*.6, v[0], v[1]], np.float32)
