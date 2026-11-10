"""
OpenGL view for the Seam viewer (moderngl).  Everything you see in the room is drawn by the GPU:

  1. background pass   one full-screen fragment shader: procedural floor texture (a different pattern for flat / hyperbolic /
                       spherical / wrapped worlds, tinted by the room's hue, with a light pool around the player), polygon walls,
                       doors and obstacles as glowing distance fields
  2. sprite pass       instanced quads: enemies, bullets, hearts, the Hunter, the player, the exit pad (additive blending)
  3. particle pass     instanced quads whose position is computed in the vertex shader from spawn position, velocity and age
  4. post              bloom (two blur levels), screen shake, chromatic hit flash, vignette, fade-in on a new floor
  5. HUD               GLCanvas: text (glyph quads from an atlas), hearts, minimap and debug overlay as instanced distance-field shapes

All shader coordinates are the game's logical pixels (960 x 600, y down), already transformed by the room's view
(rotation and mirror) on the CPU, so the shaders never flip anything: controls are untouched by any of this.
fx.py turns game events (see app/events.py: a kill, a hit, a pickup...) into particles, shake and flashes.
"""
import math, colorsys
import numpy as np
import moderngl
from seamjax.core.config import CX, CY, DOOR_GAP, PLAYER_R, HUNTER_R, CHASER_R, SHOOTER_R
from seamjax.core.types import OBST_NONE, OBST_CIRCLE, WALL_GATED
from seamjax.core.geometry.view import view_np

from seamjax.app.gl.shaders import (NSEG, NSPR, NPART, Q, VERT_FS, SCENE_FS, SPRITE_VS, SPRITE_FS, PART_VS, PART_FS, BRIGHT_FS,
                                     BLUR_FS, COMP_FS)
from seamjax.app.snapshot import hunter_shown


def rgb(h, s, l): return np.array(colorsys.hls_to_rgb((h % 360)/360., l, s), 'f4')


class GLView:
    def __init__(self, VW, VH):
        self.VW, self.VH = VW, VH
        ctx = self.ctx = moderngl.create_context()
        if ctx.version_code < 330: raise RuntimeError('OpenGL 3.3 needed, got %d' % ctx.version_code)
        mk = lambda vs, fs: ctx.program(vertex_shader=vs, fragment_shader=fs)
        self.p_scene, self.p_spr, self.p_part = mk(VERT_FS, SCENE_FS), mk(SPRITE_VS, SPRITE_FS), mk(PART_VS, PART_FS)
        self.p_bright, self.p_blur, self.p_comp = mk(VERT_FS, BRIGHT_FS), mk(VERT_FS, BLUR_FS), mk(VERT_FS, COMP_FS)
        self.tri = ctx.buffer(Q); self.quad_vao = {p: ctx.vertex_array(p, [(self.tri, '2f', 'in_pos')]) for p in (self.p_scene, self.p_bright, self.p_blur, self.p_comp)}
        self.spr_buf = ctx.buffer(reserve=NSPR*11*4, dynamic=True); self.part_buf = ctx.buffer(reserve=NPART*11*4, dynamic=True)
        self.spr_vao = ctx.vertex_array(self.p_spr, [(self.tri, '2f', 'in_corner'), (self.spr_buf, '2f 1f 1f 1f 4f 1f 1f/i', 'in_c', 'in_r', 'in_ext', 'in_kind', 'in_col', 'in_ang', 'in_len')])
        self.part_vao = ctx.vertex_array(self.p_part, [(self.tri, '2f', 'in_corner'), (self.part_buf, '2f 2f 1f 1f 4f 1f/i', 'in_p0', 'in_v', 'in_t0', 'in_life', 'in_col', 'in_size')])
        self.size = None; self.rect = (0, 0, int(VW), int(VH)); self.win = (int(VW), int(VH))

    # ---- window / framebuffers
    def resize(self, win):
        WW, WH = win; s = min(WW/self.VW, WH/self.VH); RW, RH = max(2, int(self.VW*s)), max(2, int(self.VH*s)); self.win = win; self.scale = s
        self.rect = ((WW-RW)//2, (WH-RH)//2, RW, RH)
        if self.size == (RW, RH): return
        self.size = (RW, RH); ctx = self.ctx
        def tex(w, h):
            try: t = ctx.texture((w, h), 4, dtype='f2')
            except Exception: t = ctx.texture((w, h), 4)
            t.filter = (moderngl.LINEAR, moderngl.LINEAR); t.repeat_x = t.repeat_y = False; return t, ctx.framebuffer([t])
        self.t_scene, self.f_scene = tex(RW, RH); self.t_h1, self.f_h1 = tex(RW//2, RH//2); self.t_h2, self.f_h2 = tex(RW//2, RH//2)
        self.t_q1, self.f_q1 = tex(RW//4, RH//4); self.t_q2, self.f_q2 = tex(RW//4, RH//4)

    def to_logical(self, pos):
        x, y, w, h = self.rect; return ((pos[0]-x)/self.scale, (pos[1]-y)/self.scale)

    # ---- helpers
    @staticmethod
    def _set(prog, **kv):
        for k, v in kv.items():
            if k in prog: prog[k].value = v if not isinstance(v, np.ndarray) else v.tolist()
    @staticmethod
    def _arr(prog, name, a):
        if name in prog: prog[name].write(np.ascontiguousarray(a, 'f4').tobytes())

    def _geometry(self, sn, c, M, t):
        n = c.n; V = np.array([M(*v) for v in c.corners]); hue = c.hue; hd = int(sn.hunter_door); wallc = rgb(hue, .55, .58); segs = []
        for k in range(n):
            A, B = V[(k-1) % n], V[k]
            if c.doors[k] >= 0:
                (tx, ty), (dx, dy) = c.sides.tangent[k], c.sides.door[k]
                g0 = np.array(M(dx-tx*DOOR_GAP, dy-ty*DOOR_GAP)); g1 = np.array(M(dx+tx*DOOR_GAP, dy+ty*DOOR_GAP))
                near, far = (A, B) if np.dot(A-g0, A-g0) < np.dot(B-g0, B-g0) else (B, A)
                col = np.array([1., .24, .35])*(.55+.45*math.sin(t*8)) if k == hd else np.array([1., .745, .43]) if (c.wall_kind > 0 and c.door_sec[k] > 0) else np.array([.59, .92, 1.])
                segs += [(near, g0, wallc, 4.), (g1, far, wallc, 4.), (g0, g1, col, 8. if k == hd else 5.)]
            else: segs.append((A, B, wallc, 4.))
        sa = np.zeros((NSEG, 4), 'f4'); sc = np.zeros((NSEG, 4), 'f4')
        for i, (a, b, col, w) in enumerate(segs[:NSEG]): sa[i] = [*a, *b]; sc[i] = [*col, w]
        e = V[1]-V[0]; ce = V.mean(0); wind = 1. if np.dot(ce-V[0], np.array([e[1], -e[0]])) < 0 else -1.
        vt = np.zeros((12, 2), 'f4'); vt[:n] = V
        return n, vt, wind, min(len(segs), NSEG), sa, sc

    def _obstacles(self, sn, c, M, va, mir):
        A = np.zeros((6, 4), 'f4'); B = np.zeros((6, 4), 'f4'); C = np.zeros((6, 4), 'f4'); m = 0; wopen = bool(sn.wall_open); cs, sn_ = math.cos(va), math.sin(va)
        ob = c.obstacles
        for ty, (px, py), (w, h), wall in zip(ob.kind, ob.pos, ob.size, ob.wall):
            if ty == OBST_NONE or m >= 6: continue
            if wall and wopen: continue
            col = (np.array([.35, .59, 1.]) if c.wall_kind == WALL_GATED else np.array([1., .39, .62])) if wall else rgb(c.hue, .35, .26)
            if ty == OBST_CIRCLE: ctr = M(px, py); A[m] = [*ctr, w, 0]; B[m] = [0, 1, 0, 0]
            else: ctr = M(px+w/2, py+h/2); A[m] = [*ctr, w/2, h/2]; B[m] = [1, cs, sn_, 0]
            C[m] = [*col, 1. if wall else 0.]; m += 1
        return m, A, B, C

    def _sprites(self, sn, c, M, va, mir, t):
        rows = []; r = int(sn.room)
        def add(p, rad, kind, col, inten=1., ext=None, ang=0., ln=0.):
            rows.append([p[0], p[1], rad, ext if ext is not None else rad*3.2+8, kind, col[0], col[1], col[2], inten, ang, ln])
        en = sn.enemies; alive = np.asarray(en.alive)
        if r == int(sn.exit):
            open_ = not alive.any(); pc = M(*c.pad); pulse = .8+.2*math.sin(t*4)
            add(pc, 18, 1, (.55, 1., .84) if open_ else (.5, .5, .5), pulse if open_ else .6, 34, 0, 3.)
            if open_: add(pc, 8, 0, (.55, 1., .84), .35*pulse, 40)
        pk = sn.pickups
        for i in np.where(pk.alive)[0]: add(M(pk.x[i], pk.y[i]), 11*(1+.08*math.sin(t*4+i)), 3, (1., .36, .45), 1.1, 40)
        for i in np.where(alive)[0]:
            p = M(en.x[i], en.y[i]); sh = bool(en.shooter[i]); rad = SHOOTER_R if sh else CHASER_R; tg = float(en.telegraph[i])
            if tg > 0: add(p, rad+4*tg, 1, (1., 1., 1.), .8, rad*3+10, 0, 1.6)
            else: add(p, rad, 0, (1., 1., 1.) if en.flash[i] > 0 else ((.71, .49, 1.) if sh else (1., .48, .27)), 1.)
        for bl, rad, col, inten, ext, ln in ((sn.pb, 3, (1., .89, .48), 1.5, 40, 16.), (sn.eb, 4.5, (1., .54, .82), 1.4, 36, 10.)):
            for i in np.where(bl.alive)[0]:
                vx, vy = view_np(va, mir, bl.vx[i], bl.vy[i]); add(M(bl.x[i], bl.y[i]), rad, 4, col, inten, ext, math.atan2(-vy, -vx), ln)
        if hunter_shown(sn):
            add(M(float(sn.hunter.x), float(sn.hunter.y)), HUNTER_R*1.2, 2, (1., 1., 1.) if float(sn.hunter.flash) > 0 else (1., .36, .45), 1.1, HUNTER_R*4, t*.8)
        px = M(float(sn.player.x), float(sn.player.y))
        if not (float(sn.player.inv) > 0 and int(t*20) % 2):
            add(px, PLAYER_R, 0, (.8, .9, 1.), .85, 44)
        ang = float(sn.player.aim); ax, ay = view_np(va, mir, math.cos(ang), math.sin(ang)); add(px, 1.6, 4, (1., .89, .48), 1.2, 40, math.atan2(ay, ax), 20.)
        return np.array(rows[:NSPR], 'f4').reshape(-1, 11), px

    # ---- one frame of the room (everything except the HUD)
    def scene(self, sn, c, M, va, mir, t, style, fx, now):
        import pygame
        ctx = self.ctx; self.resize(pygame.display.get_window_size())
        RW, RH = self.size; res = (self.VW, self.VH)
        n, vt, wind, ns, sa, sc = self._geometry(sn, c, M, t); no, oA, oB, oC = self._obstacles(sn, c, M, va, mir); spr, ppos = self._sprites(sn, c, M, va, mir, t)
        self.f_scene.use(); ctx.viewport = (0, 0, RW, RH); ctx.disable(moderngl.BLEND)
        P = self.p_scene
        self._set(P, u_res=res, u_ctr=(CX, CY), u_player=tuple(ppos), u_time=float(t), u_hue=float(c.hue), u_va=float(va), u_mir=float(mir), u_style=int(style), u_nv=int(n), u_wind=float(wind), u_ns=int(ns), u_no=int(no))
        for nm, arr in (('u_vtx', vt), ('u_segA', sa), ('u_segC', sc), ('u_obsA', oA), ('u_obsB', oB), ('u_obsC', oC)): self._arr(P, nm, arr)
        self.quad_vao[P].render(moderngl.TRIANGLE_STRIP)
        ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.ONE, moderngl.ONE
        if len(spr):
            self.spr_buf.write(spr.tobytes()); self._set(self.p_spr, u_res=res); self.spr_vao.render(moderngl.TRIANGLE_STRIP, instances=len(spr))
        self.part_buf.write(fx.part.tobytes()); self._set(self.p_part, u_res=res, u_now=float(now)); self.part_vao.render(moderngl.TRIANGLE_STRIP, instances=NPART)
        ctx.disable(moderngl.BLEND)
        # bloom: bright pass -> half res blur -> quarter res blur
        def blur(src, dst, tmp_t, tmp_f, w, h):
            self.p_blur['u_src'].value = 0
            tmp_f.use(); ctx.viewport = (0, 0, w, h); src.use(0); self._set(self.p_blur, u_dir=(1./w, 0.)); self.quad_vao[self.p_blur].render(moderngl.TRIANGLE_STRIP)
            dst.use(); ctx.viewport = (0, 0, w, h); tmp_t.use(0); self._set(self.p_blur, u_dir=(0., 1./h)); self.quad_vao[self.p_blur].render(moderngl.TRIANGLE_STRIP)
        self.f_h1.use(); ctx.viewport = (0, 0, RW//2, RH//2); self.t_scene.use(0); self.p_bright['u_src'].value = 0; self.quad_vao[self.p_bright].render(moderngl.TRIANGLE_STRIP)
        blur(self.t_h1, self.f_h1, self.t_h2, self.f_h2, RW//2, RH//2)
        self.f_q1.use(); ctx.viewport = (0, 0, RW//4, RH//4); self.t_h1.use(0); self.p_bright['u_src'].value = 0; self.quad_vao[self.p_bright].render(moderngl.TRIANGLE_STRIP)
        blur(self.t_q1, self.f_q1, self.t_q2, self.f_q2, RW//4, RH//4)
        # composite to the window (letterboxed)
        ctx.screen.use(); ctx.viewport = (0, 0, *self.win); ctx.screen.clear(0., 0., 0., 1.); ctx.viewport = self.rect
        self.t_scene.use(0); self.t_h1.use(1); self.t_q1.use(2)
        sh = (np.random.uniform(-1, 1, 2)*fx.shake*6.)/np.array([RW, RH])
        self._set(self.p_comp, u_scene=0, u_b1=1, u_b2=2, u_shake=tuple(sh), u_hit=float(min(1., fx.hit)), u_flash=float(fx.flash), u_fade=float(fx.fade))
        self.quad_vao[self.p_comp].render(moderngl.TRIANGLE_STRIP)

    def screenshot(self):
        import pygame
        x, y, w, h = self.rect; data = self.ctx.screen.read(viewport=(x, y, w, h), components=3)
        return pygame.image.frombytes(data, (w, h), 'RGB', True)
