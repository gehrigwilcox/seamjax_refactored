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
What everything looks like (shapes, colours, glow, textures, floor patterns, wall colours) comes from seamjax/theme.py:
sprite and floor images are loaded once, sprites into one atlas.
"""
import math, colorsys
import numpy as np
import moderngl
from seamjax.core.config import CX, CY, DOOR_GAP, PLAYER_R, HUNTER_R, SWING_SHOW
from seamjax.core import kinds
from seamjax import theme as themes
from seamjax.core.types import OBST_NONE, OBST_CIRCLE, WALL_GATED
from seamjax.core.geometry.view import view_np

from seamjax.app.gl.shaders import (NSEG, NSPR, NPART, NTEX, SHAPE_TEXTURE, FLOOR_TEXTURE, Q, VERT_FS, SCENE_FS, SPRITE_VS, SPRITE_FS,
                                     PART_VS, PART_FS, BRIGHT_FS, BLUR_FS, COMP_FS)
from seamjax.app.snapshot import hunter_shown


def rgb(h, s, l): return np.array(colorsys.hls_to_rgb((h % 360)/360., l, s), 'f4')

SHAPE = {name: i for i, name in enumerate(themes.SHAPES)}
TEX_SPRITE, TEX_FLOOR = 4, 3                            # texture units of the sprite atlas and the floor image


def _image(path):
    """an image file -> (w, h, RGBA bytes, first row = top)"""
    import pygame
    im = pygame.image.load(themes.asset(path))
    return im.get_width(), im.get_height(), pygame.image.tobytes(im, 'RGBA')


class GLView:
    def __init__(self, VW, VH, theme=None):
        self.VW, self.VH = VW, VH; self.theme = theme or themes.THEME
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
        self._load_textures()

    def _load_textures(self):
        """the theme's sprite images, packed in one atlas (a row of equal cells), and its floor images"""
        ctx = self.ctx; sprites, floors = self.theme.textures(); assert len(sprites) <= NTEX, 'at most %d sprite textures' % NTEX
        ims = [_image(p) for p in sprites]; cw = max([w for w, h, _ in ims] or [1]); chh = max([h for w, h, _ in ims] or [1])
        atlas = np.zeros((chh, cw*max(1, len(ims)), 4), np.uint8); uv = np.zeros((NTEX, 4), 'f4'); self.tex_index = {}
        for i, (p, (w, h, data)) in enumerate(zip(sprites, ims)):
            atlas[:h, i*cw:i*cw+w] = np.frombuffer(data, np.uint8).reshape(h, w, 4)
            uv[i] = [i*cw/atlas.shape[1], 0., (i*cw+w)/atlas.shape[1], h/chh]; self.tex_index[p] = i
        self.atlas = ctx.texture((atlas.shape[1], atlas.shape[0]), 4, atlas.tobytes()); self.atlas.repeat_x = self.atlas.repeat_y = False
        self.atlas.filter = (moderngl.LINEAR, moderngl.LINEAR); self._arr(self.p_spr, 'u_uv', uv)
        self.floor_tex = {}
        for p in floors:
            w, h, data = _image(p); t = ctx.texture((w, h), 4, data); t.build_mipmaps(); t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self.floor_tex[p] = t
        self.blank = ctx.texture((1, 1), 4, bytes([255, 255, 255, 255]))

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
        W = self.theme.walls; n = c.n; V = np.array([M(*v) for v in c.corners]); hue = c.hue; hd = int(sn.hunter_door); wallc = rgb(hue, *W.wall); segs = []
        for k in range(n):
            A, B = V[(k-1) % n], V[k]
            if c.doors[k] >= 0:
                (tx, ty), (dx, dy) = c.sides.tangent[k], c.sides.door[k]
                g0 = np.array(M(dx-tx*DOOR_GAP, dy-ty*DOOR_GAP)); g1 = np.array(M(dx+tx*DOOR_GAP, dy+ty*DOOR_GAP))
                near, far = (A, B) if np.dot(A-g0, A-g0) < np.dot(B-g0, B-g0) else (B, A)
                rw = bool(sn.roamer_doors[k]); pulse = .55+.45*math.sin(t*8)
                col = np.array(W.hunter_door)*pulse if k == hd else np.array(W.roamer_door)*pulse if rw else np.array(W.door_other_side) if (c.wall_kind > 0 and c.door_sec[k] > 0) else np.array(W.door)
                segs += [(near, g0, wallc, 4.), (g1, far, wallc, 4.), (g0, g1, col, 8. if k == hd or rw else 5.)]
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
            W = self.theme.walls
            col = np.array(W.gated_wall if c.wall_kind == WALL_GATED else W.solid_wall) if wall else rgb(c.hue, *W.obstacle)
            if ty == OBST_CIRCLE: ctr = M(px, py); A[m] = [*ctr, w, 0]; B[m] = [0, 1, 0, 0]
            else: ctr = M(px+w/2, py+h/2); A[m] = [*ctr, w/2, h/2]; B[m] = [1, cs, sn_, 0]
            C[m] = [*col, 1. if wall else 0.]; m += 1
        return m, A, B, C

    def _sprites(self, sn, c, M, va, mir, t):
        """one instance row per sprite: centre, radius, quad half-size, shape (or texture), colour + brightness, angle, width"""
        T = self.theme; L = T.looks; rows = []; r = int(sn.room)
        def add(look, p, rad, inten=1., ext=None, ang=0., colour=None, ln=None):
            kind, ln = (SHAPE_TEXTURE, self.tex_index[look.texture]) if look.texture else (SHAPE[look.shape], look.width if ln is None else ln)
            ext = ext if ext is not None else look.extent if look.extent is not None else rad*3.2+8
            col = colour or look.colour
            rows.append([p[0], p[1], rad, ext, kind, col[0], col[1], col[2], look.glow*inten, ang+t*look.spin, ln])
        en = sn.enemies; alive = np.asarray(en.alive)
        if r == int(sn.exit):                                                       # the exit pad: grey until the room is clear
            open_ = not alive.any(); pc = M(*c.pad); pulse = .8+.2*math.sin(t*4)
            if open_: add(L['pad'], pc, L['pad'].size, pulse); add(L['pad_glow'], pc, L['pad_glow'].size, pulse)
            else: add(L['pad_closed'], pc, L['pad_closed'].size)
        pk = sn.pickups; lk = L['pickup']
        for i in np.where(pk.alive)[0]: add(lk, M(pk.x[i], pk.y[i]), lk.size*(1+lk.pulse*math.sin(t*4+i)))
        for i in np.where(alive)[0]:
            p = M(en.x[i], en.y[i]); kind = kinds.ENEMY_KINDS[int(en.kind[i])]; look = T.enemy(kind.name).look
            rad = look.size or kind.radius; tg = float(en.telegraph[i])
            if tg > 0: add(L['telegraph'], p, rad+4*tg, ext=rad*3+10)
            else: add(look, p, rad, colour=T.flash_colour if en.flash[i] > 0 else None)
        pc = kinds.PLAYER_CLASSES[int(sn.player.cls)]; wpn = pc.weapon
        for bl, lk in ((sn.pb, T.weapon(wpn.name)), (sn.eb, L['enemy_bullet'])):
            for i in np.where(bl.alive)[0]:
                vx, vy = view_np(va, mir, bl.vx[i], bl.vy[i]); add(lk, M(bl.x[i], bl.y[i]), lk.size, ang=math.atan2(-vy, -vx))
        if hunter_shown(sn):
            lk = L['hunter']; add(lk, M(float(sn.hunter.x), float(sn.hunter.y)), lk.size or HUNTER_R*1.2, ext=lk.extent or HUNTER_R*4,
                                  colour=T.flash_colour if float(sn.hunter.flash) > 0 else None)
        px = M(float(sn.player.x), float(sn.player.y))
        if not (float(sn.player.inv) > 0 and int(t*20) % 2):                         # blinks while invulnerable
            lk = T.player(pc.name); add(lk, px, lk.size or PLAYER_R)
        sw = float(sn.player.swing)
        if wpn.melee and sw > 0:                                                    # a melee swing: a fan of streaks over the arc
            lk = T.weapon(wpn.name); aim = float(sn.player.aim)
            for j in range(11):                                                     # bright at the edge of the swing, fading inward
                a = aim + wpn.arc*(j/10-.5); dx, dy = view_np(va, mir, math.cos(a), math.sin(a))
                add(lk, (px[0]+dx*wpn.reach, px[1]+dy*wpn.reach), lk.size or 2., sw/SWING_SHOW, wpn.reach*.3+8, math.atan2(-dy, -dx), ln=wpn.reach*.25)
        ang = float(sn.player.aim); ax, ay = view_np(va, mir, math.cos(ang), math.sin(ang)); add(L['aim'], px, L['aim'].size, ang=math.atan2(ay, ax))
        return np.array(rows[:NSPR], 'f4').reshape(-1, 11), px

    # ---- one frame of the room (everything except the HUD)
    def scene(self, sn, c, M, va, mir, t, world, fx, now):
        """world: the WorldLook of this floor (seamjax/theme.py): its floor pattern or image"""
        import pygame
        ctx = self.ctx; self.resize(pygame.display.get_window_size())
        RW, RH = self.size; res = (self.VW, self.VH)
        n, vt, wind, ns, sa, sc = self._geometry(sn, c, M, t); no, oA, oB, oC = self._obstacles(sn, c, M, va, mir); spr, ppos = self._sprites(sn, c, M, va, mir, t)
        self.f_scene.use(); ctx.viewport = (0, 0, RW, RH); ctx.disable(moderngl.BLEND)
        style = FLOOR_TEXTURE if world.texture else themes.FLOORS.index(world.floor)
        (self.floor_tex[world.texture] if world.texture else self.blank).use(TEX_FLOOR); self.atlas.use(TEX_SPRITE)
        P = self.p_scene; self._set(P, u_floor=TEX_FLOOR, u_fscale=float(world.texture_scale)); self._set(self.p_spr, u_tex=TEX_SPRITE)
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
