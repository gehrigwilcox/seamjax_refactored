"""GLCanvas: the HUD drawing interface (app/canvas.py) on the GPU: distance-field shapes and text quads from a font atlas."""
import math
import numpy as np
import moderngl
from seamjax.app.gl.shaders import SPRITE_VS, HUD_FS, TEXT_VS, TEXT_FS


class GLCanvas:
    """canvas.py interface on the GPU: shapes are instanced signed-distance quads, text is instanced glyph quads from an atlas
    (rendered once at start-up with pygame's font; every glyph after that is placed and drawn by the GPU at any size)"""
    REF, FIRST, LAST, COLS, PAD = 64, 32, 126, 16, 4

    def __init__(self, gl):
        import pygame
        self.gl = gl; ctx = gl.ctx; self.bg = None; self.shapes = []; self.texts = []
        mk = lambda vs, fs: ctx.program(vertex_shader=vs, fragment_shader=fs)
        self.p_shape, self.p_text = mk(SPRITE_VS, HUD_FS), mk(TEXT_VS, TEXT_FS)
        self.shape_buf = ctx.buffer(reserve=2048*11*4, dynamic=True); self.text_buf = ctx.buffer(reserve=4096*12*4, dynamic=True)
        self.shape_vao = ctx.vertex_array(self.p_shape, [(gl.tri, '2f', 'in_corner'), (self.shape_buf, '2f 1f 1f 1f 4f 1f 1f/i', 'in_c', 'in_r', 'in_ext', 'in_kind', 'in_col', 'in_ang', 'in_len')])
        self.text_vao = ctx.vertex_array(self.p_text, [(gl.tri, '2f', 'in_corner'), (self.text_buf, '4f 4f 4f/i', 'in_rect', 'in_uv', 'in_col')])
        font = pygame.font.Font(None, self.REF); chars = [chr(i) for i in range(self.FIRST, self.LAST+1)]
        self.adv = {c: font.size(c)[0] for c in chars}; self.cw = max(self.adv.values())+2*self.PAD; self.chh = font.get_height()+2*self.PAD
        rows = -(-len(chars)//self.COLS); self.AW, self.AH = self.cw*self.COLS, self.chh*rows
        atlas = pygame.Surface((self.AW, self.AH), pygame.SRCALPHA)
        for i, c in enumerate(chars): atlas.blit(font.render(c, True, (255, 255, 255)), ((i % self.COLS)*self.cw+self.PAD, (i//self.COLS)*self.chh+self.PAD))
        data = pygame.image.tobytes(atlas, 'RGBA') if hasattr(pygame.image, 'tobytes') else pygame.image.tostring(atlas, 'RGBA')
        self.tex = ctx.texture((self.AW, self.AH), 4, data); self.tex.build_mipmaps(); self.tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR); self.tex.repeat_x = self.tex.repeat_y = False
        self.index = {c: i for i, c in enumerate(chars)}

    @staticmethod
    def _c(col): col = tuple(col); return (col[0]/255., col[1]/255., col[2]/255., (col[3] if len(col) > 3 else 255)/255.)

    def begin(self, bg=None): self.bg = bg; self.shapes = []; self.texts = []
    def width(self, s, size=22): return sum(self.adv.get(c, 0) for c in s)*size/self.REF
    def text(self, s, x, y, size=22, col=(255, 255, 255), align='l'):
        k = size/self.REF; pen = x-(self.width(s, size)/2 if align == 'c' else self.width(s, size) if align == 'r' else 0); c = self._c(col)
        for ch in s:
            i = self.index.get(ch)
            if i is not None and ch != ' ':
                u, v = (i % self.COLS)*self.cw, (i//self.COLS)*self.chh
                self.texts.append([pen-self.PAD*k, y-self.PAD*k, self.cw*k, self.chh*k, u/self.AW, v/self.AH, (u+self.cw)/self.AW, (v+self.chh)/self.AH, *c])
            pen += self.adv.get(ch, 0)*k
    def _add(self, x, y, r, ext, kind, col, ang=0., ln=0.): self.shapes.append([x, y, r, ext, kind, *self._c(col), ang, ln])
    def disk(self, x, y, r, col): self._add(x, y, r, r+3, 0, col)
    def ring(self, x, y, r, col, width=1): self._add(x, y, r, r+width/2+3, 1, col, 0., width)
    def ngon(self, x, y, r, n, col, width=0): self._add(x, y, r, r+3, 2 if not width else 3, col, 0., float(n))
    def line(self, p0, p1, col, width=1):
        dx, dy = p1[0]-p0[0], p1[1]-p0[1]; ln = math.hypot(dx, dy); self._add(p0[0], p0[1], width/2, ln+width/2+3, 5, col, math.atan2(dy, dx), ln)
    def rect(self, x, y, w, h, col, radius=0): self._add(x+w/2, y+h/2, w/2, max(w, h)/2+3, 6, col, float(min(radius, w/2, h/2)), h/2)

    def end(self):
        """draw everything recorded since begin() over the window (after GLView.scene, or alone for a loading screen)"""
        import pygame
        gl = self.gl; ctx = gl.ctx; gl.resize(pygame.display.get_window_size()); ctx.screen.use(); ctx.viewport = (0, 0, *gl.win)
        if self.bg: ctx.screen.clear(self.bg[0]/255., self.bg[1]/255., self.bg[2]/255., 1.)
        ctx.viewport = gl.rect; ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        res = (gl.VW, gl.VH)
        if self.shapes:
            a = np.array(self.shapes[:2048], 'f4'); self.shape_buf.write(a.tobytes()); gl._set(self.p_shape, u_res=res); self.shape_vao.render(moderngl.TRIANGLE_STRIP, instances=len(a))
        if self.texts:
            a = np.array(self.texts[:4096], 'f4'); self.text_buf.write(a.tobytes()); self.tex.use(0); gl._set(self.p_text, u_res=res, u_atlas=0); self.text_vao.render(moderngl.TRIANGLE_STRIP, instances=len(a))
        ctx.disable(moderngl.BLEND)
