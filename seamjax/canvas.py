"""
The 2-D drawing interface the HUD (text, hearts, minimap, debug overlay) is written against, so the same code draws on the GPU
(glview.GLCanvas) or, with --software, on a pygame surface (PygameCanvas).  Coordinates are logical pixels (960 x 600, y down);
colours are (r, g, b) or (r, g, b, a) in 0..255.

    begin(bg=None)                       start a frame (bg clears everything first)
    text(s, x, y, size=22, col, align)   align 'l' / 'c' / 'r' about x; y is the top of the text
    width(s, size)                       text width
    rect(x, y, w, h, col, radius=0)      disk(x, y, r, col)      ring(x, y, r, col, width=1)
    line(p0, p1, col, width=1)           ngon(x, y, r, n, col, width=0)   regular n-gon, one corner up; width 0 = filled
    end()                                finish the frame
"""
import math
import pygame


class PygameCanvas:
    def __init__(self, surface): self.s = surface; self._fonts = {}
    def _font(self, size):
        if size not in self._fonts: self._fonts[size] = pygame.font.Font(None, size)
        return self._fonts[size]
    def begin(self, bg=None):
        if bg: self.s.fill(tuple(bg)[:3])
    def end(self): pass
    def width(self, s, size=22): return self._font(size).size(s)[0]
    def text(self, s, x, y, size=22, col=(255, 255, 255), align='l'):
        im = self._font(size).render(s, True, tuple(col)[:3]); w = im.get_width()
        if len(col) > 3: im.set_alpha(col[3])
        self.s.blit(im, (x-(w/2 if align == 'c' else w if align == 'r' else 0), y))
    def _over(self, col, draw):                                       # translucent shapes go through a temporary layer
        if len(col) > 3 and col[3] < 255: lay = pygame.Surface(self.s.get_size(), pygame.SRCALPHA); draw(lay, tuple(col)); self.s.blit(lay, (0, 0))
        else: draw(self.s, tuple(col)[:3])
    def rect(self, x, y, w, h, col, radius=0): self._over(col, lambda s, c: pygame.draw.rect(s, c, (x, y, w, h), border_radius=int(radius)))
    def disk(self, x, y, r, col): self._over(col, lambda s, c: pygame.draw.circle(s, c, (x, y), r))
    def ring(self, x, y, r, col, width=1): self._over(col, lambda s, c: pygame.draw.circle(s, c, (x, y), r, max(1, round(width))))
    def line(self, p0, p1, col, width=1): self._over(col, lambda s, c: pygame.draw.line(s, c, p0, p1, max(1, round(width))))
    def ngon(self, x, y, r, n, col, width=0):
        pts = [(x+r*math.sin(2*math.pi*i/n), y-r*math.cos(2*math.pi*i/n)) for i in range(n)]
        self._over(col, lambda s, c: pygame.draw.polygon(s, c, pts, max(1, round(width)) if width else 0))
