"""OpenGL view smoke test: draws a synthetic room with enemies, bullets, a pickup, the Hunter and the exit pad, then fires
the event effects (kill, hit).  Needs a display with OpenGL 3.3 (run under `xvfb-run -a` on a headless machine); skips otherwise.
    python tests/gl_test.py [out.png]"""
import _setup, os, sys, math
import numpy as np, pygame
out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/gl_test.png'
pygame.init()
try:
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3); pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((960, 600), pygame.OPENGL | pygame.DOUBLEBUF)
    from seamjax.app.gl.view import GLView
    from seamjax.app.gl.canvas import GLCanvas
    from seamjax.app.gl.fx import Fx
    gl = GLView(960., 600.)
except Exception as ex:
    print('SKIP (no OpenGL 3.3 here: %s)' % ex); sys.exit(0)
n = 6; R = 230.; V = np.array([[480+R*math.sin(2*math.pi*i/n), 300-R*math.cos(2*math.pi*i/n)] for i in range(n)])
from seamjax.core.types import Sides, Obstacles, Enemies, Pickups, Bullets
from seamjax.app.snapshot import RoomDraw
from _fakes import pool, snapshot
mid, nrm, tan = np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 2))
for k in range(n):
    A, B = V[(k-1) % n], V[k]; mid[k] = (A+B)/2; tan[k] = (B-A)/np.linalg.norm(B-A); nrm[k] = [tan[k, 1], -tan[k, 0]]
doors = np.full(n, -1); doors[0] = 3; doors[3] = 5
ob = Obstacles(kind=np.array([1, 0, -1, -1, -1, -1]), pos=np.array([[380, 330], [600, 250]]+[[0, 0]]*4, float),
               size=np.array([[70, 40], [24, 0]]+[[0, 0]]*4, float), wall=np.zeros(6, bool))
c = RoomDraw(n=n, sides=Sides(mid=mid, normal=nrm, tangent=tan, door=mid.copy()), doors=doors, corners=V, obstacles=ob, wall_axis=0, wall_kind=0,
             wall_pos=0., door_sec=np.zeros(n, int), pad=np.array([480., 300.]), spawn=np.zeros((16, 2)), spawn_ok=np.zeros(16, int),
             spawn_sec=np.zeros(16, int), hue=200.)
en = pool(Enemies, 10, [dict(x=350, y=250, kind=1, alive=True), dict(x=560, y=330, alive=True), dict(x=450, y=200, telegraph=.6, alive=True),
                        dict(x=520, y=400, flash=1, alive=True)])
pk = pool(Pickups, 6, [dict(x=400, y=380, alive=True)])
pb = pool(Bullets, 24, [dict(x=430, y=290, vx=380, vy=20, life=1, alive=True)]); eb = pool(Bullets, 24, [dict(x=380, y=270, vx=100, vy=60, life=3, alive=True)])
sn = snapshot(enemies=en, pickups=pk, pb=pb, eb=eb, hunter=dict(x=300., y=330., here=True, hp=12.), hunter_door=3)
M = lambda x, y: (x, y); fx = Fx(); gl.resize((960, 600))
from seamjax.app.events import Events
ev = Events(); ev.update(sn, (0, 0), 0, False)
sn2 = sn.replace(enemies=en.replace(alive=np.where(np.arange(10) == 0, False, en.alive)), player=sn.player.replace(hp=4)); fx.apply(ev.update(sn2, (0, 0), 0, False), M, .1)        # an enemy dies and the player is hit
assert (fx.part[:, 5] > 0).sum() > 20 and fx.hit > .5 and fx.shake > .5, 'events did not make effects'
bright = {}
from seamjax import theme
for style, floor in enumerate(theme.FLOORS):
    gl.scene(sn2, c, M, 0., 1., 1.0, theme.WorldLook(floor), fx, .2); img = gl.screenshot(); a = pygame.surfarray.array3d(img).astype(float)
    bright[style] = a.mean(); assert a.max() > 200 and a.std() > 10, 'style %d looks blank' % style
    if style == 1: pygame.image.save(img, out)
mirror = gl.scene(sn2, c, M, .7, -1., 1.0, theme.WorldLook('grid'), fx, .2); gl.screenshot()                                   # mirrored, rotated view must not crash
cv = GLCanvas(gl); gl.scene(sn2, c, M, 0., 1., 1.0, theme.WorldLook('grid'), fx, .2); before = pygame.surfarray.array3d(gl.screenshot()).astype(int)
cv.begin(); cv.text('Floor 7  kills 12 -> ok', 20, 20, 22, (255, 255, 255)); cv.text('right', 940, 40, 56, (255, 120, 140), 'r'); cv.rect(20, 560, 200, 20, (255, 92, 115), 6)
cv.ring(800, 500, 40, (140, 255, 215), 2); cv.ngon(700, 500, 30, 7, (255, 255, 255), 1); cv.line((600, 450), (680, 560), (255, 200, 90), 3); cv.disk(500, 560, 8, (90, 150, 255, 120)); cv.end()
after = pygame.surfarray.array3d(gl.screenshot()).astype(int); diff = np.abs(after-before).sum(2)
assert diff[15:260, 15:60].sum() > 5000, 'text did not draw'; assert diff[20:220, 555:585].max() > 150, 'rect did not draw'; assert diff[760:840, 470:530].sum() > 1000, 'ring did not draw'
pygame.image.save(gl.screenshot(), out.replace('.png', '_hud.png'))
print('GL view PASS  mean brightness per world style:', {k: round(v, 1) for k, v in bright.items()}, '->', out)
