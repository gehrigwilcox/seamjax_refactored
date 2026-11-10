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
    from seamjax.glview import GLView, GLCanvas, Fx
    gl = GLView(960., 600., dict(CX=480., CY=300., GAP=34., PR=11., H_R=19., MAXHP=5))
except Exception as ex:
    print('SKIP (no OpenGL 3.3 here: %s)' % ex); sys.exit(0)
n = 6; R = 230.; V = np.array([[480+R*math.sin(2*math.pi*i/n), 300-R*math.cos(2*math.pi*i/n)] for i in range(n)])
side = np.zeros((12, 8))
for k in range(n):
    A, B = V[(k-1) % n], V[k]; m = (A+B)/2; tv = (B-A)/np.linalg.norm(B-A); nv = np.array([tv[1], -tv[0]])
    side[k] = [m[0], m[1], nv[0], nv[1], tv[0], tv[1], m[0], m[1]]
doors = np.full(n, -1); doors[0] = 3; doors[3] = 5
obs = np.zeros((6, 5)); ot = np.full(6, -1); obs[0] = [380, 330, 70, 40, 0]; ot[0] = 1; obs[1] = [600, 250, 24, 0, 0]; ot[1] = 0
c = dict(n=n, V=V, side=side[:n], doors=doors, obs=obs, ot=ot, wax=0, wkind=0, wpos=0., dsec=np.zeros(n, int), pad=np.array([480., 300.]), hue=200.)
en = np.zeros((10, 8)); en[0] = [350, 250, 0, 1, 0, 0, 0, 1]; en[1] = [560, 330, 0, 0, 0, 0, 0, 1]; en[2] = [450, 200, 0, 0, 0, .6, 0, 1]; en[3] = [520, 400, 0, 0, 0, 0, 1, 1]
pk = np.zeros((6, 3)); pk[0] = [400, 380, 1]; pb = np.zeros((24, 6)); pb[0] = [430, 290, 380, 20, 1, 1]; eb = np.zeros((24, 6)); eb[0] = [380, 270, 100, 60, 3, 1]
sn = dict(room=0, exit=0, hp=5, x=470., y=310., va=0., mir=1., aim=.3, inv=0., en=en, pk=pk, pb=pb, eb=eb, hx=300., hy=330., hhere=True, hroom=0, hdead=0., hflash=0., hhp=12.,
          wopen=False, hdoor=3, spec=0)
M = lambda x, y: (x, y); fx = Fx(); gl.resize((960, 600))
from seamjax.events import Events
ev = Events(); sn['pb'] = pb.copy(); sn['hdead'] = 0.; ev.update(sn, (0, 0), 0, False)
sn2 = dict(sn, en=en.copy(), hp=4); sn2['en'][0, 7] = 0; fx.apply(ev.update(sn2, (0, 0), 0, False), M, .1)        # an enemy dies and the player is hit
assert (fx.part[:, 5] > 0).sum() > 20 and fx.hit > .5 and fx.shake > .5, 'events did not make effects'
bright = {}
for style in range(4):
    gl.scene(sn2, c, M, 0., 1., 1.0, style, fx, .2); img = gl.screenshot(); a = pygame.surfarray.array3d(img).astype(float)
    bright[style] = a.mean(); assert a.max() > 200 and a.std() > 10, 'style %d looks blank' % style
    if style == 1: pygame.image.save(img, out)
mirror = gl.scene(sn2, c, M, .7, -1., 1.0, 0, fx, .2); gl.screenshot()                                   # mirrored, rotated view must not crash
cv = GLCanvas(gl); gl.scene(sn2, c, M, 0., 1., 1.0, 0, fx, .2); before = pygame.surfarray.array3d(gl.screenshot()).astype(int)
cv.begin(); cv.text('Floor 7  kills 12 -> ok', 20, 20, 22, (255, 255, 255)); cv.text('right', 940, 40, 56, (255, 120, 140), 'r'); cv.rect(20, 560, 200, 20, (255, 92, 115), 6)
cv.ring(800, 500, 40, (140, 255, 215), 2); cv.ngon(700, 500, 30, 7, (255, 255, 255), 1); cv.line((600, 450), (680, 560), (255, 200, 90), 3); cv.disk(500, 560, 8, (90, 150, 255, 120)); cv.end()
after = pygame.surfarray.array3d(gl.screenshot()).astype(int); diff = np.abs(after-before).sum(2)
assert diff[15:260, 15:60].sum() > 5000, 'text did not draw'; assert diff[20:220, 555:585].max() > 150, 'rect did not draw'; assert diff[760:840, 470:530].sum() > 1000, 'ring did not draw'
pygame.image.save(gl.screenshot(), out.replace('.png', '_hud.png'))
print('GL view PASS  mean brightness per world style:', {k: round(v, 1) for k, v in bright.items()}, '->', out)
