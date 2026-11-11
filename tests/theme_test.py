"""The theme (seamjax/theme.py) drives looks and sounds, procedural or from files. This makes a texture and a sound file,
points a theme at them, and checks that the OpenGL view (sprite atlas, textured floor), the software renderer, the
particles and the sound player all use them. The OpenGL part skips without an OpenGL 3.3 display."""
import _setup, os, sys, math, tempfile, wave
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import numpy as np, pygame
from seamjax import theme as T
from seamjax.core.types import Enemies, Sides, Obstacles
from seamjax.app.snapshot import RoomDraw
from _fakes import pool, snapshot

tmp = tempfile.mkdtemp()
# a texture: a white cross on transparent; a floor: a checkerboard
pygame.init()
s = pygame.Surface((32, 32), pygame.SRCALPHA); s.fill((0, 0, 0, 0)); pygame.draw.rect(s, (255, 255, 255, 255), (12, 0, 8, 32)); pygame.draw.rect(s, (255, 255, 255, 255), (0, 12, 32, 8))
pygame.image.save(s, os.path.join(tmp, 'cross.png'))
f = pygame.Surface((16, 16)); f.fill((40, 40, 40)); pygame.draw.rect(f, (200, 200, 200), (0, 0, 8, 8)); pygame.draw.rect(f, (200, 200, 200), (8, 8, 8, 8))
pygame.image.save(f, os.path.join(tmp, 'checker.png'))
# a sound file: 0.1 s of 440 Hz
with wave.open(os.path.join(tmp, 'beep.wav'), 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
    w.writeframes((np.sin(2*np.pi*440*np.arange(2205)/22050)*12000).astype(np.int16).tobytes())

base = T.THEME
cross, checker, beep = os.path.join(tmp, 'cross.png'), os.path.join(tmp, 'checker.png'), os.path.join(tmp, 'beep.wav')
looks = dict(base.looks, player=T.Look((1., 1., 1.), texture=cross, size=20))
enemies = dict(base.enemies, shooter=T.EnemyLook(T.Look((.2, 1., .2), texture=cross), base.enemies['shooter'].kill))
events = dict(base.events, shoot=T.Effect(T.Sound(file=beep), (T.Burst(4, (0., 1., 0.), 130, .22, 3),)))
themed = base.replace(looks=looks, enemies=enemies, events=events, worlds=dict(base.worlds, flat=T.WorldLook(texture=checker, texture_scale=64.)))
assert themed.textures() == ([cross], [checker])
assert T.Sound(file=beep) in themed.sounds()[0] and T.Sound(synth='kill_shooter') in themed.sounds()[0]

# a kind with no look of its own falls back to a grey disk
from seamjax.core import kinds
assert base.enemy('nobody') is T.DEFAULT_ENEMY

# a room with the player and a shooter in it
n = 6; R = 230.; V = np.array([[480+R*math.sin(2*math.pi*i/n), 300-R*math.cos(2*math.pi*i/n)] for i in range(n)])
mid, nrm, tan = np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 2))
for k in range(n):
    A, B = V[(k-1) % n], V[k]; mid[k] = (A+B)/2; tan[k] = (B-A)/np.linalg.norm(B-A); nrm[k] = [tan[k, 1], -tan[k, 0]]
room = RoomDraw(n=n, sides=Sides(mid=mid, normal=nrm, tangent=tan, door=mid.copy()), doors=np.full(n, -1), corners=V,
                obstacles=Obstacles(kind=np.full(6, -1), pos=np.zeros((6, 2)), size=np.zeros((6, 2)), wall=np.zeros(6, bool)), wall_axis=0, wall_kind=0,
                wall_pos=0., door_sec=np.zeros(n, int), pad=np.array([480., 300.]), spawn=np.zeros((16, 2)), spawn_ok=np.zeros(16, int), spawn_sec=np.zeros(16, int), hue=200.)
sn = snapshot(enemies=pool(Enemies, 10, [dict(x=600, y=300, kind=1, alive=True)]), player=dict(x=400., y=300.))
M = lambda x, y: (x, y)

# software renderer: the textured look is drawn as the image
screen = pygame.Surface((960, 600))
from seamjax.app.software import draw_room
draw_room(screen, sn, room, M, 0., themed); a = pygame.surfarray.array3d(screen)
assert (a[400, 290] > 200).all() and (a[388, 290] < 200).any(), 'the player is the cross image (white arm, dark corner), not a disk'
draw_room(screen, sn, room, M, 0., base); b = pygame.surfarray.array3d(screen)
assert np.abs(a.astype(int)-b.astype(int)).sum() > 0, 'the themed software render differs'
print('software renderer: textured player and shooter drawn')

# particles: the themed shoot burst is green
from seamjax.app.gl.fx import Fx
fx = Fx(); fx.apply([dict(name='shoot', x=0., y=0.)], M, 0., themed); live = fx.part[fx.part[:, 5] > 0]
assert len(live) == 4 and np.allclose(live[:, 6:9], [0., 1., 0.]), 'the theme sets the burst'

# sound: the file is loaded and played, the generated sounds are synthesised
from seamjax.app.audio import Audio
au = Audio(theme=themed)
if au.ok:
    import time
    for _ in range(600):
        if au.ready or au.err: break
        time.sleep(.1)
    assert au.ready, au.err
    assert len(au.shots[T.Sound(file=beep)]) == 1 and len(au.shots[T.Sound(synth='kill_shooter')]) == 3
    au.events([dict(name='shoot', x=0., y=0.), dict(name='kill', x=0., y=0., kind=1)], lambda x, y: 0.)
    au.set_world('flat'); au.set_hunter(2)
    print('sound: file and generated sounds loaded and played')
else: print('sound: no audio device, skipped (%s)' % au.err)

# OpenGL: the atlas holds the sprite texture, the floor is the image, and both change the picture
try:
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3); pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((960, 600), pygame.OPENGL | pygame.DOUBLEBUF)
    from seamjax.app.gl.view import GLView
    gl0, gl1 = GLView(960., 600., base), GLView(960., 600., themed)
except Exception as ex:
    print('OpenGL: SKIP (%s)' % ex); print('PASS'); sys.exit(0)
assert gl1.tex_index == {cross: 0} and checker in gl1.floor_tex
fx = Fx(); shots = []
for gl, world in ((gl0, base.worlds['flat']), (gl1, themed.worlds['flat'])):
    gl.resize((960, 600)); gl.scene(sn, room, M, 0., 1., 1., world, fx, 0.); shots.append(pygame.surfarray.array3d(gl.screenshot()).astype(int))
if len(sys.argv) > 1: pygame.image.save(gl1.screenshot(), sys.argv[1])
d = np.abs(shots[0]-shots[1]).sum(2)
assert d[380:420, 280:320].sum() > 1000, 'textured player'
assert d[580:620, 280:320].sum() > 1000, 'textured shooter'
assert d[300:340, 150:200].sum() > 1000, 'textured floor'
print('OpenGL: textured sprites and floor drawn')
print('PASS')
