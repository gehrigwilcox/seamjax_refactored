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
`Fx` turns game events (see events.py: a kill, a hit, a pickup...) into particles, shake and flashes.
"""
import math, colorsys
import numpy as np
import moderngl
from seamjax.core.config import CX, CY, DOOR_GAP, PLAYER_R, HUNTER_R, CHASER_R, SHOOTER_R
from seamjax.core.types import OBST_NONE, OBST_CIRCLE, WALL_GATED

NSEG, NSPR, NPART = 40, 1024, 2048
Q = np.array([-1, -1, 1, -1, -1, 1, 1, 1], 'f4')          # unit quad (triangle strip)

VERT_FS = '''#version 330
in vec2 in_pos; out vec2 v_uv;
void main(){ v_uv = in_pos*.5+.5; gl_Position = vec4(in_pos, 0., 1.); }'''

COMMON = '''
vec3 hsl(float h, float s, float l){ vec3 k = mod(vec3(0., 8., 4.)+h/30., 12.); float a = s*min(l, 1.-l); return l-a*max(min(min(k-3., 9.-k), 1.), -1.); }
float hash(vec2 p){ p = fract(p*vec2(123.34, 456.21)); p += dot(p, p+45.32); return fract(p.x*p.y); }
float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.-2.*f);
  return mix(mix(hash(i), hash(i+vec2(1, 0)), f.x), mix(hash(i+vec2(0, 1)), hash(i+vec2(1, 1)), f.x), f.y); }
float fbm(vec2 p){ float a = .5, s = 0.; for(int i = 0; i < 4; i++){ s += a*vnoise(p); p *= 2.03; a *= .5; } return s; }
'''

SCENE_FS = '''#version 330
in vec2 v_uv; out vec4 o;
uniform vec2 u_res, u_ctr, u_player; uniform float u_time, u_hue, u_va, u_mir; uniform int u_style;
uniform int u_nv; uniform vec2 u_vtx[12]; uniform float u_wind;
uniform int u_ns; uniform vec4 u_segA[40]; uniform vec4 u_segC[40];
uniform int u_no; uniform vec4 u_obsA[6]; uniform vec4 u_obsB[6]; uniform vec4 u_obsC[6];
''' + COMMON + '''
float dseg(vec2 p, vec2 a, vec2 b){ vec2 pa = p-a, ba = b-a; return length(pa-ba*clamp(dot(pa, ba)/max(dot(ba, ba), 1e-6), 0., 1.)); }
float polyd(vec2 p){ float d = -1e9;
  for(int i = 0; i < u_nv; i++){ vec2 a = u_vtx[i], b = u_vtx[(i+1)%u_nv], e = b-a; d = max(d, dot(p-a, normalize(vec2(e.y, -e.x))*u_wind)); } return d; }
void main(){
  vec2 p = vec2(v_uv.x, 1.-v_uv.y)*u_res;
  float c = cos(u_va), s = sin(u_va); vec2 q = p-u_ctr;                       // back to world coordinates (undo the view)
  vec2 wp = vec2(u_mir*(c*q.x+s*q.y), -s*q.x+c*q.y)+u_ctr;
  float n = fbm(wp/90.+vec2(u_time*.02, -u_time*.015));
  vec3 base = hsl(u_hue, .35, .10)*(.75+.6*n), tint = hsl(u_hue, .45, .55);
  float line = 0.; vec2 rel = wp-u_ctr;
  if(u_style == 0){ vec2 g = abs(fract(wp/44.-.5)-.5)*44.; line = 1.-smoothstep(0., 1.3, min(g.x, g.y)); }
  else if(u_style == 1){ float r = length(rel); float ring = abs(fract(r/46.)-.5)*46.; float a = atan(rel.y, rel.x);
    float sp = abs(fract(a/.5235988+.5)-.5)*r*.5235988; line = 1.-smoothstep(0., 1.3, min(ring, r > 40. ? sp : 9.)); }
  else if(u_style == 2){ vec2 g = fract(wp/44.-.5)-.5; float dd = length(g)*44.; vec2 g2 = abs(g)*44.;
    line = max(1.-smoothstep(1.4, 3.4, dd), .35*(1.-smoothstep(0., 1.1, min(g2.x, g2.y)))); }
  else { float d1 = abs(fract((wp.x+wp.y)/62.-.5)-.5)*62./1.414, d2 = abs(fract((wp.x-wp.y)/62.-.5)-.5)*62./1.414;
    line = (1.-smoothstep(0., 1.2, min(d1, d2)))*(.55+.45*sin((wp.x+wp.y)*.02-u_time*1.3)); }
  vec3 fl = base+tint*line*.16;
  fl *= 1.-.28*smoothstep(0., 340., length(p-u_ctr));
  fl += tint*.10*exp(-length(p-u_player)/170.);                                // light pool that follows the player
  vec3 outside = vec3(.016, .02, .028)+.01*n;
  float dp = polyd(p);
  vec3 col = mix(fl, outside, smoothstep(-1., 1., dp));
  col *= 1.-.35*exp(dp/18.)*step(dp, 0.);                                       // soft shade along the inside of the walls
  for(int i = 0; i < u_no; i++){
    vec4 A = u_obsA[i], B = u_obsB[i], C = u_obsC[i]; vec2 d0 = p-A.xy; float d;
    if(B.x < .5) d = length(d0)-A.z;
    else { vec2 l = vec2(dot(d0, vec2(B.y, B.z)), dot(d0, vec2(-B.z, B.y))); vec2 w = abs(l)-A.zw; d = length(max(w, 0.))+min(max(w.x, w.y), 0.); }
    bool wall = C.a > .5; vec3 fc = C.rgb*(wall ? .55 : 1.);
    if(wall) fc *= .85+.15*sin((p.x+p.y)*.5+u_time*2.);
    col = mix(col, fc, smoothstep(1., -1., d)) + C.rgb*(exp(-abs(d)/2.)*(wall ? .9 : .3)+(wall ? .25*exp(-max(d, 0.)/10.) : 0.));
  }
  for(int i = 0; i < u_ns; i++){
    vec4 A = u_segA[i], C = u_segC[i]; float d = dseg(p, A.xy, A.zw);
    col = mix(col, C.rgb, smoothstep(C.a*.5+.7, C.a*.5-.3, d)) + C.rgb*exp(-d/7.)*.3;
  }
  o = vec4(col, 1.);
}'''

SPRITE_VS = '''#version 330
in vec2 in_corner; in vec2 in_c; in float in_r; in float in_ext; in float in_kind; in vec4 in_col; in float in_ang; in float in_len;
uniform vec2 u_res; out vec2 v_l; flat out float f_r, f_kind, f_ang, f_len; flat out vec4 f_col;
void main(){ v_l = in_corner*in_ext; vec2 p = in_c+v_l; gl_Position = vec4(p.x/u_res.x*2.-1., 1.-p.y/u_res.y*2., 0., 1.);
  f_r = in_r; f_kind = in_kind; f_col = in_col; f_ang = in_ang; f_len = in_len; }'''

SPRITE_FS = '''#version 330
in vec2 v_l; flat in float f_r, f_kind, f_ang, f_len; flat in vec4 f_col; out vec4 o;
float dot2(vec2 v){ return dot(v, v); }
float sdPoly(vec2 p, float r, float n){ float an = 3.14159265/n; vec2 acs = vec2(cos(an), sin(an)); float bn = mod(atan(p.x, p.y), 2.*an)-an;
  p = length(p)*vec2(cos(bn), abs(sin(bn))); p -= r*acs; p.y += clamp(-p.y, 0., r*acs.y); return length(p)*sign(p.x); }
float sdHeart(vec2 p){ p.x = abs(p.x); if(p.y+p.x > 1.) return sqrt(dot2(p-vec2(.25, .75)))-sqrt(2.)/4.;
  return sqrt(min(dot2(p-vec2(0., 1.)), dot2(p-.5*max(p.x+p.y, 0.))))*sign(p.x-p.y); }
void main(){
  int k = int(f_kind+.5); float r = f_r, d, shade = 1.; vec2 l = v_l;
  if(k == 0){ d = length(l)-r; shade = mix(.4, 1., smoothstep(-r, 0., d)); }
  else if(k == 1){ d = abs(length(l)-r)-f_len*.5; }
  else if(k == 2){ float c = cos(f_ang), s = sin(f_ang); vec2 m = vec2(c*l.x+s*l.y, -s*l.x+c*l.y); d = sdPoly(vec2(m.x, -m.y), r, 5.); shade = .35+.9*smoothstep(-4., -1., d); }
  else if(k == 3){ float sc = 2.3*r; d = sdHeart(vec2(l.x, -l.y)/sc+vec2(0., .5))*sc; }
  else { vec2 dv = vec2(cos(f_ang), sin(f_ang)); float a = dot(l, dv), b = dot(l, vec2(-dv.y, dv.x));
         d = length(vec2(a-clamp(a, 0., f_len), b))-r; shade = mix(1., .12, clamp(a/max(f_len, 1.), 0., 1.)); }
  float fill = smoothstep(1., -1., d), glow = exp(-max(d, 0.)/(r*.6+3.))*.5;
  o = vec4(f_col.rgb*(shade*fill+glow)*f_col.a, 1.);
}'''

PART_VS = '''#version 330
in vec2 in_corner; in vec2 in_p0; in vec2 in_v; in float in_t0; in float in_life; in vec4 in_col; in float in_size;
uniform vec2 u_res; uniform float u_now; out vec2 v_c; out vec3 v_col; out float v_a;
void main(){
  float age = u_now-in_t0;
  if(in_life <= 0. || age < 0. || age > in_life){ gl_Position = vec4(3., 3., 3., 1.); v_c = vec2(0.); v_col = vec3(0.); v_a = 0.; return; }
  float k = 2.8; vec2 p = in_p0+in_v*(1.-exp(-k*age))/k; float a = 1.-age/in_life; float sz = in_size*(.35+.65*a)*2.;
  p += in_corner*sz; gl_Position = vec4(p.x/u_res.x*2.-1., 1.-p.y/u_res.y*2., 0., 1.); v_c = in_corner; v_col = in_col.rgb*in_col.a; v_a = a;
}'''

PART_FS = '''#version 330
in vec2 v_c; in vec3 v_col; in float v_a; out vec4 o;
void main(){ float d = length(v_c); float g = smoothstep(1., 0., d); o = vec4(v_col*(g*g+.5*smoothstep(.45, 0., d))*v_a*v_a*1.4, 1.); }'''

BRIGHT_FS = '''#version 330
in vec2 v_uv; out vec4 o; uniform sampler2D u_src;
void main(){ vec3 c = texture(u_src, v_uv).rgb; float l = max(max(c.r, c.g), c.b); o = vec4(c*smoothstep(.55, 1.1, l), 1.); }'''

BLUR_FS = '''#version 330
in vec2 v_uv; out vec4 o; uniform sampler2D u_src; uniform vec2 u_dir;
void main(){ vec3 c = texture(u_src, v_uv).rgb*.227027; float w[4] = float[](.1945946, .1216216, .054054, .016216);
  for(int i = 0; i < 4; i++){ vec2 off = u_dir*float(i+1)*1.4; c += (texture(u_src, v_uv+off).rgb+texture(u_src, v_uv-off).rgb)*w[i]; } o = vec4(c, 1.); }'''

COMP_FS = '''#version 330
in vec2 v_uv; out vec4 o; uniform sampler2D u_scene, u_b1, u_b2; uniform vec2 u_shake; uniform float u_hit, u_flash, u_fade;
void main(){
  vec2 uv = v_uv+u_shake; float ab = u_hit*.004;
  vec3 col = vec3(texture(u_scene, uv+vec2(ab, 0.)).r, texture(u_scene, uv).g, texture(u_scene, uv-vec2(ab, 0.)).b);
  col += texture(u_b1, uv).rgb*.45+texture(u_b2, uv).rgb*.6;
  col *= mix(.55, 1., smoothstep(1.2, .3, length((v_uv-.5)*vec2(1.25, 1.))));
  col = mix(col, col*vec3(1.5, .45, .45)+vec3(.22, 0., 0.), u_hit*.7)+vec3(1., .95, .9)*u_flash*.25;
  col *= 1.-u_fade; col = col/(1.+.25*col)*1.25; o = vec4(pow(clamp(col, 0., 1.), vec3(.92)), 1.);
}'''

HUD_FS = '''#version 330
in vec2 v_l; flat in float f_r, f_kind, f_ang, f_len; flat in vec4 f_col; out vec4 o;
float sdPoly(vec2 p, float r, float n){ float an = 3.14159265/n; vec2 acs = vec2(cos(an), sin(an)); float bn = mod(atan(p.x, p.y), 2.*an)-an;
  p = length(p)*vec2(cos(bn), abs(sin(bn))); p -= r*acs; p.y += clamp(-p.y, 0., r*acs.y); return length(p)*sign(p.x); }
void main(){
  int k = int(f_kind+.5); vec2 l = v_l; float d, cover;
  if(k == 0){ d = length(l)-f_r; cover = smoothstep(.8, -.8, d); }                                       // disk
  else if(k == 1){ d = abs(length(l)-f_r)-f_len*.5; cover = smoothstep(.8, -.8, d); }                   // ring (width f_len)
  else if(k == 2){ d = sdPoly(vec2(l.x, -l.y), f_r, f_len); cover = smoothstep(.8, -.8, d); }            // n-gon fill (n = f_len)
  else if(k == 3){ d = sdPoly(vec2(l.x, -l.y), f_r, f_len); cover = smoothstep(.8, -.8, abs(d+.7)-.7); } // n-gon outline
  else if(k == 5){ vec2 dv = vec2(cos(f_ang), sin(f_ang)); float a = dot(l, dv), b = dot(l, vec2(-dv.y, dv.x));
                   d = length(vec2(a-clamp(a, 0., f_len), b))-f_r; cover = smoothstep(.8, -.8, d); }       // line from the centre along f_ang, length f_len, half-width f_r
  else { vec2 w = abs(l)-vec2(f_r, f_len)+f_ang; d = length(max(w, 0.))+min(max(w.x, w.y), 0.)-f_ang; cover = smoothstep(.8, -.8, d); }   // rounded box (half sizes f_r, f_len; corner f_ang)
  o = vec4(f_col.rgb, f_col.a*cover);
}'''

TEXT_VS = '''#version 330
in vec2 in_corner; in vec4 in_rect; in vec4 in_uv; in vec4 in_col; uniform vec2 u_res; out vec2 v_uv; out vec4 v_col;
void main(){ vec2 c = in_corner*.5+.5; vec2 p = in_rect.xy+c*in_rect.zw; gl_Position = vec4(p.x/u_res.x*2.-1., 1.-p.y/u_res.y*2., 0., 1.);
  v_uv = mix(in_uv.xy, in_uv.zw, c); v_col = in_col; }'''

TEXT_FS = '''#version 330
in vec2 v_uv; in vec4 v_col; out vec4 o; uniform sampler2D u_atlas; void main(){ o = vec4(v_col.rgb, v_col.a*texture(u_atlas, v_uv).a); }'''


def rgb(h, s, l): return np.array(colorsys.hls_to_rgb((h % 360)/360., l, s), 'f4')


class Fx:
    """Game events -> particles, shake and flashes."""
    def __init__(self):
        self.part = np.zeros((NPART, 11), 'f4'); self.ring = 0; self.rng = np.random.default_rng(1)
        self.shake = 0.; self.hit = 0.; self.flash = 0.; self.fade = 0.

    def burst(self, p, n, col, speed, life, size, now, spread=1.):
        a = self.rng.uniform(0, 2*math.pi, n); sp = speed*self.rng.uniform(.25, 1., n)**.7
        rows = np.zeros((n, 11), 'f4'); rows[:, 0:2] = p; rows[:, 2] = np.cos(a)*sp*spread; rows[:, 3] = np.sin(a)*sp*spread
        rows[:, 4] = now; rows[:, 5] = life*self.rng.uniform(.6, 1.2, n); rows[:, 6:9] = col; rows[:, 9] = 1.; rows[:, 10] = size*self.rng.uniform(.6, 1.3, n)
        i = (self.ring+np.arange(n)) % NPART; self.part[i] = rows; self.ring = (self.ring+n) % NPART

    def apply(self, events, M, now):
        """turn events (see events.py) into particles, shake and flashes; M maps a world point to the screen"""
        for e in events:
            n, p = e['name'], np.array(M(e['x'], e['y']))
            if n == 'hit': self.shake = max(self.shake, .9); self.hit = 1.; self.burst(p, 18, (1., .35, .4), 190, .55, 6, now)
            elif n == 'heal': self.burst(p, 14, (1., .45, .6), 120, .7, 5, now)
            elif n == 'death': self.shake = 1.4; self.burst(p, 70, (1., .7, .75), 330, 1.1, 8, now)
            elif n == 'kill':
                self.burst(p, 26, (.71, .49, 1.) if e['shooter'] else (1., .48, .27), 250, .65, 6, now); self.burst(p, 8, (1., 1., 1.), 160, .35, 4, now); self.shake = max(self.shake, .28)
            elif n == 'hunter_hit': self.burst(p, 8, (1., .36, .45), 170, .4, 4, now)
            elif n == 'hunter_die': self.burst(p, 60, (1., .36, .45), 300, 1., 7, now); self.shake = max(self.shake, .8); self.flash = .6
            elif n == 'shoot': self.burst(p, 4, (1., .89, .48), 130, .22, 3, now, .6)
            elif n in ('descend', 'newrun'): self.fade = 1.

    @staticmethod
    def _v(sn, x, y):
        u = float(sn['mir'])*x; c, s = math.cos(float(sn['va'])), math.sin(float(sn['va'])); return c*u-s*y, s*u+c*y

    def decay(self, dt):
        self.shake *= math.exp(-dt*7.); self.hit *= math.exp(-dt*5.); self.flash *= math.exp(-dt*6.); self.fade = max(0., self.fade-dt*1.8)


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
        n = c['n']; V = np.array([M(*v) for v in c['V']]); hue = c['hue']; hd = int(sn['hdoor']); wallc = rgb(hue, .55, .58); segs = []
        for k in range(n):
            A, B = V[(k-1) % n], V[k]
            if c['doors'][k] >= 0:
                (tx, ty), (dx, dy) = c['side'].tangent[k], c['side'].door[k]
                g0 = np.array(M(dx-tx*DOOR_GAP, dy-ty*DOOR_GAP)); g1 = np.array(M(dx+tx*DOOR_GAP, dy+ty*DOOR_GAP))
                near, far = (A, B) if np.dot(A-g0, A-g0) < np.dot(B-g0, B-g0) else (B, A)
                col = np.array([1., .24, .35])*(.55+.45*math.sin(t*8)) if k == hd else np.array([1., .745, .43]) if (c['wkind'] > 0 and c['dsec'][k] > 0) else np.array([.59, .92, 1.])
                segs += [(near, g0, wallc, 4.), (g1, far, wallc, 4.), (g0, g1, col, 8. if k == hd else 5.)]
            else: segs.append((A, B, wallc, 4.))
        sa = np.zeros((NSEG, 4), 'f4'); sc = np.zeros((NSEG, 4), 'f4')
        for i, (a, b, col, w) in enumerate(segs[:NSEG]): sa[i] = [*a, *b]; sc[i] = [*col, w]
        e = V[1]-V[0]; ce = V.mean(0); wind = 1. if np.dot(ce-V[0], np.array([e[1], -e[0]])) < 0 else -1.
        vt = np.zeros((12, 2), 'f4'); vt[:n] = V
        return n, vt, wind, min(len(segs), NSEG), sa, sc

    def _obstacles(self, sn, c, M, va, mir):
        A = np.zeros((6, 4), 'f4'); B = np.zeros((6, 4), 'f4'); C = np.zeros((6, 4), 'f4'); m = 0; wopen = bool(sn['wopen']); cs, sn_ = math.cos(va), math.sin(va)
        ob = c['ob']
        for ty, (px, py), (w, h), wall in zip(ob.kind, ob.pos, ob.size, ob.wall):
            if ty == OBST_NONE or m >= 6: continue
            if wall and wopen: continue
            col = (np.array([.35, .59, 1.]) if c['wkind'] == WALL_GATED else np.array([1., .39, .62])) if wall else rgb(c['hue'], .35, .26)
            if ty == OBST_CIRCLE: ctr = M(px, py); A[m] = [*ctr, w, 0]; B[m] = [0, 1, 0, 0]
            else: ctr = M(px+w/2, py+h/2); A[m] = [*ctr, w/2, h/2]; B[m] = [1, cs, sn_, 0]
            C[m] = [*col, 1. if wall else 0.]; m += 1
        return m, A, B, C

    def _sprites(self, sn, c, M, va, mir, t):
        rows = []; r = int(sn['room'])
        def add(p, rad, kind, col, inten=1., ext=None, ang=0., ln=0.):
            rows.append([p[0], p[1], rad, ext if ext is not None else rad*3.2+8, kind, col[0], col[1], col[2], inten, ang, ln])
        en = sn['en']; alive = np.asarray(en.alive)
        if r == int(sn['exit']):
            open_ = not alive.any(); pc = M(*c['pad']); pulse = .8+.2*math.sin(t*4)
            add(pc, 18, 1, (.55, 1., .84) if open_ else (.5, .5, .5), pulse if open_ else .6, 34, 0, 3.)
            if open_: add(pc, 8, 0, (.55, 1., .84), .35*pulse, 40)
        pk = sn['pk']
        for i in np.where(pk.alive)[0]: add(M(pk.x[i], pk.y[i]), 11*(1+.08*math.sin(t*4+i)), 3, (1., .36, .45), 1.1, 40)
        for i in np.where(alive)[0]:
            p = M(en.x[i], en.y[i]); sh = bool(en.shooter[i]); rad = SHOOTER_R if sh else CHASER_R; tg = float(en.telegraph[i])
            if tg > 0: add(p, rad+4*tg, 1, (1., 1., 1.), .8, rad*3+10, 0, 1.6)
            else: add(p, rad, 0, (1., 1., 1.) if en.flash[i] > 0 else ((.71, .49, 1.) if sh else (1., .48, .27)), 1.)
        for bl, rad, col, inten, ext, ln in ((sn['pb'], 3, (1., .89, .48), 1.5, 40, 16.), (sn['eb'], 4.5, (1., .54, .82), 1.4, 36, 10.)):
            for i in np.where(bl.alive)[0]:
                vx, vy = Fx._v(sn, bl.vx[i], bl.vy[i]); add(M(bl.x[i], bl.y[i]), rad, 4, col, inten, ext, math.atan2(-vy, -vx), ln)
        if bool(sn['hhere']) and r == int(sn['hroom']) and float(sn['hdead']) <= 0:
            add(M(float(sn['hx']), float(sn['hy'])), HUNTER_R*1.2, 2, (1., 1., 1.) if float(sn['hflash']) > 0 else (1., .36, .45), 1.1, HUNTER_R*4, t*.8)
        px = M(float(sn['x']), float(sn['y']))
        if not (float(sn['inv']) > 0 and int(t*20) % 2):
            add(px, PLAYER_R, 0, (.8, .9, 1.), .85, 44)
        ang = float(sn['aim']); ax, ay = Fx._v(sn, math.cos(ang), math.sin(ang)); add(px, 1.6, 4, (1., .89, .48), 1.2, 40, math.atan2(ay, ax), 20.)
        return np.array(rows[:NSPR], 'f4').reshape(-1, 11), px

    # ---- one frame of the room (everything except the HUD)
    def scene(self, sn, c, M, va, mir, t, style, fx, now):
        import pygame
        ctx = self.ctx; self.resize(pygame.display.get_window_size())
        RW, RH = self.size; res = (self.VW, self.VH)
        n, vt, wind, ns, sa, sc = self._geometry(sn, c, M, t); no, oA, oB, oC = self._obstacles(sn, c, M, va, mir); spr, ppos = self._sprites(sn, c, M, va, mir, t)
        self.f_scene.use(); ctx.viewport = (0, 0, RW, RH); ctx.disable(moderngl.BLEND)
        P = self.p_scene
        self._set(P, u_res=res, u_ctr=(CX, CY), u_player=tuple(ppos), u_time=float(t), u_hue=float(c['hue']), u_va=float(va), u_mir=float(mir), u_style=int(style), u_nv=int(n), u_wind=float(wind), u_ns=int(ns), u_no=int(no))
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
