"""GLSL sources of the OpenGL view (version 330) and the buffer sizes they are written for."""
import numpy as np

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
