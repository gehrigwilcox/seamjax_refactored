import _setup
import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', False)
from seamjax.gen import *
from seamjax.env import *
SP=load_specs(); E=make_env(gen_source(SP))
B=128
keys=jax.random.split(jax.random.PRNGKey(5),B)
reset=jax.jit(jax.vmap(E['reset'])); S=reset(keys, jnp.tile(jnp.arange(1,9),B//8))
step=jax.jit(E['step'])
def chain(dn,ar,fl,n,r,c):
    cr,cc,d,steps=r,c,1,[]
    for _ in range(12):
        k=(cc+1)%n[cr] if d>0 else cc
        steps.append((cr,k))
        b=dn[cr,k]
        if b<0: return None
        s=ar[cr,k]; d=-d if fl[cr,k] else d; cr=b; cc=s if d>0 else (s-1)%n[b]
        if cr==r and cc==c and d==1: return steps
    return None
tested=fail=0; loops_by_cat={}
for i in range(B):
    dun={k:np.array(v[i]) for k,v in S['dun'].items()}; dn,ar,fl,n=dun['doors'],dun['arr'],dun['flip'],dun['n']
    seen=set(); s0=jax.tree_util.tree_map(lambda a:a[i],S)
    for r in range(int(dun['n_rooms'])):
        for c in range(n[r]):
            ch=chain(dn,ar,fl,n,r,c)
            if not ch: continue
            key=tuple(sorted(x[0] for x in ch))
            if key in seen: continue
            seen.add(key)
            if len(seen)>5: break
            s=dict(s0); s['room']=jnp.int32(r); s['va']=0.; s['mir']=1.
            s['inv']=100.; s['hdead']=1e9
            for (rr,k) in ch:
                assert int(s['room'])==rr
                sd=np.array(dun['side'][rr,k]); s['x']=jnp.float32(sd[6]+sd[2]*10); s['y']=jnp.float32(sd[7]+sd[3]*10)
                s,_,_=step(s,jnp.zeros(4))
            t=float(s['va'])%TAU; t=min(t,TAU-t)*180/np.pi; want=float(dun['hol'])%360; want=min(want,360-want)
            tested+=1
            ok=int(s['room'])==r and float(s['mir'])==1 and abs(t-want)<1e-2
            if not ok: fail+=1
print('generated-dungeon corner loops walked:',tested,'failed:',fail)
assert tested>100 and fail==0, 'corner loops (curvature / seam holonomy) do not match the tiling'
print('PASS')
