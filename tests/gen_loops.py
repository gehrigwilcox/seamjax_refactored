import _setup
import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', False)
from seamjax.gen import *
from seamjax.sim.game import Game; from seamjax.sim.queries import *; from seamjax.core.geometry.view import view_vec
from seamjax.sim.state import no_events; from seamjax.sim.systems.transit import transit as _transit
from seamjax.core.config import *
SP=load_specs(); E=Game(gen_source(SP))
B=128
keys=jax.random.split(jax.random.PRNGKey(5),B)
reset=jax.jit(jax.vmap(E.reset)); S=reset(keys, jnp.tile(jnp.arange(1,9),B//8))
step=jax.jit(E.step)
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
    dun=jax.tree_util.tree_map(lambda v:np.array(v[i]),S.dun); dn,ar,fl,n=dun.doors,dun.arrive,dun.flip,dun.n
    seen=set(); s0=jax.tree_util.tree_map(lambda a:a[i],S)
    for r in range(int(dun.n_rooms)):
        for c in range(n[r]):
            ch=chain(dn,ar,fl,n,r,c)
            if not ch: continue
            key=tuple(sorted(x[0] for x in ch))
            if key in seen: continue
            seen.add(key)
            if len(seen)>5: break
            s=s0.replace(room=jnp.int32(r),view_angle=jnp.float32(0.),mirror=jnp.float32(1.),player=s0.player.replace(inv=jnp.float32(100.)),
                         hunter=s0.hunter.replace(dead=jnp.float32(1e9)))
            for (rr,k) in ch:
                assert int(s.room)==rr
                dr,nr_=dun.sides.door[rr,k],dun.sides.normal[rr,k]; s=s.replace(player=s.player.replace(x=jnp.float32(dr[0]+nr_[0]*10),y=jnp.float32(dr[1]+nr_[1]*10)))
                s,_,_,_=step(s,jnp.zeros(4))
            t=float(s.view_angle)%TAU; t=min(t,TAU-t)*180/np.pi; want=float(dun.holonomy)%360; want=min(want,360-want)
            tested+=1
            ok=int(s.room)==r and float(s.mirror)==1 and abs(t-want)<1e-2
            if not ok: fail+=1
print('generated-dungeon corner loops walked:',tested,'failed:',fail)
assert tested>100 and fail==0, 'corner loops (curvature / seam holonomy) do not match the tiling'
print('PASS')
