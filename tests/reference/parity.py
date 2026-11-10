import json, numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seamjax.sim.game import Game; from seamjax.sim.queries import *; from seamjax.core.geometry.view import view_vec
from seamjax.sim.state import no_events; from seamjax.sim.systems.transit import transit as _transit
from pool import load_pool, pool_source
from jax import lax
P=load_pool(); P={k:(v.astype(jnp.float64) if v.dtype==jnp.float32 else v) for k,v in P.items()}
P['spawn_ok']=jnp.zeros_like(P['spawn_ok'])
E=Game(pool_source(P)); D=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'pool.json')))
@jax.jit
def run(d, acts):
    s=E.reset(jax.random.PRNGKey(0),1,d); s=s.replace(hunter=s.hunter.replace(dead=jnp.float64(1e9)))
    def body(s,a):
        s=s.replace(player=s.player.replace(inv=jnp.float64(1.),hp=jnp.int32(5)))
        s,_,_,_=E.step(s,a)
        return s,jnp.stack([s.room.astype(jnp.float64),s.player.x,s.player.y,s.view_angle,s.mirror])
    _,tr=lax.scan(body,s,acts); return tr
tot=0; bad=0; maxpos=0; maxva=0; mirbad=0; div=[]
for di,d in enumerate(D):
    for sc in d['traj']:
        acts=np.zeros((900,4))
        for f in range(900): acts[f,:2]=sc['dirs'][f//sc['every']]
        tr=np.array(run(di,jnp.asarray(acts))); js=np.array(sc['tr'])
        mis=np.where(tr[:,0]!=js[:,0])[0]; end=mis[0] if len(mis) else 900
        if len(mis): bad+=1; div.append((di,int(end)))
        tot+=end
        if end:
            maxpos=max(maxpos,np.abs(tr[:end,1:3]-js[:end,1:3]).max())
            dv=np.abs((tr[:end,3]-js[:end,3]+np.pi)%(2*np.pi)-np.pi).max(); maxva=max(maxva,dv)
            mirbad+=int((tr[:end,4]!=js[:end,4]).sum())
print('scenarios',sum(len(d['traj']) for d in D),'diverged',bad,'frames compared',tot,'max pos err',maxpos,'max view err',maxva,'mirror mismatches',mirbad)
print(div[:8])

assert bad==0 and mirbad==0 and maxpos<0.05, 'parity with the JS game failed'
print('PASS')
