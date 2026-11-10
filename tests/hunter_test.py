import _setup
import numpy as np, jax, jax.numpy as jnp
from jax import lax
from seamjax.env import *; from seamjax.gen import *
"""The Hunter always takes the true shortest route through room-sections (solid walls separate, gated ones do not)."""
SP=load_specs(); E=make_env(gen_source(SP)); B=48
S0=jax.jit(jax.vmap(E['reset']))(jax.random.split(jax.random.PRNGKey(3),B), jnp.full(B,8))
@jax.jit
def run(s):
    def body(s,_):
        s={**s,'inv':1.,'hp':jnp.int32(5)}
        s,_,_=E['step'](s,jnp.zeros(4))
        return s,(s['hroom'],s['hsec'])
    s,(hr,hs)=lax.scan(body,s,None,length=60*90)
    pn=hnode(s['dun'],s['room'],sec_point(s['dun'],s['room'],s['x'],s['y']))
    return pn,hr,hs,s['dun']['ndist'],s['dun']['wkind']
ok=tot=walled=0
for d in range(B):
    s=jax.tree_util.tree_map(lambda a:a[d],S0)
    pn,hr,hs,nd,wk=[np.array(x) for x in run(s)]; pn=int(pn)
    node=lambda i: 2*hr[i]+(hs[i] if wk[hr[i]]==1 else 0)
    ns=[node(0)]+[node(i) for i in range(1,len(hr)) if node(i)!=node(i-1)]
    dist=[nd[n,pn] for n in ns]
    good=all(dist[i]-dist[i+1]==1 for i in range(len(dist)-1)) and (dist[-1]==0 or len(ns)>=12)   # long routes may still be under way after 90 s
    walled+=int(any(wk[hr[i]]==1 for i in range(len(hr)))); tot+=1; ok+=good
print('hunter follows strict shortest section routes:',ok,'/',tot,'| games touching a solid-walled room:',walled)
assert ok==tot
print('PASS')
