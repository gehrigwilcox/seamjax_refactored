import _setup
import numpy as np, jax, jax.numpy as jnp
from jax import lax
from seamjax.sim.game import Game; from seamjax.sim.queries import *; from seamjax.core.geometry.view import view_vec
from seamjax.sim.state import no_events; from seamjax.sim.systems.transit import transit as _transit
from seamjax.gen import *
"""The Hunter always takes the true shortest route through room-sections (solid walls separate, gated ones do not)."""
SP=load_specs(); E=Game(gen_source(SP)); B=48
S0=jax.jit(jax.vmap(E.reset))(jax.random.split(jax.random.PRNGKey(3),B), jnp.full(B,8))
@jax.jit
@jax.vmap
def run(s):
    def body(s,_):
        s=s.replace(player=s.player.replace(inv=jnp.float32(1.),hp=jnp.int32(5)))
        s,_,_,_=E.step(s,jnp.zeros(4))
        return s,(s.hunter.room,s.hunter.sec)
    s,(hr,hs)=lax.scan(body,s,None,length=60*90)
    return player_node(s),hr,hs,s.dun.node_dist,s.dun.wall_kind
ok=tot=walled=0
PN,HR,HS,ND,WK=[np.array(x) for x in run(S0)]          # all games in one call
for d in range(B):
    pn,hr,hs,nd,wk=int(PN[d]),HR[d],HS[d],ND[d],WK[d]
    node=lambda i: 2*hr[i]+(hs[i] if wk[hr[i]]==1 else 0)
    ns=[node(0)]+[node(i) for i in range(1,len(hr)) if node(i)!=node(i-1)]
    dist=[nd[n,pn] for n in ns]
    good=all(dist[i]-dist[i+1]==1 for i in range(len(dist)-1)) and (dist[-1]==0 or len(ns)>=12)   # long routes may still be under way after 90 s
    walled+=int(any(wk[hr[i]]==1 for i in range(len(hr)))); tot+=1; ok+=good
print('hunter follows strict shortest section routes:',ok,'/',tot,'| games touching a solid-walled room:',walled)
assert ok==tot
print('PASS')
