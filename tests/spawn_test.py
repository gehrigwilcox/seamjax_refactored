import _setup
import numpy as np, jax, jax.numpy as jnp
from seamjax.gen import *
from seamjax.sim.game import Game; from seamjax.sim.queries import *; from seamjax.core.geometry.view import view_vec
from seamjax.sim.state import no_events; from seamjax.sim.systems.transit import transit as _transit
from seamjax.core.config import *
SP=load_specs(); E=Game(gen_source(SP))
B=96; S=jax.jit(jax.vmap(E.reset))(jax.random.split(jax.random.PRNGKey(7),B), jnp.tile(jnp.arange(1,9),B//8))
@jax.jit
def probe(s,r,k):
    s=s.replace(room=r,visited=jnp.zeros_like(s.visited),spawned=jnp.zeros_like(s.spawned),wall_open=jnp.zeros_like(s.wall_open))
    s2=(lambda s_,k_: _transit(s_, no_events(), k_)[0])(s,k); nr=s.dun.doors[r,k]
    en=s2.enemies[nr]; alive=en.alive
    d=jnp.hypot(en.x-s2.player.x,en.y-s2.player.y)
    area=s.dun.area[nr]; cap=jnp.maximum(1,jnp.round(area/AREA_PER_ENEMY).astype(jnp.int32))+(nr==s.dun.exit)
    minD=jnp.clip(s.dun.radius[nr]*.55,SPAWN_MIN,SPAWN_MAX)
    return alive.sum(),cap,jnp.min(jnp.where(alive,d,1e9)),minD,nr==s.dun.start
n=over=close=zero=0; counts=[]
for i in range(B):
    s=S[i]; dn=np.array(s.dun.doors); nr_=np.array(s.dun.n)
    for r in range(int(s.dun.n_rooms)):
        for k in range(nr_[r]):
            if dn[r,k]>=0 and dn[r,k]!=int(s.dun.start):
                a,cap,dmin,minD,_=probe(s,jnp.int32(r),jnp.int32(k)); n+=1; counts.append(int(a))
                over+=int(a>cap); close+=int(float(dmin)<.6*float(minD)-1e-3); zero+=int(a==0)
print('room entries',n,'over capacity',over,'enemy closer than 0.6*minD',close,'rooms with no enemies',zero,'mean enemies',np.mean(counts))
assert over==0 and close==0 and zero<.05*n
print('PASS')
