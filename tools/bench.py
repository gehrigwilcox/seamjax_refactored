import _path
import time, numpy as np, jax, jax.numpy as jnp
from jax import lax
from seamjax.env import *
from seamjax.gen import *
from seamjax.backend import describe
print('JAX backend:', describe())
SP=load_specs(); E=make_env(gen_source(SP))
B=512; T=300
keys=jax.random.split(jax.random.PRNGKey(1),B)
state=E['reset_batch'](keys, jnp.full(B,6))
obs_f=jax.vmap(E['observe'])
def body(c,t):
    s,k=c; k,ka,kr=jax.random.split(k,3)
    a=jax.random.uniform(ka,(B,4),minval=-1,maxval=1)
    rk=jax.random.split(kr,B)
    s,r,d,_=E['step_auto_batch'](s,a,kr)
    o=obs_f(s)
    return (s,k),(r,d,jnp.isfinite(o).all(),s.player.hp.min(),s.player.x.min(),s.player.x.max())
@jax.jit
def run(s,k): return lax.scan(body,(s,k),jnp.arange(T))
t0=time.time(); (s,k),out=run(state,jax.random.PRNGKey(2)); jax.block_until_ready(out); t1=time.time()
_,out=run(state,jax.random.PRNGKey(2)); jax.block_until_ready(out); t2=time.time()
r,d,fin,hpmin,xmin,xmax=[np.array(o) for o in out]
print(f'compile+run {t1-t0:.1f}s, second run {t2-t1:.2f}s -> {B*T/(t2-t1):,.0f} env-steps/s on {jax.devices()[0]}')
print('all obs finite',bool(fin.all()),'episodes ended',int(d.sum()),'reward mean/step',float(r.mean()),'x range',float(xmin.min()),float(xmax.max()))
print('kills total',int(np.array(s.kills).sum()),'min hp seen',int(hpmin.min()))
