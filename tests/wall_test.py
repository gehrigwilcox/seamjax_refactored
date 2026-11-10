import _setup
"""Functional checks of the wall logic in the JAX env: gate opens on clear, solid never, blocking, per-section spawn, exit rule."""
import numpy as np, jax, jax.numpy as jnp
from seamjax.sim.game import Game; from seamjax.sim.queries import *; from seamjax.core.geometry.view import view_vec
from seamjax.sim.state import no_events; from seamjax.sim.systems.transit import transit as _transit
from seamjax.gen import *
from seamjax.core.config import *
SP=load_specs(); E=Game(gen_source(SP))
B=64; S=jax.jit(jax.vmap(E.reset))(jax.random.split(jax.random.PRNGKey(11),B), jnp.full(B,8))
step=jax.jit(E.step); trans=jax.jit((lambda s_,k_: _transit(s_, no_events(), k_)[0]))
res=dict(gated=0,gate_opened=0,solid=0,solid_closed=0,blocked=0,blocked_n=0,spawn_only_own=0,spawn_n=0,reward_gate=0,exit_same_sec=0,exit_n=0)
for i in range(B):
    s=S[i]; d=s.dun; wk=np.array(d.wall_kind); wax=np.array(d.wall_axis); wp=np.array(d.wall_pos); ds=np.array(d.door_sec); dn=np.array(d.doors)
    for r in np.where(wk>0)[0][:2]:
        ks=[k for k in range(int(d.n[r])) if dn[r,k]>=0]
        for sec in (0,1):
            kk=[k for k in ks if ds[r,k]==sec]
            if not kk: continue
            k=kk[0]; nb=int(dn[r,k]); arr=int(d.arrive[r,k])
            s0=s.replace(room=jnp.int32(nb),visited=jnp.zeros_like(s.visited),spawned=jnp.zeros_like(s.spawned),wall_open=jnp.zeros_like(s.wall_open))
            s1=trans(s0,jnp.int32(arr))                                   # walk into room r from the neighbour, on door k's side
            assert int(s1.room)==r
            e=int(sec_point(d,r,s1.player.x,s1.player.y)); assert e==sec,(e,sec)
            en=jax.tree_util.tree_map(np.array,s1.enemies[r]); alive=en.alive
            sp=np.array(d.spawn[r]); ss=np.array(d.spawn_sec[r])
            # every spawned enemy sits at a spawn point of this section
            own=all(any(np.hypot(en.x[j]-sp[q,0],en.y[j]-sp[q,1])<1e-3 and ss[q]==sec for q in range(16)) for j in np.where(alive)[0])
            res['spawn_n']+=1; res['spawn_only_own']+=int(own)
            # clear the room, step once
            s2=s1.replace(enemies=s1.enemies.replace(alive=s1.enemies.alive.at[r].set(False)),player=s1.player.replace(inv=jnp.float32(5.)),hunter=s1.hunter.replace(dead=jnp.float32(1e9)))
            s3,rew,_,_=step(s2,jnp.zeros(4))
            if wk[r]==2:
                res['gated']+=1; res['gate_opened']+=int(bool(s3.wall_open[r])); res['reward_gate']+=int(float(rew)>0.04)
            else:
                res['solid']+=1; res['solid_closed']+=int(not bool(s3.wall_open[r]))
            # the wall blocks: walk straight at it from your side, you must stay on your side (solid) 
            if wk[r]==1:
                s4=s2
                for f in range(240):
                    # steer toward the other side of the wall in room coordinates, expressed as screen input
                    wantx,wanty=(0.,1.) if wax[r]==1 else (1.,0.)
                    if sec==1: wantx,wanty=-wantx,-wanty
                    vx,vy=view_vec(s4.view_angle,s4.mirror,wantx,wanty)
                    s4,_,_,_=step(s4.replace(player=s4.player.replace(inv=jnp.float32(5.)),hunter=s4.hunter.replace(dead=jnp.float32(1e9))),jnp.array([vx,vy,0.,0.]))
                    if int(s4.room)!=r: break
                if int(s4.room)==r:
                    res['blocked_n']+=1; res['blocked']+=int(int(sec_point(d,r,s4.player.x,s4.player.y))==sec)
print(res)
assert res['gated']>50 and res['solid']>30
assert res['gate_opened']==res['gated'] and res['solid_closed']==res['solid'] and res['blocked']==res['blocked_n']==res['solid'] and res['spawn_only_own']==res['spawn_n'] and res['reward_gate']==res['gated']
print('PASS')
