import _setup
import time, numpy as np, jax, jax.numpy as jnp
from seamjax.gen import *
from seamjax.core.config import MAX_ROOMS as R, N_SPAWN_POINTS as NSP
SP=load_specs()
f=jax.jit(jax.vmap(lambda k,fl: gen_dungeon(SP,k,fl)))
B=256
keys=jax.random.split(jax.random.PRNGKey(0),B); fl=jnp.tile(jnp.arange(1,9),B//8)
t0=time.time(); D=f(keys,fl); jax.block_until_ready(D); t1=time.time(); D=f(keys,fl); jax.block_until_ready(D); t2=time.time()
print(f'compile+run {t1-t0:.1f}s, run {t2-t1:.2f}s for {B} dungeons ({B/(t2-t1):.0f}/s)')
D=jax.tree_util.tree_map(np.asarray,D)
bad=dict(pad_fallback=0,sym=0,conn=0,flip=0,spawn=0,obsin=0,nrooms=0,exitfar=0)
for i in range(B):
    m=D.n_rooms[i]; dn=D.doors[i]; ar=D.arrive[i]; fl_=D.flip[i]; n=D.n[i]
    if m>R or m<2: bad['nrooms']+=1
    for r in range(m):
        for k in range(n[r]):
            b=dn[r,k]
            if b>=0:
                if dn[b,ar[r,k]]!=r or ar[b,ar[r,k]]!=k or fl_[b,ar[r,k]]!=fl_[r,k]: bad['sym']+=1
    if (D.depth[i][:m]<0).any(): bad['conn']+=1
    if D.depth[i][D.exit[i]]<2 and m>3: bad['exitfar']+=1
    # spawn / pad sanity: inside polygon
    for r in range(m):
        mid,nrm=D.sides.mid[i,r,:n[r]],D.sides.normal[i,r,:n[r]]
        def inside(x,y,mg): return all((x-mid[:,0])*nrm[:,0]+(y-mid[:,1])*nrm[:,1] <= -mg+1e-6)
        for j in range(NSP):
            if D.spawn_ok[i,r,j] and not inside(*D.spawn[i,r,j],40): bad['spawn']+=1
        if r==D.exit[i] and not inside(*D.pad[i,r],90) and not (D.pad[i,r]==[480,300]).all(): bad['pad_fallback']+=1
# independent walkability check of every exit pad and spawn point
from collections import deque
GW,GH,CS=48,30,20.
def reach_np(i,r):
    """independent flood fill; returns (cells reachable from any door, door -> component label)"""
    m_=D.n[i,r]; mid,nrm,door=D.sides.mid[i,r,:m_],D.sides.normal[i,r,:m_],D.sides.door[i,r,:m_]; ob=D.obstacles; pos,size,ot=ob.pos[i,r],ob.size[i,r],ob.kind[i,r]
    xs=(np.arange(GW)+.5)*CS; ys=(np.arange(GH)+.5)*CS; X,Y=np.meshgrid(xs,ys)
    free=np.ones((GH,GW),bool)
    for a_,n_ in zip(mid,nrm): free&=((X-a_[0])*n_[0]+(Y-a_[1])*n_[1])<=-15
    for p_,z_,t_ in zip(pos,size,ot):
        if t_==0: free&=np.hypot(X-p_[0],Y-p_[1])>=z_[0]+15
        elif t_==1:
            qx=np.clip(X,p_[0],p_[0]+z_[0]);qy=np.clip(Y,p_[1],p_[1]+z_[1]);free&=np.hypot(X-qx,Y-qy)>=15
    dk=[k for k in range(m_) if D.doors[i,r,k]>=0]
    pts={k:(door[k,0]-nrm[k,0]*30,door[k,1]-nrm[k,1]*30) for k in dk} or {-1:(480,300)}
    def snap(p):
        d=np.where(free,np.hypot(X-p[0],Y-p[1]),1e9); return np.unravel_index(d.argmin(),d.shape)
    lab=-np.ones(free.shape,int); dl={}; nl=0
    for k,p in pts.items():
        st=snap(p)
        if lab[st]<0:
            lab[st]=nl; q=deque([st])
            while q:
                a,b=q.popleft()
                for da,db in((1,0),(-1,0),(0,1),(0,-1)):
                    x,y=a+da,b+db
                    if 0<=x<GH and 0<=y<GW and free[x,y] and lab[x,y]<0: lab[x,y]=nl;q.append((x,y))
            nl+=1
        dl[k]=lab[st]
    return lab>=0,dl
bad['doors_cut']=bad['sec_mixed']=bad['wall_geom']=bad['sec_disc']=bad['spawn_sec']=bad['dsec_side']=bad['node_unreach']=0
nwalled=nsolid=ngated=0
bad['pad_unreach']=bad['spawn_unreach']=bad['over_cap']=0
for i in range(B):
    for r in range(D.n_rooms[i]):
        seen,dl=reach_np(i,r); wax=D.wall_axis[i,r]
        if wax==0:
            if len(set(dl.values()))>1: bad['doors_cut']+=1
        else:
            nwalled+=1; nsolid+=D.wall_kind[i,r]==1; ngated+=D.wall_kind[i,r]==2
            ds=D.door_sec[i,r]; P=D.wall_pos[i,r]; dr,nr_=D.sides.door[i,r],D.sides.normal[i,r]
            g=[set(dl[k] for k in dl if ds[k]==c) for c in (0,1)]
            if len(g[0])!=1 or len(g[1])!=1: bad['sec_mixed']+=1
            if g[0]==g[1]: bad['sec_disc']+=1
            for k in dl:   # door entry point lies on the side of the wall its section says
                c=(dr[k,1]-nr_[k,1]*30) if wax==1 else (dr[k,0]-nr_[k,0]*30)
                if (c>=P)!=(ds[k]==1): bad['dsec_side']+=1
            flagged=[t_ for w_,t_ in zip(D.obstacles.wall[i,r],D.obstacles.kind[i,r]) if w_]
            if len(flagged)!=1 or flagged[0]!=1 or D.wall_kind[i,r] not in (1,2): bad['wall_geom']+=1
            for j in range(NSP):
                p=D.spawn[i,r,j]
                if D.spawn_ok[i,r,j] and int(D.spawn_sec[i,r,j])!=int((p[1] if wax==1 else p[0])>=P): bad['spawn_sec']+=1
        cell=lambda p:(min(max(int(p[1]//CS),0),GH-1),min(max(int(p[0]//CS),0),GW-1))
        if r==D.exit[i] and not seen[cell(D.pad[i,r])]: bad['pad_unreach']+=1
        for j in range(NSP):
            if D.spawn_ok[i,r,j] and not seen[cell(D.spawn[i,r,j])]: bad['spawn_unreach']+=1
# the world must be walkable from the start: solid walls cut a room in two, gated ones do not
for i in range(B):
    m=D.n_rooms[i]; wk=D.wall_kind[i]; dn=D.doors[i]; ar=D.arrive[i]; ds=D.door_sec[i]
    nid=lambda r,sec: 2*r+(sec if wk[r]==1 else 0)
    k0=int(np.argmax(dn[D.start[i]]>=0)); st=nid(D.start[i],ds[D.start[i],k0]); seen={st}; q=deque([st])
    while q:
        a=q.popleft(); r=a>>1
        for k in range(D.n[i,r]):
            b=dn[r,k]
            if b>=0 and nid(r,ds[r,k])==a:
                t=nid(b,ds[b,ar[r,k]])
                if t not in seen: seen.add(t); q.append(t)
    for r in range(m):
        need=[2*r,2*r+1] if wk[r]==1 else [2*r]
        if any(x not in seen for x in need): bad['node_unreach']+=1
    nd=D.node_dist[i]
    for a in list(seen)[:6]:   # node distances are consistent with the BFS
        if nd[st,a]>=999 or nd[a,st]!=nd[st,a]: bad['node_unreach']+=1
print('walls: rooms',nwalled,'solid',nsolid,'gated',ngated,'of',int(D.n_rooms.sum()))
print('violations',bad)
cat=D.category; names=['flat','hyper','sphere','grid']
print('category mix',{names[c]:int((cat==c).sum()) for c in range(4)})
print('rooms mean/min/max',D.n_rooms.mean(),D.n_rooms.min(),D.n_rooms.max(),'| exit depth mean',D.depth[np.arange(B),D.exit].mean())
ndoor=np.array([(D.doors[i,:D.n_rooms[i]]>=0).sum()/2 for i in range(B)]); print('doors mean',ndoor.mean())

assert all(v==0 for k,v in bad.items() if k!='pad_fallback'), bad
assert bad['pad_fallback']<.1*B
print('PASS')
