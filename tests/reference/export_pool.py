"""Reference data for parity.py: dungeons + scripted walks recorded from the original JS game.
usage: python tests/reference/export_pool.py [path/to/index.html] [count]   (needs playwright)"""
import json, sys, numpy as np
from playwright.sync_api import sync_playwright
import os
HERE=os.path.dirname(os.path.abspath(__file__))
GAME=os.path.abspath(sys.argv[1]) if len(sys.argv)>1 else '/home/claude/seam/index.html'   # the original JS game
N=int(sys.argv[2]) if len(sys.argv)>2 else 200
JS="""(N)=>{
 const out=[]; const realSpawn=spawnEnemies;
 window.stick=function(role){return (role==='move'&&window.__mv)?{x:window.__mv[0],y:window.__mv[1],len:1}:null;};
 for(let i=0;i<N;i++){
  G.state='play'; G.floor=1+(i%8); G.hp=5;
  const g=genDungeon(G.floor, i%5===4?{wrap:true}:undefined);
  const traj=[];
  if(i<80){
   G.dun=g; G.state='play'; initHunter(g); window.spawnEnemies=()=>{};   // no enemies: the scripted walks must be deterministic (gates open on the first frame)
   for(let sc=0;sc<4;sc++){
     for(const r of g.rooms){ r.spawned=[false,false]; r.enemies=[]; if(r.wall){ r.wall.open=false; if(!r.obs.includes(r.wall.rect)) r.obs.push(r.wall.rect); } }
     enterRoom(g.start,-1); G.va=0; G.mir=1; G.hp=5; G.p.inv=0;
     let dirs=[],tr=[];
     for(let f=0;f<900;f++){
       if(f%75===0){const a=Math.random()*2*Math.PI; dirs.push([Math.cos(a),Math.sin(a)]);}
       const dd=dirs[dirs.length-1];
       keys.clear(); mouse.down=false;
       // feed the movement through the same path as touch: patch stick() for this frame
       window.__mv=dd; G.hp=5; G.p.inv=1;
       update(1/60);
       tr.push([G.room.id,G.p.x,G.p.y,G.va,G.mir]);
     }
     traj.push({dirs,every:75,tr});
   }
   window.__mv=null; window.spawnEnemies=realSpawn;
  }
  out.push({traj,startpos:g.start.startPos,floor:g.floor,geo:g.geo,wrapped:g.wrapped,name:g.spec.name,start:g.start.id,exit:g.exit.id,
   rooms:g.rooms.map(r=>({n:r.n,doors:r.doors.map(d=>d?d.id:-1),arr:r.arr,flip:r.flip.map(f=>f?1:0),depth:r.depth===undefined?-1:r.depth,
     sd:r.sd.map(s=>[s.mx,s.my,s.nx,s.ny,s.tx,s.ty,s.dx,s.dy]),
     obs_t:r.obs.map(o=>o.t==='c'?0:1),obs:r.obs.map(o=>o.t==='c'?[o.x,o.y,o.r,0,0]:[o.x,o.y,o.w,o.h,o.wall?1:0]),
     wax:r.wall?(r.wall.axis==='h'?1:2):0,wpos:r.wall?r.wall.pos:0,wkind:r.wall?(r.wall.kind==='solid'?1:2):0,dsec:r.dsec.slice(),spawn_sec:r.spawnPts.slice(0,16).map(p=>p[2]),pad:r.pad?[r.pad.x,r.pad.y]:null,spawn:r.spawnPts.slice(0,16).map(p=>[p[0],p[1]]),area:r.area,rb:r.Rb,rin:r.rin}))});
 }
 return out;}"""
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(); pg.goto('file://'+GAME); pg.wait_for_timeout(300)
    D=pg.evaluate(JS,N); b.close()
R=max(len(d['rooms']) for d in D); S=12; O=6
print('dungeons',len(D),'max rooms',R)
R=((R+7)//8)*8
A=lambda *s,dt=np.float32,fill=0: np.full(s,fill,dtype=dt)
T=dict(n_rooms=A(N,dt=np.int32),n=A(N,R,dt=np.int32),doors=A(N,R,S,dt=np.int32,fill=-1),arr=A(N,R,S,dt=np.int32),flip=A(N,R,S,dt=np.int32),
 side=A(N,R,S,8),obs=A(N,R,O,5),obs_t=A(N,R,O,dt=np.int32,fill=-1),pad=A(N,R,2),spawn=A(N,R,16,2),area=A(N,R),spawn_ok=A(N,R,16,dt=np.int32),rb=A(N,R),rin=A(N,R),start=A(N,dt=np.int32),exit=A(N,dt=np.int32),
 startpos=A(N,2),wax=A(N,R,dt=np.int32),wpos=A(N,R),wkind=A(N,R,dt=np.int32),dsec=A(N,R,S,dt=np.int32),spawn_sec=A(N,R,16,dt=np.int32),ndist=A(N,2*R,2*R,dt=np.int32,fill=999),floor=A(N,dt=np.int32),geo=A(N,dt=np.int32),depth=A(N,R,dt=np.int32,fill=-1),dist=A(N,R,R,dt=np.int32,fill=999))
GEO={'flat':0,'hyper':1,'sphere':2}
for i,d in enumerate(D):
    T['startpos'][i]=d['startpos'];T['n_rooms'][i]=len(d['rooms']);T['start'][i]=d['start'];T['exit'][i]=d['exit'];T['floor'][i]=d['floor'];T['geo'][i]=GEO[d['geo']]+(3 if d['wrapped'] else 0)
    for r,rm in enumerate(d['rooms']):
        n=rm['n'];T['n'][i,r]=n;T['doors'][i,r,:n]=rm['doors'];T['arr'][i,r,:n]=rm['arr'];T['flip'][i,r,:n]=rm['flip'];T['depth'][i,r]=rm['depth']
        T['side'][i,r,:n]=rm['sd'];T['rb'][i,r]=rm['rb'];T['rin'][i,r]=rm['rin']
        assert len(rm['obs'])<=O
        for k,o in enumerate(rm['obs']): T['obs'][i,r,k]=o;T['obs_t'][i,r,k]=int(rm['obs_t'][k])
        T['area'][i,r]=rm['area'];T['wax'][i,r]=rm['wax'];T['wpos'][i,r]=rm['wpos'];T['wkind'][i,r]=rm['wkind'];T['dsec'][i,r,:n]=rm['dsec']
        T['spawn_sec'][i,r,:len(rm['spawn_sec'])]=rm['spawn_sec']
        for k,pt in enumerate(rm['spawn']): T['spawn'][i,r,k]=pt;T['spawn_ok'][i,r,k]=1
        if rm['pad']: T['pad'][i,r]=rm['pad']
    # all-pairs door distances (BFS)
    nR=len(d['rooms'])
    for s in range(nR):
        dist=T['dist'][i,s];dist[s]=0;q=[s]
        while q:
            a=q.pop(0)
            for t in d['rooms'][a]['doors']:
                if t>=0 and dist[t]==999: dist[t]=dist[a]+1;q.append(t)
    # node-level distances: a node is a room-section; a gated wall does not separate (the Hunter ignores gates)
    nid=lambda r,sec: 2*r+(sec if d['rooms'][r]['wkind']==1 else 0)
    adj=[[] for _ in range(2*R)]
    for a,rm in enumerate(d['rooms']):
        for k,b in enumerate(rm['doors']):
            if b>=0: adj[nid(a,rm['dsec'][k])].append(nid(b,d['rooms'][b]['dsec'][rm['arr'][k]]))
    for s0 in range(2*R):
        nd=T['ndist'][i,s0]; nd[s0]=0; q=[s0]
        while q:
            a=q.pop(0)
            for t in adj[a]:
                if nd[t]==999: nd[t]=nd[a]+1; q.append(t)
np.savez_compressed(os.path.join(HERE,'pool.npz'),**T); json.dump(D,open(os.path.join(HERE,'pool.json'),'w'))
print({k:v.shape for k,v in T.items()})
