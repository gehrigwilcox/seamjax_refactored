"""
Game events for the viewer, found by comparing each snapshot with the previous one (the simulation itself stays pure and
knows nothing about effects or sound).  Both the visual effects (gl/fx.py) and the sound (audio.Audio) consume these.

An event is a dict: name, x, y (world position in the current room) and, for a kill, `kind` (index into core/kinds.ENEMY_KINDS).  Names:
  shoot  hit  heal  kill  death  hunter_hit  hunter_die  open  descend  newrun
"""
import math
import numpy as np


class Events:
    def __init__(self): self.prev = None

    def update(self, sn, key, gen, dead, opened=False):
        """key = (gen, room); gen changes whenever a new floor / run is generated"""
        en = sn.enemies; alive = np.asarray(en.alive); hp = int(sn.player.hp); nb = int(np.asarray(sn.pb.alive).sum())
        hh = float(sn.hunter.hp); hal = float(sn.hunter.dead) <= 0; x, y = float(sn.player.x), float(sn.player.y); P = self.prev; ev = []
        E = lambda name, px=x, py=y, **kw: ev.append(dict(name=name, x=float(px), y=float(py), **kw))
        if P is not None and P['gen'] == gen:
            if hp < P['hp']: E('hit')
            elif hp > P['hp'] and P['key'] == key: E('heal')
            if dead and not P['dead']: E('death')
            if opened: E('open')
            if P['key'] == key:
                for i in np.where(P['alive'] & ~alive)[0]: E('kill', P['ex'][i], P['ey'][i], kind=int(P['ekind'][i]))
                hx, hy = float(sn.hunter.x), float(sn.hunter.y)
                if hh < P['hh'] - .5 and bool(sn.hunter.here): E('hunter_hit', hx, hy)
                if P['hal'] and not hal: E('hunter_die', hx, hy)
                if nb > P['nb']: a = float(sn.player.aim); E('shoot', x+math.cos(a)*16, y+math.sin(a)*16)
        elif P is not None: E('newrun' if P['dead'] else 'descend')
        self.prev = dict(key=key, gen=gen, ex=np.array(en.x), ey=np.array(en.y), ekind=np.array(en.kind), alive=alive, hp=hp, nb=nb, hh=hh, hal=hal, dead=dead)
        return ev
