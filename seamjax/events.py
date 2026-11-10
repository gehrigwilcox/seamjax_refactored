"""
Game events for the viewer, found by comparing each snapshot with the previous one (the simulation itself stays pure and
knows nothing about effects or sound).  Both the visual effects (glview.Fx) and the sound (audio.Audio) consume these.

An event is a dict: name, x, y (world position in the current room) and sometimes `shooter`.  Names:
  shoot  hit  heal  kill  death  hunter_hit  hunter_die  open  descend  newrun
"""
import math
import numpy as np


class Events:
    def __init__(self): self.prev = None

    def update(self, sn, key, gen, dead, opened=False):
        """key = (gen, room); gen changes whenever a new floor / run is generated"""
        en = np.asarray(sn['en']); alive = en[:, 7] > .5; hp = int(sn['hp']); nb = int((np.asarray(sn['pb'])[:, 5] > .5).sum())
        hh = float(sn['hhp']); hal = float(sn['hdead']) <= 0; x, y = float(sn['x']), float(sn['y']); P = self.prev; ev = []
        E = lambda name, px=x, py=y, **kw: ev.append(dict(name=name, x=float(px), y=float(py), **kw))
        if P is not None and P['gen'] == gen:
            if hp < P['hp']: E('hit')
            elif hp > P['hp'] and P['key'] == key: E('heal')
            if dead and not P['dead']: E('death')
            if opened: E('open')
            if P['key'] == key:
                for i in np.where(P['alive'] & ~alive)[0]: E('kill', P['en'][i, 0], P['en'][i, 1], shooter=bool(P['en'][i, 3] > .5))
                hx, hy = float(sn['hx']), float(sn['hy'])
                if hh < P['hh'] - .5 and bool(sn['hhere']): E('hunter_hit', hx, hy)
                if P['hal'] and not hal: E('hunter_die', hx, hy)
                if nb > P['nb']: a = float(sn['aim']); E('shoot', x+math.cos(a)*16, y+math.sin(a)*16)
        elif P is not None: E('newrun' if P['dead'] else 'descend')
        self.prev = dict(key=key, gen=gen, en=en.copy(), alive=alive, hp=hp, nb=nb, hh=hh, hal=hal, dead=dead)
        return ev
