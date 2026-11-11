"""Hand-made viewer inputs (numpy Snapshots and pools) for tests of the drawing, effects and sound code."""
import numpy as np
from seamjax.core.types import Player, Enemies, Pickups, Bullets, Hunter, Roamers
from seamjax.app.snapshot import Snapshot, Flags


def pool(cls, n, rows=()):
    """a pool of capacity n whose first slots hold `rows` (dicts of field values; unset fields are 0 / False)"""
    cols = {f: np.zeros(n, bool if f in ('alive', 'ghost') else int if f in ('kind', 'pierce', 'hits', 'room', 'sec', 'way') else float) for f in cls.fields()}
    for i, row in enumerate(rows):
        for f, v in row.items(): cols[f][i] = v
    return cls(**cols)


def snapshot(player=None, hunter=None, **kw):
    """a Snapshot of room 0 with nothing in it; player / hunter: dicts of fields to set; anything else by keyword"""
    P = dict(x=470., y=310., hp=5, inv=0., fire=0., aim=.3, cls=0, swing=0.); P.update(player or {})
    H = dict(room=0, sec=0, timer=0., here=False, x=0., y=0., hp=18., dead=0., flash=0.); H.update(hunter or {})
    d = dict(room=0, floor=1, spec=0, start=0, exit=0, n_rooms=1, kills=0, view_angle=0., mirror=1., player=Player(**P),
             enemies=pool(Enemies, 10), pickups=pool(Pickups, 6), pb=pool(Bullets, 24), eb=pool(Bullets, 24), hunter=Hunter(**H),
             hunter_door=-1, hunter_dist=2, roamer_doors=np.zeros(12, bool), roamers=pool(Roamers, 6), wall_open=False, spawned=np.zeros(2, bool), visited=np.zeros(1, bool),
             flags=Flags(opened=False, died=False, descended=False))
    d.update(kw)
    return Snapshot(**d)
