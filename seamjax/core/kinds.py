"""
Enemy kinds as data. Each kind is one EnemyKind row; the simulation, the observation, the renderers and the sound all read
this table, so a new kind of enemy is a new row (plus, for a new way of moving, one more behaviour in sim/systems/enemies.py).

The index of a kind in ENEMY_KINDS is what Enemies.kind stores.
The share of each kind among spawned enemies depends on the floor: share = clip(base + per_floor*(floor-1), lo, hi);
the kind with share=None gets what is left.

Also here: the player's gun (BLASTER), as data, ready for more weapons or character classes.
"""
from dataclasses import dataclass, fields
from types import SimpleNamespace
import numpy as np

# behaviours (how a kind moves; see sim/systems/enemies.py)
CHASE = 0           # straight at you
SKIRMISH = 1        # keep a distance: back off when close, close in when far, else strafe round you


@dataclass(frozen=True)
class EnemyKind:
    name: str
    radius: float
    hp: float
    behaviour: int
    speed: float                    # px/s
    speed_per_floor: float = 0.     # added per floor (up to the difficulty cap)
    retreat: float = 230.           # SKIRMISH: back off closer than this
    approach: float = 340.          # SKIRMISH: close in further than this
    strafe: float = .6              # SKIRMISH: strafing speed as a share of `speed`
    ranged: bool = False            # shoots at you
    first_shot: float = 1.          # seconds before the first shot, plus up to first_shot_rand
    first_shot_rand: float = 1.5
    reload: float = 1.6             # seconds between shots, plus up to reload_rand
    reload_rand: float = .9
    bullet_speed: float = 190.
    bullet_life: float = 3.
    share: tuple = None             # (base, per_floor, lo, hi): share of spawns, or None for "the rest"
    colour: tuple = (255, 122, 69)  # how it is drawn (0..255)
    kill_sound: str = 'kill_charger'


CHASER = EnemyKind('chaser', radius=12., hp=3., behaviour=CHASE, speed=70., speed_per_floor=7.)
SHOOTER = EnemyKind('shooter', radius=14., hp=4., behaviour=SKIRMISH, speed=55., ranged=True, share=(.12, .03, .12, .45),
                    colour=(180, 126, 255), kill_sound='kill_shooter')

ENEMY_KINDS = (CHASER, SHOOTER)


def table(kinds=None):
    """the kinds as numpy columns: table().radius[k], table().ranged[k], ... (numeric fields only)"""
    kinds = ENEMY_KINDS if kinds is None else kinds
    cols = {f.name: np.array([getattr(k, f.name) for k in kinds]) for f in fields(EnemyKind) if f.name not in ('name', 'share', 'kill_sound')}
    cols['colour'] = cols['colour'].astype(np.float32)/255.
    return SimpleNamespace(**cols)


@dataclass(frozen=True)
class Weapon:
    delay: float        # seconds between shots
    speed: float
    life: float
    radius: float
    muzzle: float       # bullets start this far from the shooter's centre


BLASTER = Weapon(delay=.14, speed=520., life=.9, radius=3., muzzle=14.)
