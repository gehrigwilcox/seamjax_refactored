"""
Enemy kinds as data. Each kind is one EnemyKind row that the simulation and the observation read, so a new kind of enemy is
a new row (plus, for a new way of moving, one more behaviour in sim/systems/enemies.py). How a kind looks and sounds is in
seamjax/theme.py, by the kind's name (a kind without a look there is drawn as a grey disk).

The index of a kind in ENEMY_KINDS is what Enemies.kind stores.
The share of each kind among spawned enemies depends on the floor: share = clip(base + per_floor*(floor-1), lo, hi);
the kind with share=None gets what is left.

Also here: player classes (PlayerClass: speed, max hp, weapon) and their weapons (Weapon: fire rate, bullets, damage,
spread, pierce, melee). Player.cls indexes PLAYER_CLASSES; class 0 is the game as it always was.
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


CHASER = EnemyKind('chaser', radius=12., hp=3., behaviour=CHASE, speed=70., speed_per_floor=7.)
SHOOTER = EnemyKind('shooter', radius=14., hp=4., behaviour=SKIRMISH, speed=55., ranged=True, share=(.12, .03, .12, .45))

ENEMY_KINDS = (CHASER, SHOOTER)


def table(kinds=None):
    """the kinds as numpy columns: table().radius[k], table().ranged[k], ... (numeric fields only)"""
    kinds = ENEMY_KINDS if kinds is None else kinds
    cols = {f.name: np.array([getattr(k, f.name) for k in kinds]) for f in fields(EnemyKind) if f.name not in ('name', 'share')}
    return SimpleNamespace(**cols)


@dataclass(frozen=True)
class Weapon:
    name: str
    delay: float                # seconds between shots (or swings)
    speed: float = 520.         # bullet speed
    life: float = .9            # bullet life, seconds
    radius: float = 3.          # bullet radius
    muzzle: float = 14.         # bullets start this far from the player's centre
    damage: float = 1.
    pellets: int = 1            # bullets per shot, fanned evenly over `spread`
    spread: float = 0.          # radians, the whole fan
    pierce: int = 0             # enemies a bullet passes through before it stops (each one is hit once)
    ghost: bool = False         # bullets pass obstacles and inner walls (never the room's own walls)
    melee: bool = False         # no bullets: a swing hits every enemy within `reach` and `arc` of the aim
    reach: float = 0.           # melee: from the player's centre to the edge of the swing
    arc: float = 0.             # melee: radians, the whole swing


@dataclass(frozen=True)
class PlayerClass:
    name: str
    weapon: Weapon
    speed: float = 215.         # px/s
    max_hp: int = 5


BLASTER = Weapon('blaster', delay=.14)
SCATTER = Weapon('scatter', delay=.42, speed=480., life=.42, pellets=5, spread=.5)
LANCE = Weapon('lance', delay=.32, speed=700., life=.8, damage=2., pierce=2, ghost=True)
BLADE = Weapon('blade', delay=.3, damage=2., melee=True, reach=48., arc=2.2)

GUNNER = PlayerClass('gunner', BLASTER)                                     # class 0: the game as it always was
SCATTERER = PlayerClass('scatter', SCATTER, speed=205.)
LANCER = PlayerClass('lancer', LANCE, speed=205., max_hp=4)
BLADER = PlayerClass('blade', BLADE, speed=235., max_hp=6)

PLAYER_CLASSES = (GUNNER, SCATTERER, LANCER, BLADER)                        # Player.cls indexes this


def class_index(name):
    return [c.name for c in PLAYER_CLASSES].index(name)


def class_table(classes=None):
    """the classes as numpy columns: speed, max_hp, and every weapon field as w_<field> (w_delay, w_pellets, ...)"""
    classes = PLAYER_CLASSES if classes is None else classes
    cols = dict(speed=np.array([c.speed for c in classes]), max_hp=np.array([c.max_hp for c in classes], np.int32))
    for f in fields(Weapon):
        if f.name != 'name': cols['w_'+f.name] = np.array([getattr(c.weapon, f.name) for c in classes])
    return SimpleNamespace(**cols)

