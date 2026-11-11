"""
The data of the game, as typed pytrees (see struct.py). Shapes are given for ONE game; a batch adds a leading axis to every
field. R = MAX_ROOMS, NS = MAX_SIDES, NO = MAX_OBSTACLES, NE = MAX_ENEMIES (config.py).

Every entity kind is a fixed-capacity pool: an `alive` mask plus one array per field. A free slot is any slot with
alive == False.
"""
from seamjax.core.struct import pytree

# obstacle kinds
OBST_NONE, OBST_CIRCLE, OBST_RECT = -1, 0, 1
# inner wall of a room
AXIS_NONE, AXIS_H, AXIS_V = 0, 1, 2                # AXIS_H splits by y (a horizontal wall), AXIS_V by x
WALL_NONE, WALL_SOLID, WALL_GATED = 0, 1, 2        # solid never opens; gated opens once your side is cleared
# world categories
GEO_FLAT, GEO_HYPER, GEO_SPHERE, GEO_WRAPPED = 0, 1, 2, 3


# ---------------------------------------------------------------- dungeon (generated once per floor, constant during play)
@pytree
class Sides:
    """the sides of room polygons, in the room's own frame. Each field [..., NS, 2]"""
    mid: object         # midpoint of the side
    normal: object      # outward unit normal
    tangent: object     # unit tangent (along the side, clockwise)
    door: object        # centre of the door opening (on the side line; meaningful where there is a door)


@pytree
class Obstacles:
    """[..., NO] obstacles. A circle has pos = centre and size = (radius, 0); a rect has pos = top-left corner and size = (w, h).
    The inner wall of a room, if any, is a rect obstacle with wall = True; it stops counting once the wall is open."""
    kind: object        # int: OBST_NONE / OBST_CIRCLE / OBST_RECT
    pos: object         # [..., NO, 2]
    size: object        # [..., NO, 2]
    wall: object        # bool: this obstacle is the room's inner wall


@pytree
class Dungeon:
    # ---- per room, [R, ...]; slots >= n_rooms are padding (n = 0)
    n: object           # [R] sides of the room polygon
    doors: object       # [R, NS] room behind each side, -1 = no door
    arrive: object      # [R, NS] side of that room you come in through
    flip: object        # [R, NS] 1 if the door crosses a mirrored seam (your view flips)
    sides: Sides        # [R, NS, 2]
    obstacles: Obstacles  # [R, NO]
    pad: object         # [R, 2] exit pad position (exit room only)
    radius: object      # [R] bounding radius of the room
    inradius: object    # [R] distance from the centre to the nearest side
    area: object        # [R]
    spawn: object       # [R, 16, 2] pre-validated, reachable enemy spawn points
    spawn_ok: object    # [R, 16] valid spawn point
    spawn_sec: object   # [R, 16] section the spawn point lies in
    wall_axis: object   # [R] AXIS_*: orientation of the inner wall
    wall_pos: object    # [R] its coordinate (y for AXIS_H, x for AXIS_V)
    wall_kind: object   # [R] WALL_*
    door_sec: object    # [R, NS] section each door is in (0 / 1; 0 without a wall)
    depth: object       # [R] doors from the start room (-1 unreachable / padding)
    map_pos: object     # [R, 3] room centre on the model surface (only for drawing a map)
    door_seam: object   # [R, NS] door crosses a seam of a wrapped world (only for drawing a map)
    # ---- whole dungeon
    n_rooms: object
    start: object
    startpos: object    # [2] start position in the start room
    exit: object
    floor: object
    category: object    # GEO_*
    spec: object        # index of the tiling in the catalogue (specs.json)
    holonomy: object    # degrees you turn walking once round a corner
    dist: object        # [R, R] door distances between rooms
    node_dist: object   # [2R, 2R] door distances between Hunter nodes (room sections, see env.hnode)

    def room(self, r):
        """per-room fields of room r (a Dungeon whose per-room fields lost their R axis)"""
        return RoomView(n=self.n[r], doors=self.doors[r], arrive=self.arrive[r], flip=self.flip[r], sides=self.sides[r],
                        obstacles=self.obstacles[r], pad=self.pad[r], wall_axis=self.wall_axis[r], wall_pos=self.wall_pos[r],
                        wall_kind=self.wall_kind[r], door_sec=self.door_sec[r])


@pytree
class RoomView:
    """the tables of one room (see Dungeon for the meaning of each field)"""
    n: object
    doors: object
    arrive: object
    flip: object
    sides: Sides
    obstacles: Obstacles
    pad: object
    wall_axis: object
    wall_pos: object
    wall_kind: object
    door_sec: object


# ---------------------------------------------------------------- entities
@pytree
class Player:
    x: object
    y: object
    hp: object          # int
    inv: object         # invulnerability time left
    fire: object        # time until the gun can fire again
    aim: object         # aim angle in the room frame
    cls: object         # int: player class, index into core/kinds.PLAYER_CLASSES
    swing: object       # melee: time left showing the last swing (0 when not swinging)


@pytree
class Bullets:
    """[NB] bullet pool (the player's and the enemies' bullets are separate pools of this type)"""
    x: object
    y: object
    vx: object
    vy: object
    life: object        # seconds left
    alive: object
    damage: object      # damage per hit
    pierce: object      # int: enemies it can still pass through
    ghost: object       # bool: passes obstacles and inner walls
    hits: object        # int bitmask: enemy slots (bit i) and the Hunter (bit NE) it already hit


@pytree
class Enemies:
    """[R, NE] enemy pool, per room (only the current room is simulated)"""
    x: object
    y: object
    hp: object
    kind: object        # int: index into core/kinds.ENEMY_KINDS (what it is: radius, hp, speed, behaviour, gun)
    fire: object        # time until it fires (ranged kinds)
    telegraph: object   # time until it becomes active (shown as a ring until then)
    flash: object       # hit-flash time left
    alive: object
    phase: object       # int: a boss's current phase (kinds.Phase); 0 for everything else
    ptimer: object      # seconds in the current phase


@pytree
class Pickups:
    """[R, NPK] health pickups, per room"""
    x: object
    y: object
    alive: object


@pytree
class Roamers:
    """[NR] enemies following you between rooms. While travelling they are only a place on the room graph (room, section);
    when they reach your section they come in through a door and become ordinary enemies again."""
    alive: object
    kind: object        # enemy kind (core/kinds.ENEMY_KINDS)
    hp: object
    room: object
    sec: object
    timer: object       # time until the next door
    way: object         # node it heads for first (FLANK / DETOUR), -1 = straight to you


@pytree
class Hunter:
    room: object
    sec: object         # section of its room it is in (matters for solid walls)
    timer: object       # time until its next door hop
    here: object        # it is in the player's room-section and placed
    x: object
    y: object
    hp: object
    dead: object        # time until it respawns (> 0 while down)
    flash: object


@pytree
class StepEvents:
    """what happened during one step; returned by the step, not kept in the state"""
    took: object        # hits taken
    newroom: object     # entered a room for the first time
    killed: object      # enemies killed
    hit_hunter: object  # hit the Hunter
    descended: object   # took the stairs
    shake: object       # the Hunter burst in through a door
    healed: object      # pickups taken
    opened: object      # a gated wall opened


@pytree
class GameState:
    dun: Dungeon
    room: object        # the room you are in
    view_angle: object  # rotation of your view of the room
    mirror: object      # +1 / -1: your view of the room is mirrored
    player: Player
    pb: Bullets         # the player's bullets (current room only)
    eb: Bullets         # enemy bullets (current room only)
    enemies: Enemies
    pickups: Pickups
    hunter: Hunter
    roamers: Roamers
    visited: object     # [R] rooms you have been in
    wall_open: object   # [R] the inner wall of the room has opened
    spawned: object     # [R, 2] the section has spawned its enemies
    t: object           # seconds into the run
    steps: object
    kills: object
    key: object
