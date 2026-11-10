"""
Every game constant in one place: sizes, speeds, timings, reward weights, array capacities.

Plain Python numbers, no JAX import, so the viewer, the OpenGL code and the tests can read them too. Things that change
with the floor are not here but in difficulty.py.

Coordinates: each room is drawn in its own frame of VIEW_W x VIEW_H logical pixels, y down, centred on (CX, CY).
"""
import os
import numpy as np

TAU = 2*np.pi

# ---------------------------------------------------------------- frame
VIEW_W, VIEW_H = 960., 600.
CX, CY = 480., 300.

# ---------------------------------------------------------------- array capacities (fixed shapes for jit / vmap)
MAX_ROOMS = 40              # rooms per dungeon (padded)
MAX_SIDES = 12              # sides per room polygon
MAX_OBSTACLES = 6           # obstacles per room; the last slot holds the inner wall when there is one
MAX_ENEMIES = 10            # enemy slots per room
MAX_PLAYER_BULLETS = 24
MAX_ENEMY_BULLETS = 24
MAX_PICKUPS = 6             # health pickup slots per room
N_SPAWN_POINTS = 16         # pre-validated enemy spawn points per room
GEN_CHUNK = int(os.environ.get('SEAM_GEN_CHUNK', 64))     # dungeons generated at once (~4 MB scratch each): bounds GPU memory

# ---------------------------------------------------------------- doors
DOOR_GAP = 34.              # half-width of a door opening
ENTRY_INSET = 30.           # you appear this far inside the door you came through

# ---------------------------------------------------------------- player
PLAYER_R = 11.
PLAYER_SPEED = 215.
MAX_HP = 5
INVULN = 1.                 # seconds of invulnerability after a hit
AIM_DEADZONE = .25          # the gun fires while the aim stick is held further than this
FIRE_DELAY = .14
BULLET_SPEED = 520.
BULLET_LIFE = .9
BULLET_MUZZLE = 14.         # bullets start this far from the player's centre
BULLET_R = 3.               # player bullet radius
PICKUP_R = 20.              # walk this close to a health pickup to take it
PAD_R = 26.                 # stand this close to the exit pad's centre to take the stairs

# ---------------------------------------------------------------- enemies
CHASER_R, SHOOTER_R = 12., 14.
CHASER_HP, SHOOTER_HP = 3., 4.
CHASER_SPEED, CHASER_SPEED_PER_FLOOR = 70., 7.
SHOOTER_SPEED = 55.
SHOOTER_RETREAT, SHOOTER_APPROACH = 230., 340.      # shooters back off closer than this, close in further than this, else strafe
SHOOTER_STRAFE = .6
SHOOTER_FIRST_SHOT, SHOOTER_FIRST_SHOT_RAND = 1., 1.5
SHOOTER_RELOAD, SHOOTER_RELOAD_RAND = 1.6, .9
ENEMY_BULLET_SPEED = 190.
ENEMY_BULLET_LIFE = 3.
ENEMY_BULLET_R = 4.
HIT_FLASH = .1
TELEGRAPH = 1.0             # seconds an enemy is shown before it becomes active
MAX_SPAWN = 8               # enemies spawned at most per section entry
AREA_PER_ENEMY = 30000.     # room area per enemy (caps the count)
SPAWN_MIN, SPAWN_MAX = 160., 260.                  # bounds of the minimum spawn distance from the player

# ---------------------------------------------------------------- the Hunter
HUNTER_HP = 18.
HUNTER_R = 19.
HUNTER_SPEED, HUNTER_SPEED_PER_FLOOR = 88., 5.
HUNTER_HOP, HUNTER_HOP_PER_FLOOR, HUNTER_HOP_MIN = 5.2, .25, 2.8      # seconds per door hop while it travels
HUNTER_RESPAWN = 22.        # seconds it stays down after being killed
HUNTER_RESPAWN_DELAY = 8.   # seconds before it sets off again after respawning
HUNTER_GRACE = 1.6          # at most this long before it follows you out of its room
HUNTER_DOOR_INSET = 34.     # it appears this far inside the door it comes through
HUNTER_KILL_BONUS = 5       # kills credited for bringing it down

# ---------------------------------------------------------------- reward weights (RL)
R_KILL, R_HIT, R_ROOM, R_DESCEND, R_DEATH, R_HUNTER, R_HEAL, R_GATE = .1, -.2, .05, 5., -5., .5, .05, .05

# ---------------------------------------------------------------- generator
ROOM_RC = 285.              # circumradius of a room polygon before squashing
ROOM_SQUASH_MIN = .74       # rooms are squashed by a random factor in [this, 1] along x and y
DOOR_INSET = 14.            # a door keeps at least this far from a corner (plus DOOR_GAP)
GRID_W, GRID_H, GRID_CELL = 48, 30, 20.            # coarse walkability grid
WALL_GATED_P, WALL_THICK, WALL_CLEAR, WALL_MINLEN = .5, 16., 40., 140.
INF = 999                   # "unreachable" in the room / section distance tables
