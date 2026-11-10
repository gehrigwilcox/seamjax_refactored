# Seam (JAX)

A non-Euclidean roguelike twin-stick shooter, simulated **and generated** entirely in JAX. The same simulation
is what you play, what the Gymnasium environments run, and what you can put on a GPU to train thousands of games at once.

```
uv sync                                # CPU install (works everywhere); GPU installs are below
uv run seam                            # play it (WASD + mouse; --floor N to start deeper; --bot for a demo)
uv run python tests/run_all.py         # run the tests
uv run python tools/bench.py           # steps per second on your hardware
```

The project uses [uv](https://docs.astral.sh/uv/). `uv run seam` is the same as `uv run python -m seamjax.play`. Commands below that say
`python ...` mean `uv run python ...`.

## GPU backends (uv extras)

| install | hardware | notes |
|---|---|---|
| `uv sync` | CPU | works everywhere |
| `uv sync --extra cuda` | NVIDIA, CUDA 13 driver | Linux; `--extra cuda12` for an older CUDA 12 driver |
| `uv sync --extra rocm` | AMD (ROCm 7) | Linux x86_64; ROCm 7 must already be installed on the machine (JAX does not ship it) |
| `uv sync --extra metal` | Apple silicon GPU | experimental community plugin (`metaljax`), beta; Python 3.12+ |

Pick at most one GPU extra (they pin different JAX versions, so they cannot be installed together). **The backend is chosen automatically**
(`seamjax/backend.py`): with a GPU extra installed and a working device Seam runs on it, otherwise on the CPU, and it prints which one
(`JAX backend: cuda - NVIDIA ...`; the debug overlay shows it too). If a GPU plugin is installed but its device cannot start, it falls back
to the CPU and says why. To force one: `uv run seam --device cpu|cuda|rocm|metal` or `SEAM_BACKEND=...` (this also works for the Gymnasium
environments and the tests); an explicit choice that cannot start stops with the extra to install. Setting `JAX_PLATFORMS` yourself is respected.

Metal has no float64, so the one-off tiling build (which uses float64) runs on the CPU there and the results are moved to the GPU; everything
else (generation, simulation, RL) runs on the GPU. The ROCm and Metal paths could not be run by me (no such hardware here): CUDA and the
selection logic are tested only on CPU, so treat all three GPU backends as untested until you try them.

The tiling tables are built on the device by a jitted program the first time you run anything (a few seconds) and cached in `seamjax/data/`.
See `DESIGN.md` for the design document (vision, requirements, difficulty curve).

## Layout

```
seamjax/            the game (everything needed to play or train)
  specs.py          catalogue of worlds + a JAX builder that grows each tiling and closes its corner loops -> constant tables
  gen.py            GPU dungeon generator: rooms, doors, loops, obstacles, inner walls, spawn points, exit
  env.py            the simulation: movement, combat, Hunter, pickups, sections, view/mirror, observation
  difficulty.py     the difficulty curve (one function of the floor)
  render.py         pure-JAX renderer (pixel observations)
  gym_env.py        Gymnasium: SeamEnv ("Seam-v0") and SeamVectorEnv
  play.py           viewer so you can play (window, input, HUD)
  canvas.py         the 2-D drawing interface the HUD is written against (GPU canvas lives in glview.py)
  events.py         finds kills / hits / shots between snapshots (feeds effects and sound)
  audio.py          generative sound, synthesised by JAX
  glview.py         OpenGL view: floor textures, glow, particles, bloom, screen shake, all on the GPU
tests/              every test; `python tests/run_all.py`
  reference/        optional: check the JAX sim against the original JS game (needs playwright + that game)
tools/              benchmark, sample renderer
```

## Playing

WASD move, mouse aim and hold left button to fire (or I/J/K/L, or a gamepad). R restarts, Esc quits. Controls are in
screen space: up is always up, whatever the room does to your view. Pink wall: solid, never opens. Blue wall: opens
when the room is clear. Orange doors are on the second side of a wall. A flashing red door is where the Hunter arrives.
Stairs take you to the next floor (+1 health); dying starts a new run at floor 1.

**Startup:** the window opens immediately and shows what it is doing. The first launch compiles the simulation (about a minute on a CPU);
compiled programs are cached in `~/.cache/seamjax-jit`, so later launches start in a few seconds.

**Reading the world name** (debug overlay, e.g. `world 4.8.10 [hyper]`): the numbers are the regular polygons that meet at every corner, so `4.4.4.4` is 4 squares (a flat floor) and `4.4.4.4.4` is 5 squares. Their corner angles add up to exactly 360 deg on a flat floor, to more on a hyperbolic one (space curves away) and to less on a sphere (the world closes up). `grid:5x4:flip:none` is a wrapped 5 by 4 block of square rooms whose left/right edge wraps mirrored and whose top/bottom edge is a wall. The overlay spells this out on its second line.

**Debug mode:** `uv run seam --debug` shows an overlay (F1 toggles it): world type and corner-turn angle, room / start / exit,
view mirror, your section, wall kind, Hunter room and distance, enemies, frame timings, door labels (where each door leads, and which
side of a wall it is on) and the enemy spawn points. Keys while the overlay is on: **N** jump to the stairs, **H** heal, **G** god mode,
**K** kill all enemies in the room, **O** open this room's gated wall, **T** remove/restore the Hunter, **B** print floor info to the console.

## Gymnasium

```python
import gymnasium as gym, seamjax
env = gym.make("Seam-v0")                         # frame_skip=4, obs_mode="vector" | "pixels", start_floor=1
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

from seamjax import SeamVectorEnv
venv = SeamVectorEnv(num_envs=4096)               # one jitted call steps all of them (use a GPU)
obs, info = venv.reset(seed=0)
obs, reward, terminated, truncated, info = venv.step(actions)    # actions (num_envs, 4)
```

* Action: `[move_x, move_y, aim_x, aim_y]`, all in -1..1, in screen space. Firing happens while the aim stick is held.
* Observation: 194 floats, egocentric, no map: you and your view (rotation/mirror), the 12 sides (door, distance,
  Hunter warning, wall side), 8 nearest enemies, 8 nearest enemy bullets, Hunter, exit pad, nearest health pickup,
  inner wall (present, gated, open, normal, distance, your side). `obs_mode="pixels"` gives an RGB image instead.
* Reward: kill +0.1, hit taken -0.2, new room +0.05, health pickup +0.05, gate opened +0.05, stairs +5, death -5.
* The episode ends on death (`terminated`) or after `max_steps` (`truncated`). Taking the stairs does not end it:
  the run continues one floor deeper and `info["floor"]` shows where you are.
* The vector env autoresets (SAME_STEP): the last obs of a finished game is in `info["final_obs"]`.

## What is in the world

* Rooms are regular polygons on a surface of constant curvature (sphere, plane, hyperbolic plane), or the plane
  wrapped by translations / glide reflections (torus, Klein bottle). Doors follow the tiling, so walking round a corner
  turns you by the curvature of that corner. A mirrored seam flips your view, never your controls.
* Not every wall has a door. Some rooms have an inner wall: solid (permanent) or gated (opens when you clear your side).
  A solid wall splits a room into two places that are only joined through the rest of the world.
* The Hunter always takes the true shortest route through the world (section by section), warns at the door it will use,
  can be shot, and comes back.
* Health pickups: enemies sometimes drop one where they die (rate rises slowly with depth), walk over it to heal.

## Difficulty (difficulty.py)

| floor | worlds | dungeon size | rooms with walls | ranged enemies | Hunter sets off |
|---|---|---|---|---|---|
| 1 | flat only | 40% | 0% | 12% | 22 s |
| 2 | flat + wrapped | 52% | 6% | 15% | 20 s |
| 3 | + hyperbolic, spherical | 64% | 12% | 18% | 17 s |
| 5 | curved + wrapped mix | 88% | 24% | 24% | 11 s |
| 6+ | full mix | 100% | 30% | up to 45% | 10 s |

Enemy count, enemy speed and Hunter speed keep rising until floor 24.

## Notes

* Dungeons come only from `gen.py`, and the tilings it samples are built on the device by `specs.py`; nothing is exported from another program.
* `tests/specs_test.py` checks the device-built tables against an independent numpy construction (`tests/reference/specs_numpy.py`).
* `tests/reference/` records dungeons and scripted walks from the original JS game to check frame-for-frame agreement
  (`python tests/reference/export_pool.py path/to/index.html`, then `python tests/reference/parity.py`). Not needed to play or train.
* Speed on a laptop CPU is about 60k simulation steps per second (batched); on a GPU much more.
* Memory: building one dungeon needs about 4 MB of scratch, so dungeons are generated 64 at a time and only for games that
  just finished (`SEAM_GEN_CHUNK=256` on a big GPU for faster resets). 4096 parallel games need under 1 GB in total.
  JAX is told not to pre-reserve most of the GPU (`XLA_PYTHON_CLIENT_PREALLOCATE=false`); set it yourself to override.
* The viewer makes one device read per frame; `python -m seamjax.play --timing` prints where the frame time goes.

## OpenGL view

`play.py` draws the room on the GPU through `glview.py` (moderngl, OpenGL 3.3): a procedural floor shader per world type
(grid / hyperbolic rings and spokes / spherical dots / wrapped diagonals, tinted per room, with a light pool around you),
glowing walls, doors and obstacles, sprites for enemies, bullets, hearts, the Hunter and you, particle bursts for kills, hits
and pickups, bloom, screen shake, a red hit flash and a fade-in on a new floor. The HUD is on the GPU too: hearts, minimap and
debug overlay are distance-field shapes, and text is glyph quads placed by the GPU from a font atlas (the atlas is rendered once
at start-up with pygame's font, so text stays sharp at any window size). The HUD code is written against `canvas.py`, so the
same code also draws with pygame under `--software`. The view is always in screen space, so controls are never affected. The window can be resized (the
picture is letterboxed).

If OpenGL cannot start, the viewer says so and uses the old pygame renderer; `--software` forces that renderer.
`tests/gl_test.py` renders a synthetic room (run it under `xvfb-run -a` on a machine without a display; it skips if there is no OpenGL).


## Sound (audio.py)

All sound is generated by JAX at start-up (on the GPU if there is one) from formulas, nothing is recorded: sweeps, FM bells,
filtered noise and detuned oscillators. Each effect has 3 random variants so repeated shots do not sound identical. Kills ring
differently for purple and orange enemies, and effects are panned by where they happen on the screen. Each world type has its own
seamless 4 s drone (flat steady, hyperbolic beating, spherical airy, wrapped pulsing), and a heartbeat gets louder as the Hunter gets
closer. Synthesis runs in a background thread, so the window opens first and sound starts a few seconds later. `--mute` turns it off;
no sound device just means no sound. `python tools/render_sounds.py` writes the sounds to WAV files (`sampler.wav` plays them all).

## Minimap (debug mode)

With `--debug`, **M** cycles the minimap: whole dungeon / explored only / off, and **C** switches between a fixed map and one that
follows you (centred on your room, zoomed on its neighbours; start that way with `--map-follow`). Rooms are placed from the tiling's own
geometry (Poincare disk for hyperbolic worlds, azimuthal chart for spherical ones, plain plane otherwise) and the map is **turned to
match your view**: the doors of your room point on the map the way you see them on the screen. Every room shows its number, the same
number as in the door labels (`->17` means that door leads to room 17); your room's number is white, its neighbours' are yellow and
the links to them are drawn yellow. White ring = you, green ring = exit, red dot = Hunter, orange dot = a link that crosses a wrap seam
(not drawn as a line). The overlay's green dots are enemy spawn points (orange: the other side of a wall).

The minimap maths is in `minimap.py` and is tested in `tests/minimap_test.py`.
