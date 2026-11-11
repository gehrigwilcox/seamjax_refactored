"""
Play Seam on the JAX simulation (the same one the RL environments use).

    python -m seamjax.play [--floor N] [--seed S] [--bot]

Move: WASD or arrow keys   Aim + fire: mouse (hold left button), or I/J/K/L, or a gamepad   R: new run   Esc: quit.
Controls are always in screen space: up is up, whatever the room does to the view.
Pink wall = solid (never opens), blue wall = gated (opens when the room is clear). Orange doors are on the second
side of a wall. A red flashing door is where the Hunter will come through.

Debug mode (python -m seamjax.play --debug): F1 toggles the overlay (floor, world type, room, section, Hunter, enemies, timings,
door labels "room id / side of wall", spawn points) and these keys work:
  N  jump to the stairs of this floor (clears the exit room first)      H  heal to full            G  toggle god mode
  K  kill every enemy in the room       O  open this room's gated wall      T  remove / restore the Hunter
  M  minimap: off / explored / whole dungeon (turned to match your view; room numbers match the door labels)
  C  minimap follows you (centred on your room) or stays fixed        (start that way with --map-follow)
  B  print this floor's tiling name and corner-turn angle to the console
--mute turns the sound off.
Other flags: --class gunner|scatter|lancer|blade (player class, see core/kinds.py), --timing (frame-time breakdown in the console),
--floor N, --seed S, --bot.

Drawing: the room is drawn by the GPU through OpenGL (app/gl/); --software uses the pygame CPU renderer, and is also the automatic fallback.
Sound is synthesised by JAX (app/audio.py). The viewer itself lives in seamjax/app (main.py is the loop).
"""
from seamjax.app.main import main

if __name__ == '__main__':
    main()
