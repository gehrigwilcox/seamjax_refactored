"""
The playable viewer (pygame window, OpenGL or software drawing, generated sound). Nothing here is needed to train.

    main.py       window, arguments and the game loop
    session.py    the game on the JAX device: stepping, the one read per frame, next floor, debug cheats (cheats.py)
    snapshot.py   what is read from the device: Snapshot (each frame), RoomDraw (each room), the minimap layout (each floor)
    input.py      keyboard / mouse / gamepad -> action; a demo bot
    hud.py        hearts, messages, death screen, debug overlay; minimap.py: the minimap (maths and drawing)
    gl/           OpenGL view: view.py (room, sprites, bloom), fx.py (particles), canvas.py (HUD on the GPU), shaders.py
    software.py   the room drawn with pygame on the CPU; canvas.py: the HUD drawing interface on a pygame surface
    events.py     kills, hits, shots... found by comparing snapshots (they drive the effects and the sound in audio.py)
"""
