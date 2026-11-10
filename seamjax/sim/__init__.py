"""
The simulation: pure functions of a GameState.

    step.py       one frame: the systems in order -> (state, StepEvents)
    systems/      player, hunter, transit (doors), enemies, projectiles, pickups, rooms (gates, stairs)
    queries.py    read-only questions (sections, Hunter nodes, which door the Hunter comes through)
    state.py      shared building blocks (empty pools and events, conditional update, taking a hit)
    reward.py     RL reward from the events
    observe.py    RL observation
    game.py       Game: reset, next floor / new run, auto-resetting single and batched steps
    batch.py      chunked vmap, regenerating only finished games
"""
