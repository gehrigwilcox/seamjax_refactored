# Assets

Optional sound files and images that `seamjax/theme.py` can point to; paths in the theme are relative to this folder.

- `sounds/`: wav or ogg, used by `Sound(file='sounds/name.wav')` (a tuple of files = random variants). A looping sound
  (a world's drone, the heartbeat) should loop seamlessly.
- `textures/`: png. Sprites (`Look(texture='textures/name.png')`) are drawn centred on the thing, 2 x its radius wide,
  turned with it and tinted by the look's colour (white = as drawn); transparent pixels show what is behind. Floors
  (`WorldLook(texture='textures/name.png', texture_scale=128.)`) repeat every `texture_scale` pixels of the room.

Nothing here is required: the default theme is fully procedural (shapes, shaders and JAX-synthesised sound).
