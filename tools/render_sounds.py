"""Write every synthesised sound to WAV files so you can listen without starting the game.
    python tools/render_sounds.py [out_dir]     -> one file per sound plus sampler.wav (all of them in a row)"""
import _path, os, sys, wave
import numpy as np
from seamjax import audio
out = sys.argv[1] if len(sys.argv) > 1 else 'sounds'; os.makedirs(out, exist_ok=True)
def save(path, x):
    with wave.open(path, 'wb') as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(audio.SR); w.writeframes((np.clip(x, -1, 1)*32000).astype('<i2').tobytes())
B, L = audio.bank(), audio.loops(); seq = []; gap = np.zeros(int(.35*audio.SR))
for n, v in B.items(): save('%s/%s.wav' % (out, n), v[0]); seq += [v[0], gap]
for n, v in L.items(): save('%s/loop_%s.wav' % (out, n), np.tile(v, 2 if n != 'heartbeat' else 4)); seq += [np.tile(v, 1 if n != 'heartbeat' else 3), gap]
save('%s/sampler.wav' % out, np.concatenate(seq)); print('wrote %d files to %s/ (sampler.wav plays them all)' % (len(B)+len(L)+1, out))
