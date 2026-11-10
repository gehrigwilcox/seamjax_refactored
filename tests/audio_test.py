"""Sound synthesis test: every effect is finite, audible, not clipping, and the loops are seamless.  Also checks the event detector."""
import _setup, numpy as np
from seamjax import audio
from seamjax.events import Events
B = audio.bank(); L = audio.loops()
for n, v in B.items():
    assert v.shape[0] == audio.NVAR and np.isfinite(v).all(), n
    pk = np.abs(v).max(1); assert (pk > .2).all() and (pk <= 1.0).all(), (n, pk)
    assert abs(v[:, 0]).max() < .05 and abs(v[:, -1]).max() < .05, (n, 'clicks at the ends')
    assert np.abs(v[0]-v[1]).max() > .01, (n, 'variants identical')
    print('%-14s %.2f s  peak %.2f' % (n, v.shape[1]/audio.SR, pk.max()))
for n, v in L.items():
    assert np.isfinite(v).all() and np.abs(v).max() > .2, n
    step = np.abs(np.diff(v)).max(); seam = abs(v[0]-v[-1]); assert seam <= 3*step+1e-3, (n, 'seam', seam, step)
    print('%-14s loop %.1f s  seam %.4f (largest normal step %.4f)' % (n, len(v)/audio.SR, seam, step))
# events
ev = Events(); z = lambda n, v=0.: np.full(n, v)
def sn(**kw):
    en = np.zeros((10, 8)); en[0] = [100, 100, 0, 1, 0, 0, 0, 1]
    d = dict(en=en, hp=5, pb=np.zeros((24, 6)), hhp=18., hdead=0., x=10., y=20., hx=0., hy=0., hhere=False, aim=0.); d.update(kw); return d
assert ev.update(sn(), (0, 0), 0, False) == []
e2 = np.zeros((10, 8)); k = ev.update(sn(en=e2, hp=4), (0, 0), 0, False); assert [e['name'] for e in k] == ['hit', 'kill'] and k[1]['shooter'], k
assert [e['name'] for e in ev.update(sn(en=e2, hp=4), (1, 5), 1, False)] == ['descend']
print('events PASS\naudio PASS')
