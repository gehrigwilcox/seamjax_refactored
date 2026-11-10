"""
Golden trace: the simulation and the dungeon generator must keep producing exactly what they produced when the trace was
recorded. This is the safety net for refactors: a pure restructuring must not change a single number.

    python tests/golden_test.py            # compare against tests/golden/trace.npz
    python tests/golden_test.py --record   # re-record (only when a behaviour change is intended; say so in the commit)

It runs on the CPU (forced below), because floating-point results differ between devices. What is recorded and how it is
read from the state lives in tests/golden/canon.py.
"""
import os, sys
os.environ['SEAM_BACKEND'] = 'cpu'
import _setup
import time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden'))
import canon

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden', 'trace.npz')
t0 = time.time(); got = canon.run(); print('trace computed in %.0fs' % (time.time()-t0))

if '--record' in sys.argv:
    np.savez_compressed(PATH, **got)
    st = {k: got['step/'+k] for k in ('done', 'descended', 'kills', 'room', 'hp')}
    print('recorded %d arrays -> %s (%.1f MB)' % (len(got), PATH, os.path.getsize(PATH)/1e6))
    print('coverage: deaths %d, stairs %d, kills %d, room changes %d, hits taken %d' % (
        st['done'].sum(), st['descended'].sum(), st['kills'][-1].sum(), (np.diff(st['room'], axis=0) != 0).sum(), (np.diff(st['hp'], axis=0) < 0).sum()))
    sys.exit(0)

ref = dict(np.load(PATH))
problems = []
for k in sorted(set(ref) | set(got)):
    if k not in got: problems.append('missing in new trace: %s' % k); continue
    if k not in ref: problems.append('not in recorded trace: %s' % k); continue
    a, b = ref[k], got[k]
    if a.shape != b.shape: problems.append('%s: shape %s, recorded %s' % (k, b.shape, a.shape)); continue
    a = a.astype(np.float64) if a.dtype == bool else a; b = b.astype(np.float64) if b.dtype == bool else b
    if np.array_equal(a, b): continue
    diff = np.abs(a.astype(np.float64)-b.astype(np.float64)); where = np.argwhere(diff > 0)[0]
    msg = '%s: %d values differ (max %.3g), first at index %s: %r vs recorded %r' % (k, (diff > 0).sum(), diff.max(), tuple(where), b[tuple(where)], a[tuple(where)])
    if k.startswith('step/'): msg += '  [step %d, game %d]' % (where[0], where[1])
    problems.append(msg)
if problems:
    print('\n'.join(problems[:40])); print('%d arrays differ from the recorded trace' % len(problems)); sys.exit(1)
print('all %d arrays identical to the recorded trace' % len(ref))
print('PASS')
