import _setup, os, sys
import numpy as np
sys.path.insert(0, os.path.join(_setup.ROOT, 'tests', 'reference'))
"""The tiling tables built on the device equal the independent numpy construction, and satisfy the invariants."""
import specs_numpy
from seamjax.specs import build_specs
T, meta = build_specs(); O, _ = specs_numpy.build_specs()
for k in O:
    assert O[k].shape == T[k].shape, (k, O[k].shape, T[k].shape)
    assert np.allclose(O[k], T[k], atol=1e-4), k
print('device-built tables equal the numpy reference:', len(O), 'tables,', T['n'].shape[0], 'worlds')
# invariants: gluing is symmetric (the neighbour's neighbour is me, flips and seams agree)
N, R, S = T['nbr'].shape
for i in range(N):
    for r in range(T['nrooms'][i]):
        for k in range(T['n'][i, r]):
            b = T['nbr'][i, r, k]
            if b < 0: continue
            j = T['nbs'][i, r, k]
            assert T['nbr'][i, b, j] == r and T['nbs'][i, b, j] == k and T['flip'][i, b, j] == T['flip'][i, r, k] and T['seam'][i, b, j] == T['seam'][i, r, k]
# every stored corner loop closes with the right length
for i in range(N):
    for c in range(T['cyc_room'].shape[1]):
        l = T['cyc_len'][i, c]
        if l: assert 3 <= l <= 8 and (T['cyc_room'][i, c, :l] < T['nrooms'][i]).all()
print('PASS')
