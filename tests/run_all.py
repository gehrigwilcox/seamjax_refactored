"""Run the test scripts and print a summary.

    python tests/run_all.py            # every test, several at a time
    python tests/run_all.py --quick    # a few minutes: golden trace, generator, spawning, viewer pieces
    python tests/run_all.py -j 1       # one at a time (-j N: N at once; default 4)

Compiled programs are cached on disk (tests/_setup.py), so a repeat run is much faster than the first one after a change.
"""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
here = os.path.dirname(os.path.abspath(__file__))
# slowest first, so the long ones start right away
ALL = ['gym_test.py', 'wall_test.py', 'hunter_test.py', 'specs_test.py', 'regen_test.py', 'difficulty_test.py', 'gen_loops.py', 'golden_test.py',
       'spawn_test.py', 'gen_test.py', 'minimap_test.py', 'audio_test.py', 'backend_test.py', 'gl_test.py']
QUICK = ['golden_test.py', 'spawn_test.py', 'gen_test.py', 'minimap_test.py', 'audio_test.py', 'backend_test.py', 'gl_test.py']
args = sys.argv[1:]
jobs = int(args[args.index('-j')+1]) if '-j' in args else 4
names = [n for n in (QUICK if '--quick' in args else ALL) if os.path.exists(os.path.join(here, n))]

def run(n):
    t = time.time(); p = subprocess.run([sys.executable, os.path.join(here, n)], capture_output=True, text=True)
    ok = p.returncode == 0 and 'PASS' in p.stdout
    print(('PASS ' if ok else 'FAIL '), n, '%.0fs' % (time.time()-t), flush=True)
    if not ok: print(p.stdout[-1500:], p.stderr[-1500:], flush=True)
    return ok

t0 = time.time()
with ThreadPoolExecutor(jobs) as ex: res = list(ex.map(run, names))
print('%d/%d passed in %.0fs (%d at a time)' % (sum(res), len(res), time.time()-t0, jobs))
print('reference (needs tests/reference/pool.npz): python tests/reference/parity.py')
sys.exit(0 if all(res) else 1)
