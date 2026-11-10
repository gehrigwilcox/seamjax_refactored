"""Run every test script and print a summary. usage: python tests/run_all.py"""
import os, subprocess, sys, time
here = os.path.dirname(os.path.abspath(__file__))
names = ['golden_test.py', 'regen_test.py', 'specs_test.py', 'gen_test.py', 'gen_loops.py', 'spawn_test.py', 'wall_test.py', 'hunter_test.py', 'difficulty_test.py', 'gym_test.py', 'gl_test.py', 'audio_test.py', 'minimap_test.py', 'backend_test.py']
res = []
for n in names:
    if not os.path.exists(os.path.join(here, n)): continue
    t = time.time(); p = subprocess.run([sys.executable, os.path.join(here, n)], capture_output=True, text=True)
    ok = p.returncode == 0 and 'PASS' in p.stdout; res.append((n, ok, time.time()-t))
    print(('PASS ' if ok else 'FAIL '), n, '%.0fs' % (time.time()-t))
    if not ok: print(p.stdout[-1500:], p.stderr[-1500:])
print('reference (needs tests/reference/pool.npz): python tests/reference/parity.py')
sys.exit(0 if all(r[1] for r in res) else 1)
