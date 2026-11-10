"""Backend selection: auto / explicit / env override / fallback, each in a fresh process (JAX picks its backend once per process)."""
import _setup, os, sys, subprocess
import seamjax.backend as b
def run(code, **env):
    e = {k: v for k, v in os.environ.items() if k not in ('SEAM_BACKEND', 'JAX_PLATFORMS')}; e.update(env); e['PYTHONPATH'] = _setup.ROOT
    r = subprocess.run([sys.executable, '-c', 'import os\n'+code], env=e, capture_output=True, text=True); return r.returncode, r.stdout.strip().splitlines()[-1:] or [''], r.stderr
# pure decisions
assert b.auto_pick('linux') is None, 'on Linux JAX already prefers a working CUDA / ROCm GPU'
b._installed = lambda d: d == 'metaljax'; assert b.auto_pick('darwin') == 'metal,cpu' and b.auto_pick('linux') is None
b._installed = lambda d: False; assert b.auto_pick('darwin') is None
# apply_env
code = 'import seamjax.backend as b; b.apply_env(); print(os.environ.get("JAX_PLATFORMS", "<unset>"))'
for env, want in [({}, '<unset>'), ({'SEAM_BACKEND': 'cpu'}, 'cpu'), ({'SEAM_BACKEND': 'cuda'}, 'cuda,cpu'), ({'SEAM_BACKEND': 'ROCM'}, 'rocm,cpu'),
                  ({'SEAM_BACKEND': 'metal'}, 'metal,cpu'), ({'JAX_PLATFORMS': 'cpu'}, 'cpu')]:
    rc, out, err = run('import sys; sys.modules["seamjax"] = type(sys)("seamjax"); sys.modules["seamjax"].__path__ = [os.path.join(os.environ["PYTHONPATH"], "seamjax")]\n'+code, **env)
    assert rc == 0 and out[0] == want, (env, out, err[-300:])
rc, out, err = run('import sys; sys.modules["seamjax"] = type(sys)("seamjax"); sys.modules["seamjax"].__path__ = [os.path.join(os.environ["PYTHONPATH"], "seamjax")]\nimport seamjax.backend as b; b.apply_env()', SEAM_BACKEND='bogus')
assert rc != 0 and 'must be one of' in err, err[-300:]
# the CPU always works and reports itself; the float64 table build runs on the CPU
rc, out, err = run('import seamjax, seamjax.backend as b; i = b.ensure(); print(i["platform"], b.table_device().platform)', SEAM_BACKEND='cpu'); assert rc == 0 and out[0] == 'cpu cpu', (out, err[-300:])
# a platform that is not installed: explicit -> clear error with the extra to install; auto pick that fails -> falls back to the CPU with a note
import importlib.metadata as md
def have(k):
    for n in b.PLUGINS[k]:
        try: md.version(n); return True
        except md.PackageNotFoundError: pass
    return False
missing = [k for k in ('metal', 'rocm', 'cuda') if not have(k)]
if missing:
    k = missing[0]; rc, out, err = run('import seamjax, seamjax.backend as b; b.ensure()', SEAM_BACKEND=k); assert rc != 0 and 'uv sync --extra %s' % k in err, (k, err[-400:])
    rc, out, err = run('import seamjax.backend as b\nb.auto_pick = lambda platform=None: "%s,cpu"\nb.apply_env(); i = b.ensure(); print(i["platform"], "|", i["note"][:40])' % k)
    assert rc == 0 and out[0].startswith('cpu | could not start'), (out, err[-300:]); print('missing backend "%s": clear error and automatic fallback OK' % k)
print('PASS')
