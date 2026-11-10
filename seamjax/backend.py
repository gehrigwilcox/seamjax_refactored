"""
Which JAX backend (CPU / NVIDIA CUDA / AMD ROCm / Apple Metal) Seam runs on.

By default it is automatic:
  * CUDA and ROCm: JAX itself prefers a working GPU plugin over the CPU, so installing the extra (`uv sync --extra cuda` or `--extra rocm`)
    is all that is needed.  If the plugin is installed but no GPU can start, it runs on the CPU and says so.
  * Metal: the Metal plugin leaves the CPU as JAX's default, so on an Apple-silicon Mac with the `metal` extra installed Seam asks for
    `metal,cpu` itself.  If Metal fails to start it quietly falls back to the CPU.
Override with  SEAM_BACKEND=cpu|cuda|rocm|metal  (or `seam --device ...`).  An explicit choice that cannot start is an error that says
which extra to install.  Setting JAX_PLATFORMS yourself always wins in `auto` mode.

The one-off float64 tiling build needs float64, which Metal does not have; table_device() sends it to the CPU in that case.
"""
import os, sys
import importlib.metadata as md

PLATFORMS = ('cpu', 'cuda', 'rocm', 'metal')
PLUGINS = {'cuda': ('jax-cuda13-plugin', 'jax-cuda12-plugin'), 'rocm': ('jax-rocm7-plugin',), 'metal': ('metaljax',)}
HINT = {'cuda': 'uv sync --extra cuda  (CUDA 13 driver; --extra cuda12 for CUDA 12)', 'rocm': 'uv sync --extra rocm  (needs ROCm 7 already installed, Linux)',
        'metal': 'uv sync --extra metal  (Apple silicon Mac, experimental)', 'cpu': 'uv sync'}
_state = {}


class BackendError(RuntimeError): pass


def _installed(dist):
    try: md.version(dist); return True
    except md.PackageNotFoundError: return False


def auto_pick(platform=None):
    """platform string for JAX_PLATFORMS to use in auto mode, or None to leave JAX's own choice (which already prefers a working CUDA / ROCm GPU)"""
    if (platform or sys.platform) == 'darwin' and _installed('metaljax'): return 'metal,cpu'
    return None


def _set_platforms(value):
    os.environ['JAX_PLATFORMS'] = value
    if 'jax' in sys.modules: sys.modules['jax'].config.update('jax_platforms', value)       # jax already imported: the config copy must change too


def apply_env():
    """call before JAX starts a backend (seamjax/__init__ does)"""
    want = os.environ.get('SEAM_BACKEND', 'auto').strip().lower() or 'auto'; _state.clear(); _state.update(want=want, picked=None)
    if want == 'auto':
        if not os.environ.get('JAX_PLATFORMS'):
            pick = auto_pick()
            if pick: _set_platforms(pick); _state['picked'] = pick
    elif want in PLATFORMS: _set_platforms('cpu' if want == 'cpu' else want+',cpu')
    else: raise BackendError("SEAM_BACKEND / --device must be one of: auto, %s (got '%s')" % (', '.join(PLATFORMS), want))


def _clear():
    import jax
    try:
        from jax.extend import backend as xb; xb.clear_backends()
    except Exception: jax.clear_backends()


def ensure():
    """start the backend now (so a failure shows up here, with advice) and return a dict: platform, kind, count, note"""
    if 'info' in _state: return _state['info']
    import jax, jax.numpy as jnp
    if 'want' not in _state: apply_env()
    note = ''; want = _state['want']; asked = want if want != 'auto' else (_state['picked'] or '').split(',')[0]       # what was asked for, if anything
    def probe():
        jax.block_until_ready(jnp.arange(8.)*2.); d = jax.devices()[0]
        if asked not in ('', 'cpu') and not (d.platform.lower() == asked or (d.platform.lower() == 'gpu' and asked in ('cuda', 'rocm'))):      # JAX can quietly give the CPU
            raise RuntimeError("no %s device started (JAX gave: %s)" % (asked, d.platform))
        return d
    try: dev = probe()
    except Exception as ex:
        if want == 'auto' and _state['picked']:                                       # our own automatic choice did not start: use the CPU
            _set_platforms('cpu'); _clear(); asked = 'cpu'; dev = probe(); note = "could not start '%s' (%s); using the CPU" % (_state['picked'].split(',')[0], str(ex).splitlines()[0][:90])
        else:
            raise BackendError("backend '%s' could not start: %s\n  install it with: %s\n  or run on the CPU with --device cpu / SEAM_BACKEND=cpu" % (
                want if want != 'auto' else os.environ.get('JAX_PLATFORMS', '?'), str(ex).splitlines()[0][:160], HINT.get(want, HINT['cuda']))) from ex
    if dev.platform.lower() == 'cpu' and not note and _state['want'] in ('auto',):
        have = [n for k, v in PLUGINS.items() for n in v if _installed(n)]
        if have: note = '%s is installed but no device could start, so this runs on the CPU' % have[0]
    _state['info'] = dict(platform=dev.platform, kind=str(getattr(dev, 'device_kind', '')), count=jax.device_count(), note=note); return _state['info']


def describe():
    i = ensure(); s = '%s%s' % (i['platform'], ' - '+i['kind'] if i['kind'] and i['kind'].lower() != i['platform'] else '')
    return s+(' x%d' % i['count'] if i['count'] > 1 else '')


def table_device():
    """device for the one-off float64 tiling build: the default device if it can compute in float64 (CPU, CUDA, ROCm), else the CPU (Metal)"""
    import jax, jax.numpy as jnp
    ensure()
    try:
        with jax.enable_x64(True): jax.block_until_ready(jnp.arange(4, dtype=jnp.float64)*2)
        return jax.devices()[0]
    except Exception: return jax.devices('cpu')[0]
