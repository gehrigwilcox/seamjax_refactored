import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# keep compiled programs on disk between test runs (the same cache the viewer uses): a repeat run reads them instead of
# compiling again. SEAM_TEST_NO_CACHE=1 turns it off.
if not os.environ.get('SEAM_TEST_NO_CACHE'):
    os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', os.path.join(os.path.expanduser('~'), '.cache', 'seamjax-jit'))
    os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '0')
    os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES', '0')
