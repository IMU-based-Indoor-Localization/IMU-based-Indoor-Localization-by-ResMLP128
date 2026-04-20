import contextlib

with contextlib.redirect_stdout(None):
    from numba import jit
