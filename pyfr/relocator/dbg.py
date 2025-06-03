# pyfr/relocator/utils/dbg.py
from functools import partial
import os, sys, inspect, datetime, mpi4py.MPI as _MPI

_rk = _MPI.COMM_WORLD.rank
_COL = ["34","32","31","35","33","36"]   # blue, green, red, magenta, yellow, cyan

def _stamp(msg):
    ts  = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    who = f"R{_rk}"
    return f"\x1b[1;{_COL[_rk % len(_COL)]}m[{ts} {who}] {msg}\x1b[0m"

def dbg(msg, **fmt):            # always visible
    print(_stamp(msg.format(**fmt)), file=sys.stderr)

def once(fn):
    """Decorator that prints entry/exit + run-time."""
    import time, functools
    @functools.wraps(fn)
    def _wrap(*a, **k):
        dbg("↳ {f}()", f=fn.__qualname__)
        t0 = time.time()
        out = fn(*a, **k)
        dbg("↰ {f}  (Δ={dt:.3f}s)", f=fn.__qualname__, dt=time.time()-t0)
        return out
    return _wrap
