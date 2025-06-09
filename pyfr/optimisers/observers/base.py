# pyfr/optimisers/observers/base.py
from __future__ import annotations

import numpy as np
from pyfr.mpiutil import get_comm_rank_root
from pyfr.optimisers.base import HistoryMixin, FlagSyncMixin


class BaseObserver(FlagSyncMixin, HistoryMixin):
    """
    Every concrete observer computes one *row* (often per-MPI-rank data)
    whenever `should_capture()` is true.

    Row layout is decided by the child class; it must call
    `self._init_history(n_cols, colnames)` once in __init__.
    """
    name = None               # plug-in identifier

    # .................................................................
    def __init__(self, intg, cfgsect: str, suffix: str | None = None):
        FlagSyncMixin.__init__(self, intg, suffix)
        self.cfg, self.cfgsect, self.suffix = intg.cfg, cfgsect, suffix
        self.tprev = intg.tcurr

        # let the concrete class tell us the column names right now
        self._init_observer(intg)          # <── MUST call _init_history(...)

        # CSV ----------------------------------------------------------
        comm, rank, _ = get_comm_rank_root()
        if rank == 0 and intg.cfg.hasopt(cfgsect, 'file'):
            self._csv = self.init_csv(intg.cfg, cfgsect,
                                      header=','.join(self._cols_hdr))
        else:
            self._csv = None

        self._init_observer(intg)

    # -----------------------------------------------------------------
    # Public entry point called by the integrator every time-step

    def __call__(self, intg):
        """
        Concrete observer must implement:
            row = self._compute_row(intg)   (returns list / tuple of floats)
        """
        if not self.should_capture(intg.nsteps):
            return

        row_body = self._compute_row(intg)
        row = [self.tprev, intg.tcurr, *row_body]
        self.append_row(row)

        # dump immediately (cheap – one row only)
        if self._csv:
            print(*row, sep=',', file=self._csv)
            self._csv.flush()

        self.tprev = intg.tcurr      # advance window

    # -----------------------------------------------------------------
    # Child classes must override the following two hooks

    def _init_observer(self, intg):
        """Called once from the concrete observer __init__ if needed."""
        pass

    def _compute_row(self, intg) -> list[float]:
        """Return the numeric row to be stored/written."""
        raise NotImplementedError('observer must implement _compute_row')
