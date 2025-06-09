# pyfr/optimisers/hyperparameters/base.py
"""
Common base-class for all hyper-parameter helpers.
"""

from __future__ import annotations

import pathlib
from typing import Sequence, List

import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.optimisers.base import HistoryMixin, FlagSyncMixin, BoundsMixin


class BaseHyperparameter(FlagSyncMixin, BoundsMixin, HistoryMixin):
    name: str = None

    def __init__(self, intg, cfgsect: str, suffix: str | None = None):
        # Shared flags
        FlagSyncMixin.__init__(self, intg, suffix)

        self.intg, self.cfgsect, self.suffix = intg, cfgsect, suffix

        # Number of dimensions
        self.n_hparams = intg.cfg.getint(cfgsect, 'n-hyperparameters')

        # Parse bounds (prints them)
        self._init_bounds(intg.cfg, cfgsect, d_expected=self.n_hparams)

        # History initialisation (tprev, tcurr, params…)
        hdr: List[str] = ['tprev', 'tcurr'] + [
            f'param{i}' for i in range(self.n_hparams)
        ]
        self._init_history(len(hdr), hdr)

        self._tprev = intg.tcurr

        # Capture cadence
        self.interval = intg.cfg.getint(cfgsect, 'capture-interval', 0)

        # Root-rank CSV
        comm, rank, _ = get_comm_rank_root()
        self._csv_path: pathlib.Path | None = None
        if rank == 0 and intg.cfg.hasopt(cfgsect, 'file'):
            p = pathlib.Path(intg.cfg.get(cfgsect, 'file')).with_suffix('.csv')
            self._csv_path = p

        # (Optional) external sampler pushes into this list
        self._pending_updates: list[np.ndarray] = []

    # ------------------------------------------------------------------ #
    # expected from subclasses:

    #   @property
    #   def param(self)  -> tuple[float, ...]: ...

    #   @property
    #   def hparam(self): ...

    #   @hparam.setter
    #   def hparam(self, value): ...

    # ------------------------------------------------------------------ #
    def __call__(self):
        """
        One call per integrator step:
            1. capture row?  → append + dump
            2. pending update? → apply
        """
        nsteps = self.intg.nsteps

        if self.should_capture(nsteps):
            self.append_row([self._tprev, self.intg.tcurr, *self.param])
            if self._csv_path:
                self.dump_csv(self._csv_path)
            self._tprev = self.intg.tcurr

        # ..............................................................
        if self._pending_updates:
            cand = np.asarray(self._pending_updates.pop(0), dtype=float)
            print(f'[HP-{self.name}] apply candidate {cand}', flush=True)
            self.hparam = cand
            self.config_change = True

    # ------------------------------------------------------------------ #
    # small utility kept for subclasses relying on grouping

    @staticmethod
    def condense_by_group(full: Sequence[float],
                          groups: Sequence[int]) -> np.ndarray:
        arr = np.asarray(full, dtype=float)
        grp_first = {}
        for idx, g in enumerate(groups):
            if g >= 0 and g not in grp_first:
                grp_first[g] = idx
        gmax = max(grp_first, default=-1) + 1
        out = np.empty(gmax, dtype=float)
        for g, idx in grp_first.items():
            out[g] = arr[idx]
        return out

    @property
    def param(self):
        raise NotImplementedError('child must implement .param')
