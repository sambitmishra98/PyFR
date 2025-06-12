# pyfr/optimisers/hyperparameters/nelems.py
from __future__ import annotations

import numpy as np
from pyfr.mpiutil import get_comm_rank_root
from pyfr.optimisers.hyperparameters.base import BaseHyperparameter


class Nelems(BaseHyperparameter):
    """
    Hyper-parameter: number-of-elements per MPI rank
    (tuple[int] of length comm.size) One scalar integer hyper-parameter per rank.
    """
    name = 'nelems'

    # ------------------------------------------------------------------ #
    def __init__(self, intg, cfgsect: str, suffix: str | None = None):
        super().__init__(intg, cfgsect, suffix)

        # ------------------------------------------------------------------
        # Basic consistency checks
        comm, rank, _ = get_comm_rank_root()

        if self.n_hparams != comm.size:
            raise ValueError(f'[nelems] n-hyperparameters ({self.n_hparams}) '
                             f'must equal MPI size ({comm.size})')

        if self.bounds.shape[1] != comm.size:      # bounds = (4, d)
            raise ValueError('[nelems] soft/hard bounds must specify a pair '
                             'for each rank')

        if intg.nelems is None:
            raise RuntimeError('Integrator must expose .nelems before '
                               'nelems hyper-parameter can operate')

        # Keep a copy of the original distribution (may be handy later)
        self._nelems_base = tuple(intg.nelems)

    @property
    def param(self) -> tuple[int, ...]:
        """Read-only."""
        return self.intg.nelems

    @property
    def hparam(self) -> tuple[int, ...]:
        """Alias for tuning"""
        return self.intg.nelems

    @hparam.setter
    def hparam(self, value):
        value = np.asarray(value, dtype=float)

        if value.size != self.n_hparams:
            raise ValueError('length mismatch: expected '
                             f'{self.n_hparams}, got {value.size}')
        if np.any(value < 0) or np.any(value != np.floor(value)):
            raise ValueError('element counts must be non-negative integers')

        # Check sum of elements of each type remains the same before and after
        if np.sum(value) != np.sum(self._nelems_base):
            raise ValueError('Inconsistent total number of elements '
                            f'({np.sum(value)} != {np.sum(self._nelems_base)})')

        # Apply
        self.intg.nelems = tuple(int(v) for v in value)
        self.config_change = True            # shared flag → integrator notices
