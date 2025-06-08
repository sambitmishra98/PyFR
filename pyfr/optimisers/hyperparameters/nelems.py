# pyfr/optimisers/hyperparameters/nelems.py
"""
Hyper-parameter: number-of-elements per MPI rank (grouped).
Author: Sambit Mishra (feature-wait-split branch)
"""

import numpy as np
from pyfr.mpiutil import get_comm_rank_root
from pyfr.optimisers.hyperparameters.base import BaseHyperparameter


class Nelems(BaseHyperparameter):
    name = 'nelems'

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        # Original element counts (tuple[int]  length = comm.size)
        self._nelems_base = tuple(intg.nelems)

        comm, rank, _ = get_comm_rank_root()
        if self.n_hparams != comm.size:
            raise ValueError(f"[{self.name}] nhparams != {comm.size} ranks. ")


        if self.bounds.shape[1] != comm.size:
            raise ValueError(f"[{self.name}] bounds tuples "
                             f"must cover every rank ({comm.size})")

        if intg.nelems is None:
            raise ValueError(
                f"Integrator {intg.name} must have 'nelems' set before "
                f"hyperparameter '{self.name}' can be used."
            )

    @property
    def nelems_per_rank(self) -> tuple[int, ...]:
        ms = self.intg.mesh_specifications()
        # gather keys once; might be 'nelems-tri', 'nelems-quad', ...
        cols = [list(map(int, ms[k].split(',')))
                for k in ms if k.startswith('nelems-')]
        return tuple(int(np.sum(cols, axis=0)[i]) for i in range(len(cols[0])))        

    @property
    def param(self):
        return self.nelems_per_rank

    @property
    def hparam(self):
        return self.intg.nelems

    @hparam.setter
    def hparam(self, value):
        if len(value) != self.n_hparams:
            raise ValueError("Length of value must equal number of ranks")
        if any((v < 0) or (not float(v).is_integer()) for v in value):
            raise ValueError("All element counts must be non-negative ints")
        self.intg.nelems = tuple(int(v) for v in value)
        self.config_change = True