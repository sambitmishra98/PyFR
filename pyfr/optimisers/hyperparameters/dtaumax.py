import numpy as np

from pyfr.optimisers.hyperparameters.base import BaseHyperparameter
from pyfr.mpiutil import get_comm_rank_root, mpi

class PseudoTimeMax(BaseHyperparameter):
    name = 'dtaumax'

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        mgsect = 'solver-dual-time-integrator-multip'
        if mgsect not in intg.cfg.sections():
            self.pintg = intg.pseudointegrator
        else:
            self.pintg = intg.pseudointegrator.pintg

        # Verify that self.pintg.dtau_max variable exists
        if not hasattr(self.pintg, 'dtau_max'):
            raise NotImplementedError(
            f"dtau_max variable not found in {self.pintg.__class__.__name__}"
            f"Possible missing functionality: feature/variable-dt."
            f"https://github.com/sambitmishra98/PyFR/tree/feature/variable-dt"
            )

    @property
    def dtau_mats(self):
        return [mat.get() for mat in self.pintg.dtau_upts]

    @property
    def param(self):
        comm, rank, root = get_comm_rank_root()
        
        buf = np.array([np.max([np.max(m) for m in self.dtau_mats])], dtype='d')
        comm.Allreduce(mpi.IN_PLACE, buf, op=mpi.MAX)

        return [float(buf[0]),]

    @property
    def hparam(self):
        return np.atleast_1d(self.pintg.dtau_max)

    @hparam.setter
    def hparam(self, y):
        self.config_change = True
        self.pintg.dtau_max = y
