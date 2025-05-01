import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.optimisers.observers.base import BaseObserver

class PseudoTimeMax(BaseObserver):
    name = 'dtaumax'
    objective = None

    @staticmethod
    def get_dtau_mats(intg):
        sect = 'solver-dual-time-integrator-multip'
        pintg = (intg.pseudointegrator.pintg if sect in intg.cfg.sections()
           else intg.pseudointegrator)
        return [mat.get() for mat in pintg.dtau_upts]

    @staticmethod
    def observation(intg):
        mats = PseudoTimeMax.get_dtau_mats(intg)
        dtaumax = np.array([max(np.max(m) for m in mats)], dtype=np.float64)
        comm, rank, root = get_comm_rank_root()
        comm.Allreduce(mpi.IN_PLACE, dtaumax, op=mpi.MAX)
        return float(dtaumax[0])
