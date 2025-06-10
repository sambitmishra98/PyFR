# pyfr/optimisers/observers/rankwaittime.py

import numpy as np

from pyfr.optimisers.observers import BaseObserver
from pyfr.mpiutil import get_comm_rank_root


class RankWaitTime(BaseObserver):
    """
    Observer: rank-wise MPI wait-some statistics.
    Columns: send-mean-r, recv-mean-r, send-sem-r, recv-sem-r.
    """
    name = 'rankwaittime'

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        if not intg.cfg.getbool('backend', 'collect-waitsome-times', False):
            raise RuntimeError('collect-waitsome-times must be True')

    # ------------------------------------------------------------------
    def _init_observer(self, intg):
        comm, *_ = get_comm_rank_root()
        P = comm.size

        cols = ['tprev', 'tcurr']

        for strs in ('send-mean', 'recv-mean', 'send-sem', 'recv-sem'):
            for i in range(P):
                for j in range(P):  
                    cols.append(f'{strs}-{i}-{j}')

        self._init_history(len(cols), cols)

    # ------------------------------------------------------------------

    def allgather_mean_sem(self, intg):
        comm, rank, _ = get_comm_rank_root()
        P = comm.size

        def collapse(stage_arrays):
            mean = np.zeros(P)
            sem2 = np.zeros(P)

            for arr in stage_arrays:
                mean += arr[:, 0]
                sem2 += arr[:, 1]**2

            return mean.tolist(), np.sqrt(sem2).tolist()

        # Local rows (this rank is the source row i)
        smean_row, ssem_row = collapse(intg.system.rhs_wait_times_send())
        rmean_row, rsem_row = collapse(intg.system.rhs_wait_times_recv())

        return (list(it for row in comm.allgather(smean_row) for it in row) +
                list(it for row in comm.allgather(rmean_row) for it in row) +
                list(it for row in comm.allgather(ssem_row)  for it in row) +
                list(it for row in comm.allgather(rsem_row)  for it in row))

    def _local_stats(self, intg):
        comm, rank, _ = get_comm_rank_root()
        wsmean, wssem = self._collapse(intg.system.rhs_wait_times_send(), rank)
        wrmean, wrsem = self._collapse(intg.system.rhs_wait_times_recv(), rank)
        return wsmean, wrmean, wssem, wrsem

    def observation(self, intg):
        return self.allgather_mean_sem(intg)
