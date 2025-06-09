# pyfr/optimisers/observers/rankcomputetime.py
from collections import defaultdict
import math
import statistics

from pyfr.optimisers.observers import BaseObserver
from pyfr.mpiutil import get_comm_rank_root

class RankComputeTime(BaseObserver):
    name = 'rankcomputetime'

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        if not intg.cfg.getbool('backend', 'collect-waitsome-times', False):
            raise RuntimeError('collect-waitsome-times must be True')

    # initialise column names (mean-0 … sem-(N-1))
    def _init_observer(self, intg):
        comm, _, _ = get_comm_rank_root()
        cols = ['tprev', 'tcurr'] + [f'mean-{r}' for r in range(comm.size)] + \
                                    [f'sem-{r}'  for r in range(comm.size)]
        self._init_history(len(cols), cols)

    # compute one row
    def observation(self, intg):
        return self.allgather_mean_sem(intg)   # list[float]

    def rhs_compute_times(self, intg):
        # Group together timings for graphs which are semantically equivalent
        times = defaultdict(list)
        for u, f in intg.system._rhs_uin_fout:
            for i, g in enumerate(intg.system._rhs_graphs(u, f)):
                times[i].extend(g.get_compute_times())

        # Compute all statistics
        stats = []
        for t in times.values():
            n = len(t)
            mean = statistics.mean(t) if t else 0
            sem  = statistics.stdev(t, mean) / math.sqrt(n) if len(t) >= 2 else 0
            stats.append((mean, sem))
        return stats

    def compute_times(self, intg):
        """
        List of tuples (mean, sem) by rank, summed across all graphs.
        """
        stats = self.rhs_compute_times(intg)
        mean = sum(s[0] for s in stats)
        sem = math.sqrt(sum(s[1]**2 for s in stats)) if len(stats) > 1 else 0
        return (mean, sem)

    def allgather_mean_sem(self, intg) -> list:
        """
        Return a single float (seconds) computed *only* from the current rank.
        """
        comm, rank, root = get_comm_rank_root()

        local_mean, local_sem = self.compute_times(intg)

        means = comm.allgather(local_mean)
        sems  = comm.allgather(local_sem)

        # Flat vector [mean-0, mean-1, …, sem-0, sem-1, …]
        return [*means, *sems]