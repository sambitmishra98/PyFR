from collections import defaultdict
import math
import statistics

from pyfr.optimisers.observers import BaseObjective
from pyfr.mpiutil import get_comm_rank_root

class RankComputeTime(BaseObjective):
    name = 'rankcomputetime'
    objective = 'minimise'

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        # If wait-some not enabled, raise error
        if not intg.cfg.getbool('backend', 'collect-waitsome-times', False):
            raise ValueError(
                'rankcomputetime requires wait-some times collection.'
                'Verify wait-split branch addition too. ')

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
        Return a list of tuples (mean, sem) for all graphs.
        This is used to compute the objective function.
        """
        stats = self.rhs_compute_times(intg)

        # mean overall is sum of means
        mean = sum(s[0] for s in stats)
        
        # sem overall is sqrt of sum of variances
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