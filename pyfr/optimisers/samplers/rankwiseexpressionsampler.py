from __future__ import annotations
from typing import List
from pyfr.mpiutil import get_comm_rank_root
from pyfr.optimisers.samplers.base import BaseSampler


class RankwiseExpressionSampler(BaseSampler):
    """
    Reallocate each rank's *total* element count so that every rank
    ends up with the same compute-time, based purely on modeller data
    available on every rank (no MPI inside this method).
    """
    name = 'rankwiseexpression'

    # ------------------------------------------------------------------
    def sample(self) -> List[int]:
        nelems                 = list(self.modeller.X_processed[-1])
        cost_per_rank_per_elem = list(self.modeller.Y_processed[-1])

        cost_per_rank = [c*n for c, n in zip(cost_per_rank_per_elem, nelems)]

        nelems_total = sum(nelems)
        cost_total = sum(cost_per_rank)

        target = [cpr*nelems_total/cost_total for cpr in cost_per_rank]

        target = [round(n) for n in target]
        target[nelems.index(max(nelems))] += nelems_total - sum(target)
        
        return target
