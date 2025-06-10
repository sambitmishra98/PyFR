# pyfr/optimisers/observers/meshspecs.py

from collections import defaultdict
import math
import statistics

from pyfr.optimisers.observers.base import BaseObserver
from pyfr.mpiutil import get_comm_rank_root


class IntegratorSpecs(BaseObserver):
    name      = 'integratorspecs'
    objective = None

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)
        self.interval = intg.cfg.getint(cfgsect, 'capture-interval', 0)
        self._prev_nfevals = 0          # <── remember total RHS calls up to now

    def get_dofs(self, intg):
        """
            Exact degrees-of-freedom per rank for the current timestep.
        """
        comm, *_ = get_comm_rank_root()
        dofs_local = sum(intg.system.ele_ndofs)      # scalar on this rank
        return comm.allgather(dofs_local)            # list length P

    def get_dps(self, intg):
        """
        DOFs processed per second on each rank for the *last* accepted step.
        Processed-DOFs = local-dofs × (# RHS evaluations this step).
        """
        comm, *_ = get_comm_rank_root()

        # --- total RHS evaluations so far (all ranks keep identical counters)
        if hasattr(intg, 'stepper') and hasattr(intg.stepper, '_stepper_nfevals'):
            nfevals_total = intg.stepper._stepper_nfevals          # std form
        elif hasattr(intg, 'pseudointegrator'):                    # dual form
            nfevals_total = intg.pseudointegrator.pseudo_stepper_nfevals
        else:                                                      # fallback
            nfevals_total = intg.nsteps

        # Increment since last capture
        delta_nfe = nfevals_total - self._prev_nfevals
        self._prev_nfevals = nfevals_total

        # Rank-local work = DOFs × RHS calls × nvars  (per user request)
        dofs_local = sum(intg.system.ele_ndofs)
        nvars      = intg.system.nvars

        work_local = dofs_local * delta_nfe * nvars

        # Mean time for one RHS evaluation (seconds)
        ctime_mean, _ = self.compute_times(intg)

        # Total compute time for all evaluations executed since last capture
        total_ctime = ctime_mean * delta_nfe

        dps_local = work_local / total_ctime if total_ctime else 0

        return comm.allgather(dps_local)             # list length P
    
    def get_bps(self, intg):
        """
        Get the bytes per second of communication performed by integrator
        between every combination of two ranks,
        upon one time step completion.
        """
        return []

    def _init_observer(self, intg):
        # This is invoked twice by BaseObserver; guard duplicate prints
        if getattr(self, '_hdr_built', False):
            return
        self._hdr_built = True

        comm, *_ = get_comm_rank_root()
        cols = (['tprev', 'tcurr']
                + [f'dofs-{r}' for r in range(comm.size)]
                + [f'dps-{r}'  for r in range(comm.size)])
        self._init_history(len(cols), cols)

    # ------------------------------------------------------------
    def observation(self, intg):
        dofs = self.get_dofs(intg)
        dps  = self.get_dps(intg)
        return [*dofs, *dps]
    
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