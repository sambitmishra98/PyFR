import time

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi, scal_coll
from pyfr.readers.native import Solution
from pyfr.rebalance import DiffusionBalancer, make_exchangers, rebuild_mesh


class RebalanceMixin:
    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        sect = 'solver-rebalance'

        # Enable wait-time collection before the system is constructed
        if cfg.hasopt(sect, 'threshold'):
            cfg.set('backend', 'collect-wait-times', 'true')

        super().__init__(backend, systemcls, mesh, initsoln, cfg)

        comm, _, _ = get_comm_rank_root()
        self._rebal_on = comm.size >= 2 and self.cfg.hasopt(sect, 'threshold')
        if not self._rebal_on:
            return

        self._rebal_threshold = self.cfg.getfloat(sect, 'threshold', 1.3)
        self._rebal_ewma_alpha = self.cfg.getfloat(sect, 'ewma-alpha', 0.3)
        self._rebal_base_cd = self.cfg.getint(sect, 'cooldown', 50)
        self._rebal_max_cd = self.cfg.getint(sect, 'max-cooldown', 5000)
        self._rebal_nsteps = self.cfg.getint(sect, 'nsteps', 100)
        self._rebal_detail_window = self.cfg.getint(sect, 'detail-window', 20)

        # Hardware class identifier for per-class cost estimation
        self._rebal_cls = f'{self.backend.name}:{self.backend.platform_id}'

        # Runtime state
        self._rebal_ewma = 0.0
        self._rebal_last_t = time.perf_counter()
        self._rebal_cd = self._rebal_base_cd
        self._rebal_step = 0
        self._rebal_cycle = 0
        self._rebal_stall = 0
        self._rebal_trigger = 0
        self._rebal_trivial = 0
        self._rebal_settled = False
        self._rebal_did_proactive = False

        # Cache cost class info (refreshed after each rebalance)
        self._rebal_refresh_costs()

    def _rebal_refresh_costs(self):
        comm, _, _ = get_comm_rank_root()
        names, counts, static = self.system.element_cost_classes()
        etypes = self.system.mesh.etypes
        nmap = {n: i for i, n in enumerate(names)}

        # Fixed-length vectors aligned to global etype order
        self._rebal_counts = np.zeros(len(etypes), dtype=int)
        self._rebal_static = np.ones(len(etypes))
        for i, et in enumerate(etypes):
            if (j := nmap.get(et)) is not None:
                self._rebal_counts[i] = counts[j]
                self._rebal_static[i] = static[j]

        # Gather fixed per-rank info: hardware class + counts matrix
        all_cls = comm.allgather(self._rebal_cls)
        all_counts = np.array(comm.allgather(self._rebal_counts))

        # Pre-build the lstsq matrix and EWMA buffer for our hardware class
        self._rebal_cls_mask = [c == self._rebal_cls for c in all_cls]
        self._rebal_A = all_counts[self._rebal_cls_mask]
        self._rebal_all_ewma = np.empty(comm.size)

    def _rebal_load_bounds(self, my_load):
        comm, _, _ = get_comm_rank_root()
        hi = scal_coll(comm.Allreduce, my_load, op=mpi.MAX)
        lo = scal_coll(comm.Allreduce, my_load, op=mpi.MIN)
        return hi, lo

    def _rebal_mode(self, weights):
        hi, lo = self._rebal_load_bounds(weights.sum())

        # Severe imbalance; full repartition
        if lo > 0 and hi / lo > 2.0:
            mode = 'repartition'
        # First cycle; aggressive diffusion
        elif self._rebal_cycle == 0:
            mode = 'aggressive'
        # Subsequent cycles; fine-grained diffusion
        else:
            mode = 'fine'

        return mode, hi, lo

    def _rebal_measure(self):
        now = time.perf_counter()
        wall = now - self._rebal_last_t
        self._rebal_last_t = now

        # Compute time = wall minus MPI and collective waits
        graph_wait = self.system.pop_wait_time()
        coll_wait = self.pop_coll_wait()
        gpu_time = self.system.pop_gpu_elapsed()

        compute = max(wall - graph_wait - coll_wait, gpu_time or 0, 0)

        # Exponentially weighted moving average
        a = self._rebal_ewma_alpha
        if self._rebal_ewma == 0:
            self._rebal_ewma = compute
        else:
            self._rebal_ewma = a*compute + (1 - a)*self._rebal_ewma

        self.system.pop_per_neighbour_wait()

        return self._rebal_ewma

    def _rebal_weights(self):
        counts = self._rebal_counts

        # Before any runtime data, use statistical estimates
        if self._rebal_ewma <= 0:
            return np.repeat(self._rebal_static, counts)

        # Gather per-rank EWMA values
        comm, rank, _ = get_comm_rank_root()
        ewma = self._rebal_all_ewma
        ewma[rank] = self._rebal_ewma
        comm.Allgather(mpi.IN_PLACE, ewma)

        # Solve A*costs = b for our hardware class
        A, b = self._rebal_A, ewma[self._rebal_cls_mask]
        costs, _, mrank, _ = np.linalg.lstsq(A, b, rcond=None)

        # Fall back to uniform if rank-deficient or any cost < 1
        if mrank < A.shape[1] or np.any(costs < 1):
            s = b.sum() / max(A.sum(), 1)
            costs = np.full(A.shape[1], max(s, 1.0))

        return np.repeat(costs, counts)

    def _rebal_try_balance(self):
        comm, _, _ = get_comm_rank_root()

        # Compute per-element weights and check global imbalance
        weights = self._rebal_weights()
        mode, hi, lo = self._rebal_mode(weights)

        # Bail if load is balanced within threshold
        if hi / lo < self._rebal_threshold:
            return 0

        # Run the diffusion balancer and apply if any elements moved
        balancer = DiffusionBalancer(self.system.mesh)
        ownermap, n_moved = balancer.balance(weights, mode=mode,
                                             mesh=self.system.mesh)

        if (total := scal_coll(comm.Allreduce, n_moved, op=mpi.SUM)):
            self._rebal_apply(ownermap)

        return total

    def _rebal_proactive_check(self):
        if self._rebal_try_balance():
            self._rebal_cycle += 1
            self._rebal_cd = self._rebal_max_cd
            self._rebal_step = 0
            self._rebal_last_t = time.perf_counter()

    def _rebal_finish_cycle(self):
        self._rebal_cycle += 1
        self._rebal_stall = 0
        self._rebal_step = 0

        # Exponential backoff on cooldown
        self._rebal_cd = min(2*max(self._rebal_cd, self._rebal_base_cd),
                             self._rebal_max_cd)

    def _rebal_check(self):
        if not self._rebal_on:
            return

        # Fire proactive rebalance on first call
        if not self._rebal_did_proactive:
            self._rebal_did_proactive = True
            self._rebal_proactive_check()

        self._rebal_step += 1

        if self._rebal_step == 1:
            self._rebal_last_t = time.perf_counter()

        # Measurement window logic
        ns = self._rebal_nsteps
        phase = self._rebal_step % ns

        # Enable detailed MPI timing near window end
        if phase == ns - self._rebal_detail_window:
            self.system.set_mpi_timing_mode('detailed')

        if phase != 0:
            return

        # End of window: measure load and switch back to basic timing
        self.system.set_mpi_timing_mode('basic')
        my_load = self._rebal_measure()

        # Respect cooldown period
        if self._rebal_step <= self._rebal_cd or self._rebal_settled:
            return

        # Check global imbalance
        hi, lo = self._rebal_load_bounds(my_load)

        # Balanced: increment stall counter, settle after 3
        if hi / lo < self._rebal_threshold:
            self._rebal_stall += 1
            self._rebal_trigger = 0
            if self._rebal_stall >= 3:
                self._rebal_settled = True
        # Imbalanced: require sustained imbalance before acting
        else:
            self._rebal_stall = 0
            self._rebal_trigger += 1

            if self._rebal_cycle < 1 or self._rebal_trigger >= 3:
                self._rebal_trigger = 0
                self._rebal_execute()

    def _rebal_apply(self, ownermap):
        comm, _, _ = get_comm_rank_root()
        mesh = self.system.mesh

        # Build exchangers and rebuild mesh (returns permuted exchangers)
        exchangers = make_exchangers(comm, mesh.eidxs, ownermap, mesh.etypes)
        new_mesh, exchangers = rebuild_mesh(mesh, exchangers)

        # Exchange solution through permuted exchangers
        soln = dict(zip(mesh.eidxs, self.system.ele_scal_upts(self.idxcurr)))
        new_soln = {}
        for et, etex in exchangers.items():
            if (send := soln.get(et)) is None:
                nupts, nvars, _ = self.system.ele_shapes[et]
                send = np.empty((nupts, nvars, 0))

            recv = etex.Exchange(send, axis=-1)

            if recv.shape[-1]:
                new_soln[et] = recv

        # Replace the solver system and refresh cached cost info.  Boundary
        # conditions own system-level state, so preserve their serialised data
        # through the same path used by restart.
        state = {k: v for k, v in self.serialiser.serialise().items()
                 if k.startswith('bcs/')}
        soln = Solution(self.cfg, None, None, new_soln, state=state)
        self._replace_system(new_mesh, soln)
        self._rebal_refresh_costs()

        self.triggers.post_rebalance(self, exchangers)
        for p in self.plugins:
            p.post_rebalance(self, exchangers)

        self._commit_system()

    def _rebal_execute(self):
        total = self._rebal_try_balance()

        if total == 0:
            self._rebal_stall += 1
            self._rebal_settled = self._rebal_stall >= 3
            return

        # Settle if moves are trivial (< 1% of elements)
        comm, _, _ = get_comm_rank_root()
        total_neles = scal_coll(comm.Allreduce,
                                self._rebal_counts.sum(), op=mpi.SUM)
        if total < total_neles // 100:
            self._rebal_trivial += 1
            if self._rebal_trivial >= 3:
                self._rebal_settled = True
                return
        else:
            self._rebal_trivial = 0

        self._rebal_finish_cycle()
