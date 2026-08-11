import os
import time
from collections import deque

import numpy as np

from pyfr.mpiutil import DistributedDirectory, get_comm_rank_root
from pyfr.readers.native import Solution
from pyfr.rebalance import (DiffusionBalancer, IndexMesh, int_round,
                            make_exchangers, rebuild_mesh)


def _append_csv(fname, header, row):
    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            f.write(','.join(header) + '\n')

    with open(fname, 'a') as f:
        f.write(','.join(str(v) for v in row) + '\n')


# Post-rebuild physical-mesh integrity check, independent of IndexMesh.
#
# IndexMesh.etype_counts() (as logged to lb_elem_dist.csv) tracks ownership
# bookkeeping, not the physical mesh actually reconstructed by rebuild_mesh.
# A rebalance bug can silently drop elements during the collective exchange
# while IndexMesh still believes them conserved (this happened for real --
# see the 2026-08-09 collective-mismatch bug).  This check instead sums the
# ACTUAL per-etype element counts held in old_mesh.eles / new_mesh.eles on
# every rank, before and after rebuild_mesh, and asserts the global total
# per etype is unchanged.  A rebalance only moves ownership; it must never
# create or destroy elements.
#
# Enabled by default for calibration/investigation runs; set
# PYFR_REBAL_SKIP_INTEGRITY_CHECK=1 to disable for cheap production runs
# once the collective fix is fully certified.
def _assert_physical_conservation(old_mesh, new_mesh, tcurr):
    if os.environ.get('PYFR_REBAL_SKIP_INTEGRITY_CHECK'):
        return

    comm, rank, root = get_comm_rank_root()

    # old_mesh.etypes is the GLOBAL etype list (from the mesh codec), the
    # same on every rank -- must iterate this, not a locally-observed union
    # of old_mesh.eles/new_mesh.eles keys, or ranks with differing local
    # etypes would call allreduce a different number of times each and
    # deadlock. This is exactly the collective-mismatch bug class this
    # check exists to guard against.
    etypes = old_mesh.etypes
    pre = {et: len(old_mesh.eles.get(et, ())) for et in etypes}
    post = {et: len(new_mesh.eles.get(et, ())) for et in etypes}

    pre_totals = {et: comm.allreduce(pre[et]) for et in etypes}
    post_totals = {et: comm.allreduce(post[et]) for et in etypes}

    bad = {et: (pre_totals[et], post_totals[et]) for et in etypes
           if pre_totals[et] != post_totals[et]}

    # comm.allreduce above already guarantees pre_totals/post_totals -- and
    # therefore `bad` -- are identical on every rank, so every rank raises
    # together rather than only root dying while its peers hang on the next
    # collective.
    if bad:
        raise RuntimeError(
            f'Physical mesh integrity check FAILED at tcurr={tcurr:.6f}: '
            f'per-etype global element counts changed across rebuild_mesh '
            f'(etype: (pre, post)) -- {bad}. This means the rebalance '
            f'silently lost or duplicated elements in the physical mesh; '
            f'IndexMesh bookkeeping (lb_elem_dist.csv) would NOT have '
            f'caught this.'
        )

    if rank == root:
        _append_csv('lb_physical_integrity.csv',
                    ['tcurr'] + [f'total-{et}' for et in etypes],
                    [f'{tcurr:.6f}'] + [post_totals[et] for et in etypes])


class RebalanceMixin:
    '''
    Periodically repartitions the mesh to balance a measured per-rank
    cost.

    Every nsteps accepted steps the per-rank cost is assembled from
    median RHS times and per-neighbour MPI wait times, targets are set
    proportional to the current counts over the cost, and a diffusion
    balancer moves elements accordingly.  The highest-throughput
    partition seen so far is retained and restored when the hard-stop
    iteration limit is reached.
    '''
    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        sect = 'solver-rebalance'

        # Enable wait-time collection before the backend builds graphs
        if cfg.hasopt(sect, 'nsteps'):
            cfg.set('backend', 'collect-wait-times', 'true')

            if not cfg.hasopt('backend', 'collect-wait-times-len'):
                n = cfg.getint(sect, 'wait-window', 10)
                cfg.set('backend', 'collect-wait-times-len', str(n))

        super().__init__(backend, systemcls, mesh, initsoln, cfg)

        comm, rank, root = get_comm_rank_root()
        self._rebal_on = comm.size >= 2 and cfg.hasopt(sect, 'nsteps')
        if not self._rebal_on:
            return

        self._rebal_nsteps = cfg.getint(sect, 'nsteps')
        self._rebal_hard_stop = cfg.getint(sect, 'hard-stop', -1)
        self._rebal_perfwin = cfg.getint(sect, 'perf-window', 1)
        self._rebal_jitter = cfg.getfloat(sect, 'target-jitter', 0)
        self._rebal_stagnant = cfg.getint(sect, 'shuffle-if-stagnant', 0)

        # Per-rank cost model coefficients and exponents
        gf = cfg.getfloat
        self._rebal_coeffs = np.array([
            gf(sect, 'cost-coeff-compute', 1.0),
            gf(sect, 'cost-coeff-recv-wait', -1.0),
            gf(sect, 'cost-coeff-send-wait', -1.0),
            gf(sect, 'cost-coeff-caused-wait', 1.0)
        ])
        self._rebal_exps = np.array([
            gf(sect, 'cost-exp-compute', 1.0),
            gf(sect, 'cost-exp-recv-wait', 1.0),
            gf(sect, 'cost-exp-send-wait', 1.0),
            gf(sect, 'cost-exp-caused-wait', 1.0)
        ])
        self._rebal_cost_err = gf(sect, 'cost-coeff-errest', -1.0)

        self._rebal_balancer = DiffusionBalancer(cfg, sect)

        # Runtime state
        self._rebal_iter = 0
        self._rebal_stopped = self._rebal_hard_stop == 0
        self._rebal_detailed = False
        self._rebal_jitter_rank = 0

        self._rebal_best_score = -np.inf
        self._rebal_best_eidxs = None

        # Stagnation tracking for rank shuffling
        self._rebal_last_cmax = None
        self._rebal_stag = 0

        n = cfg.getint('backend', 'collect-wait-times-len')
        self._rebal_dofs = deque(maxlen=self._rebal_perfwin)
        self._rebal_errest = deque(maxlen=n)

        self._rebal_wend = time.perf_counter_ns()

        if rank == root:
            with open('lb_walltimes.csv', 'a') as f:
                f.write('tcurr,others,iterate,reinit\n')

    def _rebal_check(self):
        if not self._rebal_on:
            return

        # Switch the RHS graphs over to per-neighbour wait timing
        if not self._rebal_detailed:
            self.system.set_mpi_timing_mode('detailed')
            self._rebal_detailed = True

        ns = self._rebal_nsteps
        if ns == 1 or self.nacptsteps % ns:
            return

        if self._rebal_stopped:
            self._rebal_heartbeat()
        else:
            self._rebal_execute()

    def _rebal_heartbeat(self):
        comm, rank, root = get_comm_rank_root()
        now = time.perf_counter_ns()

        if rank == root:
            _append_csv('lb_walltimes.csv',
                        ['tcurr', 'others', 'iterate', 'reinit'],
                        [f'{self.tcurr:.6f}',
                         f'{(now - self._rebal_wend) / 1e9:.6f}', 0, 0])

        self._rebal_wend = now

    def _rebal_execute(self):
        comm, rank, root = get_comm_rank_root()
        wstart = time.perf_counter_ns()

        self._rebal_iter += 1

        # On hard stop, revert to the best partition seen and freeze
        hs = self._rebal_hard_stop
        if 0 < hs <= self._rebal_iter:
            self._rebal_stopped = True

            if rank == root:
                print(f'[rebalance] hard-stop={hs} reached; reverting to '
                      f'best partition (score={self._rebal_best_score:.6e})',
                      flush=True)

            if self._rebal_best_eidxs is not None:
                im = IndexMesh(self.system.mesh)
                self._rebal_apply(self._rebal_ownermap(
                    im, self._rebal_best_eidxs
                ))

            return

        im = IndexMesh(self.system.mesh)
        self._rebal_dist_csv(im)

        # Per-rank cost and resulting element targets
        cost = self._rebal_cost(comm)
        target = self._rebal_target(im, cost)

        # Track the highest-throughput partition seen so far
        dofs = sum(self.system.ele_ndofs) / cost[rank]
        self._rebal_dofs.append(dofs)

        scores = np.array(comm.allgather(np.median(self._rebal_dofs)))
        active = im.counts() > 0
        score = scores[active].sum()

        if self._rebal_best_eidxs is None or score > self._rebal_best_score:
            self._rebal_best_score = score
            self._rebal_best_eidxs = {
                et: np.array(v, copy=True)
                for et, v in self.system.mesh.eidxs.items()
            }

        # Rebalance in index space; shuffle the worst rank if stagnant
        if self._rebal_stagnated(cost):
            scores[~active] = np.inf
            worst = int(np.argmin(scores))

            if rank == root:
                print(f'[rebalance] stagnant; shuffling rank {worst}',
                      flush=True)

            self._rebal_balancer.shuffle(im, target, cost, worst)

            # Measurements of the shuffled partitioning start afresh
            self._rebal_dofs.clear()
            self._rebal_last_cmax = None
        else:
            self._rebal_balancer.balance(im, target)

        if not im.counts().all():
            raise RuntimeError('Rebalancing left a rank with no elements')

        witer = time.perf_counter_ns()
        self._rebal_apply(self._rebal_ownermap(im, im.eidxs()))
        self._rebal_detailed = False

        wend = time.perf_counter_ns()

        if rank == root:
            _append_csv('lb_walltimes.csv',
                        ['tcurr', 'others', 'iterate', 'reinit'],
                        [f'{self.tcurr:.6f}',
                         f'{(wstart - self._rebal_wend) / 1e9:.6f}',
                         f'{(witer - wstart) / 1e9:.6f}',
                         f'{(wend - witer) / 1e9:.6f}'])

        self._rebal_wend = time.perf_counter_ns()

    def _rebal_cost(self, comm):
        n = comm.size

        # Median per-step RHS time and per-neighbour wait times
        g1a_loc, send_med, recv_med = self.system.rhs_median_times()

        srow, rrow = np.zeros(n), np.zeros(n)
        for p, v in send_med.items():
            srow[p] = v
        for p, v in recv_med.items():
            rrow[p] = v

        g1a = np.array(comm.allgather(g1a_loc))
        g1s = np.array(comm.allgather(srow))
        g1r = np.array(comm.allgather(rrow))

        err = np.median(self._rebal_errest) if self._rebal_errest else 0.0
        err = np.array(comm.allgather(err))

        # Cost components: own compute, waits we incur on receives and
        # sends, and receive waits we cause on other ranks
        comps = np.stack([g1a, g1r.sum(axis=1), g1s.sum(axis=1),
                          g1r.sum(axis=0)])

        cost = (self._rebal_coeffs @ comps**self._rebal_exps[:, None]
                + self._rebal_cost_err*err)

        # Targets assume positive costs; clamp waits-dominated ranks
        if (pos := cost[cost > 0]).size:
            cost = np.maximum(cost, pos.min())

        self._rebal_g1_csvs(g1a, g1s, g1r)
        return cost

    def _rebal_stagnated(self, cost):
        # The partitioning is stagnant when the highest per-rank cost
        # has not improved for shuffle-if-stagnant successive checks
        if not self._rebal_stagnant:
            return False

        cmax = cost[np.isfinite(cost)].max()

        if self._rebal_last_cmax is None or cmax < self._rebal_last_cmax:
            self._rebal_last_cmax = cmax
            self._rebal_stag = 0
        else:
            self._rebal_stag += 1

            if self._rebal_stag >= self._rebal_stagnant:
                self._rebal_stag = 0
                return True

        return False

    def _rebal_target(self, im, cost):
        target = int_round(im.counts() / cost, im.nglobal)

        # Optionally perturb one rank per cycle to escape local optima
        if self._rebal_jitter > 0:
            target = target.astype(float)
            target[self._rebal_jitter_rank] *= 1 + self._rebal_jitter
            self._rebal_jitter_rank = (self._rebal_jitter_rank + 1) % len(target)

            target = int_round(target, im.nglobal)

        return target

    def _rebal_ownermap(self, im, eidxs):
        # Destination ranks for our current elements, given the desired
        # global holdings described by eidxs
        comm, _, _ = get_comm_rank_root()
        mesh = self.system.mesh

        keys = np.concatenate([
            im.goff[et] + np.asarray(v, dtype=int)
            for et, v in eidxs.items()
        ]) if eidxs else np.empty(0, dtype=int)
        directory = DistributedDirectory(comm, keys)

        ets = [et for et in mesh.etypes if et in mesh.eidxs]
        query = np.concatenate([
            im.goff[et] + np.asarray(mesh.eidxs[et], dtype=int) for et in ets
        ]) if ets else np.empty(0, dtype=int)
        dests = directory.lookup(query)

        ownermap, i = {}, 0
        for et in ets:
            ownermap[et] = dests[i:i + len(mesh.eidxs[et])]
            i += len(mesh.eidxs[et])

        return ownermap

    def _rebal_apply(self, ownermap):
        comm, _, _ = get_comm_rank_root()
        mesh = self.system.mesh

        # Build exchangers and rebuild mesh (returns permuted exchangers)
        exchangers = make_exchangers(comm, mesh.eidxs, ownermap, mesh.etypes)
        new_mesh, exchangers = rebuild_mesh(mesh, exchangers)

        # Physical-mesh integrity check, independent of IndexMesh (see
        # 2026-08-09 collective-mismatch bug -- lb_elem_dist.csv alone
        # cannot detect silent element loss during this step)
        _assert_physical_conservation(mesh, new_mesh, self.tcurr)

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

        # Replace the solver system.  Boundary conditions own system-level
        # state, so preserve their serialised data through the same path
        # used by restart.
        state = {k: v for k, v in self.serialiser.serialise().items()
                 if k.startswith('bcs/')}
        soln = Solution(self.cfg, None, None, new_soln, state=state)
        self._replace_system(new_mesh, soln)

        self.triggers.post_rebalance(self, exchangers)
        for p in self.plugins:
            p.post_rebalance(self, exchangers)

        self._commit_system()

    def _rebal_dist_csv(self, im):
        comm, rank, root = get_comm_rank_root()

        ecnt = im.etype_counts()
        mout = comm.allgather(im.mpi_out())
        npairs = len(np.unique(im.fown[(im.fown >= 0)
                                       & (im.fown != rank)]))
        npairs = comm.allreduce(npairs) // 2

        if rank != root:
            return

        header, row = ['tcurr'], [f'{self.tcurr:.6f}']
        for r in range(comm.size):
            for i, et in enumerate(im.etypes):
                header.append(f'w{r}-{et}')
                row.append(int(ecnt[r, i]))

        for key, vals in [('total_elems', ecnt.sum(axis=1)),
                          ('mpi_faces_out', mout)]:
            for r in range(comm.size):
                header.append(f'w{r}-{key}')
                row.append(int(vals[r]))

        header += ['mpi_faces_total', 'mpi_pairs']
        row += [sum(mout) // 2, npairs]

        _append_csv('lb_elem_dist.csv', header, row)

    def _rebal_g1_csvs(self, g1a, g1s, g1r):
        comm, rank, root = get_comm_rank_root()
        if rank != root:
            return

        n = comm.size
        us = lambda a: np.rint(a*1e6).astype(int).ravel().tolist()

        _append_csv('g1-all-median-ms.csv',
                    [f'r{r}' for r in range(n)], us(g1a))

        mcols = [f'i{i}-{j}' for i in range(n) for j in range(n)]
        _append_csv('g1-send-median-ms.csv', mcols, us(g1s))
        _append_csv('g1-recv-median-ms.csv', mcols, us(g1r))
