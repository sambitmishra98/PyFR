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

        score = np.array(comm.allgather(np.median(self._rebal_dofs)))
        score = score[im.counts() > 0].sum()

        if self._rebal_best_eidxs is None or score > self._rebal_best_score:
            self._rebal_best_score = score
            self._rebal_best_eidxs = {
                et: np.array(v, copy=True)
                for et, v in self.system.mesh.eidxs.items()
            }

        # Stagnation-driven rank draining requires rank removal support
        if self._rebal_stagnant and rank == root and self._rebal_iter == 1:
            print('[rebalance] shuffle-if-stagnant is not yet supported; '
                  'ignoring', flush=True)

        # Rebalance in index space and apply
        self._rebal_balancer.balance(im, target)

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

        self._rebal_g1_csvs(g1a, g1s, g1r)
        return cost

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
