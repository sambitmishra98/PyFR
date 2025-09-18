from collections import defaultdict, deque
import itertools as it
import re
import sys
import time

import numpy as np

from pyfr.cache import memoize
from pyfr.mpiutil import get_comm_rank_root, mpi, scal_coll
from pyfr.plugins import get_plugin


def _common_plugin_prop(attr):
    def wrapfn(fn):
        @property
        def newfn(self):
            if not (p := getattr(self, attr)):
                t, c = time.time(), self._plugin_wtimes['common', None]
                p = fn(self)
                self._plugin_wtimes['common', None] = c + time.time() - t
                setattr(self, attr, p)

            return p
        return newfn
    return wrapfn


class BaseIntegrator:
    def __init__(self, backend, mesh, initsoln, cfg):
        self.backend = backend
        self.isrestart = initsoln is not None
        self.cfg = cfg
        self.prevcfgs = {f: initsoln[f].tostr() for f in initsoln or []
                         if f.startswith('config-')}

        # Start time
        self.tstart = cfg.getfloat('solver-time-integrator', 'tstart', 0.0)
        self.tend = cfg.getfloat('solver-time-integrator', 'tend')

        # Current time; defaults to tstart unless restarting
        if self.isrestart:
            stats = initsoln['stats']
            self.tcurr = stats.getfloat('solver-time-integrator', 'tcurr')
        else:
            self.tcurr = self.tstart

        # List of target times to advance to
        self.tlist = deque([self.tend])

        # Accepted and rejected step counters
        self.nacptsteps = 0
        self.nrjctsteps = 0
        self.nacptchain = 0

        # Current and minimum time steps
        self._dt = cfg.getfloat('solver-time-integrator', 'dt')
        self.dtmin = cfg.getfloat('solver-time-integrator', 'dt-min', 1e-12)

        # Extract the UUID of the mesh (to be saved with solutions)
        self.mesh_uuid = mesh.uuid

        self._invalidate_caches()

        # Record the starting wall clock time
        self._wstart = time.time()
        self.walltime_last = 0.

        # Record the total amount of time spent in each plugin
        self._plugin_wtimes = defaultdict(lambda: 0)

        # Abort computation
        self._abort = False
        self._abort_reason = ''

    def plugin_abort(self, reason):
        self._abort = True
        self._abort_reason = self._abort_reason or reason

    def _get_plugins(self, initsoln):
        plugins = []

        for s in self.cfg.sections():
            if (m := re.match('(soln|solver)-plugin-(.+?)(?:-(.+))?$', s)):
                cfgsect, ptype, name, suffix = m[0], m[1], m[2], m[3]

                if ptype == 'solver' and suffix:
                    raise ValueError(f'solver-plugin-{name} cannot have a '
                                     'suffix')

                args = (ptype, name, self, cfgsect)
                if ptype == 'soln':
                    args += (suffix, )

                data = {}
                if initsoln is not None:
                    # Get the plugin data stored in the solution, if any
                    prefix = self.get_plugin_data_prefix(name, suffix)
                    for f in initsoln:
                        if f.startswith(f'{prefix}/'):
                            data[f.split('/')[2]] = initsoln[f]

                # Instantiate
                plugins.append(get_plugin(*args, **data))

        return plugins

    def _run_plugins(self):
        wtimes = self._plugin_wtimes

        self.backend.wait()

        # Fire off the plugins and tally up the runtime
        for plugin in self.plugins:
            tstart = time.time()
            tcommon = wtimes['common', None]

            plugin(self)

            dt = time.time() - tstart - wtimes['common', None] + tcommon

            pname = getattr(plugin, 'name', 'other')
            psuffix = getattr(plugin, 'suffix', None)
            wtimes[pname, psuffix] += dt

        # Abort if plugins request it
        self._check_abort()

    def _finalise_plugins(self):
        for plugin in self.plugins:
            if (finalise := getattr(plugin, 'finalise', None)):
                finalise(self)

    @staticmethod
    def get_plugin_data_prefix(name, suffix):
        if suffix:
            return f'plugins/{name}-{suffix}'
        else:
            return f'plugins/{name}'

    def call_plugin_dt(self, tstart, dt):
        ta = self.tlist
        tbegin = tstart if tstart > self.tcurr else self.tcurr
        tb = deque(np.arange(tbegin, self.tend, dt).tolist())

        self.tlist = tlist = deque()

        # Merge the current and new time lists
        while ta and tb:
            t = ta.popleft() if ta[0] < tb[0] else tb.popleft()
            if not tlist or t - tlist[-1] > self.dtmin:
                tlist.append(t)

        for t in it.chain(ta, tb):
            if not tlist or t - tlist[-1] > self.dtmin:
                tlist.append(t)

    def _invalidate_caches(self):
        self._curr_soln = None
        self._curr_grad_soln = None
        self._curr_dt_soln = None

    def step(self, t, dt):
        pass

    def advance_to(self, t):
        pass

    def run(self):
        for t in self.tlist:
            self.advance_to(t)

        self._finalise_plugins()

    @property
    def nsteps(self):
        return self.nacptsteps + self.nrjctsteps

    def collect_stats(self, stats):
        wtime = time.time() - self._wstart

        # Simulation and wall clock times
        stats.set('solver-time-integrator', 'tcurr', self.tcurr)
        stats.set('solver-time-integrator', 'wall-time-last', self.walltime_last)
        self.walltime_last = wtime
        stats.set('solver-time-integrator', 'wall-time', wtime)

        # Plugin wall clock times
        for (pname, psuffix), t in self._plugin_wtimes.items():
            k = f'plugin-wall-time-{pname}'
            if psuffix:
                k += f'-{psuffix}'

            stats.set('solver-time-integrator', k, t)

        # Step counts
        stats.set('solver-time-integrator', 'nsteps', self.nsteps)
        stats.set('solver-time-integrator', 'nacptsteps', self.nacptsteps)
        stats.set('solver-time-integrator', 'nrjctsteps', self.nrjctsteps)

        # MPI wait times
        if self.cfg.getbool('backend', 'collect-wait-times', False):
            comm, rank, root = get_comm_rank_root()

            wait_times = comm.allgather(self.system.rhs_wait_times())
            for i, ms in enumerate(zip(*wait_times)):
                for j, k in enumerate(['mean', 'stdev', 'median']):
                    stats.set('backend-wait-times', f'rhs-graph-{i}-wait-{k}',
                              ','.join(f'{v[j]:.3g}' for v in ms))

            compute_times = comm.allgather(self.system.rhs_compute_times())
            for i, ms in enumerate(zip(*compute_times)):
                for j, k in enumerate(['mean', 'sem', 
                                       'stdev', 'median']):
                    stats.set('backend-compute-times', f'rhs-graph-{i}-compute-{k}',
                              ','.join(f'{v[j]:.6g}' for v in ms))

            all_times = comm.allgather(self.system.rhs_all_times())
            for i, ms in enumerate(zip(*all_times)):
                for j, k in enumerate(['mean', 'sem', 
                                       'stdev', 'median']):
                    stats.set('backend-all-times', f'rhs-graph-{i}-all-{k}',
                              ','.join(f'{v[j]:.6g}' for v in ms))

        if self.cfg.getbool('backend', 'collect-waitsome-times', False):
            comm, rank, root = get_comm_rank_root()

            wait_times = comm.allgather(self.system.rhs_wait_times())
            for i, ms in enumerate(zip(*wait_times)):
                for j, k in enumerate(['mean', 'stdev', 'median']):
                    stats.set('backend-wait-times', f'rhs-graph-{i}-wait-{k}',
                              ','.join(f'{v[j]:.3g}' for v in ms))

            compute_times = comm.allgather(self.system.rhs_compute_times())
            for i, ms in enumerate(zip(*compute_times)):
                for j, k in enumerate(['mean', 'sem', 
                                       'stdev', 'median']):
                    stats.set('backend-compute-times', f'rhs-graph-{i}-compute-{k}',
                              ','.join(f'{v[j]:.6g}' for v in ms))

            all_times = comm.allgather(self.system.rhs_all_times())
            for i, ms in enumerate(zip(*all_times)):
                for j, k in enumerate(['mean', 'sem', 
                                       'stdev', 'median']):
                    stats.set('backend-all-times', f'rhs-graph-{i}-all-{k}',
                              ','.join(f'{v[j]:.6g}' for v in ms))

            waitsome_send = comm.allgather(self.system.rhs_wait_times_send())
            for i, ms in enumerate(zip(*waitsome_send)):
                for j, k in enumerate(['mean', 'sem','stdev', 'median']):
                    coldata = []
                    for arr in ms:
                        coldata.extend(row[j] for row in arr)

                    stats.set('backend-wait-times',
                            f'rhs-graph-{i}-send-{k}',
                            ','.join(f'{val:.3g}' for val in coldata))

            waitsome_recv = comm.allgather(self.system.rhs_wait_times_recv())
            for i, ms in enumerate(zip(*waitsome_recv)):
                for j, k in enumerate(['mean', 'sem', 'stdev', 'median']):
                    coldata = []
                    for arr in ms:
                        coldata.extend(row[j] for row in arr)

                    stats.set('backend-wait-times',
                            f'rhs-graph-{i}-recv-{k}',
                            ','.join(f'{val:.3g}' for val in coldata))

            def _make_csr(stage_lists, col):
                out = defaultdict(list)
                for r_src, sl in enumerate(stage_lists):
                    for stage, arr in enumerate(sl):
                        for r_dst, row in enumerate(arr):
                            v = row[col]
                            if r_src != r_dst and v:
                                out[stage].append(f'({r_src},{r_dst},{v:.3e})')
                return out

            for label, col in (('mean', 0), ('sem', 1), ('stdev', 2), ('median', 3)):
                for stage, trip in sorted(_make_csr(waitsome_send, col).items()):
                    stats.set('backend-wait-times', f'csr-rhs-graph-{stage}-send-{label}', ','.join(trip))

                for stage, trip in sorted(_make_csr(waitsome_recv, col).items()):
                    stats.set('backend-wait-times', f'csr-rhs-graph-{stage}-recv-{label}', ','.join(trip))

        nbs_all  = comm.allgather(self.system.nbytes_send)

        stage_to_triplets = defaultdict(list)
        for r_src, stage_dict in enumerate(nbs_all):
            for stage, vec in stage_dict.items():
                for r_dst, nb in enumerate(vec):
                    if r_src > r_dst and nb:
                        stage_to_triplets[stage].append(f'({r_src},{r_dst},{nb})')

        for stage, triplets in sorted(stage_to_triplets.items()):
            stats.set('backend-bytes', f'rhs-graph-{stage}-bytes', ','.join(triplets))

        self._collect_backend_elements(stats)

        if self.tcurr > self.tstart:
            self.load_balance_compute_weights(stats)


    def _collect_backend_elements(self, stats):
        """Collect element counts and DoFs per rank and per type, via ele_shapes."""
        comm, rank, root = get_comm_rank_root()

        # Local stats from ele_shapes
        etypes = list(self.system.ele_shapes)
        local_counts = [self.system.ele_shapes[et][2] for et in etypes]
        local_dofs   = [self.system.ele_shapes[et][0] *
                        self.system.ele_shapes[et][1] *
                        self.system.ele_shapes[et][2]
                        for et in etypes]

        # Totals per rank
        local_count_tot = sum(local_counts)
        local_dof_tot   = sum(local_dofs)

        # Gather across ranks
        all_counts = comm.allgather(local_counts)
        all_dofs   = comm.allgather(local_dofs)
        all_count_tot = comm.allgather(local_count_tot)
        all_dof_tot   = comm.allgather(local_dof_tot)

        stats.set('backend-elements', 'etypes', ','.join(etypes))

        for j, et in enumerate(etypes):
            counts = ','.join(str(c[j]) for c in all_counts)
            dofs   = ','.join(str(d[j]) for d in all_dofs)
            stats.set('backend-elements', f'elems-{et}', counts)
            stats.set('backend-elements', f'dofs-{et}', dofs)

        stats.set('backend-elements', 'elems-all',
                ','.join(str(c) for c in all_count_tot))
        stats.set('backend-elements', 'dofs-all',
                ','.join(str(d) for d in all_dof_tot))

    def load_balance_compute_weights(self, stats):
        comm, rank, root = get_comm_rank_root()

        # Grab per-rank dofs
        dofs = [int(x) for x in stats.get('backend-elements', 'dofs-all').split(',')]
        R = len(dofs)
        P = comm.size

        # Grab per-rank compute+all times (median for g0+g1)
        g0c = [float(x) for x in stats.get('backend-compute-times', 'rhs-graph-0-compute-median').split(',')]
        g1c = [float(x) for x in stats.get('backend-compute-times', 'rhs-graph-1-compute-median').split(',')]
        g0a = [float(x) for x in stats.get('backend-all-times',     'rhs-graph-0-all-median').split(',')]
        g1a = [float(x) for x in stats.get('backend-all-times',     'rhs-graph-1-all-median').split(',')]

        # NEW: read flattened send medians (length P*P): row=r (src), col=c (dst)
        def _get_send(stage):
            key = f'rhs-graph-{stage}-send-median'
            vals = [float(x) for x in stats.get('backend-wait-times', key).split(',')]
            return vals

        def _sum_offdiag(flat, P):
            # per-rank sum over c != r
            if len(flat) != P*P:
                # fallback if older stats format; keep neutral
                return [0.0]*P
            return [sum(flat[r*P + c] for c in range(P) if c != r) for r in range(P)]

        g0s_flat = _get_send(0)
        g1s_flat = _get_send(1)
        g0s_sum  = _sum_offdiag(g0s_flat, P)
        g1s_sum  = _sum_offdiag(g1s_flat, P)

        # Cost candidates
        cost_compute = [g0c[r] + g1c[r] for r in range(R)]
        cost_all     = [g0a[r] + g1a[r] for r in range(R)]

        # Normalise by dofs
        per_dof_compute = [cost_compute[r]      / dofs[r] for r in range(R)]
        per_dof_all     = [cost_all[r]          / dofs[r] for r in range(R)]

        if rank == root:
            print("[weights] dofs:", dofs)
            print("[weights] cost_compute:", cost_compute)
            print("[weights] cost_all:", cost_all)
            print("[weights] per_dof_compute:", per_dof_compute)
            print("[weights] per_dof_all:", per_dof_all)
            # NEW: show send sums and compute+send candidate
            print("[weights] send_sums_g0:", g0s_sum)
            print("[weights] send_sums_g1:", g1s_sum)

    @property
    def cfgmeta(self):
        cfg = self.cfg.tostr()

        if self.prevcfgs:
            ret = dict(self.prevcfgs, config=cfg)

            if cfg != ret[f'config-{len(self.prevcfgs) - 1}']:
                ret[f'config-{len(self.prevcfgs)}'] = cfg

            return ret
        else:
            return {'config': cfg, 'config-0': cfg}

    def _check_abort(self):
        comm, rank, root = get_comm_rank_root()

        if scal_coll(comm.Allreduce, int(self._abort), op=mpi.LOR):
            self._finalise_plugins()

            reason = self._abort_reason
            sys.exit(comm.allreduce(reason, op=lambda x, y: x or y))


class BaseCommon:
    def _get_gndofs(self):
        comm, rank, root = get_comm_rank_root()

        # Get the number of degrees of freedom in this partition
        ndofs = sum(self.system.ele_ndofs)

        # Sum to get the global number over all partitions
        return comm.allreduce(ndofs, op=mpi.SUM)

    @memoize
    def _get_axnpby_kerns(self, *rs, subdims=None):
        kerns = [self.backend.kernel('axnpby', *[em[r] for r in rs],
                                     subdims=subdims)
                 for em in self.system.ele_banks]

        return kerns

    @memoize
    def _get_reduction_kerns(self, *rs, **kwargs):
        dtau_mats = getattr(self, 'dtau_upts', [])

        kerns = []
        for em, dtaum in it.zip_longest(self.system.ele_banks, dtau_mats):
            kerns.append(self.backend.kernel('reduction', *[em[r] for r in rs],
                                             dt_mat=dtaum, **kwargs))

        return kerns

    def _addv(self, consts, regidxs, subdims=None):
        # Get a suitable set of axnpby kernels
        axnpby = self._get_axnpby_kerns(*regidxs, subdims=subdims)

        # Bind the arguments
        for k in axnpby:
            k.bind(*consts)

        self.backend.run_kernels(axnpby)

    def _add(self, *args, subdims=None):
        self._addv(args[::2], args[1::2], subdims=subdims)
