from collections import defaultdict, deque
import itertools as it
import re
import sys
import time
import gc

import numpy as np

from pyfr.cache import memoize
from pyfr.mpiutil import (initialise_new_comm, mpi, scal_coll,
                          comm, rank, root, rankmap, execute, promote_comm)
from pyfr.plugins import get_plugin

from pyfr.readers.native import NativeReader

from pyfr.partitioners.online.base import _MeshInterconnector
from pyfr.util import first

def _common_plugin_prop(attr, *, edim):
    def wrapfn(fn):
        @property
        def newfn(self: BaseIntegrator):
            if not (p := getattr(self, attr)):
                t, c = time.time(), self._plugin_wtimes['common', None]

                p = execute['compute'](lambda: fn(self), default=None)
                
                p = self.relocate_ary(self._plugins_intercon, p, edim,
                                      src_name='compute', dst_name='plugins')

                self._plugin_wtimes['common', None] = c + time.time() - t
                setattr(self, attr, p)

            return p
        return newfn
    return wrapfn


class BaseIntegrator:
    def __init__(self, backend, mmesh, initsoln, cfg):
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
        self.dt = cfg.getfloat('solver-time-integrator', 'dt')
        self.dtmin = cfg.getfloat('solver-time-integrator', 'dt-min', 1e-12)

        # Extract the UUID of the mesh (to be saved with solutions)
        self.mesh_uuid = mmesh.mesh.uuid

        # Create a dictionary to store all the meshes used throughout the simulation
        self.meshes = {'compute': mmesh.mesh, 'computebest': None}

        self._invalidate_caches()

        # Record the starting wall clock time
        self._wstart = time.time()
        self.walltime_last = 0.

        if rank['world'] == root['world']:
            with open('lb_walltimes.csv', 'a') as f: f.write('tcurr,others,iterate,reinit\n')

        self.wallt_end = time.perf_counter_ns()

        # Record the total amount of time spent in each plugin
        self._plugin_wtimes = defaultdict(lambda: 0)

        # Abort computation
        self._abort = False
        self._abort_reason = ''

        self.called_plugin_dt = False

        self.mmesh = mmesh

        self.initialise_comm_and_partition(goal='plugins')
        self._plugins_intercon = _MeshInterconnector(self.meshes['compute'].eidxs, 
                                                     self.meshes['plugins'].eidxs)
        
        # Smoothly step to target time in the last near_t steps
        self.aminf = self.cfg.getfloat('solver-time-integrator', 
                                          'dt-adjust-min-fact', 0.9)
        self.amaxf = self.cfg.getfloat('solver-time-integrator', 
                                          'dt-adjust-max-fact', 1.001)
        self.dt_fallback = cfg.getfloat('solver-time-integrator', 'dt')
        self.dt_near = None

    def copy_to_empty_system(self):

        # If already copied, do nothing
        if hasattr(self, 'convars'):
            return
       
        if comm['compute'] != mpi.COMM_NULL:
            convars = list(first(self.system.ele_map.values()).convars)
        else:
            convars = []

        convars = execute['compute'](lambda: list(first(self.system.ele_map.values()).convars),
                          default = [])

        # Ensure the empty systems have what plugins expect.
        convars = comm['world'].allgather(convars)
        self.convars = list(first(c for c in convars if c))



    def adjust_dt(self, t):
        # Time difference to traverse 
        t_diff = t - self.tcurr

        # Estimate steps to reach t upon taking self.dt_fallback steps
        est_nsteps = t_diff / self.dt_fallback
        est_nsteps_roundup = -(est_nsteps // -self.amaxf)

        if est_nsteps_roundup == 1:
            # Exactly reach t
            self.dt = t_diff
            self.dt_near = None

        elif (est_nsteps - 1) / (est_nsteps_roundup - 1) < self.aminf:
            # Modify step to the approaching t

            dt_near = t_diff / est_nsteps_roundup

            if (self.dt_near is None 
                or not self.aminf < (self.dt_near/dt_near) < self.amaxf): 
                self.dt_near = dt_near

            self.dt = self.dt_near
        else:
            # Reset step if far from t
            self.dt = self.dt_fallback

    def plugin_abort(self, reason):
        self._abort = True
        self._abort_reason = self._abort_reason or reason

    def initialise_comm_and_partition(self, goal):

        if self.cfg.hasopt('partition', f'{goal}-pname'):
            pname = self.cfg.get('partition', f'{goal}-pname')
        else:
            pname = None

        if self.mmesh.online_cfg.hasopt('partition', f'{goal}-ranklist'):
            # If pname is None, share with 'compute'
            if goal == 'plugins' and pname is None:
                pranks = self.mmesh.online_cfg.getliteral('partition', 'compute-ranklist')

                # Copy comm from 'compute' to 'plugins' by reference
                comm['plugins'] = comm['compute']
            else:
                pranks = self.mmesh.online_cfg.getliteral('partition', f'{goal}-ranklist')
                initialise_new_comm(goal, pranks)
        else:
            pranks = list(range(comm['world'].size))
            initialise_new_comm(goal, pranks)

        if goal == 'plugins':
            # If pname is None, share the mesh connectivity with 'compute'
            if pname is None:
                self.meshes[goal] = self.meshes['compute']
            else:
                reader = NativeReader(self.meshes['compute'].fname, pname,
                                    construct_con=False, comm_name=goal)
                self.meshes[goal] = reader.mesh
        else:
            reader = NativeReader(self.meshes['compute'].fname, pname,
                                construct_con=True, comm_name=goal)
            self.meshes[goal] = reader.mesh

    def relocate_ary(self, pintercon, ary, edim, *, src_name='compute', dst_name='plugins'):
        """
        Relocate per-element arrays between mesh layouts.

        Parameters
        ----------
        pintercon : _MeshInterconnector
            Interconnector built from meshes[src_name] -> meshes[dst_name].
        ary : None | dict[str, np.ndarray] | Sequence[np.ndarray]
            - dict: keyed by etype; any subset is fine (empties auto-filled).
            - sequence: ordered like meshes[src_name].eidxs iteration order.
            - None: treated as empty input.
        edim : int
            Axis index corresponding to the element dimension.
        src_name, dst_name : str
            Names in self.meshes for source/destination layouts.

        Returns
        -------
        out : dict[str, np.ndarray] | list[np.ndarray]
            dict if input was a dict; otherwise a list ordered like
            meshes[dst_name].eidxs iteration order.
        """
        # Normalize input → dict keyed by etype (only provide what we have)
        want_dict = isinstance(ary, dict)
        if ary is None:
            edict_in = {}
        elif want_dict:
            edict_in = {str(et): a for et, a in ary.items() if a is not None}
        else:
            src_order = list(self.meshes[src_name].eidxs)  # deterministic
            edict_in = {et: a for et, a in zip(src_order, ary) if a is not None}

        # Let the interconnector validate/reshape/synthesize empties and relocate
        out_dict = pintercon.relocate(edict_in, edim=edim)  # → dict[etype] = np.ndarray

        # Preserve caller’s expectation on return type
        if want_dict:
            # keep only originally requested keys in that same key order
            return {et: out_dict.get(et) for et in edict_in.keys()}
        else:
            dst_order = list(self.meshes[dst_name].eidxs)
            return [out_dict[e] for e in dst_order if e in out_dict]


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

    def _reget_plugins(self):
        plugins = []

        for s in self.cfg.sections():
            if (m := re.match('(soln|solver)-plugin-(.+?)(?:-(.+))?$', s)):
                cfgsect, ptype, name, suffix = m[0], m[1], m[2], m[3]
                plugins.append(get_plugin(ptype, name, self, cfgsect, suffix))

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
        if self.called_plugin_dt:
            return

        ta = self.tlist
        tbegin = tstart if tstart > self.tcurr else self.tcurr
        tb = deque(np.arange(self.tend - dt, self.tcurr, -dt).tolist()[::-1])

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

            if comm['compute'] != mpi.COMM_NULL:
                wait_times = comm['compute'].allgather(self.system.rhs_wait_times())
            else:
                wait_times = []

            wait_times = comm['world'].allgather(wait_times)
            
            for i, ms in enumerate(zip(*wait_times)):
                for j, k in enumerate(['mean', 'stdev', 'median']):
                    stats.set('backend-wait-times', f'rhs-graph-{i}-wait-{k}',
                              ','.join(f'{v[j]:.3g}' for v in ms))

            compute_times = comm['compute'].allgather(self.system.rhs_compute_times())
            for i, ms in enumerate(zip(*compute_times)):
                for j, k in enumerate(['mean', 'stdev', 'median']):
                    stats.set('backend-compute-times', f'rhs-graph-{i}-compute-{k}',
                              ','.join(f'{v[j]:.6g}' for v in ms))

            all_times = comm['compute'].allgather(self.system.rhs_all_times())
            for i, ms in enumerate(zip(*all_times)):
                for j, k in enumerate(['mean', 'stdev', 'median']):
                    stats.set('backend-all-times', f'rhs-graph-{i}-all-{k}',
                              ','.join(f'{v[j]:.6g}' for v in ms))

        if self.cfg.getbool('backend', 'collect-waitsome-times', False):

            if comm['compute'] == mpi.COMM_NULL:
                return 

            wait_times    = comm['compute'].allgather(self.system.rhs_wait_times())
            compute_times = comm['compute'].allgather(self.system.rhs_compute_times())
            all_times     = comm['compute'].allgather(self.system.rhs_all_times())
            waitsome_send = comm['compute'].allgather(self.system.rhs_wait_times_send())
            waitsome_recv = comm['compute'].allgather(self.system.rhs_wait_times_recv())
            #nbs_all       = comm.allgather(self.system.nbytes_send)

            for i, ms in enumerate(zip(*wait_times)):
                for j, k in enumerate(['mean', 'stdev', 'median']):
                    stats.set('backend-wait-times', f'rhs-graph-{i}-wait-{k}',
                              ','.join(f'{v[j]:.3g}' for v in ms))

            for i, ms in enumerate(zip(*compute_times)):
                for j, k in enumerate(['mean', 'stdev', 'median']):
                    stats.set('backend-compute-times', f'rhs-graph-{i}-compute-{k}',
                              ','.join(f'{v[j]:.6g}' for v in ms))

            for i, ms in enumerate(zip(*all_times)):
                for j, k in enumerate(['mean', 'stdev', 'median']):
                    stats.set('backend-all-times', f'rhs-graph-{i}-all-{k}',
                              ','.join(f'{v[j]:.6g}' for v in ms))

            for i, ms in enumerate(zip(*waitsome_send)):
                for j, k in enumerate(['mean', 'stdev', 'median']):
                    coldata = []
                    for arr in ms:
                        coldata.extend(row[j] for row in arr)

                    stats.set('backend-wait-times',
                            f'rhs-graph-{i}-send-{k}',
                            ','.join(f'{val:.3g}' for val in coldata))

            for i, ms in enumerate(zip(*waitsome_recv)):
                for j, k in enumerate(['mean', 'stdev', 'median']):
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

            for label, col in (('mean', 0), ('stdev', 1), ('median', 2)):
                for stage, trip in sorted(_make_csr(waitsome_send, col).items()):
                    stats.set('backend-wait-times', f'csr-rhs-graph-{stage}-send-{label}', ','.join(trip))

                for stage, trip in sorted(_make_csr(waitsome_recv, col).items()):
                    stats.set('backend-wait-times', f'csr-rhs-graph-{stage}-recv-{label}', ','.join(trip))

        # stage_to_triplets = defaultdict(list)
        # for r_src, stage_dict in enumerate(nbs_all):
        #     for stage, vec in stage_dict.items():
        #         for r_dst, nb in enumerate(vec):
        #             if r_src > r_dst and nb:
        #                 stage_to_triplets[stage].append(f'({r_src},{r_dst},{nb})')
        # 
        # for stage, triplets in sorted(stage_to_triplets.items()):
        #     stats.set('backend-bytes', f'rhs-graph-{stage}-bytes', ','.join(triplets))

    def get_median_matrices(self):
        # World/compute comms
        if comm['compute'] == mpi.COMM_NULL:
            return None, None, None

        P = comm['compute'].size

        all_times = comm['compute'].allgather(self.system.rhs_all_times_median())
        ws_send   = comm['compute'].allgather(self.system.rhs_wait_times_send_median())
        ws_recv   = comm['compute'].allgather(self.system.rhs_wait_times_recv_median())

        # Per-rank medians
        all_med  = np.fromiter((float(all_times[r])  for r in range(P)),                                 dtype=float, count=P)
        send_mat = np.fromiter((float(ws_send[i][j]) for i in range(P) for j in range(P)),dtype=float, count=P*P).reshape(P, P)
        recv_mat = np.fromiter((float(ws_recv[i][j]) for i in range(P) for j in range(P)),dtype=float, count=P*P).reshape(P, P)

        # Root-only concise logs
        # if rank['compute'] == root['compute']:
        #     print(f"all*1e6=\n{ np.array2string(all_med*1e6,  formatter={'float_kind':lambda x: f'{x:05.0f}'})}")
        #     print(f"send*1e6=\n{np.array2string(send_mat*1e6, formatter={'float_kind':lambda x: f'{x:03.0f}'})}")
        #     print(f"recv*1e6=\n{np.array2string(recv_mat*1e6, formatter={'float_kind':lambda x: f'{x:03.0f}'})}")

        return all_med, send_mat, recv_mat

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
        if scal_coll(comm['world'].Allreduce, int(self._abort), op=mpi.LOR):
            self._finalise_plugins()

            reason = self._abort_reason
            sys.exit(comm['world'].allreduce(reason, op=lambda x, y: x or y))

    def load_balance(self):
        # Rebalance every lb_iters accepted steps, unless lb_iters == 1 sentinel
        if self.nsteps % self.mmesh.lb_iters == 0 and not self.mmesh.lb_iters == 1:
            if rank['world'] == root['world']: print('Switching, nacptsteps = ', self.nacptsteps)
            wallt_start = time.perf_counter_ns()

            mmesh = self.mmesh ; mmesh.restart()            
            mmesh.i.info() ; mmesh.i.info_to_csv(tcurr=self.tcurr)

            mmesh.recheck_online_file()

            # Element counts per world rank
            target = mmesh.calc_target(*self.get_median_matrices())

            self.mmesh.intg_repartition(target)

            mmesh.i.info()
            
            wallt_iterate = time.perf_counter_ns() - wallt_start

            # Create newcompute per non-zero elements within mmesh
            cur0 = np.array(mmesh._cur_counts_total, dtype=np.int64)
            next_ranklist = np.flatnonzero(cur0 > 0).astype(int).tolist()
            initialise_new_comm('newcompute', next_ranklist)

            soln = self.reinit_mesh_soln(mmesh.to_mesh(mmesh.i.eidxs), self.compute_soln)
            promote_comm('newcompute', 'compute')

            self.reinit_backend_and_system(self.meshes['compute'], soln)
            wallt_reinit = time.perf_counter_ns() - wallt_start - wallt_iterate

            # Write wall times
            if rank['world'] == root['world']:
                with open('lb_walltimes.csv', 'a') as f:
                    f.write(f"{self.tcurr:.6f},"
                            f"{(wallt_start - self.wallt_end)/1e9},"
                            f"{wallt_iterate/1e9},{wallt_reinit/1e9}\n")

            self.wallt_end = time.perf_counter_ns()

    def reinit_mesh_soln(self, mesh, soln):
        # New mesh lives under the 'newcompute' logical name while we migrate.
        self.meshes['newcompute'] = mesh

        # Build interconnector from old compute layout -> newcompute layout
        self._newcompute_intercon = _MeshInterconnector(self.meshes['compute'].eidxs, 
                                                        self.meshes['newcompute'].eidxs)
        
        soln = self.relocate_ary(self._newcompute_intercon, soln,
                                 edim=2, src_name='compute', dst_name='newcompute')

        # Check fi has attribute dtau
        if hasattr(self, 'pseudointegrator'):
            if hasattr(self.pseudointegrator, 'dtau_upts'):

                # If the dtau_upts attribute exists, relocate that too.
                dtaus = [dtau.get() for dtau in self.pseudointegrator.dtau_upts]
                
                new_dtaus = self.relocate_ary(self._newcompute_intercon, dtaus,
                                                edim=2, src_name='compute', dst_name='newcompute')

                # Get shapes of the new dtau_upts
                shapes = [dtau.shape for dtau in new_dtaus]

                # Create new dtau_upts
                self.pseudointegrator.dtau_upts = [self.backend.matrix(shape, new_dtau, tags={'align'})
                                    for shape, new_dtau in zip(shapes, new_dtaus)]

        # Drop old compute mesh and plugin interconnector
        del self.meshes['compute']
        del self._plugins_intercon

        # Promote newcompute mesh to be the canonical compute mesh
        self.meshes['compute'] = self.meshes['newcompute']
        # Optionally drop the extra key to avoid confusion
        # del self.meshes['newcompute']

        return soln

class BaseCommon:
    def _get_gndofs(self):        
        # Get the number of degrees of freedom in this partition
        ndofs = sum(self.system.ele_ndofs)

        # Sum to get the global number over all partitions
        return comm['world'].allreduce(ndofs, op=mpi.SUM)

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
