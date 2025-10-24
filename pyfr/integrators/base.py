from collections import defaultdict, deque
import itertools as it
import re
import sys
import time

import numpy as np

from pyfr.cache import memoize
from pyfr.mpiutil import get_comm_rank_root, initialise_new_comm, mpi, scal_coll
from pyfr.plugins import get_plugin

from pyfr.readers.native import NativeReader, _MeshInterconnector

import os

def _append_csv_row(file_path: str, header_cols: list[str], values: list[int]):
    """Root-only: write header if missing, then append one integer row."""
    # Write header once
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            f.write(','.join(header_cols) + '\n')
        print(f"[g1csv] header_written file='{file_path}' ncols={len(header_cols)}")

    # Append row
    with open(file_path, 'a', newline='') as f:
        f.write(','.join(str(int(v)) for v in values) + '\n')
    print(f"[g1csv] row_append file='{file_path}' ncols={len(header_cols)}")


def _flatten_offdiag_labels(P: int) -> list[str]:
    """i0-1,i0-2,...,i0-(P-1), i1-0,i1-2,...,i(P-1)-(P-2); skips i==j."""
    cols = []
    for i in range(P):
        for j in range(P):
            if j != i:
                cols.append(f"i{i}-{j}")
    return cols


def _flatten_offdiag_values(M: np.ndarray) -> list[int]:
    """Row-major off-diagonal flatten to ints."""
    P = M.shape[0]
    vals = []
    for i in range(P):
        for j in range(P):
            if j != i:
                vals.append(int(M[i, j]))
    return vals



def _common_plugin_prop(attr, *, edim):
    def wrapfn(fn):
        @property
        def newfn(self):
            if not (p := getattr(self, attr)):
                t, c = time.time(), self._plugin_wtimes['common', None]

                ccomm, crank, croot = get_comm_rank_root('compute')
                comm,  rank,  root = get_comm_rank_root('world')
                if ccomm != mpi.COMM_NULL:
                    p = fn(self)
                else:
                    p = None

                p = self.relocate_ary(self._plugins_intercon, p, edim,
                                      src_name='compute', dst_name='plugins')

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

        # Create a dictionary to store all the meshes used throughout the simulation
        self.meshes = {'compute': mesh}

        self._invalidate_caches()

        # Record the starting wall clock time
        self._wstart = time.time()
        self.walltime_last = 0.

        # Record the total amount of time spent in each plugin
        self._plugin_wtimes = defaultdict(lambda: 0)

        # Abort computation
        self._abort = False
        self._abort_reason = ''

        self.called_plugin_dt = False
        self.lb_iters = self.cfg.getint('mesh', 'load-balancing-iterations', 1)
        self.lb_target_scale = self.cfg.getfloat('mesh', 'load-balancing-target-scale', 1.0)
        self.lb_flowmat_relax = self.cfg.getfloat('mesh', 'load-balancing-flowmatrix-relax', 0.5)

        # devices = 'gpu'*1+'cpu'*11
        devices = self.cfg.getliteral('mesh', 'devices')

        own_device = devices[get_comm_rank_root()[1]]

        # Get preference lists according to device type, depending on the rank
        self.etype_order = self.cfg.getliteral('mesh', f'device-preference-{own_device}')

    def plugin_abort(self, reason):
        self._abort = True
        self._abort_reason = self._abort_reason or reason

    def initialise_comm_and_partition(self, mname, construct_con=False):
        comm, rank, root = get_comm_rank_root('world')

        pname  = self.cfg.get('mesh', f'partition-{mname}', 'compute')

        # If note 'compute'
        if pname == 'compute':
            # Just duplicate the compute mesh
            self.meshes[mname] = self.meshes['compute']

        else:
           
            pranks = self.cfg.getliteral('mesh', f'partition-{mname}-ranklist',
                                        list(range(comm.size)))

            initialise_new_comm(mname, pranks)

            reader = NativeReader(self.meshes['compute'].fname, pname,
                                construct_con=construct_con, comm_name=mname)

            # Append to the meshes dictionary
            self.meshes[mname] = reader.mesh

    def initialise_interconnector(self, mname1, mname2):
        return _MeshInterconnector(self.meshes[mname1].eidxs, 
                                   self.meshes[mname2].eidxs)

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

        self.initialise_comm_and_partition('plugins')
        self._plugins_intercon = self.initialise_interconnector('compute', 
                                                                'plugins')

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
            comm, rank, root = get_comm_rank_root('compute')

            if comm != mpi.COMM_NULL:
                wait_times = comm.allgather(self.system.rhs_wait_times())
            else:
                wait_times = []

            # Finally, across all ranks in the world ....
            comm, rank, root = get_comm_rank_root()
            
            wait_times = comm.allgather(wait_times)
            
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

    def __collect_hist(self):
        """
        Append one observation: [dofs_hex, dofs_pyr, dofs_tet, ..., g1_compute_median]
        Returns the full history as a numpy array (rows = observations).
        """
        etypes = list(self.system.ele_shapes)
        dofs = [self.system.ele_shapes[et][0]*self.system.ele_shapes[et][1]*
                self.system.ele_shapes[et][2] for et in etypes]

        ct = self.system.rhs_compute_times()
        if not ct:
            raise RuntimeError("collect_hist: no compute-time data available")
        gidx = 1 if len(ct) > 1 else 0
        if len(ct[gidx]) < 4:
            raise RuntimeError("collect_hist: compute-time tuple missing median")
        tmed = float(ct[gidx][3])

        row = np.array([*map(float, dofs), tmed], dtype=float)
        if not hasattr(self, "_perf_hist"):
            self._perf_hist = []
        self._perf_hist.append(row)
        return np.asarray(self._perf_hist, dtype=float)

    def __performance_from_hist(self, hist):
        """
        Fit time ≈ sum_i alpha_i * dofs_i on the last N+1 rows (N = #etypes).
        Return {etype: perf_i} where perf_i = 1/alpha_i in dof/s.
        """

        comm, rank, root = get_comm_rank_root('world')

        etypes = list(self.system.ele_shapes)
        N = len(etypes)
        if hist is None or len(hist) < N + 1:
            return

        H = np.asarray(hist[-(N+1):], dtype=float)
        X, y = H[:, :N], H[:, N]
        alphas, *_ = np.linalg.lstsq(X, y, rcond=None)
        alphas = np.asarray(alphas, dtype=float)
        if np.any(~np.isfinite(alphas)):
            raise RuntimeError("Regression produced non-finite coefficients")

        eps = 1e-30
        perfs = 1.0 / np.maximum(alphas, eps)

        self.perf_by_etype = {et: float(p) for et, p in zip(etypes, perfs)}

        all_perf = comm.allgather(self.perf_by_etype)

        if rank == root:
            print(f"[perf] per_etype_dof_per_sec_allranks:")
            etypes = list(self.system.ele_shapes)
            header = "Rank " + " ".join(f"{et:>15}" for et in etypes)
            print(header)
            for r in range(comm.size):
                row = f"{r:>4} " + " ".join(f"{all_perf[r].get(et, 0.0):15.3e}" for et in etypes)
                print(row)

    def mask_per_etype(self, etype_performance):
        """
            Given performance numbers for each rank each etype ...
            Build a relocation mask (boolean) for downstream flow planner.
        """       
        
        pass

    def write_g1_median_csvs(self, g1a, g1s, g1r, g1idx: int = 1):
        """
        Snapshot g1 medians to CSV in integer microseconds:
        - g1-all-median-ms.csv   : columns r0,...,r{P-1}
        - g1-cmp-median-ms.csv   : columns r0,...,r{P-1}
        - g1-send-median-ms.csv  : columns i<sender>-<receiver> for all j!=i
        - g1-recv-median-ms.csv  : same pattern as send
        Header is created only if the file does not exist.
        """
        comm, rank, root = get_comm_rank_root('world')
        P = comm.size

        # Scale to microseconds and cast to int
        all_us  = np.rint(g1a * 1e6).astype(np.int64)
        send_us = np.rint(g1s * 1e6).astype(np.int64)
        recv_us = np.rint(g1r * 1e6).astype(np.int64)

        if rank != root:
            return

        # --- ALL / CMP vectors ---
        rcols = [f"r{r}" for r in range(P)]
        _append_csv_row('g1-all-median-ms.csv', rcols, all_us.tolist())

        # --- SEND / RECV full directed off-diagonal matrices ---
        mcols = _flatten_offdiag_labels(P)
        _append_csv_row('g1-send-median-ms.csv', mcols, _flatten_offdiag_values(send_us))
        _append_csv_row('g1-recv-median-ms.csv', mcols, _flatten_offdiag_values(recv_us))

    def get_target(self, ecurrs, scale=1.0):
        """
        Build target using MPI wait-split data
        - e_curr from self.meshes['compute']
        - g1c medians directly from self.system.rhs_compute_times()
        Returns the integer per-rank targets (list[int]) and sets target.
        """

        # --- Comms ---
        comm,  rank,  root  = get_comm_rank_root('world')

        Ntot   = int(ecurrs.sum())
        g1a, _, g1s, g1r = self.get_median_matrices()

        # After calling get_median_matrices(...) or inside advance loop where you snapshot:
        self.write_g1_median_csvs(g1a, g1s, g1r, g1idx=1)


        s_out = g1s.sum(axis=1) # s (sender burden): row-sum of send
        r_in  = g1r.sum(axis=1) # r as experienced locally (remove from 'all'; avoid charging receiver)
        r_out = g1r.sum(axis=0) # r attributed to the sender (transpose row-sum == column-sum of recv)

        self.oneway_mask = g1s + g1r.T > 0
        self.twoway_mask = g1s + g1r > 0

        #burden_current = g1a - (r_in + r_out)    # = (a-r)+r^T = (c+s)+r^T
        burden_target = g1a - (r_in + s_out)*scale + r_out*scale    # = (a-r)     = (c+s)

        # --- Online perf history + regression (tiny, safe to skip if not enough data)
        # hist = self.collect_hist()
        # self.performance_from_hist(hist)

        # per-element cost: exactly what you asked for
        inv_cost = ecurrs / burden_target
        N_star   = Ntot * (inv_cost / inv_cost.sum())
        N_int = np.floor(N_star).astype(np.int64)
        k = int(Ntot - N_int.sum())
        if k:
            frac = N_star - N_int
            order = np.argsort(frac) # asc; deterministic tie-break by index
            if k > 0:  N_int[order[ -k:  ]] += 1 # +1 to largest fractions
            else:      N_int[order[   :-k]] -= 1 # -1 from smallest fractions

        assert int(N_int.sum()) == Ntot, "Target sum must equal current sum"
        return N_int.tolist()

    def get_median_matrices(self, g1idx=1):
        # World/compute comms
        comm, rank, root = get_comm_rank_root('world')
        P = comm.size

        all_times = comm.allgather(self.system.rhs_all_times())
        cmp_times = comm.allgather(self.system.rhs_compute_times())
        ws_send   = comm.allgather(self.system.rhs_wait_times_send())
        ws_recv   = comm.allgather(self.system.rhs_wait_times_recv())

        # Per-rank medians
        all_med = np.fromiter((float(all_times[r][g1idx][3]) for r in range(P)), dtype=float, count=P)
        cmp_med = np.fromiter((float(cmp_times[r][g1idx][3]) for r in range(P)), dtype=float, count=P)
        send_mat = np.fromiter((float(ws_send[i][g1idx][j][3]) for i in range(P) for j in range(P)),dtype=float, count=P*P).reshape(P, P)
        recv_mat = np.fromiter((float(ws_recv[i][g1idx][j][3]) for i in range(P) for j in range(P)),dtype=float, count=P*P).reshape(P, P)

        # Root-only concise logs
        if rank == root:
            print(f"all*1e6=\n{ np.array2string(all_med*1e6,  formatter={'float_kind':lambda x: f'{x:05.0f}'})}")
            print(f"cmp*1e6=\n{ np.array2string(cmp_med*1e6,  formatter={'float_kind':lambda x: f'{x:05.0f}'})}")
            print(f"send*1e6=\n{np.array2string(send_mat*1e6, formatter={'float_kind':lambda x: f'{x:03.0f}'})}")
            print(f"recv*1e6=\n{np.array2string(recv_mat*1e6, formatter={'float_kind':lambda x: f'{x:03.0f}'})}")

        return all_med, cmp_med, send_mat, recv_mat

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
