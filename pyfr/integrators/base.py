from collections import defaultdict, deque
import itertools as it
import re
import sys
import time

import numpy as np

from pyfr.cache import memoize
from pyfr.inifile import Inifile
from pyfr.mpiutil import (initialise_new_comm, mpi, scal_coll,
                          comm, rank, root, rankmap, execute)
from pyfr.plugins import get_plugin

from pyfr.readers.native import NativeReader, _MeshInterconnector

import os

def _append_csv_row(file_path: str, header_cols: list[str], values: list[int]):
    """Root-only: write header if missing, then append one integer row."""
    # Write header once
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            f.write(','.join(header_cols) + '\n')
        #print(f"[g1csv] header_written file='{file_path}' ncols={len(header_cols)}")

    # Append row
    with open(file_path, 'a', newline='') as f:
        f.write(','.join(str(int(v)) for v in values) + '\n')
    #print(f"[g1csv] row_append file='{file_path}' ncols={len(header_cols)}")


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

                p = execute['compute'](lambda: fn(self), default=None)
                
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
        self.dt = cfg.getfloat('solver-time-integrator', 'dt')
        self.dtmin = cfg.getfloat('solver-time-integrator', 'dt-min', 1e-12)

        # Extract the UUID of the mesh (to be saved with solutions)
        self.mesh_uuid = mesh.uuid

        # Create a dictionary to store all the meshes used throughout the simulation
        self.meshes = {'compute': mesh, 'computebest': None}

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
        self.lb_iters = self.cfg.getint('partition', 'load-balancing-iterations', 1)
        self.lb_flowmat_relax = self.cfg.getfloat('partition', 'load-balancing-flowmatrix-relax', 0.5)

        self.lb_best_score       = float('inf')
        self.lb_since_best       = 0
        self.lb_stop_relocating  = False
        self.lb_patience         = 10

        # Cost scales
        self.lb_cost_scale_g1r  = self.cfg.getfloat('partition', 'lb-cost-scale-g1r',  1.0)
        self.lb_cost_scale_g1s  = self.cfg.getfloat('partition', 'lb-cost-scale-g1s',  1.0)
        self.lb_cost_scale_g1rt = self.cfg.getfloat('partition', 'lb-cost-scale-g1rt', 0.0)

        if rank['compute'] == root['compute']:
            print(f"Cost = g1a "
              f"- {self.lb_cost_scale_g1r}*R₁ - {self.lb_cost_scale_g1s}*S₁ "
              f"+ {self.lb_cost_scale_g1rt}*R₁ᵀ")

        # Smoothly step to target time in the last near_t steps
        self.aminf = self.cfg.getfloat('solver-time-integrator', 
                                          'dt-adjust-min-fact', 0.9)
        self.amaxf = self.cfg.getfloat('solver-time-integrator', 
                                          'dt-adjust-max-fact', 1.001)
        self.dt_fallback = cfg.getfloat('solver-time-integrator', 'dt')
        self.dt_near = None

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

        if self.cfg.hasopt('partition', f'online-file'):
            online_cfg = Inifile.load(self.cfg.get('partition', 'online-file'))
        else:
            online_cfg = self.cfg

        if online_cfg.hasopt('partition', f'{goal}-ranklist'):
            # If pname is None, share with 'compute'
            if goal == 'plugins' and pname is None:
                pranks = online_cfg.getliteral('partition', 'compute-ranklist')

                # Copy comm from 'compute' to 'plugins' by reference
                comm['plugins'] = comm['compute']
            else:
                pranks = online_cfg.getliteral('partition', f'{goal}-ranklist')
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

    def write_g1_median_csvs(self, g1a, g1s, g1r, g1idx: int = 1):
        """
        Snapshot g1 medians to CSV in integer microseconds.
        Now always use world-size vectors/matrices and embed compute ranks.
        """
        # Scale to microseconds and cast to int (compute index space)
        all_us  = np.rint(g1a * 1e6).astype(np.int64)
        send_us = np.rint(g1s * 1e6).astype(np.int64)
        recv_us = np.rint(g1r * 1e6).astype(np.int64)

        if rank['compute'] != root['compute']:
            return

        # ---- NEW: embed into world index space ----
        P_world   = comm['world'].size
        comp_wrs  = list(rankmap['compute'])   # world ranks in compute index order
        P_compute = len(comp_wrs)

        assert P_compute == len(all_us)
        assert send_us.shape == (P_compute, P_compute)
        assert recv_us.shape == (P_compute, P_compute)

        # Default value for non-compute ranks (can change to -1 if you prefer)
        missing_val = -1

        all_us_w  = np.full(P_world, missing_val, dtype=np.int64)
        send_us_w = np.full((P_world, P_world), missing_val, dtype=np.int64)
        recv_us_w = np.full((P_world, P_world), missing_val, dtype=np.int64)

        for ic, wr_i in enumerate(comp_wrs):
            all_us_w[wr_i] = all_us[ic]
            for jc, wr_j in enumerate(comp_wrs):
                send_us_w[wr_i, wr_j] = send_us[ic, jc]
                recv_us_w[wr_i, wr_j] = recv_us[ic, jc]

        # --- ALL vector (world size) ---
        rcols = [f"r{r}" for r in range(P_world)]
        _append_csv_row('g1-all-median-ms.csv', rcols, all_us_w.tolist())

        # --- SEND / RECV full directed off-diagonal matrices (world size) ---
        mcols = _flatten_offdiag_labels(P_world)
        _append_csv_row('g1-send-median-ms.csv', mcols, _flatten_offdiag_values(send_us_w))
        _append_csv_row('g1-recv-median-ms.csv', mcols, _flatten_offdiag_values(recv_us_w))

    def compute_cost(self, g1a, g1s, g1r):
        P_old = len(g1a)

        g1a_old = np.asarray(g1a, dtype=float)
        g1s_old = np.asarray(g1s, dtype=float)
        g1r_old = np.asarray(g1r, dtype=float)

        assert g1s_old.shape == (P_old, P_old)
        assert g1r_old.shape == (P_old, P_old)

        s_out = g1s_old.sum(axis=1)
        r_in  = g1r_old.sum(axis=1)
        r_out = g1r_old.sum(axis=0)

        cost_old = g1a_old - r_in  * self.lb_cost_scale_g1r \
                           - s_out * self.lb_cost_scale_g1s \
                           + r_out * self.lb_cost_scale_g1rt

        return cost_old

    def calc_target_ecounts(self, ecurrs, g1a, g1s, g1r):
        """
        Builds target element counts using MPI wait-split data.
        Returns integer per-rank targets in *newcompute* comm
            (world-rank list from rankmap_new).
        """
        # --- Rankmaps for old and new compute comms ---
        # New compute: communicator built from online.ini compute-ranklist
        if g1a is None or g1s is None or g1r is None:
            raise RuntimeError(
                f"[get_target] called with None g1-data on this rank; "
                "this should only be called on ranks in the old compute comm."
            )

        if comm['newcompute'] == mpi.COMM_NULL:
            raise RuntimeError(
                "[get_target] rank is not in newcompute but still reached get_target"
            )

        cost_old = self.compute_cost(g1a, g1s, g1r)

        # --- Restrict world-indexed element counts to old compute ranks ---

        # rankmap_old is a list of world ranks in old compute order
        ecurrs_old = np.asarray([ecurrs[wr] for wr in rankmap['compute']], dtype=np.int64)
        Ntot = int(ecurrs_old.sum())

        # Snapshot medians to CSV (uses whatever index space g1* live in)
        self.write_g1_median_csvs(g1a, g1s, g1r, g1idx=1)

        # Per-element inverse cost on old ranks
        inv_cost = ecurrs_old / cost_old
        N_star_old = Ntot * (inv_cost / inv_cost.sum())

        # --- Remap N_star_old from old->new using device groups (world space) ---

        # Devices from config: e.g. ['gpu', 'cpu', 'cpu', 'cpu', 'cpu']
        devices = self.cfg.getliteral('backend', 'devices')

        # rankmap_* are world-rank lists, matching ecurrs/devices indexing
        if list(rankmap['compute']) != list(rankmap['newcompute']):
            N_star_new = self.remap_targets_by_ranklist(N_star_old,
                old_ranks=rankmap['compute'], new_ranks=rankmap['newcompute'],
                devices=devices, tag="[lb-group]")
        else:
            # No rank change: old and new are the same
            N_star_new = N_star_old.copy()

        # --- Normalise and round to integer targets in newcompute order ---
        N_int = self.normalise_and_round_targets(N_star_new, Ntot=Ntot, tag="[lb-round]",)
        if comm['newcompute'] != mpi.COMM_NULL and rank['newcompute'] == root['newcompute']:
            print(f"[load-balance] N_int={N_int.tolist()} (sum={int(N_int.sum())})")

        # Return in newcompute's world-rank order (rankmap_new)
        return N_int.tolist()

    def remap_targets_by_ranklist(self, N_star_old, old_ranks, new_ranks, devices, tag="[lb-group]"):
        """
        Remap continuous targets N_star_old from old_ranks -> new_ranks using device groups.

        Parameters
        ----------
        N_star_old : array-like of float, shape (P_old,)
            Continuous targets on the old compute ranks, in old-compute index order.
        old_ranks : list[int]
            World ranks in old compute order (len == len(N_star_old)).
        new_ranks : list[int]
            World ranks in new compute order.
        devices : Sequence[str]
            Device label per *world* rank, e.g. ['gpu','cpu',...].
        tag : str
            Log prefix.

        Returns
        -------
        np.ndarray of float, shape (len(new_ranks),)
            Continuous targets in new compute order.
        """
        N_star_old = np.asarray(N_star_old, dtype=float)
        assert len(N_star_old) == len(old_ranks), \
            f"{tag} len(N_star_old)={len(N_star_old)} != len(old_ranks)={len(old_ranks)}"

        # Map world rank -> index in old-compute array
        wr_to_idx = {wr: i for i, wr in enumerate(old_ranks)}

        # Build device-group → list of old indices
        from collections import defaultdict
        group_to_indices = defaultdict(list)
        for i, wr in enumerate(old_ranks):
            dev = devices[wr]
            group_to_indices[dev].append(i)

        removed = sorted(set(old_ranks) - set(new_ranks))
        print(f"{tag} old_ranks={old_ranks} new_ranks={new_ranks}")
        print(f"{tag} removed_ranks={removed}")
        print(f"{tag} group_to_indices="
              f"{{{', '.join(f'{g}:{idxs}' for g, idxs in group_to_indices.items())}}}")

        N_star_new = np.zeros(len(new_ranks), dtype=float)

        for k, wr in enumerate(new_ranks):
            dev = devices[wr]
            if wr in wr_to_idx:
                # Rank survives: carry its own target
                i_old = wr_to_idx[wr]
                N_star_new[k] = N_star_old[i_old]
                print(f"{tag} wr={wr} (dev={dev}) reused_old idx={i_old} "
                      f"N_star_old={N_star_old[i_old]:.6e}")
            else:
                # New compute rank: use device-group average, else global average
                idxs = group_to_indices.get(dev, [])
                if idxs:
                    val = float(N_star_old[idxs].mean())
                    print(f"{tag} wr={wr} (dev={dev}) new_rank using group_avg over idxs={idxs}: "
                          f"{val:.6e}")
                else:
                    val = float(N_star_old.mean())
                    print(f"{tag} wr={wr} (dev={dev}) new_rank using global_avg: {val:.6e}")
                N_star_new[k] = val

        return N_star_new

    def normalise_and_round_targets(self, N_star, Ntot, tag="[lb-round]"):
        """
        Rescale continuous targets N_star to sum to Ntot and round to integers.

        Parameters
        ----------
        N_star : array-like of float
            Continuous targets (new compute order).
        Ntot : int
            Total element count to preserve.
        tag : str
            Log prefix.

        Returns
        -------
        np.ndarray of int
            Integer targets summing to Ntot.
        """
        N_star = np.asarray(N_star, dtype=float)
        sum_star = float(N_star.sum())

        if sum_star <= 0.0:
            raise ValueError(f"{tag} sum(N_star) <= 0 (got {sum_star})")

        scale = float(Ntot) / sum_star
        N_scaled = N_star * scale
        N_floor = np.floor(N_scaled).astype(np.int64)

        k = int(Ntot - N_floor.sum())

        #print(f"{tag} sum_star={sum_star:.6e} Ntot={Ntot} "
        #      f"sum_floor={int(N_floor.sum())} residual_k={k}")

        if k != 0:
            frac = N_scaled - N_floor
            order = np.argsort(frac)  # ascending

            if k > 0:
                idxs = order[-k:]   # bump largest fractions
                N_floor[idxs] += 1
                print(f"{tag} +1 to indices={idxs.tolist()}")
            else:
                idxs = order[:-k]   # k < 0 → drop smallest fractions
                N_floor[idxs] -= 1
                print(f"{tag} -1 from indices={idxs.tolist()}")

        #print(f"{tag} N_scaled={N_scaled.tolist()}")
        #print(f"{tag} N_int={N_floor.tolist()} (sum={int(N_floor.sum())})")

        return N_floor

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
        if rank['compute'] == root['compute']:
            print(f"all*1e6=\n{ np.array2string(all_med*1e6,  formatter={'float_kind':lambda x: f'{x:05.0f}'})}")
            print(f"send*1e6=\n{np.array2string(send_mat*1e6, formatter={'float_kind':lambda x: f'{x:03.0f}'})}")
            print(f"recv*1e6=\n{np.array2string(recv_mat*1e6, formatter={'float_kind':lambda x: f'{x:03.0f}'})}")

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
