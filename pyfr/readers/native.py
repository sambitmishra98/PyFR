from dataclasses import dataclass, field, replace
import re
from copy import deepcopy

from typing import Dict, Optional, Tuple, List

import h5py
import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import (AlltoallMixin, Scatterer, SparseScatterer, autofree, get_comm_info,
                          mpi, comm, rank, root)
from pyfr.nputil import iter_struct

from pyfr.util import subclass_where
from pyfr.shapes import BaseShape

from tabulate import tabulate

from pyfr.writers.native import NativeWriter


import time
from functools import wraps
from collections import defaultdict

import os


@dataclass
class _Mesh:
    fname: str
    raw: object

    ndims: int = None
    subset: bool = False

    creator: str = None
    codec: list = None
    uuid: str = None
    version: int = None

    etypes: list = field(default_factory=list)
    eidxs: dict = field(default_factory=dict)

    spts: dict = field(default_factory=dict)
    spts_nodes: dict = field(default_factory=dict)
    spts_curved: dict = field(default_factory=dict)

    # Required for relocation only
    faces_cidxs: dict = field(default_factory=dict)
    faces_offs: dict = field(default_factory=dict)

    con: list = field(default_factory=list)
    con_p: dict = field(default_factory=dict)
    bcon: dict = field(default_factory=dict)

_CON_UNFILLED = -2

@dataclass(slots=True)
class State:
    """Swappable per-rank element state (indices + light fields)."""
    eidxs: Dict[str, np.ndarray]       # etype -> (Ne,) int64 GIDs
    con: Dict[str, np.ndarray]         # etype -> (Ne, nfaces, ncols) int64
    spts_nodes: Dict[str, np.ndarray]  # etype -> (Ne, nverts) int64

    def clone(self) -> "State":
        # Deepcopy is fine here; these are small-ish, per-rank, and we want isolation.
        return State(deepcopy(self.eidxs), deepcopy(self.con), deepcopy(self.spts_nodes))

    def relocate_to(self, eidxs_dest: Dict[str, np.ndarray]) -> "State":
        """Relocate con and spts_nodes from self.eidxs -> eidxs_dest."""
        inter = _MeshInterconnector(self.eidxs, eidxs_dest)
        return State(
            eidxs=eidxs_dest,
            con=inter.relocate(self.con, edim=0),
            spts_nodes=inter.relocate(self.spts_nodes, edim=0),
        )


class _MetaMesh:

    __slots__ = ("etypes", "e2i", "bc2id", "mesh_src", "mesh_dest", "i", "j",
        "mpi_vertex_union", "_cache", "_ver", "_prof",
    )

    # simple inclusive-time profiler for bound methods on _MetaMesh (or others with _prof)
    def fff(fn):
        name = fn.__qualname__
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                # record on instance if available; else drop on the floor
                if args and hasattr(args[0], '_prof'):
                    prof = args[0]._prof
                    tot, cnt = prof.get(name, (0.0, 0))
                    prof[name] = (tot + dt, cnt + 1)
        return wrapper

    def _invalidate_topology(self) -> None:
        """Bump topology epoch and drop caches that depend on eidx ownership."""
        self._ver['topology'] = int(self._ver.get('topology', 0)) + 1
        # Conservative: clear all known topology-dependent entries.
        self._cache.pop('owners_map', None)
        self._cache.pop('nei_gid_sets', None)

    def __init__(self, *, mesh_src: _Mesh, etypes, e2i, bc2id, i: State, j: State):
        self.mesh_src  = mesh_src
        self.mesh_dest = None
        self.etypes    = list(etypes)
        self.e2i       = dict(e2i)
        self.bc2id     = dict(bc2id)
        self.i         = i
        self.j         = j
        self.mpi_vertex_union = None
        self._cache = {}
        self._ver = {'topology': 0}
        self._prof = {}

    # --- tiny helpers (widely reused) ---
    def _eids_i(self, et: str) -> np.ndarray:
        """Fast path: (Ne,) int64 view of current GIDs for an etype."""
        return np.asarray(self.eidxs_i.get(et, ()), dtype=np.int64)

    def _etype_order(self):
        """
        Resolve effective element-type order on this rank.

        Priority:
          1) MPICommInfo.etype_order (per-device preference),
          2) self.etypes (whatever exists in this mesh).

        Always intersect with self.etypes and preserve the order from (1)/(2).
        """
        # What exists in this mesh on this rank:
        available = set(self.etypes)

        info = get_comm_info('world')
        base = list(info.etype_order) if info.etype_order else list(self.etypes)

        # Filter to mesh-present etypes, preserving order
        return [et for et in base if et in available]

    @fff
    def _apply_plan_and_commit(self, eidxs_diff: Dict[int, Dict[str, np.ndarray]]) -> None:
        """Common relocate pattern (reset -> build j -> accept)."""
        self._reset_j_with_i()
        self.relocate_i_to_j(eidxs_diff)
        self._accept_j_into_i()

    def _local_count(self) -> int:
        """Total elements on this rank (all etypes)."""
        return int(sum(len(self.eidxs_i.get(et, ())) for et in self.etypes))

    # --- profiling helpers ---
    def _prof_reset(self):
        self._prof.clear()

    def _prof_print_and_reset(self, topk: int = 20):
        """Gather per-rank (time, calls) by function, print at root, then reset."""

        # local -> list of tuples for easier aggregation
        loc = [(k, float(v[0]), int(v[1])) for k, v in self._prof.items()]
        all_loc = comm['world'].gather(loc, root=root['world'])

        if rank['world'] == root['world']:
            agg = defaultdict(lambda: [0.0, 0])
            for lst in all_loc:
                for name, t, c in lst:
                    agg[name][0] += t
                    agg[name][1] += c

            rows = []
            total_time = sum(t for t, _ in agg.values()) or 1e-12
            for name, (t, c) in agg.items():
                rows.append([name, f"{t:8.3f}", f"{c:7d}", f"{(t/c*1000.0) if c else 0:8.3f}", f"{100.0*t/total_time:6.2f}%"])
            # sort by total time desc
            rows.sort(key=lambda r: -float(r[1]))

            headers = ["function", "time_s", "calls", "avg_ms", "share"]
            print("\n[prof] timing per iterate (aggregated across ranks)")
            print(tabulate(rows[:topk], headers=headers, tablefmt='github',
                           colalign=('left','right','right','right','right')))
        # reset everywhere
        self._prof_reset()

    @property
    def eidxs_i(self): return self.i.eidxs
    @property
    def con_i(self): return self.i.con
    @property
    def spts_nodes_i(self): return self.i.spts_nodes

    @property
    def eidxs_j(self): return self.j.eidxs
    @eidxs_j.setter
    def eidxs_j(self, v): self.j.eidxs = v
    @property
    def con_j(self): return self.j.con
    @con_j.setter
    def con_j(self, v): self.j.con = v
    @property
    def spts_nodes_j(self): return self.j.spts_nodes
    @spts_nodes_j.setter
    def spts_nodes_j(self, v): self.j.spts_nodes = v

    # -----------------
    # Looking into mesh 
    # -----------------

    # -----------------
    # Helpers
    # ----------------

    @staticmethod
    def _nfaces(etype):
        """Number of faces in an element of a given etype."""
        return len(subclass_where(BaseShape, name=etype).faces)

    @staticmethod
    def _mpi_sorted_union(local):
        """Sorted union of a locally defined set across all ranks."""
        return sorted(set().union(*comm['world'].allgather(set(local))))

    # -----------------
    # _Mesh <--> _MetaMesh

    @classmethod
    def from_mesh(cls, mesh: _Mesh) -> "_MetaMesh":

        etypes   = cls._mpi_sorted_union(set(mesh.etypes or ()))
        bc_names = cls._mpi_sorted_union(set((mesh.bcon or {}).keys()))
        e2i      = {et: i for i, et in enumerate(etypes)}
        bc2id    = {n: i for i, n in enumerate(bc_names)}

        eidxs = {et: np.asarray(mesh.eidxs.get(et, ()), dtype=np.int64)
                for et in etypes}
        con   = {et: np.full((eidxs[et].size, cls._nfaces(et), 4),
                            _CON_UNFILLED, np.int64)
                for et in etypes}
        spts_nodes = deepcopy(mesh.spts_nodes)

        mm = cls(mesh_src=mesh, etypes=etypes, e2i=e2i, bc2id=bc2id,
                i=State(eidxs=deepcopy(eidxs),
                        con=deepcopy(con),
                        spts_nodes=deepcopy(spts_nodes)),
                j=State(eidxs=deepcopy(eidxs),
                        con=deepcopy(con),
                        spts_nodes=deepcopy(spts_nodes)))

        mm._encode_con(mesh)
        mm._fill_con_mpi(mesh)
        return mm

    @fff
    def _encode_con(self, mesh):
        """
            Encode the mesh connectivity as
            con_i[etype][local_id, face_id] = (rank, i, ngid, nface)
            where
                rank  = rank owning the neighbor element ,    -1 for boundary
                i     = idx of nbr etype in self.etypes  ,    -1 for boundary
                ngid  = global ID of the neighbor element, bc_id for boundary
                nface = face ID on the neighbor element  ,    -1 for boundary
        """

        conL, conR = (mesh.con or ([], []))
        bcon = getattr(mesh, 'bcon', {}) or {}

        # Interior pairs: both sides are on *this* rank → owner_rank = rank
        for (etL, lidL, fL), (etR, lidR, fR) in zip(conL, conR):
            gR = int(self.eidxs_i[etR][int(lidR)])
            gL = int(self.eidxs_i[etL][int(lidL)])
            self.con_i[etL][lidL, fL] = (int(rank['world']), self.e2i[etR], gR, int(fR))
            self.con_i[etR][lidR, fR] = (int(rank['world']), self.e2i[etL], gL, int(fL))

        # Boundary: owner=-1, code=-1, gid=bc_id, fid=-1
        for bcname, triples in (bcon.items() if bcon else []):
            bid = self.bc2id.get(bcname)
            if bid is None:
                continue
            for et, lid, f in triples:
                self.con_i[et][lid, f] = (-1, -1, int(bid), -1)

    @fff
    def _fill_con_mpi(self, mesh):
        """
        Fill nrank faces in con_i using mesh.con_p.

        For each nbr,
            build local list of MPI faces with their GIDs, 
            allgather everyone's lists,
            zip list with the nbr's reciprocal list to tag:
            con_i[et, lid, fid] = (nbr, e2i[et_of_nbr], gid_of_nbr, fid_of_nbr)
        """
        cp = getattr(mesh, "con_p", {}) or {}

        # Our local export: {nbr: [(et, lid, fid, gid), ...]}
        local = {
            int(nbr): [(et, int(lid), int(fid), self.eidxs_i[et][int(lid)]) 
                       for (et, lid, fid) in faces]
            for nbr, faces in cp.items()
        }

        # Everyone's exports
        all_cp = comm['world'].allgather(local)

        for nbr, A in local.items():
            B = all_cp[int(nbr)].get(rank['world'], [])
            for (etA, lidA, fA, _gidA), (etB, _lidB, fB, gidB) in zip(A, B):
                self.con_i[etA][lidA, fA] = (int(nbr), self.e2i[etB], gidB, fB)

    @fff
    def to_mesh(self, eidxs_dest):
        ic = _MeshInterconnector(self.mesh_src.eidxs, eidxs_dest)

        mesh_int = replace(self.mesh_src, eidxs=eidxs_dest,
            spts=ic.relocate(self.mesh_src.spts, edim=1), 
            spts_nodes=ic.relocate(self.mesh_src.spts_nodes, edim=0),
            spts_curved=ic.relocate(self.mesh_src.spts_curved, edim=0),
            faces_cidxs=ic.relocate(self.mesh_src.faces_cidxs, edim=0),
            faces_offs=ic.relocate(self.mesh_src.faces_offs, edim=0),
            )

        self._reconstruct_con_conp_bcon(mesh_int)

        eidxs_dest = self._apply_lex_ordering(mesh_int)
        ic2 = _MeshInterconnector(mesh_int.eidxs, eidxs_dest)
        
        mesh_dest = replace(mesh_int, eidxs=eidxs_dest,
            spts=ic2.relocate(mesh_int.spts, edim=1), 
            spts_nodes=ic2.relocate(mesh_int.spts_nodes, edim=0),
            spts_curved=ic2.relocate(mesh_int.spts_curved, edim=0),
            faces_cidxs=ic2.relocate(mesh_int.faces_cidxs, edim=0),
            faces_offs=ic2.relocate(mesh_int.faces_offs, edim=0),
            )

        self._reconstruct_con_conp_bcon(mesh_dest)

        # Remove keys with empty entries
        mesh_dest.eidxs = {k: v for k, v in mesh_dest.eidxs.items() if v.size}

        return mesh_dest

    @fff
    def _reconstruct_con_conp_bcon(self, mesh):
        
        codec = list(mesh.codec or [])
        mesh.bcon = {bc.split('/')[1]: [] for bc in codec if bc.startswith('bc/')}
        
        etypes = mesh.etypes
        eidxs = {k: v.tolist() for k, v in mesh.eidxs.items()}
        glmap = [{k: j for j, k in enumerate(eidxs[etype])} if etype in eidxs else {} for etype in etypes]

        cdone, cefidx = [None]*len(codec), [None]*len(codec)
        for cidx, c in enumerate(codec):
            if (m := re.match(r'eles/(\w+)/(\d+)$', c)):
                etype, fidx = m[1], m[2]
                cdone[cidx] = set()
                cefidx[cidx] = (etype, etypes.index(etype), int(fidx))

        conl, conr = [], []
        bcon = {i: [] for i, c in enumerate(codec) if c.startswith('bc/')}
        resid = {}

        for etype in etypes:
            nfaces = self._nfaces(etype)
            
            efaces = np.empty((len(eidxs[etype]), nfaces), dtype=[('cidx', np.int16), ('off', np.int64)])
            efaces['cidx'] = mesh.faces_cidxs.get(etype, np.empty((0, nfaces), np.int16))
            efaces['off']  = mesh.faces_offs .get(etype, np.empty((0, nfaces), np.int64))

            try:
                for fidx, eface in enumerate(efaces.T):
                    efcidx = codec.index(f'eles/{etype}/{fidx}')

                    for j, (cidx, off) in enumerate(iter_struct(eface)):
                        # Boundary
                        if off == -1:
                            bcon[cidx].append((etype, j, fidx))
                        # Unpaired face
                        elif j not in cdone[efcidx]:
                            # Lookup the element type and face number
                            ketype, ketidx, kfidx = cefidx[cidx]

                            # If our rank has the element then pair it
                            if (k := glmap[ketidx].get(off)) is not None:
                                conl.append((etype, j, fidx))
                                conr.append((ketype, k, kfidx))
                                cdone[cidx].add(k)
                            # Otherwise add it to the residual dict
                            else:
                                resid[efcidx, eidxs[etype][j]] = (cidx, off)
            except:
                pass

        # Add the internal connectivity to the mesh
        mesh.con = (conl, conr)

        for k, v in bcon.items():
            if v:
                mesh.bcon[codec[k][3:]] = v
            else:
                del mesh.bcon[codec[k][3:]]

        neighbours = [i for i in range(comm['world'].size) if i != rank['world']]

        ncomm = comm['world'].Create_dist_graph_adjacent(neighbours,
                                                neighbours)

        # Create a list of our unpaired faces
        unpaired = list(resid.values())

        # Distribute this to each of our neighbours
        nunpaired = ncomm.neighbor_allgather(unpaired)

        # See which of our neighbours unpaired faces we have
        matches = [[resid[j] for j in nunp if j in resid] for nunp in nunpaired]

        # Distribute this information back to our neighbours
        nmatches = ncomm.neighbor_alltoall(matches)

        for nrank, nmatch in zip(neighbours, nmatches):
            if rank['world'] < nrank:
                ncon = [r for l, r in sorted([(resid[m], m) for m in nmatch])]
            else:
                ncon = [l for l, r in sorted([(m, resid[m]) for m in nmatch])]

            nncon = []
            for cidx, off in ncon:
                etype, etidx, fidx = cefidx[cidx]
                nncon.append((etype, glmap[etidx][off], fidx))

            # Add the connectivity to the mesh
            mesh.con_p[nrank] = nncon

        # Find elements sharing an MPI interface
        ifmpi = {etype: [] for etype in self.etypes}
        for nrank, ncon in mesh.con_p.items():
            for etype, eidx, fidx in ncon:
                ifmpi[etype].append(eidx)

        for etype in self.etypes:
            ifmpi[etype] = np.unique(ifmpi[etype])

        # Size by number of elements of this etype on this rank
        mesh.spts_internal = {
            etype: np.ones(len(mesh.eidxs.get(etype, ())), dtype=bool)
            for etype in self.etypes
        }

        for etype, eidxs_ifmpi in ifmpi.items():
            if eidxs_ifmpi.size:
                mesh.spts_internal[etype][eidxs_ifmpi.astype(int)] = False
                
        mesh.con_p = {k: v for k, v in mesh.con_p.items() if v}

    @fff
    def _apply_lex_ordering(self, mesh):
        eidxs_dest = {}

        for et in self.etypes:
            gids = np.asarray(mesh.eidxs.get(et, ()), dtype=np.int64)
            if not gids.size: eidxs_dest[et] = gids; continue
            internal = mesh.spts_internal[et].astype(np.int8, copy=False)
            curved   = mesh.spts_curved  [et].astype(np.int8, copy=False)
            order = np.lexsort((gids, curved, internal))  # primary = internal
            eidxs_dest[et] = gids[order]
        
        return eidxs_dest

    @fff
    def _compute_deltas(self, mode: str):
        if   mode in ('faces',):    return self.calc_mpi_faces_deltas()
        elif mode in ('vertices',): return self.calc_mpi_vertices_delta()
        else:
            raise ValueError(f"Unknown {mode = }' ≠ edges/vertices")

    @fff
    def _gid_owner_maps(self, eidxs) -> dict[str, dict[int, int]]:
        """
        Compact (gid -> owner) per etype via one allgather.
        Caching: if `eidxs` is exactly the current `self.eidxs_i`, return a cache
        keyed by the current topology epoch; otherwise compute without caching.
        """
        # Only cache for the canonical "current" view of indices
        if eidxs is self.eidxs_i:
            gen = int(self._ver.get('topology', 0))
            c   = self._cache.get('owners_map')
            if c is not None and c.get('gen') == gen:
                return c['map']
            out = self._gid_owner_maps_uncached(eidxs)
            self._cache['owners_map'] = {'gen': gen, 'map': out}
            return out
        # Cold path for foreign snapshots
        return self._gid_owner_maps_uncached(eidxs)

    @fff
    def _gid_owner_maps_uncached(self, eidxs) -> dict[str, dict[int, int]]:

        out: dict[str, dict[int, int]] = {}
        for et in self.etypes:
            locg = np.asarray(eidxs.get(et, ()), dtype=np.int64)
            gathered = comm['world'].allgather(locg)  # list of np.ndarrays
            d: dict[int, int] = {}
            for rr, arr in enumerate(gathered):
                if isinstance(arr, np.ndarray) and arr.size:
                    # one Python loop per contributor; no Nx2 arrays/vstack needed
                    d.update({int(g): int(rr) for g in arr})
            out[et] = d
        return out

    @fff
    def calc_mpi_vertices_delta(self) -> dict[int, dict[str, np.ndarray]]:
        mvu = self.collect_mpi_vertex_nodes()

        all_sets  = [np.asarray(v, dtype=np.int64).ravel() for v in mvu.values() if v is not None]
        union_all = np.unique(np.concatenate(all_sets)) if all_sets else np.empty(0, dtype=np.int64)

        def per_et(nds: np.ndarray, nbr_vertices: np.ndarray) -> np.ndarray:
            if nds is None or np.asarray(nds).size == 0:
                return np.empty((0, 2), dtype=np.int64)
            nds = np.asarray(nds, dtype=np.int64)

            v_in_n   = np.isin(nds, nbr_vertices, assume_unique=False)
            cnt_n    = v_in_n.sum(axis=1).astype(np.int16, copy=False)

            v_internal = np.isin(nds, union_all, invert=True, assume_unique=False) if union_all.size \
                        else np.ones_like(nds, dtype=bool)
            cnt_int  = v_internal.sum(axis=1).astype(np.int16, copy=False)

            sel = (cnt_n > 0)
            if not np.any(sel):
                return np.empty((0, 2), dtype=np.int64)

            lids  = np.nonzero(sel)[0].astype(np.int64, copy=False)
            # Make semantics consistent with face-based deltas:
            # Δ = cnt_int - cnt_n  (negative = more tied to neighbor than to interior)
            delta = (cnt_int[sel].astype(np.int32) - cnt_n[sel].astype(np.int32)).astype(np.int64, copy=False)

            mat = np.c_[lids, delta]
            order = np.lexsort((mat[:, 0], mat[:, 1]))
            return mat[order]

        out: dict[int, dict[str, np.ndarray]] = {}
        for nrank, verts in mvu.items():
            vn  = np.asarray(verts, dtype=np.int64).ravel()
            per = {et: per_et(self.spts_nodes_i.get(et), vn) for et in self.etypes}   # <<< _i here
            out[int(nrank)] = per
        return out

    @fff
    def build_moves_from_delta(self, deltas_by_rank, target_moves, etype_order=None):
        order = [et for et in (etype_order or self.etypes) if et in self.etypes]

        # Gather the min/max Δ to cap the search
        def all_deltas():
            for per_et in deltas_by_rank.values():
                for et in order:
                    a = per_et.get(et)
                    if a is not None and a.size:
                        yield from a[:, 1].tolist()

        try:
            dmin = min(all_deltas())
            dmax = max(all_deltas())
        except ValueError:
            return {}

        thresh = 0  # or start at dmin to be most permissive
        chosen = {}
        picked_once = set()

        while len(picked_once) < target_moves and thresh <= dmax:
            added = 0
            for nrank in sorted(deltas_by_rank):
                per_et = deltas_by_rank[nrank]
                for et in order:
                    arr = per_et.get(et)
                    if arr is None or arr.size == 0:
                        continue
                    # allow Δ ≤ thresh (thresh grows even if we picked nothing last round)
                    sel = arr[arr[:, 1] <= thresh]
                    if sel.size == 0:
                        continue
                    lids = sel[:, 0].astype(np.int64, copy=False)
                    gids = np.asarray(self.eidxs_i[et], np.int64)[lids]
                    take = [int(g) for g in gids if g not in picked_once]
                    if take:
                        chosen.setdefault(int(nrank), {}).setdefault(et, set()).update(take)
                        picked_once.update(take)
                        added += len(take)
                    if len(picked_once) >= target_moves:
                        break
                if len(picked_once) >= target_moves:
                    break
            # IMPORTANT: escalate even if added == 0
            thresh += 1

        return {int(n): {et: np.asarray(sorted(s), np.int64) for et, s in per.items() if s}
                for n, per in chosen.items()}

    def build_parallel_moves_to_targets(self, target_counts: List[int],
         rank_move_budget: Optional[List[int]] = None,
         etype_order: Optional[List[str]] = None, *,
         verbose: bool = True) -> Dict[int, Dict[str, np.ndarray]]:

        R = len(target_counts)
        tgt = np.asarray(target_counts, dtype=np.int64)

        # ---- global symmetric neighbor graph (one allgather) ----
        loc = set(self._mpi_faces_by_neighbor().keys())
        rows = comm['world'].allgather(loc)
        N = [set() for _ in range(R)]
        for r, ns in enumerate(rows[:R]):
            for s in ns:
                if r != s:
                    N[r].add(int(s)); N[int(s)].add(int(r))

        # ---- global current counts and surplus/need ----
        my_count = sum(len(self.eidxs_i.get(et, ())) for et in self.etypes)
        cur = np.asarray(comm['world'].allgather(int(my_count)), dtype=np.int64)

        surplus = np.maximum(0, cur - tgt)    # donors
        need    = np.maximum(0, tgt - cur)    # receivers

        if rank_move_budget is None:
            budget = surplus.copy()
        else:
            budget = np.maximum(0, np.asarray(rank_move_budget, dtype=np.int64))

        # ---- initial per-edge caps (receiver view) ----
        caps = {r: {} for r in range(R)}      # caps[r][s] = max r may send to s
        for s in range(R):
            if need[s] == 0:
                continue
            donors = [r for r in N[s] if surplus[r] > 0]
            if not donors:
                continue
            tot = int(sum(surplus[r] for r in donors))
            if tot == 0:
                continue

            # proportional to donor surplus; integerize by largest remainder to match exactly need[s]
            ideal  = {r: float(need[s]) * (float(surplus[r]) / float(tot)) for r in donors}
            floors = {r: int(np.floor(ideal[r])) for r in donors}
            left   = int(need[s] - sum(floors.values()))
            if left > 0:
                rema = sorted(((r, ideal[r] - floors[r]) for r in donors), key=lambda x: (-x[1], x[0]))
                for r2, _ in rema:
                    if left == 0: break
                    floors[r2] += 1
                    left -= 1

            for r2 in donors:
                # still edge-local for now
                caps[r2][s] = int(min(floors[r2], surplus[r2], budget[r2]))

        # ---- donor-pass clamp: enforce sum_s caps[r][s] <= budget[r] for every donor r ----
        for r in range(R):
            if not caps[r]:
                continue
            keys = sorted(caps[r])
            vals = np.asarray([caps[r][s] for s in keys], dtype=np.int64)
            total_cap = int(vals.sum())
            b = int(min(budget[r], surplus[r], total_cap))
            if total_cap <= b:
                continue  # already within budget

            # rescale proportionally, then integerize by largest remainder to sum exactly to b
            if total_cap == 0 or b == 0:
                for s in keys:
                    caps[r][s] = 0
                continue

            prop = vals.astype(float) * (float(b) / float(total_cap))
            floors = np.floor(prop).astype(int)
            left = int(b - int(floors.sum()))
            if left > 0:
                rema = prop - floors
                for i in np.argsort(-rema):
                    if left == 0: break
                    floors[i] += 1
                    left -= 1
            for i, s in enumerate(keys):
                caps[r][s] = int(floors[i])

        # ---- pick concrete elements guided by face-deltas (more negative is better) ----
        deltas_by_rank = self.calc_mpi_faces_deltas()  # {nbr: {et: int64[N,2] [lid, delta]}}
        order = [et for et in (etype_order or self.etypes) if et in self.etypes]

        moves_by_rank: Dict[int, Dict[str, np.ndarray]] = {}
        picked_gids_per_et: Dict[str, set] = {et: set() for et in order}

        # helper to push chosen gids per neighbor/etype
        def _add(nbr: int, et: str, gids: List[int]):
            if not gids: return
            a = np.asarray(sorted(gids), dtype=np.int64)
            if a.size:
                moves_by_rank.setdefault(int(nbr), {}).setdefault(et, a)

        # iterate over my neighbors only; obey *clamped* per-edge caps[rank][nbr]
        for nbr in sorted(deltas_by_rank):
            cap = int(caps.get(rank['world'], {}).get(int(nbr), 0))
            if cap <= 0:
                continue

            chosen_pairs: List[Tuple[str, int]] = []  # (etype, gid)
            picked = 0

            # gather candidates across etypes, sorted by delta ascending (most negative first)
            cand = []
            for et in order:
                arr = deltas_by_rank[nbr].get(et)
                if arr is None or arr.size == 0:
                    continue
                lids = arr[:, 0].astype(np.int64, copy=False)
                dlt  = arr[:, 1].astype(np.int64, copy=False)
                eids = np.asarray(self.eidxs_i[et], dtype=np.int64)
                for lid, d in zip(lids, dlt):
                    gid = int(eids[int(lid)])
                    if gid in picked_gids_per_et[et]:
                        continue
                    cand.append((int(d), et, int(gid)))

            cand.sort(key=lambda t: (t[0], t[2]))  # by delta, then gid for determinism

            for _, et, gid in cand:
                if picked >= cap:
                    break
                picked_gids_per_et[et].add(gid)
                chosen_pairs.append((et, gid))
                picked += 1

            if chosen_pairs:
                for et in order:
                    gs = [g for e, g in chosen_pairs if e == et]
                    _add(int(nbr), et, gs)

        if verbose:
            tot = sum(len(v) for d in moves_by_rank.values() for v in d.values())
            print(f"[parallel.plan] rank={rank['world']} cur={int(cur[rank['world']])} tgt={int(tgt[rank['world']])} "
                f"surplus={int(surplus[rank['world']])} budget={int(budget[rank['world']])} planned_moves={tot} ")

        return moves_by_rank

    @fff
    def _mpi_faces_by_neighbor(self) -> dict[int, list[tuple[str, int, int]]]:
        """
        Group *our* MPI faces by neighbor rank using self.con and gid->owner maps.
        Returns: { nbr_rank : [(etype, lid, fidx), ...], ... }
        Accepts con[..., :] with 4 cols = (owner, code, ngid, nfid).
        """

        per_nbr: dict[int, list[tuple[str, int, int]]] = {}

        for et in self.etypes:
            con_et = self.con_i.get(et)
            if con_et is None or con_et.size == 0:
                continue

            con_et = np.asarray(con_et)
            Ne, nfaces, ncols = con_et.shape

            for lid in range(Ne):
                for f in range(nfaces):
                    rec = con_et[lid, f]
                    owner, code, ngid, nf = map(int, rec)
                    if code == _CON_UNFILLED or code < 0: # boundary (-1, bc_id, -1)
                        continue

                    if owner != rank['world']:
                        per_nbr.setdefault(int(owner), []).append((et, int(lid), int(f)))

        return per_nbr

    def _face_vertex_indices(self, et: str) -> list[np.ndarray]:
        """
        For each etype, return a list whose i-th item is an np.int64 array of
        the element-vertex column indices (columns of spts_nodes[et]) that
        belong to face i.  Ordering is consistent with BaseShape.<etype>.faces
        in shapes.py.  Orientation around the face is kept consistent but is
        not relied upon here (we only need the vertex membership).
        """
        et = str(et).lower()

        if et == 'quad':
            # faces: 0:(s,-1) 1:(1,s) 2:(s,1) 3:(-1,s)
            return [np.array(a, np.int64) for a in (
                [0, 1],   # bottom
                [1, 2],   # right
                [2, 3],   # top
                [3, 0],   # left
            )]

        if et == 'tri':
            # faces: 0:(s,-1) 1:(-s,s) 2:(-1,s)
            return [np.array(a, np.int64) for a in (
                [0, 1],
                [1, 2],
                [2, 0],
            )]

        if et == 'hex':
            # vertex order assumed:
            # V0(-1,-1,-1) V1( 1,-1,-1) V2( 1, 1,-1) V3(-1, 1,-1)
            # V4(-1,-1, 1) V5( 1,-1, 1) V6( 1, 1, 1) V7(-1, 1, 1)
            # faces in shapes.py:
            # 0:(s,t,-1)  1:(s,-1,t) 2:(1,s,t) 3:(s,1,t) 4:(-1,s,t) 5:(s,t,1)
            return [np.array(a, np.int64) for a in (
                [0, 1, 2, 3],  # z = -1 (bottom)
                [0, 1, 5, 4],  # y = -1
                [1, 2, 6, 5],  # x =  1
                [3, 2, 6, 7],  # y =  1
                [0, 3, 7, 4],  # x = -1
                [4, 5, 6, 7],  # z =  1 (top)
            )]

        if et == 'tet':
            # vertex order assumed:
            # V0 base corner; V1,V2 on base plane; V3 apex
            # faces in shapes.py:
            # 0:(s,t,-1)  1:(s,-1,t)  2:(-1,t,s)  3:(s,t,-s-t-1)
            return [np.array(a, np.int64) for a in (
                [0, 1, 2],  # z = -1 base
                [0, 1, 3],  # y = -1
                [0, 2, 3],  # x = -1
                [1, 2, 3],  # slanted face
            )]

        if et == 'pri':
            # vertex order assumed:
            # bottom tri: V0,V1,V2 ; top tri: V3,V4,V5 (same (x,y), z=+1)
            # faces in shapes.py:
            # 0:(s,t,-1)  1:(s,t,1)
            # 2:(s,-1,t)  3:(-s,s,t)  4:(-1,s,t)
            return [np.array(a, np.int64) for a in (
                [0, 1, 2],      # bottom triangle
                [3, 4, 5],      # top triangle
                [0, 1, 4, 3],   # quad over edge V0-V1
                [1, 2, 5, 4],   # quad over edge V1-V2
                [2, 0, 3, 5],   # quad over edge V2-V0
            )]

        if et == 'pyr':
            # vertex order assumed:
            # base quad: V0,V1,V2,V3 (z=-1), apex V4
            # faces in shapes.py:
            # 0:(quad,s,t,-1)
            # 1:(tri, s+(t+1)/2, (t-1)/2,  t)     -> base edge V0-V1
            # 2:(tri, (1-t)/2,   -s-(t+1)/2, t)   -> base edge V1-V2
            # 3:(tri, -s-(t+1)/2,(1-t)/2,   t)    -> base edge V2-V3
            # 4:(tri, (t-1)/2,   s+(t+1)/2, t)    -> base edge V3-V0
            return [np.array(a, np.int64) for a in (
                [0, 1, 2, 3],  # base quad
                [0, 1, 4],
                [1, 2, 4],
                [2, 3, 4],
                [3, 0, 4],
            )]

        raise NotImplementedError(f"'{et}' not implemented")


    @fff
    def collect_mpi_vertex_nodes(self) -> dict[int, np.ndarray]:
        """
        Return {nbr_rank: np.ndarray[int64]} = sorted-unique global vertex-node IDs
        on MPI faces to that neighbor.
        """
        if not hasattr(self, "spts_nodes_i") or self.spts_nodes_i is None:
            raise RuntimeError("self.spts_nodes_i missing")

        # 1) which faces are MPI, grouped by neighbor
        per_nbr_faces = self._mpi_faces_by_neighbor()

        # 2) per-etype face->vertex columns
        face_vtx_by_et = {et: self._face_vertex_indices(et) for et in self.etypes}

        # 3) union vertex nodes per neighbor
        out: dict[int, np.ndarray] = {}
        for nbr, faces in per_nbr_faces.items():
            vset: set[int] = set()
            for et, lid, f in faces:
                nds = self.spts_nodes_i.get(et)
                if nds is None or nds.size == 0:
                    continue
                if lid < 0 or lid >= nds.shape[0]:
                    continue
                fv = face_vtx_by_et[et][int(f)]
                verts = np.asarray(nds[int(lid), fv], dtype=np.int64).ravel()
                for v in verts:
                    iv = int(v)
                    if iv >= 0:
                        vset.add(iv)
            out[int(nbr)] = (np.asarray(sorted(vset), dtype=np.int64)
                             if vset else np.empty(0, dtype=np.int64))

        self.mpi_vertex_union = out
        return out

    # ---------------------
    # Relocation iterations 
    # ---------------------
    
    @fff
    def plan_eidxs_dest_from_diff(self, eidxs_diff, verbose=False, *,
                                  use_alltoall=True):
        """
        Build new per-etype GID lists after a proposed relocation plan.
        """
        import os

        if use_alltoall is None:
            use_alltoall = os.getenv('PYFR_PLAN_A2A', '0') not in ('', '0')

        if not use_alltoall:
            # --- Allgather path ---
            send_plan = {
                d: {et: np.asarray(eidxs_diff.get(d, {}).get(et, ()), dtype=np.int64)
                    for et in self.etypes}
                for d in range(comm['world'].size) if d != rank['world']
            }
            all_plans = comm['world'].allgather(send_plan)

            to_send = {}
            for et in self.etypes:
                chunks = [send_plan[d][et] for d in send_plan if send_plan[d][et].size]
                to_send[et] = (np.unique(np.concatenate(chunks))
                               if chunks else np.empty(0, dtype=np.int64))

            to_recv = {et: np.empty(0, dtype=np.int64) for et in self.etypes}
            for sender, plan in enumerate(all_plans):
                if sender == rank['world']:
                    continue
                per_me = plan.get(rank['world'])
                if not per_me:
                    continue
                for et in self.etypes:
                    a = per_me.get(et)
                    if a is not None and a.size:
                        to_recv[et] = np.concatenate((to_recv[et], a))

            for et in self.etypes:
                if to_recv[et].size:
                    to_recv[et] = np.unique(to_recv[et])

        else:
            # --- Alltoallv fast path ---
            _a2a = AlltoallMixin()
            R = comm['world'].size

            to_send = {}
            to_recv = {et: np.empty(0, dtype=np.int64) for et in self.etypes}

            for et in self.etypes:
                scount = np.zeros(R, dtype=np.int32)
                chunks = []
                for d in range(R):
                    if d == rank['world']:
                        continue
                    a = np.asarray(eidxs_diff.get(d, {}).get(et, ()), dtype=np.int64)
                    scount[d] = a.size
                    if a.size:
                        chunks.append(a)
                svals = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
                to_send[et] = svals

                rvals, _ = _a2a._alltoallcv(comm['world'], svals, scount)
                to_recv[et] = (np.unique(rvals) if rvals.size else np.empty(0, dtype=np.int64))

        # --- Build new_eidxs ---
        new_eidxs = {}
        for et in self.etypes:
            cur = np.asarray(self.eidxs_i.get(et, ()), dtype=np.int64)

            ts = to_send[et]
            if cur.size and ts.size:
                keep = cur[~np.isin(cur, ts, assume_unique=True)]
            else:
                keep = cur

            rv = to_recv[et]
            if rv.size:
                add = rv[~np.isin(rv, keep, assume_unique=True)] if keep.size else rv
                new = np.concatenate((keep, add)) if add.size else keep
            else:
                new = keep

            new_eidxs[et] = new

        if verbose:
            tag = 'a2a' if use_alltoall else 'ag'
            tsz = {et: int(to_send[et].size) for et in self.etypes}
            rsz = {et: int(to_recv[et].size) for et in self.etypes}
            nsz = {et: int(new_eidxs[et].size) for et in self.etypes}
            print(f"[plan/{tag}] R{rank['world']}\tsend:{tsz}\trecv:{rsz}\ttarget:{nsz}")

        return new_eidxs
    

    def _reset_j_with_i(self):
        self.j = self.i.clone()
        self.mpi_vertex_union = None

    def _accept_j_into_i(self):
        self.i, self.j = self.j, self.i
        self._retag_con_owners_from_gid_maps()
        self.mpi_vertex_union = None
        self._invalidate_topology()

    def relocate_i_to_j(self, eidxs_diff):
        self.j = self.i.relocate_to(self.plan_eidxs_dest_from_diff(eidxs_diff))

    @fff
    def _retag_con_owners_from_gid_maps(self):
        """
            Retag owners in new con_i from current gid->owner maps
        """

        owners = self._gid_owner_maps(self.eidxs_i)              # {etype: {gid: owner_rank}}

        for et in self.etypes:
            con = self.con_i.get(et)
            if con is None or not np.asarray(con).size:
                continue

            con = np.asarray(con, dtype=np.int64, copy=False)
            codes = con[..., 1]                      # neighbor etype code (>=0 = paired)
            ngids = con[..., 2]                      # neighbor GID

            m = (codes >= 0)
            if not np.any(m):
                continue

            # Update owner column from the current gid->owner map
            # Loop by neighbor etype to avoid per-face string work
            for k, net in enumerate(self.etypes):
                mk = m & (codes == k)
                if not np.any(mk):
                    continue
                og = owners.get(net, {})
                # per-face lookup; fast enough in practice
                con[..., 0][mk] = np.array([og.get(int(g), rank['world']) for g in ngids[mk]], dtype=np.int64)

            # keep boundary/unfilled as-is (codes < 0)

    # -----------------

    def smooth_until_stagnates(self, *, max_iters: int = 20, patience: int = 1,
                            min_change: int = 0, threshold = 0,
                            target_counts: Optional[List[int]] = None,
                            restrict_src_dest: bool = False) -> list[int]:
        """
        Run diffuse_smoothing repeatedly. The stopping decision is collective:
        we allreduce the local planned-move counts so every rank takes the same
        number of iterations. Returns the GLOBAL per-iteration move counts.
        """
        history: list[int] = []
        stable = 0
        last = None

        for _ in range(int(max_iters)):
            moved_local = int(self.diffuse_smoothing(
                return_count=True,
                threshold=threshold,
                target_counts=target_counts,
                restrict_src_dest=restrict_src_dest
            ) or 0)

            moved = comm['world'].allreduce(moved_local, op=mpi.SUM)

            history.append(moved)

            if last is not None and abs(moved - last) <= int(min_change):
                stable += 1
            else:
                stable = 0
            last = moved

            # collective stop condition
            stop = (moved == 0) or (stable >= int(patience))
            # ensure every rank evaluates the same boolean
            stop_all = comm['world'].allreduce(1 if stop else 0, op=mpi.MAX)
            if stop_all:
                break

        return history

    # -----------------
    # Testing / ideas
    # -----------------

    # change signature and add gating in diffuse_smoothing

    @fff
    def diffuse_smoothing(self, *,
                        return_count: bool = False,
                        threshold: int = 0,
                        target_counts: Optional[List[int]] = None,
                        restrict_src_dest: bool = False) -> Optional[int]:

        # 0) start candidate from current
        self._reset_j_with_i()

        # Directional gating
        allowed_nbrs = None
        I_am_donor = True
        if restrict_src_dest and target_counts is not None:
            my_count = int(sum(len(self.eidxs_i.get(et, ())) for et in self.etypes))
            cur = np.asarray(comm['world'].allgather(int(my_count)), dtype=np.int64)
            tgt = np.asarray(target_counts, dtype=np.int64)
            diff = cur - tgt  # +surplus / -deficit
            I_am_donor = diff[rank['world']] > 0

            if I_am_donor:
                # donors may send only to neighbors with deficit
                allowed_nbrs = {int(n) for n in range(comm['world'].size) if diff[int(n)] < 0}
            else:
                # receivers: we will still do the collective relocate (with empty plan)
                allowed_nbrs = set()

        # 1) per-neighbor, per-etype deltas
        deltas_by_rank = self.calc_mpi_faces_deltas()

        # 2) build move plan (respect allowed_nbrs if set)
        eidxs_diff: dict[int, dict[str, np.ndarray]] = {}

        for et in self.etypes:
            parts = []
            for nbr, per_et in deltas_by_rank.items():
                if allowed_nbrs is not None and int(nbr) not in allowed_nbrs:
                    continue
                arr = per_et.get(et)
                if arr is None or arr.size == 0:
                    continue
                lids = arr[:, 0].astype(np.int64, copy=False)
                dlt  = arr[:, 1].astype(np.int64, copy=False)
                nbrs = np.full(lids.shape, int(nbr), dtype=np.int64)
                parts.append((lids, dlt, nbrs))

            if not parts:
                continue

            lids = np.concatenate([p[0] for p in parts])
            dlt  = np.concatenate([p[1] for p in parts])
            nbrs = np.concatenate([p[2] for p in parts])

            order = np.lexsort((dlt, lids))
            lids_s, dlt_s, nbrs_s = lids[order], dlt[order], nbrs[order]
            _, first_idx = np.unique(lids_s, return_index=True)
            best_lids  = lids_s[first_idx]
            best_dlt   = dlt_s[first_idx]
            best_nbrs  = nbrs_s[first_idx]

            # threshold semantics: < threshold → with threshold=1, allow Δ<=0
            mkeep = (best_dlt < int(threshold))
            if not np.any(mkeep):
                continue

            chosen_lids = best_lids[mkeep]
            chosen_nbrs = best_nbrs[mkeep]

            eids = np.asarray(self.eidxs_i[et], dtype=np.int64)
            gids = eids[chosen_lids]

            if chosen_nbrs.size:
                un, inv = np.unique(chosen_nbrs, return_inverse=True)
                for i, nbr in enumerate(un.tolist()):
                    gsel = np.asarray(sorted(gids[inv == i].tolist()), dtype=np.int64)
                    if gsel.size:
                        eidxs_diff.setdefault(int(nbr), {})[et] = gsel

        breakdown, tot = {}, 0
        for nbr, per in eidxs_diff.items():
            b = {et: int(len(g)) for et, g in per.items()}
            breakdown[int(nbr)] = b
            tot += sum(b.values())

        # Helpful trace (donor/receiver)
        role = "donor" if I_am_donor else "receiver"

        # 3) ALWAYS participate in relocate (even with empty plan)
        self._apply_plan_and_commit(eidxs_diff)

        return int(tot) if return_count else None

    def _global_mpi_faces_total(self):
        # Count MPI faces once, symmetrised, on the MetaMesh communicator

        per = {}
        for et in self.etypes:
            con = np.asarray(self.con_i.get(et, ()), dtype=np.int64)
            if con.size == 0:
                continue
            owners = con[..., 0]
            codes  = con[..., 1]
            m = (codes >= 0) & (owners != rank['world'])
            per[et] = int(m.sum())
        loc  = sum(per.values())
        glob = comm['world'].allreduce(loc, op=mpi.SUM)
        return glob // 2

    @fff
    def element_flow_plan(self, cur_counts, tgt_counts, mask, *,
                          max_iters: int = 4) -> np.ndarray:
        """
        Electrical-flow, cap-aware, integer planner on the rank graph (mask-aware).

        Parameters
        ----------
        cur_counts : sequence[int]
            Current per-rank element counts (world-length).
        tgt_counts : sequence[int | None]
            Target per-rank element counts. Entries may be None, in which case
            the target for that rank is taken to be its current count
            (i.e. unconstrained / neutral).
        mask : np.ndarray[bool] (R,R) or (R_active,R_active)
            Directional allow mask. If world-sized (R,R), it is interpreted
            directly in world rank space. If smaller (R_active,R_active), it is
            assumed to correspond to the subset of "active" ranks (those with
            not (cur==0 and tgt==0)) and is embedded into world-space.
            Diagonal is forced to False.

        Returns
        -------
        M : (R,R) int64
            Cumulative shipments in world rank space. Only neighbor edges
            nonzero; diag=0. Enforces adjacency and the provided directional mask.
        """

        # ---------- helpers ----------

        def _build_faces_matrix() -> tuple[np.ndarray, list[list[int]]]:
            """F[r,s] = symmetric MPI face counts; N[r] = neighbor list."""
            per_nbr = self._mpi_faces_by_neighbor()
            nlocal  = {int(n): len(faces) for n, faces in per_nbr.items()}

            A = np.zeros((comm['world'].size, comm['world'].size),
                         dtype=np.int64)
            for n, c in nlocal.items():
                A[rank['world'], int(n)] = int(c)

            A = comm['world'].allreduce(A, op=mpi.SUM)
            F = np.minimum(A, A.T).astype(np.float64)
            N = [np.nonzero(F[r])[0].tolist()
                 for r in range(comm['world'].size)]
            return F, N

        def _lap_solve(L: np.ndarray, b: np.ndarray, anchor: int = 0) -> np.ndarray:
            R = L.shape[0]
            if R == 0:
                return np.zeros(0, dtype=np.float64)
            if R == 1:
                return np.zeros(1, dtype=np.float64)

            idx = [i for i in range(R) if i != anchor]
            Lr  = L[np.ix_(idx, idx)]
            br  = b[idx].astype(np.float64, copy=True)

            if not Lr.size:
                return np.zeros(R, dtype=np.float64)

            try:
                pr = np.linalg.solve(Lr, br)
            except np.linalg.LinAlgError:
                # Mild regularisation in the rare case the reduced Laplacian
                # is still singular (e.g. multiple disconnected components).
                Lr_reg = Lr + 1e-8*np.eye(Lr.shape[0], dtype=Lr.dtype)
                pr = np.linalg.solve(Lr_reg, br)

            p = np.zeros(R, dtype=np.float64)
            p[idx] = pr
            return p

        def _downhill_weights(p: np.ndarray, Fm: np.ndarray, N: list[list[int]]) -> np.ndarray:
            R = Fm.shape[0]
            T = np.zeros_like(Fm, dtype=np.float64)
            for u in range(R):
                pu = p[u]
                for v in N[u]:
                    if u == v:
                        continue
                    drop = pu - p[v]
                    if drop > 0:
                        T[u, v] = Fm[u, v] * drop
            return T

        def _distribute_with_caps(
            excess: np.ndarray, deficit: np.ndarray, T: np.ndarray, Fm: np.ndarray
        ) -> np.ndarray:
            R   = T.shape[0]
            eps = 1e-12
            P   = np.zeros((R, R), dtype=np.float64)

            # 1) downhill proportional split by T
            for u in range(R):
                e = int(excess[u])
                if e <= 0:
                    continue
                w = T[u].copy()
                s = float(w.sum())
                if s > eps:
                    P[u] = e * (w / s)

            # 2) cap each sink column by remaining deficit
            sinks = np.where(deficit > 0)[0]
            if sinks.size:
                in_s  = P[:, sinks].sum(axis=0)
                alpha = np.ones_like(in_s, dtype=np.float64)
                m = in_s > (deficit[sinks].astype(np.float64) + eps)
                alpha[m] = deficit[sinks][m].astype(np.float64) / in_s[m]
                if np.any(alpha < 1 - 1e-12):
                    P[:, sinks] *= alpha

            # 3) leftover per source goes to relay neighbors, proportional to Fm
            for u in range(R):
                e = int(excess[u])
                if e <= 0:
                    continue
                rem = float(e) - float(P[u].sum())
                if rem <= 1e-9:
                    continue
                cand = [v for v in range(R)
                        if (deficit[v] == 0 and T[u, v] > eps)]
                if not cand:
                    continue
                w = Fm[u, cand].astype(np.float64)
                s = float(w.sum())
                if s > eps:
                    P[u, cand] += rem * (w / s)

            # 4) integerize per source
            S = np.zeros((R, R), dtype=np.int64)
            for u in range(R):
                e = int(excess[u])
                if e <= 0:
                    continue
                base = np.floor(P[u] + 1e-9).astype(np.int64)
                left = int(e - int(base.sum()))
                if left > 0:
                    rema  = P[u] - base
                    order = np.lexsort((np.arange(R), -Fm[u], -rema))
                    for j in order[:left]:
                        base[j] += 1
                S[u] = base
            return S

        def _cancel_anti_parallel(M: np.ndarray) -> np.ndarray:
            Mc = M.copy()
            R  = Mc.shape[0]
            for i in range(R):
                for j in range(i + 1, R):
                    a, b = int(Mc[i, j]), int(Mc[j, i])
                    if a and b:
                        if a >= b:
                            Mc[i, j], Mc[j, i] = a - b, 0
                        else:
                            Mc[j, i], Mc[i, j] = b - a, 0
            return Mc

        # ---------- world-space inputs ----------

        # cur_full / tgt_full are always world-length
        cur_full = np.asarray(cur_counts, dtype=np.int64).copy()

        if tgt_counts is None:
            tgt_full = cur_full.copy()
        else:
            try:
                tgt_full = np.asarray(tgt_counts, dtype=np.int64).copy()
            except (TypeError, ValueError):
                tmp: list[int] = []
                for i, v in enumerate(tgt_counts):
                    if v is None:
                        tmp.append(int(cur_full[i]))
                    else:
                        tmp.append(int(v))
                tgt_full = np.asarray(tmp, dtype=np.int64)

        R_full = cur_full.size

        # Faces at world resolution
        F_full, _ = _build_faces_matrix()

        # ---------- normalise mask to world shape & decide active set ----------

        mask_arr = np.array(mask, dtype=bool, copy=True)

        if mask_arr.shape == F_full.shape:
            # World-space mask: active ranks are those that actually have / will have elements
            mask_full = mask_arr
            active = [i for i, (c, t) in enumerate(zip(cur_full, tgt_full))
                      if not (c == 0 and t == 0)]
        else:
            # Treat a smaller mask as being defined over ranks [0 .. R_mask-1]
            R_mask = mask_arr.shape[0]
            if mask_arr.shape != (R_mask, R_mask):
                raise ValueError(f"mask must be square; got {mask_arr.shape}")

            # Embed the smaller mask into world space (upper-left block)
            mask_full = np.zeros_like(F_full, dtype=bool)
            mask_full[:R_mask, :R_mask] = mask_arr

            # Active ranks are exactly those covered by the mask
            active = list(range(R_mask))

        if not active:
            # No movable elements anywhere: nothing to do.
            return np.zeros((R_full, R_full), dtype=np.int64)

        np.fill_diagonal(mask_full, False)



        # ---------- restrict to active subgraph ----------

        cur = cur_full[active]
        tgt = tgt_full[active]

        F    = F_full[np.ix_(active, active)]
        maskb = mask_full[np.ix_(active, active)]

        R = len(active)

        # Undirected support for Laplacian; edges only where F>0 and mask allows something
        support = (maskb | maskb.T) & (F > 0)
        Fm      = F * support.astype(np.float64)

        # If mask disconnects a donor completely, relax by opening its strongest-F neighbor
        diff   = cur - tgt
        donors = np.where(diff > 0)[0]
        for u in donors:
            if support[u].sum() == 0:
                v = int(np.argmax(F[u]))
                if F[u, v] > 0:
                    support[u, v] = True
                    support[v, u] = True
                    Fm[u, v] = F[u, v]
                    Fm[v, u] = F[v, u]

        N = [np.nonzero(Fm[r])[0].tolist() for r in range(R)]
        L = np.diag(Fm.sum(axis=1)) - Fm

        M_sub = np.zeros_like(F, dtype=np.int64)

        for it in range(int(max_iters)):
            diff    = cur - tgt
            deficit = np.maximum(0, -diff).astype(np.int64)
            excess  = np.maximum(0,  diff).astype(np.int64)

            if excess.sum() == 0 or deficit.sum() == 0:
                print(f"[elec_cap] it={it} done; "
                      f"excess={excess.tolist()} deficit={deficit.tolist()}")
                break

            b = excess.astype(np.float64) - deficit.astype(np.float64)
            # Within active subgraph the total must still be zero
            assert abs(float(b.sum())) < 1e-6, "b must sum to zero in active subgraph"

            p = _lap_solve(L, b, anchor=0)
            T = _downhill_weights(p, Fm, N)

            # Enforce directional mask on traffic
            T *= maskb.astype(np.float64)

            S = _distribute_with_caps(excess, deficit, T, Fm)

            out = S.sum(axis=1)
            inc = S.sum(axis=0)
            cur = (cur - out + inc).astype(np.int64)
            M_sub += S

        # ---------- expand back to world shape ----------

        M_full = np.zeros_like(F_full, dtype=np.int64)
        for il, i_world in enumerate(active):
            for jl, j_world in enumerate(active):
                M_full[i_world, j_world] = M_sub[il, jl]

        np.fill_diagonal(M_full, 0)
        M_full *= (F_full > 0).astype(np.int64)
        M_full *= mask_full.astype(np.int64)

        return _cancel_anti_parallel(M_full)

    @fff
    def diffuse_smoothing2(self, flow_matrix: np.ndarray, mode, threshold, 
                           scale = 0.5, return_count = False, 
    ) -> Optional[int]:
        """
        mode: 'edge' (default) uses calc_mpi_faces_deltas()
                'vertex'        uses calc_mpi_vertices_delta()
        """

        # Validate flow matrix
        M = np.asarray(flow_matrix, dtype=np.int64)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"Need square matrix; got {M.shape = }")
        R = int(M.shape[0])
        if not (0 <= rank['world'] < R):
            raise ValueError(f"{rank['world'] = } but flow_matrix size is {R}")

        base_order = self._etype_order()

        # 0) start from current snapshot
        self._reset_j_with_i()

        # 1) precompute per-neighbor deltas once (EDGE or VERTEX)
        deltas_by_rank = self._compute_deltas(mode)

        picked_by_et: dict[str, set[int]] = {et: set() for et in base_order}

        def _pick_for_neighbor(nbr: int, need: int) -> dict[str, np.ndarray]:
            if need <= 0:
                return {}
            if not (0 <= nbr < R):
                raise IndexError(f"[itc4.pick] destination rank out of range: nbr={nbr}, R={R}")

            # For now, we keep your fixed per-destination allow-set scaffold.
            allowed = {  0: {'tet', 'pyr', 'hex'},
                         1: {'hex', 'pyr', 'tet'},
                         2: {'hex', 'pyr', 'tet'},
                         3: {'hex', 'pyr', 'tet'},
                         4: {'hex', 'pyr', 'tet'},
                         5: {'hex', 'pyr', 'tet'},
                         6: {'hex', 'pyr', 'tet'},
                         7: {'hex', 'pyr', 'tet'},
                         8: {'hex', 'pyr', 'tet'},
                         9: {'hex', 'pyr', 'tet'},
                        10: {'hex', 'pyr', 'tet'},
                        11: {'hex', 'pyr', 'tet'}, }

            if allowed is not None:
                allowed_set = set(map(str.lower, allowed.get(int(nbr), set())))
            else:
                allowed_set = None

            # Filter this destination's candidate etype order by the allow-set (if any)
            if allowed_set:
                order = [et for et in base_order if et in allowed_set]
            else:
                order = base_order

            chosen: dict[str, list[int]] = {}
            cand = []
            per_et = deltas_by_rank.get(int(nbr), {})  # (lid, delta) pairs already sorted per-et

            for et in order:
                arr = per_et.get(et)
                if arr is None or arr.size == 0:
                    continue
                lids = arr[:, 0].astype(np.int64, copy=False)
                dlt  = arr[:, 1].astype(np.int64, copy=False)
                m = (dlt <= int(threshold))
                if not np.any(m):
                    continue
                lids = lids[m]; dlt = dlt[m]
                eids = np.asarray(self.eidxs_i[et], dtype=np.int64)
                for lid, d in zip(lids, dlt):
                    gid = int(eids[int(lid)])
                    if gid not in picked_by_et[et]:
                        cand.append((int(d), et, gid))

            cand.sort(key=lambda t: (t[0], t[2]))  # delta ascending, then gid
            take = min(int(need), len(cand))
            for _, et, gid in cand[:take]:
                chosen.setdefault(et, []).append(int(gid))
                picked_by_et[et].add(int(gid))

            return {et: np.asarray(sorted(v), dtype=np.int64) for et, v in chosen.items() if v}

        # 2) build relocation diff from this rank
        eidxs_diff: dict[int, dict[str, np.ndarray]] = {}
        out_total = 0
        by_nbr_total = {}

        for nbr in range(R):
            cap = int(M[rank['world'], int(nbr)])
            if cap <= 0:
                continue

            need = int(np.ceil(cap * float(scale)))
            per = _pick_for_neighbor(int(nbr), need)
            if per:
                eidxs_diff[int(nbr)] = per
            tot = sum(len(g) for g in per.values())
            by_nbr_total[int(nbr)] = int(tot)
            out_total += int(tot)
            assert tot <= need <= cap, f"[itc4.pass] rank={rank['world']} nbr={nbr} tot={tot} need={need} cap={cap}"

        # 3) perform the relocation
        self._apply_plan_and_commit(eidxs_diff)
        return int(out_total) if return_count else None


    def seed_rank(self, rank):
        pass


    @fff
    def iterate(self, objective, target_counts, mask, *, flowmat_relax=0.5, ):
        """
        One round controller (as requested):
        1) Plan from current counts (M0).
        2) Single "nudge" pass with threshold=2 using M0.
        3) Replan (M1).
        4) One strict pass with threshold=0 using M1 (loop written as range(comm.size), break after first).
        5) Final smoothing to stagnation (default threshold in existing smoother).

        Notes:
        - We recompute the plan after each phase (no residual-cap bookkeeping).
        - threshold=0 pass does not move anything that would increase edge cuts.
        - No "-1 polish" here by request.
        """

        def _cur_counts_total():
            return list(comm['world'].allgather(self._local_count()))

        if rank['world'] == root['world']: print(f"[itc4.iter]  TARGET={target_counts}")

        if objective == 'to-target':

            exec_order = (
                        [(0, 'vertices')]*comm['world'].size*4
                    # + [(0, 'vertices')]
                    # + [(2,   'faces')]
                    # + [(2,   'faces')]
                        + [(0,   'faces')]*comm['world'].size)
            
            for step in exec_order:
                cur0 = _cur_counts_total()
                if rank['world'] == root['world']: 
                    print(f"mode={step[1]} \t thr={int(step[0])} CURRENT={cur0}")
                M0   = self.element_flow_plan(cur0, target_counts, mask)
                self.diffuse_smoothing2(M0, mode=step[1], threshold=step[0],
                                        scale=flowmat_relax)
                self.smooth_until_stagnates()
                
            self._prof_print_and_reset()   # <--- prints summary + resets counters
            
        elif objective == 'to-remove-rank':
            # Find targets with zero target counts
            zero_target = [i for i, c in enumerate(target_counts) if c == 0]

            cur0 = _cur_counts_total()
            zero_curr0_base = [i for i, c in enumerate(cur0) if c == 0]

            cur1 = [0 for _ in range(len(cur0))]

            # As long as last curr is not same as this curr 
            while cur1 != cur0:
                cur0 = _cur_counts_total()
                zero_curr0 = [i for i, c in enumerate(cur0) if c == 0]
                if len(zero_curr0_base) > len(zero_curr0):
                    if rank['world'] == root['world']:
                        print(f"[itc4.iter] to-remove-rank done; "
                              f"zero-targets={zero_target} zero-current={zero_curr0}")
                    break

                M0 = self.element_flow_plan(cur0, target_counts, mask)
                self.diffuse_smoothing2(M0, mode='vertices', threshold=6,
                                        scale=flowmat_relax)
                self.smooth_until_stagnates()

                cur1 = _cur_counts_total()
                if rank['world'] == root['world']:
                    print(f"[itc4.iter] to-remove-rank CURRENT={cur1} \t DIFF = {[cur1[i]-cur0[i] for i in range(len(cur0))]}")
        else:
            raise ValueError(f"Unknown objective '{objective}'")
            
            
    @staticmethod
    def _partition_stats(mesh):
        """Gather per-rank etype counts and symmetric MPI-face matrix F (i<->j).

        Parameters
        ----------
        mesh : _MetaMesh
            Mesh distributed over the communicator `comm_name`.
        comm_name : str
            Logical communicator name ('compute', 'plugins', etc.).
        """

        # If this rank is not part of the communicator, it contributes nothing
        if comm['compute'] == mpi.COMM_NULL:
            return None

        etypes = tuple(sorted(mesh.etypes))
        R = comm['compute'].size

        # Local element counts per etype
        loc_counts = {et: len(mesh.eidxs.get(et, ())) for et in etypes}

        # Local interface counts per neighbour rank (keys are local ranks in comm_name)
        loc_ifaces = {int(n): len(f) for n, f in mesh.con_p.items()}

        # Gather to the root['world'] of the compute communicator
        all_counts = comm['compute'].gather(loc_counts, root=root['world'])
        all_ifaces = comm['compute'].gather(loc_ifaces, root=root['world'])

        if rank['world'] != root['world']:
            return None

        # --- Build per-rank element count matrix ---
        cnt = np.zeros((R, len(etypes)), np.int64)
        for r, d in enumerate(all_counts):
            for j, et in enumerate(etypes):
                cnt[r, j] = d.get(et, 0)

        # --- Build directed interface matrix A and then symmetric F ---
        A = np.zeros((R, R), np.int64)
        for i, d in enumerate(all_ifaces):
            for j, c in d.items():
                A[i, j] = int(c)

        # Symmetric face count: faces shared between i and j
        F = np.minimum(A, A.T)

        return etypes, cnt, F

    @staticmethod
    def info(mesh):

        ps = _MetaMesh._partition_stats(mesh)

        if rank['world'] != root['world'] or ps is None:
            return

        etypes, cnt, F = ps
        R = cnt.shape[0]

        pairs = [(i, j) for i in range(R) for j in range(i + 1, R)]
        headers = (['etype'] + [f'r{r}' for r in range(R)]
                   + [f'i{i}-{j}' for (i, j) in pairs] + ['pairs', 'faces'])

        rows = [[et] + [f'{int(cnt[r, j]):,}' for r in range(R)]
                + [''] * len(pairs) + ['', '']
                for j, et in enumerate(etypes)]

        pair_vals = [int(F[i, j]) for (i, j) in pairs]
        rows.append(['mpi_faces'] + [''] * R
                    + [f'{v:,}' for v in pair_vals]
                    + [f'{sum(v > 0 for v in pair_vals):,}',
                       f'{sum(pair_vals):,}'])

        print(tabulate(rows, headers=headers, tablefmt='github',
                       colalign=('left', *('right',) * (len(headers) - 1))))

    @staticmethod
    def info_to_csv(mesh, *, tcurr: float, csv_path: str = 'lb_elem_dist.csv'):

        ps = _MetaMesh._partition_stats(mesh)

        if rank['world'] != root['world'] or ps is None:
            return

        etypes, cnt, F = ps
        R, E = cnt.shape
        pairs = [(i, j) for i in range(R) for j in range(i + 1, R)]
        pair_vals = [int(F[i, j]) for (i, j) in pairs]
        pairs_nz  = sum(v > 0 for v in pair_vals)
        faces_sum = sum(pair_vals)

        if not os.path.exists(csv_path):
            cols = (
                ['tcurr']
                + [c
                   for r in range(R)
                   for c in ([f'{r}-{et}' for et in etypes] + [f'{r}-total'])]
                + [f'i{i}-{j}' for (i, j) in pairs]
                + ['pairs', 'faces']
            )
            with open(csv_path, 'w') as f:
                f.write(','.join(cols) + '\n')
            print(f"[info_to_csv] header_written path='{csv_path}' R={R} "
                  f"etypes={list(etypes)} npairs={len(pairs)} "
                  f"ncols={len(cols)}")

        row = [f"{tcurr:.6f}"]
        for r in range(R):
            vals = cnt[r].astype(int).tolist()
            row += list(map(str, vals)) + [str(sum(vals))]
        row += list(map(str, pair_vals)) + [str(pairs_nz), str(faces_sum)]

        with open(csv_path, 'a') as f:
            f.write(','.join(row) + '\n')
        print(f"[info_to_csv] row_append tcurr={tcurr:.6f} "
              f"counts={R*(E+1)} pairs={len(pairs)} faces_sum={faces_sum}")

    # --- NEW: neighbor GID sets cache (per topology epoch) ---
    @fff
    def _neighbor_gid_sets_cached(self) -> dict[int, dict[str, np.ndarray]]:
        gen = int(self._ver.get('topology', 0))
        c   = self._cache.get('nei_gid_sets')
        if c is not None and c.get('gen') == gen:
            return c['map']

        per_nbr = self._mpi_faces_by_neighbor()
        neighbors = sorted(map(int, per_nbr.keys()))
        if not neighbors:
            m = {}
            self._cache['nei_gid_sets'] = {'gen': gen, 'map': m}
            return m

        ncomm = comm['world'].Create_dist_graph_adjacent(neighbors, neighbors)
        send = {et: np.asarray(self.eidxs_i.get(et, ()), dtype=np.int64)
                for et in self.etypes}
        recvd = ncomm.neighbor_allgather(send)

        m = {}
        for nbr, pay in zip(neighbors, recvd):
            m[int(nbr)] = {
                et: np.asarray(pay.get(et, ()), dtype=np.int64)
                for et in self.etypes
            }

        self._cache['nei_gid_sets'] = {'gen': gen, 'map': m}
        return m

    @fff
    def calc_mpi_faces_deltas(self) -> dict[int, dict[str, np.ndarray]]:
        cons = {et: np.asarray(self.con_i.get(et, ()), dtype=np.int64)
                for et in self.etypes}

        # Safe path: use global gid->owner maps
        owners_map = self._gid_owner_maps(self.eidxs_i)
        neighbor_set = sorted({
            int(owners_map[self.etypes[int(code)]].get(int(ngid), rank['world']))
            for et, con in cons.items() if con.size
            for code, ngid in zip(con[..., 1].ravel(), con[..., 2].ravel())
            if code >= 0
        } - {rank['world']})

        empty = np.empty((0, 2), np.int64)
        idx2et = self.etypes

        def per_et(et, con, nrank):
            if con.size == 0:
                return empty
            codes, ngids = con[..., 1], con[..., 2]
            valid = (codes >= 0)

            def owner_of(idx, gid):
                if idx < 0:
                    return -99
                return owners_map[idx2et[int(idx)]].get(int(gid), rank['world'])

            owners = np.vectorize(owner_of)(
                np.where(valid, codes, -1),
                np.where(valid, ngids, -1)
            )
            c_me = np.sum(valid & (owners == rank['world']), axis=1).astype(np.int16)
            c_n  = np.sum(valid & (owners == nrank), axis=1).astype(np.int16)
            sel  = (c_n > 0)
            if not np.any(sel):
                return empty
            lids  = np.nonzero(sel)[0].astype(np.int64)
            delta = (c_me - c_n)[sel].astype(np.int64)
            mat   = np.c_[lids, delta]
            return mat[np.lexsort((mat[:, 0], mat[:, 1]))]

        return {n: {et: per_et(et, cons[et], n) for et in self.etypes}
                for n in neighbor_set}
class _MeshInterconnector(AlltoallMixin):

    def __init__(self, eidxs_src, eidxs_dest):

        # Link to src and dest eidxs
        self.eidxs_src = eidxs_src
        self.eidxs_dest = eidxs_dest

        # Union of all etypes across both layouts (local then global)
        etypes = comm['world'].allgather(sorted(set(eidxs_src) | set(eidxs_dest)))
        self.etypes = sorted(set().union(*etypes))

        # WORLD-gather of per-rank global element IDs (empty for non-participants)
        self.src_all  = {et: comm['world'].allgather(self._as64(eidxs_src.get(et, ())))
                         for et in self.etypes}
        self.dest_all = {et: comm['world'].allgather(self._as64(eidxs_dest.get(et, ())))
                         for et in self.etypes}

        # Storage
        self.send_idxs, self.recv_idxs = {}, {}
        self.scount, self.sdisp = {}, {}
        self.rcount, self.rdisp = {}, {}
        self.src_pos, self.recv_to_dest = {}, {}
        self.n_dest, self.ssum, self.rsum = {}, {}, {}

        self._build_plan()

    def _as64(self, a):
        return np.asarray(a if a is not None else (), dtype=np.int64)

    def _build_plan(self):

        for et in self.etypes:
            src_local  = np.asarray(self.src_all[et][rank['world']],  dtype=np.int64)
            dest_local = np.asarray(self.dest_all[et][rank['world']], dtype=np.int64)

            send_lists = []
            for p in range(comm['world'].size):
                dest_p = np.asarray(self.dest_all[et][p], dtype=np.int64)
                sel = src_local[np.isin(src_local, dest_p, assume_unique=False)] \
                      if (src_local.size and dest_p.size) \
                      else np.empty(0, np.int64)
                send_lists.append(sel)

            sidxs  = np.concatenate(send_lists) if any(a.size for a in send_lists) else np.empty(0, np.int64)
            scount = np.array([a.size for a in send_lists], dtype=np.int32)
            sdisp  = self._count_to_disp(scount)

            # Learn receive sizes, then exchange promised global IDs
            _, (rcount, rdisp) = self._alltoallcv(comm['world'], sidxs, scount, sdisp)

            ridxs = np.empty(int(rcount.sum()), dtype=np.int64)
            self._alltoallv(comm['world'], (sidxs, (scount, sdisp)), (ridxs, (rcount, rdisp)))

            # Save counts, disps, and recv list
            self.send_idxs[et], self.recv_idxs[et] = sidxs, ridxs
            self.scount[et], self.sdisp[et] = scount, sdisp
            self.rcount[et], self.rdisp[et] = rcount, rdisp

            # Map global->local on this rank (source side)
            self.src_pos[et] = {int(g): i for i, g in enumerate(src_local)}
            rpos = {int(g): i for i, g in enumerate(self.recv_idxs[et])}
            pr = np.fromiter((rpos.get(int(g), -1) for g in dest_local),
                                                count=dest_local.size, dtype=np.int64)
            if (pr < 0).any() and dest_local.size:
                # This happens only if dest asks for a GID nobody sent -> mapping bug
                bad = [int(g) for g, p in zip(dest_local.tolist(), pr.tolist()) if p < 0][:8]
                raise ValueError(f"[plan] r={rank['world']} et={et} dest GIDs missing from recv: {bad}")

            # Book-keeping
            self.recv_to_dest[et] = pr
            self.n_dest[et] = int(dest_local.size)
            self.ssum[et]   = int(scount.sum())
            self.rsum[et]   = int(rcount.sum())

    def relocate(self, edict_src, edim):
        src0 = self.preproc_edict(edict_src, edim=edim)
        dst0 = self._relocate_edict(src0)
        return self.postproc_edict(dst0, edim=edim)

    def preproc_edict(self, edict_in, *, edim):

        edict0, specs = {}, {}

        for et in self.etypes:
            Ne = len(self.src_all[et][rank['world']])   # rows I own on source layout (may be 0)
            a = edict_in.get(et, None)
            if a is not None:
                a = np.asarray(a)
                if a.ndim <= edim:
                    raise ValueError(f"[pre] r={rank['world']} et={et} invalid edim={edim} shape={a.shape}")
                a0 = np.moveaxis(a, edim, 0) if edim else a
                if a0.shape[0] != Ne:
                    raise ValueError(f"[pre] r={rank['world']} et={et} rows={a0.shape[0]} vs Ne_src={Ne}")
                edict0[et] = np.ascontiguousarray(a0)
                specs[et] = (str(a0.dtype), a0.shape[1:])
            else:
                specs[et] = (None, None)

        # Agree on dtype/shape globally; synthesize empties where needed
        gspecs = comm['world'].allgather(specs)
        for et in self.etypes:
            gdt, gtr = None, None
            for d in gspecs:
                dt, tr = d[et]
                if dt is not None:
                    if gdt is None: gdt, gtr = dt, tr
                    elif dt != gdt or tr != gtr:
                        raise ValueError(f"[pre] dtype/shape mismatch for {et}: {(gdt,gtr)} vs {(dt,tr)}")

            if gdt is None:
                # No data anywhere for this etype; ensure no traffic planned
                if self.ssum.get(et,0) or self.rsum.get(et,0) or len(self.src_all[et][rank['world']]):
                    raise ValueError(f"[pre] no array anywhere for etype '{et}' but traffic/rows exist")
                continue

            if et not in edict0:
                Ne = len(self.src_all[et][rank['world']])
                edict0[et] = np.empty((Ne,) + tuple(gtr), dtype=np.dtype(gdt))

        return edict0

    def _relocate_edict(self, edict0):

        out0 = {}
        for et in self.etypes:
            a0 = edict0.get(et, None)
            gids = self.send_idxs[et]
            pos = np.fromiter((self.src_pos[et].get(int(g), -1) for g in gids),
                              count=self.ssum[et], dtype=np.int64)
            svals = a0[pos]
            sc, sd = self.scount[et], self.sdisp[et]
            rc, rd = self.rcount[et], self.rdisp[et]

            rtot = int(rc.sum())
            rvals = np.empty((rtot, *a0.shape[1:]), dtype=a0.dtype)
            self._alltoallv(comm['world'], (svals, (sc, sd)), (rvals, (rc, rd)))
    
            pr = self.recv_to_dest[et]
            if pr.size != self.n_dest[et]:
                raise ValueError(f"[xfer] r={rank['world']} et={et} recv_to_dest size mismatch "
                                 f"({pr.size} vs {self.n_dest[et]})")

            out0[et] = rvals[pr]

        return out0

    def postproc_edict(self, edict0, *, edim):
        out = {}
        for et in self.etypes:
            a0 = edict0.get(et, None)
            if a0.shape[0] == 0: continue
            out[et] = np.moveaxis(a0, 0, edim) if edim else a0
        return out


class NativeReader:
    def __init__(self, fname, pname=None, *, construct_con=True,
                 comm_name='world'):
        self.f = h5py.File(fname, 'r')
        self.mesh = _Mesh(fname=fname, raw=self.f)

        self.comm_name = comm_name

        self.mesh.etypes = sorted(self.f['eles'])
        self.mesh.ndims = self.f['nodes'].dtype['location'].shape[0]

        # Read in and transform the various parts of the mesh
        if comm[comm_name] != mpi.COMM_NULL:
            self._read_metadata()
            self._read_partitioning(pname)
            self._read_eles()
            self._read_nodes()

            if construct_con:
                self._construct_con()

    def close(self):
        self.f.close()

    def load_soln(self, sname, prefix=None):
        mesh, soln = self.load_subset_mesh_soln(sname, prefix)

        # Ensure the solution is not subset
        if mesh is not self.mesh:
            raise ValueError('Subset solutions are not supported')

        return soln

    def load_subset_mesh_soln(self, sname, prefix=None):

        with h5py.File(sname, 'r') as f:
            if rank['world'] == root['world']:
                # Ensure the solution is from the mesh we are using
                uuid = f['mesh-uuid'][()].decode()
                if uuid != self.mesh.uuid:
                    raise RuntimeError('Invalid solution for mesh')

                # Read any config and stats records
                soln = {fname: f[fname][()].decode()
                        for fname in f if fname.startswith('config')}
                soln['stats'] = f['stats'][()].decode()
            else:
                soln = None

            # Broadcast and parse
            soln = comm['world'].bcast(soln, root=root['world'])
            soln = {k: Inifile(v) for k, v in soln.items()}

            # Obtain the polynomial order
            order = soln['config'].getint('solver', 'order')

            # If no prefix has been specified then obtain it from the file
            if prefix is None:
                prefix = soln['stats'].get('data', 'prefix')

            # Note if any elements are subset
            subset = {}

            # Read and scatter the solution data
            for etype in self.escatter:
                # If the element is not present, mark it as completely subset
                if (ek := f'{prefix}/p{order}-{etype}') not in f:
                    subset[etype] = []
                    continue
                # If the element is partially subset use a sparse scatterer
                elif (ei := f'{ek}-idxs') in f:
                    try:
                        idxs = self.mesh.eidxs[etype]
                    except KeyError:
                        idxs = np.empty(0, dtype=int)

                    escatter = SparseScatterer(comm['world'], f[ei], idxs)
                    subset[etype] = escatter.ridx
                # Complete element present so reuse the elements scatterer
                else:
                    escatter = self.escatter[etype]

                # Read the solution
                esoln = escatter(f[ek])
                if escatter.cnt:
                    soln[etype] = esoln.swapaxes(0, 2)

                # Read the partition data
                epart = escatter(f[f'{ek}-parts'])
                if escatter.cnt:
                    soln[f'{etype}-parts'] = epart

        # If the solution is subset then subset the mesh, too
        if subset:
            return self._subset_mesh(subset), soln
        else:
            return self.mesh, soln

    def _subset_mesh(self, subset):
        eidxs, spts, spts_nodes, spts_curved = {}, {}, {}, {}

        for etype in self.mesh.spts:
            if etype in subset:
                sidx = subset[etype]
                if len(sidx):
                    eidxs[etype] = self.mesh.eidxs[etype][sidx]
                    spts[etype] = self.mesh.spts[etype][:, sidx]
                    spts_nodes[etype] = self.mesh.spts_nodes[etype][sidx]
                    spts_curved[etype] = self.mesh.spts_curved[etype][sidx]
            else:
                eidxs[etype] = self.mesh.eidxs[etype]
                spts[etype] = self.mesh.spts[etype]
                spts_nodes[etype] = self.mesh.spts_nodes[etype]
                spts_curved[etype] = self.mesh.spts_curved[etype]

        return replace(self.mesh, subset=True, eidxs=eidxs, spts=spts,
                       spts_nodes=spts_nodes, spts_curved=spts_curved,
                       con=None, con_p=None, bcon=None)

    def _read_metadata(self):
        mesh = self.mesh

        if rank[self.comm_name] == root[self.comm_name]:
            creator = self.f['creator'][()].decode()
            codec = [c.decode() for c in self.f['codec']]
            uuid = self.f['mesh-uuid'][()].decode()
            version = self.f['version'][()]

            meta = (creator, codec, uuid, version)
        else:
            meta = None

        meta = comm[self.comm_name].bcast(meta, root=root[self.comm_name])
        mesh.creator, mesh.codec, mesh.uuid, mesh.version = meta

    def _read_with_idxs(self, dset, idxs):

        # Construct a Scatterer to read in and distribute the data
        s = Scatterer(comm[self.comm_name], idxs)

        return s(dset), s

    def _select_partitioning(self, size, pname=None):
        # If a partitioning has been specified then use it
        if pname:
            pinfo = self.f[f'partitionings/{pname}']
            nparts = len(pinfo['eles'].attrs['regions'])
            if nparts != size:
                raise RuntimeError(f'Partitioning {pname} has {nparts} parts '
                                   f'but running with {size} ranks')
        # Otherwise, try to find one
        else:
            for pname, pinfo in self.f['partitionings'].items():
                nparts = len(pinfo['eles'].attrs['regions'])
                if nparts == size:
                    break
            else:
                raise RuntimeError('Mesh does not have any partitionings with '
                                   f'{size} ranks')

        return pname, pinfo

    def _read_partitioning(self, pname=None):

        size = comm[self.comm_name].size

        # Have the root rank read in the partitioning metadata
        if rank[self.comm_name] == root[self.comm_name]:
            pname, pinfo = self._select_partitioning(size, pname)

            # Read the element region data
            einfo = pinfo['eles'].attrs['regions']

            # Read the neighbours data
            if size > 1:
                ninfo = pinfo['neighbours']
                ninfo = np.split(ninfo[()], ninfo.attrs['regions'][1:-1])
            else:
                ninfo = [[]]
        else:
            pname = einfo = ninfo = None

        # Broadcast this metadata
        ppath = 'partitionings/' + comm[self.comm_name].bcast(pname, root=root[self.comm_name])
        einfo = comm[self.comm_name].scatter(einfo, root=root[self.comm_name])
        self.neighbours = comm[self.comm_name].scatter(ninfo, root=root[self.comm_name])

        # Determine the element types in the mesh
        etypes = self.mesh.etypes

        # Read our portion of the partitioning table
        peles = self.f[f'{ppath}/eles'][einfo[0]:einfo[-1]]
        peles = np.split(peles, [i - einfo[0] for i in einfo[1:-1]])

        # With this determine the indices associated with each element
        self.mesh.eidxs = {et: pe for et, pe in zip(etypes, peles) if pe.size}

    def _read_eles(self):
        self.eles, self.escatter = eles, escatter = {}, {}

        # Collectively read in and distribute each element array
        for etype in self.mesh.etypes:
            dset = self.f[f'eles/{etype}']
            idxs = self.mesh.eidxs.get(etype, [])
            einfo, escatter[etype] = self._read_with_idxs(dset, idxs)

            # If we have any elements of this type then save the einfo
            if len(idxs):
                eles[etype] = einfo

    def _read_nodes(self):
        enodes = [einfo['nodes'] for einfo in self.eles.values()]

        # Determine the overall set of nodes across all element types
        idxs = np.concatenate([en.ravel() for en in enodes])

        # Note how many dimensions we have
        self.mesh.ndims = self.f['nodes'].dtype['location'].shape[0]

        # Read in these nodes
        nodes = self._read_with_idxs(self.f['nodes'], idxs.ravel())[0]

        # Determine where each element type is in the nodes array
        eoffs = np.cumsum([en.size for en in enodes])

        # Use this to split the nodes array back up
        nodes = np.split(nodes['location'], eoffs[:-1])

        # Reshape and add to the mesh
        for (etype, einfo), n in zip(self.eles.items(), nodes):
            spts = n.reshape(*einfo['nodes'].shape, -1).swapaxes(0, 1)

            self.mesh.spts[etype] = spts
            self.mesh.spts_nodes[etype] = einfo['nodes']
            self.mesh.spts_curved[etype] = einfo['curved']
            self.mesh.faces_cidxs[etype] = einfo['faces']['cidx']
            self.mesh.faces_offs[etype]  = einfo['faces']['off']

    def _construct_con(self):
        codec = self.mesh.codec
        eidxs = {k: v.tolist() for k, v in self.mesh.eidxs.items()}
        etypes = self.mesh.etypes

        # Create a map from global to local element numbers
        glmap = [{}]*len(etypes)
        for i, etype in enumerate(etypes):
            if etype in eidxs:
                glmap[i] = {k: j for j, k in enumerate(eidxs[etype])}

        # Create cidx indexed maps
        cdone, cefidx = [None]*len(codec), [None]*len(codec)
        for cidx, c in enumerate(codec):
            if (m := re.match(r'eles/(\w+)/(\d+)$', c)):
                etype, fidx = m[1], m[2]
                cdone[cidx] = set()
                cefidx[cidx] = (etype, etypes.index(etype), int(fidx))

        conl, conr = [], []
        bcon = {i: [] for i, c in enumerate(codec) if c.startswith('bc/')}
        resid = {}

        for etype, einfo in self.eles.items():
            i = etypes.index(etype)
            for fidx, eface in enumerate(einfo['faces'].T):
                efcidx = codec.index(f'eles/{etype}/{fidx}')

                for j, (cidx, off) in enumerate(iter_struct(eface)):
                    # Boundary
                    if off == -1:
                        bcon[cidx].append((etype, j, fidx))
                    # Unpaired face
                    elif j not in cdone[efcidx]:
                        # Lookup the element type and face number
                        ketype, ketidx, kfidx = cefidx[cidx]

                        # If our rank has the element then pair it
                        if (k := glmap[ketidx].get(off)) is not None:
                            conl.append((etype, j, fidx))
                            conr.append((ketype, k, kfidx))
                            cdone[cidx].add(k)
                        # Otherwise add it to the residual dict
                        else:
                            resid[efcidx, eidxs[etype][j]] = (cidx, off)

        # Add the internal connectivity to the mesh
        self.mesh.con = (conl, conr)

        for k, v in bcon.items():
            if v:
                self.mesh.bcon[codec[k][3:]] = v

        # Handle inter-partition connectivity
        if resid:
            self._construct_mpi_con(glmap, cefidx, resid)

    def _construct_mpi_con(self, glmap, cefidx, resid):

        # Create a neighbourhood collective communicator
        ncomm = autofree(comm[self.comm_name].Create_dist_graph_adjacent(self.neighbours,
                                                         self.neighbours))

        # Create a list of our unpaired faces
        unpaired = list(resid.values())

        # Distribute this to each of our neighbours
        nunpaired = ncomm.neighbor_allgather(unpaired)

        # See which of our neighbours unpaired faces we have
        matches = [[resid[j] for j in nunp if j in resid]
                   for nunp in nunpaired]

        # Distribute this information back to our neighbours
        nmatches = ncomm.neighbor_alltoall(matches)

        for nrank, nmatch in zip(self.neighbours, nmatches):
            if rank[self.comm_name] < nrank:
                ncon = sorted([(resid[m], m) for m in nmatch])
                ncon = [r for l, r in ncon]
            else:
                ncon = sorted([(m, resid[m]) for m in nmatch])
                ncon = [l for l, r in ncon]

            nncon = []
            for cidx, off in ncon:
                etype, etidx, fidx = cefidx[cidx]
                nncon.append((etype, glmap[etidx][off], fidx))

            # Add the connectivity to the mesh
            self.mesh.con_p[nrank] = nncon
