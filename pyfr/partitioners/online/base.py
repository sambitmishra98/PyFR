from copy import deepcopy
from dataclasses import dataclass, field, replace
import os
import re
from typing import Dict, List

import h5py
import numpy as np
from tabulate import tabulate

from pyfr.inifile import Inifile
from pyfr.nputil import iter_struct
from pyfr.partitioners.base import BasePartitioner
from pyfr.partitioners.metis import METISPartitioner
from pyfr.partitioners.scotch import SCOTCHPartitioner
from pyfr.mpiutil import comm, rank, root, mpi, AlltoallMixin, rankmap, get_comm_info, execute, initialise_new_comm
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where

@dataclass
class PartitionState:
    """Data model for a *single-rank* view of a partitioned PyFR mesh.

    This object is the canonical carrier of partition state used by online
    repartitioning.  It is intentionally *algorithm-agnostic*: it stores arrays and
    provides deterministic indexing utilities, but it must not decide *how* a
    partition changes (diffusion, SCOTCH, METIS, carving, etc.) and must not perform
    MPI communication.

    Responsibilities
    ---------------
    - Store per-etype element ownership:
        * `eidxs[etype]`: global element ids owned by this rank.
    - Store per-etype element-to-element connectivity:
        * `con_idx[etype]`: global element ids of neighbours per face (or -1).
        * `con_mpi[etype]`: neighbour owner rank for each face (or -1 for non-element).
    - Store geometric/topological auxiliaries required by algorithms:
        * `spts_nodes[etype]`: global node ids for each element (used for vertex modes).
        * Optional: `centroids[etype]` if centroid-based carving is enabled.
    - Precompute a stable *flat indexing* over all local elements:
        * `eidxs_flat`: concatenation of `eidxs` in deterministic etype order.
        * `etype_slices`: mapping etype -> slice into `eidxs_flat`.

    Key invariants (must be preserved by all writers)
    -------------------------------------------------
    - Global element ids are in the PyFR global ordering and are unique *within* each
      etype (and globally unique once you apply the per-etype offsets/edisps).
    - `con_idx` uses the *same* global id space as `eidxs_flat` (and as any global
      partition vector `vparts` you apply).
    - `con_mpi[..., f]` is the owner rank of `con_idx[..., f]` (or -1 when
      `con_idx` is -1 / a boundary / invalid face).
    - All integer arrays are contiguous, have explicit dtypes (int32/int64), and
      have stable shapes across ranks (empty arrays are present, not missing keys).

    Public API (expected)
    ---------------------
    - Conversion helpers:
        * `lids_to_flat(etype, lids) -> flat_indices`
        * `flat_to_lid(flat_indices) -> (etype, lids)` or equivalent iterator
    - Lightweight diagnostics (no heavy MPI):
        * `info()` / `info_to_csv(...)` should only summarise local counts and
          top-level invariants.
    - Construction-time normalisation:
        * `_preproc_attr_dict(...)` should enforce dtype/shape policy consistently.

    Collaboration / Ownership
    -------------------------
    - `_MetaMesh` owns a *current* `PartitionState` (usually `self.i`) and a scratch
      `PartitionState` (`self.j`) to stage updates.
    - Algorithms (Diffusion/Carver/Reallocator) should treat the state as read-only
      except through `_MetaMesh` commit primitives.
    """

    eidxs:   Dict[str, np.ndarray]
    con_idx: Dict[str, np.ndarray]
    con_mpi: Dict[str, np.ndarray]

    spts_nodes: Dict[str, np.ndarray]
    centroids:  Dict[str, np.ndarray] = field(default_factory=dict)

    eidxs_flat:   np.ndarray       = field(init=False)
    etype_slices: Dict[str, slice] = field(init=False)
    etypes:       List[str]        = field(init=False)
    edisps: Dict[str, int]         = field(init=False)   # PyFR naming
    nelems_g: int                  = field(init=False)   # global total (PyFR: disp end)

    ecnts_g: Dict[str, int] = field(init=False)

    def __post_init__(self):
        local_keys = set(self.eidxs or {})
        self.etypes = sorted(set().union(*comm['world'].allgather(local_keys)))

        self.eidxs      = self._preproc_attr_dict(self.eidxs,      lead_dim=None, dtype=np.int64, gather_shape=False, force_1d=True )
        self.con_idx    = self._preproc_attr_dict(self.con_idx,    lead_dim=0   , dtype=np.int64, gather_shape= True, force_1d=False)
        self.con_mpi    = self._preproc_attr_dict(self.con_mpi,    lead_dim=0   , dtype=np.int64, gather_shape= True, force_1d=False)
        self.spts_nodes = self._preproc_attr_dict(self.spts_nodes, lead_dim=0   , dtype=None    , gather_shape= True , force_1d=False)
        self.centroids  = self._preproc_attr_dict(self.centroids,  lead_dim=0   , dtype=np.float64, gather_shape= True , force_1d=False)

        counts_loc = np.array([self.eidxs[et].size for et in self.etypes], dtype=np.int64)
        counts_g   = comm['world'].allreduce(counts_loc, op=mpi.SUM)

        self.ecnts_g = {et: int(n) for et, n in zip(self.etypes, counts_g.tolist())}

        disp = 0
        self.edisps = {}
        for et in self.etypes:
            self.edisps[et] = disp
            disp += self.ecnts_g[et]

        self.nelems_g = int(disp)
        self.eidxs_flat, self.etype_slices = PartitionState._eidxs_to_flat(self.etypes, self.eidxs, self.edisps)

    def reprocess_spts_nodes(self, spts_nodes):
        self.spts_nodes = self._preproc_attr_dict(spts_nodes, lead_dim=0, dtype=None, gather_shape=True, force_1d=False)


    def _preproc_attr_dict(
        self,
        raw: Dict[str, np.ndarray | None],
        *,
        lead_dim: int | None,
        dtype: np.dtype | str | None,
        gather_shape: bool,
        force_1d: bool,
    ) -> Dict[str, np.ndarray]:
        """
        Normalise a per-etype dict for one attribute in a generic way.

        Parameters
        ----------
        raw : dict[str, array-like or None]
            Possibly incomplete / inconsistent input dict.
        lead_dim : int or None
            Axis index corresponding to "elements" for this attribute.
            If None, we treat the entire shape as trailing (no element axis).
        dtype : np.dtype, str or None
            Target dtype. If None, we take dtype from existing data.
        gather_shape : bool
            If True, we MPI-allgather (dtype, trailing_shape) per etype and
            enforce a canonical trailing shape across ranks.
        force_1d : bool
            If True, reshape arrays to 1D (used for eidxs).

        Returns
        -------
        out : dict[str, np.ndarray]
            Fully-populated per-etype dict with normalised arrays.
        """
        world = comm['world']
        out: Dict[str, np.ndarray] = {}

        # --------------------------------------------------------------
        # 1) Optional MPI-wide dtype/trailing-shape metadata
        # --------------------------------------------------------------
        if gather_shape:
            # Local specs: (dtype_str or None, trailing_shape or None)
            specs_local: Dict[str, tuple[str | None, tuple | None]] = {}
            for et in self.etypes:
                a = raw.get(et, None)
                if a is None:
                    specs_local[et] = (None, None)
                else:
                    arr = np.asarray(a)
                    if lead_dim is None:
                        trailing = arr.shape
                    else:
                        if arr.ndim <= lead_dim:
                            trailing = ()
                        else:
                            trailing = arr.shape[lead_dim + 1:]
                    specs_local[et] = (str(arr.dtype), trailing)

            gspecs_all = world.allgather(specs_local)

            # Build lookup: et -> (canonical_dtype_str or None, trailing_shape)
            gspecs: Dict[str, tuple[str | None, tuple | None]] = {}
            for et in self.etypes:
                gdt_str, gtr = None, None
                for d in gspecs_all:
                    dt, tr = d[et]
                    if dt is not None:
                        # Assumption: all non-None entries agree on (dtype, shape).
                        gdt_str, gtr = dt, tuple(tr)
                        break
                gspecs[et] = (gdt_str, gtr)
        else:
            # No global shape unification requested
            gspecs = {et: (None, None) for et in self.etypes}

        # --------------------------------------------------------------
        # 2) Build final arrays per etype
        # --------------------------------------------------------------
        for et in self.etypes:
            a = raw.get(et, None)
            gdt_str, gtr = gspecs[et]

            # Decide dtype:
            #   * explicit "dtype" argument wins,
            #   * else MPI-agreed dtype (if any),
            #   * else dtype from local data (if any),
            #   * else fall back to float64 for truly data-less cases.
            if dtype is not None:
                dt = np.dtype(dtype)
            elif gdt_str is not None:
                dt = np.dtype(gdt_str)
            elif a is not None:
                dt = np.asarray(a).dtype
            else:
                # Degenerate case: no data anywhere for this etype.
                # Arrays will be empty so dtype is largely irrelevant.
                dt = np.dtype("float64")

            if a is None:
                # No local payload for this etype
                if lead_dim is None:
                    # No element axis semantics: just an empty 1D array
                    arr = np.empty(0, dtype=dt)
                else:
                    # Element-wise attribute: use local Ne from eidxs
                    Ne_et = int(self.eidxs.get(et, np.empty(0, dtype=np.int64)).size)
                    if gather_shape and gtr is not None:
                        trailing = gtr
                    else:
                        trailing = ()
                    shape = (Ne_et, *trailing)
                    arr = np.empty(shape, dtype=dt)
            else:
                # We have local data: convert dtype and optionally force 1D
                arr = np.asarray(a, dtype=dt)
                if force_1d:
                    arr = arr.reshape(-1)

            out[et] = arr

        return out

    @property
    def nelems(self) -> int:
        return self.nelems_total

    @property
    def eids(self) -> np.ndarray:
        # PyFR partitioner language: “global element numbers”
        return self.eidxs_flat

    @staticmethod
    def _clone_no_mpi(src: "PartitionState") -> "PartitionState":
        new = PartitionState.__new__(PartitionState)  # bypass __init__/__post_init__

        # Deep-copy the data dicts (logic preserved: independent arrays)
        new.eidxs      = {et: arr.copy() for et, arr in src.eidxs.items()}
        new.con_mpi    = {et: arr.copy() for et, arr in src.con_mpi.items()}
        new.con_idx    = {et: arr.copy() for et, arr in src.con_idx.items()}
        new.spts_nodes = {et: arr.copy() for et, arr in src.spts_nodes.items()}
        new.centroids  = {et: arr.copy() for et, arr in src.centroids.items()}

        new.eidxs_flat   = src.eidxs_flat.copy()
        new.etype_slices = dict(src.etype_slices)
        new.etypes       = list(src.etypes)


        new.edisps   = dict(src.edisps)
        new.nelems_g = int(src.nelems_g)
        new.ecnts_g  = dict(src.ecnts_g)

        return new

    def clone(self) -> "PartitionState":
        return PartitionState._clone_no_mpi(self)

    def relocate_to(self, eidxs_dest: Dict[str, np.ndarray], 
                    move_spts_nodes = True):
        inter = _MeshInterconnector(self.eidxs, eidxs_dest)
        return PartitionState(eidxs=eidxs_dest,
                        con_mpi=inter.relocate(self.con_mpi, edim=0),
                        con_idx=inter.relocate(self.con_idx, edim=0),
                     spts_nodes=inter.relocate(self.spts_nodes, edim=0) if move_spts_nodes else self.spts_nodes,
                     centroids=inter.relocate(self.centroids, edim=0))

    @staticmethod
    def _mpi_sorted_union(local):
        return sorted(set().union(*comm['world'].allgather(set(local))))

    @property
    def nelems_total(self) -> int:
        return int(self.eidxs_flat.size)

    def lids_to_flat(self, etype: str, lids: np.ndarray) -> np.ndarray:
        return self.etype_slices[etype].start + np.asarray(lids, dtype=np.int64)

    @staticmethod
    def _compute_edisps(etypes, eidxs) -> Dict[str, int]:
        W = comm['world']

        # Global counts per etype (what PyFR's edisps encodes)
        loc_cnt = np.array([np.asarray(eidxs.get(et, ())).size for et in etypes], dtype=np.int64)
        glb_cnt = W.allreduce(loc_cnt, op=mpi.SUM)

        # Sanity: ensure per-etype gids are dense 0..Ne(etype)-1 globally.
        # (If this fails, we need a compression map before using SCOTCH.)
        loc_max = np.array(
            [np.max(np.asarray(eidxs.get(et, (-1,)), dtype=np.int64)) if loc_cnt[k] else -1
            for k, et in enumerate(etypes)],
            dtype=np.int64
        )
        glb_max = np.max(np.stack(W.allgather(loc_max), axis=0), axis=0)

        disp = np.concatenate(([0], np.cumsum(glb_cnt[:-1])))
        return {et: int(disp[i]) for i, et in enumerate(etypes)}

    @staticmethod
    def _eidxs_to_flat(etypes, eidxs, edisps):
        pieces: list[np.ndarray] = []
        etype_slices: Dict[str, slice] = {}
        start = 0

        for et in etypes:
            gids = np.asarray(eidxs.get(et, ()), dtype=np.int64)
            ne = gids.size

            etype_slices[et] = slice(start, start + ne)
            start += ne

            if ne:
                pieces.append(edisps[et] + gids)

        eidxs_flat = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
        return eidxs_flat, etype_slices

    def _local_mpi_faces_out(self, *, allowed_wranks: set[int] | None = None,
                                      my_wr: int | None = None) -> tuple[int, list[int]]:
        my_wr = int(rank['world']) if my_wr is None else int(my_wr)

        nout = 0
        nbrs: set[int] = set()

        for et in self.etypes:
            cm = self.con_mpi.get(et)
            if cm is None or cm.size == 0:
                continue

            v = cm.ravel()
            m = (v >= 0) & (v != my_wr)

            if not np.any(m):
                continue

            vm = v[m]

            if allowed_wranks is not None:
                # Faster than np.isin in tight loops when wranks is a small set
                vm = vm[np.fromiter((int(x) in allowed_wranks for x in vm), dtype=bool, count=vm.size)]
                if vm.size == 0:
                    continue

            nout += int(vm.size)
            nbrs.update(int(x) for x in np.unique(vm))

        return nout, sorted(nbrs)

    def partition_stats(self):
        """
        Root-only return:
            (wranks, etypes, cnt, iface_out, nfaces_total, npairs)

        wranks:        list[int] world ranks included in the report (row order)
        etypes:        tuple[str]
        cnt:           (R, E) element counts per (rank, etype)
        iface_out:     (R,) directed MPI-face incidence per rank
        nfaces_total:  int undirected total MPI faces within wranks
        npairs:        int # undirected rank-pairs with at least one MPI face
        """
        W       = comm['world']
        root_wr = root['world']
        my_wr   = rank['world']

        etypes = tuple(self.etypes)

        wranks = list(range(W.size))
        wranks = [int(w) for w in wranks]
        wset   = set(wranks)

        # --- per-etype element counts ---
        cnt_loc = np.asarray([self.eidxs[et].size for et in etypes], dtype=np.int64)
        cnt_all = W.gather(cnt_loc, root=root_wr)

        # --- interface counts from con_mpi (restricted to wranks) ---
        out_loc, nbrs_loc = self._local_mpi_faces_out(allowed_wranks=wset, my_wr=my_wr)
        out_all  = W.gather(int(out_loc), root=root_wr)
        nbrs_all = W.gather(nbrs_loc, root=root_wr)

        out_sum = int(W.allreduce(int(out_loc), op=mpi.SUM))
        nfaces_total = out_sum // 2  # if odd -> inconsistent two-sided encoding

        if my_wr != root_wr:
            return None

        cnt_w = np.stack(cnt_all, axis=0) if cnt_all else np.zeros((W.size, len(etypes)), np.int64)
        cnt   = cnt_w[wranks, :]

        iface_out_w = np.asarray(out_all, dtype=np.int64) if out_all else np.zeros(W.size, np.int64)
        iface_out   = iface_out_w[wranks]

        pairs: set[tuple[int, int]] = set()
        for wr_i, nbrs in enumerate(nbrs_all):  # gather order is world-rank order
            if wr_i not in wset:
                continue
            for wr_j in nbrs:
                if wr_j not in wset or wr_j == wr_i:
                    continue
                a, b = (wr_i, wr_j) if wr_i < wr_j else (wr_j, wr_i)
                pairs.add((a, b))

        return wranks, etypes, cnt, iface_out, int(nfaces_total), int(len(pairs))

    def info(self, *, wranks: list[int] | None = None, which: str | None = None):
        """
        Print the compact table for THIS PartitionState.
        `which` is ignored here (kept only if you want a symmetric call site).
        """

        ps = self.partition_stats()

        if ps is None:
            return

        wranks, etypes, cnt, iface_out, nfaces_total, npairs = ps
        R, _ = cnt.shape

        headers = (["etype"] + [f"w{wr}" for wr in wranks] + ["sum"])
        rows = []

        for j, et in enumerate(etypes):
            v = cnt[:, j].astype(int)
            rows.append([et] + [f"{x:,}" for x in v] + [f"{int(v.sum()):,}"])

        tot = cnt.sum(axis=1).astype(int)
        rows.append(["total_elems"] + [f"{x:,}" for x in tot] + [f"{int(tot.sum()):,}"])

        rows.append(["mpi_faces_out"] + [f"{int(x):,}" for x in iface_out] + [f"{int(iface_out.sum()):,}"])
        rows.append(["mpi_faces_total"] + [""] * R + [f"{nfaces_total:,}"])
        rows.append(["mpi_pairs"]       + [""] * R + [f"{npairs:,}"])

        print(tabulate(rows, headers=headers, tablefmt="github",
                       colalign=("left", *("right",) * (len(headers) - 1))))

    def info_to_csv(
        self,
        *,
        tcurr: float,
        comm_name: str = "compute",
        csv_path: str = "lb_elem_dist.csv",
    ) -> None:
        """
        Append one row with:
        - per-rank per-etype counts (restricted to rankmap[comm_name])
        - per-rank total_elems
        - per-rank mpi_faces_out (directed)
        - mpi_faces_total (undirected)
        - mpi_pairs (undirected nonzero interfaces)
        """
        ps = self.partition_stats()
        if ps is None:
            return

        wranks, etypes, cnt, iface_out, nfaces_total, npairs = ps

        # Only world-root writes
        if int(rank["world"]) != int(root["world"]):
            return

        # ---- header (stable ordering) ----
        cols = ["tcurr"]

        # per-rank per-etype counts
        cols += [f"w{wr}-{et}" for wr in wranks for et in etypes]

        # per-rank totals
        cols += [f"w{wr}-total_elems" for wr in wranks]

        # per-rank directed interface incidence
        cols += [f"w{wr}-mpi_faces_out" for wr in wranks]

        # globals
        cols += ["mpi_faces_total", "mpi_pairs"]

        # ---- row ----
        row = [f"{tcurr:.6f}"]

        # counts: cnt is (len(wranks), len(etypes)) in wranks-order
        for i in range(len(wranks)):
            row += [str(int(x)) for x in cnt[i, :]]

        tot = cnt.sum(axis=1).astype(np.int64)
        row += [str(int(x)) for x in tot]

        row += [str(int(x)) for x in np.asarray(iface_out, dtype=np.int64)]
        row += [str(int(nfaces_total)), str(int(npairs))]

        # ---- write ----
        need_header = not os.path.exists(csv_path)
        if need_header:
            with open(csv_path, "w") as f:
                f.write(",".join(cols) + "\n")

        with open(csv_path, "a") as f:
            f.write(",".join(row) + "\n")


class _MetaMesh:

    """Distributed, mutable partition object providing *collective-safe* mutation
    primitives for online repartitioning.

    `_MetaMesh` is the “operational” layer: it owns `PartitionState` instances and
    offers methods to *apply* a chosen set of element moves, update connectivity
    ownership tags, and export/import to PyFR mesh objects.  Partition *decisions*
    belong in algorithms/mixins; `_MetaMesh` is the execution substrate.

    Design goals
    ------------
    - Separation of concerns:
        * algorithms compute “what to move” (plans),
        * `_MetaMesh` executes plans *correctly* and keeps state consistent.
    - Collective safety: any method that can trigger MPI communication must be safe
      when called by *all* world ranks, even if some ranks have empty partitions.
    - Determinism: given the same state and plan, results should be bitwise
      reproducible (ordering, tie-breaks, id remaps).

    Core responsibilities
    ---------------------
    - Own and maintain:
        * `self.i`: current `PartitionState`
        * `self.j`: next/scratch `PartitionState` (staging buffer)
        * topology caches (gid->owner maps, neighbour sets, etc.) with explicit
          invalidation rules.
    - Provide execution primitives:
        * `_reset_j_with_i()` / `restart()` to reinitialise staging state.
        * `_apply_plan_and_commit(eidxs_diff, ...)` as the *only* mutating entry point
          that changes element ownership across ranks.
        * `_retag_con_owners()` / `_retag_con_owners_from_*()` to restore
          `con_mpi[...,0]=owner` invariants after a move or after applying global
          partitions.
    - Provide import/export:
        * `from_mesh(...)` / `to_mesh(...)` for PyFR mesh objects.
        * `to_mesh` must be usable to rebuild compute meshes during controller
          reinitialisation.
    - Provide cheap state introspection:
        * element counts (`nelems_total`, `_cur_counts_total`), device tagging hooks,
          and debug summaries.

    Expected public surface (contract)
    ----------------------------------
    - *Plan execution*: `_apply_plan_and_commit(...)` is collective and must be called
      by all ranks every time an algorithm iteration occurs (even if the plan is
      empty on some ranks).
    - *Topology validity flags*: e.g. `_spts_nodes_valid`, `_owners_valid`, and a
      single `_invalidate_topology()` entry point.
    - *Convenience*: helpers like `_mpi_faces_by_neighbor()`, `collect_mpi_vertex_nodes()`
      may exist but should be carefully scoped: they are algorithm inputs, not
      mutation operations.

    Extensibility
    -------------
    - Subclasses/algorithms should not override commit semantics.  If specialised
      paths are needed (e.g. SCOTCH apply path), add *new* well-scoped helpers that
      still end in `_apply_plan_and_commit` or an equivalent single commit gate.
    """

    def __init__(self, mesh, cfg):
        self.mesh_src  = mesh
        self.mesh_dest = None

        self.lb_flowmat_relax = cfg.getfloat('partition', 'lb-flowmatrix-relax', 0.5)

        self.etypes = etypes = PartitionState._mpi_sorted_union(set(mesh.etypes or ()))
        bc_names = PartitionState._mpi_sorted_union(set((mesh.bcon or {}).keys()))
        self.e2i      = {et: i for i, et in enumerate(etypes)}
        self.bc2id    = {n: i for i, n in enumerate(bc_names)}

        eidxs   = {et: np.asarray(mesh.eidxs.get(et, ()),        dtype=np.int64) for et in etypes}
        con_mpi = {et: np.full((eidxs[et].size, self._nfaces(et)), -1, np.int64) for et in etypes}
        con_idx = {et: np.full((eidxs[et].size, self._nfaces(et)), -1, np.int64) for et in etypes}

        spts_nodes = deepcopy(mesh.spts_nodes)
        self._spts_nodes_valid = True

        centroids = self._init_centroids_from_mesh(mesh)

        self.i = PartitionState(eidxs=deepcopy(eidxs), con_mpi=deepcopy(con_mpi), 
                       con_idx=deepcopy(con_idx), spts_nodes=deepcopy(spts_nodes),
                       centroids=deepcopy(centroids))
        self.j = self.i.clone()

        self._encode_con(mesh)
        self._fill_con_mpi(mesh)
        self._ne_i = self._local_count

        self._cores = self._compute_cores_from_centroids()

        self._cache = {}
        self._ver = {'topology': 0}

    def _invalidate_topology(self) -> None:
        """Bump topology epoch and drop caches that depend on eidx ownership."""
        self._ver['topology'] += 1

        # TODO: more fine-grained invalidation later; for now just wipe
        self._cache.pop('owners_map', None)
        self._cache.pop('nei_gid_sets', None)

        self._cache.pop('owner_global', None)
        self._cache.pop('con_p', None)

    @property
    def mesh(self):
        return self.mesh_dest # if self.mesh_dest is not None else self.mesh_src

    @classmethod
    def from_mesh(cls, mesh) -> "_MetaMesh":
        # Create a cofig from a custom text file
        cfg = Inifile("""
                      """)

        return cls(mesh=mesh, cfg=cfg)

    def _encode_con(self, mesh):
        """
        Encode the *local* (same-rank) mesh connectivity and boundary faces as:

            con_mpi[et][lid, f, 0] = neighbour owner rank (world index), or
                                    -1 for boundary faces
            con_idx[et][lid, f, 0] = neighbour global element ID (>= 0), or
                                    eid_bc = -(bc_id + 1) for boundaries

        Global element IDs are the GIDs carried in PartitionState.eidxs_flat via
        PartitionState._eidxs_to_flat.
        """

        my_rank = int(rank['world'])

        conL, conR = (mesh.con or ([], []))
        bcon = getattr(mesh, 'bcon', {}) or {}

        # Interior pairs: both sides are on *this* rank → owner_rank = my_rank
        for (etL, lidL, fL), (etR, lidR, fR) in zip(conL, conR):
            self.i.con_mpi[etL][int(lidL), int(fL)] = my_rank
            self.i.con_idx[etL][int(lidL), int(fL)] = self.etype_lid_to_eid(etR, lidR)

            self.i.con_mpi[etR][int(lidR), int(fR)] = my_rank
            self.i.con_idx[etR][int(lidR), int(fR)] = self.etype_lid_to_eid(etL, lidL)

        # Boundary: owner=-1, eid=eid_bc<0
        for bcname, triples in (bcon.items() if bcon else []):
            bid = self.bc2id[bcname]
            if bid is None:
                continue

            for et, lid, f in triples:
                et = str(et)
                self.i.con_mpi[et][int(lid), int(f)] = -1
                self.i.con_idx[et][int(lid), int(f)] = -(int(bid) + 1)

    def _fill_con_mpi(self, mesh):
        cp = getattr(mesh, "con_p", {}) or {}

        # Local export: {nbr: [(et, lid, fid, eid_local), ...]}
        local = {
            int(nbr): [
                (
                    str(et),
                    int(lid),
                    int(fid),
                    int(self.etype_lid_to_eid(str(et), int(lid))),  # eid of THIS element
                )
                for (et, lid, fid) in faces
            ]
            for nbr, faces in cp.items()
        }

        all_cp = comm['world'].allgather(local)

        my_rank = int(rank['world'])

        for nbr, A in local.items():
            nbr = int(nbr)
            # Faces where *this* rank is the neighbour of `nbr`
            B = all_cp[nbr].get(my_rank, [])

            for (etA, lidA, fA, _eidA), (etB, _lidB, fB, eidB) in zip(A, B):
                # Store neighbour owner + neighbour global ID
                self.i.con_mpi[etA][lidA, fA] = nbr
                self.i.con_idx[etA][lidA, fA] = int(eidB)

        nfaces_mpi = sum(len(v) for v in local.values())

    def to_mesh(self, eidxs_dest):
        ic = _MeshInterconnector(self.mesh_src.eidxs, eidxs_dest)

        def reconstruct_con_conp_bcon(mesh):
            
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
                except Exception as e:
                    print(
                        f"[_reconstruct_con_conp_bcon] ERROR on rank={rank['world']} "
                        f"etype={etype}: {e.__class__.__name__}: {e}",
                        flush=True
                    )
                    raise

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

        def apply_lex_ordering(mesh):
            eidxs_dest = {}

            for et in self.etypes:
                gids = np.asarray(mesh.eidxs.get(et, ()), dtype=np.int64)
                if not gids.size: eidxs_dest[et] = gids; continue
                internal = mesh.spts_internal[et].astype(np.int8, copy=False)
                curved   = mesh.spts_curved  [et].astype(np.int8, copy=False)
                order = np.lexsort((gids, curved, internal))  # primary = internal
                eidxs_dest[et] = gids[order]
            
            return eidxs_dest

        mesh_int = replace(self.mesh_src, eidxs=eidxs_dest,
            spts       =ic.relocate(self.mesh_src.spts,        edim=1), 
            spts_nodes =ic.relocate(self.mesh_src.spts_nodes,  edim=0),
            spts_curved=ic.relocate(self.mesh_src.spts_curved, edim=0),
            faces_cidxs=ic.relocate(self.mesh_src.faces_cidxs, edim=0),
            faces_offs =ic.relocate(self.mesh_src.faces_offs,  edim=0),
            )

        reconstruct_con_conp_bcon(mesh_int)

        eidxs_dest = apply_lex_ordering(mesh_int)
        ic2 = _MeshInterconnector(mesh_int.eidxs, eidxs_dest)
        
        mesh_dest = replace(mesh_int, eidxs=eidxs_dest,
            spts       =ic2.relocate(mesh_int.spts,        edim=1), 
            spts_nodes =ic2.relocate(mesh_int.spts_nodes,  edim=0),
            spts_curved=ic2.relocate(mesh_int.spts_curved, edim=0),
            faces_cidxs=ic2.relocate(mesh_int.faces_cidxs, edim=0),
            faces_offs =ic2.relocate(mesh_int.faces_offs,  edim=0),
            )

        reconstruct_con_conp_bcon(mesh_dest)

        # Remove keys with empty entries
        mesh_dest.eidxs = {k: v for k, v in mesh_dest.eidxs.items() if v.size}

        # Set destination mesh
        self.mesh_dest = mesh_dest

        return mesh_dest

    # --------------------------------------------------------------------------

    def restart(self):
        """Reset MetaMesh, carefully resetting the rest if needed. """
        if self.mesh_dest is not None:
            self.mesh_src = self.mesh_dest
            self.mesh_dest = None

        # Re-initialise spts_nodes
        self.i.reprocess_spts_nodes(self.mesh_src.spts_nodes)
        self._cache = {}
        self._ver = {'topology': 0}
        self._spts_nodes_valid = True
        self._cores = None

    @property
    def _local_count(self):
        return sum(len(eidxs) for eidxs in self.i.eidxs.values())

    @property
    def _cur_counts_total(self):
        return list(comm['world'].allgather(self._local_count))

    @property
    def ntotal(self):
        """Total elements across all ranks."""
        return int(comm['world'].allreduce(self._local_count, op=mpi.SUM))

    @staticmethod
    def _nfaces(etype):
        """Number of faces in an element of a given etype."""
        return len(subclass_where(BaseShape, name=etype).faces)

    def int_round(self, N_star):
        """
            _MetaMesh
            Rescale targets to total elements in mesh.
        """

        Ntot = self.i.nelems_g
        N_star = np.asarray(N_star, dtype=float)
        sum_star = float(N_star.sum())

        scale = float(Ntot) / sum_star
        N_scaled = N_star * scale
        N_floor = np.floor(N_scaled).astype(np.int64)

        k = int(Ntot - N_floor.sum())

        if k != 0:
            frac = N_scaled - N_floor
            order = np.argsort(frac)  # ascending

            if k > 0:
                idxs = order[-k:]   # bump largest fractions
                N_floor[idxs] += 1
            else:
                idxs = order[:-k]   # k < 0 → drop smallest fractions
                N_floor[idxs] -= 1

        return N_floor

    # --------------------------------------------------------------------------
    # Element centroids and partition centroids (cores)
    # --------------------------------------------------------------------------
    
    def _init_centroids_from_mesh(self, mesh) -> None:
        """
        Per-etype element centroids in physical space.

        centroids_i[et]: (Ne, 3) float64, same element ordering as eidxs_i[et].
        For 2D meshes, z is padded with 0.
        """
        centroids: Dict[str, np.ndarray] = {}

        for et in self.etypes:
            gids = np.asarray(mesh.eidxs.get(et, ()), dtype=np.int64)
            Ne   = gids.size

            if Ne == 0:
                centroids[et] = np.zeros((0, 3), dtype=np.float64)
                continue

            spts = mesh.spts.get(et)
            if spts is None:
                centroids[et] = np.zeros((Ne, 3), dtype=np.float64)
                continue

            spts = np.asarray(spts, dtype=np.float64)

            if spts.ndim == 3:
                # Layout: (npts, Ne, nd)  with element index on axis 1
                npts, Ne_spts, nd = spts.shape
                # average over points -> (Ne, nd)
                c = spts.mean(axis=0)          # (Ne_spts, nd)
            elif spts.ndim == 2:
                # Degenerate: (Ne, nd) – already one point per element
                Ne_spts, nd = spts.shape
                c = spts
            else:
                raise RuntimeError(f"Unexpected spts[{et!r}].ndim={spts.ndim}")

            # Ne_spts should match Ne; if not, clip to the common prefix
            if Ne_spts != Ne:
                m = min(Ne_spts, Ne)
                c = c[:m, :]
                if m < Ne:
                    pad = np.zeros((Ne - m, c.shape[1]), dtype=np.float64)
                    c = np.vstack([c, pad])

            # Pad or trim to 3 coordinates
            nd_eff = c.shape[1]
            if nd_eff < 3:
                c3 = np.zeros((Ne, 3), dtype=np.float64)
                c3[:, :nd_eff] = c
                c = c3
            elif nd_eff > 3:
                c = c[:, :3]

            centroids[et] = c

        return centroids

    def _compute_cores_from_centroids(self) -> None:
        """
        Compute per-rank CPD cores from current element centroids.

        For rank r, core_r is the arithmetic mean of all element centroids
        owned by r (all etypes, equal weight).
        Result stored in self._cores with shape (R, nd).
        """
        R = comm['world'].size

        # Local sum and count
        nd = None
        sum_local = None
        count_local = 0

        # centroids_i[et]: (Ne, nd)
        for et in self.etypes:
            c = getattr(self.i, 'centroids', {}).get(et) if hasattr(self.i, 'centroids') else None
            if c is None:
                continue
            c = np.asarray(c, dtype=np.float64)
            if c.size == 0:
                continue

            if nd is None:
                nd = c.shape[1]
                sum_local = np.zeros(nd, dtype=np.float64)

            sum_local += c.sum(axis=0)
            count_local += c.shape[0]

        if nd is None:
            # No elements on this rank: keep a harmless default
            nd = 3
            sum_local = np.zeros(nd, dtype=np.float64)

        sums   = comm['world'].allgather(sum_local)
        counts = comm['world'].allgather(int(count_local))

        cores = np.zeros((R, nd), dtype=np.float64)
        for r in range(R):
            if counts[r] > 0:
                cores[r, :] = sums[r] / float(counts[r])
            else:
                cores[r, :] = np.nan

        return cores

    # --------------------------------------------------------------------------

    # Switch from etype and local indexing within that etype to the global flat indexing
    def etype_lid_to_eid(self, etype: str, lids) -> np.ndarray:
        lids = np.asarray(lids, dtype=np.int64)
        # eid = edisps[etype] + gid
        return self.i.edisps[etype] + np.asarray(self.i.eidxs[etype], dtype=np.int64)[lids]

    # --------------------------------------------------------------------------    

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

    def _compute_restrict_last(self, skip_last: bool, etypes_all: list[str]) -> dict[int, bool]:
        """
        Device-aware 'protected etype' policy.

        Returns
        -------
        restrict_last : dict[int, bool]
            Mapping world-rank -> True if we should *protect* the last etype
            on interfaces to that rank (i.e. do not export the last etype).
        """
        if not (skip_last and etypes_all):
            return {}

        local_order = tuple(etypes_all)
        orders_all  = comm['world'].allgather(local_order)  # list[tuple]
        restrict_last = {
            int(r): (tuple(o) != local_order)
            for r, o in enumerate(orders_all)
        }
        return restrict_last

    def _apply_plan_and_commit(self, eidxs_diff, move_spts_nodes=True) -> None:
        # self._reset_j_with_i()
        eidxs_dest = self.plan_eidxs_dest_from_diff(eidxs_diff)

        # fast path using mesh-global flat indexing from State
        fast_conn = _MetaMeshInterconnector(etypes = self.etypes,
                                            eidxs_src  = self.i.eidxs,
                                            eidxs_dest = eidxs_dest,
                                            eidxs_flat_src   = self.i.eidxs_flat,
                                            etype_slices_src = self.i.etype_slices,
        )

        self.j = PartitionState(eidxs=eidxs_dest,
                           con_mpi=fast_conn.relocate_cons(self.i.con_mpi),
                           con_idx=fast_conn.relocate_cons(self.i.con_idx),
                       spts_nodes=(fast_conn.relocate_cons(self.i.spts_nodes)
                          if move_spts_nodes else self.i.spts_nodes),
                       centroids=fast_conn.relocate_cons(self.i.centroids)
                       )

        self._accept_j_into_i()
        self._ne_i = self._local_count
        if not move_spts_nodes:
            self._spts_nodes_valid = False

    # ---------------------
    # Relocation iterations 
    # ---------------------

    def _reset_j_with_i(self):
        self.j = self.i.clone()

    def _accept_j_into_i(self) -> None:
        self.i, self.j = self.j, self.i
        self._retag_con_owners()
        self._invalidate_topology()

    def _build_eidxs_diff_from_flat(self, chosen_flat: np.ndarray,
                                          chosen_nbrs: np.ndarray,
                                    state: "PartitionState") -> dict[int, dict[str, np.ndarray]]:
        """
        Convert a flat list of chosen element indices + dest ranks into the
        {nbr: {etype: gids[]}} structure expected by _apply_plan_and_commit.

        Parameters
        ----------
        chosen_flat : (N,) int64
            Flat element indices in the *current* source State (state.eidxs_flat).
        chosen_nbrs : (N,) int64
            Destination world-rank for each chosen element.
        state : State
            The source State (typically self.i) from which we are moving
            elements. We use its eidxs and etype_slices to reconstruct per-etype
            global element IDs.

        Notes
        -----
        - We *do not* use state.eidxs_flat values here; instead we rely on the
          fact that flat indices are laid out as a concatenation of per-etype
          eidxs[et], with slices recorded in state.etype_slices.
        - This keeps eidxs_diff in the original per-etype global GID space
          that _MetaMeshInterconnector expects.
        """
        chosen_flat = np.asarray(chosen_flat, dtype=np.int64)
        chosen_nbrs = np.asarray(chosen_nbrs, dtype=np.int64)

        eidxs_diff: dict[int, dict[str, np.ndarray]] = {}

        if chosen_flat.size == 0:
            return eidxs_diff

        if chosen_flat.shape != chosen_nbrs.shape:
            raise ValueError(
                f"_build_eidxs_diff_from_flat: shape mismatch "
                f"{chosen_flat.shape} vs {chosen_nbrs.shape}"
            )

        etype_slices = state.etype_slices  # {et: slice}
        eidxs        = state.eidxs         # {et: gids[...]}

        uniq_nbrs, inv_nbr = np.unique(chosen_nbrs, return_inverse=True)

        for i_n, nbr in enumerate(uniq_nbrs.tolist()):
            mask_n = (inv_nbr == i_n)
            flat_n = chosen_flat[mask_n]
            if flat_n.size == 0:
                continue

            per_et: dict[str, np.ndarray] = {}

            # For this neighbour, split the flat indices by etype
            for et, sl in etype_slices.items():
                if sl.stop <= sl.start:
                    continue  # no local elements of this etype

                # Identify those flat indices that fall into this etype slice
                m_et = (flat_n >= sl.start) & (flat_n < sl.stop)
                if not np.any(m_et):
                    continue

                lids = (flat_n[m_et] - sl.start).astype(np.int64)
                if lids.size == 0:
                    continue

                gids = np.asarray(eidxs[et][lids], dtype=np.int64)
                per_et[et] = gids

            if per_et:
                eidxs_diff[int(nbr)] = per_et

        return eidxs_diff

    # -----------------
    # Testing / ideas
    # -----------------

    def _get_global_owner_array(self) -> np.ndarray:
        """
        Debug helper: build a global owner array

            owners_global[g] = rank that owns global element ID g,

        using PartitionState.eidxs_flat (global IDs local to this rank) and an
        allgather across ranks.
        """
        cw = comm['world']
        my_rank = int(rank['world'])
        gen = int(self._ver.get('topology', 0))

        c = self._cache.get('owner_global')
        if c is not None and c['gen'] == gen:
            return c['owners']

        ids_local = np.asarray(self.i.eidxs_flat, dtype=np.int64)

        Ne_global = int(self.i.nelems_g)
        owners_global = np.full(Ne_global, -1, dtype=np.int32)

        # Safety: eid range + uniqueness
        ids_local = np.asarray(ids_local, dtype=np.int64)
        if ids_local.size:
            if ids_local.min() < 0 or ids_local.max() >= Ne_global:
                raise ValueError(
                    f"eid out of range on rank={my_rank}: "
                    f"min={ids_local.min()} max={ids_local.max()} Ne_global={Ne_global}"
                )

        # Fill owners from gathered eid lists
        all_ids = cw.allgather(ids_local)
        for rr, eids_rr in enumerate(all_ids):
            eids_rr = np.asarray(eids_rr, dtype=np.int64)
            if eids_rr.size:
                owners_global[eids_rr] = rr

        # Final safety: every eid must be owned by exactly one rank
        if np.any(owners_global < 0):
            miss = int(np.sum(owners_global < 0))
            raise ValueError(f"owners_global has {miss} unassigned eids (non-dense or missing elements).")

        # owners_global = np.empty(Ne_global, dtype=np.int32)
        # owners_global.fill(-1)
        # # Enumerate across allgathered global IDs
        # for rr, gids_rr in enumerate(cw.allgather(ids_local)):
        #     gids_rr = np.asarray(gids_rr, dtype=np.int64)
        #     if gids_rr.size == 0:
        #         continue
        #     owners_global[gids_rr] = rr
        # self._cache['owner_global'] = {'gen': gen, 'owners': owners_global,}

        self._cache['owner_global'] = {'gen': gen, 'owners': owners_global}
        return owners_global

    def _retag_con_owners(self):
        """
        Retag connectivity owner fields using global IDs in con_idx.
        """
        owners_global = self._get_global_owner_array()
        
        for et, con_idx in self.i.con_idx.items():
            con_idx_arr = con_idx
            con_mpi_arr = self.i.con_mpi[et]

            if con_idx_arr.size == 0 or con_mpi_arr.size == 0:
                continue

            eid_flat    = con_idx_arr.reshape(-1)
            owners_flat = con_mpi_arr.reshape(-1)

            valid = (eid_flat >= 0)
            if np.any(valid):
                owners_flat[valid] = owners_global[eid_flat[valid]].astype(np.int32)

            new_owners = owners_flat.reshape(con_mpi_arr.shape)
            self.i.con_mpi[et][...] = new_owners

    def plan_eidxs_dest_from_diff(self, eidxs_diff, verbose=False):
        """
        Build new per-etype GID lists after a proposed relocation plan.
        """

        send_plan = {
            d: {
                et: np.asarray(eidxs_diff.get(d, {}).get(et, ()), dtype=np.int64)
                for et in self.etypes
            }
            for d in range(comm['world'].size)
            if d != rank['world']
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
            per_me = plan[rank['world']]
            if not per_me:
                continue
            for et in self.etypes:
                a = per_me[et]
                if a is not None and a.size:
                    to_recv[et] = np.concatenate((to_recv[et], a))

        for et in self.etypes:
            if to_recv[et].size:
                to_recv[et] = np.unique(to_recv[et])

        # --- Build new_eidxs from *flat* view only ---

        new_eidxs = {}
        for et in self.etypes:
            cur = self.i.eidxs[et]

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
            tag = 'ag'
            tsz = {et: int(to_send[et].size) for et in self.etypes}
            rsz = {et: int(to_recv[et].size) for et in self.etypes}
            nsz = {et: int(new_eidxs[et].size) for et in self.etypes}
            print(f"[plan/{tag}] R{rank['world']}\tsend:{tsz}\trecv:{rsz}\ttarget:{nsz}")

        return new_eidxs


class WaitsToTargetsModelMixin:

    """Cost-model mixin that maps measured runtime/communication signals to *target*
    element counts per rank.

    This mixin is the *performance-model* layer.  It should not mutate partitions.
    It provides:
      (i) ingestion of performance telemetry (waitsome matrices, per-rank compute),
     (ii) regularisation/smoothing and device-aware weighting,
    (iii) production of integer targets and (optionally) per-etype weights.

    Inputs (typical)
    ----------------
    - `g1a, g1s, g1r`: aggregated/median MPI Waitsome time matrices:
        * all-time, send-time, recv-time (or any consistent decomposition).
    - Per-rank compute proxies (e.g. ndofs, wall-clock kernel time, FLOP proxies).
    - Device tags and etype preference policies (CPU/GPU, hex/tet bias, etc.).
    - Online controls from config:
        * jitter fractions, smoothing windows, lb cadence, caps, min-elements, etc.

    Outputs (typical)
    -----------------
    - `targets[r]`: integer target element counts for each *world rank* (or a defined
      rank-list ordering), with exact global conservation enforced:
          sum(targets) == sum(current_counts)
    - Optional auxiliary products:
        * rank weights / speeds,
        * per-etype weighting vectors,
        * diagnostic scalars for logging (fit residuals, stability metrics).

    Contract / invariants
    ---------------------
    - Deterministic: same inputs -> same targets (seeded randomness only via an
      explicit RNG/seed parameter).
    - Conservative: targets must preserve total element count.
    - Safe for dynamic ranks: ranks with `current==0` may still get `target>0`
      (addition), and ranks with `target==0` may currently own elements (removal).

    Public API (expected)
    ---------------------
    - `calc_target_ecounts(current_counts, g1a, g1s, g1r, ...) -> targets`
    - `rank_tags()` / `device_tags` helpers to align devices with ranks.
    - Configuration exposure:
        * jitter schedule parameters,
        * fitting window/robust statistics parameters.

    Notes on architecture
    ---------------------
    If this mixin needs mesh-derived features (etype mix, ndofs), pass them in as
    arguments or via a small immutable “telemetry bundle”.  Avoid reaching into
    partition mutation internals to keep the model testable in isolation.
    """

    _LB_CYCLIC_JITTER_CTR = 0

    def __init__(self, cfg):

        # Cost scales
        self.lb_cost_scale_g1r  = cfg.getfloat('partition', 'lb-cost-scale-g1r',  1.0)
        self.lb_cost_scale_g1s  = cfg.getfloat('partition', 'lb-cost-scale-g1s',  1.0)
        self.lb_cost_scale_g1rt = cfg.getfloat('partition', 'lb-cost-scale-g1rt', 1.0)

        self.lb_iters = cfg.getint('partition', 'lb-outeriterations', 1)
        
        self.cost = None

    def write_g1_median_csvs(self, g1a, g1s, g1r, g1idx: int = 1):
        """
        Snapshot g1 medians to CSV in integer microseconds.
        Now always use world-size vectors/matrices and embed compute ranks.
        """
        if rank['compute'] != root['compute']:
            return

        # Scale to microseconds and cast to int (compute index space)
        all_us  = np.rint(g1a * 1e6).astype(np.int64)
        send_us = np.rint(g1s * 1e6).astype(np.int64)
        recv_us = np.rint(g1r * 1e6).astype(np.int64)

        print(f"g1a: {all_us}")
        print(f"g1s: \n {send_us}")
        print(f"g1r: \n {recv_us}")

        def _append_csv_row(file_path: str, header_cols: list[str], values: list[int]):
            # TODO: Connect with pyfr.writers.csv.py

            if not os.path.exists(file_path):
                with open(file_path, 'w', newline='') as f:
                    f.write(','.join(header_cols) + '\n')

            # Append row
            with open(file_path, 'a', newline='') as f:
                f.write(','.join(str(int(v)) for v in values) + '\n')

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
        mcols = [f"i{i}-{j}"                                                  for i in range(P_world) for j in range(P_world)]
        _append_csv_row('g1-send-median-ms.csv', mcols, [int(send_us_w[i, j]) for i in range(P_world) for j in range(P_world)])
        _append_csv_row('g1-recv-median-ms.csv', mcols, [int(recv_us_w[i, j]) for i in range(P_world) for j in range(P_world)])

    def _lb_next_cyclic_pulse_index(self, comm_nc, root_nc: int) -> int:
        """
        Deterministically choose a pulse rank index in the *newcompute* ordering.
        Advances a module-level counter on root_nc and broadcasts the chosen index.
        """

        if comm_nc == mpi.COMM_NULL:
            return -1

        if int(comm_nc.rank) == int(root_nc):
            idx = int(self._LB_CYCLIC_JITTER_CTR % int(comm_nc.size))
            self._LB_CYCLIC_JITTER_CTR += 1
        else:
            idx = None

        idx = comm_nc.bcast(idx, root=int(root_nc))
        return int(idx)

    def _lb_apply_single_rank_weight_bump(self, 
        N_star: np.ndarray,
        pulse_idx: int,
        frac: float,
        *,
        pattern: str = "single",     # "single" | "alternate"
        parity: int = 0,             # used only when pattern="alternate"
    ) -> np.ndarray:
        """
        - pattern="single": bump one rank = pulse_idx
        - pattern="alternate": bump half the ranks (even or odd), toggled by 'parity'
        NOTE: pulse_idx is ignored for selection in this mode.
        """
        frac = float(frac)
        if frac <= 0.0:
            return N_star

        N = np.asarray(N_star, dtype=np.float64, copy=True)
        if N.size <= 1:
            return N

        pat = str(pattern).lower()
        if pat == "single":
            j = np.array([int(pulse_idx) % int(N.size)], dtype=np.int64)
        elif pat == "alternate":
            p = int(parity) & 1
            j = np.arange(p, int(N.size), 2, dtype=np.int64)  # half ranks: p,p+2,...
        else:
            raise ValueError(f"Unknown jitter {pattern = }")

        N[j] *= (1.0 + frac)
        return N

    def compute_cost(self, g1a, g1s, g1r):

        self.write_g1_median_csvs(g1a, g1s, g1r)

        g1a_old = np.asarray(g1a, dtype=float)
        g1s_old = np.asarray(g1s, dtype=float)
        g1r_old = np.asarray(g1r, dtype=float)

        s_out = g1s_old.sum(axis=1)
        r_in  = g1r_old.sum(axis=1)
        r_out = g1r_old.sum(axis=0)

        self.cost = (g1a_old - r_in  * self.lb_cost_scale_g1r
                        - s_out * self.lb_cost_scale_g1s
                        + r_out * self.lb_cost_scale_g1rt)

        return self.cost


class _MetaMeshInterconnector(AlltoallMixin):
    """Pure index-space interconnector for *in-rank* relocation of mesh-attached arrays
    when applying a global partition vector.

    This helper is used for the “global partitioner apply” path (SCOTCH/METIS/KAHIP):
    given `eidxs_src` and `eidxs_dest` on *the same rank*, it builds local mapping
    tables that can relocate:
    - connectivity arrays (`con_idx`, `con_mpi`)
    - element-attached geometry (`spts_nodes`, centroids, etc.)

    Key property: **no MPI communication**.
    It only performs local permutation / filtering / remapping consistent with the
    new local ownership.

    Expected responsibilities
    -------------------------
    - Provide `relocate_cons(dict_of_arrays)` and similar helpers that:
        * keep shapes consistent,
        * preserve dtype,
        * remap local element ordering deterministically.
    - Expose enough metadata for fast paths:
        * `eidxs_flat_src`, `etype_slices_src` (and possibly dest analogues),
          so callers avoid recomputing expensive maps.

    Limitations / contract
    ----------------------
    - Does not decide ownership; only realises a decision already expressed as
      `eidxs_dest` (or a global `vparts` vector sliced into per-rank `eidxs_dest`).
    - Must not touch MPI collectives; correctness is local-only.
    """


    def __init__(self, etypes,
                 eidxs_src: Dict[str, np.ndarray],
                 eidxs_dest: Dict[str, np.ndarray],
                 eidxs_flat_src: np.ndarray | None = None,
                 etype_slices_src: Dict[str, slice] | None = None) -> None:
        self.eidxs_src  = eidxs_src
        self.eidxs_dest = eidxs_dest
        self.etypes     = list(etypes)

        W   = comm['world']
        rnk = int(rank['world'])

        # All-gather per-etype GID lists: same as before
        self.src_all  = {
            et: W.allgather(np.asarray(eidxs_src.get(et, ()), dtype=np.int64))
            for et in self.etypes
        }
        self.dest_all = {
            et: W.allgather(np.asarray(eidxs_dest.get(et, ()), dtype=np.int64))
            for et in self.etypes
        }

        # Storage for per-etype traffic pattern
        self.send_idxs: Dict[str, np.ndarray] = {}
        self.scount:    Dict[str, np.ndarray] = {}
        self.sdisp:     Dict[str, np.ndarray] = {}
        self.rcount:    Dict[str, np.ndarray] = {}
        self.rdisp:     Dict[str, np.ndarray] = {}
        self.recv_to_dest: Dict[str, np.ndarray] = {}
        self.n_dest:      Dict[str, int] = {}

        # ---- NEW: mesh-global flat indexing exactly as in State ----
        # Source side: either reuse from State or recompute.
        if eidxs_flat_src is not None and etype_slices_src is not None:
            self._src_flat        = np.asarray(eidxs_flat_src, dtype=np.int64)
            self._src_flat_slices = dict(etype_slices_src)
        else:
            self._src_flat, self._src_flat_slices = PartitionState._eidxs_to_flat(self.etypes,
                eidxs_src)

        # Destination side: flat view for possible later use / checks
        # self._dest_flat, self._dest_flat_slices = PartitionState._eidxs_to_flat(self.etypes,
        #     eidxs_dest)

        # Per-etype global offsets: eid = base[et] + gid
        self._eid_base: Dict[str, int] = {}
        for et in self.etypes:
            gids_local = np.asarray(self.eidxs_src.get(et, ()), dtype=np.int64)
            sl = self._src_flat_slices[et]
            if gids_local.size and sl.stop > sl.start:
                eid_local = self._src_flat[sl]
                # Base is constant: eid = base + gid
                self._eid_base[et] = int(eid_local[0] - gids_local[0])
            else:
                self._eid_base[et] = 0

        # Single sorted mapping over the flat global IDs
        if self._src_flat.size:
            order_flat = np.argsort(self._src_flat)
            self._src_flat_sorted = self._src_flat[order_flat]
            self._src_flat_idx    = order_flat.astype(np.int64)
        else:
            self._src_flat_sorted = np.empty(0, np.int64)
            self._src_flat_idx    = np.empty(0, np.int64)

        # NEW: cached local src gids and sorted mapping
        self._src_local: Dict[str, np.ndarray]   = {}
        self._src_g_sorted: Dict[str, np.ndarray] = {}
        self._src_i_sorted: Dict[str, np.ndarray] = {}

        # NEW: cached mapping from send-gids to src-row indices
        self._send_rows:      dict[str, np.ndarray] = {}

        # Build per-etype comm pattern (same logic as your _build_plan_simple)
        self._build_plan_simple()

    def _flatten_eidxs_dict(
        self,
        eidxs: Dict[str, np.ndarray],
    ) -> tuple[np.ndarray, Dict[str, slice]]:
        """
        Canonical flattening of per-etype eidxs into a 1D array of GIDs.

        The ordering matches the canonical etype order in self.etypes.

        Returns
        -------
        flat : np.ndarray
            (Ne_flat_local,) int64 global element IDs on THIS rank.
        slices : Dict[str, slice]
            Per-etype slices back into flat.
        """
        pieces: list[np.ndarray] = []
        slices: Dict[str, slice] = {}

        start = 0
        for et in self.etypes:
            arr = np.asarray(eidxs.get(et, ()), dtype=np.int64)
            n = int(arr.size)
            if n:
                pieces.append(arr)
                slices[et] = slice(start, start + n)
                start += n
            else:
                slices[et] = slice(start, start)

        if pieces:
            flat = np.concatenate(pieces)
        else:
            flat = np.empty(0, dtype=np.int64)

        return flat, slices

    def _build_plan_simple(self) -> None:
        W    = comm['world']
        rnk  = int(rank['world'])
        P    = W.size

        for et in self.etypes:
            src_local  = np.asarray(self.src_all[et][rnk],  dtype=np.int64)
            dest_local = np.asarray(self.dest_all[et][rnk], dtype=np.int64)

            self._src_local[et] = src_local

            # Precompute sorted view for fast gid -> row mapping
            if src_local.size:
                order_src = np.argsort(src_local)
                src_sorted = src_local[order_src]
                src_idx    = order_src.astype(np.int64)
            else:
                src_sorted = np.empty(0, np.int64)
                src_idx    = np.empty(0, np.int64)

            self._src_g_sorted[et] = src_sorted
            self._src_i_sorted[et] = src_idx

            # ---- per-dest send lists (which GIDs I must send to each rank p) ----
            send_lists: list[np.ndarray] = []
            for p in range(P):
                dest_p = np.asarray(self.dest_all[et][p], dtype=np.int64)
                if src_local.size and dest_p.size:
                    sel = src_local[np.isin(src_local, dest_p, assume_unique=True)]
                else:
                    sel = np.empty(0, np.int64)
                send_lists.append(sel)

            # Flatten into one send buffer of GIDs
            if any(a.size for a in send_lists):
                sidxs = np.concatenate(send_lists)
            else:
                sidxs = np.empty(0, np.int64)

            self.send_idxs[et] = sidxs

            scount = np.array([a.size for a in send_lists], dtype=np.int32)
            sdisp  = self._count_to_disp(scount)

            self.scount[et] = scount
            self.sdisp[et]  = sdisp

            # --- build recv side as before (unchanged) ---
            _, (rcount, rdisp) = self._alltoallcv(W, sidxs, scount, sdisp)
            ridxs = np.empty(int(rcount.sum()), dtype=np.int64)
            self._alltoallv(W, (sidxs, (scount, sdisp)),
                               (ridxs, (rcount, rdisp)))
            self.rcount[et] = rcount
            self.rdisp[et]  = rdisp

            # Map recv-buffer indices to local dest row indices
            n_dest = int(dest_local.size)
            pr = np.full(n_dest, -1, dtype=np.int64)
            if n_dest:
                recv_gids = ridxs
                if recv_gids.size:
                    order = np.argsort(recv_gids)
                    recv_sorted = recv_gids[order]

                    loc = np.searchsorted(recv_sorted, dest_local)
                    mask_in = loc < recv_sorted.size
                    loc_in  = loc[mask_in]
                    same    = recv_sorted[loc_in] == dest_local[mask_in]

                    mask = np.zeros_like(loc, dtype=bool)
                    mask[mask_in] = same

                    if not np.all(mask):
                        bad = dest_local[~mask][:8].tolist()
                        raise ValueError(
                            f"[MetaMeshInter] r={rnk} et={et} "
                            f"dest GIDs missing from recv: {bad}"
                        )

                    pr[mask] = order[loc_in[same]]

            self.recv_to_dest[et] = pr
            self.n_dest[et]       = n_dest

            # ---------------------------------------------------
            # NEW: map send GIDs -> local src-row indices once
            #      and cache for all relocate_cons calls.
            # ---------------------------------------------------
            if sidxs.size:
                loc = np.searchsorted(src_sorted, sidxs)
                mask = (loc < src_sorted.size) & (src_sorted[loc] == sidxs)

                if not np.all(mask):
                    bad = sidxs[~mask][:8].tolist()
                    raise ValueError(
                        f"[MetaMeshInter] r={rnk} et={et} "
                        f"send GIDs not found in src_local: {bad}"
                    )

                rows = src_idx[loc]
            else:
                rows = np.empty(0, np.int64)

            self._send_rows[et] = rows


    # ------------------------------------------------------------------
    # Data relocation: con_idx only (axis 0 = element index)
    # ------------------------------------------------------------------
    def relocate_cons(self, cons_src: Dict[str, np.ndarray]
                         ) -> Dict[str, np.ndarray]:
        W   = comm['world']
        rnk = int(rank['world'])

        out: Dict[str, np.ndarray] = {}

        for et in self.etypes:
            a0 = cons_src.get(et, None)
            src_local_gids = self._src_local.get(et, None)
            if src_local_gids is None:
                src_local_gids = np.asarray(self.src_all[et][rnk], dtype=np.int64)
                self._src_local[et] = src_local_gids

            Ne_src_local = int(src_local_gids.size)

            if a0 is None:
                # Should not happen with well-formed State, but keep a guard.
                if Ne_src_local:
                    raise ValueError(
                        f"[MetaMeshInter] r={rnk} et={et} "
                        f"con array missing but Ne_src_local={Ne_src_local}"
                    )
                out[et] = np.empty((0,), dtype=np.int32)
                continue

            # Assume np.ndarray stored by State; cheap assert only.
            if not isinstance(a0, np.ndarray):
                a0 = np.asarray(a0)

            if a0.shape[0] != Ne_src_local:
                raise ValueError(
                    f"[MetaMeshInter] r={rnk} et={et} "
                    f"rows={a0.shape[0]} vs Ne_src_local={Ne_src_local}"
                )

            trailing_shape = a0.shape[1:]

            if Ne_src_local and a0.size:
                # Normal case: at least one local element of this etype
                a0_flat = a0.reshape(Ne_src_local, -1)
            else:
                # Corner case: this rank has zero source rows of this etype.
                # We still need a well-typed, 2D empty array so that:
                #   - svals = a0_flat[rows] works (rows will be empty)
                #   - rvals uses a0_flat.shape[1] for its column count.
                if trailing_shape:
                    ncols = int(np.prod(trailing_shape, dtype=int))
                else:
                    # 1D input (shape (0,)); treat as one column
                    ncols = 1
                a0_flat = np.empty((0, ncols), dtype=a0.dtype)

            # Reuse precomputed src rows for this etype
            rows = self._send_rows[et]
            if rows.size:
                svals = a0_flat[rows]
            else:
                svals = np.empty((0, a0_flat.shape[1]), dtype=a0_flat.dtype)

            sc, sd = self.scount[et], self.sdisp[et]
            rc, rd = self.rcount[et], self.rdisp[et]

            rtot = int(rc.sum())
            rvals = np.empty((rtot, a0_flat.shape[1]), dtype=a0_flat.dtype)

            self._alltoallv(W, (svals, (sc, sd)), (rvals, (rc, rd)))

            pr    = self.recv_to_dest[et]
            ndest = self.n_dest[et]
            if pr.size != ndest:
                raise ValueError(
                    f"[MetaMeshInter] r={rnk} et={et} "
                    f"recv_to_dest size mismatch ({pr.size} vs {ndest})"
                )

            if ndest:
                out_flat = rvals[pr]
                out[et] = out_flat.reshape((ndest,) + trailing_shape)
            else:
                out[et] = np.empty((0,) + trailing_shape, dtype=a0.dtype)

        return out


class _MeshInterconnector(AlltoallMixin):

    """MPI-aware interconnector for relocating *solution arrays* and other per-element
    fields between two partition layouts.

    This is the runtime data-migration primitive used by the integrator/controller
    during a repartition event:
        old 'compute' layout  ->  staging 'newcompute' layout  ->  promoted 'compute'

    Responsibilities
    ---------------
    - Given source/destination element ownership (`eidxs_src`, `eidxs_dest`) and an
      MPI communicator, compute:
        * send lists (gids/indices per destination rank),
        * recv lists (expected gids/indices from source ranks),
        * stable ordering for packing/unpacking to preserve determinism.
    - Provide relocation methods:
        * `relocate_ary(intercon, ary, edim=..., ...)` (or equivalent)
          for ndarray payloads shaped as (nelems, ..., nvars).
    - Ensure collective correctness:
        * All ranks in the communicator participate, even if they send/recv zero.
        * Type/shape checks are strict; failures should be loud and early.
    - Support minimal introspection for debugging:
        * message volume, total bytes, max peer count, etc.

    Design notes
    ------------
    - Keep this class independent of partition algorithms: it consumes `PartitionState`
      snapshots and moves arrays; it must not mutate the mesh partition itself.
    - Prefer explicit ownership of MPI requests/buffers to avoid leaks across repeated
      load-balance events.
    """

    def __init__(self, eidxs_src, eidxs_dest):
        # Link to src and dest eidxs
        self.eidxs_src = eidxs_src
        self.eidxs_dest = eidxs_dest

        # Union of all etypes across both layouts (local then global)
        etypes = comm['world'].allgather(sorted(set(eidxs_src) | set(eidxs_dest)))
        self.etypes = sorted(set().union(*etypes))

        # REMOVE ALLGATHER BELOW !!! 
        # Data we currently own per etype
        self.src_all  = {et: comm['world'].allgather(self._as64(eidxs_src.get(et, ())))  for et in self.etypes}

        # Data we want to own per etype
        self.dest_all = {et: comm['world'].allgather(self._as64(eidxs_dest.get(et, ()))) for et in self.etypes}

        # Storage
        self.send_idxs, self.recv_idxs = {}, {}
        self.scount, self.sdisp = {}, {}
        self.rcount, self.rdisp = {}, {}
        self.src_pos, self.recv_to_dest = {}, {}
        self.n_dest, self.ssum, self.rsum = {}, {}, {}

        # NEW: vectorised lookup helpers (per etype)
        self._src_g_sorted,  self._src_i_sorted  = {}, {}
        self._recv_g_sorted, self._recv_i_sorted = {}, {}

        self._build_plan()

    def _as64(self, a):
        return np.asarray(a if a is not None else (), dtype=np.int64)

    def _flatten_eidxs_dict(
        self,
        eidxs: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, slice]]:
        """
        Canonical flattening of per-etype eidxs into a 1D array of GIDs.

        The ordering matches what MetaMesh uses:
            eidxs_flat = concat( eidxs[et] for et in self.etypes )

        Returns
        -------
        flat : np.ndarray
            (Ne_flat_local,) int64 global element IDs on THIS rank.
        slices : Dict[str, slice]
            Per-etype slices back into flat, for debugging / future use.
        """
        pieces: list[np.ndarray] = []
        slices: dict[str, slice] = {}

        start = 0
        for et in self.etypes:
            arr = self._as64(eidxs.get(et, ()))
            n = int(arr.size)
            if n:
                pieces.append(arr)
                slices[et] = slice(start, start + n)
                start += n
            else:
                # Keep an empty slice for invertibility / consistency
                slices[et] = slice(start, start)

        if pieces:
            flat = np.concatenate(pieces)
        else:
            flat = np.empty(0, dtype=np.int64)

        return flat, slices

    def relocate_eidxs_flat(
        self,
        eidxs_flat_src: np.ndarray | None = None,
        *,
        label: str = "eidxs-flat",
    ) -> tuple[np.ndarray, dict[str, slice], dict[str, slice]]:
        """
        Compute destination eidxs_flat from eidxs_dest, and optionally
        verify that the provided source eidxs_flat matches eidxs_src.

        Parameters
        ----------
        eidxs_flat_src : np.ndarray or None
            Rank-local flat GID array as built by MetaMesh on the SOURCE
            layout. If provided, we check it against our own flattening
            of eidxs_src and emit a diagnostic line.
        label : str
            Short label for log messages.

        Returns
        -------
        eidxs_flat_dest : np.ndarray
            Rank-local flat GID array for the DESTINATION layout.
        slices_src : Dict[str, slice]
            Per-etype slices describing how src-flat was built.
        slices_dest : Dict[str, slice]
            Per-etype slices describing how dest-flat was built.
        """
        # Canonical flattening of src/dest layouts
        src_flat, slices_src = self._flatten_eidxs_dict(self.eidxs_src)
        dst_flat, slices_dest = self._flatten_eidxs_dict(self.eidxs_dest)

        # --------------------------------------------------------------
        # 1) Local consistency check with provided MetaMesh eidxs_flat
        # --------------------------------------------------------------
        if eidxs_flat_src is not None:
            given = self._as64(eidxs_flat_src)

            same_shape = (given.shape == src_flat.shape)
            same_vals  = same_shape and np.array_equal(given, src_flat)

            if same_vals:
                print(
                    f"[MeshInter/{label}-check] rank={rank['world']} "
                    f"OK src_flat matches eidxs_src "
                    f"Ne_local={src_flat.size}"
                )
            else:
                print(
                    f"[MeshInter/{label}-check] rank={rank['world']} "
                    f"MISMATCH src_flat vs eidxs_src "
                    f"shape_given={given.shape} shape_expected={src_flat.shape}"
                )

        # --------------------------------------------------------------
        # 2) Global invariants to catch gross inconsistencies
        # --------------------------------------------------------------
        Ne_src_local  = int(src_flat.size)
        Ne_dest_local = int(dst_flat.size)

        # Global element counts
        Ne_src_total = comm['world'].allreduce(
            Ne_src_local, op=lambda a, b: a + b
        )
        Ne_dest_total = comm['world'].allreduce(
            Ne_dest_local, op=lambda a, b: a + b
        )

        # Simple checksum invariants (sums of GIDs)
        sum_src_local = int(src_flat.sum()) if Ne_src_local  else 0
        sum_dst_local = int(dst_flat.sum()) if Ne_dest_local else 0

        sum_src_total = comm['world'].allreduce(
            sum_src_local, op=lambda a, b: a + b
        )
        sum_dst_total = comm['world'].allreduce(
            sum_dst_local, op=lambda a, b: a + b
        )

        # Rank-0 global summary
        if rank['world'] == 0:
            print(
                f"[MeshInter/{label}-global] "
                f"Ne_src_total={Ne_src_total} Ne_dest_total={Ne_dest_total} "
                f"sum_src={sum_src_total} sum_dest={sum_dst_total}"
            )

        # Per-rank local summary
        print(
            f"[MeshInter/{label}-local] rank={rank['world']} "
            f"Ne_src_local={Ne_src_local} Ne_dest_local={Ne_dest_local}"
        )

        return dst_flat, slices_src, slices_dest

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
            self._alltoallv(comm['world'], (svals := sidxs, (scount, sdisp)), (ridxs, (rcount, rdisp)))

            # Save counts, disps, and recv list
            self.send_idxs[et], self.recv_idxs[et] = sidxs, ridxs
            self.scount[et], self.sdisp[et] = scount, sdisp
            self.rcount[et], self.rdisp[et] = rcount, rdisp

            # Map global->local on this rank (source side) – keep dict for compatibility
            self.src_pos[et] = {int(g): i for i, g in enumerate(src_local)}

            # ---- NEW: build sorted lookup tables for src and recv ----
            # For src: gid -> local row index on this rank
            if src_local.size:
                order_src = np.argsort(src_local)
                self._src_g_sorted[et] = src_local[order_src]
                self._src_i_sorted[et] = order_src.astype(np.int64)
            else:
                self._src_g_sorted[et] = np.empty(0, np.int64)
                self._src_i_sorted[et] = np.empty(0, np.int64)

            # For recv: gid -> index in flat recv buffer rvals
            recv_gids = self.recv_idxs[et]
            if recv_gids.size:
                order_recv = np.argsort(recv_gids)
                self._recv_g_sorted[et] = recv_gids[order_recv]
                self._recv_i_sorted[et] = order_recv.astype(np.int64)
            else:
                self._recv_g_sorted[et] = np.empty(0, np.int64)
                self._recv_i_sorted[et] = np.empty(0, np.int64)

            # Compute mapping dest_local -> recv index (pr) WITHOUT fromiter,
            # but preserving the "-1 for missing" convention.
            if dest_local.size:
                pr = np.full(dest_local.shape, -1, dtype=np.int64)

                recv_sorted = self._recv_g_sorted[et]
                recv_idx    = self._recv_i_sorted[et]

                if recv_sorted.size:
                    loc = np.searchsorted(recv_sorted, dest_local)
                    mask = loc < recv_sorted.size
                    if mask.any():
                        loc_m   = loc[mask]
                        dest_m  = dest_local[mask]
                        same    = recv_sorted[loc_m] == dest_m

                        pr_m = pr[mask]
                        pr_m[same] = recv_idx[loc_m[same]]
                        pr[mask] = pr_m
            else:
                pr = np.empty(0, np.int64)

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
                    raise ValueError(f"[pre] r={rank['world']} et={et} incoming-rows={a0.shape[0]} vs Ne_src={Ne}")
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
            if a0 is None:
                # Should not happen if preproc_edict did its job; keep behaviour.
                continue

            gids = self.send_idxs[et]
            ndest = self.n_dest.get(et, 0)

            # Build mapping gids -> local src row indices WITHOUT fromiter,
            # preserving "-1 for missing" semantics.
            if gids.size:
                src_sorted = self._src_g_sorted[et]
                src_idx    = self._src_i_sorted[et]

                pos = np.full(gids.shape, -1, dtype=np.int64)
                if src_sorted.size:
                    loc = np.searchsorted(src_sorted, gids)
                    mask = loc < src_sorted.size
                    if mask.any():
                        loc_m  = loc[mask]
                        gids_m = gids[mask]
                        same   = src_sorted[loc_m] == gids_m

                        pos_m = pos[mask]
                        pos_m[same] = src_idx[loc_m[same]]
                        pos[mask] = pos_m
            else:
                pos = np.empty(0, np.int64)

            # pos is now int64 with "-1" for any impossible GID,
            # exactly like dict.get(..., -1). We rely on the plan being correct
            # so in practice pos should be >= 0 always.
            svals = a0[pos]

            sc, sd = self.scount[et], self.sdisp[et]
            rc, rd = self.rcount[et], self.rdisp[et]

            rtot = int(rc.sum())
            rvals = np.empty((rtot, *a0.shape[1:]), dtype=a0.dtype)
            self._alltoallv(comm['world'], (svals, (sc, sd)), (rvals, (rc, rd)))

            pr = self.recv_to_dest[et]
            if pr.size != ndest:
                raise ValueError(f"[xfer] r={rank['world']} et={et} recv_to_dest size mismatch "
                                 f"({pr.size} vs {ndest})")

            out0[et] = rvals[pr]

        return out0

    def postproc_edict(self, edict0, *, edim):
        out = {}
        for et in self.etypes:
            a0 = edict0.get(et, None)
            if a0.shape[0] == 0: continue
            out[et] = np.moveaxis(a0, 0, edim) if edim else a0
        return out


class RankAllocatorMixin:
    """Algorithm-agnostic mixin implementing *rank addition/removal semantics* for any
    partitioning backend (diffusion, SCOTCH, METIS, ...).

    This mixin defines “how to change the *active rank set*” (grow/shrink the compute
    communicator) while delegating “how to reshape the partition” to the underlying
    partitioner/carver methods.

    Conceptual model
    ----------------
    - The world communicator is fixed; “compute ranks” are a subset (`compute-ranklist`).
    - Online changes are expressed via `online.ini` as a new `compute-ranklist`.
    - A rank is *active* if it currently owns elements or is assigned a non-zero
      target.  Rank add/remove is achieved by moving elements until:
        * added rank transitions 0 -> >0 owned elements (seeding),
        * removed rank transitions >0 -> 0 owned elements (draining).

    Responsibilities
    ---------------
    - Parse and expose online ranklist intent:
        * `self.online_cfg` provides the desired compute-ranklist.
        * helper(s) should compute `ranks_to_add` and `ranks_to_remove`.
    - Provide deterministic, collective-safe primitives:
        * `seed_rank(new_rank, targets, ...)` to bootstrap an empty rank with a small
          interface-connected patch so diffusion/global partitioning can take over.
        * `drain_rank(old_rank, targets, ...)` (or current `iterate("to-remove-rank")`)
          to evacuate elements from a rank slated for removal.
        * `swap_partitions(r0, r1)` to bubble an empty partition to a convenient
          position (e.g. end of ranklist).
    - Define selection policies as overridable hooks (pure functions preferred):
        * pick donor rank(s), pick interface neighbour, pick seed etype, pick budget.

    Required collaboration with `_MetaMesh`
    ---------------------------------------
    This mixin assumes the host class provides:
    - access to current state (`self.i`) and etype order,
    - MPI-interface queries (`_mpi_faces_by_neighbor`, `collect_mpi_vertex_nodes`),
    - collective commit (`_apply_plan_and_commit`),
    - topology invalidation/retagging helpers.

    Public API (what controllers should call)
    -----------------------------------------
    - `seed_rank(...)` and `swap_partitions(...)` are intended to be called from the
      time integrator/controller during an LB event.
    - A higher-level orchestration method is recommended:
        * `reconcile_active_ranks(current_counts, targets) -> action plan`
          returning a structured decision (add/remove/none) plus metadata for logging.

    Architecture note
    -----------------
    Multiple inheritance works here only if *initialisation is explicit* or uses
    cooperative `super()`.  Long-term, prefer composition: a `RankReallocator`
    component that receives a host “partition executor” interface.
    """

    
    def __init__(self, cfg):

        # Online partitioning file
        if cfg.hasopt('partition', f'online-file'):
            self.online_cfg_file = cfg.get('partition', 'online-file')
            self.online_cfg = Inifile.load(self.online_cfg_file)
        else:
            self.online_cfg_file = None
            self.online_cfg = cfg
            
    def recheck_online_file(self):
        if self.online_cfg_file is not None:
            self.online_cfg = Inifile.load(self.online_cfg_file)

        part_ranklist = self.online_cfg.getliteral('partition', 'compute-ranklist')

        if len(part_ranklist) != len(rankmap['compute']):
            initialise_new_comm('newcompute', part_ranklist)
        else:
            initialise_new_comm('newcompute', list(range(len(part_ranklist))))

    def swap_partitions(self, r0: int, r1: int) -> None:
        """
        Swap the per-rank mesh partition (State `i`) between two *world* ranks.

        Intended usage:
          - After a 'to-remove-rank' iterate, when some rank r0 is empty and
            you want to bubble that empty partition to the end (r1 = size - 1).
          - Once swapped, your existing "remove last rank" logic can safely
            drop the final rank without touching the MetaMesh internals.

        This:
          * exchanges `self.i` between ranks r0 and r1 via sendrecv,
          * recomputes the gid->owner maps,
          * retags con_i[..., 0] = owner_rank consistently,
          * invalidates topology-dependent caches.
        """

        commw = comm['world']
        myr   = int(rank['world'])
        nr    = commw.size

        # Normalise & validate inputs
        r0 = int(r0)
        r1 = int(r1)

        if r0 == r1:
            if myr == root['world']:
                print(f"[mm.swap] noop swap_partitions r0=r1={r0}", flush=True)
            return

        if not (0 <= r0 < nr and 0 <= r1 < nr):
            raise ValueError(
                f"swap_partitions: ranks out of range: r0={r0} r1={r1} size={nr}"
            )

        # Local bookkeeping: element count before swap
        before_local = self._local_count

        # Only the two ranks participate in the sendrecv; others are spectators
        if myr == r0 or myr == r1:
            partner    = r1 if myr == r0 else r0
            send_state = self.i
            # Send our State, receive partner's State (pickled Python object)
            recv_state = commw.sendrecv(send_state, dest=partner, source=partner)
            # Overwrite local State
            self.i = recv_state

        after_local = self._local_count

        # Collect before/after counts for a sanity log (all ranks participate)
        before_all = commw.allgather(before_local)
        after_all  = commw.allgather(after_local)

        if myr == root['world']: print(f"Swapped: {r0=} {r1=} ", flush=True)

        # Ownership tags in con_i have to be updated to reflect the new
        # gid->owner distribution. Do this once per rank.
        self._retag_con_owners()

        # Invalidate topology-dependent caches (owners_map, nei_gid_sets, ...)
        self._invalidate_topology()

    def rearrange_partitions(self) -> None:
        """
        Bubble all 0-element partitions to the end using repeated swap_partitions.

        Policy (minimal):
        - Pick the first i from the front with count[i] == 0
        - Pick the last j from the back with count[j] > 0
        - Swap i <-> j
        - Repeat until no such i exists
        """
        commw  = comm["world"]
        myr    = int(rank["world"])
        root_w = int(root["world"])
        nr     = int(commw.size)

        while True:
            # Everyone contributes their current local element count
            counts = commw.allgather(int(self._local_count))

            swap = None
            if myr == root_w:
                # Find last non-zero (ignore trailing zeros already "at the end")
                j = nr - 1
                while j >= 0 and counts[j] == 0:
                    j -= 1

                # Find first zero before j
                i = 0
                while i <= j and i < nr:
                    if counts[i] == 0:
                        # If i == j, it's already in the "end zone" effectively
                        swap = (i, j) if i < j else None
                        break
                    i += 1

                if swap is not None:
                    print(f"[mm.rearrange] swap {swap[0]} <-> {swap[1]} counts={counts}", flush=True)
                else:
                    print(f"[mm.rearrange] done counts={counts}", flush=True)

            # All ranks follow the same swap schedule
            swap = commw.bcast(swap, root=root_w)

            if swap is None:
                break

            self.swap_partitions(int(swap[0]), int(swap[1]))

    def _to_world_targets(self, targets_comp, comp_ranks):
        world = np.zeros(comm['world'].size, dtype=np.int64)
        for i, wr in enumerate(map(int, comp_ranks)):
            world[wr] = int(targets_comp[i])
        return world

    def remap_targets_by_ranklist(self, N_star_old):
        """
        Old compute-order targets -> WORLD-sized targets for the NEW ranklist.
        Policy:
        - Persisting ranks keep old mass
        - Added ranks get mean_new = Ntot / len(new_ranks)
        - Final int_round enforces sum == Ntot (over new_ranks only)
        - Ranks not in new_ranks get 0 in the returned world vector
        """
        old_ranks = list(map(int, rankmap['compute']))
        new_ranks = list(map(int, rankmap['newcompute']))

        old_mass = dict(zip(old_ranks, map(float, N_star_old)))
        mean_new = float(self.i.nelems_g) / float(len(new_ranks))

        # Newcompute-order float masses
        N_star_new = np.asarray([old_mass.get(wr, mean_new) for wr in new_ranks], dtype=float)
        return self._to_world_targets(self.int_round(N_star_new), new_ranks)

    # ----------------- Carving partitions to look better ----------------------


class CarverMixin:
    """Partition “carving” and regularisation utilities: islands, outliers/inliers, and
    interface-driven eviction/attraction.

    Carving is *topology-first* improvement applied after a primary partition move:
    - After global partitioning (SCOTCH/METIS), to remove islands and smooth seams.
    - During rank add/remove, to evacuate or attach clusters in a controlled manner.
    - During diffusion, as a stabiliser to reduce edge-cut and improve contiguity.

    Principles
    ----------
    - Collective safety: carving steps must commit an (possibly empty) plan every
      sweep so all ranks remain in lockstep.
    - Determinism: all selection must have explicit tie-breaks (gid, rank id, etype).
    - Locality: prefer moving elements on MPI interfaces using connectivity-derived
      scores (faces/vertices deltas), optionally augmented by geometry (centroids).

    Core capabilities (expected)
    ----------------------------
    1) Island detection
       - `label_islands_faces()` (and optionally a vertex-based analogue) should label
         connected components in the *local induced subgraph* of owned elements.
       - Output should be stable under permutations and provide component sizes.

    2) Cluster eviction primitives
       - `remove_cluster_step(cluster_flat, budget=..., boundary_only=...)`:
         evict a given set of elements off-rank using best neighbour heuristics.
       - Higher-level loops (e.g. `remove_small_islands_step`) should manage budgets,
         stall detection, and stopping criteria.

    3) Outlier/Inlier operations
       - Outliers: push “far” boundary elements to the neighbour they are most tied to.
       - Inliers: pull boundary elements that would fit better in a neighbour core.
       - Both should be configurable by fraction, metric (faces/vertices), and optional
         centroid-based scoring.

    4) Smoothing
       - A “local seam optimiser” that improves interface quality without changing
         global load targets; typically a bounded number of greedy moves.

    Host-class requirements
    -----------------------
    The host must provide:
    - connectivity and vertex access via `PartitionState` (`con_idx`, `con_mpi`,
      `spts_nodes`, `centroids`),
    - helper queries for MPI faces/vertices and per-neighbour interface lids,
    - a collective commit gate `_apply_plan_and_commit(...)`.

    Extensibility
    -------------
    Expose scoring as small overridable functions (or strategy objects) so carving can
    be reused for:
    - rank draining (evict all elements from a rank),
    - seeding (attach a patch to a new rank),
    - generic “make partitions look good” after any partitioner.

    Long-term recommendation: evolve this into a standalone `Carver` component with a
    minimal host interface, to avoid tight coupling with diffusion-specific methods.
    """

    def __init__(self, cfg=None):

        # Island stuff
        if cfg is not None:
            self.island_remove_fraction   = cfg.getfloat('partition', 'island-remove-fraction'  , 0.5)
            self.outlier_removal_mode     = cfg.get(     'partition', 'outlier-removal-mode'    , 'faces')
            self.outlier_removal_fraction = cfg.getfloat('partition', 'outlier-removal-fraction', 0.00)
            self.inlier_addition_mode     = cfg.get(     'partition', 'inlier-addition-mode'    , 'vertices')
            self.inlier_addition_fraction = cfg.getfloat('partition', 'inlier-addition-fraction', 0.00)
        else:
            self.island_remove_fraction   = 0.5
            self.outlier_removal_mode     = 'faces'
            self.outlier_removal_fraction = 0.00
            self.inlier_addition_mode     = 'vertices'
            self.inlier_addition_fraction = 0.00
    
    def label_islands_faces(self) -> tuple["np.ndarray", "np.ndarray"]:
        import numpy as np
        from collections import deque

        st = self.i
        gids = np.asarray(st.eidxs_flat, dtype=np.int64)
        nloc = int(gids.size)
        if nloc == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64)

        order = np.argsort(gids, kind="mergesort")
        gids_sorted = gids[order]

        uu_chunks: list[np.ndarray] = []
        vv_chunks: list[np.ndarray] = []

        for et in st.etypes:
            sl = st.etype_slices.get(et, None)
            if sl is None or sl.stop <= sl.start:
                continue
            ne = int(sl.stop - sl.start)

            nbr = np.asarray(st.con_idx[et], dtype=np.int64).reshape(ne, -1)
            m = (nbr >= 0)
            if not np.any(m):
                continue

            vv = nbr[m].astype(np.int64, copy=False)
            u_flat = np.arange(sl.start, sl.stop, dtype=np.int64)
            uu = np.broadcast_to(u_flat[:, None], nbr.shape)[m].astype(np.int64, copy=False)

            idx = np.searchsorted(gids_sorted, vv)
            ok = (idx < nloc)
            if np.any(ok):
                ok_idx = idx[ok]
                ok_vv  = vv[ok]
                ok[ok] = (gids_sorted[ok_idx] == ok_vv)

            if not np.any(ok):
                continue

            uu_chunks.append(uu[ok])
            vv_chunks.append(order[idx[ok]])

        if not uu_chunks:
            island_id = np.arange(nloc, dtype=np.int32)
            island_sizes = np.ones(nloc, dtype=np.int64)
            return island_id, island_sizes

        uu_all = np.concatenate(uu_chunks)
        vv_all = np.concatenate(vv_chunks)

        # Defensive: undirected connectivity
        uu0, vv0 = uu_all, vv_all
        uu_all = np.concatenate([uu0, vv0])
        vv_all = np.concatenate([vv0, uu0])

        perm = np.argsort(uu_all, kind="mergesort")
        uu_all = uu_all[perm]
        vv_all = vv_all[perm]

        counts = np.bincount(uu_all, minlength=nloc)
        vtab = np.empty(nloc + 1, dtype=np.int64)
        vtab[0] = 0
        np.cumsum(counts, out=vtab[1:])
        etab = vv_all.astype(np.int64, copy=False)

        island_id = np.full(nloc, -1, dtype=np.int32)
        sizes: list[int] = []
        cid = 0
        q = deque()

        for s in range(nloc):
            if island_id[s] != -1:
                continue
            island_id[s] = cid
            q.append(s)
            sz = 0

            while q:
                u = q.pop()
                sz += 1
                beg = int(vtab[u])
                end = int(vtab[u + 1])
                for v in etab[beg:end].tolist():
                    if island_id[v] == -1:
                        island_id[v] = cid
                        q.append(v)

            sizes.append(sz)
            cid += 1

        sizes_arr = np.asarray(sizes, dtype=np.int64)
        old_ids = np.arange(sizes_arr.size, dtype=np.int64)
        order2 = np.lexsort((old_ids, -sizes_arr))  # size desc, then id asc

        remap = np.empty_like(order2, dtype=np.int32)
        remap[order2] = np.arange(order2.size, dtype=np.int32)

        island_id = remap[island_id]
        island_sizes = sizes_arr[order2]
        return island_id.astype(np.int32, copy=False), island_sizes

    def _best_iface_neighbor_per_local_element(self):
        """
        For each local element (flat index), compute:
          - best_nbr[flat] : neighbour rank with max MPI-face contact
          - best_cnt[flat] : number of MPI faces to best_nbr
          - iface_mask[flat] : element touches any MPI face

        Uses _mpi_faces_by_neighbor() = {nbr: [(et, lid, fidx), ...]}.
        """
        st = self.i
        nloc = int(st.eidxs_flat.size)

        best_nbr = np.full(nloc, -1, dtype=np.int32)
        best_cnt = np.zeros(nloc, dtype=np.int16)   # face counts are small
        iface_mask = np.zeros(nloc, dtype=bool)

        per_nbr = self._mpi_faces_by_neighbor()
        if not per_nbr:
            return best_nbr, best_cnt, iface_mask

        for nbr, faces in per_nbr.items():
            nbr = int(nbr)
            if not faces:
                continue

            # Group by etype on-the-fly; count faces per (etype, lid)
            # Simple (Pythonic) approach; interface sizes are modest.
            by_et = {}
            for et, lid, _fidx in faces:
                by_et.setdefault(et, []).append(int(lid))

            for et, lids_list in by_et.items():
                if not lids_list:
                    continue

                lids = np.asarray(lids_list, dtype=np.int64)
                ulids, cnts = np.unique(lids, return_counts=True)  # counts = MPI faces to nbr

                flats = st.lids_to_flat(et, ulids)

                iface_mask[flats] = True

                # Update best neighbour by max contact; tie-break by lower nbr id
                cur_cnt = best_cnt[flats]
                cur_nbr = best_nbr[flats]

                better = (cnts > cur_cnt) | ((cnts == cur_cnt) & ((cur_nbr < 0) | (nbr < cur_nbr)))
                if np.any(better):
                    fsel = flats[better]
                    best_cnt[fsel] = cnts[better].astype(best_cnt.dtype, copy=False)
                    best_nbr[fsel] = np.int32(nbr)

        return best_nbr, best_cnt, iface_mask

    def remove_cluster_step(
        self,
        cluster_flat: np.ndarray,
        *,
        budget: int | None = None,
        boundary_only: bool = True,
        move_spts_nodes: bool = True,
    ) -> int:
        """
        One collective-safe sweep: attempt to evict elements in cluster_flat off this rank.

        IMPORTANT: Always calls _apply_plan_and_commit (even if no moves) so all ranks
        participate consistently.
        """
        W = comm["world"]
        rnk = int(rank["world"])

        st = self.i
        cluster_flat = np.asarray(cluster_flat, dtype=np.int64)

        # Always start from i->j, like diffuse()
        self._reset_j_with_i()

        # Empty cluster on this rank => still commit empty plan (collective safety)
        if cluster_flat.size == 0:
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
            return 0

        best_nbr, best_cnt, iface_mask = self._best_iface_neighbor_per_local_element()

        m = np.ones(cluster_flat.shape[0], dtype=bool)
        if boundary_only:
            m &= iface_mask[cluster_flat]
        m &= (best_nbr[cluster_flat] >= 0)

        cand = cluster_flat[m]

        # No eligible candidates => still commit empty plan (collective safety)
        if cand.size == 0:
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
            return 0

        # Deterministic priority: highest MPI-face contact first; tie-break by global gid
        gids = st.eidxs_flat[cand]
        keys0 = -best_cnt[cand].astype(np.int32)   # descending contact
        keys1 = gids.astype(np.int64)              # ascending gid
        order = np.lexsort((keys1, keys0))
        cand = cand[order]

        if budget is not None:
            budget = int(budget)
            if budget > 0 and cand.size > budget:
                cand = cand[:budget]

        chosen_flat = cand
        chosen_nbrs = best_nbr[chosen_flat].astype(np.int64, copy=False)

        # Build & apply relocation plan
        eidxs_diff = self._build_eidxs_diff_from_flat(chosen_flat, chosen_nbrs, st)

        moved_local = 0
        for _nbr, per_et in eidxs_diff.items():
            moved_local += sum(len(g) for g in per_et.values())

        # If chosen_flat non-empty but moved_local==0, something is inconsistent
        # (most likely _build_eidxs_diff_from_flat expects a different "flat" view).
        if chosen_flat.size and moved_local == 0 and rnk == int(root["world"]):
            print(
                f"[rmcluster.warn] chosen_flat={int(chosen_flat.size)} but moved_local=0; "
                "check _build_eidxs_diff_from_flat input expectations",
                flush=True,
            )

        self._apply_plan_and_commit(eidxs_diff, move_spts_nodes=move_spts_nodes)
        return int(moved_local)

    def add_ranks(self, targets, *, nseed: int = 1, verbose: bool = True) -> int:
        """
        Seed any rank with cur==0 and targets>0 by donating nseed elements.

        This does *not* try to reach targets; it just ensures ranks become non-empty.
        Your subsequent iterate_till_convergence(targets) does the heavy lifting.
        """
        W = comm["world"]
        rnk = int(rank["world"])
        root_w = int(root["world"])

        targets = np.asarray(targets, dtype=np.int64)
        cur = np.asarray(self._cur_counts_total, dtype=np.int64)

        new_ranks = [i for i in range(int(W.size)) if int(cur[i]) == 0 and int(targets[i]) > 0]

        if verbose and rnk == root_w:
            print(f"[add_ranks] cur0={cur.tolist()}", flush=True)
            print(f"[add_ranks] tgt ={targets.tolist()}", flush=True)
            print(f"[add_ranks] new_ranks={new_ranks}", flush=True)

        # Collective-safe no-op
        if not new_ranks:
            self._apply_plan_and_commit({}, move_spts_nodes=True)
            return 0

        moved_local = self.seed_new_ranks(new_ranks, targets, nseed=nseed, verbose=verbose)
        moved_glob = int(W.allreduce(int(moved_local), op=mpi.SUM))

        if verbose and rnk == root_w:
            print(f"[add_ranks] moved_glob(seeds)={moved_glob}", flush=True)

        self.iterate_till_convergence(targets, smooth=False, max_iters=10)

        return int(moved_local)

    def seed_new_ranks(self, rank_ids, targets, *, nseed: int = 1, verbose: bool = True) -> int:
        """
        Seed each rank in rank_ids (currently empty but target>0) with nseed elements.

        Output: applies relocation via _apply_plan_and_commit.
        Deterministic and collective-safe.
        """
        W = comm["world"]
        P = int(W.size)
        rnk = int(rank["world"])
        root_w = int(root["world"])

        rank_ids = sorted({int(r) for r in rank_ids})
        targets = np.asarray(targets, dtype=np.int64)
        cur = np.asarray(self._cur_counts_total, dtype=np.int64)

        # Per-rank preferred etype order (hardware-aware) gathered from each process itself
        # Works even for empty ranks because they still run code and report their preference list.
        pref_orders = W.allgather(list(self._etype_order()) if hasattr(self, "_etype_order") else [])

        # Helper: pick seed etype for a given rank, sanitized to existing mesh etypes
        mesh_etypes = list(self.i.etypes)
        def seed_etype_for_rank(rr: int) -> str:
            order = pref_orders[rr] or []
            for et in order:
                if et in mesh_etypes:
                    return et
            return mesh_etypes[0] if mesh_etypes else ""

        # Build deterministic donor mapping
        donor_for = {}
        cur_work = cur.astype(np.int64, copy=True)

        for nr in rank_ids:
            d = self.donor_ranks_for_seed(nr, targets=targets, cur=cur_work)
            donor_for[nr] = int(d)

            # pessimistically decrement donor's "available" mass so multiple new ranks don't all hit the same donor
            # (we only move nseed, but counts are in elements; this is just for donor selection stability)
            cur_work[int(d)] = max(0, int(cur_work[int(d)] - int(nseed)))

        donors = sorted(set(donor_for.values()))

        if verbose and rnk == root_w:
            print(f"[seed_new_ranks] rank_ids={rank_ids}", flush=True)
            print(f"[seed_new_ranks] donor_for={donor_for}", flush=True)

        # Build plan only on donor ranks; others keep {}.
        eidxs_diff: dict[int, dict[str, np.ndarray]] = {}

        # For each donor, broadcast which (new_rank, seed_etype) it must serve.
        for d in donors:
            req = None
            if rnk == root_w:
                req = [(nr, seed_etype_for_rank(nr)) for nr in rank_ids if donor_for[nr] == d]
            req = W.bcast(req, root=root_w)

            # Donor selects gids locally, then broadcasts chosen gids for logging consistency (optional)
            payload = None
            if rnk == d:
                chosen = []
                used = set()
                for nr, et in req:
                    et = str(et)
                    gids = self.donor_seeds(et, n=nseed, used_gids=used)
                    # gids may be empty if donor truly cannot provide; keep deterministic behavior
                    chosen.append((int(nr), et, gids.astype(np.int64, copy=False)))
                    for gg in gids.tolist():
                        used.add(int(gg))
                payload = chosen

            payload = W.bcast(payload, root=d)

            if rnk == d:
                # Convert payload into eidxs_diff format: dest -> {etype -> gids}
                for (nr, et, gids) in payload:
                    if gids.size == 0:
                        continue
                    per = eidxs_diff.setdefault(int(nr), {})
                    # Append if already exists (unlikely, but safe)
                    if et in per:
                        per[et] = np.concatenate([per[et], gids])
                    else:
                        per[et] = gids

        moved_local = int(sum(len(g) for per in eidxs_diff.values() for g in per.values()))

        if verbose and rnk == root_w:
            seeded = []
            for d in donors:
                seeded.extend([(nr, donor_for[nr]) for nr in rank_ids if donor_for[nr] == d])
            print(f"[seed_new_ranks] donors={donors} seeded_pairs={seeded}", flush=True)

        # Collective-safe commit (even if empty on this rank)
        self._apply_plan_and_commit(eidxs_diff, move_spts_nodes=True)

        return moved_local

    def donor_ranks_for_seed(self, new_rank: int, *, targets, cur=None) -> int:
        """
        Pick a donor for seeding `new_rank`.

        Policy (deterministic):
        1) Among ranks with cur>0 and r!=new_rank, choose max surplus (cur-target).
        2) If all surplus <= 0, choose rank with max cur.
        3) Tie-break: smallest rank id.
        """
        W = comm["world"]
        P = int(W.size)

        new_rank = int(new_rank)
        targets = np.asarray(targets, dtype=np.int64)

        if cur is None:
            cur = np.asarray(self._cur_counts_total, dtype=np.int64)
        else:
            cur = np.asarray(cur, dtype=np.int64)

        candidates = [r for r in range(P) if r != new_rank and int(cur[r]) > 0]

        if not candidates:
            # Degenerate: nobody has elements; return root (caller will fail to seed anyway)
            return int(root["world"])

        surplus = {r: int(cur[r] - targets[r]) for r in candidates}
        max_sur = max(surplus.values()) if surplus else None

        if max_sur is not None and max_sur > 0:
            best = [r for r in candidates if surplus[r] == max_sur]
            return int(min(best))

        # fallback: largest cur
        best = sorted(candidates, key=lambda r: (-int(cur[r]), int(r)))
        return int(best[0])

    def donor_seeds(self, seed_etype: str, *, n: int = 1, used_gids=None) -> np.ndarray:
        """
        On the donor rank: choose up to n gids to donate.

        Prefer elements of seed_etype that touch an MPI face and have high interface contact.
        Tie-break by gid for determinism. Avoid gids in used_gids.
        """
        import numpy as np

        n = int(n)
        used_gids = set() if used_gids is None else used_gids

        st = self.i
        et = str(seed_etype)

        # Fallback etype if donor has none of requested
        if et not in st.etype_slices or st.etype_slices[et].stop <= st.etype_slices[et].start:
            # pick any available etype with elements (deterministic order)
            for et2 in st.etypes:
                sl2 = st.etype_slices.get(et2)
                if sl2 and sl2.stop > sl2.start:
                    et = et2
                    break

        sl = st.etype_slices.get(et)
        if sl is None or sl.stop <= sl.start:
            return np.empty(0, dtype=np.int64)

        # Interface-aware scoring
        best_nbr, best_cnt, iface_mask = self._best_iface_neighbor_per_local_element()

        flats = np.arange(int(sl.start), int(sl.stop), dtype=np.int64)
        gids  = st.eidxs_flat[flats].astype(np.int64, copy=False)

        # Prefer interface elements, but if none exist, allow interior
        m_iface = iface_mask[flats]
        cand = flats[m_iface] if np.any(m_iface) else flats

        # Filter already-used gids
        if used_gids:
            cg = st.eidxs_flat[cand].astype(np.int64, copy=False)
            m = np.array([int(g) not in used_gids for g in cg.tolist()], dtype=bool)
            cand = cand[m]

        if cand.size == 0:
            return np.empty(0, dtype=np.int64)

        cgids = st.eidxs_flat[cand].astype(np.int64, copy=False)
        ccnt  = best_cnt[cand].astype(np.int32, copy=False)

        # Sort: max contact first, then gid ascending
        order = np.lexsort((cgids, -ccnt))
        pick = cand[order[: max(1, min(n, cand.size))]]

        return st.eidxs_flat[pick].astype(np.int64, copy=False)

    def remove_islands_step(self, *, max_move: int | None = None,
                                           max_sweeps: int = 50,
                                           patience: int = 1) -> None:
        W = comm["world"]
        rnk = int(rank["world"])
        root_w = int(root["world"])

        frac_remove = self.island_remove_fraction

        island_id, island_sizes = self.label_islands_faces()
        nis  = int(island_sizes.size)
        nloc = int(island_id.size)

        is_trivial = (nis <= 1) or (nloc == 0)

        if is_trivial:
            cluster_gids = np.empty(0, np.int64)
            nrm = 0
        else:
            cand_ids = np.arange(1, nis, dtype=np.int32)
            nrm = int(np.ceil(frac_remove * cand_ids.size))
            nrm = max(1, min(nrm, cand_ids.size))
            ids_rm = cand_ids[-nrm:]

            cluster_flat0 = np.nonzero(np.isin(island_id, ids_rm))[0].astype(np.int64, copy=False)
            cluster_gids  = np.asarray(self.i.eidxs_flat[cluster_flat0], dtype=np.int64)

        # Aggressive eviction loop (collective-safe)
        budget_left = None if max_move is None else int(max_move)
        stable = 0

        for k in range(int(max_sweeps)):
            if cluster_gids.size:
                in_cluster = np.isin(self.i.eidxs_flat, cluster_gids, assume_unique=False)
                cluster_flat = np.nonzero(in_cluster)[0].astype(np.int64, copy=False)
            else:
                cluster_flat = np.empty(0, np.int64)

            remaining_loc  = int(cluster_flat.size)
            remaining_glob = int(W.allreduce(remaining_loc, op=mpi.SUM))

            if remaining_glob == 0:
                break

            # Enforce max_move as a *total* budget across sweeps
            if budget_left is not None:
                if budget_left <= 0:
                    # still participate collectively
                    moved_local = self.remove_cluster_step(np.empty(0, np.int64), budget=0)
                else:
                    moved_local = self.remove_cluster_step(cluster_flat, budget=budget_left)
                budget_left -= int(moved_local)
            else:
                moved_local = self.remove_cluster_step(cluster_flat, budget=None)

            moved_glob = int(W.allreduce(int(moved_local), op=mpi.SUM))

            if rnk == root_w:
                print(f"[rmislands.aggr] sweep={k+1} remaining_glob={remaining_glob} moved_glob={moved_glob}", flush=True)

            if moved_glob == 0:
                stable += 1
            else:
                stable = 0

            if stable >= int(patience):
                if rnk == root_w:
                    print("[rmislands.aggr] STALL: moved_glob=0 while remaining_glob>0", flush=True)
                break

        # Optional one-shot summary (cheap)
        stats_local = (nis, int(cluster_gids.size))
        stats_all   = W.allgather(stats_local)
        if rnk == root_w:
            print(f"[rmislands.aggr] DONE {stats_all}", flush=True)


        # Return remaining islands, nrm
        nrms = W.allgather(nrm)
        return nrms

    def detect_islands(self):
        """
        Detect islands and select a subset to remove (based on island_remove_fraction).

        Returns
        -------
        cluster_gids : np.ndarray[int64]
            Local gids belonging to the selected-to-remove islands on *this* rank.
        nis_all : list[int]
            Allgathered number of islands per rank (so caller can decide to stop if all <= 1).
        """
        W      = comm["world"]
        rnk    = int(rank["world"])
        root_w = int(root["world"])

        frac_remove = float(self.island_remove_fraction)

        island_id, island_sizes = self.label_islands_faces()
        nis  = int(island_sizes.size)
        nloc = int(island_id.size)

        # "How many islands exist" (per rank); this is what you want for the stop test
        nis_all = W.allgather(nis)

        # Trivial / empty local partition
        if (nis <= 1) or (nloc == 0) or (frac_remove <= 0.0):
            cluster_gids = np.empty(0, np.int64)
            nrm_ids = 0
        else:
            # NOTE: island label 0 is assumed to be the main component; remove from 1..nis-1
            cand_ids = np.arange(1, nis, dtype=np.int32)

            # How many island-IDs to remove this round (fraction of non-main islands)
            nrm_ids = int(np.ceil(frac_remove * cand_ids.size))
            nrm_ids = max(1, min(nrm_ids, cand_ids.size))

            # Remove the last nrm_ids (assuming ordering corresponds to increasing size or some policy)
            ids_rm = cand_ids[-nrm_ids:]

            cluster_flat0 = np.nonzero(np.isin(island_id, ids_rm))[0].astype(np.int64, copy=False)
            cluster_gids  = np.asarray(self.i.eidxs_flat[cluster_flat0], dtype=np.int64)

        if rnk == root_w:
            # cheap summary; keep it small
            print(f"[rmislands.detect] nis_all={nis_all}", flush=True)

        return cluster_gids, nis_all


    def remove_islands(self, cluster_gids, *, max_move: int | None = None,
                    max_sweeps: int = 50, patience: int = 1) -> None:
        """
        Collective-safe aggressive eviction loop for the provided cluster_gids.

        Parameters
        ----------
        cluster_gids : np.ndarray[int64]
            Local gids that belong to the islands you want to evict from this rank.
            (May be empty on some ranks; still collective-safe.)
        """
        W      = comm["world"]
        rnk    = int(rank["world"])
        root_w = int(root["world"])

        cluster_gids = np.asarray(cluster_gids, dtype=np.int64)
        budget_left = None if max_move is None else int(max_move)
        stable = 0

        for k in range(int(max_sweeps)):
            if cluster_gids.size:
                in_cluster   = np.isin(self.i.eidxs_flat, cluster_gids, assume_unique=False)
                cluster_flat = np.nonzero(in_cluster)[0].astype(np.int64, copy=False)
            else:
                cluster_flat = np.empty(0, np.int64)

            remaining_loc  = int(cluster_flat.size)
            remaining_glob = int(W.allreduce(remaining_loc, op=mpi.SUM))

            if remaining_glob == 0:
                break

            # Enforce max_move as a *total* budget across sweeps
            if budget_left is not None:
                if budget_left <= 0:
                    moved_local = self.remove_cluster_step(np.empty(0, np.int64), budget=0)
                else:
                    moved_local = self.remove_cluster_step(cluster_flat, budget=budget_left)
                budget_left -= int(moved_local)
            else:
                moved_local = self.remove_cluster_step(cluster_flat, budget=None)

            moved_glob = int(W.allreduce(int(moved_local), op=mpi.SUM))

            if rnk == root_w:
                print(f"[rmislands.aggr] sweep={k+1} remaining_glob={remaining_glob} moved_glob={moved_glob}", flush=True)

            if moved_glob == 0:
                stable += 1
            else:
                stable = 0

            if stable >= int(patience):
                if rnk == root_w:
                    print("[rmislands.aggr] STALL: moved_glob=0 while remaining_glob>0", flush=True)
                break



    def _iface_from_deltas(self, mode: str, *, delta_max: int | None = None):
        st   = self.i
        nloc = int(st.eidxs_flat.size)

        deltas = self._compute_deltas(mode)  # {nbr:{et:(N,2)[lid,delta]}}
        iface_mask = np.zeros(nloc, dtype=bool)

        if not deltas:
            return (
                iface_mask,
                np.full(nloc, -1, dtype=np.int32),
                np.full(nloc,  2**30, dtype=np.int32),
            )

        best_nbr = np.full(nloc, -1, dtype=np.int32)
        best_del = np.full(nloc,  2**30, dtype=np.int32)

        for nbr, per_et in deltas.items():
            nbr = int(nbr)
            for et, mat in per_et.items():
                if mat is None or mat.size == 0:
                    continue

                lids  = mat[:, 0].astype(np.int64, copy=False)
                delt  = mat[:, 1].astype(np.int32, copy=False)
                flats = st.lids_to_flat(et, lids)

                # touches some MPI interface (regardless of delta gate)
                iface_mask[flats] = True

                if delta_max is not None:
                    ok = (delt <= int(delta_max))
                    if not np.any(ok):
                        continue
                    flats2 = flats[ok]
                    delt2  = delt[ok]
                else:
                    flats2 = flats
                    delt2  = delt

                curd = best_del[flats2]
                curn = best_nbr[flats2]
                better = (delt2 < curd) | ((delt2 == curd) & ((curn < 0) | (nbr < curn)))
                if np.any(better):
                    sel = flats2[better]
                    best_del[sel] = delt2[better]
                    best_nbr[sel] = np.int32(nbr)

        return iface_mask, best_nbr, best_del

    def _move_by_best_rank(self, chosen_flat, best_rank, *, move_spts_nodes: bool):
        st = self.i
        chosen_flat = np.asarray(chosen_flat, dtype=np.int64)
        if chosen_flat.size == 0:
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
            return 0

        chosen_nbrs = np.asarray(best_rank[chosen_flat], dtype=np.int64)
        eidxs_diff  = self._build_eidxs_diff_from_flat(chosen_flat, chosen_nbrs, st)

        moved_local = int(sum(len(g) for per in eidxs_diff.values() for g in per.values()))
        self._apply_plan_and_commit(eidxs_diff, move_spts_nodes=move_spts_nodes)
        return moved_local

    def _centroids_flat(self):
        st = self.i
        cflat = np.zeros((st.eidxs_flat.size, 3), dtype=np.float64)
        okc   = np.zeros((st.eidxs_flat.size,), dtype=bool)

        for et, sl in st.etype_slices.items():
            if sl.stop <= sl.start:
                continue
            c = st.centroids.get(et)
            if c is None or c.shape[0] != (sl.stop - sl.start):
                continue
            cflat[sl, :] = c
            okc[sl] = np.isfinite(c).all(axis=1)

        return cflat, okc

    def _select_top_fraction(self, cand_flat, score, *, frac: float, higher_is_better: bool = True):
        import numpy as np

        cand_flat = np.asarray(cand_flat, dtype=np.int64)
        if cand_flat.size == 0:
            return cand_flat

        frac = float(frac)
        if frac <= 0.0:
            return np.empty(0, np.int64)

        n = int(cand_flat.size)
        nsel = int(np.ceil(frac * n))
        nsel = max(1, min(nsel, n))

        gids = self.i.eidxs_flat[cand_flat].astype(np.int64, copy=False)
        sc   = np.asarray(score, dtype=np.float64)

        # primary score, secondary gid (deterministic)
        key0 = -sc if higher_is_better else sc
        order = np.lexsort((gids, key0))
        return cand_flat[order[:nsel]]

    def remove_outliers(
        self,
        *,
        attach_bias: float = 0.0,
        require_nonpos_delta_for_faces: bool = True,
        move_spts_nodes: bool = True,
        verbose: bool = True,
    ) -> int:
        top  = float(self.outlier_removal_fraction)
        mode = str(self.outlier_removal_mode).lower()

        W      = comm["world"]
        rnk    = int(rank["world"])
        root_w = int(root["world"])

        self._reset_j_with_i()

        moved_local = 0  # ALWAYS define, ALWAYS participate in allreduce

        if top <= 0.0:
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
        else:
            self._cores = self._compute_cores_from_centroids()
            core = np.asarray(self._cores[rnk], dtype=np.float64)

            cflat, okc = self._centroids_flat()

            is_faces = mode.startswith(("f", "e"))  # faces/edges
            dmax = 0 if (is_faces and require_nonpos_delta_for_faces) else None
            iface_mask, best_nbr, best_del = self._iface_from_deltas(mode, delta_max=dmax)

            # Default: no-op commit unless we successfully pick moves
            do_move = True

            if (not np.all(np.isfinite(core))):
                do_move = False
            else:
                cand_flat = np.nonzero(iface_mask & okc & (best_nbr >= 0))[0].astype(np.int64, copy=False)
                if cand_flat.size == 0:
                    do_move = False
                else:
                    d = np.linalg.norm(cflat[cand_flat, :] - core[None, :], axis=1)
                    mu, sig = float(d.mean()), float(d.std())
                    z = (d - mu) / sig if sig > 0 else np.zeros_like(d)

                    score = z + float(attach_bias) * (-best_del[cand_flat].astype(np.float64))

                    chosen_flat = self._select_top_fraction(
                        cand_flat, score, frac=top, higher_is_better=True
                    )

                    if chosen_flat.size == 0:
                        do_move = False
                    else:
                        moved_local = self._move_by_best_rank(
                            chosen_flat, best_nbr, move_spts_nodes=move_spts_nodes
                        )

            if not do_move:
                self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)

        moved_glob = int(W.allreduce(int(moved_local), op=mpi.SUM))
        if verbose and rnk == root_w:
            # cand size may be undefined if we never built it; keep log minimal/robust
            print(f"[rmoutliers] mode={mode} top={top} moved_glob={moved_glob}", flush=True)

        return int(moved_local)

    def _best_dest_by_core_from_deltas(
        self,
        mode: str,
        *,
        cores: np.ndarray,
        cflat: np.ndarray,
        okc: np.ndarray,
        delta_max: int | None = None,
    ):
        st   = self.i
        nloc = int(st.eidxs_flat.size)
        P    = int(cores.shape[0])

        deltas = self._compute_deltas(mode)  # {nbr:{et:(N,2)[lid,delta]}}
        iface_mask = np.zeros(nloc, dtype=bool)

        if not deltas:
            return (
                iface_mask,
                np.full(nloc, -1, dtype=np.int32),
                np.full(nloc, np.inf, dtype=np.float64),
                np.full(nloc, 2**30, dtype=np.int32),
            )

        best_dest  = np.full(nloc, -1, dtype=np.int32)
        best_dist  = np.full(nloc, np.inf, dtype=np.float64)
        best_delta = np.full(nloc, 2**30, dtype=np.int32)

        for nbr, per_et in deltas.items():
            nbr = int(nbr)
            if nbr < 0 or nbr >= P:
                continue

            core_n = cores[nbr]
            if not np.all(np.isfinite(core_n)):
                continue

            for et, mat in per_et.items():
                if mat is None or mat.size == 0:
                    continue

                lids  = mat[:, 0].astype(np.int64, copy=False)
                delt  = mat[:, 1].astype(np.int32, copy=False)
                flats = st.lids_to_flat(et, lids)

                iface_mask[flats] = True

                # centroid-valid only
                m_ok = okc[flats]
                if not np.any(m_ok):
                    continue

                flats2 = flats[m_ok]
                delt2  = delt[m_ok]

                if delta_max is not None:
                    m_d = (delt2 <= int(delta_max))
                    if not np.any(m_d):
                        continue
                    flats2 = flats2[m_d]
                    delt2  = delt2[m_d]

                d = np.linalg.norm(cflat[flats2, :] - core_n[None, :], axis=1)

                curd = best_dist[flats2]
                curk = best_dest[flats2]
                better = (d < curd) | ((d == curd) & ((curk < 0) | (nbr < curk)))
                if np.any(better):
                    sel = flats2[better]
                    best_dist[sel]  = d[better]
                    best_dest[sel]  = np.int32(nbr)
                    best_delta[sel] = delt2[better]

        return iface_mask, best_dest, best_dist, best_delta

    def add_inliers(
        self,
        *,
        margin: float = 0.0,
        move_spts_nodes: bool = True,
        verbose: bool = True,
    ) -> int:
        top  = float(self.inlier_addition_fraction)
        mode = str(self.inlier_addition_mode).lower()

        W      = comm["world"]
        rnk    = int(rank["world"])
        root_w = int(root["world"])

        self._reset_j_with_i()

        moved_local = 0

        if top <= 0.0:
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
        else:
            self._cores = self._compute_cores_from_centroids()
            cores = np.asarray(self._cores, dtype=np.float64)
            cflat, okc = self._centroids_flat()

            do_move = True
            if cores.ndim != 2:
                do_move = False
            else:
                core_self = cores[rnk]
                if not np.all(np.isfinite(core_self)):
                    do_move = False
                else:
                    iface_mask, best_dest, best_dist, _best_delta = self._best_dest_by_core_from_deltas(
                        mode, cores=cores, cflat=cflat, okc=okc, delta_max=None
                    )

                    cand_flat = np.nonzero(iface_mask & okc & (best_dest >= 0))[0].astype(np.int64, copy=False)
                    if cand_flat.size == 0:
                        do_move = False
                    else:
                        c      = cflat[cand_flat, :]
                        d_self = np.linalg.norm(c - core_self[None, :], axis=1)
                        d_dest = best_dist[cand_flat]
                        score  = d_self - d_dest  # >0 means improvement

                        m = score > float(margin)
                        cand2 = cand_flat[m]
                        if cand2.size == 0:
                            do_move = False
                        else:
                            chosen_flat = self._select_top_fraction(
                                cand2, score[m], frac=top, higher_is_better=True
                            )

                            if chosen_flat.size == 0:
                                do_move = False
                            else:
                                moved_local = self._move_by_best_rank(
                                    chosen_flat, best_dest, move_spts_nodes=move_spts_nodes
                                )

            if not do_move:
                self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)

        moved_glob = int(W.allreduce(int(moved_local), op=mpi.SUM))
        if verbose and rnk == root_w:
            print(f"[addinliers] mode={mode} top={top} moved_glob={moved_glob}", flush=True)

        return int(moved_local)


class OfflineRepartitioner(_MetaMesh):
    """Base wrapper that provides mesh/partition *construction* services common to all
    partitioners (diffusion, SCOTCH, METIS, ...).

    This class should stay small: it is the bridge between a PyFR mesh object and
    the online partition representation (`_MetaMesh` + `PartitionState`).  It should
    avoid policy decisions.

    Responsibilities
    ---------------
    - Construct initial `PartitionState` from a PyFR mesh (or from vparts), including
      global id conventions and connectivity arrays.
    - Provide utilities needed by *all* partitioners:
        * etype ordering, global counts/offsets,
        * global gid <-> (etype,lid) conventions,
        * basic validation and logging.
    - Provide target computation (`calc_target_ecounts`) *only if* it is a pure
      transformation of telemetry inputs; otherwise delegate to a cost-model mixin.

    Public API (expected)
    ---------------------
    - `restart()` / `_reset_j_with_i()` to return to a clean staging state.
    - `calc_target_ecounts(current_counts, telemetry...) -> targets`
    - `to_mesh(eidxs)` / `from_mesh(mesh)` (possibly inherited from `_MetaMesh`).

    Extensibility / inheritance
    ---------------------------
    If additional partitioners require extra data (graph caches, weights), attach it
    in subclasses (e.g. OnlineSCOTCHPartitioner) but keep this base free of solver-
    specific concerns.
    """


    def __init__(self, mesh, cfg=None):
        _MetaMesh.__init__(self, mesh, cfg)

    def _to_world_targets(self, targets_comp, comp_ranks):
        return targets_comp

    def calc_target(self, weights):
        """
            OfflineRepartitioner: 
            compute target element counts for each rank 
            using the current element counts and optional weights.
        """
        Neach = self.i.nelems
        Nall = comm['compute'].allgather(np.asarray(Neach, dtype=np.int64))
        weights = np.asarray(weights, dtype=np.float64)
        targets_unscaled = weights * Nall
        targets_comp = self.int_round(targets_unscaled)      # length = len(rankmap['compute'])
        return self._to_world_targets(targets_comp, rankmap['compute'])

class OnlinePartitioner(RankAllocatorMixin, WaitsToTargetsModelMixin, OfflineRepartitioner):
    shuffle_if_stagnant = 0

    def __init__(self, mesh, cfg):
        # Everything related to costs
        OfflineRepartitioner.__init__(self, mesh, cfg)
        WaitsToTargetsModelMixin.__init__(self, cfg)
        RankAllocatorMixin.__init__(self, cfg)

        # Initially balance elements aggressively
        self._init_aggr_iters = cfg.getint('partition',
                                             'lb-init-aggressive-iters', 
                                              comm['world'].size)

        self.jitter = cfg.getfloat('partition', 'target-jitter', 0.0)
        self.jitter_rank = 0 

    def iterate_aggressively(self, target):

        if rank["world"] == root['world']: print(f"TARGET: {target}")

        if self._init_aggr_iters > 0:
            for _ in range(self._init_aggr_iters): 
                cur0 = self._cur_counts_total
                if rank["world"] == root['world']: print(f"CURRENT: {cur0}")     
                self.iterate(target)
            self._init_aggr_iters-=1
        else:
            cur0 = self._cur_counts_total
            if rank["world"] == root['world']: print(f"CURRENT: {cur0}")     
            self.iterate(target)

    def calc_target(self, g1a, g1s, g1r):
        """
            OnlinePartitioner
            Uses parent class methods to get target distribution.
        """
        cost_old = execute['compute'](lambda: self.compute_cost(g1a, g1s, g1r), None)

        if cost_old is not None:
            target_int = super().calc_target(1/np.array(cost_old))
        else:   
            target_int = None

        target_int = comm['world'].bcast(target_int)

        target_world = self.remap_targets_by_ranklist(target_int)

        target_world_with_jitter = self.add_jitter_to_targets(target_world)

        return target_world_with_jitter

    def add_jitter_to_targets(self, target):
        """
        Cyclically add +jitter to each rank
        """
        if self.jitter > 0.0:   
            if rank["world"] == root['world']: print(f"JITTER OLD: {target}")
            target[self.jitter_rank] = target[self.jitter_rank]*(1+self.jitter)
            self.jitter_rank = (self.jitter_rank + 1) % len(target)
            target = self.int_round(target)
            if rank["world"] == root['world']: print(f"JITTER NEW: {target}")
            
        return target

    def detect_stagnation(self):

        if not self.shuffle_if_stagnant:
            return None

        patience = self.shuffle_if_stagnant

        cost = np.asarray(self.cost, dtype=np.float64)
        cmax = float(np.max(cost))

        if rank['world'] == root['world']:
            # init
            if self.last_max_cost is None:
                self.last_max_cost = cmax
                self.last_max_cost_iter = 0
                worst = None
            else:
                # “no improvement” => increment
                improved = cmax < (self.last_max_cost)
                if improved:
                    self.last_max_cost = cmax
                    self.last_max_cost_iter = 0
                    worst = None
                else:
                    self.last_max_cost_iter += 1
                    if self.last_max_cost_iter >= int(patience):
                        # deterministic argmax tie-break: lowest rank id
                        worst = int(np.argmax(cost))
                    else:
                        worst = None
        else:
            worst = None

        worst = comm['world'].bcast(worst, root=root['world'])
        return worst

class OnlineGlobalPartitioner(OnlinePartitioner):
    _backend_cls = None          # METISPartitioner or SCOTCHPartitioner
    _backend_name = "backend"
    _backend_strat = None        # SCOTCH only; e.g. "quality"
    name = None

    def __init__(self, mesh, cfg):
        super().__init__(mesh, cfg)
        self._root_cache = None

        # 1) Global opts applied to any online partitioner backend
        self.base = cfg.getliteral('partition', 'online-partitioner-opts', {})

        # 2) Backend-specific opts: scotch-opts / metis-opts
        self.per = cfg.getliteral('partition', f'{self.name}-opts', {})

        if not isinstance(self.base, dict) or not isinstance(self.per, dict):
            raise ValueError("[partition] online-partitioner-opts and <name>-opts must be dict literals")


    def _backend_opts(self, *, ufactor = None, seed = None):
        """
        Build backend opts by reusing PyFR's existing partitioner option parsing.

        Precedence (highest wins):
        1) explicit args ufactor/seed passed to partition(...)
        2) per-backend online opts in cfg, e.g. scotch-opts / metis-opts
        3) global online opts in cfg: online-partitioner-opts
        4) backend class defaults (handled inside BasePartitioner)
        """

        # Merge (per-backend overrides global)
        opts = dict(self.base)
        opts.update(self.per)

        # Apply call-time overrides last
        if ufactor is not None:
            opts['ufactor'] = int(ufactor)
        if seed is not None:
            opts['seed'] = int(seed)

        # SCOTCH: allow either 'strat' via cfg, or fall back to class attribute if set
        if self._backend_name == "scotch" and 'strat' not in opts and self._backend_strat is not None:
            opts['strat'] = self._backend_strat

        return opts

    @staticmethod
    def _as_i64_con(con):
        con = np.asarray(con, dtype=np.int64)
        return np.ascontiguousarray(con.reshape(-1, 2))

    def _canonicalize_con(self, con, *, label: str, nelems_g: int):
        """
        Canonicalize an undirected edge list in global element-id space:
          - ensure l <= r
          - drop self-loops
          - unique pairs
        """
        con0 = self._as_i64_con(con)
        n0 = int(con0.shape[0])

        if n0 == 0:
            if rank['world'] == root['world']:
                print(f"[globcache.con] {label} edges=0 selfloops=0 dups_removed=0 edges_final=0", flush=True)
            return con0

        # Ensure l <= r (canonical undirected form)
        con0.sort(axis=1)

        # Drop self-loops introduced by periodic merging
        selfloops = int(np.count_nonzero(con0[:, 0] == con0[:, 1]))
        con1 = con0[con0[:, 0] != con0[:, 1]]

        # Unique pairs (structured view trick)
        cdtype = np.dtype([('l', np.int64), ('r', np.int64)])
        cu = np.unique(con1.view(cdtype).squeeze())
        con2 = cu.view(np.int64).reshape(-1, 2)

        dups_removed = int(con1.shape[0] - con2.shape[0])

        # Range check (important to catch corrupted IDs early)
        if con2.size:
            mn = int(con2.min())
            mx = int(con2.max())
            if mn < 0 or mx >= int(nelems_g):
                raise RuntimeError(f"[globcache.con] {label} out-of-range ids: min={mn} max={mx} nelems_g={nelems_g}")

        if rank['world'] == root['world']:
            print(f"[globcache.con] {label} edges={n0} selfloops={selfloops} "
                  f"dups_removed={dups_removed} edges_final={int(con2.shape[0])}", flush=True)

        return con2

    def _validate_graph_buffers(self, graph):
        """
        Pure-Python structural checks that catch most SCOTCH-killers deterministically.
        """
        vtab = np.asarray(graph.vtab)
        etab = np.asarray(graph.etab)

        if vtab.ndim != 1 or etab.ndim != 1:
            raise RuntimeError("[globcache.graph] vtab/etab must be 1D")

        if int(vtab[0]) != 0:
            raise RuntimeError(f"[globcache.graph] vtab[0]={int(vtab[0])} != 0")

        if int(vtab[-1]) != int(etab.size):
            raise RuntimeError(f"[globcache.graph] vtab[-1]={int(vtab[-1])} != etab.size={int(etab.size)}")

        if np.any(vtab[1:] < vtab[:-1]):
            raise RuntimeError("[globcache.graph] vtab not nondecreasing")

        nverts = int(vtab.size - 1)
        if etab.size:
            mn = int(etab.min())
            mx = int(etab.max())
            if mn < 0 or mx >= nverts:
                raise RuntimeError(f"[globcache.graph] etab out of range: min={mn} max={mx} nverts={nverts}")

    def _sanitize_graph(self, graph):
        """
        Ensure numpy buffers are owned, contiguous, and stable for ctypes backends.
        """
        def as_i32_c(a):
            return np.ascontiguousarray(np.asarray(a, dtype=np.int32))

        g2 = graph._replace(
            vtab=as_i32_c(graph.vtab),
            etab=as_i32_c(graph.etab),
            vwts=as_i32_c(graph.vwts),
            ewts=as_i32_c(graph.ewts),
        )

        self._validate_graph_buffers(g2)
        return g2

    def _build_root_cache(self):
        if self._root_cache is not None:
            return

        if rank['world'] != root['world']:
            self._root_cache = None
            return

        mesh_path = self.mesh_src.fname

        with h5py.File(mesh_path, "r") as meshf:
            con, ecurved, edisps, cdisps = BasePartitioner.construct_global_con(meshf)

            nelems_file = int(len(ecurved))
            nelems_state = int(self.i.nelems_g)
            if nelems_file != nelems_state:
                raise RuntimeError(f"[globcache] nelems mismatch file={nelems_file} state={nelems_state}")

            # Equal etype weights
            elewts = {et: 1 for et in edisps.keys()}

            _elewts = np.array([elewts[etype] for etype in edisps], dtype=np.int64)
            _edisps = np.array(list(edisps.values()), dtype=np.int64)

            def elewts_fn(e):
                return _elewts[np.searchsorted(_edisps, e, side='right') - 1]

            # Periodic grouping (offline routine)
            pmcon, exwts, pmerge = BasePartitioner._group_periodic_eles(meshf, con, cdisps, elewts_fn)

        # ---- Online-only “fix” for BasePartitioner periodic merge side-effects ----
        pmcon = self._canonicalize_con(pmcon, label="pmcon", nelems_g=nelems_state)
        # ------------------------------------------------------------------------

        graph, vemap = BasePartitioner._construct_graph(pmcon, elewts_fn, exwts=exwts)
        graph = self._sanitize_graph(graph)

        print(f"[globcache] built backend={self._backend_name} nverts={graph.vwts.shape[0]} nnz={graph.etab.size} "
              f"merged={len(pmerge)} nelems_g={nelems_state}", flush=True)

        self._root_cache = (graph, vemap, pmerge, edisps, nelems_state)

    def partition(self, partwts, *, ufactor = None, seed = None):
        self._build_root_cache()
        nelems_g = int(self.i.nelems_g)

        if rank['world'] == root['world']:
            graph, vemap, pmerge, edisps, nelems_state = self._root_cache
            assert nelems_state == nelems_g

            partwts = np.asarray(partwts, dtype=np.int64)

            # Allow “rank removal” (zero target) by partitioning active subset only
            active = np.flatnonzero(partwts > 0).astype(int).tolist()
            if not active:
                raise ValueError("all partwts are zero")

            # METIS + SCOTCH both accept part weights → keep them
            partwts_active = partwts[active].tolist()

            elewts = {et: 1 for et in edisps.keys()}
            opts = self._backend_opts(ufactor=ufactor, seed=seed)

            if rank['world'] == root['world']:
                print(f"[globpart.{self._backend_name}] opts={opts}", flush=True)

            part = self._backend_cls(partwts_active, elewts=elewts, opts=opts)

            print(f"[globpart.{self._backend_name}] begin nverts={graph.vwts.shape[0]} nnz={graph.etab.size} "
                  f"nparts_active={len(active)} ufactor={ufactor} seed={seed}", flush=True)

            vparts_merged = part._partition_graph(graph, partwts_active).astype(np.int32, copy=False)

            # Sanity: partition ids must be within active part range
            if vparts_merged.size:
                pmin = int(vparts_merged.min())
                pmax = int(vparts_merged.max())
                if pmin < 0 or pmax >= len(active):
                    raise RuntimeError(f"[globpart.{self._backend_name}] invalid partition ids: "
                                       f"min={pmin} max={pmax} nparts_active={len(active)}")

            vparts = BasePartitioner._ungroup_periodic_eles(pmerge, vemap, vparts_merged)

            # Map active-part ids → world ranks
            parts_g = np.asarray([active[p] for p in vparts], dtype=np.int32)

            print(f"[globpart.{self._backend_name}] end", flush=True)
        else:
            parts_g = np.empty(nelems_g, dtype=np.int32)

        comm['world'].Bcast(parts_g, root=root['world'])
        return parts_g

    def _retag_con_owners_from_vparts(self, vparts):
        vparts = np.asarray(vparts, dtype=np.int32)

        for et in self.i.etypes:
            con_idx = self.i.con_idx[et]
            con_mpi = self.i.con_mpi[et]
            if con_idx.size == 0 or con_mpi.size == 0:
                continue

            eid = con_idx.reshape(-1)
            nbr = con_mpi.reshape(-1)

            m = (eid >= 0) & (eid < vparts.size)
            if np.any(m):
                nbr[m] = vparts[eid[m]]

    def apply_global_partition(self, vparts, *, move_spts_nodes=True):
        vparts = np.asarray(vparts, dtype=np.int32)

        if vparts.ndim != 1:
            raise ValueError("vparts must be 1D")

        nelems_g = int(self.i.nelems_g)
        if vparts.size != nelems_g:
            raise ValueError(f"vparts has size {vparts.size}, expected {nelems_g}")

        if vparts.size and (vparts.min() < 0 or vparts.max() >= comm['world'].size):
            raise ValueError("vparts contains invalid partition ids")

        me = int(rank['world'])

        # Misplacements (diagnostic)
        mine = self.i.eidxs_flat
        mis_local = int(np.count_nonzero(vparts[mine] != me))
        mis_all = comm['world'].allgather(mis_local)
        if rank['world'] == root['world']:
            print(f"[apply.pre] mis_per_rank={mis_all} total_mis={sum(mis_all)}", flush=True)

        # Counts BEFORE
        nloc0 = int(self.i.nelems_total)
        all0 = comm['world'].allgather(nloc0)
        if rank['world'] == root['world']:
            print(f"[apply.pre] nelems_per_rank={all0} sum={sum(all0)}", flush=True)

        # Build destination eidxs for this world rank
        eidxs_dest = {}
        for et in self.i.etypes:
            s = int(self.i.edisps[et])
            n = int(self.i.ecnts_g[et])
            blk = vparts[s:s + n]
            gids = np.flatnonzero(blk == me).astype(np.int64, copy=False)
            eidxs_dest[et] = gids

        # Interconnector (use State etypes; safer than self.etypes)
        fast_conn = _MetaMeshInterconnector(
            etypes=self.i.etypes,
            eidxs_src=self.i.eidxs,
            eidxs_dest=eidxs_dest,
            eidxs_flat_src=self.i.eidxs_flat,
            etype_slices_src=self.i.etype_slices,
        )

        self.j = PartitionState(
            eidxs=eidxs_dest,
            con_mpi=fast_conn.relocate_cons(self.i.con_mpi),
            con_idx=fast_conn.relocate_cons(self.i.con_idx),
            spts_nodes=(fast_conn.relocate_cons(self.i.spts_nodes)
                        if move_spts_nodes else self.i.spts_nodes)
        )

        # Swap current/next and retag owners
        self.i, self.j = self.j, self.i
        self._retag_con_owners_from_vparts(vparts)

        if not move_spts_nodes:
            self._spts_nodes_valid = False

        # Counts AFTER
        nloc1 = int(self.i.nelems_total)
        all1 = comm['world'].allgather(nloc1)
        if rank['world'] == root['world']:
            print(f"[apply.post] nelems_per_rank={all1} sum={sum(all1)}", flush=True)

        return eidxs_dest

    def intg_repartition(self, target):
        parts_g = self.partition(target)
        self.apply_global_partition(parts_g)

class OnlineMETISPartitioner(OnlineGlobalPartitioner):
    name = "metis"
    _backend_cls = METISPartitioner
    _backend_name = "metis"


class OnlineSCOTCHPartitioner(OnlineGlobalPartitioner):
    name = "scotch"
    _backend_cls = SCOTCHPartitioner
    _backend_name = "scotch"
    _backend_strat = "quality"
