from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Optional, List
from collections import defaultdict

from pyfr.partitioners.base import BasePartitioner
from pyfr.partitioners.scotch import SCOTCHPartitioner

import re
from copy import deepcopy

from typing import Dict, Optional, List

import h5py
import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import (AlltoallMixin, Scatterer, SparseScatterer, 
                          autofree, get_comm_info,
                          mpi, comm, rank, root, rankmap)
from pyfr.nputil import iter_struct

from pyfr.util import subclass_where
from pyfr.readers.gmsh import GmshReader
from pyfr.shapes import BaseShape

from tabulate import tabulate

import os


from collections import namedtuple
Graph = namedtuple("Graph", ["vtab", "etab", "vwts", "ewts"])

def _append_csv_row(file_path: str, header_cols: list[str], values: list[int]):
    # TODO: Connect with pyfr.writers.csv.py

    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            f.write(','.join(header_cols) + '\n')

    # Append row
    with open(file_path, 'a', newline='') as f:
        f.write(','.join(str(int(v)) for v in values) + '\n')

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

@dataclass
class State:
    eidxs:      Dict[str, np.ndarray]
    con_idx:    Dict[str, np.ndarray]
    con_mpi:    Dict[str, np.ndarray]
    spts_nodes: Dict[str, np.ndarray]

    eidxs_flat:   np.ndarray       = field(init=False)
    etype_slices: Dict[str, slice] = field(init=False)
    etypes:       List[str]        = field(init=False)
    edisps: Dict[str, int] = field(init=False)   # PyFR naming
    nelems_g: int          = field(init=False)   # global total (PyFR: disp end)

    ecnts_g: Dict[str, int] = field(init=False)


    def __post_init__(self):
        local_keys = set(self.eidxs or {})
        self.etypes = sorted(set().union(*comm['world'].allgather(local_keys)))

        self.eidxs      = self._preproc_attr_dict(self.eidxs,      lead_dim=None, dtype=np.int64, gather_shape=False, force_1d=True )
        self.con_idx    = self._preproc_attr_dict(self.con_idx,    lead_dim=0   , dtype=np.int64, gather_shape= True, force_1d=False)
        self.con_mpi    = self._preproc_attr_dict(self.con_mpi,    lead_dim=0   , dtype=np.int64, gather_shape= True, force_1d=False)
        self.spts_nodes = self._preproc_attr_dict(self.spts_nodes, lead_dim=0   , dtype=None    , gather_shape= True , force_1d=False)

        #self.edisps = State._compute_edisps(self.etypes, self.eidxs)
        # Global element counts per etype (PyFR-style)
        counts_loc = np.array([self.eidxs[et].size for et in self.etypes], dtype=np.int64)
        counts_g   = comm['world'].allreduce(counts_loc, op=mpi.SUM)

        self.ecnts_g = {et: int(n) for et, n in zip(self.etypes, counts_g.tolist())}

        # PyFR-style displacements (etype blocks in vparts)
        disp = 0
        self.edisps = {}
        for et in self.etypes:
            self.edisps[et] = disp
            disp += self.ecnts_g[et]

        self.nelems_g = int(disp)
        self.eidxs_flat, self.etype_slices = State._eidxs_to_flat(self.etypes, self.eidxs, self.edisps)

    def gnum(self, et: str, gids: np.ndarray) -> np.ndarray:
        # ASSUMPTION (your stated invariant): gids are dense 0..ng-1 per etype
        return self.edisps[et] + np.asarray(gids, dtype=np.int64)

    def fidx_of(self, etype: str, lids: np.ndarray) -> np.ndarray:
        # “flat index” into local concatenation
        return self.etype_slices[etype].start + np.asarray(lids, dtype=np.int64)

    def eid_of(self, etype: str, lids: np.ndarray) -> np.ndarray:
        # “global element number” (PyFR partitioner vertex id)
        lids = np.asarray(lids, dtype=np.int64)
        return self.edisps[etype] + self.eidxs[etype][lids]


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

    @property
    def nelems_etype(self) -> Dict[str, int]:
        return {et: self.eidxs.get(et, np.empty(0)).size for et in self.etypes}

    @staticmethod
    def _clone_no_mpi(src: "State") -> "State":
        new = State.__new__(State)  # bypass __init__/__post_init__

        # Deep-copy the data dicts (logic preserved: independent arrays)
        new.eidxs      = {et: arr.copy() for et, arr in src.eidxs.items()}
        new.con_mpi    = {et: arr.copy() for et, arr in src.con_mpi.items()}
        new.con_idx    = {et: arr.copy() for et, arr in src.con_idx.items()}
        new.spts_nodes = {et: arr.copy() for et, arr in src.spts_nodes.items()}

        new.eidxs_flat   = src.eidxs_flat.copy()
        new.etype_slices = dict(src.etype_slices)
        new.etypes       = list(src.etypes)


        new.edisps   = dict(src.edisps)
        new.nelems_g = int(src.nelems_g)
        new.ecnts_g  = dict(src.ecnts_g)


        return new

    def clone(self) -> "State":
        return State._clone_no_mpi(self)

    def relocate_to(self, eidxs_dest: Dict[str, np.ndarray], 
                    move_spts_nodes = True):
        inter = _MeshInterconnector(self.eidxs, eidxs_dest)
        return State(eidxs=eidxs_dest,
                        con_mpi=inter.relocate(self.con_mpi, edim=0),
                        con_idx=inter.relocate(self.con_idx, edim=0),
                     spts_nodes=inter.relocate(self.spts_nodes, edim=0) if move_spts_nodes else self.spts_nodes)

    @staticmethod
    def _mpi_sorted_union(local):
        return sorted(set().union(*comm['world'].allgather(set(local))))

    @staticmethod
    def _flat_to_eidxs(eidxs_flat, etype_slices):
        eidxs_flat = np.asarray(eidxs_flat, dtype=np.int64)
        return {et: eidxs_flat[sl].copy() for et, sl in etype_slices.items()}

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


    @staticmethod
    def _edges_to_csr(nelems: int, edges_undirected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert undirected edges (m,2) into CSR adjacency (vtab, etab),
        using vertex-id == eid (assumed dense 0..nelems-1).
        """
        if edges_undirected.size == 0:
            vtab = np.zeros(nelems + 1, dtype=np.int64)
            etab = np.empty(0, dtype=np.int64)
            return vtab, etab

        e = np.asarray(edges_undirected, dtype=np.int64)
        # directed adjacency
        e = np.vstack((e, e[:, ::-1]))

        # sort by lhs
        order = np.argsort(e[:, 0], kind="mergesort")
        lhs = e[order, 0]
        rhs = e[order, 1]

        # counts per vertex
        counts = np.bincount(lhs, minlength=nelems)
        vtab = np.empty(nelems + 1, dtype=np.int64)
        vtab[0] = 0
        np.cumsum(counts, out=vtab[1:])

        etab = rhs.astype(np.int64, copy=False)
        return vtab, etab


    def build_root_scotch_graph(
        self,
        *,
        vwts: np.ndarray | None = None,
        ewts: np.ndarray | None = None,
    ) -> Graph | None:
        """
        Gather global adjacency to root and build a Graph(vtab, etab, vwts, ewts).
        Non-root ranks return None.
        """
        W = comm["world"]
        r = int(rank["world"])
        rt = int(root["world"])

        edges_loc = self._local_eid_edges()
        edges_all = W.gather(edges_loc, root=rt)

        if r != rt:
            return None

        edges = np.concatenate(edges_all, axis=0) if edges_all else np.empty((0, 2), np.int64)

        # Unique undirected edges
        if edges.size:
            edges.sort(axis=1)  # ensure (min,max) per row
            edges = edges[np.lexsort(edges.T)]
            # unique rows
            keep = np.ones(edges.shape[0], dtype=bool)
            keep[1:] = np.any(edges[1:] != edges[:-1], axis=1)
            edges = edges[keep]

        nelems = int(W.size * 0)  # placeholder to avoid unused warnings
        nelems = int(W.allreduce(self.nelems_total, op=mpi.SUM))

        print(f"[scotch.graph] root={rt} nelems={nelems} undirected_edges={edges.shape[0]}", flush=True)

        vtab, etab = State._edges_to_csr(nelems, edges)

        # default weights: unit vertex weights (single constraint), unit edge weights
        if vwts is None:
            vwts = np.ones((nelems, 1), dtype=np.int64)
        else:
            vwts = np.asarray(vwts, dtype=np.int64).reshape(nelems, -1)

        if ewts is None:
            ewts = np.ones(etab.shape[0], dtype=np.int64)
        else:
            ewts = np.asarray(ewts, dtype=np.int64)

        return Graph(vtab=vtab, etab=etab, vwts=vwts, ewts=ewts)

    def nelems_total_global(self) -> int:
        return int(self.nelems_g)

    @staticmethod
    def edges_to_csr(nelems: int, edges_undirected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert undirected edges (m,2) into CSR adjacency (vtab, etab).
        Vertex id is eid in [0, nelems).
        """
        if edges_undirected.size == 0:
            vtab = np.zeros(nelems + 1, dtype=np.int64)
            etab = np.empty(0, dtype=np.int64)
            return vtab, etab

        e = np.asarray(edges_undirected, dtype=np.int64)
        e = np.vstack((e, e[:, ::-1]))  # directed

        order = np.argsort(e[:, 0], kind="mergesort")
        lhs = e[order, 0]
        rhs = e[order, 1]

        counts = np.bincount(lhs, minlength=nelems)
        vtab = np.empty(nelems + 1, dtype=np.int64)
        vtab[0] = 0
        np.cumsum(counts, out=vtab[1:])

        etab = rhs.astype(np.int64, copy=False)
        return vtab, etab

    def pack_local_con(self) -> np.ndarray:
        """
        Return local connectivity as pairs of *global element numbers*.

        Output: int64 array of shape (nedges_local, 2) with rows [u, v]
        where u is a local-owned element (global id) and v is its neighbour
        element (global id). Boundary faces must be encoded as -1 in con_idx.
        """
        edges = []

        # Global element count (for cheap sanity checks)
        nloc = int(self.nelems_total)
        nglb = comm['world'].allreduce(nloc, op=mpi.SUM)

        for et in self.etypes:
            sl = self.etype_slices[et]
            ne = sl.stop - sl.start
            if ne == 0:
                continue

            # Global IDs for the elements *owned by this rank* of this etype,
            # in the same row-order used by con_idx[et].
            u = self.eidxs_flat[sl]                      # (ne,)

            # Neighbour IDs per face (must already be global element numbers)
            nbr = np.asarray(self.con_idx[et], dtype=np.int64)

            # Robustify shape: treat trailing dims as "faces"
            if nbr.ndim == 1:
                nbr = nbr.reshape(ne, -1)
            elif nbr.ndim >= 2:
                nbr = nbr.reshape(ne, -1)

            if nbr.shape[0] != ne:
                raise ValueError(
                    f"con_idx[{et}] has wrong element axis: got {nbr.shape[0]} expected {ne}"
                )

            m = (nbr >= 0)
            if not np.any(m):
                continue

            uu = np.broadcast_to(u[:, None], nbr.shape)[m]
            vv = nbr[m]

            # Sanity: neighbours must be in [0, nglb)
            if vv.size:
                vmax = int(vv.max())
                if vmax >= nglb:
                    raise ValueError(
                        f"con_idx[{et}] appears NOT to store global element numbers "
                        f"(max nbr={vmax} >= nglb={nglb}). Fix con_idx encoding first."
                    )

            edges.append(np.column_stack([uu, vv]))

        if not edges:
            return np.empty((0, 2), dtype=np.int64)

        return np.vstack(edges).astype(np.int64, copy=False)


class _MetaMesh:

    # default cost scales for wait-split → targets
    lb_cost_scale_g1r  = 1.0
    lb_cost_scale_g1s  = 1.0
    lb_cost_scale_g1rt = 1.0

    def __init__(self, *, mesh_src: _Mesh, 
                 etypes, e2i, bc2id, i: State, j: State):
        self.mesh_src  = mesh_src
        self.mesh_dest = None
        self.etypes    = list(etypes)
        self.e2i       = dict(e2i)
        self.bc2id     = dict(bc2id)
        self.i         = i
        self.j         = j
        self._cache = {}
        self._ver = {'topology': 0}

        self._spts_valid = True

        self.exec_order = None

    def _invalidate_topology(self) -> None:
        """Bump topology epoch and drop caches that depend on eidx ownership."""
        self._ver['topology'] += 1

        # TODO: more fine-grained invalidation later; for now just wipe
        self._cache.pop('owners_map', None)
        self._cache.pop('nei_gid_sets', None)

        self._cache.pop('owner_global', None)
        self._cache.pop('con_p', None)


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

    @classmethod
    def from_mesh(cls, mesh: _Mesh) -> "_MetaMesh":

        etypes   = State._mpi_sorted_union(set(mesh.etypes or ()))
        bc_names = State._mpi_sorted_union(set((mesh.bcon or {}).keys()))
        e2i      = {et: i for i, et in enumerate(etypes)}
        bc2id    = {n: i for i, n in enumerate(bc_names)}

        eidxs = {et: np.asarray(mesh.eidxs.get(et, ()), dtype=np.int64)
                for et in etypes}
        con_mpi = {et: np.full((eidxs[et].size, cls._nfaces(et)), -1, np.int64) for et in etypes}
        con_idx = {et: np.full((eidxs[et].size, cls._nfaces(et)), -1, np.int64) for et in etypes}

        spts_nodes = deepcopy(mesh.spts_nodes)
        # After self.i is fully built:

        state_i = State(eidxs=deepcopy(eidxs), con_mpi=deepcopy(con_mpi), 
                        con_idx=deepcopy(con_idx), spts_nodes=deepcopy(spts_nodes),)

        mm = cls(mesh_src=mesh, etypes=etypes, e2i=e2i, bc2id=bc2id,
            i=state_i, j=state_i.clone(),)

        mm._encode_con(mesh)
        mm._fill_con_mpi(mesh)

        mm._ne_i = mm._local_count

        return mm

    # Switch from etype and local indexing within that etype to the global flat indexing
    def etype_lid_to_eid(self, etype: str, lids) -> np.ndarray:
        lids = np.asarray(lids, dtype=np.int64)
        # eid = edisps[etype] + gid
        return self.i.edisps[etype] + np.asarray(self.i.eidxs[etype], dtype=np.int64)[lids]

    # Back-compat while you transition call sites
    etype_local_to_flat = etype_lid_to_eid

    def _encode_con(self, mesh):
        """
        Encode the *local* (same-rank) mesh connectivity and boundary faces as:

            con_mpi[et][lid, f, 0] = neighbour owner rank (world index), or
                                    -1 for boundary faces
            con_idx[et][lid, f, 0] = neighbour global element ID (>= 0), or
                                    eid_bc = -(bc_id + 1) for boundaries

        Global element IDs are the GIDs carried in State.eidxs_flat via
        State._eidxs_to_flat.
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
        print(f"[con_mpi] rank={my_rank} nfaces_mpi={nfaces_mpi}")

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

        self.j = State(eidxs=eidxs_dest,
                           con_mpi=fast_conn.relocate_cons(self.i.con_mpi),
                           con_idx=fast_conn.relocate_cons(self.i.con_idx),
                       spts_nodes=(fast_conn.relocate_cons(self.i.spts_nodes)
                          if move_spts_nodes else self.i.spts_nodes))

        self._accept_j_into_i()
        self._ne_i = self._local_count
        if not move_spts_nodes:
            self._spts_valid = False

    # ---------------------
    # Relocation iterations 
    # ---------------------

    def _reset_j_with_i(self):
        self.j = self.i.clone()

    def _accept_j_into_i(self, *, vparts: np.ndarray | None = None) -> None:
        # Swap current/next
        self.i, self.j = self.j, self.i

        # Retag con_mpi (neighbor rank) using the cheapest available truth
        if vparts is None:
            self._retag_con_owners()                 # allgather-based, general
        else:
            self._retag_con_owners_from_vparts(vparts)  # O(local faces), SCOTCH path

        # One place for topology invalidation
        self._invalidate_topology()

    def _build_eidxs_diff_from_flat(self, chosen_flat: np.ndarray,
                                          chosen_nbrs: np.ndarray,
                                    state: "State") -> dict[int, dict[str, np.ndarray]]:
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

    def _compute_deltas(self, mode):
        if   mode == 'faces':    return self._calc_mpi_faces_deltas()
        elif mode == 'vertices': return self._calc_mpi_vertices_deltas()
        else: raise ValueError(f"Unknown {mode = }' ≠ edges/vertices")

    def _calc_mpi_faces_deltas(self) -> dict[int, dict[str, np.ndarray]]:
        """
        Compute per-neighbour per-etype (lid, delta) matrices using the
        owner column stored in con_mpi and neighbour eid in con_idx.

        For each element e and neighbour rank n:
            c_me(e) = # faces of e whose neighbour owner == my_rank
            c_n(e)  = # faces of e whose neighbour owner == n
            delta   = c_me(e) - c_n(e)

        For a given neighbour n and etype et we return all elements with
        c_n(e) > 0 as an int64 array of shape (N, 2):
            [ [lid_0, delta_0],
            [lid_1, delta_1],
            ...
            ]
        sorted by (delta, lid).
        """
        my_rank = int(rank['world'])
        empty = np.empty((0, 2), dtype=np.int64)

        # meta per etype: (boundary_lids, c_me_bound, owners_eff_bound)
        per_et_meta: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}
        neighbor_ids: set[int] = set()

        for et in self.etypes:
            con_mpi_et = self.i.con_mpi[et]
            con_idx_et = self.i.con_idx[et]

            if con_mpi_et.size == 0 or con_idx_et.size == 0:
                per_et_meta[et] = None
                continue

            owners = np.asarray(con_mpi_et)
            eids  = np.asarray(con_idx_et)

            # Faces that connect to another element (drop BC/invalid faces)
            valid = (eids >= 0) & (owners >= 0)

            # Elements that see at least one MPI neighbour (owner != my_rank)
            mpi_face_mask = valid & (owners != my_rank)
            has_mpi_face  = np.any(mpi_face_mask, axis=1)

            if not np.any(has_mpi_face):
                per_et_meta[et] = None
                continue

            # Local element IDs that actually sit on MPI interfaces for this etype
            lids_bnd = np.nonzero(has_mpi_face)[0].astype(np.int64, copy=False)

            # Restrict owners/valid to just these boundary elements
            owners_bnd = owners[has_mpi_face]
            valid_bnd  = valid[has_mpi_face]

            # Treat non-element faces (BCs etc.) as -1 so they do not contribute
            owners_eff_bnd = np.where(valid_bnd, owners_bnd, -1)

            # Neighbours seen anywhere on these boundary elements
            uniq = np.unique(owners_eff_bnd)
            uniq = uniq[(uniq >= 0) & (uniq != my_rank)]
            for r in uniq:
                neighbor_ids.add(int(r))

            # Count faces attached to *this* rank for each boundary element
            c_me_bnd = np.sum(owners_eff_bnd == my_rank,
                            axis=1).astype(np.int16, copy=False)

            per_et_meta[et] = (lids_bnd, c_me_bnd, owners_eff_bnd)

        neighbor_set = sorted(neighbor_ids)
        if not neighbor_set:
            return {}

        result: dict[int, dict[str, np.ndarray]] = {}

        for nrank in neighbor_set:
            nrank = int(nrank)
            per_n: dict[str, np.ndarray] = {}

            for et in self.etypes:
                meta = per_et_meta[et]
                if meta is None:
                    per_n[et] = empty
                    continue

                lids_bnd, c_me_bnd, owners_eff_bnd = meta

                # Faces to this neighbour for boundary elements only
                c_n_bnd = np.sum(owners_eff_bnd == nrank,
                                axis=1).astype(np.int16, copy=False)

                sel = (c_n_bnd > 0)
                if not np.any(sel):
                    per_n[et] = empty
                    continue

                lids  = lids_bnd[sel]
                delta = (c_me_bnd[sel] - c_n_bnd[sel]).astype(np.int64, copy=False)

                mat = np.c_[lids, delta]
                # Keep the previous ordering: by delta, then by lid
                order = np.lexsort((mat[:, 0], mat[:, 1]))
                per_n[et] = mat[order]

            result[nrank] = per_n

        return result

    def _calc_mpi_vertices_deltas(self) -> dict[int, dict[str, np.ndarray]]:
        """
        Compute per-neighbour per-etype (lid, delta) matrices using MPI vertex
        connectivity.

        For each element e and neighbour rank n:

            c_int(e) = # vertices of e that are strictly interior
                       (not on ANY MPI interface with any neighbour)
            c_n(e)   = # vertices of e that participate in an MPI interface
                       with neighbour n
            delta    = c_int(e) - c_n(e)

        For a given neighbour n and etype et we return all elements with
        c_n(e) > 0 as an int64 array of shape (N, 2):

            [ [lid_0, delta_0],
              [lid_1, delta_1],
              ...
            ]

        sorted by (delta, lid), consistent with _calc_mpi_faces_deltas.
        """

        # 1) Per-neighbour sets of MPI vertex-node IDs on faces to that neighbour
        mvu = self.collect_mpi_vertex_nodes()  # {nbr: np[int64]}

        if not mvu:
            return {}

        # 2) Union of all MPI vertex-node IDs seen on any neighbour
        all_sets = [
            np.asarray(v, dtype=np.int64).ravel()
            for v in mvu.values()
            if v is not None
        ]
        union_all = (
            np.unique(np.concatenate(all_sets))
            if all_sets else np.empty(0, dtype=np.int64)
        )

        # 2b) Precompute per-etype vertex-column indices once
        #     (union of all per-face vertex columns)
        vcols_by_et: dict[str, np.ndarray] = {}
        for et in self.etypes:
            fidx = self._face_vertex_indices(et)  # list of 1D arrays of cols
            if not fidx:
                vcols_by_et[et] = np.empty(0, dtype=np.int64)
            else:
                vcols_by_et[et] = np.unique(
                    np.concatenate(fidx).astype(np.int64, copy=False)
                )

        def per_et(et: str, nds: np.ndarray, nbr_vertices: np.ndarray) -> np.ndarray:
            """
            Compute (lid, delta) for a single etype given:
              - et: etype name
              - nds: spts_nodes[et] of shape (Ne, Nspts)
              - nbr_vertices: 1D array of vertex IDs participating in an
                              MPI interface with the *current* neighbour.
            """
            if nds is None:
                return np.empty((0, 2), dtype=np.int64)

            nds = np.asarray(nds, dtype=np.int64)
            if nds.size == 0:
                return np.empty((0, 2), dtype=np.int64)

            vcols = vcols_by_et.get(et)
            if vcols is None or vcols.size == 0:
                # No vertex columns known for this etype; nothing to do.
                return np.empty((0, 2), dtype=np.int64)

            # Restrict to vertex columns ONLY
            nds_v = nds[:, vcols]       # shape (Ne, Nvert_per_el)

            # c_n(e): # vertices of element e that belong to this neighbour's
            #         MPI vertex set.
            v_in_n = np.isin(nds_v, nbr_vertices, assume_unique=False)
            c_n = v_in_n.sum(axis=1).astype(np.int16, copy=False)

            # c_int(e): # vertices of e that are STRICTLY interior
            #           (do not appear in union_all of all MPI vertices).
            if union_all.size:
                v_is_mpi_any = np.isin(nds_v, union_all, assume_unique=False)
                v_is_int = ~v_is_mpi_any
            else:
                # No MPI vertices at all => everything is interior
                v_is_int = np.ones_like(nds_v, dtype=bool)

            c_int = v_is_int.sum(axis=1).astype(np.int16, copy=False)

            # Only elements that actually touch this neighbour by at least one
            # MPI vertex are interesting.
            sel = (c_n > 0)
            if not np.any(sel):
                return np.empty((0, 2), dtype=np.int64)

            lids = np.nonzero(sel)[0].astype(np.int64, copy=False)

            # Delta semantics consistent with faces:
            #   delta = c_int - c_n
            # Negative delta => more tied to neighbour than to interior.
            delta = (
                c_n[sel].astype(np.int32) - c_int[sel].astype(np.int32)
            ).astype(np.int64, copy=False)

            mat = np.c_[lids, delta]
            # Sort by (delta, lid) to be deterministic and compatible with faces
            order = np.lexsort((mat[:, 0], mat[:, 1]))
            return mat[order]

        # 3) Build {nbr: {etype: (N,2)}} result
        out: dict[int, dict[str, np.ndarray]] = {}
        for nrank, verts in mvu.items():
            nbr_vertices = np.asarray(verts, dtype=np.int64).ravel()
            per: dict[str, np.ndarray] = {}
            for et in self.etypes:
                nds = self.i.spts_nodes.get(et, None)
                per[et] = per_et(et, nds, nbr_vertices)
            out[int(nrank)] = per

        return out

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

        raise NotImplementedError(f"_face_vertex_indices: etype '{et}' not implemented")

    def collect_mpi_vertex_nodes(self) -> dict[int, np.ndarray]:
        """
        Return {nbr_rank: np.ndarray[int64]} = sorted-unique global vertex-node
        IDs on MPI faces to that neighbor.

        This mirrors the old implementation which used _face_vertex_indices
        and per-etype spts_nodes, but now wired to self.i.spts_nodes.
        """
        if not getattr(self, "_spts_valid", True):
            raise RuntimeError("spts_nodes are stale")

        if self.i.spts_nodes is None:
            raise RuntimeError("self.i.spts_nodes missing")

        # 1) which faces are MPI, grouped by neighbor
        per_nbr_faces = self._mpi_faces_by_neighbor()

        # 2) per-etype face -> vertex-column indices, as in the old code
        face_vtx_by_et: dict[str, list[np.ndarray]] = {
            et: self._face_vertex_indices(et) for et in self.etypes
        }

        out: dict[int, np.ndarray] = {}

        # 3) union vertex nodes per neighbor
        for nbr, faces in per_nbr_faces.items():
            verts_all: list[np.ndarray] = []

            for et, lid, f in faces:
                nds = self.i.spts_nodes.get(et)
                if nds is None or nds.size == 0:
                    continue
                if not (0 <= lid < nds.shape[0]):
                    continue

                # face vertex columns for this etype / face index
                fv = face_vtx_by_et[et][int(f)]
                verts = nds[int(lid), fv]

                if verts.size:
                    verts_all.append(verts.reshape(-1))

            if verts_all:
                vcat = np.concatenate(verts_all).astype(np.int64, copy=False)
                vcat = vcat[vcat >= 0]
                out[int(nbr)] = np.unique(vcat)
            else:
                out[int(nbr)] = np.empty(0, dtype=np.int64)

        return out

    def _mpi_faces_by_neighbor(self) -> dict[int, list[tuple[str, int, int]]]:
        """
        Return {nbr_rank: [(etype, lid, fidx), ...]} for MPI faces only.

        A face is considered MPI if:
            - neighbour owner >= 0,
            - neighbour owner != my_rank,
            - neighbour eid >= 0 (element neighbour, not BC/invalid).
        """
        per_nbr: dict[int, list[tuple[str, int, int]]] = {}

        for et in self.etypes:
            owners = self.i.con_mpi[et]
            eids  = self.i.con_idx[et]

            if owners.size == 0 or eids.size == 0: continue

            # MPI faces: element neighbour, owner >= 0, owner != my_rank
            #            and neighbour eid >= 0 (i.e. real element, not BC).
            mpi_mask = (eids >= 0) & (owners >= 0) & (owners != rank['world'])
            if not np.any(mpi_mask):
                continue

            lids, fidxs = np.nonzero(mpi_mask)

            nbrs = owners[lids, fidxs].astype(np.int64, copy=False)

            # Group by neighbour, keep the same (etype, lid, fidx) payload
            for nbr in np.unique(nbrs):
                if nbr == rank['world']:
                    continue

                sel = (nbrs == nbr)
                lids_n  = lids[sel]
                fidxs_n = fidxs[sel]

                lst = per_nbr.setdefault(nbr, [])
                lst.extend((et, lid_i, f_i) for lid_i, f_i in zip(lids_n, fidxs_n) )

        return per_nbr

    # -----------------
    # Testing / ideas
    # -----------------
    @property
    def twoway_mask(self) -> np.ndarray:
        """
        Boolean R×R adjacency matrix: entry (i, j) is True iff there is at
        least one MPI face between rank i and j, based purely on con_mpi.
        """
        R    = comm['world'].size
        r    = comm['world'].rank

        # Local 0/1 adjacency
        M_loc = np.zeros((R, R), dtype=np.uint8)

        # Use con_mpi to find neighbors
        # This helper already walks self.i.con_mpi[et] and owners:
        for nbr, faces in self._mpi_faces_by_neighbor().items():
            j = int(nbr)
            if j == r:
                continue
            # Symmetric adjacency at the *local* level
            M_loc[r, j] = 1
            M_loc[j, r] = 1

        # Parallel OR of adjacency
        M_glob = comm['world'].allreduce(M_loc, op=mpi.SUM)

        # Convert to boolean adjacency
        M_glob = M_glob > 0
        np.fill_diagonal(M_glob, False)

        return M_glob

    def element_flow_plan(self, tgt_counts, *, overshoot: float = 0.0,
                          max_iters=4) -> np.ndarray:
        cur_full = np.asarray(self._cur_counts_total, dtype=np.int64)

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

        mask_arr   = np.array(self.twoway_mask, dtype=bool, copy=True)
        mask_faces = (F_full > 0)          # authoritative: real MPI edges
        R_full     = F_full.shape[0]

        if mask_arr.shape == F_full.shape:
            # Use twoway_mask, but never forbid a true MPI edge
            mask_full = mask_arr | mask_faces
        else:
            # Treat smaller mask as being defined over ranks [0 .. R_mask-1]
            R_mask = mask_arr.shape[0]
            if mask_arr.shape != (R_mask, R_mask):
                raise ValueError(f"mask must be square; got {mask_arr.shape}")

            # Embed into world-space mask
            mask_full = np.zeros_like(F_full, dtype=bool)
            mask_full[:R_mask, :R_mask] = mask_arr

            # For ranks not covered by the mask (e.g. newly added ranks),
            # at least allow traffic wherever there is an MPI face.
            if R_mask < R_full:
                mask_full[R_mask:, :] |= mask_faces[R_mask:, :]
                mask_full[:, R_mask:] |= mask_faces[:, R_mask:]

            # And in all cases never forbid a true MPI face
            mask_full |= mask_faces

        np.fill_diagonal(mask_full, False)

        # --- DEBUG: world-space inputs and mask ---
        #if int(rank['world']) == 0:
        #    print("[elec_plan.v2] cur_full =", cur_full.tolist(), flush=True)
        #    print("[elec_plan.v2] tgt_full =", tgt_full.tolist(), flush=True)
        #    print("[elec_plan.v2] F_full (faces) =")
        #    # cast to int for cleaner printing
        #    print(F_full.astype(np.int64), flush=True)
        #    print("[elec_plan.v2] mask_full (bool as 0/1) =")
        #    print(mask_full.astype(np.int64), flush=True)



        # Active ranks: those that have or will have elements
        active = [
            i for i, (c, t) in enumerate(zip(cur_full, tgt_full))
            if not (c == 0 and t == 0)
        ]

        if not active:
            # No movable elements anywhere: nothing to do.
            M_full = np.zeros((R_full, R_full), dtype=np.int64)
            return M_full

        np.fill_diagonal(mask_full, False)

        # ---------- restrict to active subgraph ----------

        cur = cur_full[active]
        tgt = tgt_full[active]

        F     = F_full[np.ix_(active, active)]
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

        for _ in range(int(max_iters)):
            diff    = cur - tgt
            deficit = np.maximum(0, -diff).astype(np.int64)
            excess  = np.maximum(0,  diff).astype(np.int64)

            if excess.sum() == 0 or deficit.sum() == 0:
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

        M_full = _cancel_anti_parallel(M_full)

        # Optional overshoot in *flow space*
        if overshoot != 0.0:
            factor = 1.0 + float(overshoot)
            if factor <= 0.0:
                raise ValueError(f"element_flow_plan: overshoot={overshoot!r} gives factor <= 0")
            M_full = np.rint(M_full.astype(np.float64) * factor).astype(np.int64)
            M_full[M_full < 0] = 0


        #if int(rank['world']) == 0:
        #    print("[elec_plan.v2] active indices =", active, flush=True)
        #    print("[elec_plan.v2] F_active =")
        #    print(F, flush=True)
        #    print("[elec_plan.v2] mask_active (0/1) =")
        #    print(maskb.astype(np.int64), flush=True)
        #    print("[elec_plan.v2] M_full (final flow plan) =")
        #    print(M_full, flush=True)


        return M_full

    def diffuse(
        self,
        *,
        mode: str = "faces",
        interface_policy: str = "per-element",
        threshold: int = 0,
        flow_matrix: np.ndarray | None = None,
        restrict_etypes = None,
        skip_last: bool = False,
        restrict_src_dest: bool = False,
        target_counts = None,
        move_spts_nodes: bool = False,
    ) -> tuple[int, np.ndarray | None]:
        """
        Perform a *single* diffusion / smoothing sweep and apply it.

        This is intended to be the core primitive; outer drivers (iterate(),
        smooth_until_stagnates(), rank-removal logic, etc.) decide:
          - how to build / update a flow_matrix between sweeps,
          - when to stop iterating,
          - how to handle ping-pong across interfaces.

        Parameters
        ----------
        mode : {'faces', 'vertices'}
            Which imbalance metric to use. This is passed directly to
            self._compute_deltas(mode).
        interface_policy : {'per-element', 'all-or-none'}
            - 'per-element': classic face-based diffusion. Each element may
              move to at most one neighbour in this sweep. Arbitration is
              element-wise using a global min-delta rule across neighbours.
            - 'all-or-none': interface-based smoothing. For each (rank, nbr)
              interface we either move *all* candidates that pass the
              threshold, or none, subject to optional flow_matrix capacities.
        threshold : int
            Delta threshold; an element / candidate is eligible iff
                delta <= threshold
            (we use <= consistently everywhere).
        flow_matrix : (P, P) int array or None
            Residual interface capacities for *this* sweep. flow_matrix[p, q]
            is how many elements rank p is still allowed to send to rank q.
            If None, capacities are treated as infinite.
            NOTE: diffuse() does NOT mutate flow_matrix; instead it returns
            a "realisation" flow_used of the same shape, with the actual
            number of elements moved in this sweep. Outer code can then do
                M_rem -= flow_used
            or recompute a new flow_matrix as it prefers.
        restrict_etypes : sequence of etype names, optional
            If given, only these element types are considered for movement.
        skip_last : bool
            Device/protection semantics: if True, and multiple etypes exist,
            then on interfaces where self._compute_restrict_last(...) says so
            we drop the last etype from the active set (e.g. keep hex/pyr
            movable but protect tet, or vice versa).
        restrict_src_dest : bool
            If True and target_counts is not None and flow_matrix is None,
            apply simple donor/receiver gating:
                diff = cur - target
                donors are ranks with diff > 0
                receivers are ranks with diff < 0
            Donors may only send to receivers; receivers never send.
            (We purposely ignore this when a flow_matrix is supplied; in that
             regime, the flow_matrix is the single source of truth.)
        target_counts : (P,) sequence of ints, optional
            Desired per-rank element counts for donor/receiver gating.
        move_spts_nodes : bool
            Forwarded to _apply_plan_and_commit; if True we also relocate
            solution / spts nodes together with topology.

        Returns
        -------
        moved_local : int
            Number of elements moved *off this rank* in this sweep.
        flow_used : (P, P) int64 array or None
            If flow_matrix was provided, flow_used has the same shape and
            encodes how many elements were actually sent on each (p, q)
            interface in this sweep. Only row r = rank['world'] will be
            non-zero on this process; outer code can MPI-SUM this if it
            wants the global flow realisation. If flow_matrix is None,
            flow_used is None.
        """
        W    = comm['world']
        rnk  = int(rank['world'])
        P    = W.size
        thr_int = int(threshold)

        # Start candidate state for this sweep from current i -> j
        self._reset_j_with_i()
        state_i = self.i

        # ---- 1) Donor/receiver gating (optional) ------------------------
        allowed_nbrs: Optional[set[int]]
        I_am_donor = True

        if restrict_src_dest and target_counts is not None and flow_matrix is None:
            my_count = state_i.nelems_total
            cur = np.asarray(W.allgather(my_count), dtype=np.int64)
            tgt = np.asarray(target_counts, dtype=np.int64)
            diff = cur - tgt  # +surplus / -deficit

            I_am_donor = diff[rnk] > 0
            if I_am_donor:
                allowed_nbrs = {int(n) for n in range(P) if diff[int(n)] < 0}
            else:
                allowed_nbrs = set()
        else:
            allowed_nbrs = None

        # ---- 2) Compute deltas for requested mode -----------------------
        deltas_by_rank = self._compute_deltas(mode)  # {nbr:{et:[[lid,delta],...]}}

        # ---- 3) Determine active etypes per interface -------------------
        etypes_all = list(self._etype_order())  # canonical order on THIS rank

        if restrict_etypes is not None:
            allowed_set = set(restrict_etypes)
            etypes_all = [et for et in etypes_all if et in allowed_set]

        # Pre-compute "restricted last" flags as in your existing logic
        restrict_last = self._compute_restrict_last(skip_last, etypes_all)

        # ---- 4) Helper: pick candidates according to interface_policy ---
        chosen_flat_chunks: list[np.ndarray] = []
        chosen_nbr_chunks:  list[np.ndarray] = []

        # For all-or-none we must ensure an element is not picked twice.
        picked_mask: Optional[np.ndarray] = None
        if interface_policy == "all-or-none":
            picked_mask = np.zeros_like(state_i.eidxs_flat, dtype=bool)

        if interface_policy == "per-element":
            # Classic per-element arbitration (old diffuse_smoothing):
            all_flat: list[np.ndarray] = []
            all_dlt:  list[np.ndarray] = []
            all_nbrs: list[np.ndarray] = []

            for nbr, per_et in deltas_by_rank.items():
                nbr = int(nbr)
                if allowed_nbrs is not None and nbr not in allowed_nbrs:
                    continue

                for et, arr in per_et.items():
                    if arr is None or arr.size == 0:
                        continue

                    lids = arr[:, 0].astype(np.int64, copy=False)
                    dlt  = arr[:, 1].astype(np.int64, copy=False)
                    if lids.size == 0:
                        continue

                    # Threshold is applied as delta <= thr_int
                    mkeep = (dlt <= thr_int)
                    if not np.any(mkeep):
                        continue

                    lids_k = lids[mkeep]
                    dlt_k  = dlt[mkeep]
                    flat_k = state_i.lids_to_flat(et, lids_k)

                    all_flat.append(flat_k)
                    all_dlt.append(dlt_k)
                    all_nbrs.append(np.full(flat_k.shape, nbr, dtype=np.int64))

            if all_flat:
                lids_flat = np.concatenate(all_flat)
                dlt_all   = np.concatenate(all_dlt)
                nbrs_all  = np.concatenate(all_nbrs)

                # Element-wise arbitration: for each flat id pick best neighbour
                order = np.lexsort((dlt_all, lids_flat))
                lids_s = lids_flat[order]
                dlt_s  = dlt_all[order]
                nbrs_s = nbrs_all[order]

                _, first_idx = np.unique(lids_s, return_index=True)
                best_flat = lids_s[first_idx]
                best_dlt  = dlt_s[first_idx]
                best_nbrs = nbrs_s[first_idx]

                mkeep = (best_dlt <= thr_int)
                if np.any(mkeep):
                    chosen_flat = best_flat[mkeep]
                    chosen_nbrs = best_nbrs[mkeep]

                    chosen_flat_chunks.append(chosen_flat)
                    chosen_nbr_chunks.append(chosen_nbrs)

        elif interface_policy == "all-or-none":
            # Interface-based all-or-none smoothing (old edge/vertex logic):
            gids_flat = state_i.eidxs_flat  # for deterministic ordering only

            for nbr, per_et in deltas_by_rank.items():
                nbr = int(nbr)

                if allowed_nbrs is not None and nbr not in allowed_nbrs:
                    continue

                # Capacity for this interface, if flow_matrix is supplied.
                cap = None
                if flow_matrix is not None:
                    cap = int(flow_matrix[rnk, nbr])
                    if cap <= 0:
                        continue

                # Decide active etypes on THIS interface
                if skip_last and restrict_last.get(nbr, False) and len(etypes_all) > 0:
                    etypes_active = etypes_all[:-1]
                else:
                    etypes_active = etypes_all

                cand_flat: list[np.ndarray] = []
                cand_dlt:  list[np.ndarray] = []

                for et in etypes_active:
                    arr = per_et.get(et, None)
                    if arr is None or arr.size == 0:
                        continue

                    lids = arr[:, 0].astype(np.int64, copy=False)
                    dlt  = arr[:, 1].astype(np.int64, copy=False)

                    # Candidate must satisfy delta <= thr_int
                    mkeep = (dlt <= thr_int)
                    if not np.any(mkeep):
                        continue

                    lids_k = lids[mkeep]
                    dlt_k  = dlt[mkeep]
                    flat_k = state_i.lids_to_flat(et, lids_k)

                    # Drop elements already chosen for some other neighbour
                    if picked_mask is not None:
                        m_new = ~picked_mask[flat_k]
                        if not np.any(m_new):
                            continue
                        flat_k = flat_k[m_new]
                        dlt_k  = dlt_k[m_new]

                    if flat_k.size == 0:
                        continue

                    cand_flat.append(flat_k)
                    cand_dlt.append(dlt_k)

                if not cand_flat:
                    continue

                flat_all = np.concatenate(cand_flat)
                dlt_all  = np.concatenate(cand_dlt)
                n_cand   = int(flat_all.size)

                # All-or-none capacity check
                if cap is not None and n_cand > cap:
                    # Not enough budget on this interface: skip entirely.
                    continue

                # Optional deterministic ordering: sort by (delta, gid)
                order = np.lexsort((gids_flat[flat_all], dlt_all))
                flat_all = flat_all[order]

                # Mark these as chosen for this neighbour
                if picked_mask is not None:
                    picked_mask[flat_all] = True

                chosen_flat_chunks.append(flat_all)
                chosen_nbr_chunks.append(
                    np.full(flat_all.shape, nbr, dtype=np.int64)
                )
        else:
            raise ValueError(
                "diffuse: interface_policy must be 'per-element' or 'all-or-none', "
                f"got {interface_policy!r}"
            )

        # ---- 5) Build move plan and apply --------------------------------
        if not chosen_flat_chunks:
            # No movement this sweep
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
            flow_used = (
                np.zeros_like(flow_matrix, dtype=np.int64)
                if flow_matrix is not None else None
            )
            return 0, flow_used

        chosen_flat = np.concatenate(chosen_flat_chunks)
        chosen_nbrs = np.concatenate(chosen_nbr_chunks)

        eidxs_diff = self._build_eidxs_diff_from_flat(
            chosen_flat, chosen_nbrs, state_i
        )

        moved_local = 0
        flow_used: np.ndarray | None
        if flow_matrix is not None:
            flow_used = np.zeros_like(flow_matrix, dtype=np.int64)
        else:
            flow_used = None

        for nbr, per_et in eidxs_diff.items():
            n_moved = sum(len(g) for g in per_et.values())
            moved_local += n_moved
            if flow_used is not None and n_moved:
                flow_used[rnk, int(nbr)] += int(n_moved)

        # Apply the relocation (even if eidxs_diff is empty, for symmetry)
        self._apply_plan_and_commit(eidxs_diff, move_spts_nodes=move_spts_nodes)

        return int(moved_local), flow_used

    def _canonical_step(self, step: dict) -> dict:
        """
        Normalise one exec_order entry into a fully-populated config dict.

        We now *only* accept dict-style steps. The old tuple/list formats like
            ('faces', 2, 4)
        are no longer supported.
        """

        if not isinstance(step, dict):
            raise TypeError(
                f"exec_order entries must be dicts; got {type(step).__name__}: {step!r}"
            )

        # Base defaults – everything explicit so iterate/_run_step can rely on them.
        cfg = dict(
            kind=None,           # 'vertices-flow', 'faces-flow', 'faces-smooth', ...
            name=None,           # pretty-print name
            mode=None,           # 'faces' or 'vertices'
            iface="per-element", # 'per-element' or 'all-or-none'
            threshold=0,         # delta <= threshold is eligible
            use_flow=False,      # whether to call element_flow_plan()
            overshoot=0.0,       # only meaningful if use_flow=True
            max_sweeps=1,        # <=0 means "until no moves" for smoothing steps
            patience=0,          # for smoothing steps
            min_change=0,        # for smoothing steps
            restrict_src_dest=False,
            move_spts_nodes=False,
            restrict_etypes=None,
        )

        # User-specified values override defaults
        cfg.update(step)

        kind = cfg["kind"]
        if kind is None:
            raise ValueError(f"exec_order step has no 'kind': {step!r}")

        # Infer mode if missing from kind prefix
        if cfg["mode"] is None:
            if str(kind).startswith("faces-"):
                cfg["mode"] = "faces"
            elif str(kind).startswith("vertices-"):
                cfg["mode"] = "vertices"
            else:
                raise ValueError(
                    f"Cannot infer mode from kind={kind!r}; please set mode explicitly."
                )

        mode = cfg["mode"]
        if mode not in ("faces", "vertices"):
            raise ValueError(f"mode must be 'faces' or 'vertices', got {mode!r}")

        iface = cfg["iface"]
        if iface not in ("per-element", "all-or-none"):
            raise ValueError(
                f"iface must be 'per-element' or 'all-or-none', got {iface!r}"
            )

        # Flow vs non-flow constraints
        if cfg["use_flow"]:
            # Flow steps are *interface-based*; we only support all-or-none there.
            #if iface != "all-or-none":
            #    raise ValueError(
            #        f"use_flow=True requires iface='all-or-none', got iface={iface!r}"
            #    )
            if cfg["overshoot"] < 0.0:
                raise ValueError(f"overshoot must be >= 0, got {cfg['overshoot']}")
        else:
            # If we are *not* using a flow matrix, overshoot must be zero.
            if abs(cfg["overshoot"]) > 1e-14:
                raise ValueError(
                    f"overshoot={cfg['overshoot']} is only valid with use_flow=True"
                )

        # Vertex-based steps *must* keep spts_nodes in sync, otherwise later calls
        # to vertex modes will trip the "spts_nodes are stale" RuntimeError.
        if mode == "vertices" and not cfg["move_spts_nodes"]:
            cfg["move_spts_nodes"] = True

        # Normalise numerics
        cfg["threshold"]   = int(cfg["threshold"])
        cfg["max_sweeps"]  = int(cfg["max_sweeps"])
        cfg["patience"]    = int(cfg["patience"])
        cfg["min_change"]  = int(cfg["min_change"])
        cfg["overshoot"]   = float(cfg["overshoot"])

        return cfg

    def iterate(self, objective, target_counts):
        if objective == "to-target":
            # Canonicalise once per call; you can also cache this on self if desired.
            steps = [self._canonical_step(s) for s in self.exec_order]

            for step in steps:
                self._run_step(step, target_counts)

        elif objective == "to-remove-rank":
            # Keep your existing to-remove-rank logic for now;
            # we can later rewrite it on top of diffuse()/smooth_until_stagnates.
            kill_rank = [r for r, c in enumerate(target_counts) if c == 0][0]
            if rank['world'] == root['world']:
                print(f"{kill_rank = } ", flush=True)

            while True:
                cur0 = self._cur_counts_total
                M0 = self.element_flow_plan(target_counts)

                # Evacuate via vertex-based diffusion + smoothing
                # (this can also be moved to exec_order if you want)
                # vertices-flow
                self._run_step(
                    dict(
                        kind="vertices-flow",
                        name="to-remove-vertices",
                        mode="vertices",
                        iface="all-or-none",
                        threshold=0,
                        use_flow=True,
                        overshoot=0.0,
                        max_sweeps=1,
                        patience=0,
                        min_change=0,
                    ),
                    target_counts,
                )
                # faces-smooth
                self._run_step(
                    dict(
                        kind="faces-smooth",
                        name="to-remove-smooth",
                        mode="faces",
                        iface="per-element",
                        threshold=-1,
                        use_flow=False,
                        max_sweeps=50,
                        patience=1,
                        min_change=0,
                        restrict_src_dest=True,
                    ),
                    target_counts,
                )

                cur1 = self._cur_counts_total
                diff = [cur1[i] - cur0[i] for i in range(len(cur0))]

                if rank['world'] == root['world']:
                    print(f"[itc4.iter] to-remove-rank CURRENT={cur1} \t DIFF = {diff}", flush=True)

                if cur1[kill_rank] == 0:
                    if rank['world'] == root['world']:
                        print(f"to-remove-rank done; kill_rank={kill_rank} cur={cur1}", flush=True)
                    break

                if all(d == 0 for d in diff):
                    if rank['world'] == root['world']:
                        print(f"to-remove-rank stuck; kill_rank={kill_rank} cur={cur1}", flush=True)
                    break

        else:
            raise ValueError(f"Unknown objective '{objective}'")

    def _run_step(self, step: dict, target_counts: Optional[List[int]]) -> None:
        """
        Execute one canonical diffusion step described by 'step'.

        Distinguishes between:
        - flow-guided all-or-none steps (vertices/edges)
        - per-element smoothing-to-stagnation steps
        and logs flow, counts, and interface stats at the end.
        """
        W      = comm['world']
        rnk    = int(rank['world'])
        root_w = int(root['world'])
        P      = W.size

        kind      = step.get("kind")             # e.g. 'vertices-flow', 'faces-flow', 'faces-smooth'
        label     = step.get("name", kind)
        mode      = step.get("mode", "faces")    # 'faces' or 'vertices'
        iface     = step.get("iface", "all-or-none")
        thr       = int(step.get("threshold", 0))
        use_flow  = bool(step.get("use_flow", False))
        overshoot = float(step.get("overshoot", 0.0))
        max_sweeps = int(step.get("max_sweeps", 1))
        patience   = int(step.get("patience", 0))
        min_change = int(step.get("min_change", 0))
        skip_last  = bool(step.get("skip_last", False))
        restrict_src_dest = bool(step.get("restrict_src_dest", False))
        move_spts_nodes   = bool(step.get("move_spts_nodes", False))
        restrict_etypes   = step.get("restrict_etypes", None)

        if rnk == root_w:
            print(
                f"[iterate] step={label} mode={mode} iface={iface} "
                f"thr={thr} use_flow={use_flow} overshoot={overshoot} "
                f"max_sweeps={max_sweeps} patience={patience} min_change={min_change}",
                flush=True,
            )

        # ----------------- FLOW-GUIDED all-or-none steps -----------------
        flow_plan = None
        flow_used_accum = None
        sweeps = 0
        history: list[int] = []

        if use_flow:
            # Build initial flow plan
            if target_counts is None:
                raise ValueError(f"step {label}: use_flow=True but target_counts is None")

            M0 = self.element_flow_plan(target_counts)
            if overshoot != 0.0:
                factor = 1.0 + float(overshoot)
                if factor <= 0.0:
                    raise ValueError(f"{overshoot = } must be > -1")
                M_eff = np.rint(M0.astype(np.float64) * factor).astype(np.int64)
            else:
                M_eff = M0.copy()

            flow_plan = M_eff
            M_rem = M_eff.astype(np.int64, copy=True)
            flow_used_accum = np.zeros_like(M_rem, dtype=np.int64)

            # Number of sweeps for flow step
            if max_sweeps < 0:
                max_sweeps_eff = P  # a safe default upper bound
            else:
                max_sweeps_eff = max_sweeps

            for _ in range(max_sweeps_eff):
                sweeps += 1

                moved_local, flow_used_local = self.diffuse(
                    mode=mode,
                    interface_policy=iface,
                    threshold=thr,
                    flow_matrix=M_rem,
                    restrict_etypes=restrict_etypes,
                    skip_last=skip_last,
                    restrict_src_dest=False,      # gating enforced via flow
                    target_counts=None,
                    move_spts_nodes=True,
                )

                moved = W.allreduce(int(moved_local), op=mpi.SUM)
                history.append(moved)

                # Aggregate actual flow usage across all ranks
                if flow_used_local is not None:
                    flow_used_global = np.zeros_like(flow_used_local, dtype=np.int64)
                    W.Allreduce(flow_used_local, flow_used_global, op=mpi.SUM)

                    M_rem           -= flow_used_global
                    flow_used_accum += flow_used_global

                if rnk == root_w:
                    print(f"[iterate]   sweep={sweeps} moved_glob={moved}", flush=True)

                if moved == 0:
                    break

        # ----------------- per-element smoothing steps -------------------
        else:
            # Only meaningful for iface='per-element' (we can assert to be safe)
            if iface != "per-element":
                raise ValueError(
                    f"step {label}: use_flow=False but iface={iface!r} "
                    "expected 'per-element' for smoothing-to-stagnation"
                )

            # Smoothing-to-stagnation using per-element diffuse().
            # This is the inlined version of smooth_until_stagnates().
            max_iters = max_sweeps if max_sweeps > 0 else 50

            history = []
            stable = 0
            last: Optional[int] = None

            for _ in range(int(max_iters)):
                moved_local, _ = self.diffuse(
                    mode=mode,
                    interface_policy="per-element",
                    threshold=thr,
                    flow_matrix=None,
                    restrict_etypes=None,  # keep semantics identical to previous helper
                    skip_last=False,
                    restrict_src_dest=restrict_src_dest,
                    target_counts=target_counts if restrict_src_dest else None,
                    move_spts_nodes=move_spts_nodes,
                )

                moved = W.allreduce(int(moved_local), op=mpi.SUM)
                history.append(moved)

                # Track stagnation (same logic as smooth_until_stagnates)
                if last is not None and abs(moved - last) <= int(min_change):
                    stable += 1
                else:
                    stable = 0
                last = moved

                # Local stop condition
                stop = (moved == 0) or (stable >= int(patience))
                # Global agreement on stopping
                stop_all = W.allreduce(1 if stop else 0, op=mpi.MAX)

                if rnk == root_w:
                    sweep_id = len(history)
                    print(
                        f"[iterate]   sweep={sweep_id} moved_glob={moved} "
                        f"stable={stable} stop_all={bool(stop_all)}",
                        flush=True,
                    )

                if stop_all:
                    break

            sweeps = len(history)

        # ----------------- End-of-step logging ---------------------------

        cur, F_sym, nfaces_total = self.__partition_stats()
        if target_counts is not None:
            tgt = np.asarray(target_counts, dtype=np.int64)
            diff = (cur - tgt).tolist()
        else:
            diff = None

        if rnk == root_w:
            print(
                f"[iterate] done step={label} sweeps={sweeps} "
                f"CUR={cur.tolist()} "
                + (f"DIFF={diff}" if diff is not None else "")
                + f" FACES={nfaces_total}",
                flush=True,
            )

            if flow_plan is not None:
                print(f"[iterate]   flow_plan=\n{flow_plan}", flush=True)
                print(f"[iterate]   flow_used=\n{flow_used_accum}", flush=True)

    def _pick_rank_b_min_mpi_faces_local(self, rank_a: int, etype: str | None = None, ) -> Optional[int]:
        """
        Given world-rank `rank_a`, pick neighbour `rank_b` based on rank_a's
        local MPI-face connectivity.

        If `etype` is None (default):
            - Choose neighbour with the smallest MPI-face interface.

        If `etype` is not None:
            - Prefer neighbour with the largest number of elements of this
              etype on the interface; tie-break by smallest rank index.
            - If that etype is absent on all interfaces, fall back to
              "fewest faces".
        """
        commw = comm['world']
        rank_a = int(rank_a)
        myr = int(rank['world'])

        if myr == rank_a:
            per_nbr = self._mpi_faces_by_neighbor()   # {nbr: [(et, lid, fidx), ...]}

            if not per_nbr:
                rb = None
            else:
                iface_sizes = {nbr: len(faces) for nbr, faces in per_nbr.items()}

                if etype is not None:
                    et_counts: dict[int, int] = {}
                    for nbr, faces in per_nbr.items():
                        lids = [lid for (et, lid, fidx) in faces if et == etype]
                        et_counts[nbr] = len(set(lids))

                    max_count = max(et_counts.values()) if et_counts else 0

                    if max_count > 0:
                        candidates = [n for n, c in et_counts.items() if c == max_count]
                        rb = int(min(candidates))
                        # Condensed log: only chosen neighbour, no huge dicts
                        if rank_a == root['world']:
                            print(
                                f"[itc4.pickb] R{rank_a} etype={etype} "
                                f"max_iface_count={max_count} -> rb={rb}",
                                flush=True,
                            )
                    else:
                        rb = min(
                            iface_sizes.keys(),
                            key=lambda n: (iface_sizes[n], int(n)),
                        )
                        if rank_a == root['world']:
                            print(
                                f"[itc4.pickb] R{rank_a} etype={etype} "
                                "no faces of this etype; "
                                f"fallback -> rb={rb}",
                                flush=True,
                            )
                else:
                    rb = min(
                        iface_sizes.keys(),
                        key=lambda n: (iface_sizes[n], int(n)),
                    )
                    if rank_a == root['world']:
                        print(
                            f"[itc4.pickb] R{rank_a} etype=None -> rb={rb}",
                            flush=True,
                        )
        else:
            rb = None

        rb = commw.bcast(rb, root=rank_a)
        return rb

    def _pick_seed_rank_for_new_rank(self, new_rank: int) -> tuple[int, str]:
        """
        Decide which existing rank should act as rank_a for seeding `new_rank`,
        and which etype is used for seeding.

        Current policy
        --------------
        - Take the first etype in `self._etype_order()` as the top-priority
          etype for `new_rank` (e.g. 'hex' on CPU ranks).
        - Each rank counts how many local elements it has of this etype.
        - The global donor rank_a is the rank with the largest count; ties are
          broken in favour of the smallest rank index.

        Returns
        -------
        rank_a : int
            Donor rank index in world communicator.
        seed_etype : str
            The top-priority etype used for seeding.
        """
        this_rank = int(rank['world'])

        et_order = list(self._etype_order())
        if not et_order:
            if this_rank == root['world']:
                print(
                    f"[itc4.seed-info] new_rank={new_rank} has no etypes; "
                    "unable to pick rank_a",
                    flush=True,
                )
            # Fallback: no etypes -> no sensible seed_etype, caller should bail.
            return 0, ""

        seed_etype = et_order[0]

        # Local count of top-priority etype
        local_count_seed = int(len(self.i.eidxs.get(seed_etype, ())))
        counts_seed = comm['world'].allgather(local_count_seed)

        # Argmax over counts; deterministic tie-break via lowest rank
        max_count = max(counts_seed)
        rank_a_candidates = [
            r for r, c in enumerate(counts_seed) if c == max_count
        ]
        rank_a = int(min(rank_a_candidates))

        if this_rank == root['world']:
            print(
                f"[itc4.seed-info] new_rank={new_rank} "
                f"seed_etype={seed_etype} counts={counts_seed} -> rank_a={rank_a}",
                flush=True,
            )

        return rank_a, seed_etype

    def seed_rank(self, new_rank: int, targets,
        rank_a: int | None = None, n_seed_per_etype: int = 1, ) -> None:
        """
        Seed `new_rank` with a small patch of elements from a single preferred
        etype, then run a short vertex-based diffusion to grow that patch.

        Semantics
        ---------
        - Let `seed_etype` be the first etype in `self._etype_order()`
          (top of the preference list for seeding).
        - If rank_a is None, we choose rank_a globally as the rank with the
          largest number of `seed_etype` elements via `_pick_seed_rank_for_new_rank`.
        - We choose rank_b as the neighbour of rank_a with minimal MPI faces
          (via `_pick_rank_b_min_mpi_faces_local`), then broadcast rank_b.
        - On all ranks we call `collect_mpi_vertex_cluster(rank_a, rank_b)`,
          but ONLY `rank_a` actually donates up to `n_seed_per_etype` elements
          of `seed_etype` to `new_rank`.
        - Then we recompute element counts and do a short vertex-based
          diffusion with `skip_last=True` so that only the first N-1 etypes
          grow via vertex-based smoothing.
        """
        this_rank = int(rank['world'])

        # 1) Determine rank_a and seed_etype from global top-priority policy
        auto_rank_a, seed_etype = self._pick_seed_rank_for_new_rank(new_rank)

        if not seed_etype:
            # No etypes at all; nothing sensible to do
            if this_rank == root['world']:
                print(
                    f"[itc4.seed] new_rank={new_rank} no seed_etype; "
                    "skipping seeding",
                    flush=True,
                )
            return

        if rank_a is None:
            rank_a = auto_rank_a
        else:
            rank_a = int(rank_a)
            if this_rank == root['world']:
                print(
                    f"[itc4.seed] new_rank={new_rank} overriding auto "
                    f"rank_a={auto_rank_a} with user rank_a={rank_a}",
                    flush=True,
                )

        # 2) Choose rank_b: neighbour of rank_a with a suitable interface for
        #    the seed_etype (prefer neighbours with many faces of that etype).
        rb_local = self._pick_rank_b_min_mpi_faces_local(rank_a, etype=seed_etype)
        rank_b = int(comm['world'].bcast(int(rb_local), root=rank_a))

        if this_rank == root['world']:
            print(
                f"[itc4.seed] seeding R{new_rank} from interface "
                f"(R{rank_a}, R{rank_b}) seed_etype={seed_etype}",
                flush=True,
            )

        # 3) Build a tiny "cluster" everywhere, but only rank_a will donate
        cluster = self.collect_mpi_vertex_cluster(rank_a, rank_b, seed_etype=seed_etype)

        eidxs_diff: dict[int, dict[str, np.ndarray]] = {}

        if this_rank == rank_a:
            gids = np.asarray(cluster.get(seed_etype, ()), dtype=np.int64)

            if gids.size:
                cur = np.asarray(self.i.eidxs.get(seed_etype, ()), dtype=np.int64)
                if cur.size:
                    # Intersect with local gids (defensive; cluster is local)
                    mask = np.isin(gids, cur, assume_unique=False)
                    sel = gids[mask][: int(n_seed_per_etype)]

                    if sel.size:
                        eidxs_diff.setdefault(int(new_rank), {})[seed_etype] = sel
                        print(
                            f"[itc4.seed] R{this_rank} donating "
                            f"{sel.size} {seed_etype} element(s) to R{new_rank}: "
                            f"{sel.tolist()}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[itc4.seed] R{this_rank} cluster for {seed_etype} "
                            "does not intersect local gids; nothing donated",
                            flush=True,
                        )
                else:
                    print(
                        f"[itc4.seed] R{this_rank} has no local {seed_etype} "
                        "elements; nothing donated",
                        flush=True,
                    )
            else:
                # Fallback: no seed_etype on this interface, but we still want to
                # seed the new rank with *something* of seed_etype from rank_a.
                cur = np.asarray(self.i.eidxs.get(seed_etype, ()), dtype=np.int64)

                if cur.size:
                    sel = cur[: int(n_seed_per_etype)]
                    eidxs_diff.setdefault(int(new_rank), {})[seed_etype] = sel
                    print(
                        f"[itc4.seed] R{this_rank} found no {seed_etype} elements "
                        f"on interface (R{rank_a}, R{rank_b}); "
                        f"fallback donating {sel.size} local {seed_etype} "
                        f"element(s) to R{new_rank}: {sel.tolist()}",
                        flush=True,
                    )
                else:
                    print(
                        f"[itc4.seed] R{this_rank} found no {seed_etype} elements "
                        f"on interface (R{rank_a}, R{rank_b}) and has no local "
                        f"{seed_etype} elements; nothing donated",
                        flush=True,
                    )


        # 4) Apply the seeding relocation collectively
        self._apply_plan_and_commit(eidxs_diff)

        # 5) Recompute counts and perform a short vertex-based diffusion
        cur0 = self._cur_counts_total
        
        if this_rank == root['world']:
            print(f"[itc4.seed] mode=seed_rank CURRENT={cur0}", flush=True)

        M0 = self.element_flow_plan(cur0, targets)

        self.diffuse_smoothing_vertices(flow_matrix=M0, skip_last=True)

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

        if myr == root['world']:
            print(
                f"[mm.swap] swap_partitions r0={r0} r1={r1} "
                f"before={before_all} after={after_all}",
                flush=True,
            )

        # Ownership tags in con_i have to be updated to reflect the new
        # gid->owner distribution. Do this once per rank.
        self._retag_con_owners()

        # Invalidate topology-dependent caches (owners_map, nei_gid_sets, ...)
        self._invalidate_topology()

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

    def __partition_stats(self) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Returns (cur_counts, F_sym, nfaces_total):

        cur_counts : (P,) int64
            Global element counts per rank (same on all ranks).
        F_sym : (P, P) int64
            Symmetrised MPI-face matrix: F_sym[r,s] = faces on (r,s) interface.
        nfaces_total : int
            Total number of MPI faces (sum over r < s).
        """
        W   = comm['world']
        rnk = int(rank['world'])
        P   = W.size

        cur = np.asarray(self._cur_counts_total, dtype=np.int64)

        # Local one-sided face counts
        F_local = np.zeros((P, P), dtype=np.int64)
        per_nbr = self._mpi_faces_by_neighbor()
        for nbr, faces in per_nbr.items():
            F_local[rnk, int(nbr)] = int(len(faces))

        # Global; symmetrise
        F = W.allreduce(F_local, op=mpi.SUM)
        F_sym = np.minimum(F, F.T)
        nfaces_total = int(F_sym.sum() // 2)

        return cur, F_sym, nfaces_total


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
        R_comp, E = cnt.shape

        # --- NEW: embed compute-sized cnt,F into world-sized arrays ---
        R_world = comm['world'].size

        cnt_w = np.zeros((R_world, E), dtype=np.int64)
        F_w   = np.zeros((R_world, R_world), dtype=np.int64)

        # rankmap['compute'] is world-ranks in compute-rank order
        comp_wrs = list(rankmap['compute'])
        assert len(comp_wrs) == R_comp, \
            f"len(rankmap['compute'])={len(comp_wrs)} != cnt.rows={R_comp}"

        # Fill per-rank element counts and interface matrix in world index space
        for r_comp, wr_i in enumerate(comp_wrs):
            wr_i = int(wr_i)
            cnt_w[wr_i, :] = cnt[r_comp, :]

            for s_comp, wr_j in enumerate(comp_wrs):
                wr_j = int(wr_j)
                F_w[wr_i, wr_j] = F[r_comp, s_comp]

        # From here on, use world-sized cnt/F and R
        cnt, F = cnt_w, F_w
        R = R_world
        # --- END NEW ---

        pairs = [(i, j) for i in range(R) for j in range(i + 1, R)]
        pair_vals = [int(F[i, j]) for (i, j) in pairs]
        pairs_nz  = sum(v > 0 for v in pair_vals)
        faces_sum = sum(pair_vals)

        if not os.path.exists(csv_path):
            cols = (
                ['tcurr']
                + [
                    c
                    for r in range(R)
                    for c in ([f'{r}-{et}' for et in etypes] + [f'{r}-total'])
                  ]
                + [f'i{i}-{j}' for (i, j) in pairs]
                + ['pairs', 'faces']
            )
            with open(csv_path, 'w') as f:
                f.write(','.join(cols) + '\n')

        row = [f"{tcurr:.6f}"]
        for r in range(R):
            vals = cnt[r].astype(int).tolist()
            row += list(map(str, vals)) + [str(sum(vals))]
        row += list(map(str, pair_vals)) + [str(pairs_nz), str(faces_sum)]

        with open(csv_path, 'a') as f:
            f.write(','.join(row) + '\n')

    def collect_mpi_vertex_cluster(self, rank_a: int, rank_b: int, 
                                   seed_etype: str | None = None, ):
        """
        Per-rank "seed" element on a vertex interface between two ranks.

        Given two ranks (rank_a, rank_b), this returns, on each calling rank,
        a mapping {etype -> np.ndarray[int64]} of *at most one* global element
        ID on THIS rank which touches any MPI-vertices on the (rank_a, rank_b)
        interface.

        Semantics
        ---------
        - If this_rank ∉ {rank_a, rank_b}, the returned dict has the same
          keys (etypes) but all arrays are empty.
        - If this_rank ∈ {rank_a, rank_b}, we:
            * determine the relevant neighbour rank,
            * look up its MPI-vertex set,
            * scan etypes in a preference order:
                  - if seed_etype is given, try that first (if present),
                    then all remaining etypes in self._etype_order();
                  - otherwise just self._etype_order();
            * for the first etype in that scan order that has at least one
              element whose spts_nodes contain a neighbour MPI-vertex,
              select exactly ONE such element (lowest local index),
              and store its global ID in the returned dict.
        - All etypes that are not chosen have empty arrays.

        This is intentionally minimal: it is designed to support `seed_rank`,
        not to capture a full "cluster".
        """

        ra = int(rank_a)
        rb = int(rank_b)
        if ra == rb:
            raise ValueError("rank_a and rank_b must differ")

        this_rank = int(rank['world'])

        # Initialise result with per-etype empty arrays
        cluster_by_et: dict[str, np.ndarray] = {
            et: np.empty(0, dtype=np.int64) for et in self.etypes
        }

        # Base etype order (backend / device preference)
        base_order = list(self._etype_order()) or list(self.etypes)

        # Build scan order:
        # - If a seed_etype is provided, we try that type first (if it exists),
        #   then all remaining etypes in base_order.
        # - Otherwise, just use base_order.
        if seed_etype is not None:
            et_scan: list[str] = []

            # Hard preference: seed_etype first, if it exists on this mesh.
            if seed_etype in self.etypes:
                et_scan.append(seed_etype)

            # Append remaining etypes, preserving base_order but avoiding dups.
            for et in base_order:
                if et != seed_etype and et not in et_scan:
                    et_scan.append(et)

            # As an ultimate fallback (very defensive), if for some reason
            # et_scan ended up empty, fall back to all etypes.
            if not et_scan:
                et_scan = list(self.etypes)
        else:
            et_scan = base_order

        # Ensure we have up-to-date MPI-vertex sets per neighbour.
        mvu = self.collect_mpi_vertex_nodes()

        if this_rank not in (ra, rb):
            # Not part of this interface; nothing to do.
            print(
                f"[itc4.cluster] R{this_rank} not in (R{ra}, R{rb}); "
                f"cluster empty (seed_etype={seed_etype})",
                flush=True,
            )
            return cluster_by_et

        # Decide which neighbour's MPI-vertices are relevant on this rank.
        nbr = rb if this_rank == ra else ra
        nbr_vertices = np.asarray(mvu.get(nbr, ()), dtype=np.int64)

        if nbr_vertices.size == 0:
            # We *are* one of the pair, but there is no MPI-vertex interface.
            print(
                f"[itc4.cluster] R{this_rank} (ra={ra}, rb={rb}) "
                f"no MPI vertices with R{nbr}; cluster empty "
                f"(seed_etype={seed_etype})",
                flush=True,
            )
            return cluster_by_et

        chosen_total = 0
        chosen_etype: str | None = None

        # Core logic: for each etype (in preference order), pick the first
        # local element whose spts_nodes contain any of the neighbour's
        # interface vertices.
        for et in et_scan:
            nds = self.i.spts_nodes[et]
            gids = self.i.eidxs[et]

            if nds is None or nds.size == 0 or gids.size == 0:
                continue

            # Elements which contain any of the neighbour's interface vertices.
            v_in = np.isin(nds, nbr_vertices, assume_unique=False)
            sel = v_in.any(axis=1)

            lids = np.nonzero(sel)[0].astype(np.int64, copy=False)
            if lids.size == 0:
                continue

            # Pick exactly ONE element: the first in local index order
            cluster_by_et[et] = gids[lids[0]]
            chosen_total = 1
            chosen_etype = et

            # For seeding we only need one element from the most-preferred
            # etype that exists on this interface; stop after we find it.
            break

        print(
            f"[itc4.cluster] R{this_rank} (ra={ra}, rb={rb}) "
            f"seed_etype={seed_etype} chosen_total={chosen_total} "
            f"chosen_etype={chosen_etype} cluster={{"
            + ", ".join(
                f"{et}:{cluster_by_et[et].tolist()}" for et in self.etypes
            )
            + "}",
            flush=True,
        )

        return cluster_by_et

    def refine(self, mode, thr):
        self._reset_j_with_i()
        eidxs_diff = self._cpd.refine(mode=mode, thr=thr)
        self._apply_plan_and_commit(eidxs_diff)
        self.smooth_until_stagnates()

    def _get_global_owner_array(self) -> np.ndarray:
        """
        Debug helper: build a global owner array

            owners_global[g] = rank that owns global element ID g,

        using State.eidxs_flat (global IDs local to this rank) and an
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

    # --------------------------------------------------------------------------
    # Cost to targets
    # --------------------------------------------------------------------------

    # Cost targets etc

    @staticmethod
    def compute_cost(g1a, g1s, g1r):
        g1a_old = np.asarray(g1a, dtype=float)
        g1s_old = np.asarray(g1s, dtype=float)
        g1r_old = np.asarray(g1r, dtype=float)

        s_out = g1s_old.sum(axis=1)
        r_in  = g1r_old.sum(axis=1)
        r_out = g1r_old.sum(axis=0)

        return (g1a_old - r_in  * _MetaMesh.lb_cost_scale_g1r
                        - s_out * _MetaMesh.lb_cost_scale_g1s
                        + r_out * _MetaMesh.lb_cost_scale_g1rt)

    @staticmethod
    def calc_target_ecounts(ecurrs, g1a, g1s, g1r, cfg):
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

        cost_old = _MetaMesh.compute_cost(g1a, g1s, g1r)

        # --- Restrict world-indexed element counts to old compute ranks ---

        # rankmap_old is a list of world ranks in old compute order
        ecurrs_old = np.asarray([ecurrs[wr] for wr in rankmap['compute']], dtype=np.int64)
        Ntot = int(ecurrs_old.sum())

        # Snapshot medians to CSV (uses whatever index space g1* live in)
        _MetaMesh.write_g1_median_csvs(g1a, g1s, g1r, g1idx=1)

        # Per-element inverse cost on old ranks
        inv_cost = ecurrs_old / cost_old
        N_star_old = Ntot * (inv_cost / inv_cost.sum())

        # --- Remap N_star_old from old->new using device groups (world space) ---

        # Devices from config: e.g. ['gpu', 'cpu', 'cpu', 'cpu', 'cpu']
        devices = cfg.getliteral('backend', 'devices')

        # rankmap_* are world-rank lists, matching ecurrs/devices indexing
        if list(rankmap['compute']) != list(rankmap['newcompute']):
            N_star_new = _MetaMesh.remap_targets_by_ranklist(N_star_old,
                old_ranks=rankmap['compute'], new_ranks=rankmap['newcompute'],
                devices=devices, tag="[lb-group]")
        else:
            # No rank change: old and new are the same
            N_star_new = N_star_old.copy()

        # --- Normalise and round to integer targets in newcompute order ---
        N_int = _MetaMesh.normalise_and_round_targets(N_star_new, Ntot=Ntot, tag="[lb-round]",)
        if comm['newcompute'] != mpi.COMM_NULL and rank['newcompute'] == root['newcompute']:
            print(f"[load-balance] N_int={N_int} (sum={int(N_int.sum())})")



        # Return in newcompute's world-rank order (rankmap_new)
        return [26603, 15084, 15430, 15246, 13313, 15643, 15423, 14049, 12269, 21568, 17458, 16640]
        # return N_int.tolist()

    @staticmethod
    def write_g1_median_csvs(g1a, g1s, g1r, g1idx: int = 1):
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
        mcols = [f"i{i}-{j}"                                                  for i in range(P_world) for j in range(P_world)]
        _append_csv_row('g1-send-median-ms.csv', mcols, [int(send_us_w[i, j]) for i in range(P_world) for j in range(P_world)])
        _append_csv_row('g1-recv-median-ms.csv', mcols, [int(recv_us_w[i, j]) for i in range(P_world) for j in range(P_world)])

    @staticmethod
    def remap_targets_by_ranklist(N_star_old, old_ranks, new_ranks, devices, tag="[lb-group]"):
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

    @staticmethod
    def normalise_and_round_targets(N_star, Ntot, tag="[lb-round]"):
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

        if k != 0:
            frac = N_scaled - N_floor
            order = np.argsort(frac)  # ascending

            if k > 0:
                idxs = order[-k:]   # bump largest fractions
                N_floor[idxs] += 1
                print(f"{tag} +1 to indices={idxs}")
            else:
                idxs = order[:-k]   # k < 0 → drop smallest fractions
                N_floor[idxs] -= 1
                print(f"{tag} -1 from indices={idxs}")

        return N_floor

    # --------------------------------------------------------------------------
    # Mesh to MetaMesh conversion
    # --------------------------------------------------------------------------

    def to_mesh(self, eidxs_dest):
        ic = _MeshInterconnector(self.mesh_src.eidxs, eidxs_dest)

        mesh_int = replace(self.mesh_src, eidxs=eidxs_dest,
            spts       =ic.relocate(self.mesh_src.spts,        edim=1), 
            spts_nodes =ic.relocate(self.mesh_src.spts_nodes,  edim=0),
            spts_curved=ic.relocate(self.mesh_src.spts_curved, edim=0),
            faces_cidxs=ic.relocate(self.mesh_src.faces_cidxs, edim=0),
            faces_offs =ic.relocate(self.mesh_src.faces_offs,  edim=0),
            )

        self._reconstruct_con_conp_bcon(mesh_int)

        eidxs_dest = self._apply_lex_ordering(mesh_int)
        ic2 = _MeshInterconnector(mesh_int.eidxs, eidxs_dest)
        
        mesh_dest = replace(mesh_int, eidxs=eidxs_dest,
            spts       =ic2.relocate(mesh_int.spts,        edim=1), 
            spts_nodes =ic2.relocate(mesh_int.spts_nodes,  edim=0),
            spts_curved=ic2.relocate(mesh_int.spts_curved, edim=0),
            faces_cidxs=ic2.relocate(mesh_int.faces_cidxs, edim=0),
            faces_offs =ic2.relocate(mesh_int.faces_offs,  edim=0),
            )

        self._reconstruct_con_conp_bcon(mesh_dest)

        # Remove keys with empty entries
        mesh_dest.eidxs = {k: v for k, v in mesh_dest.eidxs.items() if v.size}

        return mesh_dest

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

    # --------------------------------------------------------------------------

    def diffuse_smoothing2(
        self,
        flow_matrix: np.ndarray,
        *,
        threshold: int = 0,
        etype_order: Optional[List[str]] = None,   # kept for future if you want per-etype bias
        return_count: bool = False,
        verbose: bool = False,
        scale: float = 0.5,
        score_by: str = "vertex",                  # 'edge' | 'vertex'
    ) -> Optional[int]:
        """
        One flow-guided “nudge” pass, matching the old idea of diffuse_smoothing2:

            - score_by='vertex' → vertex-based interface scoring
            - score_by='edge'   → face-based interface scoring

        Internally this just calls self.diffuse(...) with:
            iface='all-or-none', flow_matrix=M_eff, mode='vertices' or 'faces'.
        """
        W      = comm['world']
        rnk    = int(rank['world'])
        root_w = int(root['world'])

        M = np.asarray(flow_matrix, dtype=np.int64)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"[itc4] flow_matrix must be square; got shape {M.shape}")
        R = int(M.shape[0])
        if not (0 <= rnk < R):
            raise ValueError(f"[itc4] rank {rnk} out of range for flow_matrix of size {R}")

        # Scale the flow plan (relaxation in flow space, not element space)
        if float(scale) <= 0.0:
            raise ValueError(f"[itc4] scale must be > 0, got {scale!r}")
        if abs(float(scale) - 1.0) < 1e-12:
            M_eff = M.astype(np.int64, copy=True)
        else:
            M_eff = np.rint(M.astype(np.float64) * float(scale)).astype(np.int64)
            M_eff[M_eff < 0] = 0

        sb = str(score_by).lower()
        if sb in ("edge", "edges", "face", "faces"):
            mode = "faces"
        elif sb in ("vertex", "vertices", "vtx", "vtxs"):
            mode = "vertices"
        else:
            raise ValueError(f"[itc4] score_by must be 'edge' or 'vertex'; got {score_by!r}")

        moved_local, _flow_used = self.diffuse(
            mode=mode,
            interface_policy="all-or-none",
            threshold=int(threshold),
            flow_matrix=M_eff,
            restrict_etypes=None,         # we can wire etype_order→restrict_etypes later if needed
            skip_last=False,
            restrict_src_dest=False,      # flow_matrix is the single source of truth here
            target_counts=None,
            move_spts_nodes=(mode == "vertices"),
        )

        moved = int(W.allreduce(int(moved_local), op=mpi.SUM))

        if verbose and rnk == root_w:
            print(
                f"[itc4.pass] score_by={sb} mode={mode} thr={int(threshold)} "
                f"scale={float(scale):.3f} moved_glob={moved}",
                flush=True,
            )

        return moved if return_count else None

    def smooth_until_stagnates(
        self,
        *,
        max_iters: int = 20,
        patience: int = 1,
        min_change: int = 0,
        threshold: int = 0,
        target_counts: Optional[List[int]] = None,
        restrict_src_dest: bool = False,
    ) -> list[int]:
        """
        Run face-based smoothing repeatedly until global move-count stagnates.

        This matches the old smooth_until_stagnates():
            - scoring: faces (edge-based),
            - interface_policy: per-element,
            - no flow_matrix,
            - optional donor/receiver gating via target_counts.
        """
        W      = comm['world']
        rnk    = int(rank['world'])
        root_w = int(root['world'])

        history: list[int] = []
        stable = 0
        last: Optional[int] = None

        for _ in range(int(max_iters)):
            moved_local, _ = self.diffuse(
                mode="faces",
                interface_policy="per-element",
                threshold=int(threshold),
                flow_matrix=None,
                restrict_etypes=None,
                skip_last=False,
                restrict_src_dest=restrict_src_dest,
                target_counts=target_counts if restrict_src_dest else None,
                move_spts_nodes=False,
            )

            moved = int(W.allreduce(int(moved_local), op=mpi.SUM))
            history.append(moved)

            if last is not None and abs(moved - last) <= int(min_change):
                stable += 1
            else:
                stable = 0
            last = moved

            stop = (moved == 0) or (stable >= int(patience))
            stop_all = W.allreduce(1 if stop else 0, op=mpi.MAX)

            if rnk == root_w:
                sweep_id = len(history)
                print(
                    f"[smooth] sweep={sweep_id} moved_glob={moved} "
                    f"stable={stable} stop_all={bool(stop_all)}",
                    flush=True,
                )

            if stop_all:
                break

        return history

    def iterate_to_convergence(
        self,
        target_counts: List[int],
        *,
        etype_order: Optional[List[str]] = None,
        verbose: bool = False,
        flowmat_relax: float = 0.5,
    ) -> list[int]:
        """
        High-level controller with the old three-pathway structure:

            1) Vertex-based flow diffusion   (score_by='vertex')
            2) Face-based   flow diffusion   (score_by='edge')
            3) Face-based   smoothing        (no flow, to stagnation)

        All the low-level pieces are the ones you already trust:
            - element_flow_plan(...) for M
            - diffuse(...) for a single sweep
            - smooth_until_stagnates(...) for final polish
        """
        W      = comm['world']
        rnk    = int(rank['world'])
        root_w = int(root['world'])

        def _cur_counts_total_list() -> list[int]:
            # self._cur_counts_total is already used in element_flow_plan
            return list(self._cur_counts_total)

        if rnk == root_w and verbose:
            print(f"[itc4.iter] TARGET={list(target_counts)}", flush=True)

        # (threshold, score_by) pairs; matches your old exec_order idea
        exec_order = ([(2, "vertex")]+
                                             [(2, "edge")]+
                                             [(0, "edge")]*comm['world'].size
        )

        for thr, score_by in exec_order:
            cur0 = _cur_counts_total_list()
            if rnk == root_w and verbose:
                print(f"[itc4.iter] CURRENT={cur0}", flush=True)

            # New element_flow_plan(v2) already uses twoway_mask internally
            M0 = self.element_flow_plan(target_counts)

            if rnk == root_w and verbose:
                print(f"[itc4.iter] flow_plan score_by={score_by} thr={thr} =", flush=True)
                print(M0, flush=True)

            # One relaxed flow-guided pass
            self.diffuse_smoothing2(
                M0,
                threshold=thr,
                etype_order=etype_order,
                return_count=False,
                verbose=verbose,
                scale=float(flowmat_relax),
                score_by=score_by,
            )

        # Final edge-based smoothing around the converged flow result
        history = self.smooth_until_stagnates(max_iters=50, patience=1, min_change=0,
                                             threshold=0, target_counts=target_counts,
                                             restrict_src_dest=True,)
        if rnk == root_w and verbose:
            print(f"[itc4.iter] smoothing history={history}", flush=True)
        return history

    def _build_scotch_cache(self, *, ufactor: int):
        if hasattr(self, "_scotch_cache"):
            return

        if rank['world'] != root['world']:
            self._scotch_cache = None
            return

        # You need the mesh file path you are running from.
        # Use whatever your code already has (args.mesh, self.mesh_src.fname, etc.)
        mesh_path = self.mesh_src.fname

        with h5py.File(mesh_path, "r") as mesh:
            con, ecurved, edisps, cdisps = BasePartitioner.construct_global_con(mesh)

            # Sanity: global element count should match your State
            nelems_g = int(len(ecurved))
            if nelems_g != int(self.i.nelems_g):
                raise RuntimeError(f"nelems mismatch: file={nelems_g} state={self.i.nelems_g}")

            # Element weights (match your CLI -e...:1 usage)
            elewts = {et: 1 for et in edisps.keys()}

            # Dummy partwts for building elewts_fn (actual partwts are passed later)
            dummy_partwts = [1]*comm['world'].size
            part = SCOTCHPartitioner(dummy_partwts, elewts=elewts, 
                                     opts={"ufactor": ufactor})
            elewts_fn = part._get_elewts_fn(edisps)

            # This is the critical periodic grouping step you are missing
            pmcon, exwts, pmerge = BasePartitioner._group_periodic_eles(mesh, con, cdisps, elewts_fn)

        graph, vemap = BasePartitioner._construct_graph(pmcon, elewts_fn, exwts=exwts)
        graph = self._scotch_sanitize_graph(graph)

        print(f"[scotch.cache] nverts={graph.vwts.shape[0]} nnz={graph.etab.size} "
            f"(nelems_g={nelems_g})", flush=True)

        self._scotch_cache = (graph, vemap, pmerge, nelems_g, edisps)

    def _scotch_sanitize_graph(self, graph):
        """
        Return a new graph with SCOTCH-safe buffers:
        - dtype int32 (SCOTCH_numSizeof == 4)
        - C-contiguous
        - owned memory (no views/temporaries)
        Works for PyFR's namedtuple graph objects (uses _replace).
        """
        import numpy as np

        def as_i32_c(a):
            if a is None:
                return None
            return np.ascontiguousarray(np.asarray(a, dtype=np.int32))

        # Build replacement fields
        repl = {}
        for name in ("vtab", "etab", "vwts", "ewts"):
            if hasattr(graph, name):
                repl[name] = as_i32_c(getattr(graph, name))

        # Namedtuple path (PyFR)
        if hasattr(graph, "_replace"):
            g2 = graph._replace(**repl)
        else:
            # Fallback: try best-effort setattr on a mutable object
            g2 = graph
            for k, v in repl.items():
                setattr(g2, k, v)

        # Cheap invariants (catch corruption early)
        vtab = g2.vtab
        etab = g2.etab
        assert vtab.ndim == 1 and etab.ndim == 1
        assert int(vtab[0]) == 0
        assert int(vtab[-1]) == etab.size

        nverts = vtab.size - 1
        if etab.size:
            mn = int(etab.min())
            mx = int(etab.max())
            assert mn >= 0
            assert mx < nverts

        return g2

    def _scotch_copy_graph(self, graph):
        """
        Deep-copy numpy buffers of a (likely namedtuple) graph object.
        """
        import numpy as np

        repl = {}
        for name in ("vtab", "etab", "vwts", "ewts"):
            if hasattr(graph, name):
                a = getattr(graph, name)
                repl[name] = None if a is None else np.ascontiguousarray(np.asarray(a).copy())

        return graph._replace(**repl) if hasattr(graph, "_replace") else graph

    def partition_scotch(self, partwts, *, ufactor: int = 10, seed: int = 2079):
        self._build_scotch_cache(ufactor=ufactor)
        nelems_g = int(self.i.nelems_g)

        if rank['world'] == root['world']:
            graph, vemap, pmerge, nelems_file, edisps_file = self._scotch_cache
            assert nelems_file == nelems_g

            # Handle rank-removal (zero targets) here, so you can delete partition()
            partwts = np.asarray(partwts, dtype=np.int64)
            active = np.flatnonzero(partwts > 0).astype(int).tolist()
            if not active:
                raise ValueError("all partwts are zero")

            partwts_active = partwts[active].tolist()

            elewts = {et: 1 for et in edisps_file.keys()}
            part = SCOTCHPartitioner(partwts_active, elewts=elewts,
                                    opts={"ufactor": ufactor, "seed": seed,
                                          "strat": "quality"})

            print(f"[scotch.map] begin nverts={graph.vwts.shape[0]} nnz={graph.etab.size} "
                f"nparts_active={len(active)} ufactor={ufactor} seed={seed}", flush=True)

            # Debug-only: deep-copy to eliminate “ctypes sees stale pointer” hypotheses
            graph_use = self._scotch_copy_graph(graph)  # comment this out once stable

            vparts_merged = part._partition_graph(graph_use, partwts_active).astype(np.int32, copy=False)

            # Undo periodic merge
            vparts = BasePartitioner._ungroup_periodic_eles(pmerge, vemap, vparts_merged)

            # Map active-part ids -> world ranks
            parts_g = np.asarray([active[p] for p in vparts], dtype=np.int32)

            print("[scotch.map] end", flush=True)
        else:
            parts_g = np.empty(nelems_g, dtype=np.int32)

        comm['world'].Bcast(parts_g, root=root['world'])
        return parts_g


    def apply_global_partition(self, vparts, *, move_spts_nodes=True):
        """
        Apply a PyFR-ordered global partition vector to this MetaMesh.

        Parameters
        ----------
        vparts : array-like, shape (nelems_g,)
            Partition id for every global element in PyFR ordering:
            concatenate over etypes (sorted), within each etype order by global gid.
            This is the same ordering your reconstruct_by_diffusion() builds.
        move_spts_nodes : bool
            Whether to relocate spts_nodes.
        """
        vparts = np.asarray(vparts, dtype=np.int32)

        if vparts.ndim != 1:
            raise ValueError("vparts must be 1D")

        nelems_g = int(self.i.nelems_g)
        if vparts.size != nelems_g:
            raise ValueError(f"vparts has size {vparts.size}, expected {nelems_g}")

        # Optional strict sanity
        if vparts.size and (vparts.min() < 0 or vparts.max() >= comm['world'].size):
            raise ValueError("vparts contains invalid partition ids")


        me = int(rank['world'])

        # How many of my currently-owned elements does SCOTCH want to move away?
        # (Uses global IDs: self.i.eidxs_flat contains my owned global element numbers.)
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



        # Build destination eidxs for *this* world rank by slicing vparts per etype
        eidxs_dest = {}
        me = rank['world']

        for et in self.i.etypes:
            s = int(self.i.edisps[et])
            n = int(self.i.ecnts_g[et])
            sl = slice(s, s + n)
            blk = vparts[s:s+n]
            gids = np.flatnonzero(blk == me).astype(np.int64, copy=False)
            eidxs_dest[et] = gids

        # fast path using mesh-global flat indexing from State
        fast_conn = _MetaMeshInterconnector(etypes = self.etypes,
                                            eidxs_src  = self.i.eidxs,
                                            eidxs_dest = eidxs_dest,
                                            eidxs_flat_src   = self.i.eidxs_flat,
                                            etype_slices_src = self.i.etype_slices,
        )

        self.j = State(eidxs=eidxs_dest,
                           con_mpi=fast_conn.relocate_cons(self.i.con_mpi),
                           con_idx=fast_conn.relocate_cons(self.i.con_idx),
                       spts_nodes=(fast_conn.relocate_cons(self.i.spts_nodes)
                          if move_spts_nodes else self.i.spts_nodes))

        self._accept_j_into_i()

        self._ne_i = int(self.i.nelems_total)

        if not move_spts_nodes:
            self._spts_valid = False

        # Counts AFTER
        nloc1 = int(self.i.nelems_total)
        all1 = comm['world'].allgather(nloc1)
        if rank['world'] == root['world']:
            print(f"[apply.post] nelems_per_rank={all1} sum={sum(all1)}", flush=True)



        return eidxs_dest

    def _apply_eidxs_dest_and_commit(self, eidxs_dest, *, move_spts_nodes=True):
        # Build j from i by relocating arrays to match new ownership
        self.j = self.i.relocate_to(eidxs_dest, move_spts_nodes=move_spts_nodes)

        # IMPORTANT: actually make it “current”
        self._accept_j_into_i(move_spts_nodes=move_spts_nodes)

        # Any topology/flat-cache invalidation you already do
        if hasattr(self, "_invalidate_topology"):
            self._invalidate_topology()

    def _retag_con_owners_from_vparts(self, vparts):
        """
        Retag con_mpi neighbor-rank fields using global partition vector vparts.
        Assumes con_idx stores global flat element ids (same eid space as vparts).
        """
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


class _MetaMeshInterconnector(AlltoallMixin):
    """
    Fast, MetaMesh-specific interconnector for inner diffusion iterations.

    Uses the same mesh-global flat indexing as State._eidxs_to_flat.
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
            self._src_flat, self._src_flat_slices = State._eidxs_to_flat(self.etypes,
                eidxs_src)

        # Destination side: flat view for possible later use / checks
        # self._dest_flat, self._dest_flat_slices = State._eidxs_to_flat(self.etypes,
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

class NativeReader:
    def __init__(self, fname, pname=None, *, construct_con=True,
                 comm_name='world'):
        self.f = h5py.File(fname, 'r')
        self.mesh = _Mesh(fname=fname, raw=self.f)

        self.comm_name = comm_name

        self.mesh.etypes = sorted(self.f['eles'])
        self.mesh.ndims = self.f['nodes'].dtype['location'].shape[0]

        # Read in and transform the various parts of the mesh
        self._read_metadata()

        if comm[comm_name] != mpi.COMM_NULL:
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

        if rank['world'] == root['world']:
            creator = self.f['creator'][()].decode()
            codec = [c.decode() for c in self.f['codec']]
            uuid = self.f['mesh-uuid'][()].decode()
            version = self.f['version'][()]

            meta = (creator, codec, uuid, version)
        else:
            meta = None

        meta = comm['world'].bcast(meta, root=root['world'])
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
