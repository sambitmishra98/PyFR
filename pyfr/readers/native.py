from dataclasses import dataclass, field, replace

from collections import defaultdict

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
from pyfr.shapes import BaseShape

from tabulate import tabulate

import os

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

_CON_UNFILLED = -2

@dataclass
class State:
    eidxs:      Dict[str, np.ndarray]
    con_idx:    Dict[str, np.ndarray]
    con_mpi:    Dict[str, np.ndarray]

    spts_nodes: Dict[str, np.ndarray]

    eidxs_flat:   np.ndarray       = None
    etype_slices: Dict[str, slice] = None
    etypes:       List[str]        = None

    def __post_init__(self):
        self.etypes = self._mpi_sorted_union(self.eidxs.keys()) 

        self.eidxs_flat, self.etype_slices = State._eidxs_to_flat(self.eidxs)

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

        # Reuse meta-structure exactly as-is (no MPI, no recomputation)
        new.eidxs_flat   = src.eidxs_flat.copy() if src.eidxs_flat is not None else None
        new.etype_slices = dict(src.etype_slices) if src.etype_slices is not None else None
        new.etypes       = list(src.etypes) if src.etypes is not None else None

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
    def _eidxs_to_flat(eidxs):

        # Global, consistent etype ordering
        etypes = State._mpi_sorted_union(eidxs.keys())

        # Per-etype local max GID
        max_local = np.empty(len(etypes), dtype=np.int64)
        for k, et in enumerate(etypes):
            arr = np.asarray(eidxs.get(et, ()), dtype=np.int64)
            max_local[k] = arr.max() if arr.size else -1

        # Allgather to get per-etype global max GID
        all_max = comm['world'].allgather(max_local)
        if all_max:
            max_global = np.max(np.stack(all_max, axis=0), axis=0)
        else:
            max_global = np.empty(len(etypes), dtype=np.int64)

        # Fixed offsets per etype (same on all ranks, all topologies)
        offsets_et: Dict[str, int] = {}
        off = 0
        for k, et in enumerate(etypes):
            mg = int(max_global[k])
            width = (mg + 1) if mg >= 0 else 0
            offsets_et[et] = off
            off += width
        # 'off' is max_global_id + 1; owners_global will be length 'off'

        pieces: list[np.ndarray] = []
        etype_slices: Dict[str, slice] = {}
        start_local_flat = 0

        for et in etypes:
            gids = np.asarray(eidxs.get(et, ()), dtype=np.int64)
            ne_loc = gids.size

            if ne_loc == 0:
                etype_slices[et] = slice(start_local_flat, start_local_flat)
                continue

            base = int(offsets_et[et])
            gidxs = base + gids  # stable global IDs

            pieces.append(gidxs)
            etype_slices[et] = slice(start_local_flat, start_local_flat + ne_loc)
            start_local_flat += ne_loc

        eidxs_flat = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)

        Ne_loc = int(eidxs_flat.size)
        Ne_glob = int(off)  # max_global_id + 1 (may be >= true Ne_total if GIDs have holes)

        return eidxs_flat, etype_slices


    @staticmethod
    def _flat_to_eidxs(eidxs_flat, etype_slices):
        eidxs_flat = np.asarray(eidxs_flat, dtype=np.int64)
        return {et: eidxs_flat[sl].copy() for et, sl in etype_slices.items()}


class _MetaMesh:
    def __init__(self, *, mesh_src: _Mesh, etypes, e2i, bc2id, 
                 i: State, j: State):
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

        self._gids_i_flat = None      # np.ndarray[int64], shape (Ne_local,)
        self._etype_i_flat = None     # np.ndarray[int8],  shape (Ne_local,)
        self._etype_i_idx = None      # dict[str, np.ndarray[int64]] (indices into flat)

        # default cost scales for wait-split → targets
        self.lb_cost_scale_g1r  = 1.0
        self.lb_cost_scale_g1s  = 1.0
        self.lb_cost_scale_g1rt = 1.0

    def _invalidate_topology(self) -> None:
        """Bump topology epoch and drop caches that depend on eidx ownership."""
        self._ver['topology'] += 1

        # TODO: more fine-grained invalidation later; for now just wipe
        self._cache.pop('owners_map', None)
        self._cache.pop('nei_gid_sets', None)

    class _FlatView:
        __slots__ = ("gids", "etypes", "et_offsets", "et_index", "lids")

        def __init__(self, gids, etypes, et_offsets, et_index, lids):
            self.gids = gids              # (Ne_loc,) int64 global element IDs
            self.etypes = etypes          # tuple[str, ...] in canonical order
            self.et_offsets = et_offsets  # (n_et+1,) int64 prefix sums per etype
            self.et_index = et_index      # (Ne_loc,) int16 etype index per element
            self.lids = lids              # (Ne_loc,) int64 lid within that etype

    def _flat_state(self, which: str):
        if which not in ("i", "j"):
            raise ValueError(f"which must be 'i' or 'j', got {which!r}")

        # Pick the per-etype eidxs view
        if which == "i":
            eidxs = self.eidxs_i
        else:
            eidxs = self.eidxs_j

        etypes = list(self.etypes)

        gids_blocks = []
        et_index_blocks = []
        lid_blocks = []
        offsets = [0]

        for k, et in enumerate(etypes):
            arr = np.asarray(eidxs.get(et, ()), dtype=np.int64)
            n = int(arr.size)
            if n:
                gids_blocks.append(arr)
                et_index_blocks.append(np.full(n, k, dtype=np.int16))
                lid_blocks.append(np.arange(n, dtype=np.int64))
            offsets.append(offsets[-1] + n)

        if gids_blocks:
            gids = np.concatenate(gids_blocks)
            et_index = np.concatenate(et_index_blocks)
            lids = np.concatenate(lid_blocks)
        else:
            gids = np.empty(0, dtype=np.int64)
            et_index = np.empty(0, dtype=np.int16)
            lids = np.empty(0, dtype=np.int64)

        et_offsets = np.asarray(offsets, dtype=np.int64)

        return _MetaMesh._FlatView(
            gids=gids,
            etypes=tuple(etypes),
            et_offsets=et_offsets,
            et_index=et_index,
            lids=lids,
        )


    def _flat_owner_and_pos(self) -> tuple[np.ndarray, np.ndarray]:
        """
        owners_flat[g] = world rank that owns global flat index g
        pos_flat[g]    = local flat row index on that owner, or -1
        """
        eflat = self.i.eidxs_flat
        Ne_glob = comm['world'].allreduce(eflat.size, op=mpi.MAX)  # or use _eidxs_to_flat info
        # Build owners as you already do in _get_global_owner_array
        owners_flat = self._get_global_owner_array()  # length Ne_glob

        # Local pos: only fill for gids we own
        pos_flat = np.full_like(owners_flat, -1, dtype=np.int64)
        pos_flat[eflat] = np.arange(eflat.size, dtype=np.int64)

        return owners_flat, pos_flat

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
        fast_conn = _MetaMeshInterconnector(
            eidxs_src        = self.i.eidxs,
            eidxs_dest       = eidxs_dest,
            etypes           = self.etypes,
            eidxs_flat_src   = self.i.eidxs_flat,
            etype_slices_src = self.i.etype_slices,
        )

        spts_new = (
            fast_conn.relocate_cons(self.i.spts_nodes)
            if move_spts_nodes else
            self.i.spts_nodes
        )

        self.j = State(
            eidxs=eidxs_dest,
            con_mpi=fast_conn.relocate_cons(self.i.con_mpi),
            con_idx=fast_conn.relocate_cons(self.i.con_idx),
            spts_nodes=spts_new,
        )

        self._accept_j_into_i()
        self._ne_i = self._local_count()
        self._invalidate_topology()
        if not move_spts_nodes:
            self._spts_valid = False

        
    def _local_count(self) -> int:
        # Setup without flat_state
        return sum(len(eidxs) for eidxs in self.eidxs_i.values())
        
        # return int(self._flat_state('i').gids.size)

    def _cur_counts_total(self) -> list[int]:
        local_nelems = self._local_count()
        return list(comm['world'].allgather(int(local_nelems)))

    def counts_per_etype_flat(self, which: str = 'i') -> np.ndarray:
        fv = self._flat_state(which)
        net = len(self.etypes)
        if fv.gids.size == 0:
            return np.zeros(net, dtype=np.int64)

        return np.bincount(fv.et_index.astype(np.int64, copy=False), minlength=net)

    @property
    def eidxs_i(self): return self.i.eidxs

    @property
    def con_mpi_i(self): return self.i.con_mpi
    
    @property
    def con_idx_i(self): return self.i.con_idx

    @property
    def spts_nodes_i(self): return self.i.spts_nodes

    @property
    def eidxs_j(self): return self.j.eidxs
    @eidxs_j.setter
    def eidxs_j(self, v): self.j.eidxs = v

    @property
    def con_mpi_j(self): return self.j.con_mpi
    @con_mpi_j.setter
    def con_mpi_j(self, v): self.j.con_mpi = v

    @property
    def con_idx_j(self): return self.j.con_idx
    @con_idx_j.setter
    def con_idx_j(self, v): self.j.con_idx = v

    @property
    def spts_nodes_j(self): return self.j.spts_nodes
    @spts_nodes_j.setter
    def spts_nodes_j(self, v): self.j.spts_nodes = v

    @property
    def ntotal(self) -> int:
        """Total elements across all ranks."""
        local_nelems = self._local_count()
        return int(comm['world'].allreduce(local_nelems, op=mpi.SUM))

    @property
    def twoway_mask(self) -> np.ndarray:
        """
        Symmetric two-way adjacency mask between *compute* ranks, derived
        from con_mpi / con_idx.

        twoway_mask[r, s] == True  ⇔  there exists at least one face on rank r
        whose neighbour is owned by rank s (s ≠ r).

        Indexed in compute-rank order.
        """

        commc = comm['compute']
        if commc == mpi.COMM_NULL:
            return np.zeros((0, 0), dtype=np.bool_)

        R_comp = commc.size
        comp_wrs = list(rankmap['compute'])
        assert len(comp_wrs) == R_comp

        w2c = {int(wr): int(i) for i, wr in enumerate(comp_wrs)}

        my_wr = int(rank['world'])
        my_cr = w2c.get(my_wr, None)

        loc = np.zeros((R_comp, R_comp), dtype=np.bool_)

        if my_cr is not None:
            for et in self.etypes:
                con_mpi_et = self.i.con_mpi[et]
                if con_mpi_et is None:
                    continue

                owners = np.asarray(con_mpi_et, dtype=np.int64)[..., 0]
                if owners.size == 0:
                    continue

                # neighbour owner ≥ 0, and not myself
                m = (owners >= 0) & (owners != my_wr)
                if not np.any(m):
                    continue

                nbr_world = np.unique(owners[m].astype(np.int64))
                for ow in nbr_world:
                    oc = w2c[ow]
                    if oc is None or oc == my_cr:
                        continue
                    loc[my_cr, oc] = True
                    loc[oc, my_cr] = True

        mask = np.zeros_like(loc, dtype=np.bool_)
        commc.Allreduce(loc, mask, op=mpi.LOR)
        np.fill_diagonal(mask, False)
        return mask




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

    @classmethod
    def from_mesh(cls, mesh: _Mesh) -> "_MetaMesh":

        etypes   = State._mpi_sorted_union(set(mesh.etypes or ()))
        bc_names = State._mpi_sorted_union(set((mesh.bcon or {}).keys()))
        e2i      = {et: i for i, et in enumerate(etypes)}
        bc2id    = {n: i for i, n in enumerate(bc_names)}

        eidxs = {et: np.asarray(mesh.eidxs.get(et, ()), dtype=np.int64)
                for et in etypes}
        con_mpi = {et: np.full((eidxs[et].size, cls._nfaces(et)), _CON_UNFILLED, np.int64) for et in etypes}
        con_idx = {et: np.full((eidxs[et].size, cls._nfaces(et)), _CON_UNFILLED, np.int64) for et in etypes}

        spts_nodes = deepcopy(mesh.spts_nodes)
        # After self.i is fully built:

        mm = cls(mesh_src=mesh, etypes=etypes, e2i=e2i, bc2id=bc2id,
            i=State(eidxs=deepcopy(eidxs), con_mpi=deepcopy(con_mpi), con_idx=deepcopy(con_idx), spts_nodes=deepcopy(spts_nodes),),
            j=State(eidxs=deepcopy(eidxs), con_mpi=deepcopy(con_mpi), con_idx=deepcopy(con_idx), spts_nodes=deepcopy(spts_nodes),)
            )

        mm._encode_con(mesh)
        mm._fill_con_mpi(mesh)

        mm._ne_i = mm._local_count()

        return mm

    # Switch from etype and local indexing within that etype to the global flat indexing
    def etype_local_to_flat(self, etype: str, lids: np.ndarray) -> np.ndarray:
        return self.i.eidxs_flat[self.i.etype_slices[etype].start + lids]

    def _encode_con(self, mesh):
        """
        Encode the *local* (same-rank) mesh connectivity and boundary faces as:

            con_mpi[et][lid, f, 0] = neighbour owner rank (world index), or
                                    -1 for boundary faces
            con_idx[et][lid, f, 0] = neighbour global element ID (>= 0), or
                                    gidx_bc = -(bc_id + 1) for boundaries

        Global element IDs are the GIDs carried in State.eidxs_flat via
        State._eidxs_to_flat.
        """

        my_rank = int(rank['world'])

        conL, conR = (mesh.con or ([], []))
        bcon = getattr(mesh, 'bcon', {}) or {}

        # Interior pairs: both sides are on *this* rank → owner_rank = my_rank
        for (etL, lidL, fL), (etR, lidR, fR) in zip(conL, conR):
            etL = str(etL)
            etR = str(etR)

            # neighbour global IDs via flat mapping
            gR = int(self.etype_local_to_flat(etR, int(lidR)))
            gL = int(self.etype_local_to_flat(etL, int(lidL)))

            self.i.con_mpi[etL][int(lidL), int(fL)] = my_rank
            self.i.con_idx[etL][int(lidL), int(fL)] = gR

            self.i.con_mpi[etR][int(lidR), int(fR)] = my_rank
            self.i.con_idx[etR][int(lidR), int(fR)] = gL

        # Boundary: owner=-1, gidx=gidx_bc<0
        for bcname, triples in (bcon.items() if bcon else []):
            bid = self.bc2id[bcname]
            if bid is None:
                continue

            # strictly negative tag for all BCs
            gidx_bc = -(int(bid) + 1)

            for et, lid, f in triples:
                et = str(et)
                self.i.con_mpi[et][int(lid), int(f)] = -1
                self.i.con_idx[et][int(lid), int(f)] = gidx_bc

                
    def _fill_con_mpi(self, mesh):
        cp = getattr(mesh, "con_p", {}) or {}

        # Local export: {nbr: [(et, lid, fid, gidx_local), ...]}
        local = {
            int(nbr): [
                (
                    str(et),
                    int(lid),
                    int(fid),
                    int(self.etype_local_to_flat(str(et), int(lid))),  # gidx of THIS element
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

            for (etA, lidA, fA, _gidxA), (etB, _lidB, fB, gidxB) in zip(A, B):
                # Store neighbour owner + neighbour global ID
                self.i.con_mpi[etA][lidA, fA] = nbr
                self.i.con_idx[etA][lidA, fA] = int(gidxB)

        nfaces_mpi = sum(len(v) for v in local.values())
        print(f"[con_mpi] rank={my_rank} nfaces_mpi={nfaces_mpi}")


    def _compute_deltas(self, mode):
        if   mode == 'faces':    return self._calc_mpi_faces_deltas()
        elif mode == 'vertices': return self._calc_mpi_vertices_deltas()
        else:
            raise ValueError(f"Unknown {mode = }' ≠ edges/vertices")

    def _calc_mpi_faces_deltas(self) -> dict[int, dict[str, np.ndarray]]:
        """
        Compute per-neighbour per-etype (lid, delta) matrices using the
        owner column stored in con_mpi and neighbour gidx in con_idx.

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
            gidxs  = np.asarray(con_idx_et)

            # Faces that connect to another element (drop BC/invalid faces)
            valid = (gidxs >= 0) & (owners >= 0)

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




    def collect_mpi_vertex_nodes(self) -> dict[int, np.ndarray]:
        """
        Return {nbr_rank: np.ndarray[int64]} = sorted-unique global vertex-node IDs
        on MPI faces to that neighbor.
        """

        if not getattr(self, "_spts_valid", True):
            raise RuntimeError(
                "Vertex-based diffusion requested but spts_nodes are stale.\n"
                "This usually means you ran edge-only smoothing with "
                "move_spts_nodes=False and then called a vertex-based routine "
                "on the same MetaMesh. Rebuild MetaMesh from a fresh mesh, or "
                "keep move_spts_nodes=True in the preceding steps."
            )

        if not hasattr(self, "spts_nodes_i") or self.i.spts_nodes is None:
            raise RuntimeError("self.i.spts_nodes missing")

        # 1) which faces are MPI, grouped by neighbor
        per_nbr_faces = self._mpi_faces_by_neighbor()

        # 2) per-etype face->vertex columns
        face_vtx_by_et = {et: self._face_vertex_indices(et) for et in self.etypes}

        # 3) union vertex nodes per neighbor
        out: dict[int, np.ndarray] = {}
        for nbr, faces in per_nbr_faces.items():
            vset: set[int] = set()
            for et, lid, f in faces:
                nds = self.i.spts_nodes[et]
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

        return out

    def _calc_mpi_vertices_deltas(self) -> dict[int, dict[str, np.ndarray]]:
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
            per = {et: per_et(self.i.spts_nodes[et], vn) for et in self.etypes}
            out[int(nrank)] = per
        return out

    def _mpi_faces_by_neighbor(self) -> dict[int, list[tuple[str, int, int]]]:
        """
        Return {nbr_rank: [(etype, lid, fidx), ...]} for MPI faces only.

        A face is considered MPI if:
            - neighbour owner >= 0,
            - neighbour owner != my_rank,
            - neighbour gidx >= 0 (element neighbour, not BC/invalid).
        """
        my_rank = int(rank['world'])
        per_nbr: dict[int, list[tuple[str, int, int]]] = {}

        for et in self.etypes:
            con_mpi_et = self.i.con_mpi[et]
            con_idx_et = self.i.con_idx[et]

            if con_mpi_et is None or con_idx_et is None:
                continue

            con_mpi_et = np.asarray(con_mpi_et, dtype=np.int64)
            con_idx_et = np.asarray(con_idx_et, dtype=np.int64)

            if con_mpi_et.size == 0 or con_idx_et.size == 0:
                continue

            owners = con_mpi_et[...]
            gidxs  = con_idx_et[...]

            # MPI faces: element neighbour, owner >= 0, owner != my_rank
            mpi_mask = (gidxs >= 0) & (owners >= 0) & (owners != my_rank)

            if not np.any(mpi_mask):
                continue

            lids, fidxs = np.nonzero(mpi_mask)
            lids  = lids.astype(np.int64, copy=False)
            fidxs = fidxs.astype(np.int64, copy=False)

            nbrs = owners[lids, fidxs].astype(np.int64, copy=False)

            uniq_nbrs = np.unique(nbrs)
            for nbr in uniq_nbrs:
                nbr = int(nbr)
                if nbr == my_rank:
                    continue

                sel = (nbrs == nbr)
                lids_n  = lids[sel]
                fidxs_n = fidxs[sel]

                lst = per_nbr.setdefault(nbr, [])
                lst.extend((et, int(lid_i), int(f_i)) for lid_i, f_i in zip(lids_n, fidxs_n))

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

    # ---------------------
    # Relocation iterations 
    # ---------------------

    def _reset_j_with_i(self):
        self.j = self.i.clone()

    def _accept_j_into_i(self):
        self.i, self.j = self.j, self.i
        self._retag_con_owners()
        self._invalidate_topology()

    # -----------------

    def smooth_until_stagnates(self, *, max_iters: int = 20, patience: int = 1,
                            min_change: int = 0, threshold = 0,
                            target_counts: Optional[List[int]] = None,
                            restrict_src_dest: bool = False,
                            move_spts_nodes: bool = False) -> list[int]:
        """
        Run diffuse_smoothing repeatedly. The stopping decision is collective:
        we allreduce the local planned-move counts so every rank takes the same
        number of iterations. Returns the GLOBAL per-iteration move counts.
        """
        history: list[int] = []
        stable = 0
        last = None

        for _ in range(int(max_iters)):
            moved_local = int(self.diffuse_smoothing(return_count=True, threshold=threshold,
                target_counts=target_counts, restrict_src_dest=restrict_src_dest, 
                move_spts_nodes = move_spts_nodes,
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

    def _build_eidxs_diff_from_flat(self, chosen_flat, chosen_nbrs, flat):
        """
        Convert a flat list of element indices + dest ranks into the
        {nbr: {etype: gids[]}} structure expected by relocate_i_to_j.

        Parameters
        ----------
        chosen_flat : array-like of int
            Indices into the flat view (flat.gids / flat.et_index).
        chosen_nbrs : array-like of int
            Destination ranks, same length as chosen_flat.
        flat : object
            Flat view with attributes:
                - gids      : 1D array of global element IDs
                - et_index  : 1D array of integer etype indices
                - etypes    : list of etype names (str)
        """
        # Output: { nbr : { etype_name : np.ndarray[gids] } }
        eidxs_diff = {}

        chosen_flat = np.asarray(chosen_flat, dtype=np.int64)
        chosen_nbrs = np.asarray(chosen_nbrs, dtype=np.int64)

        if chosen_flat.size == 0:
            return eidxs_diff

        # Global IDs and etype indices for the selected elements
        gids_all = np.asarray(flat.gids[chosen_flat], dtype=np.int64)
        et_idx_all = np.asarray(flat.et_index[chosen_flat], dtype=np.int64)

        # Sort once by (nbr, etype_index, gid) so that:
        #   - elements for each (nbr, etype) are contiguous,
        #   - gids within each group are ascending.
        order = np.lexsort((gids_all, et_idx_all, chosen_nbrs))

        nbrs_sorted = chosen_nbrs[order]
        et_sorted   = et_idx_all[order]
        gids_sorted = gids_all[order]

        last_nbr = None
        last_et  = None
        acc_gids = []

        for nbr, et_idx, gid in zip(nbrs_sorted, et_sorted, gids_sorted):
            nbr = int(nbr)
            et_idx = int(et_idx)
            gid = int(gid)

            if last_nbr is None:
                # First entry
                last_nbr, last_et = nbr, et_idx
                acc_gids = [gid]
                continue

            if nbr != last_nbr or et_idx != last_et:
                # Flush previous group
                etname = flat.etypes[last_et]
                per_et = eidxs_diff.setdefault(last_nbr, {})
                per_et[etname] = np.asarray(acc_gids, dtype=np.int64)

                # Start new group
                last_nbr, last_et = nbr, et_idx
                acc_gids = [gid]
            else:
                acc_gids.append(gid)

        # Flush the final group
        if acc_gids:
            etname = flat.etypes[last_et]
            per_et = eidxs_diff.setdefault(last_nbr, {})
            per_et[etname] = np.asarray(acc_gids, dtype=np.int64)

        return eidxs_diff


    def diffuse_smoothing(self, *,
                        return_count: bool = False,
                        threshold: int = 0,
                        target_counts: Optional[List[int]] = None,
                        restrict_src_dest: bool = False,
                        move_spts_nodes: bool = False) -> Optional[int]:
        """
        Core diffusion step working on element faces.
        """

        # 0) start candidate from current
        self._reset_j_with_i()

        # Flat view of current state
        flat_i = self._flat_state('i')
        my_count = int(flat_i.gids.size)

        # Directional gating
        allowed_nbrs = None
        I_am_donor = True
        if restrict_src_dest and target_counts is not None:
            # If you want a one-shot debug check, keep this for a while:
            if False:  # flip to True for a debug run
                dict_count = int(sum(len(self.eidxs_i.get(et, ())) for et in self.etypes))
                print(f"[MetaMesh/debug] rank={rank['world']} flat_count={my_count} dict_count={dict_count}")
                assert my_count == dict_count

            cur = np.asarray(comm['world'].allgather(my_count), dtype=np.int64)
            tgt = np.asarray(target_counts, dtype=np.int64)
            diff = cur - tgt  # +surplus / -deficit
            I_am_donor = diff[rank['world']] > 0

            if I_am_donor:
                # donors may send only to neighbors with deficit
                allowed_nbrs = {int(n) for n in range(comm['world'].size) if diff[int(n)] < 0}
            else:
                # receivers: we will still do the collective relocate (with empty plan)
                allowed_nbrs = set()

        # 1) per-neighbor, per-etype deltas (still produced per-etype)
        deltas_by_rank = self._compute_deltas('faces')

        # Mapping etype -> flat block index
        et_to_idx = {et: i for i, et in enumerate(flat_i.etypes)}
        et_offsets = flat_i.et_offsets

        # 2) collect all candidates in flat index space
        all_flat = []
        all_dlt = []
        all_nbrs = []

        thr_int = int(threshold)

        for nbr, per_et in deltas_by_rank.items():
            nbr = int(nbr)
            if allowed_nbrs is not None and nbr not in allowed_nbrs:
                continue

            for et, arr in per_et.items():
                if arr is None or arr.size == 0:
                    continue

                et_idx = et_to_idx[et]
                if et_idx is None:
                    continue

                lids = arr[:, 0].astype(np.int64, copy=False)
                dlt  = arr[:, 1].astype(np.int64, copy=False)

                if lids.size == 0:
                    continue

                # Map (et, lid) -> flat index
                base = int(et_offsets[et_idx])
                lids_flat = base + lids

                all_flat.append(lids_flat)
                all_dlt.append(dlt)
                all_nbrs.append(np.full(lids_flat.shape, nbr, dtype=np.int64))

        if not all_flat:
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
            return 0 if return_count else None

        lids_flat = np.concatenate(all_flat)
        dlt_all   = np.concatenate(all_dlt)
        nbrs_all  = np.concatenate(all_nbrs)

        # --- arbitration per flat-id (unique element) ---
        # Sort by (flat_id, delta) so we can pick a single best neighbor.
        order = np.lexsort((dlt_all, lids_flat))
        lids_s = lids_flat[order]
        dlt_s  = dlt_all[order]
        nbrs_s = nbrs_all[order]

        _, first_idx = np.unique(lids_s, return_index=True)
        best_flat = lids_s[first_idx]
        best_dlt  = dlt_s[first_idx]
        best_nbrs = nbrs_s[first_idx]

        # threshold semantics: < threshold → with threshold=1, allow Δ<=0
        mkeep = (best_dlt < thr_int)
        if not np.any(mkeep):
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
            return 0 if return_count else None

        chosen_flat = best_flat[mkeep]
        chosen_nbrs = best_nbrs[mkeep]

        # 3) build move plan in classic {nbr: {etype: gids}} format
        eidxs_diff = self._build_eidxs_diff_from_flat(chosen_flat, chosen_nbrs, flat_i)

        # breakdown + count (for debugging / tracing)
        breakdown, tot = {}, 0
        for nbr, per in eidxs_diff.items():
            b = {et: int(len(g)) for et, g in per.items()}
            breakdown[int(nbr)] = b
            tot += sum(b.values())

        role = "donor" if I_am_donor else "receiver"

        # 4) ALWAYS participate in relocate (even with empty plan)
        self._apply_plan_and_commit(eidxs_diff, move_spts_nodes=move_spts_nodes)

        return int(tot) if return_count else None

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
            # Mask already in world space
            mask_full = mask_arr
        else:
            # Treat smaller mask as being defined over ranks [0 .. R_mask-1]
            R_mask = mask_arr.shape[0]
            if mask_arr.shape != (R_mask, R_mask):
                raise ValueError(f"mask must be square; got {mask_arr.shape}")

            # Embed into world-space mask
            mask_full = np.zeros_like(F_full, dtype=bool)
            mask_full[:R_mask, :R_mask] = mask_arr

            # For ranks not covered by the mask (e.g. newly added ranks),
            # allow traffic wherever there is an MPI face.
            if R_mask < R_full:
                mask_full[R_mask:, :] |= (F_full[R_mask:, :] > 0)
                mask_full[:, R_mask:] |= (F_full[:, R_mask:] > 0)

        # Active ranks: those that have or will have elements
        active = [
            i for i, (c, t) in enumerate(zip(cur_full, tgt_full))
            if not (c == 0 and t == 0)
        ]

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

    @staticmethod
    def _validate_flow_matrix(flow_matrix: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Common validation for flow matrices used by smoothing routines.

        Returns
        -------
        M : np.ndarray[int64], shape (R, R)
            Validated, int64 flow matrix.
        R : int
            Number of ranks (world size segment of the matrix).
        """
        M = np.asarray(flow_matrix, dtype=np.int64)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"Need square matrix; got {M.shape = }")
        R = int(M.shape[0])
        if not (0 <= rank['world'] < R):
            raise ValueError(f"{rank['world'] = } but flow_matrix size is {R}")
        return M, R


    def diffuse_smoothing_edges(self, flow_matrix, threshold=2, skip_last=False):
        """
        Edge-based (face-based) smoothing that moves *all-or-none* candidate
        elements per interface (rank -> nbr), repeatedly, until no more
        full-interface moves are possible anywhere.
        """
        # Validate flow matrix
        M0, R = self._validate_flow_matrix(flow_matrix)
        M_rem = M0.astype(np.float64, copy=True)

        # Canonical etype order on THIS rank
        etypes_all = list(self._etype_order())

        # Device-aware “protected-type” logic:
        restrict_last = self._compute_restrict_last(skip_last, etypes_all)

        if skip_last and etypes_all:
            restricted_nbrs = sorted(
                r for r, v in restrict_last.items() if v and r != rank['world']
            )
        else:
            restricted_nbrs = []

        total_moved_local = 0
        thr_int = int(threshold)

        iter_no = 0
        while True:
            iter_no += 1

            # Start from current state for this sweep
            self._reset_j_with_i()

            # 1) Compute fresh face-based deltas on *current* topology
            deltas_by_rank = self._compute_deltas('faces')

            # Flat view for mapping (et, lid) -> mesh-global gid
            flat = self._flat_state('i')
            et_to_idx  = {et: i for i, et in enumerate(flat.etypes)}
            et_offsets = flat.et_offsets
            gids_flat  = flat.gids

            # Track picked elements (by flat index) so they are not exported
            # to multiple neighbours within the same sweep.
            picked_mask = np.zeros_like(gids_flat, dtype=bool)

            eidxs_diff: dict[int, dict[str, np.ndarray]] = {}
            moved_this_sweep_local = 0

            # NEW: per-interface stopping reasons
            per_nbr_reason: dict[int, str] = {}

            for nbr, per_et in deltas_by_rank.items():
                nbr = int(nbr)
                cap_float = float(M_rem[rank['world'], nbr])
                if cap_float <= 0.0:
                    per_nbr_reason[nbr] = (
                        f"cap<=0 (M_rem[{rank['world']},{nbr}]={cap_float:.1f})"
                    )
                    continue

                # Decide which etypes are allowed on THIS interface.
                if skip_last and restrict_last.get(nbr, False) and len(etypes_all) > 0:
                    etypes_active = etypes_all[:-1]
                else:
                    etypes_active = etypes_all

                # For this neighbour, accumulate candidates per etype as flat indices
                per_nbr_flat_idx: dict[str, np.ndarray] = {}
                per_nbr_delta:    dict[str, np.ndarray] = {}

                for et in etypes_active:
                    arr = per_et[et]
                    if arr is None or arr.size == 0:
                        continue

                    lids = arr[:, 0].astype(np.int64, copy=False)
                    dlt  = arr[:, 1].astype(np.int64, copy=False)

                    # Only accept elements whose face-delta passes the threshold.
                    mkeep = (dlt <= thr_int)
                    if not np.any(mkeep):
                        continue

                    lids_k = lids[mkeep]
                    dlt_k  = dlt[mkeep]

                    et_idx = et_to_idx.get(et, None)
                    if et_idx is None:
                        continue

                    base = int(et_offsets[et_idx])
                    flat_idx = base + lids_k

                    # Drop elements already picked for some other neighbour
                    mask_new = ~picked_mask[flat_idx]
                    if not np.any(mask_new):
                        continue

                    lids_k   = lids_k[mask_new]
                    dlt_k    = dlt_k[mask_new]
                    flat_idx = flat_idx[mask_new]

                    if lids_k.size == 0:
                        continue

                    per_nbr_flat_idx[et] = flat_idx
                    per_nbr_delta[et]    = dlt_k

                if not per_nbr_flat_idx:
                    per_nbr_reason[nbr] = ("SKIP no cand")
                    continue

                # All-or-none decision on this interface
                n_cand = sum(idx.size for idx in per_nbr_flat_idx.values())
                need_float = float(n_cand)

                if cap_float + 1e-9 < need_float:
                    per_nbr_reason[nbr] = (
                        f"SKIP {need_float:4.0f}>{cap_float:4.0f}"
                    )
                    continue

                # Enough capacity: move all candidates to this neighbour.
                per_nbr_arr: dict[str, np.ndarray] = {}
                for et, flat_idx in per_nbr_flat_idx.items():
                    dlt_k = per_nbr_delta[et]

                    # Optional: keep deterministic ordering per etype (delta, then gid)
                    order = np.lexsort((gids_flat[flat_idx], dlt_k))
                    flat_idx = flat_idx[order]

                    per_nbr_arr[et] = gids_flat[flat_idx].astype(np.int64, copy=False)

                    # Mark as picked this sweep
                    picked_mask[flat_idx] = True

                eidxs_diff[nbr] = per_nbr_arr
                moved_this_sweep_local += n_cand
                per_nbr_reason[nbr] = (
                    f"MOVE {n_cand:4.0f}<{M_rem[rank['world'], nbr]:4.0f}"
                )
                M_rem[rank['world'], nbr] -= need_float


            # 3) Collectively apply this sweep's relocation
            self._apply_plan_and_commit(eidxs_diff, move_spts_nodes=False)

            # 4) Global convergence check
            moved_glob = comm['world'].allreduce(
                int(moved_this_sweep_local), op=mpi.SUM
            )
            total_moved_local += moved_this_sweep_local

            if rank['compute'] == root['compute']:
                reason_str = ", ".join(
                    f"\t {rank['world']}->{nbr}: {msg}"
                    for nbr, msg in sorted(per_nbr_reason.items())
                )
                print(
                    f"Iter: {iter_no:2.0f} moved_glob = {moved_glob:4.0f} thr_int = {thr_int} :: {reason_str}",
                    flush=True,
                )

            if moved_glob == 0:
                break



    def diffuse_smoothing_vertices(self, flow_matrix, skip_last = False, overshoot = 0.5):
        """
        Vertex-based smoothing that moves *all-or-none* candidate elements
        per interface (rank -> nbr), repeatedly, until no more full-interface
        moves are possible anywhere.
        """
        # Validate flow matrix
        M0, R = self._validate_flow_matrix(flow_matrix)

        # Apply optional overshoot to capacities:
        # M_eff = round((1 + overshoot) * M0), clamped to >= 0.
        if overshoot != 0.0:
            factor = 1.0 + float(overshoot)
            if factor <= 0.0:
                raise ValueError(f"{overshoot = } ≱ 0")
            M_eff = np.rint(M0.astype(np.float64) * factor).astype(np.int64)
            # No negative capacities allowed
            M_eff[M_eff < 0] = 0
        else:
            M_eff = M0.copy()

        # Debug: log overshoot policy once per call per rank
        if overshoot != 0.0:
            cap0_local = int(M0[rank['world']].sum())
            cap_eff_local = int(M_eff[rank['world']].sum())
            print(
                f"[itc4.vs] R{rank['world']} overshoot={overshoot:.3f} "
                f"cap_sum_old={cap0_local} cap_sum_eff={cap_eff_local}",
                flush=True,
            )

        # Remaining capacity; keep as float for safety, but interpret as counts
        M_rem = M_eff.astype(np.float64, copy=True)

        # Canonical etype order on THIS rank
        etypes_all = list(self._etype_order())

        # Device-aware “protected-type” semantics for skip_last (same as edges):
        restrict_last = self._compute_restrict_last(skip_last, etypes_all)

        if skip_last and etypes_all:
            restricted_nbrs = sorted(
                r for r, v in restrict_last.items() if v and r != rank['world']
            )
        else:
            restricted_nbrs = []

        # Debug policy (same formatting as before)
        print(
            f"R{rank['world']} setup skip_last={bool(skip_last)} "
            f"etypes_all={etypes_all} restricted_nbrs={restricted_nbrs}",
            flush=True,
        )

        iter=  0
        #for iter in range(int(len(rankmap['world']))):
        while True: 
            iter += 1
            # Start from current state for this sweep
            self._reset_j_with_i()

            # 1) Compute fresh vertex-based deltas on *current* topology
            deltas_by_rank = self._compute_deltas('vertices')

            # Flat view for mapping (et, lid) -> gid
            flat = self._flat_state('i')
            et_to_idx = {et: i for i, et in enumerate(flat.etypes)}
            et_offsets = flat.et_offsets
            gids_flat = flat.gids

            picked_by_et: dict[str, set[int]] = {
                et: set() for et in etypes_all
            }

            # 2) Build eidxs_diff for this sweep with all-or-none per interface
            eidxs_diff: dict[int, dict[str, np.ndarray]] = {}
            moved_this_sweep_local = 0

            # For each neighbor, select candidates (no threshold) and see if
            # we can move them all under remaining capacity.
            for nbr, per_et in deltas_by_rank.items():
                nbr = int(nbr)
                cap_float = float(M_rem[rank['world'], nbr])
                if cap_float <= 0.0:
                    continue

                # Decide which etypes are active on THIS interface.
                if skip_last and restrict_last.get(nbr, False) and len(etypes_all) > 0:
                    etypes_active = etypes_all[:-1]
                else:
                    etypes_active = etypes_all

                # Collect all candidate gids across *active* etypes,
                # ordered by delta (for determinism).
                cand: list[tuple[int, str, int]] = []  # (delta, etype, gid)

                for et in etypes_active:
                    arr = per_et[et]
                    if arr is None or arr.size == 0:
                        continue

                    lids = arr[:, 0].astype(np.int64, copy=False)
                    dlt  = arr[:, 1].astype(np.int64, copy=False)

                    et_idx = et_to_idx[et]
                    if et_idx is None:
                        continue
                    base = int(et_offsets[et_idx])

                    for lid, d in zip(lids, dlt):
                        gid = int(gids_flat[base + int(lid)])
                        if gid in picked_by_et[et]:
                            continue
                        cand.append((int(d), et, gid))

                if not cand:
                    continue

                # Sort by delta (for determinism); we will try to move *all*.
                cand.sort(key=lambda t: (t[0], t[2]))
                n_cand = len(cand)

                # Required capacity if we want to move ALL candidates:
                need_float = float(n_cand)

                if cap_float + 1e-9 < need_float:
                    # Cannot move all; skip this interface entirely
                    continue

                # We have enough capacity to move all candidates to this nbr
                per_nbr: dict[str, list[int]] = {}
                for _, et, gid in cand:
                    picked_by_et[et].add(gid)
                    per_nbr.setdefault(et, []).append(gid)

                # Convert to sorted arrays
                per_nbr_arr = {et: np.asarray(sorted(gids), dtype=np.int64)
                                   for et, gids in per_nbr.items() if gids}

                if per_nbr_arr:
                    eidxs_diff[nbr] = per_nbr_arr
                    moved_this_sweep_local += sum(len(g) for g in per_nbr_arr.values())
                    # Deduct capacity used (exact integer count)
                    M_rem[rank['world'], nbr] -= float(n_cand)

            # 3) Collectively apply this sweep's relocation
            self._apply_plan_and_commit(eidxs_diff, move_spts_nodes=True)

            # 4) Global convergence check
            moved_glob = comm['world'].allreduce(int(moved_this_sweep_local),
                                                 op=mpi.SUM)

            if rank['compute'] == root['compute']:
                print(f"Iter {iter+1} {moved_glob = }", flush=True)

            if moved_glob == 0:
                break

    def _locate_elements(self, etype, gids):
        """Return arrays of (owner_rank, local_idx) for given gids."""
        lut = self._gid_lookup[etype]
        owners = []
        lidxs  = []
        for gid in gids:
            owner, lid = lut[int(gid)]
            owners.append(owner)
            lidxs.append(lid)
        return np.array(owners, dtype=np.int32), np.array(lidxs, dtype=np.int64)

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

    def seed_rank(self, new_rank: int, targets, twoway_mask, 
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
        cur0 = self._cur_counts_total()
        
        if this_rank == root['world']:
            print(f"[itc4.seed] mode=seed_rank CURRENT={cur0}", flush=True)

        M0 = self.element_flow_plan(cur0, targets, twoway_mask)

        self.diffuse_smoothing_vertices(flow_matrix=M0, skip_last=True)

    def iterate(self, objective, target_counts, mask, *, flowmat_relax=0.5, ):
        if rank['world'] == root['world']: print(f"TARGET={target_counts}")

        if objective == 'to-target':
            exec_order = (#[('vertices',6)] 
                            [('faces'   ,6)]
                          + [('faces'   ,5)]
                          + [('faces'   ,4)]
                          + [('faces'   ,3)]
                          + [('faces'   ,2)]
                          + [('faces'   ,1)]
                          + [('faces'   ,0)])
            for step in exec_order:
                cur0 = self._cur_counts_total()
                if rank['world'] == root['world']: print(f"mode={step[0]} \t thr={step[1]} CURRENT={cur0}")
                M0   = self.element_flow_plan(cur0, target_counts, mask)

                if   step[0] == 'vertices': self.diffuse_smoothing_vertices(M0,skip_last=True)
                elif step[0] == 'faces':    self.diffuse_smoothing_edges(M0, threshold=step[1])
                    #self.diffuse_smoothing2(M0, mode='faces', threshold=step[1],
                    #                        scale=flowmat_relax)

                self.smooth_until_stagnates()
                
        elif objective == 'to-remove-rank':

            kill_rank = [r for r, c in enumerate(target_counts) if c == 0][0]

            if rank['world'] == root['world']: print(f"{kill_rank = } ", flush=True,)

            # Evacuate kill_rank until it reaches zero
            while True:
                cur0 = self._cur_counts_total()
                M0 = self.element_flow_plan(cur0, target_counts, mask)

                # Use vertex-based diffusion here, as in your sketch
                self.diffuse_smoothing_vertices(M0)
                self.smooth_until_stagnates(move_spts_nodes=True)

                cur1 = self._cur_counts_total()

                diff = [cur1[i] - cur0[i] for i in range(len(cur0))]
                if rank['world'] == root['world']:
                    print(f"[itc4.iter] to-remove-rank CURRENT={cur1} \t DIFF = {diff}", flush=True,)

                if cur1[kill_rank] == 0:
                    if rank['world'] == root['world']:
                        print(f"to-remove-rank done; kill_rank={kill_rank} cur={cur1}", flush=True,)
                    break

                # If diff is all zeros, we are stuck, break to avoid infinite loop
                if all(d == 0 for d in diff):
                    if rank['world'] == root['world']:
                        print(f"to-remove-rank stuck; kill_rank={kill_rank} cur={cur1}", flush=True,)
                    break
        else:
            raise ValueError(f"Unknown objective '{objective}'")

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
        before_local = self._local_count()

        # Only the two ranks participate in the sendrecv; others are spectators
        if myr == r0 or myr == r1:
            partner    = r1 if myr == r0 else r0
            send_state = self.i
            # Send our State, receive partner's State (pickled Python object)
            recv_state = commw.sendrecv(send_state, dest=partner, source=partner)
            # Overwrite local State
            self.i = recv_state

        after_local = self._local_count()

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

        # Determine Ne_global via max(global_id) + 1
        all_max = cw.allgather(int(ids_local.max()) if ids_local.size else -1)
        Ne_global = (max(all_max) + 1) if all_max else 0

        owners_global = np.empty(Ne_global, dtype=np.int32)
        owners_global.fill(-1)

        # Enumerate across allgathered global IDs
        for rr, gids_rr in enumerate(cw.allgather(ids_local)):
            gids_rr = np.asarray(gids_rr, dtype=np.int64)
            if gids_rr.size == 0:
                continue
            owners_global[gids_rr] = rr

        self._cache['owner_global'] = {'gen': gen, 'owners': owners_global,}

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

            gidx_flat   = con_idx_arr[...].reshape(-1)
            owners_flat = con_mpi_arr[...].reshape(-1)

            # Valid: element-neighbour faces (global ID >= 0)
            valid = (gidx_flat >= 0)
            nfaces_valid = int(valid.sum())

            if nfaces_valid == 0:
                continue

            owners_flat[valid] = owners_global[gidx_flat[valid]].astype(np.int32)

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
        return N_int.tolist()

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
        mcols = [f"i{i}-{j}" for i in range(P_world) for j in range(P_world) if j != i]
        _append_csv_row('g1-send-median-ms.csv', mcols, 
                        [int(send_us_w[i, j]) for i in range(P_world) for j in range(P_world) if j != i])
        _append_csv_row('g1-recv-median-ms.csv', mcols, 
                        [int(recv_us_w[i, j]) for i in range(P_world) for j in range(P_world) if j != i])

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

class _MetaMeshInterconnector(AlltoallMixin):
    """
    Fast, MetaMesh-specific interconnector for inner diffusion iterations.

    - No src_all/dest_all allgather.
    - Per-etype "request then send" protocol over alltoallv.
    - Currently specialised for element-wise arrays (axis 0 = element),
      and we only use it for con_idx.
    """
    
class _MetaMeshInterconnector(AlltoallMixin):
    """
    Fast, MetaMesh-specific interconnector for inner diffusion iterations.

    Uses the same mesh-global flat indexing as State._eidxs_to_flat.
    """

    def __init__(self,
                 eidxs_src: Dict[str, np.ndarray],
                 eidxs_dest: Dict[str, np.ndarray],
                 etypes,
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
            self._src_flat, self._src_flat_slices = State._eidxs_to_flat(eidxs_src)

        # Destination side: flat view for possible later use / checks
        self._dest_flat, self._dest_flat_slices = State._eidxs_to_flat(eidxs_dest)

        # Per-etype global offsets: gidx = base[et] + gid
        self._gidx_base: Dict[str, int] = {}
        for et in self.etypes:
            gids_local = np.asarray(self.eidxs_src.get(et, ()), dtype=np.int64)
            sl = self._src_flat_slices[et]
            if gids_local.size and sl.stop > sl.start:
                gidx_local = self._src_flat[sl]
                # Base is constant: gidx = base + gid
                self._gidx_base[et] = int(gidx_local[0] - gids_local[0])
            else:
                self._gidx_base[et] = 0

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
