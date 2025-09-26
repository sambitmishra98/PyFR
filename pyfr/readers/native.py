from __future__ import annotations

from dataclasses import dataclass, field, replace
import re

import h5py
import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import (Scatterer, SparseScatterer, autofree,
                          get_comm_rank_root, mpi,
                          AlltoallMixin)
from pyfr.nputil import iter_struct

from dataclasses import dataclass, field
from typing import Dict, List

from pyfr.relocator.utils import crprint

from pyfr.util import subclass_where
from pyfr.shapes import BaseShape

from copy import deepcopy

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

    con: list = field(default_factory=list)
    con_p: dict = field(default_factory=dict)
    bcon: dict = field(default_factory=dict)

_CON_UNFILLED = -2
_BC_NONE      = -1

@dataclass
class _MetaMesh:

    etypes: List[str]                 = field(default_factory=list)

    # Encoders/decoders for etypes and BCs
    e_to_i    : Dict[str, int]        = field(default_factory=dict)
    bc_name2id: Dict[str, int]        = field(default_factory=dict)

    eidxs: Dict[str, List[int]] = field(default_factory=dict)

    # con[etype]: (Ng(et), nfaces(et), 3): (n_code, n_gid, n_fid)
    # boundary encoded as (-1, bc_id, -1)
    con          : Dict[str, np.ndarray] = field(default_factory=dict)
    is_mpi_edge  : Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    is_mpi_vertex: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    
    _full_reloc_plan: Dict[str, dict] = field(default_factory=dict)
    _saved_perms: Dict[str, np.ndarray] = field(default_factory=dict)

    # Encode–decode helpers
    def et_id(self, et):     return self.e_to_i[et]
    def et_name(self, code): return self.etypes[int(code)]
    def bc_id(self, name):   return self.bc_name2id[name]
    def bc_name(self, bcid): 
        for n, i in self.bc_name2id.items():
            if i == int(bcid):
                return n
        raise KeyError(f"Unknown bc_id {bcid}")

    @classmethod
    def from_mesh(cls, mesh):
        comm, rank, root = get_comm_rank_root()

        local_etypes = set(getattr(mesh, "etypes", []) or [])
        etypes = _MetaMesh._mpi_sorted_union(local_etypes, comm)

        e_to_i = {et: i for i, et in enumerate(etypes)}

        local_bcnames = set((getattr(mesh, "bcon", {}) or {}).keys())
        bc_names = _MetaMesh._mpi_sorted_union(local_bcnames, comm)

        bc_name2id = {name: i for i, name in enumerate(bc_names)}

        eidxs = {et: np.array(mesh.eidxs.get(et, []), dtype=int) for et in etypes}
        con   = {et: np.full((eidxs[et].size, cls._nfaces(et), 3),
                              _CON_UNFILLED, np.int64) for et in etypes}

        mm = cls(etypes=etypes, e_to_i=e_to_i, bc_name2id=bc_name2id,
                 eidxs=eidxs, con=con)

        mm._populate_con_local(mesh)
        mm._populate_con_mpi(mesh)

        mm.is_mpi_edge   = mm.compute_is_mpi_edge(mesh)
        mm.is_mpi_vertex = mm.compute_is_mpi_vertex(mesh)

        #mv = mm.get_mpi_vertex_union(mesh)
        #touch_map = mm.get_mpi_touch_elements(mesh)
        #crprint(-1, f'MPI-touch by neighbor (GIDs): {touch_map}')

        return mm

    def compute_is_mpi_edge(self, mesh) -> Dict[int, Dict[str, np.ndarray]]:
        """
        For each neighbor rank -> {etype -> boolean mask over local elements},
        where True means the element has at least one *face* shared with that neighbor.
        """
        con_p = getattr(mesh, "con_p", {}) or {}
        out: dict[int, dict[str, np.ndarray]] = {}

        # Pre-size masks per neighbor/etype
        for nbr, faces in con_p.items():
            nbr = int(nbr)
            if nbr not in out:
                out[nbr] = {et: np.zeros(self.eidxs[et].size, dtype=bool)
                            for et in self.etypes}

            m = out[nbr]
            for et, lid, f in faces:
                et  = str(et)
                lid = int(lid)
                if 0 <= lid < m[et].size:
                    m[et][lid] = True

        return out

    def compute_is_mpi_vertex(self, mesh) -> Dict[int, Dict[str, np.ndarray]]:
        """
        For each neighbor rank -> {etype -> boolean mask over local elements},
        True if the element touches (shares a *vertex node* with) any inter-rank face
        to that neighbor.
        """
        con_p = getattr(mesh, "con_p", {}) or {}
        spts_nodes = getattr(mesh, "spts_nodes", {}) or {}

        # Precompute face→vertex columns for each etype
        face_vtx_by_et = {et: self._face_vertex_indices(et) for et in self.etypes}

        out: dict[int, dict[str, np.ndarray]] = {}

        for nbr, faces in con_p.items():
            nbr = int(nbr)

            # 1) collect vertex node IDs on MPI faces to this neighbor
            mpi_nodes: set[int] = set()
            for et, lid, f in faces:
                et  = str(et); lid = int(lid); f = int(f)
                nds = spts_nodes.get(et, None)
                if nds is None or nds.size == 0 or not (0 <= lid < nds.shape[0]):
                    continue
                fv = face_vtx_by_et[et][f]                 # vertex column indices
                verts = np.asarray(nds[lid, fv], dtype=np.int64).ravel()
                mpi_nodes.update(int(v) for v in verts)

            # 2) build per-etype boolean masks by checking vertex incidence
            out[nbr] = {}
            if not mpi_nodes:
                # still populate empty masks for consistency
                for et in self.etypes:
                    out[nbr][et] = np.zeros(self.eidxs[et].size, dtype=bool)
                continue

            mpi_nodes_arr = np.fromiter((int(v) for v in mpi_nodes),
                                        count=len(mpi_nodes), dtype=np.int64)

            for et in self.etypes:
                nds = spts_nodes.get(et, None)
                if nds is None or nds.size == 0:
                    out[nbr][et] = np.zeros(self.eidxs[et].size, dtype=bool)
                    continue

                # Each row = element’s vertex node IDs. Mark True if any vertex ∈ mpi_nodes.
                # Efficient test via broadcasting/searchsorted on sorted set:
                #  - sort node set once
                #  - for each element row, check any membership
                nn = np.unique(mpi_nodes_arr)              # sorted
                # Flatten membership check without Python loops
                # Map each vertex to membership via binary search
                idx = np.searchsorted(nn, nds, side='left')
                in_set = (idx < nn.size) & (nn.take(np.clip(idx, 0, nn.size-1)) == nds)
                out[nbr][et] = np.any(in_set, axis=1)

        return out

    def _empty_plan(self):
        """etype -> dest_rank -> empty int64 arrays (shape (0,))"""
        comm, _, _ = get_comm_rank_root()
        return {et: {r: np.empty(0, np.int64) for r in range(comm.size)}
                for et in self.etypes}

    def elements_for_transfer(self, src_rank: int, dst_rank: int, *, mode: str = "vertex"):
        """
        Return, for a given (src_rank -> dst_rank) pair, the local element GIDs
        (on src_rank) that should be sent to dst_rank under the chosen 'mode'.

        Parameters
        ----------
        src_rank : int
        Rank that would send elements.
        dst_rank : int
        Rank that would receive elements.
        mode : {"vertex","edge"}
        Use per-neighbor masks from is_mpi_vertex or is_mpi_edge.

        Returns
        -------
        dict[str, np.ndarray]
        etype -> np.int64[GIDs-to-send from src->dst] (unique, sorted).
        Returns empty arrays if nothing matches or not running on src_rank.
        """
        comm, rank, _ = get_comm_rank_root()

        # Only the source rank computes its outgoing list
        if rank != int(src_rank):
            return {et: np.empty(0, np.int64) for et in self.etypes}

        # Pick the mask dictionary (neighbor-> per-etype bool arrays)
        mpi_flag = self.is_mpi_vertex if mode == "vertex" else self.is_mpi_edge
        neigh = mpi_flag.get(int(dst_rank), {})  # per-etype boolean masks, sized to local elems

        out = {}
        for et in self.etypes:
            mask = neigh.get(et)
            if mask is None or not np.any(mask):
                out[et] = np.empty(0, np.int64)
                continue
            # Unique, sorted GIDs for local elements marked by the mask
            gids = np.unique(self.eidxs[et][mask]).astype(np.int64)
            out[et] = gids
        return out

    def create_plan_local(self, target_rank: int, *, mode: str = "vertex"):
        """
        Build a *local* relocation plan keyed by DEST rank.
        On non-target ranks: send all local elements that touch a vertex/edge
        shared with `target_rank` to that target. On target rank: send nothing.

        Returns
        -------
        dict[str, dict[int, np.ndarray]]
            etype -> dest_rank -> np.int64[GIDs-to-send]
        """
        comm, rank, _ = get_comm_rank_root()
        plan = self._empty_plan()

        if rank == int(target_rank):
            return plan  # target sends nothing

        # Use the pairwise helper to compute my src(=rank) -> dst(=target) list
        pair = self.elements_for_transfer(src_rank=rank, dst_rank=int(target_rank), mode=mode)

        for et, gids in pair.items():
            plan[et][int(target_rank)] = np.ascontiguousarray(gids, dtype=np.int64)

        return plan



    def _dbg_rank_et_counts(self, label: str) -> None:
        """Print per-rank, per-etype element counts and MPI-interface counts."""
        from pyfr.mpiutil import get_comm_rank_root
        import numpy as np
        comm, rank, _ = get_comm_rank_root()
        R = comm.size

        lines = []
        for et in self.etypes:
            own = self.placements[et][:, 0].astype(np.int64, copy=False)
            cnt = np.bincount(own, minlength=R)
            iface_mask = self.mpi_iface.get(et, np.zeros_like(own, dtype=bool))
            icnt = np.bincount(own[iface_mask], minlength=R) if iface_mask.any() else np.zeros(R, np.int64)
            lines.append(f"{et}: cnt={cnt.tolist()} iface={icnt.tolist()}")

        if rank == 0:
            print(f"[counts] {label}")
            for s in lines:
                print(f"[counts] {s}")

    def _dbg_plan_preview(self, M: np.ndarray, label: str = "plan") -> None:
        """Print nonzero move edges and totals."""
        from pyfr.mpiutil import get_comm_rank_root
        import numpy as np
        comm, rank, _ = get_comm_rank_root()
        if rank != 0: return
        edges = [(int(i), int(j), int(M[i,j])) for i in range(M.shape[0]) for j in range(M.shape[1]) if i!=j and M[i,j] > 0]
        if edges:
            items = " ".join(f"{i}->{j}:{v}" for i, j, v in edges)
            print(f"[moves] {label} {items}")
        else:
            print(f"[moves] {label} (none)")

    def _dbg_after_reloc_shapecheck(self, plan_by_etype: dict, edict_src: dict[str, np.ndarray], edim: int, label: str):
        """Sanity prints before Alltoallv and after receive ordering."""
        from pyfr.mpiutil import get_comm_rank_root
        comm, rank, _ = get_comm_rank_root()
        if rank == 0:
            print(f"[reloc] {label} edim={edim}")
            for et, p in plan_by_etype.items():
                sc = int(np.asarray(p['scount']).sum()); rc = int(np.asarray(p['rcount']).sum())
                print(f"[reloc] et={et} scount_sum={sc} rcount_sum={rc} src_local={edict_src[et].shape}")


    # Inside class _MetaMesh
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


    def get_mpi_vertices(self, mesh):
        """
        Build a dict mapping neighbor-rank -> {etype -> list of arrays},
        where each array holds the *global vertex node IDs* for one inter-rank
        face of that etype on this rank.

        Returns
        -------
        dict[int, dict[str, list[np.ndarray]]]
        """
        con_p = getattr(mesh, "con_p", {}) or {}
        spts_nodes = getattr(mesh, "spts_nodes", {}) or {}

        out: dict[int, dict[str, list[np.ndarray]]] = {}

        # Precompute per-etype face→vertex column indices
        face_vtx_by_et = {et: self._face_vertex_indices(et) for et in self.etypes}

        for nbr, faces in con_p.items():
            nbr = int(nbr)
            etmap: dict[str, list[np.ndarray]] = {}

            for et, lid, f in faces:
                et = str(et); lid = int(lid); f = int(f)

                nds = spts_nodes.get(et, None)
                if nds is None or nds.size == 0:
                    continue
                if lid < 0 or lid >= nds.shape[0]:
                    continue

                fv = face_vtx_by_et[et][f]                 # vertex-columns for this face
                # inside get_mpi_vertices, after face_nodes is built
                face_nodes = np.sort(np.asarray(nds[lid, fv], dtype=np.int64))
                etmap.setdefault(et, []).append(face_nodes)

            if etmap:
                out[nbr] = etmap

        # keep a copy for inspection
        self.mpi_vertices = out
        return out

    def get_mpi_touch_elements(self, mesh=None):
        """
        Using self.mpi_vertices / self.mpi_vertex_union, find all local elements
        that touch at least one MPI-interface vertex.

        Returns
        -------
        per_nbr_touch : dict[int, dict[str, np.ndarray]]
            {neighbor_rank -> {etype -> array of element GIDs (int64)}}
        Also sets:
            self.is_mpi_touch : dict[str, np.ndarray]   # boolean mask per etype
        """
        # Ensure we have the per-face dict and per-neighbor union
        if not getattr(self, 'mpi_vertices', None) or not getattr(self, 'mpi_vertex_union', None):
            if mesh is None:
                raise RuntimeError("get_mpi_touch_elements: need mesh to compute mpi vertices/union")
            self.get_mpi_vertices(mesh)
            self.get_mpi_vertex_union(mesh)

        spts_nodes = getattr(mesh, "spts_nodes", {}) or {}
        per_nbr_touch: dict[int, dict[str, np.ndarray]] = {}
        is_touch_agg: dict[str, np.ndarray] = {et: np.zeros(self.eidxs[et].size, dtype=bool)
                                            for et in self.etypes}

        for nbr, node_union in self.mpi_vertex_union.items():
            node_union = np.asarray(node_union, dtype=np.int64)
            etmap: dict[str, np.ndarray] = {}

            for et in self.etypes:
                nds = spts_nodes.get(et, None)
                if nds is None or nds.size == 0:
                    # no geometry nodes locally for this etype
                    etmap[et] = np.empty(0, dtype=np.int64)
                    continue

                # nds: shape (Ne_local, nverts_et).  Mark elements that use any union node.
                touch_mask = np.isin(nds, node_union, assume_unique=False).any(axis=1)

                # Record per-neighbor GIDs of touching elements
                if np.any(touch_mask):
                    lids = np.nonzero(touch_mask)[0].astype(np.int64, copy=False)
                    gids = self.eidxs[et][lids].astype(np.int64, copy=False)
                    etmap[et] = gids
                    # Update aggregate mask
                    is_touch_agg[et][lids] = True
                else:
                    etmap[et] = np.empty(0, dtype=np.int64)

            per_nbr_touch[int(nbr)] = etmap

        # Save aggregate boolean mask on the object for quick access later
        self.is_mpi_touch = is_touch_agg

        # Optional debug prints
        crprint(-1, f"[r] MPI-touch elements per neighbor (GIDs): {per_nbr_touch}")
        crprint(-1, f"[r] is_mpi_touch masks: {{et: mask.sum() for et, mask in self.is_mpi_touch.items()}}")

        return per_nbr_touch

    def get_mpi_vertex_union(self, mesh=None):
        """
        Collapse self.mpi_vertices into {neighbor_rank -> np.array(unique_node_ids)}.
        If self.mpi_vertices is empty, compute it from `mesh` first.

        Returns
        -------
        dict[int, np.ndarray]  # each array is int64, sorted unique
        """
        # Populate mpi_vertices if missing
        if not getattr(self, 'mpi_vertices', None):
            if mesh is None:
                raise RuntimeError("get_mpi_vertex_union: need mesh when mpi_vertices is empty")
            self.get_mpi_vertices(mesh)

        out: dict[int, np.ndarray] = {}

        for nbr, etmap in self.mpi_vertices.items():
            nodes_flat = []

            # etmap: {etype -> [array([v0, v1, ...]), array([...]), ...]}
            for arr_list in etmap.values():
                for arr in arr_list:
                    a = np.asarray(arr, dtype=np.int64).ravel()
                    if a.size:
                        nodes_flat.append(a)

            if nodes_flat:
                uni = np.unique(np.concatenate(nodes_flat))
            else:
                uni = np.empty(0, dtype=np.int64)

            out[int(nbr)] = uni

        # Optionally stash for later use
        self.mpi_vertex_union = out

        return out

    def compute_is_mpi(self):
        local = {et: set(self.eidxs[et].astype(np.int64).tolist()) for et in self.etypes}
        out = {}
        for et in self.etypes:
            c = self.con[et]
            if c.size == 0:
                out[et] = np.zeros(0, dtype=bool)
                continue
            ncode = c[...,0].astype(np.int64, copy=False)
            ngid  = c[...,1].astype(np.int64, copy=False)

            is_mpi_face = np.zeros_like(ncode, dtype=bool)
            for code in np.unique(ncode[ncode >= 0]):
                m = (ncode == int(code))
                gids = ngid[m]
                net = self.etypes[int(code)]
                loc = np.fromiter((g in local[net] for g in gids), count=gids.size, dtype=bool)
                tmp = np.zeros_like(m, dtype=bool); tmp[m] = ~loc
                is_mpi_face |= tmp

            out[et] = np.any(is_mpi_face, axis=1)
        self.is_mpi = out

    def curved_flags_from_mesh(self, mesh):
        # TODO: Simplify
        comm, _, _ = get_comm_rank_root()
        out = {}
        for et in self.etypes:
            # local map from mesh’s CURRENT distribution
            loc_gids = np.asarray(mesh.eidxs.get(et, ()), dtype=np.int64)
            loc_curv = np.asarray(mesh.spts_curved.get(et, np.zeros(loc_gids.size, dtype=bool)), dtype=bool)
            pairs = np.c_[loc_gids, loc_curv.view(np.uint8)] if loc_gids.size else np.empty((0,2), np.int64)
            all_pairs = comm.allgather(pairs)

            gid2curv = {}
            for a in all_pairs:
                if a.size:
                    for g, f in a:
                        gid2curv[int(g)] = bool(int(f))

            gids_now = self.eidxs[et].astype(np.int64, copy=False)
            out[et] = np.fromiter((gid2curv.get(int(g), False) for g in gids_now),
                                count=gids_now.size, dtype=bool)
        return out

    def _apply_perm_to_edict(self, edict, perms, *, edim):
        out = {}
        for et, arr in (edict or {}).items():
            if arr is None:
                continue
            a = np.asarray(arr)
            if a.size == 0:
                out[et] = a
                continue
            # move element axis to 0, permute, move back
            if edim != 0:
                a = np.moveaxis(a, edim, 0)
            p = perms.get(et, np.arange(a.shape[0], dtype=np.int64))
            a = a[p]
            if edim != 0:
                a = np.moveaxis(a, 0, edim)
            out[et] = a
        return out

    # MPI --> curved --> others reordering applied to eidxs+con ---
    def reorder_local_by_mpi_curved(self, curved_by_et, *, return_perm=False):
        perms = {}
        for et in self.etypes:
            Ne = self.eidxs[et].size
            if Ne == 0:
                perms[et] = np.arange(0, 0, dtype=np.int64)
                continue
            mpi    = self.is_mpi.get(et, np.zeros(Ne, dtype=bool))
            curved = np.asarray(curved_by_et.get(et, np.zeros(Ne, dtype=bool)), dtype=bool)
            idx    = np.arange(Ne, dtype=np.int64)
            # MPI first, then curved first, stable on original order
            perm = np.lexsort((idx, ~curved, ~mpi))
            self.eidxs[et] = self.eidxs[et][perm]
            self.con[et]   = self.con[et][perm]
            perms[et] = perm

        self._saved_perms = {et: np.asarray(p, dtype=np.int64) for et, p in perms.items()}

        return perms if return_perm else None

    # --- modify to_mesh to use the above before materializing the mesh ---
    def to_mesh(self, mesh):
        # (Optional) reorder locally first, if you want MPI-first/curved-first:
        self.compute_is_mpi()
        curved = self.curved_flags_from_mesh(mesh)

        perms = self.reorder_local_by_mpi_curved(curved, return_perm=True)

        # Relocate geometry using the saved full plan (if present)
        spts, spts_nodes, spts_curved = self.relocate_geometry_with_saved_plan(mesh)

        spts        = self._apply_perm_to_edict(spts,        perms, edim=1)
        spts_nodes  = self._apply_perm_to_edict(spts_nodes,  perms, edim=0)
        spts_curved = self._apply_perm_to_edict(spts_curved, perms, edim=0)

        # Rebuild native connectivity structs from compact con
        (conL, conR), con_p, bcon = self._to_pyfr_connectivity()

        # Finalize
        new_eidxs = {et: self.eidxs[et].astype(np.int64, copy=False) for et in self.etypes}
        return _Mesh(
            fname=mesh.fname, raw=mesh.raw, ndims=mesh.ndims, subset=mesh.subset,
            creator=mesh.creator, codec=mesh.codec, uuid=mesh.uuid, version=mesh.version,
            etypes=self.etypes,
            eidxs=new_eidxs,
            spts=spts,
            spts_nodes=spts_nodes,
            spts_curved=spts_curved,
            con=(conL, conR),
            con_p=con_p,
            bcon=bcon
        )

    def _populate_con_local(self, mesh):
        conL, conR = (mesh.con or ([], []))
        bcon = getattr(mesh, "bcon", {}) or {}

        # Connect all in interior
        # encoding: (etype_code, neighbor_gid, neighbor_fid)
        for (etL, lidL, fL), (etR, lidR, fR) in zip(conL, conR):
            self.con[etL][lidL, fL] = (self.e_to_i[etR], int(self.eidxs[etR][lidR]), int(fR))
            self.con[etR][lidR, fR] = (self.e_to_i[etL], int(self.eidxs[etL][lidL]), int(fL))

        # boundary: (-1, bc_id, -1)
        for bcname, triples in bcon.items():
            bcid = self.bc_name2id.get(bcname, None)
            if bcid is None:
                continue
            for et, lid, f in triples:
                self.con[et][lid, f] = (_BC_NONE, int(bcid), _BC_NONE)

    def _populate_con_mpi(self, mesh):
        comm, myrank, _ = get_comm_rank_root()

        # Local: con_p {nbr: [(et,lid,f), ...]}  → encode our side with *GIDs*
        cp = getattr(mesh, "con_p", {}) or {}
        local = {}
        for nbr, faces in cp.items():
            rec = []
            for et, lid, f in faces:
                gid = int(self.eidxs[et][int(lid)])
                rec.append((str(et), int(lid), int(f), int(gid)))
            local[int(nbr)] = rec

        # Allgather once
        all_cp = comm.allgather(local)

        # For each rank-pair (r,s), only write rows owned by *this* rank
        for r, rmap in enumerate(all_cp):
            for s, A in rmap.items():
                if s <= r:
                    continue
                B = all_cp[s].get(r, [])
                # A and B are order-matched by construction of con_p in PyFR

                if myrank == r:
                    for (etA, lidA, fA, gidA), (etB, lidB, fB, gidB) in zip(A, B):
                        self.con[etA][lidA, fA] = (self.e_to_i[etB], gidB, fB)

                if myrank == s:
                    for (etA, lidA, fA, gidA), (etB, lidB, fB, gidB) in zip(A, B):
                        self.con[etB][lidB, fB] = (self.e_to_i[etA], gidA, fA)

    @staticmethod
    def _mpi_sorted_union(local, comm):
        return sorted(set().union(*comm.allgather(set(local))))

    @staticmethod
    def _nfaces(et):
        return len(subclass_where(BaseShape, name=et).faces)

    def preproc_edict(self, edict_src, *, edim: int = 0):
        """
        Ensure every etype exists and the *element axis is first* (axis 0).
        If an etype is empty locally, infer (dtype, shape) from any rank that has data.
        Returns a dict {et: np.ndarray with axis-0 = elements}.
        """
        comm, _, _ = get_comm_rank_root()
        out = {}

        for et in self.etypes:
            ary = edict_src.get(et, None)
            have = (ary is not None)
            if have:
                ary = np.asarray(ary)
                if edim:
                    if ary.ndim <= edim:
                        raise ValueError(f"{et}: expected at least {edim+1} dims, got {ary.ndim}")
                    ary = np.moveaxis(ary, edim, 0)
                local_sig = (ary.dtype, ary.shape) if ary.size else (None, None)
            else:
                local_sig = (None, None)

            sigs = comm.allgather(local_sig)
            dt, sh = next(((d, s) for (d, s) in sigs if d is not None), (None, None))
            if dt is None:
                # All ranks empty → create a canonical empty (0,) array
                out[et] = np.empty((0,), dtype=np.float64)
                continue

            if have and ary.size:
                out[et] = ary.astype(dt, copy=False)
            else:
                # Create rank-local empty with the agreed trailing shape (axis-0 = elements)
                out[et] = np.empty((0,) + sh[1:], dtype=dt)

        return out

    def postproc_edict(self, edict_src, *, edim: int = 0):
        """
        Move element axis (0) back to `edim`. Drop truly empty entries.
        """
        out = {}
        for et, ary in edict_src.items():
            if ary.size == 0:
                continue
            out[et] = np.moveaxis(ary, 0, edim) if edim != 0 else ary
        return out

    @staticmethod
    def apply_relocation_plan(plan_by_etype: dict[str, dict],
                            edict_src: dict[str, np.ndarray],
                            comm) -> dict[str, np.ndarray]:
        """
        Apply precomputed relocation plan to edict with *element axis first*.
        Plan entries must provide: scount, sdisp, rcount, rdisp, send_rows, recv_to_end.
        """
        mixin = AlltoallMixin()
        out = {}

        for et, p in plan_by_etype.items():
            sc = np.asarray(p['scount'],      np.int32)
            sd = np.asarray(p['sdisp'],       np.int32)
            rc = np.asarray(p['rcount'],      np.int32)
            rd = np.asarray(p['rdisp'],       np.int32)
            sr = np.asarray(p['send_rows'],   np.int64)
            pr = np.asarray(p['recv_to_end'], np.int64)

            svals = edict_src[et][sr]                 # slice rows to send
            rtot  = int(rc.sum())
            rvals = np.empty((rtot, *svals.shape[1:]), dtype=svals.dtype)

            mixin._alltoallv(comm, (svals, (sc, sd)), (rvals, (rc, rd)))
            out[et] = rvals[pr]                       # reorder into final end-order

        return out

    def relocate_external(self, edict: dict[str, np.ndarray], *, edim: int) -> dict[str, np.ndarray]:
        """
        Minimal, silent pipeline for per-element arrays:
        preproc (axis->0) → apply_relocation_plan (with saved plan) → postproc (back to edim) → apply saved perms.
        """
        import numpy as np
        from pyfr.mpiutil import get_comm_rank_root

        # If no plan, just return a defensive copy
        if not getattr(self, "_full_reloc_plan", None):
            return {et: (None if a is None else np.array(a, copy=True))
                    for et, a in (edict or {}).items()}

        # 1) element axis first
        src0  = self.preproc_edict(edict or {}, edim=edim)

        # 2) relocate using saved plan
        comm  = get_comm_rank_root()[0]
        recv0 = self.apply_relocation_plan(self._full_reloc_plan, src0, comm)

        # 3) restore original element axis
        out   = self.postproc_edict(recv0, edim=edim)

        # 4) apply saved permutation if present
        if getattr(self, "_saved_perms", None):
            out = self._apply_perm_to_edict(out, self._saved_perms, edim=edim)

        return out

    def relocate(self, plan_by_etype, edict_src: dict[str, np.ndarray], *, edim: int = 0):
        """
        Generic relocate for any per-element edict (e.g., con with edim=0, spts with edim=1).
        """
        comm, _, _ = get_comm_rank_root()
        edict_proc = self.preproc_edict(edict_src, edim=edim)      # element axis first
        edict_loc  = self.apply_relocation_plan(plan_by_etype, edict_proc, comm)
        return self.postproc_edict(edict_loc, edim=edim)

    # ---------- FULL PLAN: build per-etype, no mutation ----------
    def plan_full_eidxs(self, et, eidxs_move):
        """
        Full-partition plan for et:
        • assign every local element a destination (default = self rank)
        • stable-partition by dest => send order
        • produce (scount,sdisp) and send_rows (permutation of all local rows)
        Returns plan dict.
        """
        comm, rank, _ = get_comm_rank_root()
        R = comm.size

        eidxs = self.eidxs[et].astype(np.int64, copy=False)
        nloc  = eidxs.size

        # dest per element (default: stay local)
        dest = np.full(nloc, rank, dtype=np.int64)

        # local gid -> lid map
        gid2lid = {int(g): i for i, g in enumerate(eidxs)} if nloc else {}

        if eidxs_move:
            for dst, gids in eidxs_move.items():
                if gids is None:
                    continue
                gg = np.asarray(gids, dtype=np.int64)
                if gg.size == 0:
                    continue

                # Keep only GIDs that are actually local on this rank
                # (avoids KeyError when someone passes foreign gids)
                is_local = np.fromiter((int(g) in gid2lid for g in gg),
                                    count=gg.size, dtype=bool)
                if not np.any(is_local):
                    continue

                gg_local = gg[is_local]
                lids = np.fromiter((gid2lid[int(g)] for g in gg_local),
                                count=gg_local.size, dtype=np.int64)

                dest[lids] = int(dst)

                # (Optional) debug: warn about non-local gids we ignored
                # if np.any(~is_local):
                #     bad = gg[~is_local].astype(int).tolist()
                #     crprint(-1, f"[plan_full_eidxs] {et}: ignored non-local gids {bad}")

        # stable partition by destination
        send_rows = np.argsort(dest, kind='mergesort')
        scount    = np.bincount(dest, minlength=R).astype(np.int32)
        sdisp     = AlltoallMixin._count_to_disp(scount)
        send_gids = eidxs[send_rows]

        return {
            'send_rows': send_rows,
            'scount': scount,
            'sdisp': sdisp,
            'send_gids': send_gids,
            'rcount': None, 'rdisp': None, 'recv_len': 0
        }


    # ---------- FULL PLAN: generic exchange of rows ----------
    def exchange_rows_with_plan(self, plan, svals):
        """
        Alltoallv rows using a full plan:
        svals must be `array[send_rows]` (element-axis first).
        Returns rvals with shape (sum(rcount), *svals.shape[1:]).
        """
        comm, _, _ = get_comm_rank_root()
        mixin = AlltoallMixin()

        scount = np.asarray(plan['scount'], np.int32)
        sdisp  = np.asarray(plan['sdisp'],  np.int32)

        rcount = np.empty_like(scount)
        comm.Alltoall([scount, mpi.INT], [rcount, mpi.INT])
        rdisp  = AlltoallMixin._count_to_disp(rcount)

        rtot   = int(rcount.sum())
        trailing = svals.shape[1:]
        rvals = np.empty((rtot, *trailing), dtype=svals.dtype)

        mixin._alltoallv(comm, (svals, (scount, sdisp)), (rvals, (rcount, rdisp)))

        plan['rcount'] = rcount
        plan['rdisp']  = rdisp
        plan['recv_len'] = rtot

        if plan.get('recv_to_end') is None:
            plan['recv_to_end'] = np.arange(rtot, dtype=np.int64)  # identity

        return rvals

    # ---------- PUBLIC: relocate eidxs + con with full plans ----------
    def relocate_eidxs_and_con_full(self, eidxs_move_by_et):
        self._full_reloc_plan.clear()
        for et in self.etypes:
            mv   = eidxs_move_by_et.get(et, {})
            plan = self.plan_full_eidxs(et, mv)          # built on pre-relocation state
            # --- eidxs ---
            recv_g = self.exchange_rows_with_plan(plan, plan['send_gids'])
            self.eidxs[et] = recv_g
            # --- con (same rows, element axis first) ---
            send_con = self.con[et][plan['send_rows']]
            recv_con = self.exchange_rows_with_plan(plan, send_con)
            self.con[et] = recv_con
            # keep the plan for geometry relocation
            self._full_reloc_plan[et] = plan

    def relocate_geometry_with_saved_plan(self, mesh):
        """
        Use self._full_reloc_plan (per etype) to relocate geometry dicts
        from the *mesh's current distribution* to the new end-order.
        """
        if not self._full_reloc_plan:
            # nothing to do; return originals (already in place)
            return mesh.spts, mesh.spts_nodes, mesh.spts_curved

        comm, _, _ = get_comm_rank_root()
        mixin = AlltoallMixin()

        def _reloc_one(edict: Dict[str, np.ndarray], edim: int):
            src = self.preproc_edict(edict or {}, edim=edim)
            out = {}
            for et in self.etypes:
                plan = self._full_reloc_plan[et]
                sc   = np.asarray(plan['scount'], np.int32)
                sd   = np.asarray(plan['sdisp'],  np.int32)
                rc   = np.asarray(plan['rcount'], np.int32)
                rd   = np.asarray(plan['rdisp'],  np.int32)
                sr   = np.asarray(plan['send_rows'], np.int64)

                svals = src[et][sr]
                rtot  = int(rc.sum())
                rvals = np.empty((rtot, *svals.shape[1:]), dtype=svals.dtype)
                mixin._alltoallv(comm, (svals, (sc, sd)), (rvals, (rc, rd)))

                pr = plan.get('recv_to_end')
                if pr is None:
                    pr = np.arange(rtot, dtype=np.int64)  # identity as fallback

                out[et] = rvals[pr]
            return self.postproc_edict(out, edim=edim)


        new_spts        = _reloc_one(mesh.spts,        edim=1)
        new_spts_nodes  = _reloc_one(mesh.spts_nodes,  edim=0)
        new_spts_curved = _reloc_one(mesh.spts_curved, edim=0)
        return new_spts, new_spts_nodes, new_spts_curved

    def _to_pyfr_connectivity(self):
        """
        Build PyFR-style connectivity from internal con:
        - con = (conL, conR) with local-local face pairs (unique, zipped)
        - con_p: {rank: [(et, lid, fid), ...]} for MPI faces
        - bcon:  {bc_name: [(et, lid, fid), ...]} for boundaries
        """
        comm, rank, _ = get_comm_rank_root()

        # gid->lid maps for this rank
        gid2lid = {et: {int(g): i for i, g in enumerate(self.eidxs[et].astype(np.int64))}
                for et in self.etypes}

        # gid->owner-rank maps (via one allgather per etype)
        owners = {}
        for et in self.etypes:
            locg = self.eidxs[et].astype(np.int64)
            pairs = np.c_[locg, np.full(locg.size, rank, np.int64)] if locg.size else np.empty((0, 2), np.int64)
            omap = {}
            for a in comm.allgather(pairs):
                if a.size:
                    for g, r in a:
                        omap[int(g)] = int(r)
            owners[et] = omap

        conL, conR = [], []
        con_p = {}
        bcon  = {}
        seen  = set()  # unique interior pairs by global (etype,gid,face)

        for et in self.etypes:
            eids   = self.eidxs[et].astype(np.int64, copy=False)
            con_et = self.con[et]
            nfaces = con_et.shape[1]

            for lid in range(eids.size):
                gid_here = int(eids[lid])
                for f in range(nfaces):
                    code, ngid, nf = map(int, con_et[lid, f])

                    # skip unfilled
                    if code == _CON_UNFILLED:
                        continue

                    # boundary: (-1, bc_id, -1)
                    if code < 0:
                        name = self.bc_name(int(ngid))
                        bcon.setdefault(name, []).append((et, int(lid), int(f)))
                        continue

                    # interior
                    net = self.etypes[int(code)]
                    ngid = int(ngid); nf = int(nf)

                    nlid = gid2lid.get(net, {}).get(ngid, None)
                    if nlid is not None:
                        # local-local: add once using a symmetric key on global ids
                        a = (et,  gid_here, f)
                        b = (net, ngid,     nf)
                        key = (a, b) if a < b else (b, a)
                        if key in seen: 
                            continue
                        seen.add(key)
                        conL.append((et,  int(lid),  int(f)))
                        conR.append((net, int(nlid), int(nf)))
                    else:
                        # mpi face → con_p entry for owner of neighbor
                        dst = owners[net][ngid]
                        if dst != rank:
                            con_p.setdefault(dst, []).append((et, int(lid), int(f)))

        return (conL, conR), con_p, bcon




class NativeReader:
    def __init__(self, fname, pname=None, *, construct_con=True):
        self.f = h5py.File(fname, 'r')
        self.mesh = _Mesh(fname=fname, raw=self.f)

        # Read in and transform the various parts of the mesh
        self._read_metadata()
        self._read_partitioning(pname)
        self._read_eles()
        self._read_nodes()

        if construct_con:
            self._construct_con()

        self.mmesh = _MetaMesh.from_mesh(self.mesh)

    def close(self):
        self.f.close()

    def load_soln(self, sname, prefix=None):
        mesh, soln = self.load_subset_mesh_soln(sname, prefix)

        # Ensure the solution is not subset
        if mesh is not self.mesh:
            raise ValueError('Subset solutions are not supported')

        return soln

    def load_subset_mesh_soln(self, sname, prefix=None):
        comm, rank, root = get_comm_rank_root()

        with h5py.File(sname, 'r') as f:
            if rank == root:
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
            soln = comm.bcast(soln, root=root)
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

                    escatter = SparseScatterer(comm, f[ei], idxs)
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
        comm, rank, root = get_comm_rank_root()

        if rank == root:
            creator = self.f['creator'][()].decode()
            codec = [c.decode() for c in self.f['codec']]
            uuid = self.f['mesh-uuid'][()].decode()
            version = self.f['version'][()]

            meta = (creator, codec, uuid, version)
        else:
            meta = None

        meta = comm.bcast(meta, root=root)
        mesh.creator, mesh.codec, mesh.uuid, mesh.version = meta

    def _read_with_idxs(self, dset, idxs):
        comm, rank, root = get_comm_rank_root()

        # Construct a Scatterer to read in and distribute the data
        s = Scatterer(comm, idxs)

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
        comm, rank, root = get_comm_rank_root()
        size = comm.size

        # Have the root rank read in the partitioning metadata
        if rank == root:
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
        ppath = 'partitionings/' + comm.bcast(pname, root=root)
        einfo = comm.scatter(einfo, root=root)
        self.neighbours = comm.scatter(ninfo, root=root)

        # Determine the element types in the mesh
        self.mesh.etypes = etypes = sorted(self.f['eles'])

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
        comm, rank, root = get_comm_rank_root()

        # Create a neighbourhood collective communicator
        ncomm = autofree(comm.Create_dist_graph_adjacent(self.neighbours,
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
            if rank < nrank:
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
