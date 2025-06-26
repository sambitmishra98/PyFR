from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, ClassVar

import numpy as np
from pyfr.readers.native import _Mesh
from pyfr.mpiutil import AlltoallMixin, get_comm_rank_root

@dataclass(frozen=True, slots=True)
class MeshMetadata:
    """
    Immutable, share-by-reference bundle of global mesh constants.

    Every MetaMesh and SubMesh instance holds one pointer (`meta`)
    instead of mutating class variables at run-time.
    """
    etypes: List[str]                     # present element types (sorted)
    edim: int                             # spatial dimension
    etype_nfaces_map: Dict[str, int]      # e.g. {'tri':3,'quad':4,…}
    nv_map: Dict[str, int]                # #verts per element type
    e_to_i: Dict[str, int]                # element-type → small int code
    i_to_e: Dict[int, str]                # inverse map


@dataclass
class SubMesh:
    # ------------------------------------------------------------------ #
    # normal instance fields
    # ------------------------------------------------------------------ #
    meta: MeshMetadata                     # global mesh constants
    etypes: list[str]                        = field(default_factory=list)
    eidxs:  dict[str, np.ndarray]            = field(default_factory=dict)
    arrays: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    # class SubMesh  (add near other ClassVars)
    array_specs: ClassVar[dict[str, tuple[np.dtype, int]]] = {
        # name          dtype      elem-axis in native mesh
        'con'        : (np.int64 ,  0),
        'spts_curved': (np.bool_ ,  0),
        'spts_nodes' : (np.int64 ,  0),
        'spts'       : (np.float64, 1),
    }
    
    # pyfr/relocator/submesh.py  (near other ClassVars)

    etype_nfaces_map: ClassVar[dict[str, int]] = {
        'tri': 3, 'quad': 4,
        'tet': 4, 'hex': 6,
        'pri': 5, 'pyr': 5,
    }

    def __getattr__(self, name: str):
        """Shortcut: allow `self.con`, `self.spts`, … to access `arrays[name]`."""
        arrays = self.__dict__.get("arrays")
        if arrays and name in arrays:
            return arrays[name]
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    # submesh.py
    @classmethod
    def blank(cls, meta: MeshMetadata) -> "SubMesh":
        """
        Allocate an *empty* sub-mesh that still knows all element types.
        Only shape matters – content will be filled later by MetaMesh.
        """
        etypes        = list(meta.etypes)               # copy → immune to aliasing
        make_empty    = lambda dt, shape: np.empty(shape, dtype=dt)

        empty_eidxs   = {et: make_empty(np.int64, (0,)) for et in etypes}
        empty_arrays  = {}

        for name, (dtype, _) in cls.array_specs.items():
            empty_arrays[name] = {
                et: make_empty(dtype, (0, *cls._trailing_shape(name, meta, et)))
                for et in etypes
            }

        return cls(meta=meta,
                etypes=etypes,
                eidxs=empty_eidxs,
                arrays=empty_arrays)

    @classmethod
    def from_native_mesh(cls,
                        mesh: _Mesh,
                        meta: MeshMetadata,
                        bc_map: dict[str, int]) -> "SubMesh":
        """
        Clone *every* per-element array from ``mesh`` into a local SubMesh.
        *   keeps axis-0 = element axis for all arrays
        *   re-uses a common copy helper to avoid boiler-plate
        """
        rank = get_comm_rank_root()[1]

        def _copy(name: str, move_axis: int | None = None) -> dict[str, np.ndarray]:
            out = {et: getattr(mesh, name)[et].copy() for et in meta.etypes}
            if move_axis is not None:                       # e.g. spts  (Nverts, dim)
                out = {et: np.moveaxis(a, move_axis, 0) for et, a in out.items()}
            return out

        arrays: dict[str, dict[str, np.ndarray]] = {
            "con"        : cls._build_con(mesh, rank, get_comm_rank_root()[0], bc_map, meta),
            "spts_curved": _copy("spts_curved"),
            "spts_nodes" : _copy("spts_nodes"),
            "spts"       : _copy("spts", move_axis=1),
        }

        # element IDs
        eidxs = {et: mesh.eidxs[et].copy() for et in meta.etypes}

        return cls(meta=meta, etypes=list(meta.etypes), eidxs=eidxs, arrays=arrays)

    @property
    def con(self):
        return self.arrays["con"]

    # ------------------------------------------------------------------ #
    # sanity helper – checks every array against its declared spec
    # ------------------------------------------------------------------ #


    # ───────────────────────────────────────────────────────
    # fast owner-column re-write (used by MetaMesh)
    # ───────────────────────────────────────────────────────
    def patch_owner_columns(self, owner_dict: dict[tuple[int, int], int]) -> None:
        """
        In-place overwrite of column 0 (owner-rank) across every face
        that links to another element (i.e. netype >= 0).

        *owner_dict* maps **(etype_code, global_eid) → owning_rank**.
        """
        for et in self.etypes:
            ten = self.con[et]                   # (Ne, Nf, 4)
            if ten.size == 0:
                continue                         # nothing to patch

            # keep only interior + MPI faces (boundary rows have netype < 0)
            mask = ten[..., 1] >= 0
            if not mask.any():
                continue

            rows, faces = np.where(mask)         # flat indices
            codes = ten[rows, faces, 1]          # neighbour etype-codes
            gids  = ten[rows, faces, 2]          # neighbour global-EIDs

            # vectorised lookup with graceful fallback
            new_owner = np.fromiter(
                (owner_dict.get((c, g), ten[r, f, 0])
                for c, g, r, f in zip(codes, gids, rows, faces)),
                dtype=ten.dtype,
                count=rows.size,
            )
            ten[rows, faces, 0] = new_owner    
            
    # ----------------------------------------------------------------------
    #  Native-like element re-ordering (call once, after sync_owner_columns)
    # ----------------------------------------------------------------------
    def restore_native_order(self, rank: int) -> None:
        """
        Re-establish the NativeReader lexicographic order:

            primary  :  is_mpi  (False → interior, True → MPI)
            secondary:  spts_curved  (False → straight,  True → curved)
        """
        is_mpi = self._compute_is_mpi(rank)

        # ---------- re-ordering -----------------------------------------
        for et in self.etypes:
            if not self.eidxs[et].size:
                continue                                 # nothing to do

            key1 = (~is_mpi[et]).astype(np.int8)          # 0 for MPI 
            key2 =  self.spts_curved[et].astype(np.int8)  # 0 for straight
            
            order = np.lexsort((key2, key1))             # two-key sort

            # eidxs
            self.eidxs[et] = self.eidxs[et][order]

            # every array listed in array_specs
            for name in SubMesh.array_specs:
                self.arrays[name][et] = self.arrays[name][et][order]

    def _slice_out_patch(self, moved: dict[str, set[int]]) -> "SubMesh":
        """
        • Build a child SubMesh containing *exactly* the rows in *moved*  
        • Remove those rows from the parent (in-place)
        """
        child_eidxs:  dict[str, np.ndarray]            = {}
        child_arrays: dict[str, dict[str, np.ndarray]] = {a: {} for a in self.array_specs}

        for et in self.etypes:
            take = np.isin(self.eidxs[et], list(moved.get(et, ())), assume_unique=True)
            keep = ~take

            # --- child ---
            child_eidxs[et] = self.eidxs[et][take].copy()
            for ary in self.array_specs:
                child_arrays[ary][et] = self.arrays[ary][et][take].copy()

            # --- parent (shrink) ---
            self.eidxs[et] = self.eidxs[et][keep]
            for ary in self.array_specs:
                self.arrays[ary][et] = self.arrays[ary][et][keep]

        return SubMesh(meta=self.meta,
                    etypes=self.etypes,
                    eidxs=child_eidxs,
                    arrays=child_arrays)


    def carveout_for_nrank(self, nrank: int, *, ntarget: int | None = None) -> "SubMesh":
        """
        Guarantee: if the interface strip contains ≥ ntarget elements the result size == ntarget.
        """
        rank = get_comm_rank_root()[1]

        # 1. interface layer -------------------------------------------------
        moved = self._find_elements_touching_rank(
            self.etypes, self.con, self.eidxs, nrank
        )

        iface_size = sum(len(v) for v in moved.values())
        if ntarget is None or ntarget <= iface_size:
            moved = self._select_top_by_iface(moved, ntarget, nrank, rank)
        else:
            # need more – keep the interface layer and grow further
            moved = self._expand_patch_to_ntarget(moved, ntarget, rank)

        if not moved:
            return SubMesh.blank(self.meta)

        return self._slice_out_patch(moved)     # existing helper factored out


    def _select_top_by_iface(
        self, moved_init: dict[str, set[int]],
        ntarget: int, nrank: int, rank: int
    ) -> dict[str, set[int]]:
        """
        Return a copy of *moved_init* trimmed to exactly `ntarget`
        using the interface-face ordering strategy.
        """
        if ntarget is None:
            return moved_init

        # flatten into (metric, et, gid) tuples
        records = []
        for et, gids in moved_init.items():
            ten = self.con[et]                       # (Ne, Nf, 4)
            curved = self.spts_curved[et]
            gl2row = {int(g): i for i, g in enumerate(self.eidxs[et])}

            for gid in gids:
                row = gl2row[gid]
                iface_faces = ((ten[row, :, 0] == nrank) & (ten[row, :, 1] >= 0)).sum()
                records.append( (-iface_faces, curved[row], gid, et) )

        # sort & pick
        records.sort()               # python tuples sort lexicographically
        keep = records[:ntarget]

        # rebuild moved dict
        moved = {}
        for _, _, gid, et in keep:
            moved.setdefault(et, set()).add(gid)

        if len(keep) < ntarget:      # underflow ⇒ impossible (should not happen)
            print(f"[select] rank={rank}: interface only {len(keep)}/{ntarget}")
        return moved

    @staticmethod
    def _find_elements_touching_rank(
        etypes: List[str],
        con: Dict[str, np.ndarray],
        eidxs: Dict[str, np.ndarray],
        nrank: int,
    ) -> Dict[str, set[int]]:
        """
        Elements having **at least one face whose owner-column == *nrank***.
        """
        hits: dict[str, set[int]] = {}
        for et in etypes:
            ten  = con[et]
            if ten.size == 0:
                continue

            rows = np.flatnonzero((ten[:, :, 0] == nrank).any(axis=1))
            if rows.size:
                hits[et] = set(map(int, eidxs[et][rows]))

        return hits


    # ─────────────────────────────────────────────────────────────
    # build connectivity tensors (owner, ncode, gid, face)
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _build_con(mesh: _Mesh,
                rank: int,
                comm,
                bc_map: dict[str, int],
                meta: MeshMetadata) -> dict[str, np.ndarray]:
        """
        Re-encode PyFR’s *triple* connectivity lists into dense tensors
        (Ne, Nf, 4).  Columns: **owner-rank, neighbour-code, gid, face-idx**.

        * interior faces  → owner == this rank
        * MPI faces       → owner  = remote rank, neighbour-gid filled later
        * boundary faces  → owner == -1, neighbour-code/gid == -1,
                            face-idx stores bc-id (small positive int)
        """
        etypes, e2i = meta.etypes, meta.e_to_i
        nf_map      = meta.etype_nfaces_map

        # allocate & build a quick gid→row map once per etype
        con   = {et: np.full((len(mesh.eidxs[et]), nf_map[et], 4),
                            -2, dtype=np.int64) for et in etypes}
        gid2r = {et: {int(g): i for i, g in enumerate(mesh.eidxs[et])}
                for et in etypes}

        # ---------- 1) interior faces ---------------------------------------
        for (etL, lidL, fL), (etR, lidR, fR) in zip(*mesh.con):
            gidL, gidR = int(mesh.eidxs[etL][lidL]), int(mesh.eidxs[etR][lidR])
            rL, rR     = gid2r[etL][gidL],            gid2r[etR][gidR]

            con[etL][rL, fL] = (rank, e2i[etR], gidR, fR)
            con[etR][rR, fR] = (rank, e2i[etL], gidL, fL)

        # ---------- 2) MPI faces  (all-to-all once) -------------------------
        send_buf: list[list[tuple[int, int, int]]] = [[] for _ in range(comm.size)]
        local_rows: list[tuple[int, str, int, int]] = []  # (peer, et, lid, f)

        for peer, triples in mesh.con_p.items():
            for et, lid, f in triples:
                local_rows.append((peer, et, lid, f))
                send_buf[peer].append((e2i[et], int(mesh.eidxs[et][lid]), f))

                # placeholder – will be overwritten after exchange
                row = gid2r[et][int(mesh.eidxs[et][lid])]
                con[et][row, f] = (peer, -1, -1, -1)

        # fixed -------------------------------------
        flat   = np.array([t for sub in send_buf for t in sub], dtype=np.int64)  # shape (N, 3)
        counts = np.fromiter((len(sub) for sub in send_buf), dtype=np.int64) 

        tx = AlltoallMixin()
        rbuf, (rcount, rdisp) = tx._alltoallcv(comm, flat, counts)

        for peer in range(comm.size):
            if rcount[peer] == 0:
                continue
            chunk = rbuf[rdisp[peer]: rdisp[peer] + rcount[peer]]

            # iterate in the same order as we sent
            for (ncode, gid_remote, f_remote), (p, etL, lidL, fL) in zip(
                    chunk, (row for row in local_rows if row[0] == peer)):
                row = gid2r[etL][int(mesh.eidxs[etL][lidL])]
                con[etL][row, fL] = (peer, ncode, gid_remote, f_remote)

        # ---------- 3) boundary faces ---------------------------------------
        for bc_name, triples in mesh.bcon.items():
            bc_id = bc_map[bc_name]
            for et, lid, f in triples:
                row = gid2r[et][int(mesh.eidxs[et][lid])]
                con[et][row, f] = (-1, -1, -1, bc_id)

        # ---------- final sanity -------------------------------------------
        for et, arr in con.items():
            assert np.all(arr != -2), f"con[{et}] still has -2 placeholders"

        return con

    @property
    def N_ei(self) -> dict[str, int]:
        """Per-etype element counts on this SubMesh."""
        # all etypes are guaranteed to exist – just take len
        return {et: len(ids) for et, ids in self.eidxs.items()}

    @property
    def N_i(self) -> int:
        """Total number of elements over all etypes."""
        return sum(self.N_ei.values())

    def _compute_is_mpi(self, rank: int) -> dict[str, np.ndarray]:
        """
        Element-wise flag: **True** ⇨ at least one face talks to another MPI rank.
        """
        out: dict[str, np.ndarray] = {}
        for et in self.etypes:
            ten = self.con[et]                                  # (Ne, Nf, 4)
            if ten.size == 0:
                out[et] = np.zeros(0, dtype=bool)
                continue

            owner, etype = ten[..., 0], ten[..., 1]
            mpi_mask      = (owner != rank) & (owner >= 0) & (etype >= 0)
            out[et]       = mpi_mask.any(axis=1)

        return out

    def _expand_patch_to_ntarget(
        self,
        moved: dict[str, set[int]],
        ntarget: int | None,
        rank: int,
    ) -> dict[str, set[int]]:
        """
        Grow *moved* deterministically until it contains **exactly**
        ``ntarget`` elements (provided this SubMesh still has that many
        elements in total).

        Strategy
        --------
        1.  repeatedly look for neighbours of the current patch which
            are **still present** on this rank                   *(safe-guard)*  
        2.  pick the candidate that  
              • shares more internal faces with the patch  
              • then has fewer MPI faces  
              • then is straight before curved
        3.  **If we can no longer find a valid neighbour but are still
            short,** fall back to filling the deficit with *any* remaining
            elements of this rank (straight-first, then curved).
        """
        if ntarget is None:
            return moved                                         # nothing to do

        from collections import defaultdict

        # ------------------------------------------------------------------ #
        def patch_size(md: dict[str, set[int]]) -> int:
            return sum(len(v) for v in md.values())

        con = self.con                                           # shorthand

        # ------------------------------------------------------------------ #
        while patch_size(moved) < ntarget:
            # ─── rebuild gid→row on EACH iteration (more robust) ─────────
            gl_to_row = {
                et: {int(g): i for i, g in enumerate(self.eidxs[et])}
                for et in self.etypes
            }

            # 1) collect neighbour candidates + how many faces they share
            cand_shared: dict[tuple[str, int], int] = defaultdict(int)

            for et in self.etypes:
                for gid in moved.get(et, ()):
                    row = gl_to_row[et].get(gid)                 # may be gone
                    if row is None:
                        continue

                    faces = con[et][row]                         # (Nf, 4)
                    mask  = (faces[:, 0] == rank) & (faces[:, 1] >= 0)
                    for ncode, ngid in faces[mask][:, 1:3]:
                        net  = self.meta.i_to_e[int(ncode)]
                        ngid = int(ngid)
                        if ngid in moved.get(net, set()):
                            continue
                        # neighbour must *still* exist
                        if ngid in gl_to_row[net]:
                            cand_shared[(net, ngid)] += 1

            # 2) if we have at least one valid neighbour → choose the best
            if cand_shared:
                keys          = list(cand_shared)
                shared_cnt    = np.asarray([cand_shared[k] for k in keys])
                mpi_cnt       = np.empty_like(shared_cnt)
                curved_flag   = np.empty_like(shared_cnt)

                for i, (et, gid) in enumerate(keys):
                    row          = gl_to_row[et][gid]
                    ten          = con[et][row]
                    mpi_cnt[i]   = ((ten[:, 0] != rank) & (ten[:, 1] >= 0)).sum()
                    curved_flag[i] = self.spts_curved[et][row]

                order_mat = np.stack([shared_cnt, -mpi_cnt, -curved_flag], axis=1)
                best      = keys[int(np.lexsort(order_mat.T)[-1])]
                moved.setdefault(best[0], set()).add(best[1])
                continue                                            # loop again

            # 3) dead-end: no *valid* neighbours left – brute-force fill
            deficit = ntarget - patch_size(moved)
            if deficit <= 0:
                break

            for et in self.etypes:                                  # straight→curved
                rows = np.arange(len(self.eidxs[et]))
                straight = rows[~self.spts_curved[et]]
                curved   = rows[self.spts_curved[et]]

                for pool in (straight, curved):
                    for row in pool:
                        gid = int(self.eidxs[et][row])
                        if gid in moved.get(et, set()):
                            continue
                        moved.setdefault(et, set()).add(gid)
                        deficit -= 1
                        if deficit == 0:
                            break
                    if deficit == 0:
                        break
                if deficit == 0:
                    break

            # if we STILL have a deficit, donor genuinely ran out of elements
            # (should be impossible given the upfront surplus check)
            if deficit > 0:
                print(f"[expand] rank={rank}: donor exhausted "
                      f"at {patch_size(moved)}/{ntarget}")
                break

        return moved

    def smallest_patch_touching(self, nrank: int) -> int:
        """
        Return the number of elements in the *first*-layer interface strip
        between this sub-mesh and `nrank`.  Used by MetaMesh to decide
        the least-overshoot it can tolerate when no exact donation is
        possible in the current sweep.
        """
        moved = self._find_elements_touching_rank(
            self.etypes, self.con, self.eidxs, nrank
        )
        return sum(len(v) for v in moved.values())



    @staticmethod
    def _trailing_shape(name: str, meta: MeshMetadata, et: str | None):
        if name == "con":
            nf = meta.etype_nfaces_map[et] if et else next(iter(meta.etype_nfaces_map.values()))
            return (nf, 4)
        if name == "spts_curved":
            return ()
        if name == "spts_nodes":
            nv = meta.nv_map[et] if et else next(iter(meta.nv_map.values()))
            return (nv,)
        if name == "spts":
            nv = meta.nv_map[et] if et else next(iter(meta.nv_map.values()))
            return (nv, meta.edim)
        raise KeyError(name)
