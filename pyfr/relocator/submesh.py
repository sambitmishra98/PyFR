from __future__ import annotations

from pyfr.relocator.utils import crpprint

"""Light-weight container for a *single* sub-partition in a parallel run.

This class purposefully owns **only** the data that is unique to the
sub-partition: element-id arrays (``eidxs``) and per-face connectivity
tensors (``con``).  All global, run-wide constants (like element-type maps)
are injected by :class:`_MetaMesh` *once* at start-up so they never live in
multiple places at once.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, ClassVar

import numpy as np
from pyfr.readers.native import _Mesh
from pyfr.mpiutil import AlltoallMixin, get_comm_rank_root


@dataclass
class SubMesh:
    # ------------------------------------------------------------------ #
    # normal instance fields
    # ------------------------------------------------------------------ #
    etypes: list[str]                        = field(default_factory=list)
    eidxs:  dict[str, np.ndarray]            = field(default_factory=dict)
    arrays: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # class-wide constants  (not dataclass fields!)
    # ------------------------------------------------------------------ #
    etype_nfaces_map: ClassVar[dict[str, int]] = {'tri': 3, 'quad': 4,
        'tet': 4, 'hex': 6, 'pri': 5, 'pyr': 5
    }
    e_to_i: ClassVar[dict[str, int]] = {}
    i_to_e: ClassVar[dict[int, str]] = {}

    # class SubMesh  (add near other ClassVars)
    array_specs: ClassVar[dict[str, tuple[callable, np.dtype, int]]] = {
        # name          trailing-shape fn(et)                  dtype        elem axis in mesh
        'con'        : (lambda et: (SubMesh.etype_nfaces_map[et], 4),  np.int64  , 0),
        'spts_curved': (lambda et: (),                                 np.bool_  , 0),
        'spts_nodes' : (lambda et: (SubMesh.nv_map[et],),              np.int64  , 0),
        'spts'       : (lambda et: (SubMesh.nv_map[et], SubMesh.edim), np.float64, 1),
    }
    
    @staticmethod
    def spec(name: str, et: str):
        """Return (trailing_shape, dtype, elem_axis) for *name*,*etype*."""
        shape_fn, dt, ax = SubMesh.array_specs[name]
        return shape_fn(et), dt, ax

    def __getattr__(self, name):
        # Called only if normal attributes fail
        if 'arrays' in self.__dict__ and name in self.__dict__['arrays']:
            return self.__dict__['arrays'][name]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    @classmethod
    def blank(cls, etypes):
        empty_eidxs = {et: np.empty(0, np.int64) for et in etypes}
        empty_arrays = {}
        for name, (shape_fn, dt, _) in cls.array_specs.items():
            empty_arrays[name] = {et: np.empty((0, *shape_fn(et)), dt) for et in etypes}
        return cls(etypes=etypes, eidxs=empty_eidxs, arrays=empty_arrays)

    @staticmethod
    def trailing_shape(name: str, et: str) -> tuple[int, ...]:
        return SubMesh.array_specs[name][0](et)

    @classmethod
    def from_native_mesh(cls, mesh: _Mesh, etypes: list[str], bc_map) -> "SubMesh":
        comm, rank, _ = get_comm_rank_root()

        arrays = {}
        for name, (shape_fn, dt, ax) in cls.array_specs.items():
            if name == 'con':
                arrays[name] = cls._build_con(mesh, rank, comm, bc_map)

            elif hasattr(mesh, name):
                raw = {et: getattr(mesh, name)[et].copy() for et in etypes}
                if ax != 0:                       # move element axis to the front
                    raw = {et: np.moveaxis(a, ax, 0) for et, a in raw.items()}
                arrays[name] = raw

            else:                                # field absent ⇒ empty placeholder
                arrays[name] = {et: np.empty((0, *shape_fn(et)), dt) for et in etypes}

        return cls(etypes=etypes,
                eidxs={et: mesh.eidxs[et].copy() for et in etypes},
                arrays=arrays)

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
            for name, (_, _, _) in SubMesh.array_specs.items():
                self.arrays[name][et] = self.arrays[name][et][order]

    def carveout_for_nrank(self, nrank: int, *, ntarget: int | None = None) -> "SubMesh":
        """
        Return a child SubMesh containing every element that touches *nrank*.
        All arrays listed in `array_specs` are sliced consistently.
        """
        from pyfr.relocator.utils import crprint


        rank = get_comm_rank_root()[1]      # my rank for prints

        # --- 1) FIRST interface layer (old behaviour) ------------------
        moved = self._find_elements_touching_rank(
            self.etypes, self.con, self.eidxs, nrank
        )

        # --- 2) (NEW) grow to reach ntarget ----------------------------
        moved = self._expand_patch_to_ntarget(moved, ntarget, rank)

        if not moved:
            return SubMesh.blank(self.etypes)

        # ---------------- locate rows to carve per etype -----------------
        idx_map = {
            et: np.nonzero(np.isin(self.eidxs[et], list(gids)))[0]
            for et, gids in moved.items()
        }

        # ---------------- build child arrays ----------------------------
        child_eidxs  = {}
        child_arrays = {ary: {} for ary in SubMesh.array_specs}

        for et in self.etypes:
            ids = idx_map.get(et, np.empty(0, np.int64))

            # element IDs
            child_eidxs[et] = (
                self.eidxs[et][ids].copy() if ids.size else np.empty(0, np.int64)
            )

            for ary in SubMesh.array_specs:
                parent_arr = self.arrays[ary][et]
                if ids.size:
                    child_arrays[ary][et] = parent_arr[ids].copy()
                else:                                           # empty slice
                    child_arrays[ary][et] = np.empty_like(parent_arr, shape=(0, *parent_arr.shape[1:]))

        for et, ids in idx_map.items():
            keep = np.ones(len(self.eidxs[et]), bool)
            keep[ids] = False

            # eidxs
            self.eidxs[et] = self.eidxs[et][keep]

            # every array
            for ary in SubMesh.array_specs:
                getattr(self, ary)[et] = getattr(self, ary)[et][keep]

        return SubMesh(etypes=self.etypes, eidxs=child_eidxs, arrays=child_arrays)


    # ------------------------------------------------------------------
    # static helpers – stateless, easily unit-testable
    # ------------------------------------------------------------------
    @staticmethod
    def _find_elements_touching_rank(
        etypes: List[str],
        con: Dict[str, np.ndarray],
        eidxs: Dict[str, np.ndarray],
        nrank: int,
    ) -> Dict[str, set[int]]:
        """Return {etype: {global-gid, …}} for elements that touch *nrank*."""
        touched = {et: set() for et in etypes}
        for et in etypes:
            ten = con[et]
            rows = np.nonzero((ten[:, :, 0] == nrank).any(axis=1))[0]
            if rows.size:
                touched[et].update(eidxs[et][rows])
        return {et: s for et, s in touched.items() if s}

    # ------------------------------------------------------------------
    # connectivity builder – unchanged except for constant refs
    # ------------------------------------------------------------------
    @staticmethod
    def _build_con(mesh: _Mesh, rank: int, comm, bc_map) -> Dict[str, np.ndarray]:
        nfaces = SubMesh.etype_nfaces_map
        e2i = SubMesh.e_to_i

        con = {et: np.full((len(mesh.eidxs[et]), nfaces[et], 4), -2, np.int64) for et in mesh.etypes}
        glmap = {et: {int(g): i for i, g in enumerate(mesh.eidxs[et])} for et in mesh.etypes}

        # internal faces -------------------------------------------------
        for (etL, lidL, fL), (etR, lidR, fR) in zip(*mesh.con):
            gidL = int(mesh.eidxs[etL][lidL])
            gidR = int(mesh.eidxs[etR][lidR])
            rL, rR = glmap[etL][gidL], glmap[etR][gidR]

            con[etL][rL, fL] = (rank, e2i[etR], gidR, fR)
            con[etR][rR, fR] = (rank, e2i[etL], gidL, fL)

        # mpi faces ------------------------------------------------------
        send_per_rank: List[List[Tuple[int, int, int]]] = [[] for _ in range(comm.size)]
        mpi_local_rows: List[Tuple[int, str, int, int]] = []  # (nrank, et, lid, f)

        for nrank, triples in mesh.con_p.items():
            for et, lid, f in triples:
                mpi_local_rows.append((nrank, et, lid, f))
                send_per_rank[nrank].append((e2i[et], int(mesh.eidxs[et][lid]), f))

                gid_local = int(mesh.eidxs[et][lid])
                r_local = glmap[et][gid_local]
                con[et][r_local, f] = (nrank, -1, -1, -1)  # placeholder

        svals = np.array([item for sub in send_per_rank for item in sub], np.int64)
        scount = np.array([len(sub) for sub in send_per_rank], np.int64)

        tx = AlltoallMixin()
        rvals, (rcount, rdisp) = tx._alltoallcv(comm, svals, scount)

        for src_rank in range(comm.size):
            if rcount[src_rank] == 0:
                continue
            start, stop = rdisp[src_rank], rdisp[src_rank] + rcount[src_rank]
            chunk = rvals[start:stop]

            for (ncode, gid_remote, f_remote), (dst_rank, etL, lidL, fL) in zip(chunk, (row for row in mpi_local_rows if row[0] == src_rank)):
                gid_local = int(mesh.eidxs[etL][lidL])
                r_local = glmap[etL][gid_local]
                con[etL][r_local, fL] = (src_rank, ncode, gid_remote, f_remote)

        # boundaries -----------------------------------------------------
        for bcname, triples in mesh.bcon.items():
            bc_id = bc_map[bcname]
            for et, lid, f in triples:
                gid = int(mesh.eidxs[et][lid])
                rrow = glmap[et][gid]
                con[et][rrow, f] = (-1, -1, -1, bc_id)

        # sanity ---------------------------------------------------------
        for et, arr in con.items():
            assert (arr == -2).sum() == 0, f"unfilled entries in con[{et}]"
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
        """Return {etype: bool[Ne]} – True if element has an MPI neighbour."""
        flags = {}
        for et in self.etypes:
            ten = self.con[et]                           # (Ne, Nf, 4)
            if ten.size == 0:
                flags[et] = np.zeros(0, bool)
                continue

            owner = ten[..., 0]                          # neighbour owner-rank
            etype = ten[..., 1]                          # -1 on boundaries
            mpi_face = (owner != rank) & (owner >= 0) & (etype >= 0)
            flags[et] = mpi_face.any(axis=1)             # OR over faces
        return flags

    # ------------------------------------------------------------------
    # helper – grow {etype: set(gids)} until |⋃| == ntarget   (self-contained)
    # ------------------------------------------------------------------
    def _expand_patch_to_ntarget(
        self,
        moved: dict[str, set[int]],
        ntarget: int | None,
        rank: int,
    ) -> dict[str, set[int]]:
        """
        Greedy, deterministic growth of *moved* until the total element
        count reaches *ntarget* (or we run out of neighbours).

        Strategy at each step
        1.  prefer candidate sharing **more patch faces**
        2.  break ties by **fewer MPI faces**
        3.  break further ties by **straight before curved**
        """
        if ntarget is None:
            return moved

        from collections import defaultdict

        # --- helpers -------------------------------------------------------
        def patch_size(md: dict[str, set[int]]) -> int:
            return sum(len(v) for v in md.values())

        # global-gid → local-row map for quick access
        gl_to_row = {
            et: {int(g): i for i, g in enumerate(self.eidxs[et])}
            for et in self.etypes
        }

        con = self.con  # convenience alias

        # ------------------------------------------------------------------
        while patch_size(moved) < ntarget:
            # 1) collect neighbour candidates + how many faces they share
            cand_shared = defaultdict(int)             # (et,gid) → shared-face-count

            for et in self.etypes:
                for gid in moved.get(et, ()):
                    row = gl_to_row[et][gid]
                    faces = con[et][row]               # (Nf,4)

                    # look at neighbours that stay on *this* rank
                    mask = (faces[:, 0] == rank) & (faces[:, 1] >= 0)
                    for ncode, ngid in faces[mask][:, 1:3]:

                        net = self.i_to_e[int(ncode)]
                        ngid = int(ngid)
                        if ngid in moved.get(net, set()):
                            continue
                        cand_shared[(net, ngid)] += 1

            if not cand_shared:
                print(f"[expand] rank={rank} dead-end at {patch_size(moved)}/{ntarget}")
                break

            # 2) build candidate metric arrays
            cand_keys       = list(cand_shared.keys())             # ordered list
            shared_counts   = np.fromiter((cand_shared[k] for k in cand_keys),
                                        int, len(cand_keys))

            mpi_counts      = np.empty_like(shared_counts)
            curved_flags    = np.empty_like(shared_counts)

            for i, (et, gid) in enumerate(cand_keys):
                row = gl_to_row[et][gid]
                ten = con[et][row]                                # (Nf,4)

                mpi_counts[i]   = ( (ten[:, 0] != rank) & (ten[:, 1] >= 0) ).sum()
                curved_flags[i] = self.spts_curved[et][row]

            # 3) deterministic best-candidate selection (lexsort)
            order_key = np.stack([ shared_counts,
                                -mpi_counts,
                                -curved_flags ], axis=1)
            best_idx          = int(np.lexsort(order_key.T)[-1])   # NumPy → int
            best_et, best_gid = cand_keys[best_idx]

            # 4) update patch + trace
            moved.setdefault(best_et, set()).add(best_gid)

            print(f"[patch] rank={rank} add ({best_et},{best_gid})  "
                f"shared={shared_counts[best_idx]}  "
                f"mpi={mpi_counts[best_idx]}  "
                f"curved={bool(curved_flags[best_idx])}  "
                f"new_size={patch_size(moved)}/{ntarget}")

        return moved
