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
from typing import Dict, List, Tuple

from pyfr.relocator.utils import crpprint, crprint


from pyfr.util import subclass_where
from pyfr.shapes import BaseShape
from copy import deepcopy


_CON_UNFILLED = -2  # sentinel for yet-unset entries
_BC_NONE      = -1  # boundary tag in con: (-1, bc_id, -1)


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

@dataclass(slots=True)
class _MetaMesh:

    ndims: int = None

    e_to_i: Dict[str, int]       = field(default_factory=dict)
    i_to_e: Dict[int, str]       = field(default_factory=dict)
    bc_name2id: Dict[str, int]   = field(default_factory=dict)
    bc_id2name: Dict[int, str]   = field(default_factory=dict)

    # global immutable topology info 
    etypes:  List[str]                 = field(default_factory=list)
    gids:    Dict[str, np.ndarray]     = field(default_factory=dict)

    # mutable elements placements ----------
    # placements[et] shape: (Ng(et), 2)  [owner_rank, owner_lid]
    placements: Dict[str, np.ndarray]  = field(default_factory=dict)

    # ---------- connectivity ----------
    # con[et] shape: (Ng(et), nfaces(et), 3): (n_code, n_gid, n_fid)
    # boundary encoded as (-1, bc_id, -1)
    con: Dict[str, np.ndarray] = field(default_factory=dict)
    # neighbor rows cache: con_nrow[et] shape: (Ng(et), nfaces(et)) with row or -1
    con_nrow: Dict[str, np.ndarray] = field(default_factory=dict)

    curved_flags: Dict[str, np.ndarray] = field(default_factory=dict)
    mpi_iface:    Dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_mesh(cls, mesh: "_Mesh") -> "_MetaMesh":
        comm, rank, _ = get_comm_rank_root()

        # 1) Global etypes
        local_et = set(getattr(mesh, "etypes", []))
        etypes = sorted(set.union(*comm.allgather(local_et))) if comm.size > 1 else sorted(local_et)
        e_to_i = {et:i for i, et in enumerate(etypes)}
        i_to_e = {i:et for et, i in e_to_i.items()}

        # 2) Global BC maps
        local_bc = set(getattr(mesh, "bcon", {}).keys())
        names = sorted(set.union(*comm.allgather(local_bc))) if comm.size > 1 else sorted(local_bc)
        bc_name2id = {n:i for i, n in enumerate(names)}
        bc_id2name = {i:n for n, i in bc_name2id.items()}

        # 3) Global gids + gid2row
        gids = {}
        for et in etypes:
            loc = np.asarray(mesh.eidxs.get(et, np.empty(0, np.int64)), dtype=np.int64)
            gathered = comm.allgather(loc)
            glob = np.unique(np.concatenate(gathered)) if gathered and len(gathered) > 1 else (np.sort(loc) if loc.size else np.empty(0, np.int64))
            gids[et] = glob


        # 4) Placements (owner rank / local lid) in global rows
        placements: Dict[str, np.ndarray] = {}
        for et in etypes:
            Ng = gids[et].size
            loc = np.zeros((Ng, 2), dtype=np.int64)
            loc_gids = np.asarray(mesh.eidxs.get(et, np.empty(0, np.int64)), dtype=np.int64)
            if loc_gids.size:
                rows = cls._rows_from_gids(gids[et], loc_gids)
                lids = np.arange(loc_gids.size, dtype=np.int64)
                loc[rows, 0] = rank + 1
                loc[rows, 1] = lids + 1
            agg = np.maximum.reduce(comm.allgather(loc)) if comm.size > 1 else loc
            placements[et] = agg - 1  # back to 0-based

        # 5) Curved flags (global)
        curved_flags: Dict[str, np.ndarray] = {}
        for et in etypes:
            Ng = gids[et].size
            loc = np.zeros(Ng, dtype=np.int8)
            loc_gids = np.asarray(mesh.eidxs.get(et, np.empty(0, np.int64)), dtype=np.int64)
            if loc_gids.size:
                rows = cls._rows_from_gids(gids[et], loc_gids)
                curv_src = mesh.spts_curved.get(et, np.zeros(loc_gids.size, dtype=bool))

                curv_arr = np.asarray(curv_src)  # no dtype here -> OK to copy if needed
                if curv_arr.dtype != np.uint8:
                    curv_arr = curv_arr.astype(np.uint8, copy=False)

                loc[rows] = curv_arr
                
            glob = np.maximum.reduce(comm.allgather(loc)) if comm.size > 1 else loc
            curved_flags[et] = glob.astype(bool, copy=False)

        con = {}
        for et in etypes:
            Ng = gids[et].size
            con[et] = np.full((Ng, _MetaMesh._nfaces(et), 3), _CON_UNFILLED, dtype=np.int64)

        mm = cls(ndims=mesh.ndims, etypes=etypes, e_to_i=e_to_i, i_to_e=i_to_e,
            bc_name2id=bc_name2id, bc_id2name=bc_id2name,
            gids=gids, placements=placements, con=con, curved_flags=curved_flags
        )

        # Ownerless interior+boundary
        updates = mm._build_con_updates_local(mesh)
        gathered = comm.allgather(updates)
        if gathered:
            total = sum(a.shape[0] for a in gathered)
            if total:
                buf = np.empty((total, 6), dtype=np.int64)
                off = 0
                for a in gathered:
                    n = a.shape[0]
                    if n:
                        buf[off:off+n] = a
                        off += n
                mm._apply_con_updates(buf)

        mm.populate_con_from_mpi(mesh)
        mm._precompute_neighbor_rows()
        mm.recompute_mpi_iface()

        return mm

    def _rank_elem_counts(self) -> np.ndarray:
        from pyfr.mpiutil import get_comm_rank_root
        comm, _, _ = get_comm_rank_root()
        R = comm.size
        out = np.zeros(R, dtype=np.int64)
        for et in self.etypes:
            own = self.placements[et][:, 0].astype(np.int64, copy=False)
            out += np.bincount(own, minlength=R)
        return out

    def _rank_iface_counts(self) -> np.ndarray:
        from pyfr.mpiutil import get_comm_rank_root
        comm, _, _ = get_comm_rank_root()
        R = comm.size
        out = np.zeros(R, dtype=np.int64)
        for et in self.etypes:
            own   = self.placements[et][:, 0].astype(np.int64, copy=False)
            iface = self.mpi_iface.get(et, np.zeros(own.size, dtype=bool))
            if np.any(iface):
                out += np.bincount(own[iface], minlength=R)
        return out

    def _apply_con_updates(self, updates: np.ndarray) -> None:
        # updates shape (k,6): [et_code, gid, fid, v0, v1, v2]
        updates = np.asarray(updates, dtype=np.int64)
        if updates.size == 0:
            return
        for code in np.unique(updates[:, 0]):
            et = self._et_name(int(code))
            mask = (updates[:, 0] == code)
            u = updates[mask]
            rows = self._rows_from_gids(self.gids[et], u[:, 1])
            # assign in one go for each face index using fancy indexing
            # scatter per unique fid
            for fid in np.unique(u[:, 2]):
                m2 = (u[:, 2] == fid)
                rr = rows[m2]
                if rr.size == 0:
                    continue
                self.con[et][rr, int(fid), :] = u[m2, 3:6]
                
        self._precompute_neighbor_rows()

    @staticmethod
    def _nfaces(et: str) -> int:
        shapecls = subclass_where(BaseShape, name=et)
        return len(shapecls.faces)

    def _build_con_updates_local(self, mesh: "_Mesh") -> np.ndarray:
        updates = []
        e_to_i = self.e_to_i
        # interior (symmetrically add both sides)
        if mesh.con:
            conL, conR = mesh.con
            for (etL, lidL, fL), (etR, lidR, fR) in zip(conL, conR):
                gidL = int(mesh.eidxs[etL][lidL]); gidR = int(mesh.eidxs[etR][lidR])
                updates.append([e_to_i[etL], gidL, int(fL), e_to_i[etR], gidR, int(fR)])
                updates.append([e_to_i[etR], gidR, int(fR), e_to_i[etL], gidL, int(fL)])
        # boundary
        if mesh.bcon:
            for bcname, triples in mesh.bcon.items():
                bcid = self.bc_name2id[bcname]
                for et, lid, f in triples:
                    gid = int(mesh.eidxs[et][lid])
                    updates.append([self.e_to_i[et], gid, int(f), _BC_NONE, bcid, _BC_NONE])
        return np.asarray(updates, dtype=np.int64) if updates else np.empty((0,6), dtype=np.int64)

    def populate_con_from_mpi(self, mesh: "_Mesh") -> None:
        comm, myrank, _ = get_comm_rank_root()
        con = self.con

        # Build lid->row tables per etype per rank (dense, -1 default)
        lid2row = {et: {} for et in self.etypes}
        for et in self.etypes:
            owners = self.placements[et][:, 0].astype(np.int64, copy=False)
            lids   = self.placements[et][:, 1].astype(np.int64, copy=False)
            rows   = np.arange(self.gids[et].size, dtype=np.int64)
            for r in np.unique(owners):
                idx = np.nonzero(owners == r)[0]
                if idx.size == 0:
                    continue
                L = lids[idx]
                table = np.full(L.max() + 1, -1, dtype=np.int64)
                table[L] = rows[idx]
                lid2row[et][int(r)] = table

        # Gather con_p and reconcile pairs
        local_cp = {int(nbr): [(str(et), int(j), int(f)) for (et, j, f) in faces]
                    for nbr, faces in getattr(mesh, "con_p", {}).items()}
        all_cp = comm.allgather(local_cp)

        # Fill con symmetrically
        for r, rmap in enumerate(all_cp):
            for s, A in rmap.items():
                if s <= r:
                    continue
                B = all_cp[s].get(r, [])
                if len(A) != len(B):
                    raise RuntimeError(f"MPI con mismatch r{s} vs s{r}: {len(A)} != {len(B)}")

                for (etA, jA, fA), (etB, jB, fB) in zip(A, B):
                    # find global rows from (rank,lid)
                    rowA = lid2row[etA][r][jA]
                    rowB = lid2row[etB][s][jB]
                    if rowA < 0 or rowB < 0:
                        raise RuntimeError("lid→row lookup failed")

                    gidA = int(self.gids[etA][rowA])
                    gidB = int(self.gids[etB][rowB])

                    con[etA][rowA, fA] = (self._et_code(etB), gidB, fB)
                    con[etB][rowB, fB] = (self._et_code(etA), gidA, fA)

        # neighbor-row cache + iface after con is final
        self._precompute_neighbor_rows()
        self.recompute_mpi_iface()

    def _precompute_neighbor_rows(self) -> None:
        """
        Build arrays['con_nrow'][et] with neighbor ROW indices per face (or -1).
        Uses n_code/n_gid from arrays['con']; costs one searchsorted per face ONCE.
        """
        out = {}
        for et in self.etypes:
            con   = self.con[et]                      # (Ng, nfaces, 3)
            ncode = con[..., 0].astype(np.int64, copy=False)
            ngid  = con[..., 1].astype(np.int64, copy=False)

            Ng, nfaces = ncode.shape
            nrow = np.full((Ng, nfaces), -1, dtype=np.int64)

            for net in self.etypes:
                code = self._et_code(net)
                m = (ncode == code)
                if not np.any(m):
                    continue
                # vectorized gid -> row once, per neighbour etype
                rows = self._rows_from_gids(self.gids[net], ngid[m])
                nrow[m] = rows
            out[et] = nrow

        self.con_nrow = out

    def recompute_mpi_iface(self) -> None:
        mask_by_et = {}
        for et in self.etypes:
            owners = self.placements[et][:, 0].astype(np.int64, copy=False)   # (Ng,)
            neigh  = self._neighbor_owners(et)                                 # (Ng, nfaces)
            # broadcast owners across faces, compare, ignore boundary (-1)
            dif = (neigh >= 0) & (neigh != owners[:, None])
            mask_by_et[et] = np.any(dif, axis=1)
        self.mpi_iface = mask_by_et

    def _ordered_rows(self, myrank: int, et: str) -> np.ndarray:
        """
        Return owned rows for etype `et` ordered like PyFR:
            • MPI-interface elements first,
            • then curved,
            • then increasing local lid (stable).
        """
        plc  = self.placements[et]
        rows = np.where(plc[:, 0] == myrank)[0]
        if rows.size == 0:
            return rows

        # Use already-built masks
        iface  = self.mpi_iface.get(et, np.zeros(self.gids[et].size, dtype=bool))[rows]
        curved = self.curved_flags.get(et, np.zeros(self.gids[et].size, dtype=bool))[rows]
        lids   = plc[rows, 1].astype(np.int64)

        # Ascending lexsort; last key is primary → (~iface) puts iface==True first, then (~curved), then lids asc
        ords = np.lexsort((lids, ~curved, ~iface))
        return rows[ords]

    def to_mesh(self, mesh: _Mesh) -> _Mesh:
        """
        Build a new PyFR-native _Mesh using current placements, relocating
        geometry from `mesh` (the previous mesh distribution).
        """
        comm, myrank, _ = get_comm_rank_root()

        # Final local order per etype (MPI-iface first, then curved, then lid)
        owned_rows = {et: self._ordered_rows(myrank, et) for et in self.etypes}
        row2lid    = {et: {int(r): i for i, r in enumerate(owned_rows[et])} for et in self.etypes}

        cons = self._build_local_connectivity(myrank, owned_rows, row2lid)
        geom, plan = self.relocate_geometry(mesh)

        return _Mesh(fname=mesh.fname, raw=mesh.raw,
                  ndims=mesh.ndims, subset=mesh.subset,
                  creator=mesh.creator, codec=mesh.codec, uuid=mesh.uuid,
                  version = mesh.version,
                  etypes  = mesh.etypes,
                  spts=geom['spts'], spts_nodes=geom['spts_nodes'], spts_curved=geom['spts_curved'],
                  con=cons['con'], con_p=cons['con_p'], bcon=cons['bcon'],
                  eidxs={et: self.gids[et][rows].astype(np.int64, copy=False) 
                         for et, rows in owned_rows.items() if rows.size}
                  ), plan

    def _build_local_connectivity(self, myrank: int,
                                owned_rows: dict[str, np.ndarray],
                                row2lid: dict[str, dict[int, int]]
                                ) -> tuple[list, list, dict, dict]:
        """
        Recreate local connectivity (con, bcon, con_p) from global con + placements.
        Returns: (conL, conR, bcon, con_p)
        """
        conL, conR = [], []
        bcon: Dict[str, List[tuple]] = {}
        con_p: Dict[int, List[tuple]] = {}

        for et, rows in owned_rows.items():
            if not rows.size:
                continue
            nfaces = _MetaMesh._nfaces(et)

            for row in rows:
                lid = row2lid[et][int(row)]
                for f in range(nfaces):
                    v0, v1, v2 = (int(x) for x in self.con[et][row, f])
                    if v0 == _BC_NONE:
                        bcname = self.bc_id2name.get(v1, str(v1))
                        bcon.setdefault(bcname, []).append((et, lid, f))
                        continue

                    net, ngid, nf = self.i_to_e[v0], v1, v2
                    nrow = self.con_nrow[et][row, f]
                    nown = int(self.placements[net][nrow, 0])

                    if nown == myrank:
                        # avoid double insert by a stable tie-break
                        if (self.e_to_i[et], row, f) < (self.e_to_i[net], nrow, nf):
                            nlid = row2lid[net][int(nrow)]
                            conL.append((et,  lid, f))
                            conR.append((net, nlid, nf))
                    else:
                        con_p.setdefault(nown, []).append((et, lid, f))

        for nbr in list(con_p):
            con_p[nbr].sort(key=lambda t: (self.e_to_i[t[0]], t[1], t[2]))

        # return conL, conR, bcon, con_p
        # Return as dict
        return {'con': (conL, conR), 'con_p': con_p, 'bcon': bcon, }

    def _et_code(self, et: str) -> int: return int(self.e_to_i[et])
    def _et_name(self, code: int) -> str: return self.i_to_e[int(code)]

    @staticmethod
    def _rows_from_gids(sorted_gids: np.ndarray, query_gids: np.ndarray) -> np.ndarray:
        # returns -1 for not-found; assumes sorted_gids unique & ascending
        idx = np.searchsorted(sorted_gids, query_gids)
        ok  = (idx < sorted_gids.size) & (sorted_gids[idx] == query_gids)
        out = np.where(ok, idx, -1).astype(np.int64, copy=False)
        return out

    def _neighbor_owners(self, et: str) -> np.ndarray:
        con   = self.con[et]
        ncode = con[..., 0].astype(np.int64, copy=False)    # neighbor etype code or -1
        nrow  = self.con_nrow[et]                 # neighbor ROW per face or -1

        Ng, nfaces = ncode.shape
        neigh = np.full((Ng, nfaces), -1, dtype=np.int64)

        # Fill in by neighbour etype in one masked slice per etype
        for net in self.etypes:
            code = self._et_code(net)
            m = (ncode == code) & (nrow >= 0)
            if not np.any(m):
                continue
            owners_net = self.placements[net][:, 0].astype(np.int64, copy=False)
            # owners for these faces are just owners_net[row]
            neigh[m] = owners_net[nrow[m]]
        return neigh

    def _compact_all_lids(self) -> None:
        for et in self.etypes:
            plc = self.placements[et]
            r   = plc[:, 0].astype(np.int64, copy=False)
            lid = plc[:, 1].astype(np.int64, copy=False)

            order = np.lexsort((lid, r))
            r_sorted = r[order]

            # starts of runs of equal rank in r_sorted
            diff = np.empty_like(r_sorted, dtype=bool); diff[0] = True
            diff[1:] = r_sorted[1:] != r_sorted[:-1]
            run_id = np.cumsum(diff) - 1                       # 0..K-1 per run
            run_pos = np.arange(r_sorted.size, dtype=np.int64) # 0..N-1 global
            # position in run = offset from first index of that run
            first_idx = np.zeros(run_id.max()+1, dtype=np.int64)
            # record first occurrence of each run id
            first_idx[run_id[diff]] = np.nonzero(diff)[0]
            pos = run_pos - first_idx[run_id]

            plc[order, 1] = pos


    def diffuse_by_matrix(self, moves: np.ndarray) -> None:
        comm, _, _ = get_comm_rank_root()
        R = comm.size

        M = np.asarray(moves, int)
        if M.shape != (R, R) or np.any(M < 0):
            raise ValueError(f"`moves` must be {R}x{R} non-negative")
        if np.any(np.diag(M)):
            M = M.copy(); np.fill_diagonal(M, 0)

        rng = np.random.default_rng(12345)

        self._precompute_neighbor_rows()

        def _owners_iface_neigh():
            self.recompute_mpi_iface()  # uses placements + con_nrow
            owners = {et: self.placements[et][:, 0].astype(np.int64, copy=False) for et in self.etypes}
            iface  = {et: self.mpi_iface.get(et, np.zeros(owners[et].size, dtype=bool)) for et in self.etypes}
            # neighbor owners computed from current placements
            neigh  = {et: self._neighbor_owners(et) for et in self.etypes}  # (Ng, nfaces) with -1 on boundary
            codes  = {et: self._et_code(et) for et in self.etypes}
            return owners, iface, neigh, codes

        for src in range(R):
            for dst in range(R):
                need = int(M[src, dst])
                if need <= 0 or src == dst:
                    continue

                # Greedy: do need single-element moves
                for _ in range(need):
                    owners, iface, neigh, codes = _owners_iface_neigh()

                    # Build candidates with Nd and Ns
                    cands = []
                    for et in self.etypes:
                        own = owners[et]
                        rows_src = np.nonzero((own == src) & iface[et])[0]
                        if rows_src.size == 0:
                            continue

                        nb = neigh[et][rows_src]          # (k, nfaces)
                        # Count faces to dst and back to src
                        Nd = np.sum(nb == dst, axis=1)
                        m  = (Nd > 0)                      # must actually touch dst
                        if not np.any(m):
                            continue

                        rows = rows_src[m]
                        Nd   = Nd[m]
                        Ns   = np.sum(nb[m] == src, axis=1)

                        cands.append(np.c_[np.full(rows.size, codes[et], np.int64), rows, Nd, Ns])

                    if not cands:
                        break  # nothing left that touches dst

                    C = np.vstack(cands)  # cols: [et_code, row, Nd, Ns]

                    # choose: max Nd, tie-break min Ns, tie-break random
                    best_Nd = C[:, 2].max()
                    I = np.flatnonzero(C[:, 2] == best_Nd)
                    if I.size > 1:
                        best_Ns = C[I, 3].min()
                        I = I[C[I, 3] == best_Ns]
                        if I.size > 1:
                            I = np.array([rng.choice(I)])
                    i = int(I[0])

                    code, row = int(C[i, 0]), int(C[i, 1])
                    et = self._et_name(code)

                    # apply single move
                    self.placements[et][row, 0] = np.int64(dst)
                    # (We defer lid compaction until the very end)

        # After all moves: compact lids once and refresh iface
        self._compact_all_lids()
        self.recompute_mpi_iface()

    def iface_pair_counts(self) -> np.ndarray:
        rec = []
        for et in self.etypes:
            own = self.placements[et][:, 0].astype(np.int64)
            nb  = self._neighbor_owners(et)              # (Ng, nfaces)
            if nb.size == 0: 
                continue
            owner_mat = np.broadcast_to(own[:, None], nb.shape)
            m = (nb >= 0) & (nb != owner_mat) & (owner_mat < nb)  # count each face once
            if not np.any(m):
                continue
            pairs = np.c_[owner_mat[m], nb[m]]
            uniq, counts = np.unique(pairs, axis=0, return_counts=True)
            rec.append(np.c_[uniq, counts.astype(np.int64)])      # counts==1 per face
        return np.vstack(rec) if rec else np.empty((0, 3), np.int64)

    def vparts_global(self) -> np.ndarray:
        """
        Return flat per-element owner ranks in PyFR's canonical order:
        concatenate over sorted(etypes), within each etype in self.gids order.
        """
        parts = []
        for et in sorted(self.etypes):
            # placements[:,0] already aligned with self.gids[et] global rows
            parts.append(self.placements[et][:, 0].astype(np.int64, copy=False))
        v = np.concatenate(parts, dtype=np.int64)
        return v

    def preproc_edict(self, edict_src: dict[str, np.ndarray], *, edim: int = 0):
        """Ensure every etype exists and element axis is first."""
        comm, _, _ = get_comm_rank_root()
        etypes = list(self.e_to_i.keys())
        edict_dst = {}

        for etype in etypes:
            ary = edict_src.get(etype, None)
            create_ary = ary is None

            if not create_ary:
                if not isinstance(ary, np.ndarray):
                    ary = np.asarray(ary)
                if edim:
                    if ary.ndim <= edim:
                        raise ValueError(f"{etype}: expected at least {edim+1} dims, got {ary.ndim}")
                    ary = np.moveaxis(ary, edim, 0)
                ary_dtype, ary_shape = ary.dtype, ary.shape
            else:
                ary_dtype, ary_shape = None, None

            if comm.allreduce(1 if create_ary else 0, op=mpi.MAX):
                ary_dtypes = comm.allgather(ary_dtype)
                ary_shapes = comm.allgather(ary_shape)
                sig = None
                for d, s in zip(ary_dtypes, ary_shapes):
                    if d is not None and s is not None:
                        sig = (d, s)
                        break
                if sig is None:
                    raise ValueError(f"All ranks empty for etype '{etype}'; cannot infer dtype/shape.")
                dt, sh = sig
                edict_dst[etype] = np.empty((0,) + sh[1:], dtype=dt) if create_ary else ary.astype(dt, copy=False)
            else:
                edict_dst[etype] = ary

        return {etype: edict_dst[etype] for etype in etypes}

    def postproc_edict(self, edict_src: dict[str, np.ndarray], *, edim: int = 0):

        edict = {}

        for etype, ary in edict_src.items():
            if ary.size > 0:
                edict[etype] = np.moveaxis(ary, 0, edim) if edim != 0 else ary

        return edict

    def _reloc_from_mesh_base(self, src_mesh: _Mesh) -> dict:
        """
        Base pack from _Mesh on this rank.
        """
        etypes = list(self.etypes)

        base_idxs = {}
        base_gid2loc = {}
        for et in etypes:
            idxs = np.asarray(src_mesh.eidxs.get(et, ()), dtype=np.int64)
            base_idxs[et] = idxs
            base_gid2loc[et] = {int(g): i for i, g in enumerate(idxs)} if idxs.size else {}

        return {"base_idxs": base_idxs, "base_gid2loc": base_gid2loc}

    def _reloc_from_mmesh_target(self) -> dict:
        """
        Target pack from _MetaMesh.
        """

        comm, rank, root = get_comm_rank_root()
        etypes = list(self.etypes)

        end_idxs = {}
        for et in etypes:
            rows = self._ordered_rows(rank, et)
            end_idxs[et] = (self.gids[et][rows].astype(np.int64, copy=False)
                            if rows.size else np.empty(0, dtype=np.int64))

        gathered = comm.allgather(end_idxs)
        end_idxs_gathered = {et: [rmap.get(et, np.empty(0, dtype=np.int64)) 
                                    for rmap in gathered]
                            for et in etypes}

        return {"end_idxs": end_idxs, "end_idxs_gathered": end_idxs_gathered}

    def _reloc_build_plan(self, base_pack: dict, target_pack: dict) -> dict:
        """
        Build per-etype Alltoallv plan:
        send_rows, scount/sdisp, rcount/rdisp, recv_to_end, end_idxs
        """
        comm, _, _ = get_comm_rank_root()
        R = comm.size

        plan = {}

        for et in self.etypes:
            base = base_pack["base_idxs"][et].astype(np.int64, copy=False)
            loc  = base_pack["base_gid2loc"][et]
            end  = target_pack["end_idxs"][et].astype(np.int64, copy=False)
            gathered = target_pack["end_idxs_gathered"][et]

            # Per-destination send GIDs in the destination's end order
            parts = []
            scount = np.zeros(R, dtype=np.int32)
            for dst in range(R):
                want = gathered[dst].astype(np.int64, copy=False)
                keep = want[np.isin(want, base)] if want.size else want
                parts.append(keep)
                scount[dst] = keep.size

            sdisp = AlltoallMixin._count_to_disp(scount)

            # Map GIDs -> local source rows
            if scount.sum():
                sg = np.concatenate(parts)
                send_rows = np.fromiter((loc[int(g)] for g in sg),
                                        count=sg.size, dtype=np.int64)
            else:
                send_rows = np.empty(0, np.int64)

            # Exchange counts
            rcount = np.empty_like(scount)
            comm.Alltoall([scount, mpi.INT], [rcount, mpi.INT])
            rdisp = AlltoallMixin._count_to_disp(rcount)

            # recv-buffer → final end-order permutation
            if end.size:
                rows   = self._rows_from_gids(self.gids[et], end)
                owners = self.placements[et][rows, 0].astype(np.int64, copy=False)
                recv_gids = (np.concatenate([end[owners == src] for src in range(R)])
                            if owners.size else np.empty(0, np.int64))
                pos = {int(g): i for i, g in enumerate(end.tolist())}
                recv_to_end = np.fromiter((pos[int(g)] for g in recv_gids),
                                        count=recv_gids.size, dtype=np.int64)
            else:
                recv_to_end = np.empty(0, np.int64)

            plan[et] = dict(send_rows=send_rows, scount=scount, sdisp=sdisp,
                                                 rcount=rcount, rdisp=rdisp,
                                                 recv_to_end=recv_to_end,
                                                 end_idxs=end,
            )

        return plan

    @staticmethod
    def apply_relocation_plan(plan_by_etype: dict[str, dict],
                              edict_src: dict[str, np.ndarray],
                              comm) -> dict[str, np.ndarray]:
        """
        Apply precomputed relocation plan to sary_edict
        """
        mixin = AlltoallMixin()
        out: dict[str, np.ndarray] = {}

        for et, p in plan_by_etype.items():
            sc = np.asarray(p['scount'],     dtype=np.int32)
            sd = np.asarray(p['sdisp'],      dtype=np.int32)
            rc = np.asarray(p['rcount'],     dtype=np.int32)
            rd = np.asarray(p['rdisp'],      dtype=np.int32)
            sr = np.asarray(p['send_rows'],  dtype=np.int64)
            pr = np.asarray(p['recv_to_end'], dtype=np.int64)

            svals = edict_src[et][sr]
            rtot  = int(rc.sum())
            rvals = np.empty((rtot, *svals.shape[1:]), dtype=svals.dtype)

            mixin._alltoallv(comm, (svals, (sc, sd)), (rvals, (rc, rd)))
            out[et] = rvals[pr]

        return out

    def relocate(self, plan, edict_src: dict[str, np.ndarray], *, edim: int = 0) -> dict[str, np.ndarray]:
        """
        Relocate edict_src from `src_mesh` per `plan`.
        """
        # element axis first for sending
        edict_src_proc = self.preproc_edict(edict_src, edim=edim)  # (nloc, ...)

        edict_loc = self.apply_relocation_plan(plan, edict_src_proc, 
                                               get_comm_rank_root()[0])

        return self.postproc_edict(edict_loc, edim=edim)

    def relocate_geometry(self, src_mesh: _Mesh) -> dict[str, dict[str, np.ndarray]]:
        """
        Relocate geometry from `src_mesh` to this _MetaMesh's current owners.
        """
        comm, _, _ = get_comm_rank_root()

        base  = self._reloc_from_mesh_base(src_mesh)
        targ  = self._reloc_from_mmesh_target()
        plan  = self._reloc_build_plan(base, targ)

        # element axis first for sending
        spts_src   = self.preproc_edict(src_mesh.spts,        edim=1)
        nodes_src  = self.preproc_edict(src_mesh.spts_nodes,  edim=0)
        curved_src = self.preproc_edict(src_mesh.spts_curved, edim=0)

        spts_loc   = self.apply_relocation_plan(plan, spts_src,   comm)
        nodes_loc  = self.apply_relocation_plan(plan, nodes_src,  comm)
        curved_loc = self.apply_relocation_plan(plan, curved_src, comm)

        return {"spts":        self.postproc_edict(spts_loc,   edim=1), 
                "spts_nodes":  self.postproc_edict(nodes_loc,  edim=0), 
                "spts_curved": self.postproc_edict(curved_loc, edim=0)}, plan

    # --- tiny helpers --------------------------------------------------------

    def _best_dst_for_row(self, face_owners: np.ndarray, cur: int) -> tuple[int, int, int]:
        """
        Pick the destination rank that appears most among neighbour faces.
        Returns (dst, best_cnt, cur_cnt). Ignores boundary faces (-1).
        """
        f = face_owners[face_owners >= 0]
        if f.size == 0:
            return cur, 0, 0
        uniq, cnts = np.unique(f, return_counts=True)
        jmax = int(cnts.argmax())
        dst  = int(uniq[jmax])
        hit  = np.where(uniq == cur)[0]
        cur_cnt = int(cnts[hit[0]]) if hit.size else 0
        return dst, int(cnts[jmax]), cur_cnt

    def _apply_owner_moves(self, moves: list[tuple[str, int, int]]) -> None:
        """Update owners in-place; compact lids once; refresh iface."""
        for et, row, dst in moves:
            self.placements[et][row, 0] = np.int64(dst)
        if moves:
            self._compact_all_lids()
            self.recompute_mpi_iface()


    def _best_dst_for_row_tieok(self, face_owners: np.ndarray, cur: int) -> tuple[int, int, int, int]:
        """
        Like _best_dst_for_row, but also returns n_other (how many distinct neighbour ranks != cur).
        Tie-break among non-cur neighbours by smallest rank id.
        Returns: (dst, best_cnt, cur_cnt, n_other)
        """
        f = face_owners[face_owners >= 0]
        if f.size == 0:
            return cur, 0, 0, 0

        uniq, cnts = np.unique(f, return_counts=True)
        # current-owner count on this element
        w = np.where(uniq == cur)[0]
        cur_cnt = int(cnts[w[0]]) if w.size else 0

        other_mask = (uniq != cur)
        n_other = int(np.sum(other_mask))
        if n_other == 0:
            return cur, cur_cnt, cur_cnt, 0

        other_uniq = uniq[other_mask]
        other_cnts = cnts[other_mask]
        # pick highest-count neighbour; smallest rank-id on ties
        best_idx = np.flatnonzero(other_cnts == other_cnts.max())[0]
        dst = int(other_uniq[best_idx])
        best_cnt = int(other_cnts[best_idx])

        return dst, best_cnt, cur_cnt, n_other

    # --- one-pass greedy smoother -------------------------------------------

    def smooth_interfaces_greedy(self) -> None:
        """
        Reduce cut faces at expense of compute-load balance:
        • For each element, move to the neighbour rank seen most on its faces.
        • If multiple neighbour ranks exist and the top neighbour ties the current owner,
        still move to that top neighbour (deterministic tie-break).
        """
        from pyfr.mpiutil import get_comm_rank_root
        self._precompute_neighbor_rows()

        moves: list[tuple[str, int, int]] = []
        tie_moves = 0

        for et in self.etypes:
            nb  = self._neighbor_owners(et)   # (Ng, nfaces), -1 on boundary
            own = self.placements[et][:, 0].astype(np.int64, copy=False)
            if nb.size == 0:
                continue

            for row, cur in enumerate(own):
                dst, best_cnt, cur_cnt, n_other = self._best_dst_for_row_tieok(nb[row], int(cur))
                if dst == cur:
                    continue

                # Move if strict reduction OR (tie with ≥2 distinct neighbour ranks)
                if best_cnt > cur_cnt or (best_cnt == cur_cnt and n_other >= 2):
                    moves.append((et, row, dst))
                    if best_cnt == cur_cnt:
                        tie_moves += 1

        self._apply_owner_moves(moves)

        # concise verification log on rank 0
        comm, rank, _ = get_comm_rank_root()
        if rank == 0:
            print(f"[smooth] greedy pass done | moves={len(moves)} tie_moves={tie_moves}")

    def _round_to_sum(self, total: int, weights: np.ndarray) -> np.ndarray:
        """Stable floor + largest remainders to match `total`."""
        w = np.asarray(weights, dtype=float).ravel()
        if w.sum() <= 0 or total <= 0:
            return np.zeros_like(w, dtype=np.int64)
        raw = (w / w.sum()) * float(total)
        base = np.floor(raw).astype(np.int64)
        rem  = int(total - base.sum())
        if rem > 0:
            frac = raw - base
            idx = np.argsort(-frac, kind='mergesort')[:rem]  # stable ties
            base[idx] += 1
        return base

    def plan_relocation(self, weights: np.ndarray, *, verbose: bool = True) -> np.ndarray:
        """
        Build an R×R integer move matrix using *rank-wise* weights only.
        - Targets E* = round_to_sum(total, weights)
        - Convert Δ = have - target to flows along feasible edges (directed candidates).
        """
        import numpy as np
        from pyfr.mpiutil import get_comm_rank_root
        comm, rank, _ = get_comm_rank_root()
        R = comm.size

        # --- counts we have now (aggregate across etypes)
        E = self._rank_elem_counts().astype(np.int64, copy=False)
        T = int(E.sum())

        w = np.asarray(weights, dtype=float).ravel()
        if w.size != R:
            raise ValueError(f"[plan-w] load-balance-weights must have {R} values, got {w.size}")

        target = self._round_to_sum(T, w)
        delta  = E - target
        surplus = np.maximum(delta, 0).astype(np.int64)
        deficit = np.maximum(-delta, 0).astype(np.int64)

        # --- directed candidate elements per edge (aggregate over etypes)
        C = np.zeros((R, R), dtype=np.int64)
        for et in self.etypes:
            owners = self.placements[et][:, 0].astype(np.int64, copy=False)
            nb     = self._neighbor_owners(et)
            if nb.size == 0:
                continue
            for row in range(owners.size):
                src = int(owners[row])
                neigh = nb[row]
                m = (neigh >= 0) & (neigh != src)
                if not np.any(m):
                    continue
                dsts = np.unique(neigh[m].astype(np.int64))
                C[src, dsts] += 1
        np.fill_diagonal(C, 0)

        # --- allocate flows ∝ directed candidate counts, capped by deficits
        M = np.zeros((R, R), dtype=np.int64)
        for src in range(R):
            ss = int(surplus[src])
            if ss <= 0:
                continue
            nbrs = np.array([dst for dst in range(R)
                            if dst != src and deficit[dst] > 0 and C[src, dst] > 0], dtype=np.int64)
            if nbrs.size == 0:
                continue
            cap = C[src, nbrs].astype(np.float64)
            base = min(ss, int(deficit[nbrs].sum()))
            if base <= 0 or cap.sum() <= 0:
                continue

            fp = (cap / cap.sum()) * float(base)
            take = np.floor(fp).astype(np.int64)
            rem = int(base - int(take.sum()))
            if rem > 0:
                frac = fp - take
                add_idx = np.argsort(-frac, kind='mergesort')[:rem]
                take[add_idx] += 1

            # cap by remaining neighbor deficits and by candidates
            take = np.minimum(take, deficit[nbrs])
            take = np.minimum(take, C[src, nbrs])

            got = int(take.sum())
            if got > 0:
                M[src, nbrs] += take
                surplus[src] -= got
                deficit[nbrs] -= take
                C[src, nbrs]  -= take

        if verbose and rank == 0:
            print(f"[plan-w] have={E.tolist()} w={(w/np.maximum(w.sum(),1e-12)).round(4).tolist()} "
                f"target={target.tolist()} delta={(E - target).tolist()}")
            nz = [(int(i), int(j), int(M[i,j])) for i in range(R) for j in range(R) if i!=j and M[i,j]>0]
            preview = " ".join(f"{i}->{j}:{v}" for i, j, v in nz) if nz else "(none)"

        return M



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
