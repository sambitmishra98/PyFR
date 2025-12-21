from collections import deque
from shutil import move
from typing import List, Optional

import numpy as np

from pyfr.mpiutil import comm, rank, root, mpi
from pyfr.partitioners.online.base import OnlinePartitioner, CarverMixin, OfflineRepartitioner

class DiffusionRepartitioner(CarverMixin, OfflineRepartitioner):
    """
        Does not use config file at all.
    """

    def __init__(self, mesh, cfg):
        OfflineRepartitioner.__init__(self, mesh, cfg)
        CarverMixin.__init__(self, cfg)

    @classmethod
    def from_vparts(cls, vparts, cfg):

        # Create a mesh object with dummy data just to satisfy the base class
        mesh = cls._mesh_from_vparts(vparts, cfg)
        mmesh = cls(mesh, cfg)
        return mmesh

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

    # ----------------- Delta computations -----------------------
    def _compute_affinity(self, mode: str):
        m = str(mode).lower()
        if m in ("face", "faces", "edge", "edges"):
            return self._calc_mpi_faces_affinity()
        elif m in ("vertex", "vertices"):
            return self._calc_mpi_vertices_affinity()
        else:
            raise ValueError(f"_compute_affinity: invalid mode {mode!r}")

    def _compute_deltas(self, score_by: str) -> dict[int, dict[str, np.ndarray]]:
        sb = str(score_by).lower()
        if sb in ("edge", "edges", "face", "faces"):
            return self._calc_mpi_faces_deltas()      # your con_mpi-based integer deltas
        elif sb in ("vertex", "vertices"):
            return self.compute_mpi_face_delta_from_vertices()
        else:
            raise ValueError(f"_compute_deltas: unknown score_by={score_by!r}")

    def compute_mpi_face_delta_from_vertices(self) -> dict[int, dict[str, "np.ndarray"]]:
        """
        Vertex deltas (legacy parity).

        Returns {nbr: {et: (N,2) int64 [[lid, delta], ...] sorted by (delta, lid)}}
        delta = c_int - c_n
        c_n   = #corner-vertices of element in nbr's MPI-vertex set
        c_int = #corner-vertices of element NOT in union of all MPI-face vertices
        Only MPI-face interface elements to nbr are eligible (lids via _iface_lids_by_neighbor_faces()).
        """
        import numpy as np

        st = self.i
        iface = self._iface_lids_by_neighbor_faces()
        if not iface:
            return {}

        mvu = self.collect_mpi_vertex_nodes()
        if not mvu:
            return {}

        allv = [np.asarray(v, dtype=np.int64).ravel() for v in mvu.values() if v is not None and len(v)]
        union_all = np.unique(np.concatenate(allv)) if allv else np.empty(0, dtype=np.int64)

        out: dict[int, dict[str, np.ndarray]] = {}

        for nbr in sorted(int(k) for k in iface.keys()):
            nbr_vertices = np.asarray(mvu.get(nbr, ()), dtype=np.int64).ravel()
            if nbr_vertices.size == 0:
                continue

            per: dict[str, np.ndarray] = {}

            for et in self._etype_order():
                lids_iface = iface.get(nbr, {}).get(et, None)
                if lids_iface is None or lids_iface.size == 0:
                    continue

                nds_full = st.spts_nodes.get(et, None)
                if nds_full is None or nds_full.size == 0:
                    continue

                vcols = self._vertex_cols(et)
                if vcols.size == 0:
                    continue

                nds_v = np.asarray(nds_full[:, vcols], dtype=np.int64, copy=False)
                nds_i = nds_v[lids_iface]                 # only interface elements
                valid = (nds_i >= 0)

                c_n = (np.isin(nds_i, nbr_vertices, assume_unique=False) & valid).sum(axis=1).astype(np.int16, copy=False)
                if union_all.size:
                    c_int = ((~np.isin(nds_i, union_all, assume_unique=False)) & valid).sum(axis=1).astype(np.int16, copy=False)
                else:
                    c_int = valid.sum(axis=1).astype(np.int16, copy=False)

                sel = (c_n > 0)
                if not np.any(sel):
                    continue

                lids  = lids_iface[sel].astype(np.int64, copy=False)
                delta = (c_int[sel].astype(np.int32) - c_n[sel].astype(np.int32)).astype(np.int64, copy=False)

                mat = np.column_stack((lids, delta)).astype(np.int64, copy=False)
                mat = mat[np.lexsort((mat[:, 0], mat[:, 1]))]  # (delta, lid)

                per[et] = mat

            if per:
                out[nbr] = per

        return out

    def _mpi_faces_by_neighbor(self) -> dict[int, list[tuple[str, int, int]]]:
        """
        {nbr: [(etype, lid, fidx), ...]} for MPI faces only.
        Deterministic: neighbour keys sorted when iterated later.
        """
        myr = int(rank["world"])
        per_nbr: dict[int, list[tuple[str, int, int]]] = {}

        for et in self.etypes:
            owners = self.i.con_mpi[et]
            eids   = self.i.con_idx[et]
            if owners.size == 0 or eids.size == 0:
                continue

            mpi_mask = (eids >= 0) & (owners >= 0) & (owners != myr)
            if not np.any(mpi_mask):
                continue

            lids, fidxs = np.nonzero(mpi_mask)
            nbrs = owners[lids, fidxs].astype(np.int64, copy=False)

            # group; keep payload
            for nbr in np.unique(nbrs):
                nbr = int(nbr)
                if nbr == myr:
                    continue
                sel = (nbrs == nbr)
                lst = per_nbr.setdefault(nbr, [])
                lst.extend((et, int(lid_i), int(f_i)) for lid_i, f_i in zip(lids[sel], fidxs[sel]))

        return per_nbr

    def _calc_mpi_faces_affinity(self) -> dict[int, dict[str, np.ndarray]]:
        myr = int(rank['world'])
        empty = np.empty((0, 3), dtype=np.int64)

        per_et_meta: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}
        nbrs: set[int] = set()

        for et in self.etypes:
            owners = self.i.con_mpi[et]
            eids   = self.i.con_idx[et]
            if owners.size == 0 or eids.size == 0:
                per_et_meta[et] = None
                continue

            # “valid element neighbour” faces
            valid = (eids >= 0) & (owners >= 0)

            # boundary elements: at least one MPI face (owner != myr)
            mpi_face_mask = valid & (owners != myr)
            has_mpi_face  = np.any(mpi_face_mask, axis=1)
            if not np.any(has_mpi_face):
                per_et_meta[et] = None
                continue

            lids_bnd       = np.nonzero(has_mpi_face)[0].astype(np.int64, copy=False)
            owners_bnd     = owners[has_mpi_face]
            valid_bnd      = valid[has_mpi_face]
            owners_eff_bnd = np.where(valid_bnd, owners_bnd, -1)

            uniq = np.unique(owners_eff_bnd)
            uniq = uniq[(uniq >= 0) & (uniq != myr)]
            for r in uniq:
                nbrs.add(int(r))

            c_me_bnd = np.sum(owners_eff_bnd == myr, axis=1).astype(np.int16, copy=False)
            per_et_meta[et] = (lids_bnd, c_me_bnd, owners_eff_bnd)

        if not nbrs:
            return {}

        out: dict[int, dict[str, np.ndarray]] = {}
        for nr in sorted(nbrs):
            per: dict[str, np.ndarray] = {}
            for et in self.etypes:
                meta = per_et_meta.get(et)
                if meta is None:
                    per[et] = empty
                    continue

                lids_bnd, c_me_bnd, owners_eff_bnd = meta
                c_n_bnd = np.sum(owners_eff_bnd == nr, axis=1).astype(np.int16, copy=False)

                sel = (c_n_bnd > 0)
                if not np.any(sel):
                    per[et] = empty
                    continue

                lids = lids_bnd[sel]
                a    = c_me_bnd[sel].astype(np.int64, copy=False)
                b    = c_n_bnd[sel].astype(np.int64, copy=False)

                per[et] = np.c_[lids, a, b]

            out[int(nr)] = per

        return out

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

    def collect_mpi_vertex_nodes(self) -> dict[int, "np.ndarray"]:
        """
        {nbr: sorted unique global vertex-node IDs that lie on MPI faces to nbr}
        """
        import numpy as np

        if not getattr(self, "_spts_valid", True):
            raise RuntimeError("spts_nodes are stale")
        if self.i.spts_nodes is None:
            raise RuntimeError("self.i.spts_nodes missing")

        per_nbr_faces = self._mpi_faces_by_neighbor()
        face_vtx_by_et = {et: self._face_vertex_indices(et) for et in self.etypes}

        out: dict[int, np.ndarray] = {}

        for nbr in sorted(int(k) for k in per_nbr_faces.keys()):
            faces = per_nbr_faces[nbr]
            verts_all: list[np.ndarray] = []

            for et, lid, f in faces:
                nds = self.i.spts_nodes.get(et)
                if nds is None or nds.size == 0:
                    continue
                if not (0 <= int(lid) < int(nds.shape[0])):
                    continue

                fv = face_vtx_by_et[str(et)][int(f)]
                verts = nds[int(lid), fv]
                if verts.size:
                    verts_all.append(np.asarray(verts, dtype=np.int64).ravel())

            if verts_all:
                vcat = np.concatenate(verts_all).astype(np.int64, copy=False)
                vcat = vcat[vcat >= 0]
                out[int(nbr)] = np.unique(vcat)
            else:
                out[int(nbr)] = np.empty(0, dtype=np.int64)

        return out

    def _metric_from_ab(self, a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
        # Always float64 for future-proofing and to support ratio cleanly.
        a = a.astype(np.float64, copy=False)
        b = b.astype(np.float64, copy=False)

        metric = str(metric).lower()
        if metric == 'delta':
            return a - b
        elif metric == 'ratio':
            den = a + b
            # ratio in [0,1] when den>0; smaller => more neighbour-tied
            out = np.full_like(a, np.inf, dtype=np.float64)
            np.divide(a, den, out=out, where=(den > 0.0))
            return out
        else:
            raise ValueError(f"Unknown metric {metric!r}; expected 'delta' or 'ratio'")

    def _calc_mpi_vertices_affinity(self) -> dict[int, dict[str, np.ndarray]]:
        """
        Vertex-based affinity for MPI-face interface elements only.
        Returns {nbr: {et: (N,3) [lid, cnt_int, cnt_nbr]}}.
        Deterministic: neighbour iteration sorted; per-et arrays lexsorted by lid.
        """

        st = self.i
        mvu = self.collect_mpi_vertex_nodes()   # {nbr: vertex IDs on MPI faces to nbr}
        if not mvu:
            return {}

        # union of all MPI-face vertices (for "interior vertex" counting)
        all_sets = [np.asarray(v, dtype=np.int64).ravel() for v in mvu.values() if v is not None and np.asarray(v).size]
        union_all = np.unique(np.concatenate(all_sets)) if all_sets else np.empty(0, dtype=np.int64)

        # Build iface lids per (nbr, et) from actual MPI faces
        per_nbr_faces = self._mpi_faces_by_neighbor()
        iface_lids: dict[tuple[int, str], np.ndarray] = {}
        for nbr, faces in per_nbr_faces.items():
            by_et = {}
            for et, lid, f in faces:
                by_et.setdefault(et, []).append(int(lid))
            for et, lids in by_et.items():
                iface_lids[(int(nbr), str(et))] = np.unique(np.asarray(lids, dtype=np.int64))

        out: dict[int, dict[str, np.ndarray]] = {}

        for nbr in sorted(int(k) for k in mvu.keys()):
            nbr_vertices = np.asarray(mvu.get(nbr, ()), dtype=np.int64).ravel()
            if nbr_vertices.size == 0:
                continue

            per_et: dict[str, np.ndarray] = {}

            for et in self._etype_order():
                nodes = st.spts_nodes.get(et, None)
                lids_iface = iface_lids.get((nbr, et), np.empty(0, dtype=np.int64))
                if nodes is None or nodes.size == 0 or lids_iface.size == 0:
                    continue

                vcols = self._vertex_cols(et)
                if vcols.size == 0:
                    continue

                nds = np.asarray(nodes[lids_iface][:, vcols], dtype=np.int64)

                # cnt_nbr: how many corner vertices belong to nbr's MPI-vertex set
                cnt_n = (np.isin(nds, nbr_vertices) & (nds >= 0)).sum(axis=1).astype(np.int32, copy=False)
                # cnt_int: how many corner vertices are NOT in union_all
                if union_all.size:
                    cnt_i = (~np.isin(nds, union_all) & (nds >= 0)).sum(axis=1).astype(np.int32, copy=False)
                else:
                    cnt_i = (nds >= 0).sum(axis=1).astype(np.int32, copy=False)

                # Elements on MPI faces to nbr should have cnt_n > 0; enforce as a debug invariant
                keep = (cnt_n > 0)

                lids = lids_iface[keep].astype(np.int64, copy=False)
                a = cnt_i[keep].astype(np.int64, copy=False)
                b = cnt_n[keep].astype(np.int64, copy=False)

                mat = np.empty((lids.size, 3), dtype=np.int64)
                mat[:, 0] = lids
                mat[:, 1] = a
                mat[:, 2] = b

                # deterministic ordering by lid (diffuse() later tie-breaks by gid anyway)
                mat = mat[np.lexsort((mat[:, 0],))]

                per_et[et] = mat

            if per_et:
                out[nbr] = per_et

        return out

    def _iface_lids_by_neighbor_faces(self) -> dict[int, dict[str, "np.ndarray"]]:
        """
        Canonical interface definition (parity anchor).
        {nbr: {et: unique_sorted_lids_on_MPI_faces_to_nbr}}
        """
        import numpy as np

        per_nbr_faces = self._mpi_faces_by_neighbor()  # {nbr: [(et,lid,fidx), ...]}
        out: dict[int, dict[str, np.ndarray]] = {}

        for nbr in sorted(int(k) for k in per_nbr_faces.keys()):
            faces = per_nbr_faces[nbr]
            by_et: dict[str, list[int]] = {}

            for et, lid, _f in faces:
                by_et.setdefault(str(et), []).append(int(lid))

            per: dict[str, np.ndarray] = {}
            for et in self._etype_order():
                lids = by_et.get(et, [])
                if lids:
                    per[et] = np.unique(np.asarray(lids, dtype=np.int64))

            if per:
                out[int(nbr)] = per

        return out

    def _vertex_cols(self, et: str) -> "np.ndarray":
        """
        Cached vertex-column indices within spts_nodes[et].
        """
        import numpy as np

        et = str(et).lower()
        cache = getattr(self, "_vcols_by_et", None)
        if cache is None:
            cache = self._vcols_by_et = {}

        v = cache.get(et, None)
        if v is None:
            fidx = self._face_vertex_indices(et)  # list[np.ndarray]
            v = np.unique(np.concatenate(fidx).astype(np.int64, copy=False)) if fidx else np.empty(0, np.int64)
            cache[et] = v

        return v

    # ----------------- Legacy-style smoothing pass ----------------------------

    def smooth(self, metric: str = "delta", 
                     etype_scale: dict[str, float] | None = None,
                     device_bias: dict[str, dict[str, float]] | None = None,
                     rank_tags: list[str] | None = None,
                     move_spts_nodes=False) -> int:
        metric = str(metric).lower()
        if metric not in ("delta", "ratio"):
            raise ValueError(f"smooth: metric must be 'delta' or 'ratio', got {metric!r}")

        if device_bias is not None and rank_tags is None:
            raise ValueError("smooth: rank_tags must be provided when device_bias is used")

        self._reset_j_with_i()
        st = self.i

        aff_by_rank = self._compute_affinity("faces")  # {nbr:{et:[lid,a,b]}}
        etypes_all  = list(self._etype_order())

        def metric_from_ab(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            a = a.astype(np.float64, copy=False)
            b = b.astype(np.float64, copy=False)
            if metric == "delta":
                return a - b
            den = a + b
            out = np.full_like(a, np.inf, dtype=np.float64)
            np.divide(a, den, out=out, where=(den > 0.0))
            return out

        def score_metric(x: np.ndarray, *, et: str, nbr: int) -> np.ndarray:
            s = x
            if etype_scale is not None:
                s = s * float(etype_scale.get(et, 1.0))
            if device_bias is not None:
                tag = rank_tags[int(nbr)]
                s = s + float(device_bias.get(tag, {}).get(et, 0.0))
            return s

        flats: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        gids: list[np.ndarray] = []
        nbrs: list[np.ndarray] = []

        for nbr in sorted(int(n) for n in aff_by_rank.keys()):
            per_et = aff_by_rank[nbr]
            for et in etypes_all:
                arr = per_et.get(et, None)
                if arr is None or arr.size == 0:
                    continue

                lids = arr[:, 0].astype(np.int64, copy=False)
                a    = arr[:, 1].astype(np.int64, copy=False)
                b    = arr[:, 2].astype(np.int64, copy=False)

                mraw = metric_from_ab(a, b)
                mkeep = (mraw <= -1)
                if not np.any(mkeep):
                    continue

                lids_k = lids[mkeep]
                flat_k = st.lids_to_flat(et, lids_k)

                mraw_k   = mraw[mkeep]
                score_k  = score_metric(mraw_k, et=et, nbr=nbr)
                gid_k    = st.eidxs[et][lids_k].astype(np.int64, copy=False)

                flats.append(flat_k)
                scores.append(score_k)
                gids.append(gid_k)
                nbrs.append(np.full(flat_k.shape, nbr, dtype=np.int64))

        if not flats:
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
            return 0

        flat_all  = np.concatenate(flats)
        score_all = np.concatenate(scores).astype(np.float64, copy=False)
        gid_all   = np.concatenate(gids)
        nbr_all   = np.concatenate(nbrs)

        # Deterministic “best neighbour per element”:
        # primary: flat id, then best score, then gid, then nbr
        order = np.lexsort((nbr_all, gid_all, score_all, flat_all))

        flat_s   = flat_all[order]
        nbr_s    = nbr_all[order]
        score_s  = score_all[order]
        gid_s    = gid_all[order]

        _, first = np.unique(flat_s, return_index=True)
        chosen_flat  = flat_s[first]
        chosen_nbr   = nbr_s[first]
        chosen_score = score_s[first]
        chosen_gid   = gid_s[first]

        eidxs_diff = self._build_eidxs_diff_from_flat(chosen_flat, chosen_nbr, st)
        self._apply_plan_and_commit(eidxs_diff, move_spts_nodes=move_spts_nodes)

        return int(chosen_flat.size)

    def smooth_until_stagnates(self, max_iters = 20,
                               move_spts_nodes=False) -> list[int]:
        stable = 0
        last: Optional[int] = None

        for _ in range(int(max_iters)):
            moved_local = self.smooth(metric="delta", move_spts_nodes=move_spts_nodes)
            moved_glob = int(comm["world"].allreduce(int(moved_local), op=mpi.SUM))

            if last is not None and abs(moved_glob - last) <= 0:
                stable += 1
            else:
                stable = 0
            last = moved_glob

            stop = (moved_glob == 0) or (stable >= 1)
            stop_all = int(comm["world"].allreduce(1 if stop else 0, op=mpi.MAX))

            if stop_all:
                break

    def diffuse_smoothing2(self, flow_matrix, *, threshold: float = 0.0,
                                 scale: float = 0.5, score_by: str = "vertex",):

        M = np.asarray(flow_matrix, dtype=np.int64)

        # legacy-style relaxation: ceil
        scale = float(scale)
        M_eff = np.ceil(M.astype(np.float64) * scale).astype(np.int64)
        M_eff[M_eff < 0] = 0
        np.fill_diagonal(M_eff, 0)

        mode = "vertices" if str(score_by).lower().startswith("v") else "faces"

        moved_local, flow_used, eidxs_diff = self.diffuse(mode=mode,
                                                          threshold=threshold,
                                                          flow_matrix=M_eff,
                                        move_spts_nodes=(mode == "vertices"),)

        moved_glob = int(comm["world"].allreduce(int(moved_local), op=mpi.SUM))

        return moved_glob, eidxs_diff

    def iterate(self, target_counts: List[int], *, flowmat_relax: float = 0.5) -> list[int]:
        """
        Debug controller (kept intentionally):
        - for now executes ONE pass: (thr=6, score_by='vertex')
        - prints key state for LEGACY parity debugging.
        """
        # Force the next debug step: one vertex-based iteration with thr=6
        exec_order =(  [(6.0, "vertex")]
                     #+ [(2.0, "face")]
                     #+ [(0.0, "face")] * comm["world"].size
                    )
        # TEST WITH/WITHOUT ABOVE FACE MOVEMENTS !!!! 
        # NO FACE MOVEMENTS GIVES BETTER ANSWER!!!!

        for thr, score_by in exec_order:
            M0 = self.element_flow_plan(target_counts)
            self.diffuse_smoothing2(M0, threshold=thr, scale=flowmat_relax, score_by=score_by)
            self.smooth_until_stagnates(move_spts_nodes=True)

    def diffuse_till_convergence(self, target_counts: List[int], *, flowmat_relax: float = 0.5, 
                                 max_iters: int = 1):
        if rank["world"] == root['world']:
            print(f"TARGET: {target_counts}")

        iters = 0

        while True:
            if max_iters != -1 and iters >= max_iters:  
                break   
            iters += 1

            cur0 = self._cur_counts_total
            if rank["world"] == root['world']:
                print(f"CURRENT: {cur0}")     

            self.iterate(target_counts, flowmat_relax=flowmat_relax)

            cur1 = self._cur_counts_total
            if cur1 == cur0:
                break

    def diffuse(self, *, mode: str = "faces", threshold: float = 0.0,
        flow_matrix, move_spts_nodes: bool = False):
        import numpy as np

        W   = comm["world"]
        rnk = int(rank["world"])
        P   = int(W.size)

        mode = str(mode).lower()
        mode = "vertices" if mode.startswith("v") else "faces"

        thr = int(threshold) if float(threshold).is_integer() else float(threshold)

        # Vertex mode must keep spts_nodes consistent for subsequent vertex scoring
        if mode == "vertices":
            move_spts_nodes = True

        M = np.asarray(flow_matrix, dtype=np.int64)
        if M.ndim != 2 or M.shape[0] != M.shape[1] or int(M.shape[0]) != P:
            raise ValueError(f"diffuse: bad flow_matrix shape {M.shape}, comm size {P}")

        # Always start from i -> j (collective safety relies on everyone committing)
        self._reset_j_with_i()
        st = self.i
        etypes_all = list(self._etype_order())

        # Canonical candidate source (per neighbour, per etype: [lid, delta])
        deltas_by_rank = self._compute_deltas(mode)  # {nbr: {et: (N,2) int64}}

        picked_by_et = {et: set() for et in etypes_all}     # gid de-dupe per etype, per sweep
        eidxs_diff: dict[int, dict[str, np.ndarray]] = {}
        flow_used = np.zeros_like(M, dtype=np.int64)

        for nbr in range(P):
            cap = int(M[rnk, nbr])
            if cap <= 0 or nbr == rnk:
                continue

            per_et = deltas_by_rank.get(int(nbr), {})
            cand: list[tuple[int, int, str]] = []  # (delta, gid, et)

            # Deterministic etype iteration
            for et in etypes_all:
                mat = per_et.get(et, None)
                if mat is None or mat.size == 0:
                    continue

                lids = mat[:, 0].astype(np.int64, copy=False)
                dlt  = mat[:, 1].astype(np.int64, copy=False)

                m_thr = (dlt <= thr)
                if not np.any(m_thr):
                    continue

                lids_t = lids[m_thr]
                dlt_t  = dlt[m_thr]

                gids_t = st.eidxs[et][lids_t].astype(np.int64, copy=False)

                # Legacy semantics: pick at most once per sweep by gid (per etype)
                pset = picked_by_et[et]
                for gid, dd in zip(gids_t.tolist(), dlt_t.tolist()):
                    if gid not in pset:
                        cand.append((int(dd), int(gid), et))

            if not cand:
                continue

            # Deterministic: sort by (delta, gid); python sort is stable
            cand.sort(key=lambda t: (t[0], t[1]))

            chosen = cand[: min(cap, len(cand))]

            per_out: dict[str, list[int]] = {}
            for dd, gid, et in chosen:
                per_out.setdefault(et, []).append(gid)
                picked_by_et[et].add(gid)

            # Finalize deterministic arrays per etype (sorted gids)
            out_per = {
                et: np.asarray(sorted(v), dtype=np.int64)
                for et, v in per_out.items()
                if v
            }

            if out_per:
                eidxs_diff[int(nbr)] = out_per
                flow_used[rnk, int(nbr)] = int(sum(len(v) for v in out_per.values()))

        moved_local = int(sum(len(g) for per in eidxs_diff.values() for g in per.values()))
        self._apply_plan_and_commit(eidxs_diff, move_spts_nodes=move_spts_nodes)
        return moved_local, flow_used, eidxs_diff

    def remove_islands_till_convergence(self, max_iters=-1):

        iters = 0

        while True:
            if max_iters != -1 and iters >= max_iters:  
                break   
            iters += 1

            cur0 = self._cur_counts_total
            nrms = self.remove_small_islands_step()
            #self.remove_outliers()
            #self.add_inliers()

            self.smooth_until_stagnates(move_spts_nodes=True)
            #if not any(nrms[i] > 1.0 for i in range(comm['world'].size) if targets[i] > 0):
            #    break
            # If no element movements happen, then exit. So compare last with now
            cur1 = self._cur_counts_total
            if cur1 == cur0:
                break
            if all(nrms[i] <= 1.0 for i in range(comm['world'].size)):
                break


class OnlineDiffusionPartitioner(DiffusionRepartitioner, OnlinePartitioner):

    def __init__(self, mesh, cfg):
        OnlinePartitioner.__init__(self, mesh, cfg)
        DiffusionRepartitioner.__init__(self, mesh, cfg)
