from collections import deque
from typing import List, Optional

import numpy as np

from pyfr.mpiutil import comm, rank, root, mpi
from pyfr.partitioners.online.base import OnlinePartitioner

class OnlineDiffusionPartitioner(OnlinePartitioner):

    def __init__(self, mesh, cfg):
        OnlinePartitioner.__init__(self, mesh, cfg)

        # Island stuff
        self.island_remove_fraction   = cfg.getfloat('partition', 'island-remove-fraction'  , 0.5)
        self.outlier_removal_mode     = cfg.get(     'partition', 'outlier-removal-mode'    , 'faces')
        self.outlier_removal_fraction = cfg.getfloat('partition', 'outlier-removal-fraction', 0.00)
        self.inlier_addition_mode     = cfg.get(     'partition', 'inlier-addition-mode'    , 'vertices')
        self.inlier_addition_fraction = cfg.getfloat('partition', 'inlier-addition-fraction', 0.00)

    # ----------------- Flow planning -----------------------

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
                     rank_tags: list[str] | None = None) -> int:
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
            self._apply_plan_and_commit({}, move_spts_nodes=False)
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
        self._apply_plan_and_commit(eidxs_diff, move_spts_nodes=False)

        return int(chosen_flat.size)

    def smooth_until_stagnates(self, max_iters = 20) -> list[int]:
        stable = 0
        last: Optional[int] = None

        for _ in range(int(max_iters)):
            moved_local = self.smooth(metric="delta")
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

    def iterate_to_convergence(self, target_counts: List[int], *,
                                     flowmat_relax: float = 0.5,
    ) -> list[int]:
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
            self.smooth_until_stagnates()

    # ----------------- Diffusion / smoothing -----------------------

    def diffuse(
        self,
        *,
        mode: str = "faces",
        threshold: float = 0.0,
        flow_matrix,
        move_spts_nodes: bool = False,
    ):
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


    def _canonical_step(self, step: dict) -> dict:
        """
        Normalise one exec_order entry into a fully-populated config dict.

        Semantics:
        - use_flow=True  -> calls self.diffuse(...) with a flow_matrix (required).
        - use_flow=False -> calls self.smooth(...) (no flow matrix).
        """
        if not isinstance(step, dict):
            raise TypeError(f"exec_order entries must be dicts; got {type(step).__name__}: {step!r}")

        cfg = dict(
            kind=None,               # e.g. 'vertices-flow', 'faces-flow', 'faces-smooth'
            name=None,               # pretty-print label
            mode=None,               # 'faces' or 'vertices' (inferred from kind if None)
            use_flow=False,          # True => diffuse; False => smooth
            metric="delta",          # 'delta' or 'ratio'
            threshold=0.0,           # float; metric <= threshold
            overshoot=0.0,           # only when use_flow=True
            max_sweeps=1,            # only when use_flow=True (<=0 => safe default upper bound)
            max_iters=50,            # only when use_flow=False
            patience=1,              # only when use_flow=False
            min_change=0,            # only when use_flow=False
            move_spts_nodes=False,   # forced True for vertices
            # future knobs; defaults replicate legacy when None
            etype_scale=None,        # dict[str,float] or None
            device_bias=None,        # dict[tag][etype] or None
            rank_tags=None,          # list[str] or None
        )
        cfg.update(step)

        kind = cfg["kind"]
        if kind is None:
            raise ValueError(f"exec_order step has no 'kind': {step!r}")

        # Infer mode if missing
        if cfg["mode"] is None:
            k = str(kind)
            if k.startswith("faces-"):
                cfg["mode"] = "faces"
            elif k.startswith("vertices-"):
                cfg["mode"] = "vertices"
            else:
                raise ValueError(f"Cannot infer mode from kind={kind!r}; set mode explicitly.")

        mode = cfg["mode"]
        if mode not in ("faces", "vertices"):
            raise ValueError(f"mode must be 'faces' or 'vertices', got {mode!r}")

        # Infer use_flow if not explicitly set
        # (keeps exec_order compact and legacy-readable)
        if "use_flow" not in step:
            cfg["use_flow"] = str(kind).endswith("-flow")

        metric = str(cfg["metric"]).lower()
        if metric not in ("delta", "ratio"):
            raise ValueError(f"metric must be 'delta' or 'ratio', got {cfg['metric']!r}")
        cfg["metric"] = metric

        # Vertex steps must keep spts_nodes consistent
        if mode == "vertices":
            cfg["move_spts_nodes"] = True

        # Normalise numerics
        cfg["threshold"] = float(cfg["threshold"])
        cfg["overshoot"] = float(cfg["overshoot"])
        cfg["max_sweeps"] = int(cfg["max_sweeps"])
        cfg["max_iters"] = int(cfg["max_iters"])
        cfg["patience"] = int(cfg["patience"])
        cfg["min_change"] = int(cfg["min_change"])

        # Flow vs non-flow constraints
        if cfg["use_flow"]:
            if cfg["overshoot"] < 0.0:
                raise ValueError(f"overshoot must be >= 0, got {cfg['overshoot']}")
            if cfg["max_sweeps"] == 0:
                raise ValueError("max_sweeps=0 is meaningless; use 1 or <=0 for default bound")
        else:
            if abs(cfg["overshoot"]) > 1e-14:
                raise ValueError(f"overshoot={cfg['overshoot']} is only valid when use_flow=True")
            if cfg["max_iters"] <= 0:
                raise ValueError(f"max_iters must be > 0 for smoothing steps, got {cfg['max_iters']}")

        # If device_bias is provided, rank_tags must exist
        if cfg["device_bias"] is not None and cfg["rank_tags"] is None:
            raise ValueError("rank_tags must be provided when device_bias is used")

        return cfg

    def iterate(self, objective, target_counts):
        if objective == "to-target":
            steps = [self._canonical_step(s) for s in self.exec_order]
            for step in steps:
                self._run_step(step, target_counts)

        elif objective == "to-remove-rank":
            # Keep this as a controller; internally it uses the same primitives.
            kill_rank = [r for r, c in enumerate(target_counts) if c == 0][0]
            if rank["world"] == root["world"]:
                print(f"{kill_rank = }", flush=True)

            # Simple loop; your stopping logic remains
            while True:
                cur0 = list(self._cur_counts_total)

                # 1) Flow-guided evacuation (vertices)
                self._run_step(
                    self._canonical_step(dict(
                        kind="vertices-flow",
                        name="to-remove-vertices",
                        threshold=0.0,
                        metric="delta",
                        overshoot=0.0,
                        max_sweeps=1,
                    )),
                    target_counts,
                )

                # 2) Local smoothing polish (faces)
                self._run_step(
                    self._canonical_step(dict(
                        kind="faces-smooth",
                        name="to-remove-smooth",
                        threshold=-1.0,
                        metric="delta",
                        max_iters=50,
                        patience=1,
                        min_change=0,
                    )),
                    target_counts,
                )

                cur1 = list(self._cur_counts_total)
                diff = [cur1[i] - cur0[i] for i in range(len(cur0))]

                if rank["world"] == root["world"]:
                    print(f"[itc4.iter] to-remove-rank CURRENT={cur1}\tDIFF={diff}", flush=True)

                if cur1[kill_rank] == 0:
                    if rank["world"] == root["world"]:
                        print(f"to-remove-rank done; kill_rank={kill_rank} cur={cur1}", flush=True)
                    break

                if all(d == 0 for d in diff):
                    if rank["world"] == root["world"]:
                        print(f"to-remove-rank stuck; kill_rank={kill_rank} cur={cur1}", flush=True)
                    break

        else:
            raise ValueError(f"Unknown objective {objective!r}")

    def _run_step(self, step: dict, target_counts: Optional[List[int]]) -> None:
        """
        Execute one canonical step.

        - use_flow=True  -> repeated diffuse() sweeps with residual M_rem.
        - use_flow=False -> repeated smooth() until stagnation criteria.
        """
        W      = comm["world"]
        rnk    = int(rank["world"])
        root_w = int(root["world"])
        P      = W.size

        kind   = step["kind"]
        label  = step.get("name") or kind
        mode   = step["mode"]
        use_flow = bool(step["use_flow"])
        thr    = float(step["threshold"])
        metric = step["metric"]

        etype_scale = step.get("etype_scale", None)
        device_bias = step.get("device_bias", None)
        rank_tags   = step.get("rank_tags", None)
        move_spts_nodes = bool(step.get("move_spts_nodes", False))

        if rnk == root_w:
            if use_flow:
                print(
                    f"[iterate] step={label} mode={mode} kind={kind} "
                    f"use_flow=True metric={metric} thr={thr:g} "
                    f"overshoot={step['overshoot']:.3f} max_sweeps={step['max_sweeps']}",
                    flush=True,
                )
            else:
                print(
                    f"[iterate] step={label} mode={mode} kind={kind} "
                    f"use_flow=False metric={metric} thr={thr:g} "
                    f"max_iters={step['max_iters']} patience={step['patience']} min_change={step['min_change']}",
                    flush=True,
                )

        # ----------------- FLOW-GUIDED sweeps -----------------
        if use_flow:
            if target_counts is None:
                raise ValueError(f"step {label}: use_flow=True but target_counts is None")

            overshoot = float(step["overshoot"])
            max_sweeps = int(step["max_sweeps"])

            M0 = self.element_flow_plan(target_counts)

            if overshoot != 0.0:
                factor = 1.0 + overshoot
                M_eff = np.rint(M0.astype(np.float64) * factor).astype(np.int64)
            else:
                M_eff = M0.copy()

            M_rem = M_eff.astype(np.int64, copy=True)
            flow_used_accum = np.zeros_like(M_rem, dtype=np.int64)

            # Safe default bound if max_sweeps <= 0
            max_sweeps_eff = P if max_sweeps <= 0 else max_sweeps

            for sweep in range(1, max_sweeps_eff + 1):
                moved_local, flow_used_local = self.diffuse(
                    mode=mode,
                    threshold=thr,
                    flow_matrix=M_rem,
                    metric=metric,
                    etype_scale=etype_scale,
                    device_bias=device_bias,
                    rank_tags=rank_tags,
                    move_spts_nodes=move_spts_nodes,
                )

                moved = int(W.allreduce(int(moved_local), op=mpi.SUM))

                # Aggregate actual flow usage across all ranks
                flow_used_global = np.zeros_like(flow_used_local, dtype=np.int64)
                W.Allreduce(flow_used_local, flow_used_global, op=mpi.SUM)

                M_rem           -= flow_used_global
                flow_used_accum += flow_used_global

                if rnk == root_w:
                    print(f"[iterate]   sweep={sweep} moved_glob={moved}", flush=True)

                if moved == 0:
                    break

            return

        # ----------------- SMOOTHING-to-stagnation sweeps -----------------
        max_iters  = int(step["max_iters"])
        patience   = int(step["patience"])
        min_change = int(step["min_change"])

        stable = 0
        last: Optional[int] = None

        for it in range(1, max_iters + 1):
            moved_local = self.smooth(
                mode=mode,
                threshold=thr,
                metric=metric,
                etype_scale=etype_scale,
                device_bias=device_bias,
                rank_tags=rank_tags,
                move_spts_nodes=move_spts_nodes,
            )

            moved = int(W.allreduce(int(moved_local), op=mpi.SUM))

            if last is not None and abs(moved - last) <= min_change:
                stable += 1
            else:
                stable = 0
            last = moved

            stop = (moved == 0) or (stable >= patience)
            stop_all = int(W.allreduce(1 if stop else 0, op=mpi.MAX))

            if rnk == root_w:
                print(
                    f"[iterate]   sweep={it} moved_glob={moved} stable={stable} stop_all={bool(stop_all)}",
                    flush=True,
                )

            if stop_all:
                break

    # ----------------- Rank addition -----------------------

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

    # ----------------- Part/Island removal ------------------------------------

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
        uu_all = np.concatenate([uu_all, vv_all])
        vv_all = np.concatenate([vv_all, uu_all[:vv_all.size]])

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

    def remove_small_islands_step(self, *, max_move: int | None = None,
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
        stats_local = (nloc, nis, nrm, int(cluster_gids.size))
        stats_all   = W.allgather(stats_local)
        if rnk == root_w:
            print(f"[rmislands.aggr] done stats={stats_all}", flush=True)


        # Return remaining islands, nrm
        nrms = W.allgather(nrm)
        return nrms

    def remove_outliers(self, *,
        attach_bias: float = 0.0,   # >0 biases toward more neighbour-tied elems (smaller delta)
        move_spts_nodes: bool = True, verbose: bool = True) -> int:
        
        top  = self.outlier_removal_fraction
        mode = self.outlier_removal_mode
        
        W      = comm["world"]
        rnk    = int(rank["world"])
        root_w = int(root["world"])
        st = self.i

        self._reset_j_with_i()

        top = float(top)
        if top <= 0.0:
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
            return 0

        self._cores = self._compute_cores_from_centroids()
        core = np.asarray(self._cores[rnk], dtype=np.float64)
        if not np.all(np.isfinite(core)):
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
            return 0

        # Flat centroids + validity
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

        iface_mask, best_nbr, best_del = self._iface_from_deltas(mode)

        mpi_flat = np.nonzero(iface_mask & (best_nbr >= 0) & okc)[0].astype(np.int64, copy=False)
        n_mpi = int(mpi_flat.size)
        if n_mpi == 0:
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
            return 0

        nsel = int(np.ceil(top * n_mpi))
        nsel = max(1, min(nsel, n_mpi))

        d = np.linalg.norm(cflat[mpi_flat, :] - core[None, :], axis=1)
        mu, sig = float(d.mean()), float(d.std())
        z = (d - mu) / sig if sig > 0 else np.zeros_like(d)

        # far-away + neighbour-tied (small/negative delta)
        score = z + float(attach_bias) * (-best_del[mpi_flat].astype(np.float64))

        gids  = st.eidxs_flat[mpi_flat].astype(np.int64, copy=False)
        order = np.lexsort((gids, -score))
        chosen_flat = mpi_flat[order[:nsel]]

        chosen_nbrs = best_nbr[chosen_flat].astype(np.int64, copy=False)
        eidxs_diff  = self._build_eidxs_diff_from_flat(chosen_flat, chosen_nbrs, st)

        moved_local = sum(len(g) for per in eidxs_diff.values() for g in per.values())
        self._apply_plan_and_commit(eidxs_diff, move_spts_nodes=move_spts_nodes)

        moved_glob = int(W.allreduce(int(moved_local), op=mpi.SUM))
        if verbose and rnk == root_w:
            print(f"[rmoutliers] mode={mode} top={top} n_mpi={n_mpi} nsel={nsel} moved_glob={moved_glob}", flush=True)

        return int(moved_local)

    def add_inliers(self, *, 
        margin: float = 0.0,         # require d_self - d_dest > margin
        move_spts_nodes: bool = True, verbose: bool = True, ) -> int:
        
        mode = self.inlier_addition_mode
        top  = self.inlier_addition_fraction
        
        
        W      = comm["world"]
        rnk    = int(rank["world"])
        root_w = int(root["world"])
        P      = int(W.size)
        st = self.i

        self._reset_j_with_i()

        top = float(top)
        if top <= 0.0:
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
            return 0

        self._cores = self._compute_cores_from_centroids()
        cores = np.asarray(self._cores, dtype=np.float64)
        if cores.ndim != 2 or cores.shape != (P, 3):
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
            return 0

        core_self = cores[rnk]
        if not np.all(np.isfinite(core_self)):
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
            return 0

        # Flat centroids + validity
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

        iface_mask, best_dest, best_dist = self._best_dest_by_core_from_deltas(
            mode, cores=cores, cflat=cflat, okc=okc
        )

        mpi_flat = np.nonzero(iface_mask & (best_dest >= 0) & okc)[0].astype(np.int64, copy=False)
        n_mpi = int(mpi_flat.size)
        if n_mpi == 0:
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
            return 0

        c      = cflat[mpi_flat, :]
        d_self = np.linalg.norm(c - core_self[None, :], axis=1)
        d_dest = best_dist[mpi_flat]
        score  = d_self - d_dest  # improvement if moved to dest core

        cand = score > float(margin)
        cand_flat = mpi_flat[cand]
        if cand_flat.size == 0:
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
            return 0

        nsel = int(np.ceil(top * int(cand_flat.size)))
        nsel = max(1, min(nsel, int(cand_flat.size)))

        gids  = st.eidxs_flat[cand_flat].astype(np.int64, copy=False)
        sc    = score[cand]
        order = np.lexsort((gids, -sc))
        chosen_flat = cand_flat[order[:nsel]]

        chosen_nbrs = best_dest[chosen_flat].astype(np.int64, copy=False)
        eidxs_diff  = self._build_eidxs_diff_from_flat(chosen_flat, chosen_nbrs, st)

        moved_local = sum(len(g) for per in eidxs_diff.values() for g in per.values())
        self._apply_plan_and_commit(eidxs_diff, move_spts_nodes=move_spts_nodes)

        moved_glob = int(W.allreduce(int(moved_local), op=mpi.SUM))
        if verbose and rnk == root_w:
            sc_sel = sc[order[:nsel]]
            print(
                f"[addinliers] mode={mode} top={top} margin={margin} n_mpi={n_mpi} "
                f"cand={int(cand_flat.size)} nsel={nsel} "
                f"score_sel=[{float(sc_sel.min()):.3e},{float(sc_sel.max()):.3e}] "
                f"moved_glob={moved_glob}",
                flush=True,
            )

        return int(moved_local)
    
    def _iface_from_deltas(self, mode: str):
        st   = self.i
        nloc = int(st.eidxs_flat.size)

        deltas = self._compute_deltas(mode)  # {nbr:{et: (N,2) [lid,delta]}}
        if not deltas:
            return (
                np.zeros(nloc, dtype=bool),
                np.full(nloc, -1, dtype=np.int32),
                np.full(nloc,  2**30, dtype=np.int32),
            )

        iface_mask = np.zeros(nloc, dtype=bool)
        best_nbr   = np.full(nloc, -1, dtype=np.int32)
        best_del   = np.full(nloc,  2**30, dtype=np.int32)

        for nbr, per_et in deltas.items():
            nbr = int(nbr)
            for et, mat in per_et.items():
                if mat is None or mat.size == 0:
                    continue

                lids  = mat[:, 0].astype(np.int64, copy=False)
                delt  = mat[:, 1].astype(np.int32, copy=False)
                flats = st.lids_to_flat(et, lids)

                iface_mask[flats] = True

                curd = best_del[flats]
                curn = best_nbr[flats]
                # choose neighbour with *smallest delta* (most tied to nbr); tie-break low nbr
                better = (delt < curd) | ((delt == curd) & ((curn < 0) | (nbr < curn)))
                if np.any(better):
                    fsel = flats[better]
                    best_del[fsel] = delt[better]
                    best_nbr[fsel] = np.int32(nbr)

        return iface_mask, best_nbr, best_del

    def _best_dest_by_core_from_deltas(
        self,
        mode: str,
        *,
        cores: np.ndarray,         # (P,3)
        cflat: np.ndarray,         # (nloc,3)
        okc: np.ndarray,           # (nloc,) bool
    ):
        st   = self.i
        nloc = int(st.eidxs_flat.size)
        P    = int(cores.shape[0])

        deltas = self._compute_deltas(mode)  # {nbr:{et:(N,2)[lid,delta]}}
        if not deltas:
            return (
                np.zeros(nloc, dtype=bool),
                np.full(nloc, -1, dtype=np.int32),
                np.full(nloc, np.inf, dtype=np.float64),
            )

        iface_mask = np.zeros(nloc, dtype=bool)
        best_dest  = np.full(nloc, -1, dtype=np.int32)
        best_dist  = np.full(nloc, np.inf, dtype=np.float64)

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
                flats = st.lids_to_flat(et, lids)

                iface_mask[flats] = True

                f2 = flats[okc[flats]]
                if f2.size == 0:
                    continue

                d = np.linalg.norm(cflat[f2, :] - core_n[None, :], axis=1)

                curd = best_dist[f2]
                curk = best_dest[f2]
                better = (d < curd) | ((d == curd) & ((curk < 0) | (nbr < curk)))
                if np.any(better):
                    sel = f2[better]
                    best_dist[sel] = d[better]
                    best_dest[sel] = np.int32(nbr)

        return iface_mask, best_dest, best_dist
