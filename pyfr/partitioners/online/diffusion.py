from collections import deque
from typing import List, Optional

import numpy as np

from pyfr.mpiutil import comm, rank, root, mpi
from pyfr.partitioners.online.base import OnlinePartitioner

class OnlineDiffusionPartitioner(OnlinePartitioner):
    """
    A collection of all diffusion related strategies.
    """

    def __init__(self, mesh, cfg):
        OnlinePartitioner.__init__(self, mesh, cfg)

        # Island stuff
        self.island_remove_fraction   = cfg.getfloat('partition', 'island-remove-fraction'  , 0.5)
        self.outlier_removal_mode     = cfg.get(     'partition', 'outlier-removal-mode'    , 'faces')
        self.outlier_removal_fraction = cfg.getfloat('partition', 'outlier-removal-fraction', 0.01)
        self.inlier_addition_mode     = cfg.get(     'partition', 'inlier-addition-mode'    , 'vertices')
        self.inlier_addition_fraction = cfg.getfloat('partition', 'inlier-addition-fraction', 0.10)

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
                c_int[sel].astype(np.int32) - c_n[sel].astype(np.int32)
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
        # deltas_by_rank = self._filter_deltas_by_src_mask(deltas_by_rank, src_mask_flat)

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
            move_spts_nodes=True, # (mode == "vertices"),
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
                move_spts_nodes=True,
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
        exec_order = ([(6, "vertex")]+
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



    def label_islands_faces(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Label face-connected components ("islands") within THIS rank.

        Returns
        -------
        island_id_flat : (nelems_local,) int32
            Island label for each local element, in the same order as
            self.i.eidxs_flat / local flat indexing.
            Labels are renumbered so that:
                - island 0 is the largest island,
                - island 1 is the 2nd largest, etc.
        island_sizes : (nislands,) int64
            Sizes of islands in descending order; island_sizes[k] is the
            number of elements with island_id_flat == k.
        """
        st = self.i
        gids = np.asarray(st.eidxs_flat, dtype=np.int64)
        nloc = int(gids.size)

        if nloc == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64)

        # --- Map neighbour global IDs -> local flat indices (or "not local") ---
        # Sort local gids once; use searchsorted for vectorized membership+mapping
        order = np.argsort(gids, kind="mergesort")
        gids_sorted = gids[order]

        # --- Build directed local adjacency edges (u_flat -> v_flat) ---
        uu_chunks: list[np.ndarray] = []
        vv_chunks: list[np.ndarray] = []

        # (Optional) sanity: ensure con_idx stores global element numbers
        nglb = int(comm["world"].allreduce(nloc, op=mpi.SUM))

        for et in st.etypes:
            sl = st.etype_slices.get(et, None)
            if sl is None:
                continue
            ne = int(sl.stop - sl.start)
            if ne == 0:
                continue

            nbr = np.asarray(st.con_idx[et], dtype=np.int64)

            # Treat trailing dims as faces; ensure shape (ne, nfaces*)
            if nbr.ndim == 1:
                nbr = nbr.reshape(ne, -1)
            else:
                nbr = nbr.reshape(ne, -1)

            if nbr.shape[0] != ne:
                raise ValueError(
                    f"[islands] con_idx[{et}] wrong element axis: got {nbr.shape[0]} expected {ne}"
                )

            m = (nbr >= 0)
            if not np.any(m):
                continue

            vv = nbr[m].astype(np.int64, copy=False)

            # Sanity: neighbour gids must be in [0, nglb)
            if vv.size and int(vv.max()) >= nglb:
                raise ValueError(
                    f"[islands] con_idx[{et}] appears not to be global eids "
                    f"(max nbr={int(vv.max())} >= nglb={nglb})."
                )

            u_flat = np.arange(sl.start, sl.stop, dtype=np.int64)
            uu = np.broadcast_to(u_flat[:, None], nbr.shape)[m].astype(np.int64, copy=False)

            # Map vv (global) -> local flat index if vv is owned locally
            idx = np.searchsorted(gids_sorted, vv)

            # SAFE membership test: avoid indexing gids_sorted with any idx == nloc
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
            # No local adjacencies: every element is its own island
            island_id = np.arange(nloc, dtype=np.int32)
            island_sizes = np.ones(nloc, dtype=np.int64)
            # All equal; "largest first" is arbitrary but already deterministic
            return island_id, island_sizes

        uu_all = np.concatenate(uu_chunks)
        vv_all = np.concatenate(vv_chunks)

        # --- CSR adjacency over local flat indices 0..nloc-1 ---
        perm = np.argsort(uu_all, kind="mergesort")
        uu_all = uu_all[perm]
        vv_all = vv_all[perm]

        counts = np.bincount(uu_all, minlength=nloc)
        vtab = np.empty(nloc + 1, dtype=np.int64)
        vtab[0] = 0
        np.cumsum(counts, out=vtab[1:])
        etab = vv_all.astype(np.int64, copy=False)

        # --- BFS/DFS connected components ---
        island_id = np.full(nloc, -1, dtype=np.int32)
        sizes: list[int] = []
        cid = 0

        q = deque()
        for s in range(nloc):
            if island_id[s] != -1:
                continue

            # Start new component
            island_id[s] = cid
            q.append(s)
            sz = 0

            while q:
                u = q.pop()
                sz += 1
                beg = int(vtab[u])
                end = int(vtab[u + 1])
                nbrs = etab[beg:end]

                # Iterate neighbours (Python loop OK; speed not your priority)
                for v in nbrs.tolist():
                    if island_id[v] == -1:
                        island_id[v] = cid
                        q.append(v)

            sizes.append(sz)
            cid += 1

        sizes_arr = np.asarray(sizes, dtype=np.int64)

        # --- Renumber so largest island is 0 (descending by size; tie-break by old id) ---
        old_ids = np.arange(sizes_arr.size, dtype=np.int64)
        order2 = np.lexsort((old_ids, -sizes_arr))  # stable: size desc, then id asc

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

    def _remove_cluster_by_mpi_vertex(self, cluster_flat: np.ndarray, *, max_sweeps: int = 200) -> None:
        W      = comm['world']
        rnk    = int(rank['world'])
        root_w = int(root['world'])

        cluster_flat = np.asarray(cluster_flat, dtype=np.int64)
        if cluster_flat.size == 0:
            return

        # Track by GIDs so the set is stable across retag/reindex
        cluster_gids = np.asarray(self.i.eidxs_flat[cluster_flat], dtype=np.int64)

        for k in range(int(max_sweeps)):
            # Rebuild local membership in current state
            in_cluster = np.isin(self.i.eidxs_flat, cluster_gids, assume_unique=False)
            nloc = int(in_cluster.sum())
            nglob = int(W.allreduce(nloc, op=mpi.SUM))

            if rnk == root_w:
                print(f"[rmcluster] sweep={k+1} remaining_glob={nglob}", flush=True)

            if nglob == 0:
                break

            src_mask_flat = in_cluster  # already bool, correct length

            moved_local, _ = self.diffuse(
                mode="vertices",
                interface_policy="per-element",
                threshold=0,
                flow_matrix=None,
                restrict_etypes=None,
                skip_last=False,
                restrict_src_dest=False,
                target_counts=None,
                move_spts_nodes=True,

                # internal-only:
                src_mask_flat=src_mask_flat,
                ignore_targets=True,
                verbose=True,   # so [diffuse.debug] prints candidate stats
            )

            moved_glob = int(W.allreduce(int(moved_local), op=mpi.SUM))
            if rnk == root_w:
                print(f"[rmcluster] sweep={k+1} moved_glob={moved_glob}", flush=True)

            # If nothing moved but cluster remains, you're genuinely stuck topologically or selection is broken.
            if moved_glob == 0:
                if rnk == root_w:
                    print("[rmcluster] STALL: moved_glob=0 while remaining_glob>0", flush=True)
                break

    def _filter_deltas_by_src_mask(self, deltas, src_mask_flat):
        if src_mask_flat is None:
            return deltas

        out = {}
        for nbr, per in deltas.items():
            per2 = {}
            for et, mat in per.items():
                if mat.size == 0:
                    per2[et] = mat
                    continue

                sl = self.i.etype_slices[et]
                mask_et = src_mask_flat[sl.start:sl.stop]
                lids = mat[:, 0]
                sel = mask_et[lids]
                per2[et] = mat[sel]

            out[int(nbr)] = per2
        return out

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

        self._compute_cores_from_centroids()
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

        self._compute_cores_from_centroids()
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
