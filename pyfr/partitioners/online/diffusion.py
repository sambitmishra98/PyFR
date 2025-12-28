from typing import List, Optional

import numpy as np

from pyfr.mpiutil import comm, rank, root, mpi
from pyfr.partitioners.online.base import OnlinePartitioner, CarverMixin, OfflineRepartitioner

class DiffusionRepartitioner(CarverMixin, OfflineRepartitioner):
    def __init__(self, mesh, cfg):
        OfflineRepartitioner.__init__(self, mesh, cfg)
        CarverMixin.__init__(self, cfg)

    @property
    def twoway_mask(self) -> np.ndarray:
        """
        A boolean adjacency matrix for determining element flow
        FUTURE DIRECTIONS: 
            - Use self.i details directly, not self._mpi_faces_by_neighbor()
            - Simplify, make concise. 
        """
        R    = comm['world'].size
        r    = comm['world'].rank

        M_loc = np.zeros((R, R), dtype=np.uint8)

        for nbr in self._mpi_faces_by_neighbor():
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

        return M_full

    # ----------------- Delta computations -----------------------
    def _compute_affinity(self, mode):
        if   mode in ['faces',]:    return self._calc_mpi_face_affinity()
        elif mode in ['vertices',]: return self._calc_mpi_vertex_affinity()
        else: raise ValueError(f"Invalid {mode = }")

    def _mpi_faces_by_neighbor(self) -> dict[int, list[tuple[str, int, int]]]:
        """
        {nbr: [(etype, lid, fidx), ...]} for MPI faces only.
        Deterministic: neighbour keys sorted when iterated later.
        """
        per_nbr: dict[int, list[tuple[str, int, int]]] = {}

        for et in self.etypes:
            owners = self.i.con_mpi[et]
            eids   = self.i.con_idx[et]
            if owners.size == 0 or eids.size == 0:
                continue

            mpi_mask = (eids >= 0) & (owners >= 0) & (owners != rank['world'])
            if not np.any(mpi_mask):
                continue

            lids, fidxs = np.nonzero(mpi_mask)
            nbrs = owners[lids, fidxs].astype(np.int64, copy=False)

            # group; keep payload
            for nbr in np.unique(nbrs):
                nbr = int(nbr)
                if nbr == rank['world']:
                    continue
                sel = (nbrs == nbr)
                lst = per_nbr.setdefault(nbr, [])
                lst.extend((et, int(lid_i), int(f_i)) for lid_i, f_i in zip(lids[sel], fidxs[sel]))

        return per_nbr

    def _calc_mpi_face_affinity(self) -> dict[int, dict[str, np.ndarray]]:
        """
            per-neighbour per-etype MPI face affinity.
        """

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

        raise NotImplementedError(f"{et} etype not implemented")

    def collect_mpi_vertex_nodes(self) -> dict[int, "np.ndarray"]:
        """
        {nbr: sorted unique global vertex-node IDs that lie on MPI faces to nbr}
        """

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

    def _metric_from_ab(self, a, b, metric):
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
            raise ValueError(f"Unknown {metric = }")

    def _calc_mpi_vertex_affinity(self) -> dict[int, dict[str, np.ndarray]]:
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
        {nbr: {et: unique_sorted_lids_on_MPI_faces_to_nbr}}
        """

        per_nbr_faces = self._mpi_faces_by_neighbor()
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

    # ----------------- Drivers ----------------------------

    def __smooth(self, metric: str = "delta", etype_scale: dict[str, float] | None = None, device_bias: dict[str, dict[str, float]] | None = None, rank_tags: list[str] | None = None, move_spts_nodes=False) -> int:
        metric = str(metric).lower()
        if metric not in ("delta", "ratio"):
            raise ValueError(f"smooth: metric must be 'delta' or 'ratio', got {metric!r}")

        if device_bias is not None and rank_tags is None:
            raise ValueError("smooth: rank_tags must be provided when device_bias is used")

        self._reset_j_with_i()
        st = self.i

        aff_by_rank = self._compute_affinity('faces')  # {nbr:{et:[lid,a,b]}}
        etypes_all  = list(self._etype_order())

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

                #mraw = metric_from_ab(a, b)
                mraw = self._metric_from_ab(a, b, metric)
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

    def __diffuse(self, *, mode='faces', metric='delta', threshold=0.0, flow_matrix=None, move_spts_nodes=False, debug=False):
        """
        Affinity-driven diffusion (no precomputed delta tables).

        Parameters
        ----------
        mode : str
            'faces'/'vertices' (also accepts 'e'->faces, 'v'->vertices).
        metric : str
            'delta' (a-b) or 'ratio' (a/(a+b), smaller => more neighbour-tied).
        threshold : float|int
            Move candidates with score <= threshold.
        flow_matrix : (P,P) int array
            Per-rank caps M[src,nbr].
        move_spts_nodes : bool
            If True, relocate spts_nodes alongside eidxs (forced True in vertex mode).
        debug : bool
            If True, print one deterministic summary line per rank.

        Returns
        -------
        moved_local : int
        flow_used   : (P,P) int array
        eidxs_diff  : dict[int, dict[str, np.ndarray]]
        """
        # ----------------- Parse mode deterministically -----------------
        m = str(mode).strip().lower()
        mode_map = {
            'faces': 'faces', 'face': 'faces', 'f': 'faces', 'e': 'faces', 'edge': 'faces', 'edges': 'faces',
            'vertices': 'vertices', 'vertex': 'vertices', 'v': 'vertices', 'vert': 'vertices', 'vtx': 'vertices'
        }
        if m not in mode_map:
            raise ValueError(f"diffuse2: unknown mode {mode!r}")
        mode = mode_map[m]

        metric = str(metric).strip().lower()
        if metric not in ('delta', 'ratio'):
            raise ValueError(f"diffuse2: metric must be 'delta' or 'ratio', got {metric!r}")

        # Threshold type: keep your existing behaviour
        thr = int(threshold) if float(threshold).is_integer() else float(threshold)

        # Vertex mode must keep spts_nodes consistent for subsequent vertex scoring
        if mode == 'vertices':
            move_spts_nodes = True

        M = np.asarray(flow_matrix, dtype=np.int64)
        if M.ndim != 2 or M.shape[0] != M.shape[1] or int(M.shape[0]) != comm['world'].size:
            raise ValueError(f"diffuse2: bad flow_matrix shape {M.shape}, comm size {comm['world'].size}")

        # Always start from i -> j
        self._reset_j_with_i()
        st = self.i
        etypes_all = list(self._etype_order())

        # Canonical candidate source: {nbr: {et: (N,3) [lid,a,b]}}
        aff_by_rank = self._compute_affinity(mode)

        picked_by_et = {et: set() for et in etypes_all}   # per-etype gid de-dupe (legacy semantics)
        eidxs_diff: dict[int, dict[str, np.ndarray]] = {}
        flow_used = np.zeros_like(M, dtype=np.int64)

        # Optional debug accumulation
        moved_to = {}

        for nbr in range(comm['world'].size):
            cap = int(M[rank['world'], nbr])
            if cap <= 0 or nbr == rank['world']:
                continue

            per_et = aff_by_rank.get(int(nbr), {})
            cand: list[tuple[float, int, int, str]] = []
            # tuple: (score, gid_u, gid, et)

            for et in etypes_all:
                arr = per_et.get(et, None)
                if arr is None or arr.size == 0:
                    continue

                lids = arr[:, 0].astype(np.int64, copy=False)
                a    = arr[:, 1].astype(np.int64, copy=False)
                b    = arr[:, 2].astype(np.int64, copy=False)

                # Derive score from (a,b)
                if metric == 'delta':
                    score = (a - b).astype(np.int64, copy=False)   # exact, integer-valued
                else:
                    # ratio: a/(a+b), smaller => more neighbour-tied
                    den = (a + b).astype(np.float64, copy=False)
                    score = np.full(a.shape, np.inf, dtype=np.float64)
                    np.divide(a.astype(np.float64, copy=False), den, out=score, where=(den > 0.0))

                # Threshold
                keep = (score <= thr)
                if not np.any(keep):
                    continue

                lids_t  = lids[keep]
                score_t = score[keep]

                # Local per-etype gid; NOT globally unique across etypes
                gids_t = st.eidxs[et][lids_t].astype(np.int64, copy=False)
                # Globally-unique id for tie-breaking only
                gids_u = (int(st.edisps[et]) + gids_t).astype(np.int64, copy=False)

                pset = picked_by_et[et]
                for gid_u, gid, sc in zip(gids_u.tolist(), gids_t.tolist(), score_t.tolist()):
                    if gid not in pset:
                        cand.append((float(sc), int(gid_u), int(gid), et))

            if not cand:
                continue

            # Deterministic: primary score, then globally-unique element id
            cand.sort(key=lambda t: (t[0], t[1]))

            chosen = cand[: min(cap, len(cand))]

            per_out: dict[str, list[int]] = {}
            for sc, gid_u, gid, et in chosen:
                per_out.setdefault(et, []).append(gid)
                picked_by_et[et].add(gid)

            out_per = {
                et: np.asarray(sorted(v), dtype=np.int64)
                for et, v in per_out.items()
                if v
            }

            if out_per:
                eidxs_diff[int(nbr)] = out_per
                used = int(sum(len(v) for v in out_per.values()))
                flow_used[rank['world'], int(nbr)] = used
                if debug:
                    moved_to[int(nbr)] = used

        moved_local = int(sum(len(g) for per in eidxs_diff.values() for g in per.values()))
        self._apply_plan_and_commit(eidxs_diff, move_spts_nodes=move_spts_nodes)

        if debug:
            # One line per rank, deterministic key ordering
            items = ",".join(f"{k}:{moved_to[k]}" for k in sorted(moved_to))
            print(f"[diffuse2] rank={int(rank['world'])} mode={mode} metric={metric} thr={thr} moved_local={moved_local} moved_to={{{items}}}")

        return moved_local, flow_used, eidxs_diff




    # ----------------- Shared helpers (drop-in) -----------------

    def _parse_mode(self, mode: str) -> str:
        """Map user mode tokens to canonical 'faces'/'vertices'."""
        m = str(mode).strip().lower()
        mode_map = {
            'faces': 'faces', 'face': 'faces', 'f': 'faces', 'e': 'faces',
            'edge': 'faces', 'edges': 'faces',
            'vertices': 'vertices', 'vertex': 'vertices', 'v': 'vertices',
            'vert': 'vertices', 'vtx': 'vertices'
        }
        if m not in mode_map:
            raise ValueError(f"unknown mode {mode!r}; expected faces/vertices (or e/v)")
        return mode_map[m]


    def _parse_metric(self, metric: str) -> str:
        met = str(metric).strip().lower()
        if met not in ('delta', 'ratio'):
            raise ValueError(f"metric must be 'delta' or 'ratio', got {metric!r}")
        return met


    def _score_from_ab(self, a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
        """
        Return score array from (a,b) where a,b are int arrays:
        - delta: a - b        (int-valued)
        - ratio: a/(a+b)      (float; smaller => more neighbour-tied)
        Always returns float64 for uniform downstream sorting.
        """
        metric = self._parse_metric(metric)

        if metric == 'delta':
            return (a.astype(np.float64, copy=False) - b.astype(np.float64, copy=False))

        den = (a + b).astype(np.float64, copy=False)
        out = np.full(a.shape, np.inf, dtype=np.float64)
        np.divide(a.astype(np.float64, copy=False), den, out=out, where=(den > 0.0))
        return out


    def _collect_affinity_candidates(
        self,
        *,
        mode: str,
        metric: str,
        threshold: float,
        etype_scale: dict[str, float] | None = None,
        device_bias: dict[str, dict[str, float]] | None = None,
        rank_tags: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a single candidate table from affinity:
        returns (flat, gid_u, gid, eti, nbr, score) as arrays, all same length.

        - flat:  local flattened element index (int64)
        - gid_u: globally-unique element id across etypes (int64)
        - gid:   per-etype gid (int64)  [needed to build eidxs_diff]
        - eti:   etype index into self._etype_order() (int16)
        - nbr:   neighbour rank (int32)
        - score: float64 score (already thresholded)
        """
        mode = self._parse_mode(mode)
        metric = self._parse_metric(metric)
        thr = float(threshold)

        if device_bias is not None and rank_tags is None:
            raise ValueError("rank_tags must be provided when device_bias is used")

        st = self.i
        etypes_all = list(self._etype_order())
        et_to_i = {et: i for i, et in enumerate(etypes_all)}

        aff_by_rank = self._compute_affinity(mode)  # {nbr: {et: [lid,a,b]}}

        flats: list[np.ndarray] = []
        gidus: list[np.ndarray] = []
        gids:  list[np.ndarray] = []
        etis:  list[np.ndarray] = []
        nbrs:  list[np.ndarray] = []
        scores:list[np.ndarray] = []

        def _bias_term(*, et: str, nbr: int) -> float:
            s = 1.0
            if etype_scale is not None:
                s *= float(etype_scale.get(et, 1.0))
            if device_bias is not None:
                tag = rank_tags[int(nbr)]
                s += float(device_bias.get(tag, {}).get(et, 0.0))
            return float(s)

        for nbr in sorted(int(n) for n in aff_by_rank.keys()):
            per_et = aff_by_rank[nbr]

            for et in etypes_all:
                arr = per_et.get(et, None)
                if arr is None or arr.size == 0:
                    continue

                lids = arr[:, 0].astype(np.int64, copy=False)
                a    = arr[:, 1].astype(np.int64, copy=False)
                b    = arr[:, 2].astype(np.int64, copy=False)

                score = self._score_from_ab(a, b, metric)

                # apply optional weighting/bias *after* defining the metric
                w = _bias_term(et=et, nbr=nbr)
                if w != 1.0:
                    score = score * w

                keep = (score <= thr)
                if not np.any(keep):
                    continue

                lids_k  = lids[keep]
                score_k = score[keep]

                gid_k  = st.eidxs[et][lids_k].astype(np.int64, copy=False)
                gid_u  = (int(st.edisps[et]) + gid_k).astype(np.int64, copy=False)
                flat_k = st.lids_to_flat(et, lids_k).astype(np.int64, copy=False)

                flats.append(flat_k)
                gidus.append(gid_u)
                gids.append(gid_k)
                etis.append(np.full(flat_k.shape, et_to_i[et], dtype=np.int16))
                nbrs.append(np.full(flat_k.shape, int(nbr), dtype=np.int32))
                scores.append(score_k.astype(np.float64, copy=False))

        if not flats:
            z = np.empty(0, dtype=np.int64)
            return z, z, z, np.empty(0, dtype=np.int16), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)

        flat_all  = np.concatenate(flats).astype(np.int64, copy=False)
        gid_u_all = np.concatenate(gidus).astype(np.int64, copy=False)
        gid_all   = np.concatenate(gids).astype(np.int64, copy=False)
        eti_all   = np.concatenate(etis).astype(np.int16, copy=False)
        nbr_all   = np.concatenate(nbrs).astype(np.int32, copy=False)
        score_all = np.concatenate(scores).astype(np.float64, copy=False)

        return flat_all, gid_u_all, gid_all, eti_all, nbr_all, score_all


    def _pack_eidxs_diff(
        self,
        chosen_nbr: np.ndarray,
        chosen_gid: np.ndarray,
        chosen_eti: np.ndarray,
    ) -> dict[int, dict[str, np.ndarray]]:
        """
        Convert chosen arrays into eidxs_diff = {nbr: {et: sorted_gids}}.
        """
        etypes_all = list(self._etype_order())

        eidxs_diff: dict[int, dict[str, np.ndarray]] = {}
        for nbr in np.unique(chosen_nbr.astype(np.int64, copy=False)):
            nbr = int(nbr)
            sel_n = (chosen_nbr == nbr)
            if not np.any(sel_n):
                continue

            per: dict[str, np.ndarray] = {}
            eti_n = chosen_eti[sel_n]
            gid_n = chosen_gid[sel_n]

            for eti in np.unique(eti_n.astype(np.int64, copy=False)):
                eti = int(eti)
                sel_e = (eti_n == eti)
                if not np.any(sel_e):
                    continue
                et = etypes_all[eti]
                per[et] = np.asarray(sorted(gid_n[sel_e].tolist()), dtype=np.int64)

            if per:
                eidxs_diff[nbr] = per

        return eidxs_diff


    # ----------------- Drivers (drop-in replacements) -----------------

    def smooth(
        self,
        metric: str = "delta",
        threshold: float | None = None,
        etype_scale: dict[str, float] | None = None,
        device_bias: dict[str, dict[str, float]] | None = None,
        rank_tags: list[str] | None = None,
        move_spts_nodes: bool = False,
        debug: bool = False,
    ) -> int:
        """
        Pick best neighbour per element (one move per element max), based on affinity.

        Defaults preserve your old semantics:
        - metric='delta' with threshold=None => thr = -1
            (i.e., move only when neighbour-tied strictly wins)
        - metric='ratio' with threshold=None => thr = 0.49
            (sensible default; override if you want stricter)
        """
        metric = self._parse_metric(metric)

        if threshold is None:
            thr = -1.0 if metric == 'delta' else 0.49
        else:
            thr = float(threshold)

        self._reset_j_with_i()
        st = self.i

        flat, gid_u, gid, eti, nbr, score = self._collect_affinity_candidates(
            mode='faces',
            metric=metric,
            threshold=thr,
            etype_scale=etype_scale,
            device_bias=device_bias,
            rank_tags=rank_tags
        )

        if flat.size == 0:
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
            if debug:
                print(f"[smooth] rank={int(rank['world'])} metric={metric} thr={thr} moved_local=0 cand=0")
            return 0

        # Deterministic “best neighbour per element”:
        # primary: flat id, then score (ascending), then gid_u, then nbr
        order = np.lexsort((nbr.astype(np.int64), gid_u, score, flat))
        flat_s  = flat[order]
        nbr_s   = nbr[order].astype(np.int64, copy=False)
        gid_s   = gid[order]
        eti_s   = eti[order]
        score_s = score[order]

        # pick first per flat
        _, first = np.unique(flat_s, return_index=True)

        chosen_flat = flat_s[first]
        chosen_nbr  = nbr_s[first]
        # chosen_gid / eti only needed for debug; plan built via your existing helper:
        chosen_gid  = gid_s[first]
        chosen_eti  = eti_s[first]
        chosen_score= score_s[first]

        eidxs_diff = self._build_eidxs_diff_from_flat(chosen_flat, chosen_nbr, st)
        self._apply_plan_and_commit(eidxs_diff, move_spts_nodes=move_spts_nodes)

        moved_local = int(chosen_flat.size)

        if debug:
            # deterministic summary
            mn = float(np.min(chosen_score)) if chosen_score.size else float('nan')
            mx = float(np.max(chosen_score)) if chosen_score.size else float('nan')
            print(f"[smooth] rank={int(rank['world'])} \t metric={metric} \t thr={thr} moved_local={moved_local} cand={int(flat.size)} score_min={mn:.6g} score_max={mx:.6g}")

        return moved_local


    def diffuse(
        self,
        *,
        mode: str = 'faces',
        metric: str = 'delta',
        threshold: float = 0.0,
        flow_matrix,
        move_spts_nodes: bool = False,
        dedupe: str = 'global',
        debug: bool = False,
    ):
        """
        Affinity-driven diffusion with per-neighbour caps M[src,nbr].

        dedupe='global' uses gid_u = edisps[et] + gid so an element cannot be moved
        twice in one sweep even across etypes.
        """
        mode = self._parse_mode(mode)
        metric = self._parse_metric(metric)

        # keep delta comparisons exact when possible
        thr = float(threshold)

        if mode == 'vertices':
            move_spts_nodes = True

        # Validate flow matrix
        M = np.asarray(flow_matrix, dtype=np.int64)
        if M.ndim != 2 or M.shape[0] != M.shape[1] or int(M.shape[0]) != comm['world'].size:
            raise ValueError(f"diffuse: bad flow_matrix shape {M.shape}, comm size {comm['world'].size}")

        r = int(rank['world'])
        cap_row = M[r].astype(np.int64, copy=False)
        # include only positive caps; self-cap doesn't matter but harmless
        cap_sum = int(np.sum(np.maximum(0, cap_row)))

        dedupe = str(dedupe).strip().lower()
        if dedupe not in ('global', 'etype'):
            raise ValueError("diffuse: dedupe must be 'global' or 'etype'")

        def _dbg(moved_local: int, cand: int, cand_cap_pos: int, dedupe_hits: int, moved_to: dict[int, int]):
            if not debug:
                return
            items = ",".join(f"{k}:{moved_to[k]}" for k in sorted(moved_to))
            print(
                f"[diffuse] rank={r} mode={mode} metric={metric} thr={thr} "
                f"dedupe={dedupe} \t cap_sum={cap_sum} \t cand={cand} \t cand_cap_pos={cand_cap_pos} "
                f"dedupe_hits={dedupe_hits} \t moved_local={moved_local} \t moved_to={{{items}}}"
            )

        # Always start from i -> j
        self._reset_j_with_i()
        st = self.i

        flat, gid_u, gid, eti, nbr, score = self._collect_affinity_candidates(
            mode=mode,
            metric=metric,
            threshold=thr
        )

        flow_used = np.zeros_like(M, dtype=np.int64)

        cand = int(getattr(flat, "size", 0))
        if cand == 0:
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
            _dbg(moved_local=0, cand=0, cand_cap_pos=0, dedupe_hits=0, moved_to={})
            return 0, flow_used, {}

        # Sort candidates grouped by nbr, then (score, gid_u) deterministically:
        # np.lexsort: last key is primary -> primary=nbr, then score, then gid_u
        order = np.lexsort((gid_u, score, nbr.astype(np.int64, copy=False)))
        nbr_s   = nbr[order].astype(np.int64, copy=False)
        gid_u_s = gid_u[order].astype(np.int64, copy=False)
        gid_s   = gid[order].astype(np.int64, copy=False)
        eti_s   = eti[order].astype(np.int64, copy=False)

        # Dedupe tracking
        dedupe_hits = 0
        if dedupe == 'global':
            moved_global: set[int] = set()
            moved_by_et: dict[int, set[int]] = {}
        else:
            moved_global = set()
            moved_by_et = {}  # created lazily per eti

        chosen_nbr: list[int] = []
        chosen_gid: list[int] = []
        chosen_eti: list[int] = []

        moved_to: dict[int, int] = {}
        cand_cap_pos = 0

        # Segment boundaries for nbr_s
        bound = np.flatnonzero(np.diff(nbr_s)) + 1
        starts = np.r_[0, bound]
        ends   = np.r_[bound, nbr_s.size]

        for s0, s1 in zip(starts, ends):
            nb = int(nbr_s[s0])
            if nb == r:
                continue

            cap = int(M[r, nb])
            if cap <= 0:
                continue

            seg_len = int(s1 - s0)
            cand_cap_pos += seg_len

            picked = 0
            for k in range(int(s0), int(s1)):
                if picked >= cap:
                    break

                eti_k = int(eti_s[k])
                gid_k = int(gid_s[k])
                gid_u_k = int(gid_u_s[k])

                if dedupe == 'global':
                    if gid_u_k in moved_global:
                        dedupe_hits += 1
                        continue
                    moved_global.add(gid_u_k)
                else:
                    ss = moved_by_et.setdefault(eti_k, set())
                    if gid_k in ss:
                        dedupe_hits += 1
                        continue
                    ss.add(gid_k)

                chosen_nbr.append(nb)
                chosen_gid.append(gid_k)
                chosen_eti.append(eti_k)
                picked += 1

            if picked:
                moved_to[nb] = int(picked)
                flow_used[r, nb] = int(picked)

        if not chosen_nbr:
            self._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
            _dbg(moved_local=0, cand=cand, cand_cap_pos=cand_cap_pos, dedupe_hits=dedupe_hits, moved_to={})
            return 0, flow_used, {}

        chosen_nbr_a = np.asarray(chosen_nbr, dtype=np.int64)
        chosen_gid_a = np.asarray(chosen_gid, dtype=np.int64)
        chosen_eti_a = np.asarray(chosen_eti, dtype=np.int16)

        eidxs_diff = self._pack_eidxs_diff(chosen_nbr_a, chosen_gid_a, chosen_eti_a)

        moved_local = int(chosen_gid_a.size)
        self._apply_plan_and_commit(eidxs_diff, move_spts_nodes=move_spts_nodes)

        _dbg(moved_local=moved_local, cand=cand, cand_cap_pos=cand_cap_pos, dedupe_hits=dedupe_hits, moved_to=moved_to)
        return moved_local, flow_used, eidxs_diff



    # ----------------- Wrappers ----------------------------

    def smooth_until_stagnates(self, max_iters = 20, move_spts_nodes=False):
        stable = 0
        last: Optional[int] = None

        for _ in range(int(max_iters)):
            moved_local = self.smooth(metric="delta", move_spts_nodes=move_spts_nodes)
            moved_glob = int(comm['world'].allreduce(int(moved_local), op=mpi.SUM))

            if last is not None and abs(moved_glob - last) <= 0: stable += 1
            else: stable = 0

            last = moved_glob

            stop = (moved_glob == 0) or (stable >= 1)
            stop_all = int(comm['world'].allreduce(1 if stop else 0, op=mpi.MAX))

            if stop_all:
                break

    def diffuse_step(self, flow_matrix, threshold, scale, mode='faces'):

        M = np.asarray(flow_matrix, dtype=np.int64)

        # legacy-style relaxation: ceil
        scale = float(scale)
        M_eff = np.ceil(M.astype(np.float64) * scale).astype(np.int64)
        M_eff[M_eff < 0] = 0
        np.fill_diagonal(M_eff, 0)

        moved_local, flow_used, eidxs_diff = self.diffuse(mode=mode,
                                                          threshold=threshold,
                                                          flow_matrix=M_eff,
                                        move_spts_nodes=True,)#(mode == 'vertices'),)

        moved_glob = int(comm['world'].allreduce(int(moved_local), op=mpi.SUM))

        return moved_glob, eidxs_diff

    # ----------------- Diffuse + Smooth wrappers ----------------------------

    def iterate(self, target_counts, flowmat_relax = 0.5, smooth=True):
        exec_order =(  [(6.0, 'vertices')] + [(2.0, 'faces')]
                     + [(0.0, 'faces')] * comm['world'].size)

        for thr, mode in exec_order:
            M0 = self.element_flow_plan(target_counts)
            self.diffuse_step(M0, thr, flowmat_relax, mode)
            if smooth==True: self.smooth_until_stagnates(move_spts_nodes=True)

    def iterate_till_convergence(self, target_counts: List[int], *, flowmat_relax: float = 0.5, 
                                 max_iters: int = 1, smooth: bool = True):

        if rank['compute'] == root['compute']: print(f"TARGET: {target_counts}")

        iters = 0

        while True:
            if max_iters != -1 and iters >= max_iters:  
                break   
            iters += 1

            cur0 = self._cur_counts_total
            if rank['compute'] == root['compute']: print(f"CURRENT: {cur0}")     

            self.iterate(target_counts, flowmat_relax=flowmat_relax, smooth=smooth)

            cur1 = self._cur_counts_total
            if cur1 == cur0:
                break

    def remove_islands_till_convergence(self, max_iters=-1, target = None):
        iters = 0
        base_counts = self._cur_counts_total if target is None else target

        self.remove_outliers()

        while True:
            if max_iters != -1 and iters >= max_iters:
                break
            iters += 1

            cur0 = self._cur_counts_total

            # 1) detect first
            cluster_gids, nis_all = self.detect_islands()

            # 2) If no islands left for all ranks, stop for sure
            if all(int(nis) == 1 for nis in nis_all):
                break

            # 3) otherwise remove + do the rest
            self.remove_islands(cluster_gids)

            self.remove_outliers()
            self.add_ranks(base_counts)
            
            self.add_inliers()
            #self.smooth_until_stagnates(move_spts_nodes=True)
            self.iterate_till_convergence(base_counts, flowmat_relax=0.5, 
                                          max_iters=comm['world'].size, smooth=True)

            cur1 = self._cur_counts_total
            if cur1 == cur0:
                break
            # 2) If 0/1  islands left for all ranks, stop after trying a bit
            if all(int(nis) <= 2 for nis in nis_all):
                break

    def drain_till_convergence(self, target: List[int], max_iters: int = 100,
                               smooth=True):
        drain_ranks = [i for i, (cnt, tgt) in enumerate(zip(self._cur_counts_total, target))
                   if tgt == 0 and cnt > 0]

        # Find all ranks that have elements to drain
        curr0_to_drain = [i for i, cnt in enumerate(self._cur_counts_total) if cnt > 0]

        # Check only those ranks and see if the have been drained
        drained = all(self._cur_counts_total[i] == 0 for i in curr0_to_drain)
        if drained:
            return

        if rank['compute'] == root['compute']:
            print(f"TARGET: {target}")

        iters = 0

        while not drained:

            cur = self._cur_counts_total
            if all(cur[i] == 0 for i in drain_ranks):
                break

            drained = all(self._cur_counts_total[i] == 0 for i in curr0_to_drain)

            if max_iters != -1 and iters >= max_iters:  
                break   
            iters += 1

            self.iterate(target, flowmat_relax=1.0, smooth=smooth)

            cur0 = self._cur_counts_total
            if rank['compute'] == root['compute']: print(f"CURRENT: {cur0}")     

class OnlineDiffusionPartitioner(DiffusionRepartitioner, OnlinePartitioner):

    def __init__(self, mesh, cfg):
        OnlinePartitioner.__init__(self, mesh, cfg)
        DiffusionRepartitioner.__init__(self, mesh, cfg)

    def intg_repartition(self, target):

        self.record_perf_sample(nfevals = 10, nvars = 5)
        stagnated = self.detect_stagnation()
        worst = None
        if stagnated:
            # active_mask: exclude ranks you refuse to drain (e.g., GPU rank 0)
            active_mask = (target > 0)
            worst = self.worst_rank_by_dofs_per_sec(window=None,
                                                    active_mask=active_mask)

        drain_target = target.copy()

        if worst is not None:
            cur = np.asarray(self._cur_counts_total, dtype=np.int64)
            cost = np.asarray(self.cost, dtype=np.float64)
            active = np.nonzero((target > 0) & (np.arange(target.size) != worst))[0]
            sink = int(active[np.argmin(cost[active])]) if active.size else None

            moved_mass = int(cur[worst])
            drain_target[worst] = 0
            if sink is not None:
                drain_target[sink] += moved_mass

            # Re-normalise by ...int... the cost after moving the mass
            drain_target = self.int_round(drain_target)

        self.add_ranks(target)
        self.drain_till_convergence(drain_target, smooth=False)
        self.remove_islands_till_convergence(target=target)
        self.iterate_aggressively(target)
