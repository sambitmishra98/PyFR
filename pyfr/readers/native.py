from dataclasses import dataclass, field, replace
from math import e
import re
from copy import deepcopy

from typing import Dict, Iterable, List, Optional, Tuple


from typing import Dict, List

import h5py
import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import (AlltoallMixin, Scatterer, SparseScatterer, autofree,
                          get_comm_rank_root, mpi)
from pyfr.nputil import iter_struct

from pyfr.util import subclass_where
from pyfr.shapes import BaseShape


import math
from typing import Dict, Iterable, Optional, Tuple, List
import numpy as np



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
_BC_NONE      = -1


@dataclass
class _MetaMesh:

    mesh_src: _Mesh = None

    etypes: List[str]                 = field(default_factory=list)

    # Encoders/decoders for etypes and BCs
    e_to_i    : Dict[str, int]        = field(default_factory=dict)
    bc_name2id: Dict[str, int]        = field(default_factory=dict)

    eidxs_i: Dict[str, List[int]]       = field(default_factory=dict)
    con_i : Dict[str, np.ndarray]       = field(default_factory=dict)
    spts_nodes_i: Dict[str, np.ndarray] = field(default_factory=dict)

    eidxs_j: Dict[str, List[int]]       = field(default_factory=dict)
    con_j  : Dict[str, np.ndarray]      = field(default_factory=dict)
    spts_nodes_j: Dict[str, np.ndarray] = field(default_factory=dict)

    mesh_dest: _Mesh = None

    @staticmethod
    def _nfaces(et):
        return len(subclass_where(BaseShape, name=et).faces)

    @staticmethod
    def _mpi_sorted_union(local, comm):
        return sorted(set().union(*comm.allgather(set(local))))

    @classmethod
    def from_mesh(cls, mesh):
        comm, rank, root = get_comm_rank_root()

        etypes     = cls._mpi_sorted_union(set(mesh.etypes or ()), comm)
        e_to_i     = {et: i for i, et in enumerate(etypes)}
        bc_names   = cls._mpi_sorted_union(set((mesh.bcon or {}).keys()), comm)
        bc_name2id = {n: i for i, n in enumerate(bc_names)}

        eidxs = {et: np.asarray(mesh.eidxs.get(et, ()), dtype=np.int64) for et in etypes}
        con = {et: np.full((eidxs[et].size, cls._nfaces(et), 4), _CON_UNFILLED, np.int64) for et in etypes}
        spts_nodes = deepcopy(mesh.spts_nodes)

        mm = cls(mesh_src=mesh, etypes=etypes, 
                e_to_i=e_to_i, bc_name2id=bc_name2id, 
                eidxs_i=deepcopy(eidxs), con_i=deepcopy(con), spts_nodes_i=deepcopy(spts_nodes),
                eidxs_j=deepcopy(eidxs), con_j=deepcopy(con), spts_nodes_j=deepcopy(spts_nodes)
                )

        mm._encode_con(mesh)
        mm._fill_con_mpi(mesh)

        return mm

    def _encode_con(self, mesh):
        comm, rank, root = get_comm_rank_root()
        conL, conR = (mesh.con or ([], []))
        bcon = getattr(mesh, 'bcon', {}) or {}

        # Interior pairs: both sides are on *this* rank → owner_rank = rank
        for (etL, lidL, fL), (etR, lidR, fR) in zip(conL, conR):
            gR = int(self.eidxs_i[etR][int(lidR)])
            gL = int(self.eidxs_i[etL][int(lidL)])
            self.con_i[etL][lidL, fL] = (int(rank), self.e_to_i[etR], gR, int(fR))
            self.con_i[etR][lidR, fR] = (int(rank), self.e_to_i[etL], gL, int(fL))

        # Boundary: owner=-1, code=-1, gid=bc_id, fid=-1
        for bcname, triples in (bcon.items() if bcon else []):
            bid = self.bc_name2id.get(bcname)
            if bid is None:
                continue
            for et, lid, f in triples:
                self.con_i[et][lid, f] = (-1, -1, int(bid), -1)

    def _fill_con_mpi(self, mesh):
        comm, rank, root = get_comm_rank_root()
        cp = getattr(mesh, "con_p", {}) or {}

        # Local export: for each nbr, list our MPI faces with their GIDs
        local = {}
        for nbr, faces in cp.items():
            rec = []
            for et, lid, f in faces:
                rec.append((str(et), int(lid), int(f), int(self.eidxs_i[et][int(lid)])))
            local[int(nbr)] = rec

        all_cp = comm.allgather(local)

        # For each pair (r,s) with s>r, zip their faces; only the owning rank writes
        for r, rmap in enumerate(all_cp):
            for s, A in rmap.items():
                if s <= r:
                    continue
                B = all_cp[s].get(r, [])

                if rank == r:
                    # On r’s local side, owner is the *other* rank s
                    for (etA, lidA, fA, gidA), (etB, _, fB, gidB) in zip(A, B):
                        self.con_i[etA][lidA, fA] = (int(s), self.e_to_i[etB], int(gidB), int(fB))

                if rank == s:
                    # On s’s local side, owner is the *other* rank r
                    for (etA, _, fA, gidA), (etB, lidB, fB, gidB) in zip(A, B):
                        self.con_i[etB][lidB, fB] = (int(r), self.e_to_i[etA], int(gidA), int(fA))
        
    def plan_eidxs_dest_from_diff(self, moves_by_rank):
        comm, rank, root = get_comm_rank_root('world')

        send_plan = {
            d: {et: np.asarray(moves_by_rank.get(d, {}).get(et, ()), dtype=np.int64)
                for et in self.etypes}
            for d in range(comm.size) if d != rank
        }
        all_plans = comm.allgather(send_plan)

        to_send = {et: set().union(*(send_plan[d][et] for d in send_plan)) for et in self.etypes}

        to_recv = {et: set() for et in self.etypes}
        for sender, plan in enumerate(all_plans):
            if sender == rank:
                continue
            ed = plan.get(rank, {})
            for et in self.etypes:
                if et in ed and ed[et].size:
                    to_recv[et].update(map(int, ed[et]))

        new_eidxs = {}
        for et in self.etypes:
            cur  = list(map(int, np.asarray(self.eidxs_i[et], dtype=np.int64)))
            keep = [g for g in cur if g not in to_send[et]]
            add  = [g for g in sorted(to_recv[et]) if g not in keep]
            new_eidxs[et] = np.asarray(keep + add, dtype=np.int64)


        # one-line sanity print per rank
        ts = {et: len(to_send[et]) for et in self.etypes}
        tr = {et: len(to_recv[et]) for et in self.etypes}
        ne = {et: len(new_eidxs[et]) for et in self.etypes}
        print(f"[plan] rank={rank} to_send={ts} to_recv={tr} new_counts={ne}")

        return new_eidxs

    def relocate_with_diff(self, moves_by_rank):
        """Plan dest → relocate → order → rebuild."""
        eidxs_dest = self.plan_eidxs_dest_from_diff(moves_by_rank)
        eidxs_dest = self._apply_lex_ordering(self.to_mesh(eidxs_dest))
        return self.to_mesh(eidxs_dest)
    
    def _apply_lex_ordering(self, md):
        comm, rank, root = get_comm_rank_root()

        # Make copy of md.eidxs
        eidxs_dest = deepcopy(md.eidxs)

        for et in self.etypes:
            gids = np.asarray(eidxs_dest[et], dtype=np.int64)
            if not gids.size:
                continue
            internal = md.spts_internal[et].astype(np.int8, copy=False)
            curved   = md.spts_curved  [et].astype(np.int8, copy=False)
            order = np.lexsort((gids, curved, internal))  # primary = internal
            eidxs_dest[et] = gids[order]
        
        return eidxs_dest

    def to_mesh(self, eidxs_dest):
        ic = _MeshInterconnector(self.mesh_src.eidxs, eidxs_dest)

        spts        = ic.relocate(self.mesh_src.spts,        edim=1)
        spts_nodes  = ic.relocate(self.mesh_src.spts_nodes,  edim=0)
        spts_curved = ic.relocate(self.mesh_src.spts_curved, edim=0)
        faces_cidxs = ic.relocate(self.mesh_src.faces_cidxs, edim=0)
        faces_offs  = ic.relocate(self.mesh_src.faces_offs,  edim=0)

        mesh_dest = replace(self.mesh_src,
                                 eidxs=eidxs_dest,
                                 
                                 spts=spts, 
                                 spts_nodes=spts_nodes, 
                                 spts_curved=spts_curved,
                                 
                                 faces_cidxs=faces_cidxs, 
                                 faces_offs=faces_offs)

        self._reconstruct_con_conp_bcon(mesh_dest)
        return mesh_dest

    def _reconstruct_con_conp_bcon(self, mesh):
        comm, rank, root = get_comm_rank_root()
        
        mesh.bcon = {bc.split('/')[1]: [] for bc in mesh.codec if bc.startswith('bc/')}
        
        codec = mesh.codec
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
            except:
                pass

        # Add the internal connectivity to the mesh
        mesh.con = (conl, conr)

        for k, v in bcon.items():
            if v:
                mesh.bcon[codec[k][3:]] = v
            else:
                del mesh.bcon[codec[k][3:]]

        # MPI connectivity
        comm, rank, root = get_comm_rank_root('world')

        neighbours = [i for i in range(comm.size) if i != rank]

        ncomm = comm.Create_dist_graph_adjacent(neighbours,
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
            if rank < nrank:
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

    def compare_meshes(self, ref: _Mesh, test: _Mesh) -> bool:
        import numpy as np

        def _aeq(a, b, name):
            if (a is None) != (b is None): raise AssertionError(f"{name}: one is None")
            if a is None: return
            a, b = np.asarray(a), np.asarray(b)
            if a.shape != b.shape: raise AssertionError(f"{name}: shape {a.shape} != {b.shape}")
            if (np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating)):
                if not np.allclose(a, b, rtol=0.0, atol=0.0, equal_nan=True):
                    raise AssertionError(f"{name}: float values differ")
            else:
                if not np.array_equal(a, b): raise AssertionError(f"{name}: values differ")

        # meta
        for k in ("ndims","subset","creator","codec","uuid","version"):
            if getattr(ref, k) != getattr(test, k): raise AssertionError(f"{k} differs")
        if sorted(ref.etypes) != sorted(test.etypes): raise AssertionError("etypes differ")

        etypes = list(dict.fromkeys([*self.etypes, *sorted(set(ref.eidxs)|set(test.eidxs))]))

        # per-etype arrays
        for et in etypes:
            _aeq(ref.eidxs.get(et, np.empty(0,int)),        test.eidxs.get(et, np.empty(0,int)),        f"eidxs[{et}]")
            _aeq(ref.spts.get(et),                          test.spts.get(et),                          f"spts[{et}]")
            _aeq(ref.spts_nodes.get(et),                    test.spts_nodes.get(et),                    f"spts_nodes[{et}]")
            _aeq(ref.spts_curved.get(et),                   test.spts_curved.get(et),                   f"spts_curved[{et}]")
            _aeq(ref.faces_cidxs.get(et),                   test.faces_cidxs.get(et),                   f"faces_cidxs[{et}]")
            _aeq(ref.faces_offs.get(et),                    test.faces_offs.get(et),                    f"faces_offs[{et}]")

        # interior connectivity (order-insensitive)
        def _norm_con(con):
            if not con: return ((), ())
            L, R = con
            return (tuple(sorted(tuple(L))), tuple(sorted(tuple(R))))
        if _norm_con(ref.con) != _norm_con(test.con): raise AssertionError("con differs")

        # mpi connectivity (order-insensitive)
        def _norm_con_p(cp):
            return {int(k): tuple(sorted(tuple(v))) for k, v in (cp or {}).items()}
        if _norm_con_p(ref.con_p) != _norm_con_p(test.con_p): raise AssertionError("con_p differs")

        # boundary connectivity (order-insensitive)
        def _norm_bcon(bc):
            return {str(k): tuple(sorted(tuple(v))) for k, v in (bc or {}).items()}
        if _norm_bcon(ref.bcon) != _norm_bcon(test.bcon): raise AssertionError("bcon differs")

        return True

    @staticmethod
    def info(mesh):
        # MPI setup
        from tabulate import tabulate

        comm, rank, root = get_comm_rank_root('world')
        etypes = tuple(sorted(mesh.etypes))
        nranks = comm.size

        # Local tallies
        loc_counts = {et: len(mesh.eidxs.get(et, ())) for et in etypes}
        loc_ifaces = {int(nbr): len(faces) for nbr, faces in mesh.con_p.items()}

        # Gather to root
        all_counts = comm.gather(loc_counts, root=root)
        all_ifaces = comm.gather(loc_ifaces, root=root)
        if rank != root:
            return

        # Counts matrix [rank, etype]
        cnt = np.zeros((nranks, len(etypes)), dtype=np.int64)
        for r, d in enumerate(all_counts):
            for j, et in enumerate(etypes):
                cnt[r, j] = d.get(et, 0)

        # Symmetrised faces per pair i<j
        A = np.zeros((nranks, nranks), dtype=np.int64)
        for i, d in enumerate(all_ifaces):
            for j, c in d.items():
                A[i, j] = int(c)
        F = np.minimum(A, A.T)

        # Build table
        pairs = [(i, j) for i in range(nranks) for j in range(i + 1, nranks)]
        headers = (['etype']
                + [f'r{r}' for r in range(nranks)]
                + [f'i{i}-{j}' for (i, j) in pairs]
                + ['pairs', 'faces'])

        def fmt(x): return f'{int(x):,}'

        rows = []
        for j, et in enumerate(etypes):
            rows.append(
                [et] + [fmt(cnt[r, j]) for r in range(nranks)]
                + [''] * len(pairs) + ['', '']
            )

        pair_vals = [int(F[i, j]) for (i, j) in pairs]
        rows.append(
            ['mpi_faces']
            + [''] * nranks
            + [fmt(v) for v in pair_vals]
            + [fmt(sum(v > 0 for v in pair_vals)), fmt(sum(pair_vals))]
        )

        print(tabulate(rows, headers=headers, tablefmt='github',
                    colalign=('left', *('right',) * (len(headers) - 1))))

    def _gid_owner_maps(self) -> dict[str, dict[int, int]]:
        """Compact (gid -> owner) per etype via one allgather."""
        import numpy as np
        from pyfr.mpiutil import get_comm_rank_root

        comm, rank, _ = get_comm_rank_root('world')
        out = {}
        for et in self.etypes:
            locg = np.asarray(self.eidxs_i.get(et, ()), dtype=np.int64)
            pairs_local = np.c_[locg, np.full(locg.size, rank, np.int64)] if locg.size else np.empty((0,2), np.int64)
            gathered = comm.allgather(pairs_local)
            arrs = [a for a in gathered if isinstance(a, np.ndarray) and a.size]
            allpairs = np.vstack(arrs) if arrs else np.empty((0,2), np.int64)
            out[et] = {int(g): int(r) for g, r in allpairs}
        return out

    def compute_mpi_face_deltas(self) -> dict[int, dict[str, np.ndarray]]:
        _, myrank, _ = get_comm_rank_root('world')
        cons = {et: np.asarray(self.con_i.get(et, ()), dtype=np.int64) for et in self.etypes}

        # Fresh gid->owner maps (after the latest swaps)
        owners_map = self._gid_owner_maps()  # {et: {gid: owner_rank}}

        # Discover neighbors using fresh ownership, not stored owner column
        neighbor_set = sorted({
            int(owners_map[self.etypes[int(code)]].get(int(ngid), myrank))
            for et, con in cons.items() if con.size
            for code, ngid in zip(con[...,1].ravel(), con[...,2].ravel())
            if code >= 0  # interior
        } - {myrank})

        def per_et(et: str, con: np.ndarray, nrank: int) -> np.ndarray:
            if con.size == 0:
                return np.empty((0,2), np.int64)

            codes = con[..., 1]                  # et-index of neighbor
            ngids = con[..., 2]                  # neighbor global id
            valid = (codes >= 0)

            # Recompute owner for each face via gid->owner map
            # Build an array same shape as codes with the current owner ranks
            netypes = np.where(valid, codes, -1)
            # Map (etype index -> etype name)
            idx2et = self.etypes
            # Vectorized: fallback to myrank if unknown
            def owner_of(idx, gid):
                if idx < 0:
                    return -99  # ignored by 'valid'
                return owners_map[idx2et[int(idx)]].get(int(gid), myrank)

            # Build owners array
            owners = np.vectorize(owner_of)(netypes, np.where(valid, ngids, -1))

            c_me = np.sum(valid & (owners ==  myrank), axis=1).astype(np.int16, copy=False)
            c_n  = np.sum(valid & (owners ==  nrank ), axis=1).astype(np.int16, copy=False)

            sel = (c_n > 0)
            if not np.any(sel):
                return np.empty((0,2), np.int64)

            lids  = np.nonzero(sel)[0].astype(np.int64, copy=False)
            delta = (c_me - c_n)[sel].astype(np.int64, copy=False)
            mat   = np.c_[lids, delta]
            order = np.lexsort((mat[:,0], mat[:,1]))   # primary = delta
            return mat[order]

        return {n: {et: per_et(et, cons[et], n) for et in self.etypes} for n in neighbor_set}

    def build_moves_from_delta(self, deltas_by_rank, target_moves, etype_order=None):
        order = [et for et in (etype_order or self.etypes) if et in self.etypes]

        # Gather the min/max Δ to cap the search
        def all_deltas():
            for per_et in deltas_by_rank.values():
                for et in order:
                    a = per_et.get(et)
                    if a is not None and a.size:
                        yield from a[:, 1].tolist()

        try:
            dmin = min(all_deltas())
            dmax = max(all_deltas())
        except ValueError:
            return {}

        thresh = 0  # or start at dmin to be most permissive
        chosen = {}
        picked_once = set()

        while len(picked_once) < target_moves and thresh <= dmax:
            added = 0
            for nrank in sorted(deltas_by_rank):
                per_et = deltas_by_rank[nrank]
                for et in order:
                    arr = per_et.get(et)
                    if arr is None or arr.size == 0:
                        continue
                    # allow Δ ≤ thresh (thresh grows even if we picked nothing last round)
                    sel = arr[arr[:, 1] <= thresh]
                    if sel.size == 0:
                        continue
                    lids = sel[:, 0].astype(np.int64, copy=False)
                    gids = np.asarray(self.eidxs_i[et], np.int64)[lids]
                    take = [int(g) for g in gids if g not in picked_once]
                    if take:
                        chosen.setdefault(int(nrank), {}).setdefault(et, set()).update(take)
                        picked_once.update(take)
                        added += len(take)
                    if len(picked_once) >= target_moves:
                        break
                if len(picked_once) >= target_moves:
                    break
            # IMPORTANT: escalate even if added == 0
            thresh += 1

        return {int(n): {et: np.asarray(sorted(s), np.int64) for et, s in per.items() if s}
                for n, per in chosen.items()}


    def build_parallel_moves_to_targets(
        self,
        target_counts: List[int],
        rank_move_budget: Optional[List[int]] = None,
        etype_order: Optional[List[str]] = None,
        *,
        verbose: bool = True,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        import numpy as np
        comm, rank, _ = get_comm_rank_root('world')
        R = len(target_counts)
        tgt = np.asarray(target_counts, dtype=np.int64)

        # ---- global symmetric neighbor graph (one allgather) ----
        loc = set(self._mpi_faces_by_neighbor().keys())
        rows = comm.allgather(loc)
        N = [set() for _ in range(R)]
        for r, ns in enumerate(rows[:R]):
            for s in ns:
                if r != s:
                    N[r].add(int(s)); N[int(s)].add(int(r))

        # ---- global current counts and surplus/need ----
        my_count = sum(len(self.eidxs_i.get(et, ())) for et in self.etypes)
        cur = np.asarray(comm.allgather(int(my_count)), dtype=np.int64)

        surplus = np.maximum(0, cur - tgt)    # donors
        need    = np.maximum(0, tgt - cur)    # receivers

        if rank_move_budget is None:
            budget = surplus.copy()
        else:
            budget = np.maximum(0, np.asarray(rank_move_budget, dtype=np.int64))

        # ---- initial per-edge caps (receiver view) ----
        caps = {r: {} for r in range(R)}      # caps[r][s] = max r may send to s
        for s in range(R):
            if need[s] == 0:
                continue
            donors = [r for r in N[s] if surplus[r] > 0]
            if not donors:
                continue
            tot = int(sum(surplus[r] for r in donors))
            if tot == 0:
                continue

            # proportional to donor surplus; integerize by largest remainder to match exactly need[s]
            ideal  = {r: float(need[s]) * (float(surplus[r]) / float(tot)) for r in donors}
            floors = {r: int(np.floor(ideal[r])) for r in donors}
            left   = int(need[s] - sum(floors.values()))
            if left > 0:
                rema = sorted(((r, ideal[r] - floors[r]) for r in donors), key=lambda x: (-x[1], x[0]))
                for r2, _ in rema:
                    if left == 0: break
                    floors[r2] += 1
                    left -= 1

            for r2 in donors:
                # still edge-local for now
                caps[r2][s] = int(min(floors[r2], surplus[r2], budget[r2]))

        # ---- donor-pass clamp: enforce sum_s caps[r][s] <= budget[r] for every donor r ----
        for r in range(R):
            if not caps[r]:
                continue
            keys = sorted(caps[r])
            vals = np.asarray([caps[r][s] for s in keys], dtype=np.int64)
            total_cap = int(vals.sum())
            b = int(min(budget[r], surplus[r], total_cap))
            if total_cap <= b:
                continue  # already within budget

            # rescale proportionally, then integerize by largest remainder to sum exactly to b
            if total_cap == 0 or b == 0:
                for s in keys:
                    caps[r][s] = 0
                continue

            prop = vals.astype(float) * (float(b) / float(total_cap))
            floors = np.floor(prop).astype(int)
            left = int(b - int(floors.sum()))
            if left > 0:
                rema = prop - floors
                for i in np.argsort(-rema):
                    if left == 0: break
                    floors[i] += 1
                    left -= 1
            for i, s in enumerate(keys):
                caps[r][s] = int(floors[i])

        # ---- pick concrete elements guided by face-deltas (more negative is better) ----
        deltas_by_rank = self.compute_mpi_face_deltas()  # {nbr: {et: int64[N,2] [lid, delta]}}
        order = [et for et in (etype_order or self.etypes) if et in self.etypes]

        moves_by_rank: Dict[int, Dict[str, np.ndarray]] = {}
        picked_gids_per_et: Dict[str, set] = {et: set() for et in order}

        # helper to push chosen gids per neighbor/etype
        def _add(nbr: int, et: str, gids: List[int]):
            if not gids: return
            a = np.asarray(sorted(gids), dtype=np.int64)
            if a.size:
                moves_by_rank.setdefault(int(nbr), {}).setdefault(et, a)

        # iterate over my neighbors only; obey *clamped* per-edge caps[rank][nbr]
        for nbr in sorted(deltas_by_rank):
            cap = int(caps.get(rank, {}).get(int(nbr), 0))
            if cap <= 0:
                continue

            chosen_pairs: List[Tuple[str, int]] = []  # (etype, gid)
            picked = 0

            # gather candidates across etypes, sorted by delta ascending (most negative first)
            cand = []
            for et in order:
                arr = deltas_by_rank[nbr].get(et)
                if arr is None or arr.size == 0:
                    continue
                lids = arr[:, 0].astype(np.int64, copy=False)
                dlt  = arr[:, 1].astype(np.int64, copy=False)
                eids = np.asarray(self.eidxs_i[et], dtype=np.int64)
                for lid, d in zip(lids, dlt):
                    gid = int(eids[int(lid)])
                    if gid in picked_gids_per_et[et]:
                        continue
                    cand.append((int(d), et, int(gid)))

            cand.sort(key=lambda t: (t[0], t[2]))  # by delta, then gid for determinism

            for _, et, gid in cand:
                if picked >= cap:
                    break
                picked_gids_per_et[et].add(gid)
                chosen_pairs.append((et, gid))
                picked += 1

            if chosen_pairs:
                for et in order:
                    gs = [g for e, g in chosen_pairs if e == et]
                    _add(int(nbr), et, gs)

        if verbose:
            tot = sum(len(v) for d in moves_by_rank.values() for v in d.values())
            print(f"[parallel.plan] rank={rank} cur={int(cur[rank])} tgt={int(tgt[rank])} "
                f"surplus={int(surplus[rank])} budget={int(budget[rank])} planned_moves={tot} ")

        return moves_by_rank



    def _mpi_faces_by_neighbor(self) -> dict[int, list[tuple[str, int, int]]]:
        """
        Group *our* MPI faces by neighbor rank using self.con and gid->owner maps.
        Returns: { nbr_rank : [(etype, lid, fidx), ...], ... }
        Accepts con[..., :] with 3 cols = (code, ngid, nfid) or
                            with 4 cols = (owner, code, ngid, nfid).
        """
        from pyfr.mpiutil import get_comm_rank_root
        import numpy as np

        comm, rank, _ = get_comm_rank_root()

        # Build gid->owner maps once (collective). Used if con is 3-col.
        owners = self._gid_owner_maps()

        per_nbr: dict[int, list[tuple[str, int, int]]] = {}

        for et in self.etypes:
            con_et = self.con_i.get(et)
            if con_et is None or con_et.size == 0:
                continue

            con_et = np.asarray(con_et)
            Ne, nfaces, ncols = con_et.shape
            if ncols not in (3, 4):
                raise AssertionError(f"{et}: self.con must have 3 or 4 columns; got {ncols}")

            for lid in range(Ne):
                for f in range(nfaces):
                    rec = con_et[lid, f]

                    # decode
                    if ncols == 4:
                        owner, code, ngid, nf = map(int, rec)
                    else:  # ncols == 3
                        code, ngid, nf = map(int, rec)
                        owner = rank  # default; may be replaced below

                    # skip unfilled / boundary
                    if code == _CON_UNFILLED:
                        continue
                    if code < 0:  # boundary (-1, bc_id, -1)
                        continue

                    net = self.etypes[int(code)]

                    # if owner not provided (3-col), look it up
                    if ncols == 3:
                        owner = owners.get(net, {}).get(int(ngid), rank)

                    if owner != rank:
                        per_nbr.setdefault(int(owner), []).append((et, int(lid), int(f)))

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

        raise NotImplementedError(f"_face_vertex_indices: etype '{et}' not implemented")

    def collect_mpi_vertex_nodes(self) -> dict[int, np.ndarray]:
        """
        Return { nbr_rank : np.ndarray[int64] } = sorted-unique global vertex-node IDs
        on MPI faces to that neighbor.  Uses self.con (+owner if present) and
        self.spts_nodes_i; no con_p dependency.
        """

        if not hasattr(self, "spts_nodes_i") or self.spts_nodes_i is None:
            raise RuntimeError("self.spts_nodes_i missing")

        # 1) which faces are MPI, grouped by neighbor
        per_nbr_faces = self._mpi_faces_by_neighbor()

        # 2) per-etype face->vertex columns
        face_vtx_by_et = {et: self._face_vertex_indices(et) for et in self.etypes}

        # 3) union vertex nodes per neighbor
        out: dict[int, np.ndarray] = {}
        for nbr, faces in per_nbr_faces.items():
            vset: set[int] = set()
            for et, lid, f in faces:
                nds = self.spts_nodes_i.get(et)
                if nds is None or nds.size == 0:
                    continue
                if lid < 0 or lid >= nds.shape[0]:
                    continue

                fv = face_vtx_by_et[et][int(f)]             # vertex column indices
                verts = np.asarray(nds[int(lid), fv], dtype=np.int64).ravel()
                # filter out any negative/placeholder node ids defensively
                for v in verts:
                    iv = int(v)
                    if iv >= 0:
                        vset.add(iv)

            out[int(nbr)] = (np.asarray(sorted(vset), dtype=np.int64)
                            if vset else np.empty(0, dtype=np.int64))

        # optional: stash
        self.mpi_vertex_union = out
        return out

    def compute_mpi_face_delta_from_vertices(self) -> dict[int, dict[str, np.ndarray]]:
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
            delta = (cnt_n[sel].astype(np.int32) - cnt_int[sel].astype(np.int32)).astype(np.int64, copy=False)

            mat = np.c_[lids, delta]
            order = np.lexsort((mat[:, 0], mat[:, 1]))
            return mat[order]

        out: dict[int, dict[str, np.ndarray]] = {}
        for nrank, verts in mvu.items():
            vn  = np.asarray(verts, dtype=np.int64).ravel()
            per = {et: per_et(self.spts_nodes_i.get(et), vn) for et in self.etypes}   # <<< _i here
            out[int(nrank)] = per
        return out

    def _reset_j_from_i(self):
        """Initialize candidate state _j from current _i."""
        self.eidxs_j       = deepcopy(self.eidxs_i)
        self.spts_nodes_j  = deepcopy(self.spts_nodes_i)
        self.con_j         = deepcopy(self.con_i)

    def _lenmap(self, dic):  # {'hex': len(...), ...}
        return {et: int(len(dic.get(et, ()))) for et in self.etypes}

    def _print_swap_probe(self, where):
        from pyfr.mpiutil import get_comm_rank_root
        _, rank, _ = get_comm_rank_root('world')
        li, lj = self._lenmap(self.eidxs_i), self._lenmap(self.eidxs_j)
        print(f"[swap_probe:{where}] rank={rank} | eidxs_i={li} | eidxs_j={lj}")

    def _accept_j(self):
        """Commit _j as the new _i (single in-place swap)."""
        self.eidxs_i,      self.eidxs_j      = self.eidxs_j,      self.eidxs_i
        self.spts_nodes_i, self.spts_nodes_j = self.spts_nodes_j, self.spts_nodes_i
        self.con_i,        self.con_j        = self.con_j,        self.con_i

    def diffuse_by_mpi_vertex(self, *, target_moves=1000, chooser_rank=1):
        """
        One diffusion step driven by MPI-face vertex overlap.

        Parameters
        ----------
        target_moves : int
            Soft cap on total GIDs to move across all neighbors.
        chooser_rank : int
            Rank that proposes the plan (others send empty plans).
        last : bool
            If True, skip relocating con/spts_nodes and return a materialized mesh
            for the final step. If False, commit _j -> _i and return None.

        Returns
        -------
        mesh | None
            If last=True, returns a rebuilt mesh (ready for compute).
            Otherwise returns None (state committed internally).
        """
        comm, rank, _ = get_comm_rank_root('world')

        # 0) candidate from current
        self._reset_j_from_i()

        # 1) deltas from *_i* using face-vertex unions
        deltas = self.compute_mpi_face_delta_from_vertices()

        # 2) propose plan (only chooser_rank proposes; others send {})
        plan = self.build_moves_from_delta(deltas, target_moves) if rank == chooser_rank else {}

        # 3) build _j from plan
        self.eidxs_j = self.plan_eidxs_dest_from_diff(plan)



        # Relocate lightweight state between iterations and accept
        inter = _MeshInterconnector(self.eidxs_i, self.eidxs_j)
        self.con_j = inter.relocate(self.con_i, edim=0)
        self.spts_nodes_j = inter.relocate(self.spts_nodes_i, edim=0)

        self._accept_j()
        self._retag_con_owners_from_gid_maps()

    def _retag_con_owners_from_gid_maps(self):
        owners = self._gid_owner_maps()              # {etype: {gid: owner_rank}}
        _, myrank, _ = get_comm_rank_root('world')

        for et in self.etypes:
            con = self.con_i.get(et)
            if con is None or not np.asarray(con).size:
                continue

            con = np.asarray(con, dtype=np.int64, copy=False)
            codes = con[..., 1]                      # neighbor etype code (>=0 = paired)
            ngids = con[..., 2]                      # neighbor GID

            m = (codes >= 0)
            if not np.any(m):
                continue

            # Update owner column from the current gid->owner map
            # Loop by neighbor etype to avoid per-face string work
            for k, net in enumerate(self.etypes):
                mk = m & (codes == k)
                if not np.any(mk):
                    continue
                og = owners.get(net, {})
                # per-face lookup; fast enough in practice
                con[..., 0][mk] = np.array([og.get(int(g), myrank) for g in ngids[mk]], dtype=np.int64)

            # keep boundary/unfilled as-is (codes < 0)

    def diffuse_smoothing(self, *, return_count: bool = False) -> None:
        """
        Greedy edge-cut smoothing (vectorised):
        For each MPI-touching element, pick the neighbor with the most negative
        delta = (#faces to myrank) - (#faces to neighbor). If no negative deltas,
        do nothing for that element.

        Effects
        -------
        Commits the step internally (_accept_j). Returns None.
        """
        comm, rank, _ = get_comm_rank_root('world')

        # 0) start candidate from current
        self._reset_j_from_i()

        # 1) get per-neighbor, per-etype deltas: {nbr: {et: int64[N,2] [lid, delta]}}
        deltas_by_rank = self.compute_mpi_face_deltas()

        # 2) build a move plan in a single vectorised pass
        moves_by_rank: dict[int, dict[str, np.ndarray]] = {}

        for et in self.etypes:
            # collect (lids, deltas, nbrs) across all neighbors for this etype
            parts = []
            for nbr, per_et in deltas_by_rank.items():
                arr = per_et.get(et)
                if arr is None or arr.size == 0:
                    continue
                lids = arr[:, 0].astype(np.int64, copy=False)
                dlt  = arr[:, 1].astype(np.int64, copy=False)
                nbrs = np.full(lids.shape, int(nbr), dtype=np.int64)
                parts.append((lids, dlt, nbrs))

            if not parts:
                continue

            lids = np.concatenate([p[0] for p in parts])
            dlt  = np.concatenate([p[1] for p in parts])
            nbrs = np.concatenate([p[2] for p in parts])

            # sort by (lid, delta) so first index per lid is the *minimum* delta
            order = np.lexsort((dlt, lids))                 # primary key = dlt, grouped by lids
            lids_s, dlt_s, nbrs_s = lids[order], dlt[order], nbrs[order]

            # first occurrence per lid after sort → minimal delta neighbor
            _, first_idx = np.unique(lids_s, return_index=True)
            best_lids  = lids_s[first_idx]
            best_dlt   = dlt_s[first_idx]
            best_nbrs  = nbrs_s[first_idx]

            # keep only negative deltas
            mneg = best_dlt < 0
            if not np.any(mneg):
                continue

            chosen_lids = best_lids[mneg]
            chosen_nbrs = best_nbrs[mneg]

            # lids -> gids (stable)
            eids = np.asarray(self.eidxs_i[et], dtype=np.int64)
            gids = eids[chosen_lids]

            # group by neighbor with a tiny vectorised split
            if chosen_nbrs.size:
                un, inv = np.unique(chosen_nbrs, return_inverse=True)
                for i, nbr in enumerate(un.tolist()):
                    gsel = np.asarray(sorted(gids[inv == i].tolist()), dtype=np.int64)
                    if gsel.size:
                        moves_by_rank.setdefault(int(nbr), {})[et] = gsel

        # Debug/trace line (compact but high-signal)
        # Example: "[diffuse_smoothing] rank=2 planned_moves total=37 by_nbr={1:{hex:12,tri:7}, 3:{tri:18}}"
        breakdown = {}
        tot = 0
        for nbr, per in moves_by_rank.items():
            b = {et: int(len(g)) for et, g in per.items()}
            breakdown[int(nbr)] = b
            tot += sum(b.values())
        print(f"[diffuse_smoothing] rank={rank} planned_moves total={tot} by_nbr={breakdown}")

        # 3) compute destination layout from plan, relocate lightweight state, accept
        self.eidxs_j = self.plan_eidxs_dest_from_diff(moves_by_rank)
        inter = _MeshInterconnector(self.eidxs_i, self.eidxs_j)
        self.con_j = inter.relocate(self.con_i, edim=0)
        self.spts_nodes_j = inter.relocate(self.spts_nodes_i, edim=0)

        # 4) commit

        # before accept
        self._accept_j()
        self._retag_con_owners_from_gid_maps()
        
        return int(tot) if return_count else None

    def smooth_until_stagnates(self, *, max_iters: int = 20,
                            patience: int = 1,
                            min_change: int = 0) -> list[int]:
        """
        Run diffuse_smoothing repeatedly. The stopping decision is collective:
        we allreduce the local planned-move counts so every rank takes the same
        number of iterations. Returns the GLOBAL per-iteration move counts.
        """
        comm, _, _ = get_comm_rank_root('world')
        history: list[int] = []
        stable = 0
        last = None

        for _ in range(int(max_iters)):
            moved_local = int(self.diffuse_smoothing(return_count=True) or 0)

            moved = comm.allreduce(moved_local, op=mpi.SUM)

            history.append(moved)

            if last is not None and abs(moved - last) <= int(min_change):
                stable += 1
            else:
                stable = 0
            last = moved

            # collective stop condition
            stop = (moved == 0) or (stable >= int(patience))
            # ensure every rank evaluates the same boolean
            stop_all = comm.allreduce(1 if stop else 0, op=mpi.MAX)
            if stop_all:
                break

        return history

    def iterate(self, iters=2, *, target_moves=1000, chooser_rank=1):
        """
        Run a small pipeline of diffusion passes and return a fresh mesh.

        Current pipeline:
        - (iters - 1) internal commits of diffuse_by_mpi_vertex
        - 1 final diffuse_by_mpi_vertex that materializes and returns a mesh

        Parameters
        ----------
        iters : int
            Total number of diffusion passes of the current algorithm.
        target_moves : int
            Passed to diffuse_by_mpi_vertex.
        chooser_rank : int
            Passed to diffuse_by_mpi_vertex.

        Returns
        -------
        mesh
            New compute-ready mesh.
        """
        iters = max(1, int(iters))
        for _ in range(iters):
            self.diffuse_by_mpi_vertex(target_moves=target_moves, chooser_rank=chooser_rank,)
            self.smooth_until_stagnates(max_iters=1)

        self.smooth_until_stagnates()

        # Final pass materializes and returns the mesh
        print(f"[iterate] materializing from eidxs_i; etypes={ {et: len(self.eidxs_i.get(et, ())) for et in self.etypes} }")

        return self.to_mesh(self.eidxs_i)

    def plan_send_matrix_simple(
        self,
        current: List[int],
        target:  List[int],
        neighbors: Dict[int, Iterable[int]] | None = None,
        *,
        verbose: bool = True
    ) -> List[List[Optional[int]]]:
        """
        Simplest send-matrix planner (compact):
        - Donors split surplus to *deficit* neighbors.
        - Split ∝ neighbor TARGET (not deficit), cap by remaining need.
        - Integerize by floor + largest remainder.
        - If neighbors=None, derive the MPI graph from current mesh connectivity.
        """
        import numpy as np
        comm, _, _ = get_comm_rank_root('world')

        cur = np.asarray(current, dtype=np.int64)
        tgt = np.asarray(target,  dtype=np.int64)
        R   = int(cur.size)

        # Build symmetric adjacency N from mesh if not provided (one allgather).
        loc = set(self._mpi_faces_by_neighbor().keys())  # my neighbors now
        rows = comm.allgather(loc)
        N = [set() for _ in range(R)]
        for r, ns in enumerate(rows[:R]):
            for s in ns:
                if r != s:
                    N[r].add(int(s)); N[int(s)].add(r)

        # Matrix init: None for non-neighbors, 0 for neighbors
        M: List[List[Optional[int]]] = [[None]*R for _ in range(R)]
        for r in range(R):
            for s in N[r]:
                M[r][s] = 0

        diff      = (cur - tgt)                # +surplus / -deficit
        rem_need  = np.maximum(0, -diff)       # mutable receiver deficits
        donors    = [int(r) for r in np.argsort(-diff) if diff[int(r)] > 0]

        if verbose:
            print(f"[simple.init] R={R} sum_current={int(cur.sum())} sum_target={int(tgt.sum())} tot_diff={int(cur.sum()-tgt.sum())}")
            for r in range(R):
                print(f"[simple.diff] rank={r} curr={int(cur[r])} tgt={int(tgt[r])} diff={int(diff[r]):+d}")
            print(f"[simple.order] donors={donors}")

        for r in donors:
            surr = int(diff[r])
            cands = [s for s in sorted(N[r]) if rem_need[s] > 0]
            if not cands or surr <= 0:
                if verbose: print(f"[simple.alloc] rank={r} surplus={surr} candidates=[] sends={{}}")
                continue

            send = min(surr, int(rem_need[cands].sum()))
            w    = tgt[cands].astype(float);  sw = float(w.sum()) or float(len(cands))
            ideal = send * (w / sw)

            floors = np.minimum(np.floor(ideal).astype(int), rem_need[cands].astype(int))
            left   = int(send - int(floors.sum()))
            if left > 0:
                rema = ideal - np.floor(ideal)
                for i in np.argsort(-rema):
                    if left == 0: break
                    if floors[i] < rem_need[cands[i]]:
                        floors[i] += 1
                        left -= 1

            sends = {int(cands[i]): int(floors[i]) for i in range(len(cands)) if floors[i] > 0}
            for s, a in sends.items():
                M[r][s] += a
                rem_need[s] -= a
                diff[r]     -= a

            if verbose:
                print(f"[simple.alloc] rank={r} surplus={surr} candidates={cands} sends={{{', '.join(f'{k}:{v}' for k,v in sorted(sends.items()))}}}")

        if verbose:
            for r in range(R):
                out = {s: M[r][s] for s in range(R) if M[r][s] not in (None, 0)}
                print(f"[simple.matrix] rank={r} -> {out}")

        return M


    def iterate_to_convergence(
        self,
        target_counts,
        *,
        move_fraction: float = 1.0,
        max_iters: int = 20,
        etype_order=None,
        materialize: bool = False,
        verbose: bool = True,
        tol: int = 0,           # stop if sum(|cur - tgt|) <= tol
        min_step: int = 1,      # at least this many per donor (if surplus>0)
    ):
        import numpy as np
        comm, rank, _ = get_comm_rank_root('world')
        tgt = np.asarray(target_counts, dtype=np.int64)

        for it in range(int(max_iters)):
            my_count = int(sum(len(self.eidxs_i.get(et, ())) for et in self.etypes))
            cur = np.array([int(v) for v in comm.allgather(my_count)], dtype=np.int64)

            diff = cur - tgt
            if np.sum(np.abs(diff)) <= int(tol):
                if verbose and rank == 0:
                    print(f"[itc] converged at iter={it} counts={cur.tolist()}")
                break

            surplus = np.maximum(0, diff)
            # ceil(move_fraction*surplus), but at least min_step when surplus>0
            raw = np.ceil(move_fraction * surplus.astype(float)).astype(np.int64)
            budget = np.where(surplus > 0, np.maximum(raw, int(min_step)), 0)

            moves = self.build_parallel_moves_to_targets(
                target_counts=target_counts,
                rank_move_budget=budget.tolist(),
                etype_order=etype_order,
                verbose=False,
            )

            planned_local = sum(len(v) for d in moves.values() for v in d.values())
            planned = comm.allreduce(int(planned_local > 0), op=mpi.SUM)
            if planned == 0:
                if verbose and rank == 0:
                    print(f"[itc] no-op at iter={it}; stopping. counts={cur.tolist()} target={tgt.tolist()}")
                break

            eidxs_next = self.plan_eidxs_dest_from_diff(moves)
            inter = _MeshInterconnector(self.eidxs_i, eidxs_next)
            self.con_j        = inter.relocate(self.con_i,        edim=0)
            self.spts_nodes_j = inter.relocate(self.spts_nodes_i, edim=0)
            self.eidxs_j      = eidxs_next
            self._accept_j()
            self._retag_con_owners_from_gid_maps()


            if verbose and rank == 0:
                print(f"[itc] iter={it} planned_any=True budget_sum={int(budget.sum())} global_diff={diff.tolist()}")

        self.smooth_until_stagnates()

        if materialize:
            # optional lexicographic ordering on the final mesh
            return self.to_mesh(self._apply_lex_ordering(self.to_mesh(self.eidxs_i)))
        return None


class _MeshInterconnector(AlltoallMixin):

    def __init__(self, eidxs_src, eidxs_dest):
        comm, rank, root = get_comm_rank_root('world')

        # Link to src and dest eidxs
        self.eidxs_src = eidxs_src
        self.eidxs_dest = eidxs_dest

        # Union of all etypes across both layouts (local then global)
        etypes = comm.allgather(sorted(set(eidxs_src) | set(eidxs_dest)))
        self.etypes = sorted(set().union(*etypes))

        # WORLD-gather of per-rank global element IDs (empty for non-participants)
        self.src_all  = {et: comm.allgather(self._as64(eidxs_src.get(et, ())))
                         for et in self.etypes}
        self.dest_all = {et: comm.allgather(self._as64(eidxs_dest.get(et, ())))
                         for et in self.etypes}

        # Storage
        self.send_idxs, self.recv_idxs = {}, {}
        self.scount, self.sdisp = {}, {}
        self.rcount, self.rdisp = {}, {}
        self.src_pos, self.recv_to_dest = {}, {}
        self.n_dest, self.ssum, self.rsum = {}, {}, {}

        self._build_plan()

    def _as64(self, a):
        return np.asarray(a if a is not None else (), dtype=np.int64)

    def _build_plan(self):
        comm, rank, root = get_comm_rank_root('world')

        for et in self.etypes:
            src_local  = np.asarray(self.src_all[et][rank],  dtype=np.int64)
            dest_local = np.asarray(self.dest_all[et][rank], dtype=np.int64)

            send_lists = []
            for p in range(comm.size):
                dest_p = np.asarray(self.dest_all[et][p], dtype=np.int64)
                sel = src_local[np.isin(src_local, dest_p, assume_unique=False)] \
                      if (src_local.size and dest_p.size) \
                      else np.empty(0, np.int64)
                send_lists.append(sel)

            sidxs  = np.concatenate(send_lists) if any(a.size for a in send_lists) else np.empty(0, np.int64)
            scount = np.array([a.size for a in send_lists], dtype=np.int32)
            sdisp  = self._count_to_disp(scount)

            # Learn receive sizes, then exchange promised global IDs
            _, (rcount, rdisp) = self._alltoallcv(comm, sidxs, scount, sdisp)

            ridxs = np.empty(int(rcount.sum()), dtype=np.int64)
            self._alltoallv(comm, (sidxs, (scount, sdisp)), (ridxs, (rcount, rdisp)))

            # Save counts, disps, and recv list
            self.send_idxs[et], self.recv_idxs[et] = sidxs, ridxs
            self.scount[et], self.sdisp[et] = scount, sdisp
            self.rcount[et], self.rdisp[et] = rcount, rdisp

            # Map global->local on this rank (source side)
            self.src_pos[et] = {int(g): i for i, g in enumerate(src_local)}
            rpos = {int(g): i for i, g in enumerate(self.recv_idxs[et])}
            pr = np.fromiter((rpos.get(int(g), -1) for g in dest_local),
                                                count=dest_local.size, dtype=np.int64)
            if (pr < 0).any() and dest_local.size:
                # This happens only if dest asks for a GID nobody sent -> mapping bug
                bad = [int(g) for g, p in zip(dest_local.tolist(), pr.tolist()) if p < 0][:8]
                raise ValueError(f"[plan] r={rank} et={et} dest GIDs missing from recv: {bad}")

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
        comm, rank, root = get_comm_rank_root('world')
        edict0, specs = {}, {}

        for et in self.etypes:
            Ne = len(self.src_all[et][rank])   # rows I own on source layout (may be 0)
            a = edict_in.get(et, None)
            if a is not None:
                a = np.asarray(a)
                if a.ndim <= edim:
                    raise ValueError(f"[pre] r={rank} et={et} invalid edim={edim} shape={a.shape}")
                a0 = np.moveaxis(a, edim, 0) if edim else a
                if a0.shape[0] != Ne:
                    raise ValueError(f"[pre] r={rank} et={et} rows={a0.shape[0]} vs Ne_src={Ne}")
                edict0[et] = np.ascontiguousarray(a0)
                specs[et] = (str(a0.dtype), a0.shape[1:])
            else:
                specs[et] = (None, None)

        # Agree on dtype/shape globally; synthesize empties where needed
        gspecs = comm.allgather(specs)
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
                if self.ssum.get(et,0) or self.rsum.get(et,0) or len(self.src_all[et][rank]):
                    raise ValueError(f"[pre] no array anywhere for etype '{et}' but traffic/rows exist")
                continue

            if et not in edict0:
                Ne = len(self.src_all[et][rank])
                edict0[et] = np.empty((Ne,) + tuple(gtr), dtype=np.dtype(gdt))

        return edict0

    def _relocate_edict(self, edict0):
        comm, rank, root = get_comm_rank_root('world')
        
        out0 = {}
        for et, a0 in (edict0 or {}).items():
            if et not in self.send_idxs or et not in self.recv_to_dest:
                raise ValueError(f"[xfer] mapping not ready for etype '{et}'")
            if a0.ndim == 0:
                raise ValueError(f"[xfer] scalar array for etype '{et}'")

            gids = self.send_idxs[et]
            pos = np.fromiter((self.src_pos[et].get(int(g), -1) for g in gids),
                              count=self.ssum[et], dtype=np.int64)
            if (pos < 0).any():
                miss = [int(g) for g, p in zip(gids.tolist(), pos.tolist()) if p < 0][:8]
                raise ValueError(f"[xfer] r={rank} et={et} missing local GIDs: {miss}")

            svals = a0[pos]
            sc, sd = self.scount[et], self.sdisp[et]
            rc, rd = self.rcount[et], self.rdisp[et]

            rtot = int(rc.sum())
            rvals = np.empty((rtot, *a0.shape[1:]), dtype=a0.dtype)
            self._alltoallv(comm, (svals, (sc, sd)), (rvals, (rc, rd)))

            pr = self.recv_to_dest[et]
            if pr.size != self.n_dest[et]:
                raise ValueError(f"[xfer] r={rank} et={et} recv_to_dest size mismatch "
                                 f"({pr.size} vs {self.n_dest[et]})")

            out0[et] = rvals[pr]

        return out0

    def postproc_edict(self, edict0, *, edim):
        out = {}
        for et, a0 in (edict0 or {}).items():
            if a0.shape[0] == 0: continue
            out[et] = np.moveaxis(a0, 0, edim) if edim else a0
        return out


class NativeReader:
    def __init__(self, fname, pname=None, *, construct_con=True, 
                 comm_name = 'compute'):
        self.f = h5py.File(fname, 'r')
        self.mesh = _Mesh(fname=fname, raw=self.f)

        self.comm_name = comm_name

        comm, rank, root = get_comm_rank_root(self.comm_name)

        self.mesh.etypes = sorted(self.f['eles'])
        self.mesh.ndims = self.f['nodes'].dtype['location'].shape[0]

        # Read in and transform the various parts of the mesh
        if comm != mpi.COMM_NULL:
            self._read_metadata()
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
        comm, rank, root = get_comm_rank_root(self.comm_name)

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
        comm, rank, root = get_comm_rank_root(self.comm_name)

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
        comm, rank, root = get_comm_rank_root(self.comm_name)

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
        comm, rank, root = get_comm_rank_root(self.comm_name)
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
        comm, rank, root = get_comm_rank_root(self.comm_name)

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
