from dataclasses import dataclass, field, replace
import os
import re
from copy import deepcopy

from typing import Dict, Optional, Tuple, List

import h5py
import numpy as np

from tabulate import tabulate

from pyfr.inifile import Inifile
from pyfr.mpiutil import (AlltoallMixin, Scatterer, SparseScatterer, autofree,
                          get_comm_rank_root, mpi)
from pyfr.nputil import iter_struct

from pyfr.util import subclass_where
from pyfr.shapes import BaseShape

from pyfr.writers.native import NativeWriter


import time
from functools import wraps
from collections import defaultdict


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

@dataclass(slots=True)
class State:
    """Swappable per-rank element state (indices + light fields)."""
    eidxs:      Dict[str, np.ndarray] # etype -> (Ne,) int64 GIDs
    con:        Dict[str, np.ndarray] # etype -> (Ne, nfaces, ncols) int64
    spts_nodes: Dict[str, np.ndarray] # etype -> (Ne, nverts) int64

    def clone(self) -> "State":
        return State(deepcopy(self.eidxs), 
                     deepcopy(self.con), deepcopy(self.spts_nodes))

    def relocate_to(self, eidxs_dest: Dict[str, np.ndarray]) -> "State":
        """Relocate con and spts_nodes from self.eidxs -> eidxs_dest."""
        inter = _MeshInterconnector(self.eidxs, eidxs_dest)
        return State(
            eidxs=eidxs_dest,
            con=inter.relocate(self.con, edim=0),
            spts_nodes=inter.relocate(self.spts_nodes, edim=0),
        )

class _MetaMesh:

    __slots__ = ("etypes", "e2i", "bc2id", "mesh_src", "mesh_dest", "i", "j",
                 "_cache", "_ver", "_prof")


    # simple inclusive-time profiler for bound methods on _MetaMesh (or others with _prof)
    def fff(fn):
        name = fn.__qualname__
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                # record on instance if available; else drop on the floor
                if args and hasattr(args[0], '_prof'):
                    prof = args[0]._prof
                    tot, cnt = prof.get(name, (0.0, 0))
                    prof[name] = (tot + dt, cnt + 1)
        return wrapper


    def __init__(self, *, mesh_src: _Mesh, etypes, e2i, bc2id, i: State, j: State):
        self.mesh_src, self.mesh_dest = mesh_src, None
        self.etypes    = list(etypes)
        self.e2i       = dict(e2i)
        self.bc2id     = dict(bc2id)
        self.i, self.j = i, j
        self._cache = {}
        self._ver = {'topology': 0, 'vertices': 0}
        self._prof = {}

    # --- profiling helpers ---
    def _prof_reset(self):
        self._prof.clear()

    def _prof_print_and_reset(self, topk: int = 20):
        """Gather per-rank (time, calls) by function, print at root, then reset."""
        comm, rank, root = get_comm_rank_root('world')

        # local -> list of tuples for easier aggregation
        loc = [(k, float(v[0]), int(v[1])) for k, v in self._prof.items()]
        all_loc = comm.gather(loc, root=root)

        if rank == root:
            agg = defaultdict(lambda: [0.0, 0])
            for lst in all_loc:
                for name, t, c in lst:
                    agg[name][0] += t
                    agg[name][1] += c

            rows = []
            total_time = sum(t for t, _ in agg.values()) or 1e-12
            for name, (t, c) in agg.items():
                rows.append([name, f"{t:8.3f}", f"{c:7d}", f"{(t/c*1000.0) if c else 0:8.3f}", f"{100.0*t/total_time:6.2f}%"])
            # sort by total time desc
            rows.sort(key=lambda r: -float(r[1]))

            headers = ["function", "time_s", "calls", "avg_ms", "share"]
            print("\n[prof] timing per iterate_to_convergence (aggregated across ranks)")
            print(tabulate(rows[:topk], headers=headers, tablefmt='github',
                           colalign=('left','right','right','right','right')))
        # reset everywhere
        self._prof_reset()


    @property
    def eidxs_i(self): return self.i.eidxs
    @property
    def con_i(self): return self.i.con
    @property
    def spts_nodes_i(self): return self.i.spts_nodes

    @property
    def eidxs_j(self): return self.j.eidxs
    @eidxs_j.setter
    def eidxs_j(self, v): self.j.eidxs = v
    @property
    def con_j(self): return self.j.con
    @con_j.setter
    def con_j(self, v): self.j.con = v
    @property
    def spts_nodes_j(self): return self.j.spts_nodes
    @spts_nodes_j.setter
    def spts_nodes_j(self, v): self.j.spts_nodes = v

    def _bump_topology(self):
        self._ver['topology'] += 1
        self._cache.clear()

    def _reset_j_with_i(self):
        self.j = self.i.clone()
        self._bump_topology()

    def _accept_j_into_i(self):
        self.i, self.j = self.j, self.i
        self._retag_con_owners_from_gid_maps()
        self._bump_topology()

    def _array_views(self):
        """Cache numpy views of eidxs and a gid->lid map per etype."""
        key = ('views', self._ver['topology'])
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        eidxs_np = {et: np.asarray(self.eidxs_i.get(et, ()), dtype=np.int64)
                    for et in self.etypes}
        # Python dict is fine here (built once per topology bump)
        gid2lid  = {et: {int(g): i for i, g in enumerate(arr.tolist())}
                    for et, arr in eidxs_np.items()}

        self._cache[key] = (eidxs_np, gid2lid)
        return self._cache[key]


    @staticmethod
    def _mpi_sorted_union(local, comm):
        """Sorted union of a locally defined set across all ranks."""
        return sorted(set().union(*comm.allgather(set(local))))

    # -----------------
    # _Mesh <--> _MetaMesh

    @classmethod
    def from_mesh(cls, mesh: _Mesh):
        comm, rank, root = get_comm_rank_root()

        etypes   = cls._mpi_sorted_union(set(mesh.etypes or ()), comm)
        bc_names = cls._mpi_sorted_union(set((mesh.bcon or {}).keys()), comm)
        e2i      = {et: i for i, et in enumerate(etypes)}
        bc2id    = {n: i for i, n in enumerate(bc_names)}

        eidxs = {et: np.asarray(mesh.eidxs.get(et, ()), dtype=np.int64) for et in etypes}
        con   = {et: np.full((eidxs[et].size, cls._nfaces(et), 4), _CON_UNFILLED, np.int64) for et in etypes}
        spts_nodes = deepcopy(mesh.spts_nodes)

        mm = cls(mesh_src=mesh, etypes=etypes, e2i=e2i, bc2id=bc2id,
                 i=State(eidxs=deepcopy(eidxs), con=deepcopy(con), 
                         spts_nodes=deepcopy(spts_nodes)),
                 j=State(eidxs=deepcopy(eidxs), con=deepcopy(con), 
                         spts_nodes=deepcopy(spts_nodes)))

        mm._encode_con(mesh)      # writes into mm.i.con
        mm._fill_con_mpi(mesh)    # writes into mm.i.con
        return mm

    def _encode_con(self, mesh):
        """
            Encode the mesh connectivity as
            con_i[etype][local_id, face_id] = (rank, i, ngid, nface)
            where
                rank  = rank owning the neighbor element ,    -1 for boundary
                i     = idx of nbr etype in self.etypes  ,    -1 for boundary
                ngid  = global ID of the neighbor element, bc_id for boundary
                nface = face ID on the neighbor element  ,    -1 for boundary
        """

        comm, rank, root = get_comm_rank_root()
        conL, conR = (mesh.con or ([], []))
        bcon = getattr(mesh, 'bcon', {}) or {}

        # Interior pairs: both sides are on *this* rank → owner_rank = rank
        for (etL, lidL, fL), (etR, lidR, fR) in zip(conL, conR):
            gR = int(self.eidxs_i[etR][int(lidR)])
            gL = int(self.eidxs_i[etL][int(lidL)])
            self.con_i[etL][lidL, fL] = (int(rank), self.e2i[etR], gR, int(fR))
            self.con_i[etR][lidR, fR] = (int(rank), self.e2i[etL], gL, int(fL))

        # Boundary: owner=-1, code=-1, gid=bc_id, fid=-1
        for bcname, triples in (bcon.items() if bcon else []):
            bid = self.bc2id.get(bcname)
            if bid is None:
                continue
            for et, lid, f in triples:
                self.con_i[et][lid, f] = (-1, -1, int(bid), -1)

    def _fill_con_mpi(self, mesh):
        """
        Fill nrank faces in con_i using mesh.con_p.

        For each nbr,
            build local list of MPI faces with their GIDs, 
            allgather everyone's lists,
            zip list with the nbr's reciprocal list to tag:
            con_i[et, lid, fid] = (nbr, e2i[et_of_nbr], gid_of_nbr, fid_of_nbr)
        """
        comm, rank, root = get_comm_rank_root()
        cp = getattr(mesh, "con_p", {}) or {}

        # Our local export: {nbr: [(et, lid, fid, gid), ...]}
        local = {
            int(nbr): [(et, int(lid), int(fid), self.eidxs_i[et][int(lid)]) 
                       for (et, lid, fid) in faces]
            for nbr, faces in cp.items()
        }

        # Everyone's exports
        all_cp = comm.allgather(local)

        for nbr, A in local.items():
            B = all_cp[int(nbr)].get(rank, [])
            for (etA, lidA, fA, _gidA), (etB, _lidB, fB, gidB) in zip(A, B):
                self.con_i[etA][lidA, fA] = (int(nbr), self.e2i[etB], gidB, fB)

    def to_mesh(self, eidxs_dest):
        ic = _MeshInterconnector(self.mesh_src.eidxs, eidxs_dest)

        mesh_int = replace(self.mesh_src, eidxs=eidxs_dest,
            spts=ic.relocate(self.mesh_src.spts, edim=1), 
            spts_nodes=ic.relocate(self.mesh_src.spts_nodes, edim=0),
            spts_curved=ic.relocate(self.mesh_src.spts_curved, edim=0),
            faces_cidxs=ic.relocate(self.mesh_src.faces_cidxs, edim=0),
            faces_offs=ic.relocate(self.mesh_src.faces_offs, edim=0),
            )

        self._reconstruct_con_conp_bcon(mesh_int)

        eidxs_dest = self._apply_lex_ordering(mesh_int)
        ic2 = _MeshInterconnector(mesh_int.eidxs, eidxs_dest)
        
        mesh_dest = replace(mesh_int, eidxs=eidxs_dest,
            spts=ic2.relocate(mesh_int.spts, edim=1), 
            spts_nodes=ic2.relocate(mesh_int.spts_nodes, edim=0),
            spts_curved=ic2.relocate(mesh_int.spts_curved, edim=0),
            faces_cidxs=ic2.relocate(mesh_int.faces_cidxs, edim=0),
            faces_offs=ic2.relocate(mesh_int.faces_offs, edim=0),
            )

        self._reconstruct_con_conp_bcon(mesh_dest)

        # Remove keys with empty entries
        mesh_dest.eidxs = {k: v for k, v in mesh_dest.eidxs.items() if v.size}

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
                
        mesh.con_p = {k: v for k, v in mesh.con_p.items() if v}

    def _apply_lex_ordering(self, mesh):
        comm, rank, root = get_comm_rank_root()

        eidxs_dest = {}

        for et in self.etypes:
            gids = np.asarray(mesh.eidxs.get(et, ()), dtype=np.int64)
            if not gids.size: eidxs_dest[et] = gids; continue
            internal = mesh.spts_internal[et].astype(np.int8, copy=False)
            curved   = mesh.spts_curved  [et].astype(np.int8, copy=False)
            order = np.lexsort((gids, curved, internal))  # primary = internal
            eidxs_dest[et] = gids[order]
        
        return eidxs_dest

    # -----------------
    # Small geometry helpers
    # ----------------

    @staticmethod
    def _nfaces(etype):
        """Number of faces in an element of a given etype."""
        return len(subclass_where(BaseShape, name=etype).faces)

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

    def _face_vtx_by_et_cached(self):
        key = ('face_vtx_by_et', tuple(self.etypes))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        m = {et: self._face_vertex_indices(et) for et in self.etypes}
        self._cache[key] = m
        return m

    
    #-----------------
    # Topology helpers
    #-----------------

    @fff
    def _mpi_faces_by_neighbor(self) -> dict[int, list[tuple[str, int, int]]]:
        """
        Group *our* MPI faces by neighbor rank using self.con and gid->owner maps.
        Returns: { nbr_rank : [(etype, lid, fidx), ...], ... }
        Accepts con[..., :] with 3 cols = (code, ngid, nfid) or
                            with 4 cols = (owner, code, ngid, nfid).
        """
        comm, rank, _ = get_comm_rank_root()

        # Build gid->owner maps once (collective). Used if con is 3-col or owner stale.
        owners = self._gid_owner_maps(self.eidxs_i)

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
                        owner = rank  # default; will be recomputed below if interior

                    # skip unfilled / boundary
                    if code == _CON_UNFILLED:
                        continue
                    if code < 0:  # boundary (-1, bc_id, -1)
                        continue

                    net = self.etypes[int(code)]

                    # recompute owner from the gid->owner map for robust neighbor tagging
                    owner = owners.get(net, {}).get(int(ngid), owner)

                    if owner != rank:
                        per_nbr.setdefault(int(owner), []).append((et, int(lid), int(f)))

        # Trace for deterministic debugging
        print(f"[impl.swap] rank={rank} using owner-map (legacy) in _mpi_faces_by_neighbor")
        return per_nbr


    @fff
    def _faces_matrix_cached(self):
        """Symmetric faces matrix F built once per topology."""
        comm, rank, _ = get_comm_rank_root('world')
        key = ('F_symmetric', self._ver['topology'], comm.size)
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        per_nbr = self._mpi_faces_by_neighbor()
        nlocal  = {int(n): len(faces) for n, faces in per_nbr.items()}

        A = np.zeros((comm.size, comm.size), dtype=np.int64)
        for n, c in nlocal.items():
            A[rank, int(n)] = int(c)
        A = comm.allreduce(A, op=mpi.SUM)
        F = np.minimum(A, A.T).astype(np.float64)
        self._cache[key] = F
        if rank == 0:
            nnz = int((F > 0).sum())
            print(f"[cache.F] topo={self._ver['topology']} nnz={nnz}")
        return F

    def _gid_owner_maps(self, eidxs) -> dict[str, dict[int, int]]:
        """Compact (gid -> owner) per etype via one allgather."""
        comm, rank, root = get_comm_rank_root('world')
        out = {}
        for et in self.etypes:
            locg = np.asarray(eidxs.get(et, ()), dtype=np.int64)
            pairs_local = np.c_[locg, np.full(locg.size, rank, np.int64)] if locg.size else np.empty((0,2), np.int64)
            gathered = comm.allgather(pairs_local)
            arrs = [a for a in gathered if isinstance(a, np.ndarray) and a.size]
            allpairs = np.vstack(arrs) if arrs else np.empty((0,2), np.int64)
            out[et] = {int(g): int(r) for g, r in allpairs}
        return out

    @fff
    def collect_mpi_vertex_nodes(self) -> dict[int, np.ndarray]:
        key = (self._ver['topology'],)
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        per_nbr_faces = self._mpi_faces_by_neighbor()
        face_vtx_by_et = self._face_vtx_by_et_cached()
        out: dict[int, np.ndarray] = {}
        for nbr, faces in per_nbr_faces.items():
            vset = set()
            for et, lid, f in faces:
                nds = self.spts_nodes_i.get(et)
                if nds is None or nds.size == 0:
                    continue
                fv = face_vtx_by_et[et][int(f)]
                verts = np.asarray(nds[int(lid), fv], dtype=np.int64).ravel()
                vset.update(int(v) for v in verts if int(v) >= 0)
            out[int(nbr)] = (np.fromiter(sorted(vset), dtype=np.int64)
                            if vset else np.empty(0, np.int64))

        self._cache[key] = out
        sizes = {int(k): int(v.size) for k, v in out.items()}
        print(f"[cache.vunion] topo={self._ver['topology']} sizes={sizes}")
        return out


    #-----------------
    # Delta builders
    #-----------------

    def _face_owner_views_cached(self):
        key = ('owners_valid', self._ver['topology'])
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        views = {}
        for et in self.etypes:
            con = np.asarray(self.con_i.get(et, ()), dtype=np.int64)
            if con.size == 0:
                continue
            owners = con[..., 0]
            valid  = (con[..., 1] >= 0)
            views[et] = (owners, valid)
        self._cache[key] = views
        return views

    @fff
    def _deltas_edge_cached(self) -> dict[int, dict[str, np.ndarray]]:
        key = ('deltas.edge', self._ver['topology'])
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        comm, rank, _ = get_comm_rank_root('world')
        nbrs = sorted(self._mpi_faces_by_neighbor().keys())
        out: dict[int, dict[str, np.ndarray]] = {int(n): {} for n in nbrs}

        for et in self.etypes:
            con = np.asarray(self.con_i.get(et, ()), dtype=np.int64)
            if con.size == 0:
                continue
            owners = con[..., 0]
            codes  = con[..., 1]
            valid  = (codes >= 0)
            c_me   = np.sum(valid & (owners == rank), axis=1).astype(np.int16, copy=False)
            for nrank in nbrs:
                c_n = np.sum(valid & (owners == int(nrank)), axis=1).astype(np.int16, copy=False)
                sel = (c_n > 0)
                if not np.any(sel):
                    continue
                lids  = np.nonzero(sel)[0].astype(np.int64, copy=False)
                delta = (c_me[sel] - c_n[sel]).astype(np.int64, copy=False)
                mat   = np.c_[lids, delta]
                order = np.lexsort((mat[:, 0], mat[:, 1]))
                out[int(nrank)][et] = mat[order]

        self._cache[key] = out
        print(f"[cache.deltas.edge] topo={self._ver['topology']} "
            f"nbrs={sorted(out)}")
        return out

    @fff
    def _deltas_vertex_cached(self) -> dict[int, dict[str, np.ndarray]]:
        key = ('deltas.vtx', self._ver['topology'])
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        mvu = self.collect_mpi_vertex_nodes()
        all_sets  = [np.asarray(v, dtype=np.int64).ravel() for v in mvu.values() if v is not None]
        union_all = np.unique(np.concatenate(all_sets)) if all_sets else np.empty(0, dtype=np.int64)

        def per_et(nds: np.ndarray, nbr_vertices: np.ndarray) -> np.ndarray:
            if nds is None or np.asarray(nds).size == 0:
                return np.empty((0, 2), dtype=np.int64)
            nds = np.asarray(nds, dtype=np.int64)
            v_in_n   = np.isin(nds, nbr_vertices, assume_unique=False)
            cnt_n    = v_in_n.sum(axis=1).astype(np.int16, copy=False)
            if union_all.size:
                v_internal = np.isin(nds, union_all, invert=True, assume_unique=False)
            else:
                v_internal = np.ones_like(nds, dtype=bool)
            cnt_int = v_internal.sum(axis=1).astype(np.int16, copy=False)
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
            per = {et: per_et(self.spts_nodes_i.get(et), vn) for et in self.etypes}
            out[int(nrank)] = per

        self._cache[key] = out
        print(f"[cache.deltas.vtx] topo={self._ver['topology']} "
            f"nbrs={sorted(out)} union_all={int(union_all.size)}")
        return out

    def _compute_deltas(self, score_by: str):
        m = str(score_by).lower()
        if m in ('edge','edges','face','faces'):
            return self.compute_mpi_face_deltas()              # legacy behavior
        elif m in ('vertex','vertices','vtx','vtxs'):
            return self.compute_mpi_face_delta_from_vertices() # unchanged
        else:
            raise ValueError(f"[itc4.mode] unknown score_by='{score_by}'; use 'edge' or 'vertex'")


    @fff
    def compute_mpi_face_deltas(self) -> dict[int, dict[str, np.ndarray]]:
        comm, rank, _ = get_comm_rank_root('world')
        cons = {et: np.asarray(self.con_i.get(et, ()), dtype=np.int64) for et in self.etypes}

        # Fresh gid->owner maps (after the latest swaps)
        owners_map = self._gid_owner_maps(self.eidxs_i)  # {et: {gid: owner_rank}}

        # Discover neighbors using fresh ownership, not stored owner column
        neighbor_set = sorted({
            int(owners_map[self.etypes[int(code)]].get(int(ngid), rank))
            for et, con in cons.items() if con.size
            for code, ngid in zip(con[...,1].ravel(), con[...,2].ravel())
            if code >= 0  # interior
        } - {rank})

        def per_et(et: str, con: np.ndarray, nrank: int) -> np.ndarray:
            if con.size == 0:
                return np.empty((0,2), np.int64)

            codes = con[..., 1]                  # et-index of neighbor
            ngids = con[..., 2]                  # neighbor global id
            valid = (codes >= 0)

            # Recompute owner for each face via gid->owner map (vectorized wrapper)
            idx2et = self.etypes
            def owner_of(idx, gid):
                if int(idx) < 0:
                    return -99  # ignored by 'valid'
                return owners_map[idx2et[int(idx)]].get(int(gid), rank)

            owners = np.vectorize(owner_of)(np.where(valid, codes, -1), np.where(valid, ngids, -1))

            c_me = np.sum(valid & (owners ==  rank), axis=1).astype(np.int16, copy=False)
            c_n  = np.sum(valid & (owners ==  nrank), axis=1).astype(np.int16, copy=False)

            sel = (c_n > 0)
            if not np.any(sel):
                return np.empty((0,2), np.int64)

            lids  = np.nonzero(sel)[0].astype(np.int64, copy=False)
            delta = (c_me - c_n)[sel].astype(np.int64, copy=False)
            mat   = np.c_[lids, delta]
            order = np.lexsort((mat[:,0], mat[:,1]))   # primary = delta
            return mat[order]

        out = {n: {et: per_et(et, cons[et], n) for et in self.etypes} for n in neighbor_set}
        print(f"[impl.swap] rank={rank} using owner-map (legacy) in compute_mpi_face_deltas")
        return out


    @fff
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

    # -----------------
    
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

    @fff
    def plan_eidxs_dest_from_diff(self, eidxs_diff, verbose=False):
        comm, rank, root = get_comm_rank_root('world')

        send_plan = {
            d: {et: np.asarray(eidxs_diff.get(d, {}).get(et, ()), dtype=np.int64)
                for et in self.etypes}
            for d in range(comm.size) if d != rank
        }
        all_plans = comm.allgather(send_plan)

        to_send = {et: set().union(*(send_plan[d][et] for d in send_plan)) 
                   for et in self.etypes}

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

        if verbose:
            # one-line sanity print per rank
            ts = {et: len(to_send[et]) for et in self.etypes}
            tr = {et: len(to_recv[et]) for et in self.etypes}
            ne = {et: len(new_eidxs[et]) for et in self.etypes}
            print(f"[plan] R{rank} \t send:{ts} \t recv:{tr} \t target:{ne}")

        return new_eidxs

    @fff
    def relocate_i_to_j(self, eidxs_diff):
        # materialize j directly from i using the relocation plan
        self.j = self.i.relocate_to(self.plan_eidxs_dest_from_diff(eidxs_diff))

    @fff
    def _retag_con_owners_from_gid_maps(self):
        """
            Retag owners in new con_i from current gid->owner maps
        """

        owners = self._gid_owner_maps(self.eidxs_i)              # {etype: {gid: owner_rank}}
        comm, rank, root = get_comm_rank_root('world')

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
                con[..., 0][mk] = np.array([og.get(int(g), rank) for g in ngids[mk]], dtype=np.int64)

            # keep boundary/unfilled as-is (codes < 0)

    # -----------------
    # Topology/adjacency helpers
    # -----------------
    
    # ---------------------
    # Iteration controller 
    # ---------------------

    @fff
    def element_flow_plan(self, cur_counts, tgt_counts, mask, *, 
                          max_iters: int = 4) -> np.ndarray:
        """
        Electrical-flow, cap-aware, integer planner on the rank graph (mask-aware).

        Parameters
        ----------
        cur_counts : sequence[int]
            Current per-rank element counts.
        tgt_counts : sequence[int]
            Target per-rank element counts.
        mask : np.ndarray[bool] (R,R)
            Directional allow mask. True means u->v shipments are permitted.
            Diagonal should be False. This function WILL NOT create a mask.

        Returns
        -------
        M : (R,R) int64
            Cumulative shipments. Only neighbor edges nonzero; diag=0.
            Enforces adjacency and the provided directional mask.
        """
        # MPI
        comm, rank, _ = get_comm_rank_root('world')

        # ---------- helpers ----------

        def _build_faces_matrix() -> tuple[np.ndarray, list[list[int]]]:
            F = self._faces_matrix_cached()
            N = [np.nonzero(F[r])[0].tolist() for r in range(comm.size)]
            return F, N

        def _lap_solve(L: np.ndarray, b: np.ndarray, anchor: int = 0) -> np.ndarray:
            """Solve L p = b with p[anchor] = 0 (tiny dense solve; R is small)."""
            R = L.shape[0]
            idx = [i for i in range(R) if i != anchor]
            Lr  = L[np.ix_(idx, idx)]
            pr  = np.linalg.solve(Lr, b[idx]) if Lr.size else np.empty(0)
            p   = np.zeros(R, dtype=np.float64)
            p[idx] = pr
            return p  # p[anchor] stays 0

        def _downhill_weights(p: np.ndarray, Fm: np.ndarray, N: list[list[int]]) -> np.ndarray:
            """T[u,v] = Fm[u,v] * max(p[u]-p[v], 0) for neighbors; else 0."""
            R = Fm.shape[0]
            T = np.zeros_like(Fm, dtype=np.float64)
            for u in range(R):
                pu = p[u]
                for v in N[u]:
                    if u != v:
                        drop = pu - p[v]
                        if drop > 0:
                            T[u, v] = Fm[u, v] * drop
            return T

        def _distribute_with_caps(
            excess: np.ndarray, deficit: np.ndarray, T: np.ndarray, Fm: np.ndarray
        ) -> np.ndarray:
            """
            Integer shipments S per source row respecting:
            - per-sink caps (incoming ≤ deficit),
            - per-source integer totals (largest remainder),
            - relay split proportional to Fm among downhill non-sinks.
            """
            R   = T.shape[0]
            eps = 1e-12
            P   = np.zeros((R, R), dtype=np.float64)

            # 1) downhill proportional split by T
            for u in range(R):
                e = int(excess[u])
                if e <= 0: continue
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
                    P[:, sinks] *= alpha  # broadcast scaling per sink

            # 3) leftover per source goes to relay neighbors (non-sinks), proportional to Fm
            for u in range(R):
                e = int(excess[u])
                if e <= 0: continue
                rem = float(e) - float(P[u].sum())
                if rem <= 1e-9: continue
                cand = [v for v in range(R) if (deficit[v] == 0 and T[u, v] > eps)]
                if not cand: continue
                w = Fm[u, cand].astype(np.float64)
                s = float(w.sum())
                if s > eps:
                    P[u, cand] += rem * (w / s)

            # 4) integerize per source; tie-break: larger Fm[u,v], then smaller index
            S = np.zeros((R, R), dtype=np.int64)
            for u in range(R):
                e = int(excess[u])
                if e <= 0: continue
                base = np.floor(P[u] + 1e-9).astype(np.int64)
                left = int(e - int(base.sum()))
                if left > 0:
                    rema = P[u] - base
                    order = np.lexsort((np.arange(R), -Fm[u], -rema))
                    for j in order[:left]:
                        base[j] += 1
                S[u] = base
            return S

        def _cancel_anti_parallel(M: np.ndarray) -> np.ndarray:
            """Net i→j and j→i so the plan is easy to reason about."""
            Mc = M.copy()
            R  = Mc.shape[0]
            for i in range(R):
                for j in range(i + 1, R):
                    a, b = int(Mc[i, j]), int(Mc[j, i])
                    if a and b:
                        if a >= b: Mc[i, j], Mc[j, i] = a - b, 0
                        else:      Mc[j, i], Mc[i, j] = b - a, 0
            return Mc

        # --- inputs as arrays
        cur = np.asarray(cur_counts, dtype=np.int64).copy()
        tgt = np.asarray(tgt_counts, dtype=np.int64).copy()

        # --- faces and base conductance
        F, _ = _build_faces_matrix()

        # --- REQUIRED mask handling (no mask creation here)
        maskb = np.array(mask, dtype=bool, copy=True)
        if maskb.shape != F.shape:
            raise ValueError(f"mask shape {maskb.shape} != {F.shape}")
        np.fill_diagonal(maskb, False)

        # UNDIRECTED support for Laplacian; keep only edges that exist in F
        support = (maskb | maskb.T) & (F > 0)
        Fm = F * support.astype(np.float64)

        # If mask disconnects a donor completely, minimally relax by opening its strongest-F neighbor
        diff    = cur - tgt
        donors  = np.where(diff > 0)[0]
        relaxed = []
        for u in donors:
            if support[u].sum() == 0:
                v = int(np.argmax(F[u]))
                if F[u, v] > 0:
                    support[u, v] = True
                    support[v, u] = True
                    Fm[u, v] = F[u, v]
                    Fm[v, u] = F[v, u]
                    relaxed.append((u, v))

        # neighbor list and Laplacian on masked conductance
        N = [np.nonzero(Fm[r])[0].tolist() for r in range(comm.size)]
        L = np.diag(Fm.sum(axis=1)) - Fm

        M = np.zeros_like(F, dtype=np.int64)

        for it in range(int(max_iters)):
            diff    = cur - tgt
            deficit = np.maximum(0, -diff).astype(np.int64)
            excess  = np.maximum(0,  diff).astype(np.int64)

            if excess.sum() == 0 or deficit.sum() == 0:
                print(f"[elec_cap] it={it} done; excess={excess.tolist()} deficit={deficit.tolist()}")
                break

            b = excess.astype(np.float64) - deficit.astype(np.float64)
            assert abs(float(b.sum())) < 1e-6, "b must sum to zero"

            p = _lap_solve(L, b, anchor=0)
            T = _downhill_weights(p, Fm, N)

            # Enforce DIRECTIONAL mask on traffic
            T *= maskb.astype(np.float64)

            S = _distribute_with_caps(excess, deficit, T, Fm)

            out = S.sum(axis=1); inc = S.sum(axis=0)
            cur = (cur - out + inc).astype(np.int64)
            M  += S

            # Debug (kept concise; uncomment per need)
            # print(f"[elec_cap.step] it={it} rank={rank} out={int(out[rank])} in={int(inc[rank])} diff={int((cur-tgt)[rank])}")

        # Final enforcement: adjacency and directional mask; clear diag
        np.fill_diagonal(M, 0)
        M *= (F > 0).astype(np.int64)
        M *= maskb.astype(np.int64)

        return _cancel_anti_parallel(M)

    @fff
    def diffuse_smoothing(self, *,
                        return_count: bool = False,
                        threshold: int = 0,
                        target_counts: Optional[List[int]] = None,
                        restrict_src_dest: bool = False) -> Optional[int]:
        comm, rank, root = get_comm_rank_root('world')

        # 0) start candidate from current
        self._reset_j_with_i()

        # Directional gating
        allowed_nbrs = None
        I_am_donor = True
        if restrict_src_dest and target_counts is not None:
            my_count = int(sum(len(self.eidxs_i.get(et, ())) for et in self.etypes))
            cur = np.asarray(comm.allgather(int(my_count)), dtype=np.int64)
            tgt = np.asarray(target_counts, dtype=np.int64)
            diff = cur - tgt  # +surplus / -deficit
            I_am_donor = diff[rank] > 0

            if I_am_donor:
                # donors may send only to neighbors with deficit
                allowed_nbrs = {int(n) for n in range(comm.size) if diff[int(n)] < 0}
            else:
                # receivers: we will still do the collective relocate (with empty plan)
                allowed_nbrs = set()

        # 1) per-neighbor, per-etype deltas
        deltas_by_rank = self.compute_mpi_face_deltas()

        # 2) build move plan (respect allowed_nbrs if set)
        eidxs_diff: dict[int, dict[str, np.ndarray]] = {}

        for et in self.etypes:
            arrs = []
            nbr_ids = []
            for nbr, per_et in deltas_by_rank.items():
                a = per_et.get(et)
                if a is not None and a.size:
                    arrs.append(a)          # shape (Mi, 2) with [lid, delta]
                    nbr_ids.append(int(nbr))

            if not arrs:
                continue

            # Stack into one big array and compute best-by-lid without Python unique
            big = np.vstack(arrs)                   # (M, 2)
            nbrs_for_big = np.repeat(nbr_ids, [a.shape[0] for a in arrs]).astype(np.int64)

            # big is already sorted by (delta, lid) from builder; first occurrence wins.
            # Keep first occurrence per lid:
            lids = big[:, 0].astype(np.int64)
            first = np.empty_like(lids, dtype=bool); first[:] = False
            # np.unique on sorted “lids with stable=True” returns first indices
            ulids, idx = np.unique(lids, return_index=True)
            first[idx] = True

            best_lids  = big[first, 0]
            best_dlt   = big[first, 1]
            best_nbrs  = nbrs_for_big[first]

            mkeep = (best_dlt < int(threshold))
            if not np.any(mkeep):
                continue
            chosen_lids = best_lids[mkeep]
            chosen_nbrs = best_nbrs[mkeep]
            # ... then group by chosen_nbrs like you already do


            eids = np.asarray(self.eidxs_i[et], dtype=np.int64)
            gids = eids[chosen_lids]

            if chosen_nbrs.size:
                un, inv = np.unique(chosen_nbrs, return_inverse=True)
                for i, nbr in enumerate(un.tolist()):
                    gsel = np.asarray(sorted(gids[inv == i].tolist()), dtype=np.int64)
                    if gsel.size:
                        eidxs_diff.setdefault(int(nbr), {})[et] = gsel

        breakdown, tot = {}, 0
        for nbr, per in eidxs_diff.items():
            b = {et: int(len(g)) for et, g in per.items()}
            breakdown[int(nbr)] = b
            tot += sum(b.values())

        # Helpful trace (donor/receiver)
        role = "donor" if I_am_donor else "receiver"

        # 3) ALWAYS participate in relocate (even with empty plan)
        self.relocate_i_to_j(eidxs_diff)
        self._accept_j_into_i()

        return int(tot) if return_count else None

    @fff
    def diffuse_smoothing2(self, flow_matrix: np.ndarray, *, threshold = 0, 
        etype_order: List[str] = None, return_count = False, verbose = False, 
        scale = 0.5, score_by: str = 'vertex',
    ) -> Optional[int]:
        """Use a pre-planned flow matrix to pick elements by Δ and relocate."""

        # MPI info
        comm, rank, root = get_comm_rank_root('world')

        # Validate flow matrix
        M = np.asarray(flow_matrix, dtype=np.int64)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"[itc4] flow_matrix must be square; got shape {M.shape}")
        R = int(M.shape[0])
        if not (0 <= rank < R):
            raise ValueError(f"[itc4] rank {rank} out of range for flow_matrix of size {R}")

        # 0) snapshot _i -> _j
        self._reset_j_with_i()

        # 0a) mode line
        if rank == root:
            print(f"[itc4.mode] rank={rank} score_by={str(score_by).lower()} thr={int(threshold)} scale={float(scale):.3f}")

        # Lightweight cached views
        eidxs_np, _gid2lid = self._array_views()

        # Restrict to requested order (deterministic)
        base_order = [et for et in (etype_order or self.etypes) if et in self.etypes]

        # One boolean “picked” mask per etype so we never double-pick a lid
        picked_mask_by_et = {et: np.zeros(eidxs_np[et].size, dtype=bool) for et in base_order}

        # Precomputed per-neighbor deltas (cached under the current topology)
        deltas_by_rank = self._compute_deltas(score_by)  # {nbr: {et: int64[N,2] (lid, Δ)}}

        # Helper: pick up to `need` elements for this neighbor using masks only
        def _pick_for_neighbor(nbr: int, need: int, *, deltas_by_rank, threshold: int,
                            base_order: list[str], verbose=False) -> dict[str, np.ndarray]:
            if need <= 0:
                return {}
            per = deltas_by_rank.get(int(nbr), {})
            chosen: dict[str, np.ndarray] = {}
            left = int(need)
            for et in base_order:
                if left <= 0:
                    break
                arr = per.get(et)
                if arr is None or arr.size == 0:
                    continue

                lids = arr[:, 0]; dlt = arr[:, 1]
                if threshold == 0:
                    m = (dlt < 0)
                    if not np.any(m): 
                        continue
                    lids = lids[m]
                    # no sorting needed; keep stable (as produced by cached deltas)
                else:
                    m = (dlt <= int(threshold))
                    if not np.any(m):
                        continue
                    # already ordered “best first” from builder; no extra sort here
                    lids = lids[m]

                # mask out already-picked lids
                pm = picked_mask_by_et[et]
                if pm.size:
                    lids = lids[(lids >= 0) & (lids < pm.size)]
                    if lids.size == 0:
                        continue
                    lids = lids[~pm[lids]]

                if lids.size == 0:
                    continue

                # take in pre-sorted order (Δ asc, lid asc), up to what's left
                take = lids[:left]
                if take.size:
                    picked_mask_by_et[et][take] = True
                    gids = eidxs_np[et][take]
                    chosen[et] = gids.astype(np.int64, copy=False)
                    left -= int(take.size)
                    if verbose:
                        print(f"[itc4.pick] rank={rank} -> nbr={int(nbr)} et={et} "
                            f"take={int(take.size)} left={left}")
            return chosen

        # 1) build relocation diff from this rank
        eidxs_diff: dict[int, dict[str, np.ndarray]] = {}
        out_total = 0
        by_nbr_total = {}

        for nbr in range(R):
            cap = int(M[rank, int(nbr)])
            if cap <= 0:
                continue
            need = int(np.ceil(cap * float(scale)))
            per = _pick_for_neighbor(int(nbr), need,
                                    deltas_by_rank=deltas_by_rank,
                                    threshold=int(threshold),
                                    base_order=base_order,
                                    verbose=verbose)
            if per:
                eidxs_diff[int(nbr)] = per
            tot = sum(len(g) for g in per.values())
            by_nbr_total[int(nbr)] = int(tot)
            out_total += int(tot)
            assert tot <= need <= cap, f"[itc4.pass] rank={rank} nbr={nbr} tot={tot} need={need} cap={cap}"

        if verbose:
            print(f"[itc4.pass] rank={rank} thr={int(threshold)} scale={float(scale):.3f} "
                f"out_total={int(out_total)} by_nbr_total={by_nbr_total}")

        # 2) relocate
        self.relocate_i_to_j(eidxs_diff)
        self._accept_j_into_i()

        return int(out_total) if return_count else None

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
        comm, rank, root = get_comm_rank_root('world')

        # 0) candidate from current
        self._reset_j_with_i()

        # 1) deltas from *_i* using face-vertex unions
        deltas = self.compute_mpi_face_delta_from_vertices()

        # 2) propose plan (only chooser_rank proposes; others send {})
        eidxs_diff = self.build_moves_from_delta(deltas, target_moves) if rank == chooser_rank else {}

        # 3) build _j from plan
        self.relocate_i_to_j(eidxs_diff)
        self._accept_j_into_i()

    def smooth_until_stagnates(self, *, max_iters: int = 20,
                            patience: int = 1,
                            min_change: int = 0,
                            threshold = 0,
                            target_counts: Optional[List[int]] = None,
                            restrict_src_dest: bool = False) -> list[int]:
        """
        Run diffuse_smoothing repeatedly. The stopping decision is collective:
        we allreduce the local planned-move counts so every rank takes the same
        number of iterations. Returns the GLOBAL per-iteration move counts.
        """
        comm, rank, root = get_comm_rank_root('world')
        history: list[int] = []
        stable = 0
        last = None

        for _ in range(int(max_iters)):
            moved_local = int(self.diffuse_smoothing(
                return_count=True,
                threshold=threshold,
                target_counts=target_counts,
                restrict_src_dest=restrict_src_dest
            ) or 0)

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

    def iterate_to_convergence(self, target_counts, mask, *, 
        etype_order: Optional[List[str]] = None, verbose = False, flowmat_relax=0.5, ):
        """
        One round controller (as requested):
        1) Plan from current counts (M0).
        2) Single "nudge" pass with threshold=2 using M0.
        3) Replan (M1).
        4) One strict pass with threshold=0 using M1 (loop written as range(comm.size), break after first).
        5) Final smoothing to stagnation (default threshold in existing smoother).

        Notes:
        - We recompute the plan after each phase (no residual-cap bookkeeping).
        - threshold=0 pass does not move anything that would increase edge cuts.
        - No "-1 polish" here by request.
        """
        comm, rank, root = get_comm_rank_root('world')

        def _cur_counts_total():
            my_count = int(sum(map(len, self.eidxs_i.values())))
            return list(comm.allgather(my_count))

        if rank == root: print(f"[itc4.iter]  TARGET={target_counts}")

        # cur0 = _cur_counts_total()
        # if rank == root: print(f"[itc4.REV] CURRENT={cur0}")
        # M0   = self.element_flow_plan(cur0, target_counts, mask)
        # self.diffuse_smoothing2(M0.T, threshold=2, etype_order=etype_order, scale=flowmat_relax, score_by='edge')

        # prefer tuples for immutable steps
        order_of_execution = ([(2, 'vertex')] + [(2, 'edge')] + [(0, 'edge')] * comm.size )
        
        for exec in order_of_execution:
            cur0 = _cur_counts_total()
            if rank == root: print(f"[itc4.iter] CURRENT={cur0}")
            M0   = self.element_flow_plan(cur0, target_counts, mask)
            self.diffuse_smoothing2(M0, threshold=exec[0], etype_order=etype_order, 
                                    scale=flowmat_relax, score_by=exec[1])
            self.smooth_until_stagnates()


        # ... existing code ...
        self._prof_print_and_reset()   # <--- prints summary + resets counters


    # -----------------
    # Partition info 
    # -----------------

    def _partition_stats(self, mesh):
        """Gather per-rank etype counts and symmetric MPI-face matrix F (i<->j)."""
        comm, rank, root = get_comm_rank_root('world')
        etypes = tuple(sorted(mesh.etypes))
        R = comm.size

        # local
        loc_counts = {et: len(mesh.eidxs.get(et, ())) for et in etypes}
        loc_ifaces = {int(n): len(f) for n, f in mesh.con_p.items()}

        all_counts = comm.gather(loc_counts, root=root)
        all_ifaces = comm.gather(loc_ifaces, root=root)
        if rank != root:
            return None

        cnt = np.zeros((R, len(etypes)), np.int64)
        for r, d in enumerate(all_counts):
            for j, et in enumerate(etypes):
                cnt[r, j] = d.get(et, 0)

        A = np.zeros((R, R), np.int64)
        for i, d in enumerate(all_ifaces):
            for j, c in d.items():
                A[i, j] = int(c)
        F = np.minimum(A, A.T)  # symmetric faces
        return etypes, cnt, F

    @staticmethod
    def info(mesh):
        comm, rank, root = get_comm_rank_root('world')
        ps = _MetaMesh._partition_stats(_MetaMesh, mesh)  # call through class
        if rank != root:
            return
        etypes, cnt, F = ps
        R = cnt.shape[0]
        pairs = [(i, j) for i in range(R) for j in range(i + 1, R)]
        headers = (['etype'] + [f'r{r}' for r in range(R)]
                + [f'i{i}-{j}' for (i, j) in pairs] + ['pairs', 'faces'])
        rows = [[et] + [f'{int(cnt[r, j]):,}' for r in range(R)] + ['']*len(pairs) + ['', '']
                for j, et in enumerate(etypes)]
        pair_vals = [int(F[i, j]) for (i, j) in pairs]
        rows.append(['mpi_faces'] + ['']*R + [f'{v:,}' for v in pair_vals]
                    + [f'{sum(v > 0 for v in pair_vals):,}', f'{sum(pair_vals):,}'])
        print(tabulate(rows, headers=headers, tablefmt='github',
                    colalign=('left', *('right',)*(len(headers)-1))))

    @staticmethod
    def info_to_csv(mesh, *, tcurr: float, csv_path: str = 'lb_elem_dist.csv'):
        comm, rank, root = get_comm_rank_root('world')
        ps = _MetaMesh._partition_stats(_MetaMesh, mesh)
        if rank != root:
            return
        etypes, cnt, F = ps
        R, E = cnt.shape
        pairs = [(i, j) for i in range(R) for j in range(i + 1, R)]
        pair_vals = [int(F[i, j]) for (i, j) in pairs]
        pairs_nz  = sum(v > 0 for v in pair_vals)
        faces_sum = sum(pair_vals)

        if not os.path.exists(csv_path):
            cols = ['tcurr'] + [f'{r}-{et}' for r in range(R) for et in etypes] \
                + [f'{r}-total' for r in range(R)] \
                + [f'i{i}-{j}' for (i, j) in pairs] + ['pairs', 'faces']
            with open(csv_path, 'w') as f: f.write(','.join(cols) + '\n')
            print(f"[info_to_csv] header_written path='{csv_path}' R={R} etypes={list(etypes)} "
                f"npairs={len(pairs)} ncols={len(cols)}")

        row = [f"{tcurr:.6f}"]
        for r in range(R):
            vals = cnt[r].astype(int).tolist()
            row += list(map(str, vals)) + [str(sum(vals))]
        row += list(map(str, pair_vals)) + [str(pairs_nz), str(faces_sum)]
        with open(csv_path, 'a') as f: f.write(','.join(row) + '\n')
        print(f"[info_to_csv] row_append tcurr={tcurr:.6f} counts={R*(E+1)} pairs={len(pairs)} faces_sum={faces_sum}")

    # -----------------
    # Partition Writer
    # -----------------


    def init_part_pyfrs(self, intg, *, fields=('part_id', 'is_curved'), async_timeout=0.0):
        """
        Initialise a NativeWriter to dump *partition* snapshots only.
        Data layout mimics the standard writer: (ne_local, nvars, nupts).
        fields:
            - 'part_id'  : MPI rank (int) replicated across nupts
            - 'is_curved': {0,1} replicated across nupts (if available; else zeros)
        """
        # ---- MPI + basic config ----
        comm, rank, root = get_comm_rank_root('world')
        basedir  = intg.cfg.get('mesh', 'basedir', '.')
        basename = intg.cfg.get('mesh', 'basename')

        # ---- resolve local element counts per etype ----
        mesh = intg.system.meshes['compute']
        etypes = tuple(sorted(mesh.eidxs))
        erdata = {et: np.arange(len(mesh.eidxs.get(et, ())), dtype=np.int64) for et in etypes}

        # ---- shapes per etype (use real nupts so VTU exporters remain happy) ----
        # nvars = number of partition fields we're going to write
        nvars = len(fields)
        nupts = getattr(intg.system, 'nupts')  # dict: etype -> nupts
        ershapes = {et: (nvars, int(nupts[et])) for et in erdata if len(erdata[et])}

        # ---- construct the writer (prefix='partition') ----
        self._writer = NativeWriter.from_integrator(
            intg, basedir, basename, 'partition', fpdtype=np.float64
        )
        self._writer.set_shapes_eidxs(ershapes, erdata)

        # ---- book-keeping for subsequent write calls ----
        self._part_fields = tuple(fields)
        self._part_async_timeout = float(async_timeout)
        self._part_nvars = int(nvars)
        self.nupts = {et: int(nupts[et]) for et in etypes}

        # ---- trace (one line per rank) ----
        counts = {et: int(len(erdata[et])) for et in etypes}
        print(f"[part.init] rank={rank} fields={list(self._part_fields)} nvars={self._part_nvars} "
            f"etypes={list(etypes)} counts={counts}")

    def write_to_pyfrs(self, intg, iter_round=1, relocation_round=1):
        """
        Write one HDF5 'partition' snapshot at current tcurr.
        Data per etype is shaped (ne_local, nvars, nupts) of float64.
        Field 0: part_id (MPI rank); Field 1: is_curved (if requested, else zeros).
        """
        comm, rank, root = get_comm_rank_root('world')
        t = float(intg.tcurr)

        # Sanity: we must have been initialised
        if not hasattr(self, '_writer'):
            raise RuntimeError("init_part_pyfrs must be called before write_to_pyfrs")

        mesh = intg.system.mesh
        etypes = tuple(sorted(mesh.eidxs))

        # Build data dict: etype -> (ne_local, nvars, nupts)
        data = {}
        want_curved = ('is_curved' in self._part_fields)

        # Try to fetch curved flags per etype if present; else zeros
        curved_map = {}
        for et in etypes:
            cf = getattr(mesh, 'spts_curved', {}).get(et, None)
            if cf is None:
                curved_map[et] = None
            else:
                # cf is expected shape (ne_local,) boolean/0/1; cast to int64
                curved_map[et] = np.asarray(cf, dtype=np.int64).ravel()

        for et in etypes:
            gids_local = mesh.eidxs.get(et, ())
            ne = int(len(gids_local))
            if ne == 0 or et not in self.nupts:
                continue

            nv = self._part_nvars
            npu = int(self.nupts[et])

            # Allocate (ne, nvars, nupts)
            arr = np.empty((ne, nv, npu), dtype=np.float64)

            # Field 0: part_id = rank, replicated over nupts
            arr[:, 0, :] = float(rank)

            # Field 1 (optional): is_curved
            if want_curved and nv >= 2:
                if curved_map[et] is not None and len(curved_map[et]) == ne:
                    arr[:, 1, :] = curved_map[et][:, None]
                else:
                    arr[:, 1, :] = 0.0

            # If more fields are ever added in the future, zero-fill the rest
            if nv > 2:
                arr[:, 2:, :] = 0.0

            data[et] = arr

        # ---- metadata (mirror WriterPlugin style) ----
        stats = Inifile()
        stats.set('data', 'fields', ','.join(self._part_fields))
        stats.set('data', 'prefix', 'partition')
        intg.collect_stats(stats)  # keep mesh/solver meta consistent

        if rank == root:
            metadata = {**intg.cfgmeta,
                        'stats': stats.tostr(),
                        'mesh-uuid': intg.mesh_uuid,
                        'partition/iter_round': np.array(str(iter_round), dtype='S'),
                        'partition/relocation_round': np.array(str(relocation_round), dtype='S')}
        else:
            metadata = None

        # ---- friendly trace ----
        loc_counts = {et: int(data[et].shape[0]) for et in data}
        print(f"[part.write] rank={rank} tcurr={t:.6f} iter={int(iter_round)} reloc={int(relocation_round)} "
            f"ne={loc_counts}")

        # ---- write (sync by default; async if _part_async_timeout>0) ----
        self._writer.write(data, t, metadata, timeout=self._part_async_timeout)

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
        for et in self.etypes:
            a0 = edict0.get(et, None)
            gids = self.send_idxs[et]
            pos = np.fromiter((self.src_pos[et].get(int(g), -1) for g in gids),
                              count=self.ssum[et], dtype=np.int64)
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
        for et in self.etypes:
            a0 = edict0.get(et, None)
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
