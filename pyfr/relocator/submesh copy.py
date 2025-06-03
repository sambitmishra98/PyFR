from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from pyfr.readers.native import _Mesh
from pyfr.mpiutil import AlltoallMixin, get_comm_rank_root
# from pyfr.relocator.submesharymanager import SubMeshAryManager, AryKey, AryEtDict

def checked(fn):
    """Run self.validate() after *every* successful call to fn(self, …)."""
    from functools import wraps
    @wraps(fn)
    def _wrap(self, *a, **k):
        try:
            out = fn(self, *a, **k)      # run original method
        except Exception:
            # a failed mutation should leave the object unchanged,
            # so run validation and re-raise for easier debugging
            self.validate()
            raise
        self.validate()                  # post-condition check
        return out
    return _wrap

@dataclass
class SubMesh:
    """
    Container for a single sub-partition in a parallel run:
    - eidxs, conn
    - A dictionary mapping str names (spts, spts_nodes, spts_curved) to arrays
    - minimal local operations (_move_smesh_elements, carveout_for_nrank, etc.)
    """
    etypes:     list[str]             = field(default_factory=list)
    eidxs_base: dict[str, np.ndarray] = field(default_factory=dict)
    eidxs:      dict[str, np.ndarray] = field(default_factory=dict)
    con:        dict[str, np.ndarray] = field(default_factory=dict)
    conn:       np.ndarray            = field(default_factory=lambda: np.zeros((0,7),dtype=np.int64))
    #arrays:     dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    # Provide class-level or static mapping from i->etype, etype->i
    e_to_i = {"tri": 0, "quad": 1, "tet": 2, "hex": 3, "pri": 4, "pyr": 5}
    i_to_e = {v: k for k, v in e_to_i.items()}

    #def register_array(self, name: str, edict: dict[str, np.ndarray]):
    #    self.arrays[name] = edict

    def __post_init__(self):
        self.validate()
        object.__setattr__(self, "_locked", True)

    @classmethod
    def from_native_mesh(cls, 
                         mesh: _Mesh, 
                         etypes: list[str],
                         bc_map: dict[str, int],
    #                     arrays_preprocessed: dict[str, np.ndarray],
                         ):
        """
        Build a sub-mesh that initially contains all rank-local elements

        Parameters
        ----------
        mesh
            The PyFR `_Mesh` as returned by `NativeReader`.
        rank, comm
            MPI housekeeping, needed for the MPI-face setup.
        bc_map
            ``{'wall': 0, 'inlet': 1, ...}`` – already agreed across ranks.
        """
        
        comm, rank, root = get_comm_rank_root()
        
        eidxs = {et: mesh.eidxs[et].copy() for et in etypes}
        conn = cls._build_unified_connector(mesh, rank, comm, bc_map)

        sm = cls(etypes = etypes,
                 eidxs  = eidxs, 
                 eidxs_base = deepcopy(eidxs), 
                 con = cls._build_con(mesh, rank, comm, bc_map),
                 conn = conn,
                 #arrays = {name: arr for name, arr in arrays_preprocessed.items()}
                 )

        return sm
    
    @staticmethod
    def carve_idxs(rows: np.ndarray) -> dict[str, np.ndarray]:
        """
        Convert rows -> {etype_string: np.array of global EIDs}
        using self.i_to_e. (We can also make it an instance method.)
        """
        from collections import defaultdict
        d = defaultdict(set)
        for itype, eidx, *_ in rows:
            et_str = SubMesh.i_to_e[itype]
            d[et_str].add(eidx)
        return {et: np.array(sorted(vals), dtype=np.int64) for et, vals in d.items()}

    def recreate_mesh(self, base_mesh, rank: int):
        """
        Build a new _Mesh from the contents of this SubMesh.
        """
        mesh = _Mesh.__new__(_Mesh)

        # Copy over metadata from base_mesh
        mesh.fname    = base_mesh.fname
        mesh.raw      = base_mesh.raw
        mesh.ndims    = base_mesh.ndims
        mesh.subset   = False
        mesh.creator  = base_mesh.creator
        mesh.codec    = base_mesh.codec
        mesh.uuid     = base_mesh.uuid
        mesh.version  = base_mesh.version

        # Etypes, eidxs
        mesh.etypes = list(self.eidxs.keys())
        mesh.eidxs  = {et: arr.copy() for et, arr in self.eidxs.items()}

        # Get from arrays
        #mesh.spts        = {et: self.arrays['spts'       ][et].copy() for et in mesh.etypes}
        #mesh.spts_nodes  = {et: self.arrays['spts_nodes' ][et].copy() for et in mesh.etypes}
        #mesh.spts_curved = {et: self.arrays['spts_curved'][et].copy() for et in mesh.etypes}

        # Rebuild con, con_p, bcon from self.conn
        conl, conr = [], []
        mesh.con_p = {}
        mesh.bcon  = {}

        for row in self.conn:
            itype, eid, fno, nrank, rtype, relem, rface = row
            et = SubMesh.i_to_e[itype]

            if nrank == rank and rtype >= 0:
                # Internal
                conl.append((et, eid, fno))
                net = SubMesh.i_to_e[rtype]
                conr.append((net, relem, rface))
            elif nrank >= 0 and nrank != rank:
                mesh.con_p.setdefault(int(nrank), []).append((et, eid, fno))
            else:
                # boundary
                bc_face = int(rface)
                mesh.bcon.setdefault(bc_face, []).append((et, eid, fno))

        mesh.con = (conl, conr)
        return mesh

    @property
    def N_ei(self) -> dict[str, int]:
        return {et: len(arr) for et, arr in self.eidxs.items()}

    @property
    def N_i(self) -> int:
        return sum(self.N_ei.values())

    @property
    def N_c(self) -> dict[str, dict[int, int]]:
        """
        Count connector rows by MPI ranks and by boundary IDs.
        Return e.g. {'mpi': {r0: count, r1: ...}, 'bc': {bc0: count, ...}}.
        """
        mpi_counts, bc_counts = {}, {}
        for row in self.conn:
            nrank, nface = int(row[3]), int(row[6])
            if nrank >= 0:
                mpi_counts[nrank] = mpi_counts.get(nrank, 0) + 1
            else:
                bc_counts[nface] = bc_counts.get(nface, 0) + 1
        return {'mpi': mpi_counts, 'bc': bc_counts}

    def validate_conn_counts(self) -> None:
        """
        Check that connectivity row count matches #elements * faces.
        Raise AssertionError if mismatch is found.
        """
        face_map = {'tri': 3, 'quad': 4, 'tet': 4, 'hex': 6, 'pri': 5, 'pyr': 5}
        expected = sum(len(self.eidxs.get(et, [])) * face_map.get(et, 0)
                       for et in self.etypes)
        actual = self.conn.shape[0]
        if expected != actual:
            from pyfr.mpiutil import get_comm_rank_root
            _, rank, _ = get_comm_rank_root()
            details = []
            for et in self.etypes:
                code = self.e_to_i[et]
                for gid in self.eidxs.get(et, []):
                    c = int(((self.conn[:, 0] == code) & (self.conn[:, 1] == gid)).sum())
                    if c != face_map[et]:
                        details.append((et, gid, c, face_map[et]))
            msg = (f"[Rank {rank}] mismatch in connectivity rows: "
                   f"{actual} != {expected}, details={details}")
            raise AssertionError(msg)

    def _move_smesh_elements(self,
                            target_smesh: 'SubMesh',
                            new_eidxs_dict: dict[str, np.ndarray]) -> None:
        """
        Remove from `self.eidxs` those listed in `new_eidxs_dict`,
        for each etype in `target_smesh.etypes`. This is the old logic
        from _MetaMesh._move_smesh_elements, now local to submesh.
        """
        for etype in target_smesh.etypes:
            if etype in self.eidxs:
                old_e = self.eidxs[etype]
                remove_e = new_eidxs_dict[etype]
                self.eidxs[etype] = np.array([e for e in old_e if e not in remove_e],
                                             dtype=np.int64)

    @checked
    def carveout_for_nrank(self, nrank: int):
        """
        Move every element that has (at least) one face whose neighbour-rank
        equals *nrank*, together with the connectivity rows **originating from**
        those elements.

        Internal faces between two moved elements are copied twice (both
        orientations) because both left-hand elements belong to the set.
        Faces between a moved and a non-moved element become MPI interfaces:
            • the row that starts from the moved element is transferred,
            • the opposite-orientation row stays in the parent sub-mesh.
        """

        # ------------------------------------------------------------------ #
        # 1)  gather the elements whose neighbour-rank == nrank
        # ------------------------------------------------------------------ #
        moved_pairs = {
            (int(it), int(gid))
            for it, gid in self.conn[self.conn[:, 3] == nrank][:, :2]
        }
        if not moved_pairs:                # nothing touches that rank
            return SubMesh(
                etypes=self.etypes,
                eidxs={et: np.empty(0, np.int64) for et in self.etypes},
                conn=np.zeros((0, 7), np.int64),
                #spts={}, spts_nodes={}, spts_curved={}
            )

        # ------------------------------------------------------------------ #
        # 2)  select rows whose *left-hand* element is in moved_pairs
        # ------------------------------------------------------------------ #
        mask_left = np.zeros(self.conn.shape[0], dtype=bool)
        for it, gid in moved_pairs:
            mask_left |= (self.conn[:, 0] == it) & (self.conn[:, 1] == gid)

        rows_to_move = self.conn[mask_left]

        # ------------------------------------------------------------------ #
        # 3)  build eidx lists for the new sub-mesh (just the moved elements)
        # ------------------------------------------------------------------ #
        new_eidxs_raw = self.carve_idxs(rows_to_move)
        new_eidxs = {et: new_eidxs_raw.get(et, np.empty(0, np.int64))
                    for et in self.etypes}

        # --- NEW: map global-EID → row index inside each per-etype block ----
        gid2row = {et: {gid: i for i, gid in enumerate(self.eidxs[et])} for et in self.etypes}
        idx_map = {et: np.fromiter((gid2row[et][g] for g in new_eidxs[et]), dtype=np.intp) for et in self.etypes}
        new_con = SubMesh._slice_fields(self.con, self.etypes, idx_map)

        sm_new = SubMesh(etypes = self.etypes,
                         eidxs  = new_eidxs,
                         con= new_con,
                         conn   = rows_to_move.copy(),
                         )

        # 5) remove transferred rows & elements from parent ------------------
        self.conn = self.conn[~mask_left]
        self._move_smesh_elements(sm_new, new_eidxs)

        
        
        #for dname in ('spts', 'spts_nodes', 'spts_curved'):
        #    d = getattr(self, dname)
        #    for et in self.etypes:
        #        keep = np.setdiff1d(np.arange(len(d[et])), idx_map[et])
        #        d[et] = d[et][keep]

        return sm_new

    # put this right after SubMesh.carve_idxs(...)
    @staticmethod
    def _slice_fields(arrdict, etypes, idx_map):
        """
        Pick rows from every etype-array in *arrdict* according to *idx_map*.

        Parameters
        ----------
        arrdict : dict[str, np.ndarray]
            One of spts / spts_nodes / spts_curved from the **parent** smesh.
        etypes : list[str]
        idx_map : dict[str, np.ndarray]   # global-EID → row index inside arrdict[et]

        Returns
        -------
        sliced : dict[str, np.ndarray]    # rows belonging to the carve-out
        """
        out = {}
        for et in etypes:
            ids   = idx_map.get(et)
            a_src = arrdict[et]
            if ids is None or ids.size == 0:
                out[et] = np.empty_like(a_src, shape=(0,) + a_src.shape[1:])
            else:
                out[et] = a_src[ids]
        return out

    def validate(self, *, full: bool = True):
        """
        When *full* is False we skip the per-etype 'rows == #eidxs' test
        so a geometry-less shell passes.  Every public mutator that later
        inserts spts/spts_nodes/spts_curved must finish with validate(full=True)
        (automatically applied by the @checked decorator, see previous message).
        """
        # 1. no duplicate eids per etype
        for et, arr in self.eidxs.items():
            assert len(arr) == len(set(arr)), f"duplicate gid in eidxs[{et}]"

        # 2b. connectivity row-count (faces per element)
        if full:
            self.validate_conn_counts()

        # 3. every connector-LHS element is owned by the mesh
        own = {(self.e_to_i[et], int(g)) for et in self.etypes for g in self.eidxs[et]}
        assert all(tuple(row[:2]) in own for row in self.conn), \
            "conn row refers to element not in sub-mesh"
            
        # 4. neighbour columns are well-formed
        bad = np.where((self.conn[:,3] >= 0) & (self.conn[:,4] < 0))[0]
        assert bad.size == 0, "`nrank>=0` rows with missing neighbour info"            
            
    def mpi_pair_check(self, owner_rank: int, partner_rows=None):
        """
        Cheap runtime check: every row pointing at another rank must have
        a twin row on that partner rank that points back here.
        Pass *partner_rows* as the partner’s conn to avoid extra comms.
        """
        if partner_rows is None:
            return  # only callable when both sides are in memory
        for it,eid,fc,nr,nt,nid,nfc in self.conn:
            if nr < 0 or nr == owner_rank:
                continue
            key = (nt,nid,nfc, owner_rank, it,eid,fc)
            assert (partner_rows == key).all(axis=1).any(), f"missing MPI twin {key}"

    @staticmethod
    def _build_unified_connector(mesh: _Mesh, rank, comm, bc_map) -> np.ndarray:
        """
           The `conn` array in each smesh is a unified connectivity array 
           It handles both left and right sides of every single interface
           of every element in smeshes. Each row is a 7-tuple:

              [itype, eid, face, nrank, netype, neid, nface]

            All connectors are encoded for each of MPI transfers:
                - integer-coded etype (instead of str), 
                - global element ID (instead of local), and
                - face index.

           Where:
               - itype, eid, face: Local element's interface
               - nrank: The neighbor rank.  
                        If nrank == self.rank
                            it's an internal face; 
                        Else, if nrank == -1, 
                            it's a boundary face (boundary condition set later)
                        Else
                            It's an external-directed face, towards an smesh
               - netype, neid, nface: The neighboring element (or boundary ID).  
                 For internal or MPI faces,
                 these columns specify the other element's etype, global ID, and face index.

           This design unifies internal, MPI, and boundary data into a single structure,
           making it straightforward to slice or move selected faces during submesh 'carveouts'
           and parallel exchanges.  Future extensions can add columns to hold boundary
           metadata or geometric attributes, without changing the overall table layout.
        """
        def face_arr(triples):
            a = np.zeros((len(triples), 3), np.int64)
            for i, (et, lid, f) in enumerate(triples):
                a[i] = (SubMesh.e_to_i[et], mesh.eidxs[et][lid], f)
            return a

        # 1. internal faces (two orientations)
        cL, cR = face_arr(mesh.con[0]), face_arr(mesh.con[1])
        internal = np.vstack([
            np.column_stack([cL, np.full((len(cL), 1), rank), cR]),
            np.column_stack([cR, np.full((len(cR), 1), rank), cL]),
        ]) if cL.size else np.empty((0, 7), np.int64)

        # 2. MPI faces (rows missing [nitype, ngid, nface] → filled later)
        mpi_rows = []
        for nrank, triples in mesh.con_p.items():
            if not triples:
                continue
            left = face_arr(triples)
            fill = np.tile(np.array([nrank, -1, -1, -1], np.int64), (len(triples), 1))
            mpi_rows.append(np.hstack([left, fill]))
        mpi_rows = np.vstack(mpi_rows) if mpi_rows else np.empty((0, 7), np.int64)

        # alltoallcv: fill columns [4,5,6] for MPI faces
        mpi_rows = SubMesh._alltoallcv_con_p_setup(comm, mpi_rows)

        # 3. boundary faces  (nrank = -1, last column stores bc-id from mesh.bcon key)
        # --- boundary faces --------------------------------------------------------
        bc_rows = []
        for bcname, triples in mesh.bcon.items():          # <- keep the *name*
            if not triples:
                continue
            left   = face_arr(triples)
            bc_id  = bc_map[bcname]                  # <- translate to int
            fill   = np.tile(np.array([-1, -1, -1, bc_id], np.int64),
                            (len(triples), 1))
            bc_rows.append(np.hstack([left, fill]))
        bc_rows = np.vstack(bc_rows) if bc_rows else np.empty((0, 7), np.int64)

        # 4. concatenate & sort for deterministic ordering
        unified = np.vstack([internal, mpi_rows, bc_rows])
        if unified.size:
            order = np.lexsort((unified[:, 2], unified[:, 1], unified[:, 0]))
            unified = unified[order]

        return unified

    # ------------------------------------------------------------------ #
    # New helper: build per-etype 3-D connectivity arrays (con)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_con(mesh: _Mesh, rank, comm, bc_map) -> dict[str, np.ndarray]:
        """
        Return
        -------
        con : dict[str, np.ndarray]
            One entry per etype, each of shape
                (Nelements(etype), Nfaces(etype), 4)
            storing (nrank, netype, nelem, nface).

        Conventions
        -----------
        • Interior face nrank == rank
        • MPI face      nrank >= 0 and != rank
        • Boundary face nrank == -1,
                        netype = -1, nelem = -1, nface = bc_id
        """

        # 1. face-count lookup
        nfaces = {'tri': 3, 'quad': 4, 'tet': 4,
                  'hex': 6, 'pri': 5, 'pyr': 5}

        # 2. allocate output arrays, fill with -1 for easier sanity checks
        con = {et: np.full((len(mesh.eidxs[et]), nfaces[et], 4), -2, dtype=np.int64)
               for et in mesh.etypes}

        # 3. build id→row-index maps  (global-eid → axis-0 index)
        gid2row = {et: {int(g): i for i, g in enumerate(mesh.eidxs[et])}
                   for et in mesh.etypes}

        # 4. obtain the flat 7-column table (re-use existing routine)
        rows = SubMesh._build_unified_connector(mesh, rank, comm, bc_map)

        # 5. fill the 3-D arrays
        for itype, gid, face, nr, nt, nid, nfc in rows:
            et  = SubMesh.i_to_e[int(itype)]
            r   = gid2row[et][int(gid)]
            con[et][r, int(face)] = (int(nr), int(nt), int(nid), int(nfc))

        return con



    @staticmethod
    def _alltoallcv_con_p_setup(comm, conn: np.ndarray) -> np.ndarray:
        """
        For MPI faces, fill columns [4,5,6] by exchanging local references.
        """

        if conn.size == 0:
            return conn
        mask = (conn[:, 6] == -1)
        svals = np.array(conn[mask][:, :3], dtype=np.int64)
        scount = np.bincount(conn[mask][:, 3], minlength=comm.size)
        rvals, _ = AlltoallMixin()._alltoallcv(comm, svals, scount)
        conn[mask, 4:] = rvals
        return conn
            
            