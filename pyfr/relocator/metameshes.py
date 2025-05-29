# Location: pyfr/relocator/metameshes.py

from typing import Callable   # <-- add near the top of the file

from copy import deepcopy
from dataclasses import dataclass, field

from pprint import pprint
import numpy as np
from mpi4py import MPI

from pyfr.mpiutil import get_comm_rank_root, AlltoallMixin
from pyfr.readers.native import _Mesh, NativeReader
from pyfr.relocator.utils import crpprint

class MeshExchanger:
    """
        Perform all MPI AlltoallMixin data exchanges.
    """
    def __init__(self, comm, transporter):
        self.comm = comm
        self.transporter = transporter
        self.alltoall_data = {}

    def exchange_eidxs(self, packed: dict[str, dict[str, np.ndarray]]) -> dict[str, list[np.ndarray]]:
        """Exchange element-index arrays across ranks."""
        recv = {}
        for et, info in packed.items():
            rvals, (rcount, rdisps) = self.transporter._alltoallcv(
                self.comm, info['svals'], info['scount'])
            # split per-rank
            recv[et] = [rvals[rdisps[i]:rdisps[i]+rcount[i]] for i in range(self.comm.size)]
            # store metadata for subsequent array exchanges
            self.alltoall_data[et] = {
                'scount': info['scount'],
                'sdisp': self.transporter._count_to_disp(info['scount']),
                'rcount': rcount,
                'rdisps': rdisps
            }
        return recv

    def exchange_conn(self, packed_conn: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
        """Exchange connector arrays for each destination rank."""
        rvals, (rcount, rdisps) = self.transporter._alltoallcv(
            self.comm, packed_conn['svals'], packed_conn['scount'])
        conn_recv = {}
        for i in range(self.comm.size):
            if rcount[i] > 0:
                conn_recv[i] = rvals[rdisps[i]:rdisps[i]+rcount[i]].reshape(-1, 7)
            else:
                conn_recv[i] = np.zeros((0, 7), dtype=np.int64)
        return conn_recv

    def exchange_ary(self, packed: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Exchange array-type data using previously stored displacements."""
        recv = {}
        for et, info in packed.items():
            meta = self.alltoall_data[et]
            svals = info['svals']
            # allocate receive buffer
            shape = (meta['rcount'].sum(),) + svals.shape[1:]
            buf = np.empty(shape, dtype=svals.dtype)
            # perform all-to-all-v
            self.transporter._alltoallv(self.comm,
                                        (svals, (info['scount'], meta['sdisp'])),
                                        (buf,   (meta['rcount'], meta['rdisps']))
                                       )
            recv[et] = buf
        return recv
    
def mirror_conn(conn: np.ndarray) -> np.ndarray:
    """
    Mirror connector columns for symmetric interfaces.
        Legend: 
            0: element type (own)
            1: element ID   (own)
            2: face number  (own)
            3: rank (or) -1 for boundary
            4: element type (neighbour)
            5: element ID   (neighbour)
            6: face number  (neighbour)
    
    """
    return conn[:, [4, 5, 6, 3, 0, 1, 2]]

def carve_idxs(rows: np.ndarray, i_to_e: dict[int, str]) -> dict[str, np.ndarray]:
    """Extract element indices per etype from connector rows."""
    from collections import defaultdict
    d = defaultdict(set)
    for itype, eidx, *_ in rows:
        d[i_to_e[itype]].add(eidx)
    return {et: np.array(sorted(vals), dtype=np.int64) for et, vals in d.items()}
 
def build_block(base: np.ndarray, fill: tuple[int, int, int, int]) -> np.ndarray:
    """Append fill columns to base array for connector blocks."""
    n = base.shape[0]
    return np.hstack([base, np.tile(fill, (n, 1))])

@dataclass
class _SubPartitionedMesh:
    """
    A sub-partitioned mesh carved out of mesh in a rank.

    etypes
        List of element-types present in global mesh.
    eidxs_base
        Mapping {etype: global element IDs} in base partition.
    eidxs
        Mapping {etype: global element IDs} in this submesh.
    conn
        Unified (Nfaces,7) connector array.
    spts, spts_nodes, spts_curved
        Per-etype point data.
    """
    etypes:      list[str]                         = field(default_factory=list)
    eidxs_base:  dict[str, np.ndarray[np.int64]]   = field(default_factory=dict)
    eidxs:       dict[str, np.ndarray[np.int64]]   = field(default_factory=dict)
    conn:        np.ndarray[np.int64]              = field(default_factory=lambda: np.zeros((0, 7), dtype=np.int64))
    spts:        dict[str, np.ndarray[np.float64]] = field(default_factory=dict)
    spts_nodes:  dict[str, np.ndarray[np.int64]]   = field(default_factory=dict)
    spts_curved: dict[str, np.ndarray[np.bool_]]   = field(default_factory=dict)

    @property
    def N_ei(self) -> dict[str,int]:
        return {et: len(e) for et,e in self.eidxs.items()}

    @property
    def N_i(self) -> int:
        return sum(self.N_ei.values())

    @property
    def N_c(self) -> dict[str,dict[int,int]]:
        mpi_counts, bc_counts = {},{}
        for row in self.conn:
            nrank = int(row[3]); nface = int(row[6])
            if nrank >= 0:
                mpi_counts[nrank] = mpi_counts.get(nrank,0)+1
            else:
                bc_counts[nface] = bc_counts.get(nface,0)+1
        return {'mpi':mpi_counts,'bc':bc_counts}

    def summary(self) -> str:
        elems = ','.join(f"{et}={len(ids)}" for et,ids in self.eidxs.items())
        nc = self.N_c
        mpis = ','.join(f"{r}:{c}" for r,c in nc['mpi'].items()) or 'none'
        bcs  = ','.join(f"{b}:{c}" for b,c in nc['bc'].items()) or 'none'
        return f"SubMesh | elems[{elems}] | mpi-ifaces[{mpis}] | bc-ifaces[{bcs}] | curved-spts={self.nspts_curved}"

    def validate_conn_counts(self) -> None:
        face_map = {'tri':3,'quad':4,'tet':4,'hex':6,'pri':5,'pyr':5}
        exp = sum(len(self.eidxs.get(et,[]))*face_map.get(et,0) for et in self.etypes)
        act = self.conn.shape[0]
        if exp!=act:
            errs=[]
            for et in self.etypes:
                code = _MetaMesh.e_to_i[et]
                for gid in self.eidxs.get(et,[]):
                    cnt=int(((self.conn[:,0]==code)&(self.conn[:,1]==gid)).sum())
                    if cnt!=face_map[et]: errs.append((et,gid,cnt,face_map[et]))
            raise AssertionError(f"conn count mismatch {act}!={exp}, details {errs}")

    @property
    def nspts_curved(self) -> int:
        return sum(np.sum(self.spts_curved.get(et,[])) for et in self.etypes)

class _MetaMesh:
    """
    Initialise, modify and transfer _SubPartitionedMesh instances.
        mesh: a weak-link reference to original mesh object.
        comm: towards smesh re-distribution.
        smeshes: {nrank: smesh} mapping within each rank
    """

    e_to_i = {"tri": 0, "quad": 1, "tet": 2, "hex": 3, "pri": 4, "pyr": 5,}
    i_to_e = {v: k for k, v in e_to_i.items()}


    def __init__(self, mesh: _Mesh, comm: MPI.Comm):
        self.comm, self.rank, root = get_comm_rank_root()
        self.mesh = mesh

        # map submesh ID → [src_rank, dest_rank]
        self.src_dest = {self.rank: [self.rank, self.rank]}
        self.smeshes = {self.rank: None}

        self.transporter = AlltoallMixin()
        self.glmap = self.get_global_local_mapping()
        self.lgmap = {etype: {gid: i for i, gid in v.items()} for etype, v in self.glmap.items()}
        self.etypes = self.gather_etypes(self.mesh)
        self.bcnames = self.gather_bc_names(self.mesh)
        self.bc_map = {name: i for i, name in enumerate(self.bcnames)}

        unified_conn = self._build_unified_connector(self.mesh)
        self.smeshes[self.rank] = self._initialise_int_smesh(unified_conn)

    def export_input_mesh(self, fname='mesh.npz'):
        """
        Write out the original mesh self.mesh to an .npz file, encoding:
          - etypes
          - eidxs_<etype>
          - conl, conr
          - bcon_<name>
          - spts_<etype>, spts_nodes_<etype>, spts_curved_<etype>
        Only valid when comm.size == 1.
        """
        # Only export for single‐rank runs
        if self.comm.size != 1:
            raise RuntimeError(f"Cannot export input mesh when comm.size={self.comm.size}")

        import numpy as _np

        mesh = self.mesh
        etypes  = mesh.etypes
        et2code = {et: i for i, et in enumerate(etypes)}

        # Gather arrays to save
        save_dict = {
            'etypes': _np.array(etypes, dtype='<U10')
        }

        # Element indices per type
        for et in etypes:
            save_dict[f'eidxs_{et}'] = mesh.eidxs[et]

        # Internal connectivity
        conl = _np.array([[et2code[et], eid, face]
                          for et, eid, face in mesh.con[0]],
                         dtype=_np.int64)
        conr = _np.array([[et2code[et], eid, face]
                          for et, eid, face in mesh.con[1]],
                         dtype=_np.int64)
        save_dict['conl'] = conl
        save_dict['conr'] = conr

        # Boundary faces (no MPI faces since size=1)
        for bcname, trips in mesh.bcon.items():
            arr = _np.array([[et2code[et], eid, face] for et, eid, face in trips],
                            dtype=_np.int64)
            save_dict[f'bcon_{bcname}'] = arr

        # Point‐data arrays
        for et in etypes:
            save_dict[f'spts_{et}']        = mesh.spts[et]
            save_dict[f'spts_nodes_{et}']  = mesh.spts_nodes[et]
            save_dict[f'spts_curved_{et}'] = mesh.spts_curved[et]

        # Write compressed NPZ
        _np.savez_compressed(fname, **save_dict)


    def export_submeshes(self, fname='smeshes.npz'):
        """
        Write out every _SubPartitionedMesh in self.smeshes to an .npz file,
        encoding for each submesh:
          - eidxs_<et> per element type
          - conn             the (N,7) connector array
          - spts_<et>, spts_nodes_<et>, spts_curved_<et>
        Only valid when comm.size == 1.
        """
        if self.comm.size != 1:
            raise RuntimeError(f"Cannot export submeshes when comm.size={self.comm.size}")

        import numpy as _np

        save = {}
        # Record the list of submesh IDs
        smids = list(self.smeshes.keys())
        save['smesh_ids'] = _np.array(smids, dtype=_np.int64)

        # Loop over each submesh
        for smid, sm in self.smeshes.items():
            pref = f's{smid}_'

            # Element indices
            for et in self.etypes:
                arr = sm.eidxs.get(et, _np.empty((0,), dtype=_np.int64))
                save[f'{pref}eidxs_{et}'] = arr

            # Connector
            save[f'{pref}conn'] = sm.conn

            # Point‐data
            for et in self.etypes:
                save[f'{pref}spts_{et}']        = sm.spts.get(et,        _np.empty((0,)))
                save[f'{pref}spts_nodes_{et}']  = sm.spts_nodes.get(et,  _np.empty((0,)))
                save[f'{pref}spts_curved_{et}'] = sm.spts_curved.get(et, _np.empty((0,)))

        # Write compressed NPZ
        _np.savez_compressed(fname, **save)

    def _complete_con_p_setup(self, conn: np.ndarray[np.int64]) -> np.ndarray[np.int64]:

        mpi_mask = conn[:, 6] == -1
        svals = np.array(conn[mpi_mask][:, :3], dtype=np.int64)
        scount = np.bincount(conn[mpi_mask][:, 3], minlength=self.comm.size)

        rvals, _ = self.transporter._alltoallcv(self.comm, svals, scount)
        conn[mpi_mask, 4:] = rvals

        return conn

    def get_global_local_mapping(self):
        glmap = {}
        for etype in self.mesh.etypes:
            g_ids = self.mesh.eidxs[etype]
            glmap[etype] = {gid: i for i, gid in enumerate(g_ids)}
        return glmap

    def gather_etypes(self, mesh: _Mesh):
        return sorted(set.union(*self.comm.allgather(set(mesh.etypes))))

    def gather_bc_names(self, mesh: _Mesh):
        return sorted(set.union(*self.comm.allgather(set(mesh.bcon.keys()))))

    def con_e_list_to_face_i_array(self, lst):
        """
        Convert a Python list of (etype_string, local_index, face_no)
        to a NumPy array of shape (N, 3).
        """
        if not lst:
            return np.zeros((0, 3), dtype=np.int64)
        arr = np.zeros((len(lst), 3), dtype=np.int64)
        for i, (etstr, loc_eid, face_no) in enumerate(lst):
            itype = self.e_to_i[etstr]
            global_eid = self.lgmap[etstr][loc_eid]
            arr[i] = (itype, global_eid, face_no)
        return arr

    def _initialise_int_smesh(self, unified_conn):
        """
        Create a single _SubPartitionedMesh from all local elements
        in base_mesh.  This is a trivial 'no sub-partition' approach
        to get started; the real logic might be more nuanced.
        """
        eidxs = {etype: self.mesh.eidxs[etype].copy() 
                    if etype in self.mesh.etypes
                    else np.array([], dtype=np.int64)
                 for etype in self.etypes
                    }

        return _SubPartitionedMesh(etypes=self.etypes, eidxs=eidxs,
            eidxs_base=deepcopy(eidxs),
            conn=unified_conn,
            spts=self.preprocess_ary("spts", edim=1),
            spts_nodes=self.preprocess_ary("spts_nodes"),
            spts_curved=self.preprocess_ary("spts_curved")
        )

    def _build_unified_connector(self, mesh: _Mesh) -> np.ndarray:
        """
        Build the unified (N,7) connector array.
        Rows: [itype, elem, face, nrank, nitype, nelem, nface]
        """
        # Internal faces
        conl = self.con_e_list_to_face_i_array(mesh.con[0])
        conr = self.con_e_list_to_face_i_array(mesh.con[1])
        # Internal faces
        internal_lr = np.column_stack([
            conl[:, 0], conl[:, 1], conl[:, 2],
            np.full(conl.shape[0], self.rank, dtype=np.int64),
            conr[:, 0], conr[:, 1], conr[:, 2]
        ])

        internal_rl = np.column_stack([
            conr[:, 0], conr[:, 1], conr[:, 2],
            np.full(conr.shape[0], self.rank, dtype=np.int64),
            conl[:, 0], conl[:, 1], conl[:, 2]
        ])

        internal = np.vstack((internal_lr, internal_rl))

        # MPI faces
        mpi_blocks = [
            self._build_section(trips, (nrank, -1, -1, -1))
            for nrank, trips in mesh.con_p.items() if trips
        ]
        # MPI faces
        mpi = np.vstack(mpi_blocks) if mpi_blocks else np.zeros((0, 7), dtype=np.int64)
        mpi = self._complete_con_p_setup(mpi)

        # Boundary faces
        bc_blocks = [
            self._build_section(trips, (-1, -1, -1, self.bc_map[name]))
            for name, trips in mesh.bcon.items() if trips
        ]
        # Boundary faces
        bc = np.vstack(bc_blocks) if bc_blocks else np.zeros((0, 7), dtype=np.int64)

        # Combine all faces and then sort by (itype, elem, face)
        unified = np.vstack((internal, mpi, bc))
        # lexsort: last key first → face, elem, itype
        order = np.lexsort((unified[:, 2], unified[:, 1], unified[:, 0]))
        return unified[order]

    def carveout_mpi_smesh(self, int_smesh: '_SubPartitionedMesh', nrank: int):
        """Carve out submesh for neighbor rank `nrank`."""
        # Extract rows corresponding to nrank
        nrank_interfaces = int_smesh.conn[int_smesh.conn[:, 3] == nrank]

        # Derive new eidxs
        from pyfr.relocator.metameshes3 import carve_idxs
        new_eidxs_dict = carve_idxs(nrank_interfaces, self.i_to_e)

        # Create a new submesh
        self.smeshes[nrank] = _SubPartitionedMesh(
            etypes=list(new_eidxs_dict.keys()),
            eidxs=new_eidxs_dict
        )

        # Record in src_dest
        self.src_dest[nrank] = [self.rank, nrank]

        # Move elements from int_smesh -> new smesh
        self.move_smesh_elements(int_smesh, self.smeshes[nrank], new_eidxs_dict)

        # ---> CHANGED: pass nrank into update_smesh_connectivity
        self.update_smesh_connectivity(int_smesh, self.smeshes[nrank], nrank)
        # <-----

    def grow_smesh(self,  src_smesh: _SubPartitionedMesh, 
                         dest_smesh: _SubPartitionedMesh):    
        
        srank_elements           = dest_smesh.conn[:][:, 0:2]
        srank_interface_elements = dest_smesh.conn[dest_smesh.conn[:, 3] == self.rank][:, 4:6]

        new_eidxs = {etype: set() for etype in self.etypes}
        for (itype, eidx) in srank_elements:
            new_eidxs[self.i_to_e[itype]].add(eidx)
        for (itype, eidx) in srank_interface_elements:
            new_eidxs[self.i_to_e[itype]].add(eidx)

        new_eidxs_dict = {etype: np.array(sorted(eids), dtype=np.int64) for etype, eids in new_eidxs.items()}        

        dest_smesh.eidxs = new_eidxs_dict

        self.move_smesh_elements(src_smesh, dest_smesh, new_eidxs_dict)
        self.update_smesh_connectivity(src_smesh, dest_smesh)        

    def move_smesh_elements(self, int_smesh: _SubPartitionedMesh, 
                                       mpi_smesh: _SubPartitionedMesh, 
                                  new_eidxs_dict: dict[str, np.ndarray]):

        # Remove eidxs from int_smesh
        for etype in mpi_smesh.etypes:
            if etype in int_smesh.eidxs:
                int_smesh.eidxs[etype] = np.array([e for e in int_smesh.eidxs[etype] if e not in new_eidxs_dict[etype]],dtype=np.int64)
            
    def update_smesh_connectivity(self, int_smesh: '_SubPartitionedMesh',
                                  mpi_smesh: '_SubPartitionedMesh',
                                  nrank: int):
        """
        Move and orient interface rows between int_smesh and mpi_smesh,
        using nrank as the submesh's rank.
        """
        conn = int_smesh.conn

        # select rows where neighbor-rank is nrank on either side
        sel_mask = (conn[:, 3] == nrank) | (conn[:, 4] == nrank)
        selected = conn[sel_mask]
        remaining = conn[~sel_mask]

        rows = selected.copy()
        # flip rows where column-3 != nrank
        flip = rows[:, 3] != nrank
        rows[flip] = mirror_conn(rows[flip])

        # store oriented rows in mpi_smesh, remainder stays in int_smesh
        mpi_smesh.conn = rows
        int_smesh.conn = remaining

    def pack_eidxs(self) -> dict[str, dict[str, np.ndarray]]:
        """Pack eidxs ∀smesh into dict of scoutn and svals for MPI exchange."""

        info = {}
        for et in self.etypes:
            local_arrays = [
                self.smeshes[r].eidxs.get(et, np.array([], dtype=np.int64)) if r in self.smeshes and r != self.rank
                                         else np.array([], dtype=np.int64)
                for r in range(self.comm.size) 
            ]

            info[et]  = {'svals': np.concatenate(local_arrays, dtype=np.int64), 
                      'scount': np.array([arr.size for arr in local_arrays], dtype=np.int64)
                     }

        return info

    def exchange_eidxs(self, packd_eidxs: dict) -> None:

        self.alltoall_data = {}

        eidxs_recv = {}
        for et, info in packd_eidxs.items():
            rvals, (rcount, rdisps) = self.transporter._alltoallcv(self.comm, info['svals'], info['scount'])
            eidxs_recv[et] = [ rvals[rdisps[i] : rdisps[i] + rcount[i]] for i in range(self.comm.size) ]
            self.alltoall_data[et] = {
                'scount': info['scount'], 
                'sdisp': self.transporter._count_to_disp(info['scount']),
                'rcount': rcount, 
                'rdisps': rdisps
                }
        return eidxs_recv

    def pack_conn(self):
        """Pack each **destination** submesh’s conn array for MPI exchange."""
        svals_list = []
        scount = []

        for dest in range(self.comm.size):
            if dest != self.rank and dest in self.smeshes:
                # pull *that* submesh’s connector rows
                arr = self.smeshes[dest].conn
            else:
                arr = np.zeros((0, 7), dtype=np.int64)

            svals_list.append(arr)
            scount.append(arr.shape[0])

        svals = np.concatenate(svals_list, axis=0)
        scount = np.array(scount, dtype=np.int64)
        sdisp = self.transporter._count_to_disp(scount)

        return {'svals': svals, 'scount': scount, 'sdisp': sdisp}

    def exchange_conn(self, packd_conn: dict) -> dict[int, np.ndarray]:
        """
        Perform MPI all-to-all exchange of unified connector arrays.
        Returns a dictionary mapping each rank (0...comm.size-1) to its received connector array.
        """
        rvals, (rcount, rdisps) = self.transporter._alltoallcv(self.comm,
                                                                packd_conn['svals'],
                                                                packd_conn['scount'])
        conn_recv = {}
        for i in range(self.comm.size):
            if rcount[i] > 0:
                conn_recv[i] = rvals[rdisps[i] : rdisps[i] + rcount[i]].reshape(-1, 7)
            else:
                conn_recv[i] = np.zeros((0, 7), dtype=np.int64)
        return conn_recv

    def drop_empty_smeshes(self):
        """ Remove any smesh with N_i = 0. """
        for smid in [smid for smid, smesh in self.smeshes.items() if smesh.N_i == 0]:
            del self.smeshes[smid]

    @staticmethod
    def _preproc_edict(etypes: list[str],
                       edict_in: dict[str, np.ndarray],
                       *,
                       edim: int = 0) -> dict[str, np.ndarray]:
        """
        Ensure that for every etype in `etypes`, edict_out[etype] exists.
        Missing ones are filled with empty arrays matching the
        non‐empty ones’ dtype and shape.
        """
        from pyfr.mpiutil import get_comm_rank_root, mpi
        comm, rank, root = get_comm_rank_root()

        # shallow copy
        edict = {e: edict_in[e].copy() for e in edict_in}

        # first, figure out dtype+shape on any non‑empty entry
        # and note which etypes are actually missing
        missing = []
        candidate_dtype = None
        candidate_shape = None

        for et in etypes:
            arr = edict.get(et)
            if arr is None or arr.size == 0:
                missing.append(et)
            else:
                # if we need to move axis to front, do it here, but remember
                if edim != 0 and arr.ndim > edim:
                    arr = np.moveaxis(arr, edim, 0)
                edict[et] = arr
                candidate_dtype = arr.dtype
                candidate_shape = arr.shape

        # did *any* rank have to create an array?
        do_create = bool(missing)
        # reduce across ranks
        if comm.allreduce(do_create, op=mpi.MAX):
            # gather all dtypes/shapes so we pick a real one
            dtypes = comm.allgather(candidate_dtype)
            shapes = comm.allgather(candidate_shape)
            for dt, sh in zip(dtypes, shapes):
                if dt is not None and sh is not None:
                    candidate_dtype = dt
                    candidate_shape = sh
                    break
            # now create empty arrays for all missing
            for et in missing:
                edict[et] = np.empty((0,)+candidate_shape[1:], dtype=candidate_dtype)

        # finally, return in sorted‐by‐etype order
        return {et: edict[et] for et in sorted(etypes)}

    def preprocess_ary(self,
                       ary_name: str,
                       *,
                       edim: int = 0) -> None:
        """
        Call this in __post_init__ to ensure
        mesh.[ary_name] has an entry for every etype.
        edim=1 for 'spts' (so that we treat axis‑1 as the element axis),
        edim=0 for the rest.
        """
        # pick off the right dict
        arrdict = getattr(self.mesh, ary_name)

        # run the generic fill‐in routine
        return self._preproc_edict(self.etypes, arrdict, edim=edim)

    def move_ary(self, ary_name: str) -> None:
        """
        Move array data from the internal submesh to each destination submesh.
        Uses base-array indexing and explicit empty-case handling to avoid OOB.
        """
        base_smesh = self.smeshes[self.rank]
        base_dict  = getattr(base_smesh, ary_name)
        base_eidxs = base_smesh.eidxs_base  # global IDs → base-array ordering

        for dest_rank, sm in self.smeshes.items():
            if dest_rank == self.rank:
                continue

            new_dict = {}
            for et in self.etypes:
                arr_base = base_dict.get(et)
                dest_ids = sm.eidxs.get(et, np.array([], dtype=np.int64))

                # —— Empty cases first ——
                if arr_base is None or arr_base.size == 0 or dest_ids.size == 0:
                    # create an empty array with the same trailing shape & dtype
                    if arr_base is None:
                        new_arr = np.empty((0,), dtype=float)
                    else:
                        new_arr = np.empty((0,) + arr_base.shape[1:], dtype=arr_base.dtype)
                else:
                    # map global IDs → local positions in arr_base
                    idx_map = {gid: i for i, gid in enumerate(base_eidxs[et])}
                    # collect the local indices for these global IDs
                    local_idxs = [idx_map[gid] for gid in dest_ids]
                    new_arr = arr_base[local_idxs]

                new_dict[et] = new_arr

            setattr(sm, ary_name, new_dict)


    def pack_ary(self, ary_name: str) -> dict[str, dict[str, np.ndarray]]:
        """
        Revamped pack_ary: only pack data for submeshes that need exchange (dest != local rank).
        Assumes all arrays have element ID as first dimension.
        Returns mapping of each etype to:
          'svals': concatenated send buffer,
          'scount': per-destination element counts.
        """
        from pyfr.mpiutil import get_comm_rank_root
        import numpy as np

        comm, rank, _ = get_comm_rank_root()
        info: dict[str, dict[str, np.ndarray]] = {}

        for et in self.etypes:
            # Local template to infer trailing shape and dtype
            local_arr = getattr(self.smeshes[rank], ary_name).get(et)
            if local_arr is None:
                raise ValueError(f"Missing local {ary_name} for etype '{et}'")
            trailing_shape = local_arr.shape[1:]
            dtype = local_arr.dtype

            send_list = []
            counts = []

            # Only pack into ranks with existing submeshes (excluding self)
            for r in range(comm.size):
                if r != rank and r in self.smeshes:
                    arr = getattr(self.smeshes[r], ary_name).get(et)
                    if arr is None:
                        arr = np.empty((0,) + trailing_shape, dtype=dtype)
                else:
                    arr = np.empty((0,) + trailing_shape, dtype=dtype)

                send_list.append(arr)
                counts.append(arr.shape[0])

            svals = np.concatenate(send_list, axis=0)
            scount = np.array(counts, dtype=np.int64)

            info[et] = {'svals': svals, 'scount': scount}

        return info

    def exchange_ary(self, packd: dict[str, dict[str, np.ndarray]], ) -> dict[str, np.ndarray]:
        """
        Perform an all‑to‑all‑v exchange of the arrays packed by pack_ary.
        Must be called *after* exchange_eidxs (so that self.alltoall_data is set).

        Arguments:
          packd    the dict returned by pack_ary()
          leaddim  the axis along which elements were laid out
                   (1 for 'spts', 0 for the others).

        Returns:
          relocated: dict mapping each etype → received array
        """
        relocated: dict[str, np.ndarray] = {}

        for et in self.etypes:
            # pull out the count/displacement info you saved earlier
            info = self.alltoall_data[et]
            scount, sdisp = info['scount'], info['sdisp']
            rcount, rdisp = info['rcount'], info['rdisps']

            svals = packd[et]['svals']
            # build receive buffer
            # total_recv × (…trailing dims…)
            recv_shape = (rcount.sum(),) + svals.shape[1:]
            rbuf = np.empty(recv_shape, dtype=svals.dtype)

            # do the real work
            self.transporter._alltoallv(self.comm,
                                        (svals, (scount, sdisp)),
                                        (rbuf,  (rcount, rdisp)))

            relocated[et] = rbuf

        return relocated

    def combine_smeshes(self) -> None:
        """
        Combine all submeshes back into the internal smesh (rank==self.rank).
        Merges eidxs, conn, and array fields (spts, spts_nodes, spts_curved).
        After combining, only the internal smesh remains.
        """
        import numpy as np
        from pyfr.mpiutil import get_comm_rank_root

        comm, rank, _ = get_comm_rank_root()
        main = self.smeshes.get(rank)
        # Collect other ranks
        for smid in list(self.smeshes.keys()):
            if smid == rank:
                continue
            other = self.smeshes.pop(smid)
            # merge element indices
            for et in self.etypes:
                a = main.eidxs.get(et, np.array([], dtype=np.int64))
                b = other.eidxs.get(et, np.array([], dtype=np.int64))
                combined = np.concatenate([a, b], axis=0)
                main.eidxs[et] = np.unique(combined)
            # merge connectors
            main.conn = np.concatenate([main.conn, other.conn], axis=0)
            # merge arrays
            for ary_name in ['spts', 'spts_nodes', 'spts_curved']:
                main_dict = getattr(main, ary_name)
                other_dict = getattr(other, ary_name)
                merged = {}
                for et in self.etypes:
                    arr1 = main_dict.get(et)
                    arr2 = other_dict.get(et)
                    if arr1 is None or arr1.size == 0:
                        merged[et] = arr2.copy() if arr2 is not None else arr1
                    elif arr2 is None or arr2.size == 0:
                        merged[et] = arr1
                    else:
                        merged[et] = np.concatenate([arr1, arr2], axis=0)
                setattr(main, ary_name, merged)

    def recreate_mesh(self, rank: int) -> _Mesh:
        """
        Recreate a new _Mesh instance from the combined internal smesh at `rank`.
        """
        # Access combined submesh
        sm = self.smeshes[rank]

        # Instantiate _Mesh without calling __init__
        mesh = _Mesh.__new__(_Mesh)

        # Copy metadata
        mesh.fname   = self.mesh.fname
        mesh.raw     = self.mesh.raw
        mesh.ndims   = self.mesh.ndims
        mesh.subset  = False
        mesh.creator = self.mesh.creator
        mesh.codec   = self.mesh.codec
        mesh.uuid    = self.mesh.uuid
        mesh.version = self.mesh.version

        # Element types and indices
        mesh.etypes = list(sm.eidxs.keys())
        mesh.eidxs = {et: sm.eidxs[et].copy() for et in mesh.etypes}

        # Node data
        mesh.spts        = {et: sm.spts.get(et).copy() for et in mesh.etypes}
        mesh.spts_nodes  = {et: sm.spts_nodes.get(et).copy() for et in mesh.etypes}
        mesh.spts_curved = {et: sm.spts_curved.get(et).copy() for et in mesh.etypes}

        # Connectivity
        conl, conr = [], []
        mesh.con_p = {}
        mesh.bcon = {}

        for row in sm.conn:
            itype, eid, fno, nrank, rtype, relem, rface = row.tolist()
            etype = self.i_to_e[itype]

            # Internal faces
            if nrank == rank and rtype >= 0:
                conl.append((etype, eid, fno))
                conr.append((self.i_to_e[int(rtype)], int(relem), int(rface)))

            # MPI faces
            elif nrank >= 0 and nrank != rank:
                mesh.con_p.setdefault(int(nrank), []).append((etype, eid, fno))

            # Boundary faces
            else:
                # For boundary, nrank == -1 or rtype < 0; use rface as bc key
                bc_key = int(rface)
                mesh.bcon.setdefault(bc_key, []).append((etype, eid, fno))

        mesh.con = (conl, conr)
        return mesh

    @staticmethod
    def _pop_rows(arr: np.ndarray,
                  mask: np.ndarray,
                  transform: Callable[[np.ndarray], np.ndarray] = lambda x: x
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Detach rows where *mask* is True from *arr*, optionally applying
        *transform* to the slice. Returns ``(selected, remaining)``.
        """
        selected = transform(arr[mask])
        remaining = arr[~mask]
        return selected, remaining

    def _build_section(self,
                       trips: list,
                       fill: tuple[int, int, int, int],
                       *,
                       post: Callable[[np.ndarray], np.ndarray] | None = None
                       ) -> np.ndarray:
        """Generic builder for a connector block given *trips*.
        *fill* is the 4‑tuple for columns 3‑6 (nrank, nitype, nelem, nface).
        """
        if not trips:
            return np.zeros((0, 7), dtype=np.int64)
        base = self.con_e_list_to_face_i_array(trips)
        section = np.hstack([base, np.tile(fill, (base.shape[0], 1))])
        return post(section) if post else section

def run_function(mesh, comm):
    mmesh = _MetaMesh(mesh, comm)
    int_smesh = mmesh.smeshes[comm.rank]

    # Example carveout
    if comm.rank == 2:
        mmesh.carveout_mpi_smesh(int_smesh, nrank=1)
        mmesh.carveout_mpi_smesh(int_smesh, nrank=0)
        # etc.

    print(f"[Rank {comm.rank}] after carveout:")
    for smid, sm in mmesh.smeshes.items():
        if sm is not None:
            print(f"  Submesh {smid}, eidxs: {sm.eidxs}, conn.shape={sm.conn.shape}")

    mmesh.grow_smesh(mmesh.smeshes[comm.rank], mmesh.smeshes[1])
    mmesh.grow_smesh(mmesh.smeshes[comm.rank], mmesh.smeshes[0])
    
    mmesh.move_ary('spts') 
    mmesh.move_ary('spts_nodes')
    mmesh.move_ary('spts_curved')

    packed_eidxs       = mmesh.pack_eidxs()
    packed_con         = mmesh.pack_conn()
    packed_spts        = mmesh.pack_ary('spts')
    packed_spts_nodes  = mmesh.pack_ary('spts_nodes')
    packed_spts_curved = mmesh.pack_ary('spts_curved')

    # Perform all-to-all exchanges via exchanger
    exchanger = MeshExchanger(mmesh.comm, mmesh.transporter)
    packed_eidxs_exchanged       = exchanger.exchange_eidxs(packed_eidxs)
    packed_con_exchanged         = exchanger.exchange_conn(packed_con)
    packed_spts_exchanged        = exchanger.exchange_ary(packed_spts)
    packed_spts_nodes_exchanged  = exchanger.exchange_ary(packed_spts_nodes)
    packed_spts_curved_exchanged = exchanger.exchange_ary(packed_spts_curved)

    # Create remaining submeshes from packed_recv
    for nrank in range(mmesh.comm.size):
        if nrank != mmesh.comm.rank:
            mmesh.smeshes[nrank] = _SubPartitionedMesh(
                etypes=mmesh.etypes,
                eidxs={etype: packed_eidxs_exchanged[etype][nrank]
                       for etype in mmesh.etypes},
                conn=packed_con_exchanged[nrank],
            )
            # these come from (src=original rank → dest=nrank)
            mmesh.src_dest[nrank] = [mmesh.rank, nrank]


    # Drop any empty submeshes
    # mmesh.drop_empty_smeshes()

    # Now, for each submesh, set the spts, spts_nodes, and spts_curved
    for id, smesh in mmesh.smeshes.items():
        if id == mmesh.comm.rank:
            continue

        for et in mmesh.etypes:
            smesh.spts[et]        = packed_spts_exchanged[et]
            smesh.spts_nodes[et]  = packed_spts_nodes_exchanged[et]
            smesh.spts_curved[et] = packed_spts_curved_exchanged[et]

    # Combine all submeshes into a single smesh, into smesh with ID == rank
    mmesh.combine_smeshes()

    # Return a _Mesh instance with the combined smesh
    return mmesh.recreate_mesh(comm.rank)
    
if __name__ == "__main__":
    comm, rank, _ = get_comm_rank_root()

    mesh = NativeReader('/scratch/EFFORTS/LoadBalancer3/couette/squaremesh/square.pyfrm', 
                        pname='pp3', 
                        construct_con=True).mesh
    mmesh = _MetaMesh(mesh, comm=comm)

    run_function(mesh, comm)

    # Testing boilerplate code
    crpprint(-1, mmesh.smeshes)
    # Validate
    pprint(mmesh.smeshes[rank].validate_conn_counts())
    #print(mmesh.smeshes[rank].summary())

    # 8) Write out compressed NPZ
    #mmesh.export_input_mesh('mesh.npz')
    #mmesh.export_submeshes('smeshes.npz')
