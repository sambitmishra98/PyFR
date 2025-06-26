from collections import defaultdict
import numpy as np
from pyfr.mpiutil import AlltoallMixin, get_comm_rank_root
from pyfr.relocator.submesh import SubMesh

class SubMeshExchanger:
    """
    Generic element-ID + field/connector exchanger.

    Steps
    -----
    1.  Build _sent_eids ledger directly from `mmesh.smeshes[dest]`.
    2.  pack_eidxs  →  _alltoallcv  (counts barrier, recv disp cached)
    3.  pack_ary('con') / exchange_ary   (single _alltoallv)
    4.  Re-inflate tensors and merge into local sub-mesh.
    """

    def __init__(self, comm):
        self.comm = comm
        self.tx   = AlltoallMixin()
        self.alltoall_data = {}                 # etype → scount/sdisp/rcount/rdisp
        self._sent_eids   = [defaultdict(list) for _ in range(comm.size)]

    # ────────────────────────────────────────────────────────────────
    #  PUBLIC  –  pure data-exchange, no object creation / merging
    # ────────────────────────────────────────────────────────────────
    # ───── replace SubMeshExchanger.exchange() completely ────────────────
    def exchange(self, mmesh, ary_names=None):
        """
        Move element IDs + chosen arrays (default: all in SubMesh.array_specs).
        Returns:
            eidx_by_src : {src: {et: 1-D int64}}
            ary_by_src  : {src: {et: {ary: ndarray}}}
        """
        ary_names = ary_names or list(SubMesh.array_specs)

        # 0. book-keeping
        self._reset_ledger()
        self._ledger_remote_slices(mmesh)

        # 1. IDs
        eidx_recv   = self.exchange_eidxs(self.pack_eidxs(mmesh))
        eidx_by_src = self._unpack_eidxs(eidx_recv, mmesh)

        # 2. arrays (generic loop)
        ary_by_src = {s: {et: {} for et in mmesh.etypes}
                    for s in range(self.comm.size)}

        for ary in ary_names:
            recv_buf   = self.exchange_ary(mmesh, self.pack_ary(mmesh, ary), ary)
            chunk_dict = self._unpack_ary(recv_buf, mmesh, ary)
            for s in range(self.comm.size):
                for et in mmesh.etypes:
                    ary_by_src[s][et][ary] = chunk_dict[s][et]

        return eidx_by_src, ary_by_src
    # ────────────────────────────────────────────────────────────────
    #  HELPERS  ─ new
    # ────────────────────────────────────────────────────────────────
    def _unpack_eidxs(self, eidx_recv, mmesh):
        """Return {src-rank: {et: ndarray}} built from exchange_eidxs output."""
        out = {src: {} for src in range(self.comm.size)}
        for et in mmesh.etypes:
            for src, arr in enumerate(eidx_recv[et]):
                out[src][et] = arr                       # may be empty
        return out

    # ───── change _unpack_ary() signature + add ary_name param ──────────
    def _unpack_ary(self, ary_recv, mmesh, ary_name):
        out = {src: {} for src in range(self.comm.size)}
        for et in mmesh.etypes:
            meta   = self.alltoall_data[et]
            shape  = ary_recv[et].shape[1:]          # trailing dims
            dtype  = ary_recv[et].dtype
            empty  = lambda: np.empty((0, *shape), dtype)
            for src in range(self.comm.size):
                rd, nrow = meta['rdisp'][src], meta['rcount'][src]
                out[src][et] = empty() if nrow == 0 else ary_recv[et][rd: rd + nrow]
        return out


    # ────────────────────────────────────────────────────────────────
    #  HELPERS (new)
    # ────────────────────────────────────────────────────────────────
    def _ledger_remote_slices(self, mmesh):
        """Populate _sent_eids from all remote SubMeshes (no data moved yet)."""
        for dest, sm in mmesh.smeshes.items():
            if dest == mmesh.rank:
                continue
            for et in mmesh.etypes:
                self._sent_eids[dest][et].extend(sm.eidxs[et])


    def _blank_remote_slots(self, mmesh):
        """After a successful exchange, reset every remote SubMesh to blank."""
        for dest in range(self.comm.size):
            if dest != mmesh.rank:
                mmesh.smeshes[dest] = SubMesh.blank(mmesh.etypes)


    # ──────────────────────────────────────────────────────────────
    #  INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────────
    def _reset_ledger(self):
        for b in self._sent_eids:
            b.clear()

    def _merge_into_local(self, mmesh, inc):
        """Concatenate inc into mmesh.smeshes[self.rank] in-place."""
        dst = mmesh.smeshes.setdefault(
            mmesh.rank,
            SubMesh(etypes=mmesh.etypes,
                    eidxs  ={et: np.empty(0, np.int64) for et in mmesh.etypes},
                    arrays={"con": {et: np.empty(
                        (0, SubMesh.etype_nfaces_map[et], 4), np.int64)
                        for et in mmesh.etypes}})
        )
        for et in mmesh.etypes:
            if inc.eidxs[et].size:
                dst.eidxs[et] = np.concatenate([dst.eidxs[et], inc.eidxs[et]])
                dst.con  [et] = np.concatenate([dst.con  [et], inc.con[et]], axis=0)

    # ──────────────────────────────────────────────────────────────
    #  THIN WRAPPERS AROUND MPI HELPERS
    # ──────────────────────────────────────────────────────────────
    def exchange_eidxs(self, packed):
        recv = {}
        for et, info in packed.items():
            sdisp = self.tx._count_to_disp(info['scount'])
            rvals, (rcount, rdisp) = self.tx._alltoallcv(
                self.comm, info['svals'], info['scount'])
            recv[et] = [rvals[rdisp[i]: rdisp[i] + rcount[i]]
                        for i in range(self.comm.size)]
            self.alltoall_data[et] = {'scount': info['scount'], 'sdisp': sdisp,
                                      'rcount': rcount,         'rdisp': rdisp}
        return recv

    # ───── change exchange_ary() to accept ary_name (for debug prints) ──
    def exchange_ary(self, mmesh, packd, ary_name):
        relocated = {}
        for et, buf in packd.items():
            meta = self.alltoall_data[et]
            rbuf = np.empty((meta['rcount'].sum(),) + buf['svals'].shape[1:],
                            dtype=buf['svals'].dtype)
            self.tx._alltoallv(
                self.comm,
                (buf['svals'], (meta['scount'], meta['sdisp'])),
                (rbuf,         (meta['rcount'], meta['rdisp']))
            )
            relocated[et] = rbuf

        return relocated

    def pack_eidxs(self, mmesh):
        """Build {et: svals/scount} from self._sent_eids."""
        info = {}
        for et in mmesh.etypes:
            pieces, counts = [], []
            for r in range(self.comm.size):
                ids = np.asarray(self._sent_eids[r][et], np.int64)
                pieces.append(ids)
                counts.append(ids.size)
            info[et] = {
                'svals' : np.concatenate(pieces) if pieces else np.empty(0, np.int64),
                'scount': np.asarray(counts, np.int64)
            }
        return info

    def pack_ary(self, mmesh, ary_name):
        """
        Serialise one field/array for every remote rank.

        * honours the *current* 2-tuple spec:  (dtype, elem_axis)
        * trailing shape is taken from SubMesh._trailing_shape(...)
        """
        comm, rank, _ = get_comm_rank_root()
        info = {}

        dt, _elem_axis = SubMesh.array_specs[ary_name]     # <- now only two values
        meta = mmesh.meta

        for et in mmesh.etypes:
            trailing = SubMesh._trailing_shape(ary_name, meta, et)

            pieces, counts = [], []
            for r in range(comm.size):
                arr = mmesh.smeshes[r].arrays[ary_name][et]

                # sanity-check dtype & shape
                if arr.dtype != dt or arr.shape[1:] != trailing:
                    raise TypeError(
                        f"[pack_ary] rank={rank} {ary_name}[{et}] "
                        f"has dtype {arr.dtype}, shape {arr.shape}; "
                        f"expected dtype {dt}, trailing {trailing}"
                    )

                if r == rank:
                    counts.append(0)          # keep local rows
                else:
                    pieces.append(arr)
                    counts.append(arr.shape[0])

            svals = np.concatenate(pieces) if pieces else np.empty((0, *trailing), dt)

            info[et] = {
                "svals":  svals,
                "scount": np.asarray(counts, np.int64),
            }

        return info
