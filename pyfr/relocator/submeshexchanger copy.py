from collections import defaultdict

import numpy as np

from pyfr.mpiutil import AlltoallMixin
from pyfr.relocator.submesh import SubMesh

class SubMeshExchanger:
    """
    Exchange submesh data with _alltoallcv and _alltoallv.
    After eidxs exchange, use recv info with _alltoallv for any array. 
    """
    
    def __init__(self, comm):
        self.comm = comm
        self.tx = AlltoallMixin()
        self.alltoall_data = {}

        self._meta: dict[str, dict[str, np.ndarray]] = {}   # etype → rcount/rdisp

        self._sent_eids = [defaultdict(list) for _ in range(comm.size)]

    def exchange_eidxs(self, packed: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, list[np.ndarray]]:
        recv = {}
        for et, info in packed.items():
            sdisp = self.tx._count_to_disp(info['scount'])
            rvals, (rcount, rdisps) = self.tx._alltoallcv(self.comm, info['svals'], info['scount'] )
            recv[et] = [ rvals[rdisps[i] : rdisps[i] + rcount[i]] for i in range(self.comm.size) ]
            self.alltoall_data[et] = {'scount': info['scount'], 'sdisp': sdisp, 
                                      'rcount': rcount, 'rdisps': rdisps, }
        return recv

    def exchange_conn(self, packd):
        rvals, (rcount, rdisp) = self.tx._alltoallcv(self.comm, packd['svals'], packd['scount'])
        return {r: (rvals[rdisp[r]: rdisp[r]+rcount[r]].reshape(-1, 7) if rcount[r] else np.zeros((0, 7), np.int64)) for r in range(self.comm.size)}
        
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

        for et in packd:
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
            self.tx._alltoallv(self.comm, (svals, (scount, sdisp)),
                                          (rbuf,  (rcount, rdisp)))

            relocated[et] = rbuf

        return relocated

    def _reset_ledger(self):
        for bucket in self._sent_eids:
            bucket.clear()

    def _build_remote_submesh(self, mmesh, rank, eidx_in, conn_in, a_by_rank):
        """Compose a SubMesh assembled from the pieces received from *rank*."""
        return SubMesh(
            etypes        = mmesh.etypes,
            eidxs         = {et: eidx_in[et][rank]          for et in mmesh.etypes},
            conn          = conn_in[rank],
            #spts          = {et: a_by_rank['spts'][et][rank]        for et in mmesh.etypes},
            #spts_nodes    = {et: a_by_rank['spts_nodes'][et][rank]  for et in mmesh.etypes},
            #spts_curved   = {et: a_by_rank['spts_curved'][et][rank] for et in mmesh.etypes},
            strict_geometry=False
        )

    def _merge_into_mesh(self, mmesh, rank, inc):
        """Combine *inc* into mmesh.smeshes[rank], creating/expanding as needed."""
        # Re-stamp owner column (col-3) from sender→local
        inc.conn[inc.conn[:, 3] == rank, 3] = mmesh.rank

        dst = mmesh.smeshes.get(rank)
        if dst is None or dst.conn.size == 0:
            mmesh.smeshes[rank] = inc
        else:
            dst.conn = np.concatenate((dst.conn, inc.conn))
            for et in mmesh.etypes:
                dst.eidxs[et] = np.unique(np.concatenate((dst.eidxs[et], inc.eidxs[et])))

    def perform_all_exchanges(self, mmesh):
        # reset book-keeping for this exchange pass
        self._reset_ledger()

        conn_in = self.exchange_conn(self.pack_conn(mmesh))
        eidx_in = self.exchange_eidxs(self.pack_eidxs(mmesh))

        #arrays_in = {name: self.exchange_ary(mmesh.pack_ary(name))for name in ('spts', 'spts_nodes', 'spts_curved')}
        #arys_by_rank = {name: {et: self._split_by_rank(arrays_in[name][et], et) for et in mmesh.etypes} for name in arrays_in}
        
        # merge what came from each remote rank
        for r in range(mmesh.comm.size):
            if r == mmesh.rank:
                continue
         
            inc = SubMesh(etypes = mmesh.etypes, 
                          eidxs = {et: eidx_in[et][r] for et in mmesh.etypes}, conn = conn_in[r], 
                        
                          #**{name: {et: arys_by_rank[name][et][r] for et in mmesh.etypes} for name in ('spts', 'spts_nodes', 'spts_curved') }, 
                          )

            inc.validate(full=True)

            mask = inc.conn[:, 3] == r
            inc.conn[mask, 3] = mmesh.rank

            dst = mmesh.smeshes.get(r)
            if dst is None or dst.conn.size == 0:
                mmesh.smeshes[r] = inc
            else:
                dst.conn = np.concatenate([dst.conn, inc.conn])
                for et in mmesh.etypes:
                    dst.eidxs[et] = np.unique(np.concatenate([dst.eidxs[et], inc.eidxs[et]]))

            for sm in mmesh.smeshes.values():
                sm.validate(full=True)

    def _split_by_rank(self, buf: np.ndarray, et: str) -> list[np.ndarray]:
        """
        Slice *buf* into one view per MPI rank using the rcount/rdisp info
        saved earlier in self.alltoall_data[et] (populated in exchange_eidxs).
        Returns a list of length comm.size.
        """
        info   = self.alltoall_data[et]
        rdisp  = info['rdisps']
        rcount = info['rcount']
        return [buf[rdisp[i]: rdisp[i] + rcount[i]]   if rcount[i]
                else buf[0:0]                         # empty  view, preserves dtype
                for i in range(self.comm.size)]

    def pack_eidxs(self, mmesh):
        info = {}
        for et in mmesh.etypes:
            counts, pieces = [], []
            for r in range(mmesh.comm.size):
                ids = np.asarray(self._sent_eids[r][et], dtype=np.int64)
                counts.append(ids.size)
                pieces.append(ids)
            info[et] = {
                'svals' : np.concatenate(pieces) if pieces else np.empty(0, np.int64),
                'scount': np.array(counts, dtype=np.int64)
            }
        return info

    def pack_conn(self, mmesh):
        """
        Serialise every *remote* sub-mesh destined for a different rank.
        After the buffers are built the corresponding sub-meshes are deleted
        so they will no longer be merged back locally.
        """
        svals, scount = [], np.zeros(mmesh.comm.size, np.int64)

        for dest in range(mmesh.comm.size):
            if dest == mmesh.rank:
                continue                        # nothing to send to yourself
            sm = mmesh.smeshes.get(dest)
            if sm is None or sm.conn.size == 0:
                continue

            buf = sm.conn.ravel().astype(np.int64)
            svals.append(buf)
            scount[dest] = buf.size

            for et in mmesh.etypes:
                self._sent_eids[dest][et].extend(sm.eidxs.get(et, []))

            # --------------------------------------------------------------
            # ❷ discard local copy – ownership is being transferred
            # --------------------------------------------------------------
            del mmesh.smeshes[dest]

        return {'svals': np.concatenate(svals) if svals else np.empty(0, np.int64),
                'scount': scount}

    def _rebuild_nrank(self, mmesh) -> None:
        """
        Make column-3 of every conn row reflect the *current* owner of the
        right-hand element (netype, neid).
        """
        comm = mmesh.comm
        etypes, e2i = mmesh.etypes, mmesh.e_to_i

        # 1) local list  [(itype, gid, my_rank), ...]
        local_pairs = []
        main = mmesh.smeshes[mmesh.rank]          # after combine_smeshes
        for et in etypes:
            it = e2i[et]
            local_pairs.extend((it, int(gid), mmesh.rank) for gid in main.eidxs[et])

        # 2) gather and build dict  (itype, gid) → owner
        all_pairs = comm.allgather(np.asarray(local_pairs, dtype=np.int64))
        owner = {}
        for arr in all_pairs:
            for it, gid, rnk in arr:
                owner[(int(it), int(gid))] = int(rnk)

        # 3) patch every local sub-mesh
        for sm in mmesh.smeshes.values():
            # rows whose RHS is an element, not a boundary
            mask = sm.conn[:, 4] >= 0
            rhs  = sm.conn[mask][:, 4:6]                  # (netype, neid)
            sm.conn[mask, 3] = [owner[tuple(k)] for k in rhs]
