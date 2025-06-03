# ─── submesharymanager.py (was array_manager.py) ──────────────────────────
from typing import Dict
import numpy as np
from pyfr.mpiutil import get_comm_rank_root, mpi

AryEtDict = Dict[str, np.ndarray]          # etype -> ndarray

class SubMeshAryManager:
    """
    Handles arbitrary element-centric arrays.

    self.arrays      : Dict[str, AryEtDict]        # 'spts' -> {'tri': …}
    self._ary_edim   : Dict[str, int]              # 'spts' -> 1
    """

    # ────────────────────────────────────────────────────────────────────
    #  Registration
    # ────────────────────────────────────────────────────────────────────
# ── submesharymanager.py ────────────────────────────────────────────────
from typing import Dict
import numpy as np
from pyfr.mpiutil import get_comm_rank_root, mpi

AryEtDict = Dict[str, np.ndarray]          # etype → ndarray


class SubMeshAryManager:
    # … (rest unchanged) …

    # ───────────────────────────────────────────────────────────────────
    #  Robust, MPI-safe registration
    # ───────────────────────────────────────────────────────────────────
    def register_array(self, name: str, *, edim: int, edict: AryEtDict) -> None:
        """
        Parameters
        ----------
        name   : str        logical family name, e.g. 'spts'
        edim   : int        position of the element-axis in *incoming* arrays
                            (0 for everything except spts, which is 1)
        edict  : {etype → ndarray}
                  geometry blocks for as many element types as are present
        """
        comm, _, _ = get_comm_rank_root()

        # -----------------------------------------------------------------
        # 1)  Decide on a prototype  (dtype, trailing-shape)
        #     • try local non-empty block
        #     • else gather candidates from other ranks
        #     • else default to float64 row-vector  (legacy behaviour)
        # -----------------------------------------------------------------
        proto = next((a for a in edict.values() if a.size), None)

        if proto is None:
            # ask everyone – first rank that *has* data supplies prototype
            proto = next(
                (p for p in comm.allgather(None if proto is None else proto) if p is not None),
                None
            )

        if proto is None:                          # truly no data anywhere
            dtype, shp1 = np.float64, (1,)         #  ➜ (0,1) sentinel
        else:
            dtype, shp1 = proto.dtype, proto.shape[1:]

        empty_like = lambda: np.empty((0,) + shp1, dtype)

        # -----------------------------------------------------------------
        # 2)  Build a fully-populated per-etype dictionary
        #     (axis permutation done here so *all* stored arrays have edim=0)
        # -----------------------------------------------------------------
        fixed: AryEtDict = {}
        for et in self.etypes:
            arr = edict.get(et)
            if arr is None or arr.size == 0:
                arr = empty_like()
            if edim != 0 and arr.ndim > edim:      # bring element axis forward
                arr = np.moveaxis(arr, edim, 0)
            fixed[et] = arr.copy()

        # -----------------------------------------------------------------
        # 3)  Stash away
        # -----------------------------------------------------------------
        self.arrays[name]    = fixed
        self._ary_edim[name] = edim



    # ────────────────────────────────────────────────────────────────────
    #  Packing for alltoallv
    # ────────────────────────────────────────────────────────────────────
    def pack_arrays(self, names: list[str]):
        """
        Return a *per-array, per-etype* payload identical in spirit to the
        classic `pack_ary`:

            {
              'spts' : {
                  'tri' : {'svals': ndarray, 'scount': counts},   # per dest-rank
                  'quad': {...},
              },
              'spts_nodes' : { ... },
            }
        """
        comm, rank, _ = get_comm_rank_root()
        info = {}

        for name in names:
            ary_info = {}
            for et in self.etypes:
                # concatenate chunks that go to each rank  (old logic)
                pieces, counts = [], []
                for r in range(comm.size):
                    gids   = self._dest_gid_lists(self.parent_mmesh)[r][et]
                    arrsrc = self.arrays[name][et]
                    if gids.size:
                        # global-id -> local row lookup once per call
                        g2row = {int(g): i for i, g in enumerate(self.eidxs[et])}
                        rows  = [g2row[int(g)] for g in gids]
                        blk   = arrsrc[rows]
                    else:
                        blk   = arrsrc[0:0]   # typed empty
                    pieces.append(blk)
                    counts.append(blk.shape[0])

                ary_info[et] = {
                    'svals' : np.concatenate(pieces) if pieces else arrsrc[0:0],
                    'scount': np.asarray(counts, np.int64)
                }

            info[name] = ary_info
        return info



    # --------------- UNPACK (postprocess) --------------------------
    @staticmethod
    def unpack_split(buf: np.ndarray, counts: np.ndarray, disps: np.ndarray):
        """Return &[buf slice] per rank – helper for Exchanger."""
        out = []
        for r in range(len(counts)):
            n = counts[r]
            out.append(buf[disps[r]: disps[r]+n] if n else buf[0:0])
        return out

    # helper: which global element IDs must go to each rank?
    def _dest_gid_lists(self, mmesh):
        """Return {rank → {etype → np.ndarray[gids]}} using the exchanger ledger."""
        # The ledger is filled in SubMeshExchanger.pack_conn (_sent_eids)
        return getattr(mmesh, "_exch_sent_eids")      # injected right before packing
