import atexit
import ctypes
import math
import os
import sys
import weakref

import numpy as np

from typing import List, Optional

comm_rank_roots = {}  # will now store MPICommInfo objects

class MPICommInfo:
    """
    Thin wrapper around an MPI communicator plus rank-map and device tags.

    Attributes
    ----------
    name : str
        Logical name ('world', 'compute', 'newcompute', ...).
    comm : mpi4py.MPI.Comm or MPI.COMM_NULL
    rank : Optional[int]
        Local rank in 'comm' or None if COMM_NULL.
    root : Optional[int]
        Designated root rank in 'comm' or None if COMM_NULL.
    rankmap : Optional[List[int]]
        List of world ranks in communicator order. For 'world' this is
        [0, 1, ..., size-1].
    devices : Optional[List[str]]
        Device tags per *local* rank in this communicator,
        e.g. ['gpu', 'cpu', 'cpu', ...].
    """

    # ---- per-PROCESS state (same across comms) ----
    _device: Optional[str] = None
    _etype_order: Optional[List[str]] = None
    _devices_world: Optional[List[str]] = None

    def __init__(self, name: str, comm,
                 rank: Optional[int], root: Optional[int],
                 rankmap: Optional[List[int]] = None,
                 ):
        from mpi4py import MPI

        self.name = name
        self.comm = comm
        self.rank = rank
        self.root = root
        self.rankmap = list(rankmap) if rankmap is not None else None

        # Normalise COMM_NULL <-> inactive ranks
        if comm is None or comm == MPI.COMM_NULL:
            self.comm = MPI.COMM_NULL
            self.rank = None
            self.root = None

    # ---------------- basic properties ----------------

    @property
    def device(self) -> Optional[str]:
        return type(self)._device

    @property
    def etype_order(self) -> Optional[List[str]]:
        eo = type(self)._etype_order
        return list(eo) if eo is not None else None

    @property
    def active(self) -> bool:
        from mpi4py import MPI
        return self.comm is not None and self.comm != MPI.COMM_NULL

    @property
    def size(self) -> int:
        return self.comm.Get_size() if self.active else 0

    @property
    def local_rank(self) -> Optional[int]:
        return self.rank

    def local_to_world(self, local_rank: Optional[int] = None) -> Optional[int]:
        """Map local rank -> world rank using rankmap."""
        if self.rankmap is None:
            return None
        if local_rank is None:
            local_rank = self.rank
        if local_rank is None:
            return None
        if 0 <= local_rank < len(self.rankmap):
            return self.rankmap[local_rank]
        return None

    def world_to_local(self, world_rank: int) -> Optional[int]:
        """Slow O(P) lookup; sufficient for small partitions."""
        if self.rankmap is None:
            return None
        try:
            return self.rankmap.index(world_rank)
        except ValueError:
            return None

    # ---------------- safe collectives ----------------

    def barrier(self):
        """Barrier that is a no-op on COMM_NULL."""
        if self.active:
            self.comm.Barrier()

    def allgather(self, value, default=None):
        """
        Safe allgather: if COMM_NULL returns 'default' on this rank.
        Intended for patterns where only participating ranks use the result.
        """
        if not self.active:
            return default
        return self.comm.allgather(value)

    def allreduce(self, value, op=None, default=None):
        """
        Safe allreduce wrapper. If COMM_NULL, returns 'default'.
        """
        if not self.active:
            return default

        return self.comm.allreduce(value, op=op)

    def bcast(self, value, root: int = 0, default=None):
        """
        Safe broadcast. On COMM_NULL ranks, returns 'default'.
        """
        if not self.active:
            return default
        return self.comm.bcast(value, root=root)

    def gather(self, value, root: int = 0, default=None):
        """
        Safe gather: returns list on root, default elsewhere for COMM_NULL.
        """
        if not self.active:
            return default
        return self.comm.gather(value, root=root)

    # ---------------- constructors / factories ----------------

    @property
    def devices_world(self) -> Optional[List[str]]:
        dv = type(self)._devices_world
        return list(dv) if dv is not None else None

    def device_of_world_rank(self, world_rank: int) -> Optional[str]:
        dv = type(self)._devices_world
        if dv is None:
            return None
        if 0 <= int(world_rank) < len(dv):
            return dv[int(world_rank)]
        return None

    @classmethod
    def world(cls, comm, cfg):
        size = comm.Get_size()
        rank = comm.Get_rank()
        rankmap = list(range(size))

        device = None
        devices = None
        if cfg is not None and cfg.hasopt('backend', 'devices'):
            devices = cfg.getliteral('backend', 'devices')
            if len(devices) != size:
                raise ValueError(f"devices count {len(devices)} != {size}")
            device = devices[rank]

        cls._device = device
        cls._devices_world = list(devices) if devices is not None else None  # NEW

        etype_order = None
        if cfg is not None and device is not None:
            key = f'device-preference-{device}'
            if cfg.hasopt('backend', key):
                etype_order = cfg.getliteral('backend', key)

        if etype_order is None:
            etype_order = ['hex', 'pyr', 'tet']
        cls._etype_order = list(etype_order)

        return cls('world', comm, rank, root=0, rankmap=rankmap)
    @classmethod
    def from_ranklist(cls, name: str, ranklist_world: List[int]):
        """
        Build a new communicator as a subset of MPI.COMM_WORLD,
        with rank order given by 'ranklist_world'.
        Per-process device/etype_order are inherited automatically
        via the class-level fields.
        """
        from mpi4py import MPI

        world = MPI.COMM_WORLD
        w_rank = world.Get_rank()

        # Decide if this world rank participates.
        if w_rank not in ranklist_world:
            color = MPI.UNDEFINED
            key = MPI.UNDEFINED
        else:
            color = 0
            key = ranklist_world.index(w_rank)

        new_comm = world.Split(color, key=key)

        if new_comm == MPI.COMM_NULL:
            new_rank = None
            root = None
        else:
            new_rank = new_comm.Get_rank()
            root = 0

        info = cls(
            name=name,
            comm=new_comm,
            rank=new_rank,
            root=root,
            rankmap=ranklist_world
        )
        comm_rank_roots[name] = info
        return info

    def run(self, fn, default=None):
        """
        Execute `fn()` only if this rank is active in this communicator.
        Otherwise return `default`.
        """
        if self.active:
            return fn()
        else:
            return default


def promote_comm(src_name: str, dest_name: str) -> None:
    """
    Atomically replace logical communicator 'dest_name' with 'src_name'.

    After this call:
        - comm[dest_name], rank[dest_name], rankmap[dest_name], etc.
        all refer to the communicator that was previously 'src_name'.
        - The mapping under 'src_name' is removed.

    All world ranks must call this collectively.
    """
    src_info = get_comm_info(src_name)

    # Rebind: dest_name now points at src_info.
    comm_rank_roots[dest_name] = src_info
    src_info.name = dest_name

    # Remove the old src_name key to avoid accidental reuse.
    if src_name in comm_rank_roots and src_name != dest_name:
        del comm_rank_roots[src_name]

def init_mpi(cfg=None):

    global comm_rank_roots

    import mpi4py.rc
    from mpi4py import MPI

    # Prefork to allow us to exec processes after MPI is initialised
    if hasattr(os, 'fork'):
        from pytools.prefork import enable_prefork

        enable_prefork()

    # Manually initialise MPI with thread support
    MPI.Init_thread()

    # Prevent mpi4py from calling MPI_Finalize
    mpi4py.rc.finalize = False

    comm = MPI.COMM_WORLD

    comm_rank_roots['world'] = MPICommInfo.world(comm, cfg)

    # Intercept any uncaught exceptions
    class ExceptHook:
        def __init__(self):
            self.exception = None

            self._orig_excepthook = sys.excepthook
            sys.excepthook = self._excepthook

        def _excepthook(self, exc_type, exc, *args):
            self.exception = exc
            self._orig_excepthook(exc_type, exc, *args)

    # Register our exception hook
    excepthook = ExceptHook()

    def onexit():
        if not MPI.Is_initialized() or MPI.Is_finalized():
            return

        # Get the current exception (if any)
        exc = excepthook.exception

        # If we are exiting normally then call MPI_Finalize
        if (comm.size == 1 or exc is None or
            isinstance(exc, KeyboardInterrupt) or
            (isinstance(exc, SystemExit) and exc.code == 0)):
            import gc
            gc.collect()

            MPI.Finalize()
        # Otherwise forcefully abort
        else:
            sys.stderr.flush()
            MPI.COMM_WORLD.Abort(1)

    # Register our exit handler
    atexit.register(onexit)


def autofree(obj):
    def callfree(fromhandle, handle):
        fromhandle(handle).free()

    weakref.finalize(obj, callfree, obj.fromhandle, obj.handle)
    return obj


class _CommView:
    def __getitem__(self, name):
        return comm_rank_roots[name].comm

    def __getattr__(self, name: str):
        return self[name]


class _RankView:
    """
    View over MPICommInfo.rank

    rank['world']      -> int world-local rank
    rank['compute']    -> int local rank in 'compute', or None if COMM_NULL
    rank.world         -> same as rank['world']
    """
    def __getitem__(self, name):
        return comm_rank_roots[name].rank

    def __getattr__(self, name: str):
        return self[name]


class _RootView:
    """
    View over MPICommInfo.root

    root['world']      -> root rank for 'world' (typically 0)
    root['compute']    -> root for 'compute', or None if COMM_NULL
    root.world         -> same as root['world']
    """
    def __getitem__(self, name: str):
        return comm_rank_roots[name].root

    def __getattr__(self, name: str):
        return self[name]


class _RankMapView:
    """
    View over MPICommInfo.rankmap (list of world ranks in communicator order).

    rankmap['world']      -> [0, 1, 2, ..., size-1]
    rankmap['compute']    -> e.g. [0, 2, 4]
    rankmap.world         -> same as rankmap['world']
    """
    def __getitem__(self, name: str):
        return comm_rank_roots[name].rankmap

    def __getattr__(self, name: str):
        return self[name]


class _ExecView:
    """
    executor['compute'](lambda: fn(...), default=...)
    executes fn() only on ranks active in communicator 'compute'.
    """
    def __getitem__(self, name: str):
        info = comm_rank_roots[name]

        def _run(fn, default=None):
            if info.active:
                return fn()
            else:
                return default

        return _run


def get_comm_rank_root():

    info = comm_rank_roots.get('world')

    return info.comm, info.rank, info.root


def append_comm_rank_root(comm_name, comm, rank, root, rank_mapping):
    """
    Low-level hook used in some places. Now stores an MPICommInfo.
    Per-process device / etype_order are taken from MPICommInfo class.
    """
    comm_rank_roots[comm_name] = MPICommInfo(
        name=comm_name,
        comm=comm,
        rank=rank,
        root=root,
        rankmap=rank_mapping
    )

def get_comm_info(comm_name='world') -> MPICommInfo:
    """Return the MPICommInfo object for a logical communicator."""
    info = comm_rank_roots.get(comm_name)
    if info is None:
        raise KeyError(f"Unknown MPI communicator name '{comm_name}'")
    return info


def get_local_rank():
    envs = [
        'MV2_COMM_WORLD_LOCAL_RANK',
        'OMPI_COMM_WORLD_LOCAL_RANK',
        'SLURM_LOCALID'
    ]

    for ev in envs:
        if ev in os.environ:
            return int(os.environ[ev])
    else:
        from mpi4py import MPI

        return autofree(MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)).rank


def scal_coll(colfn, v, *args, **kwargs):
    dtype = int if isinstance(v, (int, np.integer)) else float
    v = np.array([v], dtype=dtype)
    colfn(mpi.IN_PLACE, v, *args, **kwargs)
    return dtype(v[0])


def get_start_end_csize(comm, n):
    rank, size = comm.rank, comm.size

    # Determine how much data each rank is responsible for
    csize = max(-(-n // size), 1)

    # Determine which part of the dataset we should handle
    return min(rank*csize, n), min((rank + 1)*csize, n), csize


class AlltoallMixin:
    @staticmethod
    def _count_to_disp(count):
        return np.concatenate(([0], np.cumsum(count[:-1])))

    @staticmethod
    def _disp_to_count(disp, n):
        return np.concatenate((disp[1:] - disp[:-1], [n - disp[-1]]))

    def _alltoallv(self, comm, sbuf, rbuf):
        svals = sbuf[0]

        # If we are dealing with scalar data then call Alltoallv directly
        if svals.dtype.names is None and svals.ndim == 1:
            comm.Alltoallv(sbuf, rbuf)
        # Else, we need to create a suitable derived datatype
        else:
            from mpi4py.util.dtlib import from_numpy_dtype

            dtype = svals.dtype

            if svals.ndim > 1:
                dtype = [('', dtype, svals.shape[1:])]

            dtype = autofree(from_numpy_dtype(dtype).Commit())
            comm.Alltoallv((*sbuf, dtype), (*rbuf, dtype))

    def _alltoallcv(self, comm, svals, scount, sdisps=None):
        # Exchange counts
        rcount = np.empty_like(scount)
        comm.Alltoall(scount, rcount)

        # Compute displacements
        rdisps = self._count_to_disp(rcount)
        sdisps = self._count_to_disp(scount) if sdisps is None else sdisps

        # Exchange values
        rvals = np.empty((rcount.sum(), *svals.shape[1:]), dtype=svals.dtype)
        rbuf = (rvals, (rcount, rdisps))
        self._alltoallv(comm, (svals, (scount, sdisps)), rbuf)

        return rbuf


class BaseGathererScatterer(AlltoallMixin):
    def __init__(self, comm, aidx):
        self.comm = comm

        # Determine array size
        n = aidx[-1] if len(aidx) else -1
        n = scal_coll(comm.Allreduce, n, op=mpi.MAX) + 1

        # Determine which part of the dataset we should handle
        self.start, self.end, csize = get_start_end_csize(comm, n)

        # Map each index to its associated rank
        adisps = np.searchsorted(aidx, csize*np.arange(comm.size))
        acount = np.diff(adisps, append=len(aidx))

        # Exchange the indices
        bidx, (bcount, bdisps) = self._alltoallcv(comm, aidx, acount, adisps)

        # Save the count and displacement information
        self.acountdisps = (acount, adisps)
        self.bcountdisps = (bcount, bdisps)

        # Return the index information
        return bidx


class Scatterer(BaseGathererScatterer):
    def __init__(self, comm, idx):
        idx = np.asanyarray(idx, dtype=int)

        # Eliminate duplicates from our index array
        ridx, self.rinv = np.unique(idx, return_inverse=True)

        self.sidx = super().__init__(comm, ridx) - self.start

        # Save the receive count
        self.cnt = len(ridx)

    def __call__(self, dset, didxs=(...,)):
        # Read the data
        svals = dset[self.start:self.end, *didxs][self.sidx]

        # Allocate space for receiving the data
        rvals = np.empty((self.cnt, *svals.shape[1:]), dtype=svals.dtype)

        # Perform the exchange
        self._alltoallv(self.comm, (svals, self.bcountdisps),
                        (rvals, self.acountdisps))

        # Unpack the data
        return rvals[self.rinv]


class Gatherer(BaseGathererScatterer):
    def __init__(self, comm, idx):
        idx = np.asanyarray(idx, dtype=int)

        # Sort our send array
        self.sinv = np.argsort(idx)
        self.sidx = idx[self.sinv]

        bidx = super().__init__(comm, self.sidx)

        # Determine how to sort the data we will receive
        self.rinv = np.argsort(bidx)
        self.ridx = bidx[self.rinv]

        # Note the source rank of each received element
        self.rsrc = np.repeat(np.arange(comm.size), self.bcountdisps[0])
        self.rsrc = self.rsrc[self.rinv].astype(np.int32)

        # Compute the total number of items and our offset
        self.cnt = cnt = len(self.ridx)
        self.tot = scal_coll(comm.Allreduce, cnt, op=mpi.SUM)
        self.off = scal_coll(comm.Exscan, cnt, op=mpi.SUM)
        self.off = self.off if comm.rank else 0

    def __call__(self, dset):
        # Sort the data we are going to be sending
        svals = np.ascontiguousarray(dset[self.sinv])

        # Allocate space for the data we will receive
        rvals = np.empty((self.cnt, *dset.shape[1:]), dtype=dset.dtype)

        # Perform the exchange
        self._alltoallv(self.comm, (svals, self.acountdisps),
                        (rvals, self.bcountdisps))

        # Sort our received data
        return rvals[self.rinv]


class SparseScatterer(AlltoallMixin):
    def __init__(self, comm, iset, aidx):
        self.comm = comm

        # Sort our indices
        ainv = np.argsort(aidx)
        bidx = aidx[ainv]

        # Determine the array size
        n = len(iset)

        # Determine which part of the dataset we should handle
        self.start, self.end, _ = get_start_end_csize(comm, n)

        # Read our portion of the sorted index table
        cidx = iset[self.start:self.end]

        # Tell other ranks what region we have
        region = np.array([cidx.min(initial=n), cidx.max(initial=n) + 1])
        minmax = np.empty(2*comm.size, dtype=int)
        comm.Allgather(region, minmax)

        # Determine which rank, if any, has each of our desired indices
        didx = np.split(bidx, np.searchsorted(bidx, minmax))[1::2]
        dcount = np.array([len(s) for s in didx])

        # Exchange indices
        eidx, (ecount, edisps) = self._alltoallcv(comm, np.concatenate(didx),
                                                  dcount)

        # See which of these indices are present
        mask = np.isin(eidx, cidx, assume_unique=True)
        sidx = eidx[mask]
        scount = np.array([m.sum() for m in np.split(mask, edisps[1:])])
        sdisps = self._count_to_disp(scount)

        # Make a note of which indices we have
        self.sidx = np.searchsorted(cidx, sidx)
        self.scountdisps = (scount, sdisps)

        # Exchange the present indices
        ridx, self.rcountdisps = self._alltoallcv(comm, sidx, scount,
                                                  sdisps)

        self.ridx = ainv[np.searchsorted(bidx, ridx)]
        self.cnt = self.rcountdisps[0].sum()

    def __call__(self, dset, didxs=(...,)):
        # Read and appropriately reorder our send data
        svals = dset[self.start:self.end, *didxs][self.sidx]

        # Allocate space for receiving the data
        rvals = np.empty((self.cnt, *svals.shape[1:]), dtype=svals.dtype)

        # Perform the exchange
        self._alltoallv(self.comm, (svals, self.scountdisps),
                        (rvals, self.rcountdisps))

        return rvals

def initialise_new_comm(comm_name, rank_mapping):
    """
    Create / update a logical communicator as a subset of MPI.COMM_WORLD.
    """
    MPICommInfo.from_ranklist(comm_name, rank_mapping)


class Sorter(AlltoallMixin):
    typemap = {
        'int8': np.uint8, 'int16': np.uint16,
        'int32': np.uint32, 'int64': np.uint64,
        'float32': np.int32, 'float64': np.int64
    }

    def __init__(self, comm, keys):
        self.comm = comm

        # Locally sort our outbound keys
        self.sidx = np.argsort(keys)
        skeys = keys[self.sidx]

        # Determine the total size of the array
        size = scal_coll(comm.Allreduce, len(keys))

        self.start, end, csize = get_start_end_csize(comm, size)
        self.cnt = end - self.start

        # Determine what to send to each rank
        sdisps = self._splitters(skeys, self.start)
        scount = self._disp_to_count(sdisps, len(keys))
        self.scountdisps = (scount, sdisps)

        # Exchange the keys
        rkeys, self.rcountdisps = self._alltoallcv(comm, skeys, scount,
                                                   sdisps)

        # Locally sort our inbound keys
        self.ridx = np.argsort(rkeys)
        self.keys = rkeys[self.ridx]

    def _transform_keys(self, skeys):
        dtype = skeys.dtype

        if np.issubdtype(dtype, np.unsignedinteger):
            return skeys
        elif np.issubdtype(dtype, np.signedinteger):
            udtype = self.typemap[dtype.name]
            return skeys.view(udtype) ^ udtype(np.iinfo(dtype).max + 1)
        elif np.issubdtype(dtype, np.floating):
            shift = 8*dtype.itemsize - 1
            idtype = self.typemap[dtype.name]
            udtype = self.typemap[np.dtype(idtype).name]

            mask = (skeys.view(idtype) >> shift).view(udtype)
            mask |= udtype(1 << shift)
            return skeys.view(udtype) ^ mask
        else:
            raise ValueError('Unsupported dtype')

    def _splitters(self, skeys, r):
        # Transform the keys so they're unsigned integers
        skeys = self._transform_keys(skeys)

        # Determine the minimum and maximum values in the array
        kmin = self.comm.allreduce(int(skeys[0]), op=mpi.MIN)
        kmax = self.comm.allreduce(int(skeys[-1]), op=mpi.MAX)

        # Compute the number of bits in the key space
        W = math.ceil(math.log2(kmax - kmin + 1))

        e, rt = kmin, 0
        q = np.empty(self.comm.size, dtype=skeys.dtype)

        for i in range(W - 1, -1, -1):
            # Compute and gather the probes
            q[self.comm.rank] = e + 2**i
            self.comm.Allgather(mpi.IN_PLACE, q)

            # Obtain the global location of each probe
            t = np.searchsorted(skeys, q)
            self.comm.Reduce_scatter_block(mpi.IN_PLACE, t)

            if t[0] <= r:
                e, rt = e + 2**i, t[0]

        q[self.comm.rank] = e
        self.comm.Allgather(mpi.IN_PLACE, q)

        # Count the occurances of each probe in skeys
        ubnd = np.searchsorted(skeys, q, side='right')
        lbnd = np.searchsorted(skeys, q, side='left')
        ld = ubnd - lbnd

        # Compute the global position of each probe
        gd = np.zeros_like(ld)
        self.comm.Exscan(ld, gd)

        q[self.comm.rank] = r - rt
        self.comm.Allgather(mpi.IN_PLACE, q)

        return lbnd + np.maximum(0, np.minimum(ld, q.astype(int) - gd))

    def __call__(self, svals):
        # Locally sort our data
        svals = svals[self.sidx]

        # Allocate space for receiving the data
        rvals = np.empty((self.cnt, *svals.shape[1:]), dtype=svals.dtype)

        # Perform the exchange
        self._alltoallv(self.comm, (svals, self.scountdisps),
                        (rvals, self.rcountdisps))

        # Locally sort our received data
        return rvals[self.ridx]

    @property
    def argidx(self):
        svals = self.start + np.argsort(self.ridx)
        rvals = np.empty(len(self.sidx), dtype=svals.dtype)

        self._alltoallv(self.comm, (svals, self.rcountdisps),
                        (rvals, self.scountdisps))

        return rvals[np.argsort(self.sidx)]


class _MPI_Funcs:
    def __init__(self):
        from mpi4py import MPI

        self._lib = ctypes.CDLL(MPI.__file__)

    def __getattr__(self, attr):
        func = getattr(self._lib, f'MPI_{attr}')
        return ctypes.cast(func, ctypes.c_void_p).value


class _MPI:
    def __init__(self):
        self.funcs = _MPI_Funcs()

    def addrof(self, obj):
        from mpi4py import MPI

        return MPI._addressof(obj)

    def __getattr__(self, attr):
        from mpi4py import MPI

        return getattr(MPI, attr)


mpi = _MPI()

comm = _CommView()
rank = _RankView()
root = _RootView()
rankmap = _RankMapView()
execute = _ExecView()