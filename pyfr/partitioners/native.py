import h5py
import numpy as np

from pyfr.mpiutil import comm, rank, root
from pyfr.partitioners.base import BasePartitioner, write_partitioning
from pyfr.progress import NullProgressSequence


def _write_random_partitioning(mesh_path: str, nparts: int, *, seed: int = 2079) -> str:
    """
    Root-only: write a temporary random partitioning with `nparts` parts into mesh_path.
    Returns the new partitioning name.
    """
    pname = f"__startup_rand_{nparts}_seed{int(seed)}"

    partwts = [1]*int(nparts)
    part = RandomPartitioner(partwts, elewts=None, opts={'seed': int(seed)})

    with h5py.File(mesh_path, 'r+') as mesh:
        mesh.require_group('partitionings')
        pinfo = part.partition(mesh, NullProgressSequence())
        write_partitioning(mesh, pname, pinfo)

    return pname


class RandomPartitioner(BasePartitioner):
    """
    Graph partitioner: uniform random assignment of vertices to partitions.
    This is a real BasePartitioner (usable by pyfr partition if you register it).
    """
    name = 'random'
    has_part_weights = False
    has_multiple_constraints = True  # permits elewts=None

    dflt_opts = {'seed': -1}
    int_opts = {'seed'}
    enum_opts = {}

    def _partition_graph(self, graph, partwts):
        nparts = len(partwts)
        nvert = len(graph.vtab) - 1

        if nparts == 1:
            return np.zeros(nvert, dtype=np.int32)

        seed = int(self.opts.get('seed', -1))
        rng = np.random.default_rng(None if seed < 0 else seed)
        return rng.integers(0, nparts, size=nvert, dtype=np.int32)


class ContiguousPartitioner(RandomPartitioner):
    """
    Placeholder for a future *graph* contiguous partitioner.
    For now we keep contiguity as an OnlineDiffusionPartitioner startup postprocess,
    which is the correct level for your island-removal logic.
    """
    name = 'contiguous'
    has_part_weights = True
    has_multiple_constraints = True


# ------------------------ Startup init policy (used by __main__) ----------------

class _StartupInitBase:
    name = 'none'

    def __init__(self, cfg):
        self.cfg = cfg

    def prepare_pname(self, *, mesh_path: str, compute_pname: str) -> str:
        return compute_pname

    def postprocess(self, *, mmesh, partitioner: str) -> None:
        return


class _StartupInitRandom(_StartupInitBase):
    name = 'random'

    def prepare_pname(self, *, mesh_path: str, compute_pname: str) -> str:
        # Strict: current implementation assumes compute == world
        if comm['compute'].size != comm['world'].size:
            raise NotImplementedError(
                "[startup-rand] compute comm != world comm not supported yet"
            )

        seed = 2079
        if self.cfg is not None and self.cfg.hasopt('partition', 'startup-rand-seed'):
            seed = self.cfg.getint('partition', 'startup-rand-seed')

        if rank['world'] == root['world']:
            tmp = _write_random_partitioning(mesh_path, comm['compute'].size, seed=seed)
            print(
                f"[startup-rand] wrote partitioning {tmp!r} "
                f"(nparts={comm['compute'].size}, seed={seed})",
                flush=True
            )
        else:
            tmp = None

        compute_pname = comm['world'].bcast(tmp, root=root['world'])
        comm['world'].barrier()
        return str(compute_pname)


class _StartupInitContiguous(_StartupInitRandom):
    name = 'contiguous'

    def postprocess(self, *, mmesh, partitioner: str) -> None:
        if partitioner != 'diffusion':
            raise NotImplementedError(
                "[startup-contig] initial-partitioner=contiguous requires partitioner=diffusion"
            )

        if not hasattr(mmesh, 'startup_make_contiguous'):
            raise NotImplementedError(
                "[startup-contig] diffusion partitioner missing startup_make_contiguous()"
            )

        mmesh.startup_make_contiguous(max_iters=100)

class _StartupInitContiguousDeviceType(_StartupInitRandom):
    name = 'contiguous-device-type'

    def postprocess(self, *, mmesh, partitioner: str) -> None:
        if partitioner != 'diffusion':
            raise NotImplementedError(
                "[startup-contig] initial-partitioner=contiguous-device-type "
                "requires partitioner=diffusion"
            )

        if not hasattr(mmesh, 'startup_make_contiguous'):
            raise NotImplementedError(
                "[startup-contig] diffusion partitioner missing startup_make_contiguous()"
            )

        mmesh.startup_make_contiguous(max_iters=100,
                                      maintain_cluster_by_device_types=True)


def get_startup_init(cfg, *, startup_from_one: bool):
    """
    Factory used by __main__.
    Preserves old behavior: if startup_from_one and no explicit setting,
    default to contiguous (random + contiguity cleanup).
    """
    init_partitioner = None
    if cfg is not None and cfg.hasopt('partition', 'initial-partitioner'):
        init_partitioner = cfg.get('partition', 'initial-partitioner').strip().lower()
    elif startup_from_one:
        init_partitioner = 'contiguous'

    if not startup_from_one:
        return _StartupInitBase(cfg)

    if init_partitioner == 'random':
        return _StartupInitRandom(cfg)
    elif init_partitioner == 'random-etypes':
        return _StartupInitRandomEtypes(cfg)
    elif init_partitioner == 'contiguous-etypes':
        return _StartupInitContiguousEtypes(cfg)
    elif init_partitioner == 'contiguous':
        return _StartupInitContiguous(cfg)
    elif init_partitioner == 'contiguous-device-type':
        return _StartupInitContiguousDeviceType(cfg)
    else:
        raise NotImplementedError(
            f"[startup] unknown initial-partitioner={init_partitioner!r}"
        )


def _write_startup_etypes_partitioning(mesh_path: str, nparts: int, *,
                                      seed: int,
                                      devices_world: list[str],
                                      etype_to_device: dict[str, str]) -> str:
    """
    Root-only: write a startup partitioning where each element-type is assigned
    only to ranks whose device tag matches etype_to_device[etype].

    Assignment within a device-group is balanced by round-robin after shuffling.
    """
    pname = f"__startup_etypes_{nparts}_seed{int(seed)}"
    rng = np.random.default_rng(None if int(seed) < 0 else int(seed))

    if len(devices_world) != int(nparts):
        raise ValueError(f"[startup-etypes] devices_world len {len(devices_world)} != nparts {nparts}")

    # Build device->ranks map
    dev_to_ranks = {}
    for r, d in enumerate(devices_world):
        dev_to_ranks.setdefault(str(d), []).append(int(r))

    with h5py.File(mesh_path, 'r+') as mesh:
        mesh.require_group('partitionings')

        con, ecurved, edisps, _ = BasePartitioner.construct_global_con(mesh)
        ntotal = int(len(ecurved))
        vparts = np.empty(ntotal, dtype=np.int32)

        # For each etype, assign its global element range to allowed ranks
        for etype, disp in edisps.items():
            einfo = mesh['eles'][etype]['curved', 'faces'][()]
            n = int(len(einfo))
            s = int(disp)
            e = s + n

            if etype not in etype_to_device:
                raise ValueError(f"[startup-etypes] missing etype_to_device for etype={etype!r}")

            dev = str(etype_to_device[etype])
            ranks = dev_to_ranks.get(dev, [])
            if not ranks:
                raise ValueError(f"[startup-etypes] no ranks for device tag {dev!r} (etype={etype!r})")

            # Balanced assignment within the allowed ranks
            idx = np.arange(s, e, dtype=np.int64)
            rng.shuffle(idx)
            ranks = np.asarray(ranks, dtype=np.int32)
            vparts[idx] = ranks[np.arange(idx.size, dtype=np.int64) % ranks.size]

        # Construct + write partitioning using existing PyFR logic
        pinfo = BasePartitioner.construct_partitioning(mesh, ecurved, edisps, con, vparts)
        write_partitioning(mesh, pname, pinfo)

    return pname

class _StartupInitRandomEtypes(_StartupInitBase):
    name = 'random-etypes'

    def prepare_pname(self, *, mesh_path: str, compute_pname: str) -> str:
        if comm['compute'].size != comm['world'].size:
            raise NotImplementedError("[startup-etypes] compute comm != world comm not supported yet")

        seed = 2079
        if self.cfg is not None and self.cfg.hasopt('partition', 'startup-rand-seed'):
            seed = self.cfg.getint('partition', 'startup-rand-seed')

        # Use MPICommInfo if available; fallback to cfg
        from pyfr.mpiutil import comm_rank_roots
        winfo = comm_rank_roots.get('world')
        devices = None if winfo is None else winfo.devices_world
        if devices is None and self.cfg is not None and self.cfg.hasopt('backend', 'devices'):
            devices = self.cfg.getliteral('backend', 'devices')
        if devices is None:
            raise ValueError("[startup-etypes] no devices list found")

        if self.cfg is None or not self.cfg.hasopt('partition', 'startup-etype-to-device'):
            raise ValueError("[startup-etypes] missing [partition] startup-etype-to-device mapping")
        et2d = self.cfg.getliteral('partition', 'startup-etype-to-device')

        if rank['world'] == root['world']:
            print(f"[startup-etypes] etype_to_device={et2d}", flush=True)
            tmp = _write_startup_etypes_partitioning(mesh_path, comm['compute'].size,
                                                    seed=seed, devices_world=devices,
                                                    etype_to_device=et2d)
            print(f"[startup-etypes] wrote partitioning {tmp!r}", flush=True)
        else:
            tmp = None

        compute_pname = comm['world'].bcast(tmp, root=root['world'])
        comm['world'].barrier()
        return str(compute_pname)

class _StartupInitContiguousEtypes(_StartupInitRandomEtypes):
    name = 'contiguous-etypes'

    def postprocess(self, *, mmesh, partitioner: str) -> None:
        if partitioner != 'diffusion':
            raise NotImplementedError("[startup-contig] contiguous-etypes requires partitioner=diffusion")
        mmesh.startup_make_contiguous(max_iters=100, maintain_cluster_by_device_types=True)


