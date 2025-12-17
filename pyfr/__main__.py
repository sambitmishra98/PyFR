#!/usr/bin/env python
from argparse import ArgumentParser, FileType
import csv
import io
from pathlib import Path
import re

import h5py
import mpi4py.rc
import numpy as np
mpi4py.rc.initialize = False

from pyfr._version import __version__
from pyfr.backends import BaseBackend, get_backend
from pyfr.inifile import Inifile
from pyfr.mpiutil import (get_comm_rank_root, init_mpi, initialise_new_comm,
                          comm, rank, root)
from pyfr.partitioners import (BasePartitioner, get_partitioner,
                               reconstruct_partitioning, write_partitioning,
                               reconstruct_by_diffusion)
from pyfr.plugins import BaseCLIPlugin
from pyfr.progress import (NullProgressSequence, ProgressBar,
                           ProgressSequenceAction)
from pyfr.readers import BaseReader, get_reader_by_name, get_reader_by_extn
from pyfr.readers.native import NativeReader
from pyfr.partitioners.online.base import _MetaMesh
from pyfr.readers.stl import read_stl
from pyfr.resamplers import (BaseInterpolator, NativeCloudResampler,
                             get_interpolator)
from pyfr.solvers import get_solver
from pyfr.util import first, subclasses
from pyfr.writers import BaseWriter, get_writer_by_extn, get_writer_by_name
from pyfr.writers.native import NativeWriter


def main():
    ap = ArgumentParser(prog='pyfr')
    sp = ap.add_subparsers(help='sub-command help', metavar='command')

    # Common options
    ap.add_argument('-v', '--verbose', action='count',
                    help='increase verbosity')
    ap.add_argument('-V', '--version', action='version',
                    version=f'%(prog)s {__version__}')
    ap.add_argument('-p', '--progress', action=ProgressSequenceAction,
                    help='show progress')

    # Import command
    ap_import = sp.add_parser('import', help='import --help')
    ap_import.add_argument('inmesh', type=FileType('r'),
                           help='input mesh file')
    ap_import.add_argument('outmesh', help='output PyFR mesh file')
    types = sorted(cls.name for cls in subclasses(BaseReader))
    ap_import.add_argument('-t', dest='type', choices=types,
                           help='input file type; this is usually inferred '
                           'from the extension of inmesh')
    ap_import.add_argument('-l', dest='lintol', type=float, default=1e-5,
                           help='linearisation tolerance')
    ap_import.set_defaults(process=process_import)

    # Partition subcommand
    ap_partition = sp.add_parser('partition', help='partition --help')
    ap_partition = ap_partition.add_subparsers()

    # List partitionings
    ap_partition_list = ap_partition.add_parser('list',
                                                help='partition list --help')
    ap_partition_list.add_argument('mesh', help='input mesh file')
    ap_partition_list.add_argument('-s', '--sep', default='\t',
                                   help='separator')
    ap_partition_list.set_defaults(process=process_partition_list)

    # Get info about a partitioning
    ap_partition_info = ap_partition.add_parser('info',
                                                help='partition info --help')
    ap_partition_info.add_argument('mesh', help='input mesh file')
    ap_partition_info.add_argument('name', help='partitioning name')
    ap_partition_info.add_argument('-s', '--sep', default='\t',
                                   help='separator')
    ap_partition_info.set_defaults(process=process_partition_info)

    # Add partitioning
    ap_partition_add = ap_partition.add_parser('add',
                                               help='partition add --help')
    ap_partition_add.add_argument('mesh', help='input mesh file')
    ap_partition_add.add_argument('np', help='number of partitions or a colon '
                                  'delimited list of weights')
    ap_partition_add.add_argument('name', nargs='?', help='partitioning name')
    ap_partition_add.add_argument('-f', '--force', action='count',
                                  help='overwrite existing partitioning')
    partitioners = sorted(cls.name for cls in subclasses(BasePartitioner))
    ap_partition_add.add_argument(
        '-p', dest='partitioner', choices=partitioners,
        help='partitioner to use'
    )
    ap_partition_add.add_argument(
        '-e', dest='elewts', action='append', default=[],
        metavar='shape:weight', help='element weighting factor or "balanced"'
    )
    ap_partition_add.add_argument(
        '--popt', dest='popts', action='append', default=[],
        metavar='key:value', help='partitioner-specific option'
    )
    ap_partition_add.set_defaults(process=process_partition_add)

    # Construct partitioning
    ap_partition_construct = ap_partition.add_parser(
        'construct', help='partition construct --help'
    )
    ap_partition_construct.add_argument('mesh', help='input mesh file')
    ap_partition_construct.add_argument('name', help='partitioning name')
    ap_partition_construct.add_argument('-f', '--force', action='count',
                                        help='overwrite existing partitioning')
    ap_partition_construct.add_argument('--seed', type=int, default=-1,
                                    help='RNG seed (-1: nondeterministic)')

    ap_partition_construct.set_defaults(process=process_partition_construct)

    # Reconstruct partitioning
    ap_partition_reconstruct = ap_partition.add_parser(
        'reconstruct', help='partition reconstruct --help'
    )
    ap_partition_reconstruct.add_argument('mesh', help='input mesh file')
    ap_partition_reconstruct.add_argument('soln', help='input solution file')
    ap_partition_reconstruct.add_argument('name', help='partitioning name')
    ap_partition_reconstruct.add_argument(
        '-f', '--force', action='count', help='overwrite existing partitioning'
    )
    ap_partition_reconstruct.set_defaults(
        process=process_partition_reconstruct
    )

    # Reconstruct partitioning by diffusion from an existing partitioning
    ap_partition_diffuse = ap_partition.add_parser(
        'diffuse', help='partition diffuse --help'
    )

    ap_partition_diffuse.add_argument('mesh', help='input mesh file')
    ap_partition_diffuse.add_argument('np', 
                                      help='number of partitions or a colon '
                                           'delimited list of weights')
    ap_partition_diffuse.add_argument('name', help='Existing partitioning name')
    ap_partition_diffuse.add_argument('dpname', help='Diffused partitioning name')
    ap_partition_diffuse.add_argument(
        '-f', '--force', action='count', help='overwrite existing partitioning'
    )

    ap_partition_diffuse.set_defaults(process=process_partition_diffuse)

    # Remove partitioning
    ap_partition_remove = ap_partition.add_parser(
        'remove', help='partition remove --help'
    )
    ap_partition_remove.add_argument('mesh', help='input mesh file')
    ap_partition_remove.add_argument('name', help='partitioning')
    ap_partition_remove.set_defaults(process=process_partition_remove)

    # Export command
    ap_export = sp.add_parser('export', help='export --help')
    ap_export = ap_export.add_subparsers()

    for etype in ('boundary', 'stl', 'volume'):
        ap_export_type = ap_export.add_parser(etype,
                                              help=f'export {etype} --help')

        ap_export_type.add_argument('meshf', help='input mesh file')
        ap_export_type.add_argument('solnf', help='input solution file')
        ap_export_type.add_argument('outf', help='output file')

        if etype == 'boundary':
            ap_export_type.add_argument('eargs', nargs='+', metavar='boundary',
                                        help='boundary to output')
        elif etype == 'stl':
            ap_export_type.add_argument('eargs', nargs='+', metavar='stl',
                                        help='STL region to output')

        ap_export_type.add_argument('-b', '--batchfile', type=FileType('r'),
                                    default='-', help='batch export file')

        ftypes = [c.name for c in subclasses(BaseWriter) if c.type == etype]
        ap_export_type.add_argument(
            '-t', dest='ftype', choices=ftypes, required=False,
            help='output file type; this is usually inferred from the '
            'extension of outf'
        )
        ap_export_type.add_argument(
            '-f', '--field', dest='fields', action='append', metavar='FIELD',
            help='what fields should be output; may be repeated, by default '
            'all fields are output'
        )
        ap_export_type.add_argument(
            '-p', '--precision', choices=['single', 'double'],
            default='single', help='output number precision; defaults to '
            'single'
        )
        ap_export_type.add_argument(
            '--eopt', dest='eopts', action='append', default=[],
            metavar='key:value', help='exporter-specific option'
        )
        ap_export_type.add_argument('-P', '--pname',
                                    help='partitioning to use')
        ap_export_type.set_defaults(etype=etype, process=process_export)

    # Region subcommand
    ap_region = sp.add_parser('region', help='region --help')
    ap_region = ap_region.add_subparsers()

    # Add region
    ap_region_add = ap_region.add_parser('add', help='region add --help')
    ap_region_add.add_argument('mesh', help='input mesh file')
    ap_region_add.add_argument('stl', type=FileType('rb'), help='STL file')
    ap_region_add.add_argument('name', help='region name')
    ap_region_add.set_defaults(process=process_region_add)

    # List regions
    ap_region_list = ap_region.add_parser('list', help='region list --help')
    ap_region_list.add_argument('mesh', help='input mesh file')
    ap_region_list.add_argument('-s', '--sep', default='\t', help='separator')
    ap_region_list.set_defaults(process=process_region_list)

    # Remove region
    ap_region_remove = ap_region.add_parser('remove',
                                            help='region remove --help')
    ap_region_remove.add_argument('mesh', help='input mesh file')
    ap_region_remove.add_argument('name', help='region name')
    ap_region_remove.set_defaults(process=process_region_remove)

    # Resample command
    ap_resample = sp.add_parser('resample', help='resample --help')
    ap_resample.add_argument('srcmesh', help='source mesh file')
    ap_resample.add_argument('srcsoln', help='source solution file')
    ap_resample.add_argument('tgtmesh', help='target mesh file')
    ap_resample.add_argument('tgtcfg', help='target config file')
    ap_resample.add_argument('tgtsoln', help='target solution file')
    itypes = [i.name for i in subclasses(BaseInterpolator, just_leaf=True)]
    ap_resample.add_argument('-i', '--interpolator', choices=itypes,
                             required=True, help='interpolator to use')
    ap_resample.add_argument(
        '--iopt', dest='iopts', action='append', default=[],
        metavar='key:value', help='interpolator-specific option'
    )
    ap_resample.add_argument('-P', '--pname', help='partitioning to use')
    ap_resample.set_defaults(process=process_resample)

    # Run command
    ap_run = sp.add_parser('run', help='run --help')
    ap_run.add_argument('mesh', help='mesh file')
    ap_run.add_argument('cfg', type=FileType('r'), help='config file')
    ap_run.set_defaults(process=process_run)

    # Restart command
    ap_restart = sp.add_parser('restart', help='restart --help')
    ap_restart.add_argument('mesh', help='mesh file')
    ap_restart.add_argument('soln', help='solution file')
    ap_restart.add_argument('cfg', nargs='?', type=FileType('r'),
                            help='new config file')
    ap_restart.set_defaults(process=process_restart)

    # Options common to run and restart
    backends = sorted(cls.name for cls in subclasses(BaseBackend))
    for p in [ap_run, ap_restart]:
        p.add_argument('-b', '--backend', choices=backends, required=True,
                       help='backend to use')
        p.add_argument('-P', '--pname', help='partitioning to use')

    # Plugin commands
    for scls in subclasses(BaseCLIPlugin, just_leaf=True):
        scls.add_cli(sp.add_parser(scls.name, help=f'{scls.name} --help'))

    # Parse the arguments
    args = ap.parse_args()

    # Invoke the process method
    if hasattr(args, 'process'):
        args.process(args)
    else:
        ap.print_help()


def process_import(args):
    # Get a suitable mesh reader instance
    if args.type:
        reader = get_reader_by_name(args.type, args.inmesh, args.progress)
    else:
        extn = Path(args.inmesh.name).suffix
        reader = get_reader_by_extn(extn, args.inmesh, args.progress)

    # Write out the mesh
    reader.write(args.outmesh, args.lintol)


def process_partition_list(args):
    with h5py.File(args.mesh, 'r') as mesh:
        print('name', 'parts', sep=args.sep)

        for name, part in sorted(mesh['partitionings'].items()):
            nparts = len(part['eles'].attrs['regions'])
            print(name, nparts, sep=args.sep)


def process_partition_info(args):
    with h5py.File(args.mesh, 'r') as mesh:
        # Read the partition region info from the mesh
        regions = mesh[f'partitionings/{args.name}/eles'].attrs['regions']

        # Print out the header
        print('part', *mesh['eles'], sep=args.sep)

        # Compute and output the number of elements in each partition
        for i, neles in enumerate(regions[:, 1:] - regions[:, :-1]):
            print(i, *neles, sep=args.sep)


def process_partition_add(args):
    with h5py.File(args.mesh, 'r+') as mesh:
        # Determine the element types
        etypes = list(mesh['eles'])

        # Partition weights
        if ':' in args.np or '*' in args.np:
            def psub(m): return ':'.join([m[1]]*int(m[2]))
            pwts = re.sub(r'(\d+)\*(\d+)', psub, args.np)
            pwts = [int(w) for w in pwts.split(':')]
        else:
            pwts = [1]*int(args.np)

        # Element weights
        if args.elewts == ['balanced']:
            ewts = None
        elif len(etypes) == 1:
            ewts = {etypes[0]: 1}
        else:
            ewts = (ew.split(':') for ew in args.elewts)
            ewts = {e: int(w) for e, w in ewts}

        # Ensure all weights have been provided
        if ewts is not None and len(ewts) != len(etypes):
            missing = ', '.join(set(etypes) - set(ewts))
            raise ValueError(f'Missing element weights for: {missing}')

        # Get the partitioning name
        pname = args.name or str(len(pwts))
        if not re.match(r'\w+$', pname):
            raise ValueError('Invalid partitioning name')

        # Check it does not already exist unless --force is given
        if pname in mesh['partitionings'] and not args.force:
            raise ValueError('Partitioning already exists; use -f to replace')

        # Partitioner-specific options
        opts = dict(s.split(':', 1) for s in args.popts)

        # Create the partitioner
        if args.partitioner:
            part = get_partitioner(args.partitioner, pwts, ewts, opts=opts)
        else:
            parts = sorted(cls.name for cls in subclasses(BasePartitioner))
            for name in parts:
                try:
                    part = get_partitioner(name, pwts, ewts)
                    break
                except OSError:
                    pass
            else:
                raise RuntimeError('No partitioners available')

        # Partition the mesh
        pinfo = part.partition(mesh, args.progress)

        # Write out the new partitioning
        with args.progress.start('Write partitioning'):
            write_partitioning(mesh, pname, pinfo)



def _counts_from_edisps(edisps: dict[str, int], ntotal: int) -> dict[str, int]:
    # edisps preserves insertion order from construct_global_con (sorted etypes)
    ets = list(edisps.keys())
    ds  = [edisps[et] for et in ets]
    cnt = {}
    for k, et in enumerate(ets):
        d0 = ds[k]
        d1 = ds[k + 1] if k + 1 < len(ds) else ntotal
        cnt[et] = int(d1 - d0)
    return cnt


def _random_vparts_nonempty(n: int, nparts: int, *, seed: int) -> np.ndarray:
    """
    Return vparts (int32, shape (n,)) with all parts non-empty if n >= nparts.
    If n < nparts, empties are unavoidable; we allow them.
    """
    rng = np.random.default_rng(None if seed < 0 else seed)

    if nparts <= 1:
        return np.zeros(n, dtype=np.int32)

    if n == 0:
        return np.empty(0, dtype=np.int32)

    if n < nparts:
        # Not enough elements to populate every part; accept empties.
        return rng.integers(0, nparts, size=n, dtype=np.int32)

    # Reroll until every part appears at least once (simple + robust for now)
    # In practice, success probability is very high once n >> nparts.
    while True:
        v = rng.integers(0, nparts, size=n, dtype=np.int32)
        if np.unique(v).size == nparts:
            return v


def _vparts_to_scatter_eidxs(
    vparts: np.ndarray,
    etypes_order: list[str],
    edisps: dict[str, int],
    ecnts: dict[str, int],
    nparts: int,
) -> list[dict[str, np.ndarray]]:
    """
    Convert global vparts into a list (len=nparts) of per-rank dicts:
      out[r][etype] = gids (int64) owned by rank r for that etype.

    Uses an argsort+split method per etype (O(N log N) per etype) but avoids
    O(nparts*N) loops.
    """
    out = [{et: np.empty(0, dtype=np.int64) for et in etypes_order} for _ in range(nparts)]

    for et in etypes_order:
        d0 = int(edisps[et])
        ne = int(ecnts[et])
        if ne == 0:
            continue

        parts_et = np.asarray(vparts[d0:d0 + ne], dtype=np.int32)

        # Sort element gids by owning part id
        order = np.argsort(parts_et, kind='mergesort')
        p_sorted = parts_et[order]

        # Split positions for each rank in [0, nparts)
        # idx[r]..idx[r+1] in `order` are gids with part == r
        cuts = np.searchsorted(p_sorted, np.arange(1, nparts, dtype=np.int32))
        blocks = np.split(order, cuts)

        for r in range(nparts):
            gids_r = blocks[r].astype(np.int64, copy=False)
            out[r][et] = gids_r

    return out


def _pick_local_seed(eidxs_local: dict[str, np.ndarray], *, seed: int, rank: int) -> tuple[str, int] | None:
    """
    Pick one (etype, gid) from local ownership for printing.
    Returns None if the rank has no elements.
    """
    ets = [et for et, a in eidxs_local.items() if a is not None and len(a) > 0]
    if not ets:
        return None

    # Deterministic across ranks if seed >= 0
    if seed < 0:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed + 1_000_003 * int(rank))

    et = ets[int(rng.integers(0, len(ets)))]
    gids = np.asarray(eidxs_local[et], dtype=np.int64)
    gid = int(gids[int(rng.integers(0, gids.size))])

    return (et, gid)


def process_partition_construct(args):
    # Validate partitioning name
    if not re.match(r'\w+$', args.name):
        raise ValueError('Invalid partitioning name')

    init_mpi()
    comm, rank, root = get_comm_rank_root()
    nparts = int(comm.size)

    # ----------------------------
    # Root: read mesh + make vparts
    # ----------------------------
    err = None
    payload = None

    if rank == root:
        try:
            with h5py.File(args.mesh, 'r') as mesh:
                # Ensure partitionings group exists
                if 'partitionings' not in mesh:
                    raise ValueError("Mesh is missing 'partitionings' group")

                # Refuse overwrite unless --force
                if args.name in mesh['partitionings'] and not args.force:
                    raise ValueError('Partitioning already exists; use -f to replace')

                # Build what PyFR needs to write a proper partitioning
                con, ecurved, edisps, _ = BasePartitioner.construct_global_con(mesh)

                etypes_order = list(edisps.keys())
                ntotal = int(len(ecurved))
                ecnts = _counts_from_edisps(edisps, ntotal)

                # Phase 0: random partition
                vparts = _random_vparts_nonempty(ntotal, nparts, seed=args.seed)

                # Scatter payload: per-rank per-etype gids
                scatter_eidxs = _vparts_to_scatter_eidxs(vparts, etypes_order, edisps, ecnts, nparts)

                payload = dict(
                    con=con,
                    ecurved=ecurved,
                    edisps=edisps,
                    etypes_order=etypes_order,
                    ecnts=ecnts,
                    vparts=vparts,
                    scatter_eidxs=scatter_eidxs,
                )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

    # Broadcast errors to avoid deadlocks
    err = comm.bcast(err, root=root)
    if err:
        raise RuntimeError(err)

    # Broadcast small metadata
    etypes_order = comm.bcast(payload['etypes_order'] if rank == root else None, root=root)
    edisps       = comm.bcast(payload['edisps']       if rank == root else None, root=root)

    # Scatter local eidxs ownership to every rank
    eidxs_local = comm.scatter(payload['scatter_eidxs'] if rank == root else None, root=root)

    # ----------------------------
    # All ranks: pick local seed, gather and print on root
    # ----------------------------
    seed_local = _pick_local_seed(eidxs_local, seed=args.seed, rank=rank)
    seeds = comm.gather(seed_local, root=root)

    if rank == root:
        print(f"[construct] nparts={nparts} name={args.name!r} seed={args.seed}", flush=True)
        for r, s in enumerate(seeds):
            if s is None:
                print(f"[construct.seed] r{r}: <empty>", flush=True)
            else:
                et, gid = s
                eid = int(edisps[et] + int(gid))
                print(f"[construct.seed] r{r}: etype={et} gid={gid} eid={eid}", flush=True)

    comm.barrier()

    # ----------------------------
    # Root: write the partitioning
    # ----------------------------
    if rank == root:
        con     = payload['con']
        ecurved = payload['ecurved']
        edisps2 = payload['edisps']
        vparts  = payload['vparts']

        with h5py.File(args.mesh, 'r+') as mesh:
            # Re-check overwrite (race-safe)
            if args.name in mesh['partitionings'] and not args.force:
                raise ValueError('Partitioning already exists; use -f to replace')

            pinfo = BasePartitioner.construct_partitioning(mesh, ecurved, edisps2, con, vparts)
            write_partitioning(mesh, args.name, pinfo)

            print(f"[construct] wrote partitionings/{args.name}", flush=True)

    comm.barrier()

def process_partition_reconstruct(args):
    with (h5py.File(args.mesh, 'r+') as mesh,
          h5py.File(args.soln, 'r') as soln):
        # Validate the partitioning name
        if not re.match(r'\w+$', args.name):
            raise ValueError('Invalid partitioning name')

        # Check it does not already exist unless --force is given
        if args.name in mesh['partitionings'] and not args.force:
            raise ValueError('Partitioning already exists; use -f to replace')

        # Reconstruct the partitioning used in the solution
        pinfo = reconstruct_partitioning(mesh, soln, args.progress)

        # Write out the new partitioning
        with args.progress.start('Write partitioning'):
            write_partitioning(mesh, args.name, pinfo)


def process_partition_diffuse(args):
    # Validate the partitioning name
    if not re.match(r'\w+$', args.name):
        raise ValueError('Invalid partitioning name')

    init_mpi()

    # If running with n MPI ranks and mesh args.name has m partitions, then
    # create new comms with m ranks each
    comm, rank, root = get_comm_rank_root()

    etypes = []
    pwts   = []
    if rank == root:
        with (h5py.File(args.mesh, 'r+') as mesh):
            etypes = list(mesh['eles'])

            # Partition weights
            if ':' in args.np or '*' in args.np:
                def psub(m): return ':'.join([m[1]]*int(m[2]))
                pwts = re.sub(r'(\d+)\*(\d+)', psub, args.np)
                pwts = [int(w) for w in pwts.split(':')]
            else:
                pwts = [1]*int(args.np)

    etypes = comm.bcast(etypes, root=root)
    pwts   = comm.bcast(pwts  , root=root)

    if comm.size < 2:
        raise ValueError('Diffusion is meaningless for nranks > 1.')

    reader = NativeReader(args.mesh, pname=args.name)
    read_only_mesh = reader.mesh
    reader.close()

    # Reconstruct the partitioning used in the solution
    vparts = reconstruct_by_diffusion(read_only_mesh, pwts, args.progress)

    if rank == root:
        with args.progress.start('Repartition'):
            if rank == root:
                with (h5py.File(args.mesh, 'r+') as mesh):
                    # Check it does not already exist unless --force is given
                    if args.dpname in mesh['partitionings'] and not args.force:
                        raise ValueError('Partitioning already exists; use -f to replace')

                    con, ecurved, edisps, _ = BasePartitioner.construct_global_con(mesh)

                    pinfo = BasePartitioner.construct_partitioning(mesh, ecurved, edisps,
                                                                con, vparts)

                    # Write out the new partitioning
                    with args.progress.start('Write partitioning'):
                            write_partitioning(mesh, args.dpname, pinfo)

def process_partition_remove(args):
    with h5py.File(args.mesh, 'r+') as mesh:
        mparts = mesh['partitionings']

        if args.name not in mparts:
            raise ValueError(f'Partitioning {args.name} does not exist')

        del mparts[args.name]


def process_region_add(args):
    # Read the STL file
    stl = read_stl(args.stl)

    # See if the surface is closed
    edges = np.vstack([stl[:, 1:3], stl[:, 2:4], stl[:, [3, 1]]])
    edges.view('f4,f4,f4').sort(axis=1)
    closed = (np.unique(edges, axis=0, return_counts=True)[1] == 2).all()

    # Validate the name
    if not re.match(r'\w+$', args.name):
        raise ValueError('Invalid region name')

    with h5py.File(args.mesh, 'r+') as mesh:
        g = mesh.require_group('regions/stl')

        if args.name in g:
            del g[args.name]

        g[args.name] = stl
        g[args.name].attrs['closed'] = closed


def process_region_list(args):
    with h5py.File(args.mesh, 'r') as mesh:
        print('name', 'tris', 'closed', sep=args.sep)

        for name, tris in sorted(mesh.get('regions/stl', {}).items()):
            print(name, len(tris), str(tris.attrs['closed']).lower(),
                  sep=args.sep)


def process_region_remove(args):
    with h5py.File(args.mesh, 'r+') as mesh:
        rparts = mesh.get('regions/stl')

        if rparts is None or args.name not in rparts:
            raise ValueError(f'Region {args.name} does not exist')

        del rparts[args.name]


def process_export(args):
    # Manually initialise MPI
    init_mpi()

    comm, rank, root = get_comm_rank_root()

    # Common arguments
    kargs = [args.eargs] if 'eargs' in args else []
    kwargs = {'fields': args.fields, 'prec': args.precision,
              'pname': args.pname}

    # Process any exporter-specific options
    for e in args.eopts:
        k, v = e.split(':', 1)
        kwargs[k] = int(v) if re.match(r'\d+', v) else v

    # Obtain files to export from a batch file
    if args.solnf == '-' and args.outf == '-':
        if rank == root:
            batch = args.batchfile.read()

            dialect = csv.Sniffer().sniff(batch)
            batch = csv.reader(io.StringIO(batch), dialect=dialect)
            batch = comm.bcast([l for l in batch if l], root=root)
        else:
            batch = comm.bcast(None, root=root)
    # Obtain files to export from the command line
    else:
        batch = [[args.solnf, args.outf]]

    # Get writer instance by specified type or output file extension
    if args.ftype:
        writer = get_writer_by_name(args.ftype, args.etype, args.meshf,
                                    *kargs, **kwargs)
    else:
        extn = Path(batch[0][1]).suffix
        writer = get_writer_by_extn(extn, args.etype, args.meshf, *kargs,
                                    **kwargs)

    # Process the files
    progress = args.progress if rank == root else NullProgressSequence()
    with progress.start_with_bar('Process solutions') as pbar:
        for solnf, outf in pbar.start_with_iter(batch):
            writer.process(solnf, outf)


def process_resample(args):
    # Manually initialise MPI
    init_mpi()

    comm, rank, root = get_comm_rank_root()
    progress = args.progress if rank == root else NullProgressSequence()

    with progress.start('Load source and target meshes'):
        # Load the source mesh
        sreader = NativeReader(args.srcmesh, args.pname, construct_con=False)
        smesh, ssoln = sreader.load_subset_mesh_soln(args.srcsoln)

        # Load the target mesh and config file
        treader = NativeReader(args.tgtmesh, args.pname, construct_con=False)
        tcfg = Inifile.load(args.tgtcfg)

    # Get the interpolator
    opts = dict(s.split(':', 1) for s in args.iopts)
    interp = get_interpolator(args.interpolator, smesh.ndims, opts)

    # Perform the resampling
    resampler = NativeCloudResampler(smesh, ssoln, interp, progress)
    tsoln = resampler.sample_with_mesh_config(treader.mesh, tcfg)
    tshapes = {k: v.shape[1:] for k, v in tsoln.items()}
    fpdtype = first(tsoln.values()).dtype

    # Get the output file path
    tpath = Path(args.tgtsoln).absolute()

    # Get the data field prefix
    prefix = ssoln['stats'].get('data', 'prefix')

    # Have the root rank prepare a stats record
    if rank == root:
        stats = Inifile()
        stats.set('data', 'prefix', prefix)
        stats.set('data', 'fields', ssoln['stats'].get('data', 'fields'))
        stats.set('solver-time-integrator', 'tcurr',
                  ssoln['stats'].get('solver-time-integrator', 'tcurr'))
        metadata = {'config': tcfg.tostr(), 'stats': stats.tostr(),
                    'mesh-uuid': treader.mesh.uuid}
    else:
        metadata = None

    with progress.start('Write target solution'):
        # Write out the new solution
        writer = NativeWriter(treader.mesh, tcfg, fpdtype, tpath.parent,
                              tpath.name, prefix)
        writer.set_shapes_eidxs(tshapes, treader.mesh.eidxs)
        writer.write(tsoln, None, metadata)


def _process_common(args, soln, cfg):
    # Manually initialise MPI
    init_mpi(cfg)

    # If cfg provides a partitioning ranklist, use it
    if cfg is not None and cfg.hasopt('partition', 'compute-ranklist'):
        part_ranklist = cfg.getliteral('partition', 'compute-ranklist')
    else:
        part_ranklist = list(range(comm['world'].size))

    initialise_new_comm('compute', part_ranklist)

    compute_pname = None
    
    if cfg.hasopt('partition', 'compute-pname'):
        compute_pname = cfg.get('partition', 'compute-pname')
    
    # Read the mesh
    reader = NativeReader(args.mesh, pname=compute_pname,
                          comm_name='compute')
    mesh = reader.mesh
    mmesh = _MetaMesh(mesh, cfg)

    # Load a provided solution, if any
    if soln is not None:
        soln = reader.load_soln(soln)

    # If we do not have a config file then take it from the solution
    if cfg is None:
        cfg = soln['config']

    # Create a backend
    backend = get_backend(args.backend, cfg)

    # Construct the solver
    solver = get_solver(backend, mmesh, soln, cfg)

    # If we are running interactively then create a progress bar
    if args.progress and rank['world'] == root['world']:
        pbar = ProgressBar()
        pbar.start(solver.tend, start=solver.tstart, curr=solver.tcurr)

        # Register a callback to update the bar after each step
        solver.plugins.append(lambda intg: pbar(intg.tcurr))

    # Execute!
    solver.run()


def process_run(args):
    _process_common(args, None, Inifile.load(args.cfg))


def process_restart(args):
    cfg = Inifile.load(args.cfg) if args.cfg else None
    _process_common(args, args.soln, cfg)


if __name__ == '__main__':
    main()
