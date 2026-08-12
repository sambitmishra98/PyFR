"""Offline diffusion-based repartitioning for the new fork.

`newversion-rebased` can only diffuse ONLINE, from inside a running solver
(RebalanceMixin). The old monolith had a `pyfr partition diffuse` CLI; this
restores the equivalent so a diffused partitioning can be produced -- and
timed -- without running any physics at all.

Design mirrors the old branch's process_partition_diffuse:
  * runs under `mpirun -np P`, one rank per target partition
  * reads the mesh under an EXISTING partitioning (the starting point)
  * runs the diffusion schedule toward a target element distribution
  * writes the result back as a new named partitioning

Deliberate difference from the online path: the target here comes from the
requested weights, NOT from a measured per-rank cost model, because offline
there are no solver timings to measure. That makes the run deterministic and
repeatable, which is what we want for a timing comparison against METIS.

This module is imported by pyfr/__main__.py; it is kept separate so the
addition to __main__ stays small.
"""
import re
import time

import h5py
import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, init_mpi, mpi
from pyfr.partitioners.base import BasePartitioner, write_partitioning
from pyfr.readers.native import NativeReader
from pyfr.rebalance import DiffusionBalancer, IndexMesh, int_round


def _parse_weights(spec, nparts):
    # Same syntax as `partition add`: a count, or a colon list, with n*k
    # repetition -- but here the count must match the launched rank count.
    if spec in (None, '-'):
        return [1]*nparts

    if ':' in spec or '*' in spec:
        def psub(m):
            return ':'.join([m[1]]*int(m[2]))

        wts = re.sub(r'(\d+)\*(\d+)', psub, spec)
        wts = [int(w) for w in wts.split(':')]
    else:
        n = int(spec)
        if n != nparts:
            raise ValueError(
                f'partition diffuse: np={n} but launched with {nparts} ranks; '
                f'run under `mpirun -np {n}`'
            )
        wts = [1]*n

    if len(wts) != nparts:
        raise ValueError(
            f'partition diffuse: {len(wts)} weights given but launched with '
            f'{nparts} ranks'
        )

    return wts


def _cut(im, comm):
    # Global count of partition-crossing face sides. Each cut face is counted
    # once from each side, so this is 2x the true edge cut; it is used only as
    # a relative quality metric, and the factor is constant.
    #
    # COLLECTIVE: every rank must call this the same number of times. Never
    # put it behind a `rank == root` guard (that bug cost an MPI_ERR_TRUNCATE
    # hunt earlier).
    return int(comm.allreduce(im.mpi_out(), op=mpi.SUM))


def _islands(bal, im):
    # Max connected components owned by any one rank. 1 per rank is ideal;
    # more means a rank owns disconnected blobs. COLLECTIVE via detect_islands.
    return int(max(bal.detect_islands(im)[1]))


def _partitioning_nparts(mesh, pname):
    if pname not in mesh['partitionings']:
        raise ValueError(f'Partitioning {pname!r} does not exist')

    # NB: nparts == len(regions), not len(regions) - 1 -- matching
    # process_partition_list in __main__.py. (Got this wrong first time.)
    return len(mesh[f'partitionings/{pname}/eles'].attrs['regions'])


def process_partition_diffuse(args):
    # The `partition` CLI path does not initialise MPI (only run/restart do),
    # and this command is inherently parallel -- one rank per partition.
    init_mpi()

    comm, rank, root = get_comm_rank_root()
    nparts = comm.size

    if nparts < 2:
        raise ValueError('Diffusion is meaningless for fewer than 2 ranks')

    if not re.match(r'\w+$', args.dpname):
        raise ValueError('Invalid partitioning name')

    # Validate on root, then fan out, so a bad request fails identically on
    # every rank instead of deadlocking a subset of them
    err, pwts = None, None
    if rank == root:
        try:
            pwts = _parse_weights(args.np, nparts)

            with h5py.File(args.mesh, 'r') as mesh:
                have = _partitioning_nparts(mesh, args.name)
                if have != nparts:
                    raise ValueError(
                        f'Initial partitioning {args.name!r} has {have} parts '
                        f'but this job has {nparts} ranks'
                    )

                if args.dpname in mesh['partitionings'] and not args.force:
                    raise ValueError('Partitioning already exists; use -f '
                                     'to replace')
        except Exception as e:
            err = f'{type(e).__name__}: {e}'

    err = comm.bcast(err, root=root)
    if err:
        raise RuntimeError(err)

    pwts = comm.bcast(pwts, root=root)

    # A minimal cfg so DiffusionBalancer can read its options; every knob has
    # a default, so an empty section is a valid "stock behaviour" request.
    # Inifile has no hassect(); its get*() with a default adds the section on
    # demand, and every DiffusionBalancer knob has a default, so an ini
    # without a [diffuse] section simply means "stock behaviour".
    cfg = Inifile(args.cfg.read() if args.cfg else '[diffuse]\n')
    sect = 'diffuse'

    t0 = time.perf_counter()

    reader = NativeReader(args.mesh, pname=args.name)
    mesh = reader.mesh
    t_read = time.perf_counter() - t0

    # Constructed BEFORE the IndexMesh: it knows, from the schedule and the
    # outlier/inlier settings, which index tables will ever be read, and the
    # unread ones are the bulk of both the build and the migration cost.
    bal = DiffusionBalancer(cfg, sect)

    t1 = time.perf_counter()
    im = IndexMesh(mesh, vaff=bal.needs_vaff, geom=bal.needs_geom)
    t_im = time.perf_counter() - t1

    # Release the read-only handle before root reopens the file 'r+' below,
    # otherwise h5py raises "file is already open for read-only".
    reader.close()

    # Target element counts from the requested weights
    target = int_round(np.array(pwts, dtype=float), im.nglobal)

    start = im.counts()
    cut = cut0 = _cut(im, comm)
    isl = isl0 = _islands(bal, im)

    # t_bal accumulates ONLY the balancer itself; t_diffuse additionally
    # covers this loop's diagnostics (two counts() allgathers and a _cut()
    # allreduce per cycle). Measured: the gap is under 1%, so the per-cycle
    # cut reporting is free and can stay on by default. Both are reported so
    # nobody has to take that on trust.
    t_bal = 0.0

    t2 = time.perf_counter()
    for i in range(args.niters):
        before = im.counts()

        tb = time.perf_counter()
        bal.balance(im, target)
        t_bal += time.perf_counter() - tb

        # im.counts() is COLLECTIVE (an allgather), so it must be called by
        # every rank the same number of times -- calling it inside a
        # `rank == root` guard for the progress print desynchronises the
        # communicator and shows up as MPI_ERR_TRUNCATE on a later collective.
        after = im.counts()
        cut = _cut(im, comm)
        isl = _islands(bal, im)

        if rank == root:
            print(f'[diffuse] cycle {i + 1}/{args.niters} '
                  f'spread={int(after.max() - after.min())} '
                  f'maxdev={int(np.abs(after - target).max())} '
                  f'cut={cut} ({cut/cut0:.3f}x)',
                  flush=True)

        if (after == before).all():
            if rank == root:
                print(f'[diffuse] converged after {i + 1} cycle(s)',
                      flush=True)
            break
    t_diffuse = time.perf_counter() - t2

    final = im.counts()

    # Gather the global element -> rank map. im.gids are global flat element
    # ids in `mesh.etypes` order, which is the same order
    # construct_global_con/construct_partitioning use, so the flat index is
    # directly usable as a vparts index.
    t3 = time.perf_counter()
    allg = comm.gather(im.gids, root=root)

    if rank == root:
        vparts = np.full(im.nglobal, -1, dtype=np.int32)
        for r, g in enumerate(allg):
            vparts[g] = r

        if (vparts < 0).any():
            raise RuntimeError(
                f'partition diffuse: {int((vparts < 0).sum())} elements were '
                f'not claimed by any rank'
            )

        with h5py.File(args.mesh, 'r+') as h5:
            # NB: the new fork returns FIVE values here
            # (conn, ecurved, etags, edisps, cdisps); the old branch returned
            # four. Unpacking four silently raises "too many values to unpack".
            con, ecurved, _etags, edisps, _cdisps = \
                BasePartitioner.construct_global_con(h5)
            pinfo = BasePartitioner.construct_partitioning(
                h5, ecurved, edisps, con, vparts
            )

            if args.dpname in h5['partitionings']:
                del h5['partitionings'][args.dpname]

            write_partitioning(h5, args.dpname, pinfo)
    t_write = time.perf_counter() - t3

    if rank == root:
        print(f'[diffuse] start  counts: {list(start)}')
        print(f'[diffuse] target counts: {list(target)}')
        print(f'[diffuse] final  counts: {list(final)}')
        print(f'[diffuse] cut {cut0} -> {cut} ({cut/cut0:.3f}x)')
        print(f'[diffuse] islands/rank {isl0} -> {isl} '
              f'(1 per rank is ideal)')
        print(f'[diffuse] TIMING read={t_read:.3f}s indexmesh={t_im:.3f}s '
              f'balance={t_bal:.3f}s diffuse={t_diffuse:.3f}s '
              f'write={t_write:.3f}s '
              f'total={time.perf_counter() - t0:.3f}s')
        print(f'[diffuse] wrote partitionings/{args.dpname}', flush=True)

    comm.barrier()
