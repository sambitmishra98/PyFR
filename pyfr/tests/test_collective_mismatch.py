#!/usr/bin/env python3
'''
Deterministic regression test for the collective-participation bug in
pyfr/rebalance/mesh.py::_exchange_eles_spts.

The C200 'imbalanced4' partition already puts every quad on rank 0, so
ranks 1-3 have no 'quad' key in their local mesh.eles while mesh.etypes
is globally ['quad', 'tri'].  The unpatched code guards three collective
Exchange() calls with a rank-local `if et not in old_mesh.eles: continue`,
so ranks 1-3 skip collectives that rank 0 executes.

This test moves a handful of quads from rank 0 to rank 1, which is a
legitimate rebalance decision, and which rank 1 cannot possibly service
while it is skipping the exchange.  Run under `mpirun -np 4`; use a
wall-clock timeout, because the unpatched failure mode can be a deadlock
rather than an exception.
'''
import pathlib
import sys
import traceback

import mpi4py.rc
mpi4py.rc.initialize = False

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, init_mpi

MESH = str(pathlib.Path(__file__).parent / 'rebalance-fixtures' / 'c200.pyfrm')
from pyfr.readers.native import NativeReader
from pyfr.rebalance.mesh import _exchange_eles_spts
from pyfr.rebalance.redistribute import make_exchangers

NMOVE = 10


def main():
    init_mpi()
    comm, rank, root = get_comm_rank_root()

    _rdr = NativeReader(MESH, pname='imbalanced4')
    mesh = getattr(_rdr, 'mesh', _rdr)

    has_quad = 'quad' in mesh.eles
    nquad = len(mesh.eles['quad']) if has_quad else 0
    if rank == root:
        print(f'mesh.etypes (global) = {list(mesh.etypes)}', flush=True)
    comm.barrier()
    print(f'  rank {rank}: local etypes={sorted(mesh.eles)} nquad={nquad}',
          flush=True)
    comm.barrier()

    # Keep everything in place except NMOVE quads, which rank 0 hands to
    # rank 1 -- a rank that currently owns no quads at all.
    ownermap = {}
    for et in mesh.etypes:
        n = len(mesh.eidxs.get(et, []))
        dests = np.full(n, rank, dtype=int)
        if et == 'quad' and rank == 0 and n >= NMOVE:
            dests[:NMOVE] = 1
        ownermap[et] = dests

    if rank == root:
        print(f'\nmoving {NMOVE} quads from rank 0 -> rank 1 '
              f'(rank 1 currently owns 0 quads)', flush=True)
        print('calling _exchange_eles_spts ...', flush=True)

    exchangers = make_exchangers(comm, mesh.eidxs, ownermap, mesh.etypes)
    eles, spts, nvals = _exchange_eles_spts(mesh, exchangers)

    got = len(eles['quad']) if 'quad' in eles else 0
    allgot = comm.allgather((rank, got))

    if rank == root:
        print('\nRESULT: exchange returned on all ranks (no deadlock, no error)',
              flush=True)
        print('  post-exchange quad counts per rank:', flush=True)
        ok = True
        for r, g in sorted(allgot):
            exp = {0: nquad - NMOVE, 1: NMOVE}.get(r, 0)
            flag = 'ok' if g == exp else f'MISMATCH (expected {exp})'
            if g != exp:
                ok = False
            print(f'    rank {r}: {g:>4}   {flag}', flush=True)
        print(f'\n{"PASS" if ok else "FAIL"}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        from mpi4py import MPI
        MPI.COMM_WORLD.Abort(1)
