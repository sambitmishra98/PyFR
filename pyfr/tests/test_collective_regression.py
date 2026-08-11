#!/usr/bin/env python3
'''
Regression suite for the collective-participation bug in
pyfr/rebalance/mesh.py::_exchange_eles_spts (banked per Sol's response23
review). Run under `mpirun -np 4` against the C200 'imbalanced4' partition,
where mesh.etypes = ['quad', 'tri'] and only rank 0 owns any quads.

Cases covered (numbering matches Sol's list):
  1. Receive into an absent etype               (rank 1 gets quads it had none of)
  2. Absent FIRST etype ('quad'), present later  ('tri') -- verifies the tri
     collectives immediately after a skipped/participated quad round stay
     correctly paired, since quad is etypes[0].
  3. Zero local send, nonzero receive            (same move as #1, from rank 1's
     receiving side)
  4. Zero send AND zero receive on one rank      (rank 2/3: no quads before,
     none assigned, must still just pass through)
  5. Multiple fields verified, not just counts   (spts geometry + valency
     content checked to actually match the source element, not merely sized
     right)
  6. Global conservation                         (per-etype and total, vs the
     known pristine mesh totals)
'''
import pathlib
import sys
import traceback

import mpi4py.rc
mpi4py.rc.initialize = False

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, init_mpi
from pyfr.readers.native import NativeReader
from pyfr.rebalance.mesh import _exchange_eles_spts
from pyfr.rebalance.redistribute import make_exchangers

MESH = str(pathlib.Path(__file__).parent / 'rebalance-fixtures' / 'c200.pyfrm')
NMOVE = 10
PRISTINE_TOTALS = {'quad': 196, 'tri': 3231}  # sum of tri across all 4 ranks
FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    if not cond:
        FAILS.append(label)
    print(f'  [{tag}] {label}' + (f'  ({detail})' if detail else ''), flush=True)


def main():
    init_mpi()
    comm, rank, root = get_comm_rank_root()

    mesh = getattr(NativeReader(MESH, pname='imbalanced4'), 'mesh', None)
    if mesh is None:
        mesh = NativeReader(MESH, pname='imbalanced4')

    assert list(mesh.etypes) == ['quad', 'tri'], (
        'test assumes quad is etypes[0] -- fixture changed, review needed')

    pre_quad = mesh.eles['quad'].copy() if 'quad' in mesh.eles else None
    pre_spts_quad = mesh.spts['quad'].copy() if 'quad' in mesh.eles else None

    # Move NMOVE quads: rank0 -> rank1 (was zero, #1/#3), leave rank2/rank3
    # untouched at zero (#4). tri stays fully in place on every rank.
    ownermap = {}
    for et in mesh.etypes:
        n = len(mesh.eidxs.get(et, []))
        dests = np.full(n, rank, dtype=int)
        if et == 'quad' and rank == 0 and n >= NMOVE:
            dests[:NMOVE] = 1
        ownermap[et] = dests

    moved_ids = pre_quad['nodes'][:NMOVE].copy() if rank == 0 else None
    moved_spts = pre_spts_quad[:, :NMOVE, :].copy() if rank == 0 else None
    moved_ids = comm.bcast(moved_ids, root=0)
    moved_spts = comm.bcast(moved_spts, root=0)

    exchangers = make_exchangers(comm, mesh.eidxs, ownermap, mesh.etypes)
    eles, spts, nvals = _exchange_eles_spts(mesh, exchangers)

    n_quad = len(eles.get('quad', []))
    n_tri = len(eles.get('tri', []))

    print(f'rank {rank}: post-exchange quad={n_quad} tri={n_tri}', flush=True)
    comm.barrier()

    if rank == root:
        print('\n=== case 1/3: receive into a previously-absent etype ===', flush=True)
    allq = comm.allgather((rank, n_quad))
    if rank == root:
        got = dict(allq)
        check('rank 1 (was 0 quads) now owns exactly NMOVE quads',
              got.get(1) == NMOVE, f'got {got.get(1)}')
        check('rank 0 correctly reduced by NMOVE',
              got.get(0) == 196 - NMOVE, f'got {got.get(0)}')

    if rank == root:
        print('\n=== case 2: quad (etypes[0]) skip/participate does not '
              'corrupt the immediately-following tri (etypes[1]) exchange ===',
              flush=True)
    allt = comm.allgather((rank, n_tri))
    if rank == root:
        gott = dict(allt)
        expected_tri = {0: 2717, 1: 172, 2: 171, 3: 171}
        ok = all(gott.get(r) == expected_tri[r] for r in range(4))
        check('tri counts unchanged on every rank despite quad exchange first',
              ok, f'got {gott}, expected {expected_tri}')

    if rank == root:
        print('\n=== case 4: ranks with zero quads before AND after just '
              'pass through cleanly ===', flush=True)
    if rank in (2, 3):
        check(f'rank {rank}: 0 quads before and after, no error reaching here',
              n_quad == 0, f'got {n_quad}')

    if rank == 1:
        print('\n=== case 5: content check, not just counts (rank 1) ===', flush=True)
        recv_ids = eles['quad']['nodes']
        recv_spts = spts['quad']
        ids_ok = np.array_equal(np.sort(recv_ids, axis=0),
                                np.sort(moved_ids, axis=0)) if False else \
                 sorted(map(tuple, recv_ids.tolist())) == \
                 sorted(map(tuple, moved_ids.tolist()))
        check('received element node-connectivity matches the moved quads exactly',
              ids_ok)
        # geometry: match up by node tuple since exchange order isn't
        # guaranteed to preserve source order
        src_by_nodes = {tuple(moved_ids[i]): moved_spts[:, i, :]
                        for i in range(NMOVE)}
        geom_ok = True
        for i in range(len(recv_ids)):
            key = tuple(recv_ids[i])
            if key not in src_by_nodes or not np.allclose(recv_spts[:, i, :],
                                                          src_by_nodes[key]):
                geom_ok = False
                break
        check('received spts geometry matches the source element exactly '
              '(not just correctly counted/shaped)', geom_ok)

    comm.barrier()
    if rank == root:
        print('\n=== case 6: global conservation ===', flush=True)
    tot_quad = comm.allreduce(n_quad)
    tot_tri = comm.allreduce(n_tri)
    if rank == root:
        check('global quad total conserved', tot_quad == PRISTINE_TOTALS['quad'],
              f'got {tot_quad}, expected {PRISTINE_TOTALS["quad"]}')
        check('global tri total conserved', tot_tri == PRISTINE_TOTALS['tri'],
              f'got {tot_tri}, expected {PRISTINE_TOTALS["tri"]}')
        check('grand total conserved',
              tot_quad + tot_tri == sum(PRISTINE_TOTALS.values()))

    comm.barrier()
    n_fail = comm.allreduce(len(FAILS))
    if rank == root:
        print(f'\n{"ALL PASS" if n_fail == 0 else f"{n_fail} FAILURE(S)"}',
              flush=True)
    return 1 if n_fail else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        from mpi4py import MPI
        MPI.COMM_WORLD.Abort(1)
