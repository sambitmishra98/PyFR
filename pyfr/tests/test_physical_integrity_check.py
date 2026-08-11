#!/usr/bin/env python3
'''
Unit test for the new _assert_physical_conservation() integrity check in
pyfr/integrators/rebalance.py. Run under `mpirun -np 4`: proves the check
(a) passes silently on a real conserved rebuild, and (b) actually raises
when elements are dropped -- i.e. it would have caught the 2026-08-09
collective-mismatch bug's silent data loss, which lb_elem_dist.csv could
not.
'''
import sys
import traceback
from types import SimpleNamespace

import mpi4py.rc
mpi4py.rc.initialize = False

from pyfr.mpiutil import get_comm_rank_root, init_mpi
from pyfr.integrators.rebalance import _assert_physical_conservation

FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    if not cond:
        FAILS.append(label)
    print(f'  [{tag}] {label}' + (f'  ({detail})' if detail else ''),
          flush=True)


def main():
    init_mpi()
    comm, rank, root = get_comm_rank_root()

    # Case A: conserved rebuild (rank 0 -> rank 1 moves 10 quads, same as
    # test_collective_mismatch.py's legitimate scenario). Global totals
    # unchanged -> must not raise.
    pre_counts = {0: {'quad': 196, 'tri': 2717}, 1: {'tri': 172},
                  2: {'tri': 171}, 3: {'tri': 171}}
    post_counts_ok = {0: {'quad': 186, 'tri': 2717}, 1: {'quad': 10, 'tri': 172},
                       2: {'tri': 171}, 3: {'tri': 171}}

    etypes = ['quad', 'tri']
    old_mesh = SimpleNamespace(etypes=etypes,
                                eles={et: [None] * n
                                      for et, n in pre_counts[rank].items()})
    new_mesh_ok = SimpleNamespace(etypes=etypes,
                                   eles={et: [None] * n
                                        for et, n in post_counts_ok[rank].items()})

    raised = False
    try:
        _assert_physical_conservation(old_mesh, new_mesh_ok, tcurr=1.0)
    except RuntimeError:
        raised = True
    comm.barrier()
    check('conserved rebuild: check passes without raising', not raised)

    # Case B: silent loss (rank 1 should receive 10 quads but gets 0 --
    # exactly the old collective-mismatch bug's behaviour). Global total
    # drops by 10 -> must raise on root.
    post_counts_bad = {0: {'quad': 186, 'tri': 2717}, 1: {'tri': 172},
                        2: {'tri': 171}, 3: {'tri': 171}}
    new_mesh_bad = SimpleNamespace(etypes=etypes,
                                    eles={et: [None] * n
                                         for et, n in post_counts_bad[rank].items()})

    raised = False
    err = None
    try:
        _assert_physical_conservation(old_mesh, new_mesh_bad, tcurr=2.0)
    except RuntimeError as e:
        raised = True
        err = e
    comm.barrier()

    check('silent-loss case: check raises RuntimeError on every rank '
          '(not just root -- avoids root dying while peers hang)',
          raised, str(err) if err else 'no exception')

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
