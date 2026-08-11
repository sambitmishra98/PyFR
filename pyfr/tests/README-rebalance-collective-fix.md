# Rebalance MPI collective regression tests

These are real `mpirun -np 4` integration tests against the online rebalancer
(`pyfr/rebalance/`), not standard single-process pytest unit tests. They
exercise `pyfr/rebalance/mesh.py::_exchange_eles_spts` and the physical-mesh
integrity check in `pyfr/integrators/rebalance.py`.

Run with:

    mpirun -np 4 python3 pyfr/tests/test_collective_mismatch.py
    mpirun -np 4 python3 pyfr/tests/test_collective_regression.py
    mpirun -np 4 python3 pyfr/tests/test_physical_integrity_check.py

## Background

`_exchange_eles_spts` guarded its three collective `Alltoallv`/`Exchange`
calls with a rank-local `if et not in old_mesh.eles: continue`. Since a rank
can legitimately own zero elements of an etype (the balancer drains them)
while still being a valid destination for that etype, this caused a real
collective-participation mismatch: ranks skipping the guarded body executed
fewer collectives than their peers, which MPI matches positionally. Depending
on buffer compatibility this produced either a hang/crash, or -- worse --
**silent element loss** with no error at all (confirmed via
`test_collective_mismatch.py`: 10 elements moved between ranks vanished with
no exception, only a UCX "unexpected tag-receive descriptor" warning).

The fix makes every rank agree on each etype's array schema via a cheap
`comm.allgather`, then participate in every collective with correctly-typed
empty buffers when they locally own none of that etype, rather than skipping.

`_assert_physical_conservation()` in `pyfr/integrators/rebalance.py` is a
second, independent safety net: it checks the ACTUAL physical mesh element
counts (not `IndexMesh` bookkeeping, which is structurally blind to this
failure mode) before and after every rebuild, and raises on every rank
simultaneously if elements were silently lost or duplicated. Toggle off with
`PYFR_REBAL_SKIP_INTEGRITY_CHECK=1` for cheap production runs once the fix is
fully certified; on by default.

`rebalance-fixtures/c200.pyfrm` is a small mesh with an `imbalanced4`
partition where rank 0 owns every quad and ranks 1-3 own none -- a real,
deterministic reproducer for the bug (no synthetic mocking of the failure
condition).

See `Efforts/profession/Iterate performance investigation worksheet.md`
(vault, 2026-08-09/2026-08-11 entries) for the full investigation history,
including cluster-level confirmation on both Launch (native + Apptainer
container, real GPU compute node under Slurm) and this local proof.
