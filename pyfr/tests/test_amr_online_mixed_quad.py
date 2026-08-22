import gc
from io import StringIO

import h5py
import numpy as np
import pytest

import pyfr.amrtransaction as amrtx
import pyfr.amrwriter as amrwriter
from pyfr.amr import encode_quad_leaf_tree
from pyfr.amrmesh import (
    AMRMeshError, materialize_native_mixed_quad_tree,
)
from pyfr.amrtransaction import (
    AMRTransactionError, perform_indicator_mixed_quad_amr_transaction,
)
from pyfr.amrwriter import (
    build_adapted_mixed_quad_mesh, write_adapted_mixed_quad_mesh,
)
from pyfr.backends import get_backend
from pyfr.inifile import Inifile
from pyfr.progress import NullProgressSequence
from pyfr.readers.gmsh import GmshReader
from pyfr.readers.native import NativeReader, Solution
from pyfr.shapes import QuadShape, TriShape
from pyfr.solvers import get_solver


FIELDS = ['rho', 'rhou', 'rhov', 'E']


def _mixed_gmsh():
    return '''\
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
3
2 1 "fluid"
1 2 "qwall"
1 3 "far"
$EndPhysicalNames
$Nodes
6
1 0 0 0
2 1 0 0
3 2 0 0
4 0 1 0
5 1 1 0
6 2 1 0
$EndNodes
$Elements
9
1 1 2 3 3 1 2
2 1 2 2 2 2 3
3 1 2 3 3 3 6
4 1 2 3 3 6 5
5 1 2 3 3 5 4
6 1 2 3 3 4 1
7 2 2 1 1 1 2 5
8 2 2 1 1 1 5 4
9 3 2 1 1 2 3 6 5
$EndElements
'''


def _buffered_mixed_gmsh():
    return '''\
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
3
2 1 "fluid"
1 2 "qwall"
1 3 "far"
$EndPhysicalNames
$Nodes
8
1 0 0 0
2 1 0 0
3 2 0 0
4 3 0 0
5 0 1 0
6 1 1 0
7 2 1 0
8 3 1 0
$EndNodes
$Elements
12
1 1 2 3 3 1 2
2 1 2 3 3 2 3
3 1 2 3 3 3 4
4 1 2 2 2 4 8
5 1 2 3 3 8 7
6 1 2 3 3 7 6
7 1 2 3 3 6 5
8 1 2 3 3 5 1
9 2 2 1 1 1 2 6
10 2 2 1 1 1 6 5
11 3 2 1 1 2 3 7 6
12 3 2 1 1 3 4 8 7
$EndElements
'''


def _cfg(system='euler', *, wall_min_level=0, order=1):
    cfg = Inifile(f'''\
[backend]
precision = double
[backend-openmp]
cc = gcc
[constants]
gamma = 1.4
mu = 0.001
Pr = 0.72
[solver]
system = {system}
order = {order}
anti-alias = none
viscosity-correction = none
[solver-time-integrator]
formulation = explicit
scheme = rk4
controller = none
tstart = 0
tend = 2e-8
dt = 1e-8
[solver-interfaces]
riemann-solver = rusanov
ldg-beta = 0.0
ldg-tau = 0.1
mortar-implementation = staged
mortar-geom-tol = 1e-10
[solver-interfaces-line]
flux-pts = gauss-legendre
[solver-elements-quad]
soln-pts = gauss-legendre
[solver-elements-tri]
soln-pts = alpha-opt
[soln-bcs-qwall]
type = {'no-slp-adia-wall' if system == 'navier-stokes' else 'slp-adia-wall'}
[soln-bcs-far]
type = {'no-slp-adia-wall' if system == 'navier-stokes' else 'slp-adia-wall'}
[solver-amr]
indicator = density-velocity-variation
refine-threshold = 0.05
max-level = 2
wall-min-level = {wall_min_level}
wall-boundaries = {'qwall' if wall_min_level else ''}
[soln-ics]
rho = 1
u = 0
v = 0
p = 10
''')
    return cfg


def _root_reader(tmp_path):
    path = tmp_path / 'mixed-root.pyfrm'
    GmshReader(
        StringIO(_mixed_gmsh()), NullProgressSequence()
    ).write(str(path), 1e-5)
    return NativeReader(str(path))


def _buffered_root_reader(tmp_path):
    path = tmp_path / 'mixed-buffered-root.pyfrm'
    GmshReader(
        StringIO(_buffered_mixed_gmsh()), NullProgressSequence()
    ).write(str(path), 1e-5)
    return NativeReader(str(path))


def _refined_tree(mesh, level=1):
    root = int(mesh.eidxs['quad'][0])
    if level == 1:
        leaves = tuple((root, (q,)) for q in range(4))
    elif level == 2:
        leaves = tuple((root, (q, r)) for q in range(4) for r in range(4))
    else:
        raise ValueError('unsupported test level')
    return encode_quad_leaf_tree(mesh.uuid, leaves)


def _state(cfg, *, nquad=1, ntri=2, varying=True):
    qb = QuadShape(None, cfg)
    tb = TriShape(None, cfg)

    rhoq = np.ones((qb.nupts, nquad))
    if varying:
        rhoq[:, 0] = np.linspace(0.94, 1.06, qb.nupts)
    zq = np.zeros_like(rhoq)
    pq = np.full_like(rhoq, 10.0)
    eq = pq/0.4
    qstate = np.concatenate([
        rhoq[:, None, :], zq[:, None, :], zq[:, None, :], eq[:, None, :]
    ], axis=1)

    rhot = np.ones((tb.nupts, ntri))
    zt = np.zeros_like(rhot)
    pt = np.full_like(rhot, 10.0)
    et = pt/0.4
    tstate = np.stack((rhot, zt, zt, et), axis=1)
    return {'quad': qstate, 'tri': tstate}


def _integrator(tmp_path, system='euler', *, wall_min_level=0):
    reader = _root_reader(tmp_path)
    cfg = _cfg(system, wall_min_level=wall_min_level)
    stats = Inifile()
    stats.set('solver-time-integrator', 'tcurr', 0.0)
    soln = Solution(
        config=cfg, stats=stats, fields=FIELDS, data=_state(cfg), state={}
    )
    intg = get_solver(get_backend('openmp', cfg), reader.mesh, soln, cfg)
    intg.advance_to(1e-8)
    return reader, intg


def _assert_con_equal(lhs, rhs):
    assert np.array_equal(lhs.cidxs, rhs.cidxs)
    assert np.array_equal(lhs.eidxs, rhs.eidxs)
    assert lhs.cidxmap == rhs.cidxmap


def _line_mcons(mesh):
    return tuple(
        mcon for mcon in mesh.mcon.values()
        if mcon.format == 'one-to-many-v1'
        and mcon.template == 'line-1x2'
    )


def test_mixed_quad_materializer_builds_tri_line1x2_and_roundtrips(tmp_path):
    root = _root_reader(tmp_path)
    adapted = None
    try:
        raw = materialize_native_mixed_quad_tree(
            root.mesh, _refined_tree(root.mesh)
        )
        assert len(raw.tri_nodes) == 2
        assert len(raw.quad_nodes) == 4
        assert len(raw.mortars) == 1
        mortar = raw.mortars[0]
        assert mortar.left_etype == 'tri'
        assert mortar.right_etype == ('quad', 'quad')
        assert mortar.format == 'one-to-many-v1'
        assert mortar.template == 'line-1x2'

        direct = build_adapted_mixed_quad_mesh(raw)
        path = tmp_path / 'mixed-adapted.pyfrm'
        write_adapted_mixed_quad_mesh(raw, str(path))
        adapted = NativeReader(str(path))
        native = adapted.mesh

        assert direct.uuid == native.uuid
        assert direct.etypes == native.etypes == ['quad', 'tri']
        assert np.array_equal(direct.spts['tri'], root.mesh.spts['tri'])
        assert np.array_equal(direct.tags['tri'], root.mesh.tags['tri'])
        assert tuple(direct.amr_tree.leaves()) == tuple(
            native.amr_tree.leaves()
        )
        assert direct.amr_tree.root_mesh_uuid == root.mesh.uuid

        assert direct.cidxmap == native.cidxmap
        for lhs, rhs in zip(direct.con, native.con):
            _assert_con_equal(lhs, rhs)
        assert direct.bcon.keys() == native.bcon.keys()
        for name in native.bcon:
            _assert_con_equal(direct.bcon[name], native.bcon[name])

        assert tuple(direct.mcon) == tuple(native.mcon)
        assert len(_line_mcons(direct)) == 1
        for name in direct.mcon:
            lhs = direct.mcon[name]
            rhs = native.mcon[name]
            assert np.array_equal(lhs.records, rhs.records)
            assert lhs.format == rhs.format == 'one-to-many-v1'
            assert lhs.template == rhs.template == 'line-1x2'
    finally:
        if adapted is not None:
            adapted.close()
        root.close()


def test_mixed_quad_persistence_splits_homogeneous_mortar_batches(tmp_path):
    root = _root_reader(tmp_path)
    adapted = None
    try:
        leaves = (
            (0, (0,)),
            *((0, (1, child)) for child in range(4)),
            (0, (2,)),
            *((0, (3, child)) for child in range(4)),
        )
        raw = materialize_native_mixed_quad_tree(
            root.mesh, encode_quad_leaf_tree(root.mesh.uuid, leaves)
        )
        assert len(raw.mortars) == 3

        direct = build_adapted_mixed_quad_mesh(raw)
        assert set(direct.mcon) == {
            'line-1x2-quad-to-quad', 'line-1x2-tri-to-quad'
        }
        assert sorted(map(len, _line_mcons(direct))) == [1, 2]

        path = tmp_path / 'mixed-l2-buffered.pyfrm'
        write_adapted_mixed_quad_mesh(raw, str(path))
        adapted = NativeReader(str(path))
        assert tuple(adapted.mesh.amr_tree.leaves()) == leaves
        assert tuple(adapted.mesh.mcon) == tuple(direct.mcon)
        for name in direct.mcon:
            assert np.array_equal(
                direct.mcon[name].records, adapted.mesh.mcon[name].records
            )
    finally:
        if adapted is not None:
            adapted.close()
        root.close()


def test_mixed_quad_materializer_rejects_l2_against_immutable_tri(tmp_path):
    root = _root_reader(tmp_path)
    try:
        with pytest.raises(
            AMRMeshError, match='2:1 against immutable Tri neighbour'
        ):
            materialize_native_mixed_quad_tree(
                root.mesh, _refined_tree(root.mesh, level=2)
            )
    finally:
        root.close()


@pytest.mark.parametrize('system', ['euler', 'navier-stokes'])
def test_mixed_quad_live_event_preserves_tris_and_continues(tmp_path, system):
    reader, intg = _integrator(tmp_path, system)
    try:
        old_intg = id(intg)
        old_system = id(intg.system)
        old_tcurr = intg.tcurr
        triidx = intg.system.ele_types.index('tri')
        old_tri = np.array(
            intg.system.ele_scal_upts(intg.idxcurr)[triidx], copy=True
        )

        result = perform_indicator_mixed_quad_amr_transaction(intg)
        tx = result.transaction

        assert result.decision.action == 'refine'
        assert result.decision.marks == ((0, ()),)
        assert id(intg) == old_intg
        assert id(intg.system) != old_system
        assert intg.tcurr == old_tcurr
        assert tx.stage_leaf_count == 4
        assert tx.stage_tri_count == 2
        assert tx.stage_mortar_count == 1
        assert tx.stage_mortar_formats == (
            ('one-to-many-v1', 'line-1x2'),
        )
        assert tx.fixed_blocked_marks == ()
        assert tx.bank_drift == 0.0
        assert np.max(np.abs(tx.conservation_error)) < 1e-12
        assert all(np.all(np.isfinite(rhs)) for rhs in tx.rhs)

        triidx = intg.system.ele_types.index('tri')
        new_tri = intg.system.ele_scal_upts(intg.idxcurr)[triidx]
        assert np.array_equal(new_tri, old_tri)

        intg.advance_to(2e-8)
        assert intg.tcurr == 2e-8
        assert all(
            np.all(np.isfinite(state))
            for state in intg.system.ele_scal_upts(intg.idxcurr)
        )
    finally:
        reader.close()
        del intg
        gc.collect()


def test_mixed_quad_live_transaction_uses_no_file_io(monkeypatch, tmp_path):
    reader, intg = _integrator(tmp_path)
    try:
        def forbidden(*args, **kwargs):
            pytest.fail('MIX2D1 live transaction attempted file I/O')

        monkeypatch.setattr(h5py, 'File', forbidden)
        monkeypatch.setattr(amrtx, 'NativeReader', forbidden)
        monkeypatch.setattr(
            amrwriter, 'write_adapted_mixed_quad_mesh', forbidden
        )

        result = perform_indicator_mixed_quad_amr_transaction(intg)
        assert result.transaction.stage_leaf_count == 4
    finally:
        reader.close()
        del intg
        gc.collect()


def test_mixed_quad_two_root_layers_admit_level2_wall_floor(tmp_path):
    reader = _buffered_root_reader(tmp_path)
    cfg = _cfg('navier-stokes', wall_min_level=2)
    stats = Inifile()
    stats.set('solver-time-integrator', 'tcurr', 0.0)
    soln = Solution(
        config=cfg, stats=stats, fields=FIELDS,
        data=_state(cfg, nquad=2, varying=False), state={},
    )
    intg = get_solver(
        get_backend('openmp', cfg), reader.mesh, soln, cfg
    )
    try:
        intg.advance_to(1e-8)
        triidx = intg.system.ele_types.index('tri')
        old_tri = np.array(
            intg.system.ele_scal_upts(intg.idxcurr)[triidx], copy=True
        )

        result = perform_indicator_mixed_quad_amr_transaction(intg)
        tx = result.transaction
        assert result.decision.action == 'refine'
        assert result.decision.marks == ()
        assert tx.stage_leaf_count == 20
        assert tx.stage_tri_count == 2
        assert tx.stage_mortar_count == 3
        assert len(tx.wall_splits) == 5
        assert len(tx.closure_splits) == 1
        assert tx.bank_drift == 0.0
        assert np.max(np.abs(tx.conservation_error)) < 1e-12
        assert all(np.all(np.isfinite(rhs)) for rhs in tx.rhs)

        leaves = tuple(intg.system.mesh.amr_tree.leaves())
        assert sum(len(path) == 1 for _, path in leaves) == 4
        assert sum(len(path) == 2 for _, path in leaves) == 16
        assert sum(map(len, _line_mcons(intg.system.mesh))) == 3
        triidx = intg.system.ele_types.index('tri')
        assert np.array_equal(
            intg.system.ele_scal_upts(intg.idxcurr)[triidx], old_tri
        )
    finally:
        reader.close()
        del intg
        gc.collect()


def test_mixed_quad_wall_floor_conflict_fails_closed(tmp_path):
    reader, intg = _integrator(tmp_path, wall_min_level=2)
    try:
        old_system = intg.system
        old_state = [
            np.array(state, copy=True)
            for state in old_system.ele_scal_upts(intg.idxcurr)
        ]
        with pytest.raises(
            AMRTransactionError,
            match='wall minimum level conflicts with immutable Tri 2:1',
        ):
            perform_indicator_mixed_quad_amr_transaction(intg)

        assert intg.system is old_system
        for old, new in zip(
            old_state, intg.system.ele_scal_upts(intg.idxcurr)
        ):
            assert np.array_equal(old, new)
    finally:
        reader.close()
        del intg
        gc.collect()


@pytest.mark.parametrize(
    'system,order',
    [('euler', 1), ('navier-stokes', 1), ('navier-stokes', 3)],
)
def test_mixed_quad_native_schedule_runs_live_event(
    tmp_path, system, order
):
    reader = _root_reader(tmp_path)
    try:
        cfg = _cfg(system, order=order)
        cfg.set('solver-time-integrator', 'tend', 3e-8)
        cfg.set('solver-amr', 'schedule-dt', 1e-8)
        stats = Inifile()
        stats.set('solver-time-integrator', 'tcurr', 0.0)
        soln = Solution(
            config=cfg, stats=stats, fields=FIELDS,
            data=_state(cfg), state={},
        )
        intg = get_solver(
            get_backend('openmp', cfg), reader.mesh, soln, cfg
        )
        intg.run()

        assert intg.amr_schedule.mode == 'mixed-quad-1r'
        assert intg.tcurr == pytest.approx(3e-8)
        history = intg.amr_schedule.history
        assert [event.time for event in history] == pytest.approx(
            [1e-8, 2e-8]
        )
        assert [event.action for event in history] == ['refine', 'refine']
        assert [event.leaf_count for event in history] == [4, 10]
        assert [event.mortar_count for event in history] == [1, 3]
        assert intg._amr_root_mesh is reader.mesh
        assert len(intg.system.mesh.eidxs['tri']) == 2
        assert intg.system.mesh.amr_tree is not None

        leaves = tuple(intg.system.mesh.amr_tree.leaves())
        assert (0, (0,)) in leaves and (0, (2,)) in leaves
        assert not any(
            len(path) > 1 and path[0] in {0, 2}
            for _, path in leaves
        )
        assert set(intg.system.mesh.mcon) == {
            'line-1x2-quad-to-quad', 'line-1x2-tri-to-quad'
        }
        assert sum(map(len, _line_mcons(intg.system.mesh))) == 3
        if order == 3:
            assert intg.system.ele_shapes['quad'] == (16, 4, 10)
            assert intg.system.ele_shapes['tri'] == (10, 4, 2)
        assert all(
            np.all(np.isfinite(state))
            for state in intg.system.ele_scal_upts(intg.idxcurr)
        )
    finally:
        reader.close()
        del intg
        gc.collect()
