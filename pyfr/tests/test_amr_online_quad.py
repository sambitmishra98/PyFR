from io import StringIO

import h5py
import numpy as np
import pytest

import pyfr.amrtransaction as amrtx
import pyfr.amrwriter as amrwriter
from pyfr.amr import encode_quad_leaf_tree
from pyfr.amrmesh import materialize_native_quad_tree
from pyfr.amrtransaction import (
    AMRTransactionError, _quad_online_settings,
    _validate_quad_online_integrator, perform_indicator_quad_amr_transaction,
)
from pyfr.amrwriter import build_adapted_quad_mesh, write_adapted_quad_mesh
from pyfr.backends import get_backend
from pyfr.inifile import Inifile
from pyfr.progress import NullProgressSequence
from pyfr.readers.gmsh import GmshReader
from pyfr.readers.native import NativeReader, Solution
from pyfr.shapes import QuadShape
from pyfr.solvers import get_solver


FIELDS = ['rho', 'rhou', 'rhov', 'E']


def _two_quad_gmsh():
    coords = [
        (1, 0, 0, 0), (2, 1, 0, 0), (3, 2, 0, 0),
        (4, 0, 1, 0), (5, 1, 1, 0), (6, 2, 1, 0),
    ]
    quads = [(1, 2, 5, 4), (2, 3, 6, 5)]
    boundary = [
        (1, 2), (2, 3), (3, 6),
        (6, 5), (5, 4), (4, 1),
    ]
    lines = [
        '$MeshFormat', '2.2 0 8', '$EndMeshFormat',
        '$PhysicalNames', '2', '2 1 "fluid"', '1 2 "wall"',
        '$EndPhysicalNames', '$Nodes', str(len(coords)),
    ]
    lines.extend(f'{i} {x} {y} {z}' for i, x, y, z in coords)
    lines.extend([
        '$EndNodes', '$Elements', str(len(boundary) + len(quads)),
    ])

    eid = 1
    for edge in boundary:
        lines.append(f'{eid} 1 2 2 2 ' + ' '.join(map(str, edge)))
        eid += 1
    for quad in quads:
        lines.append(f'{eid} 3 2 1 1 ' + ' '.join(map(str, quad)))
        eid += 1
    lines.append('$EndElements')
    return '\n'.join(lines) + '\n'


def _cfg():
    return Inifile('''\
[backend]
precision = double
[backend-openmp]
cc = gcc
[constants]
gamma = 1.4
[solver]
system = euler
order = 1
anti-alias = none
[solver-time-integrator]
formulation = explicit
scheme = rk4
controller = none
tstart = 0
tend = 2e-5
dt = 1e-5
[solver-interfaces]
riemann-solver = rusanov
mortar-implementation = staged
mortar-geom-tol = 1e-10
[solver-interfaces-line]
flux-pts = gauss-legendre
[solver-elements-quad]
soln-pts = gauss-legendre
[soln-bcs-wall]
type = slp-adia-wall
[solver-amr]
indicator = density-velocity-variation
refine-threshold = 0.05
max-level = 2
wall-min-level = 0
[soln-ics]
rho = 1
u = 0
v = 0
p = 10
''')


def _root_reader(tmp_path):
    path = tmp_path / 'root.pyfrm'
    GmshReader(
        StringIO(_two_quad_gmsh()), NullProgressSequence()
    ).write(str(path), 1e-5)
    return NativeReader(str(path))


def _one_root_raw(mesh):
    roots = tuple(map(int, sorted(mesh.eidxs['quad'])))
    leaves = [(roots[0], (q,)) for q in range(4)] + [(roots[1], ())]
    return materialize_native_quad_tree(
        mesh, encode_quad_leaf_tree(mesh.uuid, leaves)
    )


def _assert_con_equal(lhs, rhs):
    assert np.array_equal(lhs.cidxs, rhs.cidxs)
    assert np.array_equal(lhs.eidxs, rhs.eidxs)
    assert lhs.cidxmap == rhs.cidxmap


def _online_integrator(tmp_path):
    reader = _root_reader(tmp_path)
    cfg = _cfg()
    basis = QuadShape(None, cfg)
    nupts = basis.nupts

    rho = np.ones((nupts, 2))
    u = np.zeros_like(rho)
    u[:, 0] = np.linspace(0.0, 0.4, nupts)
    v = np.zeros_like(rho)
    p = np.full_like(rho, 10.0)
    energy = p/0.4 + 0.5*rho*(u*u + v*v)
    state = np.stack((rho, rho*u, rho*v, energy), axis=1)

    stats = Inifile()
    stats.set('solver-time-integrator', 'tcurr', 0.0)
    soln = Solution(
        config=cfg, stats=stats, fields=FIELDS,
        data={'quad': state}, state={},
    )
    intg = get_solver(get_backend('openmp', cfg), reader.mesh, soln, cfg)
    intg.advance_to(1e-5)
    assert intg.nacptsteps == 1 and intg.stepinfo == []
    return reader, intg


def test_online_quad_in_memory_mesh_matches_native_roundtrip(tmp_path):
    root = _root_reader(tmp_path)
    adapted = None
    try:
        raw = _one_root_raw(root.mesh)
        direct = build_adapted_quad_mesh(raw)
        path = tmp_path / 'adapted.pyfrm'
        write_adapted_quad_mesh(raw, str(path))
        adapted = NativeReader(str(path))
        native = adapted.mesh

        for name in ('ndims', 'creator', 'codec', 'uuid', 'version', 'etypes'):
            assert getattr(direct, name) == getattr(native, name)
        for name in (
            'eidxs', 'spts', 'spts_nodes', 'spts_curved', 'colours', 'tags'
        ):
            assert np.array_equal(
                getattr(direct, name)['quad'], getattr(native, name)['quad']
            )

        assert direct.cidxmap == native.cidxmap
        for lhs, rhs in zip(direct.con, native.con):
            _assert_con_equal(lhs, rhs)
        assert direct.bcon.keys() == native.bcon.keys()
        for name in native.bcon:
            _assert_con_equal(direct.bcon[name], native.bcon[name])

        assert direct.mcon.keys() == native.mcon.keys()
        for name in native.mcon:
            lhs, rhs = direct.mcon[name], native.mcon[name]
            assert np.array_equal(lhs.records, rhs.records)
            assert lhs.format == rhs.format
            assert lhs.template == rhs.template

        for name in ('node_idxs', 'node_valency', 'node_locs'):
            assert np.array_equal(getattr(direct, name), getattr(native, name))
        assert tuple(direct.amr_tree.leaves()) == tuple(
            native.amr_tree.leaves()
        )
        assert direct.amr_tree.root_mesh_uuid == native.amr_tree.root_mesh_uuid
    finally:
        if adapted is not None:
            adapted.close()
        root.close()


def test_online_quad_transaction_has_no_file_io_and_continues(monkeypatch,
                                                               tmp_path):
    reader, intg = _online_integrator(tmp_path)
    try:
        old_intg = id(intg)
        old_system = intg.system
        old_tcurr = intg.tcurr
        old_nacpt = intg.nacptsteps
        old_idx = intg.idxcurr
        old_nrhs = intg.nrhsevals
        old_state = np.array(
            old_system.ele_scal_upts(old_idx)[0], copy=True
        )

        def forbidden(*args, **kwargs):
            pytest.fail('ONLINE2D-1 transaction attempted file I/O')

        monkeypatch.setattr(h5py, 'File', forbidden)
        monkeypatch.setattr(amrtx, 'NativeReader', forbidden)
        monkeypatch.setattr(amrtx, 'write_adapted_mesh', forbidden)
        monkeypatch.setattr(amrwriter, 'write_adapted_quad_mesh', forbidden)

        result = perform_indicator_quad_amr_transaction(intg)
        tx = result.transaction

        assert id(intg) == old_intg
        assert intg.system is not old_system
        assert intg.tcurr == old_tcurr
        assert intg.nacptsteps == old_nacpt
        assert intg.idxcurr == old_idx
        assert intg.nrhsevals == old_nrhs
        assert np.array_equal(old_system.ele_scal_upts(old_idx)[0], old_state)
        assert result.decision.marks == ((0, ()),)
        assert tx.stage_leaf_count == 5
        assert tx.stage_mortar_count == 1
        assert tx.stage_mortar_formats == (
            ('one-to-many-v1', 'line-1x2'),
        )
        assert tx.bank_drift == 0
        assert np.max(np.abs(tx.conservation_error)) < 1e-12
        staged, = intg.system.ele_scal_upts(intg.idxcurr)
        assert np.array_equal(staged, tx.transferred_state)

        intg.advance_to(2e-5)
        assert intg.tcurr == pytest.approx(2e-5)
        assert intg.nacptsteps == old_nacpt + 1
        continued, = intg.system.ele_scal_upts(intg.idxcurr)
        assert np.isfinite(continued).all()
    finally:
        reader.close()


def test_online_quad_repeated_live_event_uses_root_anchor_and_continues(
        tmp_path):
    reader, intg = _online_integrator(tmp_path)
    try:
        first = perform_indicator_quad_amr_transaction(intg)
        assert first.transaction.stage_leaf_count == 5
        assert first.transaction.stage_mortar_count == 1
        assert intg._amr_root_mesh is reader.mesh

        intg.advance_to(2e-5)
        before_second_steps = intg.nacptsteps
        before_second_rhs = intg.nrhsevals
        first_adapted_system = intg.system

        second = perform_indicator_quad_amr_transaction(intg)
        tx = second.transaction
        assert second.decision.action == 'refine'
        assert tx.refine_marks == (
            (0, (0,)), (0, (1,)), (0, (2,)), (0, (3,))
        )
        assert tx.closure_splits == ((1, ()),)
        assert tx.stage_leaf_count == 20
        assert tx.stage_mortar_count == 2
        assert intg.system is not first_adapted_system
        assert intg._amr_root_mesh is reader.mesh
        assert intg.nacptsteps == before_second_steps
        assert intg.nrhsevals == before_second_rhs
        assert np.max(np.abs(tx.conservation_error)) < 1e-12

        intg.tend = 3e-5
        intg.advance_to(3e-5)
        assert intg.tcurr == pytest.approx(3e-5)
        assert intg.nacptsteps == before_second_steps + 1
        continued, = intg.system.ele_scal_upts(intg.idxcurr)
        assert np.isfinite(continued).all()
    finally:
        reader.close()


def test_online_quad_second_event_late_failure_keeps_first_adapted_solver(
        monkeypatch, tmp_path):
    reader, intg = _online_integrator(tmp_path)
    try:
        first = perform_indicator_quad_amr_transaction(intg)
        assert first.transaction.stage_leaf_count == 5
        intg.advance_to(2e-5)

        adapted_system = intg.system
        accepted_bank = intg.idxcurr
        accepted_state = np.array(
            adapted_system.ele_scal_upts(accepted_bank)[0], copy=True
        )
        snapshot = (
            intg.tcurr, intg.nacptsteps, intg.nrjctsteps, intg.nrhsevals,
            intg.mesh_uuid, intg.serialiser, intg.gndofs,
        )
        orig_scratch_rhs = amrtx._scratch_rhs

        def fail_late(*args, **kwargs):
            raise RuntimeError('second-event late staged validation failure')

        monkeypatch.setattr(amrtx, '_scratch_rhs', fail_late)
        with pytest.raises(RuntimeError, match='second-event late staged'):
            perform_indicator_quad_amr_transaction(intg)

        assert intg.system is adapted_system
        assert intg._amr_root_mesh is reader.mesh
        assert np.array_equal(
            adapted_system.ele_scal_upts(accepted_bank)[0], accepted_state
        )
        assert (
            intg.tcurr, intg.nacptsteps, intg.nrjctsteps, intg.nrhsevals,
            intg.mesh_uuid, intg.serialiser, intg.gndofs,
        ) == snapshot

        monkeypatch.setattr(amrtx, '_scratch_rhs', orig_scratch_rhs)
        intg.tend = 3e-5
        intg.advance_to(3e-5)
        continued, = intg.system.ele_scal_upts(intg.idxcurr)
        assert intg.tcurr == pytest.approx(3e-5)
        assert np.isfinite(continued).all()
    finally:
        reader.close()


def test_online_quad_native_schedule_repeats_then_noops(tmp_path):
    reader = _root_reader(tmp_path)
    try:
        cfg = _cfg()
        cfg.set('solver-time-integrator', 'tend', 4e-5)
        cfg.set('solver-amr', 'schedule-dt', 1e-5)
        basis = QuadShape(None, cfg)
        nupts = basis.nupts

        rho = np.ones((nupts, 2))
        u = np.zeros_like(rho)
        u[:, 0] = np.linspace(0.0, 0.4, nupts)
        v = np.zeros_like(rho)
        p = np.full_like(rho, 10.0)
        energy = p/0.4 + 0.5*rho*(u*u + v*v)
        state = np.stack((rho, rho*u, rho*v, energy), axis=1)

        stats = Inifile()
        stats.set('solver-time-integrator', 'tcurr', 0.0)
        soln = Solution(
            config=cfg, stats=stats, fields=FIELDS,
            data={'quad': state}, state={},
        )
        intg = get_solver(
            get_backend('openmp', cfg), reader.mesh, soln, cfg
        )
        intg.run()

        assert intg.tcurr == pytest.approx(4e-5)
        assert intg.nacptsteps >= 4
        assert intg.nrhsevals == 4*intg.nacptsteps
        history = intg.amr_schedule.history
        assert [event.time for event in history] == pytest.approx(
            [1e-5, 2e-5, 3e-5]
        )
        assert [event.action for event in history] == [
            'refine', 'refine', 'none'
        ]
        assert [event.leaf_count for event in history] == [5, 20, 20]
        assert [event.mortar_count for event in history] == [1, 2, None]
        assert intg._amr_root_mesh is reader.mesh

        leaves = tuple(intg.system.mesh.amr_tree.leaves())
        assert len(leaves) == 20
        assert len(intg.system.mesh.mcon['line-1x2']) == 2
        continued, = intg.system.ele_scal_upts(intg.idxcurr)
        assert np.isfinite(continued).all()
    finally:
        reader.close()


def test_online_quad_late_failure_rolls_back_and_old_solver_advances(
        monkeypatch):
    import pyfr.amrindicator as indicator
    import pyfr.amroffline as offline

    old_leaves = ((0, ()),)
    new_leaves = tuple((0, (q,)) for q in range(4))
    old_state = np.arange(8, dtype=float).reshape(2, 4, 1) + 1.0
    transferred = np.repeat(old_state, 4, axis=2)

    class Matrix:
        def __init__(self, value):
            self.value = np.array(value, copy=True)
            self.ioshape = self.value.shape

        def set(self, value):
            self.value[...] = value

    class Backend:
        name = 'openmp'

        @staticmethod
        def wait():
            pass

    class Elements:
        @staticmethod
        def convars(ndims, cfg):
            return FIELDS

    class FakeSystem:
        name = 'euler'
        ndims = 2
        elementscls = Elements

        def __init__(self, backend, mesh, initsoln=None, registers=None,
                     cfg=None, serialiser=None, needs_cfl=False):
            self.backend = backend
            self.mesh = mesh
            self.cfg = cfg
            state = (initsoln.data['quad'] if initsoln is not None
                     else old_state)
            self.ele_types = ['quad']
            self.ele_shapes = {'quad': tuple(state.shape)}
            self.nrhs = 2
            self.ele_banks = [[Matrix(state), Matrix(np.zeros_like(state))]]
            self.ele_ndofs = [int(np.prod(state.shape))]
            self.rhs_calls = []

        def ele_scal_upts(self, bank):
            return (self.ele_banks[0][bank].value,)

        def commit(self):
            pass

        def preproc(self, t, bank):
            pass

        def rhs(self, t, uin, uout):
            self.rhs_calls.append((t, uin, uout))
            self.ele_banks[0][uout].value[...] = (
                self.ele_banks[0][uin].value + 1
            )

    class RootMesh:
        uuid = 'root-quad'
        bcon = {}

    class StageMesh:
        uuid = 'stage-quad'
        amr_tree = encode_quad_leaf_tree(RootMesh.uuid, new_leaves)
        mcon = {}
        eidxs = {'quad': np.arange(4, dtype=np.int64)}

    class Raw:
        root_mesh_uuid = RootMesh.uuid
        leaf_order = new_leaves
        mortars = ()

    cfg = Inifile('[constants]\ngamma = 1.4\n')
    root_mesh = RootMesh()
    system = FakeSystem(Backend(), root_mesh, cfg=cfg)

    class Intg:
        def __init__(self):
            self.system = system
            self.backend = system.backend
            self.cfg = cfg
            self._registers = object()
            self.idxcurr = 0
            self.tcurr = 1.0
            self.nacptsteps = 3
            self.nrjctsteps = 0
            self.nrhsevals = 12
            self.mesh_uuid = root_mesh.uuid
            self.serialiser = object()
            self.gndofs = 8

        def advance_to(self, target):
            self.system.rhs(self.tcurr, self.idxcurr, 1)
            self.tcurr = target

    intg = Intg()
    snapshot = (
        intg.system, intg.tcurr, intg.idxcurr, intg.nacptsteps,
        intg.nrjctsteps, intg.nrhsevals, intg.mesh_uuid,
        intg.serialiser, intg.gndofs,
    )

    monkeypatch.setattr(
        amrtx, '_validate_quad_online_integrator',
        lambda i, restart_root_mesh=None: (system, root_mesh, root_mesh),
    )
    monkeypatch.setattr(
        amrtx, '_quad_online_settings',
        lambda cfg: (0.5, 1, (), 0, 1e-14, 1e-14),
    )
    monkeypatch.setattr(
        amrtx, '_quad_local_maps',
        lambda mesh: ({0: 0}, {0: object()}, None),
    )
    monkeypatch.setattr(
        amrtx, '_derive_quad_root_face_topology',
        lambda *args: ({}, {}),
    )
    monkeypatch.setattr(
        indicator, 'density_velocity_variation_scores',
        lambda *args, **kwargs: np.array([1.0]),
    )
    monkeypatch.setattr(offline, '_current_column_leaves',
                        lambda *args: old_leaves)
    monkeypatch.setattr(offline, '_validate_state',
                        lambda *args: (0, 1, 2, 3))
    monkeypatch.setattr(offline, '_split_nodes', lambda leaves: set())
    monkeypatch.setattr(offline, '_wall_floor_splits',
                        lambda *args: set())
    monkeypatch.setattr(offline, '_close_2to1',
                        lambda *args: (new_leaves, ()))
    monkeypatch.setattr(offline, '_prolongate_to_tree',
                        lambda *args: transferred)
    monkeypatch.setattr(offline, '_global_integral',
                        lambda *args: np.ones(4))
    monkeypatch.setattr(
        amrtx, 'QuadShape',
        type('FakeQuadShape', (), {
            '__init__': lambda self, *args: setattr(self, 'nupts', 2)
        }),
    )
    monkeypatch.setattr(amrtx, '_eos_ranges',
                        lambda *args: ((1.0, 1.0), (1.0, 1.0)))
    monkeypatch.setattr(amrtx, 'materialize_native_quad_tree',
                        lambda *args, **kwargs: Raw())
    monkeypatch.setattr(amrtx, 'build_adapted_quad_mesh',
                        lambda raw: StageMesh())

    def fail_late(*args, **kwargs):
        raise RuntimeError('late staged validation failure')

    monkeypatch.setattr(amrtx, '_scratch_rhs', fail_late)
    with pytest.raises(RuntimeError, match='late staged validation'):
        perform_indicator_quad_amr_transaction(intg)

    assert snapshot == (
        intg.system, intg.tcurr, intg.idxcurr, intg.nacptsteps,
        intg.nrjctsteps, intg.nrhsevals, intg.mesh_uuid,
        intg.serialiser, intg.gndofs,
    )
    assert np.array_equal(system.ele_scal_upts(0)[0], old_state)

    intg.advance_to(2.0)
    assert intg.system is system
    assert intg.tcurr == 2.0
    assert system.rhs_calls == [(1.0, 0, 1)]
    assert np.array_equal(system.ele_scal_upts(0)[0], old_state)


class _TriggerFlag:
    def __init__(self, active=False):
        self.active = active

    def __bool__(self):
        return self.active


class _ValidationIntegrator:
    def __init__(self):
        self.cfg = _cfg()
        mesh = type('Mesh', (), {
            'uuid': 'root-uuid', 'etypes': ['quad'], 'con_p': {}, 'mcon': {},
            'amr_tree': None,
            'spts_curved': {'quad': np.zeros(1, dtype=bool)},
        })()
        self.system = type(
            'System', (), {'name': 'euler', 'ndims': 2, 'mesh': mesh}
        )()
        self.backend = type('Backend', (), {'name': 'openmp'})()
        self.serialiser = type('Serialiser', (), {'_serialfns': {}})()
        self.triggers = _TriggerFlag()
        self.formulation = 'explicit'
        self.controller_name = 'none'
        self.stepper_name = 'rk4'
        self.nacptsteps = 1
        self.stepinfo = []
        self.plugins = ()


def test_online_quad_scope_fails_closed(monkeypatch):
    intg = _ValidationIntegrator()
    assert _validate_quad_online_integrator(intg) == (
        intg.system, intg.system.mesh, intg.system.mesh
    )

    intg.backend.name = 'cuda'
    assert _validate_quad_online_integrator(intg) == (
        intg.system, intg.system.mesh, intg.system.mesh
    )
    intg.backend.name = 'hip'
    with pytest.raises(AMRTransactionError, match='OpenMP and CUDA only'):
        _validate_quad_online_integrator(intg)
    intg.backend.name = 'openmp'

    intg.system.mesh.etypes = ['tri']
    with pytest.raises(AMRTransactionError, match='pure Quad'):
        _validate_quad_online_integrator(intg)
    intg.system.mesh.etypes = ['quad']

    intg.system.mesh.spts_curved['quad'][0] = True
    with pytest.raises(AMRTransactionError, match='affine Quad'):
        _validate_quad_online_integrator(intg)
    intg.system.mesh.spts_curved['quad'][0] = False

    intg.system.mesh.amr_tree = object()
    with pytest.raises(AMRTransactionError, match='typed Quad ancestry'):
        _validate_quad_online_integrator(intg)
    intg.system.mesh.amr_tree = None

    root_mesh = intg.system.mesh
    adapted_mesh = type('Mesh', (), {
        'uuid': 'adapted-uuid', 'etypes': ['quad'], 'con_p': {}, 'mcon': {},
        'amr_tree': encode_quad_leaf_tree('root-uuid', [(0, ())]),
        'spts_curved': {'quad': np.zeros(1, dtype=bool)},
    })()
    intg.system.mesh = adapted_mesh
    with pytest.raises(AMRTransactionError, match='requires its root anchor'):
        _validate_quad_online_integrator(intg)
    assert _validate_quad_online_integrator(
        intg, root_mesh
    ) == (intg.system, adapted_mesh, root_mesh)
    intg._amr_root_mesh = root_mesh
    assert _validate_quad_online_integrator(intg) == (
        intg.system, adapted_mesh, root_mesh
    )

    adapted_mesh.amr_tree = encode_quad_leaf_tree('other-root', [(0, ())])
    with pytest.raises(AMRTransactionError, match='persisted ancestry'):
        _validate_quad_online_integrator(intg)
    adapted_mesh.amr_tree = encode_quad_leaf_tree('root-uuid', [(0, ())])

    adapted_mesh.mcon = {
        'bad': type('MCon', (), {
            'format': 'one-to-many-v1', 'template': 'quad-2x2', 'nright': 4,
        })()
    }
    with pytest.raises(AMRTransactionError, match='unsupported mortar'):
        _validate_quad_online_integrator(intg)
    adapted_mesh.mcon = {}

    intg.system.mesh = root_mesh

    intg.cfg.set('solver-interfaces', 'mortar-implementation', 'fused')
    with pytest.raises(AMRTransactionError, match='staged mortar'):
        _validate_quad_online_integrator(intg)

    intg = _ValidationIntegrator()
    intg.plugins = (object(),)
    with pytest.raises(AMRTransactionError, match='live plugins do not match'):
        _validate_quad_online_integrator(intg)

    intg = _ValidationIntegrator()
    intg.triggers.active = True
    with pytest.raises(AMRTransactionError, match='live triggers'):
        _validate_quad_online_integrator(intg)

    intg = _ValidationIntegrator()
    intg.cfg.set('solver', 'shock-capturing', 'entropy-filter')
    with pytest.raises(AMRTransactionError, match='shock-capturing'):
        _validate_quad_online_integrator(intg)

    intg = _ValidationIntegrator()
    intg.cfg.set('solver-order-quad', 'order', '2')
    with pytest.raises(AMRTransactionError, match='mixed-p'):
        _validate_quad_online_integrator(intg)

    intg = _ValidationIntegrator()
    intg.cfg.set('solver-amr', 'max-level', '3')
    with pytest.raises(AMRTransactionError, match='maximum level'):
        _quad_online_settings(intg.cfg)

    class Comm:
        size = 2

    intg = _ValidationIntegrator()
    monkeypatch.setattr(amrtx, 'get_comm_rank_root', lambda: (Comm(), 0, 0))
    with pytest.raises(AMRTransactionError, match='one MPI rank'):
        _validate_quad_online_integrator(intg)
