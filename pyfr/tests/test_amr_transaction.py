from io import StringIO

import numpy as np
import pytest

from pyfr.amrmesh import materialize_native_hex_tree
from pyfr.amrtransaction import (
    AMRTransactionError, _close_coarsening, _close_refinement,
    _conserved_totals, _current_tree, _inject_staged_hex_bank,
    _integration_weights, _mesh_volumes, _raw_volumes, _root_tree,
    _scratch_bank, _scratch_rhs, _single_hex_bank, _transfer_state,
    _validate_integrator, perform_one_amr_transaction,
)
from pyfr.amrwriter import write_adapted_mesh
from pyfr.inifile import Inifile
from pyfr.progress import NullProgressSequence
from pyfr.readers.gmsh import GmshReader
from pyfr.readers.native import NativeReader
from pyfr.solvers.base.system import BaseSystem


def _two_hex_gmsh():
    coords = []
    nid = {}
    n = 1
    for z in (0, 1):
        for y in (0, 1):
            for x in (0, 1, 2):
                nid[x, y, z] = n
                coords.append((n, x, y, z))
                n += 1

    def hnodes(x0, x1):
        return [
            nid[x0, 0, 0], nid[x1, 0, 0],
            nid[x1, 1, 0], nid[x0, 1, 0],
            nid[x0, 0, 1], nid[x1, 0, 1],
            nid[x1, 1, 1], nid[x0, 1, 1],
        ]

    hexes = [hnodes(0, 1), hnodes(1, 2)]
    fmaps = [
        [0, 1, 2, 3], [0, 1, 5, 4], [1, 2, 6, 5],
        [3, 2, 6, 7], [0, 3, 7, 4], [4, 5, 6, 7],
    ]
    quads = []
    for hi, h in enumerate(hexes):
        for fi, fmap in enumerate(fmaps):
            if (hi, fi) not in {(0, 2), (1, 4)}:
                quads.append([h[i] for i in fmap])

    lines = [
        '$MeshFormat', '2.2 0 8', '$EndMeshFormat',
        '$PhysicalNames', '2', '3 1 "fluid"', '2 2 "wall"',
        '$EndPhysicalNames', '$Nodes', str(len(coords)),
    ]
    lines.extend(f'{i} {x} {y} {z}' for i, x, y, z in coords)
    lines.extend(['$EndNodes', '$Elements', str(len(quads) + 2)])

    eid = 1
    for q in quads:
        lines.append(f'{eid} 3 2 2 2 ' + ' '.join(map(str, q)))
        eid += 1
    for h in hexes:
        lines.append(f'{eid} 5 2 1 1 ' + ' '.join(map(str, h)))
        eid += 1
    lines.append('$EndElements')

    return '\n'.join(lines) + '\n'


def _cfg():
    return Inifile('''
[backend]
precision = double
[backend-openmp]
cc = gcc-13
[constants]
gamma = 1.4
[solver]
system = euler
order = 3
anti-alias = none
[solver-time-integrator]
scheme = rk4
controller = none
tstart = 0
tend = 3e-6
dt = 1e-6
[solver-interfaces]
riemann-solver = rusanov
mortar-implementation = staged
[solver-interfaces-quad]
flux-pts = gauss-legendre
[solver-elements-hex]
soln-pts = gauss-legendre
[soln-bcs-wall]
type = slp-adia-wall
[soln-ics]
rho = 1
u = 0.05
v = 0
w = 0
p = 1
''')


def _ns_cfg():
    cfg = _cfg()
    cfg.set('solver', 'system', 'navier-stokes')
    cfg.set('solver', 'viscosity-correction', 'none')
    cfg.set('constants', 'mu', '0.01')
    cfg.set('constants', 'Pr', '0.72')
    cfg.set('solver-interfaces', 'ldg-beta', '0.0')
    cfg.set('solver-interfaces', 'ldg-tau', '0.1')
    return cfg


def _root_mesh(tmp_path):
    fname = tmp_path / 'root.pyfrm'
    GmshReader(StringIO(_two_hex_gmsh()), NullProgressSequence()).write(
        str(fname), 1e-5
    )
    return NativeReader(str(fname))


def test_d6a_native_closure_is_one_root_to_nine_leaves(tmp_path):
    reader = _root_mesh(tmp_path)
    try:
        old, local_by_leaf = _root_tree(reader.mesh)
        target = int(sorted(reader.mesh.eidxs['hex'])[0])
        proposed = _close_refinement(reader.mesh, old, (target, ()))
        raw = materialize_native_hex_tree(reader.mesh, proposed)

        assert list(old.leaves()) == [(0, ()), (1, ())]
        assert proposed.nleaves == 9
        assert len(raw.mortars) == 1
        assert all(m.format == 'one-to-many-v1' for m in raw.mortars)
        assert all(m.template == 'quad-2x2' for m in raw.mortars)
    finally:
        reader.close()


def test_d6a_transfer_uses_accepted_state_and_conserves_components(tmp_path):
    reader = _root_mesh(tmp_path)
    try:
        cfg = _cfg()
        old, local_by_leaf = _root_tree(reader.mesh)
        target = int(sorted(reader.mesh.eidxs['hex'])[0])
        proposed = _close_refinement(reader.mesh, old, (target, ()))

        state = np.ones((64, 5, 2))
        state[:, 1] = 0.05
        state[:, 2:4] = 0
        state[:, 4] = 2.5
        fake_system = type('FakeSystem', (), {'cfg': cfg})()
        transferred, basis = _transfer_state(
            fake_system, old, proposed, state, local_by_leaf
        )
        raw = materialize_native_hex_tree(reader.mesh, proposed)
        weights = _integration_weights(cfg, basis)
        before = _conserved_totals(
            state, _mesh_volumes(reader.mesh), weights
        )
        after = _conserved_totals(
            transferred, _raw_volumes(raw), weights
        )

        assert np.allclose(before, [2.0, 0.1, 0.0, 0.0, 5.0], rtol=0,
                           atol=2e-12)
        assert np.allclose(before, after, rtol=0, atol=2e-12)
        assert np.isclose(
            _mesh_volumes(reader.mesh).sum(), _raw_volumes(raw).sum(),
            rtol=0, atol=2e-12
        )
    finally:
        reader.close()


class _TriggerFlag:
    def __init__(self, active=False):
        self.active = active

    def __bool__(self):
        return self.active


class _ValidationIntegrator:
    def __init__(self, cfg=None):
        mesh = type('Mesh', (), {
            'amr_tree': None,
            'etypes': ['hex'],
            'con_p': {},
            'mcon': [],
            'spts_curved': {'hex': np.zeros(1, dtype=bool)},
        })()
        cfg = cfg or _cfg()
        system_name = cfg.get('solver', 'system')
        self.system = type(
            'System', (), {'name': system_name, 'mesh': mesh}
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
        self.cfg = cfg


def test_d6a_integrator_validation_accepts_supported_safe_point():
    intg = _ValidationIntegrator()

    system, mesh, root_mesh = _validate_integrator(intg)

    assert system is intg.system
    assert mesh is intg.system.mesh
    assert root_mesh is mesh


def test_d6d_integrator_validation_accepts_constant_viscosity_ns():
    intg = _ValidationIntegrator(_ns_cfg())

    system, mesh, root_mesh = _validate_integrator(intg)

    assert system is intg.system
    assert system.name == 'navier-stokes'
    assert mesh is intg.system.mesh
    assert root_mesh is mesh


def test_d6d_integrator_validation_rejects_variable_viscosity_ns():
    cfg = _ns_cfg()
    cfg.set('solver', 'viscosity-correction', 'sutherland')
    intg = _ValidationIntegrator(cfg)

    with pytest.raises(AMRTransactionError, match='constant viscosity'):
        _validate_integrator(intg)


@pytest.mark.parametrize(
    'setup, match', [
        (lambda i: setattr(i.backend, 'name', 'cuda'), 'OpenMP only'),
        (lambda i: setattr(i, 'nacptsteps', 0), 'completed accepted'),
        (lambda i: setattr(i, 'stepinfo', [object()]), 'accepted-step safe'),
        (lambda i: setattr(i.triggers, 'active', True),
         'trigger-managed state'),
        (lambda i: i.serialiser._serialfns.update({'triggers': object()}),
         'trigger-managed serialised state'),
        (lambda i: i.serialiser._serialfns.update(
            {'trigger-src/example': object()}),
         'trigger-managed serialised state'),
        (lambda i: i.serialiser._serialfns.update({'bcs/wall': object()}),
         'stateless boundary conditions'),
        (lambda i: i.serialiser._serialfns.update({'intg/unknown': object()}),
         'unsupported serialised mutable state'),
    ]
)
def test_d6a_integrator_validation_fails_closed(setup, match):
    intg = _ValidationIntegrator()
    setup(intg)

    with pytest.raises(AMRTransactionError, match=match):
        _validate_integrator(intg)


def test_d6a_trigger_rejection_occurs_before_transaction_preparation(
        monkeypatch):
    import pyfr.amrtransaction as amrtx

    intg = _ValidationIntegrator()
    intg.triggers.active = True

    def unexpected_prepare(*args, **kwargs):
        pytest.fail('transaction preparation ran after trigger rejection')

    monkeypatch.setattr(amrtx, '_root_tree', unexpected_prepare)

    with pytest.raises(AMRTransactionError, match='trigger-managed state'):
        perform_one_amr_transaction(intg, [(0, ())])


class _Matrix:
    def __init__(self, value):
        self.value = np.array(value, copy=True)
        self.ioshape = self.value.shape
        self.set_calls = 0

    def set(self, value):
        if tuple(value.shape) != self.ioshape:
            raise ValueError('Invalid matrix shape')
        self.set_calls += 1
        self.value[...] = value

    def get(self):
        return np.array(self.value, copy=True)


class _Backend:
    def __init__(self):
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1


class _ScratchSystem:
    ele_scal_upts = BaseSystem.ele_scal_upts

    def __init__(self, nrhs=3, shape=(2, 5, 1)):
        self.nrhs = nrhs
        self.backend = _Backend()
        self.ele_types = ['hex']
        self.ele_shapes = {'hex': tuple(shape)}
        self.ele_banks = [[
            _Matrix(np.full(shape, i, dtype=float))
            for i in range(nrhs)
        ]]
        self.rhs_calls = []

    def ele_scal_upts(self, bank):
        return (self.ele_banks[0][bank].value,)

    def rhs(self, t, uinbank, foutbank):
        self.rhs_calls.append((t, uinbank, foutbank))
        self.ele_banks[0][foutbank].value[...] = (
            self.ele_banks[0][uinbank].value + 3.0
        )


def test_d6a_scratch_rhs_keeps_staged_accepted_bank_untouched():
    system = _ScratchSystem()
    before = np.array(system.ele_scal_upts(0)[0], copy=True)
    scratch = _scratch_bank(system, 0)

    rhs, drift = _scratch_rhs(system, 1.25, 0, scratch)

    assert scratch == 1
    assert system.rhs_calls == [(1.25, 0, 1)]
    assert np.array_equal(system.ele_scal_upts(0)[0], before)
    assert np.array_equal(rhs[0], before + 3.0)
    assert drift == 0


def test_d6a_scratch_rhs_requires_a_distinct_rhs_register():
    with pytest.raises(AMRTransactionError, match='distinct RHS scratch'):
        _scratch_bank(_ScratchSystem(nrhs=1), 0)


def test_d6a_single_hex_bank_fails_closed_on_group_or_shape_mismatch():
    system = _ScratchSystem()
    system.ele_types = ['hex', 'quad']
    with pytest.raises(AMRTransactionError, match='exactly one Hex'):
        _single_hex_bank(system, 0, 'staged')

    system = _ScratchSystem()
    system.ele_banks[0][0].ioshape = (2, 5, 2)
    with pytest.raises(AMRTransactionError, match='shape does not match'):
        _single_hex_bank(system, 0, 'staged')


def test_d6a_staged_hex_bank_maps_physical_leaves_to_runtime_order(tmp_path):
    reader = _root_mesh(tmp_path)
    stage_reader = None
    try:
        cfg = _cfg()
        old, local_by_leaf = _root_tree(reader.mesh)
        target = int(sorted(reader.mesh.eidxs['hex'])[0])
        proposed = _close_refinement(reader.mesh, old, (target, ()))

        # Each root has a different coefficient field, while the nonconstant
        # pointwise term makes every refined child detectable by its own data.
        state = np.empty((64, 5, 2))
        coeff = np.arange(state.shape[0], dtype=float)
        for root_col in range(state.shape[2]):
            for var in range(state.shape[1]):
                state[:, var, root_col] = (
                    100000.0*root_col + 1000.0*var +
                    (var + 1)*coeff + coeff*coeff/100.0
                )

        fake_system = type('FakeSystem', (), {'cfg': cfg})()
        transferred, _ = _transfer_state(
            fake_system, old, proposed, state, local_by_leaf
        )
        assert transferred.shape == (64, 5, 9)
        raw = materialize_native_hex_tree(reader.mesh, proposed)
        stage_path = tmp_path / 'proposed.pyfrm'
        write_adapted_mesh(raw, str(stage_path))
        stage_reader = NativeReader(str(stage_path))
        stage_system = _ScratchSystem(shape=transferred.shape)

        staged_state, leaf_to_bank, bank_shape = _inject_staged_hex_bank(
            stage_system, stage_reader.mesh, raw, transferred, 0
        )

        # Reconstruct local runtime order independently from the materialized
        # D5 leaf-ordinal contract rather than from the injection helper.
        leaf_at_native_eidx = dict(enumerate(raw.leaf_order))
        runtime_leaves = tuple(
            leaf_at_native_eidx[int(eidx)]
            for eidx in stage_reader.mesh.eidxs['hex']
        )
        assert leaf_to_bank == tuple(
            zip(runtime_leaves, map(int, stage_reader.mesh.eidxs['hex']))
        )
        assert bank_shape == (64, 5, 9)
        assert stage_system.ele_banks[0][0].ioshape == (64, 5, 9)
        assert stage_system.ele_banks[0][0].set_calls == 1
        assert stage_system.backend.wait_count == 1

        # This uses the normal PyFR BaseSystem state-read interface.
        readback = stage_system.ele_scal_upts(0)[0]
        assert np.array_equal(readback, staged_state)
        fingerprints = {
            transferred[:, :, i].tobytes()
            for i in range(transferred.shape[2])
        }
        assert len(fingerprints) == len(runtime_leaves)
        for local_col, leaf in enumerate(runtime_leaves):
            physical_col = raw.leaf_order.index(leaf)
            assert np.array_equal(
                readback[:, :, local_col], transferred[:, :, physical_col]
            )
    finally:
        if stage_reader is not None:
            stage_reader.close()
        reader.close()


def test_d6a_stage_failure_leaves_old_system_untouched(monkeypatch):
    import pyfr.amrtransaction as amrtx

    class Tree:
        def leaves(self):
            return [(0, ())]

    class Intg:
        def __init__(self, system):
            self.system = system
            self.idxcurr = 0
            self.tcurr = 1.0
            self.cfg = object()

    system = _ScratchSystem()
    intg = Intg(system)
    old_state = np.array(system.ele_scal_upts(0)[0], copy=True)
    tree = Tree()
    transferred = np.array(old_state, copy=True)
    raw = type('Raw', (), {'leaf_order': [(0, ())]})()

    def fail_stage_write(raw, path):
        raise RuntimeError('staging failure')

    validate = lambda intg, restart_root_mesh=None: (
        system, object(), object()
    )
    monkeypatch.setattr(amrtx, '_validate_integrator', validate)
    monkeypatch.setattr(amrtx, '_current_tree',
                        lambda mesh: (tree, {(0, ()): 0}))
    monkeypatch.setattr(amrtx, '_close_refinement',
                        lambda mesh, old, mark: tree)
    monkeypatch.setattr(amrtx, '_eos_ranges',
                        lambda system, state: ((1.0, 1.0), (1.0, 1.0)))
    monkeypatch.setattr(amrtx, '_transfer_state',
                        lambda *args: (transferred, object()))
    monkeypatch.setattr(amrtx, '_mesh_volumes', lambda mesh: np.array([1.0]))
    monkeypatch.setattr(amrtx, '_raw_volumes', lambda raw: np.array([1.0]))
    monkeypatch.setattr(amrtx, '_integration_weights',
                        lambda cfg, basis: np.array([1.0]))
    monkeypatch.setattr(amrtx, '_conserved_totals',
                        lambda state, volumes, weights: np.ones(5))
    monkeypatch.setattr(amrtx, 'materialize_native_hex_tree',
                        lambda mesh, tree: raw)
    monkeypatch.setattr(amrtx, 'write_adapted_mesh', fail_stage_write)

    with pytest.raises(RuntimeError, match='staging failure'):
        perform_one_amr_transaction(intg, [(0, ())])

    assert intg.system is system
    assert intg.tcurr == 1.0
    assert intg.idxcurr == 0
    assert system.rhs_calls == []
    assert system.ele_banks[0][0].set_calls == 0
    assert np.array_equal(system.ele_scal_upts(0)[0], old_state)



def _materialize_reader(tmp_path, root_mesh, tree, name):
    raw = materialize_native_hex_tree(root_mesh, tree)
    path = tmp_path / name
    write_adapted_mesh(raw, path)
    return NativeReader(str(path)), raw


def test_d6b_repeated_refinement_closes_to_recursive_23_leaf_tree(tmp_path):
    root = _root_mesh(tmp_path)
    stage1 = None
    try:
        old, _ = _root_tree(root.mesh)
        target = int(sorted(root.mesh.eidxs['hex'])[0])
        tree1 = _close_refinement(root.mesh, old, (target, ()))
        stage1, raw1 = _materialize_reader(
            tmp_path, root.mesh, tree1, 'stage1.pyfrm'
        )
        persisted, local = _current_tree(stage1.mesh)
        assert tuple(persisted.leaves()) == tuple(tree1.leaves())
        assert set(local) == set(tree1.leaves())

        mark = (target, (1,))
        tree2 = _close_refinement(root.mesh, persisted, mark)
        raw2 = materialize_native_hex_tree(root.mesh, tree2)

        assert tree2.nleaves == 23
        assert len(raw2.mortars) == 4
        assert all(
            (m.format, m.template) == ('one-to-many-v1', 'quad-2x2')
            for m in raw2.mortars
        )
    finally:
        if stage1 is not None:
            stage1.close()
        root.close()


def test_d6b_legal_recursive_sibling_coarsening_to_16_leaves(tmp_path):
    root = _root_mesh(tmp_path)
    try:
        old, _ = _root_tree(root.mesh)
        target = int(sorted(root.mesh.eidxs['hex'])[0])
        tree1 = _close_refinement(root.mesh, old, (target, ()))
        parent = (target, (1,))
        tree2 = _close_refinement(root.mesh, tree1, parent)
        marks = {(target, (1, o)) for o in range(8)}
        tree3 = _close_coarsening(root.mesh, tree2, marks)
        raw3 = materialize_native_hex_tree(root.mesh, tree3)

        assert tree3.nleaves == 16
        assert parent in set(tree3.leaves())
        assert not (marks & set(tree3.leaves()))
        assert len(raw3.mortars) == 0
    finally:
        root.close()


def test_d6b_refine_then_restrict_reproduces_representable_parent(tmp_path):
    root = _root_mesh(tmp_path)
    try:
        cfg = _cfg()
        fake_system = type('FakeSystem', (), {'cfg': cfg})()
        old, map0 = _root_tree(root.mesh)
        target = int(sorted(root.mesh.eidxs['hex'])[0])
        tree1 = _close_refinement(root.mesh, old, (target, ()))

        # Distinguishable representable p3 state on each root.
        nupts = 64
        state0 = np.empty((nupts, 5, 2))
        x = np.arange(nupts, dtype=float)
        for e in range(2):
            for v in range(5):
                state0[:, v, e] = 10*e + v + 1e-3*x + 1e-6*x*x

        state1, _ = _transfer_state(
            fake_system, old, tree1, state0, map0
        )
        map1 = {leaf: i for i, leaf in enumerate(tree1.leaves())}
        parent = (target, (1,))
        tree2 = _close_refinement(root.mesh, tree1, parent)
        state2, _ = _transfer_state(
            fake_system, tree1, tree2, state1, map1
        )
        map2 = {leaf: i for i, leaf in enumerate(tree2.leaves())}
        marks = {(target, (1, o)) for o in range(8)}
        tree3 = _close_coarsening(root.mesh, tree2, marks)
        state3, _ = _transfer_state(
            fake_system, tree2, tree3, state2, map2
        )

        i1 = map1[parent]
        i3 = tuple(tree3.leaves()).index(parent)
        assert np.allclose(
            state3[:, :, i3], state1[:, :, i1], rtol=0, atol=2e-11
        )
    finally:
        root.close()


def test_d6b_repeated_adaptation_without_root_anchor_fails_closed():
    intg = _ValidationIntegrator()
    intg.system.mesh.amr_tree = type(
        'Tree', (), {'root_mesh_uuid': 'root-uuid'}
    )()
    intg.system.mesh.mcon = {}

    with pytest.raises(AMRTransactionError, match='immutable root mesh'):
        _validate_integrator(intg)


def test_d6b_repeated_adaptation_rejects_wrong_root_anchor():
    intg = _ValidationIntegrator()
    intg.system.mesh.amr_tree = type(
        'Tree', (), {'root_mesh_uuid': 'expected-root'}
    )()
    intg.system.mesh.mcon = {}
    intg._amr_root_mesh = type(
        'RootMesh', (), {'amr_tree': None, 'uuid': 'wrong-root'}
    )()

    with pytest.raises(AMRTransactionError, match='does not match'):
        _validate_integrator(intg)



def test_d6c_adapted_restart_accepts_matching_supplied_root_mesh():
    intg = _ValidationIntegrator()
    intg.system.mesh.amr_tree = type(
        'Tree', (), {'root_mesh_uuid': 'expected-root'}
    )()
    intg.system.mesh.mcon = {}
    root = type(
        'RootMesh', (), {'amr_tree': None, 'uuid': 'expected-root'}
    )()

    system, mesh, root_mesh = _validate_integrator(intg, root)

    assert system is intg.system
    assert mesh is intg.system.mesh
    assert root_mesh is root
    assert not hasattr(intg, '_amr_root_mesh')


def test_d6c_adapted_restart_rejects_wrong_supplied_root_mesh():
    intg = _ValidationIntegrator()
    intg.system.mesh.amr_tree = type(
        'Tree', (), {'root_mesh_uuid': 'expected-root'}
    )()
    intg.system.mesh.mcon = {}
    root = type(
        'RootMesh', (), {'amr_tree': None, 'uuid': 'wrong-root'}
    )()

    with pytest.raises(AMRTransactionError, match='does not match'):
        _validate_integrator(intg, root)


def test_d6c_adapted_restart_rejects_adapted_supplied_root_mesh():
    intg = _ValidationIntegrator()
    intg.system.mesh.amr_tree = type(
        'Tree', (), {'root_mesh_uuid': 'expected-root'}
    )()
    intg.system.mesh.mcon = {}
    root = type(
        'RootMesh', (), {
            'amr_tree': object(), 'uuid': 'expected-root'
        }
    )()

    with pytest.raises(AMRTransactionError, match='unadapted native mesh'):
        _validate_integrator(intg, root)


def test_d6c_rejects_supplied_root_conflicting_with_live_anchor():
    intg = _ValidationIntegrator()
    intg.system.mesh.amr_tree = type(
        'Tree', (), {'root_mesh_uuid': 'live-root'}
    )()
    intg.system.mesh.mcon = {}
    intg._amr_root_mesh = type(
        'RootMesh', (), {'amr_tree': None, 'uuid': 'live-root'}
    )()
    supplied = type(
        'RootMesh', (), {'amr_tree': None, 'uuid': 'other-root'}
    )()

    with pytest.raises(AMRTransactionError, match='differs from the live'):
        _validate_integrator(intg, supplied)
