from dataclasses import replace
from io import StringIO

import numpy as np
import pytest

from pyfr.amr import encode_hex_leaf_tree
from pyfr.amrmesh import materialize_native_mixed_hex_tree
from pyfr.amrcheckpoint import write_online_mixed_hex_checkpoint
from pyfr.amrtransaction import (
    AMRTransactionError, perform_indicator_mixed_hex_amr_transaction,
    perform_scripted_mixed_hex_amr_transaction,
)
from pyfr.amrwriter import (
    build_adapted_mixed_hex_mesh, write_adapted_mixed_hex_mesh,
)
from pyfr.backends import get_backend
from pyfr.inifile import Inifile
from pyfr.progress import NullProgressSequence
from pyfr.readers.gmsh import GmshReader
from pyfr.readers.native import NativeReader, Solution
from pyfr.shapes import HexShape, PyrShape, TetShape
from pyfr.solvers import get_solver


FIELDS = ['rho', 'rhou', 'rhov', 'rhow', 'E']


def _mixed_hex_gmsh():
    return '''\
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
3 1 "fluid"
2 2 "wall"
$EndPhysicalNames
$Nodes
14
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 0 0 1
6 1 0 1
7 1 1 1
8 0 1 1
9 2 0 0
10 2 1 0
11 2 0 1
12 2 1 1
13 3 0.5 0.5
14 2.5 0.5 -0.5
$EndNodes
$Elements
19
1 3 2 2 2 1 2 3 4
2 3 2 2 2 1 2 6 5
3 3 2 2 2 4 3 7 8
4 3 2 2 2 1 4 8 5
5 3 2 2 2 5 6 7 8
6 3 2 2 2 2 9 10 3
7 3 2 2 2 2 9 11 6
8 3 2 2 2 3 10 12 7
9 3 2 2 2 6 11 12 7
10 2 2 2 2 10 12 13
11 2 2 2 2 12 11 13
12 2 2 2 2 11 9 13
13 2 2 2 2 9 10 14
14 2 2 2 2 10 13 14
15 2 2 2 2 13 9 14
16 5 2 1 1 1 2 3 4 5 6 7 8
17 5 2 1 1 2 9 10 3 6 11 12 7
18 7 2 1 1 9 10 12 11 13
19 4 2 1 1 9 10 13 14
$EndElements
'''


def _root_reader(tmp_path):
    path = tmp_path / 'mixed3d-root.pyfrm'
    GmshReader(
        StringIO(_mixed_hex_gmsh()), NullProgressSequence()
    ).write(str(path), 1e-5)
    return NativeReader(str(path))


def _refined_tree(mesh):
    roots = tuple(map(int, mesh.eidxs['hex']))
    leaves = ((roots[0], ()),) + tuple((roots[1], (o,)) for o in range(8))
    return encode_hex_leaf_tree(mesh.uuid, leaves)


def _q2_curved_hex_root(mesh):
    """Elevate the tiny Hex region to a conforming curved q2 geometry."""
    h1 = HexShape.std_ele(1)
    h2 = HexShape.std_ele(2)
    c2 = HexShape.corner_pts_idxs(len(h2))
    next_id = int(np.max(mesh.node_idxs)) + 1
    coords = {
        int(i): np.array(c, copy=True)
        for i, c in zip(mesh.node_idxs, mesh.node_locs)
    }
    by_coord = {
        tuple(np.round(c, 14)): int(i) for i, c in coords.items()
    }
    rows = []
    spts = []
    for eidx, oldrow in enumerate(mesh.spts_nodes['hex']):
        corners = mesh.spts['hex'][:, eidx]
        lhs = np.column_stack((h1, np.ones(len(h1))))
        coeff = np.linalg.lstsq(lhs, corners, rcond=None)[0]
        phys = np.column_stack((h2, np.ones(len(h2)))) @ coeff
        x, y, z = phys[:, 0], phys[:, 1], phys[:, 2].copy()
        phys[:, 2] += 0.25*x*(2 - x)*y*(1 - y)*z*(1 - z)

        row = np.empty(len(h2), dtype=np.int64)
        for hi, pnt in enumerate(phys):
            if hi in c2:
                ci = int(np.flatnonzero(c2 == hi)[0])
                row[hi] = int(oldrow[ci])
                continue
            key = tuple(np.round(pnt, 14))
            nid = by_coord.get(key)
            if nid is None:
                nid = next_id
                next_id += 1
                by_coord[key] = nid
                coords[nid] = np.array(pnt, copy=True)
            row[hi] = nid
        rows.append(row)
        spts.append(phys)

    out = replace(mesh)
    out.spts_nodes = dict(mesh.spts_nodes)
    out.spts_nodes['hex'] = np.asarray(rows, dtype=np.int64)
    out.spts = dict(mesh.spts)
    out.spts['hex'] = np.stack(spts, axis=1)
    out.spts_curved = dict(mesh.spts_curved)
    out.spts_curved['hex'] = np.ones(len(rows), dtype=bool)
    out.node_idxs = np.array(sorted(coords), dtype=np.int64)
    out.node_locs = np.array([coords[i] for i in out.node_idxs])
    return out


def _cfg(system='euler'):
    wall = 'slp-adia-wall' if system == 'euler' else 'no-slp-adia-wall'
    ldg = '' if system == 'euler' else 'ldg-beta = 0.5\nldg-tau = 0.1\n'
    return Inifile(f'''\
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
order = 1
anti-alias = none
viscosity-correction = none
shock-capturing = none
[solver-time-integrator]
formulation = explicit
scheme = rk4
controller = none
tstart = 0
tend = 3e-8
dt = 1e-8
[solver-interfaces]
riemann-solver = rusanov
{ldg}mortar-implementation = staged
mortar-geom-tol = 1e-10
[solver-interfaces-tri]
flux-pts = williams-shunn
[solver-interfaces-quad]
flux-pts = gauss-legendre
[solver-elements-tet]
soln-pts = shunn-ham
[solver-elements-pyr]
soln-pts = gauss-legendre
[solver-elements-hex]
soln-pts = gauss-legendre
[soln-bcs-wall]
type = {wall}
[soln-ics]
rho = 1
u = 0.05
v = 0
w = 0
p = 1
''')


def _state(cfg, mesh):
    rho, u, p = 1.0, 0.05, 1.0
    energy = p/0.4 + 0.5*rho*u*u
    shapes = {'hex': HexShape, 'pyr': PyrShape, 'tet': TetShape}
    result = {}
    for etype, shape in shapes.items():
        basis = shape(None, cfg)
        state = np.zeros((basis.nupts, 5, len(mesh.eidxs[etype])))
        state[:, 0] = rho
        state[:, 1] = rho*u
        state[:, 4] = energy
        result[etype] = state
    return result


def _integrator(tmp_path, system='euler'):
    reader = _root_reader(tmp_path)
    cfg = _cfg(system)
    stats = Inifile()
    stats.set('solver-time-integrator', 'tcurr', 0.0)
    soln = Solution(
        config=cfg, stats=stats, fields=FIELDS,
        data=_state(cfg, reader.mesh), state={},
    )
    intg = get_solver(get_backend('openmp', cfg), reader.mesh, soln, cfg)
    intg.advance_to(1e-8)
    return reader, intg


def _quad_mcons(mesh):
    return tuple(
        mcon for mcon in mesh.mcon.values()
        if mcon.format == 'one-to-many-v1'
        and mcon.template == 'quad-2x2'
    )


def test_mixed_hex_materializer_builds_hex_and_pyr_quad2x2(tmp_path):
    root = _root_reader(tmp_path)
    adapted = None
    try:
        raw = materialize_native_mixed_hex_tree(
            root.mesh, _refined_tree(root.mesh)
        )
        assert len(raw.hex_nodes) == 9
        assert len(raw.fixed_nodes['pyr']) == 1
        assert len(raw.fixed_nodes['tet']) == 1
        assert [(m.left_etype, m.right_etype) for m in raw.mortars] == [
            ('hex', ('hex',)*4), ('pyr', ('hex',)*4)
        ]

        direct = build_adapted_mixed_hex_mesh(raw)
        path = tmp_path / 'mixed3d-adapted.pyfrm'
        write_adapted_mixed_hex_mesh(raw, str(path))
        adapted = NativeReader(str(path))
        native = adapted.mesh

        assert direct.uuid == native.uuid
        assert direct.etypes == native.etypes == ['hex', 'pyr', 'tet']
        for etype in ('pyr', 'tet'):
            assert np.array_equal(direct.spts[etype], root.mesh.spts[etype])
            assert np.array_equal(direct.tags[etype], root.mesh.tags[etype])
        assert tuple(native.amr_tree.leaves()) == tuple(raw.leaf_order)
        assert native.amr_tree.root_mesh_uuid == root.mesh.uuid
        assert len(_quad_mcons(native)) == 2
        names = tuple(native.mcon)
        assert 'quad-2x2-hex-to-hex' in names
        assert 'quad-2x2-pyr-to-hex' in names
    finally:
        root.close()
        if adapted is not None:
            adapted.close()


@pytest.mark.parametrize('system', ['euler', 'navier-stokes'])
def test_mixed_hex_live_event_first_rhs_and_continuation(tmp_path, system):
    root, intg = _integrator(tmp_path, system)
    try:
        iid = id(intg)
        old_system = intg.system
        accepted_before = {
            et: np.array(a, copy=True)
            for et, a in zip(
                old_system.ele_types,
                old_system.ele_scal_upts(intg.idxcurr),
            )
        }
        target = int(root.mesh.eidxs['hex'][1])
        tx = perform_scripted_mixed_hex_amr_transaction(
            intg, (target, ())
        )

        assert id(intg) == iid
        assert tx.old_system_id == id(old_system)
        assert tx.new_system_id == id(intg.system)
        assert tx.stage_leaf_count == 9
        assert tx.stage_pyr_count == tx.stage_tet_count == 1
        assert tx.stage_mortar_count == 2
        assert tx.bank_drift == 0
        assert np.max(np.abs(tx.conservation_error)) < 2e-12
        assert tx.rho_range[0] > 0 and tx.pressure_range[0] > 0
        assert all(np.isfinite(a).all() for a in tx.rhs)
        for etype in ('pyr', 'tet'):
            assert np.array_equal(
                tx.transferred_state[etype], accepted_before[etype]
            )

        intg.advance_to(2e-8)
        assert intg.tcurr == pytest.approx(2e-8)
        assert intg.nacptsteps == 2
        assert all(
            np.isfinite(a).all()
            for a in intg.system.ele_scal_upts(intg.idxcurr)
        )
    finally:
        root.close()


def test_mixed_hex_l2_against_pyramid_fails_with_exact_rollback(tmp_path):
    root, intg = _integrator(tmp_path)
    try:
        target = int(root.mesh.eidxs['hex'][1])
        perform_scripted_mixed_hex_amr_transaction(intg, (target, ()))

        system = intg.system
        mesh_uuid = intg.mesh_uuid
        before = {
            et: np.array(a, copy=True)
            for et, a in zip(
                system.ele_types, system.ele_scal_upts(intg.idxcurr)
            )
        }
        with pytest.raises(AMRTransactionError, match='L2 Hex contact'):
            perform_scripted_mixed_hex_amr_transaction(
                intg, (target, (1,))
            )

        assert intg.system is system
        assert intg.mesh_uuid == mesh_uuid
        after = {
            et: np.array(a, copy=True)
            for et, a in zip(
                system.ele_types, system.ele_scal_upts(intg.idxcurr)
            )
        }
        assert all(np.array_equal(before[et], after[et]) for et in before)
    finally:
        root.close()


def test_mixed_hex_repeated_refinement_away_from_pyramid(tmp_path):
    root, intg = _integrator(tmp_path)
    try:
        target = int(root.mesh.eidxs['hex'][0])
        first = perform_scripted_mixed_hex_amr_transaction(
            intg, (target, ())
        )
        assert first.stage_leaf_count == 9
        intg.advance_to(2e-8)

        second = perform_scripted_mixed_hex_amr_transaction(
            intg, (target, (0,))
        )
        assert second.stage_leaf_count == 16
        assert max(len(path) for _, path in second.proposed_tree.leaves()) == 2
        assert second.stage_pyr_count == second.stage_tet_count == 1
        assert second.bank_drift == 0
        assert np.max(np.abs(second.conservation_error)) < 2e-12

        intg.advance_to(3e-8)
        assert intg.tcurr == pytest.approx(3e-8)
        assert intg.nacptsteps == 3
    finally:
        root.close()


def test_mixed_hex_d9q_indicator_marks_only_hex(tmp_path):
    root, intg = _integrator(tmp_path)
    try:
        cfg = intg.cfg
        cfg.set('solver-amr', 'indicator', 'density-velocity-variation')
        cfg.set('solver-amr', 'refine-threshold', 0.01)
        cfg.set('solver-amr', 'max-level', 2)
        cfg.set('solver-amr', 'wall-min-level', 0)
        cfg.set('solver-amr', 'wall-boundaries', '')

        hidx = intg.system.ele_types.index('hex')
        state = np.array(
            intg.system.ele_scal_upts(intg.idxcurr)[hidx], copy=True
        )
        state[0, 0, 0] *= 1.05
        intg.system.ele_banks[hidx][intg.idxcurr].set(state)
        intg.backend.wait()

        result = perform_indicator_mixed_hex_amr_transaction(intg)
        target = int(root.mesh.eidxs['hex'][0])
        assert result.decision.action == 'refine'
        assert result.decision.marks == ((target, ()),)
        assert result.transaction.stage_leaf_count == 9
        assert result.transaction.stage_pyr_count == 1
        assert result.transaction.stage_tet_count == 1
        assert result.transaction.bank_drift == 0
    finally:
        root.close()


def test_mixed_hex_checkpoint_restart_and_schedule_rearm(tmp_path):
    root = _root_reader(tmp_path)
    adapted = None
    try:
        cfg = _cfg('euler')
        cfg.set('solver-amr', 'indicator', 'density-velocity-variation')
        cfg.set('solver-amr', 'refine-threshold', 1.0)
        cfg.set('solver-amr', 'max-level', 2)
        cfg.set('solver-amr', 'wall-boundaries', 'wall')
        cfg.set('solver-amr', 'wall-min-level', 1)
        cfg.set('solver-amr', 'schedule-dt', 1e-8)
        cfg.set('solver-amr', 'checkpoint-dir', str(tmp_path / 'checkpoints'))

        stats = Inifile()
        stats.set('solver-time-integrator', 'tcurr', 0.0)
        soln = Solution(
            config=cfg, stats=stats, fields=FIELDS,
            data=_state(cfg, root.mesh), state={},
        )
        intg = get_solver(get_backend('openmp', cfg), root.mesh, soln, cfg)
        intg.advance_to(1e-8)

        assert intg.amr_schedule.mode == 'mixed-hex-1r'
        assert len(intg.amr_schedule.history) == 1
        assert intg.amr_schedule.history[0].action == 'refine'
        assert len(intg.amr_schedule.checkpoints) == 1
        checkpoint = intg.amr_schedule.checkpoints[0]
        assert checkpoint.hex_leaf_count == 16
        assert checkpoint.pyramid_count == checkpoint.tet_count == 1

        explicit = write_online_mixed_hex_checkpoint(
            intg, tmp_path / 'explicit.pyfrm', tmp_path / 'explicit.pyfrs'
        )
        assert explicit.hex_leaf_count == 16

        adapted = NativeReader(checkpoint.mesh_path)
        restart = adapted.load_soln(checkpoint.soln_path)
        restart.config.set(
            'solver-amr', 'root-mesh', str(tmp_path / 'mixed3d-root.pyfrm')
        )
        restart.config.set('solver-amr', 'checkpoint-dir', str(tmp_path / 'rchk'))
        restarted = get_solver(
            get_backend('openmp', restart.config), adapted.mesh,
            restart, restart.config,
        )
        assert restarted.amr_schedule.mode == 'mixed-hex-1r'
        assert restarted._amr_root_mesh.uuid == root.mesh.uuid
        restarted.advance_to(2e-8)
        assert restarted.tcurr == pytest.approx(2e-8)
        assert len(restarted.amr_schedule.history) == 1
        assert restarted.amr_schedule.history[0].action == 'none'
        assert all(
            np.isfinite(a).all()
            for a in restarted.system.ele_scal_upts(restarted.idxcurr)
        )
    finally:
        root.close()
        if adapted is not None:
            adapted.close()


def test_mixed_hex_q2_root_restriction_and_p3_curved_mortar(tmp_path):
    root = _root_reader(tmp_path)
    try:
        q2root = _q2_curved_hex_root(root.mesh)
        raw = materialize_native_mixed_hex_tree(q2root, _refined_tree(q2root))
        assert raw.hex_nodes.shape == (9, 27)
        assert np.all(raw.hex_curved)
        direct = build_adapted_mixed_hex_mesh(raw)
        assert np.count_nonzero(direct.spts_curved['hex']) == 9
        assert np.array_equal(direct.spts['pyr'], q2root.spts['pyr'])
        assert np.array_equal(direct.spts['tet'], q2root.spts['tet'])

        cfg3 = _cfg('navier-stokes')
        cfg3.set('solver', 'order', 3)
        soln3 = Solution(
            config=cfg3, stats=Inifile(), fields=FIELDS,
            data=_state(cfg3, direct), state={},
        )
        soln3.stats.set('solver-time-integrator', 'tcurr', 0.0)
        intg = get_solver(get_backend('openmp', cfg3), direct, soln3, cfg3)
        intg.advance_to(1e-8)
        assert intg.tcurr == pytest.approx(1e-8)
        assert all(
            np.isfinite(a).all()
            for a in intg.system.ele_scal_upts(intg.idxcurr)
        )
    finally:
        root.close()
