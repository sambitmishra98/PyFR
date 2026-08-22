from io import StringIO
from pathlib import Path

import h5py
import numpy as np
import pytest

from pyfr.amroffline import OfflineQuadAMRError, perform_offline_quad_amr
from pyfr.inifile import Inifile
from pyfr.progress import NullProgressSequence
from pyfr.readers.gmsh import GmshReader
from pyfr.readers.native import NativeReader
from pyfr.shapes import QuadShape
from pyfr.writers.native import NativeWriter


FIELDS = ('rho', 'rhou', 'rhov', 'E')


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
[constants]
gamma = 1.4

[solver]
order = 1
anti-alias = none

[solver-elements-quad]
soln-pts = gauss-legendre
''')


def _root_and_solution(tmp_path):
    rootf = tmp_path / 'root.pyfrm'
    GmshReader(
        StringIO(_two_quad_gmsh()), NullProgressSequence()
    ).write(str(rootf), 1e-5)

    reader = NativeReader(str(rootf), construct_con=False)
    cfg = _cfg()
    basis = QuadShape(None, cfg)
    nupts = basis.nupts

    rho = np.ones((nupts, 2))
    u = np.zeros((nupts, 2))
    # One root has a clear D9Q velocity-spread mark; the other is constant.
    u[:, 0] = np.linspace(0.0, 0.4, nupts)
    v = np.zeros_like(u)
    p = np.full_like(u, 10.0)
    energy = p/0.4 + 0.5*rho*(u*u + v*v)
    state = np.stack((rho, rho*u, rho*v, energy), axis=1)

    stats = Inifile()
    stats.set('data', 'prefix', 'soln')
    stats.set('solver-time-integrator', 'tcurr', 1.0)

    solnf = tmp_path / 'root.pyfrs'
    writer = NativeWriter(
        reader.mesh, cfg, np.float64, tmp_path, solnf.name, 'soln'
    )
    writer.set_shapes_eidxs(
        {'quad': (4, nupts)}, reader.mesh.eidxs, {'soln': FIELDS}
    )
    writer.write(
        {'quad': {'soln': state.transpose(2, 1, 0)}}, 1.0,
        {'config': cfg.tostr(), 'stats': stats.tostr(),
         'mesh-uuid': reader.mesh.uuid},
    )
    writer.flush()
    reader.close()

    return rootf, solnf


def test_offline_quad_event_writes_restart_pair(tmp_path):
    rootf, solnf = _root_and_solution(tmp_path)
    outm, outs = tmp_path / 'event1.pyfrm', tmp_path / 'event1.pyfrs'

    result = perform_offline_quad_amr(
        rootf, solnf, outm, outs,
        refine_threshold=0.05, max_level=2,
    )

    assert result.old_leaves == 2
    assert result.new_leaves == 5
    assert result.refine_marks == 1
    assert result.closure_splits == 0
    assert result.max_score > 0.05
    assert result.conservation_residual < 1e-12

    reader = NativeReader(str(outm))
    try:
        assert len(tuple(reader.mesh.amr_tree.leaves())) == 5
        assert len(reader.mesh.mcon['line-1x2']) == 1
        soln = reader.load_soln(str(outs))
        assert np.isfinite(soln.data['quad']).all()
    finally:
        reader.close()


def test_offline_quad_repeated_event_uses_persisted_tree_and_root(tmp_path):
    rootf, solnf = _root_and_solution(tmp_path)
    event1m, event1s = tmp_path / 'event1.pyfrm', tmp_path / 'event1.pyfrs'
    perform_offline_quad_amr(
        rootf, solnf, event1m, event1s,
        refine_threshold=0.05, max_level=2,
    )

    missing_m = tmp_path / 'missing.pyfrm'
    missing_s = tmp_path / 'missing.pyfrs'
    with pytest.raises(OfflineQuadAMRError, match='immutable root mesh'):
        perform_offline_quad_amr(
            event1m, event1s, missing_m, missing_s,
            refine_threshold=0.01, max_level=2,
        )
    assert not missing_m.exists() and not missing_s.exists()

    event2m, event2s = tmp_path / 'event2.pyfrm', tmp_path / 'event2.pyfrs'
    result = perform_offline_quad_amr(
        event1m, event1s, event2m, event2s,
        root_mesh_path=rootf, refine_threshold=0.01, max_level=2,
    )
    assert result.old_leaves == 5
    assert result.new_leaves > result.old_leaves

    reader = NativeReader(str(event2m))
    try:
        leaves = tuple(reader.mesh.amr_tree.leaves())
        assert max(len(path) for _, path in leaves) == 2
        soln = reader.load_soln(str(event2s))
        assert np.isfinite(soln.data['quad']).all()
    finally:
        reader.close()


def test_offline_quad_wall_floor_is_topological(tmp_path):
    rootf, solnf = _root_and_solution(tmp_path)
    outm, outs = tmp_path / 'wall.pyfrm', tmp_path / 'wall.pyfrs'

    result = perform_offline_quad_amr(
        rootf, solnf, outm, outs,
        refine_threshold=1e9, max_level=1,
        wall_boundaries=('wall',), wall_min_level=1,
    )
    assert result.refine_marks == 0
    assert result.new_leaves == 8

    reader = NativeReader(str(outm))
    try:
        assert not reader.mesh.mcon
        assert all(
            len(path) == 1 for _, path in reader.mesh.amr_tree.leaves()
        )
    finally:
        reader.close()


def test_offline_quad_noop_fails_without_outputs(tmp_path):
    rootf, solnf = _root_and_solution(tmp_path)
    outm, outs = tmp_path / 'noop.pyfrm', tmp_path / 'noop.pyfrs'

    with pytest.raises(OfflineQuadAMRError, match='no refinement'):
        perform_offline_quad_amr(
            rootf, solnf, outm, outs,
            refine_threshold=1e9, max_level=2,
        )
    assert not outm.exists() and not outs.exists()


def test_offline_quad_rejects_wrong_root_anchor(tmp_path):
    rootf, solnf = _root_and_solution(tmp_path)
    event1m, event1s = tmp_path / 'event1.pyfrm', tmp_path / 'event1.pyfrs'
    perform_offline_quad_amr(
        rootf, solnf, event1m, event1s,
        refine_threshold=0.05, max_level=2,
    )

    other = tmp_path / 'other.pyfrm'
    GmshReader(
        StringIO(_two_quad_gmsh()), NullProgressSequence()
    ).write(str(other), 1e-5)
    with h5py.File(other, 'a') as f:
        del f['mesh-uuid']
        f['mesh-uuid'] = np.array('wrong-root-uuid', dtype='S')

    outm, outs = tmp_path / 'bad.pyfrm', tmp_path / 'bad.pyfrs'
    with pytest.raises(OfflineQuadAMRError, match='root anchor UUID mismatch'):
        perform_offline_quad_amr(
            event1m, event1s, outm, outs, root_mesh_path=other,
            refine_threshold=0.01, max_level=2,
        )
    assert not outm.exists() and not outs.exists()


def test_offline_quad_rejects_multiple_ranks(monkeypatch, tmp_path):
    class FakeComm:
        size = 2

    monkeypatch.setattr(
        'pyfr.amroffline.get_comm_rank_root',
        lambda: (FakeComm(), 0, 0),
    )

    outm, outs = tmp_path / 'out.pyfrm', tmp_path / 'out.pyfrs'
    with pytest.raises(OfflineQuadAMRError, match='one MPI rank'):
        perform_offline_quad_amr(
            'unused.pyfrm', 'unused.pyfrs', outm, outs,
            refine_threshold=0.05, max_level=2,
        )

    assert not outm.exists() and not outs.exists()


def test_offline_quad_does_not_overwrite_outputs(tmp_path):
    rootf, solnf = _root_and_solution(tmp_path)
    outm, outs = tmp_path / 'existing.pyfrm', tmp_path / 'new.pyfrs'
    outm.write_bytes(b'keep-me')

    with pytest.raises(OfflineQuadAMRError, match='already exists'):
        perform_offline_quad_amr(
            rootf, solnf, outm, outs,
            refine_threshold=0.05, max_level=2,
        )

    assert outm.read_bytes() == b'keep-me'
    assert not outs.exists()
