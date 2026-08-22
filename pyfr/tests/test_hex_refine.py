from io import StringIO

import h5py
import numpy as np
import pytest

from pyfr.progress import NullProgressSequence
from pyfr.readers.gmsh import GmshReader
from pyfr.readers.native import NativeReader


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
            if (hi, fi) in {(0, 2), (1, 4)}:
                continue
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


def _reader():
    return GmshReader(StringIO(_two_hex_gmsh()), NullProgressSequence())


def _hex_volume(points, row):
    p = points[row]
    return abs(np.linalg.det(np.column_stack((
        p[1] - p[0], p[3] - p[0], p[4] - p[0]
    ))))


def test_one_hex_octree_refinement_generates_quad_2x2_mortar(tmp_path):
    reader = _reader()
    parent = reader._elenodes[5, (1,)][1].copy()
    parent_volume = _hex_volume(reader._nodepts, parent)

    summary = reader.refine_hexes('12')

    assert summary == {
        'selected': 1,
        'generated-hexes': 8,
        'mortars': 1,
        'boundary-splits': 5,
        'conforming-refined-faces': 0,
    }
    assert len(reader._elenodes[5, (1,)]) == 9
    assert len(reader._elenodes[3, (2,)]) == 25

    children = reader._elenodes[5, (1,)][1:]
    child_volume = sum(_hex_volume(reader._nodepts, c) for c in children)
    assert child_volume == pytest.approx(parent_volume, rel=5e-13, abs=5e-13)

    outf = tmp_path / 'refined.pyfrm'
    reader.write(outf, 1e-5)

    with h5py.File(outf, 'r') as f:
        assert len(f['eles/hex']) == 9
        assert list(f['mortars']) == ['quad-2x2']
        m = f['mortars/quad-2x2']
        assert m.attrs['format'] == 'one-to-many-v1'
        assert m.attrs['template'] == 'quad-2x2'
        assert len(m) == 1
        assert m.dtype['right_cidx'].shape == (4,)

    native = NativeReader(outf)
    try:
        mcon = native.mesh.mcon['quad-2x2']
        assert mcon.format == 'one-to-many-v1'
        assert mcon.template == 'quad-2x2'
        assert mcon.nright == 4
    finally:
        native.close()


def test_adjacent_refined_hexes_are_conforming():
    reader = _reader()
    summary = reader.refine_hexes('11,12')

    assert summary['selected'] == 2
    assert summary['generated-hexes'] == 16
    assert summary['mortars'] == 0
    assert summary['conforming-refined-faces'] == 1
    assert len(reader._elenodes[5, (1,)]) == 16

    raw = reader._to_raw_mesh(1e-5)
    assert len(raw[1]['hex']) == 16
    assert raw[-1] == {}


def test_hex_refinement_all_selects_every_hex():
    reader = _reader()
    summary = reader.refine_hexes('all')
    assert summary['selected'] == 2
    assert summary['mortars'] == 0


def test_hex_refinement_rejects_unknown_tag():
    reader = _reader()
    with pytest.raises(ValueError, match='Unknown Gmsh element tags'):
        reader.refine_hexes('999')


def test_hex_refinement_rejects_non_volume_hex_tag():
    reader = _reader()
    with pytest.raises(ValueError, match='only complete volume Hex8'):
        reader.refine_hexes('1')


def test_hex_refinement_rejects_non_affine_geometry():
    reader = _reader()
    row = reader._elenodes[5, (1,)][1]
    reader._nodepts[row[6]] += [0.1, 0.0, 0.0]

    with pytest.raises(ValueError, match='requires affine Hex8 geometry'):
        reader.refine_hexes('12')


def test_hex_refinement_children_use_parent_3x3x3_grid():
    reader = _reader()
    parent = reader._elenodes[5, (1,)][1].copy()
    p = reader._nodepts[parent]
    origin = p[0]
    axes = np.column_stack((p[1] - p[0], p[3] - p[0], p[4] - p[0]))

    reader.refine_hexes('12')
    children = reader._elenodes[5, (1,)][1:]
    child_nodes = np.unique(children)
    actual = reader._nodepts[child_nodes]
    expected = np.array([
        origin + axes @ np.array([i, j, k], dtype=float) / 2
        for k in range(3) for j in range(3) for i in range(3)
    ])

    assert len(child_nodes) == 27
    for point in expected:
        assert np.min(np.linalg.norm(actual - point, axis=1)) <= 5e-13


def test_hex_refinement_rejects_high_order_hex_target():
    reader = _reader()
    ekey = (12, (1,))
    reader._elenodes[ekey] = np.zeros((1, 27), dtype=np.int64)
    reader._eletags[ekey] = np.array([99], dtype=np.int64)

    with pytest.raises(ValueError, match='only complete volume Hex8'):
        reader.refine_hexes('99')


def test_hex_refinement_rejects_periodic_touching_cell():
    reader = _reader()
    boundary = reader._boundary_quad_lookup().copy()
    row = reader._elenodes[5, (1,)][1]

    for fmap in reader._petype_fnmap['hex']['quad']:
        qkey = tuple(sorted(int(n) for n in row[fmap]))
        if qkey in boundary:
            _, bkey, bidx = boundary[qkey]
            boundary[qkey] = ('periodic', bkey, bidx)
            break
    else:
        raise AssertionError('No selected exterior face found')

    reader._boundary_quad_lookup = lambda: boundary
    with pytest.raises(ValueError, match='does not refine periodic'):
        reader.refine_hexes('12')


def test_hex_refinement_rejects_high_order_boundary_face():
    reader = _reader()
    boundary = reader._boundary_quad_lookup().copy()
    row = reader._elenodes[5, (1,)][1]

    for fmap in reader._petype_fnmap['hex']['quad']:
        qkey = tuple(sorted(int(n) for n in row[fmap]))
        if qkey in boundary:
            boundary[qkey] = ('fixed', (10, (2,)), 0)
            break
    else:
        raise AssertionError('No selected exterior face found')

    reader._boundary_quad_lookup = lambda: boundary
    with pytest.raises(ValueError, match='requires Quad4'):
        reader.refine_hexes('12')


def test_hex_refinement_rejects_nonhex_interior_neighbour():
    reader = _reader()
    faces = reader._volume_face_lookup()
    row = reader._elenodes[5, (1,)][1]

    for fmap in reader._petype_fnmap['hex']['quad']:
        qkey = tuple(sorted(int(n) for n in row[fmap]))
        if len(faces[qkey]) == 2:
            owners = list(faces[qkey])
            oi = 0 if owners[0][1] != 1 else 1
            owners[oi] = (*owners[oi][:3], 'tet', 4)
            faces[qkey] = owners
            break
    else:
        raise AssertionError('No selected interior face found')

    reader._volume_face_lookup = lambda: faces
    with pytest.raises(ValueError, match='unrefined Hex8 neighbour'):
        reader.refine_hexes('12')


def test_recursive_hex_refinement_balances_face_neighbours(tmp_path):
    reader = _reader()
    summary = reader.refine_hexes('12/4')

    assert summary == {
        'selected': 1,
        'generated-hexes': 23,
        'mortars': 4,
        'boundary-splits': 10,
        'conforming-refined-faces': 0,
    }

    rows = reader._elenodes[5, (1,)]
    volumes = np.array([
        _hex_volume(reader._nodepts, row) for row in rows
    ])
    assert np.count_nonzero(np.isclose(volumes, 1/8)) == 15
    assert np.count_nonzero(np.isclose(volumes, 1/64)) == 8

    outf = tmp_path / 'recursive.pyfrm'
    reader.write(outf, 1e-5)
    native = NativeReader(outf)
    try:
        mcon = native.mesh.mcon['quad-2x2']
        assert mcon.nright == 4
        assert len(mcon) == 4
    finally:
        native.close()


def test_deep_recursive_hex_refinement_never_emits_1_to_16(tmp_path):
    reader = _reader()
    summary = reader.refine_hexes('12/4/0')

    assert summary['selected'] == 1
    assert summary['generated-hexes'] == 44
    assert summary['mortars'] == 13

    rows = reader._elenodes[5, (1,)]
    volumes = np.array([
        _hex_volume(reader._nodepts, row) for row in rows
    ])
    assert np.count_nonzero(np.isclose(volumes, 1/8)) == 13
    assert np.count_nonzero(np.isclose(volumes, 1/64)) == 23
    assert np.count_nonzero(np.isclose(volumes, 1/512)) == 8

    outf = tmp_path / 'deep.pyfrm'
    reader.write(outf, 1e-5)
    with h5py.File(outf, 'r') as f:
        assert list(f['mortars']) == ['quad-2x2']
        assert len(f['mortars/quad-2x2']) == 13

    native = NativeReader(outf)
    try:
        assert native.mesh.mcon['quad-2x2'].nright == 4
    finally:
        native.close()


def test_recursive_hex_refinement_selector_order_is_deterministic():
    lhs, rhs = _reader(), _reader()
    lsummary = lhs.refine_hexes('12/4,11')
    rsummary = rhs.refine_hexes('11,12/4,12/4')

    assert lsummary == rsummary
    assert np.array_equal(lhs._nodepts, rhs._nodepts)
    assert lhs._hex_refine_mortars == rhs._hex_refine_mortars
    assert lhs._elenodes.keys() == rhs._elenodes.keys()
    for key in lhs._elenodes:
        assert np.array_equal(lhs._elenodes[key], rhs._elenodes[key])
        assert np.array_equal(lhs._eletags[key], rhs._eletags[key])


def test_recursive_hex_refinement_is_idempotent():
    reader = _reader()
    summary = reader.refine_hexes('12/4')
    nodepts = reader._nodepts.copy()
    elenodes = {k: v.copy() for k, v in reader._elenodes.items()}

    assert reader.refine_hexes('12/4/0') == summary
    assert np.array_equal(reader._nodepts, nodepts)
    for key, rows in elenodes.items():
        assert np.array_equal(reader._elenodes[key], rows)



def test_balance_closure_rejects_non_affine_coarse_neighbour():
    reader = _reader()
    row = reader._elenodes[5, (1,)][0]
    reader._nodepts[row[6]] += [0.1, 0.0, 0.0]

    with pytest.raises(ValueError, match='requires affine Hex8 geometry'):
        reader.refine_hexes('12/4')


def test_balance_closure_rejects_periodic_coarse_neighbour():
    reader = _reader()
    boundary = reader._boundary_quad_lookup().copy()
    row = reader._elenodes[5, (1,)][0]

    for fmap in reader._petype_fnmap['hex']['quad']:
        qkey = tuple(sorted(int(n) for n in row[fmap]))
        if qkey in boundary:
            _, bkey, bidx = boundary[qkey]
            boundary[qkey] = ('periodic', bkey, bidx)
            break
    else:
        raise AssertionError('No coarse-neighbour exterior face found')

    reader._boundary_quad_lookup = lambda: boundary
    with pytest.raises(ValueError, match='does not refine periodic'):
        reader.refine_hexes('12/4')

def test_recursive_hex_refinement_rejects_bad_octant_path():
    reader = _reader()
    with pytest.raises(ValueError, match='octants must be in 0..7'):
        reader.refine_hexes('12/8')

    reader = _reader()
    with pytest.raises(ValueError, match='selector must be'):
        reader.refine_hexes('12//3')
