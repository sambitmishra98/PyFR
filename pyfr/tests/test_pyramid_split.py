from collections import Counter

import numpy as np
import pytest

from pyfr.polys import get_polybasis
from pyfr.readers.gmsh import GmshReader
from pyfr.shapes import PyrShape, TetShape, TriShape


_PYR_ETYPES = {1: 7, 2: 14, 3: 118, 4: 119}
_QUAD_ETYPES = {1: 3, 2: 10, 3: 36, 4: 37}
_TRI_ETYPES = {1: 2, 2: 9, 3: 21, 4: 23}


def _gmsh_row(petype, ids):
    nodemap = GmshReader._nodemaps[petype, len(ids)]
    return np.asarray(ids)[np.argsort(nodemap)]


def _pyramid(order, offset=0, *, zsign=1.0):
    points = PyrShape.std_ele(order)
    r, s, t = points.T
    coords = np.column_stack((
        1.2*r + 0.05*r*s,
        0.9*s + 0.03*r*t,
        zsign*(1.1*t + 0.02*r*s),
    ))
    coords += [0.3, -0.2, 0.1]

    ids = np.arange(offset, offset + len(points))
    return coords, _gmsh_row('pyr', ids)


def _tet_rows(reader):
    keys = [
        key for key in reader._elenodes
        if reader._etype_map[key[0]][0] == 'tet'
    ]
    assert len(keys) == 1
    return keys[0], reader._elenodes[keys[0]]


def _tet_random_points(npts=250):
    rng = np.random.default_rng(1741)
    weights = rng.exponential(size=(npts, 4))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ TetShape.std_ele(1)


def _child_geometry_error(reader, parent, row):
    nnodes = len(row)
    order = TetShape.order_from_npts(nnodes)
    tpts = TetShape.std_ele(order)
    tids = row[reader._nodemaps['tet', nnodes]]
    tcoords = reader._nodepts[tids]
    tbasis = get_polybasis('tet', order, tpts)

    porder = PyrShape.order_from_npts(len(parent))
    ppts = PyrShape.std_ele(porder)
    pids = parent[reader._nodemaps['pyr', len(parent)]]
    pcoords = reader._nodepts[pids]
    pbasis = get_polybasis('pyr', porder, ppts)

    pref = PyrShape.std_ele(1)
    gref = pref[np.argsort(reader._nodemaps['pyr', 5])]
    refbyid = {
        int(node): point for node, point in zip(parent[:5], gref)
    }
    corners = tids[TetShape.corner_pts_idxs(nnodes)]
    childref = np.asarray([refbyid[int(node)] for node in corners])
    childmap = reader._affine_map(TetShape.std_ele(1), childref)

    rpts = _tet_random_points()
    expected = pbasis.nodal_basis_at(childmap(rpts)) @ pcoords
    actual = tbasis.nodal_basis_at(rpts) @ tcoords
    return np.max(np.abs(actual - expected))


@pytest.mark.parametrize('order', [1, 2, 3, 4])
def test_complete_pyramid_split_geometry_orders(order):
    points, parent = _pyramid(order)
    reader = GmshReader.__new__(GmshReader)
    reader._nodepts = points
    reader._elenodes = {
        (_PYR_ETYPES[order], 7): parent[None, :]
    }

    reader.split_pyramids()

    _, children = _tet_rows(reader)
    target_order = 2*order
    assert reader._pyramid_target_tet_order == target_order
    assert children.shape == (
        2, TetShape.npts_from_order(target_order)
    )
    assert len(reader._pyramid_mortars) == 1
    assert reader._pyramid_mortars[0]['quality'] > 0

    errors = [
        _child_geometry_error(reader, parent, child)
        for child in children
    ]
    assert max(errors) < 2e-9


@pytest.mark.parametrize('order', [1, 2, 3, 4])
def test_boundary_pyramid_split_inherits_boundary_tag(order):
    points, parent = _pyramid(order)
    qetype = _QUAD_ETYPES[order]
    qnnodes = GmshReader._etype_map[qetype][1]
    qrow = np.resize(parent[:4], qnnodes)

    reader = GmshReader.__new__(GmshReader)
    reader._nodepts = points
    reader._bfacespents = {'wall': 1}
    reader._pfacespents = {}
    reader._elenodes = {
        (_PYR_ETYPES[order], (2,)): parent[None, :],
        (qetype, (1,)): qrow[None, :],
    }

    reader.split_pyramids()

    assert (qetype, (1,)) not in reader._elenodes
    trietype = _TRI_ETYPES[order]
    triangles = reader._elenodes[trietype, (1,)]
    assert triangles.shape == (
        2, TriShape.npts_from_order(order)
    )

    _, children = _tet_rows(reader)
    child_faces = {
        frozenset(corners[:3]) for corners in children[:, :4]
    }
    boundary_faces = {
        frozenset(tri[:3]) for tri in triangles
    }
    assert boundary_faces == child_faces
    assert reader._pyramid_mortars == []
    assert len(reader._pyramid_boundary_splits) == 1


@pytest.mark.parametrize('existing_order', [1, 2, 3, 4])
def test_existing_tets_promote_to_generated_order(existing_order):
    ppoints, parent = _pyramid(2)

    tpoints = TetShape.std_ele(existing_order)
    tcoords = tpoints @ np.array([
        [0.8, 0.0, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, 0.0, 0.9],
    ])
    tcoords += [3.0, 0.0, 0.0]
    tids = np.arange(len(ppoints), len(ppoints) + len(tpoints))
    trow = _gmsh_row('tet', tids)

    reader = GmshReader.__new__(GmshReader)
    reader._nodepts = np.vstack((ppoints, tcoords))
    reader._elenodes = {
        (14, 7): parent[None, :],
        ({1: 4, 2: 11, 3: 29, 4: 30}[existing_order], 7):
            trow[None, :],
    }

    reader.split_pyramids()

    _, tets = _tet_rows(reader)
    assert reader._pyramid_target_tet_order == 4
    assert tets.shape == (3, TetShape.npts_from_order(4))


@pytest.mark.parametrize('bad_order', [2, 3])
def test_split_rejects_invalid_high_order_children(bad_order):
    points, parent = _pyramid(bad_order)
    pids = parent[GmshReader._nodemaps['pyr', len(parent)]]

    # Pull one non-corner geometry node far into the volume.  Both diagonal
    # choices then contain a nonpositive high-order tetrahedral child map.
    points[pids[1]] += [1.5, 0.0, 0.0]

    reader = GmshReader.__new__(GmshReader)
    reader._nodepts = points
    reader._elenodes = {
        (_PYR_ETYPES[bad_order], 7): parent[None, :]
    }

    with pytest.raises(ValueError, match='near-singular tetrahedral child'):
        reader.split_pyramids()


def test_pyramid_pair_uses_one_conforming_diagonal():
    points = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.8, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.9, 0.5, 1.0],
        [0.9, 0.5, -1.0],
    ])
    pyramids = np.array([
        [0, 1, 2, 3, 4],
        [0, 3, 2, 1, 5],
    ])

    reader = GmshReader.__new__(GmshReader)
    reader._nodepts = points
    reader._elenodes = {(7, 7): pyramids}

    reader.split_pyramids()

    _, children = _tet_rows(reader)
    assert children.shape == (4, 10)
    assert reader._pyramid_mortars == []
    assert len(reader._pyramid_conforming_pairs) == 1
    pair = reader._pyramid_conforming_pairs[0]
    assert pair['geometry_error'] == 0
    assert pair['quality'] > 0

    base_faces = [
        frozenset(corners[:3])
        for corners in children[:, :4]
        if set(corners[:3]) <= {0, 1, 2, 3}
    ]
    counts = Counter(base_faces)
    assert len(counts) == 2
    assert sorted(counts.values()) == [2, 2]


def test_periodic_pyramid_bases_use_matching_diagonals():
    order = 2
    lpoints, lparent = _pyramid(order)
    rpoints, rparent = _pyramid(order, offset=len(lpoints))
    rpoints += [3.0, 0.0, 0.0]

    qetype = _QUAD_ETYPES[order]
    qnnodes = GmshReader._etype_map[qetype][1]
    lquad = np.resize(lparent[:4], qnnodes)
    rquad = np.resize(rparent[:4], qnnodes)

    reader = GmshReader.__new__(GmshReader)
    reader._nodepts = np.vstack((lpoints, rpoints))
    reader._bfacespents = {}
    reader._pfacespents = {'x': [1, 2]}
    reader._elenodes = {
        (_PYR_ETYPES[order], (3,)): np.vstack((lparent, rparent)),
        (qetype, (1,)): lquad[None, :],
        (qetype, (2,)): rquad[None, :],
    }

    reader.split_pyramids()

    trietype = _TRI_ETYPES[order]
    assert reader._elenodes[trietype, (1,)].shape[0] == 2
    assert reader._elenodes[trietype, (2,)].shape[0] == 2
    assert reader._pyramid_mortars == []
    assert len(reader._pyramid_periodic_pairs) == 1
    pair = reader._pyramid_periodic_pairs[0]
    assert pair['name'] == 'x'
    assert pair['geometry_error'] < 1e-12
    assert np.allclose(pair['translation'], [3.0, 0.0, 0.0])


def _affine_pyramid(order, offset=0, shift=(0.0, 0.0, 0.0)):
    points = PyrShape.std_ele(order)
    coords = points @ np.array([
        [1.2, 0.1, 0.0],
        [0.0, 0.9, 0.1],
        [0.0, 0.0, 1.1],
    ])
    coords += np.asarray(shift)

    ids = np.arange(offset, offset + len(points))
    return coords, _gmsh_row('pyr', ids)


def test_selective_split_retains_compatible_pyramid():
    points, parent = _affine_pyramid(2)
    reader = GmshReader.__new__(GmshReader)
    reader._nodepts = points
    reader._elenodes = {(14, 7): parent[None, :]}

    summary = reader.split_pyramids('incompatible')

    assert np.array_equal(reader._elenodes[14, 7], parent[None, :])
    assert not any(
        reader._etype_map[key[0]][0] == 'tet'
        for key in reader._elenodes
    )
    assert summary == {
        'policy': 'incompatible', 'input': 1,
        'split': 0, 'retained': 1, 'generated-tets': 0,
        'mortars': 0, 'boundary-splits': 0,
        'conforming-pairs': 0, 'periodic-pairs': 0,
    }


def test_selective_split_converts_only_incompatible_pyramid():
    apoints, aparent = _affine_pyramid(2)
    wpoints, wparent = _pyramid(2, offset=len(apoints))
    wpoints += [4.0, 0.0, 0.0]

    reader = GmshReader.__new__(GmshReader)
    reader._nodepts = np.vstack((apoints, wpoints))
    reader._elenodes = {
        (14, 7): np.vstack((aparent, wparent)),
    }

    summary = reader.split_pyramids('incompatible')

    assert np.array_equal(reader._elenodes[14, 7], aparent[None, :])
    _, children = _tet_rows(reader)
    assert children.shape == (2, TetShape.npts_from_order(4))
    assert len(reader._pyramid_mortars) == 1
    assert summary['input'] == 2
    assert summary['split'] == 1
    assert summary['retained'] == 1
    assert summary['generated-tets'] == 2


def test_split_all_converts_compatible_pyramid():
    points, parent = _affine_pyramid(2)
    reader = GmshReader.__new__(GmshReader)
    reader._nodepts = points
    reader._elenodes = {(14, 7): parent[None, :]}

    summary = reader.split_pyramids('all')

    assert (14, 7) not in reader._elenodes
    _, children = _tet_rows(reader)
    assert children.shape == (2, TetShape.npts_from_order(4))
    assert summary['split'] == 1
    assert summary['retained'] == 0


def test_invalid_pyramid_split_policy():
    points, parent = _affine_pyramid(1)
    reader = GmshReader.__new__(GmshReader)
    reader._nodepts = points
    reader._elenodes = {(7, 7): parent[None, :]}

    with pytest.raises(ValueError, match='Invalid pyramid splitting policy'):
        reader.split_pyramids('sometimes')
