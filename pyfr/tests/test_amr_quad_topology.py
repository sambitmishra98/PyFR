from collections import defaultdict
from fractions import Fraction

import numpy as np
import pytest

from pyfr.amr import (
    QuadNodeStore, quad_affine_map, quad_c2_transform,
    quad_child_face_nodes, quad_face_axis_side,
    quad_face_corner_indices, quad_interval_overlap,
    quad_order_line1x2_faces, quad_refined_children,
    quad_root_face_u, quad_transform_interval,
    quad_tree_cell_index, quad_tree_face_groups,
    quad_tree_face_pairs, quad_tree_leaves,
)


class _Store:
    def __init__(self, coords):
        self.coords_by_id = {
            int(i): np.asarray(p, dtype=float)
            for i, p in enumerate(coords)
        }
        self.next_id = len(coords)
        self.store = QuadNodeStore(self.coords, self.allocate)

    def coords(self, ids):
        return np.asarray([
            self.coords_by_id[int(i)] for i in np.asarray(ids)
        ])

    def allocate(self, points):
        out = []
        for point in np.asarray(points):
            idx = self.next_id
            self.next_id += 1
            self.coords_by_id[idx] = np.asarray(point, dtype=float)
            out.append(idx)
        return np.asarray(out, dtype=np.int64)


def _unit_store():
    return _Store([
        [-1, -1], [1, -1], [-1, 1], [1, 1],
    ])


def _rootfaces_one_root():
    return {
        (0, fidx): {
            'qkey': ('boundary', fidx),
            'transform': (1, 0),
        }
        for fidx in range(4)
    }


def test_quad_face_corner_indices_match_native_faces():
    assert tuple(quad_face_corner_indices(i) for i in range(4)) == (
        (0, 1), (1, 3), (2, 3), (0, 2),
    )


def test_quad_tree_cell_index_uses_native_quadrant_bits():
    assert quad_tree_cell_index(()) == (0, (0, 0))
    assert quad_tree_cell_index((0,)) == (1, (0, 0))
    assert quad_tree_cell_index((1,)) == (1, (1, 0))
    assert quad_tree_cell_index((2,)) == (1, (0, 1))
    assert quad_tree_cell_index((3,)) == (1, (1, 1))
    assert quad_tree_cell_index((3, 0, 2)) == (3, (4, 5))


def test_quad_face_axis_side_matches_native_face_order():
    assert tuple(quad_face_axis_side(i) for i in range(4)) == (
        (1, 0), (0, 1), (1, 1), (0, 0),
    )


def test_quad_root_face_u_identity_and_reversal():
    pids = np.array([10, 11, 12, 13])
    low = quad_root_face_u(pids, 0)
    high = quad_root_face_u(pids, 2)

    assert low == {10: 0, 11: 1}
    assert high == {12: 0, 13: 1}
    assert quad_c2_transform(low, low) == (1, 0)
    assert quad_c2_transform(low, {10: 1, 11: 0}) == (-1, 1)


def test_quad_c2_transform_rejects_invalid_orientation():
    with pytest.raises(ValueError, match='orientation'):
        quad_c2_transform({1: 0, 2: 1}, {1: 0})
    with pytest.raises(ValueError, match='orientation'):
        quad_c2_transform({1: 0, 2: 1}, {1: 0, 2: 2})


def test_quad_transform_interval_identity_and_reversal():
    interval = (Fraction(1, 4), Fraction(1, 2))
    assert quad_transform_interval(interval, (1, 0)) == interval
    assert quad_transform_interval(interval, (-1, 1)) == (
        Fraction(1, 2), Fraction(3, 4)
    )


def test_quad_interval_overlap_is_open():
    assert quad_interval_overlap((0, Fraction(1, 2)),
                                 (Fraction(1, 4), 1))
    assert not quad_interval_overlap((0, Fraction(1, 2)),
                                     (Fraction(1, 2), 1))


def test_quad_tree_leaves_canonical_recursive_order():
    split = {
        (0, ()),
        (0, (2,)),
    }
    leaves = quad_tree_leaves([1, 0], split)

    assert leaves[:2] == [(0, (0,)), (0, (1,))]
    assert leaves[2:6] == [(0, (2, q)) for q in range(4)]
    assert leaves[6:] == [(0, (3,)), (1, ())]


def test_quad_tree_face_groups_pair_one_level_split():
    leaves = [(0, (q,)) for q in range(4)]
    groups = quad_tree_face_groups(leaves, _rootfaces_one_root())
    pairs = list(quad_tree_face_pairs(groups, {}))

    assert len(pairs) == 4
    assert all(a['level'] == b['level'] == 1 for a, b in pairs)


def test_quad_tree_face_groups_reversed_shared_root_pairing():
    rootfaces = _rootfaces_one_root()
    rootfaces.update({
        (1, 0): {'qkey': ('boundary', 10), 'transform': (1, 0)},
        (1, 1): {'qkey': ('boundary', 11), 'transform': (1, 0)},
        (1, 2): {'qkey': ('boundary', 12), 'transform': (1, 0)},
        (1, 3): {'qkey': ('interior', 0), 'transform': (-1, 1)},
    })
    rootfaces[0, 1] = {
        'qkey': ('interior', 0), 'transform': (1, 0)
    }
    rootkinds = {('root', ('interior', 0)): 'interior'}

    leaves = [(root, (q,)) for root in (0, 1) for q in range(4)]
    groups = quad_tree_face_groups(leaves, rootfaces)
    pairs = list(quad_tree_face_pairs(groups, rootkinds))
    shared = [
        (a, b) for a, b in pairs
        if a['surface'] == ('root', ('interior', 0))
    ]

    assert len(shared) == 2
    assert {
        (a['leaf'], b['leaf']) for a, b in shared
    } == {
        ((0, (1,)), (1, (2,))),
        ((0, (3,)), (1, (0,))),
    }


def test_quad_tree_face_pairs_rejects_incomplete_internal():
    groups = defaultdict(lambda: defaultdict(list))
    surface = ('internal', 0, 0, Fraction(1, 2))
    groups[surface][0].append({
        'leaf': (0, (0,)), 'fidx': 1, 'level': 1,
        'interval': (0, 1), 'surface': surface,
    })
    with pytest.raises(ValueError, match='Incomplete internal'):
        list(quad_tree_face_pairs(groups, {}))


def test_quad_tree_face_pairs_rejects_incomplete_shared_root():
    groups = defaultdict(lambda: defaultdict(list))
    surface = ('root', ('interior', 0))
    groups[surface][0].append({
        'leaf': (0, ()), 'fidx': 1, 'level': 0,
        'interval': (0, 1), 'surface': surface,
    })
    with pytest.raises(ValueError, match='Incomplete shared'):
        list(quad_tree_face_pairs(groups, {surface: 'interior'}))


def test_quad_tree_face_pairs_rejects_nonmanifold_surface():
    groups = defaultdict(lambda: defaultdict(list))
    surface = ('root', ('interior', 0))
    for side in range(3):
        groups[surface][side].append({
            'leaf': (side, ()), 'fidx': 1, 'level': 0,
            'interval': (0, 1), 'surface': surface,
        })
    with pytest.raises(ValueError, match='Non-manifold'):
        list(quad_tree_face_pairs(groups, {surface: 'interior'}))


def test_quad_affine_map_reproduces_affine_geometry():
    st = _Store([
        [2, 3], [5, 4], [1, 7], [4, 8],
    ])
    amap = quad_affine_map(np.arange(4), st.store)
    got = amap([[0, 0], [0.5, -0.5]])

    assert np.allclose(got[0], [3, 5.5], rtol=0, atol=2e-15)
    assert np.all(np.isfinite(got))


def test_quad_affine_map_rejects_nonaffine_geometry():
    st = _Store([
        [-1, -1], [1, -1], [-1, 1], [1.1, 1],
    ])
    with pytest.raises(ValueError, match='affine'):
        quad_affine_map(np.arange(4), st.store)


def test_quad_affine_map_rejects_nonpositive_geometry():
    st = _Store([
        [1, -1], [-1, -1], [1, 1], [-1, 1],
    ])
    with pytest.raises(ValueError, match='positive'):
        quad_affine_map(np.arange(4), st.store)


def test_quad_refined_children_geometry_and_quadrant_order():
    st = _unit_store()
    children = quad_refined_children(
        np.arange(4), st.store, {}, {}
    )

    assert [(ix, iy) for ix, iy, _ in children] == [
        (0, 0), (1, 0), (0, 1), (1, 1)
    ]
    centres = []
    for _, _, row in children:
        centres.append(st.store.coords(row).mean(axis=0))
    assert np.allclose(centres, [
        [-0.5, -0.5], [0.5, -0.5],
        [-0.5, 0.5], [0.5, 0.5],
    ])
    assert len(st.coords_by_id) == 9


def test_quad_refined_children_reuses_shared_edge_nodes():
    st = _Store([
        [0, 0], [1, 0], [0, 1], [1, 1],
        [2, 0], [2, 1],
    ])
    cache = {}
    ccache = {}
    left = quad_refined_children(
        [0, 1, 2, 3], st.store, cache, ccache
    )
    right = quad_refined_children(
        [1, 4, 3, 5], st.store, cache, ccache
    )

    lnodes = {
        n for *_, row in left
        for n in row if np.isclose(st.coords_by_id[int(n)][0], 1)
    }
    rnodes = {
        n for *_, row in right
        for n in row if np.isclose(st.coords_by_id[int(n)][0], 1)
    }
    assert lnodes == rnodes
    assert len(lnodes) == 3


def test_quad_refined_children_rejects_shared_geometry_disagreement():
    st = _unit_store()
    cache = {(0, 1): 99}
    ccache = {(0, 1): np.array([0.25, -1.0])}
    st.coords_by_id[99] = np.array([0.0, -1.0])

    with pytest.raises(ValueError, match='disagree'):
        quad_refined_children(np.arange(4), st.store, cache, ccache)


def test_quad_child_face_nodes_returns_two_halves():
    st = _unit_store()
    children = quad_refined_children(
        np.arange(4), st.store, {}, {}
    )

    for fidx in range(4):
        faces = quad_child_face_nodes(
            children, fidx, quad_face_corner_indices(fidx)
        )
        assert len(faces) == 2
        assert all(len(face) == 2 for face in faces)


def test_quad_order_line1x2_faces_is_coarse_reference_low_high():
    st = _unit_store()
    children = quad_refined_children(
        np.arange(4), st.store, {}, {}
    )
    fidx = 0
    fine = quad_child_face_nodes(
        children, fidx, quad_face_corner_indices(fidx)
    )
    ordered = quad_order_line1x2_faces(
        np.arange(4), fidx, list(reversed(fine)), st.store
    )

    midx = [
        st.store.coords(face).mean(axis=0)[0] for face in ordered
    ]
    assert midx[0] < midx[1]


def test_quad_order_line1x2_faces_rejects_incomplete_coverage():
    st = _unit_store()
    children = quad_refined_children(
        np.arange(4), st.store, {}, {}
    )
    fine = quad_child_face_nodes(
        children, 0, quad_face_corner_indices(0)
    )
    with pytest.raises(ValueError, match='Incomplete'):
        quad_order_line1x2_faces(np.arange(4), 0, fine[:1], st.store)


def test_quad_order_line1x2_faces_rejects_bad_half():
    st = _unit_store()
    bad0 = st.store.allocate([[0.25, -1]])[0]
    bad1 = st.store.allocate([[0.75, -1]])[0]
    with pytest.raises(ValueError, match='does not match'):
        quad_order_line1x2_faces(
            np.arange(4), 0, [(int(bad0), int(bad1))], st.store
        )
