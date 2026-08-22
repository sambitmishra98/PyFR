import numpy as np
import pytest

from pyfr.amr import (
    HexLeafTree, QuadLeafTree, _hex_transform_rect_cached,
    encode_hex_leaf_tree, encode_quad_leaf_tree,
    hex_leaf_tree_from_arrays, hex_transform_rect,
    quad_leaf_tree_from_arrays,
)


ROOT_UUID = 'e3b0c442-98fc-1c14-9afb-f4c8996fb924'


def _children(path, count=8):
    return [(0, path + (o,)) for o in range(count)]


def test_encode_single_root_single_leaf():
    tree = encode_hex_leaf_tree(ROOT_UUID, [(0, ())])
    assert tree.nleaves == 1
    assert list(tree.leaves()) == [(0, ())]
    assert tree.version == 1
    assert tree.root_mesh_uuid == ROOT_UUID


def test_encode_one_level_split_reconstructs_exactly():
    leaves = _children(())
    tree = encode_hex_leaf_tree(ROOT_UUID, leaves)
    assert tree.nleaves == 8
    assert sorted(tree.leaves()) == sorted(leaves)


def test_encode_multi_level_split_reconstructs_exactly():
    # Root splits into 8; octant 3 splits again into 8 grandchildren.
    leaves = [(0, p) for p in [(0,), (1,), (2,), (4,), (5,), (6,), (7,)]]
    leaves += [(0, (3, o)) for o in range(8)]
    tree = encode_hex_leaf_tree(ROOT_UUID, leaves)
    assert tree.nleaves == 15
    assert sorted(tree.leaves()) == sorted(leaves)


def test_encode_multiple_roots():
    leaves = [(0, ())] + [(1, (o,)) for o in range(8)]
    tree = encode_hex_leaf_tree(ROOT_UUID, leaves)
    assert tree.nleaves == 9
    assert sorted(tree.leaves()) == sorted(leaves)


def test_encode_bytes_independent_of_input_order():
    leaves = _children(())
    a = encode_hex_leaf_tree(ROOT_UUID, leaves)
    b = encode_hex_leaf_tree(ROOT_UUID, list(reversed(leaves)))
    c = encode_hex_leaf_tree(ROOT_UUID, sorted(leaves, key=repr))
    assert np.array_equal(a.root_eidx, b.root_eidx)
    assert np.array_equal(a.path_offsets, b.path_offsets)
    assert np.array_equal(a.path_data, b.path_data)
    assert np.array_equal(a.root_eidx, c.root_eidx)
    assert np.array_equal(a.path_offsets, c.path_offsets)
    assert np.array_equal(a.path_data, c.path_data)


def test_encode_canonical_lexicographic_ordering():
    # Legal trees only: unsplit roots 1 and 2, plus a complete one-level
    # split under root 0 (so no isolated/incomplete-split leaf).
    leaves = _children(()) + [(2, ()), (1, ())]
    tree = encode_hex_leaf_tree(ROOT_UUID, leaves)
    ordered = list(tree.leaves())
    assert ordered == sorted(leaves)


def test_encoded_arrays_read_only():
    tree = encode_hex_leaf_tree(ROOT_UUID, [(0, ())])
    assert not tree.root_eidx.flags.writeable
    assert not tree.path_offsets.flags.writeable
    assert not tree.path_data.flags.writeable


def test_from_arrays_round_trips_exactly():
    leaves = [(0, p) for p in [(0,), (1,), (2,), (4,), (5,), (6,), (7,)]]
    leaves += [(0, (3, o)) for o in range(8)]
    original = encode_hex_leaf_tree(ROOT_UUID, leaves)
    rebuilt = hex_leaf_tree_from_arrays(
        original.version, original.root_mesh_uuid, original.root_eidx,
        original.path_offsets, original.path_data,
    )
    assert np.array_equal(original.root_eidx, rebuilt.root_eidx)
    assert np.array_equal(original.path_offsets, rebuilt.path_offsets)
    assert np.array_equal(original.path_data, rebuilt.path_data)


def test_reject_empty_root_mesh_uuid():
    with pytest.raises(ValueError, match='non-empty'):
        encode_hex_leaf_tree('', [(0, ())])


def test_reject_unsupported_version():
    with pytest.raises(ValueError, match='version'):
        encode_hex_leaf_tree(ROOT_UUID, [(0, ())], version=2)
    tree = encode_hex_leaf_tree(ROOT_UUID, [(0, ())])
    with pytest.raises(ValueError, match='version'):
        hex_leaf_tree_from_arrays(
            2, ROOT_UUID, tree.root_eidx, tree.path_offsets, tree.path_data
        )


def test_reject_negative_root_index():
    with pytest.raises(ValueError, match='root'):
        encode_hex_leaf_tree(ROOT_UUID, [(-1, ())])


def test_reject_octant_outside_range():
    with pytest.raises(ValueError, match='octant'):
        encode_hex_leaf_tree(ROOT_UUID, [(0, (8,))])
    with pytest.raises(ValueError, match='octant'):
        encode_hex_leaf_tree(ROOT_UUID, [(0, (-1,))])


def test_reject_duplicate_active_leaf():
    with pytest.raises(ValueError, match='duplicate'):
        encode_hex_leaf_tree(ROOT_UUID, [(0, (1,)), (0, (1,))])


def test_reject_active_ancestor_plus_active_descendant():
    with pytest.raises(ValueError, match='ancestor'):
        encode_hex_leaf_tree(ROOT_UUID, [(0, ()), (0, (0,))])
    with pytest.raises(ValueError, match='ancestor'):
        encode_hex_leaf_tree(ROOT_UUID, [(0, (1,)), (0, (1, 2))])


def test_reject_incomplete_eight_way_split():
    with pytest.raises(ValueError, match='incomplete'):
        encode_hex_leaf_tree(ROOT_UUID, _children(())[:7])


def test_reject_malformed_path_offsets_length():
    tree = encode_hex_leaf_tree(ROOT_UUID, [(0, ())])
    with pytest.raises(ValueError, match='length'):
        hex_leaf_tree_from_arrays(
            1, ROOT_UUID, tree.root_eidx,
            np.array([0, 0, 0], dtype=np.int64), tree.path_data,
        )


def test_reject_malformed_path_offsets_not_starting_at_zero():
    # Legal complete one-level split so path-data is non-empty.
    tree = encode_hex_leaf_tree(ROOT_UUID, _children(()))
    with pytest.raises(ValueError, match='start at 0'):
        hex_leaf_tree_from_arrays(
            1, ROOT_UUID, tree.root_eidx,
            tree.path_offsets + 1, tree.path_data,
        )


def test_reject_malformed_path_offsets_decreasing():
    root_eidx = np.array([0, 0], dtype=np.int64)
    path_data = np.array([1, 2], dtype=np.uint8)
    bad_offsets = np.array([0, 2, 1], dtype=np.int64)
    with pytest.raises(ValueError, match='non-decreasing'):
        hex_leaf_tree_from_arrays(
            1, ROOT_UUID, root_eidx, bad_offsets, path_data
        )


def test_reject_malformed_path_offsets_final_mismatch():
    # Legal complete one-level split so path-data is non-empty.
    tree = encode_hex_leaf_tree(ROOT_UUID, _children(()))
    bad_offsets = tree.path_offsets.copy()
    bad_offsets[-1] += 5  # starts at 0, non-decreasing, but wrong total
    with pytest.raises(ValueError, match='final entry'):
        hex_leaf_tree_from_arrays(
            1, ROOT_UUID, tree.root_eidx, bad_offsets, tree.path_data
        )


def test_reject_malformed_path_data_octant_range():
    root_eidx = np.array([0], dtype=np.int64)
    path_offsets = np.array([0, 1], dtype=np.int64)
    bad_path_data = np.array([9], dtype=np.uint8)
    with pytest.raises(ValueError, match='octant'):
        hex_leaf_tree_from_arrays(
            1, ROOT_UUID, root_eidx, path_offsets, bad_path_data
        )


def test_reject_malformed_shapes_not_1d():
    root_eidx = np.zeros((1, 1), dtype=np.int64)
    path_offsets = np.array([0, 0], dtype=np.int64)
    path_data = np.zeros((0,), dtype=np.uint8)
    with pytest.raises(ValueError, match='shapes'):
        hex_leaf_tree_from_arrays(
            1, ROOT_UUID, root_eidx, path_offsets, path_data
        )


def test_hex_transform_rect_reuses_normalised_sequences():
    _hex_transform_rect_cached.cache_clear()
    rect = [0, 1, 0, 1]
    transform = np.array([0, 1, 0, -1, 0, 1])

    expected = hex_transform_rect(rect, transform)
    actual = hex_transform_rect(tuple(rect), tuple(transform))

    assert actual == expected == (0, 1, 0, 1)
    info = _hex_transform_rect_cached.cache_info()
    assert info.misses == 1
    assert info.hits == 1
    assert info.currsize == 1
    assert info.maxsize == 4096


def test_hex_transform_rect_cache_distinguishes_orientation():
    _hex_transform_rect_cached.cache_clear()
    rect = (0, 0.5, 0, 0.5)
    identity = (1, 0, 0, 0, 1, 0)
    rotate = (0, 1, 0, -1, 0, 1)

    assert hex_transform_rect(rect, identity) == (0, 0.5, 0, 0.5)
    assert hex_transform_rect(rect, rotate) == (0, 0.5, 0.5, 1)
    assert _hex_transform_rect_cached.cache_info().misses == 2


def _quad_children(path, count=4):
    return [(0, path + (q,)) for q in range(count)]


def test_quad_encode_single_root_single_leaf():
    tree = encode_quad_leaf_tree(ROOT_UUID, [(0, ())])
    assert isinstance(tree, QuadLeafTree)
    assert tree.nleaves == 1
    assert list(tree.leaves()) == [(0, ())]
    assert tree.version == 1
    assert tree.root_mesh_uuid == ROOT_UUID


def test_quad_encode_one_level_split_reconstructs_exactly():
    leaves = _quad_children(())
    tree = encode_quad_leaf_tree(ROOT_UUID, leaves)
    assert tree.nleaves == 4
    assert list(tree.leaves()) == sorted(leaves)


def test_quad_encode_multi_level_split_reconstructs_exactly():
    leaves = [(0, p) for p in [(0,), (1,), (3,)]]
    leaves += [(0, (2, q)) for q in range(4)]
    tree = encode_quad_leaf_tree(ROOT_UUID, leaves)
    assert tree.nleaves == 7
    assert list(tree.leaves()) == sorted(leaves)


def test_quad_encode_multiple_roots_and_canonical_order():
    leaves = _quad_children(()) + [(2, ()), (1, ())]
    tree = encode_quad_leaf_tree(ROOT_UUID, reversed(leaves))
    assert tree.nleaves == 6
    assert list(tree.leaves()) == sorted(leaves)


def test_quad_encode_bytes_independent_of_input_order():
    leaves = _quad_children(())
    a = encode_quad_leaf_tree(ROOT_UUID, leaves)
    b = encode_quad_leaf_tree(ROOT_UUID, list(reversed(leaves)))
    assert np.array_equal(a.root_eidx, b.root_eidx)
    assert np.array_equal(a.path_offsets, b.path_offsets)
    assert np.array_equal(a.path_data, b.path_data)


def test_quad_encoded_arrays_read_only():
    tree = encode_quad_leaf_tree(ROOT_UUID, [(0, ())])
    assert not tree.root_eidx.flags.writeable
    assert not tree.path_offsets.flags.writeable
    assert not tree.path_data.flags.writeable


def test_quad_from_arrays_round_trips_exactly():
    leaves = [(0, p) for p in [(0,), (1,), (3,)]]
    leaves += [(0, (2, q)) for q in range(4)]
    original = encode_quad_leaf_tree(ROOT_UUID, leaves)
    rebuilt = quad_leaf_tree_from_arrays(
        original.version, original.root_mesh_uuid, original.root_eidx,
        original.path_offsets, original.path_data,
    )
    assert np.array_equal(original.root_eidx, rebuilt.root_eidx)
    assert np.array_equal(original.path_offsets, rebuilt.path_offsets)
    assert np.array_equal(original.path_data, rebuilt.path_data)


def test_quad_reject_empty_root_mesh_uuid_and_version():
    with pytest.raises(ValueError, match='non-empty'):
        encode_quad_leaf_tree('', [(0, ())])
    with pytest.raises(ValueError, match='version'):
        encode_quad_leaf_tree(ROOT_UUID, [(0, ())], version=2)


def test_quad_reject_negative_root_index():
    with pytest.raises(ValueError, match='root'):
        encode_quad_leaf_tree(ROOT_UUID, [(-1, ())])


def test_quad_reject_quadrant_outside_range():
    with pytest.raises(ValueError, match='quadrant'):
        encode_quad_leaf_tree(ROOT_UUID, [(0, (4,))])
    with pytest.raises(ValueError, match='quadrant'):
        encode_quad_leaf_tree(ROOT_UUID, [(0, (-1,))])


def test_quad_reject_duplicate_active_leaf():
    with pytest.raises(ValueError, match='duplicate'):
        encode_quad_leaf_tree(ROOT_UUID, [(0, (1,)), (0, (1,))])


def test_quad_reject_active_ancestor_plus_active_descendant():
    with pytest.raises(ValueError, match='ancestor'):
        encode_quad_leaf_tree(ROOT_UUID, [(0, ()), (0, (0,))])
    with pytest.raises(ValueError, match='ancestor'):
        encode_quad_leaf_tree(ROOT_UUID, [(0, (1,)), (0, (1, 2))])


@pytest.mark.parametrize('count', [1, 2, 3])
def test_quad_reject_incomplete_four_way_split(count):
    with pytest.raises(ValueError, match=rf'{count}/4'):
        encode_quad_leaf_tree(ROOT_UUID, _quad_children((), count))


def test_quad_reject_malformed_path_offsets_length():
    tree = encode_quad_leaf_tree(ROOT_UUID, [(0, ())])
    with pytest.raises(ValueError, match='length'):
        quad_leaf_tree_from_arrays(
            1, ROOT_UUID, tree.root_eidx,
            np.array([0, 0, 0], dtype=np.int64), tree.path_data,
        )


def test_quad_reject_malformed_path_offsets_not_starting_at_zero():
    tree = encode_quad_leaf_tree(ROOT_UUID, _quad_children(()))
    with pytest.raises(ValueError, match='start at 0'):
        quad_leaf_tree_from_arrays(
            1, ROOT_UUID, tree.root_eidx,
            tree.path_offsets + 1, tree.path_data,
        )


def test_quad_reject_malformed_path_offsets_decreasing():
    root_eidx = np.array([0, 0], dtype=np.int64)
    path_data = np.array([1, 2], dtype=np.uint8)
    bad_offsets = np.array([0, 2, 1], dtype=np.int64)
    with pytest.raises(ValueError, match='non-decreasing'):
        quad_leaf_tree_from_arrays(
            1, ROOT_UUID, root_eidx, bad_offsets, path_data
        )


def test_quad_reject_malformed_path_offsets_final_mismatch():
    tree = encode_quad_leaf_tree(ROOT_UUID, _quad_children(()))
    bad_offsets = tree.path_offsets.copy()
    bad_offsets[-1] += 5
    with pytest.raises(ValueError, match='final entry'):
        quad_leaf_tree_from_arrays(
            1, ROOT_UUID, tree.root_eidx, bad_offsets, tree.path_data
        )


def test_quad_reject_malformed_path_data_quadrant_range():
    root_eidx = np.array([0], dtype=np.int64)
    path_offsets = np.array([0, 1], dtype=np.int64)
    bad_path_data = np.array([4], dtype=np.uint8)
    with pytest.raises(ValueError, match='quadrant'):
        quad_leaf_tree_from_arrays(
            1, ROOT_UUID, root_eidx, path_offsets, bad_path_data
        )


def test_quad_reject_malformed_shapes_not_1d():
    root_eidx = np.zeros((1, 1), dtype=np.int64)
    path_offsets = np.array([0, 0], dtype=np.int64)
    path_data = np.zeros((0,), dtype=np.uint8)
    with pytest.raises(ValueError, match='shapes'):
        quad_leaf_tree_from_arrays(
            1, ROOT_UUID, root_eidx, path_offsets, path_data
        )
