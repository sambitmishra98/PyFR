import pytest

from pyfr.amr import (
    hex_coarsen_candidates, hex_octree_children, hex_octree_level,
)


ROOT = 10
PPARENT = (ROOT, (0,))                   # level 1
PCHILDREN = hex_octree_children(PPARENT)  # level 2 leaves


def test_hex_coarsen_complete_marked_family_accepted():
    result = hex_coarsen_candidates(PCHILDREN, PCHILDREN, [])
    assert result == (PPARENT,)


def test_hex_coarsen_candidate_ordering_is_deterministic():
    a = hex_coarsen_candidates(PCHILDREN, PCHILDREN, [])
    b = hex_coarsen_candidates(
        PCHILDREN, sorted(PCHILDREN, key=repr, reverse=True), []
    )
    c = hex_coarsen_candidates(PCHILDREN, list(PCHILDREN)[::-1], [])
    assert a == b == c


def test_hex_coarsen_seven_of_eight_marked_rejected():
    result = hex_coarsen_candidates(PCHILDREN, list(PCHILDREN)[:7], [])
    assert result == ()


def test_hex_coarsen_duplicate_marks_identical_result():
    baseline = hex_coarsen_candidates(PCHILDREN, PCHILDREN, [])
    dup = hex_coarsen_candidates(
        PCHILDREN, list(PCHILDREN) + list(PCHILDREN), []
    )
    assert dup == baseline


def test_hex_coarsen_mark_referencing_non_leaf_is_ignored():
    """A mark naming a leaf that is not itself in the active leaf set
    (e.g. a would-be split child) never manufactures a false candidate.
    """
    baseline = hex_coarsen_candidates(PCHILDREN, PCHILDREN, [])
    not_a_leaf = (ROOT, (0, 0, 0))
    result = hex_coarsen_candidates(
        PCHILDREN, list(PCHILDREN) + [not_a_leaf], []
    )
    assert result == baseline


def test_hex_coarsen_level_diff_two_rejected():
    """A level-3 neighbour makes collapsing level-2 siblings to level 1
    illegal."""
    qparent = (20, (3, 2))
    qchildren = hex_octree_children(qparent)
    edge_pair = [(next(iter(PCHILDREN)), next(iter(qchildren)))]

    result = hex_coarsen_candidates(
        PCHILDREN | qchildren, PCHILDREN, edge_pair
    )
    assert result == ()


def test_hex_coarsen_joint_neighbour_collapse_restores_2to1():
    qparent = (20, (3, 2))
    qchildren = hex_octree_children(qparent)
    edge_pair = [(next(iter(PCHILDREN)), next(iter(qchildren)))]

    result = hex_coarsen_candidates(
        PCHILDREN | qchildren, PCHILDREN | qchildren, edge_pair
    )
    assert set(result) == {PPARENT, qparent}


def test_hex_coarsen_same_level_neighbour_accepted():
    neighbour = (30, (5,))  # level 1, same as PPARENT post-collapse
    result = hex_coarsen_candidates(
        PCHILDREN | {neighbour}, PCHILDREN,
        [(next(iter(PCHILDREN)), neighbour)]
    )
    assert result == (PPARENT,)


def test_hex_coarsen_cross_root_face_uses_level_rule():
    cross_root = (99, (7, 6, 5))
    result = hex_coarsen_candidates(
        PCHILDREN | {cross_root}, PCHILDREN,
        [(next(iter(PCHILDREN)), cross_root)]
    )
    assert result == ()


def test_hex_coarsen_internal_sibling_face_pair_is_not_a_violation():
    internal = list(PCHILDREN)[:2]
    result = hex_coarsen_candidates(
        PCHILDREN, PCHILDREN, [(internal[0], internal[1])]
    )
    assert result == (PPARENT,)


def test_hex_coarsen_preexisting_unbalanced_input_fails_closed():
    with pytest.raises(ValueError, match='2:1'):
        hex_coarsen_candidates(
            {(1, (0,)), (2, (0, 0, 0))}, set(),
            [((1, (0,)), (2, (0, 0, 0)))]
        )


def test_hex_coarsen_face_pair_outside_active_set_fails_closed():
    not_a_leaf = (ROOT, (0, 0, 0))
    with pytest.raises(ValueError, match='outside the current active'):
        hex_coarsen_candidates(
            PCHILDREN, PCHILDREN, [(not_a_leaf, PPARENT)]
        )


def test_hex_coarsen_one_level_per_transaction():
    """64 marked grandchildren produce exactly eight direct-parent
    candidates in one transaction; the grandparent never collapses in
    the same call."""
    deep_leaves = {(7, (a, b)) for a in range(8) for b in range(8)}
    result = hex_coarsen_candidates(deep_leaves, deep_leaves, [])
    assert len(result) == 8
    assert all(hex_octree_level(p) == 1 for p in result)


def test_hex_coarsen_incomplete_family_never_bridged_fails_closed():
    incomplete = set(list(PCHILDREN)[:5])
    result = hex_coarsen_candidates(incomplete, incomplete, [])
    assert result == ()
