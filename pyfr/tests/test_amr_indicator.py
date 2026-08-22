import numpy as np
import pytest

from pyfr.amr import encode_hex_leaf_tree
from pyfr.amrindicator import (
    AMRIndicatorError, density_variation_scores,
    density_velocity_variation_scores, indicator_settings,
    select_hex_indicator_decision,
)
from pyfr.inifile import Inifile


def _tree(leaves):
    return encode_hex_leaf_tree('root', leaves)


def test_density_variation_scores_constant_and_known_spread():
    state = np.zeros((4, 3, 2))
    state[:, 0, 0] = 2
    state[:, 0, 1] = [1, 2, 3, 2]

    scores = density_variation_scores(state)
    assert scores[0] == 0
    assert scores[1] == pytest.approx(1.0)


def test_density_variation_scores_fail_closed():
    with pytest.raises(AMRIndicatorError, match='shape'):
        density_variation_scores(np.zeros((4, 3)))
    with pytest.raises(AMRIndicatorError, match='finite'):
        state = np.zeros((4, 3, 1)); state[0, 0, 0] = np.nan
        density_variation_scores(state)
    with pytest.raises(AMRIndicatorError, match='floor'):
        density_variation_scores(np.zeros((4, 3, 1)), density_floor=0)


def _euler_state(rho, velocity, p=1.0, gamma=1.4):
    rho = np.asarray(rho, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if velocity.ndim == 1:
        velocity = velocity[:, None]
    momentum = rho[:, None]*velocity
    energy = p/(gamma - 1) + 0.5*rho*np.sum(velocity*velocity, axis=1)
    return np.column_stack((rho, momentum, energy))[:, :, None]


def test_density_velocity_variation_detects_constant_density_shear():
    state = _euler_state(np.ones(4), [0.0, 0.1, 0.05, 0.0])

    density = density_variation_scores(state, 0)
    composite = density_velocity_variation_scores(
        state, density_index=0, momentum_indices=(1,), energy_index=2,
        gamma=1.4,
    )

    assert density[0] == 0
    assert composite[0] == pytest.approx(0.1/np.sqrt(1.4))
    assert composite[0] > 0.05


def test_density_velocity_variation_keeps_density_signal():
    rho = np.array([1.0, 1.2, 1.1, 1.0])
    state = _euler_state(rho, np.zeros(4))

    density = density_variation_scores(state, 0)
    composite = density_velocity_variation_scores(
        state, density_index=0, momentum_indices=(1,), energy_index=2,
        gamma=1.4,
    )

    np.testing.assert_array_equal(composite, density)


def test_density_velocity_variation_fails_closed_on_nonphysical_state():
    state = _euler_state(np.ones(4), np.zeros(4))
    state[0, 0, 0] = 0
    with pytest.raises(AMRIndicatorError, match='density'):
        density_velocity_variation_scores(
            state, density_index=0, momentum_indices=(1,), energy_index=2,
            gamma=1.4,
        )

    state = _euler_state(np.ones(4), np.zeros(4))
    state[:, 2, 0] = -1
    with pytest.raises(AMRIndicatorError, match='pressure'):
        density_velocity_variation_scores(
            state, density_index=0, momentum_indices=(1,), energy_index=2,
            gamma=1.4,
        )

    state = _euler_state(np.ones(4), np.zeros(4))
    with pytest.raises(AMRIndicatorError, match='acoustic floor'):
        density_velocity_variation_scores(
            state, density_index=0, momentum_indices=(1,), energy_index=2,
            gamma=1.4, acoustic_floor=0,
        )


def test_indicator_refine_priority_score_and_tie_break():
    tree = _tree([(0, ()), (1, ()), (2, ())])
    scores = {(0, ()): 0.2, (1, ()): 0.4, (2, ()): 0.4}
    decision = select_hex_indicator_decision(
        tree, scores, refine_threshold=0.3, coarsen_threshold=0.1,
        max_level=2,
    )
    assert decision.action == 'refine'
    assert decision.marks == ((1, ()),)
    assert decision.trigger_score == 0.4


def test_indicator_refine_batch_orders_score_then_leaf():
    tree = _tree([(0, ()), (1, ()), (2, ()), (3, ())])
    scores = {
        (0, ()): 0.4,
        (1, ()): 0.6,
        (2, ()): 0.5,
        (3, ()): 0.6 + 2e-16,
    }
    decision = select_hex_indicator_decision(
        tree, scores, refine_threshold=0.3, coarsen_threshold=0.05,
        tie_tolerance=1e-12, max_level=2, max_refine_marks=3,
    )
    assert decision.action == 'refine'
    assert decision.marks == ((1, ()), (3, ()), (2, ()))
    assert decision.trigger_score == 0.5


def test_indicator_near_tie_uses_canonical_leaf():
    tree = _tree([(0, ()), (1, ()), (2, ())])
    scores = {(0, ()): 0.4, (1, ()): 0.4 + 2e-16, (2, ()): 0.1}
    decision = select_hex_indicator_decision(
        tree, scores, refine_threshold=0.3, coarsen_threshold=0.05,
        tie_tolerance=1e-12, max_level=2,
    )
    assert decision.marks == ((0, ()),)


def test_indicator_refine_respects_max_level():
    leaves = ([(0, (0, o)) for o in range(8)] +
              [(0, (o,)) for o in range(1, 8)] + [(1, ())])
    tree = _tree(leaves)
    scores = {leaf: 0.0 for leaf in leaves}
    scores[(0, (0, 0))] = 1.0
    scores[(1, ())] = 0.5
    decision = select_hex_indicator_decision(
        tree, scores, refine_threshold=0.3, coarsen_threshold=0.1,
        max_level=2,
    )
    assert decision.marks == ((1, ()),)


def test_indicator_coarsens_complete_low_score_family():
    children = [(0, (o,)) for o in range(8)]
    tree = _tree(children + [(1, ())])
    scores = {leaf: 0.01 for leaf in children}
    scores[(1, ())] = 0.15
    decision = select_hex_indicator_decision(
        tree, scores, refine_threshold=0.3, coarsen_threshold=0.05,
        max_level=2,
    )
    assert decision.action == 'coarsen'
    assert set(decision.marks) == set(children)
    assert decision.trigger_score == 0.01


def test_indicator_coarsen_batch_orders_score_then_parent():
    leaves = (
        [(0, (o,)) for o in range(8)] +
        [(1, (o,)) for o in range(8)] +
        [(2, (o,)) for o in range(8)]
    )
    tree = _tree(leaves)
    scores = {}
    for leaf in leaves:
        root = leaf[0]
        scores[leaf] = {0: 0.02, 1: 0.01, 2: 0.01 + 2e-16}[root]

    default = select_hex_indicator_decision(
        tree, scores, refine_threshold=0.3, coarsen_threshold=0.05,
        tie_tolerance=1e-12, max_level=2,
    )
    assert default.action == 'coarsen'
    assert default.marks == tuple((1, (o,)) for o in range(8))
    assert default.trigger_score == pytest.approx(0.01)

    batch = select_hex_indicator_decision(
        tree, scores, refine_threshold=0.3, coarsen_threshold=0.05,
        tie_tolerance=1e-12, max_level=2, max_coarsen_families=2,
    )
    assert batch.action == 'coarsen'
    assert batch.marks == tuple(
        [(1, (o,)) for o in range(8)] +
        [(2, (o,)) for o in range(8)]
    )
    assert batch.trigger_score == pytest.approx(0.01 + 2e-16)


def test_indicator_hysteresis_returns_none_inside_band():
    tree = _tree([(0, ()), (1, ())])
    scores = {(0, ()): 0.15, (1, ()): 0.2}
    decision = select_hex_indicator_decision(
        tree, scores, refine_threshold=0.3, coarsen_threshold=0.1,
        max_level=2,
    )
    assert decision.action == 'none'
    assert decision.marks == ()
    assert decision.trigger_score is None


def test_indicator_requires_exact_score_coverage():
    tree = _tree([(0, ()), (1, ())])
    with pytest.raises(AMRIndicatorError, match='cover'):
        select_hex_indicator_decision(
            tree, {(0, ()): 1.0}, refine_threshold=0.3,
            coarsen_threshold=0.1,
        )


def test_indicator_settings_are_explicit_and_hysteretic():
    cfg = Inifile('''
[solver-amr]
indicator = density-variation
refine-threshold = 0.3
coarsen-threshold = 0.1
max-level = 3
max-refine-marks = 4
max-coarsen-families = 3
''')
    settings = indicator_settings(cfg)
    assert settings.refine_threshold == 0.3
    assert settings.coarsen_threshold == 0.1
    assert settings.acoustic_floor == 1e-14
    assert settings.tie_tolerance == 1e-12
    assert settings.min_level == 0
    assert settings.max_level == 3
    assert settings.max_refine_marks == 4
    assert settings.max_coarsen_families == 3

    bad = Inifile('''
[solver-amr]
refine-threshold = 0.1
coarsen-threshold = 0.1
''')
    with pytest.raises(AMRIndicatorError, match='hysteresis'):
        indicator_settings(bad)


def test_indicator_settings_reject_nonpositive_batch_limit():
    cfg = Inifile('''
[solver-amr]
refine-threshold = 0.3
coarsen-threshold = 0.1
max-refine-marks = 0
''')
    with pytest.raises(AMRIndicatorError, match='positive integer'):
        indicator_settings(cfg)


def test_indicator_settings_reject_nonpositive_coarsen_family_limit():
    cfg = Inifile("""
[solver-amr]
refine-threshold = 0.3
coarsen-threshold = 0.1
max-coarsen-families = 0
""")
    with pytest.raises(AMRIndicatorError, match='positive integer'):
        indicator_settings(cfg)


def test_indicator_settings_accept_composite_and_validate_acoustic_floor():
    cfg = Inifile("""
[solver-amr]
indicator = density-velocity-variation
refine-threshold = 0.05
coarsen-threshold = 0.01
acoustic-floor = 1e-12
""")
    settings = indicator_settings(cfg)
    assert settings.indicator == 'density-velocity-variation'
    assert settings.acoustic_floor == 1e-12

    bad = Inifile("""
[solver-amr]
indicator = density-velocity-variation
refine-threshold = 0.05
coarsen-threshold = 0.01
acoustic-floor = 0
""")
    with pytest.raises(AMRIndicatorError, match='acoustic floor'):
        indicator_settings(bad)


def test_indicator_settings_reject_unknown_indicator():
    cfg = Inifile("""
[solver-amr]
indicator = velocity-gradient
refine-threshold = 0.05
coarsen-threshold = 0.01
""")
    with pytest.raises(AMRIndicatorError, match='unsupported'):
        indicator_settings(cfg)


def test_density_settings_ignore_unused_acoustic_floor():
    cfg = Inifile("""
[solver-amr]
indicator = density-variation
refine-threshold = 0.3
coarsen-threshold = 0.1
acoustic-floor = deliberately-not-a-number
""")
    settings = indicator_settings(cfg)
    assert settings.indicator == 'density-variation'
    assert settings.acoustic_floor == 1e-14
