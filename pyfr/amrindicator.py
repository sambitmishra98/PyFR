"""Deterministic solution indicators for explicit online Hex AMR events.

D9A keeps indicator evaluation outside the normal RHS path.  The indicator
selects marks only; accepted D7 topology, transfer, ownership, migration,
validation, COMMIT, and rollback remain authoritative.
"""
from dataclasses import dataclass

import numpy as np

from pyfr.amr import hex_octree_children
from pyfr.amrmpi import (
    MPIAMRTransactionError, _collective_error, _distributed_current_tree,
    _distributed_mixed_current_tree, _validate_mpi_integrator,
    _validate_mpi_mixed_hex_integrator, perform_one_mpi_amr_transaction,
    perform_one_mpi_mixed_hex_amr_transaction,
)
from pyfr.amrtransaction import _copy_state, _single_hex_bank
from pyfr.mpiutil import get_comm_rank_root


class AMRIndicatorError(ValueError):
    """An AMR indicator or deterministic mark decision is invalid."""


@dataclass(frozen=True)
class AMRIndicatorSettings:
    indicator: str
    refine_threshold: float
    coarsen_threshold: float
    density_floor: float
    acoustic_floor: float
    tie_tolerance: float
    min_level: int
    max_level: int
    max_refine_marks: int
    max_coarsen_families: int


@dataclass(frozen=True)
class AMRIndicatorDecision:
    action: str
    marks: tuple
    trigger_score: float | None


@dataclass(frozen=True)
class IndicatorMPIAMRResult:
    decision: AMRIndicatorDecision
    scores: tuple
    transaction: object | None


def density_variation_scores(state, density_index=0, density_floor=1e-14):
    """Return a relative within-element density variation for one Hex group."""
    state = np.asarray(state)
    if state.ndim != 3:
        raise AMRIndicatorError(
            'density indicator state must have shape (nupts, nvars, neles)'
        )
    if not isinstance(density_index, (int, np.integer)):
        raise AMRIndicatorError('density indicator index must be an integer')
    density_index = int(density_index)
    if not 0 <= density_index < state.shape[1]:
        raise AMRIndicatorError('density indicator index is outside nvars')
    if not np.isfinite(density_floor) or density_floor <= 0:
        raise AMRIndicatorError('density floor must be finite and positive')

    rho = state[:, density_index, :]
    if not np.isfinite(rho).all():
        raise AMRIndicatorError('density indicator input is not finite')

    rmin = np.min(rho, axis=0)
    rmax = np.max(rho, axis=0)
    rmean = np.mean(rho, axis=0)
    denom = np.maximum(np.abs(rmean), float(density_floor))
    scores = (rmax - rmin)/denom
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise AMRIndicatorError('density indicator produced invalid scores')

    return scores


def density_velocity_variation_scores(
    state, *, density_index, momentum_indices, energy_index, gamma,
    density_floor=1e-14, acoustic_floor=1e-14,
):
    """Return max density/velocity variation for one conserved Hex bank."""
    state = np.asarray(state)
    if state.ndim != 3:
        raise AMRIndicatorError(
            'composite indicator state must have shape (nupts, nvars, neles)'
        )
    if not isinstance(density_index, (int, np.integer)):
        raise AMRIndicatorError('density indicator index must be an integer')
    if not isinstance(energy_index, (int, np.integer)):
        raise AMRIndicatorError('energy indicator index must be an integer')
    density_index, energy_index = int(density_index), int(energy_index)
    momentum_indices = tuple(momentum_indices)
    if not momentum_indices or any(
        not isinstance(i, (int, np.integer)) for i in momentum_indices
    ):
        raise AMRIndicatorError(
            'momentum indicator indices must be nonempty integers'
        )
    momentum_indices = tuple(map(int, momentum_indices))
    indices = (density_index, *momentum_indices, energy_index)
    if any(not 0 <= i < state.shape[1] for i in indices):
        raise AMRIndicatorError('composite indicator index is outside nvars')
    if len(set(indices)) != len(indices):
        raise AMRIndicatorError('composite indicator indices must be distinct')
    if not np.isfinite(gamma) or gamma <= 1:
        raise AMRIndicatorError('gamma must be finite and greater than one')
    if not np.isfinite(acoustic_floor) or acoustic_floor <= 0:
        raise AMRIndicatorError(
            'indicator acoustic floor must be finite and positive'
        )

    density = density_variation_scores(
        state, density_index, density_floor
    )
    rho = state[:, density_index, :]
    if np.any(rho <= 0):
        raise AMRIndicatorError('composite indicator density is nonpositive')

    mom = state[:, momentum_indices, :]
    energy = state[:, energy_index, :]
    if not np.isfinite(mom).all() or not np.isfinite(energy).all():
        raise AMRIndicatorError('composite indicator input is not finite')

    velocity = mom/rho[:, None, :]
    kinetic = 0.5*np.sum(mom*mom, axis=1)/rho
    pressure = (float(gamma) - 1)*(energy - kinetic)
    if not np.isfinite(pressure).all() or np.any(pressure <= 0):
        raise AMRIndicatorError('composite indicator pressure is nonpositive')

    acoustic = np.sqrt(float(gamma)*pressure/rho)
    amean = np.mean(acoustic, axis=0)
    denom = np.maximum(amean, float(acoustic_floor))
    vspread = np.sqrt(np.sum(
        (np.max(velocity, axis=0) - np.min(velocity, axis=0))**2, axis=0
    ))
    velocity_score = vspread/denom
    scores = np.maximum(density, velocity_score)
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise AMRIndicatorError('composite indicator produced invalid scores')

    return scores


def _parent(leaf):
    root, path = leaf
    if not path:
        return None
    return root, path[:-1]


def select_hex_indicator_decision(
    tree, scores, *, refine_threshold, coarsen_threshold,
    tie_tolerance=1e-12, min_level=0, max_level=2,
    max_refine_marks=1, max_coarsen_families=1,
):
    """Select one deterministic refine/coarsen/no-op decision.

    Refinement has priority and may select a bounded batch of leaves.
    Otherwise a bounded batch of complete sibling families may be selected
    for coarsening.  D7 remains responsible for 2:1 closure and coarsening
    legality.
    """
    if (not np.isfinite(refine_threshold) or
            not np.isfinite(coarsen_threshold)):
        raise AMRIndicatorError('indicator thresholds must be finite')
    if not 0 <= coarsen_threshold < refine_threshold:
        raise AMRIndicatorError(
            'indicator hysteresis requires 0 <= coarsen < refine'
        )
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise AMRIndicatorError(
            'indicator tie tolerance must be finite and nonnegative'
        )
    tie_tolerance = float(tie_tolerance)
    if not isinstance(min_level, (int, np.integer)) or min_level < 0:
        raise AMRIndicatorError('indicator min level must be nonnegative')
    if not isinstance(max_level, (int, np.integer)) or max_level < 1:
        raise AMRIndicatorError('indicator max level must be positive')
    min_level, max_level = int(min_level), int(max_level)
    if min_level >= max_level:
        raise AMRIndicatorError('indicator min level must be below max level')
    if (not isinstance(max_refine_marks, (int, np.integer)) or
            max_refine_marks < 1):
        raise AMRIndicatorError(
            'indicator max refine marks must be a positive integer'
        )
    max_refine_marks = int(max_refine_marks)
    if (not isinstance(max_coarsen_families, (int, np.integer)) or
            max_coarsen_families < 1):
        raise AMRIndicatorError(
            'indicator max coarsen families must be a positive integer'
        )
    max_coarsen_families = int(max_coarsen_families)

    leaves = tuple(tree.leaves())
    if set(scores) != set(leaves):
        raise AMRIndicatorError(
            'indicator scores must cover the active tree exactly'
        )
    vals = {}
    for leaf in leaves:
        val = scores[leaf]
        if not np.isscalar(val) or not np.isfinite(val) or val < 0:
            raise AMRIndicatorError(
                f'indicator score for {leaf!r} is invalid'
            )
        vals[leaf] = float(val)

    refine = [
        leaf for leaf in leaves
        if len(leaf[1]) < max_level and vals[leaf] >= refine_threshold
    ]
    if refine:
        pending = list(refine)
        selected = []
        while pending and len(selected) < max_refine_marks:
            best = max(vals[leaf] for leaf in pending)
            atol = tie_tolerance*max(1.0, abs(best))
            tied = sorted(
                leaf for leaf in pending if best - vals[leaf] <= atol
            )
            selected.extend(tied[:max_refine_marks - len(selected)])
            tied = set(tied)
            pending = [leaf for leaf in pending if leaf not in tied]

        selected = tuple(selected)
        trigger = min(vals[leaf] for leaf in selected)
        return AMRIndicatorDecision('refine', selected, trigger)

    active = set(leaves)
    parents = sorted({p for leaf in leaves if (p := _parent(leaf))})
    candidates = []
    for parent in parents:
        if len(parent[1]) < min_level:
            continue
        children = hex_octree_children(parent)
        if children <= active:
            worst = max(vals[c] for c in children)
            if worst <= coarsen_threshold:
                candidates.append((worst, parent, tuple(sorted(children))))

    if candidates:
        pending = list(candidates)
        selected = []
        while pending and len(selected) < max_coarsen_families:
            best = min(c[0] for c in pending)
            atol = tie_tolerance*max(1.0, abs(best))
            tied = sorted(
                (c for c in pending if c[0] - best <= atol),
                key=lambda x: x[1],
            )
            selected.extend(
                tied[:max_coarsen_families - len(selected)]
            )
            tied_parents = {c[1] for c in tied}
            pending = [c for c in pending if c[1] not in tied_parents]

        marks = tuple(sorted(
            child for _, _, children in selected for child in children
        ))
        trigger = max(worst for worst, _, _ in selected)
        return AMRIndicatorDecision('coarsen', marks, trigger)

    return AMRIndicatorDecision('none', (), None)


def indicator_settings(cfg):
    section = 'solver-amr'
    if section not in cfg.sections():
        raise AMRIndicatorError('indicator-driven AMR requires [solver-amr]')
    required = ('refine-threshold', 'coarsen-threshold')
    if any(not cfg.hasopt(section, opt) for opt in required):
        raise AMRIndicatorError(
            '[solver-amr] must set refine-threshold and coarsen-threshold'
        )

    indicator = cfg.get(section, 'indicator', 'density-variation')
    supported = {'density-variation', 'density-velocity-variation'}
    if indicator not in supported:
        raise AMRIndicatorError(f'unsupported AMR indicator {indicator!r}')
    acoustic_floor = (
        cfg.getfloat(section, 'acoustic-floor', 1e-14)
        if indicator == 'density-velocity-variation' else 1e-14
    )

    settings = AMRIndicatorSettings(
        indicator=indicator,
        refine_threshold=cfg.getfloat(section, 'refine-threshold'),
        coarsen_threshold=cfg.getfloat(section, 'coarsen-threshold'),
        density_floor=cfg.getfloat(section, 'density-floor', 1e-14),
        acoustic_floor=acoustic_floor,
        tie_tolerance=cfg.getfloat(section, 'tie-tolerance', 1e-12),
        min_level=cfg.getint(section, 'min-level', 0),
        max_level=cfg.getint(section, 'max-level', 2),
        max_refine_marks=cfg.getint(section, 'max-refine-marks', 1),
        max_coarsen_families=cfg.getint(
            section, 'max-coarsen-families', 1
        ),
    )

    # Reuse decision validation for the numerical policy parameters.
    class _EmptyTree:
        @staticmethod
        def leaves():
            return ()

    select_hex_indicator_decision(
        _EmptyTree(), {}, refine_threshold=settings.refine_threshold,
        coarsen_threshold=settings.coarsen_threshold,
        tie_tolerance=settings.tie_tolerance, min_level=settings.min_level,
        max_level=settings.max_level,
        max_refine_marks=settings.max_refine_marks,
        max_coarsen_families=settings.max_coarsen_families,
    )
    if (not np.isfinite(settings.density_floor) or
            settings.density_floor <= 0):
        raise AMRIndicatorError(
            'indicator density floor must be finite and positive'
        )
    if (settings.indicator == 'density-velocity-variation' and
            (not np.isfinite(settings.acoustic_floor) or
             settings.acoustic_floor <= 0)):
        raise AMRIndicatorError(
            'indicator acoustic floor must be finite and positive'
        )
    return settings


def _merge_global_scores(comm, local_scores, tree):
    chunks = comm.allgather(local_scores)
    merged = {}
    for chunk in chunks:
        for leaf, score in chunk.items():
            if leaf in merged:
                raise AMRIndicatorError(
                    f'indicator leaf {leaf!r} is owned by multiple ranks'
                )
            merged[leaf] = score
    if set(merged) != set(tree.leaves()):
        raise AMRIndicatorError(
            'indicator score ownership does not cover the active tree exactly'
        )
    return merged


def evaluate_mpi_density_indicator(intg):
    """Collectively evaluate density scores and choose one global action."""
    comm, _, _ = get_comm_rank_root()
    try:
        comm, system, mesh, _ = _validate_mpi_integrator(intg)
        settings = indicator_settings(intg.cfg)
        tree, local_by_leaf = _distributed_current_tree(mesh, comm)
    except Exception as exc:
        _collective_error(comm, 'indicator validation', exc)
        raise AssertionError('unreachable')
    _collective_error(comm, 'indicator validation')

    cfgs = comm.allgather(settings)
    if len(set(cfgs)) != 1:
        _collective_error(
            comm, 'indicator configuration agreement',
            AMRIndicatorError('MPI ranks disagree on [solver-amr] settings'),
        )
    settings = cfgs[0]

    error = None
    try:
        bank = intg.idxcurr
        shape, _ = _single_hex_bank(system, bank, 'indicator accepted')
        parts = _copy_state(system, bank)
        if len(parts) != 1 or tuple(parts[0].shape) != shape:
            raise AMRIndicatorError('indicator accepted bank is invalid')
        state = parts[0]
        if state.shape[2] != len(local_by_leaf):
            raise AMRIndicatorError(
                'indicator bank columns do not match local leaf ownership'
            )

        convars = system.elementscls.convars(system.ndims, intg.cfg)
        density_index = convars.index('rho')
        if settings.indicator == 'density-variation':
            values = density_variation_scores(
                state, density_index, settings.density_floor
            )
        else:
            momentum_indices = tuple(
                i for i, name in enumerate(convars)
                if name.startswith('rho') and name != 'rho'
            )
            energy_index = convars.index('E')
            gamma = intg.cfg.getfloat('constants', 'gamma')
            values = density_velocity_variation_scores(
                state, density_index=density_index,
                momentum_indices=momentum_indices,
                energy_index=energy_index, gamma=gamma,
                density_floor=settings.density_floor,
                acoustic_floor=settings.acoustic_floor,
            )
        local_scores = {
            leaf: float(values[col]) for leaf, col in local_by_leaf.items()
        }
    except Exception as exc:
        error = exc
        local_scores = None
    _collective_error(comm, 'indicator evaluation', error)

    error = None
    try:
        scores = _merge_global_scores(comm, local_scores, tree)
        decision = select_hex_indicator_decision(
            tree, scores,
            refine_threshold=settings.refine_threshold,
            coarsen_threshold=settings.coarsen_threshold,
            tie_tolerance=settings.tie_tolerance,
            min_level=settings.min_level,
            max_level=settings.max_level,
            max_refine_marks=settings.max_refine_marks,
            max_coarsen_families=settings.max_coarsen_families,
        )
    except Exception as exc:
        error = exc
        decision = None
        scores = None
    _collective_error(comm, 'indicator decision', error)

    decisions = comm.allgather(decision)
    if len(set(decisions)) != 1:
        _collective_error(
            comm, 'indicator decision agreement',
            AMRIndicatorError('MPI ranks produced different AMR decisions'),
        )
    return decision, scores, local_by_leaf



def evaluate_mpi_mixed_hex_indicator(intg):
    """Collectively evaluate D9Q on a distributed mixed Hex topology."""
    comm, _, _ = get_comm_rank_root()
    try:
        comm, system, mesh, _ = _validate_mpi_mixed_hex_integrator(intg)
        settings = indicator_settings(intg.cfg)
        tree, local_by_leaf = _distributed_mixed_current_tree(mesh, comm)
    except Exception as exc:
        _collective_error(comm, 'mixed indicator validation', exc)
        raise AssertionError('unreachable')
    _collective_error(comm, 'mixed indicator validation')

    cfgs = comm.allgather(settings)
    if len(set(cfgs)) != 1:
        _collective_error(
            comm, 'mixed indicator configuration agreement',
            AMRIndicatorError(
                'MPI ranks disagree on mixed [solver-amr] settings'
            ),
        )
    settings = cfgs[0]

    error = None
    try:
        bank = intg.idxcurr
        states = dict(zip(system.ele_types, _copy_state(system, bank)))
        state = states.get('hex')
        if state is None:
            if local_by_leaf:
                raise AMRIndicatorError(
                    'mixed indicator Hex ownership/state mismatch'
                )
            local_scores = {}
        else:
            if state.ndim != 3 or state.shape[2] != len(local_by_leaf):
                raise AMRIndicatorError(
                    'mixed indicator bank columns do not match local Hex '
                    'ownership'
                )

            convars = system.elementscls.convars(system.ndims, intg.cfg)
            density_index = convars.index('rho')
            if settings.indicator == 'density-variation':
                values = density_variation_scores(
                    state, density_index, settings.density_floor
                )
            else:
                momentum_indices = tuple(
                    i for i, name in enumerate(convars)
                    if name.startswith('rho') and name != 'rho'
                )
                energy_index = convars.index('E')
                gamma = intg.cfg.getfloat('constants', 'gamma')
                values = density_velocity_variation_scores(
                    state, density_index=density_index,
                    momentum_indices=momentum_indices,
                    energy_index=energy_index, gamma=gamma,
                    density_floor=settings.density_floor,
                    acoustic_floor=settings.acoustic_floor,
                )
            local_scores = {
                leaf: float(values[col])
                for leaf, col in local_by_leaf.items()
            }
    except Exception as exc:
        error = exc
        local_scores = None
    _collective_error(comm, 'mixed indicator evaluation', error)

    error = None
    try:
        scores = _merge_global_scores(comm, local_scores, tree)
        decision = select_hex_indicator_decision(
            tree, scores,
            refine_threshold=settings.refine_threshold,
            coarsen_threshold=settings.coarsen_threshold,
            tie_tolerance=settings.tie_tolerance,
            min_level=settings.min_level, max_level=settings.max_level,
            max_refine_marks=settings.max_refine_marks,
            max_coarsen_families=settings.max_coarsen_families,
        )
        if decision.action == 'coarsen':
            raise AMRIndicatorError(
                'V10K distributed mixed Hex D9Q is refinement-only'
            )
    except Exception as exc:
        error = exc
        decision = None
        scores = None
    _collective_error(comm, 'mixed indicator decision', error)

    decisions = comm.allgather(decision)
    if len(set(decisions)) != 1:
        _collective_error(
            comm, 'mixed indicator decision agreement',
            AMRIndicatorError(
                'MPI ranks produced different mixed AMR decisions'
            ),
        )
    return decision, scores, local_by_leaf


def perform_indicator_mpi_mixed_hex_amr_transaction(
    intg, *, shared_stage_dir, repartition=True
):
    """Evaluate D9Q then execute the V10K mixed-Hex MPI transaction."""
    decision, scores, local_by_leaf = evaluate_mpi_mixed_hex_indicator(intg)
    score_items = tuple((leaf, scores[leaf]) for leaf in sorted(scores))
    if decision.action == 'none':
        return IndicatorMPIAMRResult(decision, score_items, None)

    local_marks = tuple(
        leaf for leaf in decision.marks if leaf in local_by_leaf
    )
    result = perform_one_mpi_mixed_hex_amr_transaction(
        intg, local_marks, shared_stage_dir=shared_stage_dir,
        repartition=repartition,
    )
    return IndicatorMPIAMRResult(decision, score_items, result)

def perform_indicator_mpi_amr_transaction(
    intg, *, shared_stage_dir, ownership_policy='balanced-affinity-v1'
):
    """Evaluate D9A and, if marked, execute the accepted D7 transaction."""
    decision, scores, local_by_leaf = evaluate_mpi_density_indicator(intg)
    score_items = tuple((leaf, scores[leaf]) for leaf in sorted(scores))
    if decision.action == 'none':
        return IndicatorMPIAMRResult(decision, score_items, None)

    local_marks = tuple(
        leaf for leaf in decision.marks if leaf in local_by_leaf
    )
    result = perform_one_mpi_amr_transaction(
        intg, local_marks, shared_stage_dir=shared_stage_dir,
        action=decision.action, ownership_policy=ownership_policy,
    )
    return IndicatorMPIAMRResult(decision, score_items, result)
