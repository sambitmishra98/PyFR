from dataclasses import dataclass
import os
import shutil
import tempfile
import weakref

import numpy as np

from pyfr.amr import (
    QuadLeafTree, apply_hex_refine_transfer, apply_hex_restrict_transfer,
    build_hex_refine_transfer, build_hex_restrict_transfer,
    encode_hex_leaf_tree, encode_quad_leaf_tree, hex_coarsen_candidates,
    hex_octree_children, hex_tree_face_groups, hex_tree_face_pairs,
    hex_tree_leaves,
)
from pyfr.amrmesh import (
    _derive_quad_root_face_topology, _derive_root_face_topology,
    _derive_mixed_quad_root_face_topology, _hex_local_maps, _quad_local_maps,
    materialize_native_hex_tree, materialize_native_mixed_quad_tree,
    materialize_native_quad_tree,
)
from pyfr.amrwriter import (
    build_adapted_mixed_quad_mesh, build_adapted_quad_mesh,
    write_adapted_mesh,
)
from pyfr.cache import clear_memoize
from pyfr.mpiutil import get_comm_rank_root
from pyfr.quadrules import get_quadrule
from pyfr.readers.native import NativeReader, Solution
from pyfr.shapes import HexShape, QuadShape, TriShape
from pyfr.writers.serialise import Serialiser


class AMRTransactionError(RuntimeError):
    pass


@dataclass(frozen=True)
class AMRTransactionResult:

    action: str
    old_tree: object
    proposed_tree: object
    raw_mesh: object
    transferred_state: np.ndarray
    rhs: tuple
    conservation_before: np.ndarray
    conservation_after: np.ndarray
    conservation_error: np.ndarray
    old_volume: float
    proposed_volume: float
    old_rho_range: tuple
    old_pressure_range: tuple
    rho_range: tuple
    pressure_range: tuple
    stage_mesh_uuid: str
    stage_leaf_count: int
    stage_mortar_count: int
    stage_mortar_formats: tuple
    bank_drift: float
    old_system_id: int
    new_system_id: int
    accepted_bank: int
    scratch_bank: int
    stage_leaf_to_bank: tuple
    stage_accepted_shape: tuple
    tcurr: float


@dataclass(frozen=True)
class QuadAMRDecision:

    action: str
    marks: tuple
    trigger_score: float | None


@dataclass(frozen=True)
class QuadIndicatorAMRResult:

    decision: QuadAMRDecision
    scores: tuple
    transaction: object | None


@dataclass(frozen=True)
class QuadAMRTransactionResult:

    old_tree: object
    proposed_tree: object
    raw_mesh: object
    transferred_state: np.ndarray
    rhs: tuple
    refine_marks: tuple
    wall_splits: tuple
    closure_splits: tuple
    conservation_before: np.ndarray
    conservation_after: np.ndarray
    conservation_error: np.ndarray
    old_rho_range: tuple
    old_pressure_range: tuple
    rho_range: tuple
    pressure_range: tuple
    stage_mesh_uuid: str
    stage_leaf_count: int
    stage_mortar_count: int
    stage_mortar_formats: tuple
    bank_drift: float
    old_system_id: int
    new_system_id: int
    accepted_bank: int
    scratch_bank: int
    stage_accepted_shape: tuple
    tcurr: float


@dataclass(frozen=True)
class MixedQuadAMRTransactionResult:

    old_tree: object
    proposed_tree: object
    raw_mesh: object
    transferred_state: dict
    rhs: tuple
    refine_marks: tuple
    wall_splits: tuple
    closure_splits: tuple
    fixed_blocked_marks: tuple
    conservation_before: np.ndarray
    conservation_after: np.ndarray
    conservation_error: np.ndarray
    old_rho_range: tuple
    old_pressure_range: tuple
    rho_range: tuple
    pressure_range: tuple
    stage_mesh_uuid: str
    stage_leaf_count: int
    stage_tri_count: int
    stage_mortar_count: int
    stage_mortar_formats: tuple
    bank_drift: float
    old_system_id: int
    new_system_id: int
    accepted_bank: int
    scratch_bank: int
    stage_accepted_shapes: tuple
    tcurr: float


def _copy_state(system, bank):
    return tuple(np.array(a, copy=True, order='C')
                 for a in system.ele_scal_upts(bank))


def _max_abs_diff(lhs, rhs):
    return max(
        (float(np.max(np.abs(a - b), initial=0.0))
         for a, b in zip(lhs, rhs)), default=0.0
    )


def _scratch_bank(system, accepted_bank):
    if accepted_bank < 0 or accepted_bank >= system.nrhs:
        raise AMRTransactionError('D6A accepted solution bank is invalid')

    try:
        return next(i for i in range(system.nrhs) if i != accepted_bank)
    except StopIteration:
        raise AMRTransactionError(
            'D6A requires a distinct RHS scratch register'
        ) from None


def _single_hex_bank(system, bank, label):
    if list(getattr(system, 'ele_types', ())) != ['hex']:
        raise AMRTransactionError(
            f'D6A {label} system requires exactly one Hex element group'
        )
    if len(getattr(system, 'ele_banks', ())) != 1:
        raise AMRTransactionError(
            f'D6A {label} system requires exactly one solution bank group'
        )
    if bank < 0 or bank >= getattr(system, 'nrhs', 0):
        raise AMRTransactionError(f'D6A {label} solution bank is invalid')

    try:
        group_bank = system.ele_banks[0][bank]
        group_shape = tuple(system.ele_shapes['hex'])
        bank_shape = tuple(group_bank.ioshape)
    except (AttributeError, IndexError, KeyError, TypeError):
        raise AMRTransactionError(
            f'D6A {label} Hex bank layout is not available'
        ) from None

    if len(group_shape) != 3 or any(n <= 0 for n in group_shape):
        raise AMRTransactionError(
            f'D6A {label} Hex group has an invalid logical shape'
        )
    if bank_shape != group_shape:
        raise AMRTransactionError(
            f'D6A {label} Hex bank shape does not match its element group'
        )

    return group_shape, group_bank


def _stage_hex_native_leaf_order(stage_mesh, raw):
    if list(stage_mesh.etypes) != ['hex']:
        raise AMRTransactionError(
            'D6A staged mesh requires exactly one active Hex element group'
        )
    if stage_mesh.amr_tree is None:
        raise AMRTransactionError('D6A staged mesh lost persistent ancestry')

    physical_leaves = tuple(raw.leaf_order)
    native_leaves = tuple(stage_mesh.amr_tree.leaves())
    if (not physical_leaves or
        len(set(physical_leaves)) != len(physical_leaves) or
        len(set(native_leaves)) != len(native_leaves) or
        native_leaves != physical_leaves):
        raise AMRTransactionError(
            'D6A physical leaf order does not match staged native ordering'
        )

    native_eidxs = np.asarray(stage_mesh.eidxs.get('hex', ()),
                              dtype=np.int64)
    expected_eidxs = np.arange(len(physical_leaves), dtype=np.int64)
    if native_eidxs.ndim != 1 or not np.array_equal(native_eidxs,
                                                     expected_eidxs):
        raise AMRTransactionError(
            'D6A staged Hex eidxs do not equal canonical leaf ordinals'
        )

    return native_leaves, native_eidxs


def _stage_hex_leaf_mapping(stage_system, stage_mesh, raw, transferred, bank):
    group_shape, group_bank = _single_hex_bank(stage_system, bank, 'staged')
    native_leaves, native_eidxs = _stage_hex_native_leaf_order(
        stage_mesh, raw
    )
    if group_shape[2] != len(native_leaves):
        raise AMRTransactionError(
            'D6A staged Hex group does not contain every proposed leaf'
        )
    if tuple(transferred.shape) != group_shape:
        raise AMRTransactionError(
            'D6A transferred state does not match the staged Hex bank shape'
        )

    return tuple(zip(native_leaves, map(int, native_eidxs))), (
        group_shape, group_bank
    )


def _inject_staged_hex_bank(stage_system, stage_mesh, raw, transferred, bank):
    leaf_to_bank, (group_shape, group_bank) = _stage_hex_leaf_mapping(
        stage_system, stage_mesh, raw, transferred, bank
    )
    injected = np.array(transferred, copy=True, order='C')
    if tuple(injected.shape) != group_shape:
        raise AMRTransactionError(
            'D6A staged Hex injection changed the validated bank shape'
        )

    # Matrix.set requires an exact full ioshape; no solution-point iteration
    # or broadcasting participates in this group-level state bridge.
    group_bank.set(injected)
    stage_system.backend.wait()

    staged_parts = _copy_state(stage_system, bank)
    if len(staged_parts) != 1 or tuple(staged_parts[0].shape) != group_shape:
        raise AMRTransactionError(
            'D6A staged Hex state-read interface returned an invalid shape'
        )
    staged_state = staged_parts[0]
    if not np.array_equal(staged_state, transferred):
        raise AMRTransactionError(
            'D6A staged accepted bank differs from injected transfer'
        )

    return staged_state, leaf_to_bank, group_shape


def _normalise_mark(mark):
    if not isinstance(mark, (tuple, list)) or len(mark) != 2:
        raise AMRTransactionError(
            'D6A scripted marks must be (root_eidx, octant_path) leaf ids'
        )

    root, path = mark
    if not isinstance(root, (int, np.integer)) or root < 0:
        raise AMRTransactionError(f'Invalid scripted root Hex id: {root!r}')

    try:
        path = tuple(int(o) for o in path)
    except (TypeError, ValueError):
        raise AMRTransactionError('Invalid scripted octant path') from None

    if any(o < 0 or o > 7 for o in path):
        raise AMRTransactionError('Scripted octants must be in 0..7')

    return int(root), path


def _validate_integrator(intg, restart_root_mesh=None):
    if getattr(intg, 'formulation', None) != 'explicit':
        raise AMRTransactionError('D6A requires the explicit formulation')
    if getattr(intg, 'controller_name', None) != 'none':
        raise AMRTransactionError(
            'D6A requires the fixed-step none controller'
        )
    if getattr(intg, 'stepper_name', None) != 'rk4':
        raise AMRTransactionError(
            'D6A requires the standard explicit RK4 stepper'
        )
    if getattr(getattr(intg, 'backend', None), 'name', None) != 'openmp':
        raise AMRTransactionError('D6A currently supports OpenMP only')
    if getattr(intg, 'nacptsteps', 0) < 1:
        raise AMRTransactionError(
            'D6A requires a completed accepted physical timestep'
        )
    if getattr(intg, 'stepinfo', None) != []:
        raise AMRTransactionError(
            'D6A requires the post-advance accepted-step safe point'
        )

    cfg = intg.cfg
    solver_system = cfg.get('solver', 'system')
    if solver_system not in {'euler', 'navier-stokes'}:
        raise AMRTransactionError(
            'D6D currently supports Euler and Navier-Stokes only'
        )
    if (solver_system == 'navier-stokes' and
            cfg.get('solver', 'viscosity-correction', 'none') != 'none'):
        raise AMRTransactionError(
            'D6D Navier-Stokes requires constant viscosity'
        )
    if (cfg.get('solver-time-integrator', 'formulation', 'explicit')
            != 'explicit'):
        raise AMRTransactionError('D6A requires an explicit time integrator')
    if cfg.get('solver-time-integrator', 'controller', 'none') != 'none':
        raise AMRTransactionError('D6A requires controller = none')
    if cfg.get('solver-time-integrator', 'scheme', 'rk4') != 'rk4':
        raise AMRTransactionError('D6A requires scheme = rk4')
    if cfg.get('solver', 'shock-capturing', 'none') != 'none':
        raise AMRTransactionError('D6A requires shock-capturing = none')
    if any(s.startswith('solver-order-') for s in cfg.sections()):
        raise AMRTransactionError('D6A does not support mixed-p systems')

    if getattr(intg, 'plugins', ()):
        raise AMRTransactionError(
            'D6A does not carry generic plugins across the topology event'
        )
    if any(s.startswith(('soln-plugin-', 'solver-plugin-'))
           for s in cfg.sections()):
        raise AMRTransactionError(
            'D6A requires a no-plugin transaction configuration'
        )
    if getattr(intg, 'triggers', None):
        raise AMRTransactionError(
            'D6A does not carry trigger-managed state across topology changes'
        )

    serialfns = getattr(getattr(intg, 'serialiser', None), '_serialfns', {})
    if any(k.startswith('bcs/') for k in serialfns):
        raise AMRTransactionError('D6A requires stateless boundary conditions')
    if 'triggers' in serialfns or any(
        k.startswith(('trigger-src/', 'trigger-pub/')) for k in serialfns
    ):
        raise AMRTransactionError(
            'D6A does not carry trigger-managed serialised state'
        )
    if serialfns:
        raise AMRTransactionError(
            'D6A does not carry unsupported serialised mutable state'
        )

    comm, _, _ = get_comm_rank_root()
    if comm.size != 1:
        raise AMRTransactionError('D6A requires a single MPI rank')

    system = getattr(intg, 'system', None)
    mesh = getattr(system, 'mesh', None)
    if system is None or mesh is None:
        raise AMRTransactionError('D6A requires a live PyFR system and mesh')
    if getattr(system, 'name', None) != solver_system:
        raise AMRTransactionError(
            'D6D live system does not match the configured solver system'
        )
    if mesh.etypes != ['hex']:
        raise AMRTransactionError('D6B requires a pure Hex native mesh')
    if mesh.con_p:
        raise AMRTransactionError('D6B does not support MPI connectivity')
    if np.any(mesh.spts_curved.get('hex', ())):
        raise AMRTransactionError('D6B requires affine Hex geometry')

    if mesh.amr_tree is None:
        if mesh.mcon:
            raise AMRTransactionError(
                'D6B root mesh does not support pre-existing mortars'
            )
        root_mesh = mesh
    else:
        for mcon in mesh.mcon.values():
            if (mcon.format != 'one-to-many-v1' or
                mcon.template != 'quad-2x2' or mcon.nright != 4):
                raise AMRTransactionError(
                    'D6B adapted mesh contains an unsupported mortar'
                )

        live_root_mesh = getattr(intg, '_amr_root_mesh', None)
        if live_root_mesh is not None and restart_root_mesh is not None:
            if live_root_mesh.uuid != restart_root_mesh.uuid:
                raise AMRTransactionError(
                    'D6C supplied root mesh differs from the live root anchor'
                )

        root_mesh = live_root_mesh or restart_root_mesh
        if root_mesh is None:
            raise AMRTransactionError(
                'D6C adapted restart requires the immutable root mesh'
            )
        if getattr(root_mesh, 'amr_tree', None) is not None:
            raise AMRTransactionError(
                'D6C root-mesh anchor must be an unadapted native mesh'
            )
        if root_mesh.uuid != mesh.amr_tree.root_mesh_uuid:
            raise AMRTransactionError(
                'D6C root-mesh anchor does not match persisted ancestry'
            )

    return system, mesh, root_mesh


def _root_tree(mesh):
    geidx = np.asarray(mesh.eidxs.get('hex', ()), dtype=np.int64)
    if geidx.ndim != 1 or len(geidx) == 0:
        raise AMRTransactionError('D6A root mesh has no local Hex elements')
    if len(np.unique(geidx)) != len(geidx):
        raise AMRTransactionError('D6A root Hex global ids are not unique')

    leaves = [(int(eidx), ()) for eidx in geidx]
    tree = encode_hex_leaf_tree(mesh.uuid, leaves)

    # Root identity, native row, and solution column must map one-to-one.
    local_by_leaf = {(int(eidx), ()): i for i, eidx in enumerate(geidx)}
    if len(local_by_leaf) != len(geidx):
        raise AMRTransactionError(
            'D6A root-to-runtime Hex mapping is ambiguous'
        )

    return tree, local_by_leaf


def _current_tree(mesh):
    if mesh.amr_tree is None:
        return _root_tree(mesh)

    tree = mesh.amr_tree
    leaves = tuple(tree.leaves())
    geidx = np.asarray(mesh.eidxs.get('hex', ()), dtype=np.int64)
    if len(geidx) != len(leaves):
        raise AMRTransactionError(
            'D6B adapted Hex count differs from persisted leaf ancestry'
        )
    if set(map(int, geidx)) != set(range(len(leaves))):
        raise AMRTransactionError(
            'D6B adapted global Hex ids are not canonical leaf ordinals'
        )

    local_by_leaf = {leaves[int(eidx)]: i for i, eidx in enumerate(geidx)}
    if len(local_by_leaf) != len(leaves):
        raise AMRTransactionError(
            'D6B adapted leaf-to-runtime mapping is ambiguous'
        )
    return tree, local_by_leaf


def _tree_split_nodes(tree):
    return {
        (root, path[:k])
        for root, path in tree.leaves()
        for k in range(len(path))
    }


def _close_refinement(mesh, old_tree, mark):
    l2g, _, pids_by_g, _ = _hex_local_maps(mesh)
    rootfaces, rootkinds = _derive_root_face_topology(mesh, pids_by_g, l2g)
    roots = sorted(pids_by_g)
    old_leaves = tuple(old_tree.leaves())
    if mark not in old_leaves:
        raise AMRTransactionError(
            f'Scripted mark is not an active leaf: {mark!r}'
        )

    # Reconstruct split nodes and add neighbours required for 2:1 balance.
    split = _tree_split_nodes(old_tree) | {mark}
    while True:
        leaves = hex_tree_leaves(roots, split)
        groups = hex_tree_face_groups(leaves, rootfaces)
        try:
            pairs = list(hex_tree_face_pairs(groups, rootkinds))
        except ValueError as exc:
            raise AMRTransactionError(str(exc)) from exc

        added = False
        for lhs, rhs in pairs:
            if abs(lhs['level'] - rhs['level']) <= 1:
                continue
            coarse = (lhs['leaf'] if lhs['level'] < rhs['level']
                      else rhs['leaf'])
            if coarse not in leaves:
                raise AMRTransactionError(
                    'D6A closure selected a non-active coarse leaf'
                )
            if coarse not in split:
                split.add(coarse)
                added = True

        if not added:
            break

    try:
        return encode_hex_leaf_tree(mesh.uuid, leaves)
    except ValueError as exc:
        raise AMRTransactionError(str(exc)) from exc


def _close_coarsening(mesh, old_tree, marks):
    l2g, _, pids_by_g, _ = _hex_local_maps(mesh)
    rootfaces, rootkinds = _derive_root_face_topology(mesh, pids_by_g, l2g)
    leaves = tuple(old_tree.leaves())
    groups = hex_tree_face_groups(leaves, rootfaces)
    try:
        pairs = list(hex_tree_face_pairs(groups, rootkinds))
        face_pairs = [(lhs['leaf'], rhs['leaf']) for lhs, rhs in pairs]
        approved = hex_coarsen_candidates(leaves, marks, face_pairs)
    except ValueError as exc:
        raise AMRTransactionError(str(exc)) from exc

    if len(approved) != 1:
        raise AMRTransactionError(
            'D6B requires exactly one legal sibling family to coarsen'
        )
    parent = approved[0]
    children = hex_octree_children(parent)
    if set(marks) != children:
        raise AMRTransactionError(
            'D6B coarsening marks must be exactly eight direct siblings'
        )

    proposed = (set(leaves) - children) | {parent}
    try:
        return encode_hex_leaf_tree(old_tree.root_mesh_uuid, proposed)
    except ValueError as exc:
        raise AMRTransactionError(str(exc)) from exc


def _affine_volume(pids, node_loc):
    src = HexShape.std_ele(1)
    dst = np.array([node_loc[int(i)] for i in pids], dtype=float)
    lhs = np.column_stack((src, np.ones(len(src))))
    coeff, _, rank, _ = np.linalg.lstsq(lhs, dst, rcond=None)
    if rank != 4:
        raise AMRTransactionError('D6A encountered degenerate Hex geometry')

    fitted = lhs @ coeff
    scale = max(1.0, float(np.max(np.abs(dst), initial=0.0)))
    if np.max(np.abs(fitted - dst), initial=0.0) > 1e-10*scale:
        raise AMRTransactionError('D6A requires affine Hex geometry')

    det = float(np.linalg.det(coeff[:3].T))
    if det <= 1e-10*scale**3:
        raise AMRTransactionError('D6A requires positive Hex Jacobians')
    return 8.0*det


def _mesh_volumes(mesh):
    loc = {int(i): c for i, c in zip(mesh.node_idxs, mesh.node_locs)}
    return np.array([
        _affine_volume(row, loc) for row in mesh.spts_nodes['hex']
    ])


def _raw_volumes(raw):
    loc = {int(i): c for i, c in zip(raw.node_ids, raw.node_locs)}
    return np.array([_affine_volume(row, loc) for row in raw.hex_nodes])


def _physical_hex_conserved_totals(state, spts, basis):
    from pyfr.polys import get_polybasis

    spts = np.asarray(spts, dtype=float)
    gorder = HexShape.order_from_npts(spts.shape[0])
    n1d = max(basis.order + 2, 2*gorder + 2)
    qrule = get_quadrule(
        'hex', rule='gauss-legendre', npts=n1d**3
    )

    sint = basis.ubasis.nodal_basis_at(qrule.pts)
    gbasis = get_polybasis(
        'hex', gorder, HexShape.std_ele(gorder)
    )
    gdiff = gbasis.jac_nodal_basis_at(qrule.pts)
    jac = np.einsum('dsk,sec->kedc', gdiff, spts)
    det = np.linalg.det(jac)
    scale = max(1.0, float(np.max(np.abs(jac), initial=0.0)))
    if not np.isfinite(det).all() or np.min(det) <= 1e-13*scale**3:
        raise AMRTransactionError(
            'V10J represented Hex geometry has a nonpositive Jacobian'
        )

    values = np.einsum('ku,uvn->kvn', sint, state)
    return np.einsum('k,kn,kvn->v', qrule.wts, det, values)


def _raw_hex_spts(raw):
    loc = {int(i): c for i, c in zip(raw.node_ids, raw.node_locs)}
    values = np.array([
        [loc[int(i)] for i in row] for row in raw.hex_nodes
    ], dtype=float)
    return np.swapaxes(values, 0, 1)


def _integration_weights(cfg, basis):
    order = basis.order
    qrule = get_quadrule('hex', rule='gauss-legendre', npts=(order + 1)**3)
    return qrule.wts @ basis.ubasis.nodal_basis_at(qrule.pts)


def _conserved_totals(state, volumes, weights):
    # ``weights`` integrate on the PyFR reference Hex [-1, 1]^3, whose
    # volume is eight.  For an affine physical Hex, det(J) = volume / 8.
    return np.einsum('u,uvn,n->v', weights, state, volumes / 8.0)


def _eos_ranges(system, state):
    if not all(np.isfinite(a).all() for a in state):
        raise AMRTransactionError('D6A conservative state is not finite')

    cons = np.moveaxis(state, 1, 0)
    pri = system.elementscls.con_to_pri(cons, system.cfg)
    rho = np.asarray(pri[0])
    pressure = np.asarray(pri[-1])
    if not np.isfinite(rho).all() or not np.isfinite(pressure).all():
        raise AMRTransactionError('D6D primitive state is not finite')
    if np.min(rho) <= 0 or np.min(pressure) <= 0:
        raise AMRTransactionError(
            'D6D compressible state requires rho > 0 and p > 0'
        )
    return (float(np.min(rho)), float(np.max(rho))), (
        float(np.min(pressure)), float(np.max(pressure))
    )


def _transfer_state(
    system, old_tree, proposed_tree, old_state, local_by_leaf
):
    basis = HexShape(None, system.cfg)
    refine = build_hex_refine_transfer(basis)
    restrict = build_hex_restrict_transfer(basis)
    if old_state.shape[0] != refine.nupts:
        raise AMRTransactionError('D6B solution point mapping is ambiguous')

    old_leaves = set(old_tree.leaves())
    if set(local_by_leaf) != old_leaves:
        raise AMRTransactionError(
            'D6B persisted leaves do not match the live solution mapping'
        )
    old_by_leaf = {leaf: old_state[:, :, i]
                   for leaf, i in local_by_leaf.items()}
    result = np.empty((old_state.shape[0], old_state.shape[1],
                       proposed_tree.nleaves), dtype=old_state.dtype)
    refined = {}
    restricted = {}

    for i, leaf in enumerate(proposed_tree.leaves()):
        if leaf in old_by_leaf:
            result[:, :, i] = old_by_leaf[leaf]
            continue

        root, path = leaf
        parent = (root, path[:-1]) if path else None
        if parent in old_leaves:
            if parent not in refined:
                pidx = local_by_leaf[parent]
                refined[parent] = apply_hex_refine_transfer(
                    refine, old_state[:, :, pidx:pidx + 1]
                )
            result[:, :, i] = refined[parent][path[-1], :, :, 0]
            continue

        children = hex_octree_children(leaf)
        if children <= old_leaves:
            if leaf not in restricted:
                cstate = np.stack([
                    old_by_leaf[(root, path + (o,))]
                    for o in range(8)
                ])[:, :, :, None]
                restricted[leaf] = apply_hex_restrict_transfer(
                    restrict, cstate
                )[:, :, 0]
            result[:, :, i] = restricted[leaf]
            continue

        raise AMRTransactionError(
            f'D6B has no accepted transfer source for proposed leaf {leaf!r}'
        )

    result.setflags(write=False)
    return result, basis


def _scratch_rhs(system, t, accepted_bank, scratch_bank):
    if scratch_bank == accepted_bank:
        raise AMRTransactionError(
            'D6A scratch RHS requires distinct input and output banks'
        )

    before = _copy_state(system, accepted_bank)
    system.rhs(t, accepted_bank, scratch_bank)
    system.backend.wait()
    rhs = _copy_state(system, scratch_bank)
    if not all(np.isfinite(a).all() for a in rhs):
        raise AMRTransactionError('D6A first RHS is not finite')

    after = _copy_state(system, accepted_bank)
    drift = _max_abs_diff(before, after)
    if not all(np.array_equal(a, b) for a, b in zip(before, after)):
        raise AMRTransactionError(
            f'D6A accepted bank changed during scratch RHS by {drift:.3e}'
        )
    return rhs, drift


def _cleanup_stage(reader, stage_dir):
    try:
        reader.close()
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def _discard_stage(stage_system, reader, stage_dir):
    if stage_system is not None:
        finalizer = getattr(stage_system, '_amr_stage_finalizer', None)
        if finalizer is not None:
            finalizer()
            return
    if reader is not None:
        _cleanup_stage(reader, stage_dir)
    elif stage_dir:
        shutil.rmtree(stage_dir, ignore_errors=True)


def _validate_staged_hex_transfer(
    stage_system, staged_state, old_state, old_volumes, proposed_volumes,
    old_weights
):
    before = _conserved_totals(old_state, old_volumes, old_weights)
    after = _conserved_totals(staged_state, proposed_volumes, old_weights)
    error = after - before
    tol = 8192*np.finfo(old_state.dtype).eps
    scale = np.maximum(1.0, np.abs(before))
    if np.any(np.abs(error) > tol*scale):
        raise AMRTransactionError(
            'D6A componentwise conservative transfer gate failed'
        )
    if not np.isclose(
        np.sum(old_volumes), np.sum(proposed_volumes), rtol=0,
        atol=tol*max(1.0, np.sum(old_volumes))
    ):
        raise AMRTransactionError('D6A physical volume gate failed')

    rho_range, pressure_range = _eos_ranges(stage_system, staged_state)
    return before, after, error, rho_range, pressure_range


def _validate_staged_hex_rhs(
    stage_system, system, intg, bank, scratch_bank, staged_state, transferred
):
    stage_system.preproc(intg.tcurr, bank)
    intg.backend.wait()
    staged_after_preproc = _copy_state(stage_system, bank)[0]
    if not np.array_equal(staged_after_preproc, staged_state):
        raise AMRTransactionError(
            'D6A staged accepted bank changed during preprocessing'
        )

    if (stage_system.nrhs != system.nrhs or
            len(stage_system.ele_banks) != len(system.ele_banks) or
            any(len(a) != len(b) for a, b in zip(
                stage_system.ele_banks, system.ele_banks))):
        raise AMRTransactionError(
            'D6A staged register-bank layout changed'
        )

    rhs, bank_drift = _scratch_rhs(
        stage_system, intg.tcurr, bank, scratch_bank
    )
    staged_after_rhs = _copy_state(stage_system, bank)[0]
    if not np.array_equal(staged_after_rhs, transferred):
        raise AMRTransactionError(
            'D6A staged accepted bank changed after scratch RHS'
        )

    return rhs, bank_drift, sum(stage_system.ele_ndofs)


def _propose_scripted_hex_tree(root_mesh, mesh, scripted_marks, action):
    if action not in {'refine', 'coarsen'}:
        raise AMRTransactionError(
            f'Unknown D6B transaction action: {action!r}'
        )

    marks = tuple(_normalise_mark(m) for m in scripted_marks)
    if action == 'refine':
        if len(marks) != 1 or len(set(marks)) != 1:
            raise AMRTransactionError(
                'D6B refinement requires exactly one unique scripted mark'
            )
    elif len(marks) != 8 or len(set(marks)) != 8:
        raise AMRTransactionError(
            'D6B coarsening requires exactly eight unique scripted marks'
        )

    old_tree, local_by_leaf = _current_tree(mesh)
    active = set(old_tree.leaves())
    if not set(marks) <= active:
        missing = sorted(set(marks) - active, key=repr)
        raise AMRTransactionError(
            f'Scripted marks are not active leaves: {missing!r}'
        )

    if action == 'refine':
        proposed_tree = _close_refinement(root_mesh, old_tree, marks[0])
    else:
        proposed_tree = _close_coarsening(root_mesh, old_tree, marks)

    return marks, old_tree, local_by_leaf, proposed_tree


def perform_one_amr_transaction(
    intg, scripted_marks, *, action='refine', restart_root_mesh=None
):
    system, mesh, root_mesh = _validate_integrator(
        intg, restart_root_mesh
    )
    marks, old_tree, local_by_leaf, proposed_tree = (
        _propose_scripted_hex_tree(root_mesh, mesh, scripted_marks, action)
    )

    bank = intg.idxcurr
    old_shape, _ = _single_hex_bank(system, bank, 'accepted')
    scratch_bank = _scratch_bank(system, bank)
    old_parts = _copy_state(system, bank)
    if len(old_parts) != 1:
        raise AMRTransactionError('D6A requires one uniform Hex solution bank')
    old_state = old_parts[0]
    if tuple(old_state.shape) != old_shape:
        raise AMRTransactionError(
            'D6A accepted state does not match its Hex bank shape'
        )
    if old_shape[2] != len(local_by_leaf):
        raise AMRTransactionError(
            'D6A Hex leaf identity does not map to the runtime solution bank'
        )

    old_rho, old_pressure = _eos_ranges(system, old_state)
    transferred, basis = _transfer_state(
        system, old_tree, proposed_tree, old_state, local_by_leaf
    )
    if not np.isfinite(transferred).all():
        raise AMRTransactionError(
            'D6A transferred conservative state is not finite'
        )

    old_volumes = _mesh_volumes(mesh)
    old_weights = _integration_weights(intg.cfg, basis)
    raw = materialize_native_hex_tree(root_mesh, proposed_tree)
    if tuple(raw.leaf_order) != tuple(proposed_tree.leaves()):
        raise AMRTransactionError(
            'D6A materialized leaf order differs from the proposed tree'
        )
    proposed_volumes = _raw_volumes(raw)

    stage_dir = tempfile.mkdtemp(prefix='pyfr-d6a-')
    stage_path = os.path.join(stage_dir, 'proposed.pyfrm')
    reader = None
    stage_system = None
    try:
        write_adapted_mesh(raw, stage_path)
        reader = NativeReader(stage_path)
        stage_mesh = reader.mesh
        _stage_hex_native_leaf_order(stage_mesh, raw)

        staged = Solution(
            config=intg.cfg, stats=None, fields=None,
            data={'hex': np.array(transferred, copy=True)}, state={}
        )
        stage_serialiser = Serialiser()
        stage_system = type(system)(
            intg.backend, stage_mesh, staged, intg._registers,
            intg.cfg, stage_serialiser, needs_cfl=False
        )
        stage_system.commit()
        staged_state, stage_leaf_to_bank, stage_accepted_shape = (
            _inject_staged_hex_bank(
                stage_system, stage_mesh, raw, transferred, bank
            )
        )

        # Validate after transfer crosses the native/PyFR bank boundary.
        before, after, error, staged_rho, staged_pressure = (
            _validate_staged_hex_transfer(
                stage_system, staged_state, old_state, old_volumes,
                proposed_volumes, old_weights
            )
        )
        rhs, bank_drift, staged_gndofs = _validate_staged_hex_rhs(
            stage_system, system, intg, bank, scratch_bank, staged_state,
            transferred
        )
        finalizer = weakref.finalize(
            stage_system, _cleanup_stage, reader, stage_dir
        )
        stage_system._amr_stage_finalizer = finalizer
    except Exception:
        _discard_stage(stage_system, reader, stage_dir)
        raise

    # Commit point; all fallible proposal and validation work is above.
    clear_memoize(intg)
    intg.system = stage_system
    intg._amr_root_mesh = root_mesh
    intg.mesh_uuid = stage_mesh.uuid
    intg.serialiser = stage_serialiser
    intg.gndofs = staged_gndofs
    intg._invalidate_caches()

    return AMRTransactionResult(
        action=action, old_tree=old_tree, proposed_tree=proposed_tree,
        raw_mesh=raw,
        transferred_state=transferred, rhs=tuple(rhs),
        conservation_before=before, conservation_after=after,
        conservation_error=error, old_volume=float(np.sum(old_volumes)),
        proposed_volume=float(np.sum(proposed_volumes)),
        old_rho_range=old_rho, old_pressure_range=old_pressure,
        rho_range=staged_rho, pressure_range=staged_pressure,
        stage_mesh_uuid=stage_mesh.uuid,
        stage_leaf_count=len(raw.leaf_order),
        stage_mortar_count=len(raw.mortars),
        stage_mortar_formats=tuple(
            (m.format, m.template) for m in raw.mortars
        ),
        bank_drift=bank_drift,
        old_system_id=id(system), new_system_id=id(stage_system),
        accepted_bank=bank, scratch_bank=scratch_bank,
        stage_leaf_to_bank=stage_leaf_to_bank,
        stage_accepted_shape=stage_accepted_shape,
        tcurr=float(intg.tcurr)
    )


def _quad_online_settings(cfg):
    section = 'solver-amr'
    if section not in cfg.sections():
        raise AMRTransactionError('ONLINE2D-1 requires [solver-amr]')
    if not cfg.hasopt(section, 'refine-threshold'):
        raise AMRTransactionError(
            'ONLINE2D-1 requires solver-amr refine-threshold'
        )

    indicator = cfg.get(section, 'indicator', 'density-velocity-variation')
    if indicator != 'density-velocity-variation':
        raise AMRTransactionError(
            'ONLINE2D-1 requires the accepted D9Q composite indicator'
        )

    threshold = cfg.getfloat(section, 'refine-threshold')
    if not np.isfinite(threshold) or threshold < 0:
        raise AMRTransactionError(
            'ONLINE2D-1 refine threshold must be finite and nonnegative'
        )

    max_level = cfg.getint(section, 'max-level', 2)
    if not 1 <= max_level <= 2:
        raise AMRTransactionError(
            'ONLINE2D-1 maximum level must be one or two'
        )

    wall_min_level = cfg.getint(section, 'wall-min-level', 0)
    if not 0 <= wall_min_level <= max_level:
        raise AMRTransactionError(
            'ONLINE2D-1 wall minimum level must be in 0..max-level'
        )

    wall_text = cfg.get(section, 'wall-boundaries', '')
    walls = tuple(v for v in wall_text.replace(',', ' ').split() if v)
    if len(set(walls)) != len(walls):
        raise AMRTransactionError(
            'ONLINE2D-1 wall boundary names must be unique'
        )

    density_floor = cfg.getfloat(section, 'density-floor', 1e-14)
    acoustic_floor = cfg.getfloat(section, 'acoustic-floor', 1e-14)
    if not np.isfinite(density_floor) or density_floor <= 0:
        raise AMRTransactionError(
            'ONLINE2D-1 density floor must be finite and positive'
        )
    if not np.isfinite(acoustic_floor) or acoustic_floor <= 0:
        raise AMRTransactionError(
            'ONLINE2D-1 acoustic floor must be finite and positive'
        )

    return (
        threshold, max_level, walls, wall_min_level,
        density_floor, acoustic_floor,
    )


def _quad_online_writer_plugins(intg, *, require_checkpoint_schedule=False):
    from pyfr.plugins.soln.writer import WriterPlugin

    cfg = intg.cfg
    plugin_sections = [
        s for s in cfg.sections()
        if s.startswith(('soln-plugin-', 'solver-plugin-', 'trigger-'))
    ]
    if plugin_sections and plugin_sections != ['soln-plugin-writer']:
        raise AMRTransactionError(
            'ONLINE2D writer lifecycle supports one soln-plugin-writer only'
        )

    plugins = tuple(getattr(intg, 'plugins', ()))
    if len(plugins) != len(plugin_sections) or any(
        type(p) is not WriterPlugin for p in plugins
    ):
        raise AMRTransactionError(
            'ONLINE2D live plugins do not match the writer-only scope'
        )
    if not plugins:
        return plugins

    if not cfg.hasopt('solver-amr', 'checkpoint-dir'):
        raise AMRTransactionError(
            'ONLINE2D writer lifecycle requires solver-amr checkpoint-dir'
        )
    if require_checkpoint_schedule:
        schedule = getattr(intg, 'amr_schedule', None)
        if (schedule is None or
                getattr(schedule, 'checkpoint_dir', None) is None):
            raise AMRTransactionError(
                'ONLINE2D writer lifecycle requires checkpointed scheduling'
            )

    for plugin in plugins:
        if (plugin.trigger or plugin.trigger_write_name or
                plugin.trigger_fire_name):
            raise AMRTransactionError(
                'ONLINE2D writer lifecycle does not support triggers'
            )
        if plugin.postact:
            raise AMRTransactionError(
                'ONLINE2D writer lifecycle does not support post-actions'
            )
        if plugin._write_grads:
            raise AMRTransactionError(
                'ONLINE2D writer lifecycle does not support gradient output'
            )
        if cfg.get(plugin.cfgsect, 'region', '*') != '*':
            raise AMRTransactionError(
                'ONLINE2D writer lifecycle requires full-domain output'
            )

    return plugins


def _validate_quad_online_integrator(intg, restart_root_mesh=None):
    if getattr(intg, 'formulation', None) != 'explicit':
        raise AMRTransactionError('ONLINE2D-1 requires explicit formulation')
    if getattr(intg, 'controller_name', None) != 'none':
        raise AMRTransactionError('ONLINE2D-1 requires controller none')
    if getattr(intg, 'stepper_name', None) != 'rk4':
        raise AMRTransactionError('ONLINE2D-1 requires explicit RK4')
    backend_name = getattr(getattr(intg, 'backend', None), 'name', None)
    if backend_name not in {'openmp', 'cuda'}:
        raise AMRTransactionError(
            'ONLINE2D-1 currently supports OpenMP and CUDA only'
        )
    if getattr(intg, 'nacptsteps', 0) < 1:
        raise AMRTransactionError(
            'ONLINE2D-1 requires a completed accepted physical timestep'
        )
    if getattr(intg, 'stepinfo', None) != []:
        raise AMRTransactionError(
            'ONLINE2D-1 requires the post-advance accepted-step safe point'
        )

    cfg = intg.cfg
    solver_system = cfg.get('solver', 'system')
    if solver_system not in {'euler', 'navier-stokes'}:
        raise AMRTransactionError(
            'ONLINE2D-1 supports Euler and Navier-Stokes only'
        )
    if (solver_system == 'navier-stokes' and
            cfg.get('solver', 'viscosity-correction', 'none') != 'none'):
        raise AMRTransactionError(
            'ONLINE2D-1 Navier-Stokes requires constant viscosity'
        )
    if (cfg.get('solver-time-integrator', 'formulation', 'explicit') !=
            'explicit'):
        raise AMRTransactionError(
            'ONLINE2D-1 requires an explicit time integrator'
        )
    if cfg.get('solver-time-integrator', 'controller', 'none') != 'none':
        raise AMRTransactionError('ONLINE2D-1 requires controller = none')
    if cfg.get('solver-time-integrator', 'scheme', 'rk4') != 'rk4':
        raise AMRTransactionError('ONLINE2D-1 requires scheme = rk4')
    if cfg.get('solver', 'shock-capturing', 'none') != 'none':
        raise AMRTransactionError('ONLINE2D-1 requires shock-capturing = none')
    mortar_impl = cfg.get(
        'solver-interfaces', 'mortar-implementation', 'fused'
    )
    if mortar_impl != 'staged':
        raise AMRTransactionError(
            'ONLINE2D-1 requires staged mortar execution'
        )
    if any(s.startswith('solver-order-') for s in cfg.sections()):
        raise AMRTransactionError('ONLINE2D-1 does not support mixed-p')

    _quad_online_writer_plugins(intg)
    if getattr(intg, 'triggers', None):
        raise AMRTransactionError('ONLINE2D-1 excludes live triggers')
    if getattr(getattr(intg, 'serialiser', None), '_serialfns', {}):
        raise AMRTransactionError(
            'ONLINE2D-1 excludes unsupported serialised mutable state'
        )

    comm, _, _ = get_comm_rank_root()
    if comm.size != 1:
        raise AMRTransactionError('ONLINE2D-1 requires one MPI rank')

    system = getattr(intg, 'system', None)
    mesh = getattr(system, 'mesh', None)
    if system is None or mesh is None:
        raise AMRTransactionError(
            'ONLINE2D-1 requires a live PyFR system and mesh'
        )
    if getattr(system, 'name', None) != solver_system:
        raise AMRTransactionError(
            'ONLINE2D-1 live system does not match solver configuration'
        )
    if getattr(system, 'ndims', None) != 2:
        raise AMRTransactionError('ONLINE2D-1 requires a 2D system')
    if mesh.etypes != ['quad']:
        raise AMRTransactionError('ONLINE2D-1 requires a pure Quad mesh')
    if mesh.con_p:
        raise AMRTransactionError('ONLINE2D-1 does not support MPI topology')
    if mesh.amr_tree is None:
        if mesh.mcon:
            raise AMRTransactionError(
                'ONLINE2D-1 root mesh must not contain pre-existing mortars'
            )
        root_mesh = mesh
    else:
        if not isinstance(mesh.amr_tree, QuadLeafTree):
            raise AMRTransactionError(
                'ONLINE2D-1 adapted mesh requires typed Quad ancestry'
            )
        for mcon in mesh.mcon.values():
            if (mcon.format != 'one-to-many-v1' or
                    mcon.template != 'line-1x2' or mcon.nright != 2):
                raise AMRTransactionError(
                    'ONLINE2D-1 adapted mesh contains an unsupported mortar'
                )

        live_root_mesh = getattr(intg, '_amr_root_mesh', None)
        if live_root_mesh is not None and restart_root_mesh is not None:
            if live_root_mesh.uuid != restart_root_mesh.uuid:
                raise AMRTransactionError(
                    'ONLINE2D-1 supplied root mesh differs from the live '
                    'root anchor'
                )

        root_mesh = live_root_mesh or restart_root_mesh
        if root_mesh is None:
            raise AMRTransactionError(
                'ONLINE2D-1 adapted live mesh requires its root anchor'
            )
        if getattr(root_mesh, 'amr_tree', None) is not None:
            raise AMRTransactionError(
                'ONLINE2D-1 root anchor must be an unadapted native mesh'
            )
        if root_mesh.uuid != mesh.amr_tree.root_mesh_uuid:
            raise AMRTransactionError(
                'ONLINE2D-1 root anchor does not match persisted ancestry'
            )
        if root_mesh.etypes != ['quad'] or root_mesh.con_p or root_mesh.mcon:
            raise AMRTransactionError(
                'ONLINE2D-1 root anchor is outside the pure-Quad scope'
            )
        if np.any(root_mesh.spts_curved.get('quad', ())):
            raise AMRTransactionError(
                'ONLINE2D-1 root anchor requires affine Quad geometry'
            )

    if np.any(mesh.spts_curved.get('quad', ())):
        raise AMRTransactionError('ONLINE2D-1 requires affine Quad geometry')

    return system, mesh, root_mesh


def _single_quad_bank(system, bank, label):
    if list(getattr(system, 'ele_types', ())) != ['quad']:
        raise AMRTransactionError(
            f'ONLINE2D-1 {label} system requires one Quad element group'
        )
    if len(getattr(system, 'ele_banks', ())) != 1:
        raise AMRTransactionError(
            f'ONLINE2D-1 {label} system requires one solution bank group'
        )
    if bank < 0 or bank >= getattr(system, 'nrhs', 0):
        raise AMRTransactionError(
            f'ONLINE2D-1 {label} solution bank is invalid'
        )

    try:
        group_bank = system.ele_banks[0][bank]
        group_shape = tuple(system.ele_shapes['quad'])
        bank_shape = tuple(group_bank.ioshape)
    except (AttributeError, IndexError, KeyError, TypeError):
        raise AMRTransactionError(
            f'ONLINE2D-1 {label} Quad bank layout is unavailable'
        ) from None

    if len(group_shape) != 3 or any(n <= 0 for n in group_shape):
        raise AMRTransactionError(
            f'ONLINE2D-1 {label} Quad group has an invalid logical shape'
        )
    if bank_shape != group_shape:
        raise AMRTransactionError(
            f'ONLINE2D-1 {label} Quad bank shape does not match the group'
        )
    return group_shape, group_bank


def _inject_staged_quad_bank(stage_system, stage_mesh, transferred, bank):
    shape, group_bank = _single_quad_bank(stage_system, bank, 'staged')
    leaves = tuple(stage_mesh.amr_tree.leaves())
    eidxs = np.asarray(stage_mesh.eidxs.get('quad', ()), dtype=np.int64)
    if not np.array_equal(eidxs, np.arange(len(leaves), dtype=np.int64)):
        raise AMRTransactionError(
            'ONLINE2D-1 staged Quad eidxs are not canonical leaf ordinals'
        )
    if tuple(transferred.shape) != shape or shape[2] != len(leaves):
        raise AMRTransactionError(
            'ONLINE2D-1 transferred state does not match staged Quad bank'
        )

    group_bank.set(np.array(transferred, copy=True, order='C'))
    stage_system.backend.wait()
    parts = _copy_state(stage_system, bank)
    if len(parts) != 1 or tuple(parts[0].shape) != shape:
        raise AMRTransactionError(
            'ONLINE2D-1 staged state readback has an invalid shape'
        )
    if not np.array_equal(parts[0], transferred):
        raise AMRTransactionError(
            'ONLINE2D-1 staged accepted bank differs from transfer'
        )
    return parts[0], shape


def _quad_wall_roots(root_mesh, wall_boundaries, l2g):
    roots = set()
    for name in wall_boundaries:
        try:
            bcon = root_mesh.bcon[name]
        except KeyError as exc:
            raise AMRTransactionError(
                f'ONLINE2D-1 root mesh has no boundary {name!r}'
            ) from exc

        for etype, _, eidxs in bcon.items():
            if etype != 'quad':
                raise AMRTransactionError(
                    'ONLINE2D-1 wall floor requires Quad boundaries'
                )
            roots.update(l2g[int(eidx)] for eidx in eidxs)
    return roots


def _validate_staged_quad_mesh(stage_mesh, raw, new_leaves):
    if tuple(stage_mesh.amr_tree.leaves()) != new_leaves:
        raise AMRTransactionError(
            'ONLINE2D-1 staged native ancestry changed leaf order'
        )
    if set(stage_mesh.mcon) - {'line-1x2'}:
        raise AMRTransactionError(
            'ONLINE2D-1 staged mesh contains an unsupported mortar'
        )
    mcon = stage_mesh.mcon.get('line-1x2', ())
    if len(mcon) != len(raw.mortars):
        raise AMRTransactionError(
            'ONLINE2D-1 staged Line-1x2 mortar count changed'
        )
    for rec, expected in zip(getattr(mcon, 'records', ()), raw.mortars):
        if (int(rec['left_eidx']) != expected.left_eidx or
                tuple(map(int, rec['right_eidx'])) != expected.right_eidx):
            raise AMRTransactionError(
                'ONLINE2D-1 staged Line-1x2 low/high ordering changed'
            )


def _materialize_staged_quad_mesh(root_mesh, proposed_tree, new_leaves):
    try:
        raw = materialize_native_quad_tree(
            root_mesh, proposed_tree, comm_size=1
        )
        stage_mesh = build_adapted_quad_mesh(raw)
    except Exception as exc:
        raise AMRTransactionError(
            f'ONLINE2D-1 native materialization failed: {exc}'
        ) from exc

    _validate_staged_quad_mesh(stage_mesh, raw, new_leaves)
    return raw, stage_mesh


def _prepare_quad_transfer(
    root_mesh, old_leaves, new_leaves, old_state, cfg, fields, gamma
):
    from pyfr.amroffline import (
        _global_integral, _prolongate_to_tree, _validate_state
    )

    basis = QuadShape(None, cfg)
    if basis.nupts != old_state.shape[0]:
        raise AMRTransactionError(
            'ONLINE2D-1 solution points do not match the Quad basis'
        )

    transferred = _prolongate_to_tree(
        old_state, old_leaves, new_leaves, basis
    )
    _validate_state(transferred, fields, gamma)

    before = _global_integral(root_mesh, old_leaves, old_state, basis)
    proposed = _global_integral(root_mesh, new_leaves, transferred, basis)
    error = proposed - before
    scale = max(1.0, float(np.max(np.abs(before), initial=0.0)))
    tol = 65536*np.finfo(np.result_type(old_state, float)).eps*scale
    if float(np.max(np.abs(error), initial=0.0)) > tol:
        raise AMRTransactionError(
            'ONLINE2D-1 conservative prolongation gate failed'
        )

    return basis, transferred, before, tol


def _validate_staged_quad_rhs(
    stage_system, system, intg, bank, scratch_bank, transferred, staged_state
):
    stage_system.preproc(intg.tcurr, bank)
    intg.backend.wait()
    after_preproc = _copy_state(stage_system, bank)
    if (len(after_preproc) != 1 or
            not np.array_equal(after_preproc[0], staged_state)):
        raise AMRTransactionError(
            'ONLINE2D-1 staged bank changed during preprocessing'
        )

    if (stage_system.nrhs != system.nrhs or
            len(stage_system.ele_banks) != len(system.ele_banks) or
            any(len(a) != len(b) for a, b in zip(
                stage_system.ele_banks, system.ele_banks
            ))):
        raise AMRTransactionError(
            'ONLINE2D-1 staged register-bank layout changed'
        )

    rhs, bank_drift = _scratch_rhs(
        stage_system, intg.tcurr, bank, scratch_bank
    )
    after_rhs = _copy_state(stage_system, bank)
    if (len(after_rhs) != 1 or
            not np.array_equal(after_rhs[0], transferred)):
        raise AMRTransactionError(
            'ONLINE2D-1 staged accepted bank changed after first RHS'
        )

    return rhs, bank_drift, sum(stage_system.ele_ndofs)


def _build_staged_quad_system(
    intg, system, stage_mesh, transferred, bank, writer_plugins
):
    staged = Solution(
        config=intg.cfg, stats=None, fields=None,
        data={'quad': np.array(transferred, copy=True)}, state={}
    )
    stage_serialiser = Serialiser()
    stage_system = type(system)(
        intg.backend, stage_mesh, staged, intg._registers,
        intg.cfg, stage_serialiser, needs_cfl=False
    )

    # WriterPlugin needs the construction-time element map, matching the
    # ordinary integrator lifecycle where plugins precede system.commit().
    staged_plugins = [
        plugin.prepare_amr_rebind(intg, stage_system)
        for plugin in writer_plugins
    ]
    stage_system.commit()
    staged_state, stage_shape = _inject_staged_quad_bank(
        stage_system, stage_mesh, transferred, bank
    )
    return (
        stage_system, stage_serialiser, staged_plugins, staged_state,
        stage_shape
    )


def _validate_staged_quad_transfer(
    root_mesh, new_leaves, stage_system, staged_state, basis, before, tol
):
    from pyfr.amroffline import _global_integral

    after = _global_integral(root_mesh, new_leaves, staged_state, basis)
    error = after - before
    if float(np.max(np.abs(error), initial=0.0)) > tol:
        raise AMRTransactionError(
            'ONLINE2D-1 staged conservative transfer gate failed'
        )
    rho_range, pressure_range = _eos_ranges(stage_system, staged_state)
    return after, error, rho_range, pressure_range


def _prepare_quad_state(intg, system, old_leaves, bank):
    from pyfr.amroffline import _validate_state

    old_shape, _ = _single_quad_bank(system, bank, 'accepted')
    scratch_bank = _scratch_bank(system, bank)
    parts = _copy_state(system, bank)
    if len(parts) != 1 or tuple(parts[0].shape) != old_shape:
        raise AMRTransactionError(
            'ONLINE2D-1 accepted Quad bank readback is invalid'
        )

    old_state = parts[0]
    if old_shape[2] != len(old_leaves):
        raise AMRTransactionError(
            'ONLINE2D-1 Quad leaves do not map to the accepted bank'
        )

    fields = tuple(system.elementscls.convars(system.ndims, intg.cfg))
    gamma = intg.cfg.getfloat('constants', 'gamma')
    state_indices = _validate_state(old_state, fields, gamma)
    rho_range, pressure_range = _eos_ranges(system, old_state)
    return (
        scratch_bank, old_state, fields, gamma, state_indices, rho_range,
        pressure_range
    )


def perform_indicator_quad_amr_transaction(
    intg, *, restart_root_mesh=None
):
    from pyfr.amrindicator import density_velocity_variation_scores
    from pyfr.amroffline import (
        _close_2to1, _current_column_leaves, _split_nodes,
        _wall_floor_splits,
    )

    system, mesh, root_mesh = _validate_quad_online_integrator(
        intg, restart_root_mesh
    )
    writer_plugins = _quad_online_writer_plugins(
        intg, require_checkpoint_schedule=True
    )
    (
        refine_threshold, max_level, wall_boundaries, wall_min_level,
        density_floor, acoustic_floor,
    ) = _quad_online_settings(intg.cfg)

    l2g, pids_by_g, _ = _quad_local_maps(root_mesh)
    roots = tuple(sorted(pids_by_g))
    rootfaces, rootkinds = _derive_quad_root_face_topology(
        root_mesh, pids_by_g, l2g
    )
    old_leaves = _current_column_leaves(mesh, roots)
    old_tree = encode_quad_leaf_tree(root_mesh.uuid, old_leaves)
    if tuple(old_tree.leaves()) != tuple(old_leaves):
        raise AMRTransactionError(
            'ONLINE2D-1 live Quad bank order is not canonical leaf order'
        )

    bank = intg.idxcurr
    (
        scratch_bank, old_state, fields, gamma, state_indices, old_rho,
        old_pressure
    ) = _prepare_quad_state(intg, system, old_leaves, bank)
    irho, irhou, irhov, ienergy = state_indices
    scores = density_velocity_variation_scores(
        old_state, density_index=irho,
        momentum_indices=(irhou, irhov), energy_index=ienergy,
        gamma=gamma, density_floor=density_floor,
        acoustic_floor=acoustic_floor,
    )
    score_items = tuple(zip(old_leaves, map(float, scores)))
    marks = {
        leaf for leaf, score in score_items
        if len(leaf[1]) < max_level and score >= refine_threshold
    }

    wall_roots = _quad_wall_roots(root_mesh, wall_boundaries, l2g)
    existing = _split_nodes(old_leaves)
    wall_floor = _wall_floor_splits(wall_roots, wall_min_level)
    wall_splits = tuple(sorted(wall_floor - existing - marks))
    split = existing | marks | wall_floor
    try:
        new_leaves, closure = _close_2to1(
            roots, split, rootfaces, rootkinds, max_level
        )
    except Exception as exc:
        if isinstance(exc, AMRTransactionError):
            raise
        raise AMRTransactionError(str(exc)) from exc
    new_leaves = tuple(new_leaves)
    marks = tuple(sorted(marks))
    trigger = max(
        (score for leaf, score in score_items if leaf in set(marks)),
        default=None,
    )

    if new_leaves == tuple(old_leaves):
        return QuadIndicatorAMRResult(
            QuadAMRDecision('none', marks, trigger), score_items, None
        )

    proposed_tree = encode_quad_leaf_tree(root_mesh.uuid, new_leaves)
    basis, transferred, before, tol = _prepare_quad_transfer(
        root_mesh, old_leaves, new_leaves, old_state, intg.cfg, fields, gamma
    )

    raw, stage_mesh = _materialize_staged_quad_mesh(
        root_mesh, proposed_tree, new_leaves
    )

    stage_system = None
    try:
        (
            stage_system, stage_serialiser, staged_plugins, staged_state,
            stage_shape
        ) = _build_staged_quad_system(
            intg, system, stage_mesh, transferred, bank, writer_plugins
        )

        after, error, staged_rho, staged_pressure = (
            _validate_staged_quad_transfer(
                root_mesh, new_leaves, stage_system, staged_state, basis,
                before, tol
            )
        )

        rhs, bank_drift, staged_gndofs = _validate_staged_quad_rhs(
            stage_system, system, intg, bank, scratch_bank, transferred,
            staged_state
        )
    except Exception:
        stage_system = None
        raise

    # COMMIT.  All fallible proposal and validation operations are above.
    clear_memoize(intg)
    old_system_id = id(system)
    intg.system = stage_system
    intg._amr_root_mesh = root_mesh
    intg.mesh_uuid = stage_mesh.uuid
    intg.serialiser = stage_serialiser
    intg.gndofs = staged_gndofs
    if writer_plugins:
        intg.plugins = staged_plugins
    intg._invalidate_caches()

    tx = QuadAMRTransactionResult(
        old_tree=old_tree, proposed_tree=proposed_tree, raw_mesh=raw,
        transferred_state=transferred, rhs=tuple(rhs),
        refine_marks=marks, wall_splits=wall_splits,
        closure_splits=tuple(closure), conservation_before=before,
        conservation_after=after, conservation_error=error,
        old_rho_range=old_rho, old_pressure_range=old_pressure,
        rho_range=staged_rho, pressure_range=staged_pressure,
        stage_mesh_uuid=stage_mesh.uuid,
        stage_leaf_count=len(raw.leaf_order),
        stage_mortar_count=len(raw.mortars),
        stage_mortar_formats=tuple(
            (m.format, m.template) for m in raw.mortars
        ),
        bank_drift=bank_drift, old_system_id=old_system_id,
        new_system_id=id(stage_system), accepted_bank=bank,
        scratch_bank=scratch_bank, stage_accepted_shape=stage_shape,
        tcurr=float(intg.tcurr),
    )
    return QuadIndicatorAMRResult(
        QuadAMRDecision('refine', marks, trigger), score_items, tx
    )




def _validate_mixed_quad_online_integrator(intg, restart_root_mesh=None):
    if getattr(intg, 'formulation', None) != 'explicit':
        raise AMRTransactionError('MIX2D1 requires explicit formulation')
    if getattr(intg, 'controller_name', None) != 'none':
        raise AMRTransactionError('MIX2D1 requires controller none')
    if getattr(intg, 'stepper_name', None) != 'rk4':
        raise AMRTransactionError('MIX2D1 requires explicit RK4')
    if getattr(getattr(intg, 'backend', None), 'name', None) != 'openmp':
        raise AMRTransactionError('MIX2D1 currently requires OpenMP')
    if (getattr(intg, 'nacptsteps', 0) < 1 or
            getattr(intg, 'stepinfo', None) != []):
        raise AMRTransactionError(
            'MIX2D1 requires a completed post-advance accepted timestep'
        )

    cfg = intg.cfg
    solver_system = cfg.get('solver', 'system')
    if solver_system not in {'euler', 'navier-stokes'}:
        raise AMRTransactionError('MIX2D1 supports Euler/Navier-Stokes only')
    if (solver_system == 'navier-stokes' and
            cfg.get('solver', 'viscosity-correction', 'none') != 'none'):
        raise AMRTransactionError(
            'MIX2D1 Navier-Stokes requires constant viscosity'
        )
    if (cfg.get('solver-time-integrator', 'formulation', 'explicit') !=
            'explicit'):
        raise AMRTransactionError('MIX2D1 requires explicit time integration')
    if cfg.get('solver-time-integrator', 'controller', 'none') != 'none':
        raise AMRTransactionError('MIX2D1 requires controller = none')
    if cfg.get('solver-time-integrator', 'scheme', 'rk4') != 'rk4':
        raise AMRTransactionError('MIX2D1 requires scheme = rk4')
    if cfg.get('solver', 'shock-capturing', 'none') != 'none':
        raise AMRTransactionError('MIX2D1 requires shock-capturing = none')
    if (cfg.get('solver-interfaces', 'mortar-implementation', 'fused') !=
            'staged'):
        raise AMRTransactionError('MIX2D1 requires staged mortars')
    if any(s.startswith('solver-order-') for s in cfg.sections()):
        raise AMRTransactionError('MIX2D1 does not implement mixed-p')

    plugin_sections = [
        s for s in cfg.sections()
        if s.startswith(('soln-plugin-', 'solver-plugin-', 'trigger-'))
    ]
    if (plugin_sections or getattr(intg, 'plugins', None) or
            getattr(intg, 'triggers', None)):
        raise AMRTransactionError(
            'MIX2D1 first seam excludes plugins/triggers'
        )
    if getattr(getattr(intg, 'serialiser', None), '_serialfns', {}):
        raise AMRTransactionError(
            'MIX2D1 excludes unsupported serialised mutable state'
        )

    comm, _, _ = get_comm_rank_root()
    if comm.size != 1:
        raise AMRTransactionError('MIX2D1 requires one MPI rank')

    system = getattr(intg, 'system', None)
    mesh = getattr(system, 'mesh', None)
    if system is None or mesh is None:
        raise AMRTransactionError('MIX2D1 requires a live PyFR system')
    if getattr(system, 'name', None) != solver_system or system.ndims != 2:
        raise AMRTransactionError('MIX2D1 live system/configuration mismatch')
    if set(mesh.etypes) != {'tri', 'quad'} or mesh.con_p:
        raise AMRTransactionError(
            'MIX2D1 requires one-rank mixed Tri+Quad topology'
        )

    if mesh.amr_tree is None:
        if mesh.mcon:
            raise AMRTransactionError(
                'MIX2D1 root mesh must not contain pre-existing mortars'
            )
        root_mesh = mesh
    else:
        if not isinstance(mesh.amr_tree, QuadLeafTree):
            raise AMRTransactionError(
                'MIX2D1 adapted mesh requires typed Quad ancestry'
            )
        for mcon in mesh.mcon.values():
            if (mcon.format != 'one-to-many-v1' or
                    mcon.template != 'line-1x2' or mcon.nright != 2):
                raise AMRTransactionError(
                    'MIX2D1 adapted mesh contains an unsupported mortar'
                )

        live_root_mesh = getattr(intg, '_amr_root_mesh', None)
        if live_root_mesh is not None and restart_root_mesh is not None:
            if live_root_mesh.uuid != restart_root_mesh.uuid:
                raise AMRTransactionError(
                    'MIX2D1 supplied root differs from live root anchor'
                )
        root_mesh = live_root_mesh or restart_root_mesh
        if root_mesh is None:
            raise AMRTransactionError(
                'MIX2D1 adapted live mesh requires immutable root anchor'
            )
        if getattr(root_mesh, 'amr_tree', None) is not None:
            raise AMRTransactionError('MIX2D1 root anchor must be unadapted')
        if root_mesh.uuid != mesh.amr_tree.root_mesh_uuid:
            raise AMRTransactionError('MIX2D1 root anchor UUID mismatch')

    if (set(root_mesh.etypes) != {'tri', 'quad'} or root_mesh.con_p or
            root_mesh.mcon):
        raise AMRTransactionError(
            'MIX2D1 root anchor is outside Tri+Quad scope'
        )
    if any(
        np.any(root_mesh.spts_curved.get(et, ())) for et in ('tri', 'quad')
    ):
        raise AMRTransactionError(
            'MIX2D1 requires affine Tri+Quad root geometry'
        )
    if any(np.any(mesh.spts_curved.get(et, ())) for et in ('tri', 'quad')):
        raise AMRTransactionError('MIX2D1 requires affine live geometry')

    # Immutable Tri ordering, coordinates, and tags may not drift.
    if mesh.amr_tree is not None:
        if not np.array_equal(mesh.spts['tri'], root_mesh.spts['tri']):
            raise AMRTransactionError('MIX2D1 immutable Tri geometry changed')
        if not np.array_equal(mesh.tags['tri'], root_mesh.tags['tri']):
            raise AMRTransactionError('MIX2D1 immutable Tri tags changed')
        if not np.array_equal(
            mesh.eidxs['tri'],
            np.arange(len(mesh.eidxs['tri']), dtype=np.int64)
        ):
            raise AMRTransactionError(
                'MIX2D1 immutable Tri order is noncanonical'
            )

    return system, mesh, root_mesh


def _mixed_quad_column_leaves(mesh):
    eidxs = tuple(int(i) for i in mesh.eidxs['quad'])
    if mesh.amr_tree is None:
        return tuple((eidx, ()) for eidx in eidxs)
    leaves = tuple(mesh.amr_tree.leaves())
    try:
        return tuple(leaves[eidx] for eidx in eidxs)
    except IndexError as exc:
        raise AMRTransactionError(
            'MIX2D1 Quad eidxs do not index persistent ancestry'
        ) from exc


def _mixed_etype_bank_map(system, bank, label, etypes, prefix, groups):
    if bank < 0 or bank >= getattr(system, 'nrhs', 0):
        raise AMRTransactionError(f'{prefix} {label} solution bank is invalid')
    if set(system.ele_types) != etypes:
        raise AMRTransactionError(
            f'{prefix} {label} system requires {groups}'
        )

    result = {}
    for etype, banks in zip(system.ele_types, system.ele_banks):
        try:
            shape = tuple(system.ele_shapes[etype])
            gbank = banks[bank]
            bshape = tuple(gbank.ioshape)
        except (KeyError, IndexError, AttributeError, TypeError):
            raise AMRTransactionError(
                f'{prefix} {label} {etype} bank layout unavailable'
            ) from None
        if len(shape) != 3 or any(n <= 0 for n in shape) or bshape != shape:
            raise AMRTransactionError(
                f'{prefix} {label} {etype} bank shape mismatch'
            )
        result[etype] = shape, gbank
    return result


def _uniform_etype_bank_map(system, bank, label):
    return _mixed_etype_bank_map(
        system, bank, label, {'tri', 'quad'}, 'MIX2D1',
        'uniform-p Tri+Quad groups'
    )


def _mixed_hex_bank_map(system, bank, label):
    return _mixed_etype_bank_map(
        system, bank, label, {'hex', 'pyr', 'tet'}, 'V10J',
        'uniform-p Tet+Pyramid+Hex groups'
    )


def _copy_state_by_etype(system, bank):
    return dict(zip(system.ele_types, _copy_state(system, bank)))


def _inject_staged_mixed_bank(
    stage_system, states, bank, bank_map, prefix
):
    bankmap = bank_map(stage_system, bank, 'staged')
    for etype, state in states.items():
        shape, gbank = bankmap[etype]
        if tuple(state.shape) != shape:
            raise AMRTransactionError(
                f'{prefix} transferred {etype} state shape mismatch'
            )
        gbank.set(np.array(state, copy=True, order='C'))
    stage_system.backend.wait()

    readback = _copy_state_by_etype(stage_system, bank)
    if set(readback) != set(states):
        raise AMRTransactionError(f'{prefix} staged state group mismatch')
    for etype in states:
        if not np.array_equal(readback[etype], states[etype]):
            raise AMRTransactionError(
                f'{prefix} staged {etype} bank differs from injected state'
            )
    return readback, tuple(
        (etype, tuple(readback[etype].shape)) for etype in sorted(readback)
    )


def _tri_global_integral(mesh, state, basis):
    qrule = get_quadrule('tri', qdeg=2*basis.order + 2)
    iwts = qrule.wts @ basis.ubasis.nodal_basis_at(qrule.pts)
    src = TriShape.std_ele(1)
    lhs = np.column_stack((src, np.ones(len(src))))
    total = np.zeros(state.shape[1], dtype=np.result_type(state, float))

    if state.shape[2] != len(mesh.eidxs['tri']):
        raise AMRTransactionError('MIX2D1 Tri state/geometry count mismatch')
    for li in range(state.shape[2]):
        dst = np.asarray(mesh.spts['tri'][:, li, :], dtype=float)
        coeff, _, rank, _ = np.linalg.lstsq(lhs, dst, rcond=None)
        if rank != 3:
            raise AMRTransactionError('MIX2D1 immutable Tri is degenerate')
        det = abs(float(np.linalg.det(coeff[:2].T)))
        if not np.isfinite(det) or det <= 0:
            raise AMRTransactionError('MIX2D1 immutable Tri Jacobian invalid')
        total += det*np.einsum('i,iv->v', iwts, state[:, :, li])
    return total


def _combined_eos_ranges(system, states):
    rr, pr = [], []
    for state in states.values():
        rho, pressure = _eos_ranges(system, state)
        rr.append(rho)
        pr.append(pressure)
    return (
        (min(v[0] for v in rr), max(v[1] for v in rr)),
        (min(v[0] for v in pr), max(v[1] for v in pr)),
    )


def _fixed_blocked_quad_leaves(leaves, rootfaces, fixed):
    from pyfr.amr import quad_tree_face_groups

    groups = quad_tree_face_groups(tuple(leaves), rootfaces)
    blocked = set()
    for surface in fixed:
        for fragments in groups.get(surface, {}).values():
            for frag in fragments:
                if frag['level'] >= 1:
                    blocked.add(frag['leaf'])
    return blocked


def _validate_staged_mixed_quad_mesh(
    stage_mesh, raw, root_mesh, new_leaves
):
    if tuple(stage_mesh.amr_tree.leaves()) != new_leaves:
        raise AMRTransactionError('MIX2D1 staged ancestry changed leaf order')
    if set(stage_mesh.etypes) != {'tri', 'quad'}:
        raise AMRTransactionError('MIX2D1 staged element topology changed')
    if not np.array_equal(stage_mesh.spts['tri'], root_mesh.spts['tri']):
        raise AMRTransactionError(
            'MIX2D1 staged immutable Tri geometry changed'
        )

    staged_mortars = tuple(
        mcon for mcon in stage_mesh.mcon.values()
        if mcon.format == 'one-to-many-v1'
        and mcon.template == 'line-1x2'
    )
    if sum(map(len, staged_mortars)) != len(raw.mortars):
        raise AMRTransactionError('MIX2D1 staged mortar count changed')


def _materialize_staged_mixed_quad_mesh(root_mesh, proposed_tree, new_leaves):
    try:
        raw = materialize_native_mixed_quad_tree(
            root_mesh, proposed_tree, comm_size=1
        )
        stage_mesh = build_adapted_mixed_quad_mesh(raw)
    except Exception as exc:
        raise AMRTransactionError(
            f'MIX2D1 native materialization failed: {exc}'
        ) from exc

    _validate_staged_mixed_quad_mesh(stage_mesh, raw, root_mesh, new_leaves)
    return raw, stage_mesh


def _prepare_mixed_quad_transfer(
    root_mesh, old_leaves, new_leaves, old, cfg, fields, gamma
):
    from pyfr.amroffline import (
        _global_integral, _prolongate_to_tree, _validate_state
    )

    qbasis = QuadShape(None, cfg)
    tbasis = TriShape(None, cfg)
    if qbasis.nupts != old['quad'].shape[0]:
        raise AMRTransactionError('MIX2D1 Quad solution-point mismatch')
    if tbasis.nupts != old['tri'].shape[0]:
        raise AMRTransactionError('MIX2D1 Tri solution-point mismatch')

    transferred = {
        'quad': _prolongate_to_tree(
            old['quad'], old_leaves, new_leaves, qbasis
        ),
        'tri': np.array(old['tri'], copy=True, order='C'),
    }
    _validate_state(transferred['quad'], fields, gamma)
    _validate_state(transferred['tri'], fields, gamma)
    if not np.array_equal(transferred['tri'], old['tri']):
        raise AMRTransactionError('MIX2D1 immutable Tri transfer is not exact')

    qbefore = _global_integral(root_mesh, old_leaves, old['quad'], qbasis)
    qafter = _global_integral(
        root_mesh, new_leaves, transferred['quad'], qbasis
    )
    tibefore = _tri_global_integral(root_mesh, old['tri'], tbasis)
    before = qbefore + tibefore
    error = qafter + tibefore - before
    scale = max(1.0, float(np.max(np.abs(before), initial=0.0)))
    tol = 65536*np.finfo(
        np.result_type(old['quad'], old['tri'], float)
    ).eps*scale
    if float(np.max(np.abs(error), initial=0.0)) > tol:
        raise AMRTransactionError(
            'MIX2D1 conservative prolongation gate failed'
        )

    return qbasis, tbasis, transferred, before, tol


def _build_staged_mixed_system(
    intg, system, stage_mesh, transferred, bank, bank_map, prefix
):
    staged = Solution(
        config=intg.cfg, stats=None, fields=None,
        data={et: np.array(v, copy=True) for et, v in transferred.items()},
        state={},
    )
    stage_serialiser = Serialiser()
    stage_system = type(system)(
        intg.backend, stage_mesh, staged, intg._registers,
        intg.cfg, stage_serialiser, needs_cfl=False,
    )
    stage_system.commit()
    staged_states, stage_shapes = _inject_staged_mixed_bank(
        stage_system, transferred, bank, bank_map, prefix
    )
    return stage_system, stage_serialiser, staged_states, stage_shapes


def _validate_staged_mixed_quad_transfer(
    root_mesh, new_leaves, stage_system, staged_states, qbasis, tbasis,
    before, tol
):
    from pyfr.amroffline import _global_integral

    after = (
        _global_integral(
            root_mesh, new_leaves, staged_states['quad'], qbasis
        ) + _tri_global_integral(
            root_mesh, staged_states['tri'], tbasis
        )
    )
    error = after - before
    if float(np.max(np.abs(error), initial=0.0)) > tol:
        raise AMRTransactionError(
            'MIX2D1 staged conservative transfer gate failed'
        )

    rho_range, pressure_range = _combined_eos_ranges(
        stage_system, staged_states
    )
    return after, error, rho_range, pressure_range


def _validate_staged_mixed_rhs(
    stage_system, system, intg, bank, scratch_bank, transferred, label
):
    stage_system.preproc(intg.tcurr, bank)
    intg.backend.wait()
    after_preproc = _copy_state_by_etype(stage_system, bank)
    for etype in transferred:
        if not np.array_equal(after_preproc[etype], transferred[etype]):
            raise AMRTransactionError(
                f'{label} staged {etype} bank changed during preprocessing'
            )

    old_layout = {
        et: len(banks)
        for et, banks in zip(system.ele_types, system.ele_banks)
    }
    new_layout = {
        et: len(banks)
        for et, banks in zip(stage_system.ele_types, stage_system.ele_banks)
    }
    if stage_system.nrhs != system.nrhs or new_layout != old_layout:
        raise AMRTransactionError(f'{label} register-bank layout changed')

    rhs, bank_drift = _scratch_rhs(
        stage_system, intg.tcurr, bank, scratch_bank
    )
    after_rhs = _copy_state_by_etype(stage_system, bank)
    for etype in transferred:
        if not np.array_equal(after_rhs[etype], transferred[etype]):
            raise AMRTransactionError(
                f'{label} staged {etype} bank changed after first RHS'
            )

    return rhs, bank_drift, sum(stage_system.ele_ndofs)


def _prepare_mixed_quad_state(intg, system, root_mesh, old_leaves, bank):
    from pyfr.amroffline import _validate_state

    bankmap = _uniform_etype_bank_map(system, bank, 'accepted')
    scratch_bank = _scratch_bank(system, bank)
    old = _copy_state_by_etype(system, bank)
    qshape, tshape = bankmap['quad'][0], bankmap['tri'][0]
    if tuple(old['quad'].shape) != qshape or tuple(old['tri'].shape) != tshape:
        raise AMRTransactionError('MIX2D1 accepted bank readback mismatch')
    if qshape[2] != len(old_leaves):
        raise AMRTransactionError('MIX2D1 Quad bank/leaf count mismatch')
    if tshape[2] != len(root_mesh.eidxs['tri']):
        raise AMRTransactionError('MIX2D1 immutable Tri bank count mismatch')

    fields = tuple(system.elementscls.convars(system.ndims, intg.cfg))
    gamma = intg.cfg.getfloat('constants', 'gamma')
    state_indices = _validate_state(old['quad'], fields, gamma)
    _validate_state(old['tri'], fields, gamma)
    rho_range, pressure_range = _combined_eos_ranges(system, old)
    return (
        scratch_bank, old, fields, gamma, state_indices, rho_range,
        pressure_range
    )


def _close_mixed_quad_refinement(
    root_mesh, roots, old_leaves, rootfaces, rootkinds, fixed, marks,
    wall_boundaries, wall_min_level, l2g, max_level
):
    from pyfr.amr import quad_tree_face_groups, quad_tree_leaves
    from pyfr.amroffline import _close_2to1, _split_nodes, _wall_floor_splits

    # A level-1 Quad touching an immutable Tri cannot be refined to level 2.
    blocked = _fixed_blocked_quad_leaves(old_leaves, rootfaces, fixed)
    fixed_blocked_marks = tuple(sorted(marks & blocked))
    marks -= blocked

    wall_roots = _quad_wall_roots(root_mesh, wall_boundaries, l2g)
    existing = _split_nodes(old_leaves)
    wall_floor = _wall_floor_splits(wall_roots, wall_min_level)

    # Do not weaken the accepted wall-floor policy against an immutable Tri.
    wall_leaves = tuple(quad_tree_leaves(roots, existing | wall_floor))
    wall_groups = quad_tree_face_groups(wall_leaves, rootfaces)
    if any(
        frag['level'] > 1
        for surface in fixed
        for fragments in wall_groups.get(surface, {}).values()
        for frag in fragments
    ):
        raise AMRTransactionError(
            'MIX2D1 wall minimum level conflicts with immutable Tri 2:1 '
            'constraint; add a Quad buffer layer'
        )

    wall_splits = tuple(sorted(wall_floor - existing - marks))
    split = existing | marks | wall_floor
    try:
        new_leaves, closure = _close_2to1(
            roots, split, rootfaces, rootkinds, max_level
        )
    except Exception as exc:
        raise AMRTransactionError(str(exc)) from exc

    return (
        tuple(new_leaves), tuple(sorted(marks)), wall_splits, tuple(closure),
        fixed_blocked_marks
    )


def perform_indicator_mixed_quad_amr_transaction(
    intg, *, restart_root_mesh=None
):
    from pyfr.amrindicator import density_velocity_variation_scores
    system, mesh, root_mesh = _validate_mixed_quad_online_integrator(
        intg, restart_root_mesh
    )
    (
        refine_threshold, max_level, wall_boundaries, wall_min_level,
        density_floor, acoustic_floor,
    ) = _quad_online_settings(intg.cfg)

    l2g, pids_by_g, _ = _quad_local_maps(root_mesh)
    roots = tuple(sorted(pids_by_g))
    rootfaces, rootkinds, fixed = _derive_mixed_quad_root_face_topology(
        root_mesh, pids_by_g, l2g
    )
    old_leaves = _mixed_quad_column_leaves(mesh)
    if {root for root, _ in old_leaves} != set(roots):
        raise AMRTransactionError(
            'MIX2D1 Quad ancestry root coverage mismatch'
        )
    old_tree = encode_quad_leaf_tree(root_mesh.uuid, old_leaves)

    bank = intg.idxcurr
    (
        scratch_bank, old, fields, gamma, state_indices, old_rho,
        old_pressure
    ) = _prepare_mixed_quad_state(
        intg, system, root_mesh, old_leaves, bank
    )
    irho, irhou, irhov, ienergy = state_indices

    scores = density_velocity_variation_scores(
        old['quad'], density_index=irho,
        momentum_indices=(irhou, irhov), energy_index=ienergy,
        gamma=gamma, density_floor=density_floor,
        acoustic_floor=acoustic_floor,
    )
    score_items = tuple(zip(old_leaves, map(float, scores)))
    marks = {
        leaf for leaf, score in score_items
        if len(leaf[1]) < max_level and score >= refine_threshold
    }

    (
        new_leaves, marks, wall_splits, closure, fixed_blocked_marks
    ) = _close_mixed_quad_refinement(
        root_mesh, roots, old_leaves, rootfaces, rootkinds, fixed, marks,
        wall_boundaries, wall_min_level, l2g, max_level
    )
    trigger = max(
        (score for leaf, score in score_items if leaf in set(marks)),
        default=None,
    )

    if new_leaves == tuple(old_leaves):
        return QuadIndicatorAMRResult(
            QuadAMRDecision('none', marks, trigger), score_items, None
        )

    proposed_tree = encode_quad_leaf_tree(root_mesh.uuid, new_leaves)
    qbasis, tbasis, transferred, before, tol = _prepare_mixed_quad_transfer(
        root_mesh, old_leaves, new_leaves, old, intg.cfg, fields, gamma
    )

    raw, stage_mesh = _materialize_staged_mixed_quad_mesh(
        root_mesh, proposed_tree, new_leaves
    )

    stage_system = None
    try:
        stage_system, stage_serialiser, staged_states, stage_shapes = (
            _build_staged_mixed_system(
                intg, system, stage_mesh, transferred, bank,
                _uniform_etype_bank_map, 'MIX2D1'
            )
        )

        after, error, staged_rho, staged_pressure = (
            _validate_staged_mixed_quad_transfer(
                root_mesh, new_leaves, stage_system, staged_states,
                qbasis, tbasis, before, tol
            )
        )

        rhs, bank_drift, staged_gndofs = _validate_staged_mixed_rhs(
            stage_system, system, intg, bank, scratch_bank, transferred,
            'MIX2D1'
        )
    except Exception:
        stage_system = None
        raise

    clear_memoize(intg)
    old_system_id = id(system)
    intg.system = stage_system
    intg._amr_root_mesh = root_mesh
    intg.mesh_uuid = stage_mesh.uuid
    intg.serialiser = stage_serialiser
    intg.gndofs = staged_gndofs
    intg._invalidate_caches()

    tx = MixedQuadAMRTransactionResult(
        old_tree=old_tree, proposed_tree=proposed_tree, raw_mesh=raw,
        transferred_state=transferred, rhs=tuple(rhs), refine_marks=marks,
        wall_splits=wall_splits, closure_splits=tuple(closure),
        fixed_blocked_marks=fixed_blocked_marks,
        conservation_before=before, conservation_after=after,
        conservation_error=error, old_rho_range=old_rho,
        old_pressure_range=old_pressure, rho_range=staged_rho,
        pressure_range=staged_pressure, stage_mesh_uuid=stage_mesh.uuid,
        stage_leaf_count=len(raw.leaf_order),
        stage_tri_count=len(raw.tri_nodes),
        stage_mortar_count=len(raw.mortars),
        stage_mortar_formats=tuple(
            (m.format, m.template) for m in raw.mortars
        ),
        bank_drift=bank_drift, old_system_id=old_system_id,
        new_system_id=id(stage_system), accepted_bank=bank,
        scratch_bank=scratch_bank, stage_accepted_shapes=stage_shapes,
        tcurr=float(intg.tcurr),
    )
    return QuadIndicatorAMRResult(
        QuadAMRDecision('refine', marks, trigger), score_items, tx
    )


@dataclass(frozen=True)
class MixedHexAMRTransactionResult:

    old_tree: object
    proposed_tree: object
    raw_mesh: object
    transferred_state: dict
    rhs: tuple
    refine_marks: tuple
    wall_splits: tuple
    closure_splits: tuple
    fixed_blocked_marks: tuple
    conservation_before: np.ndarray
    conservation_after: np.ndarray
    conservation_error: np.ndarray
    old_rho_range: tuple
    old_pressure_range: tuple
    rho_range: tuple
    pressure_range: tuple
    stage_mesh_uuid: str
    stage_leaf_count: int
    stage_pyr_count: int
    stage_tet_count: int
    stage_mortar_count: int
    stage_mortar_formats: tuple
    bank_drift: float
    old_system_id: int
    new_system_id: int
    accepted_bank: int
    scratch_bank: int
    stage_accepted_shapes: tuple
    tcurr: float

    @property
    def refine_mark(self):
        return self.refine_marks[0] if len(self.refine_marks) == 1 else None


@dataclass(frozen=True)
class MixedHexAMRDecision:

    action: str
    marks: tuple
    trigger_score: float | None


@dataclass(frozen=True)
class MixedHexIndicatorAMRResult:

    decision: MixedHexAMRDecision
    scores: tuple
    transaction: MixedHexAMRTransactionResult | None






def _validate_mixed_hex_online_integrator(intg, restart_root_mesh=None):
    if getattr(intg, 'formulation', None) != 'explicit':
        raise AMRTransactionError('V10J requires explicit formulation')
    if getattr(intg, 'controller_name', None) != 'none':
        raise AMRTransactionError('V10J requires controller none')
    if getattr(intg, 'stepper_name', None) != 'rk4':
        raise AMRTransactionError('V10J requires explicit RK4')
    if getattr(getattr(intg, 'backend', None), 'name', None) not in {
        'openmp', 'cuda'
    }:
        raise AMRTransactionError('V10J currently requires OpenMP or CUDA')
    if (getattr(intg, 'nacptsteps', 0) < 1 or
            getattr(intg, 'stepinfo', None) != []):
        raise AMRTransactionError(
            'V10J requires a completed post-advance accepted timestep'
        )

    cfg = intg.cfg
    solver_system = cfg.get('solver', 'system')
    if solver_system not in {'euler', 'navier-stokes'}:
        raise AMRTransactionError('V10J supports Euler/Navier-Stokes only')
    if (solver_system == 'navier-stokes' and
            cfg.get('solver', 'viscosity-correction', 'none') != 'none'):
        raise AMRTransactionError(
            'V10J Navier-Stokes requires constant viscosity'
        )
    if cfg.get('solver-time-integrator', 'controller', 'none') != 'none':
        raise AMRTransactionError('V10J requires controller = none')
    if cfg.get('solver-time-integrator', 'scheme', 'rk4') != 'rk4':
        raise AMRTransactionError('V10J requires scheme = rk4')
    if cfg.get('solver', 'shock-capturing', 'none') != 'none':
        raise AMRTransactionError('V10J requires shock-capturing = none')
    if (cfg.get('solver-interfaces', 'mortar-implementation', 'fused') !=
            'staged'):
        raise AMRTransactionError('V10J requires staged mortars')
    if any(s.startswith('solver-order-') for s in cfg.sections()):
        raise AMRTransactionError('V10J does not implement mixed-p')

    plugin_sections = [
        s for s in cfg.sections()
        if s.startswith(('soln-plugin-', 'solver-plugin-', 'trigger-'))
    ]
    if (plugin_sections or getattr(intg, 'plugins', None) or
            getattr(intg, 'triggers', None)):
        raise AMRTransactionError(
            'V10J first live seam excludes plugins/triggers'
        )
    if getattr(getattr(intg, 'serialiser', None), '_serialfns', {}):
        raise AMRTransactionError(
            'V10J excludes unsupported serialised mutable state'
        )

    comm, _, _ = get_comm_rank_root()
    if comm.size != 1:
        raise AMRTransactionError('V10J requires one MPI rank')

    system = getattr(intg, 'system', None)
    mesh = getattr(system, 'mesh', None)
    if system is None or mesh is None:
        raise AMRTransactionError('V10J requires a live PyFR system')
    if (getattr(system, 'name', None) != solver_system or
            system.ndims != 3):
        raise AMRTransactionError('V10J live system/configuration mismatch')
    if set(mesh.etypes) != {'hex', 'pyr', 'tet'} or mesh.con_p:
        raise AMRTransactionError(
            'V10J requires one-rank mixed Tet+Pyramid+Hex topology'
        )

    if mesh.amr_tree is None:
        if mesh.mcon:
            raise AMRTransactionError(
                'V10J root mesh must not contain pre-existing mortars'
            )
        root_mesh = mesh
    else:
        from pyfr.amr import HexLeafTree
        if not isinstance(mesh.amr_tree, HexLeafTree):
            raise AMRTransactionError(
                'V10J adapted mesh requires typed Hex ancestry'
            )
        for mcon in mesh.mcon.values():
            if (mcon.format != 'one-to-many-v1' or
                    mcon.template != 'quad-2x2' or mcon.nright != 4):
                raise AMRTransactionError(
                    'V10J adapted mesh contains an unsupported mortar'
                )

        live_root_mesh = getattr(intg, '_amr_root_mesh', None)
        if live_root_mesh is not None and restart_root_mesh is not None:
            if live_root_mesh.uuid != restart_root_mesh.uuid:
                raise AMRTransactionError(
                    'V10J supplied root differs from live root anchor'
                )
        root_mesh = live_root_mesh or restart_root_mesh
        if root_mesh is None:
            raise AMRTransactionError(
                'V10J adapted live mesh requires immutable root anchor'
            )
        if getattr(root_mesh, 'amr_tree', None) is not None:
            raise AMRTransactionError('V10J root anchor must be unadapted')
        if root_mesh.uuid != mesh.amr_tree.root_mesh_uuid:
            raise AMRTransactionError('V10J root anchor UUID mismatch')

    if (set(root_mesh.etypes) != {'hex', 'pyr', 'tet'} or
            root_mesh.con_p or root_mesh.mcon):
        raise AMRTransactionError(
            'V10J root anchor is outside Tet+Pyramid+Hex scope'
        )
    if mesh.amr_tree is not None:
        for etype in ('pyr', 'tet'):
            if not np.array_equal(mesh.spts[etype], root_mesh.spts[etype]):
                raise AMRTransactionError(
                    f'V10J immutable {etype} geometry changed'
                )
            if not np.array_equal(mesh.tags[etype], root_mesh.tags[etype]):
                raise AMRTransactionError(
                    f'V10J immutable {etype} tags changed'
                )
            if not np.array_equal(
                mesh.eidxs[etype],
                np.arange(len(mesh.eidxs[etype]), dtype=np.int64)
            ):
                raise AMRTransactionError(
                    f'V10J immutable {etype} order is noncanonical'
                )

    return system, mesh, root_mesh


def _close_mixed_hex_refinements(
    root_mesh, old_tree, marks, *, forced_splits=()
):
    from pyfr.amrmesh import _derive_mixed_hex_root_face_topology

    l2g, _, pids_by_g, _ = _hex_local_maps(root_mesh)
    rootfaces, rootkinds, fixed = _derive_mixed_hex_root_face_topology(
        root_mesh, pids_by_g, l2g
    )
    roots = sorted(pids_by_g)
    old_leaves = tuple(old_tree.leaves())
    marks = tuple(marks)
    bad = sorted(set(marks) - set(old_leaves))
    if bad:
        raise AMRTransactionError(
            f'V10J refinement mark is not an active leaf: {bad[0]!r}'
        )

    split = _tree_split_nodes(old_tree) | set(marks) | set(forced_splits)
    closure = []
    while True:
        leaves = tuple(hex_tree_leaves(roots, split))
        groups = hex_tree_face_groups(leaves, rootfaces)
        try:
            pairs = list(hex_tree_face_pairs(groups, rootkinds))
        except ValueError as exc:
            raise AMRTransactionError(str(exc)) from exc

        add = set()
        for lhs, rhs in pairs:
            if abs(lhs['level'] - rhs['level']) <= 1:
                continue
            coarse = lhs['leaf'] if lhs['level'] < rhs['level'] else rhs['leaf']
            add.add(coarse)
        add -= split
        if not add:
            break
        ordered = sorted(add)
        split.update(ordered)
        closure.extend(ordered)

    # Immutable Pyramid is permanently level 0.  Any direct L2 Hex contact
    # would violate the accepted 2:1 trace contract and is rejected.
    groups = hex_tree_face_groups(leaves, rootfaces)
    for surface in fixed:
        if any(
            frag['level'] > 1
            for fragments in groups.get(surface, {}).values()
            for frag in fragments
        ):
            raise AMRTransactionError(
                'V10J refinement would create L2 Hex contact with immutable '
                'L0 Pyramid'
            )

    try:
        tree = encode_hex_leaf_tree(root_mesh.uuid, leaves)
    except ValueError as exc:
        raise AMRTransactionError(str(exc)) from exc
    return tree, tuple(closure)


def _close_mixed_hex_refinement(root_mesh, old_tree, mark):
    return _close_mixed_hex_refinements(root_mesh, old_tree, (mark,))


def _validate_staged_mixed_hex_mesh(
    stage_mesh, raw, root_mesh, proposed_tree
):
    if tuple(stage_mesh.amr_tree.leaves()) != tuple(proposed_tree.leaves()):
        raise AMRTransactionError('V10J staged ancestry changed leaf order')
    if set(stage_mesh.etypes) != {'hex', 'pyr', 'tet'}:
        raise AMRTransactionError('V10J staged element topology changed')
    for etype in ('pyr', 'tet'):
        if not np.array_equal(stage_mesh.spts[etype], root_mesh.spts[etype]):
            raise AMRTransactionError(
                f'V10J staged immutable {etype} geometry changed'
            )
        if not np.array_equal(stage_mesh.tags[etype], root_mesh.tags[etype]):
            raise AMRTransactionError(
                f'V10J staged immutable {etype} tags changed'
            )

    staged_mortars = tuple(
        mcon for mcon in stage_mesh.mcon.values()
        if mcon.format == 'one-to-many-v1'
        and mcon.template == 'quad-2x2'
    )
    if sum(map(len, staged_mortars)) != len(raw.mortars):
        raise AMRTransactionError('V10J staged mortar count changed')


def _materialize_staged_mixed_hex_mesh(root_mesh, proposed_tree):
    from pyfr.amrmesh import materialize_native_mixed_hex_tree
    from pyfr.amrwriter import build_adapted_mixed_hex_mesh

    try:
        raw = materialize_native_mixed_hex_tree(root_mesh, proposed_tree)
        stage_mesh = build_adapted_mixed_hex_mesh(raw)
    except Exception as exc:
        raise AMRTransactionError(
            f'V10J native materialization failed: {exc}'
        ) from exc

    _validate_staged_mixed_hex_mesh(
        stage_mesh, raw, root_mesh, proposed_tree
    )
    return raw, stage_mesh


def _validate_proposed_mixed_hex_transfer(
    transferred, raw, basis, before, dtype
):
    proposed_after = _physical_hex_conserved_totals(
        transferred['hex'], _raw_hex_spts(raw), basis
    )
    error = proposed_after - before
    scale = np.maximum(1.0, np.abs(before))
    tol = 8192*np.finfo(dtype).eps
    if np.any(np.abs(error) > tol*scale):
        raise AMRTransactionError(
            'V10J componentwise conservative Hex transfer gate failed'
        )

    return tol, scale


def _validate_staged_mixed_hex_transfer(
    stage_system, stage_mesh, staged_states, basis, before, tol, scale
):
    after = _physical_hex_conserved_totals(
        staged_states['hex'], stage_mesh.spts['hex'], basis
    )
    error = after - before
    if np.any(np.abs(error) > tol*scale):
        raise AMRTransactionError(
            'V10J staged conservative Hex transfer gate failed'
        )

    rho_range, pressure_range = _combined_eos_ranges(
        stage_system, staged_states
    )
    return after, error, rho_range, pressure_range


def _commit_mixed_hex_refinement(
    intg, system, mesh, root_mesh, old_tree, local_by_leaf, proposed_tree,
    *, refine_marks=(), wall_splits=(), closure_splits=(),
    fixed_blocked_marks=(),
):
    bank = intg.idxcurr
    bankmap = _mixed_hex_bank_map(system, bank, 'accepted')
    scratch_bank = _scratch_bank(system, bank)
    old = _copy_state_by_etype(system, bank)
    for etype, (shape, _) in bankmap.items():
        if tuple(old[etype].shape) != shape:
            raise AMRTransactionError(
                f'V10J accepted {etype} bank readback mismatch'
            )
    if old['hex'].shape[2] != len(local_by_leaf):
        raise AMRTransactionError('V10J Hex bank/leaf count mismatch')
    for etype in ('pyr', 'tet'):
        if old[etype].shape[2] != len(root_mesh.eidxs[etype]):
            raise AMRTransactionError(
                f'V10J immutable {etype} bank count mismatch'
            )

    old_rho, old_pressure = _combined_eos_ranges(system, old)
    transferred_hex, basis = _transfer_state(
        system, old_tree, proposed_tree, old['hex'], local_by_leaf
    )
    transferred = {
        'hex': transferred_hex,
        'pyr': np.array(old['pyr'], copy=True, order='C'),
        'tet': np.array(old['tet'], copy=True, order='C'),
    }
    for etype in ('pyr', 'tet'):
        if not np.array_equal(transferred[etype], old[etype]):
            raise AMRTransactionError(
                f'V10J immutable {etype} transfer is not exact'
            )
    _combined_eos_ranges(system, transferred)

    before = _physical_hex_conserved_totals(
        old['hex'], mesh.spts['hex'], basis
    )

    raw, stage_mesh = _materialize_staged_mixed_hex_mesh(
        root_mesh, proposed_tree
    )

    tol, scale = _validate_proposed_mixed_hex_transfer(
        transferred, raw, basis, before, old['hex'].dtype
    )

    stage_system = None
    try:
        stage_system, stage_serialiser, staged_states, stage_shapes = (
            _build_staged_mixed_system(
                intg, system, stage_mesh, transferred, bank,
                _mixed_hex_bank_map, 'V10J'
            )
        )

        after, error, staged_rho, staged_pressure = (
            _validate_staged_mixed_hex_transfer(
                stage_system, stage_mesh, staged_states, basis, before, tol,
                scale
            )
        )

        rhs, bank_drift, staged_gndofs = _validate_staged_mixed_rhs(
            stage_system, system, intg, bank, scratch_bank, transferred,
            'V10J'
        )
    except Exception:
        stage_system = None
        raise

    clear_memoize(intg)
    old_system_id = id(system)
    intg.system = stage_system
    intg._amr_root_mesh = root_mesh
    intg.mesh_uuid = stage_mesh.uuid
    intg.serialiser = stage_serialiser
    intg.gndofs = staged_gndofs
    intg._invalidate_caches()

    return MixedHexAMRTransactionResult(
        old_tree=old_tree, proposed_tree=proposed_tree, raw_mesh=raw,
        transferred_state=transferred, rhs=tuple(rhs),
        refine_marks=tuple(refine_marks), wall_splits=tuple(wall_splits),
        closure_splits=tuple(closure_splits),
        fixed_blocked_marks=tuple(fixed_blocked_marks),
        conservation_before=before, conservation_after=after,
        conservation_error=error, old_rho_range=old_rho,
        old_pressure_range=old_pressure, rho_range=staged_rho,
        pressure_range=staged_pressure, stage_mesh_uuid=stage_mesh.uuid,
        stage_leaf_count=len(raw.leaf_order),
        stage_pyr_count=len(raw.fixed_nodes['pyr']),
        stage_tet_count=len(raw.fixed_nodes['tet']),
        stage_mortar_count=len(raw.mortars),
        stage_mortar_formats=tuple(
            (m.format, m.template) for m in raw.mortars
        ),
        bank_drift=bank_drift, old_system_id=old_system_id,
        new_system_id=id(stage_system), accepted_bank=bank,
        scratch_bank=scratch_bank, stage_accepted_shapes=stage_shapes,
        tcurr=float(intg.tcurr),
    )


def perform_scripted_mixed_hex_amr_transaction(
    intg, scripted_mark, *, restart_root_mesh=None
):
    system, mesh, root_mesh = _validate_mixed_hex_online_integrator(
        intg, restart_root_mesh
    )
    mark = _normalise_mark(scripted_mark)

    old_tree, local_by_leaf = _current_tree(mesh)
    if mark not in set(old_tree.leaves()):
        raise AMRTransactionError(
            f'V10J scripted mark is not active: {mark!r}'
        )
    proposed_tree, closure = _close_mixed_hex_refinement(
        root_mesh, old_tree, mark
    )
    return _commit_mixed_hex_refinement(
        intg, system, mesh, root_mesh, old_tree, local_by_leaf,
        proposed_tree, refine_marks=(mark,), closure_splits=closure,
    )


def _mixed_hex_online_settings(cfg):
    section = 'solver-amr'
    if section not in cfg.sections():
        raise AMRTransactionError('V10J D9Q requires [solver-amr]')
    if not cfg.hasopt(section, 'refine-threshold'):
        raise AMRTransactionError(
            'V10J D9Q requires solver-amr refine-threshold'
        )
    if cfg.get(section, 'indicator', 'density-velocity-variation') != \
            'density-velocity-variation':
        raise AMRTransactionError(
            'V10J D9Q requires density-velocity-variation'
        )

    threshold = cfg.getfloat(section, 'refine-threshold')
    if not np.isfinite(threshold) or threshold < 0:
        raise AMRTransactionError(
            'V10J D9Q refine threshold must be finite and nonnegative'
        )
    max_level = cfg.getint(section, 'max-level', 2)
    if not 1 <= max_level <= 2:
        raise AMRTransactionError('V10J D9Q max-level must be one or two')
    wall_min_level = cfg.getint(section, 'wall-min-level', 0)
    if not 0 <= wall_min_level <= max_level:
        raise AMRTransactionError(
            'V10J D9Q wall-min-level must be in 0..max-level'
        )

    wall_text = cfg.get(section, 'wall-boundaries', '')
    walls = tuple(v for v in wall_text.replace(',', ' ').split() if v)
    if len(set(walls)) != len(walls):
        raise AMRTransactionError(
            'V10J D9Q wall boundary names must be unique'
        )

    density_floor = cfg.getfloat(section, 'density-floor', 1e-14)
    acoustic_floor = cfg.getfloat(section, 'acoustic-floor', 1e-14)
    if not np.isfinite(density_floor) or density_floor <= 0:
        raise AMRTransactionError(
            'V10J D9Q density floor must be finite and positive'
        )
    if not np.isfinite(acoustic_floor) or acoustic_floor <= 0:
        raise AMRTransactionError(
            'V10J D9Q acoustic floor must be finite and positive'
        )
    return (
        threshold, max_level, walls, wall_min_level,
        density_floor, acoustic_floor,
    )


def _mixed_hex_wall_roots(root_mesh, wall_boundaries, l2g):
    roots = set()
    for name in wall_boundaries:
        try:
            bcon = root_mesh.bcon[name]
        except KeyError as exc:
            raise AMRTransactionError(
                f'V10J D9Q root mesh has no boundary {name!r}'
            ) from exc
        found = False
        for etype, _, eidxs in bcon.items():
            if etype == 'hex':
                found = True
                roots.update(l2g[int(eidx)] for eidx in eidxs)
        if not found:
            raise AMRTransactionError(
                f'V10J D9Q wall boundary {name!r} has no Hex faces'
            )
    return roots


def _hex_wall_floor_splits(wall_roots, min_level):
    split = set()
    frontier = [(root, ()) for root in sorted(wall_roots)]
    for _ in range(min_level):
        split.update(frontier)
        frontier = [
            (root, path + (octant,))
            for root, path in frontier for octant in range(8)
        ]
    return split


def _fixed_blocked_hex_leaves(leaves, rootfaces, fixed):
    groups = hex_tree_face_groups(tuple(leaves), rootfaces)
    blocked = set()
    for surface in fixed:
        for fragments in groups.get(surface, {}).values():
            for frag in fragments:
                if frag['level'] >= 1:
                    blocked.add(frag['leaf'])
    return blocked


def perform_indicator_mixed_hex_amr_transaction(
    intg, *, restart_root_mesh=None
):
    from pyfr.amrindicator import density_velocity_variation_scores
    from pyfr.amrmesh import _derive_mixed_hex_root_face_topology

    system, mesh, root_mesh = _validate_mixed_hex_online_integrator(
        intg, restart_root_mesh
    )
    (
        threshold, max_level, wall_boundaries, wall_min_level,
        density_floor, acoustic_floor,
    ) = _mixed_hex_online_settings(intg.cfg)

    old_tree, local_by_leaf = _current_tree(mesh)
    old_leaves = tuple(
        sorted(local_by_leaf, key=local_by_leaf.__getitem__)
    )
    bank = intg.idxcurr
    old = _copy_state_by_etype(system, bank)
    _combined_eos_ranges(system, old)

    fields = tuple(system.elementscls.convars(system.ndims, intg.cfg))
    try:
        irho = fields.index('rho')
        imom = tuple(fields.index(v) for v in ('rhou', 'rhov', 'rhow'))
        ienergy = fields.index('E')
    except ValueError as exc:
        raise AMRTransactionError(
            'V10J D9Q conserved-field layout is unsupported'
        ) from exc
    gamma = intg.cfg.getfloat('constants', 'gamma')
    scores = density_velocity_variation_scores(
        old['hex'], density_index=irho, momentum_indices=imom,
        energy_index=ienergy, gamma=gamma, density_floor=density_floor,
        acoustic_floor=acoustic_floor,
    )
    score_items = tuple(zip(old_leaves, map(float, scores)))
    marks = {
        leaf for leaf, score in score_items
        if len(leaf[1]) < max_level and score >= threshold
    }

    l2g, _, pids_by_g, _ = _hex_local_maps(root_mesh)
    rootfaces, _, fixed = _derive_mixed_hex_root_face_topology(
        root_mesh, pids_by_g, l2g
    )
    blocked = _fixed_blocked_hex_leaves(old_leaves, rootfaces, fixed)
    fixed_blocked_marks = tuple(sorted(marks & blocked))
    marks -= blocked

    existing = _tree_split_nodes(old_tree)
    wall_roots = _mixed_hex_wall_roots(
        root_mesh, wall_boundaries, l2g
    )
    wall_floor = _hex_wall_floor_splits(wall_roots, wall_min_level)
    wall_splits = tuple(sorted(wall_floor - existing - marks))
    proposed_tree, closure = _close_mixed_hex_refinements(
        root_mesh, old_tree, tuple(sorted(marks)),
        forced_splits=wall_floor,
    )
    marks = tuple(sorted(marks))
    trigger = max(
        (score for leaf, score in score_items if leaf in set(marks)),
        default=None,
    )

    if tuple(proposed_tree.leaves()) == tuple(old_tree.leaves()):
        return MixedHexIndicatorAMRResult(
            MixedHexAMRDecision('none', marks, trigger), score_items, None
        )

    tx = _commit_mixed_hex_refinement(
        intg, system, mesh, root_mesh, old_tree, local_by_leaf,
        proposed_tree, refine_marks=marks, wall_splits=wall_splits,
        closure_splits=closure, fixed_blocked_marks=fixed_blocked_marks,
    )
    return MixedHexIndicatorAMRResult(
        MixedHexAMRDecision('refine', marks, trigger), score_items, tx
    )
