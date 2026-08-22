"""Offline monotone Quad h-adaptation for native PyFR restart pairs."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyfr.amr import (
    QuadLeafTree, apply_quad_refine_transfer, build_quad_refine_transfer,
    encode_quad_leaf_tree, quad_tree_face_groups, quad_tree_face_pairs,
    quad_tree_leaves,
)
from pyfr.amrindicator import density_velocity_variation_scores
from pyfr.amrmesh import (
    _derive_quad_root_face_topology, _quad_local_maps,
    materialize_native_quad_tree,
)
from pyfr.amrwriter import write_adapted_quad_mesh
from pyfr.mpiutil import get_comm_rank_root
from pyfr.quadrules import get_quadrule
from pyfr.readers.native import NativeReader
from pyfr.shapes import QuadShape
from pyfr.writers.native import NativeWriter


class OfflineQuadAMRError(ValueError):
    """An offline Quad AMR transform is invalid or outside scope."""


@dataclass(frozen=True)
class OfflineQuadAMRResult:
    old_leaves: int
    new_leaves: int
    refine_marks: int
    closure_splits: int
    max_score: float
    conservation_residual: float
    mesh_path: str
    soln_path: str


def _split_nodes(leaves):
    return {
        (root, path[:depth])
        for root, path in leaves
        for depth in range(len(path))
    }


def _wall_floor_splits(wall_roots, min_level):
    split = set()
    frontier = [(root, ()) for root in sorted(wall_roots)]

    for _ in range(min_level):
        split.update(frontier)
        frontier = [
            (root, path + (quadrant,))
            for root, path in frontier
            for quadrant in range(4)
        ]

    return split


def _close_2to1(roots, split, rootfaces, rootkinds, max_level):
    split = set(split)
    closure = []

    while True:
        leaves = tuple(quad_tree_leaves(roots, split))
        groups = quad_tree_face_groups(leaves, rootfaces)
        add = set()

        for lhs, rhs in quad_tree_face_pairs(groups, rootkinds):
            if abs(lhs['level'] - rhs['level']) <= 1:
                continue

            coarse = lhs if lhs['level'] < rhs['level'] else rhs
            leaf = coarse['leaf']
            if len(leaf[1]) >= max_level:
                raise OfflineQuadAMRError(
                    '2:1 closure would exceed the configured maximum level'
                )
            add.add(leaf)

        add -= split
        if not add:
            return leaves, tuple(closure)

        ordered = sorted(add)
        split.update(ordered)
        closure.extend(ordered)


def _conserved_indices(fields):
    fields = tuple(fields)
    required = ('rho', 'rhou', 'rhov', 'E')
    missing = [name for name in required if name not in fields]
    if missing:
        raise OfflineQuadAMRError(
            f'2D compressible Quad AMR requires fields {required}; '
            f'missing {tuple(missing)}'
        )

    return tuple(fields.index(name) for name in required)


def _validate_state(state, fields, gamma):
    state = np.asarray(state)
    if state.ndim != 3:
        raise OfflineQuadAMRError(
            'Quad solution state must have shape (nupts, nvars, neles)'
        )
    if not np.isfinite(state).all():
        raise OfflineQuadAMRError('Quad solution state is not finite')

    irho, irhou, irhov, ienergy = _conserved_indices(fields)
    rho = state[:, irho]
    if np.any(rho <= 0):
        raise OfflineQuadAMRError('Quad solution density is nonpositive')

    rhou, rhov = state[:, irhou], state[:, irhov]
    energy = state[:, ienergy]
    pressure = (gamma - 1)*(
        energy - 0.5*(rhou*rhou + rhov*rhov)/rho
    )
    if not np.isfinite(pressure).all() or np.any(pressure <= 0):
        raise OfflineQuadAMRError('Quad solution pressure is nonpositive')

    return irho, irhou, irhov, ienergy


def _current_column_leaves(mesh, roots):
    if 'quad' not in mesh.eidxs or len(mesh.etypes) != 1:
        raise OfflineQuadAMRError(
            'offline Quad AMR requires a pure-Quad current mesh'
        )

    eidxs = tuple(int(i) for i in mesh.eidxs['quad'])
    if mesh.amr_tree is None:
        if mesh.uuid is None:
            raise OfflineQuadAMRError('current root mesh UUID is missing')
        return tuple((eidx, ()) for eidx in eidxs)

    if not isinstance(mesh.amr_tree, QuadLeafTree):
        raise OfflineQuadAMRError(
            'offline Quad AMR requires typed Quad ancestry on adapted input'
        )

    leaves = tuple(mesh.amr_tree.leaves())
    try:
        return tuple(leaves[eidx] for eidx in eidxs)
    except IndexError as exc:
        raise OfflineQuadAMRError(
            'adapted Quad element indices do not index persistent leaves'
        ) from exc


def _prolongate_to_tree(state, old_leaves, new_leaves, basis):
    old_leaves = tuple(old_leaves)
    new_leaves = tuple(new_leaves)
    oldset, newset = set(old_leaves), set(new_leaves)
    if len(oldset) != len(old_leaves) or len(newset) != len(new_leaves):
        raise OfflineQuadAMRError('active Quad leaves must be unique')

    transfer = build_quad_refine_transfer(basis)
    oldidx = {leaf: i for i, leaf in enumerate(old_leaves)}
    result = {}

    def descend(leaf, values):
        if leaf in newset:
            result[leaf] = values
            return
        if len(leaf[1]) >= max(map(lambda x: len(x[1]), new_leaves)):
            raise OfflineQuadAMRError(
                f'proposed tree does not descend from active leaf {leaf}'
            )

        children = apply_quad_refine_transfer(
            transfer, values[:, :, None]
        )[..., 0]
        for quadrant in range(4):
            descend(
                (leaf[0], leaf[1] + (quadrant,)), children[quadrant]
            )

    for leaf in old_leaves:
        descend(leaf, state[:, :, oldidx[leaf]])

    if set(result) != newset:
        raise OfflineQuadAMRError(
            'proposed tree is not a monotone refinement of the current tree'
        )

    return np.stack([result[leaf] for leaf in new_leaves], axis=2)


def _quad_integral_weights(basis):
    qrule = get_quadrule(
        'quad', rule='gauss-legendre', npts=(basis.order + 2)**2
    )
    return qrule.wts @ basis.ubasis.nodal_basis_at(qrule.pts)


def _root_jacobians(mesh):
    src = QuadShape.std_ele(1)
    lhs = np.column_stack((src, np.ones(len(src))))
    geidxs = tuple(int(i) for i in mesh.eidxs['quad'])
    jac = {}

    for li, eidx in enumerate(geidxs):
        dst = np.asarray(mesh.spts['quad'][:, li, :], dtype=float)
        coeff, _, rank, _ = np.linalg.lstsq(lhs, dst, rcond=None)
        if rank != 3:
            raise OfflineQuadAMRError('root Quad geometry is degenerate')
        det = float(np.linalg.det(coeff[:2].T))
        if not np.isfinite(det) or det <= 0:
            raise OfflineQuadAMRError(
                'root Quad geometry has a nonpositive affine Jacobian'
            )
        jac[eidx] = det

    return jac


def _global_integral(root_mesh, leaves, state, basis):
    iwts = _quad_integral_weights(basis)
    jac = _root_jacobians(root_mesh)
    total = np.zeros(state.shape[1], dtype=np.result_type(state, float))

    for col, (root, path) in enumerate(leaves):
        try:
            scale = jac[root]/(4**len(path))
        except KeyError as exc:
            raise OfflineQuadAMRError(
                f'leaf references unknown root Quad {root}'
            ) from exc
        total += scale*np.einsum('i,iv->v', iwts, state[:, :, col])

    return total


def _write_solution(mesh, soln, state, outpath):
    outpath = Path(outpath)
    tcurr = soln.stats.getfloat('solver-time-integrator', 'tcurr')
    fields = tuple(soln.fields)

    writer = NativeWriter(
        mesh, soln.config, state.dtype, outpath.parent, outpath.name,
        'soln', isrestart=False,
    )
    writer.set_shapes_eidxs(
        {'quad': (state.shape[1], state.shape[0])}, mesh.eidxs,
        {'soln': fields},
    )

    metadata = {
        'config': soln.config.tostr(),
        'stats': soln.stats.tostr(),
        'mesh-uuid': mesh.uuid,
        **{name: cfg.tostr() for name, cfg in soln.prevcfgs.items()},
        **soln.state,
    }
    writer.write(
        {'quad': {'soln': state.transpose(2, 1, 0)}},
        tcurr, metadata, timeout=0,
    )
    writer.flush()


def perform_offline_quad_amr(
    current_mesh_path, current_soln_path, out_mesh_path, out_soln_path, *,
    root_mesh_path=None, refine_threshold, max_level,
    wall_boundaries=(), wall_min_level=0, density_floor=1e-14,
    acoustic_floor=1e-14, lintol=1e-5,
):
    """Perform one monotone offline Quad refinement event.

    Adapted inputs require ``root_mesh_path`` to reference the immutable
    original native root mesh whose UUID is stored in the persistent quadtree.
    The operation is intentionally single-rank and refinement-only.
    """
    comm, _, _ = get_comm_rank_root()
    if comm.size != 1:
        raise OfflineQuadAMRError(
            'offline Quad AMR currently requires one MPI rank'
        )

    if not np.isfinite(refine_threshold) or refine_threshold < 0:
        raise OfflineQuadAMRError(
            'refine threshold must be finite and nonnegative'
        )
    if not isinstance(max_level, (int, np.integer)) or max_level < 1:
        raise OfflineQuadAMRError('maximum level must be a positive integer')
    max_level = int(max_level)
    if (not isinstance(wall_min_level, (int, np.integer)) or
            not 0 <= wall_min_level <= max_level):
        raise OfflineQuadAMRError(
            'wall minimum level must be in 0..max_level'
        )
    wall_min_level = int(wall_min_level)
    wall_boundaries = tuple(wall_boundaries)

    out_mesh_path, out_soln_path = map(
        Path, (out_mesh_path, out_soln_path)
    )
    if out_mesh_path == out_soln_path:
        raise OfflineQuadAMRError(
            'offline AMR mesh and solution output paths must differ'
        )
    if out_mesh_path.exists() or out_soln_path.exists():
        raise OfflineQuadAMRError('offline AMR output path already exists')
    out_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    out_soln_path.parent.mkdir(parents=True, exist_ok=True)

    current_reader = NativeReader(str(current_mesh_path))
    root_reader = None
    out_reader = None
    success = False
    try:
        current_mesh = current_reader.mesh
        soln = current_reader.load_soln(str(current_soln_path))

        if current_mesh.amr_tree is None:
            if root_mesh_path is not None:
                root_reader = NativeReader(str(root_mesh_path))
                if root_reader.mesh.uuid != current_mesh.uuid:
                    raise OfflineQuadAMRError(
                        'unadapted current mesh and root anchor UUID differ'
                    )
            else:
                root_reader = current_reader
        else:
            if root_mesh_path is None:
                raise OfflineQuadAMRError(
                    'adapted Quad input requires the immutable root mesh'
                )
            root_reader = NativeReader(str(root_mesh_path))
            if root_reader.mesh.amr_tree is not None:
                raise OfflineQuadAMRError(
                    'immutable root mesh anchor must not itself be adapted'
                )
            if (root_reader.mesh.uuid !=
                    current_mesh.amr_tree.root_mesh_uuid):
                raise OfflineQuadAMRError('root anchor UUID mismatch')

        root_mesh = root_reader.mesh
        l2g, pids_by_g, _ = _quad_local_maps(root_mesh)
        roots = tuple(sorted(pids_by_g))
        rootfaces, rootkinds = _derive_quad_root_face_topology(
            root_mesh, pids_by_g, l2g
        )

        old_leaves = _current_column_leaves(current_mesh, roots)
        current_level = max((len(path) for _, path in old_leaves), default=0)
        if current_level > max_level:
            raise OfflineQuadAMRError(
                'maximum level is below the current adapted tree depth'
            )
        if {root for root, _ in old_leaves} != set(roots):
            raise OfflineQuadAMRError(
                'current Quad ancestry does not cover the immutable root'
            )

        if set(soln.data) != {'quad'}:
            raise OfflineQuadAMRError(
                'offline Quad AMR requires one uniform Quad solution group'
            )
        state = np.asarray(soln.data['quad'])
        if state.ndim != 3 or state.shape[2] != len(old_leaves):
            raise OfflineQuadAMRError(
                'current solution columns do not match active Quad leaves'
            )
        gamma = soln.config.getfloat('constants', 'gamma')
        irho, irhou, irhov, ienergy = _validate_state(
            state, soln.fields, gamma
        )
        scores = density_velocity_variation_scores(
            state, density_index=irho,
            momentum_indices=(irhou, irhov), energy_index=ienergy,
            gamma=gamma, density_floor=density_floor,
            acoustic_floor=acoustic_floor,
        )
        marks = {
            leaf for leaf, score in zip(old_leaves, scores)
            if len(leaf[1]) < max_level and score >= refine_threshold
        }

        wall_roots = set()
        for name in wall_boundaries:
            try:
                bcon = root_mesh.bcon[name]
            except KeyError as exc:
                raise OfflineQuadAMRError(
                    f'root mesh has no boundary {name!r}'
                ) from exc
            for etype, _, eidxs in bcon.items():
                if etype != 'quad':
                    raise OfflineQuadAMRError(
                        'wall minimum-level policy requires Quad boundaries'
                    )
                wall_roots.update(l2g[int(eidx)] for eidx in eidxs)

        split = _split_nodes(old_leaves)
        split.update(marks)
        split.update(_wall_floor_splits(wall_roots, wall_min_level))
        new_leaves, closure = _close_2to1(
            roots, split, rootfaces, rootkinds, max_level
        )
        new_leaves = tuple(new_leaves)
        if new_leaves == tuple(sorted(old_leaves)):
            raise OfflineQuadAMRError(
                'offline Quad AMR produced no refinement'
            )

        tree = encode_quad_leaf_tree(root_mesh.uuid, new_leaves)
        basis = QuadShape(None, soln.config)
        if basis.nupts != state.shape[0]:
            raise OfflineQuadAMRError(
                'solution point count does not match the configured Quad basis'
            )
        new_state = _prolongate_to_tree(
            state, old_leaves, new_leaves, basis
        )
        _validate_state(new_state, soln.fields, gamma)

        before = _global_integral(root_mesh, old_leaves, state, basis)
        after = _global_integral(root_mesh, new_leaves, new_state, basis)
        residual = float(np.max(np.abs(after - before), initial=0.0))
        scale = max(1.0, float(np.max(np.abs(before), initial=0.0)))
        tol = 65536*np.finfo(np.result_type(state, float)).eps*scale
        if residual > tol:
            raise OfflineQuadAMRError(
                'offline Quad prolongation failed conservation validation: '
                f'{residual} > {tol}'
            )

        raw = materialize_native_quad_tree(root_mesh, tree, comm_size=1)
        write_adapted_quad_mesh(raw, str(out_mesh_path), lintol=lintol)

        out_reader = NativeReader(str(out_mesh_path))
        out_mesh = out_reader.mesh
        if not isinstance(out_mesh.amr_tree, QuadLeafTree):
            raise OfflineQuadAMRError(
                'written adapted mesh did not recover typed Quad ancestry'
            )
        if tuple(out_mesh.amr_tree.leaves()) != new_leaves:
            raise OfflineQuadAMRError(
                'written adapted mesh ancestry changed leaf ordering'
            )

        _write_solution(out_mesh, soln, new_state, out_soln_path)
        written = out_reader.load_soln(str(out_soln_path))
        recovered = np.asarray(written.data['quad'])
        if not np.array_equal(recovered, new_state):
            raise OfflineQuadAMRError(
                'written adapted solution does not reload bit-for-bit'
            )
        _validate_state(recovered, written.fields, gamma)

        result = OfflineQuadAMRResult(
            len(old_leaves), len(new_leaves), len(marks), len(closure),
            float(np.max(scores, initial=0.0)), residual,
            str(out_mesh_path), str(out_soln_path),
        )
        success = True
        return result
    finally:
        if out_reader is not None:
            out_reader.close()
        if root_reader is not None and root_reader is not current_reader:
            root_reader.close()
        current_reader.close()
        if not success:
            out_mesh_path.unlink(missing_ok=True)
            out_soln_path.unlink(missing_ok=True)
