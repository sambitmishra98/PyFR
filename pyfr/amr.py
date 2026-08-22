from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np

from pyfr.quadrules import get_quadrule
from pyfr.shapes import HexShape, LineShape, QuadShape


@dataclass(frozen=True)
class HexRefineTransferKey:
    etype: str
    order: int
    soln_pts: str
    template: str = 'octree-2x2x2-v1'


@dataclass(frozen=True, eq=False)
class HexRefineTransfer:
    key: HexRefineTransferKey
    interp: np.ndarray

    @property
    def order(self):
        return self.key.order

    @property
    def nupts(self):
        return self.interp.shape[1]


def hex_child_to_parent(octant, pts):
    """Map child Hex reference points into a parent octant."""
    if not isinstance(octant, (int, np.integer)) or not 0 <= octant < 8:
        raise ValueError('Hex refinement octant must be an integer in 0..7')

    pts = np.asarray(pts)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('Hex reference points must have shape (npts, 3)')

    shift = np.array([
        1 if octant & 1 else -1,
        1 if octant & 2 else -1,
        1 if octant & 4 else -1,
    ], dtype=pts.dtype)

    return 0.5*(pts + shift)


_hex_refine_transfer_cache = {}


def build_hex_refine_transfer(basis):
    """Build exact same-order parent-to-octant Hex interpolation."""
    if basis.name != 'hex':
        raise ValueError('Octree prolongation requires a Hex basis')

    soln_pts = basis.cfg.get('solver-elements-hex', 'soln-pts')
    key = HexRefineTransferKey(basis.name, basis.order, soln_pts)

    try:
        return _hex_refine_transfer_cache[key]
    except KeyError:
        pass

    interp = np.stack([
        basis.ubasis.nodal_basis_at(hex_child_to_parent(octant, basis.upts))
        for octant in range(8)
    ])
    interp.setflags(write=False)

    transfer = HexRefineTransfer(key, interp)
    _hex_refine_transfer_cache[key] = transfer
    return transfer


def apply_hex_refine_transfer(transfer, parent):
    """Prolongate parent conservative solution banks to eight children.

    Parameters
    ----------
    transfer : HexRefineTransfer
        Same-order parent-to-child interpolation operators.
    parent : ndarray
        Array with shape ``(nupts, nvars, nparents)``.

    Returns
    -------
    ndarray
        Array with shape ``(8, nupts, nvars, nparents)``.  Axis zero uses
        the D2 octant identity ``ix + 2*iy + 4*iz``.
    """
    parent = np.asarray(parent)
    if parent.ndim != 3:
        raise ValueError(
            'Parent solution must have shape (nupts, nvars, nparents)'
        )
    if parent.shape[0] != transfer.nupts:
        raise ValueError('Parent solution point count does not match transfer')

    return np.einsum('oji,ivn->ojvn', transfer.interp, parent, optimize=True)


@dataclass(frozen=True)
class QuadRefineTransferKey:
    etype: str
    order: int
    soln_pts: str
    template: str = 'quadtree-2x2-v1'


@dataclass(frozen=True, eq=False)
class QuadRefineTransfer:
    key: QuadRefineTransferKey
    interp: np.ndarray

    @property
    def order(self):
        return self.key.order

    @property
    def nupts(self):
        return self.interp.shape[1]


def quad_child_to_parent(quadrant, pts):
    """Map child Quad reference points into a parent quadrant."""
    if not isinstance(quadrant, (int, np.integer)) or not 0 <= quadrant < 4:
        raise ValueError('Quad refinement quadrant must be an integer in 0..3')

    pts = np.asarray(pts)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError('Quad reference points must have shape (npts, 2)')

    shift = np.array([
        1 if quadrant & 1 else -1,
        1 if quadrant & 2 else -1,
    ], dtype=pts.dtype)

    return 0.5*(pts + shift)


_quad_refine_transfer_cache = {}


def build_quad_refine_transfer(basis):
    """Build exact same-order parent-to-quadrant Quad interpolation."""
    if basis.name != 'quad':
        raise ValueError('Quadtree prolongation requires a Quad basis')

    soln_pts = basis.cfg.get('solver-elements-quad', 'soln-pts')
    key = QuadRefineTransferKey(basis.name, basis.order, soln_pts)

    try:
        return _quad_refine_transfer_cache[key]
    except KeyError:
        pass

    interp = np.stack([
        basis.ubasis.nodal_basis_at(quad_child_to_parent(quadrant, basis.upts))
        for quadrant in range(4)
    ])
    interp.setflags(write=False)

    transfer = QuadRefineTransfer(key, interp)
    _quad_refine_transfer_cache[key] = transfer
    return transfer


def apply_quad_refine_transfer(transfer, parent):
    """Prolongate parent conservative solution banks to four children.

    Parameters
    ----------
    transfer : QuadRefineTransfer
        Same-order parent-to-child interpolation operators.
    parent : ndarray
        Array with shape ``(nupts, nvars, nparents)``.

    Returns
    -------
    ndarray
        Array with shape ``(4, nupts, nvars, nparents)``. Axis zero uses
        the quadtree quadrant identity ``ix + 2*iy``.
    """
    parent = np.asarray(parent)
    if parent.ndim != 3:
        raise ValueError(
            'Parent solution must have shape (nupts, nvars, nparents)'
        )
    if parent.shape[0] != transfer.nupts:
        raise ValueError('Parent solution point count does not match transfer')

    return np.einsum('oji,ivn->ojvn', transfer.interp, parent, optimize=True)


@dataclass(frozen=True, eq=False)
class HexRestrictTransfer:
    """Same-order affine Hex eight-child -> parent L2 restriction.

    Reuses the accepted D3 parent -> child interpolation `T_o` (`refine`)
    and its octant identity rather than inventing an independent
    convention.  ``restrict[o]`` is ``R_o = (1/8) M^-1 T_o^T M``, the exact
    L2 projection for an affine octree child whose reference-to-parent
    Jacobian is ``1/8``.
    """

    key: HexRefineTransferKey
    restrict: np.ndarray
    mass: np.ndarray
    refine: HexRefineTransfer

    @property
    def order(self):
        return self.key.order

    @property
    def nupts(self):
        return self.restrict.shape[1]


_hex_restrict_transfer_cache = {}


def build_hex_restrict_transfer(basis):
    """Build the exact same-order affine Hex child -> parent restriction.

    The reference Hex mass matrix for the nodal solution basis is built
    from ``(order + 1)**3`` tensor Gauss-Legendre points, which integrate
    products of ``Q_p`` nodal basis functions (degree ``2*order`` per
    dimension) exactly.
    """
    refine = build_hex_refine_transfer(basis)

    try:
        return _hex_restrict_transfer_cache[refine.key]
    except KeyError:
        pass

    order = basis.order
    qrule = get_quadrule('hex', rule='gauss-legendre', npts=(order + 1)**3)
    vq = basis.ubasis.nodal_basis_at(qrule.pts)
    mass = vq.T @ (qrule.wts[:, None]*vq)

    # R_o = (1/8) M^-1 T_o^T M, via numpy.linalg.solve (never an explicit
    # inverse).  The loop below runs over the fixed 8 octants at
    # construction time only; it is not a loop over parent families.
    restrict = np.stack([
        np.linalg.solve(mass, refine.interp[o].T @ mass)/8
        for o in range(8)
    ])
    restrict.setflags(write=False)
    mass.setflags(write=False)

    transfer = HexRestrictTransfer(refine.key, restrict, mass, refine)
    _hex_restrict_transfer_cache[refine.key] = transfer
    return transfer


def apply_hex_restrict_transfer(transfer, children):
    """Restrict eight child conservative solution banks to their parent.

    Parameters
    ----------
    transfer : HexRestrictTransfer
        Same-order child-to-parent L2 restriction operators.
    children : ndarray
        Array with shape ``(8, nupts, nvars, nparents)``.  Axis zero uses
        the D2 octant identity ``ix + 2*iy + 4*iz``, matching
        ``apply_hex_refine_transfer``.

    Returns
    -------
    ndarray
        Array with shape ``(nupts, nvars, nparents)``.
    """
    children = np.asarray(children)
    if children.ndim != 4:
        raise ValueError(
            'Child solution must have shape (8, nupts, nvars, nparents)'
        )
    if children.shape[0] != 8:
        raise ValueError('Child solution must carry exactly 8 octants')
    if children.shape[1] != transfer.nupts:
        raise ValueError(
            'Child solution point count does not match transfer'
        )

    return np.einsum(
        'oij,ojvn->ivn', transfer.restrict, children, optimize=True
    )


def hex_restriction_projection_loss(transfer, children, parent):
    """Per (var, parent) L2 projection-loss diagnostic.

    ``E^2 = sum_o (1/8) r_o^T M r_o``, with ``r_o = U_C,o - T_o U_P``. This
    is roundoff-zero for a child state generated by
    ``apply_hex_refine_transfer`` from a representable parent, and positive
    for a genuinely incompatible piecewise child field. It is a diagnostic
    only, not an acceptance criterion for whether coarsening is physically
    safe.
    """
    children = np.asarray(children)
    parent = np.asarray(parent)
    if children.ndim != 4 or children.shape[0] != 8:
        raise ValueError(
            'Child solution must have shape (8, nupts, nvars, nparents)'
        )
    if children.shape[1] != transfer.nupts:
        raise ValueError(
            'Child solution point count does not match transfer'
        )
    if parent.ndim != 3 or parent.shape[0] != transfer.nupts:
        raise ValueError(
            'Parent solution must have shape (nupts, nvars, nparents)'
        )

    reproj = np.einsum(
        'oji,ivn->ojvn', transfer.refine.interp, parent, optimize=True
    )
    residual = children - reproj
    return np.einsum(
        'oivn,ij,ojvn->vn', residual, transfer.mass, residual,
        optimize=True
    )/8


def hex_octree_parent(leaf):
    """Return the parent leaf identity, or ``None`` at a tree root.

    A leaf identity is ``(root, path)`` where ``path`` is a tuple of
    octant indices (D2 identity ``ix + 2*iy + 4*iz``) from the root to the
    leaf.  This is a pure topology identity; no persisted tree ABI or
    mesh coupling is implied (that is D5's responsibility).
    """
    root, path = leaf
    if not path:
        return None
    return root, path[:-1]


def hex_octree_children(parent):
    """Return the eight direct child leaf identities of ``parent``."""
    root, path = parent
    return {(root, path + (o,)) for o in range(8)}


def hex_octree_level(leaf):
    """Return the refinement level (root is level 0) of ``leaf``."""
    return len(leaf[1])


def hex_coarsen_candidates(leaves, marked, face_pairs):
    """Select a deterministic, 2:1-legal one-level sibling-collapse set.

    Pure topology only: no mesh mutation, no persisted tree/restart
    metadata, no solution transfer, no MPI/repartition, no backend
    rebuild. Those remain D5/D6/D7 responsibilities.

    Parameters
    ----------
    leaves : iterable of (root, path)
        The CURRENT active leaf set.
    marked : iterable of (root, path)
        Leaves proposed for coarsening. Leaves not in ``leaves`` are
        ignored.
    face_pairs : iterable of (leaf, leaf)
        The CURRENT balanced physical face-neighbour relation between
        active leaves, including cross-root neighbours.

    Returns
    -------
    tuple of (root, path)
        Deterministically ordered (independent of input mark/pair
        ordering) parent identities whose eight direct children are
        legally collapsed by this transaction: all eight are active
        leaves, all eight are marked, and the collapse does not violate
        2:1 balance under the current face adjacency once internal
        sibling faces are ignored. At most one level per lineage is
        collapsed in a single call; a fully marked deeper family never
        collapses its grandparent here.

    Raises
    ------
    ValueError
        If ``face_pairs`` references a leaf outside ``leaves``, or a
        pre-existing >1-level face imbalance exists whose coarse side is
        not itself a coarsening candidate. Such input is fail-closed
        rather than silently repaired.
    """
    leaves = set(leaves)
    marks = set(marked) & leaves
    face_pairs = list(face_pairs)

    candidates = set()
    for leaf in marks:
        parent = hex_octree_parent(leaf)
        if parent is not None:
            siblings = hex_octree_children(parent)
            if siblings <= leaves and siblings <= marks:
                candidates.add(parent)

    approved = set(candidates)
    while True:
        def mapped(leaf):
            parent = hex_octree_parent(leaf)
            return parent if parent in approved else leaf

        rejected = set()
        for lhs, rhs in face_pairs:
            if lhs not in leaves or rhs not in leaves:
                raise ValueError(
                    'face pair references a leaf outside the current '
                    'active leaf set'
                )
            ml, mr = mapped(lhs), mapped(rhs)
            if ml == mr:
                continue

            dl, dr = hex_octree_level(ml), hex_octree_level(mr)
            if abs(dl - dr) <= 1:
                continue

            coarse = ml if dl < dr else mr
            if coarse not in approved:
                raise ValueError(
                    'pre-existing leaf face adjacency is not 2:1 '
                    'balanced and no proposed coarsening is responsible'
                )
            rejected.add(coarse)

        if not rejected:
            return tuple(sorted(approved, key=repr))
        approved -= rejected


AMR_LEAF_TREE_VERSION = 1


@dataclass(frozen=True, eq=False)
class HexLeafTree:
    """Immutable native octree active-leaf identity codec.

    Runtime leaf identity is ``(root_mesh_uuid, root_hex_global_eidx,
    octant_path)``, matching the accepted D2 octant convention
    ``ix + 2*iy + 4*iz``.  ``root_mesh_uuid`` and ``version`` belong to
    the tree, not to each leaf.  Canonical ordering is lexicographic by
    ``(root_eidx, octant_path)``.

    Persisted schema (reference R1):
    - ``amr/version = 1``
    - ``amr/root-mesh-uuid``
    - ``amr/leaves/root-eidx``     little-endian int64, one per leaf
    - ``amr/leaves/path-offsets``  little-endian int64, length nleaves+1
    - ``amr/leaves/path-data``     uint8, flattened octants
    """

    version: int
    root_mesh_uuid: str
    root_eidx: np.ndarray
    path_offsets: np.ndarray
    path_data: np.ndarray

    @property
    def nleaves(self):
        return self.root_eidx.shape[0]

    def leaves(self):
        """Yield each active leaf as ``(root_eidx, path)`` in canonical
        order."""
        for i in range(self.nleaves):
            s, e = self.path_offsets[i], self.path_offsets[i + 1]
            yield (
                int(self.root_eidx[i]),
                tuple(int(o) for o in self.path_data[s:e]),
            )


def _canonicalize_hex_leaves(leaves):
    """Validate and canonically order a set of ``(root_eidx, path)``
    active-leaf identities.  Fails closed on any structurally invalid
    input; see module docstring gates for the exact rejected cases.
    """
    seen = set()
    by_root = {}
    for root_eidx, path in leaves:
        if not isinstance(root_eidx, (int, np.integer)) or root_eidx < 0:
            raise ValueError(
                f'invalid root Hex global element index {root_eidx!r}'
            )
        root_eidx = int(root_eidx)
        path = tuple(int(o) for o in path)
        for o in path:
            if not 0 <= o < 8:
                raise ValueError(
                    f'octant {o} outside 0..7 in path for root {root_eidx}'
                )

        key = (root_eidx, path)
        if key in seen:
            raise ValueError(f'duplicate active leaf {key}')
        seen.add(key)
        by_root.setdefault(root_eidx, set()).add(path)

    for root_eidx, paths in by_root.items():
        for p in paths:
            for k in range(len(p)):
                if p[:k] in paths:
                    raise ValueError(
                        f'active ancestor (root={root_eidx}, path={p[:k]}) '
                        f'and active descendant (root={root_eidx}, '
                        f'path={p}) cannot both be active leaves'
                    )

        # Every node implied as a strict prefix of some leaf path (i.e.
        # every internal split node) must have exactly 8 children also
        # implied; a 1..7-of-8 count is an incomplete eight-way split.
        implied = set()
        for p in paths:
            for k in range(len(p) + 1):
                implied.add(p[:k])
        for node in implied:
            if node in paths:
                continue
            nchildren = sum(
                1 for o in range(8) if (node + (o,)) in implied
            )
            if nchildren != 8:
                raise ValueError(
                    f'incomplete eight-way split at root {root_eidx} '
                    f'node {node}: {nchildren}/8 children present'
                )

    return sorted(seen)


def encode_hex_leaf_tree(root_mesh_uuid, leaves, version=AMR_LEAF_TREE_VERSION):
    """Build a validated, canonically ordered, read-only
    :class:`HexLeafTree` from an iterable of ``(root_eidx, path)`` active
    leaves.
    """
    if version != AMR_LEAF_TREE_VERSION:
        raise ValueError(f'unsupported amr leaf-tree version {version}')
    if not root_mesh_uuid:
        raise ValueError('root mesh UUID must be non-empty')

    canonical = _canonicalize_hex_leaves(leaves)

    root_eidx = np.array([r for r, _ in canonical], dtype=np.int64)
    lens = np.array([len(p) for _, p in canonical], dtype=np.int64)
    path_offsets = np.zeros(len(canonical) + 1, dtype=np.int64)
    if len(lens):
        np.cumsum(lens, out=path_offsets[1:])
    path_data = np.array(
        [o for _, p in canonical for o in p], dtype=np.uint8
    )

    root_eidx.setflags(write=False)
    path_offsets.setflags(write=False)
    path_data.setflags(write=False)

    return HexLeafTree(version, root_mesh_uuid, root_eidx, path_offsets,
                        path_data)


def hex_leaf_tree_from_arrays(version, root_mesh_uuid, root_eidx,
                               path_offsets, path_data):
    """Reconstruct and fully validate a :class:`HexLeafTree` from raw
    persisted arrays (e.g. read back from HDF5).  Malformed input -
    including corrupt or tampered offsets/shapes - fails closed.
    """
    root_eidx = np.asarray(root_eidx)
    path_offsets = np.asarray(path_offsets)
    path_data = np.asarray(path_data)

    if version != AMR_LEAF_TREE_VERSION:
        raise ValueError(f'unsupported amr leaf-tree version {version}')
    if root_eidx.ndim != 1 or path_offsets.ndim != 1 or path_data.ndim != 1:
        raise ValueError('malformed leaf-tree array shapes')
    if path_offsets.shape[0] != root_eidx.shape[0] + 1:
        raise ValueError('path-offsets length must be nleaves + 1')
    if path_offsets.shape[0] and path_offsets[0] != 0:
        raise ValueError('path-offsets must start at 0')
    if np.any(np.diff(path_offsets) < 0):
        raise ValueError('path-offsets must be non-decreasing')
    if path_offsets.shape[0] and path_offsets[-1] != path_data.shape[0]:
        raise ValueError(
            'path-offsets final entry must equal path-data length'
        )
    if path_data.size and (int(path_data.min()) < 0
                            or int(path_data.max()) > 7):
        raise ValueError('path-data octants must be in 0..7')

    leaves = [
        (int(root_eidx[i]),
         tuple(int(o) for o in path_data[path_offsets[i]:path_offsets[i + 1]]))
        for i in range(root_eidx.shape[0])
    ]
    # Re-run full semantic validation (duplicates / ancestor+descendant /
    # incomplete splits) through the single source of truth used by
    # encode_hex_leaf_tree, rather than duplicating it here.
    return encode_hex_leaf_tree(root_mesh_uuid, leaves, version=version)


@dataclass(frozen=True, eq=False)
class QuadLeafTree:
    """Immutable native quadtree active-leaf identity codec.

    Runtime leaf identity is ``(root_mesh_uuid, root_quad_global_eidx,
    quadrant_path)``, using the accepted 2D quadrant convention
    ``ix + 2*iy``. ``root_mesh_uuid`` and ``version`` belong to the tree,
    not to each leaf. Canonical ordering is lexicographic by
    ``(root_eidx, quadrant_path)``.
    """

    version: int
    root_mesh_uuid: str
    root_eidx: np.ndarray
    path_offsets: np.ndarray
    path_data: np.ndarray

    @property
    def nleaves(self):
        return self.root_eidx.shape[0]

    def leaves(self):
        """Yield each active leaf as ``(root_eidx, path)`` in canonical
        order."""
        for i in range(self.nleaves):
            s, e = self.path_offsets[i], self.path_offsets[i + 1]
            yield (
                int(self.root_eidx[i]),
                tuple(int(q) for q in self.path_data[s:e]),
            )


def _canonicalize_quad_leaves(leaves):
    """Validate and canonically order ``(root_eidx, path)`` active leaves."""
    seen = set()
    by_root = {}
    for root_eidx, path in leaves:
        if not isinstance(root_eidx, (int, np.integer)) or root_eidx < 0:
            raise ValueError(
                f'invalid root Quad global element index {root_eidx!r}'
            )
        root_eidx = int(root_eidx)
        path = tuple(int(q) for q in path)
        for q in path:
            if not 0 <= q < 4:
                raise ValueError(
                    f'quadrant {q} outside 0..3 in path for root {root_eidx}'
                )

        key = (root_eidx, path)
        if key in seen:
            raise ValueError(f'duplicate active leaf {key}')
        seen.add(key)
        by_root.setdefault(root_eidx, set()).add(path)

    for root_eidx, paths in by_root.items():
        for p in paths:
            for k in range(len(p)):
                if p[:k] in paths:
                    raise ValueError(
                        f'active ancestor (root={root_eidx}, path={p[:k]}) '
                        f'and active descendant (root={root_eidx}, '
                        f'path={p}) cannot both be active leaves'
                    )

        implied = set()
        for p in paths:
            for k in range(len(p) + 1):
                implied.add(p[:k])
        for node in implied:
            if node in paths:
                continue
            nchildren = sum(
                1 for q in range(4) if (node + (q,)) in implied
            )
            if nchildren != 4:
                raise ValueError(
                    f'incomplete four-way split at root {root_eidx} '
                    f'node {node}: {nchildren}/4 children present'
                )

    return sorted(seen)


def encode_quad_leaf_tree(root_mesh_uuid, leaves,
                          version=AMR_LEAF_TREE_VERSION):
    """Build a validated, canonically ordered, read-only Quad leaf tree."""
    if version != AMR_LEAF_TREE_VERSION:
        raise ValueError(f'unsupported amr leaf-tree version {version}')
    if not root_mesh_uuid:
        raise ValueError('root mesh UUID must be non-empty')

    canonical = _canonicalize_quad_leaves(leaves)

    root_eidx = np.array([r for r, _ in canonical], dtype=np.int64)
    lens = np.array([len(p) for _, p in canonical], dtype=np.int64)
    path_offsets = np.zeros(len(canonical) + 1, dtype=np.int64)
    if len(lens):
        np.cumsum(lens, out=path_offsets[1:])
    path_data = np.array(
        [q for _, p in canonical for q in p], dtype=np.uint8
    )

    root_eidx.setflags(write=False)
    path_offsets.setflags(write=False)
    path_data.setflags(write=False)

    return QuadLeafTree(version, root_mesh_uuid, root_eidx, path_offsets,
                        path_data)


def quad_leaf_tree_from_arrays(version, root_mesh_uuid, root_eidx,
                               path_offsets, path_data):
    """Reconstruct and fully validate a :class:`QuadLeafTree` from arrays."""
    root_eidx = np.asarray(root_eidx)
    path_offsets = np.asarray(path_offsets)
    path_data = np.asarray(path_data)

    if version != AMR_LEAF_TREE_VERSION:
        raise ValueError(f'unsupported amr leaf-tree version {version}')
    if root_eidx.ndim != 1 or path_offsets.ndim != 1 or path_data.ndim != 1:
        raise ValueError('malformed leaf-tree array shapes')
    if path_offsets.shape[0] != root_eidx.shape[0] + 1:
        raise ValueError('path-offsets length must be nleaves + 1')
    if path_offsets.shape[0] and path_offsets[0] != 0:
        raise ValueError('path-offsets must start at 0')
    if np.any(np.diff(path_offsets) < 0):
        raise ValueError('path-offsets must be non-decreasing')
    if path_offsets.shape[0] and path_offsets[-1] != path_data.shape[0]:
        raise ValueError(
            'path-offsets final entry must equal path-data length'
        )
    if path_data.size and (int(path_data.min()) < 0
                           or int(path_data.max()) > 3):
        raise ValueError('path-data quadrants must be in 0..3')

    leaves = [
        (int(root_eidx[i]),
         tuple(int(q) for q in path_data[path_offsets[i]:path_offsets[i + 1]]))
        for i in range(root_eidx.shape[0])
    ]
    return encode_quad_leaf_tree(root_mesh_uuid, leaves, version=version)


# ===========================================================================
# D5B1 - shared pure Hex-octree core.
#
# This is the octree mathematics extracted from the accepted D2 Gmsh
# refinement path (`pyfr.readers.gmsh.GmshReader.refine_hexes` and its
# helpers), made independent of Gmsh-specific state so both the Gmsh
# importer and a native materializer call exactly one implementation.
#
# Import-time-only responsibilities are deliberately NOT here: selector
# parsing, ancestor-implication/split-set planning, the 2:1 *closure*
# mutation loop, and Gmsh boundary-element reconstruction all remain in
# `GmshReader`. This module only supplies the reusable geometry/topology
# primitives; callers own their own orchestration.
#
# All functions here operate on node ids already in PyFR canonical Hex8
# node order (`HexShape` ordering). A caller reading a format whose node
# order differs (e.g. Gmsh) is responsible for permuting at its own
# boundary; a caller whose data is already canonical (a native PyFR mesh)
# passes it straight through.
# ===========================================================================

HEX_QUAD_FNMAP = (
    (0, 1, 2, 3), (0, 1, 4, 5), (1, 2, 5, 6),
    (2, 3, 6, 7), (0, 3, 4, 7), (4, 5, 6, 7),
)


@lru_cache(maxsize=None)
def hex_face_corner_indices(fidx):
    """The four canonical Hex8 vertex indices bounding face ``fidx``, in
    the same corner order as :class:`~pyfr.shapes.QuadShape`."""
    hpts = HexShape.std_ele(1)
    qpts = QuadShape.std_ele(1)
    project = HexShape.faces[fidx][1]
    fpts = np.asarray([project(*p) for p in qpts])

    idxs = []
    for p in fpts:
        match = np.flatnonzero(np.all(np.isclose(hpts, p), axis=1))
        if len(match) != 1:
            raise ValueError('Unable to identify Hex face corners')
        idxs.append(int(match[0]))

    return tuple(idxs)


def hex_tree_cell_index(path):
    """Map an octant path to its dyadic ``(level, (i, j, k))`` cell
    index, using the accepted D2 octant convention
    ``ix + 2*iy + 4*iz``."""
    i = j = k = 0
    for octant in path:
        i = 2*i + (octant & 1)
        j = 2*j + ((octant >> 1) & 1)
        k = 2*k + ((octant >> 2) & 1)

    return len(path), (i, j, k)


def hex_face_axis_side(fidx):
    """Return ``(axis, is_high_side)`` for canonical Hex8 face
    ``fidx``."""
    return (
        (2, 0), (1, 0), (0, 1),
        (1, 1), (0, 0), (2, 1),
    )[fidx]


def hex_root_face_uv(pids, fidx):
    """Integer in-plane ``(u, v) in {0, 1}^2`` corner coordinates for each
    node on face ``fidx``, keyed by node id. ``pids`` are the element's
    node ids in PyFR canonical Hex8 order."""
    pids = np.asarray(pids)
    hpts = HexShape.std_ele(1)
    axis, _ = hex_face_axis_side(fidx)
    taxes = tuple(d for d in range(3) if d != axis)

    uv = {}
    for cidx in hex_face_corner_indices(fidx):
        node = int(pids[cidx])
        uv[node] = tuple(
            int(round((hpts[cidx, d] + 1)/2)) for d in taxes
        )

    return uv


def hex_d4_transform(src, dst):
    """Derive the affine D4 orientation transform mapping ``src`` corner
    ``(u, v)`` coordinates to ``dst`` corner ``(u, v)`` coordinates, both
    keyed by the shared node id."""
    mapping = {uv: dst[node] for node, uv in src.items()}
    try:
        c = mapping[0, 0]
        pu = mapping[1, 0]
        pv = mapping[0, 1]
        puv = mapping[1, 1]
    except KeyError:
        raise ValueError('Invalid Hex root-face corner orientation') \
            from None

    au, av = pu[0] - c[0], pv[0] - c[0]
    bu, bv = pu[1] - c[1], pv[1] - c[1]
    if (c[0] + au + av, c[1] + bu + bv) != puv:
        raise ValueError('Non-affine Hex root-face orientation')

    return au, av, c[0], bu, bv, c[1]


@lru_cache(maxsize=4096)
def _hex_transform_rect_cached(rect, transform):
    u0, u1, v0, v1 = rect
    au, av, cu, bu, bv, cv = transform
    points = [
        (au*u + av*v + cu, bu*u + bv*v + cv)
        for u in (u0, u1) for v in (v0, v1)
    ]
    us, vs = zip(*points)

    return min(us), max(us), min(vs), max(vs)


def hex_transform_rect(rect, transform):
    """Apply a D4 transform to a dyadic face rectangle
    ``(u0, u1, v0, v1)``."""
    return _hex_transform_rect_cached(tuple(rect), tuple(transform))


def hex_rect_overlap(a, b):
    """Whether two open dyadic rectangles ``(u0, u1, v0, v1)`` overlap."""
    return (
        max(a[0], b[0]) < min(a[1], b[1]) and
        max(a[2], b[2]) < min(a[3], b[3])
    )


def hex_tree_leaves(roots, split):
    """Recursively expand ``split`` (a set of ``(root, path)`` internal
    split markers) into the ordered list of active ``(root, path)``
    leaves. ``roots`` is the iterable of distinct root identities, sorted
    in their natural (e.g. numeric) order - matching the accepted D2
    behaviour exactly, since callers may rely on the resulting leaf
    order (e.g. for deterministic generated-tag assignment)."""
    leaves = []

    def walk(root, path):
        if (root, path) in split:
            for octant in range(8):
                walk(root, path + (octant,))
        else:
            leaves.append((root, path))

    for root in sorted(roots):
        walk(root, ())

    return leaves


def hex_tree_face_groups(leaves, rootfaces):
    """Group every leaf face by abstract surface identity: a root
    boundary (``('root', qkey)``) or an internal dyadic split plane
    (``('internal', root, axis, plane)``).  ``rootfaces`` maps
    ``(root, fidx) -> {'qkey', 'transform', ...}`` for each root's six
    faces; ``transform`` is the D4 transform (identity for an
    unshared/boundary face)."""
    groups = defaultdict(lambda: defaultdict(list))

    for leaf in leaves:
        root, path = leaf
        level, idx = hex_tree_cell_index(path)
        den = 1 << level
        lo = tuple(Fraction(i, den) for i in idx)
        hi = tuple(Fraction(i + 1, den) for i in idx)

        for fidx in range(6):
            axis, high = hex_face_axis_side(fidx)
            plane = hi[axis] if high else lo[axis]
            taxes = tuple(d for d in range(3) if d != axis)
            rect = (
                lo[taxes[0]], hi[taxes[0]],
                lo[taxes[1]], hi[taxes[1]],
            )

            atroot = plane == (1 if high else 0)
            if atroot:
                face = rootfaces[root, fidx]
                surface = ('root', face['qkey'])
                side = root
                rect = hex_transform_rect(rect, face['transform'])
            else:
                surface = ('internal', root, axis, plane)
                side = high

            groups[surface][side].append({
                'leaf': leaf, 'fidx': fidx, 'level': level,
                'rect': rect, 'surface': surface,
            })

    return groups


def hex_tree_face_pairs(groups, rootkinds):
    """Yield ``(lface, rface)`` descriptor pairs for every physically
    adjacent leaf-face pair implied by ``groups`` (as produced by
    :func:`hex_tree_face_groups`). ``rootkinds`` maps
    ``('root', qkey) -> kind`` so an incomplete shared root surface can be
    distinguished from an incomplete-but-external one.  Raises on
    incomplete internal faces, incomplete shared root faces, or
    non-manifold (more than two sides) surfaces."""
    for surface in sorted(groups, key=repr):
        sides = groups[surface]
        if len(sides) == 1:
            if surface[0] == 'internal':
                raise ValueError('Incomplete internal Hex tree face')
            if rootkinds.get(surface) == 'interior':
                raise ValueError('Incomplete shared Hex root face')
            continue
        if len(sides) != 2:
            raise ValueError('Non-manifold Hex tree face')

        skeys = sorted(sides, key=repr)
        lhs = sorted(
            sides[skeys[0]], key=lambda d: (d['leaf'], d['fidx'])
        )
        rhs = sorted(
            sides[skeys[1]], key=lambda d: (d['leaf'], d['fidx'])
        )
        for lface in lhs:
            for rface in rhs:
                if hex_rect_overlap(lface['rect'], rface['rect']):
                    yield lface, rface


class HexNodeStore:
    """Injectable node-coordinate store/allocator.

    Both the Gmsh importer and a native materializer supply one of these
    so the shared subdivision code below uses exactly one node-allocation
    algorithm, preserving D2's cross-root shared-node behaviour rather
    than reimplementing it per caller.
    """

    def __init__(self, coords, allocate):
        """
        Parameters
        ----------
        coords : callable(node_ids) -> (n, 3) float ndarray
            Physical coordinates of existing node ids.
        allocate : callable(points) -> (n,) int64 ndarray
            Allocate new node ids for the given ``(n, 3)`` physical
            points, returning their assigned ids.
        """
        self._coords = coords
        self._allocate = allocate

    def coords(self, node_ids):
        return self._coords(node_ids)

    def allocate(self, points):
        return self._allocate(points)


def hex_affine_map(pids, store, tol=1e-10):
    """Validate affine, positive-Jacobian Hex8 geometry for the element
    with canonical-order node ids ``pids``, and return an evaluator
    mapping reference points to physical coordinates."""
    pids = np.asarray(pids)
    src = HexShape.std_ele(1)
    dst = store.coords(pids)
    lhs = np.column_stack((src, np.ones(len(src))))
    coeff, _, rank, _ = np.linalg.lstsq(lhs, dst, rcond=None)

    if rank != 4:
        raise ValueError('Degenerate Hex8 refinement geometry')

    fitted = lhs @ coeff
    scale = max(1.0, float(np.max(np.abs(dst), initial=0.0)))
    error = np.max(np.abs(fitted - dst), initial=0.0)
    if error > tol*scale:
        raise ValueError(
            'V10D1 Hex refinement requires affine Hex8 geometry'
        )

    jac = coeff[:3].T
    if np.linalg.det(jac) <= tol*scale**3:
        raise ValueError(
            'V10D1 Hex refinement requires positive affine geometry'
        )

    def apply(points):
        points = np.asarray(points, dtype=float)
        plhs = np.column_stack((points, np.ones(len(points))))
        return plhs @ coeff

    return apply


def hex_refined_children(pids, store, node_cache, coord_cache):
    """Recursively subdivide one affine Hex8 (canonical-order node ids
    ``pids``) into its eight D2-octant children.

    New subdivision nodes are shared with previously processed
    neighbours via ``node_cache``/``coord_cache`` (both caller-owned,
    keyed by the sorted tuple of coincident parent-node support),
    reproducing D2's cross-root shared-node behaviour exactly.

    Returns ``[(ix, iy, iz, child_pids), ...]`` with each ``child_pids``
    in PyFR canonical Hex8 order.
    """
    amap = hex_affine_map(pids, store)
    pids = np.asarray(pids)
    hpts = HexShape.std_ele(1)
    refs = (-1.0, 0.0, 1.0)
    grid = {}

    for z in refs:
        for y in refs:
            for x in refs:
                ref = np.array([x, y, z])
                mask = np.ones(len(hpts), dtype=bool)
                for d, value in enumerate(ref):
                    if value:
                        mask &= np.isclose(hpts[:, d], value)

                support = tuple(sorted(int(i) for i in pids[mask]))
                if len(support) == 1:
                    node = support[0]
                elif support in node_cache:
                    node = node_cache[support]
                    coord = amap([ref])[0]
                    if not np.allclose(
                        coord_cache[support], coord,
                        rtol=0, atol=2e-10*max(1.0, abs(coord).max())
                    ):
                        raise ValueError(
                            'Refined Hex neighbours disagree on shared '
                            'subdivision-node geometry'
                        )
                else:
                    coord = amap([ref])[0]
                    node = int(store.allocate(coord[None])[0])
                    node_cache[support] = node
                    coord_cache[support] = coord

                grid[tuple(ref)] = node

    children = []
    for iz in range(2):
        for iy in range(2):
            for ix in range(2):
                centre = np.array([
                    -0.5 if ix == 0 else 0.5,
                    -0.5 if iy == 0 else 0.5,
                    -0.5 if iz == 0 else 0.5,
                ])
                crefs = 0.5*hpts + centre
                row = np.array(
                    [grid[tuple(p)] for p in crefs], dtype=np.int64
                )
                children.append((ix, iy, iz, row))

    return children


def hex_child_face_nodes(children, fidx, fmap):
    """Given the eight ``(ix, iy, iz, pids)`` children of one refined
    Hex8 (as returned by :func:`hex_refined_children`), return the four
    child face-node tuples lying on Hex8 face ``fidx``.

    ``fmap`` is the 4-entry face-corner local-index table for ``fidx``,
    in whatever node-order convention ``children`` rows themselves use
    (e.g. ``HEX_QUAD_FNMAP[fidx]`` for a Gmsh-file-order caller, or
    ``hex_face_corner_indices(fidx)`` for a canonical-order caller).
    Note that these two conventions are NOT interchangeable: they agree
    at 4 of the 6 face indices and diverge at the other 2, so ``fmap``
    must always be drawn from the same convention as ``children``.
    """
    faces = []
    for ix, iy, iz, row in children:
        on_face = (
            (fidx == 0 and iz == 0) or
            (fidx == 1 and iy == 0) or
            (fidx == 2 and ix == 1) or
            (fidx == 3 and iy == 1) or
            (fidx == 4 and ix == 0) or
            (fidx == 5 and iz == 1)
        )
        if on_face:
            row = np.asarray(row)
            faces.append(tuple(int(n) for n in row[list(fmap)]))

    if len(faces) != 4:
        raise ValueError('Hex refinement did not produce four face patches')

    return faces


def hex_order_quad2x2_faces(coarse_pids, fidx, fine, store, tol=1e-10):
    """Order four fine (child) face-node tuples against one coarse Hex8
    face into the accepted quad-2x2 quadrant order.  ``coarse_pids`` are
    the coarse element's node ids in PyFR canonical Hex8 order."""
    coarse_pids = np.asarray(coarse_pids)
    cids = coarse_pids[list(hex_face_corner_indices(fidx))]
    qpts = QuadShape.std_ele(1)
    cpts = store.coords(cids)
    lhs = np.column_stack((qpts, np.ones(len(qpts))))
    coeff, _, rank, _ = np.linalg.lstsq(lhs, cpts, rcond=None)
    if rank != 3:
        raise ValueError('Degenerate coarse Hex mortar face')

    grid = np.array([
        (u, v) for v in (-1.0, 0.0, 1.0)
        for u in (-1.0, 0.0, 1.0)
    ])
    gphys = np.column_stack((grid, np.ones(len(grid)))) @ coeff
    quadrants = (
        frozenset({(-1.0, -1.0), (0.0, -1.0),
                   (-1.0, 0.0), (0.0, 0.0)}),
        frozenset({(0.0, -1.0), (1.0, -1.0),
                   (0.0, 0.0), (1.0, 0.0)}),
        frozenset({(-1.0, 0.0), (0.0, 0.0),
                   (-1.0, 1.0), (0.0, 1.0)}),
        frozenset({(0.0, 0.0), (1.0, 0.0),
                   (0.0, 1.0), (1.0, 1.0)}),
    )

    scale = max(1.0, float(np.max(np.abs(cpts), initial=0.0)))
    ordered = [None]*4
    for face in fine:
        target = []
        for node in face:
            coord = store.coords(np.array([node]))[0]
            errors = np.max(np.abs(gphys - coord), axis=1)
            match = np.flatnonzero(errors <= tol*scale)
            if len(match) != 1:
                raise ValueError(
                    'Refined Hex face does not match coarse 2x2 grid'
                )
            target.append(tuple(float(v) for v in grid[int(match[0])]))

        try:
            slot = quadrants.index(frozenset(target))
        except ValueError:
            raise ValueError(
                'Refined Hex face is not a valid 2x2 quadrant'
            ) from None
        if ordered[slot] is not None:
            raise ValueError('Duplicate refined Hex face quadrant')
        ordered[slot] = face

    if any(face is None for face in ordered):
        raise ValueError('Incomplete refined Hex face coverage')

    return tuple(ordered)


# ===========================================================================
# Recursive Quad-tree topology / subdivision core
#
# This is the direct 2D dimensional sibling of the accepted Hex core above.
# It intentionally stays topology/geometry-only: callers own mark selection,
# 2:1 closure mutation, native materialization, and persistence.
# ===========================================================================


@lru_cache(maxsize=None)
def quad_face_corner_indices(fidx):
    """Canonical Quad4 vertex indices bounding Line face ``fidx``."""
    qpts = QuadShape.std_ele(1)
    lpts = LineShape.std_ele(1)
    project = QuadShape.faces[fidx][1]
    fpts = np.asarray([project(*p) for p in lpts])

    idxs = []
    for p in fpts:
        match = np.flatnonzero(np.all(np.isclose(qpts, p), axis=1))
        if len(match) != 1:
            raise ValueError('Unable to identify Quad face corners')
        idxs.append(int(match[0]))

    return tuple(idxs)


def quad_tree_cell_index(path):
    """Map a quadrant path to its dyadic ``(level, (i, j))`` cell.

    Quadrants use the accepted native-reference convention ``ix + 2*iy``.
    """
    i = j = 0
    for quadrant in path:
        i = 2*i + (quadrant & 1)
        j = 2*j + ((quadrant >> 1) & 1)

    return len(path), (i, j)


def quad_face_axis_side(fidx):
    """Return ``(axis, is_high_side)`` for canonical Quad4 face ``fidx``."""
    return ((1, 0), (0, 1), (1, 1), (0, 0))[fidx]


def quad_root_face_u(pids, fidx):
    """Integer Line coordinate ``u in {0, 1}`` keyed by shared node id."""
    pids = np.asarray(pids)
    qpts = QuadShape.std_ele(1)
    axis, _ = quad_face_axis_side(fidx)
    taxis = 1 - axis

    return {
        int(pids[cidx]): int(round((qpts[cidx, taxis] + 1)/2))
        for cidx in quad_face_corner_indices(fidx)
    }


def quad_c2_transform(src, dst):
    """Derive the identity/reversal map from one shared root Line to
    another."""
    try:
        mapping = {u: dst[node] for node, u in src.items()}
    except KeyError:
        raise ValueError('Invalid Quad root-face corner orientation') \
            from None
    if set(mapping) != {0, 1}:
        raise ValueError('Invalid Quad root-face corner orientation')

    c = mapping[0]
    p = mapping[1]
    a = p - c
    if a not in (-1, 1) or c not in (0, 1):
        raise ValueError('Non-affine Quad root-face orientation')

    return a, c


@lru_cache(maxsize=4096)
def _quad_transform_interval_cached(interval, transform):
    u0, u1 = interval
    a, c = transform
    v0, v1 = a*u0 + c, a*u1 + c

    return min(v0, v1), max(v0, v1)


def quad_transform_interval(interval, transform):
    """Apply a Line identity/reversal transform to a dyadic interval."""
    return _quad_transform_interval_cached(tuple(interval), tuple(transform))


def quad_interval_overlap(a, b):
    """Whether two open dyadic Line intervals overlap."""
    return max(a[0], b[0]) < min(a[1], b[1])


def quad_tree_leaves(roots, split):
    """Expand internal four-way split markers into canonical active leaves."""
    leaves = []

    def walk(root, path):
        if (root, path) in split:
            for quadrant in range(4):
                walk(root, path + (quadrant,))
        else:
            leaves.append((root, path))

    for root in sorted(roots):
        walk(root, ())

    return leaves


def quad_tree_face_groups(leaves, rootfaces):
    """Group leaf faces by shared root or internal dyadic surface identity."""
    groups = defaultdict(lambda: defaultdict(list))

    for leaf in leaves:
        root, path = leaf
        level, idx = quad_tree_cell_index(path)
        den = 1 << level
        lo = tuple(Fraction(i, den) for i in idx)
        hi = tuple(Fraction(i + 1, den) for i in idx)

        for fidx in range(4):
            axis, high = quad_face_axis_side(fidx)
            plane = hi[axis] if high else lo[axis]
            taxis = 1 - axis
            interval = (lo[taxis], hi[taxis])

            atroot = plane == (1 if high else 0)
            if atroot:
                face = rootfaces[root, fidx]
                surface = ('root', face['qkey'])
                side = root
                interval = quad_transform_interval(
                    interval, face['transform']
                )
            else:
                surface = ('internal', root, axis, plane)
                side = high

            groups[surface][side].append({
                'leaf': leaf, 'fidx': fidx, 'level': level,
                'interval': interval, 'surface': surface,
            })

    return groups


def quad_tree_face_pairs(groups, rootkinds):
    """Yield physically adjacent leaf-face pairs implied by face groups."""
    for surface in sorted(groups, key=repr):
        sides = groups[surface]
        if len(sides) == 1:
            if surface[0] == 'internal':
                raise ValueError('Incomplete internal Quad tree face')
            if rootkinds.get(surface) == 'interior':
                raise ValueError('Incomplete shared Quad root face')
            continue
        if len(sides) != 2:
            raise ValueError('Non-manifold Quad tree face')

        skeys = sorted(sides, key=repr)
        lhs = sorted(
            sides[skeys[0]], key=lambda d: (d['leaf'], d['fidx'])
        )
        rhs = sorted(
            sides[skeys[1]], key=lambda d: (d['leaf'], d['fidx'])
        )

        for lface in lhs:
            for rface in rhs:
                if quad_interval_overlap(
                    lface['interval'], rface['interval']
                ):
                    yield lface, rface


class QuadNodeStore:
    """Injectable node-coordinate store/allocator for Quad subdivision."""

    def __init__(self, coords, allocate):
        self._coords = coords
        self._allocate = allocate

    def coords(self, node_ids):
        return self._coords(node_ids)

    def allocate(self, points):
        return self._allocate(points)


def quad_affine_map(pids, store, tol=1e-10):
    """Validate positive affine Quad4 geometry and return its map."""
    pids = np.asarray(pids)
    src = QuadShape.std_ele(1)
    dst = store.coords(pids)
    lhs = np.column_stack((src, np.ones(len(src))))
    coeff, _, rank, _ = np.linalg.lstsq(lhs, dst, rcond=None)

    if rank != 3:
        raise ValueError('Degenerate Quad4 refinement geometry')

    fitted = lhs @ coeff
    scale = max(1.0, float(np.max(np.abs(dst), initial=0.0)))
    error = np.max(np.abs(fitted - dst), initial=0.0)
    if error > tol*scale:
        raise ValueError('Quad refinement requires affine Quad4 geometry')

    jac = coeff[:2].T
    if np.linalg.det(jac) <= tol*scale**2:
        raise ValueError('Quad refinement requires positive affine geometry')

    def apply(points):
        points = np.asarray(points, dtype=float)
        plhs = np.column_stack((points, np.ones(len(points))))
        return plhs @ coeff

    return apply


def quad_refined_children(pids, store, node_cache, coord_cache):
    """Subdivide one affine Quad4 into four canonical quadrant children."""
    amap = quad_affine_map(pids, store)
    pids = np.asarray(pids)
    qpts = QuadShape.std_ele(1)
    refs = (-1.0, 0.0, 1.0)
    grid = {}

    for y in refs:
        for x in refs:
            ref = np.array([x, y])
            mask = np.ones(len(qpts), dtype=bool)
            for d, value in enumerate(ref):
                if value:
                    mask &= np.isclose(qpts[:, d], value)

            support = tuple(sorted(int(i) for i in pids[mask]))
            if len(support) == 1:
                node = support[0]
            elif support in node_cache:
                node = node_cache[support]
                coord = amap([ref])[0]
                scale = max(
                    1.0, float(np.max(np.abs(coord), initial=0.0))
                )
                if not np.allclose(
                    coord_cache[support], coord, rtol=0,
                    atol=2e-10*scale
                ):
                    raise ValueError(
                        'Refined Quad neighbours disagree on shared '
                        'subdivision-node geometry'
                    )
            else:
                coord = amap([ref])[0]
                node = int(store.allocate(coord[None])[0])
                node_cache[support] = node
                coord_cache[support] = coord

            grid[tuple(ref)] = node

    children = []
    for iy in range(2):
        for ix in range(2):
            centre = np.array([
                -0.5 if ix == 0 else 0.5,
                -0.5 if iy == 0 else 0.5,
            ])
            crefs = 0.5*qpts + centre
            row = np.array(
                [grid[tuple(p)] for p in crefs], dtype=np.int64
            )
            children.append((ix, iy, row))

    return children


def quad_child_face_nodes(children, fidx, fmap):
    """Return the two child Line face-node tuples on parent face ``fidx``."""
    faces = []
    for ix, iy, row in children:
        on_face = (
            (fidx == 0 and iy == 0) or
            (fidx == 1 and ix == 1) or
            (fidx == 2 and iy == 1) or
            (fidx == 3 and ix == 0)
        )
        if on_face:
            row = np.asarray(row)
            faces.append(tuple(int(n) for n in row[list(fmap)]))

    if len(faces) != 2:
        raise ValueError('Quad refinement did not produce two face halves')

    return faces


def quad_order_line1x2_faces(coarse_pids, fidx, fine, store, tol=1e-10):
    """Order two fine Lines in coarse-reference low/high half order."""
    coarse_pids = np.asarray(coarse_pids)
    cids = coarse_pids[list(quad_face_corner_indices(fidx))]
    lpts = LineShape.std_ele(1)
    cpts = store.coords(cids)
    lhs = np.column_stack((lpts, np.ones(len(lpts))))
    coeff, _, rank, _ = np.linalg.lstsq(lhs, cpts, rcond=None)
    if rank != 2:
        raise ValueError('Degenerate coarse Quad mortar face')

    grid = np.array([[-1.0], [0.0], [1.0]])
    gphys = np.column_stack((grid, np.ones(len(grid)))) @ coeff
    halves = (
        frozenset({-1.0, 0.0}),
        frozenset({0.0, 1.0}),
    )

    scale = max(1.0, float(np.max(np.abs(cpts), initial=0.0)))
    ordered = [None]*2
    for face in fine:
        target = []
        for node in face:
            coord = store.coords(np.array([node]))[0]
            errors = np.max(np.abs(gphys - coord), axis=1)
            match = np.flatnonzero(errors <= tol*scale)
            if len(match) != 1:
                raise ValueError(
                    'Refined Quad face does not match coarse 1x2 grid'
                )
            target.append(float(grid[int(match[0]), 0]))

        try:
            slot = halves.index(frozenset(target))
        except ValueError:
            raise ValueError(
                'Refined Quad face is not a valid 1x2 half'
            ) from None
        if ordered[slot] is not None:
            raise ValueError('Duplicate refined Quad face half')
        ordered[slot] = face

    if any(face is None for face in ordered):
        raise ValueError('Incomplete refined Quad face coverage')

    return tuple(ordered)
