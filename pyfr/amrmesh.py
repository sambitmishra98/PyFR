from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from pyfr.amr import (
    HexNodeStore, QuadNodeStore, hex_face_corner_indices, hex_octree_level,
    hex_order_quad2x2_faces, hex_refined_children, hex_tree_face_groups,
    hex_tree_face_pairs, quad_affine_map, quad_c2_transform,
    quad_face_corner_indices,
    quad_order_line1x2_faces, quad_refined_children, quad_root_face_u,
    quad_tree_face_groups, quad_tree_face_pairs,
)


class AMRMeshError(ValueError):
    pass


@dataclass(frozen=True)
class AdaptedRawMesh:

    root_mesh_uuid: str
    node_ids: np.ndarray
    node_locs: np.ndarray
    hex_nodes: np.ndarray       # (nleaves, 8) canonical-order node ids
    hex_faces_cidx: np.ndarray  # (nleaves, 6) int16 codec index per face
    hex_faces_off: np.ndarray   # (nleaves, 6) int64: >=0 nbr eidx, -1 bc, -2 mortar
    codec: tuple
    mortars: tuple              # tuple of MortarRecord
    leaf_order: tuple           # tuple of (root_eidx, path), index == adapted eidx
    leaf_tags: np.ndarray       # (nleaves,) uint64, inherited from root; NOT colour


@dataclass(frozen=True)
class MortarRecord:
    name: str
    left_eidx: int
    left_fidx: int
    right_eidx: tuple   # 4 adapted eidx, quad-2x2 order
    right_fidx: tuple   # 4 fidx (all equal for an affine octree face)
    format: str = 'one-to-many-v1'
    template: str = 'quad-2x2'


# ---------------------------------------------------------------------------
# Scope validation - fail closed, never work around.
# ---------------------------------------------------------------------------

def _validate_scope(mesh, tree, comm_size):
    if comm_size != 1:
        raise AMRMeshError(
            'D5B2 native materializer requires a single-rank communicator'
        )
    if mesh.uuid != tree.root_mesh_uuid:
        raise AMRMeshError(
            'HexLeafTree root_mesh_uuid does not match the native root mesh'
        )
    if list(mesh.etypes) != ['hex'] and any(
        et != 'hex' and et in mesh.eidxs and len(mesh.eidxs[et])
        for et in mesh.etypes
    ):
        raise AMRMeshError(
            'D5B2 supports pure-Hex root meshes only; other element types '
            'are present'
        )
    if 'hex' not in mesh.spts_curved or np.any(mesh.spts_curved['hex']):
        raise AMRMeshError(
            'D5B2 requires first-order affine (uncurved) root Hex geometry'
        )
    if mesh.con_p:
        raise AMRMeshError(
            'D5B2 requires no MPI connectivity in the root mesh'
        )
    if mesh.mcon:
        raise AMRMeshError(
            'D5B2 requires no pre-existing mortar connectivity in the '
            'root mesh'
        )
    for name in mesh.bcon:
        if 'periodic' in name.lower():
            raise AMRMeshError(
                f'D5B2 requires no periodic connectivity; found boundary '
                f'{name!r}'
            )


# ---------------------------------------------------------------------------
# Root topology derivation from native Mesh.con / Mesh.bcon - do not
# rediscover neighbours geometrically.
# ---------------------------------------------------------------------------

def _hex_local_maps(mesh):
    if 'hex' not in mesh.eidxs:
        raise AMRMeshError('Root mesh has no Hex elements')

    geidx = mesh.eidxs['hex']
    l2g = {li: int(gi) for li, gi in enumerate(geidx)}
    g2l = {int(gi): li for li, gi in enumerate(geidx)}
    pids_by_g = {
        int(gi): np.asarray(mesh.spts_nodes['hex'][li])
        for li, gi in enumerate(geidx)
    }
    tags_by_g = {
        int(gi): mesh.tags['hex'][li] for li, gi in enumerate(geidx)
    }
    return l2g, g2l, pids_by_g, tags_by_g


def _face_uv(pids_by_g, geidx, fidx):
    from pyfr.amr import hex_face_axis_side
    from pyfr.shapes import HexShape

    pids = np.asarray(pids_by_g[geidx])
    order = HexShape.order_from_npts(len(pids))
    hpts = HexShape.std_ele(order)
    axis, _ = hex_face_axis_side(fidx)
    taxes = tuple(d for d in range(3) if d != axis)
    idxs = HexShape.face_corner_pts_idxs(fidx, len(pids))

    return {
        int(pids[cidx]): tuple(
            int(round((hpts[cidx, d] + 1)/2)) for d in taxes
        )
        for cidx in idxs
    }


def _derive_root_face_topology(mesh, pids_by_g, l2g):
    from pyfr.amr import hex_d4_transform

    rootfaces = {}
    rootkinds = {}
    identity = (1, 0, 0, 0, 1, 0)

    def cidx_face(cidx):
        etype, fidx = mesh.cidxmap[int(cidx)]
        if etype != 'hex':
            raise AMRMeshError(
                'D5B2 requires all root connectivity to be Hex-Hex'
            )
        return fidx

    if mesh.con:
        lhs, rhs = mesh.con
        for i in range(len(lhs)):
            lg = l2g[int(lhs.eidxs[i])]
            rg = l2g[int(rhs.eidxs[i])]
            lf = cidx_face(lhs.cidxs[i])
            rf = cidx_face(rhs.cidxs[i])

            a, b = (lg, lf), (rg, rf)
            primary, secondary = (a, b) if a < b else (b, a)
            qkey = ('interior', primary)
            rootkinds[('root', qkey)] = 'interior'

            if secondary == a:
                # a is secondary; recompute consistently below
                pass

            uv_primary = _face_uv(pids_by_g, *primary)
            for side in (primary, secondary):
                if side == primary:
                    rootfaces[side] = {'qkey': qkey, 'transform': identity}
                else:
                    uv_side = _face_uv(pids_by_g, *side)
                    transform = hex_d4_transform(uv_side, uv_primary)
                    rootfaces[side] = {'qkey': qkey, 'transform': transform}

    for name, con in mesh.bcon.items():
        for i in range(len(con)):
            g = l2g[int(con.eidxs[i])]
            f = cidx_face(con.cidxs[i])
            qkey = ('boundary', name, g, f)
            rootkinds[('root', qkey)] = 'boundary'
            rootfaces[g, f] = {'qkey': qkey, 'transform': identity}

    for g in pids_by_g:
        for fidx in range(6):
            if (g, fidx) not in rootfaces:
                raise AMRMeshError(
                    f'Root Hex {g} face {fidx} is neither interior '
                    f'Hex-Hex connectivity nor a named boundary; '
                    f'unsupported root topology'
                )

    return rootfaces, rootkinds


# ---------------------------------------------------------------------------
# Root coverage invariant.
# ---------------------------------------------------------------------------

def _validate_root_coverage(tree, pids_by_g):
    tree_roots = {r for r, _ in tree.leaves()}
    mesh_roots = set(pids_by_g)
    if tree_roots != mesh_roots:
        missing = mesh_roots - tree_roots
        unknown = tree_roots - mesh_roots
        raise AMRMeshError(
            f'HexLeafTree root coverage does not exactly match the native '
            f'root Hex set (missing={sorted(missing)}, '
            f'unknown={sorted(unknown)})'
        )


# ---------------------------------------------------------------------------
# 2:1 balance validation - reject, never silently close.
# ---------------------------------------------------------------------------

def _validate_balanced_and_pair(leaves, rootfaces, rootkinds):
    groups = hex_tree_face_groups(leaves, rootfaces)
    pairs = list(hex_tree_face_pairs(groups, rootkinds))

    for lface, rface in pairs:
        dl = abs(lface['level'] - rface['level'])
        if dl > 1:
            raise AMRMeshError(
                'Supplied HexLeafTree is not physically 2:1 balanced '
                f'({lface["leaf"]}, fidx={lface["fidx"]}) vs '
                f'({rface["leaf"]}, fidx={rface["fidx"]})'
            )

    return groups, pairs


# ---------------------------------------------------------------------------
# Leaf geometry materialization - reuse the shared D1/D2 subdivision core.
# ---------------------------------------------------------------------------

class _NativeNodeStore:
    def __init__(self, node_ids, node_locs):
        self._coords = {
            int(i): tuple(float(x) for x in c)
            for i, c in zip(node_ids, node_locs)
        }
        self._next_id = (int(node_ids.max()) + 1) if len(node_ids) else 0
        self.new_nodes = []

    def coords(self, ids):
        return np.array([self._coords[int(i)] for i in np.asarray(ids)])

    def allocate(self, points):
        ids = []
        for p in np.asarray(points):
            nid = self._next_id
            self._next_id += 1
            coord = tuple(float(x) for x in p)
            self._coords[nid] = coord
            self.new_nodes.append((nid, coord))
            ids.append(nid)
        return np.array(ids, dtype=np.int64)


def _materialize_leaf_pids(pids_by_g, leaves, store):
    node_cache, coord_cache = {}, {}
    result = {}

    by_root = defaultdict(list)
    for root, path in leaves:
        by_root[root].append(path)

    for root, paths in by_root.items():
        cache = {(): pids_by_g[root]}
        needed = {p[:k] for p in paths for k in range(len(p) + 1)}
        for node in sorted(needed, key=len):
            if node in cache:
                continue
            parent = node[:-1]
            children = hex_refined_children(
                cache[parent], store, node_cache, coord_cache
            )
            for ix, iy, iz, row in children:
                octant = ix + 2*iy + 4*iz
                cache[parent + (octant,)] = row

        for p in paths:
            result[root, p] = cache[p]

    return result


def _hex_face_nodes(row, fidx):
    from pyfr.shapes import HexShape

    row = np.asarray(row)
    idxs = HexShape.face_corner_pts_idxs(fidx, len(row))
    return tuple(int(row[i]) for i in idxs)


def _materialize_isoparametric_hex_leaves(mesh, pids_by_g, leaves, store):
    from pyfr.amr import hex_tree_cell_index
    from pyfr.polys import get_polybasis
    from pyfr.shapes import HexShape

    result = {}
    by_root = defaultdict(list)
    for root, path in leaves:
        by_root[root].append(path)

    for root, paths in by_root.items():
        root_pids = np.asarray(pids_by_g[root], dtype=np.int64)
        order = HexShape.order_from_npts(len(root_pids))
        spts = np.asarray(HexShape.std_ele(order), dtype=float)
        rcoords = store.coords(root_pids)
        basis = get_polybasis('hex', order, spts)

        # Parametric node cache makes sibling leaves share their common
        # shape points exactly.  Existing root shape points retain their ids.
        pcache = {
            tuple(np.round(pt, 14)): int(nid)
            for pt, nid in zip(spts, root_pids)
        }

        for path in sorted(paths):
            if not path:
                result[root, path] = np.array(root_pids, copy=True)
                continue

            level, ijk = hex_tree_cell_index(path)
            scale = 2**level
            ijk = np.asarray(ijk, dtype=float)
            rpts = (spts + 2*ijk + 1 - scale)/scale
            coords = basis.nodal_basis_at(rpts) @ rcoords

            row = np.empty(len(spts), dtype=np.int64)
            for i, (rpt, coord) in enumerate(zip(rpts, coords)):
                key = tuple(np.round(rpt, 14))
                nid = pcache.get(key)
                if nid is None:
                    nid = int(store.allocate(np.asarray([coord]))[0])
                    pcache[key] = nid
                else:
                    old = store.coords(np.asarray([nid]))[0]
                    gscale = max(1.0, float(np.max(np.abs(coord))))
                    if np.max(np.abs(old - coord)) > 5e-12*gscale:
                        raise AMRMeshError(
                            'Restricted Hex geometry disagrees at a shared '
                            'parametric shape point'
                        )
                row[i] = nid

            result[root, path] = row

    return result


# ---------------------------------------------------------------------------
# Top-level entry point.
# ---------------------------------------------------------------------------

def materialize_native_hex_tree(mesh, tree, *, comm_size=1):
    _validate_scope(mesh, tree, comm_size)

    l2g, g2l, pids_by_g, tags_by_g = _hex_local_maps(mesh)
    _validate_root_coverage(tree, pids_by_g)

    rootfaces, rootkinds = _derive_root_face_topology(mesh, pids_by_g, l2g)

    leaves = list(tree.leaves())
    groups, pairs = _validate_balanced_and_pair(leaves, rootfaces, rootkinds)

    store = _NativeNodeStore(mesh.node_idxs, mesh.node_locs)
    leaf_pids = _materialize_leaf_pids(pids_by_g, leaves, store)

    # Canonical leaf -> adapted global Hex eidx mapping: leaf ordinal in
    # the tree's own canonical (root_eidx, path) lexicographic order.
    leaf_order = tuple(leaves)
    adapted_eidx = {leaf: i for i, leaf in enumerate(leaf_order)}

    codec = ['eles/hex/face/0', 'eles/hex/face/1', 'eles/hex/face/2',
             'eles/hex/face/3', 'eles/hex/face/4', 'eles/hex/face/5']
    codec_index = {c: i for i, c in enumerate(codec)}

    def bc_cidx(name):
        key = f'bc/{name}'
        if key not in codec_index:
            codec_index[key] = len(codec)
            codec.append(key)
        return codec_index[key]

    n = len(leaf_order)
    faces_cidx = np.zeros((n, 6), dtype=np.int16)
    faces_off = np.zeros((n, 6), dtype=np.int64)
    faces_set = np.zeros((n, 6), dtype=bool)

    def set_face(leaf, fidx, cidx, off):
        i = adapted_eidx[leaf]
        if faces_set[i, fidx]:
            raise AMRMeshError(
                f'Leaf {leaf} face {fidx} assigned more than once'
            )
        faces_cidx[i, fidx] = cidx
        faces_off[i, fidx] = off
        faces_set[i, fidx] = True

    # Boundary faces: any singleton surface classified 'boundary'.
    for surface, sides in groups.items():
        if surface[0] != 'root' or rootkinds.get(surface) != 'boundary':
            continue
        (_, qkey), = [(surface[0], surface[1])]
        name = qkey[1]
        for side, fragments in sides.items():
            for frag in fragments:
                set_face(frag['leaf'], frag['fidx'], bc_cidx(name), -1)

    # Equal-level and 1:4 mortar pairs.
    mortar_groups = defaultdict(list)  # (coarse leaf, coarse fidx) -> fine fragments
    equal_pairs_seen = set()

    for lface, rface in pairs:
        ll, lf = lface['leaf'], lface['fidx']
        rl, rf = rface['leaf'], rface['fidx']
        if lface['level'] == rface['level']:
            key = frozenset({(ll, lf), (rl, rf)})
            if key in equal_pairs_seen:
                continue
            equal_pairs_seen.add(key)
            lcidx = codec_index[f'eles/hex/face/{rf}']
            rcidx = codec_index[f'eles/hex/face/{lf}']
            set_face(ll, lf, lcidx, adapted_eidx[rl])
            set_face(rl, rf, rcidx, adapted_eidx[ll])
        else:
            coarse, fine = (lface, rface) if lface['level'] < rface['level'] \
                else (rface, lface)
            mortar_groups[coarse['leaf'], coarse['fidx']].append(fine)

    mortars = []
    for mi, ((cleaf, cfidx), fines) in enumerate(sorted(
        mortar_groups.items(), key=lambda kv: (adapted_eidx[kv[0][0]], kv[0][1])
    )):
        if len(fines) != 4:
            raise AMRMeshError(
                f'Coarse leaf {cleaf} face {cfidx} has {len(fines)} fine '
                f'neighbours (expected exactly 4 for a quad-2x2 mortar)'
            )

        coarse_pids = leaf_pids[cleaf]
        fine_faces = []
        for frag in fines:
            fmap = hex_face_corner_indices(frag['fidx'])
            row = np.asarray(leaf_pids[frag['leaf']])
            fine_faces.append(tuple(int(x) for x in row[list(fmap)]))

        ordered = hex_order_quad2x2_faces(
            coarse_pids, cfidx, fine_faces, store
        )
        # Map each ordered face-node tuple back to its owning fine fragment.
        by_face = {}
        for frag in fines:
            fmap = hex_face_corner_indices(frag['fidx'])
            row = np.asarray(leaf_pids[frag['leaf']])
            key = tuple(int(x) for x in row[list(fmap)])
            by_face[key] = frag

        right_eidx, right_fidx = [], []
        for face in ordered:
            frag = by_face[face]
            right_eidx.append(adapted_eidx[frag['leaf']])
            right_fidx.append(frag['fidx'])
            set_face(frag['leaf'], frag['fidx'], -1, -2)

        set_face(cleaf, cfidx, -1, -2)

        name = f'm{mi}'
        mortars.append(MortarRecord(
            name=name, left_eidx=adapted_eidx[cleaf], left_fidx=cfidx,
            right_eidx=tuple(right_eidx), right_fidx=tuple(right_fidx),
        ))

    if not np.all(faces_set):
        missing = np.flatnonzero(~faces_set.all(axis=1))
        raise AMRMeshError(
            f'{len(missing)} adapted leaves have unresolved faces after '
            f'materialization'
        )

    hex_nodes = np.array(
        [leaf_pids[leaf] for leaf in leaf_order], dtype=np.int64
    )

    all_ids = set(int(i) for i in mesh.node_idxs) | {
        i for i, _ in store.new_nodes
    }
    node_ids = np.array(sorted(all_ids), dtype=np.int64)
    node_locs = store.coords(node_ids)

    leaf_tags = np.array(
        [tags_by_g[leaf[0]] for leaf in leaf_order], dtype=np.uint64
    )

    return AdaptedRawMesh(
        root_mesh_uuid=mesh.uuid, node_ids=node_ids, node_locs=node_locs,
        hex_nodes=hex_nodes, hex_faces_cidx=faces_cidx,
        hex_faces_off=faces_off, codec=tuple(codec), mortars=tuple(mortars),
        leaf_order=leaf_order, leaf_tags=leaf_tags,
    )


# ===========================================================================
# OFFLINE-2D 0053 - native Quad quadtree materializer
#
# This is the direct 2D dimensional sibling of the accepted D5B2 Hex
# materializer above.  It consumes an already-valid, already-2:1-balanced
# QuadLeafTree and deliberately stops before HDF5 persistence.
# ===========================================================================


@dataclass(frozen=True)
class AdaptedQuadRawMesh:

    root_mesh_uuid: str
    node_ids: np.ndarray
    node_locs: np.ndarray
    quad_nodes: np.ndarray
    quad_faces_cidx: np.ndarray
    quad_faces_off: np.ndarray
    codec: tuple
    mortars: tuple
    leaf_order: tuple
    leaf_tags: np.ndarray


@dataclass(frozen=True)
class LineMortarRecord:

    name: str
    left_eidx: int
    left_fidx: int
    right_eidx: tuple
    right_fidx: tuple
    format: str = 'one-to-many-v1'
    template: str = 'line-1x2'


@dataclass(frozen=True)
class MixedLineMortarRecord:

    name: str
    left_etype: str
    left_eidx: int
    left_fidx: int
    right_etype: tuple
    right_eidx: tuple
    right_fidx: tuple
    format: str = 'one-to-many-v1'
    template: str = 'line-1x2'


@dataclass(frozen=True)
class AdaptedMixedQuadRawMesh:

    root_mesh_uuid: str
    node_ids: np.ndarray
    node_locs: np.ndarray
    tri_nodes: np.ndarray
    tri_faces_cidx: np.ndarray
    tri_faces_off: np.ndarray
    tri_tags: np.ndarray
    quad_nodes: np.ndarray
    quad_faces_cidx: np.ndarray
    quad_faces_off: np.ndarray
    codec: tuple
    mortars: tuple
    leaf_order: tuple
    leaf_tags: np.ndarray


def _validate_quad_scope(mesh, tree, comm_size):
    if comm_size != 1:
        raise AMRMeshError(
            'Quad native materializer requires a single-rank communicator'
        )
    if mesh.uuid != tree.root_mesh_uuid:
        raise AMRMeshError(
            'QuadLeafTree root_mesh_uuid does not match the native root mesh'
        )
    if 'quad' not in mesh.eidxs or any(
        et != 'quad' and et in mesh.eidxs and len(mesh.eidxs[et])
        for et in mesh.etypes
    ):
        raise AMRMeshError(
            'Quad native materializer supports pure-Quad root meshes only'
        )
    if 'quad' not in mesh.spts_curved or np.any(mesh.spts_curved['quad']):
        raise AMRMeshError(
            'Quad native materializer requires first-order affine '
            '(uncurved) root geometry'
        )
    if mesh.con_p:
        raise AMRMeshError(
            'Quad native materializer requires no MPI root connectivity'
        )
    if mesh.mcon:
        raise AMRMeshError(
            'Quad native materializer requires no pre-existing root mortars'
        )
    if mesh.raw is not None and 'periodic' in mesh.raw:
        raise AMRMeshError(
            'Quad native materializer requires no periodic connectivity'
        )
    for name in mesh.bcon:
        if 'periodic' in name.lower():
            raise AMRMeshError(
                'Quad native materializer requires no periodic '
                f'connectivity; found boundary {name!r}'
            )


def _quad_local_maps(mesh):
    if 'quad' not in mesh.eidxs:
        raise AMRMeshError('Root mesh has no Quad elements')

    geidx = mesh.eidxs['quad']
    l2g = {li: int(gi) for li, gi in enumerate(geidx)}
    pids_by_g = {
        int(gi): np.asarray(mesh.spts_nodes['quad'][li])
        for li, gi in enumerate(geidx)
    }
    tags_by_g = {
        int(gi): mesh.tags['quad'][li] for li, gi in enumerate(geidx)
    }

    return l2g, pids_by_g, tags_by_g


def _derive_quad_root_face_topology(mesh, pids_by_g, l2g):
    rootfaces = {}
    rootkinds = {}
    identity = (1, 0)

    def cidx_face(cidx):
        etype, fidx = mesh.cidxmap[int(cidx)]
        if etype != 'quad':
            raise AMRMeshError(
                'Quad native materializer requires Quad-Quad connectivity'
            )
        return fidx

    if mesh.con:
        lhs, rhs = mesh.con
        for i in range(len(lhs)):
            lg = l2g[int(lhs.eidxs[i])]
            rg = l2g[int(rhs.eidxs[i])]
            lf = cidx_face(lhs.cidxs[i])
            rf = cidx_face(rhs.cidxs[i])

            a, b = (lg, lf), (rg, rf)
            primary, secondary = (a, b) if a < b else (b, a)
            qkey = ('interior', primary)
            rootkinds['root', qkey] = 'interior'

            pu = quad_root_face_u(
                pids_by_g[primary[0]], primary[1]
            )
            su = quad_root_face_u(
                pids_by_g[secondary[0]], secondary[1]
            )
            rootfaces[primary] = {
                'qkey': qkey, 'transform': identity
            }
            rootfaces[secondary] = {
                'qkey': qkey,
                'transform': quad_c2_transform(su, pu),
            }

    for name, con in mesh.bcon.items():
        for i in range(len(con)):
            g = l2g[int(con.eidxs[i])]
            fidx = cidx_face(con.cidxs[i])
            qkey = ('boundary', name, g, fidx)
            rootkinds['root', qkey] = 'boundary'
            rootfaces[g, fidx] = {
                'qkey': qkey, 'transform': identity
            }

    for g in pids_by_g:
        for fidx in range(4):
            if (g, fidx) not in rootfaces:
                raise AMRMeshError(
                    f'Root Quad {g} face {fidx} is neither interior '
                    'Quad-Quad connectivity nor a named boundary'
                )

    return rootfaces, rootkinds


def _validate_quad_root_geometry(mesh, pids_by_g):
    store = _NativeNodeStore(mesh.node_idxs, mesh.node_locs)
    qstore = QuadNodeStore(store.coords, store.allocate)

    for root, pids in pids_by_g.items():
        try:
            quad_affine_map(pids, qstore)
        except ValueError as exc:
            raise AMRMeshError(
                f'Root Quad {root} is outside the affine geometry scope: '
                f'{exc}'
            ) from exc


def _validate_quad_root_coverage(tree, pids_by_g):
    tree_roots = {root for root, _ in tree.leaves()}
    mesh_roots = set(pids_by_g)
    if tree_roots != mesh_roots:
        missing = mesh_roots - tree_roots
        unknown = tree_roots - mesh_roots
        raise AMRMeshError(
            'QuadLeafTree root coverage does not exactly match the native '
            f'root Quad set (missing={sorted(missing)}, '
            f'unknown={sorted(unknown)})'
        )


def _validate_quad_balanced_and_pair(leaves, rootfaces, rootkinds):
    groups = quad_tree_face_groups(leaves, rootfaces)
    pairs = list(quad_tree_face_pairs(groups, rootkinds))

    for lface, rface in pairs:
        if abs(lface['level'] - rface['level']) > 1:
            raise AMRMeshError(
                'Supplied QuadLeafTree is not physically 2:1 balanced '
                f'({lface["leaf"]}, fidx={lface["fidx"]}) vs '
                f'({rface["leaf"]}, fidx={rface["fidx"]})'
            )

    return groups, pairs


def _materialize_quad_leaf_pids(pids_by_g, leaves, store):
    node_cache = {}
    coord_cache = {}
    result = {}
    qstore = QuadNodeStore(store.coords, store.allocate)

    by_root = defaultdict(list)
    for root, path in leaves:
        by_root[root].append(path)

    for root, paths in by_root.items():
        cache = {(): pids_by_g[root]}
        needed = {
            path[:depth]
            for path in paths
            for depth in range(len(path) + 1)
        }
        for node in sorted(needed, key=lambda p: (len(p), p)):
            if node in cache:
                continue

            parent = node[:-1]
            children = quad_refined_children(
                cache[parent], qstore, node_cache, coord_cache
            )
            for ix, iy, row in children:
                quadrant = ix + 2*iy
                cache[parent + (quadrant,)] = row

        for path in paths:
            result[root, path] = cache[path]

    return result


def materialize_native_quad_tree(mesh, tree, *, comm_size=1):
    _validate_quad_scope(mesh, tree, comm_size)

    l2g, pids_by_g, tags_by_g = _quad_local_maps(mesh)
    _validate_quad_root_geometry(mesh, pids_by_g)
    _validate_quad_root_coverage(tree, pids_by_g)
    rootfaces, rootkinds = _derive_quad_root_face_topology(
        mesh, pids_by_g, l2g
    )

    leaves = list(tree.leaves())
    groups, pairs = _validate_quad_balanced_and_pair(
        leaves, rootfaces, rootkinds
    )

    store = _NativeNodeStore(mesh.node_idxs, mesh.node_locs)
    leaf_pids = _materialize_quad_leaf_pids(pids_by_g, leaves, store)

    leaf_order = tuple(leaves)
    adapted_eidx = {leaf: i for i, leaf in enumerate(leaf_order)}

    codec = [f'eles/quad/face/{fidx}' for fidx in range(4)]
    codec_index = {value: i for i, value in enumerate(codec)}

    def bc_cidx(name):
        key = f'bc/{name}'
        if key not in codec_index:
            codec_index[key] = len(codec)
            codec.append(key)
        return codec_index[key]

    nleaves = len(leaf_order)
    faces_cidx = np.zeros((nleaves, 4), dtype=np.int16)
    faces_off = np.zeros((nleaves, 4), dtype=np.int64)
    faces_set = np.zeros((nleaves, 4), dtype=bool)

    def set_face(leaf, fidx, cidx, off):
        eidx = adapted_eidx[leaf]
        if faces_set[eidx, fidx]:
            raise AMRMeshError(
                f'Leaf {leaf} face {fidx} assigned more than once'
            )
        faces_cidx[eidx, fidx] = cidx
        faces_off[eidx, fidx] = off
        faces_set[eidx, fidx] = True

    for surface, sides in groups.items():
        if surface[0] != 'root' or rootkinds.get(surface) != 'boundary':
            continue

        name = surface[1][1]
        for fragments in sides.values():
            for frag in fragments:
                set_face(
                    frag['leaf'], frag['fidx'], bc_cidx(name), -1
                )

    mortar_groups = defaultdict(list)
    equal_pairs_seen = set()

    for lface, rface in pairs:
        lleaf, lfidx = lface['leaf'], lface['fidx']
        rleaf, rfidx = rface['leaf'], rface['fidx']

        if lface['level'] == rface['level']:
            key = frozenset({(lleaf, lfidx), (rleaf, rfidx)})
            if key in equal_pairs_seen:
                continue
            equal_pairs_seen.add(key)

            set_face(
                lleaf, lfidx,
                codec_index[f'eles/quad/face/{rfidx}'],
                adapted_eidx[rleaf],
            )
            set_face(
                rleaf, rfidx,
                codec_index[f'eles/quad/face/{lfidx}'],
                adapted_eidx[lleaf],
            )
        else:
            coarse, fine = (
                (lface, rface)
                if lface['level'] < rface['level']
                else (rface, lface)
            )
            mortar_groups[coarse['leaf'], coarse['fidx']].append(fine)

    mortars = []
    for mi, ((cleaf, cfidx), fines) in enumerate(sorted(
        mortar_groups.items(),
        key=lambda item: (
            adapted_eidx[item[0][0]], item[0][1]
        ),
    )):
        if len(fines) != 2:
            raise AMRMeshError(
                f'Coarse leaf {cleaf} face {cfidx} has {len(fines)} fine '
                'neighbours (expected exactly 2 for a line-1x2 mortar)'
            )

        fine_faces = []
        by_face = {}
        for frag in fines:
            fmap = quad_face_corner_indices(frag['fidx'])
            row = np.asarray(leaf_pids[frag['leaf']])
            face = tuple(int(x) for x in row[list(fmap)])
            fine_faces.append(face)
            if face in by_face:
                raise AMRMeshError('Duplicate fine Line face in mortar')
            by_face[face] = frag

        ordered = quad_order_line1x2_faces(
            leaf_pids[cleaf], cfidx, fine_faces,
            QuadNodeStore(store.coords, store.allocate),
        )

        right_eidx = []
        right_fidx = []
        for face in ordered:
            frag = by_face[face]
            right_eidx.append(adapted_eidx[frag['leaf']])
            right_fidx.append(frag['fidx'])
            set_face(frag['leaf'], frag['fidx'], -1, -2)

        set_face(cleaf, cfidx, -1, -2)
        mortars.append(LineMortarRecord(
            name=f'm{mi}',
            left_eidx=adapted_eidx[cleaf],
            left_fidx=cfidx,
            right_eidx=tuple(right_eidx),
            right_fidx=tuple(right_fidx),
        ))

    if not np.all(faces_set):
        missing = np.argwhere(~faces_set)
        first = [tuple(int(v) for v in row) for row in missing[:8]]
        raise AMRMeshError(
            'Adapted Quad faces remain unresolved after materialization '
            f'(first={first})'
        )

    quad_nodes = np.array(
        [leaf_pids[leaf] for leaf in leaf_order], dtype=np.int64
    )
    all_ids = set(int(i) for i in mesh.node_idxs) | {
        node for node, _ in store.new_nodes
    }
    node_ids = np.array(sorted(all_ids), dtype=np.int64)
    node_locs = store.coords(node_ids)
    leaf_tags = np.array(
        [tags_by_g[leaf[0]] for leaf in leaf_order], dtype=np.uint64
    )

    return AdaptedQuadRawMesh(
        root_mesh_uuid=mesh.uuid,
        node_ids=node_ids,
        node_locs=node_locs,
        quad_nodes=quad_nodes,
        quad_faces_cidx=faces_cidx,
        quad_faces_off=faces_off,
        codec=tuple(codec),
        mortars=tuple(mortars),
        leaf_order=leaf_order,
        leaf_tags=leaf_tags,
    )


# ===========================================================================
# MIX2D1 - one-rank affine Tri+Quad materialization
#
# Quads retain the accepted QuadLeafTree identity/materialization mechanics.
# Triangles are immutable volume elements.  A Tri-Quad root interface is a
# fixed level-0 topology boundary for 2:1 purposes and becomes either an
# ordinary conforming interface or the accepted Line-1x2 trace mortar.
# ===========================================================================


def _validate_mixed_quad_scope(mesh, tree, comm_size):
    if comm_size != 1:
        raise AMRMeshError(
            'Mixed Quad materializer requires a single-rank communicator'
        )
    if mesh.uuid != tree.root_mesh_uuid:
        raise AMRMeshError(
            'QuadLeafTree root_mesh_uuid does not match the mixed root mesh'
        )
    if set(mesh.etypes) != {'tri', 'quad'}:
        raise AMRMeshError(
            'Mixed Quad materializer requires exactly Tri+Quad root topology'
        )
    if any(np.any(mesh.spts_curved.get(et, ())) for et in ('tri', 'quad')):
        raise AMRMeshError(
            'Mixed Quad materializer requires affine (uncurved) Tri+Quad '
            'geometry'
        )
    if mesh.con_p:
        raise AMRMeshError(
            'Mixed Quad materializer requires no MPI root connectivity'
        )
    if mesh.mcon:
        raise AMRMeshError(
            'Mixed Quad materializer requires no pre-existing root mortars'
        )
    if mesh.raw is not None and 'periodic' in mesh.raw:
        raise AMRMeshError(
            'Mixed Quad materializer requires no periodic connectivity'
        )
    for name in mesh.bcon:
        if 'periodic' in name.lower():
            raise AMRMeshError(
                'Mixed Quad materializer requires no periodic connectivity; '
                f'found boundary {name!r}'
            )


def _derive_mixed_quad_root_face_topology(mesh, pids_by_g, l2g):
    rootfaces = {}
    rootkinds = {}
    fixed = {}
    identity = (1, 0)

    def face(cidx):
        try:
            return mesh.cidxmap[int(cidx)]
        except KeyError as exc:
            raise AMRMeshError('Invalid mixed root face codec') from exc

    if mesh.con:
        lhs, rhs = mesh.con
        for i in range(len(lhs)):
            let, lf = face(lhs.cidxs[i])
            ret, rf = face(rhs.cidxs[i])
            le, re = int(lhs.eidxs[i]), int(rhs.eidxs[i])

            if let == ret == 'quad':
                lg, rg = l2g[le], l2g[re]
                a, b = (lg, lf), (rg, rf)
                primary, secondary = (a, b) if a < b else (b, a)
                qkey = ('interior', primary)
                rootkinds['root', qkey] = 'interior'

                pu = quad_root_face_u(pids_by_g[primary[0]], primary[1])
                su = quad_root_face_u(pids_by_g[secondary[0]], secondary[1])
                rootfaces[primary] = {'qkey': qkey, 'transform': identity}
                rootfaces[secondary] = {
                    'qkey': qkey, 'transform': quad_c2_transform(su, pu)
                }
            elif 'quad' in (let, ret):
                if let == 'quad':
                    qet, qe, qf = let, le, lf
                    fet, fe, ff = ret, re, rf
                else:
                    qet, qe, qf = ret, re, rf
                    fet, fe, ff = let, le, lf

                if qet != 'quad' or fet != 'tri':
                    raise AMRMeshError(
                        'Mixed Quad materializer only supports Tri-Quad '
                        'immutable interfaces'
                    )
                qg = l2g[qe]
                qkey = ('fixed', qg, qf, fet, fe, ff)
                surface = ('root', qkey)
                rootkinds[surface] = 'fixed'
                rootfaces[qg, qf] = {
                    'qkey': qkey, 'transform': identity
                }
                fixed[surface] = (fet, fe, ff)

    for name, con in mesh.bcon.items():
        for etype, fidx, eidxs in con.items():
            if etype != 'quad':
                continue
            for eidx in eidxs:
                g = l2g[int(eidx)]
                qkey = ('boundary', name, g, fidx)
                rootkinds['root', qkey] = 'boundary'
                rootfaces[g, fidx] = {
                    'qkey': qkey, 'transform': identity
                }

    for g in pids_by_g:
        for fidx in range(4):
            if (g, fidx) not in rootfaces:
                raise AMRMeshError(
                    f'Root Quad {g} face {fidx} is unresolved in mixed '
                    'root topology'
                )

    return rootfaces, rootkinds, fixed


def _order_fixed_line1x2_faces(coarse_nodes, fine_faces, store, tol=1e-10):
    coarse_nodes = tuple(map(int, coarse_nodes))
    if len(coarse_nodes) != 2 or len(fine_faces) != 2:
        raise AMRMeshError('Mixed line-1x2 requires one Line and two halves')

    cpts = store.coords(np.asarray(coarse_nodes, dtype=np.int64))
    direction = cpts[1] - cpts[0]
    denom = float(np.dot(direction, direction))
    if denom <= tol*tol:
        raise AMRMeshError('Degenerate immutable coarse Line interface')

    scale = max(1.0, float(np.max(np.abs(cpts), initial=0.0)))
    slots = [None, None]
    for face in fine_faces:
        fpts = store.coords(np.asarray(face, dtype=np.int64))
        t = ((fpts - cpts[0]) @ direction)/denom
        expected = np.array([0.0, 0.5, 1.0])
        matched = []
        for value, point in zip(t, fpts):
            ix = int(np.argmin(np.abs(expected - value)))
            if abs(expected[ix] - value) > tol*scale:
                raise AMRMeshError(
                    'Fine Quad Line does not match immutable coarse Line'
                )
            # Orthogonal geometry mismatch is checked explicitly as well.
            proj = cpts[0] + expected[ix]*direction
            if np.max(np.abs(point - proj), initial=0.0) > tol*scale:
                raise AMRMeshError(
                    'Fine Quad Line is not collinear with immutable Line'
                )
            matched.append(ix)

        key = frozenset(matched)
        if key == frozenset((0, 1)):
            slot = 0
        elif key == frozenset((1, 2)):
            slot = 1
        else:
            raise AMRMeshError('Invalid immutable Line-1x2 half coverage')
        if slots[slot] is not None:
            raise AMRMeshError('Duplicate immutable Line-1x2 half')
        slots[slot] = tuple(map(int, face))

    if any(face is None for face in slots):
        raise AMRMeshError('Incomplete immutable Line-1x2 coverage')
    return tuple(slots)


def materialize_native_mixed_quad_tree(mesh, tree, *, comm_size=1):
    _validate_mixed_quad_scope(mesh, tree, comm_size)

    l2g, pids_by_g, tags_by_g = _quad_local_maps(mesh)
    _validate_quad_root_geometry(mesh, pids_by_g)
    _validate_quad_root_coverage(tree, pids_by_g)
    rootfaces, rootkinds, fixed = _derive_mixed_quad_root_face_topology(
        mesh, pids_by_g, l2g
    )

    leaves = list(tree.leaves())
    groups, pairs = _validate_quad_balanced_and_pair(
        leaves, rootfaces, rootkinds
    )

    # Immutable non-Quad neighbours are level 0.  A touching active Quad leaf
    # may therefore be level 0 or 1, but never level 2.
    for surface in fixed:
        fragments = [
            frag for side in groups.get(surface, {}).values() for frag in side
        ]
        if not fragments:
            raise AMRMeshError('Mixed immutable interface has no Quad side')
        if max(f['level'] for f in fragments) > 1:
            raise AMRMeshError(
                'Quad refinement exceeds 2:1 against immutable Tri neighbour'
            )

    store = _NativeNodeStore(mesh.node_idxs, mesh.node_locs)
    leaf_pids = _materialize_quad_leaf_pids(pids_by_g, leaves, store)
    leaf_order = tuple(leaves)
    adapted_eidx = {leaf: i for i, leaf in enumerate(leaf_order)}

    codec = (
        [f'eles/tri/face/{fidx}' for fidx in range(3)] +
        [f'eles/quad/face/{fidx}' for fidx in range(4)]
    )
    codec_index = {value: i for i, value in enumerate(codec)}

    def bc_cidx(name):
        key = f'bc/{name}'
        if key not in codec_index:
            codec_index[key] = len(codec)
            codec.append(key)
        return codec_index[key]

    ntri = len(mesh.spts_nodes['tri'])
    nquad = len(leaf_order)
    tri_cidx = np.zeros((ntri, 3), dtype=np.int16)
    tri_off = np.zeros((ntri, 3), dtype=np.int64)
    tri_set = np.zeros((ntri, 3), dtype=bool)
    quad_cidx = np.zeros((nquad, 4), dtype=np.int16)
    quad_off = np.zeros((nquad, 4), dtype=np.int64)
    quad_set = np.zeros((nquad, 4), dtype=bool)

    def set_tri(eidx, fidx, cidx, off):
        if tri_set[eidx, fidx]:
            raise AMRMeshError(
                f'Immutable Tri {eidx} face {fidx} assigned more than once'
            )
        tri_cidx[eidx, fidx] = cidx
        tri_off[eidx, fidx] = off
        tri_set[eidx, fidx] = True

    def set_quad(leaf, fidx, cidx, off):
        eidx = adapted_eidx[leaf]
        if quad_set[eidx, fidx]:
            raise AMRMeshError(
                f'Quad leaf {leaf} face {fidx} assigned more than once'
            )
        quad_cidx[eidx, fidx] = cidx
        quad_off[eidx, fidx] = off
        quad_set[eidx, fidx] = True

    # Preserve immutable Tri-Tri connectivity exactly.  Tri-Quad root faces
    # are reconstructed below against the adapted Quad side.
    if mesh.con:
        lhs, rhs = mesh.con
        for i in range(len(lhs)):
            let, lf = mesh.cidxmap[int(lhs.cidxs[i])]
            ret, rf = mesh.cidxmap[int(rhs.cidxs[i])]
            le, re = int(lhs.eidxs[i]), int(rhs.eidxs[i])
            if let == ret == 'tri':
                set_tri(le, lf, codec_index[f'eles/tri/face/{rf}'], re)
                set_tri(re, rf, codec_index[f'eles/tri/face/{lf}'], le)

    # Preserve immutable Tri boundary identity.
    for name, con in mesh.bcon.items():
        for etype, fidx, eidxs in con.items():
            if etype == 'tri':
                for eidx in eidxs:
                    set_tri(int(eidx), fidx, bc_cidx(name), -1)

    # Adapted Quad physical boundaries.
    for surface, sides in groups.items():
        if surface[0] != 'root' or rootkinds.get(surface) != 'boundary':
            continue
        name = surface[1][1]
        for fragments in sides.values():
            for frag in fragments:
                set_quad(frag['leaf'], frag['fidx'], bc_cidx(name), -1)

    mortar_groups = defaultdict(list)
    equal_pairs_seen = set()
    for lface, rface in pairs:
        lleaf, lfidx = lface['leaf'], lface['fidx']
        rleaf, rfidx = rface['leaf'], rface['fidx']
        if lface['level'] == rface['level']:
            key = frozenset({(lleaf, lfidx), (rleaf, rfidx)})
            if key in equal_pairs_seen:
                continue
            equal_pairs_seen.add(key)
            set_quad(
                lleaf, lfidx, codec_index[f'eles/quad/face/{rfidx}'],
                adapted_eidx[rleaf]
            )
            set_quad(
                rleaf, rfidx, codec_index[f'eles/quad/face/{lfidx}'],
                adapted_eidx[lleaf]
            )
        else:
            coarse, fine = (
                (lface, rface) if lface['level'] < rface['level']
                else (rface, lface)
            )
            mortar_groups[('quad', coarse['leaf'], coarse['fidx'])].append(
                fine
            )

    mortars = []

    # Quad-Quad coarse/fine interfaces retain accepted ordering verbatim.
    for _, cleaf, cfidx in sorted(
        mortar_groups, key=lambda key: (adapted_eidx[key[1]], key[2])
    ):
        fines = mortar_groups['quad', cleaf, cfidx]
        if len(fines) != 2:
            raise AMRMeshError(
                f'Coarse Quad leaf {cleaf} face {cfidx} has {len(fines)} '
                'fine neighbours'
            )
        fine_faces, by_face = [], {}
        for frag in fines:
            fmap = quad_face_corner_indices(frag['fidx'])
            row = np.asarray(leaf_pids[frag['leaf']])
            face = tuple(int(x) for x in row[list(fmap)])
            fine_faces.append(face)
            by_face[face] = frag

        ordered = quad_order_line1x2_faces(
            leaf_pids[cleaf], cfidx, fine_faces,
            QuadNodeStore(store.coords, store.allocate),
        )
        reidx, rfidx = [], []
        for face in ordered:
            frag = by_face[face]
            reidx.append(adapted_eidx[frag['leaf']])
            rfidx.append(frag['fidx'])
            set_quad(frag['leaf'], frag['fidx'], -1, -2)
        set_quad(cleaf, cfidx, -1, -2)
        mortars.append(MixedLineMortarRecord(
            name='', left_etype='quad', left_eidx=adapted_eidx[cleaf],
            left_fidx=cfidx, right_etype=('quad', 'quad'),
            right_eidx=tuple(reidx), right_fidx=tuple(rfidx),
        ))

    # Fixed Tri-Quad root interfaces are either ordinary conforming Line
    # interfaces or one accepted Line-1x2 mortar with Tri as coarse side.
    for surface, (fet, fe, ff) in sorted(fixed.items(), key=repr):
        fragments = sorted(
            (frag for side in groups[surface].values() for frag in side),
            key=lambda d: (d['interval'], d['leaf'], d['fidx'])
        )
        if len(fragments) == 1 and fragments[0]['level'] == 0:
            frag = fragments[0]
            set_tri(
                fe, ff, codec_index[f'eles/quad/face/{frag["fidx"]}'],
                adapted_eidx[frag['leaf']]
            )
            set_quad(
                frag['leaf'], frag['fidx'],
                codec_index[f'eles/tri/face/{ff}'], fe
            )
            continue

        if len(fragments) != 2 or any(f['level'] != 1 for f in fragments):
            raise AMRMeshError(
                'Immutable Tri interface requires one L0 or two L1 Quad faces'
            )

        from pyfr.shapes import TriShape
        tri_row = np.asarray(mesh.spts_nodes['tri'][fe])
        tids = TriShape.face_corner_pts_idxs(ff, len(tri_row))
        coarse_nodes = tuple(int(x) for x in tri_row[tids])
        fine_faces, by_face = [], {}
        for frag in fragments:
            fmap = quad_face_corner_indices(frag['fidx'])
            row = np.asarray(leaf_pids[frag['leaf']])
            face = tuple(int(x) for x in row[list(fmap)])
            fine_faces.append(face)
            by_face[face] = frag

        ordered = _order_fixed_line1x2_faces(
            coarse_nodes, fine_faces, store
        )
        reidx, rfidx = [], []
        for face in ordered:
            frag = by_face[face]
            reidx.append(adapted_eidx[frag['leaf']])
            rfidx.append(frag['fidx'])
            set_quad(frag['leaf'], frag['fidx'], -1, -2)
        set_tri(fe, ff, -1, -2)
        mortars.append(MixedLineMortarRecord(
            name='', left_etype=fet, left_eidx=fe, left_fidx=ff,
            right_etype=('quad', 'quad'), right_eidx=tuple(reidx),
            right_fidx=tuple(rfidx),
        ))

    if not np.all(tri_set):
        missing = [tuple(map(int, r)) for r in np.argwhere(~tri_set)[:8]]
        raise AMRMeshError(
            f'Immutable Tri faces remain unresolved (first={missing})'
        )
    if not np.all(quad_set):
        missing = [tuple(map(int, r)) for r in np.argwhere(~quad_set)[:8]]
        raise AMRMeshError(
            f'Adapted Quad faces remain unresolved (first={missing})'
        )

    # Mortar names are assigned only after the complete deterministic mixed
    # list is known, independent of which topology branch produced a record.
    mortars = tuple(
        MixedLineMortarRecord(
            name=f'm{i}', left_etype=m.left_etype,
            left_eidx=m.left_eidx, left_fidx=m.left_fidx,
            right_etype=m.right_etype, right_eidx=m.right_eidx,
            right_fidx=m.right_fidx,
        )
        for i, m in enumerate(mortars)
    )

    quad_nodes = np.array(
        [leaf_pids[leaf] for leaf in leaf_order], dtype=np.int64
    )
    tri_nodes = np.array(mesh.spts_nodes['tri'], copy=True, dtype=np.int64)
    all_ids = set(int(i) for i in mesh.node_idxs) | {
        node for node, _ in store.new_nodes
    }
    node_ids = np.array(sorted(all_ids), dtype=np.int64)
    node_locs = store.coords(node_ids)
    leaf_tags = np.array(
        [tags_by_g[leaf[0]] for leaf in leaf_order], dtype=np.uint64
    )

    return AdaptedMixedQuadRawMesh(
        root_mesh_uuid=mesh.uuid, node_ids=node_ids, node_locs=node_locs,
        tri_nodes=tri_nodes, tri_faces_cidx=tri_cidx,
        tri_faces_off=tri_off,
        tri_tags=np.array(mesh.tags['tri'], copy=True, dtype=np.uint64),
        quad_nodes=quad_nodes, quad_faces_cidx=quad_cidx,
        quad_faces_off=quad_off, codec=tuple(codec), mortars=mortars,
        leaf_order=leaf_order, leaf_tags=leaf_tags,
    )

# ===========================================================================
# V10J - mixed Tet+Pyramid+Hex materialization with Hex-only AMR
# ===========================================================================

@dataclass(frozen=True)
class MixedQuadMortarRecord:

    name: str
    left_etype: str
    left_eidx: int
    left_fidx: int
    right_etype: tuple
    right_eidx: tuple
    right_fidx: tuple
    format: str = 'one-to-many-v1'
    template: str = 'quad-2x2'


@dataclass(frozen=True)
class AdaptedMixedHexRawMesh:

    root_mesh_uuid: str
    node_ids: np.ndarray
    node_locs: np.ndarray
    fixed_nodes: dict
    fixed_faces_cidx: dict
    fixed_faces_off: dict
    fixed_tags: dict
    fixed_curved: dict
    hex_nodes: np.ndarray
    hex_curved: np.ndarray
    hex_faces_cidx: np.ndarray
    hex_faces_off: np.ndarray
    codec: tuple
    mortars: tuple
    leaf_order: tuple
    leaf_tags: np.ndarray


def _mixed_hex_fixed_info():
    from pyfr.shapes import PyrShape, TetShape

    return {
        'pyr': (PyrShape, len(PyrShape.faces)),
        'tet': (TetShape, len(TetShape.faces)),
    }


def _validate_mixed_hex_scope(mesh, tree, comm_size):
    if comm_size != 1:
        raise AMRMeshError(
            'Mixed Hex materializer requires a single-rank communicator'
        )
    if mesh.uuid != tree.root_mesh_uuid:
        raise AMRMeshError(
            'HexLeafTree root_mesh_uuid does not match the mixed root mesh'
        )
    if set(mesh.etypes) != {'hex', 'pyr', 'tet'}:
        raise AMRMeshError(
            'Mixed Hex materializer requires exactly Tet+Pyramid+Hex root '
            'topology'
        )
    # Immutable Tet/Pyramid geometry may be high-order/curved and Hex roots
    # may be high-order.  Hex children are exact restrictions of each
    # immutable root PyFR geometry map; fixed elements are copied verbatim.
    from pyfr.shapes import HexShape, PyrShape, TetShape
    shapes = {'hex': HexShape, 'pyr': PyrShape, 'tet': TetShape}
    for etype, shape in shapes.items():
        try:
            shape.order_from_npts(mesh.spts_nodes[etype].shape[1])
        except ValueError as exc:
            raise AMRMeshError(
                f'Unsupported {etype} shape-point count in mixed Hex root'
            ) from exc
    if mesh.con_p:
        raise AMRMeshError(
            'Mixed Hex materializer requires no MPI root connectivity'
        )
    if mesh.mcon:
        raise AMRMeshError(
            'Mixed Hex materializer requires no pre-existing root mortars'
        )
    if mesh.raw is not None and 'periodic' in mesh.raw:
        raise AMRMeshError(
            'Mixed Hex materializer requires no periodic connectivity'
        )
    for name in mesh.bcon:
        if 'periodic' in name.lower():
            raise AMRMeshError(
                'Mixed Hex materializer requires no periodic connectivity; '
                f'found boundary {name!r}'
            )


def _fixed_face_corner_nodes(mesh, etype, eidx, fidx):
    info = _mixed_hex_fixed_info()
    try:
        shape, _ = info[etype]
    except KeyError as exc:
        raise AMRMeshError(
            f'Unsupported immutable mixed Hex element type {etype!r}'
        ) from exc

    row = np.asarray(mesh.spts_nodes[etype][eidx])
    idxs = shape.face_corner_pts_idxs(fidx, len(row))
    return tuple(int(x) for x in row[np.asarray(idxs, dtype=int)])


def _derive_mixed_hex_root_face_topology(mesh, pids_by_g, l2g):
    from pyfr.amr import hex_d4_transform

    rootfaces = {}
    rootkinds = {}
    fixed = {}
    identity = (1, 0, 0, 0, 1, 0)

    def face(cidx):
        try:
            return mesh.cidxmap[int(cidx)]
        except KeyError as exc:
            raise AMRMeshError('Invalid mixed Hex root face codec') from exc

    if mesh.con:
        lhs, rhs = mesh.con
        for i in range(len(lhs)):
            let, lf = face(lhs.cidxs[i])
            ret, rf = face(rhs.cidxs[i])
            le, re = int(lhs.eidxs[i]), int(rhs.eidxs[i])

            if let == ret == 'hex':
                lg, rg = l2g[le], l2g[re]
                a, b = (lg, lf), (rg, rf)
                primary, secondary = (a, b) if a < b else (b, a)
                qkey = ('interior', primary)
                rootkinds['root', qkey] = 'interior'

                pu = _face_uv(pids_by_g, *primary)
                su = _face_uv(pids_by_g, *secondary)
                rootfaces[primary] = {'qkey': qkey, 'transform': identity}
                rootfaces[secondary] = {
                    'qkey': qkey,
                    'transform': hex_d4_transform(su, pu),
                }
            elif 'hex' in (let, ret):
                if let == 'hex':
                    he, hf = le, lf
                    fet, fe, ff = ret, re, rf
                else:
                    he, hf = re, rf
                    fet, fe, ff = let, le, lf

                if fet != 'pyr':
                    raise AMRMeshError(
                        'Mixed Hex materializer only supports Hex-Pyramid '
                        'immutable interfaces; direct Hex-Tet contact is '
                        'unsupported'
                    )
                coarse = _fixed_face_corner_nodes(mesh, fet, fe, ff)
                if len(coarse) != 4:
                    raise AMRMeshError(
                        'Hex-Pyramid coupling requires the Pyramid '
                        'quadrilateral base'
                    )

                hg = l2g[he]
                qkey = ('fixed', hg, hf, fet, fe, ff)
                surface = ('root', qkey)
                rootkinds[surface] = 'fixed'
                rootfaces[hg, hf] = {
                    'qkey': qkey, 'transform': identity
                }
                fixed[surface] = (fet, fe, ff)

    for name, con in mesh.bcon.items():
        for etype, fidx, eidxs in con.items():
            if etype != 'hex':
                continue
            for eidx in eidxs:
                g = l2g[int(eidx)]
                qkey = ('boundary', name, g, fidx)
                rootkinds['root', qkey] = 'boundary'
                rootfaces[g, fidx] = {
                    'qkey': qkey, 'transform': identity
                }

    for g in pids_by_g:
        for fidx in range(6):
            if (g, fidx) not in rootfaces:
                raise AMRMeshError(
                    f'Root Hex {g} face {fidx} is unresolved in mixed root '
                    'topology'
                )

    return rootfaces, rootkinds, fixed


def _order_hex_quad2x2_faces(
    coarse_row, fidx, fine_faces, store, tol=1e-10
):
    from pyfr.polys import get_polybasis
    from pyfr.shapes import HexShape, QuadShape, proj_pts

    coarse_row = np.asarray(coarse_row)
    order = HexShape.order_from_npts(len(coarse_row))
    hspts = HexShape.std_ele(order)
    basis = get_polybasis('hex', order, hspts)
    qpts = QuadShape.std_ele(2)
    project = HexShape.faces[fidx][1]
    fpts = proj_pts(project, qpts)
    gphys = basis.nodal_basis_at(fpts) @ store.coords(coarse_row)

    scale = max(1.0, float(np.max(np.abs(gphys), initial=0.0)))
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

    ordered = [None]*4
    for face in fine_faces:
        target = []
        seen = set()
        for node in face:
            coord = store.coords(np.asarray([node]))[0]
            errors = np.max(np.abs(gphys - coord), axis=1)
            match = np.flatnonzero(errors <= tol*scale)
            if len(match) != 1:
                raise AMRMeshError(
                    'Fine Hex face does not match restricted coarse Hex '
                    '2x2 grid'
                )
            gi = int(match[0])
            if gi in seen:
                raise AMRMeshError('Duplicate fine Hex mortar corner')
            seen.add(gi)
            target.append(tuple(float(v) for v in qpts[gi]))

        try:
            slot = quadrants.index(frozenset(target))
        except ValueError:
            raise AMRMeshError(
                'Fine Hex face is not a valid restricted 2x2 quadrant'
            ) from None
        if ordered[slot] is not None:
            raise AMRMeshError('Duplicate restricted Hex 2x2 quadrant')
        ordered[slot] = tuple(map(int, face))

    if any(face is None for face in ordered):
        raise AMRMeshError('Incomplete restricted Hex 2x2 coverage')
    return tuple(ordered)


def _order_fixed_quad2x2_faces(coarse_nodes, fine_faces, store, tol=1e-10):
    from pyfr.shapes import QuadShape

    coarse_nodes = tuple(map(int, coarse_nodes))
    if len(coarse_nodes) != 4 or len(fine_faces) != 4:
        raise AMRMeshError(
            'Mixed quad-2x2 requires one Quad and four fine patches'
        )

    qpts = QuadShape.std_ele(1)
    cpts = store.coords(np.asarray(coarse_nodes, dtype=np.int64))
    lhs = np.column_stack((qpts, np.ones(len(qpts))))
    coeff, _, rank, _ = np.linalg.lstsq(lhs, cpts, rcond=None)
    if rank != 3:
        raise AMRMeshError('Degenerate immutable coarse Quad interface')

    fitted = lhs @ coeff
    scale = max(1.0, float(np.max(np.abs(cpts), initial=0.0)))
    if np.max(np.abs(fitted - cpts), initial=0.0) > tol*scale:
        raise AMRMeshError(
            'Immutable coarse Quad interface is outside affine mortar scope'
        )

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

    ordered = [None]*4
    for face in fine_faces:
        target = []
        seen = set()
        for node in face:
            coord = store.coords(np.array([node]))[0]
            errors = np.max(np.abs(gphys - coord), axis=1)
            match = np.flatnonzero(errors <= tol*scale)
            if len(match) != 1:
                raise AMRMeshError(
                    'Fine Hex face does not match immutable coarse 2x2 grid'
                )
            gi = int(match[0])
            if gi in seen:
                raise AMRMeshError('Duplicate fine quad mortar corner')
            seen.add(gi)
            target.append(tuple(float(v) for v in grid[gi]))

        try:
            slot = quadrants.index(frozenset(target))
        except ValueError:
            raise AMRMeshError(
                'Fine Hex face is not a valid immutable 2x2 quadrant'
            ) from None
        if ordered[slot] is not None:
            raise AMRMeshError('Duplicate immutable quad-2x2 quadrant')
        ordered[slot] = tuple(map(int, face))

    if any(face is None for face in ordered):
        raise AMRMeshError('Incomplete immutable quad-2x2 coverage')
    return tuple(ordered)


def _validate_mixed_hex_fixed_balance(groups, fixed):
    for surface in fixed:
        fragments = [
            frag for side in groups.get(surface, {}).values() for frag in side
        ]
        if not fragments:
            raise AMRMeshError('Mixed immutable interface has no Hex side')
        if max(f['level'] for f in fragments) > 1:
            raise AMRMeshError(
                'Hex refinement exceeds 2:1 against immutable Pyramid '
                'neighbour'
            )


def materialize_native_mixed_hex_tree(mesh, tree, *, comm_size=1):
    _validate_mixed_hex_scope(mesh, tree, comm_size)

    l2g, _, pids_by_g, tags_by_g = _hex_local_maps(mesh)
    _validate_root_coverage(tree, pids_by_g)
    rootfaces, rootkinds, fixed = _derive_mixed_hex_root_face_topology(
        mesh, pids_by_g, l2g
    )

    leaves = list(tree.leaves())
    groups, pairs = _validate_balanced_and_pair(leaves, rootfaces, rootkinds)

    _validate_mixed_hex_fixed_balance(groups, fixed)

    store = _NativeNodeStore(mesh.node_idxs, mesh.node_locs)
    leaf_pids = _materialize_isoparametric_hex_leaves(
        mesh, pids_by_g, leaves, store
    )
    leaf_order = tuple(leaves)
    adapted_eidx = {leaf: i for i, leaf in enumerate(leaf_order)}

    fixed_info = _mixed_hex_fixed_info()
    fixed_etypes = ('pyr', 'tet')
    codec = []
    for etype in (*fixed_etypes, 'hex'):
        nfaces = fixed_info[etype][1] if etype in fixed_info else 6
        codec.extend(f'eles/{etype}/face/{fidx}' for fidx in range(nfaces))
    codec_index = {value: i for i, value in enumerate(codec)}

    def bc_cidx(name):
        key = f'bc/{name}'
        if key not in codec_index:
            codec_index[key] = len(codec)
            codec.append(key)
        return codec_index[key]

    fixed_cidx = {}
    fixed_off = {}
    fixed_set = {}
    for etype in fixed_etypes:
        nfaces = fixed_info[etype][1]
        neles = len(mesh.spts_nodes[etype])
        fixed_cidx[etype] = np.zeros((neles, nfaces), dtype=np.int16)
        fixed_off[etype] = np.zeros((neles, nfaces), dtype=np.int64)
        fixed_set[etype] = np.zeros((neles, nfaces), dtype=bool)

    nhex = len(leaf_order)
    hex_cidx = np.zeros((nhex, 6), dtype=np.int16)
    hex_off = np.zeros((nhex, 6), dtype=np.int64)
    hex_set = np.zeros((nhex, 6), dtype=bool)

    def set_fixed(etype, eidx, fidx, cidx, off):
        if fixed_set[etype][eidx, fidx]:
            raise AMRMeshError(
                f'Immutable {etype} {eidx} face {fidx} assigned more than '
                'once'
            )
        fixed_cidx[etype][eidx, fidx] = cidx
        fixed_off[etype][eidx, fidx] = off
        fixed_set[etype][eidx, fidx] = True

    def set_hex(leaf, fidx, cidx, off):
        eidx = adapted_eidx[leaf]
        if hex_set[eidx, fidx]:
            raise AMRMeshError(
                f'Hex leaf {leaf} face {fidx} assigned more than once'
            )
        hex_cidx[eidx, fidx] = cidx
        hex_off[eidx, fidx] = off
        hex_set[eidx, fidx] = True

    # Preserve all immutable-immutable connectivity exactly.  Hex-Pyramid
    # root faces are reconstructed below against the adapted Hex side.
    if mesh.con:
        lhs, rhs = mesh.con
        for i in range(len(lhs)):
            let, lf = mesh.cidxmap[int(lhs.cidxs[i])]
            ret, rf = mesh.cidxmap[int(rhs.cidxs[i])]
            le, re = int(lhs.eidxs[i]), int(rhs.eidxs[i])
            if let != 'hex' and ret != 'hex':
                set_fixed(
                    let, le, lf, codec_index[f'eles/{ret}/face/{rf}'], re
                )
                set_fixed(
                    ret, re, rf, codec_index[f'eles/{let}/face/{lf}'], le
                )

    # Preserve all immutable boundary identities.
    for name, con in mesh.bcon.items():
        for etype, fidx, eidxs in con.items():
            if etype in fixed_info:
                for eidx in eidxs:
                    set_fixed(etype, int(eidx), fidx, bc_cidx(name), -1)

    # Adapted Hex physical boundaries.
    for surface, sides in groups.items():
        if surface[0] != 'root' or rootkinds.get(surface) != 'boundary':
            continue
        name = surface[1][1]
        for fragments in sides.values():
            for frag in fragments:
                set_hex(frag['leaf'], frag['fidx'], bc_cidx(name), -1)

    mortar_groups = defaultdict(list)
    equal_pairs_seen = set()
    for lface, rface in pairs:
        lleaf, lfidx = lface['leaf'], lface['fidx']
        rleaf, rfidx = rface['leaf'], rface['fidx']
        if lface['level'] == rface['level']:
            key = frozenset({(lleaf, lfidx), (rleaf, rfidx)})
            if key in equal_pairs_seen:
                continue
            equal_pairs_seen.add(key)
            set_hex(
                lleaf, lfidx, codec_index[f'eles/hex/face/{rfidx}'],
                adapted_eidx[rleaf]
            )
            set_hex(
                rleaf, rfidx, codec_index[f'eles/hex/face/{lfidx}'],
                adapted_eidx[lleaf]
            )
        else:
            coarse, fine = (
                (lface, rface) if lface['level'] < rface['level']
                else (rface, lface)
            )
            mortar_groups[coarse['leaf'], coarse['fidx']].append(fine)

    mortars = []

    # Hex-Hex coarse/fine faces use the accepted pure-Hex quad-2x2 ordering.
    for (cleaf, cfidx), fines in sorted(
        mortar_groups.items(), key=lambda kv: (
            adapted_eidx[kv[0][0]], kv[0][1]
        )
    ):
        if len(fines) != 4:
            raise AMRMeshError(
                f'Coarse Hex leaf {cleaf} face {cfidx} has {len(fines)} '
                'fine neighbours'
            )
        fine_faces, by_face = [], {}
        for frag in fines:
            row = np.asarray(leaf_pids[frag['leaf']])
            face = _hex_face_nodes(row, frag['fidx'])
            fine_faces.append(face)
            by_face[face] = frag

        ordered = _order_hex_quad2x2_faces(
            leaf_pids[cleaf], cfidx, fine_faces, store
        )
        reidx, rfidx = [], []
        for face in ordered:
            frag = by_face[face]
            reidx.append(adapted_eidx[frag['leaf']])
            rfidx.append(frag['fidx'])
            set_hex(frag['leaf'], frag['fidx'], -1, -2)
        set_hex(cleaf, cfidx, -1, -2)
        mortars.append(MixedQuadMortarRecord(
            name='', left_etype='hex', left_eidx=adapted_eidx[cleaf],
            left_fidx=cfidx, right_etype=('hex',)*4,
            right_eidx=tuple(reidx), right_fidx=tuple(rfidx),
        ))

    # Fixed Pyramid-Hex interfaces are conforming at L0 or one 1:4 mortar at
    # L1, with the immutable Pyramid quadrilateral base as coarse side.
    for surface, (fet, fe, ff) in sorted(fixed.items(), key=repr):
        fragments = sorted(
            (frag for side in groups[surface].values() for frag in side),
            key=lambda d: (d['rect'], d['leaf'], d['fidx'])
        )
        if len(fragments) == 1 and fragments[0]['level'] == 0:
            frag = fragments[0]
            set_fixed(
                fet, fe, ff, codec_index[f'eles/hex/face/{frag["fidx"]}'],
                adapted_eidx[frag['leaf']]
            )
            set_hex(
                frag['leaf'], frag['fidx'],
                codec_index[f'eles/{fet}/face/{ff}'], fe
            )
            continue

        if len(fragments) != 4 or any(f['level'] != 1 for f in fragments):
            raise AMRMeshError(
                'Immutable Pyramid interface requires one L0 or four L1 '
                'Hex faces'
            )

        coarse_nodes = _fixed_face_corner_nodes(mesh, fet, fe, ff)
        fine_faces, by_face = [], {}
        for frag in fragments:
            row = np.asarray(leaf_pids[frag['leaf']])
            face = _hex_face_nodes(row, frag['fidx'])
            fine_faces.append(face)
            by_face[face] = frag

        ordered = _order_fixed_quad2x2_faces(
            coarse_nodes, fine_faces, store
        )
        reidx, rfidx = [], []
        for face in ordered:
            frag = by_face[face]
            reidx.append(adapted_eidx[frag['leaf']])
            rfidx.append(frag['fidx'])
            set_hex(frag['leaf'], frag['fidx'], -1, -2)
        set_fixed(fet, fe, ff, -1, -2)
        mortars.append(MixedQuadMortarRecord(
            name='', left_etype=fet, left_eidx=fe, left_fidx=ff,
            right_etype=('hex',)*4, right_eidx=tuple(reidx),
            right_fidx=tuple(rfidx),
        ))

    for etype in fixed_etypes:
        if not np.all(fixed_set[etype]):
            missing = [
                tuple(map(int, row))
                for row in np.argwhere(~fixed_set[etype])[:8]
            ]
            raise AMRMeshError(
                f'Immutable {etype} faces remain unresolved (first={missing})'
            )
    if not np.all(hex_set):
        missing = [tuple(map(int, row)) for row in np.argwhere(~hex_set)[:8]]
        raise AMRMeshError(
            f'Adapted Hex faces remain unresolved (first={missing})'
        )

    mortars = tuple(
        MixedQuadMortarRecord(
            name=f'm{i}', left_etype=m.left_etype,
            left_eidx=m.left_eidx, left_fidx=m.left_fidx,
            right_etype=m.right_etype, right_eidx=m.right_eidx,
            right_fidx=m.right_fidx,
        )
        for i, m in enumerate(mortars)
    )

    hex_nodes = np.array(
        [leaf_pids[leaf] for leaf in leaf_order], dtype=np.int64
    )
    fixed_nodes = {
        et: np.array(mesh.spts_nodes[et], copy=True, dtype=np.int64)
        for et in fixed_etypes
    }
    fixed_tags = {
        et: np.array(mesh.tags[et], copy=True, dtype=np.uint64)
        for et in fixed_etypes
    }
    fixed_curved = {
        et: np.array(mesh.spts_curved[et], copy=True, dtype=bool)
        for et in fixed_etypes
    }
    all_ids = set(int(i) for i in mesh.node_idxs) | {
        node for node, _ in store.new_nodes
    }
    node_ids = np.array(sorted(all_ids), dtype=np.int64)
    node_locs = store.coords(node_ids)
    leaf_tags = np.array(
        [tags_by_g[leaf[0]] for leaf in leaf_order], dtype=np.uint64
    )
    g2l = {int(g): i for i, g in enumerate(mesh.eidxs['hex'])}
    hex_curved = np.array(
        [mesh.spts_curved['hex'][g2l[leaf[0]]] for leaf in leaf_order],
        dtype=bool
    )

    return AdaptedMixedHexRawMesh(
        root_mesh_uuid=mesh.uuid, node_ids=node_ids, node_locs=node_locs,
        fixed_nodes=fixed_nodes, fixed_faces_cidx=fixed_cidx,
        fixed_faces_off=fixed_off, fixed_tags=fixed_tags,
        fixed_curved=fixed_curved, hex_nodes=hex_nodes,
        hex_curved=hex_curved, hex_faces_cidx=hex_cidx,
        hex_faces_off=hex_off, codec=tuple(codec), mortars=mortars,
        leaf_order=leaf_order, leaf_tags=leaf_tags,
    )
