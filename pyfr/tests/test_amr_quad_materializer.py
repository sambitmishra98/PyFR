from io import StringIO

import numpy as np
import pytest

from pyfr.amr import encode_quad_leaf_tree
from pyfr.amrmesh import AMRMeshError, materialize_native_quad_tree
from pyfr.progress import NullProgressSequence
from pyfr.readers.gmsh import GmshReader
from pyfr.readers.native import NativeReader


def _two_quad_gmsh():
    coords = [
        (1, 0, 0, 0), (2, 1, 0, 0), (3, 2, 0, 0),
        (4, 0, 1, 0), (5, 1, 1, 0), (6, 2, 1, 0),
    ]
    quads = [
        (1, 2, 5, 4),
        (2, 3, 6, 5),
    ]
    boundary = [
        (1, 2), (2, 3), (3, 6),
        (6, 5), (5, 4), (4, 1),
    ]

    lines = [
        '$MeshFormat', '2.2 0 8', '$EndMeshFormat',
        '$PhysicalNames', '2', '2 1 "fluid"', '1 2 "wall"',
        '$EndPhysicalNames', '$Nodes', str(len(coords)),
    ]
    lines.extend(f'{i} {x} {y} {z}' for i, x, y, z in coords)
    lines.extend([
        '$EndNodes', '$Elements', str(len(boundary) + len(quads)),
    ])

    eid = 1
    for edge in boundary:
        lines.append(f'{eid} 1 2 2 2 ' + ' '.join(map(str, edge)))
        eid += 1
    for quad in quads:
        lines.append(f'{eid} 3 2 1 1 ' + ' '.join(map(str, quad)))
        eid += 1
    lines.append('$EndElements')

    return '\n'.join(lines) + '\n'


def _root_mesh(tmp_path):
    fname = tmp_path / 'root.pyfrm'
    GmshReader(StringIO(_two_quad_gmsh()), NullProgressSequence()).write(
        str(fname), 1e-5
    )
    return NativeReader(str(fname))


def _roots(mesh):
    return tuple(int(i) for i in sorted(mesh.eidxs['quad']))


def _one_root_split(mesh):
    left, right = _roots(mesh)
    leaves = [(left, (q,)) for q in range(4)] + [(right, ())]
    return encode_quad_leaf_tree(mesh.uuid, leaves)


def test_quad_materializer_identity_tree(tmp_path):
    reader = _root_mesh(tmp_path)
    try:
        roots = _roots(reader.mesh)
        tree = encode_quad_leaf_tree(
            reader.mesh.uuid, [(root, ()) for root in roots]
        )
        raw = materialize_native_quad_tree(reader.mesh, tree)

        assert raw.leaf_order == tuple(tree.leaves())
        assert raw.quad_nodes.shape == (2, 4)
        assert len(raw.mortars) == 0
        assert np.count_nonzero(raw.quad_faces_off >= 0) == 2
        assert np.count_nonzero(raw.quad_faces_off == -1) == 6
        assert np.array_equal(
            raw.leaf_tags,
            reader.mesh.tags['quad'][np.argsort(reader.mesh.eidxs['quad'])],
        )
    finally:
        reader.close()


def test_quad_materializer_one_to_two_mortar(tmp_path):
    reader = _root_mesh(tmp_path)
    try:
        raw = materialize_native_quad_tree(
            reader.mesh, _one_root_split(reader.mesh)
        )
        mortar, = raw.mortars

        assert len(raw.leaf_order) == 5
        assert len(raw.node_ids) == 11
        assert mortar.format == 'one-to-many-v1'
        assert mortar.template == 'line-1x2'
        assert len(mortar.right_eidx) == len(mortar.right_fidx) == 2
        assert np.count_nonzero(raw.quad_faces_off == -2) == 3
        assert np.count_nonzero(raw.quad_faces_off == -1) == 9
        assert np.count_nonzero(raw.quad_faces_off >= 0) // 2 == 4
        assert 4*5 == 2*4 + 9 + 3

        coords = {
            int(i): p for i, p in zip(raw.node_ids, raw.node_locs)
        }
        mids = []
        for eidx, fidx in zip(mortar.right_eidx, mortar.right_fidx):
            row = raw.quad_nodes[eidx]
            from pyfr.amr import quad_face_corner_indices
            nodes = row[list(quad_face_corner_indices(fidx))]
            mids.append(np.mean([coords[int(i)] for i in nodes], axis=0))

        # Coarse face is the right root's x-low face; low/high means its
        # canonical Line coordinate, which increases in y for this fixture.
        assert mids[0][1] < mids[1][1]
    finally:
        reader.close()


def test_quad_materializer_both_roots_refined_is_conforming(tmp_path):
    reader = _root_mesh(tmp_path)
    try:
        roots = _roots(reader.mesh)
        tree = encode_quad_leaf_tree(
            reader.mesh.uuid,
            [(root, (q,)) for root in roots for q in range(4)],
        )
        raw = materialize_native_quad_tree(reader.mesh, tree)

        assert len(raw.leaf_order) == 8
        assert len(raw.mortars) == 0
        assert np.count_nonzero(raw.quad_faces_off >= 0) // 2 == 10
        assert np.count_nonzero(raw.quad_faces_off == -1) == 12
    finally:
        reader.close()


def test_quad_materializer_rejects_unbalanced_tree(tmp_path):
    reader = _root_mesh(tmp_path)
    try:
        left, right = _roots(reader.mesh)
        leaves = [(right, ())]
        for q in range(4):
            if q in (1, 3):
                leaves.extend((left, (q, c)) for c in range(4))
            else:
                leaves.append((left, (q,)))
        tree = encode_quad_leaf_tree(reader.mesh.uuid, leaves)

        with pytest.raises(AMRMeshError, match='2:1'):
            materialize_native_quad_tree(reader.mesh, tree)
    finally:
        reader.close()


def test_quad_materializer_rejects_wrong_root_coverage(tmp_path):
    reader = _root_mesh(tmp_path)
    try:
        left, _ = _roots(reader.mesh)
        tree = encode_quad_leaf_tree(reader.mesh.uuid, [(left, ())])
        with pytest.raises(AMRMeshError, match='coverage'):
            materialize_native_quad_tree(reader.mesh, tree)
    finally:
        reader.close()


def test_quad_materializer_rejects_uuid_mismatch(tmp_path):
    reader = _root_mesh(tmp_path)
    try:
        roots = _roots(reader.mesh)
        tree = encode_quad_leaf_tree(
            'not-the-root', [(root, ()) for root in roots]
        )
        with pytest.raises(AMRMeshError, match='root_mesh_uuid'):
            materialize_native_quad_tree(reader.mesh, tree)
    finally:
        reader.close()


def test_quad_materializer_rejects_multi_rank(tmp_path):
    reader = _root_mesh(tmp_path)
    try:
        roots = _roots(reader.mesh)
        tree = encode_quad_leaf_tree(
            reader.mesh.uuid, [(root, ()) for root in roots]
        )
        with pytest.raises(AMRMeshError, match='single-rank'):
            materialize_native_quad_tree(reader.mesh, tree, comm_size=2)
    finally:
        reader.close()


def test_quad_materializer_rejects_preexisting_mortars(tmp_path):
    reader = _root_mesh(tmp_path)
    try:
        roots = _roots(reader.mesh)
        tree = encode_quad_leaf_tree(
            reader.mesh.uuid, [(root, ()) for root in roots]
        )
        reader.mesh.mcon = [object()]
        with pytest.raises(AMRMeshError, match='pre-existing'):
            materialize_native_quad_tree(reader.mesh, tree)
    finally:
        reader.close()


def test_quad_materializer_rejects_nonaffine_root(tmp_path):
    gmsh = '''$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
2 1 "fluid"
1 2 "wall"
$EndPhysicalNames
$Nodes
4
1 0 0 0
2 2 0 0
3 0 1 0
4 1.5 1 0
$EndNodes
$Elements
5
1 1 2 2 2 1 2
2 1 2 2 2 2 4
3 1 2 2 2 4 3
4 1 2 2 2 3 1
5 3 2 1 1 1 2 4 3
$EndElements
'''
    fname = tmp_path / 'nonaffine.pyfrm'
    GmshReader(StringIO(gmsh), NullProgressSequence()).write(
        str(fname), 1e-5
    )
    reader = NativeReader(str(fname))
    try:
        root, = _roots(reader.mesh)
        tree = encode_quad_leaf_tree(reader.mesh.uuid, [(root, ())])
        with pytest.raises(AMRMeshError, match='affine geometry scope'):
            materialize_native_quad_tree(reader.mesh, tree)
    finally:
        reader.close()


def test_quad_materializer_rejects_periodic_metadata(tmp_path):
    reader = _root_mesh(tmp_path)
    oldraw = reader.mesh.raw
    try:
        roots = _roots(reader.mesh)
        tree = encode_quad_leaf_tree(
            reader.mesh.uuid, [(root, ()) for root in roots]
        )
        reader.mesh.raw = {'periodic': object()}
        with pytest.raises(AMRMeshError, match='periodic'):
            materialize_native_quad_tree(reader.mesh, tree)
    finally:
        reader.mesh.raw = oldraw
        reader.close()
