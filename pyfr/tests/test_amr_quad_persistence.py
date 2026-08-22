from io import StringIO

import h5py
import numpy as np
import pytest

from pyfr.amr import encode_quad_leaf_tree
from pyfr.amrmesh import materialize_native_quad_tree
from pyfr.amrwriter import write_adapted_quad_mesh
from pyfr.progress import NullProgressSequence
from pyfr.readers.gmsh import GmshReader
from pyfr.readers.native import NativeReader


def _two_quad_gmsh():
    coords = [
        (1, 0, 0, 0), (2, 1, 0, 0), (3, 2, 0, 0),
        (4, 0, 1, 0), (5, 1, 1, 0), (6, 2, 1, 0),
    ]
    quads = [(1, 2, 5, 4), (2, 3, 6, 5)]
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


def _build_adapted_quad(tmp_path):
    rootf = tmp_path / 'root.pyfrm'
    GmshReader(
        StringIO(_two_quad_gmsh()), NullProgressSequence()
    ).write(str(rootf), 1e-5)

    root = NativeReader(str(rootf))
    roots = tuple(int(i) for i in sorted(root.mesh.eidxs['quad']))
    leaves = [(roots[0], (q,)) for q in range(4)] + [(roots[1], ())]
    tree = encode_quad_leaf_tree(root.mesh.uuid, leaves)
    raw = materialize_native_quad_tree(root.mesh, tree)

    adaptedf = tmp_path / 'adapted.pyfrm'
    write_adapted_quad_mesh(raw, str(adaptedf), lintol=1e-5)
    root_uuid = root.mesh.uuid
    root.close()

    return adaptedf, tuple(tree.leaves()), root_uuid


def test_adapted_quad_mesh_round_trips_typed_ancestry(tmp_path):
    adaptedf, leaves, root_uuid = _build_adapted_quad(tmp_path)

    with h5py.File(adaptedf) as f:
        assert f['amr/template'][()].decode() == 'quadtree-2x2-v1'
        assert list(f['mortars']) == ['line-1x2']
        mortar = f['mortars/line-1x2']
        assert mortar.attrs['format'] == 'one-to-many-v1'
        assert mortar.attrs['template'] == 'line-1x2'
        assert 'mortar/general-v1' in [c.decode() for c in f['codec']]

    reader = NativeReader(str(adaptedf))
    try:
        assert reader.mesh.etypes == ['quad']
        assert reader.mesh.amr_tree.root_mesh_uuid == root_uuid
        assert tuple(reader.mesh.amr_tree.leaves()) == leaves
        assert len(reader.mesh.eidxs['quad']) == len(leaves) == 5

        mcon = reader.mesh.mcon['line-1x2']
        assert mcon.format == 'one-to-many-v1'
        assert mcon.template == 'line-1x2'
        assert mcon.nright == 2
        assert len(mcon) == 1
    finally:
        reader.close()


def test_adapted_quad_reader_rejects_unknown_template(tmp_path):
    adaptedf, _, _ = _build_adapted_quad(tmp_path)

    with h5py.File(adaptedf, 'a') as f:
        del f['amr/template']
        f['amr/template'] = np.array('unknown-template', dtype='S')

    with pytest.raises(
        ValueError, match='Unsupported persistent AMR template'
    ):
        NativeReader(str(adaptedf))


def test_adapted_quad_reader_rejects_leaf_count_mismatch(tmp_path):
    adaptedf, leaves, root_uuid = _build_adapted_quad(tmp_path)
    wrong = encode_quad_leaf_tree(root_uuid, [(leaves[-1][0], ())])

    with h5py.File(adaptedf, 'a') as f:
        for name, data in [
            ('root-eidx', wrong.root_eidx),
            ('path-offsets', wrong.path_offsets),
            ('path-data', wrong.path_data),
        ]:
            del f[f'amr/leaves/{name}']
            f[f'amr/leaves/{name}'] = data

    with pytest.raises(RuntimeError, match='leaf count'):
        NativeReader(str(adaptedf))


def test_adapted_quad_reader_rejects_corrupt_quadrants(tmp_path):
    adaptedf, _, _ = _build_adapted_quad(tmp_path)

    with h5py.File(adaptedf, 'a') as f:
        path = f['amr/leaves/path-data'][()]
        path[0] = 7
        del f['amr/leaves/path-data']
        f['amr/leaves/path-data'] = path

    with pytest.raises(ValueError, match='quadrants'):
        NativeReader(str(adaptedf))


def test_adapted_quad_identity_writer_has_no_mortar_metadata(tmp_path):
    rootf = tmp_path / 'root.pyfrm'
    GmshReader(
        StringIO(_two_quad_gmsh()), NullProgressSequence()
    ).write(str(rootf), 1e-5)

    root = NativeReader(str(rootf))
    try:
        roots = tuple(int(i) for i in sorted(root.mesh.eidxs['quad']))
        tree = encode_quad_leaf_tree(
            root.mesh.uuid, [(root_eidx, ()) for root_eidx in roots]
        )
        raw = materialize_native_quad_tree(root.mesh, tree)
        adaptedf = tmp_path / 'identity.pyfrm'
        write_adapted_quad_mesh(raw, str(adaptedf), lintol=1e-5)
    finally:
        root.close()

    with h5py.File(adaptedf) as f:
        assert 'mortars' not in f
        assert 'mortar/general-v1' not in [c.decode() for c in f['codec']]

    reader = NativeReader(str(adaptedf))
    try:
        assert not reader.mesh.mcon
        assert tuple(reader.mesh.amr_tree.leaves()) == tuple(tree.leaves())
    finally:
        reader.close()
