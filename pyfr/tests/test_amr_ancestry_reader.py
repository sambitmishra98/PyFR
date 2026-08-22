import shutil
from io import StringIO

import h5py
import numpy as np
import pytest

from pyfr.amr import encode_hex_leaf_tree, encode_quad_leaf_tree
from pyfr.amrmesh import materialize_native_hex_tree
from pyfr.amrwriter import write_adapted_mesh
from pyfr.progress import NullProgressSequence
from pyfr.readers.gmsh import GmshReader
from pyfr.readers.native import NativeReader


def _two_hex_gmsh():
    coords = []
    nid = {}
    n = 1
    for z in (0, 1):
        for y in (0, 1):
            for x in (0, 1, 2):
                nid[x, y, z] = n
                coords.append((n, x, y, z))
                n += 1

    def hnodes(x0, x1):
        return [
            nid[x0, 0, 0], nid[x1, 0, 0],
            nid[x1, 1, 0], nid[x0, 1, 0],
            nid[x0, 0, 1], nid[x1, 0, 1],
            nid[x1, 1, 1], nid[x0, 1, 1],
        ]

    hexes = [hnodes(0, 1), hnodes(1, 2)]
    fmaps = [
        [0, 1, 2, 3], [0, 1, 5, 4], [1, 2, 6, 5],
        [3, 2, 6, 7], [0, 3, 7, 4], [4, 5, 6, 7],
    ]
    quads = []
    for hi, h in enumerate(hexes):
        for fi, fmap in enumerate(fmaps):
            if (hi, fi) in {(0, 2), (1, 4)}:
                continue
            quads.append([h[i] for i in fmap])

    lines = [
        '$MeshFormat', '2.2 0 8', '$EndMeshFormat',
        '$PhysicalNames', '2', '3 1 "fluid"', '2 2 "wall"',
        '$EndPhysicalNames', '$Nodes', str(len(coords)),
    ]
    lines.extend(f'{i} {x} {y} {z}' for i, x, y, z in coords)
    lines.extend(['$EndNodes', '$Elements', str(len(quads) + 2)])

    eid = 1
    for q in quads:
        lines.append(f'{eid} 3 2 2 2 ' + ' '.join(map(str, q)))
        eid += 1
    for h in hexes:
        lines.append(f'{eid} 5 2 1 1 ' + ' '.join(map(str, h)))
        eid += 1
    lines.append('$EndElements')

    return '\n'.join(lines) + '\n'


def _build_adapted(tmp_path, nsplit_leaves):
    reader = GmshReader(StringIO(_two_hex_gmsh()), NullProgressSequence())
    rootf = tmp_path / 'root.pyfrm'
    reader.write(str(rootf), 1e-5)
    nroot = NativeReader(str(rootf))
    mesh = nroot.mesh
    g = sorted(mesh.eidxs['hex'])

    if nsplit_leaves == 9:
        leaves = [(g[0], (o,)) for o in range(8)] + [(g[1], ())]
    else:
        raise ValueError('unsupported fixture size')

    tree = encode_hex_leaf_tree(mesh.uuid, leaves)
    raw = materialize_native_hex_tree(mesh, tree)
    adaptedf = tmp_path / 'adapted.pyfrm'
    write_adapted_mesh(raw, str(adaptedf), lintol=1e-5)
    nroot.close()

    return str(adaptedf), leaves, mesh.uuid


def test_non_amr_native_mesh_has_no_amr_tree(tmp_path):
    reader = GmshReader(StringIO(_two_hex_gmsh()), NullProgressSequence())
    rootf = tmp_path / 'root.pyfrm'
    reader.write(str(rootf), 1e-5)

    nreader = NativeReader(str(rootf))
    assert nreader.mesh.amr_tree is None
    nreader.close()


def test_adapted_mesh_round_trips_ancestry_from_disk(tmp_path):
    adaptedf, leaves, root_uuid = _build_adapted(tmp_path, 9)

    with h5py.File(adaptedf) as f:
        assert 'template' not in f['amr']

    nreader = NativeReader(adaptedf)
    tree = nreader.mesh.amr_tree
    assert tree is not None
    assert tree.nleaves == 9
    assert tree.root_mesh_uuid == root_uuid
    assert sorted(tree.leaves()) == sorted(leaves)
    nreader.close()


def test_reader_rejects_tree_leaf_count_not_matching_mesh(tmp_path):
    adaptedf, leaves, root_uuid = _build_adapted(tmp_path, 9)

    # Overwrite the persisted ancestry with a structurally VALID tree
    # (passes the D5A validator) whose leaf count does not match this
    # mesh's actual 9 Hexes.
    wrong_tree = encode_hex_leaf_tree(root_uuid, [(leaves[0][0], ())])
    with h5py.File(adaptedf, 'a') as f:
        del f['amr/leaves/root-eidx']
        del f['amr/leaves/path-offsets']
        del f['amr/leaves/path-data']
        f['amr/leaves/root-eidx'] = wrong_tree.root_eidx
        f['amr/leaves/path-offsets'] = wrong_tree.path_offsets
        f['amr/leaves/path-data'] = wrong_tree.path_data

    with pytest.raises(RuntimeError, match='leaf count'):
        NativeReader(adaptedf)


def test_reader_rejects_corrupted_ancestry_via_d5a_validator(tmp_path):
    adaptedf, leaves, root_uuid = _build_adapted(tmp_path, 9)

    with h5py.File(adaptedf, 'a') as f:
        pd = f['amr/leaves/path-data'][()]
        pd[:] = 0  # collapses distinct leaf paths -> duplicate/incomplete
        del f['amr/leaves/path-data']
        f['amr/leaves/path-data'] = pd

    with pytest.raises(ValueError):
        NativeReader(adaptedf)


def test_reader_rejects_quad_ancestry_on_hex_mesh(tmp_path):
    adaptedf, leaves, root_uuid = _build_adapted(tmp_path, 9)
    qtree = encode_quad_leaf_tree(
        root_uuid, [(i, ()) for i in range(len(leaves))]
    )

    with h5py.File(adaptedf, 'a') as f:
        f['amr/template'] = np.array('quadtree-2x2-v1', dtype='S')
        for name, data in [
            ('root-eidx', qtree.root_eidx),
            ('path-offsets', qtree.path_offsets),
            ('path-data', qtree.path_data),
        ]:
            del f[f'amr/leaves/{name}']
            f[f'amr/leaves/{name}'] = data

    with pytest.raises(RuntimeError, match='pure-Quad'):
        NativeReader(adaptedf)
