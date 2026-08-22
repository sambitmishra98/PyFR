from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from pyfr.partitioners.baseline import BaselinePartitioner
from pyfr.readers.shared_nodes import SharedNodesFinder


def _write_partition_mesh(path):
    fdtype = [('cidx', np.int16), ('off', np.int64)]
    hdtype = [
        ('curved', bool), ('faces', fdtype, 6), ('tags', np.uint64)
    ]
    tdtype = [
        ('curved', bool), ('faces', fdtype, 4), ('tags', np.uint64)
    ]

    codec = [
        'eles/hex',
        *[f'eles/hex/face/{i}' for i in range(6)],
        'eles/tet',
        *[f'eles/tet/face/{i}' for i in range(4)],
        'mortar/quad-tri', 'bc/wall', 'tag/fluid',
    ]
    cidx = {name: i for i, name in enumerate(codec)}

    hexes = np.zeros(2, dtype=hdtype)
    tets = np.zeros(4, dtype=tdtype)
    for eles in (hexes, tets):
        eles['faces']['cidx'] = cidx['bc/wall']
        eles['faces']['off'] = -1
        eles['tags'] = 1

    # A conforming hex-to-hex face joins the two mortar groups.
    hexes[0]['faces'][2] = (cidx['eles/hex/face/4'], 1)
    hexes[1]['faces'][4] = (cidx['eles/hex/face/2'], 0)

    # Mark two coarse and four fine faces as mortar faces.
    hexes['faces'][:, 5] = (cidx['mortar/quad-tri'], -2)
    tets['faces'][:, 0] = (cidx['mortar/quad-tri'], -2)

    mdtype = [
        ('coarse_cidx', np.int16),
        ('coarse_eidx', np.int64),
        ('fine_cidx', np.int16, 2),
        ('fine_eidx', np.int64, 2),
        ('diagonal', np.uint8),
        ('quality', np.float64),
    ]
    mortars = np.zeros(2, dtype=mdtype)
    mortars['coarse_cidx'] = cidx['eles/hex/face/5']
    mortars['coarse_eidx'] = [0, 1]
    mortars['fine_cidx'] = cidx['eles/tet/face/0']
    mortars['fine_eidx'] = [[0, 1], [2, 3]]
    mortars['quality'] = 1

    with h5py.File(path, 'w') as mesh:
        mesh['codec'] = np.asarray(codec, dtype='S')
        mesh['eles/hex'] = hexes
        mesh['eles/tet'] = tets
        mesh['mortars/quad-tri'] = mortars


def test_mortar_participants_remain_on_one_partition(tmp_path):
    path = tmp_path / 'mesh.pyfrm'
    _write_partition_mesh(path)

    partitioner = BaselinePartitioner(
        [1, 1], elewts={'hex': 1, 'tet': 1}
    )
    with h5py.File(path) as mesh:
        pinfo = partitioner.partition(mesh)

    (peles, pregions), (neighbours, nregions) = pinfo
    owners = {'hex': np.empty(2, dtype=int),
              'tet': np.empty(4, dtype=int)}
    for rank, region in enumerate(pregions):
        owners['hex'][peles[region[0]:region[1]]] = rank
        owners['tet'][peles[region[1]:region[2]]] = rank

    groups = [
        [owners['hex'][0], owners['tet'][0], owners['tet'][1]],
        [owners['hex'][1], owners['tet'][2], owners['tet'][3]],
    ]
    assert all(len(set(group)) == 1 for group in groups)
    assert groups[0][0] != groups[1][0]
    assert neighbours.tolist() == [1, 0]
    assert nregions.tolist() == [0, 1, 2]


def test_shared_nodes_empty_response():
    finder = object.__new__(SharedNodesFinder)
    finder.comm = SimpleNamespace(size=3)

    data, counts = finder._build_responses(
        np.empty(0, dtype=int), np.empty(0, dtype=np.int32)
    )

    assert data.size == 0
    assert counts.tolist() == [0, 0, 0]


def _write_general_partition_mesh(path):
    fdtype = [('cidx', np.int16), ('off', np.int64)]
    hdtype = [
        ('curved', bool), ('faces', fdtype, 6), ('tags', np.uint64)
    ]

    codec = [
        'eles/hex',
        *[f'eles/hex/face/{i}' for i in range(6)],
        'mortar/general-v1', 'bc/wall', 'tag/fluid',
    ]
    cidx = {name: i for i, name in enumerate(codec)}

    hexes = np.zeros(10, dtype=hdtype)
    hexes['faces']['cidx'] = cidx['bc/wall']
    hexes['faces']['off'] = -1
    hexes['tags'] = 1

    # Join the two five-element groups weakly through their left elements.
    hexes[0]['faces'][2] = (cidx['eles/hex/face/4'], 5)
    hexes[5]['faces'][4] = (cidx['eles/hex/face/2'], 0)

    # Each group is one left plus four right participants.
    for base in (0, 5):
        hexes[base]['faces'][5] = (cidx['mortar/general-v1'], -2)
        hexes['faces'][base + 1:base + 5, 0] = (
            cidx['mortar/general-v1'], -2
        )

    mdtype = [
        ('left_cidx', np.int16),
        ('left_eidx', np.int64),
        ('right_cidx', np.int16, 4),
        ('right_eidx', np.int64, 4),
    ]
    mortars = np.zeros(2, dtype=mdtype)
    mortars['left_cidx'] = cidx['eles/hex/face/5']
    mortars['left_eidx'] = [0, 5]
    mortars['right_cidx'] = cidx['eles/hex/face/0']
    mortars['right_eidx'] = [[1, 2, 3, 4], [6, 7, 8, 9]]

    with h5py.File(path, 'w') as mesh:
        mesh['codec'] = np.asarray(codec, dtype='S')
        mesh['eles/hex'] = hexes
        dset = mesh.create_dataset('mortars/h-quad2x2', data=mortars)
        dset.attrs['format'] = 'one-to-many-v1'
        dset.attrs['template'] = 'quad-2x2'


def test_general_mortar_participants_remain_on_one_partition(tmp_path):
    path = tmp_path / 'mesh-general.pyfrm'
    _write_general_partition_mesh(path)

    partitioner = BaselinePartitioner([1, 1], elewts={'hex': 1})
    with h5py.File(path) as mesh:
        pinfo = partitioner.partition(mesh)

    (peles, pregions), (_, _) = pinfo
    owners = np.empty(10, dtype=int)
    for rank, region in enumerate(pregions):
        owners[peles[region[0]:region[1]]] = rank

    groups = [owners[[0, 1, 2, 3, 4]], owners[[5, 6, 7, 8, 9]]]
    assert all(len(set(group)) == 1 for group in groups)
    assert groups[0][0] != groups[1][0]


def test_general_mortar_connectivity_sides():
    from pyfr.readers.native import MortarConnectivity

    dtype = [
        ('left_cidx', np.int16), ('left_eidx', np.int64),
        ('right_cidx', np.int16, 4), ('right_eidx', np.int64, 4),
    ]
    records = np.zeros(2, dtype=dtype)
    records['left_cidx'] = 0
    records['left_eidx'] = [2, 3]
    records['right_cidx'] = 1
    records['right_eidx'] = [[4, 5, 6, 7], [8, 9, 10, 11]]
    mcon = MortarConnectivity(
        'h', records, {0: ('hex', 5), 1: ('hex', 0)},
        format='one-to-many-v1', template='quad-2x2'
    )

    assert mcon.nright == 4
    assert mcon.format == 'one-to-many-v1'
    assert mcon.template == 'quad-2x2'
    assert np.array_equal(mcon.side('left').eidxs, [2, 3])
    assert np.array_equal(mcon.side('right', 2).eidxs, [6, 10])


def test_general_line_mortar_connectivity_sides():
    from pyfr.readers.native import MortarConnectivity

    dtype = [
        ('left_cidx', np.int16), ('left_eidx', np.int64),
        ('right_cidx', np.int16, 2), ('right_eidx', np.int64, 2),
    ]
    records = np.zeros(2, dtype=dtype)
    records['left_cidx'] = 0
    records['left_eidx'] = [2, 3]
    records['right_cidx'] = 1
    records['right_eidx'] = [[4, 5], [6, 7]]
    mcon = MortarConnectivity(
        'h', records, {0: ('quad', 1), 1: ('quad', 3)},
        format='one-to-many-v1', template='line-1x2'
    )

    assert mcon.nright == 2
    assert mcon.format == 'one-to-many-v1'
    assert mcon.template == 'line-1x2'
    assert np.array_equal(mcon.side('left').eidxs, [2, 3])
    assert np.array_equal(mcon.side('right', 1).eidxs, [5, 7])


def test_native_reader_constructs_general_mortar_connectivity(tmp_path):
    from pyfr.readers.native import Mesh, NativeReader

    path = tmp_path / 'mesh-general-reader.pyfrm'
    _write_general_partition_mesh(path)

    reader = object.__new__(NativeReader)
    reader.f = h5py.File(path, 'r')
    reader.mesh = Mesh(fname=str(path), raw=reader.f)

    codec = [c.decode() for c in reader.f['codec'][()]]
    cidxmap = {
        i: ('hex', int(name.rsplit('/', 1)[1]))
        for i, name in enumerate(codec)
        if name.startswith('eles/hex/face/')
    }
    lcidx = np.array([
        codec.index('eles/hex/face/5'),
        *([codec.index('eles/hex/face/0')]*4),
    ], dtype=np.int16)
    leidx = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    is_mortar = np.ones(5, dtype=bool)
    gids = np.arange(5, dtype=np.int64)
    g2l = {'hex': (gids, gids, gids)}

    try:
        reader._construct_mortar_con(
            g2l, cidxmap, is_mortar, lcidx, leidx
        )
        mcon = reader.mesh.mcon['h-quad2x2']
        assert mcon.format == 'one-to-many-v1'
        assert mcon.template == 'quad-2x2'
        assert mcon.nright == 4
        assert np.array_equal(mcon.records['left_eidx'], [0])
        assert np.array_equal(mcon.records['right_eidx'][0], [1, 2, 3, 4])
    finally:
        reader.f.close()


def test_amr_hex_root_family_is_exposed_as_atomic_group(tmp_path):
    from io import StringIO

    from pyfr.amr import encode_hex_leaf_tree
    from pyfr.amrmesh import materialize_native_hex_tree
    from pyfr.amrmpi import _serial_root_mesh
    from pyfr.amrwriter import write_adapted_mesh
    from pyfr.partitioners.base import BasePartitioner
    from pyfr.progress import NullProgressSequence
    from pyfr.readers.gmsh import GmshReader
    from pyfr.tests.test_amr_transaction import _two_hex_gmsh

    root_path = tmp_path / 'root.pyfrm'
    adapted_path = tmp_path / 'adapted.pyfrm'
    GmshReader(StringIO(_two_hex_gmsh()), NullProgressSequence()).write(
        str(root_path), 1e-5
    )
    root = _serial_root_mesh(str(root_path))
    tree = encode_hex_leaf_tree(
        root.uuid, [(0, (o,)) for o in range(8)] + [(1, ())]
    )
    raw = materialize_native_hex_tree(root, tree)
    write_adapted_mesh(raw, str(adapted_path))

    with h5py.File(adapted_path) as mesh:
        _, _, _, edisps, _ = BasePartitioner.construct_global_con(mesh)
        groups = BasePartitioner._amr_hex_family_groups(mesh, edisps)

    assert len(groups) == 1
    assert groups[0].tolist() == list(range(8))

    # On this deliberately tiny two-root mesh the root family and its
    # coarse-side mortar neighbour form one hard affinity component.  A
    # two-way split is therefore impossible and must fail rather than break
    # either locality contract.
    partitioner = BaselinePartitioner([1, 1], elewts={'hex': 1})
    with h5py.File(adapted_path) as mesh:
        with pytest.raises(RuntimeError, match='mesh has 1 parts'):
            partitioner.partition(mesh)


def test_amr_hex_family_group_rejects_mismatched_ancestry(tmp_path):
    path = tmp_path / 'mesh.pyfrm'
    _write_general_partition_mesh(path)

    with h5py.File(path, 'r+') as mesh:
        mesh['amr/version'] = np.int64(1)
        mesh['amr/root-mesh-uuid'] = np.bytes_('root')
        mesh['amr/leaves/root-eidx'] = np.arange(9, dtype=np.int64)
        mesh['amr/leaves/path-offsets'] = np.zeros(10, dtype=np.int64)
        mesh['amr/leaves/path-data'] = np.empty(0, dtype=np.uint8)

    partitioner = BaselinePartitioner([1, 1], elewts={'hex': 1})
    with h5py.File(path) as mesh:
        with pytest.raises(ValueError, match='ancestry does not match'):
            partitioner.partition(mesh)
