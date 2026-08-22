import numpy as np

from pyfr.readers.base import NodalMeshAssembler


def test_first_order_blocks_with_mixed_geometry_orders_are_merged():
    assembler = NodalMeshAssembler.__new__(NodalMeshAssembler)
    assembler._etype_map = {2: ('tri', 3), 9: ('tri', 6)}

    elements = {
        (2, (1,)): np.array([[0, 1, 2]]),
        (9, (1,)): np.array([[3, 4, 5, 6, 7, 8]]),
    }

    first_order = assembler._to_first_order(elements)

    assert np.array_equal(
        first_order['tri', (1,)],
        [[0, 1, 2], [3, 4, 5]],
    )


def test_single_volume_face_remains_one_dimensional():
    assembler = NodalMeshAssembler.__new__(NodalMeshAssembler)
    assembler._petype_fnmap = {
        'pyr': {'quad': [[0, 1, 2, 3]], 'tri': []},
    }

    codec = [
        'eles/pyr',
        'eles/pyr/face/0',
        'eles/pyr/face/1',
        'eles/pyr/face/2',
        'eles/pyr/face/3',
        'eles/pyr/face/4',
    ]
    foeles = np.array([[0, 1, 2, 3, 4]])

    petype, (_, _, eidx), nodes = assembler._foface_info(
        'pyr', 'quad', codec, foeles
    )

    assert petype == 'pyr'
    assert np.array_equal(eidx, [0])
    assert nodes.shape == (1,)
    assert nodes.tolist() == [(0, 1, 2, 3)]

    fdtype = [('cidx', np.int16), ('off', np.int64)]
    edtype = [
        ('nodes', np.int64, 5), ('curved', bool),
        ('faces', fdtype, 5), ('colour', np.uint8),
        ('tags', np.uint64),
    ]
    eles = {'pyr': np.zeros(1, dtype=edtype)}
    resid, _ = assembler._pair_volume_faces(
        {'quad': [(petype, (_, _, eidx), nodes)]}, codec, eles
    )

    assert resid == {(0, 1, 2, 3): (0, 0)}
