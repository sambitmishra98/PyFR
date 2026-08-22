from types import SimpleNamespace

import numpy as np
import pytest

from pyfr.inifile import Inifile
from pyfr.solvers.base.groups import ElementGroupKey
from pyfr.writers.native import NativeWriter


def _cfg():
    return Inifile('''\
[solver-elements-hex]
soln-pts = gauss-legendre
''')


def _writer():
    writer = object.__new__(NativeWriter)
    writer.cfg = _cfg()
    writer.ndims = 3
    writer.fpdtype = np.dtype('float64')
    writer._global_ecounts = {'hex': 4}
    return writer


def test_native_writer_accepts_mixed_group_keys():
    p2 = ElementGroupKey('hex', 2)
    p3 = ElementGroupKey('hex', 3)
    writer = _writer()

    shapes = {p2: (5, 27), p3: (5, 64)}
    eidxs = {p2: np.array([10, 11]), p3: np.array([12, 13])}
    writer.set_shapes_eidxs(shapes, eidxs, {'soln': list('abcde')})

    assert set(writer._einfo) == {'p2-hex', 'p3-hex'}
    assert writer._einfo['p2-hex'][1]
    assert writer._einfo['p3-hex'][1]
    assert writer._einfo['p2-hex'][2] == p2
    assert writer._einfo['p3-hex'][2] == p3


def test_native_writer_rejects_group_point_count_mismatch():
    p3 = ElementGroupKey('hex', 3)
    writer = _writer()

    with pytest.raises(ValueError, match='incompatible solution point count'):
        writer.set_shapes_eidxs(
            {p3: (5, 27)}, {p3: np.array([10, 11])},
            {'soln': list('abcde')}
        )
