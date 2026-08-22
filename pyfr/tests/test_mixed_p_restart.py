from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader, SolutionGroup
from pyfr.shapes import HexShape
from pyfr.solvers.base.elements import BaseElements
from pyfr.solvers.base.groups import ElementGroupKey, ElementGroupMap


def _cfg():
    return Inifile('''\
[solver]
order = 2
anti-alias = none

[solver-elements-hex]
soln-pts = gauss-legendre

[solver-order-high]
tag = high
order = 3
''')


def _mesh(tags=(0, 0, 1, 1)):
    return SimpleNamespace(
        codec=['tag/high'], etypes=['hex'], con_p={},
        tags={'hex': np.asarray(tags, dtype=np.uint64)},
        eidxs={'hex': np.arange(len(tags), dtype=np.int64)}
    )


def _dtype(nupts):
    soln = [(n, np.float64, (nupts,)) for n in 'abcde']
    return np.dtype([('soln', soln)])


def _reader_file(tmp_path, p2=(0, 1), p3=(2, 3), p3_nupts=64):
    path = tmp_path / 'mixed.h5'
    f = h5py.File(path, 'w')
    f.create_dataset('eles/hex', (4,), dtype=np.uint8)
    f.create_dataset('soln/p2-hex', (len(p2),), dtype=_dtype(27))
    f.create_dataset('soln/p2-hex-idxs', data=np.asarray(p2))
    f.create_dataset('soln/p3-hex', (len(p3),), dtype=_dtype(p3_nupts))
    f.create_dataset('soln/p3-hex-idxs', data=np.asarray(p3))

    reader = object.__new__(NativeReader)
    reader.f = f
    reader.mesh = SimpleNamespace(etypes=['hex'])
    return reader, f


def test_reader_discovers_valid_mixed_groups(tmp_path):
    reader, f = _reader_file(tmp_path)
    try:
        groups, byetype = reader._mixed_soln_groups(f, 'soln')
    finally:
        f.close()

    assert sorted(groups) == ['p2-hex', 'p3-hex']
    assert byetype == {'hex': ['p2-hex', 'p3-hex']}
    assert groups['p2-hex'][:3] == ('hex', 2, pytest.approx([0, 1]))
    assert groups['p3-hex'][:3] == ('hex', 3, pytest.approx([2, 3]))


def test_reader_rejects_overlapping_mixed_groups(tmp_path):
    reader, f = _reader_file(tmp_path, p3=(1, 3))
    try:
        with pytest.raises(ValueError, match='overlapping persistent groups'):
            reader._mixed_soln_groups(f, 'soln')
    finally:
        f.close()


def test_reader_rejects_persistent_point_count_mismatch(tmp_path):
    reader, f = _reader_file(tmp_path, p3_nupts=27)
    try:
        with pytest.raises(ValueError, match='order/point-count mismatch'):
            reader._mixed_soln_groups(f, 'soln')
    finally:
        f.close()


def test_reader_rejects_duplicate_group_indices(tmp_path):
    reader, f = _reader_file(tmp_path, p2=(0, 0))
    try:
        with pytest.raises(ValueError, match='duplicate element indices'):
            reader._mixed_soln_groups(f, 'soln')
    finally:
        f.close()


def test_reader_rejects_out_of_range_group_indices(tmp_path):
    reader, f = _reader_file(tmp_path, p3=(2, 4))
    try:
        with pytest.raises(ValueError, match='element index out of range'):
            reader._mixed_soln_groups(f, 'soln')
    finally:
        f.close()


def test_group_map_accepts_exact_persistent_membership():
    gmap = ElementGroupMap(_mesh(), _cfg())
    groups = {
        'p2-hex': SolutionGroup('hex', 2, np.array([0, 1])),
        'p3-hex': SolutionGroup('hex', 3, np.array([2, 3]))
    }

    expected = gmap.validate_solution_groups(groups)
    assert expected['p2-hex'] == ElementGroupKey('hex', 2)
    assert expected['p3-hex'] == ElementGroupKey('hex', 3)


def test_group_map_rejects_swapped_persistent_membership():
    gmap = ElementGroupMap(_mesh(), _cfg())
    groups = {
        'p2-hex': SolutionGroup('hex', 2, np.array([2, 3])),
        'p3-hex': SolutionGroup('hex', 3, np.array([0, 1]))
    }

    with pytest.raises(RuntimeError, match='element/order map mismatch'):
        gmap.validate_solution_groups(groups)


def test_group_map_rejects_missing_persistent_group():
    gmap = ElementGroupMap(_mesh(), _cfg())
    groups = {
        'p2-hex': SolutionGroup('hex', 2, np.array([0, 1]))
    }

    with pytest.raises(RuntimeError, match='restart group mismatch'):
        gmap.validate_solution_groups(groups)


def test_restart_source_basis_accepts_explicit_order():
    cfg = _cfg()
    basis = HexShape(None, cfg, order=3)
    ele = SimpleNamespace(basis=basis, nupts=64, neles=1, nvars=1)
    state = np.arange(64, dtype=float).reshape(64, 1, 1)

    out = BaseElements.set_ics_from_soln(ele, state, cfg, order=3)
    assert np.allclose(out, state)



def test_group_map_accepts_globally_present_locally_empty_group(monkeypatch):
    import pyfr.solvers.base.groups as groupsmod

    class FakeComm:
        def allgather(self, value):
            if not value or isinstance(value[0], int):
                return [value, [3]]
            return [value, [('hex', 3)]]

    monkeypatch.setattr(
        groupsmod, 'get_comm_rank_root', lambda: (FakeComm(), 0, 0)
    )
    mesh = _mesh(tags=(0, 0))
    gmap = ElementGroupMap(mesh, _cfg())
    groups = {
        'p2-hex': SolutionGroup('hex', 2, np.array([0, 1])),
        'p3-hex': SolutionGroup('hex', 3, np.array([], dtype=np.int64)),
    }

    expected = gmap.validate_solution_groups(groups)
    assert expected['p3-hex'] == ElementGroupKey('hex', 3)
