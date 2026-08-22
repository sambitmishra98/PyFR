from types import SimpleNamespace

import numpy as np
import pytest

from pyfr.inifile import Inifile
from pyfr.readers.native import Connectivity, Mesh
from pyfr.shapes import HexShape
from pyfr.solvers.base.groups import ElementGroupKey, ElementGroupMap
from pyfr.solvers.base.system import BaseSystem


def _cfg(extra=''):
    return Inifile(f'''\
[solver]
order = 2
anti-alias = none

[solver-elements-hex]
soln-pts = gauss-legendre

[solver-order-high]
tag = high
order = 3
{extra}
''')


def _mesh(tags):
    return Mesh(
        fname='test', raw=None, codec=['tag/high'], etypes=['hex'],
        tags={'hex': np.asarray(tags, dtype=np.uint64)},
        eidxs={'hex': np.arange(10, 10 + len(tags))}, con_p={}
    )


def _con(eidxs):
    cidxs = np.zeros(len(eidxs), dtype=np.int16)
    return Connectivity(cidxs, np.asarray(eidxs), {0: ('hex', 0)})


def test_shape_explicit_solution_order():
    cfg = _cfg()
    assert HexShape(None, cfg).order == 2
    assert HexShape(None, cfg, order=3).order == 3


def test_element_group_map_builds_adjacent_groups():
    gmap = ElementGroupMap(_mesh([0, 1, 1, 0]), _cfg())
    p2 = ElementGroupKey('hex', 2)
    p3 = ElementGroupKey('hex', 3)

    assert gmap.active_orders == (2, 3)
    assert gmap.group_eidxs[p2].tolist() == [0, 3]
    assert gmap.group_eidxs[p3].tolist() == [1, 2]
    assert gmap.group_global_eidxs[p2].tolist() == [10, 13]
    assert gmap.group_global_eidxs[p3].tolist() == [11, 12]
    assert gmap.resolve('hex', 3) == (p2, 1)
    assert gmap.resolve('hex', 2) == (p3, 1)



def test_element_group_map_rejects_zero_default_order():
    cfg = Inifile(_cfg().tostr().replace('order = 2', 'order = 0', 1))
    with pytest.raises(ValueError, match='orders must be >= 1'):
        ElementGroupMap(_mesh([0, 1]), cfg)

def test_element_group_map_rejects_nonadjacent_orders():
    cfg = _cfg().tostr().replace('order = 3', 'order = 4')
    with pytest.raises(ValueError, match='two adjacent'):
        ElementGroupMap(_mesh([0, 1]), Inifile(cfg))


def test_element_group_map_rejects_unknown_tag():
    cfg = _cfg().tostr().replace('tag = high', 'tag = absent')
    with pytest.raises(ValueError, match='Unknown mixed-p mesh tag'):
        ElementGroupMap(_mesh([0, 1]), Inifile(cfg))


def test_element_group_map_rejects_overlapping_tags():
    mesh = _mesh([0, 3])
    mesh.codec = ['tag/high', 'tag/other']
    cfg = Inifile(_cfg().tostr() + '''\
[solver-order-other]
tag = other
order = 3
''')

    with pytest.raises(ValueError, match='override tags overlap'):
        ElementGroupMap(mesh, cfg)


def test_grouped_connectivity_uses_local_indices():
    gmap = ElementGroupMap(_mesh([0, 1, 1, 0]), _cfg())
    con = gmap.remap_connectivity(_con([0, 1, 2, 3]))

    items = list(con.items())
    assert items[0][0] == ElementGroupKey('hex', 2)
    assert items[0][2].tolist() == [0, 1]
    assert items[1][0] == ElementGroupKey('hex', 3)
    assert items[1][2].tolist() == [0, 1]


def test_same_order_internal_connectivity_is_remapped():
    gmap = ElementGroupMap(_mesh([0, 0, 1, 1]), _cfg())
    lhs, rhs = gmap.remap_internal(_con([0, 2]), _con([1, 3]))

    assert [k.order for k, _, _ in lhs.items()] == [2, 3]
    assert [k.order for k, _, _ in rhs.items()] == [2, 3]


def test_p_mismatch_internal_connectivity_fails_closed():
    gmap = ElementGroupMap(_mesh([0, 1]), _cfg())
    with pytest.raises(RuntimeError, match='requires V10C2 p-mortar'):
        gmap.remap_internal(_con([0]), _con([1]))


def _validate_cfg(cfg, initsoln=None):
    system = object.__new__(BaseSystem)
    system.cfg = cfg
    mesh = SimpleNamespace(mcon={}, con_p={})
    return system._validate_mixed_p_mode(mesh, initsoln)


def test_uniform_restart_into_mixed_p_fails_closed():
    initsoln = SimpleNamespace(groups={})
    match = 'Uniform solution cannot initialize'
    with pytest.raises(RuntimeError, match=match):
        _validate_cfg(_cfg(), initsoln)


def test_mixed_p_plugins_fail_closed():
    cfg = Inifile(
        _cfg().tostr() + '\n[soln-plugin-nancheck]\nnsteps = 1\n'
    )
    with pytest.raises(RuntimeError, match='plugins are not supported'):
        _validate_cfg(cfg)


def test_mixed_p_implicit_fails_closed():
    cfg = Inifile(
        _cfg().tostr() +
        '\n[solver-time-integrator]\nformulation = implicit\n'
    )
    with pytest.raises(RuntimeError, match='implicit mode is not supported'):
        _validate_cfg(cfg)


def test_mixed_p_region_split_full_and_subset():
    gmap = ElementGroupMap(_mesh([0, 1, 1, 0]), _cfg())
    p2 = ElementGroupKey('hex', 2)
    p3 = ElementGroupKey('hex', 3)

    full = gmap.split_region({'hex': slice(None)})
    assert full[p2].tolist() == [0, 1]
    assert full[p3].tolist() == [0, 1]

    subset = gmap.split_region({'hex': np.array([1, 3])})
    assert subset[p2].tolist() == [1]
    assert subset[p3].tolist() == [0]


def test_mixed_p_writer_plugin_is_allowed():
    cfg = Inifile(
        _cfg().tostr() + '''
[soln-plugin-writer]
dt-out = 1
basedir = .
basename = out-{t}
'''
    )
    _validate_cfg(cfg)


def test_mixed_p_nonwriter_plugin_still_fails_closed():
    cfg = Inifile(
        _cfg().tostr() + '''
[soln-plugin-writer]
dt-out = 1
basedir = .
basename = out-{t}

[soln-plugin-nancheck]
nsteps = 1
'''
    )
    with pytest.raises(RuntimeError, match='soln-plugin-writer'):
        _validate_cfg(cfg)


def test_split_internal_separates_same_p_and_p_mortars():
    gmap = ElementGroupMap(_mesh([0, 1, 0, 1]), _cfg())
    lhs = _con([0, 1, 2])
    rhs = _con([2, 0, 3])

    (slhs, srhs), pgroups = gmap.split_internal(lhs, rhs)

    assert len(slhs.cidxs) == len(srhs.cidxs) == 1
    assert list(slhs.items())[0][0] == ElementGroupKey('hex', 2)
    assert list(slhs.items())[0][2].tolist() == [0]
    assert list(srhs.items())[0][2].tolist() == [1]

    assert len(pgroups) == 2
    assert [g.left.elekey.order for g in pgroups] == [3, 2]
    assert [g.right[0].elekey.order for g in pgroups] == [2, 3]
    assert pgroups[0].left.eidxs == (0,)
    assert pgroups[0].right[0].eidxs == (0,)
    assert pgroups[1].left.eidxs == (1,)
    assert pgroups[1].right[0].eidxs == (1,)
    assert all(g.left.face_topology == 'quad' for g in pgroups)


def test_split_internal_all_p_mismatch_leaves_no_native_interface():
    gmap = ElementGroupMap(_mesh([0, 1]), _cfg())
    (lhs, rhs), pgroups = gmap.split_internal(_con([0]), _con([1]))

    assert len(lhs.cidxs) == len(rhs.cidxs) == 0
    assert len(pgroups) == 1
    assert pgroups[0].left.elekey == ElementGroupKey('hex', 2)
    assert pgroups[0].right[0].elekey == ElementGroupKey('hex', 3)


def test_system_collects_p_mortar_and_keeps_unsupported_execution_closed():
    gmap = ElementGroupMap(_mesh([0, 1]), _cfg())
    system = object.__new__(BaseSystem)
    system.ele_group_map = gmap
    system.backend = object()
    system.cfg = _cfg()
    system.intinterscls = lambda *args: pytest.fail(
        'ordinary interface must not be constructed for pure p mismatch'
    )
    system.mortarinterscls = object

    mesh = SimpleNamespace(con=(_con([0]), _con([1])), mcon={})
    assert system._load_int_inters(mesh, {}) == []
    assert len(system._p_mortar_groups) == 1

    with pytest.raises(RuntimeError, match='PDE execution is not supported'):
        system._load_mortar_inters(mesh, {})


def test_system_dispatches_p_mortar_constructor():
    gmap = ElementGroupMap(_mesh([0, 1]), _cfg())
    system = object.__new__(BaseSystem)
    system.ele_group_map = gmap
    system.backend = object()
    system.cfg = _cfg()
    system.intinterscls = lambda *args: pytest.fail(
        'ordinary interface must not be constructed for pure p mismatch'
    )

    calls = []

    class PMortar:
        @classmethod
        def from_p_mortar(cls, be, mesh, group, elemap, cfg, name):
            calls.append((be, mesh, group, elemap, cfg, name))
            return name

    system.mortarinterscls = PMortar
    mesh = SimpleNamespace(con=(_con([0]), _con([1])), mcon={})
    elemap = object()

    assert system._load_int_inters(mesh, {}) == []
    assert system._load_mortar_inters(mesh, elemap) == ['p-mortar-0']
    assert len(calls) == 1
    assert calls[0][2] == system._p_mortar_groups[0]
    assert calls[0][3] is elemap



def test_element_group_map_global_identity_allows_local_single_order(
    monkeypatch
):
    import pyfr.solvers.base.groups as groupsmod

    class FakeComm:
        def allgather(self, value):
            if not value or isinstance(value[0], int):
                return [value, [3]]
            return [value, [('hex', 3)]]

    monkeypatch.setattr(
        groupsmod, 'get_comm_rank_root', lambda: (FakeComm(), 0, 0)
    )
    mesh = _mesh([0, 0])
    gmap = ElementGroupMap(mesh, _cfg())

    assert gmap.global_active_orders == (2, 3)
    assert gmap.keys == (ElementGroupKey('hex', 2),)
    assert gmap.global_keys == (
        ElementGroupKey('hex', 2), ElementGroupKey('hex', 3)
    )


def test_element_group_map_global_identity_rejects_third_order(monkeypatch):
    import pyfr.solvers.base.groups as groupsmod

    class FakeComm:
        def allgather(self, value):
            return [value, [3, 4]]

    monkeypatch.setattr(
        groupsmod, 'get_comm_rank_root', lambda: (FakeComm(), 0, 0)
    )
    with pytest.raises(ValueError, match='two adjacent'):
        ElementGroupMap(_mesh([0, 0]), _cfg())


def test_mixed_p_mode_allows_mpi_setup_boundary():
    system = object.__new__(BaseSystem)
    system.cfg = _cfg()
    mesh = SimpleNamespace(mcon={}, con_p={1: object()})
    system._validate_mixed_p_mode(mesh, None)
