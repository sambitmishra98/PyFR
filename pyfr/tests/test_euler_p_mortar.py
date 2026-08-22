from types import SimpleNamespace

import pytest

from pyfr.inifile import Inifile
from pyfr.mortars import MortarGroup, MortarSide
from pyfr.solvers.base.groups import ElementGroupKey
from pyfr.solvers.euler import mortars as emortars


def _cfg(impl='staged'):
    return Inifile(f'''\
[solver-interfaces]
mortar-implementation = {impl}
''')


def _group():
    p2 = ElementGroupKey('hex', 2)
    p3 = ElementGroupKey('hex', 3)
    return MortarGroup(
        MortarSide('hex', 'quad', (2,), (0,), p2),
        (MortarSide('hex', 'quad', (4,), (0,), p3),),
    )


class _GeneralMCon:
    format = 'one-to-many-v1'

    def __init__(self, template):
        self.name = f'{template}-mortar'
        self.template = template

    def __len__(self):
        return 1


def _general_group(template):
    if template == 'quad-2x2':
        etype, ftopology, ndims, nvars, nright = 'hex', 'quad', 3, 5, 4
    elif template == 'line-1x2':
        etype, ftopology, ndims, nvars, nright = 'quad', 'line', 2, 4, 2
    else:
        raise ValueError(f'Unsupported test template {template!r}')

    key = ElementGroupKey(etype, 3)
    group = MortarGroup(
        MortarSide(etype, ftopology, (0,), (0,), key),
        tuple(
            MortarSide(etype, ftopology, (i + 1,), (i + 1,), key)
            for i in range(nright)
        ),
    )
    return group, {key: SimpleNamespace(ndims=ndims, nvars=nvars)}


@pytest.mark.parametrize(
    ('template', 'builder_name', 'nright'),
    [('quad-2x2', 'quad', 4), ('line-1x2', 'line', 2)],
)
def test_euler_general_mortar_dispatches_to_accepted_builder(
    monkeypatch, template, builder_name, nright
):
    group, elemap = _general_group(template)
    geometry = SimpleNamespace(backend_nbytes=4321)
    ops = {
        'geometry': geometry,
        'max_geom_error': 2e-15,
        'max_normal_error': 3e-15,
        'mortar_group': group,
        'nops': 1,
        'shared_operator_bytes': 1234,
    }
    build_calls, init_calls = [], []

    def make_builder(name):
        def builder(mesh, emap, mcon, cfg):
            build_calls.append((name, mesh, emap, mcon, cfg))
            return ops

        return builder

    monkeypatch.setattr(
        emortars, 'build_quad_quad4_operators', make_builder('quad')
    )
    monkeypatch.setattr(
        emortars, 'build_line_line2_operators', make_builder('line')
    )
    monkeypatch.setattr(
        emortars, 'local_mortar_ownership',
        lambda: SimpleNamespace(kind='local')
    )
    monkeypatch.setattr(
        emortars.EulerMortarInters, '_init_general_staged',
        lambda self, be, emap, cfg, gotops: init_calls.append(
            (self, be, emap, cfg, gotops)
        ),
    )

    be, mesh, mcon = object(), object(), _GeneralMCon(template)
    inter = emortars.EulerMortarInters(be, mesh, mcon, elemap, _cfg())

    assert len(group.right) == nright
    assert build_calls == [(builder_name, mesh, elemap, mcon, inter.cfg)]
    assert init_calls == [(inter, be, elemap, inter.cfg, ops)]
    assert inter.coarse_etype == group.left.etype
    assert inter.fine_etype == group.right[0].etype
    assert inter.noperator_sets == 1
    assert inter.operator_bytes == 1234


def test_euler_general_mortar_rejects_unknown_template():
    _, elemap = _general_group('line-1x2')
    match = 'Unsupported general mortar template'

    with pytest.raises(ValueError, match=match):
        emortars.EulerMortarInters(
            object(), object(), _GeneralMCon('line-1x3'), elemap, _cfg()
        )


def test_euler_general_line_mortar_rejects_fused_execution():
    _, elemap = _general_group('line-1x2')

    with pytest.raises(ValueError, match='require staged execution'):
        emortars.EulerMortarInters(
            object(), object(), _GeneralMCon('line-1x2'), elemap,
            _cfg('fused')
        )


def test_euler_p_mortar_constructor_reuses_general_staged(monkeypatch):
    group = _group()
    elemap = {
        group.left.runtime_key: SimpleNamespace(ndims=3, nvars=5),
        group.right[0].runtime_key: SimpleNamespace(ndims=3, nvars=5),
    }
    geometry = SimpleNamespace(backend_nbytes=4321)
    ops = {
        'geometry': geometry,
        'max_geom_error': 2e-15,
        'max_normal_error': 3e-15,
        'nops': 1,
        'shared_operator_bytes': 1234,
    }
    calls = []

    monkeypatch.setattr(
        emortars, 'build_quad_p_operators',
        lambda mesh, emap, grp, cfg: ops,
    )
    monkeypatch.setattr(
        emortars, 'local_mortar_ownership',
        lambda: SimpleNamespace(kind='local')
    )
    monkeypatch.setattr(
        emortars.EulerMortarInters, '_init_general_staged',
        lambda self, be, emap, cfg, gotops: calls.append(
            (self, be, emap, cfg, gotops)
        ),
    )

    be, mesh = object(), object()
    inter = emortars.EulerMortarInters.from_p_mortar(
        be, mesh, group, elemap, _cfg(), 'p-mortar-7'
    )

    assert inter.name == 'p-mortar-7'
    assert inter.ninters == 1
    assert inter.ndims == 3
    assert inter.nvars == 5
    assert inter.mortar_implementation == 'staged'
    assert inter._geometry is geometry
    assert inter.noperator_sets == 1
    assert inter.shared_operator_bytes == 1234
    assert inter.fused_operator_bytes == 0
    assert inter.operator_bytes == 1234
    assert inter.geometry_bytes == 4321
    assert calls == [(inter, be, elemap, inter.cfg, ops)]


def test_euler_p_mortar_rejects_fused_execution():
    group = _group()
    elemap = {
        group.left.runtime_key: SimpleNamespace(ndims=3, nvars=5),
        group.right[0].runtime_key: SimpleNamespace(ndims=3, nvars=5),
    }

    with pytest.raises(ValueError, match='require staged execution'):
        emortars.EulerMortarInters.from_p_mortar(
            object(), object(), group, elemap, _cfg('fused'), 'p-mortar-0'
        )
