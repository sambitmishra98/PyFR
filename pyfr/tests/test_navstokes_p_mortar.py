from types import SimpleNamespace

import pytest

from pyfr.inifile import Inifile
from pyfr.mortars import MortarGroup, MortarSide
from pyfr.solvers.base.groups import ElementGroupKey
from pyfr.solvers.navstokes import mortars as nmortars


def _cfg(impl='staged', visc_corr='none'):
    return Inifile(f'''\
[solver]
viscosity-correction = {visc_corr}
[solver-interfaces]
mortar-implementation = {impl}
''')


def _group(reverse=False):
    p2 = ElementGroupKey('hex', 2)
    p3 = ElementGroupKey('hex', 3)
    if reverse:
        p2, p3 = p3, p2
    return MortarGroup(
        MortarSide('hex', 'quad', (2,), (0,), p2),
        (MortarSide('hex', 'quad', (4,), (0,), p3),),
    )


@pytest.mark.parametrize('reverse', [False, True])
def test_ns_p_mortar_constructor_reuses_general_ldg(monkeypatch, reverse):
    group = _group(reverse)
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
    build_calls = []
    init_calls = []

    def build(mesh, emap, grp, cfg, *, state_projection=False):
        build_calls.append((mesh, emap, grp, cfg, state_projection))
        return ops

    monkeypatch.setattr(nmortars, 'build_quad_p_operators', build)
    monkeypatch.setattr(
        nmortars, 'local_mortar_ownership',
        lambda: SimpleNamespace(kind='local')
    )
    monkeypatch.setattr(
        nmortars.NavierStokesMortarInters, '_init_general_staged',
        lambda self, be, emap, cfg, gotops: init_calls.append(
            (self, be, emap, cfg, gotops)
        ),
    )

    be, mesh = object(), object()
    inter = nmortars.NavierStokesMortarInters.from_p_mortar(
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
    assert build_calls == [(mesh, elemap, group, inter.cfg, True)]
    assert init_calls == [(inter, be, elemap, inter.cfg, ops)]


def test_ns_p_mortar_rejects_fused_execution():
    group = _group()
    elemap = {
        group.left.runtime_key: SimpleNamespace(ndims=3, nvars=5),
        group.right[0].runtime_key: SimpleNamespace(ndims=3, nvars=5),
    }

    with pytest.raises(ValueError, match='require staged execution'):
        nmortars.NavierStokesMortarInters.from_p_mortar(
            object(), object(), group, elemap, _cfg('fused'), 'p-mortar-0'
        )


def test_ns_p_mortar_rejects_variable_viscosity():
    group = _group()
    elemap = {
        group.left.runtime_key: SimpleNamespace(ndims=3, nvars=5),
        group.right[0].runtime_key: SimpleNamespace(ndims=3, nvars=5),
    }

    with pytest.raises(ValueError, match='require constant viscosity'):
        nmortars.NavierStokesMortarInters.from_p_mortar(
            object(), object(), group, elemap, _cfg(visc_corr='sutherland'),
            'p-mortar-0'
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
    ('template', 'builder_name', 'ndims'),
    [('quad-2x2', 'quad', 3), ('line-1x2', 'line', 2)],
)
def test_ns_general_mortar_dispatches_to_accepted_builder(
    monkeypatch, template, builder_name, ndims
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
        def builder(mesh, emap, mcon, cfg, *, state_projection=False):
            build_calls.append(
                (name, mesh, emap, mcon, cfg, state_projection)
            )
            return ops

        return builder

    monkeypatch.setattr(
        nmortars, 'build_quad_quad4_operators', make_builder('quad')
    )
    monkeypatch.setattr(
        nmortars, 'build_line_line2_operators', make_builder('line')
    )
    monkeypatch.setattr(
        nmortars, 'local_mortar_ownership',
        lambda: SimpleNamespace(kind='local')
    )
    monkeypatch.setattr(
        nmortars.NavierStokesMortarInters, '_init_general_staged',
        lambda self, be, emap, cfg, gotops: init_calls.append(
            (self, be, emap, cfg, gotops)
        ),
    )

    be, mesh, mcon = object(), object(), _GeneralMCon(template)
    inter = nmortars.NavierStokesMortarInters(
        be, mesh, mcon, elemap, _cfg()
    )

    assert inter.ndims == ndims
    assert build_calls == [
        (builder_name, mesh, elemap, mcon, inter.cfg, True)
    ]
    assert init_calls == [(inter, be, elemap, inter.cfg, ops)]
    assert inter.coarse_etype == group.left.etype
    assert inter.fine_etype == group.right[0].etype


def test_ns_general_line_mortar_rejects_fused_execution():
    _, elemap = _general_group('line-1x2')

    with pytest.raises(ValueError, match='require staged execution'):
        nmortars.NavierStokesMortarInters(
            object(), object(), _GeneralMCon('line-1x2'), elemap,
            _cfg('fused')
        )


def test_ns_general_line_mortar_requires_2d():
    group, _ = _general_group('line-1x2')
    key = group.left.runtime_key
    elemap = {key: SimpleNamespace(ndims=3, nvars=5)}

    with pytest.raises(ValueError, match='Line mortars require 2D'):
        nmortars.NavierStokesMortarInters(
            object(), object(), _GeneralMCon('line-1x2'), elemap, _cfg()
        )


@pytest.mark.parametrize('ndims', [2, 3])
def test_ns_general_gradient_gather_uses_active_dimensions(ndims):
    inter = object.__new__(nmortars.NavierStokesMortarInters)
    inter.ndims = ndims
    inter.nvars = 4 if ndims == 2 else 5
    calls = []

    class Backend:
        def kernel(self, name, **kwargs):
            calls.append((name, kwargs))
            return object()

    inter._be = Backend()
    inter._general_grad_buffer_views = (
        lambda bufs, nfpts, batch: tuple(
            f'view-{d}' for d in range(ndims)
        )
    )

    inter._general_grad_side_gather(
        object(), tuple(object() for _ in range(ndims)), 4, {'n': 2}, {}
    )

    name, kwargs = calls[0]
    assert name == 'mortargradsidegather'
    assert kwargs['dims'] == [8]
    assert kwargs['g'] is not None
    assert {key for key in kwargs if key.startswith('b')} == {
        f'b{d}' for d in range(ndims)
    }
