from types import SimpleNamespace

import numpy as np
import pytest

from pyfr.inifile import Inifile
from pyfr.mortars import MortarSide
from pyfr.solvers.base.groups import ElementGroupKey
from pyfr.solvers.baseadvecdiff.inters import mpi_ldg_beta
from pyfr.solvers.navstokes import mortars as nmortars


def _cfg(beta=0.0):
    return Inifile(f'''\
[constants]
gamma = 1.4
mu = 0.01
Pr = 0.72

[solver]
viscosity-correction = none
shock-capturing = none

[solver-interfaces]
riemann-solver = rusanov
mortar-implementation = staged
ldg-beta = {beta}
ldg-tau = 0.1
''')


class _Pointwise:
    def register(self, name):
        pass


class _Matrix:
    _next_mid = 1

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.nbytes = int(np.prod(shape))*8
        self.mid = self._next_mid
        _Matrix._next_mid += 1

    def sendreq(self, comm, rank, tag):
        return 'send', comm.rank, rank, tag, self.mid

    def recvreq(self, comm, rank, tag):
        return 'recv', comm.rank, rank, tag, self.mid

    def slice(self, ra=None, rb=None, ca=None, cb=None):
        return _Matrix((rb - ra, *self.shape[1:]))


class _Backend:
    fpdtype_eps = np.finfo(float).eps

    def __init__(self):
        self.pointwise = _Pointwise()
        self.xchg = []

    def xchg_matrix(self, shape, tags=None):
        m = _Matrix(shape)
        self.xchg.append(m)
        return m

    def matrix(self, shape, tags=None):
        return _Matrix(shape)

    def const_matrix(self, value, tags=None):
        return _Matrix(np.asarray(value).shape)

    def view(self, *args, **kwargs):
        return object()

    def kernel(self, name, *args, **kwargs):
        return name, args, kwargs

    def unordered_meta_kernel(self, kernels):
        return 'unordered', kernels

    def ordered_meta_kernel(self, kernels):
        return 'ordered', kernels


def _entry(local_is_owner, local_order, eidx=0):
    owner_order, nonowner_order = 2, 3
    lkey = ElementGroupKey('hex', local_order)
    local = MortarSide('hex', 'quad', (2,), (eidx,), lkey)
    face = SimpleNamespace(
        local_is_owner=local_is_owner,
        local_side=local,
        geometry=SimpleNamespace(
            scaled_normals=(np.ones((3, 16, 1)),)
        ),
        face_pair=(('hex', 10 + eidx, 2), ('hex', 11 + eidx, 4)),
    )
    onfpts = (owner_order + 1)**2
    nnfpts = (nonowner_order + 1)**2
    nmpts = 16
    ops = {
        'operator_keys': (('p',),),
        'operator_sets': ({
            'left_interp': (np.ones((nmpts, onfpts)),),
            'right_interp': (np.ones((nmpts, nnfpts)),),
            'left_proj': (np.ones((onfpts, nmpts)),),
            'right_proj': (np.ones((nnfpts, nmpts)),),
            'right_state_proj': (np.ones((nnfpts, nmpts)),),
        },),
        'nleftfpts': onfpts,
        'nrightfpts': (nnfpts,),
        'nmpts': nmpts,
    }
    return face, ops


def _plan(owner_rank, pairs=None):
    pairs = pairs or ((('hex', 10, 2), ('hex', 11, 4)),)
    return SimpleNamespace(
        neighbour_rank=1 - owner_rank,
        ownership=SimpleNamespace(
            owner_rank=owner_rank, participant_ranks=(0, 1)
        ),
        tags=(
            ('state', 16), ('common-state', 17),
            ('gradient', 18), ('flux', 19)
        ),
        face_pairs=pairs,
    )


def _elemap(order):
    return {
        ElementGroupKey('hex', order): SimpleNamespace(ndims=3, nvars=5)
    }


def _patch_views(monkeypatch):
    monkeypatch.setattr(nmortars, 'mortar_side_view', lambda *args: object())
    monkeypatch.setattr(
        nmortars, 'mortar_side_point_view', lambda *args: object()
    )


def test_mpi_ldg_beta_preserves_native_rank_rule():
    assert mpi_ldg_beta(0.5, 1, 0) == 0.5
    assert mpi_ldg_beta(0.5, 0, 1) == -0.5
    assert mpi_ldg_beta(0.5, 2, 1) == 0.5
    assert mpi_ldg_beta(0.5, 1, 2) == -0.5
    assert mpi_ldg_beta(-0.5, 1, 0) == -0.5
    assert mpi_ldg_beta(0.0, 0, 1) == 0.0


@pytest.mark.parametrize('beta', [-0.5, 0.0, 0.5])
def test_mpi_ldg_beta_two_sided_algebra(beta):
    bo = mpi_ldg_beta(beta, 1, 0)
    bn = mpi_ldg_beta(beta, 0, 1)
    assert bo == -bn

    uo = np.array([1.1, -0.2, 0.3, 0.4, 2.8])
    un = np.array([0.9, 0.1, -0.4, 0.2, 2.5])
    co = (0.5 - bo)*uo + (0.5 + bo)*un
    cn = (0.5 - bn)*un + (0.5 + bn)*uo
    np.testing.assert_array_equal(co, cn)

    go = np.array([0.4, -0.1, 0.7])
    gn = np.array([-0.3, 0.2, 0.5])
    fo = (0.5 + bo)*go + (0.5 - bo)*gn
    fn = (0.5 + bn)*gn + (0.5 - bn)*go
    np.testing.assert_array_equal(fo, fn)


@pytest.mark.parametrize(
    'beta,extra', [(0.0, True), (-0.5, True), (0.5, False)]
)
def test_navierstokes_mpi_p_owner_is_authoritative(
    monkeypatch, beta, extra
):
    _patch_views(monkeypatch)
    obe = _Backend()
    owner = nmortars._NavierStokesMPIPMortarBatch(
        obe, _plan(1), (_entry(True, 2),), _elemap(2),
        _cfg(beta), SimpleNamespace(rank=1), 0
    )
    assert 'common_state_eval' in owner.kernels
    assert 'flux_eval' in owner.kernels
    assert 'flux_scatter' not in owner.kernels
    expected = {'mpi_p_state_recv', 'mpi_p_flux_send'}
    if extra:
        expected |= {'mpi_p_common_send', 'mpi_p_grad_recv'}
    assert set(owner.mpireqs) == expected

    nbe = _Backend()
    nonowner = nmortars._NavierStokesMPIPMortarBatch(
        nbe, _plan(1), (_entry(False, 3),), _elemap(3),
        _cfg(beta), SimpleNamespace(rank=0), 0
    )
    assert 'common_state_eval' not in nonowner.kernels
    assert 'flux_eval' not in nonowner.kernels
    assert 'flux_scatter' in nonowner.kernels
    expected = {'mpi_p_state_send', 'mpi_p_flux_recv'}
    if extra:
        expected |= {'mpi_p_common_recv', 'mpi_p_grad_send'}
    assert set(nonowner.mpireqs) == expected


@pytest.mark.parametrize('beta', [0.0, 0.5])
def test_navierstokes_mpi_p_messages_are_per_batch(monkeypatch, beta):
    _patch_views(monkeypatch)
    face1, ops = _entry(True, 2, 0)
    face2, _ = _entry(True, 2, 1)
    pairs = (face1.face_pair, face2.face_pair)

    be = _Backend()
    batch = nmortars._NavierStokesMPIPMortarBatch(
        be, _plan(1, pairs), ((face1, ops), (face2, ops)),
        _elemap(2), _cfg(beta), SimpleNamespace(rank=1), 0
    )
    assert batch.ninters == 2
    assert len(be.xchg) == 2
    assert be.xchg[0].shape == (16, 5, 2)
    assert be.xchg[1].shape == (48, 5, 2)
    assert sum(name.endswith('_send') for name in batch.mpireqs) <= 2
    assert sum(name.endswith('_recv') for name in batch.mpireqs) <= 2


def test_navierstokes_mpi_p_requires_constant_viscosity(monkeypatch):
    _patch_views(monkeypatch)
    cfg = _cfg()
    cfg.set('solver', 'viscosity-correction', 'sutherland')
    with pytest.raises(ValueError, match='constant viscosity'):
        nmortars._NavierStokesMPIPMortarBatch(
            _Backend(), _plan(1), (_entry(True, 2),), _elemap(2),
            cfg, SimpleNamespace(rank=1), 0
        )
