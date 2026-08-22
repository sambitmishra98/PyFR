from types import SimpleNamespace

import numpy as np
import pytest

from pyfr.inifile import Inifile
from pyfr.mortars import MortarSide
from pyfr.solvers.base.groups import ElementGroupKey
from pyfr.solvers.base.system import BaseSystem
from pyfr.solvers.euler import mortars as emortars


def _cfg():
    return Inifile('''\
[constants]
gamma = 1.4

[solver-interfaces]
riemann-solver = rusanov
''')


class _Pointwise:
    def register(self, name):
        pass


class _Matrix:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.nbytes = int(np.prod(shape))*8
        self.requests = []

    def sendreq(self, comm, rank, tag):
        req = ('send', comm.rank, rank, tag, id(self))
        self.requests.append(req)
        return req

    def recvreq(self, comm, rank, tag):
        req = ('recv', comm.rank, rank, tag, id(self))
        self.requests.append(req)
        return req


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

    def kernel(self, name, *args, **kwargs):
        return name, args, kwargs

    def unordered_meta_kernel(self, kernels):
        return 'unordered', kernels

    def ordered_meta_kernel(self, kernels):
        return 'ordered', kernels


def _entry(
    local_is_owner, local_order, owner_order=None, nonowner_order=None
):
    owner_order = owner_order if owner_order is not None else local_order
    nonowner_order = (
        nonowner_order if nonowner_order is not None else local_order + 1
    )
    lkey = ElementGroupKey('hex', local_order)
    local = MortarSide('hex', 'quad', (2,), (0,), lkey)
    face = SimpleNamespace(
        local_is_owner=local_is_owner,
        local_side=local,
        geometry=SimpleNamespace(
            scaled_normals=(np.ones((3, 16, 1)),)
        ),
        face_pair=(('hex', 10, 2), ('hex', 11, 4)),
    )
    eye_l = np.eye((owner_order + 1)**2)
    eye_r = np.eye((nonowner_order + 1)**2)
    nmpts = 16
    ops = {
        'operator_keys': (('p',),),
        'operator_sets': ({
            'left_interp': (np.ones((nmpts, len(eye_l))),),
            'right_interp': (np.ones((nmpts, len(eye_r))),),
            'left_proj': (np.ones((len(eye_l), nmpts)),),
            'right_proj': (np.ones((len(eye_r), nmpts)),),
        },),
        'nleftfpts': len(eye_l),
        'nrightfpts': (len(eye_r),),
        'nmpts': nmpts,
    }
    return face, ops


def _plan(owner_rank):
    return SimpleNamespace(
        neighbour_rank=1 - owner_rank,
        ownership=SimpleNamespace(owner_rank=owner_rank),
        tags=(('state', 17), ('flux', 19)),
        face_pairs=((('hex', 10, 2), ('hex', 11, 4)),),
    )


def test_euler_mpi_p_batch_has_one_owner_flux(monkeypatch):
    monkeypatch.setattr(emortars, 'mortar_side_view', lambda *args: object())
    elemap = {ElementGroupKey('hex', 2): SimpleNamespace(ndims=3, nvars=5)}

    obe = _Backend()
    owner = emortars._EulerMPIPMortarBatch(
        obe, _plan(0), (_entry(True, 2, 2, 3),), elemap,
        _cfg(), SimpleNamespace(rank=0), 0
    )
    assert 'flux_eval' in owner.kernels
    assert 'flux_scatter' not in owner.kernels
    assert set(owner.mpireqs) == {'mpi_p_state_recv', 'mpi_p_flux_send'}

    nbe = _Backend()
    nonowner = emortars._EulerMPIPMortarBatch(
        nbe, _plan(0), (_entry(False, 3, 2, 3),), elemap,
        _cfg(), SimpleNamespace(rank=1), 0
    )
    assert 'flux_eval' not in nonowner.kernels
    assert 'flux_scatter' in nonowner.kernels
    assert set(nonowner.mpireqs) == {'mpi_p_state_send', 'mpi_p_flux_recv'}

    assert len(obe.xchg) == len(nbe.xchg) == 1
    assert obe.xchg[0].shape == nbe.xchg[0].shape == (16, 5, 1)


def test_euler_mpi_p_constructor_preserves_plan_order(monkeypatch):
    pairs = ((('hex', 10, 2), ('hex', 11, 4)),)
    face, ops = _entry(True, 2, 2, 3)
    plan = _plan(0)
    calls = []

    class Batch:
        def __init__(self, *args):
            calls.append(args)

    monkeypatch.setattr(emortars, '_EulerMPIPMortarBatch', Batch)
    got = emortars.EulerMortarInters.from_mpi_p_mortars(
        object(), (face,), (ops,), (plan,), {}, _cfg(), object()
    )

    assert len(got) == 1
    assert calls[0][1] is plan
    assert calls[0][2] == ((face, ops),)
    assert plan.face_pairs == pairs


def test_euler_mpi_p_constructor_rejects_unknown_face(monkeypatch):
    face, ops = _entry(True, 2, 2, 3)
    plan = _plan(0)
    plan.face_pairs = ((('hex', 20, 2), ('hex', 21, 4)),)

    with pytest.raises(ValueError, match='unknown face'):
        emortars.EulerMortarInters.from_mpi_p_mortars(
            object(), (face,), (ops,), (plan,), {}, _cfg(), object()
        )


def test_system_collectively_activates_mpi_p_on_empty_local_rank(monkeypatch):
    from pyfr.solvers.base import system as bsystem

    class Comm:
        rank = 2

        def allreduce(self, value, op=None):
            assert value is False
            return True

        def Dup(self):
            return SimpleNamespace(rank=self.rank)

    comm = Comm()
    monkeypatch.setattr(
        bsystem, 'get_comm_rank_root', lambda: (comm, comm.rank, 0)
    )
    monkeypatch.setattr(bsystem, 'autofree', lambda obj: obj)

    system = object.__new__(BaseSystem)
    system.ele_group_map = SimpleNamespace(
        split_mpi=lambda mesh, elemap, cfg: ({}, ())
    )
    system.backend = object()
    system.cfg = _cfg()
    calls = []

    class Mortar:
        @classmethod
        def from_mpi_p_mortars(cls, *args):
            calls.append(args)
            return ()

    system.mortarinterscls = Mortar
    mesh = SimpleNamespace(con_p={})

    assert system._load_mpi_inters(mesh, {}) == []
    assert system._has_mpi_p_mortars is True
    assert calls and calls[0][1:4] == ((), (), ())
    assert system._mpi_p_mortar_comm.rank == 2


def test_system_distributed_p_requires_equation_opt_in(monkeypatch):
    from pyfr.solvers.base import system as bsystem

    class Comm:
        rank = 3

        def allreduce(self, value, op=None):
            return True

    monkeypatch.setattr(
        bsystem, 'get_comm_rank_root', lambda: (Comm(), 3, 0)
    )

    system = object.__new__(BaseSystem)
    system.ele_group_map = SimpleNamespace(
        split_mpi=lambda mesh, elemap, cfg: ({}, ())
    )
    system.backend = object()
    system.cfg = _cfg()
    system.mortarinterscls = object

    with pytest.raises(RuntimeError, match='PDE execution is not supported'):
        system._load_mpi_inters(SimpleNamespace(con_p={}), {})


def test_euler_mpi_p_batch_messages_are_per_batch(monkeypatch):
    monkeypatch.setattr(emortars, 'mortar_side_view', lambda *args: object())
    elemap = {ElementGroupKey('hex', 2): SimpleNamespace(ndims=3, nvars=5)}

    face1, ops = _entry(True, 2, 2, 3)
    face2 = SimpleNamespace(**vars(face1))
    face2.face_pair = (('hex', 20, 2), ('hex', 21, 4))
    face2.local_side = MortarSide(
        'hex', 'quad', (2,), (1,), ElementGroupKey('hex', 2)
    )
    face2.geometry = SimpleNamespace(
        scaled_normals=(np.ones((3, 16, 1)),)
    )

    be = _Backend()
    batch = emortars._EulerMPIPMortarBatch(
        be, _plan(0), ((face1, ops), (face2, ops)), elemap,
        _cfg(), SimpleNamespace(rank=0), 0
    )

    assert batch.ninters == 2
    assert len(be.xchg) == 1
    assert be.xchg[0].shape == (16, 5, 2)
    assert set(batch.mpireqs) == {'mpi_p_state_recv', 'mpi_p_flux_send'}
