from types import SimpleNamespace

import pyfr.amrschedule as amrschedule

import pytest

from pyfr.amrschedule import (
    AMRScheduleError, AMRScheduleMixin, _future_schedule_targets,
)


def test_future_schedule_targets_absolute_grid_and_final_exclusion():
    targets = _future_schedule_targets(
        0.0, 1.0, 1e-12, regular_start=0.0, schedule_dt=0.2,
        initial_time=0.0025,
    )
    assert targets == pytest.approx((0.0025, 0.2, 0.4, 0.6, 0.8))


def test_future_schedule_targets_restart_keeps_absolute_phase():
    targets = _future_schedule_targets(
        0.45, 1.0, 1e-12, regular_start=0.0, schedule_dt=0.2,
        initial_time=0.0025,
    )
    assert targets == pytest.approx((0.6, 0.8))


def test_future_schedule_targets_deduplicates_initial_time():
    targets = _future_schedule_targets(
        0.0, 1.0, 1e-12, regular_start=0.2, schedule_dt=0.2,
        initial_time=0.2,
    )
    assert targets == pytest.approx((0.2, 0.4, 0.6, 0.8))


@pytest.mark.parametrize('dt', [0.0, -0.1, float('nan')])
def test_future_schedule_targets_rejects_invalid_dt(dt):
    with pytest.raises(AMRScheduleError):
        _future_schedule_targets(
            0.0, 1.0, 1e-12, regular_start=0.2, schedule_dt=dt,
        )


def test_schedule_mixin_splits_requested_advance_at_targets():
    calls = []

    class Schedule:
        def __init__(self):
            self.targets = (0.2, 0.4)
            self.done = set()

        def next_target(self, tcurr, requested, dtmin):
            return next(
                (t for t in self.targets
                 if t not in self.done and t <= requested + dtmin),
                None,
            )

        def after_advance_to(self, intg, target):
            calls.append(('amr', target))
            self.done.add(target)

    class Parent:
        def __init__(self):
            self.tcurr = 0.0
            self.dtmin = 1e-12
            self.amr_schedule = Schedule()

        def advance_to(self, target):
            calls.append(('advance', target))
            self.tcurr = target

    class Mixed(AMRScheduleMixin, Parent):
        def __init__(self):
            Parent.__init__(self)

    Mixed().advance_to(0.5)
    assert calls == [
        ('advance', 0.2), ('amr', 0.2),
        ('advance', 0.4), ('amr', 0.4),
        ('advance', 0.5),
    ]


def test_after_advance_matches_roundoff_target_once(monkeypatch):
    calls = []
    result = SimpleNamespace(
        decision=SimpleNamespace(action='none', marks=(), trigger_score=None),
        scores=(((0, ()), 0.0),), transaction=None,
    )
    monkeypatch.setattr(
        amrschedule, 'perform_indicator_mpi_amr_transaction',
        lambda *args, **kwargs: calls.append(args[0].tcurr) or result,
    )

    schedule = amrschedule.NativeAMRSchedule.__new__(
        amrschedule.NativeAMRSchedule
    )
    schedule.targets = (0.6000000000000001,)
    schedule._completed_targets = set()
    schedule.stage_dir = '/tmp'
    schedule.history = []
    intg = SimpleNamespace(tcurr=0.6, dtmin=1e-12)

    schedule.after_advance_to(intg, 0.6)
    schedule.after_advance_to(intg, 0.6000000000000001)
    assert calls == [0.6]
    assert len(schedule.history) == 1


def test_quad_checkpoint_failure_is_post_commit_and_target_stays_done(
        monkeypatch, tmp_path):
    tx = SimpleNamespace(stage_leaf_count=5, stage_mortar_count=1)
    result = SimpleNamespace(
        decision=SimpleNamespace(
            action='refine', marks=((0, ()),), trigger_score=0.2
        ),
        scores=(((0, ()), 0.2),), transaction=tx,
    )
    calls = []

    monkeypatch.setattr(
        'pyfr.amrtransaction.perform_indicator_quad_amr_transaction',
        lambda intg: calls.append(('amr', intg.tcurr)) or result,
    )

    def fail_checkpoint(intg, mesh_path, soln_path):
        calls.append(('checkpoint', mesh_path, soln_path))
        raise RuntimeError('injected post-commit checkpoint failure')

    monkeypatch.setattr(
        'pyfr.amrcheckpoint.write_online_quad_checkpoint', fail_checkpoint
    )

    schedule = amrschedule.NativeAMRSchedule.__new__(
        amrschedule.NativeAMRSchedule
    )
    schedule.mode = 'quad-1r'
    schedule.targets = (0.2,)
    schedule._completed_targets = set()
    schedule.stage_dir = None
    schedule.checkpoint_dir = tmp_path
    schedule.history = []
    schedule.checkpoints = []
    intg = SimpleNamespace(tcurr=0.2, dtmin=1e-12)

    with pytest.raises(RuntimeError, match='post-commit checkpoint'):
        schedule.after_advance_to(intg, 0.2)

    assert schedule._completed_targets == {0.2}
    assert len(schedule.history) == 1
    assert schedule.checkpoints == []
    assert calls[0] == ('amr', 0.2)
    assert calls[1][1:] == (
        tmp_path / 'online-amr-t0.20000000000000001.pyfrm',
        tmp_path / 'online-amr-t0.20000000000000001.pyfrs',
    )

    # A caller which catches an output failure must not replay the committed
    # adaptation at the same physical time.
    assert schedule.after_advance_to(intg, 0.2) is None
    assert calls.count(('amr', 0.2)) == 1


def test_schedule_mixin_rejects_plugin_config_before_parent_init():
    calls = []

    class Cfg:
        @staticmethod
        def sections():
            return ['solver-amr', 'soln-plugin-nancheck']

    class Parent:
        def __init__(self, *args, **kwargs):
            calls.append('parent-init')

    class Mixed(AMRScheduleMixin, Parent):
        pass

    args = (object(), object(), object(), None, Cfg())
    with pytest.raises(AMRScheduleError, match='one Quad writer'):
        Mixed(*args)
    assert calls == []


def test_schedule_mixin_rejects_writer_without_checkpoint_before_parent_init(
        monkeypatch):
    calls = []

    class Cfg:
        @staticmethod
        def sections():
            return ['solver-amr', 'soln-plugin-writer']

        @staticmethod
        def hasopt(section, option):
            return False

    class Mesh:
        etypes = ['quad']

    class Comm:
        size = 1

    class Parent:
        def __init__(self, *args, **kwargs):
            calls.append('parent-init')

    class Mixed(AMRScheduleMixin, Parent):
        pass

    monkeypatch.setattr(
        'pyfr.amrschedule.get_comm_rank_root', lambda: (Comm(), 0, 0)
    )
    args = (object(), object(), Mesh(), None, Cfg())
    with pytest.raises(AMRScheduleError, match='checkpoint-dir'):
        Mixed(*args)
    assert calls == []


def test_schedule_mixin_rejects_writer_on_nonquad_before_parent_init(
        monkeypatch):
    calls = []

    class Cfg:
        @staticmethod
        def sections():
            return ['solver-amr', 'soln-plugin-writer']

        @staticmethod
        def hasopt(section, option):
            return section == 'solver-amr' and option == 'checkpoint-dir'

    class Mesh:
        etypes = ['hex']

    class Comm:
        size = 1

    class Parent:
        def __init__(self, *args, **kwargs):
            calls.append('parent-init')

    class Mixed(AMRScheduleMixin, Parent):
        pass

    monkeypatch.setattr(
        'pyfr.amrschedule.get_comm_rank_root', lambda: (Comm(), 0, 0)
    )
    args = (object(), object(), Mesh(), None, Cfg())
    with pytest.raises(AMRScheduleError, match='one-rank Quad'):
        Mixed(*args)
    assert calls == []


def test_next_target_rejects_missed_pending_time():
    schedule = amrschedule.NativeAMRSchedule.__new__(
        amrschedule.NativeAMRSchedule
    )
    schedule.targets = (0.2, 0.4)
    schedule._completed_targets = set()
    with pytest.raises(AMRScheduleError, match='passed without adaptation'):
        schedule.next_target(0.3, 0.5, 1e-12)


def test_schedule_mixin_does_not_take_dtmin_step_after_near_target():
    calls = []

    class Schedule:
        def __init__(self):
            self.done = False

        def next_target(self, tcurr, requested, dtmin):
            if not self.done and 1.2 <= requested + dtmin:
                return 1.2
            return None

        def after_advance_to(self, intg, target):
            calls.append(('amr', target))
            self.done = True

    class Parent:
        def __init__(self):
            self.tcurr = 1.0
            self.dtmin = 1e-12
            self.amr_schedule = Schedule()

        def advance_to(self, target):
            calls.append(('advance', target))
            self.tcurr = target

    class Mixed(AMRScheduleMixin, Parent):
        def __init__(self):
            Parent.__init__(self)

    Mixed().advance_to(1.2000000000000002)
    assert calls == [('advance', 1.2), ('amr', 1.2)]


def test_native_schedule_requires_rank_agreement():
    class Comm:
        @staticmethod
        def allgather(signature):
            return [signature, (*signature[:-1], (0.25,))]

    with pytest.raises(AMRScheduleError, match='ranks disagree'):
        amrschedule._check_schedule_agreement(
            Comm(), (0.2, 0.0, 0.0025, (0.0025, 0.2))
        )


def test_integrator_without_schedule_dt_has_no_amr_mixin(monkeypatch):
    import pyfr.integrators as integrators

    class Controller:
        def __init__(self, *args, **kwargs):
            self.constructed = True

    class Stepper:
        pass

    def select(base, **kwargs):
        if 'controller_name' in kwargs:
            return Controller
        if 'stepper_name' in kwargs:
            return Stepper
        raise AssertionError('unexpected integrator component lookup')

    class Cfg:
        @staticmethod
        def get(section, option, default=None):
            values = {
                ('solver-time-integrator', 'controller'): 'none',
                ('solver-time-integrator', 'scheme'): 'rk4',
                ('solver-time-integrator', 'formulation'): 'explicit',
            }
            return values.get((section, option), default)

        @staticmethod
        def hasopt(section, option):
            return False

    monkeypatch.setattr(integrators, 'subclass_where', select)
    intg = integrators.get_integrator(
        object(), object(), object(), None, Cfg()
    )

    assert intg.constructed
    assert not isinstance(intg, AMRScheduleMixin)
    assert not hasattr(intg, 'amr_schedule')


def _adapted_quad_restart_intg(tmp_path, *, with_root=True, backend='openmp'):
    from pyfr.inifile import Inifile

    cfg = Inifile(
        '[solver]\n'
        'system = euler\n'
        '[solver-time-integrator]\n'
        'formulation = explicit\n'
        'scheme = rk4\n'
        'controller = none\n'
        'tstart = 0\n'
        'tend = 3\n'
        'dt = 0.1\n'
        '[solver-amr]\n'
        'refine-threshold = 0.05\n'
        'max-level = 2\n'
        'schedule-start = 0\n'
        'schedule-dt = 1\n'
    )
    if with_root:
        cfg.set('solver-amr', 'root-mesh', tmp_path / 'root.pyfrm')

    tree = SimpleNamespace(root_mesh_uuid='root-uuid')
    mesh = SimpleNamespace(etypes=['quad'], amr_tree=tree)
    system = SimpleNamespace(mesh=mesh)
    return SimpleNamespace(
        cfg=cfg, system=system, formulation='explicit',
        controller_name='none', stepper_name='rk4',
        backend=SimpleNamespace(name=backend), plugins=(), triggers=None,
        serialiser=SimpleNamespace(_serialfns={}), tstart=0.0, tend=3.0,
        tcurr=1.0, dtmin=1e-12, isrestart=True,
    )


def test_adapted_quad_restart_schedule_recovers_root_anchor(monkeypatch,
                                                             tmp_path):
    intg = _adapted_quad_restart_intg(tmp_path)
    root = SimpleNamespace(
        uuid='root-uuid', amr_tree=None, etypes=['quad'], con_p={}, mcon={},
        spts_curved={'quad': ()},
    )

    class Reader:
        def __init__(self, path):
            assert path == str((tmp_path / 'root.pyfrm').absolute())
            self.mesh = root

        def close(self):
            pass

    import pyfr.readers.native as native
    monkeypatch.setattr(native, 'NativeReader', Reader)
    schedule = amrschedule.NativeAMRSchedule(intg)

    assert intg._amr_root_mesh is root
    assert schedule.mode == 'quad-1r'
    assert schedule.targets == pytest.approx((2.0,))


def test_adapted_quad_restart_schedule_allows_cuda(monkeypatch, tmp_path):
    intg = _adapted_quad_restart_intg(tmp_path, backend='cuda')
    root = SimpleNamespace(
        uuid='root-uuid', amr_tree=None, etypes=['quad'], con_p={}, mcon={},
        spts_curved={'quad': ()},
    )

    class Reader:
        def __init__(self, path):
            assert path == str((tmp_path / 'root.pyfrm').absolute())
            self.mesh = root

        def close(self):
            pass

    import pyfr.readers.native as native
    monkeypatch.setattr(native, 'NativeReader', Reader)
    schedule = amrschedule.NativeAMRSchedule(intg)

    assert schedule.mode == 'quad-1r'
    assert schedule.targets == pytest.approx((2.0,))


def test_adapted_quad_restart_schedule_rejects_hip(monkeypatch, tmp_path):
    intg = _adapted_quad_restart_intg(tmp_path, backend='hip')
    root = SimpleNamespace(
        uuid='root-uuid', amr_tree=None, etypes=['quad'], con_p={}, mcon={},
        spts_curved={'quad': ()},
    )

    class Reader:
        def __init__(self, path):
            self.mesh = root

        def close(self):
            pass

    import pyfr.readers.native as native
    monkeypatch.setattr(native, 'NativeReader', Reader)
    with pytest.raises(AMRScheduleError, match='OpenMP or CUDA'):
        amrschedule.NativeAMRSchedule(intg)


def test_adapted_quad_restart_schedule_requires_root_anchor(tmp_path):
    intg = _adapted_quad_restart_intg(tmp_path, with_root=False)
    with pytest.raises(AMRScheduleError, match='requires.*root-mesh'):
        amrschedule.NativeAMRSchedule(intg)


def test_adapted_restart_dispatch(monkeypatch):
    calls = []
    intg = SimpleNamespace(
        isrestart=True,
        system=SimpleNamespace(mesh=SimpleNamespace(amr_tree=object())),
    )
    monkeypatch.setattr(
        amrschedule, '_recover_quad_root', lambda intg: calls.append('quad')
    )
    monkeypatch.setattr(
        amrschedule, '_recover_mixed_hex_root',
        lambda intg: calls.append('mixed-hex'),
    )

    amrschedule._recover_schedule_root(intg, 'quad-1r')
    amrschedule._recover_schedule_root(intg, 'mixed-hex-1r')
    assert calls == ['quad', 'mixed-hex']

    with pytest.raises(AMRScheduleError, match='mixed Quad.*adapted restart'):
        amrschedule._recover_schedule_root(intg, 'mixed-quad-1r')
    with pytest.raises(AMRScheduleError, match='Hex AMR.*adapted restart'):
        amrschedule._recover_schedule_root(intg, 'hex-mpi')

    intg.isrestart = False
    amrschedule._recover_schedule_root(intg, 'quad-1r')
    assert calls == ['quad', 'mixed-hex']


def test_quad_writer_transaction_error_is_schedule_error(monkeypatch):
    from pyfr.amrtransaction import AMRTransactionError

    intg = SimpleNamespace(
        formulation='explicit', controller_name='none', stepper_name='rk4',
        backend=SimpleNamespace(name='openmp'), plugins=(), triggers=None,
        serialiser=SimpleNamespace(_serialfns={}),
    )
    monkeypatch.setattr(
        'pyfr.amrtransaction._quad_online_writer_plugins',
        lambda intg: (_ for _ in ()).throw(
            AMRTransactionError('expected writer rejection')
        ),
    )

    with pytest.raises(AMRScheduleError, match='expected writer rejection'):
        amrschedule._validate_schedule_integrator(intg, 'quad-1r')


def test_native_hex_schedule_remains_openmp_only(monkeypatch):
    from pyfr.inifile import Inifile

    cfg = Inifile(
        '[solver-amr]\n'
        'schedule-dt = 1\n'
        'refine-threshold = 0.5\n'
        'coarsen-threshold = 0.25\n'
    )
    comm = SimpleNamespace(size=2)
    monkeypatch.setattr(
        amrschedule, 'get_comm_rank_root', lambda: (comm, 0, 0)
    )
    intg = SimpleNamespace(
        cfg=cfg, system=SimpleNamespace(
            mesh=SimpleNamespace(etypes=['hex'], amr_tree=None)
        ),
        formulation='explicit', controller_name='none', stepper_name='rk4',
        backend=SimpleNamespace(name='cuda'), plugins=(), triggers=None,
        serialiser=SimpleNamespace(_serialfns={}),
    )

    with pytest.raises(AMRScheduleError, match='Hex AMR.*OpenMP'):
        amrschedule.NativeAMRSchedule(intg)


def test_native_mixed_hex_mpi_schedule_selects_mixed_mode(monkeypatch,
                                                           tmp_path):
    from pyfr.inifile import Inifile

    cfg = Inifile(
        '[solver-amr]\n'
        'schedule-dt = 1\n'
        'refine-threshold = 0.5\n'
        'coarsen-threshold = 0.25\n'
        f'stage-dir = {tmp_path}\n'
    )
    comm = SimpleNamespace(size=2, allgather=lambda x: [x, x])
    monkeypatch.setattr(
        amrschedule, 'get_comm_rank_root', lambda: (comm, 0, 0)
    )
    intg = SimpleNamespace(
        cfg=cfg, system=SimpleNamespace(
            mesh=SimpleNamespace(
                etypes=['hex', 'pyr', 'tet'], amr_tree=None
            )
        ),
        formulation='explicit', controller_name='none', stepper_name='rk4',
        backend=SimpleNamespace(name='openmp'), plugins=(), triggers=None,
        serialiser=SimpleNamespace(_serialfns={}),
        tstart=0.0, tend=3.0, tcurr=0.0, dtmin=1e-12, isrestart=False,
    )

    schedule = amrschedule.NativeAMRSchedule(intg)
    assert schedule.mode == 'mixed-hex-mpi'
    assert schedule.stage_dir == tmp_path.absolute()


def test_after_advance_routes_mixed_hex_mpi(monkeypatch):
    calls = []
    tx = SimpleNamespace(stage_leaf_count=9, stage_mortar_count=1)
    result = SimpleNamespace(
        decision=SimpleNamespace(
            action='refine', marks=((0, ()),), trigger_score=0.2
        ),
        scores=(((0, ()), 0.2),), transaction=tx,
    )
    monkeypatch.setattr(
        amrschedule, 'perform_indicator_mpi_mixed_hex_amr_transaction',
        lambda *args, **kwargs: calls.append((args, kwargs)) or result,
    )

    schedule = amrschedule.NativeAMRSchedule.__new__(
        amrschedule.NativeAMRSchedule
    )
    schedule.mode = 'mixed-hex-mpi'
    schedule.targets = (0.2,)
    schedule._completed_targets = set()
    schedule.stage_dir = '/tmp/stage'
    schedule.checkpoint_dir = None
    schedule.history = []
    schedule.checkpoints = []
    intg = SimpleNamespace(tcurr=0.2, dtmin=1e-12)

    schedule.after_advance_to(intg, 0.2)
    assert len(calls) == 1
    assert calls[0][0] == (intg,)
    assert calls[0][1] == {
        'shared_stage_dir': '/tmp/stage', 'repartition': True
    }
    assert schedule.history[0].marks == ((0, ()),)
