from dataclasses import dataclass
import math

import numpy as np

from pyfr.amrindicator import (
    AMRIndicatorError, indicator_settings,
    perform_indicator_mpi_amr_transaction,
    perform_indicator_mpi_mixed_hex_amr_transaction,
)
from pyfr.mpiutil import get_comm_rank_root


class AMRScheduleError(AMRIndicatorError):
    pass


@dataclass(frozen=True)
class ScheduledAMREvent:
    time: float
    action: str
    marks: tuple
    trigger_score: float | None
    leaf_count: int
    mortar_count: int | None


def _future_schedule_targets(tcurr, tend, dtmin, *, regular_start,
                             schedule_dt, initial_time=None):
    values = (tcurr, tend, dtmin, regular_start, schedule_dt)
    if not all(np.isfinite(v) for v in values):
        raise AMRScheduleError('AMR schedule times must be finite')
    if schedule_dt <= 0:
        raise AMRScheduleError('AMR schedule-dt must be positive')
    if dtmin <= 0:
        raise AMRScheduleError('integrator dt-min must be positive')

    targets = []
    if initial_time is not None:
        if not np.isfinite(initial_time):
            raise AMRScheduleError('AMR initial-time must be finite')
        if tcurr + dtmin < initial_time < tend - dtmin:
            targets.append(float(initial_time))

    # regular_start is the origin of the absolute schedule grid.  Find the
    # first grid member strictly after the current physical time.
    k = max(0, math.floor((tcurr - regular_start)/schedule_dt) + 1)
    t = regular_start + k*schedule_dt
    while t < tend - dtmin:
        if t > tcurr + dtmin:
            targets.append(float(t))
        k += 1
        t = regular_start + k*schedule_dt

    targets.sort()
    unique = []
    for t in targets:
        if not unique or t - unique[-1] > dtmin:
            unique.append(t)
    return tuple(unique)


def _check_schedule_agreement(comm, signature):
    signatures = comm.allgather(signature)
    if len(set(signatures)) != 1:
        raise AMRScheduleError('MPI ranks disagree on native AMR schedule')



def _schedule_mode(intg, comm, mesh_etypes):
    cfg = intg.cfg

    if comm.size == 1 and mesh_etypes in {
        ('quad',), ('quad', 'tri'), ('hex', 'pyr', 'tet')
    }:
        from pyfr.amrtransaction import AMRTransactionError

        try:
            if mesh_etypes == ('hex', 'pyr', 'tet'):
                from pyfr.amrtransaction import _mixed_hex_online_settings
                _mixed_hex_online_settings(cfg)
                return 'mixed-hex-1r'

            from pyfr.amrtransaction import _quad_online_settings
            _quad_online_settings(cfg)
            return ('quad-1r' if mesh_etypes == ('quad',)
                    else 'mixed-quad-1r')
        except AMRTransactionError as exc:
            raise AMRScheduleError(str(exc)) from exc

    indicator_settings(cfg)
    if comm.size < 2:
        raise AMRScheduleError('native D9 Hex AMR scheduling requires MPI')

    return ('mixed-hex-mpi' if mesh_etypes == ('hex', 'pyr', 'tet')
            else 'hex-mpi')


def _validate_schedule_integrator(intg, mode):
    if getattr(intg, 'formulation', None) != 'explicit':
        raise AMRScheduleError('native D9 AMR scheduling requires explicit')
    if getattr(intg, 'controller_name', None) != 'none':
        raise AMRScheduleError(
            'native D9 AMR scheduling requires controller none'
        )
    if getattr(intg, 'stepper_name', None) != 'rk4':
        raise AMRScheduleError('native D9 AMR scheduling requires RK4')

    backend_name = getattr(getattr(intg, 'backend', None), 'name', None)
    if mode == 'quad-1r':
        if backend_name not in {'openmp', 'cuda'}:
            raise AMRScheduleError(
                'native Quad AMR scheduling requires OpenMP or CUDA'
            )
    elif mode == 'mixed-quad-1r':
        if backend_name != 'openmp':
            raise AMRScheduleError(
                'native mixed Quad AMR scheduling requires OpenMP'
            )
    elif mode == 'mixed-hex-1r':
        if backend_name not in {'openmp', 'cuda'}:
            raise AMRScheduleError(
                'native mixed Hex AMR scheduling requires OpenMP or CUDA'
            )
    elif backend_name != 'openmp':
        raise AMRScheduleError('native D9 Hex AMR scheduling requires OpenMP')

    if mode == 'quad-1r':
        from pyfr.amrtransaction import (
            AMRTransactionError, _quad_online_writer_plugins,
        )
        try:
            _quad_online_writer_plugins(intg)
        except AMRTransactionError as exc:
            raise AMRScheduleError(str(exc)) from exc
    elif (getattr(intg, 'plugins', ()) or
          getattr(intg, 'triggers', None)):
        raise AMRScheduleError(
            'native D9 AMR scheduling excludes plugins/triggers'
        )

    if getattr(getattr(intg, 'serialiser', None), '_serialfns', {}):
        raise AMRScheduleError(
            'native D9 AMR scheduling excludes serialised state'
        )


def _schedule_times(intg, comm):
    cfg = intg.cfg
    section = 'solver-amr'
    schedule_dt = cfg.getfloat(section, 'schedule-dt')
    regular_start = cfg.getfloat(section, 'schedule-start', intg.tstart)
    initial_time = (
        cfg.getfloat(section, 'initial-time')
        if cfg.hasopt(section, 'initial-time') else None
    )
    targets = _future_schedule_targets(
        intg.tcurr, intg.tend, intg.dtmin, regular_start=regular_start,
        schedule_dt=schedule_dt, initial_time=initial_time,
    )
    _check_schedule_agreement(
        comm, (schedule_dt, regular_start, initial_time, targets)
    )

    return schedule_dt, regular_start, initial_time, targets


def _schedule_paths(cfg, mode):
    section = 'solver-amr'
    stage_dir = (
        None if mode in {'quad-1r', 'mixed-quad-1r', 'mixed-hex-1r'}
        else cfg.getpath(section, 'stage-dir', abs=True)
    )

    if not cfg.hasopt(section, 'checkpoint-dir'):
        return stage_dir, None
    if mode not in {'quad-1r', 'mixed-hex-1r'}:
        raise AMRScheduleError(
            'native AMR event checkpoints require pure Quad or '
            'mixed Hex one-rank mode'
        )

    return stage_dir, cfg.getpath(section, 'checkpoint-dir', abs=True)


def _recover_quad_root(intg):
    from pyfr.readers.native import NativeReader

    cfg = intg.cfg
    section = 'solver-amr'
    if not cfg.hasopt(section, 'root-mesh'):
        raise AMRScheduleError(
            'adapted Quad restart requires solver-amr root-mesh'
        )

    root_path = cfg.getpath(section, 'root-mesh', abs=True)
    try:
        root_reader = NativeReader(str(root_path))
    except Exception as exc:
        raise AMRScheduleError(
            f'unable to open adapted Quad root mesh: {root_path}'
        ) from exc

    try:
        root_mesh = root_reader.mesh
        tree = intg.system.mesh.amr_tree
        if getattr(root_mesh, 'amr_tree', None) is not None:
            raise AMRScheduleError(
                'adapted Quad root-mesh anchor must be unadapted'
            )
        if root_mesh.uuid != tree.root_mesh_uuid:
            raise AMRScheduleError(
                'adapted Quad root-mesh anchor UUID mismatch'
            )
        if (root_mesh.etypes != ['quad'] or root_mesh.con_p or
                root_mesh.mcon or
                np.any(root_mesh.spts_curved.get('quad', ()))):
            raise AMRScheduleError(
                'adapted Quad root-mesh anchor is outside scope'
            )
        intg._amr_root_mesh = root_mesh
    finally:
        root_reader.close()


def _recover_mixed_hex_root(intg):
    from pyfr.amr import HexLeafTree
    from pyfr.readers.native import NativeReader

    cfg = intg.cfg
    section = 'solver-amr'
    if not cfg.hasopt(section, 'root-mesh'):
        raise AMRScheduleError(
            'adapted mixed Hex restart requires solver-amr root-mesh'
        )

    root_path = cfg.getpath(section, 'root-mesh', abs=True)
    try:
        root_reader = NativeReader(str(root_path))
    except Exception as exc:
        raise AMRScheduleError(
            f'unable to open mixed Hex root mesh: {root_path}'
        ) from exc

    try:
        root_mesh = root_reader.mesh
        tree = intg.system.mesh.amr_tree
        if not isinstance(tree, HexLeafTree):
            raise AMRScheduleError(
                'adapted mixed Hex restart requires Hex ancestry'
            )
        if getattr(root_mesh, 'amr_tree', None) is not None:
            raise AMRScheduleError(
                'mixed Hex root-mesh anchor must be unadapted'
            )
        if root_mesh.uuid != tree.root_mesh_uuid:
            raise AMRScheduleError(
                'mixed Hex root-mesh anchor UUID mismatch'
            )
        if (set(root_mesh.etypes) != {'hex', 'pyr', 'tet'} or
                root_mesh.con_p or root_mesh.mcon):
            raise AMRScheduleError(
                'mixed Hex root-mesh anchor is outside V10J scope'
            )
        intg._amr_root_mesh = root_mesh
    finally:
        root_reader.close()


def _recover_schedule_root(intg, mode):
    if (not intg.isrestart or
            getattr(intg.system.mesh, 'amr_tree', None) is None):
        return

    if mode == 'quad-1r':
        _recover_quad_root(intg)
    elif mode == 'mixed-quad-1r':
        raise AMRScheduleError(
            'native mixed Quad AMR scheduling does not yet support '
            'adapted restart'
        )
    elif mode == 'mixed-hex-1r':
        _recover_mixed_hex_root(intg)
    else:
        raise AMRScheduleError(
            'native D9 Hex AMR scheduling does not support adapted restart'
        )


def _perform_scheduled_amr(intg, mode, stage_dir):
    if mode == 'quad-1r':
        from pyfr.amrtransaction import (
            perform_indicator_quad_amr_transaction,
        )
        return perform_indicator_quad_amr_transaction(intg)
    if mode == 'mixed-quad-1r':
        from pyfr.amrtransaction import (
            perform_indicator_mixed_quad_amr_transaction,
        )
        return perform_indicator_mixed_quad_amr_transaction(intg)
    if mode == 'mixed-hex-1r':
        from pyfr.amrtransaction import (
            perform_indicator_mixed_hex_amr_transaction,
        )
        return perform_indicator_mixed_hex_amr_transaction(intg)
    if mode == 'mixed-hex-mpi':
        return perform_indicator_mpi_mixed_hex_amr_transaction(
            intg, shared_stage_dir=stage_dir, repartition=True
        )

    return perform_indicator_mpi_amr_transaction(
        intg, shared_stage_dir=stage_dir,
        ownership_policy='balanced-affinity-v1',
    )


def _write_schedule_checkpoint(intg, mode, checkpoint_dir, scheduled):
    tname = format(float(scheduled), '.17g')
    stem = f'online-amr-t{tname}'
    mesh_path = checkpoint_dir / f'{stem}.pyfrm'
    soln_path = checkpoint_dir / f'{stem}.pyfrs'

    if mode == 'mixed-hex-1r':
        from pyfr.amrcheckpoint import write_online_mixed_hex_checkpoint
        return write_online_mixed_hex_checkpoint(intg, mesh_path, soln_path)

    from pyfr.amrcheckpoint import write_online_quad_checkpoint
    return write_online_quad_checkpoint(intg, mesh_path, soln_path)


class NativeAMRSchedule:

    def __init__(self, intg):
        cfg = intg.cfg
        section = 'solver-amr'
        if not cfg.hasopt(section, 'schedule-dt'):
            raise AMRScheduleError(
                'native AMR scheduling requires schedule-dt'
            )

        comm, _, _ = get_comm_rank_root()
        mesh_etypes = tuple(getattr(intg.system.mesh, 'etypes', ()))
        self.mode = _schedule_mode(intg, comm, mesh_etypes)
        _validate_schedule_integrator(intg, self.mode)

        self.stage_dir, self.checkpoint_dir = _schedule_paths(cfg, self.mode)
        (self.schedule_dt, self.regular_start, self.initial_time,
         self.targets) = _schedule_times(intg, comm)
        _recover_schedule_root(intg, self.mode)

        self._completed_targets = set()
        self.history = []
        self.checkpoints = []

    def next_target(self, tcurr, requested, dtmin):
        for target in self.targets:
            if target in self._completed_targets:
                continue
            if target <= tcurr + dtmin:
                raise AMRScheduleError(
                    'an AMR target was passed without adaptation'
                )
            if target <= requested + dtmin:
                return target
            break
        return None

    def after_advance_to(self, intg, target):
        matches = [t for t in self.targets if abs(target - t) <= intg.dtmin]
        if not matches:
            return None
        scheduled = min(matches, key=lambda t: abs(target - t))
        if scheduled in self._completed_targets:
            return None
        if abs(intg.tcurr - scheduled) > intg.dtmin:
            raise AMRScheduleError(
                'integrator did not reach the AMR target time'
            )

        mode = getattr(self, 'mode', 'hex-mpi')
        result = _perform_scheduled_amr(intg, mode, self.stage_dir)
        tx = result.transaction
        event = ScheduledAMREvent(
            time=float(intg.tcurr),
            action=result.decision.action,
            marks=result.decision.marks,
            trigger_score=result.decision.trigger_score,
            leaf_count=(tx.stage_leaf_count if tx is not None
                        else len(result.scores)),
            mortar_count=tx.stage_mortar_count if tx is not None else None,
        )

        # The AMR transaction has already committed at this point.  Mark the
        # target complete before any optional output work so that a checkpoint
        # failure can never cause the same physical-time event to be replayed.
        self._completed_targets.add(scheduled)
        self.history.append(event)

        if tx is not None and self.checkpoint_dir is not None:
            checkpoint = _write_schedule_checkpoint(
                intg, mode, self.checkpoint_dir, scheduled
            )
            self.checkpoints.append(checkpoint)
        return result


class AMRScheduleMixin:

    def __init__(self, *args, **kwargs):
        cfg = kwargs.get('cfg', args[4] if len(args) > 4 else None)
        if cfg is None:
            raise AMRScheduleError('native D9 AMR scheduling requires config')
        plugin_sections = [
            s for s in cfg.sections()
            if s.startswith(('soln-plugin-', 'solver-plugin-', 'trigger-'))
        ]
        if plugin_sections:
            mesh = kwargs.get('mesh', args[2] if len(args) > 2 else None)
            comm, _, _ = get_comm_rank_root()
            if plugin_sections != ['soln-plugin-writer']:
                raise AMRScheduleError(
                    'native D9 AMR scheduling supports one Quad writer '
                    'plugin only'
                )
            if (comm.size != 1 or
                    tuple(getattr(mesh, 'etypes', ())) != ('quad',)):
                raise AMRScheduleError(
                    'native D9 writer scheduling requires one-rank Quad mode'
                )
            if not cfg.hasopt('solver-amr', 'checkpoint-dir'):
                raise AMRScheduleError(
                    'ONLINE2D writer lifecycle requires solver-amr '
                    'checkpoint-dir'
                )

        super().__init__(*args, **kwargs)
        self.amr_schedule = NativeAMRSchedule(self)

    def advance_to(self, target):
        schedule = self.amr_schedule
        while (scheduled := schedule.next_target(
            self.tcurr, target, self.dtmin
        )) is not None:
            super().advance_to(scheduled)
            schedule.after_advance_to(self, scheduled)

        if abs(target - self.tcurr) <= self.dtmin:
            return None
        return super().advance_to(target)
