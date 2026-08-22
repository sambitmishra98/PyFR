import numpy as np
import pytest

from pyfr.amr import QuadLeafTree
from pyfr.amrcheckpoint import (
    OnlineQuadCheckpointError, write_online_quad_checkpoint,
)
from pyfr.amrtransaction import perform_indicator_quad_amr_transaction
from pyfr.backends import get_backend
from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader
from pyfr.readers.native import Solution
from pyfr.shapes import QuadShape
from pyfr.solvers import get_solver
from pyfr.tests.test_amr_online_quad import (
    FIELDS, _cfg, _online_integrator, _root_reader,
)


def test_committed_repeated_quad_state_writes_ordinary_checkpoint(tmp_path):
    reader, intg = _online_integrator(tmp_path)
    out_reader = None
    try:
        perform_indicator_quad_amr_transaction(intg)
        intg.advance_to(2e-5)
        second = perform_indicator_quad_amr_transaction(intg)
        assert second.transaction.stage_leaf_count == 20

        state, = intg.system.ele_scal_upts(intg.idxcurr)
        state = np.array(state, copy=True)
        leaves = tuple(intg.system.mesh.amr_tree.leaves())
        root_uuid = intg._amr_root_mesh.uuid

        outm = tmp_path / 'online-event2.pyfrm'
        outs = tmp_path / 'online-event2.pyfrs'
        result = write_online_quad_checkpoint(intg, outm, outs)

        assert result.tcurr == pytest.approx(2e-5)
        assert result.leaf_count == 20
        assert result.root_mesh_uuid == root_uuid
        assert outm.is_file() and outs.is_file()

        out_reader = NativeReader(str(outm))
        assert isinstance(out_reader.mesh.amr_tree, QuadLeafTree)
        assert out_reader.mesh.amr_tree.root_mesh_uuid == root_uuid
        assert tuple(out_reader.mesh.amr_tree.leaves()) == leaves
        soln = out_reader.load_soln(str(outs))
        assert np.array_equal(soln.data['quad'], state)
        assert soln.stats.getfloat(
            'solver-time-integrator', 'tcurr'
        ) == pytest.approx(2e-5)
    finally:
        if out_reader is not None:
            out_reader.close()
        reader.close()


def test_online_checkpoint_failure_leaves_live_solver_and_no_partials(
        monkeypatch, tmp_path):
    reader, intg = _online_integrator(tmp_path)
    try:
        perform_indicator_quad_amr_transaction(intg)
        system = intg.system
        state, = system.ele_scal_upts(intg.idxcurr)
        state = np.array(state, copy=True)
        snapshot = (
            intg.tcurr, intg.idxcurr, intg.nacptsteps, intg.nrhsevals,
            intg.mesh_uuid, intg.serialiser, intg.gndofs,
        )

        def fail_write(*args, **kwargs):
            raise RuntimeError('injected checkpoint mesh-write failure')

        monkeypatch.setattr(
            'pyfr.amrcheckpoint.write_adapted_quad_mesh', fail_write
        )
        outm = tmp_path / 'failed.pyfrm'
        outs = tmp_path / 'failed.pyfrs'
        with pytest.raises(RuntimeError, match='injected checkpoint'):
            write_online_quad_checkpoint(intg, outm, outs)

        assert intg.system is system
        assert snapshot == (
            intg.tcurr, intg.idxcurr, intg.nacptsteps, intg.nrhsevals,
            intg.mesh_uuid, intg.serialiser, intg.gndofs,
        )
        current, = intg.system.ele_scal_upts(intg.idxcurr)
        assert np.array_equal(current, state)
        assert not outm.exists() and not outs.exists()
    finally:
        reader.close()


def test_online_checkpoint_rejects_unadapted_root(tmp_path):
    reader, intg = _online_integrator(tmp_path)
    try:
        with pytest.raises(
            OnlineQuadCheckpointError, match='adapted pure-Quad'
        ):
            write_online_quad_checkpoint(
                intg, tmp_path / 'root.pyfrm', tmp_path / 'root.pyfrs'
            )
    finally:
        reader.close()


def test_native_schedule_checkpoints_each_committed_quad_event(tmp_path):
    reader = _root_reader(tmp_path)
    outdir = tmp_path / 'event-checkpoints'
    out_readers = []
    try:
        cfg = _cfg()
        cfg.set('solver-time-integrator', 'tend', 3e-5)
        cfg.set('solver-amr', 'schedule-dt', 1e-5)
        cfg.set('solver-amr', 'checkpoint-dir', str(outdir))
        basis = QuadShape(None, cfg)
        nupts = basis.nupts

        rho = np.ones((nupts, 2))
        u = np.zeros_like(rho)
        u[:, 0] = np.linspace(0.0, 0.4, nupts)
        v = np.zeros_like(rho)
        p = np.full_like(rho, 10.0)
        energy = p/0.4 + 0.5*rho*(u*u + v*v)
        state = np.stack((rho, rho*u, rho*v, energy), axis=1)

        stats = Inifile()
        stats.set('solver-time-integrator', 'tcurr', 0.0)
        soln = Solution(
            config=cfg, stats=stats, fields=FIELDS,
            data={'quad': state}, state={},
        )
        intg = get_solver(
            get_backend('openmp', cfg), reader.mesh, soln, cfg
        )
        intg.run()

        assert [event.action for event in intg.amr_schedule.history] == [
            'refine', 'refine'
        ]
        assert len(intg.amr_schedule.checkpoints) == 2

        for checkpoint, leaves in zip(
            intg.amr_schedule.checkpoints, (5, 20)
        ):
            assert checkpoint.leaf_count == leaves
            assert checkpoint.root_mesh_uuid == reader.mesh.uuid
            assert checkpoint.mesh_path.endswith('.pyfrm')
            assert checkpoint.soln_path.endswith('.pyfrs')

            out_reader = NativeReader(checkpoint.mesh_path)
            out_readers.append(out_reader)
            out_soln = out_reader.load_soln(checkpoint.soln_path)
            assert len(tuple(out_reader.mesh.amr_tree.leaves())) == leaves
            assert out_soln.stats.getfloat(
                'solver-time-integrator', 'tcurr'
            ) == pytest.approx(checkpoint.tcurr)

        assert intg.tcurr == pytest.approx(3e-5)
        assert intg.tcurr > intg.amr_schedule.checkpoints[-1].tcurr
        current, = intg.system.ele_scal_upts(intg.idxcurr)
        assert np.isfinite(current).all()
    finally:
        for out_reader in out_readers:
            out_reader.close()
        reader.close()


def test_native_writer_plugin_rebinds_across_committed_quad_events(tmp_path):
    reader = _root_reader(tmp_path)
    outdir = tmp_path / 'event-checkpoints'
    writer_dir = tmp_path / 'writer-output'
    writer_dir.mkdir()
    opened = []
    try:
        cfg = _cfg()
        cfg.set('solver-time-integrator', 'tend', 4e-5)
        cfg.set('solver-amr', 'schedule-dt', 1e-5)
        cfg.set('solver-amr', 'checkpoint-dir', str(outdir))
        cfg.set('soln-plugin-writer', 'basedir', str(writer_dir))
        cfg.set(
            'soln-plugin-writer', 'basename', 'writer-{n:02d}-{t:.8f}'
        )
        cfg.set('soln-plugin-writer', 'dt-out', 1e-5)
        basis = QuadShape(None, cfg)
        nupts = basis.nupts

        rho = np.ones((nupts, 2))
        u = np.zeros_like(rho)
        u[:, 0] = np.linspace(0.0, 0.4, nupts)
        v = np.zeros_like(rho)
        p = np.full_like(rho, 10.0)
        energy = p/0.4 + 0.5*rho*(u*u + v*v)
        state = np.stack((rho, rho*u, rho*v, energy), axis=1)

        stats = Inifile()
        stats.set('solver-time-integrator', 'tcurr', 0.0)
        soln = Solution(
            config=cfg, stats=stats, fields=FIELDS,
            data={'quad': state}, state={},
        )
        intg = get_solver(
            get_backend('openmp', cfg), reader.mesh, soln, cfg
        )
        intg.run()

        assert [event.action for event in intg.amr_schedule.history] == [
            'refine', 'refine', 'none'
        ]
        assert len(intg.amr_schedule.checkpoints) == 2

        anchors = [(0.0, reader.mesh.fname)] + [
            (checkpoint.tcurr, checkpoint.mesh_path)
            for checkpoint in intg.amr_schedule.checkpoints
        ]
        outputs = sorted(writer_dir.glob('writer-*.pyfrs'))
        assert outputs
        assert [int(path.name.split('-')[1]) for path in outputs] == list(
            range(len(outputs))
        )

        seen = []
        for path in outputs:
            matches = []
            for event_time, mesh_path in anchors:
                native = NativeReader(str(mesh_path))
                opened.append(native)
                try:
                    written = native.load_soln(str(path))
                except RuntimeError as exc:
                    if 'Invalid solution for mesh' not in str(exc):
                        raise
                else:
                    tcurr = written.stats.getfloat(
                        'solver-time-integrator', 'tcurr'
                    )
                    matches.append((event_time, tcurr))

            assert len(matches) == 1
            seen.append(matches[0])

        # The writer runs at accepted-step completion, before an AMR event at
        # the same time.  Subsequent writes must bind to the newly committed
        # topology until the next event replaces it.
        assert (0.0, pytest.approx(1e-5)) in seen
        assert (1e-5, pytest.approx(2e-5)) in seen
        assert (2e-5, pytest.approx(3e-5)) in seen
    finally:
        for native in opened:
            native.close()
        reader.close()
