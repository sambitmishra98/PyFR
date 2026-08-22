"""Ordinary native checkpoints for committed online Quad AMR states."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyfr.amr import HexLeafTree, QuadLeafTree
from pyfr.amrmesh import (
    materialize_native_mixed_hex_tree, materialize_native_quad_tree,
)
from pyfr.amrwriter import (
    write_adapted_mixed_hex_mesh, write_adapted_quad_mesh,
)
from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root
from pyfr.readers.native import NativeReader
from pyfr.writers.native import NativeWriter


class OnlineQuadCheckpointError(RuntimeError):
    """A committed online Quad state cannot be checkpointed safely."""


@dataclass(frozen=True)
class OnlineQuadCheckpointResult:
    mesh_path: str
    soln_path: str
    mesh_uuid: str
    root_mesh_uuid: str
    tcurr: float
    leaf_count: int
    mortar_count: int


@dataclass(frozen=True)
class OnlineMixedHexCheckpointResult:
    mesh_path: str
    soln_path: str
    mesh_uuid: str
    root_mesh_uuid: str
    tcurr: float
    hex_leaf_count: int
    pyramid_count: int
    tet_count: int
    mortar_count: int


def _checkpoint_state(intg):
    comm, _, _ = get_comm_rank_root()
    if comm.size != 1:
        raise OnlineQuadCheckpointError(
            'online Quad checkpoints currently require one MPI rank'
        )
    if getattr(intg, 'stepinfo', None) != []:
        raise OnlineQuadCheckpointError(
            'online Quad checkpoint requires an accepted-step safe point'
        )

    system = getattr(intg, 'system', None)
    mesh = getattr(system, 'mesh', None)
    if system is None or mesh is None:
        raise OnlineQuadCheckpointError(
            'online Quad checkpoint requires a live PyFR system'
        )
    if mesh.etypes != ['quad'] or not isinstance(mesh.amr_tree, QuadLeafTree):
        raise OnlineQuadCheckpointError(
            'online Quad checkpoint requires an adapted pure-Quad mesh'
        )
    if mesh.con_p:
        raise OnlineQuadCheckpointError(
            'online Quad checkpoint does not support MPI topology'
        )

    root_mesh = getattr(intg, '_amr_root_mesh', None)
    if root_mesh is None or getattr(root_mesh, 'amr_tree', None) is not None:
        raise OnlineQuadCheckpointError(
            'online Quad checkpoint requires its immutable root anchor'
        )
    if root_mesh.uuid != mesh.amr_tree.root_mesh_uuid:
        raise OnlineQuadCheckpointError(
            'online Quad checkpoint root anchor disagrees with ancestry'
        )

    if getattr(getattr(intg, 'serialiser', None), '_serialfns', {}):
        raise OnlineQuadCheckpointError(
            'online Quad checkpoint excludes serialised mutable state'
        )

    try:
        state, = system.ele_scal_upts(intg.idxcurr)
    except (AttributeError, TypeError, ValueError):
        raise OnlineQuadCheckpointError(
            'online Quad checkpoint cannot read the accepted solution bank'
        ) from None
    state = np.array(state, copy=True, order='C')

    leaves = tuple(mesh.amr_tree.leaves())
    eidxs = np.asarray(mesh.eidxs.get('quad', ()), dtype=np.int64)
    if (state.ndim != 3 or state.shape[2] != len(leaves) or
            not np.array_equal(eidxs, np.arange(len(leaves)))):
        raise OnlineQuadCheckpointError(
            'online Quad checkpoint state/ancestry ordering is inconsistent'
        )

    return mesh, root_mesh, leaves, state


def _write_solution(intg, mesh, state, outpath):
    fields = tuple(intg.system.elementscls.convars(2, intg.cfg))
    stats = Inifile()
    stats.set('data', 'prefix', 'soln')
    intg.collect_stats(stats)

    metadata = {
        **intg.cfgmeta,
        'stats': stats.tostr(),
        'mesh-uuid': mesh.uuid,
    }
    metadata |= intg.serialiser.serialise()

    writer = NativeWriter(
        mesh, intg.cfg, state.dtype, outpath.parent, outpath.name,
        'soln', isrestart=False,
    )
    writer.set_shapes_eidxs(
        {'quad': (state.shape[1], state.shape[0])}, mesh.eidxs,
        {'soln': fields},
    )
    writer.write(
        {'quad': {'soln': state.transpose(2, 1, 0)}},
        intg.tcurr, metadata, timeout=0,
    )
    writer.flush()


def write_online_quad_checkpoint(intg, out_mesh_path, out_soln_path, *,
                                 lintol=1e-5):
    """Write an ordinary native mesh/solution pair after a committed event.

    This operation is deliberately separate from the AMR transaction.  It
    never participates in PREPARE/BUILD/VALIDATE/COMMIT and therefore cannot
    turn an output failure into an adaptation rollback claim.
    """
    mesh, root_mesh, leaves, state = _checkpoint_state(intg)

    out_mesh_path = Path(out_mesh_path)
    out_soln_path = Path(out_soln_path)
    if out_mesh_path == out_soln_path:
        raise OnlineQuadCheckpointError(
            'online Quad mesh and solution checkpoint paths must differ'
        )
    if out_mesh_path.exists() or out_soln_path.exists():
        raise OnlineQuadCheckpointError(
            'online Quad checkpoint output path already exists'
        )
    out_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    out_soln_path.parent.mkdir(parents=True, exist_ok=True)

    reader = None
    success = False
    try:
        raw = materialize_native_quad_tree(
            root_mesh, mesh.amr_tree, comm_size=1
        )
        if tuple(raw.leaf_order) != leaves:
            raise OnlineQuadCheckpointError(
                'checkpoint materialization changed active leaf ordering'
            )
        write_adapted_quad_mesh(raw, str(out_mesh_path), lintol=lintol)

        reader = NativeReader(str(out_mesh_path))
        out_mesh = reader.mesh
        if not isinstance(out_mesh.amr_tree, QuadLeafTree):
            raise OnlineQuadCheckpointError(
                'checkpoint mesh did not recover typed Quad ancestry'
            )
        if (out_mesh.amr_tree.root_mesh_uuid != root_mesh.uuid or
                tuple(out_mesh.amr_tree.leaves()) != leaves):
            raise OnlineQuadCheckpointError(
                'checkpoint mesh changed root identity or leaf ordering'
            )

        _write_solution(intg, out_mesh, state, out_soln_path)
        soln = reader.load_soln(str(out_soln_path))
        recovered = np.asarray(soln.data['quad'])
        if not np.array_equal(recovered, state):
            raise OnlineQuadCheckpointError(
                'checkpoint solution did not reload bit-for-bit'
            )

        result = OnlineQuadCheckpointResult(
            str(out_mesh_path), str(out_soln_path), out_mesh.uuid,
            root_mesh.uuid, float(intg.tcurr), len(leaves),
            sum(len(m.records) for m in out_mesh.mcon.values()),
        )
        success = True
        return result
    finally:
        if reader is not None:
            reader.close()
        if not success:
            out_mesh_path.unlink(missing_ok=True)
            out_soln_path.unlink(missing_ok=True)


def _mixed_hex_checkpoint_state(intg):
    comm, _, _ = get_comm_rank_root()
    if comm.size != 1:
        raise OnlineQuadCheckpointError(
            'mixed Hex checkpoints currently require one MPI rank'
        )
    if getattr(intg, 'stepinfo', None) != []:
        raise OnlineQuadCheckpointError(
            'mixed Hex checkpoint requires an accepted-step safe point'
        )

    system = getattr(intg, 'system', None)
    mesh = getattr(system, 'mesh', None)
    if system is None or mesh is None:
        raise OnlineQuadCheckpointError(
            'mixed Hex checkpoint requires a live PyFR system'
        )
    if (set(mesh.etypes) != {'hex', 'pyr', 'tet'} or
            not isinstance(mesh.amr_tree, HexLeafTree)):
        raise OnlineQuadCheckpointError(
            'mixed Hex checkpoint requires adapted Tet+Pyramid+Hex topology'
        )
    if mesh.con_p:
        raise OnlineQuadCheckpointError(
            'mixed Hex checkpoint does not support MPI topology'
        )

    root_mesh = getattr(intg, '_amr_root_mesh', None)
    if root_mesh is None or getattr(root_mesh, 'amr_tree', None) is not None:
        raise OnlineQuadCheckpointError(
            'mixed Hex checkpoint requires its immutable root anchor'
        )
    if root_mesh.uuid != mesh.amr_tree.root_mesh_uuid:
        raise OnlineQuadCheckpointError(
            'mixed Hex checkpoint root anchor disagrees with ancestry'
        )
    if getattr(getattr(intg, 'serialiser', None), '_serialfns', {}):
        raise OnlineQuadCheckpointError(
            'mixed Hex checkpoint excludes serialised mutable state'
        )

    states = {
        etype: np.array(state, copy=True, order='C')
        for etype, state in zip(
            system.ele_types, system.ele_scal_upts(intg.idxcurr)
        )
    }
    if set(states) != {'hex', 'pyr', 'tet'}:
        raise OnlineQuadCheckpointError(
            'mixed Hex checkpoint solution groups are incomplete'
        )
    leaves = tuple(mesh.amr_tree.leaves())
    if states['hex'].shape[2] != len(leaves):
        raise OnlineQuadCheckpointError(
            'mixed Hex checkpoint Hex state/ancestry ordering differs'
        )
    for etype in ('pyr', 'tet'):
        if states[etype].shape[2] != len(root_mesh.eidxs[etype]):
            raise OnlineQuadCheckpointError(
                f'mixed Hex checkpoint {etype} state count differs'
            )
    return mesh, root_mesh, leaves, states


def _write_mixed_hex_solution(intg, mesh, states, outpath):
    fields = tuple(intg.system.elementscls.convars(3, intg.cfg))
    stats = Inifile()
    stats.set('data', 'prefix', 'soln')
    intg.collect_stats(stats)
    metadata = {
        **intg.cfgmeta,
        'stats': stats.tostr(),
        'mesh-uuid': mesh.uuid,
    }
    metadata |= intg.serialiser.serialise()

    dtype = np.result_type(*(state.dtype for state in states.values()))
    writer = NativeWriter(
        mesh, intg.cfg, dtype, outpath.parent, outpath.name,
        'soln', isrestart=False,
    )
    writer.set_shapes_eidxs(
        {
            etype: (state.shape[1], state.shape[0])
            for etype, state in states.items()
        },
        mesh.eidxs, {'soln': fields},
    )
    writer.write(
        {
            etype: {'soln': state.transpose(2, 1, 0)}
            for etype, state in states.items()
        },
        intg.tcurr, metadata, timeout=0,
    )
    writer.flush()


def write_online_mixed_hex_checkpoint(
    intg, out_mesh_path, out_soln_path, *, lintol=1e-5
):
    """Write an ordinary native checkpoint after mixed-3D Hex AMR."""
    mesh, root_mesh, leaves, states = _mixed_hex_checkpoint_state(intg)
    out_mesh_path = Path(out_mesh_path)
    out_soln_path = Path(out_soln_path)
    if out_mesh_path == out_soln_path:
        raise OnlineQuadCheckpointError(
            'mixed Hex mesh and solution checkpoint paths must differ'
        )
    if out_mesh_path.exists() or out_soln_path.exists():
        raise OnlineQuadCheckpointError(
            'mixed Hex checkpoint output path already exists'
        )
    out_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    out_soln_path.parent.mkdir(parents=True, exist_ok=True)

    reader = None
    success = False
    try:
        raw = materialize_native_mixed_hex_tree(root_mesh, mesh.amr_tree)
        if tuple(raw.leaf_order) != leaves:
            raise OnlineQuadCheckpointError(
                'mixed Hex checkpoint changed active leaf ordering'
            )
        write_adapted_mixed_hex_mesh(raw, str(out_mesh_path), lintol=lintol)

        reader = NativeReader(str(out_mesh_path))
        out_mesh = reader.mesh
        if not isinstance(out_mesh.amr_tree, HexLeafTree):
            raise OnlineQuadCheckpointError(
                'mixed Hex checkpoint did not recover Hex ancestry'
            )
        if (out_mesh.amr_tree.root_mesh_uuid != root_mesh.uuid or
                tuple(out_mesh.amr_tree.leaves()) != leaves):
            raise OnlineQuadCheckpointError(
                'mixed Hex checkpoint changed root identity or leaves'
            )
        for etype in ('pyr', 'tet'):
            if not np.array_equal(out_mesh.spts[etype], root_mesh.spts[etype]):
                raise OnlineQuadCheckpointError(
                    f'mixed Hex checkpoint changed immutable {etype}'
                )

        _write_mixed_hex_solution(intg, out_mesh, states, out_soln_path)
        soln = reader.load_soln(str(out_soln_path))
        if set(soln.data) != set(states):
            raise OnlineQuadCheckpointError(
                'mixed Hex checkpoint solution groups changed'
            )
        for etype, state in states.items():
            if not np.array_equal(np.asarray(soln.data[etype]), state):
                raise OnlineQuadCheckpointError(
                    f'mixed Hex checkpoint {etype} did not reload bitwise'
                )

        result = OnlineMixedHexCheckpointResult(
            str(out_mesh_path), str(out_soln_path), out_mesh.uuid,
            root_mesh.uuid, float(intg.tcurr), len(leaves),
            len(out_mesh.eidxs['pyr']), len(out_mesh.eidxs['tet']),
            sum(len(m.records) for m in out_mesh.mcon.values()),
        )
        success = True
        return result
    finally:
        if reader is not None:
            reader.close()
        if not success:
            out_mesh_path.unlink(missing_ok=True)
            out_soln_path.unlink(missing_ok=True)
