from dataclasses import dataclass
import os
import re
import tempfile

import h5py
import numpy as np

from pyfr.amr import (
    apply_hex_refine_transfer, apply_hex_restrict_transfer,
    build_hex_refine_transfer, build_hex_restrict_transfer,
    encode_hex_leaf_tree, hex_coarsen_candidates, hex_octree_children,
    hex_tree_face_groups, hex_tree_face_pairs, hex_tree_leaves,
)
from pyfr.amrmesh import (
    _derive_root_face_topology, _hex_local_maps, materialize_native_hex_tree,
    materialize_native_mixed_hex_tree,
)
from pyfr.amrpolicy import balanced_affinity_destination_parts
from pyfr.amrtransaction import (
    AMRTransactionError, _conserved_totals, _copy_state, _eos_ranges,
    _integration_weights, _mesh_volumes, _normalise_mark, _scratch_bank,
    _scratch_rhs, _single_hex_bank, _tree_split_nodes,
    _close_mixed_hex_refinements, _physical_hex_conserved_totals,
)
from pyfr.amrwriter import write_adapted_mesh, write_adapted_mixed_hex_mesh
from pyfr.cache import clear_memoize
from pyfr.mpiutil import AlltoallMixin, get_comm_rank_root, mpi, scal_coll
from pyfr.partitioners.base import BasePartitioner, write_partitioning
from pyfr.partitioners.baseline import BaselinePartitioner
from pyfr.readers.native import Connectivity, Mesh, NativeReader, Solution
from pyfr.shapes import HexShape
from pyfr.writers.serialise import Serialiser


class MPIAMRTransactionError(AMRTransactionError):
    pass


class _CollectiveFailure(MPIAMRTransactionError):
    pass


@dataclass(frozen=True)
class MPIAMRTransactionResult:
    old_tree: object
    proposed_tree: object
    global_marks: tuple
    local_old_eidxs: tuple
    local_stage_eidxs: tuple
    local_transferred_state: np.ndarray
    rhs: tuple
    conservation_before: np.ndarray
    conservation_after: np.ndarray
    conservation_error: np.ndarray
    old_volume: float
    proposed_volume: float
    rho_range: tuple
    pressure_range: tuple
    stage_mesh_uuid: str
    stage_path: str
    stage_leaf_count: int
    stage_mortar_count: int
    stage_mpi_face_count: int
    bank_drift: float
    migration_send_counts: tuple
    migration_recv_counts: tuple
    accepted_bank: int
    scratch_bank: int
    tcurr: float


@dataclass(frozen=True)
class MPIMixedHexAMRTransactionResult:
    old_tree: object
    proposed_tree: object
    global_marks: tuple
    local_old_eidxs: tuple
    local_stage_eidxs: tuple
    conservation_before: np.ndarray
    conservation_after: np.ndarray
    conservation_error: np.ndarray
    old_volume: float
    proposed_volume: float
    rho_range: tuple
    pressure_range: tuple
    stage_mesh_uuid: str
    stage_path: str
    stage_leaf_count: int
    stage_pyr_count: int
    stage_tet_count: int
    stage_mortar_count: int
    stage_mpi_face_count: int
    bank_drift: float
    accepted_bank: int
    scratch_bank: int
    old_epoch: int
    new_epoch: int
    old_system_id: int
    new_system_id: int
    tcurr: float
    migration_send_counts: tuple = ()
    migration_recv_counts: tuple = ()
    repartitioned: bool = False


class _LeafStateExchanger(AlltoallMixin):
    pass


def _collective_error(comm, label, error=None):
    msg = '' if error is None else f'{type(error).__name__}: {error}'
    msgs = comm.allgather(msg)
    if any(msgs):
        detail = '; '.join(f'rank {i}: {m}' for i, m in enumerate(msgs) if m)
        raise _CollectiveFailure(f'D7A {label} failed: {detail}')


def _normalise_destination_parts(parts, proposed_tree, size):
    if not hasattr(parts, 'items'):
        raise MPIAMRTransactionError(
            'D7A destination ownership must be a leaf -> rank mapping'
        )

    norm = {}
    for leaf, rank in parts.items():
        nleaf = _normalise_mark(leaf)
        if (not isinstance(rank, (int, np.integer)) or
                not 0 <= int(rank) < size):
            raise MPIAMRTransactionError(
                f'D7A invalid destination rank for leaf {nleaf!r}: {rank!r}'
            )
        if nleaf in norm:
            raise MPIAMRTransactionError(
                f'D7A duplicate destination leaf: {nleaf!r}'
            )
        norm[nleaf] = int(rank)

    leaves = tuple(proposed_tree.leaves())
    if set(norm) != set(leaves):
        missing = sorted(set(leaves) - set(norm), key=repr)
        extra = sorted(set(norm) - set(leaves), key=repr)
        raise MPIAMRTransactionError(
            'D7A destination ownership must cover the proposed tree exactly '
            f'(missing={missing!r}, extra={extra!r})'
        )

    vparts = np.array([norm[leaf] for leaf in leaves], dtype=np.int32)
    if set(vparts.tolist()) != set(range(size)):
        raise MPIAMRTransactionError(
            'D7A initial mechanics gate requires every MPI rank to own '
            'at least one proposed leaf'
        )
    return norm, vparts


def _validate_mortar_affinity(raw, vparts):
    for mortar in raw.mortars:
        eidxs = (mortar.left_eidx, *mortar.right_eidx)
        owners = {int(vparts[int(i)]) for i in eidxs}
        if len(owners) != 1:
            raise MPIAMRTransactionError(
                f'D7A destination ownership splits mortar {mortar.name!r} '
                f'across ranks {sorted(owners)}'
            )


def _serial_root_mesh(fname, expected_uuid=None):
    with h5py.File(fname, 'r') as f:
        if 'amr' in f:
            raise MPIAMRTransactionError(
                'D7A immutable root file must be an unadapted native mesh'
            )
        if 'periodic' in f or 'mortars' in f:
            raise MPIAMRTransactionError(
                'D7A root file does not support periodic or pre-existing '
                'mortar topology'
            )
        etypes = sorted(f['eles'])
        if etypes != ['hex']:
            raise MPIAMRTransactionError('D7A requires a pure-Hex root file')

        uuid = f['mesh-uuid'][()].decode()
        if expected_uuid is not None and uuid != expected_uuid:
            raise MPIAMRTransactionError(
                'D7A root file UUID does not match persisted ancestry'
            )
        codec = [c.decode() for c in f['codec'][()]]
        cidxmap = {}
        for cidx, name in enumerate(codec):
            if m := re.fullmatch(r'eles/(\w+)/face/(\d+)', name):
                cidxmap[cidx] = (m[1], int(m[2]))

        einfo = f['eles/hex'][()]
        if np.any(einfo['curved']):
            raise MPIAMRTransactionError('D7A requires affine root Hexes')
        nodes = f['nodes'][()]
        node_locs = np.asarray(nodes['location'])
        neles = len(einfo)

        # Build complete serial native connectivity from the authoritative face
        # records.  Interior records appear twice, so retain one orientation.
        lc, le, rc, reidx = [], [], [], []
        btmp = {}
        for geidx, efaces in enumerate(einfo['faces']):
            for fidx, face in enumerate(efaces):
                own_cidx = codec.index(f'eles/hex/face/{fidx}')
                off, cidx = int(face['off']), int(face['cidx'])
                if off >= 0:
                    if geidx < off:
                        lc.append(own_cidx); le.append(geidx)
                        rc.append(cidx); reidx.append(off)
                elif off == -1:
                    name = codec[cidx]
                    if not name.startswith('bc/'):
                        raise MPIAMRTransactionError(
                            'D7A root boundary face has invalid native codec'
                        )
                    btmp.setdefault(name[3:], ([], []))[0].append(own_cidx)
                    btmp[name[3:]][1].append(geidx)
                else:
                    raise MPIAMRTransactionError(
                        'D7A root file contains unsupported face connectivity'
                    )

        con = (
            Connectivity(np.asarray(lc, np.int16), np.asarray(le, np.int64),
                         cidxmap),
            Connectivity(np.asarray(rc, np.int16), np.asarray(reidx, np.int64),
                         cidxmap),
        )
        bcon = {
            name: Connectivity(np.asarray(cs, np.int16),
                               np.asarray(es, np.int64), cidxmap)
            for name, (cs, es) in btmp.items()
        }
        spts_nodes = np.asarray(einfo['nodes'], dtype=np.int64)
        spts = node_locs[spts_nodes].swapaxes(0, 1)

        return Mesh(
            fname=fname, raw=None, ndims=node_locs.shape[1], creator='d7-root',
            codec=codec, uuid=uuid, version=int(f['version'][()]),
            etypes=['hex'], eidxs={'hex': np.arange(neles, dtype=np.int64)},
            spts={'hex': spts}, spts_nodes={'hex': spts_nodes},
            spts_curved={'hex': np.asarray(einfo['curved'])},
            colours={'hex': np.asarray(einfo['colour'])},
            tags={'hex': np.asarray(einfo['tags'])}, con=con, con_p={},
            bcon=bcon, mcon={}, cidxmap=cidxmap,
            node_idxs=np.arange(len(nodes), dtype=np.int64),
            node_valency=np.asarray(nodes['valency']), node_locs=node_locs,
            amr_tree=None,
        )


def _serial_mixed_root_mesh(fname, expected_uuid=None):
    with h5py.File(fname, 'r') as f:
        if 'amr' in f:
            raise MPIAMRTransactionError(
                'V10K immutable mixed root file must be unadapted'
            )
        if 'periodic' in f or 'mortars' in f:
            raise MPIAMRTransactionError(
                'V10K mixed root does not support periodic or pre-existing '
                'mortar topology'
            )

        etypes = sorted(f['eles'])
        if set(etypes) != {'hex', 'pyr', 'tet'}:
            raise MPIAMRTransactionError(
                'V10K requires exactly Tet+Pyramid+Hex root topology'
            )

        uuid = f['mesh-uuid'][()].decode()
        if expected_uuid is not None and uuid != expected_uuid:
            raise MPIAMRTransactionError(
                'V10K mixed root UUID does not match persisted ancestry'
            )

        codec = [c.decode() for c in f['codec'][()]]
        cidxmap = {}
        for cidx, name in enumerate(codec):
            if m := re.fullmatch(r'eles/(\w+)/face/(\d+)', name):
                cidxmap[cidx] = (m[1], int(m[2]))

        nodes = f['nodes'][()]
        node_locs = np.asarray(nodes['location'])
        einfo = {etype: f[f'eles/{etype}'][()] for etype in etypes}

        lc, le, rc, reidx = [], [], [], []
        btmp = {}
        for etype in etypes:
            for geidx, efaces in enumerate(einfo[etype]['faces']):
                for fidx, face in enumerate(efaces):
                    own_cidx = codec.index(f'eles/{etype}/face/{fidx}')
                    off, cidx = int(face['off']), int(face['cidx'])
                    if off >= 0:
                        try:
                            ret, rfidx = cidxmap[cidx]
                        except KeyError as exc:
                            raise MPIAMRTransactionError(
                                'V10K mixed root has invalid interior codec'
                            ) from exc
                        lhs = (etype, geidx, fidx)
                        rhs = (ret, off, rfidx)
                        if lhs < rhs:
                            lc.append(own_cidx)
                            le.append(geidx)
                            rc.append(cidx)
                            reidx.append(off)
                    elif off == -1:
                        name = codec[cidx]
                        if not name.startswith('bc/'):
                            raise MPIAMRTransactionError(
                                'V10K mixed root boundary has invalid codec'
                            )
                        btmp.setdefault(name[3:], ([], []))[0].append(
                            own_cidx
                        )
                        btmp[name[3:]][1].append(geidx)
                    else:
                        raise MPIAMRTransactionError(
                            'V10K mixed root has unsupported face connectivity'
                        )

        con = (
            Connectivity(np.asarray(lc, np.int16), np.asarray(le, np.int64),
                         cidxmap),
            Connectivity(np.asarray(rc, np.int16),
                         np.asarray(reidx, np.int64), cidxmap),
        )
        bcon = {
            name: Connectivity(np.asarray(cs, np.int16),
                               np.asarray(es, np.int64), cidxmap)
            for name, (cs, es) in btmp.items()
        }

        spts_nodes = {
            etype: np.asarray(einfo[etype]['nodes'], dtype=np.int64)
            for etype in etypes
        }
        spts = {
            etype: node_locs[spts_nodes[etype]].swapaxes(0, 1)
            for etype in etypes
        }
        eidxs = {
            etype: np.arange(len(einfo[etype]), dtype=np.int64)
            for etype in etypes
        }

        return Mesh(
            fname=fname, raw=None, ndims=node_locs.shape[1],
            creator='v10k-mixed-root', codec=codec, uuid=uuid,
            version=int(f['version'][()]), etypes=etypes, eidxs=eidxs,
            spts=spts, spts_nodes=spts_nodes,
            spts_curved={et: np.asarray(einfo[et]['curved']) for et in etypes},
            colours={et: np.asarray(einfo[et]['colour']) for et in etypes},
            tags={et: np.asarray(einfo[et]['tags']) for et in etypes},
            con=con, con_p={}, bcon=bcon, mcon={}, cidxmap=cidxmap,
            node_idxs=np.arange(len(nodes), dtype=np.int64),
            node_valency=np.asarray(nodes['valency']), node_locs=node_locs,
            amr_tree=None,
        )


def _distributed_mixed_current_tree(mesh, comm):
    geidx = np.asarray(mesh.eidxs.get('hex', ()), dtype=np.int64)
    if geidx.ndim != 1 or len(np.unique(geidx)) != len(geidx):
        raise MPIAMRTransactionError('V10K local Hex ids are invalid')

    if mesh.amr_tree is None:
        uuids = comm.allgather(mesh.uuid)
        if len(set(uuids)) != 1:
            raise MPIAMRTransactionError('V10K ranks disagree on mesh UUID')
        owned = comm.allgather(geidx)
        flat = np.concatenate(owned)
        if len(np.unique(flat)) != len(flat):
            raise MPIAMRTransactionError('V10K root Hex ownership overlaps')
        roots = np.sort(flat)
        if not len(roots):
            raise MPIAMRTransactionError('V10K mixed root has no Hexes')
        tree = encode_hex_leaf_tree(
            mesh.uuid, [(int(eidx), ()) for eidx in roots]
        )
        local = {(int(eidx), ()): i for i, eidx in enumerate(geidx)}
    else:
        tree = mesh.amr_tree
        leaves = tuple(tree.leaves())
        if np.any((geidx < 0) | (geidx >= len(leaves))):
            raise MPIAMRTransactionError(
                'V10K adapted local Hex ordinal is outside ancestry'
            )
        local = {leaves[int(eidx)]: i for i, eidx in enumerate(geidx)}
        owned = comm.allgather(geidx)
        flat = np.concatenate(owned)
        if (len(flat) != len(leaves) or
                not np.array_equal(np.sort(flat), np.arange(len(leaves)))):
            raise MPIAMRTransactionError(
                'V10K adapted Hex ownership is not an exact partition'
            )

    return tree, local


def _assemble_typed_ownership(mesh, old_tree, local_by_leaf, comm):
    owned = {'hex': tuple(local_by_leaf)}
    for etype in ('pyr', 'tet'):
        owned[etype] = tuple(map(int, mesh.eidxs.get(etype, ())))

    gathered = comm.allgather(owned)
    parts = {'hex': {}, 'pyr': {}, 'tet': {}}
    for rank, rank_owned in enumerate(gathered):
        for leaf in rank_owned['hex']:
            if leaf in parts['hex']:
                raise MPIAMRTransactionError(
                    f'V10K duplicate Hex owner for {leaf!r}'
                )
            parts['hex'][leaf] = rank
        for etype in ('pyr', 'tet'):
            for eidx in rank_owned[etype]:
                if eidx in parts[etype]:
                    raise MPIAMRTransactionError(
                        f'V10K duplicate {etype} owner for {eidx}'
                    )
                parts[etype][eidx] = rank

    if set(parts['hex']) != set(old_tree.leaves()):
        raise MPIAMRTransactionError(
            'V10K global Hex ownership does not cover the current tree'
        )
    return parts


def _inherited_mixed_ownership(root_mesh, old_tree, proposed_tree, old_parts,
                               size):
    owner = {'hex': np.empty(proposed_tree.nleaves, dtype=np.int32)}
    old_hex = old_parts['hex']
    for i, leaf in enumerate(proposed_tree.leaves()):
        ancestor = leaf
        while ancestor not in old_hex:
            root, path = ancestor
            if not path:
                raise MPIAMRTransactionError(
                    f'V10K proposed leaf {leaf!r} has no old owner'
                )
            ancestor = (root, path[:-1])
        owner['hex'][i] = old_hex[ancestor]

    for etype in ('pyr', 'tet'):
        neles = len(root_mesh.eidxs[etype])
        if set(old_parts[etype]) != set(range(neles)):
            raise MPIAMRTransactionError(
                f'V10K {etype} ownership does not cover immutable roots'
            )
        owner[etype] = np.array(
            [old_parts[etype][i] for i in range(neles)], dtype=np.int32
        )

    for etype, vparts in owner.items():
        if np.any((vparts < 0) | (vparts >= size)):
            raise MPIAMRTransactionError(
                f'V10K {etype} ownership contains an invalid rank'
            )
    counts = np.zeros(size, dtype=np.int64)
    for vparts in owner.values():
        counts += np.bincount(vparts, minlength=size)
    if np.any(counts == 0):
        raise MPIAMRTransactionError(
            'V10K K2 requires every rank to own a real volume element'
        )
    return owner


def _validate_mixed_mortar_affinity(raw, owner):
    for mortar in raw.mortars:
        ranks = {int(owner[mortar.left_etype][mortar.left_eidx])}
        ranks.update(
            int(owner[etype][eidx])
            for etype, eidx in zip(mortar.right_etype, mortar.right_eidx)
        )
        if len(ranks) != 1:
            raise MPIAMRTransactionError(
                f'V10K ownership splits mortar {mortar.name!r} across '
                f'ranks {sorted(ranks)}'
            )


def _typed_ownership_from_partitioning(mesh, pinfo, size):
    (peidx, pregions), _ = pinfo
    etypes = sorted(mesh['eles'])
    if pregions.shape != (size, len(etypes) + 1):
        raise MPIAMRTransactionError(
            'V10K partition regions have an unexpected shape'
        )

    owner = {}
    for i, etype in enumerate(etypes):
        neles = len(mesh[f'eles/{etype}'])
        parts = np.full(neles, -1, dtype=np.int32)
        for rank in range(size):
            s, e = map(int, pregions[rank, i:i + 2])
            eidxs = np.asarray(peidx[s:e], dtype=np.int64)
            if np.any((eidxs < 0) | (eidxs >= neles)):
                raise MPIAMRTransactionError(
                    f'V10K partition contains invalid {etype} indices'
                )
            if np.any(parts[eidxs] != -1):
                raise MPIAMRTransactionError(
                    f'V10K partition assigns {etype} elements twice'
                )
            parts[eidxs] = rank
        if np.any(parts < 0):
            raise MPIAMRTransactionError(
                f'V10K partition does not cover every {etype} element'
            )
        owner[etype] = parts

    counts = np.zeros(size, dtype=np.int64)
    for parts in owner.values():
        counts += np.bincount(parts, minlength=size)
    if np.any(counts == 0):
        raise MPIAMRTransactionError(
            'V10K partition leaves an MPI rank without volume elements'
        )
    return owner


def _prepare_mixed_stage_root(root_fname, root_uuid, old_tree, marks,
                              old_parts, size, shared_stage_dir,
                              repartition=False):
    root_mesh = _serial_mixed_root_mesh(root_fname, root_uuid)
    proposed_tree, closure = _close_mixed_hex_refinements(
        root_mesh, old_tree, marks
    )
    raw = materialize_native_mixed_hex_tree(root_mesh, proposed_tree)

    if not repartition:
        owner = _inherited_mixed_ownership(
            root_mesh, old_tree, proposed_tree, old_parts, size
        )
        _validate_mixed_mortar_affinity(raw, owner)

    os.makedirs(shared_stage_dir, exist_ok=True)
    stage_path = None
    try:
        fd, stage_path = tempfile.mkstemp(
            prefix='pyfr-v10k-', suffix='.pyfrm', dir=shared_stage_dir
        )
        os.close(fd)
        os.unlink(stage_path)
        write_adapted_mixed_hex_mesh(raw, stage_path)

        with h5py.File(stage_path, 'r+') as f:
            if repartition:
                # The baseline partitioner hard-merges every complete mortar
                # group before graph partitioning.  Keep this fail-closed
                # validation as an independent ownership check afterwards.
                part = BaselinePartitioner(
                    [1]*size, elewts='balanced'
                )
                pinfo = part.partition(f)
                owner = _typed_ownership_from_partitioning(f, pinfo, size)
                _validate_mixed_mortar_affinity(raw, owner)
                pname = f'amr-v10k-baseline-{size}'
            else:
                con, ecurved, _, edisps, _ = (
                    BasePartitioner.construct_global_con(f)
                )
                etypes = sorted(f['eles'])
                vparts = np.concatenate([owner[e] for e in etypes])
                pinfo = BasePartitioner.construct_partitioning(
                    f, ecurved, edisps, con, vparts
                )
                pname = f'amr-v10k-{size}'

            write_partitioning(f, pname, pinfo)
            stage_uuid = f['mesh-uuid'][()].decode()
    except Exception:
        if stage_path:
            try:
                os.unlink(stage_path)
            except FileNotFoundError:
                pass
        raise

    return (proposed_tree, closure, owner, stage_path, pname, stage_uuid,
            raw)


def _distributed_current_tree(mesh, comm):
    geidx = np.asarray(mesh.eidxs.get('hex', ()), dtype=np.int64)
    if geidx.ndim != 1 or not len(geidx):
        raise MPIAMRTransactionError(
            'D7A initial mechanics gate requires local Hexes on every rank'
        )
    if len(np.unique(geidx)) != len(geidx):
        raise MPIAMRTransactionError('D7A local Hex ids are not unique')

    if mesh.amr_tree is None:
        uuids = comm.allgather(mesh.uuid)
        if len(set(uuids)) != 1:
            raise MPIAMRTransactionError(
                'D7A ranks disagree on root mesh UUID'
            )
        all_eidxs = comm.allgather(geidx)
        flat = np.concatenate(all_eidxs)
        if len(np.unique(flat)) != len(flat):
            raise MPIAMRTransactionError('D7A root Hex ownership overlaps')
        roots = np.sort(flat)
        tree = encode_hex_leaf_tree(mesh.uuid,
                                    [(int(e), ()) for e in roots])
        local = {(int(e), ()): i for i, e in enumerate(geidx)}
    else:
        tree = mesh.amr_tree
        leaves = tuple(tree.leaves())
        if np.any((geidx < 0) | (geidx >= len(leaves))):
            raise MPIAMRTransactionError(
                'D7A adapted local Hex ordinal is outside persisted ancestry'
            )
        local = {leaves[int(e)]: i for i, e in enumerate(geidx)}
        owned = comm.allgather(geidx)
        flat = np.concatenate(owned)
        if (len(flat) != len(leaves) or
                not np.array_equal(np.sort(flat), np.arange(len(leaves)))):
            raise MPIAMRTransactionError(
                'D7A adapted leaf ownership is not a global exact partition'
            )

    return tree, local


def _close_refinements(root_mesh, old_tree, marks):
    l2g, _, pids_by_g, _ = _hex_local_maps(root_mesh)
    rootfaces, rootkinds = _derive_root_face_topology(
        root_mesh, pids_by_g, l2g
    )
    roots = sorted(pids_by_g)
    old_leaves = set(old_tree.leaves())
    marks = set(marks)
    if not marks or not marks <= old_leaves:
        raise MPIAMRTransactionError(
            'D7A global refinement marks must be active leaves'
        )

    split = _tree_split_nodes(old_tree) | marks
    while True:
        leaves = hex_tree_leaves(roots, split)
        groups = hex_tree_face_groups(leaves, rootfaces)
        try:
            pairs = list(hex_tree_face_pairs(groups, rootkinds))
        except ValueError as exc:
            raise MPIAMRTransactionError(str(exc)) from exc

        added = False
        for lhs, rhs in pairs:
            if abs(lhs['level'] - rhs['level']) <= 1:
                continue
            coarse = (lhs['leaf'] if lhs['level'] < rhs['level']
                      else rhs['leaf'])
            if coarse not in leaves:
                raise MPIAMRTransactionError(
                    'D7A closure selected a non-active coarse leaf'
                )
            if coarse not in split:
                split.add(coarse)
                added = True
        if not added:
            break

    return encode_hex_leaf_tree(old_tree.root_mesh_uuid, leaves)


def _close_coarsening(root_mesh, old_tree, marks):
    l2g, _, pids_by_g, _ = _hex_local_maps(root_mesh)
    rootfaces, rootkinds = _derive_root_face_topology(
        root_mesh, pids_by_g, l2g
    )
    leaves = tuple(old_tree.leaves())
    marks = set(marks)
    groups = hex_tree_face_groups(leaves, rootfaces)
    try:
        pairs = list(hex_tree_face_pairs(groups, rootkinds))
        face_pairs = [(lhs['leaf'], rhs['leaf']) for lhs, rhs in pairs]
        approved = hex_coarsen_candidates(leaves, marks, face_pairs)
    except ValueError as exc:
        raise MPIAMRTransactionError(str(exc)) from exc

    if not approved:
        raise MPIAMRTransactionError(
            'D9J requires at least one legal sibling family to coarsen'
        )

    retiring = set().union(*(hex_octree_children(p) for p in approved))
    if marks != retiring:
        raise MPIAMRTransactionError(
            'D9J coarsening marks must be complete legal sibling families'
        )

    proposed = (set(leaves) - retiring) | set(approved)
    return encode_hex_leaf_tree(old_tree.root_mesh_uuid, proposed)


def _produce_local_refined_records(system, old_tree, proposed_tree,
                                   old_state, local_by_leaf):
    basis = HexShape(None, system.cfg)
    refine = build_hex_refine_transfer(basis)
    if old_state.shape[:2] != (refine.nupts, old_state.shape[1]):
        raise MPIAMRTransactionError('D7A local state point layout is invalid')
    if old_state.shape[2] != len(local_by_leaf):
        raise MPIAMRTransactionError(
            'D7A local leaf ownership does not match its solution bank'
        )

    proposed = set(proposed_tree.leaves())
    ordinal = {leaf: i for i, leaf in enumerate(proposed_tree.leaves())}
    ords, states = [], []
    for leaf, col in local_by_leaf.items():
        if leaf in proposed:
            ords.append(ordinal[leaf])
            states.append(np.array(old_state[:, :, col], copy=True))
            continue

        root, path = leaf
        children = {(root, path + (o,)) for o in range(8)}
        if not children <= proposed:
            raise MPIAMRTransactionError(
                f'D7A proposed tree is not a direct refinement of {leaf!r}'
            )
        child = apply_hex_refine_transfer(
            refine, old_state[:, :, col:col + 1]
        )
        for o in range(8):
            cleaf = (root, path + (o,))
            ords.append(ordinal[cleaf])
            states.append(np.array(child[o, :, :, 0], copy=True))

    if states:
        sarr = np.stack(states)
    else:
        sarr = np.empty((0, old_state.shape[0], old_state.shape[1]),
                        dtype=old_state.dtype)
    return np.asarray(ords, dtype=np.int64), sarr


def _coarsened_parents(old_tree, proposed_tree):
    old = set(old_tree.leaves())
    proposed = set(proposed_tree.leaves())
    parents = tuple(sorted(
        leaf for leaf in proposed - old
        if hex_octree_children(leaf) <= old
    ))
    if not parents:
        raise MPIAMRTransactionError(
            'D9J proposed tree must contain restricted parents'
        )

    retiring = set().union(*(hex_octree_children(p) for p in parents))
    if old - proposed != retiring:
        raise MPIAMRTransactionError(
            'D9J proposed tree is not a direct sibling-collapse batch'
        )
    return parents


def _produce_local_coarsened_records(
    comm, system, old_tree, proposed_tree, old_state, local_by_leaf, vparts
):
    parents = _coarsened_parents(old_tree, proposed_tree)
    ordinal = {leaf: i for i, leaf in enumerate(proposed_tree.leaves())}
    parent_by_ord = {ordinal[parent]: parent for parent in parents}
    owner_by_ord = {po: int(vparts[po]) for po in parent_by_ord}
    child_info = {}
    retiring = set()
    for po, parent in parent_by_ord.items():
        root, path = parent
        for o in range(8):
            child = (root, path + (o,))
            child_info[child] = (po, o, owner_by_ord[po])
            retiring.add(child)

    # Route only retiring child states to each proposed parent owner.  The
    # tag is (proposed-parent ordinal, octant); solution data is never
    # allgathered.
    tags, lstates, dests = [], [], []
    for child, (po, octant, owner) in child_info.items():
        if child in local_by_leaf:
            tags.append((po, octant))
            dests.append(owner)
            lstates.append(np.array(
                old_state[:, :, local_by_leaf[child]], copy=True
            ))

    if tags:
        tags = np.asarray(tags, dtype=np.int64)
        family = np.stack(lstates)
        dests = np.asarray(dests, dtype=np.int32)
        order = np.lexsort((tags[:, 1], tags[:, 0], dests))
        tags = np.ascontiguousarray(tags[order])
        family = np.ascontiguousarray(family[order])
        dests = dests[order]
    else:
        tags = np.empty((0, 2), dtype=np.int64)
        family = np.empty(
            (0, old_state.shape[0], old_state.shape[1]),
            dtype=old_state.dtype
        )
        dests = np.empty(0, dtype=np.int32)
    scount = np.bincount(dests, minlength=comm.size).astype(np.int64)

    ex = _LeafStateExchanger()
    rtags, (rcount, rdisp) = ex._alltoallcv(comm, tags, scount)
    rstates, (rcount_s, rdisp_s) = ex._alltoallcv(comm, family, scount)

    owned = {po for po, owner in owner_by_ord.items() if owner == comm.rank}
    local_error = None
    if (not np.array_equal(rcount, rcount_s) or
            not np.array_equal(rdisp, rdisp_s)):
        local_error = MPIAMRTransactionError(
            'D9J family tag/state exchanges disagree on layout'
        )
    else:
        received = set(map(int, rtags[:, 0])) if len(rtags) else set()
        if received != owned:
            local_error = MPIAMRTransactionError(
                'D9J restriction owner received the wrong parent families'
            )
        else:
            for po in owned:
                octants = rtags[rtags[:, 0] == po, 1]
                if (len(octants) != 8 or len(np.unique(octants)) != 8 or
                        set(map(int, octants)) != set(range(8))):
                    local_error = MPIAMRTransactionError(
                        'D9J restriction owner did not receive all children'
                    )
                    break
    _collective_error(comm, 'coarsen family collection', local_error)

    parent_states = {}
    local_error = None
    try:
        restrict = build_hex_restrict_transfer(HexShape(None, system.cfg))
        for po in sorted(owned):
            rows = np.flatnonzero(rtags[:, 0] == po)
            pos = {int(rtags[i, 1]): int(i) for i in rows}
            cstate = np.stack([rstates[pos[o]] for o in range(8)])
            cstate = cstate[:, :, :, None]
            parent_states[po] = apply_hex_restrict_transfer(
                restrict, cstate
            )[:, :, 0]
    except Exception as exc:
        local_error = exc
    _collective_error(comm, 'coarsen restriction', local_error)

    ords, states = [], []
    proposed = set(proposed_tree.leaves())
    for leaf, col in local_by_leaf.items():
        if leaf in retiring:
            continue
        if leaf not in proposed:
            raise MPIAMRTransactionError(
                f'D9J has no transfer target for old leaf {leaf!r}'
            )
        ords.append(ordinal[leaf])
        states.append(np.array(old_state[:, :, col], copy=True))

    for po in sorted(owned):
        ords.append(po)
        states.append(np.array(parent_states[po], copy=True))

    if states:
        sarr = np.stack(states)
    else:
        sarr = np.empty(
            (0, old_state.shape[0], old_state.shape[1]),
            dtype=old_state.dtype
        )
    return np.asarray(ords, dtype=np.int64), sarr


def _exchange_leaf_states(comm, ords, states, vparts):
    schema = (states.dtype.str, states.shape[1], states.shape[2])
    schemas = comm.allgather(schema)
    if len(set(schemas)) != 1:
        raise MPIAMRTransactionError(
            f'D7A ranks disagree on migrated-state schema: {schemas!r}'
        )

    dest = vparts[ords] if len(ords) else np.empty(0, dtype=np.int32)
    order = np.lexsort((ords, dest)) if len(ords) else np.empty(0, dtype=int)
    sords = np.ascontiguousarray(ords[order])
    sstates = np.ascontiguousarray(states[order])
    dest = dest[order]
    scount = np.bincount(dest, minlength=comm.size).astype(np.int64)

    ex = _LeafStateExchanger()
    rords, (rcount, rdisp) = ex._alltoallcv(comm, sords, scount)
    rstates, (rcount_s, rdisp_s) = ex._alltoallcv(comm, sstates, scount)
    if (not np.array_equal(rcount, rcount_s) or
            not np.array_equal(rdisp, rdisp_s)):
        raise MPIAMRTransactionError(
            'D7A ordinal/state exchanges disagree on receive layout'
        )
    return rords, rstates, scount, rcount


def _stage_local_bank(stage_mesh, rords, rstates):
    eidxs = np.asarray(stage_mesh.eidxs.get('hex', ()), dtype=np.int64)
    if not len(eidxs):
        raise MPIAMRTransactionError(
            'D7A initial mechanics gate requires local proposed Hexes'
        )
    if (len(rords) != len(eidxs) or len(np.unique(rords)) != len(rords) or
            set(map(int, rords)) != set(map(int, eidxs))):
        raise MPIAMRTransactionError(
            'D7A received leaf ordinals do not match NativeReader ownership'
        )
    pos = {int(e): i for i, e in enumerate(rords)}
    local = np.stack([rstates[pos[int(e)]] for e in eidxs], axis=2)
    return local, eidxs


def _validate_mpi_integrator(intg):
    comm, _, _ = get_comm_rank_root()
    if comm.size < 2:
        raise MPIAMRTransactionError('D7A requires at least two MPI ranks')
    if getattr(intg, 'formulation', None) != 'explicit':
        raise MPIAMRTransactionError('D7A requires explicit formulation')
    if getattr(intg, 'controller_name', None) != 'none':
        raise MPIAMRTransactionError('D7A requires controller = none')
    if getattr(intg, 'stepper_name', None) != 'rk4':
        raise MPIAMRTransactionError('D7A requires RK4')
    if getattr(getattr(intg, 'backend', None), 'name', None) != 'openmp':
        raise MPIAMRTransactionError('D7A currently supports OpenMP only')
    if (getattr(intg, 'nacptsteps', 0) < 1 or
            getattr(intg, 'stepinfo', None) != []):
        raise MPIAMRTransactionError(
            'D7A requires a completed accepted-step safe point'
        )

    cfg = intg.cfg
    solver_system = cfg.get('solver', 'system')
    if solver_system not in {'euler', 'navier-stokes'}:
        raise MPIAMRTransactionError(
            'D7A supports Euler/constant-viscosity NS'
        )
    if (solver_system == 'navier-stokes' and
            cfg.get('solver', 'viscosity-correction', 'none') != 'none'):
        raise MPIAMRTransactionError(
            'D7A Navier-Stokes requires constant viscosity'
        )
    if cfg.get('solver', 'shock-capturing', 'none') != 'none':
        raise MPIAMRTransactionError('D7A requires shock-capturing = none')
    if any(s.startswith('solver-order-') for s in cfg.sections()):
        raise MPIAMRTransactionError('D7A does not support mixed-p')
    if getattr(intg, 'plugins', ()) or getattr(intg, 'triggers', None):
        raise MPIAMRTransactionError('D7A does not carry plugins/triggers')
    serialfns = getattr(getattr(intg, 'serialiser', None), '_serialfns', {})
    if serialfns:
        raise MPIAMRTransactionError(
            'D7A does not carry serialised mutable runtime state'
        )

    system = getattr(intg, 'system', None)
    mesh = getattr(system, 'mesh', None)
    if (system is None or mesh is None or
            getattr(system, 'name', None) != solver_system):
        raise MPIAMRTransactionError(
            'D7A requires a matching live PyFR system'
        )
    if mesh.etypes != ['hex'] or np.any(mesh.spts_curved.get('hex', ())):
        raise MPIAMRTransactionError('D7A requires pure affine Hex geometry')
    for mcon in mesh.mcon.values():
        if (mcon.format != 'one-to-many-v1' or
                mcon.template != 'quad-2x2' or mcon.nright != 4):
            raise MPIAMRTransactionError(
                'D7A live mesh has unsupported mortars'
            )

    root_mesh = getattr(intg, '_amr_root_mesh', None)
    if mesh.amr_tree is None:
        if mesh.mcon:
            raise MPIAMRTransactionError(
                'D7A unadapted root does not support pre-existing mortars'
            )
        root_mesh = mesh
    elif root_mesh is None:
        raise MPIAMRTransactionError(
            'D7A adapted restart requires a live immutable root anchor'
        )

    return comm, system, mesh, root_mesh


def _physical_hex_volume(spts, basis):
    spts = np.asarray(spts)
    state = np.ones((basis.nupts, 1, spts.shape[1]), dtype=spts.dtype)
    return float(_physical_hex_conserved_totals(state, spts, basis)[0])


def _prepare_local_mixed_hex_transfer(
    intg, mesh, stage_mesh, stage_system, old_states, readback, repartition
):
    nvars = next(iter(old_states.values())).shape[1]
    basis = HexShape(None, intg.cfg)
    if 'hex' in old_states:
        old_local = _physical_hex_conserved_totals(
            old_states['hex'], mesh.spts['hex'], basis
        )
        old_vol_local = _physical_hex_volume(mesh.spts['hex'], basis)
    else:
        old_local = np.zeros(nvars, dtype=float)
        old_vol_local = 0.0
    if 'hex' in readback:
        new_local = _physical_hex_conserved_totals(
            readback['hex'], stage_mesh.spts['hex'], basis
        )
        new_vol_local = _physical_hex_volume(stage_mesh.spts['hex'], basis)
    else:
        new_local = np.zeros(nvars, dtype=float)
        new_vol_local = 0.0

    rr, pr = [], []
    for state in readback.values():
        rho, pressure = _eos_ranges(stage_system, state)
        rr.append(rho)
        pr.append(pressure)
    local_rho = min(r[0] for r in rr), max(r[1] for r in rr)
    local_pressure = min(p[0] for p in pr), max(p[1] for p in pr)

    if not repartition:
        for etype in ('pyr', 'tet'):
            if etype in readback:
                expected = _reorder_fixed_state(
                    mesh, stage_mesh, old_states, etype
                )
                if not np.array_equal(readback[etype], expected):
                    raise MPIAMRTransactionError(
                        f'V10K immutable {etype} state drifted'
                    )

    return (
        old_local, new_local, old_vol_local, new_vol_local, local_rho,
        local_pressure
    )


def _validate_global_mixed_hex_transfer(
    comm, old_local, new_local, old_vol_local, new_vol_local, local_rho,
    local_pressure, dtype
):
    before = np.array(old_local, copy=True)
    after = np.array(new_local, copy=True)
    comm.Allreduce(mpi.IN_PLACE, before, op=mpi.SUM)
    comm.Allreduce(mpi.IN_PLACE, after, op=mpi.SUM)
    cons_error = after - before
    old_volume = scal_coll(comm.Allreduce, old_vol_local, op=mpi.SUM)
    proposed_volume = scal_coll(comm.Allreduce, new_vol_local, op=mpi.SUM)
    rho_range = (
        scal_coll(comm.Allreduce, local_rho[0], op=mpi.MIN),
        scal_coll(comm.Allreduce, local_rho[1], op=mpi.MAX),
    )
    pressure_range = (
        scal_coll(comm.Allreduce, local_pressure[0], op=mpi.MIN),
        scal_coll(comm.Allreduce, local_pressure[1], op=mpi.MAX),
    )

    tol = 8192*np.finfo(dtype).eps
    scale = np.maximum(1.0, np.abs(before))
    gate_error = None
    if np.any(np.abs(cons_error) > tol*scale):
        gate_error = MPIAMRTransactionError(
            'V10K global Hex conservation gate failed'
        )
    elif not np.isclose(
        old_volume, proposed_volume, rtol=0,
        atol=tol*max(1.0, old_volume)
    ):
        gate_error = MPIAMRTransactionError(
            'V10K global Hex volume gate failed'
        )
    _collective_error(comm, 'V10K global conservation/EOS', gate_error)

    return (
        before, after, cons_error, old_volume, proposed_volume, rho_range,
        pressure_range
    )



def _validate_staged_mixed_hex_rhs(
    comm, intg, stage_system, transferred, bank
):
    stage_system.preproc(intg.tcurr, bank)
    stage_system.backend.wait()
    preproc_error = None
    try:
        after_preproc = _local_mixed_state(
            stage_system, bank, 'post-preproc'
        )
        for etype in transferred:
            if not np.array_equal(after_preproc[etype], transferred[etype]):
                raise MPIAMRTransactionError(
                    f'V10K staged {etype} bank changed in preproc'
                )
    except Exception as exc:
        preproc_error = exc
    _collective_error(comm, 'V10K staged preprocessing', preproc_error)

    scratch_bank = _scratch_bank(stage_system, bank)
    rhs, bank_drift = _scratch_rhs(
        stage_system, intg.tcurr, bank, scratch_bank
    )
    rhs_error = None
    try:
        after_rhs = _local_mixed_state(stage_system, bank, 'post-RHS')
        for etype in transferred:
            if not np.array_equal(after_rhs[etype], transferred[etype]):
                raise MPIAMRTransactionError(
                    f'V10K staged {etype} bank changed after first RHS'
                )
        if not all(np.isfinite(a).all() for a in rhs):
            raise MPIAMRTransactionError('V10K first staged RHS is not finite')
    except Exception as exc:
        rhs_error = exc
    _collective_error(comm, 'V10K first staged RHS', rhs_error)

    return rhs, bank_drift, scratch_bank


def _validate_mixed_hex_precommit(
    comm, stage_mesh, stage_system, stage_mortar_expected, stage_uuid,
    new_epoch, proposed_tree, repartition
):
    local_mortars = sum(len(m) for m in stage_mesh.mcon.values())
    mortar_count = scal_coll(comm.Allreduce, local_mortars, op=mpi.SUM)
    local_mpi_faces = sum(len(c) for c in stage_mesh.con_p.values())
    mpi_incidence = scal_coll(
        comm.Allreduce, local_mpi_faces, op=mpi.SUM
    )

    final_error = None
    if mortar_count != stage_mortar_expected:
        final_error = MPIAMRTransactionError(
            'V10K distributed mortar count differs from materialization'
        )
    elif mpi_incidence % 2 or mpi_incidence == 0:
        final_error = MPIAMRTransactionError(
            'V10K staged MPI connectivity is missing or asymmetric'
        )
    _collective_error(comm, 'V10K pre-COMMIT validation', final_error)

    commit_signature = (
        stage_uuid, new_epoch, tuple(proposed_tree.leaves()), bool(repartition)
    )
    commit_signatures = comm.allgather(commit_signature)
    commit_error = None
    if len(set(commit_signatures)) != 1:
        commit_error = MPIAMRTransactionError(
            'V10K ranks disagree on the proposed COMMIT identity'
        )
    _collective_error(comm, 'V10K COMMIT agreement', commit_error)

    staged_gndofs = scal_coll(
        comm.Allreduce, sum(stage_system.ele_ndofs), op=mpi.SUM
    )
    return mortar_count, mpi_incidence // 2, staged_gndofs

def _validate_mpi_mixed_hex_integrator(intg):
    comm, _, _ = get_comm_rank_root()
    if comm.size not in {2, 4}:
        raise MPIAMRTransactionError(
            'V10K mixed Hex MPI AMR requires two or four MPI ranks'
        )
    if getattr(intg, 'formulation', None) != 'explicit':
        raise MPIAMRTransactionError('V10K K2 requires explicit formulation')
    if getattr(intg, 'controller_name', None) != 'none':
        raise MPIAMRTransactionError('V10K K2 requires controller = none')
    if getattr(intg, 'stepper_name', None) != 'rk4':
        raise MPIAMRTransactionError('V10K K2 requires RK4')
    if getattr(getattr(intg, 'backend', None), 'name', None) != 'openmp':
        raise MPIAMRTransactionError('V10K K2 requires OpenMP')
    if (getattr(intg, 'nacptsteps', 0) < 1 or
            getattr(intg, 'stepinfo', None) != []):
        raise MPIAMRTransactionError(
            'V10K K2 requires a completed accepted-step safe point'
        )

    cfg = intg.cfg
    solver_system = cfg.get('solver', 'system')
    if solver_system not in {'euler', 'navier-stokes'}:
        raise MPIAMRTransactionError('V10K K2 supports Euler/NS only')
    if (solver_system == 'navier-stokes' and
            cfg.get('solver', 'viscosity-correction', 'none') != 'none'):
        raise MPIAMRTransactionError(
            'V10K K2 Navier-Stokes requires constant viscosity'
        )
    if cfg.get('solver', 'shock-capturing', 'none') != 'none':
        raise MPIAMRTransactionError('V10K K2 requires no shock capturing')
    if (cfg.get('solver-interfaces', 'mortar-implementation', 'fused') !=
            'staged'):
        raise MPIAMRTransactionError('V10K K2 requires staged mortars')
    if any(s.startswith('solver-order-') for s in cfg.sections()):
        raise MPIAMRTransactionError('V10K K2 does not support mixed-p')
    if getattr(intg, 'plugins', ()) or getattr(intg, 'triggers', None):
        raise MPIAMRTransactionError('V10K K2 excludes plugins/triggers')
    if getattr(getattr(intg, 'serialiser', None), '_serialfns', {}):
        raise MPIAMRTransactionError(
            'V10K K2 excludes serialised mutable runtime state'
        )

    system = getattr(intg, 'system', None)
    mesh = getattr(system, 'mesh', None)
    if (system is None or mesh is None or system.ndims != 3 or
            getattr(system, 'name', None) != solver_system):
        raise MPIAMRTransactionError('V10K K2 live system mismatch')
    if set(mesh.etypes) != {'hex', 'pyr', 'tet'}:
        raise MPIAMRTransactionError(
            'V10K K2 requires mixed Tet+Pyramid+Hex topology'
        )
    if not mesh.con_p:
        raise MPIAMRTransactionError(
            'V10K K2 requires real cross-rank conforming connectivity'
        )
    for mcon in mesh.mcon.values():
        if (mcon.format != 'one-to-many-v1' or
                mcon.template != 'quad-2x2' or mcon.nright != 4):
            raise MPIAMRTransactionError(
                'V10K K2 live mesh contains an unsupported mortar'
            )

    root_mesh = getattr(intg, '_amr_root_mesh', None)
    if mesh.amr_tree is None:
        if mesh.mcon:
            raise MPIAMRTransactionError(
                'V10K K2 root mesh must not contain mortars'
            )
        root_mesh = mesh
    elif root_mesh is None:
        raise MPIAMRTransactionError(
            'V10K adapted live mesh requires its immutable root anchor'
        )
    if root_mesh.uuid != (mesh.amr_tree.root_mesh_uuid
                          if mesh.amr_tree is not None else mesh.uuid):
        raise MPIAMRTransactionError('V10K immutable root UUID mismatch')

    return comm, system, mesh, root_mesh


def _local_mixed_state(system, bank, label):
    if bank < 0 or bank >= getattr(system, 'nrhs', 0):
        raise MPIAMRTransactionError(f'V10K {label} bank is invalid')
    if not system.ele_types or not set(system.ele_types) <= {
        'hex', 'pyr', 'tet'
    }:
        raise MPIAMRTransactionError(
            f'V10K {label} local element families are invalid'
        )

    copied = dict(zip(system.ele_types, _copy_state(system, bank)))
    if set(copied) != set(system.ele_types):
        raise MPIAMRTransactionError(
            f'V10K {label} local bank readback is incomplete'
        )
    for etype, state in copied.items():
        shape = tuple(system.ele_shapes[etype])
        if tuple(state.shape) != shape:
            raise MPIAMRTransactionError(
                f'V10K {label} {etype} bank shape mismatch'
            )
    return copied


def _reorder_fixed_state(mesh, stage_mesh, old_states, etype):
    old_eidxs = tuple(map(int, mesh.eidxs.get(etype, ())))
    new_eidxs = tuple(map(int, stage_mesh.eidxs.get(etype, ())))
    if set(old_eidxs) != set(new_eidxs):
        raise MPIAMRTransactionError(
            f'V10K immutable {etype} ownership changed during K2'
        )
    if not old_eidxs:
        return None
    if etype not in old_states:
        raise MPIAMRTransactionError(
            f'V10K immutable {etype} state is missing locally'
        )
    pos = {eidx: i for i, eidx in enumerate(old_eidxs)}
    return np.stack(
        [old_states[etype][:, :, pos[eidx]] for eidx in new_eidxs], axis=2
    )


def _stage_local_mixed_transfer(system, mesh, stage_mesh, old_tree,
                                proposed_tree, local_by_leaf, old_states):
    schema = None
    if 'hex' in old_states:
        schema = (old_states['hex'].dtype.str, old_states['hex'].shape[0],
                  old_states['hex'].shape[1])
    comm, _, _ = get_comm_rank_root()
    schemas = [s for s in comm.allgather(schema) if s is not None]
    if not schemas or len(set(schemas)) != 1:
        raise MPIAMRTransactionError(
            f'V10K ranks disagree on Hex state schema: {schemas!r}'
        )
    dtype, nupts, nvars = schemas[0]

    if local_by_leaf:
        ords, records = _produce_local_refined_records(
            system, old_tree, proposed_tree, old_states['hex'], local_by_leaf
        )
    else:
        ords = np.empty(0, dtype=np.int64)
        records = np.empty((0, nupts, nvars), dtype=np.dtype(dtype))

    local_hex = tuple(map(int, stage_mesh.eidxs.get('hex', ())))
    if (len(ords) != len(local_hex) or len(np.unique(ords)) != len(ords) or
            set(map(int, ords)) != set(local_hex)):
        raise MPIAMRTransactionError(
            'V10K inherited Hex ownership differs from local transfer'
        )

    transferred = {}
    if local_hex:
        pos = {int(eidx): i for i, eidx in enumerate(ords)}
        transferred['hex'] = np.stack(
            [records[pos[eidx]] for eidx in local_hex], axis=2
        )
    for etype in ('pyr', 'tet'):
        state = _reorder_fixed_state(mesh, stage_mesh, old_states, etype)
        if state is not None:
            transferred[etype] = state

    if set(transferred) != set(stage_mesh.eidxs):
        raise MPIAMRTransactionError(
            'V10K transferred families differ from staged ownership'
        )
    return transferred


def _exchange_typed_records(comm, etype, eidxs, records, owner,
                            expected_eidxs):
    eidxs = np.asarray(eidxs, dtype=np.int64)
    expected_eidxs = np.asarray(expected_eidxs, dtype=np.int64)
    owner = np.asarray(owner, dtype=np.int32)

    if eidxs.ndim != 1 or len(np.unique(eidxs)) != len(eidxs):
        raise MPIAMRTransactionError(
            f'V10K migrated {etype} ids are invalid'
        )
    if np.any((eidxs < 0) | (eidxs >= len(owner))):
        raise MPIAMRTransactionError(
            f'V10K migrated {etype} id is outside proposed ownership'
        )

    schema = None
    if records is not None:
        records = np.asarray(records)
        if records.ndim != 3 or len(records) != len(eidxs):
            raise MPIAMRTransactionError(
                f'V10K migrated {etype} record layout is invalid'
            )
        schema = (records.dtype.str, records.shape[1], records.shape[2])

    schemas = [s for s in comm.allgather(schema) if s is not None]
    if not schemas or len(set(schemas)) != 1:
        raise MPIAMRTransactionError(
            f'V10K ranks disagree on {etype} migration schema: {schemas!r}'
        )
    dtype, nupts, nvars = schemas[0]
    if records is None:
        records = np.empty(
            (0, nupts, nvars), dtype=np.dtype(dtype)
        )

    dest = owner[eidxs] if len(eidxs) else np.empty(0, dtype=np.int32)
    if np.any((dest < 0) | (dest >= comm.size)):
        raise MPIAMRTransactionError(
            f'V10K {etype} migration has an invalid destination rank'
        )
    order = np.lexsort((eidxs, dest)) if len(eidxs) else np.empty(0, int)
    seidxs = np.ascontiguousarray(eidxs[order])
    srecords = np.ascontiguousarray(records[order])
    dest = dest[order]
    scount = np.bincount(dest, minlength=comm.size).astype(np.int64)

    ex = _LeafStateExchanger()
    reidxs, (rcount, rdisp) = ex._alltoallcv(comm, seidxs, scount)
    rrecords, (rcount_s, rdisp_s) = ex._alltoallcv(
        comm, srecords, scount
    )
    if (not np.array_equal(rcount, rcount_s) or
            not np.array_equal(rdisp, rdisp_s)):
        raise MPIAMRTransactionError(
            f'V10K {etype} id/state migration layouts disagree'
        )

    if (len(reidxs) != len(expected_eidxs) or
            len(np.unique(reidxs)) != len(reidxs) or
            set(map(int, reidxs)) != set(map(int, expected_eidxs))):
        raise MPIAMRTransactionError(
            f'V10K migrated {etype} ids differ from staged ownership'
        )

    local = None
    if len(expected_eidxs):
        pos = {int(eidx): i for i, eidx in enumerate(reidxs)}
        local = np.stack(
            [rrecords[pos[int(eidx)]] for eidx in expected_eidxs], axis=2
        )
    return local, tuple(map(int, scount)), tuple(map(int, rcount))


def _stage_local_mixed_migration(system, mesh, stage_mesh, old_tree,
                                 proposed_tree, local_by_leaf, old_states,
                                 owner):
    transferred, sends, recvs = {}, {}, {}

    if 'hex' in old_states:
        heidxs, hrecords = _produce_local_refined_records(
            system, old_tree, proposed_tree, old_states['hex'], local_by_leaf
        )
    else:
        heidxs, hrecords = np.empty(0, np.int64), None
    local, scount, rcount = _exchange_typed_records(
        get_comm_rank_root()[0], 'hex', heidxs, hrecords, owner['hex'],
        stage_mesh.eidxs.get('hex', ())
    )
    sends['hex'], recvs['hex'] = scount, rcount
    if local is not None:
        transferred['hex'] = local

    comm, _, _ = get_comm_rank_root()
    for etype in ('pyr', 'tet'):
        eidxs = np.asarray(mesh.eidxs.get(etype, ()), dtype=np.int64)
        records = None
        if etype in old_states:
            records = np.moveaxis(old_states[etype], 2, 0)
        local, scount, rcount = _exchange_typed_records(
            comm, etype, eidxs, records, owner[etype],
            stage_mesh.eidxs.get(etype, ())
        )
        sends[etype], recvs[etype] = scount, rcount
        if local is not None:
            transferred[etype] = local

    if set(transferred) != set(stage_mesh.eidxs):
        raise MPIAMRTransactionError(
            'V10K migrated families differ from staged ownership'
        )
    return transferred, sends, recvs


def _inject_local_mixed_state(stage_system, states, bank):
    if set(stage_system.ele_types) != set(states):
        raise MPIAMRTransactionError(
            'V10K staged system families differ from transferred state'
        )
    for etype, banks in zip(stage_system.ele_types, stage_system.ele_banks):
        state = states[etype]
        shape = tuple(stage_system.ele_shapes[etype])
        if tuple(state.shape) != shape:
            raise MPIAMRTransactionError(
                f'V10K staged {etype} transfer shape mismatch'
            )
        banks[bank].set(np.array(state, copy=True, order='C'))
    stage_system.backend.wait()

    readback = _local_mixed_state(stage_system, bank, 'staged')
    for etype, state in states.items():
        if not np.array_equal(readback[etype], state):
            raise MPIAMRTransactionError(
                f'V10K staged {etype} bank differs from transferred state'
            )
    return readback


def _prepare_stage_root(root_fname, root_uuid, old_tree, marks, parts, size,
                        shared_stage_dir, action, old_parts=None,
                        ownership_policy=None):
    root_mesh = _serial_root_mesh(root_fname, root_uuid)
    if action == 'refine':
        proposed_tree = _close_refinements(root_mesh, old_tree, marks)
    else:
        proposed_tree = _close_coarsening(root_mesh, old_tree, marks)
    if parts is None:
        raw = materialize_native_hex_tree(root_mesh, proposed_tree)
        if ownership_policy != 'balanced-affinity-v1':
            raise MPIAMRTransactionError(
                f'D7D unsupported ownership policy {ownership_policy!r}'
            )
        parts = balanced_affinity_destination_parts(
            old_tree, proposed_tree, raw, old_parts, size
        )
        parts, vparts = _normalise_destination_parts(
            parts, proposed_tree, size
        )
    else:
        parts, vparts = _normalise_destination_parts(
            parts, proposed_tree, size
        )
        raw = materialize_native_hex_tree(root_mesh, proposed_tree)
    _validate_mortar_affinity(raw, vparts)

    os.makedirs(shared_stage_dir, exist_ok=True)
    stage_path = None
    try:
        fd, stage_path = tempfile.mkstemp(
            prefix='pyfr-d7a-', suffix='.pyfrm', dir=shared_stage_dir
        )
        os.close(fd)
        os.unlink(stage_path)
        write_adapted_mesh(raw, stage_path)
        pname = f'amr-d7a-{size}'
        with h5py.File(stage_path, 'r+') as f:
            con, ecurved, _, edisps, _ = (
                BasePartitioner.construct_global_con(f)
            )
            pinfo = BasePartitioner.construct_partitioning(
                f, ecurved, edisps, con, vparts
            )
            write_partitioning(f, pname, pinfo)
            stage_uuid = f['mesh-uuid'][()].decode()
    except Exception:
        if stage_path:
            try:
                os.unlink(stage_path)
            except FileNotFoundError:
                pass
        raise

    return proposed_tree, vparts, stage_path, pname, stage_uuid, raw


def _remove_stage(comm, path):
    comm.barrier()
    if comm.rank == 0:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    comm.barrier()


def perform_one_mpi_mixed_hex_amr_transaction(
    intg, local_scripted_marks, *, shared_stage_dir, repartition=False
):
    comm, _, _ = get_comm_rank_root()
    try:
        comm, system, mesh, root_mesh = _validate_mpi_mixed_hex_integrator(
            intg
        )
    except Exception as exc:
        _collective_error(comm, 'V10K integrator validation', exc)
        raise AssertionError('unreachable')
    _collective_error(comm, 'V10K integrator validation')

    repartitions = comm.allgather(bool(repartition))
    if len(set(repartitions)) != 1:
        _collective_error(
            comm, 'V10K repartition agreement',
            MPIAMRTransactionError(
                f'V10K ranks disagree on repartition mode: {repartitions!r}'
            )
        )
    repartition = repartitions[0]

    try:
        old_tree, local_by_leaf = _distributed_mixed_current_tree(mesh, comm)
        marks = tuple(_normalise_mark(m) for m in local_scripted_marks)
        if not set(marks) <= set(local_by_leaf):
            raise MPIAMRTransactionError(
                'V10K local marks must be owned by the calling rank'
            )
        old_parts = _assemble_typed_ownership(
            mesh, old_tree, local_by_leaf, comm
        )
    except Exception as exc:
        _collective_error(comm, 'V10K initial topology', exc)
        raise AssertionError('unreachable')
    _collective_error(comm, 'V10K initial topology')

    gathered_marks = comm.allgather(marks)
    flat_marks = [m for rank_marks in gathered_marks for m in rank_marks]
    mark_error = None
    if not flat_marks:
        mark_error = MPIAMRTransactionError(
            'V10K refinement requires at least one global mark'
        )
    elif len(flat_marks) != len(set(flat_marks)):
        mark_error = MPIAMRTransactionError(
            'V10K distributed marks must be globally unique'
        )
    _collective_error(comm, 'V10K mark agreement', mark_error)
    global_marks = tuple(sorted(flat_marks, key=repr))

    old_epoch = getattr(intg, '_amr_mpi_epoch', 0)
    signature = (
        mesh.uuid, old_tree.root_mesh_uuid, tuple(old_tree.leaves()),
        int(old_epoch), float(intg.tcurr), int(intg.idxcurr),
    )
    signatures = comm.allgather(signature)
    if len(set(signatures)) != 1:
        _collective_error(
            comm, 'V10K epoch/topology agreement',
            MPIAMRTransactionError(
                f'V10K ranks disagree on topology/epoch: {signatures!r}'
            )
        )
    old_epoch = int(old_epoch)
    new_epoch = old_epoch + 1

    root_fnames = comm.allgather(os.path.abspath(root_mesh.fname))
    if len(set(root_fnames)) != 1:
        _collective_error(
            comm, 'V10K root path agreement',
            MPIAMRTransactionError('V10K ranks disagree on root mesh path')
        )
    stage_dirs = comm.allgather(os.path.abspath(shared_stage_dir))
    if len(set(stage_dirs)) != 1:
        _collective_error(
            comm, 'V10K stage path agreement',
            MPIAMRTransactionError('V10K ranks disagree on stage directory')
        )

    error = None
    if comm.rank == 0:
        try:
            prepared = _prepare_mixed_stage_root(
                root_fnames[0], old_tree.root_mesh_uuid, old_tree,
                global_marks, old_parts, comm.size, stage_dirs[0],
                repartition=repartition
            )
            (proposed_tree, closure, owner, stage_path, pname, stage_uuid,
             raw) = prepared
            meta = (
                proposed_tree, closure, owner, stage_path, pname, stage_uuid,
                len(raw.leaf_order), len(raw.fixed_nodes['pyr']),
                len(raw.fixed_nodes['tet']), len(raw.mortars),
            )
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            meta = None
    else:
        meta = None
    error = comm.bcast(error, root=0)
    if error:
        raise MPIAMRTransactionError(
            f'V10K stage preparation failed: {error}'
        )
    meta = comm.bcast(meta, root=0)
    (proposed_tree, closure, owner, stage_path, pname, stage_uuid,
     stage_leaf_count, stage_pyr_count, stage_tet_count,
     stage_mortar_expected) = meta

    old_stage_reader = getattr(intg, '_amr_mpi_stage_reader', None)
    old_stage_path = getattr(intg, '_amr_mpi_stage_path', None)
    reader = None
    stage_system = None
    committed = False
    try:
        visibility_error = None
        try:
            with h5py.File(stage_path, 'r') as f:
                if f['mesh-uuid'][()].decode() != stage_uuid:
                    raise MPIAMRTransactionError(
                        'V10K staged mesh UUID changed before read'
                    )
                if f'partitionings/{pname}' not in f:
                    raise MPIAMRTransactionError(
                        'V10K staged partitioning is not visible'
                    )
        except Exception as exc:
            visibility_error = exc
        _collective_error(comm, 'V10K shared stage visibility',
                          visibility_error)

        reader = NativeReader(stage_path, pname=pname)
        stage_mesh = reader.mesh
        identity_error = None
        try:
            if (stage_mesh.uuid != stage_uuid or
                    stage_mesh.amr_tree.root_mesh_uuid !=
                    proposed_tree.root_mesh_uuid or
                    tuple(stage_mesh.amr_tree.leaves()) !=
                    tuple(proposed_tree.leaves())):
                raise MPIAMRTransactionError(
                    'V10K staged reader lost mesh/tree identity'
                )
            local_owner = {
                etype: tuple(map(int, stage_mesh.eidxs.get(etype, ())))
                for etype in ('hex', 'pyr', 'tet')
            }
            expected = {
                etype: tuple(
                    i for i, rank in enumerate(owner[etype])
                    if int(rank) == comm.rank
                )
                for etype in ('hex', 'pyr', 'tet')
            }
            if any(set(local_owner[e]) != set(expected[e]) for e in expected):
                raise MPIAMRTransactionError(
                    'V10K NativeReader ownership differs from proposal'
                )
        except Exception as exc:
            identity_error = exc
        _collective_error(comm, 'V10K staged NativeReader', identity_error)

        bank = intg.idxcurr
        bank_error = None
        try:
            old_states = _local_mixed_state(system, bank, 'accepted')
            old_snapshot = {
                etype: np.array(state, copy=True)
                for etype, state in old_states.items()
            }
            for etype, state in old_states.items():
                if state.shape[2] != len(mesh.eidxs[etype]):
                    raise MPIAMRTransactionError(
                        f'V10K accepted {etype} bank/ownership mismatch'
                    )
        except Exception as exc:
            bank_error = exc
        _collective_error(comm, 'V10K accepted bank', bank_error)

        transfer_error = None
        migration_sends = migration_recvs = {}
        try:
            if repartition:
                transferred, migration_sends, migration_recvs = (
                    _stage_local_mixed_migration(
                        system, mesh, stage_mesh, old_tree, proposed_tree,
                        local_by_leaf, old_states, owner
                    )
                )
            else:
                transferred = _stage_local_mixed_transfer(
                    system, mesh, stage_mesh, old_tree, proposed_tree,
                    local_by_leaf, old_states
                )
        except Exception as exc:
            transfer_error = exc
        _collective_error(comm, 'V10K conservative transfer', transfer_error)

        staged = Solution(
            config=intg.cfg, stats=None, fields=None,
            data={e: np.array(v, copy=True) for e, v in transferred.items()},
            state={},
        )
        stage_serialiser = Serialiser()
        stage_system = type(system)(
            intg.backend, stage_mesh, staged, intg._registers,
            intg.cfg, stage_serialiser, needs_cfl=False
        )
        stage_system.commit()

        build_error = None
        try:
            readback = _inject_local_mixed_state(
                stage_system, transferred, bank
            )
        except Exception as exc:
            build_error = exc
        _collective_error(comm, 'V10K staged system build', build_error)

        local_error = None
        try:
            (
                old_local, new_local, old_vol_local, new_vol_local,
                local_rho, local_pressure
            ) = _prepare_local_mixed_hex_transfer(
                intg, mesh, stage_mesh, stage_system, old_states, readback,
                repartition
            )
        except Exception as exc:
            local_error = exc
        _collective_error(comm, 'V10K local physics gates', local_error)

        (
            before, after, cons_error, old_volume, proposed_volume,
            rho_range, pressure_range
        ) = _validate_global_mixed_hex_transfer(
            comm, old_local, new_local, old_vol_local, new_vol_local,
            local_rho, local_pressure, next(iter(old_states.values())).dtype
        )

        rhs, bank_drift, scratch_bank = _validate_staged_mixed_hex_rhs(
            comm, intg, stage_system, transferred, bank
        )

        old_error = None
        try:
            current_old = _local_mixed_state(system, bank, 'old-live')
            for etype, state in old_snapshot.items():
                if not np.array_equal(current_old[etype], state):
                    raise MPIAMRTransactionError(
                        f'V10K old live {etype} state changed before COMMIT'
                    )
        except Exception as exc:
            old_error = exc
        _collective_error(comm, 'V10K old-system rollback guard', old_error)

        mortar_count, mpi_faces, staged_gndofs = _validate_mixed_hex_precommit(
            comm, stage_mesh, stage_system, stage_mortar_expected, stage_uuid,
            new_epoch, proposed_tree, repartition
        )

        comm.barrier()
        clear_memoize(intg)
        old_system_id = id(system)
        intg.system = stage_system
        intg._amr_root_mesh = root_mesh
        intg.mesh_uuid = stage_mesh.uuid
        intg.serialiser = stage_serialiser
        intg.gndofs = staged_gndofs
        intg._invalidate_caches()
        intg._amr_mpi_stage_reader = reader
        intg._amr_mpi_stage_path = stage_path
        intg._amr_mpi_epoch = new_epoch
        committed = True
        comm.barrier()

        if old_stage_reader is not None:
            try:
                old_stage_reader.close()
            except Exception:
                pass
        comm.barrier()
        if (comm.rank == 0 and old_stage_path and
                old_stage_path != stage_path):
            try:
                os.unlink(old_stage_path)
            except FileNotFoundError:
                pass
        comm.barrier()

        return MPIMixedHexAMRTransactionResult(
            old_tree=old_tree, proposed_tree=proposed_tree,
            global_marks=global_marks,
            local_old_eidxs=tuple(
                (e, tuple(map(int, mesh.eidxs.get(e, ()))))
                for e in ('hex', 'pyr', 'tet')
            ),
            local_stage_eidxs=tuple(
                (e, tuple(map(int, stage_mesh.eidxs.get(e, ()))))
                for e in ('hex', 'pyr', 'tet')
            ),
            conservation_before=before, conservation_after=after,
            conservation_error=cons_error, old_volume=float(old_volume),
            proposed_volume=float(proposed_volume), rho_range=rho_range,
            pressure_range=pressure_range, stage_mesh_uuid=stage_uuid,
            stage_path=stage_path, stage_leaf_count=stage_leaf_count,
            stage_pyr_count=stage_pyr_count, stage_tet_count=stage_tet_count,
            stage_mortar_count=int(mortar_count),
            stage_mpi_face_count=int(mpi_faces), bank_drift=bank_drift,
            accepted_bank=bank, scratch_bank=scratch_bank,
            old_epoch=old_epoch, new_epoch=new_epoch,
            old_system_id=old_system_id, new_system_id=id(stage_system),
            tcurr=float(intg.tcurr),
            migration_send_counts=tuple(
                (e, migration_sends.get(e, ()))
                for e in ('hex', 'pyr', 'tet')
            ),
            migration_recv_counts=tuple(
                (e, migration_recvs.get(e, ()))
                for e in ('hex', 'pyr', 'tet')
            ),
            repartitioned=bool(repartition),
        )
    except Exception as exc:
        if isinstance(exc, _CollectiveFailure):
            failure = exc
        else:
            try:
                _collective_error(comm, 'V10K transaction', exc)
            except _CollectiveFailure as cexc:
                failure = cexc

        if not committed:
            if stage_system is not None:
                try:
                    stage_system.close()
                except Exception:
                    pass
            if reader is not None:
                reader.close()
            _remove_stage(comm, stage_path)
        raise failure


def perform_one_mpi_amr_transaction(
    intg, local_scripted_marks, destination_parts=None, *, shared_stage_dir,
    action='refine', ownership_policy=None
):
    comm, _, _ = get_comm_rank_root()
    try:
        comm, system, mesh, root_mesh = _validate_mpi_integrator(intg)
    except Exception as exc:
        _collective_error(comm, 'integrator validation', exc)
        raise AssertionError('unreachable')
    _collective_error(comm, 'integrator validation')

    actions = comm.allgather(action)
    if len(set(actions)) != 1 or actions[0] not in {'refine', 'coarsen'}:
        _collective_error(
            comm, 'action agreement',
            MPIAMRTransactionError(
                f'D7C ranks disagree on transaction action: {actions!r}'
            )
        )
    action = actions[0]

    try:
        old_tree, local_by_leaf = _distributed_current_tree(mesh, comm)
        marks = tuple(_normalise_mark(m) for m in local_scripted_marks)
        if not set(marks) <= set(local_by_leaf):
            raise MPIAMRTransactionError(
                'D7A local marks must be owned by the calling rank'
            )
    except Exception as exc:
        _collective_error(comm, 'initial validation', exc)
        raise AssertionError('unreachable')
    _collective_error(comm, 'initial validation')

    gathered_marks = comm.allgather(marks)
    flat_marks = [m for ms in gathered_marks for m in ms]
    mark_error = None
    if len(flat_marks) != len(set(flat_marks)):
        mark_error = MPIAMRTransactionError(
            'D7 distributed marks must be globally unique'
        )
    elif action == 'refine' and not flat_marks:
        mark_error = MPIAMRTransactionError(
            'D7 refinement requires at least one global mark'
        )
    elif action == 'coarsen' and (not flat_marks or
                                   len(flat_marks) % 8):
        mark_error = MPIAMRTransactionError(
            'D9J coarsening requires a positive multiple of eight marks'
        )
    _collective_error(comm, 'mark agreement', mark_error)
    global_marks = tuple(sorted(flat_marks, key=repr))

    policy_mode = destination_parts is None
    policy_modes = comm.allgather(policy_mode)
    if len(set(policy_modes)) != 1:
        _collective_error(
            comm, 'ownership mode agreement',
            MPIAMRTransactionError(
                'D7D ranks disagree on explicit versus policy ownership'
            )
        )
    policy_error = None
    if ownership_policy is not None and not isinstance(ownership_policy, str):
        policy_error = MPIAMRTransactionError(
            'D7D ownership policy must be a string or None'
        )
    _collective_error(comm, 'ownership policy validation', policy_error)

    policies = comm.allgather(ownership_policy)
    if len(set(policies)) != 1:
        _collective_error(
            comm, 'ownership policy agreement',
            MPIAMRTransactionError(
                f'D7D ranks disagree on ownership policy: {policies!r}'
            )
        )
    ownership_policy = policies[0]
    if policy_mode and ownership_policy is None:
        ownership_policy = 'balanced-affinity-v1'
    if not policy_mode and ownership_policy is not None:
        _collective_error(
            comm, 'ownership policy agreement',
            MPIAMRTransactionError(
                'D7D explicit destination ownership cannot also name a policy'
            )
        )

    owned_by_rank = comm.allgather(tuple(local_by_leaf))
    old_parts = {
        leaf: r for r, leaves in enumerate(owned_by_rank) for leaf in leaves
    }
    if set(old_parts) != set(old_tree.leaves()):
        _collective_error(
            comm, 'current ownership assembly',
            MPIAMRTransactionError(
                'D7D current ownership does not cover the old tree exactly'
            )
        )

    # The immutable root file path and ownership mode must agree on every
    # rank before root-only materialisation begins.
    root_fnames = comm.allgather(os.path.abspath(root_mesh.fname))
    if len(set(root_fnames)) != 1:
        _collective_error(
            comm, 'root path agreement',
            MPIAMRTransactionError('D7A ranks disagree on immutable root path')
        )
    stage_dirs = comm.allgather(os.path.abspath(shared_stage_dir))
    if len(set(stage_dirs)) != 1:
        _collective_error(
            comm, 'stage path agreement',
            MPIAMRTransactionError('D7A ranks disagree on shared stage path')
        )

    # Normalise the destination map locally against a provisional broadcast
    # tree signature only after rank 0 has performed global closure.
    prepared = None
    error = None
    if comm.rank == 0:
        try:
            prepared = _prepare_stage_root(
                root_fnames[0], old_tree.root_mesh_uuid, old_tree,
                global_marks, destination_parts, comm.size, stage_dirs[0],
                action, old_parts, ownership_policy
            )
            (proposed_tree, vparts, stage_path, pname, stage_uuid, raw) = (
                prepared
            )
            meta = (proposed_tree, vparts, stage_path, pname, stage_uuid,
                    len(raw.leaf_order), len(raw.mortars),
                    tuple((m.format, m.template) for m in raw.mortars))
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            meta = None
    else:
        meta = None
    error = comm.bcast(error, root=0)
    if error:
        raise MPIAMRTransactionError(f'D7A stage preparation failed: {error}')
    meta = comm.bcast(meta, root=0)
    (proposed_tree, vparts, stage_path, pname, stage_uuid,
     stage_leaf_count, stage_mortar_expected, mortar_formats) = meta

    # Explicit maps must be identical on every rank.  Policy ownership is
    # computed once from globally assembled old ownership and the canonical
    # proposed tree, then broadcast with the staged partition metadata.
    dest_error = None
    if not policy_mode:
        try:
            _, local_vparts = _normalise_destination_parts(
                destination_parts, proposed_tree, comm.size
            )
            if not np.array_equal(local_vparts, vparts):
                raise MPIAMRTransactionError(
                    'D7A destination maps differ by rank'
                )
        except Exception as exc:
            dest_error = exc
    try:
        _collective_error(comm, 'destination agreement', dest_error)
    except _CollectiveFailure:
        _remove_stage(comm, stage_path)
        raise

    old_stage_reader = getattr(intg, '_amr_mpi_stage_reader', None)
    old_stage_path = getattr(intg, '_amr_mpi_stage_path', None)
    reader = None
    stage_system = None
    committed = False
    try:
        # Verify the shared stage file is locally visible on every rank before
        # any rank enters NativeReader's collectives.  A node-local staging
        # path must fail closed rather than strand peers inside HDF5/MPI setup.
        visibility_error = None
        try:
            with h5py.File(stage_path, 'r') as f:
                if f['mesh-uuid'][()].decode() != stage_uuid:
                    raise MPIAMRTransactionError(
                        'D7A staged mesh UUID changed before distributed read'
                    )
                if f'partitionings/{pname}' not in f:
                    raise MPIAMRTransactionError(
                        'D7A staged partitioning is not visible on this rank'
                    )
        except Exception as exc:
            visibility_error = exc
        _collective_error(comm, 'shared stage visibility', visibility_error)

        # Real NativeReader partition construction is collective.
        reader = NativeReader(stage_path, pname=pname)
        stage_mesh = reader.mesh
        if (stage_mesh.uuid != stage_uuid or
                stage_mesh.amr_tree.root_mesh_uuid !=
                proposed_tree.root_mesh_uuid or
                tuple(stage_mesh.amr_tree.leaves()) !=
                tuple(proposed_tree.leaves())):
            raise MPIAMRTransactionError(
                'D7A staged reader lost mesh/tree identity'
            )
        _collective_error(comm, 'staged NativeReader')

        bank = intg.idxcurr
        bank_error = None
        try:
            old_shape, _ = _single_hex_bank(system, bank, 'accepted')
            old_parts = _copy_state(system, bank)
            if (len(old_parts) != 1 or
                    tuple(old_parts[0].shape) != old_shape):
                raise MPIAMRTransactionError(
                    'D7A accepted local bank is invalid'
                )
            old_state = old_parts[0]
            old_state_snapshot = np.array(old_state, copy=True)
            if old_shape[2] != len(local_by_leaf):
                raise MPIAMRTransactionError(
                    'D7A live local bank does not match local leaf ownership'
                )
        except Exception as exc:
            bank_error = exc
        _collective_error(comm, 'accepted local bank', bank_error)

        transfer_error = None
        try:
            if action == 'refine':
                ords, states = _produce_local_refined_records(
                    system, old_tree, proposed_tree, old_state,
                    local_by_leaf
                )
            else:
                ords, states = _produce_local_coarsened_records(
                    comm, system, old_tree, proposed_tree, old_state,
                    local_by_leaf, vparts
                )
        except _CollectiveFailure:
            raise
        except Exception as exc:
            transfer_error = exc
        _collective_error(
            comm, 'distributed transfer preparation', transfer_error
        )
        produced = comm.allgather(ords)
        all_ords = np.concatenate(produced)
        if (len(all_ords) != proposed_tree.nleaves or
                not np.array_equal(np.sort(all_ords),
                                   np.arange(proposed_tree.nleaves))):
            raise MPIAMRTransactionError(
                'D7A distributed transfer does not produce every proposed '
                'leaf exactly once'
            )
        _collective_error(comm, 'distributed transfer coverage')

        rords, rstates, scount, rcount = _exchange_leaf_states(
            comm, ords, states, vparts
        )
        local_transfer, stage_eidxs = _stage_local_bank(
            stage_mesh, rords, rstates
        )
        _collective_error(comm, 'state migration')

        staged = Solution(
            config=intg.cfg, stats=None, fields=None,
            data={'hex': np.array(local_transfer, copy=True)}, state={}
        )
        stage_serialiser = Serialiser()
        stage_system = type(system)(
            intg.backend, stage_mesh, staged, intg._registers,
            intg.cfg, stage_serialiser, needs_cfl=False
        )
        stage_system.commit()
        group_shape, group_bank = _single_hex_bank(
            stage_system, bank, 'staged'
        )
        if tuple(local_transfer.shape) != group_shape:
            raise MPIAMRTransactionError(
                'D7A migrated state does not match staged local bank shape'
            )
        group_bank.set(local_transfer)
        stage_system.backend.wait()
        readback = _copy_state(stage_system, bank)[0]
        if not np.array_equal(readback, local_transfer):
            raise MPIAMRTransactionError(
                'D7A staged accepted bank differs from migrated state'
            )
        _collective_error(comm, 'staged system build')

        # Compute every rank-local physics diagnostic first, then vote before
        # entering the reductions.  A local EOS failure must not strand peers
        # inside a later collective.
        local_error = None
        try:
            basis = HexShape(None, intg.cfg)
            weights = _integration_weights(intg.cfg, basis)
            old_volumes = _mesh_volumes(mesh)
            new_volumes = _mesh_volumes(stage_mesh)
            old_local = _conserved_totals(old_state, old_volumes, weights)
            new_local = _conserved_totals(readback, new_volumes, weights)
            old_vol_local = float(np.sum(old_volumes))
            new_vol_local = float(np.sum(new_volumes))
            local_rho, local_pressure = _eos_ranges(stage_system, readback)
        except Exception as exc:
            local_error = exc
        _collective_error(comm, 'local conservation and EOS', local_error)

        before = np.array(old_local, copy=True)
        after = np.array(new_local, copy=True)
        comm.Allreduce(mpi.IN_PLACE, before, op=mpi.SUM)
        comm.Allreduce(mpi.IN_PLACE, after, op=mpi.SUM)
        error = after - before
        old_volume = scal_coll(comm.Allreduce, old_vol_local, op=mpi.SUM)
        proposed_volume = scal_coll(comm.Allreduce, new_vol_local, op=mpi.SUM)
        rho_range = (
            scal_coll(comm.Allreduce, local_rho[0], op=mpi.MIN),
            scal_coll(comm.Allreduce, local_rho[1], op=mpi.MAX),
        )
        pressure_range = (
            scal_coll(comm.Allreduce, local_pressure[0], op=mpi.MIN),
            scal_coll(comm.Allreduce, local_pressure[1], op=mpi.MAX),
        )

        tol = 8192*np.finfo(old_state.dtype).eps
        scale = np.maximum(1.0, np.abs(before))
        if np.any(np.abs(error) > tol*scale):
            raise MPIAMRTransactionError(
                'D7A global componentwise conservation gate failed'
            )
        if not np.isclose(old_volume, proposed_volume, rtol=0,
                          atol=tol*max(1.0, old_volume)):
            raise MPIAMRTransactionError(
                'D7A global physical volume gate failed'
            )
        _collective_error(comm, 'global conservation and EOS')

        stage_system.preproc(intg.tcurr, bank)
        stage_system.backend.wait()
        if not np.array_equal(_copy_state(stage_system, bank)[0], readback):
            raise MPIAMRTransactionError(
                'D7A staged accepted bank changed during preprocessing'
            )
        _collective_error(comm, 'staged preprocessing')

        scratch_bank = _scratch_bank(stage_system, bank)
        rhs, bank_drift = _scratch_rhs(
            stage_system, intg.tcurr, bank, scratch_bank
        )
        _collective_error(comm, 'first staged RHS')

        if not np.array_equal(
            _copy_state(system, bank)[0], old_state_snapshot
        ):
            raise MPIAMRTransactionError(
                'D7A old live accepted state changed before COMMIT'
            )
        local_mortars = sum(len(m) for m in stage_mesh.mcon.values())
        mortar_count = scal_coll(comm.Allreduce, local_mortars, op=mpi.SUM)
        if mortar_count != stage_mortar_expected:
            raise MPIAMRTransactionError(
                'D7A distributed mortar count differs from global '
                'materialisation'
            )
        local_mpi_faces = sum(len(c) for c in stage_mesh.con_p.values())
        mpi_face_incidence = scal_coll(
            comm.Allreduce, local_mpi_faces, op=mpi.SUM
        )
        if mpi_face_incidence % 2:
            raise MPIAMRTransactionError(
                'D7A MPI face incidence is not two-sided'
            )
        mpi_faces = mpi_face_incidence // 2
        if mpi_faces <= 0:
            raise MPIAMRTransactionError(
                'D7A initial mechanics gate requires real MPI connectivity'
            )
        staged_gndofs = scal_coll(
            comm.Allreduce, sum(stage_system.ele_ndofs), op=mpi.SUM
        )
        _collective_error(comm, 'pre-COMMIT validation')

        # All-rank COMMIT.  No rank may decide independently after this
        # barrier.
        comm.barrier()
        clear_memoize(intg)
        intg.system = stage_system
        intg._amr_root_mesh = root_mesh
        intg.mesh_uuid = stage_mesh.uuid
        intg.serialiser = stage_serialiser
        intg.gndofs = staged_gndofs
        intg._invalidate_caches()
        intg._amr_mpi_stage_reader = reader
        intg._amr_mpi_stage_path = stage_path
        committed = True
        comm.barrier()

        # A prior D7-created stage file is no longer the live mesh after the
        # all-rank ownership switch.  Close every rank's reader first, then
        # let rank 0 remove that retired shared file.  External restart files
        # are never tracked here and are therefore never deleted.
        if old_stage_reader is not None:
            old_stage_reader.close()
        comm.barrier()
        if (comm.rank == 0 and old_stage_path and
                old_stage_path != stage_path):
            try:
                os.unlink(old_stage_path)
            except FileNotFoundError:
                pass
        comm.barrier()

        local_transfer.setflags(write=False)
        return MPIAMRTransactionResult(
            old_tree=old_tree, proposed_tree=proposed_tree,
            global_marks=global_marks,
            local_old_eidxs=tuple(map(int, mesh.eidxs['hex'])),
            local_stage_eidxs=tuple(map(int, stage_eidxs)),
            local_transferred_state=local_transfer,
            rhs=tuple(rhs), conservation_before=before,
            conservation_after=after, conservation_error=error,
            old_volume=float(old_volume),
            proposed_volume=float(proposed_volume),
            rho_range=rho_range, pressure_range=pressure_range,
            stage_mesh_uuid=stage_uuid, stage_path=stage_path,
            stage_leaf_count=stage_leaf_count,
            stage_mortar_count=int(mortar_count),
            stage_mpi_face_count=int(mpi_faces), bank_drift=bank_drift,
            migration_send_counts=tuple(map(int, scount)),
            migration_recv_counts=tuple(map(int, rcount)),
            accepted_bank=bank, scratch_bank=scratch_bank,
            tcurr=float(intg.tcurr),
        )
    except Exception as exc:
        # Do not vote twice.  A _CollectiveFailure means this rank already
        # participated in the same all-rank failure vote as its peers.  A
        # raw local exception must join the peers at their next checkpoint
        # exactly once.
        if isinstance(exc, _CollectiveFailure):
            failure = exc
        else:
            try:
                _collective_error(comm, 'transaction', exc)
            except _CollectiveFailure as cexc:
                failure = cexc

        if not committed:
            if reader is not None:
                reader.close()
            _remove_stage(comm, stage_path)
        raise failure
