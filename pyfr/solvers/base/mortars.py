from dataclasses import dataclass

import numpy as np

from pyfr.mortars import (
    MortarGeometry, MortarGroup, MortarOperatorKey, MortarSide,
)


def mortar_face_view(be, elemap, etype, fidxs, eidxs, field, vshape,
                     *, row_offset=0):
    eles = elemap[etype]
    n = len(eidxs)
    nfp = eles.nfacefpts[fidxs[0]]
    rmap = np.array([
        eles.basis.facefpts[fidx][0] + row_offset for fidx in fidxs
    ], dtype=np.int32)

    return be.view(
        np.full(n, getattr(eles, field).mid), rmap, eidxs,
        np.ones(n, dtype=np.int32), vshape=(nfp, *vshape)
    )


def mortar_side_view(be, elemap, side, field, vshape, *, row_offset=0):
    return mortar_face_view(
        be, elemap, side.runtime_key,
        np.asarray(side.fidxs, dtype=np.int32),
        np.asarray(side.eidxs, dtype=np.int64), field, vshape,
        row_offset=row_offset
    )


def mortar_side_point_view(be, elemap, side, getter, vshape):
    fidxs = np.asarray(side.fidxs, dtype=np.int32)
    eidxs = np.asarray(side.eidxs, dtype=np.int64)
    eles = elemap[side.runtime_key]
    parts = [
        getattr(eles, getter)(np.array([eidx]), int(fidx))
        for fidx, eidx in zip(fidxs, eidxs)
    ]
    maps = [np.concatenate(values) for values in zip(*parts)]

    return be.view(*maps, vshape=vshape)


@dataclass(frozen=True)
class MortarOwnership:
    kind: str
    owner_rank: int
    participant_ranks: tuple

    @classmethod
    def local(cls, rank):
        rank = int(rank)
        return cls('local', rank, (rank,))

    @classmethod
    def distributed(cls, owner_rank, participant_ranks):
        owner_rank = int(owner_rank)
        participants = tuple(sorted(map(int, participant_ranks)))
        return cls('distributed', owner_rank, participants)

    def __post_init__(self):
        if self.kind == 'local':
            if self.participant_ranks != (self.owner_rank,):
                raise ValueError(
                    'Local mortar ownership requires one participant rank'
                )
        elif self.kind == 'distributed':
            if (
                len(self.participant_ranks) != 2
                or len(set(self.participant_ranks)) != 2
                or self.owner_rank not in self.participant_ranks
            ):
                raise ValueError(
                    'Distributed mortar ownership requires two distinct '
                    'participant ranks including the owner'
                )
        else:
            raise ValueError(
                f'Unsupported mortar ownership kind {self.kind!r}'
            )


def local_mortar_ownership():
    from pyfr.mpiutil import get_comm_rank_root

    _, rank, _ = get_comm_rank_root()
    return MortarOwnership.local(rank)


@dataclass(frozen=True, order=True)
class MPIMortarSetupSignature:
    owner_etype: str
    owner_order: int
    owner_geometry_order: int
    owner_fidx: int
    nonowner_etype: str
    nonowner_order: int
    nonowner_geometry_order: int
    nonowner_fidx: int
    reference_target: tuple
    trace_sampling: tuple


@dataclass(frozen=True)
class MPIMortarBatchPlan:
    neighbour_rank: int
    ownership: MortarOwnership
    signature: MPIMortarSetupSignature
    face_pairs: tuple
    tags: tuple


_MPI_MORTAR_CHANNELS = ('state', 'common-state', 'gradient', 'flux')


def build_mpi_p_mortar_batch_plans(face_ops):
    from mpi4py import MPI

    from pyfr.mpiutil import get_comm_rank_root

    comm, rank, _ = get_comm_rank_root()
    grouped = {}
    for face, ops in face_ops:
        key = ops['operator_keys'][0]
        patch = key.patches[0]
        sig = MPIMortarSetupSignature(
            key.left.etype, key.left.solution_order,
            face.owner_geometry_order, key.left.fidx,
            patch.right.etype, patch.right.solution_order,
            face.nonowner_geometry_order, patch.right.fidx,
            patch.left_map.target, key.trace_sampling,
        )
        gkey = face.neighbour_rank, face.owner_rank, sig
        grouped.setdefault(gkey, []).append(face)

    tag_ub = comm.Get_attr(MPI.TAG_UB)
    plans = []
    neighbours = sorted({key[0] for key in grouped})
    for nrank in neighbours:
        keys = sorted(
            key for key in grouped if key[0] == nrank
        )
        for ordinal, key in enumerate(keys):
            _, owner_rank, sig = key
            faces = sorted(grouped[key], key=lambda face: face.face_pair)
            tags = tuple(
                (channel, ordinal*len(_MPI_MORTAR_CHANNELS) + ci)
                for ci, channel in enumerate(_MPI_MORTAR_CHANNELS)
            )
            if tags and tags[-1][1] > tag_ub:
                raise RuntimeError(
                    'Distributed p-mortar MPI tag exceeds TAG_UB'
                )

            ownership = MortarOwnership.distributed(
                owner_rank, (rank, nrank)
            )
            plans.append(MPIMortarBatchPlan(
                nrank, ownership, sig,
                tuple(face.face_pair for face in faces), tags
            ))

    return tuple(plans)


@dataclass(frozen=True)
class MortarExecutionSignature:
    operator_key: MortarOperatorKey
    geometry_orders: tuple
    anti_aliasing: tuple
    equation: str
    ownership: str
    implementation: str


@dataclass(frozen=True)
class MortarExecutionBatch:
    signature: MortarExecutionSignature
    indices: tuple
    group: MortarGroup
    operator_set: int
    geometry: MortarGeometry
    ownership: MortarOwnership


def _subset_mortar_side(side, indices):
    return MortarSide(
        side.etype, side.face_topology,
        tuple(side.fidxs[i] for i in indices),
        tuple(side.eidxs[i] for i in indices), side.elekey,
    )


def _subset_mortar_group(group, indices):
    return MortarGroup(
        _subset_mortar_side(group.left, indices),
        tuple(_subset_mortar_side(side, indices) for side in group.right),
    )


def build_mortar_execution_batches(
    ops, elemap, equation, implementation, ownership
):
    group = ops['mortar_group']
    sides = (group.left, *group.right)
    geometry_orders = tuple(
        elemap[side.runtime_key].basis.nsptsord for side in sides
    )
    anti_aliasing = tuple(
        tuple(sorted(elemap[side.runtime_key].antialias)) for side in sides
    )

    batches = []
    for oi, (key, indices) in enumerate(zip(
        ops['operator_keys'], ops['operator_groups']
    )):
        indices = tuple(int(i) for i in indices)
        signature = MortarExecutionSignature(
            key, geometry_orders, anti_aliasing, equation, ownership.kind,
            implementation
        )
        batches.append(MortarExecutionBatch(
            signature, indices, _subset_mortar_group(group, indices), oi,
            ops['geometry'].subset(indices), ownership
        ))

    return tuple(batches)


class BaseMortarInters:
    @property
    def stats(self):
        return {
            'name': self.name,
            'ninters': self.ninters,
            'coarse-etype': self.coarse_etype,
            'fine-etype': self.fine_etype,
            'implementation': self.mortar_implementation,
            'operator-sets': self.noperator_sets,
            'batches': self.nbatches,
            'operator-bytes': self.operator_bytes,
            'shared-operator-bytes': self.shared_operator_bytes,
            'fused-operator-bytes': self.fused_operator_bytes,
            'geometry-bytes': self.geometry_bytes,
            'staged-buffer-bytes': self.staged_buffer_bytes,
            'staged-allocated-bytes': self.staged_allocated_bytes,
            'max-geometry-error': self.max_geom_error,
            'max-normal-error': self.max_normal_error,
        }
