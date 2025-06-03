from __future__ import annotations

from typing import TYPE_CHECKING
from collections import defaultdict
import re

import numpy as np

from pyfr.inifile import Inifile
from pyfr.partitioners.base import BasePartitioner
from pyfr.progress import NullProgressSequence

from pyfr.relocator.metamesh import MetaMesh

if TYPE_CHECKING:
    from pyfr.readers.native import _Mesh

from pyfr.mpiutil import get_comm_rank_root

def reconstruct_partitioning(mesh, soln, progress=NullProgressSequence):
    if mesh['mesh-uuid'][()] != soln['mesh-uuid'][()]:
        raise ValueError('Invalid solution for mesh')

    prefix = Inifile(soln['stats'][()].decode()).get('data', 'prefix')
    sparts = defaultdict(list)

    # Read the partition data from the solution
    for k in soln[prefix]:
        if (m := re.match(r'(p(?:\d+)-(\w+))-parts$', k)):
            parts = soln[f'{prefix}/{k}'][:]

            idxs = soln.get(f'{prefix}/{m[1]}-idxs')
            if idxs is None:
                idxs = np.arange(len(parts))

            sparts[m[2]].append((idxs, parts))

    # Group the data together by element type
    for etype, sp in sparts.items():
        idxs, parts = map(np.concatenate, zip(*sp))

        sparts[etype] = parts[np.argsort(idxs)]

    vparts = np.concatenate([p for _, p in sorted(sparts.items())])

    # Construct the global connectivity array
    with progress.start('Construct global connectivity array'):
        con, ecurved, edisps, _ = BasePartitioner.construct_global_con(mesh)

    # Ensure that the solution has not been subset
    if len(vparts) != len(ecurved):
        raise ValueError('Can not reconstruct partitioning from subsetted '
                         'solution')

    # Construct the partitioning data
    with progress.start('Construct partitioning'):
        pinfo = BasePartitioner.construct_partitioning(mesh, ecurved, edisps,
                                                       con, vparts)

    return pinfo

def reconstruct_by_relocation(mesh: _Mesh, targets: list[int], 
                              progress=NullProgressSequence):
    """
    Relocate elements towards exactly `targets[rank]` elements per rank

    Parameters
    ----------
    mesh : pyfr.readers.native._Mesh          # already duplicated on all ranks
    targets : list[int]                       # desired element-counts per rank
    progress : ProgressSequence or NullProgressSequence
    """
    comm, rank, root = get_comm_rank_root()

    R = comm.size
    targets = list(map(int, targets))        # make sure they are plain ints

    # ---------------- sanity on root, then broadcast -------------------
    loc_nelems = sum(len(mesh.eidxs[et]) for et in mesh.etypes)

    # global element count (same on every rank after the reduction)
    from mpi4py import MPI
    glob_nelems = comm.allreduce(loc_nelems, op=MPI.SUM)

    if rank == root:
        if len(targets) != R:
            raise ValueError('Length of target list must equal #MPI ranks')
        if sum(targets) != glob_nelems:
            raise ValueError(f'Target counts sum to {sum(targets)} but mesh '
                             f'has {glob_nelems} elements')

    # broadcast the (possibly-root-modified) targets list
    targets = comm.bcast(targets, root=root)

    # ---------------- run one donor→receivers round per sweep ----------
    with progress.start('Relocate elements'):
        mmesh = MetaMesh(mesh)                            # build SubMeshes
        mmesh.redistribute_sweeps(targets)                # exactly R-1 MPI
        mesh = mmesh.smeshes[rank]                        # local relocated

    # ---------------- package (gid , part) for every element -----------
    sparts = {et: (ids,
                   rank*np.ones(len(ids), dtype=np.int64))
              for et, ids in mesh.eidxs.items()}

    # gather to root, assemble vparts in global (etype, gid) order
    sparts = {k: comm.gather(v, root=root) for k, v in sparts.items()}
    if rank == root:
        for et, plist in sparts.items():
            idxs, parts = map(np.concatenate, zip(*plist))
            sparts[et]  = parts[np.argsort(idxs)]

        vparts = np.concatenate([p for _, p in sorted(sparts.items())])
    else:
        vparts = None

    return vparts
