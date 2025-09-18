from collections import defaultdict
import re

import numpy as np

from pyfr.inifile import Inifile
from pyfr.partitioners.base import BasePartitioner
from pyfr.progress import NullProgressSequence


from typing import Sequence
import numpy as np
from pyfr.mpiutil import get_comm_rank_root

from pyfr.readers.native import _Mesh, _MetaMesh


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

def reconstruct_by_diffusion(mesh: _Mesh, name: str, part_wts: Sequence[float],
                             progress=NullProgressSequence,
                             initialise_only: bool = False):
    """
    New implementation atop _MetaMesh:
    • build global ownerless view,
    • apportion integer targets from weights,
    • (optionally) diffuse to meet those targets,
    • return vparts on root (None elsewhere), matching old API.
    """
    comm, rank, root = get_comm_rank_root()

    with progress.start('Build MetaMesh'):
        mm = _MetaMesh.from_mesh(mesh)  # global, same on all ranks

    with progress.start('Targets from weights'):
        targets = mm.apportion_counts_by_weights(part_wts)
        cur = mm.counts_per_rank()
        if rank == root:
            print(f"[reconstruct_by_diffusion] part_wts={list(map(float, part_wts))}", flush=True)
            print(f"[reconstruct_by_diffusion] cur={cur.tolist()} targets={targets.tolist()}", flush=True)
        # hard guard: totals must match
        assert int(targets.sum()) == int(cur.sum()), \
            f"Target sum {int(targets.sum())} must equal total elements {int(cur.sum())}"

    if not initialise_only:
        with progress.start('Diffuse to targets'):
            mm.diffuse_to_targets(targets)
    else:
        if rank == root:
            print("[reconstruct_by_diffusion] initialise_only=True -> skipping diffusion", flush=True)

    with progress.start('Assemble vparts'):
        vparts_all = mm.vparts_global().astype(np.int32, copy=False)
        # keep old behavior: only root returns a value
        vparts = vparts_all if rank == root else None

        if rank == root:
            print(f"[reconstruct_by_diffusion] vparts size={int(vparts_all.size)}", flush=True)

    return vparts