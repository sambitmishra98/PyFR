from collections import defaultdict
import re

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import comm, rank, root, rankmap, initialise_new_comm
from pyfr.partitioners.base import BasePartitioner
from pyfr.partitioners.online.diffusion import DiffusionRepartitioner
from pyfr.progress import NullProgressSequence

from pyfr.partitioners.online.base import _MetaMesh

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

def construct_by_diffusion(mesh, part_wts, progress=NullProgressSequence):
    initialise_new_comm('compute', rankmap['world'])

    pw = np.asarray(part_wts, dtype=np.float64).ravel()

    if pw.size != comm['compute'].size:
        raise ValueError(
            f"[diffuse] weights length mismatch: got {pw.size} weights "
            f"but MPI has {comm['compute'].size} ranks; provide exactly one weight per rank"
        )

    if not np.all(np.isfinite(pw)):
        raise ValueError("[diffuse] weights contain non-finite values")

    if np.any(pw < 0):
        raise ValueError("[diffuse] weights must be non-negative")

    if float(pw.sum()) <= 0.0:
        raise ValueError("[diffuse] weights sum must be > 0")

    with progress.start('Initialise relocator'):
        mmesh = DiffusionRepartitioner.from_mesh(mesh)
        mmesh.i.info()
        target = mmesh.calc_target(pw)

        if rank['compute'] == root['compute']:
            print(f"{target = }", flush=True)

    with progress.start('Remove islands'):
        mmesh.remove_islands_till_convergence()

    with progress.start('Diffuse elements'):
        mmesh.i.info()
        mmesh.diffuse_till_convergence(target, max_iters=100)

    with progress.start('Create relocated mesh'):
        mesh = mmesh.to_mesh(mmesh.i.eidxs)
        local_ecounts = sum(len(eidxs) for eidxs in mesh.eidxs.values())
        all_ecounts = comm['compute'].gather(local_ecounts, root=root['compute'])
        if rank['compute'] == root['compute']:
            print(f"[itc4.part] FINAL_ECOUNTS={all_ecounts}", flush=True)

        mmesh.i.info()

    eidxs = mesh.eidxs
    sparts = {}
    for etype in mesh.etypes:
        idxs = np.asarray(eidxs.get(etype, ()), dtype=np.int64)
        parts = np.full(idxs.size, rank['compute'], dtype=np.int32)
        sparts[etype] = (idxs, parts)

    sparts = {
        etype: comm['compute'].gather(v, root=root['compute'])
        for etype, v in sparts.items()
    }

    if rank['compute'] == root['compute']:
        for etype, sp in sparts.items():
            idxs_list, parts_list = zip(*sp)

            if all(len(ix) == 0 for ix in idxs_list):
                sparts[etype] = np.empty(0, dtype=np.int32)
                continue

            idxs  = np.concatenate(idxs_list)
            parts = np.concatenate(parts_list)

            sparts[etype] = parts[np.argsort(idxs)]

        # Concatenate across etypes in sorted etype order
        vparts = np.concatenate([p for _, p in sorted(sparts.items())])
    else:
        vparts = None

    return vparts
