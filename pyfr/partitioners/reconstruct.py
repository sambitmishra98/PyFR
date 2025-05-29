from collections import defaultdict
import re

import numpy as np

from pyfr.inifile import Inifile
from pyfr.partitioners.base import BasePartitioner
from pyfr.relocator.diffuser import LoadRelocator
from pyfr.progress import NullProgressSequence

from pyfr.relocator.metameshes2 import run_function

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

def reconstruct_by_diffusion(mesh, name, part_wts, progress=NullProgressSequence):

    comm, rank, root = get_comm_rank_root()

    load_relocator = LoadRelocator(mesh, bmmesh='existing', 
                                   cmmesh='diffuse', cnmmesh='diffuse_new')

    with progress.start('Initialise relocator'):
        load_relocator.move_priority = list(range(comm.size))

        load_relocator.curr_nelems = comm.allgather(
            load_relocator.mm.get_mmesh('diffuse').nelems)
        t_nelems = load_relocator.firstguess_target_nelems(part_wts)

    with progress.start('Diffuse elements'):
        print(f't_nelems_byrank: {t_nelems}')
        mesh = load_relocator.diffuse_computation('diffuse', t_nelems, 
                                                    cli=True)

    # Group partition number and element idx from mesh of each rank
    sparts = {etype: [] for etype in mesh.etypes}
    eidxs = load_relocator.mm.get_mmesh('diffuse_new').eidxs
    for etype, idxs in eidxs.items():
        sparts[etype] = (idxs, rank*np.ones(len(idxs), dtype=np.int32))

    # Gather sparts data from all ranks, by element type
    sparts = {k: comm.gather(v, root=root) for k, v in sparts.items()}
    if rank == root:
        print(f'Mesh {name} created with partition weights: {part_wts}\n')
        for etype, sp in sparts.items():
            idxs, parts = map(np.concatenate, zip(*sp))

            sparts[etype] = parts[np.argsort(idxs)] 

        vparts = np.concatenate([p for _, p in sorted(sparts.items())])
    else:
        vparts = None

    return vparts

def reconstruct_by_relocation(mesh, progress=NullProgressSequence):

    comm, rank, root = get_comm_rank_root()

    with progress.start('Example move elements from rank 2 to 1'):
        # Move elements from rank 2 to rank 1
        mesh = run_function(mesh, comm)
    # Group partition number and element idx from mesh of each rank
    sparts = {etype: [] for etype in mesh.etypes}
    eidxs = mesh.eidxs
    for etype, idxs in eidxs.items():
        sparts[etype] = (idxs, rank*np.ones(len(idxs), dtype=np.int64))

    # Gather sparts data from all ranks, by element type
    sparts = {k: comm.gather(v, root=root) for k, v in sparts.items()}
    if rank == root:
        for etype, sp in sparts.items():
            idxs, parts = map(np.concatenate, zip(*sp))

            sparts[etype] = parts[np.argsort(idxs)] 

        vparts = np.concatenate([p for _, p in sorted(sparts.items())])
    else:
        vparts = None

    return vparts


# mpirun -n 3 pyfr partition relocate -f square.pyfrm 3 3 pp3
# mpirun -n 3 pyfr run -b openmp -Ppp3 square.pyfrm couette.ini 
# mpirun -n 3 pyfr export volume -Ppp3 square.pyfrm plugin-lb.pyfrs lb.vtu
