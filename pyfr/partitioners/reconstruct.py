from collections import defaultdict
import re

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import comm, rank, root, rankmap, initialise_new_comm
from pyfr.partitioners.base import BasePartitioner
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

def reconstruct_by_diffusion(mesh, part_wts, progress=NullProgressSequence):
    initialise_new_comm('compute', rankmap['world'])

    mmesh = _MetaMesh.from_mesh(mesh)

    exec_order= [
        dict(kind="all-verts",    name="vertex-push",      mode="vertices", iface="all-or-none", threshold=  0, use_flow=True , overshoot=1.0, max_sweeps=  1, patience=0, min_change=0, restrict_src_dest=False, move_spts_nodes=True), # 1a) vertex push along flow
        dict(kind="smooth-faces", name="faces-final-flow", mode="faces",    iface="per-element", threshold= -1, use_flow=False, overshoot=0.0, max_sweeps=100, patience=0, min_change=0, restrict_src_dest=False, move_spts_nodes=True), # 3a) final flow-based faces push (thr=0, as in your old exec_order)
        dict(kind="smooth-faces", name="faces-final-flow", mode="faces",    iface="per-element", threshold= -1, use_flow=False, overshoot=0.0, max_sweeps=100, patience=0, min_change=0, restrict_src_dest=False, move_spts_nodes=True), # 3a) final flow-based faces push (thr=0, as in your old exec_order)
        dict(kind="smooth-faces", name="faces-final-flow", mode="faces",    iface="per-element", threshold= -1, use_flow=False, overshoot=0.0, max_sweeps=100, patience=0, min_change=0, restrict_src_dest=False, move_spts_nodes=True), # 3a) final flow-based faces push (thr=0, as in your old exec_order)
        dict(kind="all-faces",    name="faces-push",       mode="faces",    iface="per-element", threshold=  0, use_flow=True , overshoot=0.0, max_sweeps=  4, patience=0, min_change=0, restrict_src_dest=False, move_spts_nodes=True), # 3a) final flow-based faces push (thr=0, as in your old exec_order)
        dict(kind="smooth-faces", name="faces-final-flow", mode="faces",    iface="per-element", threshold= -1, use_flow=False, overshoot=0.0, max_sweeps=100, patience=0, min_change=0, restrict_src_dest=False, move_spts_nodes=True), # 3a) final flow-based faces push (thr=0, as in your old exec_order)
        ]

    mmesh.exec_order = exec_order

    mmesh.info(mesh)

    pw = np.asarray(part_wts, dtype=np.float64)

    if len(pw) < comm['compute'].size:
        npad = comm['compute'].size - len(pw)
        pw = np.pad(pw, (0, npad), 'constant', constant_values=0.0)

    with progress.start('Initialise relocator'):
        target_counts = _MetaMesh.normalise_and_round_targets(
            pw, Ntot=mmesh.ntotal, tag="[normalise]")

        if rank['compute'] == root['compute']:
            print(f"{target_counts = }", flush=True)

    with progress.start('Diffuse elements'):
        #mmesh.iterate(objective='to-target', target_counts=target_counts)
        mmesh.iterate_to_convergence(target_counts)

        # Also refine
        # for _ in range(10):
        #     mmesh.refine('cpd', mode='edges', thr = 5)
    
    # --- rebuild a "relocated" mesh for this rank and extract vparts ---
    with progress.start('Create relocated mesh'):
        mesh = mmesh.to_mesh(mmesh.i.eidxs)
        # Print final element coutns for each rank from compute_rank
        local_ecounts = sum(len(eidxs) for eidxs in mesh.eidxs.values())
        all_ecounts = comm['compute'].gather(local_ecounts, root=root['compute'])
        if rank['compute'] == root['compute']:
            print(f"[itc4.part] FINAL_ECOUNTS={all_ecounts}", flush=True)

        mmesh.info(mesh)

    # Group per-rank element indices and owning partition (compute rank)
    # For every etype, every rank contributes a *pair* of arrays:
    #   (local element indices of this etype, local partition IDs == compute rank)
    eidxs = mesh.eidxs
    sparts = {}
    for etype in mesh.etypes:
        idxs = np.asarray(eidxs.get(etype, ()), dtype=np.int64)
        parts = np.full(idxs.size, rank['compute'], dtype=np.int32)
        sparts[etype] = (idxs, parts)

    # Gather sparts data from all ranks, by element type
    sparts = {
        etype: comm['compute'].gather(v, root=root['compute'])
        for etype, v in sparts.items()
    }

    if rank['compute'] == root['compute']:
        for etype, sp in sparts.items():
            # sp is now a list of (idxs, parts) pairs, one per rank
            idxs_list, parts_list = zip(*sp)

            # If this etype is globally empty, keep an empty array and continue
            if all(len(ix) == 0 for ix in idxs_list):
                sparts[etype] = np.empty(0, dtype=np.int32)
                continue

            idxs  = np.concatenate(idxs_list)
            parts = np.concatenate(parts_list)

            # Sort by element index to obtain consistent global vparts ordering
            sparts[etype] = parts[np.argsort(idxs)]

        # Concatenate across etypes in sorted etype order
        vparts = np.concatenate([p for _, p in sorted(sparts.items())])
    else:
        vparts = None

    return vparts
