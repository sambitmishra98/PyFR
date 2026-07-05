from dataclasses import replace

import numpy as np

from pyfr.mpiutil import DistributedDirectory, get_comm_rank_root
from pyfr.readers.native import construct_con, sort_eles
from pyfr.readers.shared_nodes import SharedNodesFinder


def _exchange_eles_spts(old_mesh, exchangers):
    eles, spts, nvals = {}, {}, {}

    # Build a lookup for node valency from the old mesh
    oidxs, ovals = old_mesh.node_idxs, old_mesh.node_valency

    for et, etex in exchangers.items():
        if et not in old_mesh.eles:
            continue

        # Exchange element structured arrays, geometry, and valency
        new_e = etex.Exchange(old_mesh.eles[et])
        new_s = etex.Exchange(old_mesh.spts[et], axis=1)

        old_enodes = old_mesh.eles[et]['nodes']
        new_v = etex.Exchange(ovals[np.searchsorted(oidxs, old_enodes)])

        if len(new_e):
            eles[et] = new_e
            spts[et] = new_s
            nvals[et] = new_v

    return eles, spts, nvals


def _populate_nodes(mesh, eles, spts, nvals):
    if not eles:
        return

    # Derive node indices, locations, and valency from exchanged data
    idxs_parts, locs_parts, vals_parts = [], [], []
    for et in eles:
        idxs_parts.append(eles[et]['nodes'].ravel())
        locs_parts.append(spts[et].transpose(1, 0, 2).reshape(-1, mesh.ndims))
        vals_parts.append(nvals[et].ravel())

    idxs = np.concatenate(idxs_parts)
    locs = np.concatenate(locs_parts)
    vals = np.concatenate(vals_parts)

    unique_idxs, first_occ = np.unique(idxs, return_index=True)
    mesh.node_idxs = unique_idxs
    mesh.node_locs = locs[first_occ]
    mesh.node_valency = vals[first_occ]

    # Populate per-etype mesh fields
    for et, einfo in eles.items():
        mesh.spts[et] = spts[et]
        mesh.spts_nodes[et] = einfo['nodes']
        mesh.spts_curved[et] = einfo['curved']
        mesh.colours[et] = einfo['colour']
        mesh.tags[et] = einfo['tags']


def rebuild_mesh(old_mesh, exchangers):
    comm, _, _ = get_comm_rank_root()

    new_eidxs = {et: ex.newgeidxs for et, ex in exchangers.items()
                 if len(ex.newgeidxs)}

    mesh = replace(old_mesh, eidxs=new_eidxs, eles={}, spts={}, spts_nodes={},
                   spts_curved={}, colours={}, tags={}, con=(), con_p={},
                   bcon={}, cidxmap={}, node_idxs=None, node_valency=None,
                   node_locs=None, shared_nodes=None, neighbours=None)

    # Exchange element data, geometry, and node valency in memory
    eles, spts, nvals = _exchange_eles_spts(old_mesh, exchangers)
    mesh.eles = eles

    # Sort elements and apply perm to geometry
    perm = sort_eles(mesh, eles)
    for et, order in perm.items():
        spts[et] = spts[et][:, order]
        nvals[et] = nvals[et][order]

    _populate_nodes(mesh, eles, spts, nvals)

    etype_owner = {et: DistributedDirectory(comm, new_eidxs.get(et, []))
                   for et in mesh.etypes}
    construct_con(mesh, eles, etype_owner=etype_owner)

    snf = SharedNodesFinder(eles, mesh.node_idxs, mesh.node_valency)
    mesh.shared_nodes = snf.compute()

    new_ex = {et: ex.with_perm(perm[et]) if et in perm else ex
              for et, ex in exchangers.items()}

    return mesh, new_ex
