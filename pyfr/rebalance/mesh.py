from dataclasses import replace

import numpy as np

from pyfr.mpiutil import (DestExchanger, DistributedDirectory,
                          get_comm_rank_root, mpi)
from pyfr.readers.native import (build_g2l, construct_con, parse_codec,
                                 sort_eles)
from pyfr.readers.shared_nodes import SharedNodesFinder


def _row_intersect(a, b, blk=1 << 16):
    # Row-wise set intersection of two -1 padded index arrays
    outs, kmax = [], 1

    for i in range(0, len(a), blk):
        aa, bb = a[i:i + blk], b[i:i + blk]

        m = (aa[:, :, None] == bb[:, None, :]).any(axis=2) & (aa >= 0)
        kmax = max(kmax, int(m.sum(axis=1).max(initial=1)))

        # Sort matches to the front of each row
        key = np.argsort(~m, axis=1, kind='stable')
        outs.append(np.take_along_axis(np.where(m, aa, -1), key, axis=1))

    if outs:
        return np.vstack(outs)[:, :kmax]
    else:
        return np.empty((0, 1), dtype=int)


def _con_flat(con, etype_off, cfidx):
    cidx_off = np.empty(len(cfidx), dtype=int)
    for cidx, off in etype_off.items():
        cidx_off[cidx] = off

    return cidx_off[con.cidxs] + con.eidxs, cfidx[con.cidxs]


def _frozen_flat(mesh, etype_off):
    # Flat indices of elements in periodic groups; these must remain
    # co-resident and so are pinned to their current rank
    if 'periodic' not in mesh.raw:
        return np.empty(0, dtype=int)

    cidxmap, _ = parse_codec(mesh.codec, mesh.etypes)
    g2l = build_g2l(mesh.etypes, mesh.eidxs)

    def flat_idx(rec):
        etype, _ = cidxmap[int(rec['cidx'])]
        if etype not in g2l:
            return None

        ordgi, perm = g2l[etype][1:]
        pos = np.searchsorted(ordgi, rec['off'])
        if pos < len(ordgi) and ordgi[pos] == rec['off']:
            return etype_off[etype] + perm[pos]
        else:
            return None

    flats = set()
    for pcon in mesh.raw['periodic'].values():
        for left, right in pcon[()].reshape(-1, 2):
            li = flat_idx(left)
            ri = flat_idx(right)

            if li is None and ri is None:
                continue
            elif li is None or ri is None:
                raise RuntimeError('Periodic elements split across ranks')
            elif li != ri:
                flats.update((li, ri))

    return np.fromiter(flats, dtype=int, count=len(flats))


class IndexMesh:
    '''
    Index-space view of the element partitioning.

    Elements are identified by globally-unique flat ids and carry their
    face adjacency (neighbour ids and owner ranks), per-face shared node
    ids, element node ids, and centroids.  Ownership changes are applied
    with move(), which physically relocates these index records between
    ranks; the underlying solver mesh is untouched.
    '''
    def __init__(self, mesh):
        comm, rank, root = get_comm_rank_root()
        self.comm, self.rank = comm, rank

        self.etypes = etypes = mesh.etypes
        loc_ets = [et for et in etypes if et in mesh.eidxs]

        # Globally-unique flat element ids
        ecount = np.array([len(mesh.eidxs.get(et, ())) for et in etypes])
        gcount = np.empty_like(ecount)
        comm.Allreduce(ecount, gcount)

        goffs = np.concatenate(([0], gcount[:-1].cumsum()))
        self.goff = dict(zip(etypes, goffs))
        self.nglobal = int(gcount.sum())

        etype_off, off = {}, 0
        for et in loc_ets:
            etype_off[et] = off
            off += len(mesh.eidxs[et])

        ne = self.neles = off

        self.gids = np.concatenate(
            [self.goff[et] + np.asarray(mesh.eidxs[et], dtype=int)
             for et in loc_ets]
        ) if ne else np.empty(0, dtype=int)
        self.eti = np.concatenate(
            [np.full(len(mesh.eidxs[et]), etypes.index(et), dtype=np.int16)
             for et in loc_ets]
        ) if ne else np.empty(0, dtype=np.int16)

        # Element node ids, -1 padded to a globally consistent width
        nn = max((mesh.eles[et]['nodes'].shape[1] for et in loc_ets),
                 default=0)
        nn = comm.allreduce(nn, op=mpi.MAX)

        self.nodes = np.full((ne, nn), -1, dtype=int)
        for et in loc_ets:
            i, enodes = etype_off[et], mesh.eles[et]['nodes']
            self.nodes[i:i + len(enodes), :enodes.shape[1]] = enodes

        # Element centroids
        self.cents = np.empty((ne, mesh.ndims))
        for et in loc_ets:
            i, spts = etype_off[et], mesh.spts[et]
            self.cents[i:i + spts.shape[1]] = spts.mean(axis=0)

        # Codec index to face index lookup
        cidxmap, _ = parse_codec(mesh.codec, etypes)
        cfidx = np.full(len(mesh.codec), -1, dtype=int)
        for cidx, (et, fidx) in cidxmap.items():
            cfidx[cidx] = fidx

        nf = 1 + max((f for f in cfidx if f >= 0), default=0)

        # Per-face neighbour element ids and owner ranks
        self.fgid = np.full((ne, nf), -1, dtype=int)
        self.fown = np.full((ne, nf), -1, dtype=np.int32)

        rec_flat, rec_fidx, rec_other = [], [], []

        # Codec offsets for converting connectivity to flat indices
        con_off = {c: etype_off[et] for c, (et, _) in cidxmap.items()
                   if et in etype_off}

        if mesh.con:
            lhs, rhs = mesh.con
            lf, lfidx = _con_flat(lhs, con_off, cfidx)
            rf, rfidx = _con_flat(rhs, con_off, cfidx)

            for mf, mfidx, of in [(lf, lfidx, rf), (rf, rfidx, lf)]:
                self.fgid[mf, mfidx] = self.gids[of]
                self.fown[mf, mfidx] = rank

                rec_flat.append(mf)
                rec_fidx.append(mfidx)
                rec_other.append(self.nodes[of])

        # Exchange ids and node rows across each MPI interface; the face
        # orderings of con_p are symmetric between neighbouring ranks
        reqs, recvs = [], []
        for nrank in sorted(mesh.con_p):
            mf, mfidx = _con_flat(mesh.con_p[nrank], con_off, cfidx)

            send = np.ascontiguousarray(
                np.hstack([self.gids[mf, None], self.nodes[mf]])
            )
            recv = np.empty_like(send)

            reqs.append(comm.Isend(send, nrank))
            reqs.append(comm.Irecv(recv, nrank))
            recvs.append((nrank, mf, mfidx, send, recv))

        mpi.Request.Waitall(reqs)

        for nrank, mf, mfidx, _, recv in recvs:
            self.fgid[mf, mfidx] = recv[:, 0]
            self.fown[mf, mfidx] = nrank

            rec_flat.append(mf)
            rec_fidx.append(mfidx)
            rec_other.append(recv[:, 1:])

        # Per-face shared node ids, -1 padded
        if rec_flat:
            rflat = np.concatenate(rec_flat)
            rfidx = np.concatenate(rec_fidx)
            rother = np.vstack([r[:, :nn] for r in rec_other])
            shared = _row_intersect(self.nodes[rflat], rother)
        else:
            rflat = rfidx = np.empty(0, dtype=int)
            shared = np.empty((0, 1), dtype=int)

        nfv = comm.allreduce(shared.shape[1], op=mpi.MAX)

        self.fnodes = np.full((ne, nf, nfv), -1, dtype=int)
        self.fnodes[rflat, rfidx, :shared.shape[1]] = shared

        # Periodic elements are pinned to their current rank
        self.frozen = np.zeros(ne, dtype=bool)
        self.frozen[_frozen_flat(mesh, etype_off)] = True

    def counts(self):
        return np.array(self.comm.allgather(self.neles))

    def etype_counts(self):
        cnt = np.bincount(self.eti, minlength=len(self.etypes))
        return np.array(self.comm.allgather(cnt))

    def mpi_out(self):
        return int(((self.fown >= 0) & (self.fown != self.rank)).sum())

    def eidxs(self):
        eidxs = {}
        for i, et in enumerate(self.etypes):
            gids = np.sort(self.gids[self.eti == i]) - self.goff[et]
            if len(gids):
                eidxs[et] = gids

        return eidxs

    def move(self, dests):
        comm, rank = self.comm, self.rank
        dests = np.asarray(dests, dtype=int)

        # Publish the movers so all ranks can retag face owners
        mv = dests != rank
        pairs = np.vstack(comm.allgather(
            np.column_stack([self.gids[mv], dests[mv]])
        ))

        if not len(pairs):
            return

        ex = DestExchanger(comm, dests)
        for name in ('gids', 'eti', 'nodes', 'cents', 'fgid', 'fown',
                     'fnodes', 'frozen'):
            setattr(self, name, ex.Exchange(getattr(self, name)))

        self.neles = len(self.gids)

        # Retag the owners of faces adjoining moved elements
        mgid, mdst = pairs[:, 0], pairs[:, 1]
        order = np.argsort(mgid)
        mgid, mdst = mgid[order], mdst[order]

        pos = np.searchsorted(mgid, self.fgid)
        np.clip(pos, 0, len(mgid) - 1, out=pos)
        hit = (mgid[pos] == self.fgid) & (self.fgid >= 0)
        self.fown[hit] = mdst[pos[hit]].astype(self.fown.dtype)


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
