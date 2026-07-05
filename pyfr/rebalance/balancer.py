import numpy as np

from pyfr.graphutil import (Graph, GraphPartitioner, build_csr, con_to_flat,
                            dedup_weighted_edges, graph_edge_sources)
from pyfr.mpiutil import (AlltoallMixin, RootGatherer, autofree,
                          get_comm_rank_root, mpi, pcg_scalar, scal_coll)
from pyfr.nputil import range_offsets
from pyfr.readers.native import build_g2l, parse_codec
from pyfr.util import DisjointSet


class DiffusionBalancer(AlltoallMixin):
    def __init__(self, mesh):
        self.comm, self.rank, _ = get_comm_rank_root()
        self.neighbours = mesh.neighbours

        # Neighbour communicator for the partition graph
        self._etype_off, self._neles = range_offsets(
            sorted(mesh.eidxs.items())
        )

        if self._neles == 0:
            raise ValueError('Rank has no elements')

        if len(self.neighbours):
            self.ncomm = autofree(self.comm.Create_dist_graph_adjacent(
                self.neighbours, self.neighbours
            ))
        else:
            self.ncomm = None

        # Internal face adjacency as CSR
        if mesh.con:
            lhs, rhs = mesh.con
            li = con_to_flat(lhs, self._etype_off)
            ri = con_to_flat(rhs, self._etype_off)
        else:
            li = ri = np.empty(0, dtype=int)

        self._vtab, self._etab, _ = build_csr(self._neles, li, ri,
                                               symmetrise=True)
        self._int_deg = np.diff(self._vtab).astype(np.int32)

        # Per-neighbour external face degree (sparse: indices + values)
        self._ext_deg = {}
        for nrank, con in mesh.con_p.items():
            idx = [self._etype_off[et] + eidxs
                   for et, _, eidxs in con.items()]
            if idx:
                flat = np.concatenate(idx)
                deg = np.bincount(flat, minlength=self._neles).astype(np.int32)
                nz = np.flatnonzero(deg)
                self._ext_deg[nrank] = (nz, deg[nz])
            else:
                self._ext_deg[nrank] = (np.empty(0, dtype=int),
                                        np.empty(0, dtype=np.int32))

        self._periodic_groups = _periodic_groups(mesh, self._etype_off)

    def compute_flow(self, my_load):
        comm = self.comm

        if self.ncomm is None or comm.size < 2:
            return {}

        nbrs = self.neighbours
        mean_load = scal_coll(comm.Allreduce, my_load) / comm.size

        # Solve L*phi = b via CG on the partition graph Laplacian
        diag = float(len(nbrs))
        nbuf = np.empty(len(nbrs))

        def lap(x):
            self.ncomm.Neighbor_allgather(np.array([x]), nbuf)
            return diag*x - nbuf.sum()

        dinv = 1.0 / max(diag, 1.0)
        phi = pcg_scalar(comm, lap, lambda r: r*dinv, my_load - mean_load)

        # Gather neighbour potentials
        nbr_phi = np.empty(len(nbrs))
        self.ncomm.Neighbor_allgather(np.array([phi]), nbr_phi)

        # Only overloaded ranks emit flow
        excess = my_load - mean_load
        if excess <= 0:
            return {}

        # Compute per-neighbour flow proportional to potential diff
        diff = phi - nbr_phi
        mask = diff > 0
        if not mask.any():
            return {}

        # Cap total outflow at our excess load
        flow = diff[mask]
        flow *= excess / flow.sum()

        return dict(zip(nbrs[mask], flow))

    def select_elements(self, flow, weights, max_neg_gain=-1, new_owner=None,
                        check_islands=True):
        rank, neles = self.rank, self._neles
        vtab, etab = self._vtab, self._etab

        if new_owner is None:
            new_owner = np.full(neles, rank, dtype=int)

        if not flow:
            return new_owner, 0

        moving = np.zeros(neles, dtype=bool)
        has_adj = check_islands and len(etab) > 0
        esrc = graph_edge_sources(vtab)

        # Process neighbours in order of decreasing flow
        for nrank in sorted(flow, key=flow.get, reverse=True):
            remaining = flow[nrank]
            if remaining <= 0:
                continue

            # Expand wavefront until flow is satisfied
            for _wave in range(neles):
                if remaining <= 0:
                    break

                # Count adjacency to target rank (MPI + internal)
                adj = np.zeros(neles, dtype=np.int32)

                if (sp := self._ext_deg.get(nrank)) is not None:
                    adj[sp[0]] += sp[1]

                np.add.at(adj, esrc[new_owner[etab] == nrank], 1)

                # Find owned elements adjacent to target rank
                mine = new_owner == rank
                cands = np.flatnonzero(mine & (adj > 0))
                if len(cands) == 0:
                    break

                # Compute gain = faces_to_target - internal_faces
                gains = adj[cands] - self._int_deg[cands]

                # Filter by gain threshold
                keep = gains >= max_neg_gain
                cands = cands[keep]
                gains = gains[keep]
                if len(cands) == 0:
                    break

                # Sort by gain descending, select up to flow delta
                cands = cands[np.argsort(-gains)]

                cut = np.searchsorted(np.cumsum(weights[cands]), remaining)
                n_move = min(cut + 1, len(cands))
                batch = cands[:n_move]

                # Reject moves that would create islands
                if has_adj:
                    moving[batch] = True
                    safe = _check_islands(batch, moving, vtab, etab)
                    moving[batch[~safe]] = False
                    batch = batch[safe]

                new_owner[batch] = nrank
                remaining -= float(weights[batch].sum())

        return new_owner, (new_owner != rank).sum()

    def refine(self, new_owner, weights, nrounds=10):
        rank = self.rank
        neles = self._neles
        nbrs = self.neighbours
        vtab, etab = self._vtab, self._etab

        if len(etab) == 0 or len(nbrs) == 0:
            return

        # Map rank IDs to local column indices
        nparts = len(nbrs) + 1
        rank_to_col = {rank: 0}
        for i, nr in enumerate(nbrs):
            rank_to_col[nr] = i + 1

        owner_col = np.full(neles, -1, dtype=np.int32)
        for r, c in rank_to_col.items():
            owner_col[new_owner == r] = c

        parts = owner_col.copy()
        parts[parts < 0] = 0

        # Build local CSR including MPI boundary edges
        vwts = np.maximum(np.rint(weights).astype(np.int32), 1)
        partwts = np.ones(nparts) / nparts

        GraphPartitioner().refine(parts, vtab, etab, vwts, partwts, nrounds)

        # Map column indices back to rank IDs
        col_to_rank = np.array([rank] + list(nbrs))
        new_owner[:] = col_to_rank[parts]

    def _run_sub_iters(self, weights, new_owner, n_sub, neg_gain, islands):
        comm = self.comm
        for _ in range(n_sub):
            cur_load = float(weights[new_owner == self.rank].sum())
            flow = self.compute_flow(cur_load)
            new_owner, moved = self.select_elements(
                flow, weights, max_neg_gain=neg_gain,
                new_owner=new_owner, check_islands=islands
            )
            if not scal_coll(comm.Allreduce, moved, op=mpi.SUM):
                break

        return new_owner

    def repartition(self, weights, mesh):
        comm = self.comm
        rank = self.rank
        neles = self._neles
        vtab, etab = self._vtab, self._etab

        new_owner = np.full(neles, rank, dtype=int)
        if comm.size < 2 or neles == 0:
            return new_owner, 0

        def gather(arr):
            return RootGatherer(comm, len(arr))(arr)

        vwts = np.maximum(np.rint(weights).astype(np.int32), 1)

        # Coarsen the local graph to reduce gather size
        hierarchy, mpi_map = self._local_coarsen(vtab, etab, vwts)

        c_vtab, c_etab, c_ewts, c_vwts = hierarchy[-1][:4]
        c_nv = len(c_vtab) - 1

        # Compute global offsets for coarse vertex numbering
        counts = np.empty(comm.size, dtype=int)
        comm.Allgather(np.array([c_nv]), counts)
        goff = np.concatenate(([0], counts.cumsum()))
        my_off = goff[rank]

        # Pack local coarsened edges as (src, dst, weight) triples
        c_esrc = graph_edge_sources(c_vtab, np.int32)
        local_triples = np.column_stack([c_esrc + my_off,
                                         c_etab + my_off, c_ewts])

        # Build weighted MPI edge triples
        mpi_triples = self._coarse_mpi_edges(mesh, my_off, mpi_map)

        # Gather coarsened graph to root
        all_local = gather(local_triples)
        all_mpi = gather(mpi_triples)
        all_vwts = gather(c_vwts.ravel())

        # Partition the coarsened graph on root and broadcast
        all_parts = np.empty(goff[-1], dtype=np.int32)
        if rank == 0:
            all_parts[:] = _partition_on_root(goff[-1], all_local, all_mpi,
                                                all_vwts, comm.size)
        comm.Bcast(all_parts)

        # Project coarse partition back through hierarchy
        c_parts = all_parts[my_off:my_off + c_nv]

        for _, _, _, _, match, _ in reversed(hierarchy[1:]):
            c_parts = c_parts[match]

        new_owner[:] = c_parts
        return new_owner, (new_owner != rank).sum()

    def _local_coarsen(self, vtab, etab, vwts):
        nv = len(vtab) - 1
        ewts = np.ones(len(etab), dtype=np.int32)
        esrc = graph_edge_sources(vtab, np.int32)

        hierarchy = [(vtab, etab, ewts, vwts.reshape(-1, 1), None, esrc)]

        # Coarsen until small enough to gather
        target = max(20000, 500*self.comm.size)
        gp = GraphPartitioner(seed=2079 + self.rank)

        while nv > target:
            cv, ce, cw, cwv = hierarchy[-1][:4]
            result = gp.coarsen(cv, ce, cw, cwv)
            if result is None:
                break

            nv_c = len(result[0]) - 1
            if nv_c >= 0.8*nv:
                break

            hierarchy.append(result)
            nv = nv_c

        # Build fine-to-coarse element mapping for MPI edges
        mpi_map = np.arange(self._neles, dtype=np.int32)
        for _, _, _, _, match, _ in hierarchy[1:]:
            if match is not None:
                mpi_map = match[mpi_map]

        return hierarchy, mpi_map

    def _coarse_mpi_edges(self, mesh, my_off, mpi_map):
        nbrs = self.neighbours

        # Map local MPI face elements to coarse global indices
        my_gidx = {}
        for nrank, con in mesh.con_p.items():
            fine = con_to_flat(con, self._etype_off)
            my_gidx[nrank] = mpi_map[fine] + my_off

        # Exchange coarse indices with neighbours
        send = [my_gidx.get(nr, np.empty(0, dtype=int)) for nr in nbrs]

        if self.ncomm is not None and len(nbrs):
            scount = np.array([len(s) for s in send], dtype=int)
            rcount = np.empty_like(scount)
            self.ncomm.Neighbor_alltoall(scount, rcount)

            sdisps = self._count_to_disp(scount)
            rdisps = self._count_to_disp(rcount)

            svals = np.concatenate(send)

            rvals = np.empty(rcount.sum(), dtype=int)
            self.ncomm.Neighbor_alltoallv((svals, (scount, sdisps)),
                                          (rvals, (rcount, rdisps)))
            recv = np.split(rvals, rdisps[1:])
        else:
            recv = [np.empty(0, dtype=int) for _ in nbrs]

        # Build edge pairs from matched faces
        pairs = []
        for mine, theirs in zip(send, recv):
            if (n := min(len(mine), len(theirs))) > 0:
                p = np.column_stack([mine[:n], theirs[:n]])
                keep = p[:, 0] != p[:, 1]
                if keep.any():
                    pairs.append(p[keep])

        if not pairs:
            return np.empty((0, 3), dtype=int)

        edges = np.vstack(pairs)

        # Deduplicate and count parallel edges
        ones = np.ones(len(edges), dtype=np.int32)
        s, d, w = dedup_weighted_edges(*edges.T, ones)

        return np.column_stack([s, d, w])

    def balance(self, weights, mode='aggressive', **kwargs):
        neles = self._neles

        if self.ncomm is None or neles == 0:
            return np.full(neles, self.rank, dtype=int), 0

        new_owner = np.full(neles, self.rank, dtype=int)

        # Full multilevel repartition via coarsen-gather-partition
        if mode == 'repartition':
            new_owner, _ = self.repartition(weights, kwargs['mesh'])
            do_refine = True
        # Diffusion with wavefront expansion
        elif mode == 'aggressive':
            nf = max(self._vtab[-1] // self._neles + 1, 3)
            new_owner = self._run_sub_iters(weights, new_owner, 20, -nf, False)
            do_refine = False
        # Fine diffusion with island checking
        else:
            new_owner = self._run_sub_iters(weights, new_owner, 10, -1, True)
            do_refine = True

        n_moved = (new_owner != self.rank).sum()
        if n_moved > 0 and do_refine:
            self.refine(new_owner, weights)
            n_moved = (new_owner != self.rank).sum()

        self._align_periodic(new_owner, weights)
        n_moved = (new_owner != self.rank).sum()

        # Convert flat owner array to per-etype dict
        etypes = sorted(self._etype_off, key=self._etype_off.get)
        bounds = [self._etype_off[et] for et in etypes] + [self._neles]
        ownermap = {et: new_owner[bounds[i]:bounds[i + 1]]
                     for i, et in enumerate(etypes)}

        return ownermap, n_moved

    def _align_periodic(self, new_owner, weights):
        for grp in self._periodic_groups:
            owners = new_owner[grp]
            if np.all(owners == owners[0]):
                continue

            uown, inv = np.unique(owners, return_inverse=True)
            counts = np.bincount(inv)
            loads = np.bincount(inv, weights=weights[grp], minlength=len(uown))
            order = np.lexsort((uown, -loads, -counts))
            new_owner[grp] = uown[order[0]]


def _partition_on_root(nv, local_triples, mpi_triples, vwts, nparts):
    src, dst, wts = np.vstack([local_triples, mpi_triples]).T

    vtab, etab, ewts = build_csr(nv, src, dst, wts, symmetrise=True,
                                 dedup=True)
    graph = Graph(vtab, etab, vwts.reshape(-1, 1).astype(np.int32), ewts)

    return GraphPartitioner().partition(graph, np.ones(nparts) / nparts)


def _periodic_groups(mesh, etype_off):
    if 'periodic' not in mesh.raw:
        return []

    cidxmap, _ = parse_codec(mesh.codec, mesh.etypes)
    g2l = build_g2l(mesh.etypes, mesh.eidxs)
    ds = DisjointSet()

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

    for pcon in mesh.raw['periodic'].values():
        for left, right in pcon[()].reshape(-1, 2):
            li = flat_idx(left)
            ri = flat_idx(right)

            if li is None and ri is None:
                continue
            elif li is None or ri is None:
                raise RuntimeError('Periodic elements split across ranks')
            elif li != ri:
                ds.union(li, ri)

    groups = {}
    for child, root in ds.merges().items():
        groups.setdefault(root, []).append(child)

    return [np.array([root, *children], dtype=int)
            for root, children in groups.items()]


def _check_islands(batch, moving, vtab, etab):
    safe = np.ones(len(batch), dtype=bool)

    for i, eidx in enumerate(batch):
        lo, hi = vtab[eidx], vtab[eidx + 1]
        if lo < hi:
            nbrs = etab[lo:hi]
            if np.all(moving[nbrs]):
                safe[i] = False
                moving[eidx] = False

    return safe
