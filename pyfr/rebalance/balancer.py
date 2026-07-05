import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi, scal_coll


def int_round(x, total):
    # Rescale to the requested total with largest-remainder rounding,
    # breaking ties stably by rank index
    x = np.asarray(x, dtype=float)
    xs = x*(total / x.sum())

    xf = np.floor(xs).astype(int)
    k = int(total - xf.sum())

    if k > 0:
        xf[np.argsort(-(xs - xf), kind='stable')[:k]] += 1
    elif k < 0:
        xf[np.argsort(xs - xf, kind='stable')[:-k]] -= 1

    return xf


def _lap_solve(lap, b, anchor=0):
    # Solve the anchored graph Laplacian system L p = b
    n = len(lap)
    p = np.zeros(n)

    if n > 1:
        idx = np.arange(n) != anchor
        lr = lap[np.ix_(idx, idx)]
        p[idx] = np.linalg.lstsq(lr, b[idx], rcond=None)[0]

    return p


def _distribute_with_caps(excess, deficit, twts, fm):
    # Split each source's excess over its downhill edges, cap sink columns
    # at their deficits, and integerise per source
    n = len(twts)
    plan = np.zeros((n, n))

    rs = twts.sum(axis=1)
    np.divide(excess[:, None]*twts, rs[:, None], out=plan,
              where=rs[:, None] > 0)

    inflow = plan.sum(axis=0)
    m = (deficit > 0) & (inflow > deficit)
    if m.any():
        plan[:, m] *= deficit[m] / inflow[m]

    iplan = np.zeros((n, n), dtype=int)
    for u in np.flatnonzero(excess > 0):
        base = np.floor(plan[u]).astype(int)

        if (left := int(excess[u] - base.sum())) > 0:
            cand = np.flatnonzero(fm[u])
            frac = plan[u, cand] - base[cand]
            base[cand[np.lexsort((cand, -fm[u, cand], -frac))[:left]]] += 1

        iplan[u] = base

    return iplan


class DiffusionBalancer:
    '''
    Diffusion-based repartitioner operating on an IndexMesh.

    Each cycle drains any zero-target ranks, removes islands and
    geometric outliers, and then runs a fixed schedule of capped
    diffusion passes with greedy smoothing until the per-rank element
    counts settle on the requested target.
    '''
    def __init__(self, cfg, sect):
        self.comm, self.rank, self.root = get_comm_rank_root()

        # Fraction of non-main islands to evict per healing round
        self.island_frac = cfg.getfloat(sect, 'island-fraction', 0.5)

        # Outlier eviction and inlier attraction
        self.outlier_frac = cfg.getfloat(sect, 'outlier-fraction', 0)
        self.outlier_mode = cfg.get(sect, 'outlier-mode', 'faces')
        self.inlier_frac = cfg.getfloat(sect, 'inlier-fraction', 0)
        self.inlier_mode = cfg.get(sect, 'inlier-mode', 'vertices')

        # Flow matrix relaxation and initial aggressive iterations
        self.flow_relax = cfg.getfloat(sect, 'flow-relax', 0.5)
        self.aggr_iters = cfg.getint(sect, 'init-aggressive-iters', 1)

    # -- Flow planning

    def flow_plan(self, im, target, max_iters=4):
        comm = self.comm
        n = comm.size

        cur = im.counts()
        tgt = np.asarray(target, dtype=int)

        # Symmetrised MPI face-count matrix
        row = np.zeros(n, dtype=int)
        fown = im.fown[(im.fown >= 0) & (im.fown != self.rank)]
        row += np.bincount(fown, minlength=n)

        amat = np.empty((n, n), dtype=int)
        comm.Allgather(row, amat)
        fmat = np.minimum(amat, amat.T).astype(float)

        # Plan on the root rank and broadcast
        mfull = np.zeros((n, n), dtype=int)

        if self.rank == self.root:
            active = np.flatnonzero((cur > 0) | (tgt > 0))
            fm = fmat[np.ix_(active, active)]
            lap = np.diag(fm.sum(axis=1)) - fm

            acur, atgt = cur[active].copy(), tgt[active]
            msub = np.zeros_like(fm, dtype=int)

            for _ in range(max_iters):
                diff = acur - atgt
                excess = np.maximum(diff, 0)
                deficit = np.maximum(-diff, 0)

                if not excess.sum() or not deficit.sum():
                    break

                # Route flow down the potential gradient, weighted by
                # face counts
                p = _lap_solve(lap, (excess - deficit).astype(float))
                twts = fm*np.maximum(p[:, None] - p[None, :], 0)

                step = _distribute_with_caps(excess, deficit, twts, fm)

                acur += step.sum(axis=0) - step.sum(axis=1)
                msub += step

            mfull[np.ix_(active, active)] = msub
            mfull *= fmat > 0
            np.fill_diagonal(mfull, 0)

            # Cancel anti-parallel flows
            mfull = np.maximum(mfull - mfull.T, 0)

        comm.Bcast(mfull, root=self.root)
        return mfull

    # -- Affinity

    def _candidates(self, im, mode, thr):
        '''
        Affinity candidates as (flat, gid, nbr, score) arrays, where for
        each boundary element and adjoining rank the score is a - b with
        a the ties to our own region and b the ties to the neighbour,
        counted in faces or shared vertices.  Entries with score > thr
        are dropped.
        '''
        rank = self.rank
        movable = ~im.frozen

        mpi_face = (im.fown >= 0) & (im.fown != rank)
        nbrs = np.unique(im.fown[mpi_face])

        if mode == 'faces':
            aown = (im.fown == rank).sum(axis=1)
        else:
            # Interface vertex sets per neighbouring rank
            vsets = {}
            for nb in nbrs:
                fv = im.fnodes[im.fown == nb]
                vsets[nb] = np.unique(fv[fv >= 0])

            vall = (np.unique(np.concatenate(list(vsets.values())))
                    if vsets else np.empty(0, dtype=int))

            valid = im.nodes >= 0
            aown = (~np.isin(im.nodes, vall) & valid).sum(axis=1)

        flats, gids, dsts, scores = [], [], [], []
        for nb in nbrs:
            if mode == 'faces':
                b = (im.fown == nb).sum(axis=1)
                sel = (b > 0) & movable
            else:
                b = (np.isin(im.nodes, vsets[nb]) & valid).sum(axis=1)
                sel = (im.fown == nb).any(axis=1) & (b > 0) & movable

            score = (aown[sel] - b[sel]).astype(float)
            keep = score <= thr

            flat = np.flatnonzero(sel)[keep]
            flats.append(flat)
            gids.append(im.gids[flat])
            dsts.append(np.full(len(flat), nb, dtype=int))
            scores.append(score[keep])

        if flats:
            return tuple(map(np.concatenate, (flats, gids, dsts, scores)))
        else:
            z = np.empty(0, dtype=int)
            return z, z, z, z.astype(float)

    def _best_nbr(self, im, mode, score_max=None):
        # Per element: adjoining rank with the lowest affinity score
        flat, _, nbr, score = self._candidates(im, mode, np.inf)

        best_nbr = np.full(im.neles, -1, dtype=int)
        best_score = np.full(im.neles, np.inf)

        if score_max is not None:
            keep = score <= score_max
            flat, nbr, score = flat[keep], nbr[keep], score[keep]

        # Ascending neighbour order makes ties resolve to the lower rank
        for i in np.argsort(nbr, kind='stable'):
            if score[i] < best_score[flat[i]]:
                best_score[flat[i]] = score[i]
                best_nbr[flat[i]] = nbr[i]

        return best_nbr, best_score

    def _contact_best(self, im):
        # Per element: adjoining rank with the highest MPI face contact
        best_nbr = np.full(im.neles, -1, dtype=int)
        best_cnt = np.zeros(im.neles, dtype=int)

        mpi_face = (im.fown >= 0) & (im.fown != self.rank)
        for nb in np.unique(im.fown[mpi_face]):
            cnt = (im.fown == nb).sum(axis=1)
            better = cnt > best_cnt

            best_cnt[better] = cnt[better]
            best_nbr[better] = nb

        return best_nbr, best_cnt

    # -- Movement passes

    def _move(self, im, flats, dests):
        moved = np.full(im.neles, self.rank, dtype=int)
        moved[flats] = dests

        im.move(moved)
        return len(flats)

    def smooth(self, im, thr=-1.0):
        # Move each element with a strictly winning neighbour
        flat, gid, nbr, score = self._candidates(im, 'faces', thr)

        if len(flat):
            order = np.lexsort((nbr, gid, score, flat))
            first = np.unique(flat[order], return_index=True)[1]

            chosen = order[np.sort(first)]
            return self._move(im, flat[chosen], nbr[chosen])
        else:
            im.move(np.full(im.neles, self.rank, dtype=int))
            return 0

    def smooth_until_stagnant(self, im, max_iters=20):
        comm, last, stable = self.comm, None, 0

        for _ in range(max_iters):
            moved = scal_coll(comm.Allreduce, self.smooth(im), op=mpi.SUM)

            stable = stable + 1 if moved == last else 0
            last = moved

            if not moved or stable >= 1:
                break

    def diffuse(self, im, caps, thr, mode):
        # Fill each destination's cap with the best-scoring candidates
        flat, gid, nbr, score = self._candidates(im, mode, thr)

        order = np.lexsort((gid, score, nbr))
        flat, gid, nbr = flat[order], gid[order], nbr[order]

        chosen_flat, chosen_nbr = [], []
        taken = np.empty(0, dtype=int)

        for nb in np.unique(nbr):
            if nb == self.rank or (cap := int(caps[nb])) <= 0:
                continue

            seg = nbr == nb
            fs, gs = flat[seg], gid[seg]

            # An element may only be moved once per sweep
            keep = ~np.isin(gs, taken)
            fs, gs = fs[keep][:cap], gs[keep][:cap]

            chosen_flat.append(fs)
            chosen_nbr.append(np.full(len(fs), nb, dtype=int))
            taken = np.concatenate([taken, gs])

        if chosen_flat:
            return self._move(im, np.concatenate(chosen_flat),
                              np.concatenate(chosen_nbr))
        else:
            im.move(np.full(im.neles, self.rank, dtype=int))
            return 0

    # -- Schedules

    def iterate(self, im, target, relax=None, smooth=True):
        relax = self.flow_relax if relax is None else relax
        schedule = [(6.0, 'vertices'), (2.0, 'faces')] + [(0.0, 'faces')]*10

        for thr, mode in schedule:
            mflow = self.flow_plan(im, target)

            caps = np.ceil(mflow[self.rank]*relax).astype(int)
            self.diffuse(im, np.maximum(caps, 0), thr, mode)

            if smooth:
                self.smooth_until_stagnant(im)

    def converge(self, im, target, max_iters=1, smooth=True):
        for _ in range(max_iters):
            cur = im.counts()
            self.iterate(im, target, smooth=smooth)

            if (im.counts() == cur).all():
                break

    def drain(self, im, target, max_iters=100):
        # Empty any ranks with a zero target
        tgt = np.asarray(target, dtype=int)

        for _ in range(max_iters):
            if not im.counts()[tgt == 0].sum():
                break

            self.iterate(im, target, relax=1.0, smooth=False)

    # -- Island and outlier handling

    def label_islands(self, im):
        # Connected components of the owned-face subgraph, labelled in
        # order of decreasing size
        ne = im.neles
        own = im.fown == self.rank

        src = np.broadcast_to(np.arange(ne)[:, None], im.fgid.shape)[own]

        order = np.argsort(im.gids)
        dst = order[np.searchsorted(im.gids[order], im.fgid[own])]

        vtab = np.zeros(ne + 1, dtype=int)
        vtab[1:] = np.bincount(src, minlength=ne).cumsum()
        etab = dst[np.argsort(src, kind='stable')]

        labels = np.full(ne, -1, dtype=int)
        sizes, cid = [], 0

        for s in range(ne):
            if labels[s] >= 0:
                continue

            stack, sz = [s], 0
            labels[s] = cid

            while stack:
                u = stack.pop()
                sz += 1

                for v in etab[vtab[u]:vtab[u + 1]]:
                    if labels[v] < 0:
                        labels[v] = cid
                        stack.append(v)

            sizes.append(sz)
            cid += 1

        sizes = np.array(sizes, dtype=int)
        order = np.lexsort((np.arange(len(sizes)), -sizes))

        remap = np.empty_like(order)
        remap[order] = np.arange(len(order))

        return remap[labels] if len(sizes) else labels, sizes[order]

    def detect_islands(self, im):
        labels, sizes = self.label_islands(im)
        nis = len(sizes)

        nis_all = self.comm.allgather(nis)

        # Evict a fraction of the smallest non-main islands
        if nis > 1 and self.island_frac > 0:
            nrm = int(np.ceil(self.island_frac*(nis - 1)))
            cluster = im.gids[labels >= nis - min(max(nrm, 1), nis - 1)]
        else:
            cluster = np.empty(0, dtype=int)

        return cluster, nis_all

    def remove_islands(self, im, cluster, max_sweeps=50, patience=1):
        comm, stable = self.comm, 0

        for _ in range(max_sweeps):
            flat = np.flatnonzero(np.isin(im.gids, cluster))

            if not scal_coll(comm.Allreduce, len(flat), op=mpi.SUM):
                break

            # Evict boundary cluster elements to their best neighbour
            best_nbr, best_cnt = self._contact_best(im)

            cand = flat[(best_cnt[flat] > 0) & ~im.frozen[flat]]
            cand = cand[np.lexsort((im.gids[cand], -best_cnt[cand]))]

            moved = self._move(im, cand, best_nbr[cand])
            moved = scal_coll(comm.Allreduce, moved, op=mpi.SUM)

            stable = 0 if moved else stable + 1
            if stable >= patience:
                break

    def _cores(self, im):
        # Per-rank arithmetic mean of the owned element centroids
        nd = im.cents.shape[1]

        sums = np.array(self.comm.allgather(im.cents.sum(axis=0)))
        cnts = im.counts()

        with np.errstate(invalid='ignore'):
            return sums / np.where(cnts > 0, cnts, np.nan)[:, None]

    def _top_fraction(self, im, cand, score, frac):
        nsel = min(max(int(np.ceil(frac*len(cand))), 1), len(cand))
        order = np.lexsort((im.gids[cand], -score))

        return cand[order[:nsel]]

    def remove_outliers(self, im):
        # Evict boundary elements far from our own centroid which have a
        # neighbour they are at least as tied to as ourselves
        if self.outlier_frac <= 0:
            return

        gate = 0.0 if self.outlier_mode == 'faces' else None
        best_nbr, _ = self._best_nbr(im, self.outlier_mode, score_max=gate)

        cores = self._cores(im)
        cand = np.flatnonzero((best_nbr >= 0) & ~im.frozen)

        if len(cand) and np.isfinite(cores[self.rank]).all():
            d = np.linalg.norm(im.cents[cand] - cores[self.rank], axis=1)
            z = (d - d.mean()) / d.std() if d.std() > 0 else np.zeros_like(d)

            chosen = self._top_fraction(im, cand, z, self.outlier_frac)
            self._move(im, chosen, best_nbr[chosen])
        else:
            im.move(np.full(im.neles, self.rank, dtype=int))

    def add_inliers(self, im):
        # Pull boundary elements towards the neighbour with the closest
        # centroid, when it is closer than our own
        if self.inlier_frac <= 0:
            return

        flat, _, nbr, _ = self._candidates(im, self.inlier_mode, np.inf)

        cores = self._cores(im)
        best_dst = np.full(im.neles, -1, dtype=int)
        best_d = np.full(im.neles, np.inf)

        for i in np.argsort(nbr, kind='stable'):
            if not np.isfinite(cores[nbr[i]]).all():
                continue

            d = np.linalg.norm(im.cents[flat[i]] - cores[nbr[i]])
            if d < best_d[flat[i]]:
                best_d[flat[i]] = d
                best_dst[flat[i]] = nbr[i]

        cand = np.flatnonzero((best_dst >= 0) & ~im.frozen)

        if len(cand) and np.isfinite(cores[self.rank]).all():
            d_self = np.linalg.norm(im.cents[cand] - cores[self.rank],
                                    axis=1)
            score = d_self - best_d[cand]

            cand, score = cand[score > 0], score[score > 0]
            if len(cand):
                chosen = self._top_fraction(im, cand, score,
                                            self.inlier_frac)
                self._move(im, chosen, best_dst[chosen])
                return

        im.move(np.full(im.neles, self.rank, dtype=int))

    def heal(self, im, target, max_iters=-1):
        # Alternate island eviction, outlier/inlier moves, and capped
        # diffusion until every rank is down to one main component
        self.remove_outliers(im)

        iters = 0
        while max_iters == -1 or iters < max_iters:
            iters += 1

            cluster, nis_all = self.detect_islands(im)
            if all(n == 1 for n in nis_all):
                break

            self.remove_islands(im, cluster)
            self.remove_outliers(im)
            self.add_inliers(im)

            cur = im.counts()
            self.converge(im, target, max_iters=10)

            if (im.counts() == cur).all() or all(n <= 2 for n in nis_all):
                break

    # -- Top level

    def balance(self, im, target):
        self.drain(im, target)
        self.heal(im, target)

        # Iterate, aggressively so on early calls
        if self.aggr_iters > 0:
            for _ in range(self.aggr_iters):
                self.iterate(im, target)

            self.aggr_iters -= 1
        else:
            self.iterate(im, target)

    def seed(self, im, new_rank, target):
        # Give an empty rank one element, taken from the interface of
        # the most overloaded donor rank
        surplus = im.counts() - np.asarray(target, dtype=int)
        surplus[new_rank] = np.iinfo(int).min
        donor = int(np.argmax(surplus))

        dests = np.full(im.neles, self.rank, dtype=int)
        if self.rank == donor and im.neles:
            _, contact = self._contact_best(im)

            cand = np.flatnonzero(~im.frozen)
            if not len(cand):
                cand = np.arange(im.neles)

            best = cand[np.lexsort((im.gids[cand], -contact[cand]))[0]]
            dests[best] = new_rank

        im.move(dests)

    def shuffle(self, im, target, cost, worst):
        '''
        Escape a local optimum by draining the worst-performing rank
        into the cheapest one, then re-seeding it with a single element
        from which subsequent balancing cycles can regrow it.
        '''
        cur = im.counts()
        tgt = np.asarray(target, dtype=int)
        active = np.flatnonzero((tgt > 0) & (np.arange(len(tgt)) != worst))

        if not len(active):
            return self.balance(im, target)

        # Rank removal: retarget the worst rank's mass at the cheapest
        drain = tgt.astype(float)
        drain[active[np.argmin(cost[active])]] += cur[worst]
        drain[worst] = 0
        drain = int_round(drain, im.nglobal)

        self.drain(im, drain)
        self.iterate(im, drain)

        # Rank addition: mean target, one seed element, short settle
        readd = drain.astype(float)
        readd[worst] = readd[active].mean()
        readd = int_round(readd, im.nglobal)

        if im.counts()[worst] == 0:
            self.seed(im, worst, readd)

        self.converge(im, readd, max_iters=5)
