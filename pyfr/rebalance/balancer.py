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

        # Affinity mode for the opening aggressive pass of the schedule.
        # 'faces' is the default because it matches 'vertices' on final
        # spread, edge cut and time, while letting IndexMesh omit the node
        # tables entirely -- those are 81 of the 99 values per element that
        # move() would otherwise migrate.
        self.open_mode = cfg.get(sect, 'open-mode', 'faces')

        # Smooth after every k-th schedule step. Smoothing after every step
        # fights the diffusion and stops it reaching the target at all: on
        # c3900 it stalls at maxdev 420 where any stride >= 2 lands exactly,
        # and it is several times slower for the privilege. The final step
        # always smooths regardless, so a schedule never ends on an
        # unrefined boundary.
        self.smooth_every = cfg.getint(sect, 'smooth-every', 4)

        # Which IndexMesh tables this configuration will ever consult, so the
        # caller can tell IndexMesh to skip building and migrating the rest
        modes = {m for _, m in self.schedule()}
        if self.outlier_frac > 0:
            modes.add(self.outlier_mode)
        if self.inlier_frac > 0:
            modes.add(self.inlier_mode)

        self.needs_vaff = modes != {'faces'}
        self.needs_geom = self.outlier_frac > 0 or self.inlier_frac > 0

        # heal()'s island count is not guaranteed to descend monotonically to
        # one per rank: it has been observed to rise on the first iteration
        # and then oscillate, so neither of the original exit conditions ever
        # fires and the default max_iters=-1 loops forever.  Cap it, and stop
        # once the island count has failed to improve on its best for
        # `heal-patience` consecutive iterations.  0 disables heal entirely.
        #
        # Default is 0 (off). heal() runs before iterate(), so the islands
        # that the final iterate() creates are never seen by it: measured on
        # c3900 (drift and violent cases, offline and online), enabling or
        # disabling heal leaves the final partitioning bit-identical (same
        # cut, same island count) -- it only costs time (~2x on the drift
        # case offline). Confirmed with Sambit 2026-08-12 to default off;
        # outlier-fraction and inlier-fraction both default to 0 and so were
        # never exercised by this measurement, which is the reason this stays
        # a config option rather than being removed outright.
        self.heal_maxiters = cfg.getint(sect, 'heal-max-iters', 0)
        self.heal_patience = cfg.getint(sect, 'heal-patience', 3)

        # Moot while heal-max-iters defaults to 0, above. If heal is turned
        # back on, this runs it after iterate() instead of before, where it
        # can actually see the islands iterate() creates rather than ones
        # that no longer exist.
        self.heal_after = cfg.getbool(sect, 'heal-after-iterate', False)

    def schedule(self):
        return ([(6.0, self.open_mode), (2.0, 'faces')]
                + [(0.0, 'faces')]*10)

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

        if mode != 'faces' and im.nodes is None:
            raise RuntimeError(
                f'Affinity mode {mode!r} needs the IndexMesh node tables, '
                f'but it was built without them; see DiffusionBalancer.'
                f'needs_vaff'
            )

        fown = im.fown

        # Distinct face owners in one linear pass; the -1 padding is shifted
        # into bin 0
        tot = np.bincount(fown.ravel() + 1, minlength=self.comm.size + 1)
        owners = np.flatnonzero(tot) - 1
        nbrs = owners[(owners >= 0) & (owners != rank)]

        if mode == 'faces':
            # Only an element with a face to another rank can be a candidate,
            # so restrict to those before the per-neighbour passes
            isnbr = (fown >= 0) & (fown != rank)
            bnd = np.flatnonzero(isnbr.any(axis=1) & movable)

            fb = fown[bnd]
            aown = (fb == rank).sum(axis=1)
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
                b = (fb == nb).sum(axis=1)

                # aown and b are indexed in boundary space, so sel is too;
                # bnd maps back out, and movable is already folded into it
                sel = b > 0
                idx = bnd[sel]
            else:
                b = (np.isin(im.nodes, vsets[nb]) & valid).sum(axis=1)
                sel = (im.fown == nb).any(axis=1) & (b > 0) & movable
                idx = np.flatnonzero(sel)

            score = (aown[sel] - b[sel]).astype(float)
            keep = score <= thr

            flat = idx[keep]
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
        schedule = self.schedule()

        for i, (thr, mode) in enumerate(schedule):
            mflow = self.flow_plan(im, target)

            caps = np.ceil(mflow[self.rank]*relax).astype(int)
            self.diffuse(im, np.maximum(caps, 0), thr, mode)

            # Always smooth on the last step, whatever the stride
            last = i == len(schedule) - 1

            if smooth and (last or (i + 1) % self.smooth_every == 0):
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

        # Connected components by label propagation with path compression.
        # This was a pure-Python BFS over every owned element, and it runs at
        # least once per balance() even when there are no islands to find,
        # which made it the single largest cost in heal().
        #
        # Each round every vertex takes the smallest label in its
        # neighbourhood, then labels are compressed by pointer jumping, so
        # components collapse in O(log n) fully vectorised rounds.
        lab = np.arange(ne)

        # etab is grouped by source, so segment minima come from a reduceat.
        # Zero-degree vertices would give repeated segment starts (for which
        # reduceat returns the wrong thing), so they are excluded; they
        # contribute nothing anyway, and the segments either side stay
        # correctly bounded.
        has = np.diff(vtab) > 0
        starts = vtab[:-1][has]

        while True:
            new = lab.copy()

            if len(starts):
                new[has] = np.minimum(lab[has],
                                      np.minimum.reduceat(lab[etab], starts))

            # Pointer jumping: follow each label to its root
            while True:
                nxt = new[new]
                if np.array_equal(nxt, new):
                    break

                new = nxt

            if np.array_equal(new, lab):
                break

            lab = new

        # Compact root ids to 0..ncomp-1. np.unique returns roots ascending,
        # and a component's root is its smallest member index, so this
        # reproduces the old BFS's numbering (which assigned ids in ascending
        # order of first-encountered vertex) exactly.
        _, labels = np.unique(lab, return_inverse=True)
        labels = labels.reshape(-1)
        sizes = np.bincount(labels)

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
        nis_best, nis_stall = None, 0
        while max_iters == -1 or iters < max_iters:
            iters += 1

            cluster, nis_all = self.detect_islands(im)
            if all(n == 1 for n in nis_all):
                break

            # Stagnation criterion on the island count itself.  The existing
            # exits below test element movement and nis <= 2, neither of which
            # fires while nis_all merely oscillates.
            if self.heal_patience > 0:
                nis_now = sum(nis_all)

                if nis_best is None or nis_now < nis_best:
                    nis_best, nis_stall = nis_now, 0
                else:
                    nis_stall += 1

                if nis_stall >= self.heal_patience:
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

        if self.heal_maxiters and not self.heal_after:
            self.heal(im, target, max_iters=self.heal_maxiters)

        # Iterate, aggressively so on early calls
        if self.aggr_iters > 0:
            for _ in range(self.aggr_iters):
                self.iterate(im, target)

            self.aggr_iters -= 1
        else:
            self.iterate(im, target)

        if self.heal_maxiters and self.heal_after:
            self.heal(im, target, max_iters=self.heal_maxiters)

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
