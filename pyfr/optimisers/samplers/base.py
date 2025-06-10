# pyfr/optimisers/samplers/base.py
from __future__ import annotations
import pathlib
from typing import Optional, List

from pyfr.mpiutil import get_comm_rank_root
from pyfr.optimisers.base import FlagSyncMixin, HistoryMixin, find_instance


class BaseSampler(FlagSyncMixin, HistoryMixin):
    """
    Common base class for samplers (Bayesian, random, grid, …).

    Sub-classes must override :meth:`propose` and may override :meth:`accept`.
    """
    name: str | None = None   # set in subclass

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def __init__(self, intg, cfgsect: str, suffix: Optional[str] = None):
        super().__init__(intg, suffix)          # shared flags!

        self.intg     = intg
        self.cfg      = intg.cfg
        self.cfgsect  = cfgsect
        self.suffix   = suffix

        # link to modeller & hyper-parameter
        mname = self.cfg.get(cfgsect, 'modeller')
        self.modeller = find_instance(intg.modellers, mname, suffix)
        self.hparam   = self.modeller.hparam
        self.n_hparams = self.hparam.n_hparams

        # history = iteration, *hparams
        header = ['iter'] + [f'h{i}' for i in range(self.n_hparams)]
        self._init_history(len(header), header)

        # cadence
        self.interval = self.cfg.getint(cfgsect, 'capture-interval', 1)

        # CSV (root rank only)
        comm, rank, root = get_comm_rank_root()
        self.csv_path: pathlib.Path | None = None
        if rank == root and self.cfg.hasopt(cfgsect, 'file'):
            self.csv_path = pathlib.Path(
                self.cfg.get(cfgsect, 'file')
            ).with_suffix('.csv')

        if rank == root:
            print(f"[Sampler:init] {self.name}-{suffix or ''} "
                  f"→ modeller={mname}-{suffix or ''}  interval={self.interval}",
                  flush=True)

    # ------------------------------------------------------------------ #
    # runtime entry
    # ------------------------------------------------------------------ #
    def __call__(self):
        # capture aligned via shared FlagSyncMixin.interval
        if not self.should_capture(self.intg.nsteps):
            return

        comm, rank, root = get_comm_rank_root()

        if rank == root:
            cand = self.propose()                     # subclass method
            if len(cand) != self.n_hparams:
                raise ValueError('proposed vector length mismatch')

            # record + dump
            self.append_row([self.intg.nsteps, *cand])
            if self.csv_path:
                self.dump_csv(self.csv_path, flush=True)

        else:
            cand = None

        # broadcast candidate to all ranks
        cand = comm.bcast(cand, root=root)

        # queue the update for the hyper-parameter helper
        self.hparam._pending_updates.append(cand)

        # flag chain
        self.config_prepare = True   # shared dict → modeller + HP see it

    # ------------------------------------------------------------------ #
    # API hooks
    # ------------------------------------------------------------------ #
    def propose(self) -> List[float]:
        """Return a new hyper-parameter candidate (len = n_hparams)."""
        raise NotImplementedError

    def accept(self, cand: List[float], loss_old: float, loss_new: float):
        """Optional acceptance test (e.g., MCMC)."""
        return True


class EmptySampler(BaseSampler):
    name = 'empty'

    def propose(self):
        return list(self.modeller._best_candidate())
