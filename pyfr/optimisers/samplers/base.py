# pyfr/optimisers/samplers/base.py
from __future__ import annotations
import pathlib
from typing import Optional, List

from pyfr.mpiutil import get_comm_rank_root
from pyfr.optimisers.base import FlagSyncMixin, HistoryMixin, find_instance


class BaseSampler(FlagSyncMixin, HistoryMixin):
    name: str | None = None

    def __init__(self, intg, cfgsect: str, suffix: Optional[str] = None):
        FlagSyncMixin.__init__(self, intg, suffix)

        self.intg     = intg
        self.cfg      = intg.cfg
        self.cfgsect  = cfgsect
        self.suffix   = suffix

        # link to modeller & hyper-parameter
        mname = self.cfg.get(cfgsect, 'modeller')
        self.modeller = find_instance(intg.modellers, mname, suffix)
        self.observer = self.modeller.observer
        self.hparam   = self.modeller.hparam
        self.n_hparams = self.hparam.n_hparams

        # history = iteration, *hparams
        header = [f'h{i}' for i in range(self.n_hparams)]
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

    def __call__(self):
        if not self.should_capture(self.intg.nsteps):
            return

        if not self.observer.history:
            return

        # comm, rank, root = get_comm_rank_root()

        candidate = self.sample()

        self.append_row([*candidate])
        if self.csv_path:
            self.dump_csv(self.csv_path, flush=True)

        # queue the update for the hyper-parameter helper
        self.hparam._pending_updates.append(candidate)

        # flag chain
        self.config_prepare = True

    def sample(self) -> List[float]:
        raise NotImplementedError


class EmptySampler(BaseSampler):
    name = 'empty'

    def sample(self):
        return list(self.modeller._best_candidate or self.hparam.param)
