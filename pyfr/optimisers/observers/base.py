# pyfr/optimisers/observers/base.py
from __future__ import annotations

from pyfr.mpiutil import get_comm_rank_root
from pyfr.optimisers.base import HistoryMixin, FlagSyncMixin

class BaseObserver(FlagSyncMixin, HistoryMixin):
    name = None
    objective = None

    def __init__(self, intg, cfgsect: str, suffix: str | None = None):
        FlagSyncMixin.__init__(self, intg, suffix)
        self.cfg, self.cfgsect, self.suffix = intg.cfg, cfgsect, suffix
        self.tprev = intg.tcurr

        self._init_observer(intg)

        comm, rank, root = get_comm_rank_root()
        if rank == 0 and intg.cfg.hasopt(cfgsect, 'file'):
            self._csv = self.init_csv(intg.cfg, cfgsect, 
                                      header=','.join(self._cols_hdr))
        else:
            self._csv = None

    def __call__(self, intg):
        if not self.should_capture(intg.nsteps):
            return

        row_body = self.observation(intg)
        row = [self.tprev, intg.tcurr, *row_body]
        self.append_row(row)

        # dump immediately (cheap – one row only)
        if self._csv:
            print(*row, sep=',', file=self._csv)
            self._csv.flush()

        self.tprev = intg.tcurr      # advance window

    def _init_observer(self, intg):
        pass

    def observation(self, intg) -> list[float]:
        raise NotImplementedError('observer must implement observation')
