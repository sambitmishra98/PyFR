from __future__ import annotations

from typing import Optional
import pathlib

from pyfr.mpiutil import get_comm_rank_root
from pyfr.optimisers.base import FlagSyncMixin, BoundsMixin, HistoryMixin, find_instance

class BaseModeller(FlagSyncMixin, BoundsMixin, HistoryMixin):
    name: str | None = None

    def __init__(self, intg, cfgsect: str, suffix: Optional[str] = None):
        FlagSyncMixin.__init__(self, intg, suffix)

        self.intg, self.suffix = intg, suffix

        obs_name = self.intg.cfg.get(cfgsect, "observer")
        hyp_name = self.intg.cfg.get(cfgsect, "hyperparameter")

        self.observer = find_instance(intg.observers,       obs_name, suffix)
        self.hparam   = find_instance(intg.hyperparameters, hyp_name, suffix)
        self.n_hparams = self.hparam.n_hparams

        colnames = [f"h{i}" for i in range(self.n_hparams)] + ["y", "ystd"]
        self._init_history(self.n_hparams + 2, colnames=colnames)

        self.interval = self.intg.cfg.getint(cfgsect, "capture-interval", 0)

        comm, rank, root = get_comm_rank_root()
        self._csv_path: pathlib.Path | None = None
        if rank == root and intg.cfg.hasopt(cfgsect, 'file'):
            p = pathlib.Path(intg.cfg.get(cfgsect, 'file')).with_suffix('.csv')
            self._csv_path = p

        if rank == root:
            print(
                f"[Modeller:init] '{self.name or '<unnamed>'}' "
                f"obs={self.observer.name}:{self.observer.suffix or ''}  "
                f"hp={self.hparam.name}:{self.hparam.suffix or ''}  "
                f"interval={self.interval}",
                flush=True,
            )

    def _best_candidate(self):
        """
        Return the hyper-parameter vector with the smallest y seen so far.
        If no rows have been recorded yet, return None.
        """

        Xs = self.hparam.history
        Ys = self.observer.history
        # history row = [h0 … h_{d-1}, y, ystd]
        best_X = min(
            (Xs[i] for i in range(len(Xs)) if Ys[i][0] is not None),
            key=lambda x: Ys[Xs.index(x)][0],
            default=None
        )
        return best_X

    def __call__(self):
        if not self.should_capture(self.intg.nsteps):
            return

        # bail out gracefully if data not ready yet
        if not self.hparam.history or not self.observer.history:
            return

        Xs = self.hparam.history[-1]
        Ys = self.observer.history[-1][2:]
        
        self.append_row([*Xs, *Ys])
        
        if self._csv_path:
            self.dump_csv(self._csv_path, flush=True)

    def fit_model(self, x, y, ystd):
        raise NotImplementedError

class EmptyModeller(BaseModeller):
    name = "empty"

    def fit_model(self, x, y, ystd):
        pass
