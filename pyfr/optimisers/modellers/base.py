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

        colnames = self.initialise_csv_colnames()

        colnames = [f"h{i}" for i in range(self.n_hparams)] + ["y", "ystd"]
        self._init_history(self.n_hparams + 2, colnames=colnames)

        # processed views – subclasses may fill these
        self.X_processed: list[tuple] = []
        self.Y_processed: list[tuple] = []
        self.Ystd_processed: list[tuple] = []

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

    def initialise_csv_colnames(self):
        return [f"h{i}" for i in range(self.n_hparams)] + ["y", "ystd"]

    @property
    def _best_candidate(self):
        """
        Return the hyper-parameter vector (tuple) that achieved
        the *lowest* y so far.  None if no data yet.
        """
        if not self.history:
            return None

        d = self.n_hparams              # last two cols are y, ystd
        best_row = min(self.history, key=lambda r: r[d])   # r[d] is y
        return best_row[2:2+d]             # tuple(h0, h1, …)

    def __call__(self):
        if not self.should_capture(self.intg.nsteps):
            return

        # bail out gracefully if data not ready yet
        if not self.hparam.history or not self.observer.history:
            return

        iteration_data = self.hparam.history[-1][:2]
        X = self.hparam.history[-1][2:]
        Y = self.observer.history[-1][2:]

        x,y,s = self.process_model(X, Y)
        self.X_processed.append(   x)
        self.Y_processed.append(   y)
        self.Ystd_processed.append(s)

        self.append_row([*iteration_data, *x, *y, *s])
        
        if self._csv_path:
            self.dump_csv(self._csv_path, flush=True)

    def fit_model(self, x, y, ystd):
        raise NotImplementedError

    def process_model(self, Xs, Ys):
        NotImplementedError(f"Must implement process_model()!")

class EmptyModeller(BaseModeller):
    name = "empty"

    def fit_model(self, x, y, ystd):
        pass
