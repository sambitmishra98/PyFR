from typing import Sequence

import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import init_csv

class BaseHyperparameter:
    name = None
    
    def __init__(self, intg, cfgsect, suffix=None):
        self.intg = intg
        self.cfgsect = cfgsect
        
        self.suffix = suffix

        self.n_hparams = intg.cfg.getint(cfgsect, 'n-hyperparameters')
        sbounds = intg.cfg.getliteral(cfgsect, 'soft-bounds')
        hbounds = intg.cfg.getliteral(cfgsect, 'hard-bounds')
        self.bounds = np.array(sbounds).transpose().reshape(2, -1)
        
        print(f'Hyperparameter soft-bounds: {self.bounds}')

        self._hist = []
        self.tprev = intg.tcurr

        # Initialise 
        self.config_prepare = False 
        self.config_change = False 

        self.interval = intg.cfg.getint(self.cfgsect, 'capture-interval', 0)

        # Candidate obtained from sampler
        self.hparam_new = None

        # MPI info
        comm, rank, root = get_comm_rank_root()

        if rank == root and intg.cfg.hasopt(cfgsect, 'file'):
                self.outf = init_csv(intg.cfg, cfgsect, header=self.csv_header)
        else:
            self.outf = None

        self.hparam_pending_update = []

    @property
    def csv_header(self):
        if self.n_hparams == 1:
            return 'tprev,tcurr,param'
        else:
            # Create a header with the number of hyperparameters
            return 'tprev,tcurr,' + ','.join([f'param{i}' for i in range(self.n_hparams)])

    @property
    def config_change(self):
        return self._config_change
    
    @config_change.setter
    def config_change(self, y):
        self._config_change = y

    @property
    def config_prepare(self):
        return self._config_prepare
    
    @config_prepare.setter
    def config_prepare(self, y):
        self._config_prepare = y

    @property
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, y):
        self._interval = y

    @property
    def stats_hist(self) -> np.ndarray:
        print(f"stats_hist: {self._hist}")
        return np.stack(self._hist, axis=0)

    def __call__(self):

        if self.__update_condition(self.intg):
            # If float, then do not try to copy
            self._hist.append(self.hparam)

            if self.outf:
                print(self.tprev, self.intg.tcurr, 
                      *self.param, sep=',', file=self.outf)
                self.outf.flush()

            # Update the previous time
            self.tprev = self.intg.tcurr

        if len(self.hparam_pending_update):
            pending = self.hparam_pending_update.pop(0)
            self.hparam = np.asarray(pending, dtype=float)
            self.config_change = True

    def __update_condition(self, intg):

        if self.interval == 0:
            return self.config_prepare
        else:
            return intg.nsteps % self.interval == 0

    @staticmethod
    def condense_by_group(full: Sequence[float], groups: Sequence[int]) -> np.ndarray:
        """
        full: tuple/list of length N
        groups: same length, labels in -1,0,1,...,G-1
        Returns: array of length G with first-occurrence values.
        """
        arr = np.array(full, dtype=float)
        # Map each group≥0 to its first index
        grp_to_idx = {}
        for idx, g in enumerate(groups):
            if g < 0:
                continue
            if g not in grp_to_idx:
                grp_to_idx[g] = idx

        G = max(grp_to_idx) + 1 if grp_to_idx else 0
        cc = np.empty(G, dtype=float)
        for g, idx in grp_to_idx.items():
            cc[g] = arr[idx]
        return cc    