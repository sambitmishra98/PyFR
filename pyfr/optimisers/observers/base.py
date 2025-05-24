import statistics

import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import init_csv

class BaseObserver:
    name = None
    objective = None
    
    def __init__(self, intg, cfgsect, suffix=None):
        self.cfg = intg.cfg
        self.cfgsect = cfgsect
        
        self.suffix = suffix

        self.tprev = intg.tcurr
        self._hist = np.empty((0, 4), dtype=np.float64)

        # Initialise 
        self.cost_list = []
        self.config_prepare = False 
        self.config_change = False 

        self.interval = self.cfg.getint(self.cfgsect, 'capture-interval', 0)

        # MPI info
        comm, rank, root = get_comm_rank_root()

        if rank == root and intg.cfg.hasopt(cfgsect, 'file'):
            self.outf = init_csv(intg.cfg, cfgsect, 
                                header='tprev,tcurr,count,mean,stdev,median')
        else:
            self.outf = None

    def __call__(self, intg):

        self.accumulate_cost(intg)

        if self.__update_condition(intg):
            # Config change always performed by sampler
            # If no sampler, periodically print to csv for offline optimisation            

            stats = self.calculate_cost_stats()
            self._hist = np.append(self._hist, np.array([stats]), axis=0)

            if self.outf:
                print(self.tprev, intg.tcurr, *stats, sep=',', file=self.outf)
                self.outf.flush()

            self.cost_list = []

            # Update the previous time
            self.tprev = intg.tcurr

    def __update_condition(self, intg):

        if self.interval == 0:
            return self.config_prepare
        else:
            return intg.nsteps % self.interval == 0

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
        # Skip first 30% of the interval
        self._skip_initial = int(0.5 * y)

        self._interval = y

    def accumulate_cost(self, intg):
        self.cost_list.append(self.observation(intg))

    def calculate_cost_stats(self):
        clist = self.cost_list[self._skip_initial:]

        mean = statistics.mean(clist) if clist else 0
        stdev = statistics.stdev(clist, mean) if len(clist) >= 2 else 0
        median = statistics.median(clist) if clist else 0

        stats = [len(self.cost_list), mean, stdev, median]

        return stats

    @property
    def stats_hist(self) -> np.ndarray[np.float64]:
        return self._hist

    def reset_cost(self):
        self.cost_list = []

class BaseObjective(BaseObserver):

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        # Get the objective function
        if self.objective not in ['minimise', 'maximise', 'equalise']:
            raise ValueError(f'Invalid objective: {self.objective}')
