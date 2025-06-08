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
        self._hist = []

        # Initialise 
        self.cost_list = []
        self.config_prepare = False 
        self.config_change = False 

        self.interval = self.cfg.getint(self.cfgsect, 'capture-interval', 0)

        # MPI info
        comm, rank, root = get_comm_rank_root()

        if rank == root and intg.cfg.hasopt(cfgsect, 'file'):
            cols = [f'mean-{r}' for r in range(comm.size)] + \
                   [f'sem-{r}'  for r in range(comm.size)]
            self.outf = init_csv(intg.cfg, cfgsect, 
                                 header='tprev,tcurr,' + ','.join(cols),)
        else:
            self.outf = None

    def __call__(self, intg):
        if self.__update_condition(intg):
            stats = self.allgather_mean_sem(intg)   # flat list [means… sems…]
            self._hist.append(stats)                # just append to Python list

            if self.outf:
                print(self.tprev, intg.tcurr, *stats, sep=',', file=self.outf)
                self.outf.flush()

            # Update
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
        self._interval = y

    def reset_cost(self):
        self.cost_list = []

class BaseObjective(BaseObserver):

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        # Get the objective function
        if self.objective not in ['minimise', 'maximise', 'equalise']:
            raise ValueError(f'Invalid objective: {self.objective}')
