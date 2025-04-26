import itertools as it

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BaseSolnPlugin, init_csv

import numpy as np

class PseudoResidualStatsPlugin(BaseSolnPlugin):

    """Outputs statistics about the pseudo-timestep (res) field.

    This plugin is designed for dual time-stepping schemes.
    It tracks min-max pseudo-timestep values across the entire field
    pseudo-iterations within each physical time step.

    The statistics can be output at different levels of granularity controlled
    by the `abstraction` parameter in the configuration file.

    Configuration options in the .ini file under `[soln-plugin-resstats]`:

    - `abstraction`: (integer, default=0) Controls the level of detail in the
      output CSV file.
        - 0: Outputs global min/max res across all variables and element types.
        - 1: Outputs min/max res per variable, aggregated across element types.
        - 2: Outputs min/max res per variable and per element type.
          (Note: Higher values are not currently implemented).
    - `flushsteps`: (integer, default=500) Number of physical time steps
      after which the output CSV file is flushed to disk.
    - `level`: (integer, optional) Specifies the multigrid level to analyze
      when using the `solver-dual-time-integrator-multip` integrator.
      Defaults to the main solver order (`[solver]order`).

    The output is written to a CSV file named based on the plugin configuration
    prefix (e.g., `resstats_*.csv`). The columns include the pseudo-step count,
    current physical time, and the requested res statistics based on the
    `abstraction` level.
    """

    name = 'pseudoresidualstats'
    systems = ['*']
    formulations = ['dual']
    dimensions = [2,3]

    def __init__(self, intg, cfgsect, prefix):
        super().__init__(intg, cfgsect, prefix)

        self.stats = []

        self.t_p = intg.tcurr

        self.fvars   = intg.system.elementscls.convars(self.ndims, self.cfg)
        self.e_types = intg.system.ele_types

        # Maximum of 3 Levels of abstraction for the stats of pseudo-dt field

        self.res_stats = { 'n' : {'all':0},
                           'min': {'all':0}|{p:{'all':0}|{e:{'all':0} for e in self.e_types} for p in self.fvars}, 
                           'max': {'all':0}|{p:{'all':0}|{e:{'all':0} for e in self.e_types} for p in self.fvars},
                         }

        self.abstraction = self.cfg.getint(self.cfgsect, 'abstraction', 0)

        if self.abstraction > 3:
            raise NotImplementedError(f'abstraction > {self.abstraction} not implemented')

        if 'solver-dual-time-integrator-multip' in intg.cfg.sections():
            self.level = self.cfg.getint(self.cfgsect, 
                'level', intg.cfg.getint('solver','order'))

        csv_header =  'pseudo-steps, tcurr'

        for k, v in self.res_stats.items():
            if k != 'all':
                csv_header += f',{k}'
                for vk, vv in v.items():
                    if vk != 'all' and self.abstraction>1:
                        csv_header += f',{k}_{vk}'
                        for vvk in vv.keys():
                            if vvk != 'all' and self.abstraction>2:
                                csv_header += f',{k}_{vk}_{vvk}'

        # MPI info
        self.comm, self.rank, self.root = get_comm_rank_root()

        # The root rank needs to open the output file
        if (self.rank == self.root):
            self.flushsteps = self.cfg.getint(self.cfgsect, 'flushsteps', 500)
            self.outf = init_csv(self.cfg, cfgsect, csv_header)
        else:
            self.outf = None

        self.stored_t_p = None
        self.last_appendable = None

    def get_pseudo_residual_field(self, intg):
        """
        Get difference of registers at the highest polynomial level
        """

        if 'solver-dual-time-integrator-multip' in intg.cfg.sections():
            pintg = intg.pseudointegrator.pintgs[self.level]
        else:
            pintg = intg.pseudointegrator
            
        res = []

        for em, dtaum in it.zip_longest(pintg.system.ele_banks, pintg.dtau_upts):
            r_prev = em[pintg._idxprev].get()
            r_curr = em[pintg._idxcurr].get()
            res.append((r_curr - r_prev)/dtaum.get())

        return res

    def __call__(self, intg):
        # Process the sequence of pseudo-residuals

        for (npiter, iternr, resid) in intg.pseudostepinfo:

            if iternr == 1:           # We can store the last step's data

                if self.last_appendable != None:
                    if self.stored_t_p != self.t_p:
                        self.stats.append((f for f in self.last_appendable))
                        self.prev_npiter = npiter - 1
                else:
                    self.prev_npiter = 0 

                self.stored_t_p = self.t_p

            self.res_stats['n']['all'] = npiter-self.prev_npiter

            self.pseudo_res_statistics(intg)
            #self.residual_statistics(intg, resid)
            self.last_appendable = (npiter, intg.tcurr, 
                                    *self.pseudores_stats_as_list(self.res_stats))

        self.t_p = intg.tcurr

        # If we're the root rank then output
        if self.outf:
            for s in self.stats:
                print(*s, sep=f',', file=self.outf)

            # Periodically flush to disk
            if intg.nacptsteps % self.flushsteps == 0:
                self.outf.flush()

        # Reset the stats
        self.stats = []

    def _reduce(self, arr, op):
        """MPI-reduce wrapper: MIN or MAX into in-place on root."""
        if self.rank == self.root:
            self.comm.Reduce(mpi.IN_PLACE, arr, op=op, root=self.root)
        else:
            self.comm.Reduce(arr,        None, op=op, root=self.root)

    def pseudo_res_statistics(self, intg):
        '''
            self.deltau_mats is a list of matrices, one for each element type.
            Each matrix is a 3D array of shape (nupts, nvars, neles)
        '''

        deltau_mats = self.get_pseudo_residual_field(intg)


        for j, var in enumerate(self.fvars):
            for i, e_type in enumerate(self.e_types):

                # each element type, each soln point in element, each variable in (p, u, v, w)
                # Stats obtained over all elements
                self.res_stats['min'][var][e_type]['each'] = (deltau_mats[i][:, j, :].min(1))
                self.res_stats['max'][var][e_type]['each'] = (deltau_mats[i][:, j, :].max(1))

                # each element type, each variable in (p, u, v, w)
                # Stats obtained over all elements and element soln points

                self.res_stats['min'][var][e_type]['all'] = (self.res_stats['min'][var][e_type]['each'].min())
                self.res_stats['max'][var][e_type]['all'] = (self.res_stats['max'][var][e_type]['each'].max())

            # each variable in (p, u, v, w)
            # Stats obtained over all element types, elements and element soln points
            self.res_stats['min'][var]['all'] = min([self.res_stats['min'][var][e_type]['all'] for e_type in self.e_types])
            self.res_stats['max'][var]['all'] = max([self.res_stats['max'][var][e_type]['all'] for e_type in self.e_types])

            t_min = np.array(self.res_stats['min'][var]['all']) ; self._reduce(t_min, mpi.MIN) ; self.res_stats['min'][var][e_type]['all'] = t_min
            t_max = np.array(self.res_stats['max'][var]['all']) ; self._reduce(t_max, mpi.MAX) ; self.res_stats['max'][var][e_type]['all'] = t_max

        # Stats obtained over all element types, elements, variable in (p, u, v, w) and element soln points
        self.res_stats['min']['all'] = min([self.res_stats['min'][var]['all'] for var in self.fvars])
        self.res_stats['max']['all'] = max([self.res_stats['max'][var]['all'] for var in self.fvars])

    def pseudores_stats_as_list(self, deltau_stats):
        deltau_stats_list = []
        for v in deltau_stats.values():
            deltau_stats_list.append(v['all'])               
            for vk, vv in v.items():
                if self.abstraction>1:                   
                    if isinstance(vv, dict):    
                        deltau_stats_list.append(vv['all'])
                if vk != 'all':
                    for vvv in vv.values():
                        if self.abstraction>2:
                            if isinstance(vvv, dict):
                                deltau_stats_list.append(vvv['all'])

        return deltau_stats_list
