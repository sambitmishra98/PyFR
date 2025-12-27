
import gc
import math

from pyfr.mpiutil import execute, comm, mpi
from pyfr.partitioners.online.base import _MeshInterconnector
from pyfr.util import first

from pyfr.integrators.base import BaseIntegrator, _common_plugin_prop
from pyfr.integrators.dual.pseudo import get_pseudo_integrator


class BaseDualIntegrator(BaseIntegrator):
    formulation = 'dual'

    def __init__(self, backend, systemcls, mmesh, initsoln, cfg):
        super().__init__(backend, mmesh, initsoln, cfg)

        # Get the pseudo-integrator
        self.pseudointegrator = get_pseudo_integrator(
            backend, systemcls, mmesh, initsoln, cfg, self.stepper_nregs,
            self.stage_nregs, self.dt
        )

        # Copy over some system attributes to ranks with empty system
        self.copy_to_empty_system()

        self.ele_map_plugins = self.system._setup_elemap(self.meshes['plugins'])

        # Event handlers for advance_to
        self.plugins = self._get_plugins(initsoln)

        # Commit the pseudo integrators now we have the plugins
        execute['compute'](lambda: self.pseudointegrator.commit())


    @property
    def system(self):
        return self.pseudointegrator.system

    @property
    def pseudostepinfo(self):
        return self.pseudointegrator.pseudostepinfo

    @_common_plugin_prop('_curr_soln', edim=2)
    def soln(self):
        return self.system.ele_scal_upts(self.pseudointegrator._idxcurr)

    @property
    def compute_soln(self):
        p = execute['compute'](lambda: self.system.ele_scal_upts(self.pseudointegrator._idxcurr),
                    default = None)

        return p

    def copy_to_empty_system(self):

        # If already copied, do nothing
        if hasattr(self, 'convars'):
            return
       
        if comm['compute'] != mpi.COMM_NULL:
            convars = list(first(self.system.ele_map.values()).convars)
        else:
            convars = []

        convars = execute['compute'](lambda: list(first(self.system.ele_map.values()).convars),
                          default = [])

        # Ensure the empty systems have what plugins expect.
        convars = comm['world'].allgather(convars)
        self.convars = list(first(c for c in convars if c))


    def reinit_backend_and_system(self, mesh, soln):

        # Carefully switch all work into pseudointegrator
        self._invalidate_caches()

        for attr in dir(self):
           if attr.startswith('_memoize_cache@'):
               delattr(self, attr) 

        self.pseudointegrator.reinit_backend_and_system(mesh, soln)

        self.copy_to_empty_system()

        # Re-initialise plugin comm and interconnector
        self.initialise_comm_and_partition(goal='plugins')
        self._plugins_intercon = _MeshInterconnector(self.meshes['compute'].eidxs, 
                                                     self.meshes['plugins'].eidxs)

        self.plugins = self._reget_plugins()
        comm['world'].barrier()

    @_common_plugin_prop('_curr_grad_soln', edim=3)
    def grad_soln(self):
        self.system.compute_grads(self.tcurr, self.pseudointegrator._idxcurr)
        return [e.get() for e in self.system.eles_vect_upts]

    @_common_plugin_prop('_curr_dt_soln', edim=2)
    def dt_soln(self):
        soln = self.soln

        idx = self.pseudointegrator._idxcurr
        self.system.rhs(self.tcurr, idx, idx)

        dt_soln = self.system.ele_scal_upts(idx)

        # Reset current register with original contents
        for e, s in zip(self.system.ele_banks, soln):
            e[idx].set(s)

        return dt_soln

    def call_plugin_dt(self, tstart, dt):
        rem = math.fmod(dt, self.dt)
        tol = 5.0*self.dtmin
        if rem > tol and (self.dt - rem) > tol:
            raise ValueError('Plugin call times must be multiples of dt')
        
        rem_tstart = math.fmod(tstart, self.dt)
        if rem_tstart > tol and (self.dt - rem_tstart) > tol:
            raise ValueError('Plugin start times must be multiples of dt')

        super().call_plugin_dt(tstart, dt)

    def collect_stats(self, stats):
        super().collect_stats(stats)

        self.pseudointegrator.collect_stats(stats)
