import gc

from pyfr.integrators.base import BaseIntegrator, _common_plugin_prop
from pyfr.integrators.base import BaseCommon
from pyfr.mpiutil import execute, comm
from pyfr.partitioners.online.base import _MeshInterconnector

class BaseStdIntegrator(BaseCommon, BaseIntegrator):
    formulation = 'std'

    def __init__(self, backend, systemcls, mmesh, initsoln, cfg):
        super().__init__(backend, mmesh, initsoln, cfg)

        # Sanity checks
        if self.controller_needs_errest and not self.stepper_has_errest:
            raise TypeError('Incompatible stepper/controller combination')

        # Determine the amount of temp storage required by this method
        self.nregs = self.stepper_nregs

        # Construct the relevant system
        self.system = systemcls(backend, mmesh.mesh, initsoln, nregs=self.nregs,
                                cfg=cfg)

        self._systemcls = systemcls

        # Copy over some system attributes to ranks with empty system
        self.copy_to_empty_system()

        # Register index list and current index
        self._regidx = list(range(self.nregs))
        self._idxcurr = 0

        self.ele_map_plugins = self.system._setup_elemap(self.meshes['plugins'])

        # Event handlers for advance_to
        self.plugins = self._get_plugins(initsoln)

        execute['compute'](lambda: self.system.commit())
        execute['compute'](lambda: self.system.preproc(self.tcurr, self._idxcurr))

        # Global degree of freedom count
        self._gndofs = self._get_gndofs()

    def reinit_backend_and_system(self, mesh, soln):
        self._invalidate_caches()

        del self.system

        for attr in dir(self):
           if attr.startswith('_memoize_cache@'):
               delattr(self, attr) 

        gc.collect()

        comm['world'].barrier()

        self.system = self._systemcls(self.backend, mesh, soln, 
                                      nregs=self.nregs, cfg=self.cfg)

        self.copy_to_empty_system()

        self._idxcurr = 0

        # Re-initialise plugin comm and interconnector
        self.initialise_comm_and_partition(goal='plugins')
        self._plugins_intercon = _MeshInterconnector(self.meshes['compute'].eidxs, 
                                                     self.meshes['plugins'].eidxs)

        self.plugins = self._reget_plugins()

        execute['compute'](lambda: self.system.commit())
        execute['compute'](lambda: self.system.preproc(self.tcurr, self._idxcurr))

        comm['world'].barrier()

    @_common_plugin_prop('_curr_soln', edim=2)
    def soln(self):
        self.system.postproc(self._idxcurr)
        return self.system.ele_scal_upts(self._idxcurr)

    @property
    def compute_soln(self):

        execute['compute'](lambda: self.system.postproc(self._idxcurr))
        p = execute['compute'](lambda: self.system.ele_scal_upts(self._idxcurr),
                    default = None)

        return p

    @_common_plugin_prop('_curr_grad_soln', edim=3)
    def grad_soln(self):
        self.system.postproc(self._idxcurr)
        self.system.compute_grads(self.tcurr, self._idxcurr)
        return [e.get() for e in self.system.eles_vect_upts]

    @_common_plugin_prop('_curr_dt_soln', edim=2)
    def dt_soln(self):
        soln = self.soln

        self.system.rhs(self.tcurr, self._idxcurr, self._idxcurr)
        dt_soln = self.system.ele_scal_upts(self._idxcurr)

        # Reset current register with original contents
        for e, s in zip(self.system.ele_banks, soln):
            e[self._idxcurr].set(s)

        return dt_soln

    @property
    def controller_needs_errest(self):
        pass
