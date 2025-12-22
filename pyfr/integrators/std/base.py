from pyfr.integrators.base import BaseIntegrator, _common_plugin_prop
from pyfr.integrators.base import BaseCommon
from pyfr.util import first

from pyfr.mpiutil import mpi, comm, execute

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

    def copy_to_empty_system(self):
        
        if comm['compute'] != mpi.COMM_NULL:
            convars = list(first(self.system.ele_map.values()).convars)
        else:
            convars = []

        convars = execute['compute'](lambda: list(first(self.system.ele_map.values()).convars),
                          default = [])

        # Ensure the empty systems have what plugins expect.
        convars = comm['world'].allgather(convars)
        self.convars = list(first(c for c in convars if c))

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
