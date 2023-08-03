from pyfr.integrators.base import BaseIntegrator
from pyfr.integrators.base import BaseCommon
from pyfr.integrators.dual.pseudo import get_pseudo_integrator


class BaseDualIntegrator(BaseCommon, BaseIntegrator):
    formulation = 'dual'

    def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg):
        super().__init__(backend, rallocs, mesh, initsoln, cfg)

        # Get the pseudo-integrator
        self.pseudointegrator = get_pseudo_integrator(
            backend, systemcls, rallocs, mesh, initsoln, cfg,
            self.stepper_nregs, self.stage_nregs, self._dt
        )

        # Event handlers for advance_to
        self.plugins = self._get_plugins(initsoln)

        # Commit the pseudo integrators now we have the plugins
        self.pseudointegrator.commit()

    @property
    def system(self):
        return self.pseudointegrator.system

    @property
    def pseudostepinfo(self):
        return self.pseudointegrator.pseudostepinfo

    @property
    def soln(self):
        if not self._curr_soln:
            self._curr_soln = self.system.ele_scal_upts(
                self.pseudointegrator._idxcurr
            )

        return self._curr_soln

    @property
    def grad_soln(self):
        system = self.system

        if not self._curr_grad_soln:
            system.compute_grads(self.tcurr, self.pseudointegrator._idxcurr)
            self._curr_grad_soln = [e.get() for e in system.eles_vect_upts]

        return self._curr_grad_soln

    def collect_stats(self, stats):
        super().collect_stats(stats)

        self.pseudointegrator.collect_stats(stats)
