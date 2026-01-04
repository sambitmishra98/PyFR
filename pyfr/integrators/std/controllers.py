import math

import numpy as np

from time import perf_counter_ns

from pyfr.integrators.std.base import BaseStdIntegrator
from pyfr.mpiutil import (mpi, comm, execute)

class BaseStdController(BaseStdIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure the system is compatible with our formulation/controller
        self.system.elementscls.validate_formulation(self)

        # Solution filtering frequency
        self._fnsteps = self.cfg.getint('soln-filter', 'nsteps', '0')

        # Stats on the most recent step
        self.stepinfo = []

        # Fire off any event handlers if not restarting
        if not self.isrestart:
            self._run_plugins()

    def _accept_step(self, dt, idxcurr, err=None):
        self._note_step_accepted(True)
        self.tcurr += dt
        self.nacptsteps += 1
        self.nacptchain += 1
        self.stepinfo.append((dt, 'accept', err))

        self._idxcurr = idxcurr

        # Filter
        if self._fnsteps and self.nacptsteps % self._fnsteps == 0:
            self.system.filt(idxcurr)

        self._invalidate_caches()

        # Run any plugins
        self._run_plugins()

        # Clear the step info
        self.stepinfo = []

    def _reject_step(self, dt, idxold, err=None):
        self._note_step_accepted(False)
        if dt <= self.dtmin:
            raise RuntimeError('Minimum sized time step rejected')

        self.nacptchain = 0
        self.nrjctsteps += 1
        self.stepinfo.append((dt, 'reject', err))

        self._idxcurr = idxold


class StdNoneController(BaseStdController):
    controller_name = 'none'
    controller_has_variable_dt = False

    @property
    def controller_needs_errest(self):
        return False

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t:

            # Decide on the time step
            self.adjust_dt(t)

            idxcurr = execute['compute'](lambda: self.step(self.tcurr, self.dt),
                                         default=-1)
            idxcurr = comm['world'].allreduce(idxcurr, op=mpi.MAX)

            self._errest_tdiff_hist.append(0)
            # We are not adaptive, so accept every step
            self._accept_step(self.dt, idxcurr)

            self.load_balance()


class StdPIController(BaseStdController):
    controller_name = 'pi'
    controller_has_variable_dt = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sect = 'solver-time-integrator'

        # Error tolerances
        self._atol = self.cfg.getfloat(sect, 'atol')
        self._rtol = self.cfg.getfloat(sect, 'rtol')

        if self._atol < 10*self.backend.fpdtype_eps:
            raise ValueError('Absolute tolerance too small')

        if self._rtol < 10*self.backend.fpdtype_eps:
            raise ValueError('Relative tolerance too small')

        # Error norm
        self._norm = self.cfg.get(sect, 'errest-norm', 'l2')
        if self._norm not in {'l2', 'uniform'}:
            raise ValueError('Invalid error norm')

        # PI control values
        self._alpha = self.cfg.getfloat(sect, 'pi-alpha', 0.58)
        self._beta = self.cfg.getfloat(sect, 'pi-beta', 0.42)

        # Estimate of previous error
        self._errprev = 1.0

        # Step size adjustment factors
        self._saffac = self.cfg.getfloat(sect, 'safety-fact', 0.8)
        self._maxfac = self.cfg.getfloat(sect, 'max-fact', 2.5)
        self._minfac = self.cfg.getfloat(sect, 'min-fact', 0.3)

        if not self._minfac < 1 <= self._maxfac:
            raise ValueError('Invalid max-fact, min-fact')

    @property
    def controller_needs_errest(self):
        return True

    def _errest(self, rcurr, rprev, rerr):

        # Get a set of kernels to estimate the integration error
        ekerns = self._get_reduction_kerns(rcurr, rprev, rerr, method='errest',
                                           norm=self._norm)

        # Bind the dynamic arguments
        for kern in ekerns:
            kern.bind(self._atol, self._rtol)

        # Run the kernels
        self.backend.run_kernels(ekerns, wait=True)

        # Pseudo L2 norm
        if self._norm == 'l2':
            # Reduce locally (element types + field variables)
            err = np.array([sum(v for k in ekerns for v in k.retval)])

            # Reduce globally (MPI ranks)
            comm['world'].Allreduce(mpi.IN_PLACE, err, op=mpi.SUM)

            # Normalise
            err = math.sqrt(float(err) / self._gndofs)
        # Uniform norm
        else:
            # Reduce locally (element types + field variables)
            err = np.array([max(v for k in ekerns for v in k.retval)])

            # Reduce globally (MPI ranks)
            comm['world'].Allreduce(mpi.IN_PLACE, err, op=mpi.MAX)

            # Normalise
            err = math.sqrt(float(err))

        return err if not math.isnan(err) else 100

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        # Constants
        maxf = self._maxfac
        minf = self._minfac
        saff = self._saffac
        sord = self.stepper_order

        expa = self._alpha / sord
        expb = self._beta / sord

        while self.tcurr < t:
            # Adjust current time step per target t
            self.adjust_dt(t)

            self.dt = max(self.dt, self.dtmin)

            # Decide on the time step
            dt = max(min(t - self.tcurr, self.dt), self.dtmin)

            # Take the step
            idxcurr, idxprev, idxerr = execute['compute'](lambda: self.step(self.tcurr, dt),
                                                          default=(-1, -1, -1))

            idxcurr = comm['world'].allreduce(idxcurr, op=mpi.MAX)
            idxprev = comm['world'].allreduce(idxprev, op=mpi.MAX)
            idxerr  = comm['world'].allreduce(idxerr,  op=mpi.MAX)

            #self.backend.wait()
            tstart = perf_counter_ns()

            # Estimate the error
            err = self._errest(idxcurr, idxprev, idxerr)

            #self.backend.wait()
            self._errest_tdiff_hist.append((perf_counter_ns() - tstart)*1e-9)

            # Decide if to accept or reject the step
            if err < 1.0:
                self._errprev = err
                self._accept_step(self.dt, idxcurr, err=err)
                self.load_balance()
            else:
                self._reject_step(self.dt, idxprev, err=err)

            # Adjust time step per PI controller
            fac = err**-expa * self._errprev**expb
            fac = min(maxf, max(minf, saff*fac))

            # Compute the next time step
            self.dt_fallback = fac*self.dt

