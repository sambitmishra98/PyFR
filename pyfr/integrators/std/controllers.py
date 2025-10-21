import math
import gc

from copy import deepcopy

import numpy as np

from pyfr.backends import get_backend

from pyfr.backends.base import backend
from pyfr.integrators.std.base import BaseStdIntegrator
from pyfr.mpiutil import get_comm_rank_root, initialise_new_comm, mpi

from pyfr.readers.native import _MetaMesh, NativeReader


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
            comm,  rank,  root  = get_comm_rank_root('world')
            ccomm, crank, croot = get_comm_rank_root('compute')

            # Decide on the time step
            dt = max(min(t - self.tcurr, self._dt), self.dtmin)

            if ccomm != mpi.COMM_NULL:
                # Take the step
                idxcurr = self.step(self.tcurr, dt)

            else:
                idxcurr = -1    
            # allreduce MAX idxcurr, we only need to keep one
            idxcurr = comm.allreduce(idxcurr, op=mpi.MAX)

            # We are not adaptive, so accept every step
            self._accept_step(dt, idxcurr)

            # Switch mesh here, after 5 steps
            if self.nacptsteps % self.lb_iters == 0 and not self.lb_iters == 1:

                print('Switching, nacptsteps = ', self.nacptsteps)

                mmesh = _MetaMesh.from_mesh(self.meshes['compute'])

                _MetaMesh.info(self.meshes['compute'])

                current_local = sum(len(eidxs) for eidxs in self.meshes['compute'].eidxs.values())
                ecurrs = np.asarray(comm.allgather(int(current_local)), dtype=np.int64)

                # Get target element distribution
                targets = self.get_target(ecurrs, scale=self.lb_target_scale)

                mmesh.iterate_to_convergence(targets, etype_order=self.etype_order,
                    flowmat_relax=self.lb_flowmat_relax, mask=self.twoway_mask)
                soln = self.reinit_mesh_soln(mmesh.to_mesh(mmesh.eidxs_j), self.compute_soln)
                self.reinit_backend_and_system(self.meshes['newcompute'], soln)

    def reinit_mesh_soln(self, mesh, soln):
        self.meshes['newcompute'] = mesh
        _MetaMesh.info(self.meshes['newcompute'])
        self._newcompute_intercon = self.initialise_interconnector('compute', 'newcompute')
        soln = self.relocate_ary(self._newcompute_intercon, soln, edim=2,
                                    src_name='compute', dst_name='newcompute')

        del self.meshes['compute']
        del self._plugins_intercon

        self.meshes['compute'] = self.meshes['newcompute']

        return soln

    def reinit_backend_and_system(self, mesh, soln):
        ccomm, crank, croot = get_comm_rank_root('compute')
        comm,  rank,  root  = get_comm_rank_root('world')

        self._invalidate_caches()

        del self.system

        for attr in dir(self):
           if attr.startswith('_memoize_cache@'):
               delattr(self, attr) 

        gc.collect()

        comm.barrier()

        self.system = self._systemcls(self.backend, mesh, soln, nregs=self.nregs, cfg=self.cfg)

        self.copy_to_empty_system()

        self._idxcurr = 0

        # Re-initialise plugin comm and interconnector
        self.initialise_comm_and_partition('plugins', construct_con=False)
        self._plugins_intercon = self.initialise_interconnector('compute', 'plugins')

        self.plugins = self._reget_plugins()

        if ccomm != mpi.COMM_NULL:
            # Commit the sytem
            self.system.commit()

            # Pre-process solution
            self.system.preproc(self.tcurr, self._idxcurr)

        comm.barrier()


class StdPIController(BaseStdController):
    controller_name = 'pi'
    controller_has_variable_dt = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sect = 'solver-time-integrator'

        # Maximum time step
        self.dtmax = self.cfg.getfloat(sect, 'dt-max', 1e2)

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
        comm, rank, root = get_comm_rank_root('compute')

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
            comm.Allreduce(mpi.IN_PLACE, err, op=mpi.SUM)

            # Normalise
            err = math.sqrt(float(err) / self._gndofs)
        # Uniform norm
        else:
            # Reduce locally (element types + field variables)
            err = np.array([max(v for k in ekerns for v in k.retval)])

            # Reduce globally (MPI ranks)
            comm.Allreduce(mpi.IN_PLACE, err, op=mpi.MAX)

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
            # Decide on the time step
            dt = max(min(t - self.tcurr, self._dt, self.dtmax), self.dtmin)

            # Take the step
            idxcurr, idxprev, idxerr = self.step(self.tcurr, dt)

            # Estimate the error
            err = self._errest(idxcurr, idxprev, idxerr)

            # Determine time step adjustment factor
            fac = err**-expa * self._errprev**expb
            fac = min(maxf, max(minf, saff*fac))

            # Compute the size of the next step
            self._dt = fac*dt

            # Decide if to accept or reject the step
            if err < 1.0:
                self._errprev = err
                self._accept_step(dt, idxcurr, err=err)
            else:
                self._reject_step(dt, idxprev, err=err)
