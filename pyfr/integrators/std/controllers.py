import math
import gc

from time import perf_counter_ns
import numpy as np

from pyfr.integrators.std.base import BaseStdIntegrator
from pyfr.mpiutil import (mpi, initialise_new_comm, 
                          comm, rank, root, rankmap, execute,
                          promote_comm)

from pyfr.readers.native import _MetaMesh

from pyfr.inifile import Inifile


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

        # Global union of etypes (keeps header complete)
        ets_local = tuple(self.meshes['compute'].eidxs.keys())
        etypes = sorted({et for ets in comm['world'].allgather(ets_local) for et in ets})
        self._lb_etypes = etypes  # cache for rows

        if rank['world'] == root['world']:
            with open('lb_walltimes.csv', 'a') as f:
                f.write('tcurr,others,iterate,reinit\n')
                self.wallt_end = perf_counter_ns()

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

    def load_balance(self):
        # Rebalance every lb_iters accepted steps, unless lb_iters == 1 sentinel
        if self.nacptsteps % self.lb_iters == 0 and not self.lb_iters == 1:
            # Read inifile  from /scratch/EFFORTS/LoadBalancer3/c3900/online.ini
            online_cfg = Inifile.load(self.cfg.get('partition', 'online-file'))
            part_ranklist = online_cfg.getliteral('partition', 'compute-ranklist')

            # Build / update 'newcompute' communicator
            initialise_new_comm('newcompute', part_ranklist)


            # 🔍 NEW: print communicator state after creating newcompute
            if rank['world'] == root['world']:
                print(
                    f"[comm-debug] init-newcompute: "
                    f"world_size={comm['world'].size} "
                    f"compute_size={comm['compute'].size} "
                    f"rankmap_compute={rankmap['compute']} "
                    f"newcompute_size={comm['newcompute'].size} "
                    f"rankmap_newcompute={rankmap['newcompute']}",
                    flush=True
                )

            wallt_start = perf_counter_ns()

            g1a, g1s, g1r = self.get_median_matrices()

            print('Switching, nacptsteps = ', self.nacptsteps)

            # Build MetaMesh over the *world* communicator
            mmesh = _MetaMesh.from_mesh(self.meshes['compute'])

            _MetaMesh.info(self.meshes['compute'])
            _MetaMesh.info_to_csv(self.meshes['compute'], tcurr=self.tcurr)

            # Element counts per world rank
            current_local = sum(len(eidxs) for eidxs in self.meshes['compute'].eidxs.values())
            ecurrs = np.asarray(comm['world'].allgather(int(current_local)), dtype=np.int64)

            print(f"[lb] world-rank {rank['world']} ecurrs={ecurrs.tolist()}")

            # After this, we measure only iterate+reinit
            self.wallt_end = perf_counter_ns()

            # Only newcompute ranks compute targets / mask
            targets = execute['newcompute'](lambda: self.calc_target_ecounts
                                            (ecurrs, g1a, g1s, g1r), default=None)
            twoway_mask = execute['newcompute'](lambda: g1s + g1r > 0, default=None)

            # Broadcast to all world ranks so everyone agrees
            targets     = comm['world'].bcast(targets)
            twoway_mask = comm['world'].bcast(twoway_mask)

            if rank['world'] == root['world']:
                print(f"mask:\n{twoway_mask.astype(int)}", flush=True)

            # Extend targets to world size using rankmap['newcompute']
            targets = [
                targets[rankmap['newcompute'].index(i)] if i in rankmap['newcompute'] else 0
                for i in range(comm['world'].size)
            ]

            # RANK REMOVAL STRATEGY:
            # Get all ranks with zero target and non-zero current, 
            # load balance until one of the ranks reaches the zero target.
                        
            ranks_to_clear = [i for i, (n, t) in enumerate(zip(ecurrs, targets))
                              if t == 0 and n > 0 ]

            if ranks_to_clear:
                mmesh.iterate("to-remove-rank", targets, mask=twoway_mask,
                              flowmat_relax=self.lb_flowmat_relax)
            

            else:

                # RANK ADDITION STRATEGY:
                # Get all ranks with non-zero target and zero current,
                # sequentially seed the mesh until
                #   all ranks are populated by at least one element.
                # Then continue with the normal load balancing.
                
                for i in range(comm['world'].size):
                    if targets[i] > 0 and ecurrs[i] == 0:
                        print(f"Seeding rank {i}", flush=True)
                        mmesh.seed_rank(i)
    
                # Run diffusion on MetaMesh
                mmesh.iterate("to-target", targets, mask=twoway_mask,
                                           flowmat_relax=self.lb_flowmat_relax)
    

            # Convert back to mesh, relocate solution, and reinit system
            soln = self.reinit_mesh_soln(mmesh.to_mesh(mmesh.eidxs_j), self.compute_soln)

            wallt_iterate = perf_counter_ns() - wallt_start

            # Optional safety check: new element counts per world rank
            current_local_new = sum(len(eidxs)
                                    for eidxs in self.meshes['compute'].eidxs.values())
            ecurrs_new = np.asarray(
                comm['world'].allgather(int(current_local_new)),
                dtype=np.int64
            )

            # At this point, rankmap['newcompute'] is still valid.
            new_active = set(rankmap['newcompute'])

            # Assert: any world-rank not in new_active has zero elements.
            # (If you want to be looser, just drop the assert.)
            if all((i in new_active) or (ecurrs_new[i] == 0)
                       for i in range(comm['world'].size)):


                # 🔍 NEW: print state just before and after promote_comm
                if rank['world'] == root['world']:
                    print(
                        f"[comm-debug] before-promote: "
                        f"compute_size={comm['compute'].size} "
                        f"rankmap_compute={rankmap['compute']} "
                        f"newcompute_size={comm['newcompute'].size} "
                        f"rankmap_newcompute={rankmap['newcompute']}",
                        flush=True
                    )

                # Now make 'newcompute' the canonical 'compute' communicator.
                promote_comm('newcompute', 'compute')

                if rank['world'] == root['world']:
                    print(
                        f"[comm-debug] after-promote: "
                        f"compute_size={comm['compute'].size} "
                        f"rankmap_compute={rankmap['compute']}",
                        flush=True
                    )

            # Reinitialise backend+system on the new 'compute' layout
            self.reinit_backend_and_system(self.meshes['compute'], soln)

            wallt_reinit = perf_counter_ns() - wallt_start - wallt_iterate

            # Write wall times
            if rank['world'] == root['world']:
                with open('lb_walltimes.csv', 'a') as f:
                    f.write(f"{self.tcurr:.6f},"
                            f"{(wallt_start - self.wallt_end)/1e9},"
                            f"{wallt_iterate/1e9},"
                            f"{wallt_reinit/1e9}\n")

            self.wallt_end = perf_counter_ns()

    def reinit_mesh_soln(self, mesh, soln):
        # New mesh lives under the 'newcompute' logical name while we migrate.
        self.meshes['newcompute'] = mesh
        _MetaMesh.info(self.meshes['newcompute'])

        # Build interconnector from old compute layout -> newcompute layout
        self._newcompute_intercon = self.initialise_interconnector('compute', 'newcompute')
        soln = self.relocate_ary(self._newcompute_intercon, soln,
                                 edim=2, src_name='compute', dst_name='newcompute')

        # Drop old compute mesh and plugin interconnector
        del self.meshes['compute']
        del self._plugins_intercon

        # Promote newcompute mesh to be the canonical compute mesh
        self.meshes['compute'] = self.meshes['newcompute']
        # Optionally drop the extra key to avoid confusion
        # del self.meshes['newcompute']

        return soln


    def reinit_backend_and_system(self, mesh, soln):


        # 🔍 NEW: log communicator state at entry
        if comm['compute'] != mpi.COMM_NULL:
            print(
                f"[reinit-debug] world_rank={rank['world']} "
                f"compute_size={comm['compute'].size} "
                f"rankmap_compute={rankmap['compute']}",
                flush=True
            )

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
        self._plugins_intercon = self.initialise_interconnector('compute', 'plugins')

        self.plugins = self._reget_plugins()

        #if comm['compute'] != mpi.COMM_NULL:
        #    self.system.commit()
        #    self.system.preproc(self.tcurr, self._idxcurr)

        execute['compute'](lambda: self.system.commit())
        execute['compute'](lambda: self.system.preproc(self.tcurr, self._idxcurr))

        comm['world'].barrier()


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
            dt = max(min(t - self.tcurr, self._dt, self.dtmax), self.dtmin)

            # Take the step
            idxcurr, idxprev, idxerr = execute['compute'](lambda: self.step(self.tcurr, dt),
                                                          default=(-1, -1, -1))

            idxcurr = comm['world'].allreduce(idxcurr, op=mpi.MAX)
            idxprev = comm['world'].allreduce(idxprev, op=mpi.MAX)
            idxerr  = comm['world'].allreduce(idxerr,  op=mpi.MAX)

            # Estimate the error
            err = self._errest(idxcurr, idxprev, idxerr)

            # Decide if to accept or reject the step
            if err < 1.0:
                self._errprev = err
                self._accept_step(self.dt, idxcurr, err=err)
            else:
                self._reject_step(self.dt, idxprev, err=err)

            # Adjust time step per PI controller
            fac = err**-expa * self._errprev**expb
            fac = min(maxf, max(minf, saff*fac))

            # Compute the next time step
            self.dt_fallback = fac*self.dt
            self.load_balance()
