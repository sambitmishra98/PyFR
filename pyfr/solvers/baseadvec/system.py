from itertools import zip_longest

from pyfr.cache import memoize
from pyfr.solvers.base import BaseSystem
from pyfr.solvers.baseadvec.entfilter import EntropyFilter


class BaseAdvectionSystem(BaseSystem):
    _shock_capturing_modes = {'none', 'entropy-filter'}

    def __init__(self, backend, mesh, initsoln, registers, cfg, serialiser,
                 *, needs_cfl=False):
        super().__init__(backend, mesh, initsoln, registers, cfg, serialiser)
        self._needs_cfl = needs_cfl

        if needs_cfl:
            for eles in self.ele_map.values():
                eles.init_wavespeed()


        # Create scal_fpts views and MPI exchange at the system level
        scal_fpts = {et: e._scal_fpts for et, e in self.ele_map.items()}
        iint_v, mpi_v, bc_v = self.make_field_views(scal_fpts,
                                                    vshape=(self.nvars,))
        for i, (lhs, rhs) in zip(self._int_inters, iint_v):
            i.scal_lhs = lhs
            i.scal_rhs = rhs
        for m, (lhs, rhs) in zip(self._mpi_inters, mpi_v):
            m.scal_lhs = lhs
            m.scal_rhs = rhs
        for b, lhs in zip(self._bc_inters, bc_v):
            b.scal_lhs = lhs
        self.register_mpi_exchange('scal_fpts', mpi_v)

        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')
        if self._mortar_inters and shock_capturing != 'none':
            raise ValueError(
                'Mortars currently require shock-capturing = none'
            )

        # Handle entropy filtering
        if (shock_capturing == 'entropy-filter' and
            self.cfg.getint('solver', 'order') > 0):
            self._ef = EntropyFilter(self.backend, self.cfg, self,
                                     self._int_inters, self._mpi_inters,
                                     self._bc_inters)
        else:
            self._ef = None

    def commit(self):
        self.backend.commit()

        # Reduction kernels to find max wavespeed across each element type
        if self._needs_cfl:
            self._wspd_red_kerns = [
                self.backend.kernel('reduction', 'max', ['x'], {'x': e._wspd})
                for e in self.ele_map.values()
            ]

        super().commit()

        # Populate the per-element wavespeed
        if self._needs_cfl:
            self.compute_max_wavespeed(0)

    @memoize
    def _rhs_graphs(self, uinbank, foutbank):
        if self._has_mpi_p_mortars:
            return self._rhs_graphs_mpi_p(uinbank, foutbank)

        m = self._mpireqs
        k, *_ = self._get_kernels(uinbank, foutbank)

        def deps(dk, *names): return self._kdeps(k, dk, *names)

        # Graph 1: interpolate solution, exchange, compute local flux
        g_intf = self.backend.graph()
        g_intf.add_mpi_reqs(m['scal_fpts_recv'])

        # Interpolate the solution to the flux points
        g_intf.add_all(k['eles/disu'], deps=k['eles/entropy_filter'])

        # EF adds its kernels (topo sort handles ordering)
        if self._ef:
            self._ef.add_to_graph_pre_recv(g_intf, k, m)

        # Pack and send these interpolated solutions to our neighbours
        g_intf.add_all(k['mpiint/scal_fpts_pack'], deps=k['eles/disu'])
        for send, pack in zip(m['scal_fpts_send'],
                              k['mpiint/scal_fpts_pack']):
            g_intf.add_mpi_req(send, deps=[pack])

        # Compute the common normal flux at our internal/boundary interfaces
        g_intf.add_all(k['iint/comm_flux'],
                       deps=k['eles/disu'] + k['mpiint/scal_fpts_pack'])
        g_intf.add_all(k['mint/comm_flux'], deps=k['eles/disu'])

        mprev = None
        for stage in ('comm_flux_gather', 'comm_flux_interp',
                      'comm_flux_eval', 'comm_flux_project',
                      'comm_flux_scatter'):
            skerns = k[f'mint/{stage}']
            if mprev is None:
                g_intf.add_all(skerns, deps=k['eles/disu'])
            else:
                for l in skerns:
                    g_intf.add(l, deps=deps(l, f'mint/{mprev}'))
            mprev = stage

        g_intf.add_all(k['bcint/comm_flux'],
                       deps=k['eles/disu'] + k['bcint/comm_entropy'])

        # Make a copy of the solution (if used by source terms)
        g_intf.add_all(k['eles/copy_soln'], deps=k['eles/entropy_filter'])

        g_intf.commit()

        # Graph 2: receive MPI solution, compute flux and divergence
        g_flux_div = self.backend.graph()

        # Interpolate the solution to the quadrature points
        g_flux_div.add_all(k['eles/qptsu'])

        # Compute the transformed flux
        for l in k['eles/tdisf']:
            g_flux_div.add(l, deps=deps(l, 'eles/qptsu'))

        # Compute the transformed divergence of the partially corrected flux
        for l in k['eles/tdivtpcorf']:
            g_flux_div.add(l, deps=deps(l, 'eles/tdisf'))

        # Unpack MPI face data (may be empty when unpack is a no-op)
        g_flux_div.add_all(k['mpiint/scal_fpts_unpack'])

        # Compute the common normal flux at our MPI interfaces
        for l in k['mpiint/comm_flux']:
            g_flux_div.add(l, deps=deps(l, 'mpiint/scal_fpts_unpack'))

        # EF: unpack and compute comm_entropy at MPI interfaces
        if self._ef:
            self._ef.add_to_graph_post_recv(g_flux_div, k, deps)

        # Compute the transformed divergence of the corrected flux
        for l in k['eles/tdivtconf']:
            ldeps = deps(l, 'eles/tdivtpcorf') + k['mpiint/comm_flux']
            g_flux_div.add(l, deps=ldeps)

        # Obtain the physical divergence of the corrected flux
        for l in k['eles/negdivconf']:
            g_flux_div.add(l, deps=deps(l, 'eles/tdivtconf'))

        kgroup = [k['eles/qptsu'], k['eles/tdisf'], k['eles/tdivtpcorf'],
                  k['eles/tdivtconf'], k['eles/negdivconf']]
        for ks in zip_longest(*kgroup):
            self._group(g_flux_div, ks, subs=[
                [(ks[0], 'out'), (ks[1], 'u')],
                [(ks[1], 'f'), (ks[2], 'b')],
            ])

        g_flux_div.commit()

        return g_intf, g_flux_div

    @memoize
    def _rhs_graphs_mpi_p(self, uinbank, foutbank):
        m = self._mpireqs
        k, *_ = self._get_kernels(uinbank, foutbank)

        def deps(dk, *names):
            return self._kdeps(k, dk, *names)

        # Graph 1: interpolate, ordinary exchange/local flux, and send the
        # nonowner native trace to the authoritative distributed-p owner.
        g_state = self.backend.graph()
        g_state.add_mpi_reqs(m['scal_fpts_recv'])
        g_state.add_mpi_reqs(m['mpi_p_state_recv'])

        g_state.add_all(k['eles/disu'], deps=k['eles/entropy_filter'])

        g_state.add_all(
            k['mpiint/scal_fpts_pack'], deps=k['eles/disu']
        )
        for send, pack in zip(
            m['scal_fpts_send'], k['mpiint/scal_fpts_pack']
        ):
            g_state.add_mpi_req(send, deps=[pack])

        g_state.add_all(
            k['iint/comm_flux'],
            deps=k['eles/disu'] + k['mpiint/scal_fpts_pack']
        )
        g_state.add_all(k['mint/comm_flux'], deps=k['eles/disu'])

        mprev = None
        for stage in (
            'comm_flux_gather', 'comm_flux_interp', 'comm_flux_eval',
            'comm_flux_project', 'comm_flux_scatter'
        ):
            skerns = k[f'mint/{stage}']
            if mprev is None:
                g_state.add_all(skerns, deps=k['eles/disu'])
            else:
                for kern in skerns:
                    g_state.add(
                        kern, deps=deps(kern, f'mint/{mprev}')
                    )
            mprev = stage

        g_state.add_all(
            k['bcint/comm_flux'],
            deps=k['eles/disu'] + k['bcint/comm_entropy']
        )

        g_state.add_all(
            k['mpimint/state_gather'], deps=k['eles/disu']
        )
        g_state.add_all(
            k['mpimint/state_pack'], deps=k['mpimint/state_gather']
        )
        state_send_deps = (
            k['mpimint/state_pack'] or k['mpimint/state_gather']
        )
        for send in m['mpi_p_state_send']:
            g_state.add_mpi_req(send, deps=state_send_deps)

        g_state.add_all(
            k['eles/copy_soln'], deps=k['eles/entropy_filter']
        )
        g_state.commit()

        # Graph 2: consume the completed remote state on owners, evaluate
        # exactly one common Euler flux, and return the signed projected
        # nonowner native-face contribution.
        g_flux = self.backend.graph()
        g_flux.add_mpi_reqs(m['mpi_p_flux_recv'])

        g_flux.add_all(k['eles/qptsu'])
        for kern in k['eles/tdisf']:
            g_flux.add(kern, deps=deps(kern, 'eles/qptsu'))
        for kern in k['eles/tdivtpcorf']:
            g_flux.add(kern, deps=deps(kern, 'eles/tdisf'))

        g_flux.add_all(k['mpiint/scal_fpts_unpack'])
        for kern in k['mpiint/comm_flux']:
            g_flux.add(
                kern, deps=deps(kern, 'mpiint/scal_fpts_unpack')
            )

        g_flux.add_all(k['mpimint/state_unpack'])
        g_flux.add_all(
            k['mpimint/state_interp'], deps=k['mpimint/state_unpack']
        )
        g_flux.add_all(
            k['mpimint/flux_eval'], deps=k['mpimint/state_interp']
        )
        g_flux.add_all(
            k['mpimint/flux_project'], deps=k['mpimint/flux_eval']
        )
        g_flux.add_all(
            k['mpimint/owner_scatter'],
            deps=k['mpimint/flux_project']
        )
        g_flux.add_all(
            k['mpimint/flux_pack'], deps=k['mpimint/flux_project']
        )
        flux_send_deps = (
            k['mpimint/flux_pack'] or k['mpimint/flux_project']
        )
        for send in m['mpi_p_flux_send']:
            g_flux.add_mpi_req(send, deps=flux_send_deps)

        kgroup = [
            k['eles/qptsu'], k['eles/tdisf'], k['eles/tdivtpcorf']
        ]
        for ks in zip_longest(*kgroup):
            self._group(g_flux, ks, subs=[
                [(ks[0], 'out'), (ks[1], 'u')],
                [(ks[1], 'f'), (ks[2], 'b')],
            ])

        g_flux.commit()

        # Graph 3: the projected-flux receive is now complete. Scatter it
        # on nonowners before forming the corrected divergence everywhere.
        g_div = self.backend.graph()
        g_div.add_all(k['mpimint/flux_unpack'])
        g_div.add_all(
            k['mpimint/flux_scatter'], deps=k['mpimint/flux_unpack']
        )

        for kern in k['eles/tdivtconf']:
            g_div.add(kern, deps=k['mpimint/flux_scatter'])
        for kern in k['eles/negdivconf']:
            g_div.add(kern, deps=deps(kern, 'eles/tdivtconf'))

        for ks in zip_longest(
            k['eles/tdivtconf'], k['eles/negdivconf']
        ):
            self._group(g_div, ks)

        g_div.commit()
        return g_state, g_flux, g_div

    def _preproc_graphs(self, uinbank):
        if self._ef:
            return self._preproc_graphs_ef(uinbank)
        else:
            return ()

    @memoize
    def _preproc_graphs_ef(self, uinbank):
        m = self._mpireqs
        k, *_ = self._get_kernels(uinbank, None)
        def deps(dk, *names): return self._kdeps(k, dk, *names)
        return self._ef.preproc_graphs(self.backend, k, m, deps)

    def postproc(self, uinbank):
        if self._ef:
            if uinbank >= self.nrhs:
                raise ValueError('Invalid register number')
            k, *_ = self._get_kernels(uinbank, None)
            self._ef.postproc(self.backend, k)

    def compute_max_wavespeed(self, uinbank):
        k, *_ = self._get_kernels(uinbank, None)
        kerns = k['eles/wavespeed'] + self._wspd_red_kerns
        self.backend.run_kernels(kerns, wait=True)
        return max(k.retval[0] for k in self._wspd_red_kerns)
