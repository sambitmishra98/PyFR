import numpy as np

from pyfr.mortars import (
    MortarSide, build_line_line2_operators, build_quad_p_operators,
    build_quad_quad4_operators, build_quad_tri_operators,
)
from pyfr.solvers.base.mortars import (
    BaseMortarInters, build_mortar_execution_batches, local_mortar_ownership,
    mortar_face_view, mortar_side_point_view, mortar_side_view,
)
from pyfr.solvers.baseadvecdiff.inters import mpi_ldg_beta


class _NavierStokesMPIPMortarBatch:
    def __init__(self, be, plan, entries, elemap, cfg, comm, index):
        self._be = be
        self.cfg = cfg
        self.plan = plan
        self.ownership = plan.ownership
        self.name = f'p{plan.neighbour_rank}-mortar-{index}'
        self.kernels = {}
        self.mpireqs = {}
        self.ninters = len(entries)

        if not entries:
            raise ValueError('Empty distributed p-mortar execution batch')
        if cfg.get('solver', 'shock-capturing', 'none') != 'none':
            raise ValueError(
                'Distributed mixed-p Navier-Stokes requires '
                'shock-capturing = none'
            )
        if cfg.get('solver', 'viscosity-correction', 'none') != 'none':
            raise ValueError(
                'Distributed mixed-p Navier-Stokes requires constant '
                'viscosity'
            )
        if cfg.get(
            'solver-interfaces', 'mortar-implementation', 'fused'
        ) != 'staged':
            raise ValueError(
                'Distributed mixed-p Navier-Stokes requires staged '
                'execution'
            )

        rank = comm.rank
        owner = rank == plan.ownership.owner_rank
        local_owners = {entry[0].local_is_owner for entry in entries}
        if local_owners != {owner}:
            raise ValueError('Inconsistent distributed p-mortar ownership')
        self.local_is_owner = owner

        faces = tuple(entry[0] for entry in entries)
        face_ops = tuple(entry[1] for entry in entries)
        ops = face_ops[0]
        if any(o['operator_keys'] != ops['operator_keys'] for o in face_ops):
            raise ValueError('Inconsistent distributed p-mortar operators')
        if len(ops['operator_sets']) != 1:
            raise ValueError(
                'Distributed p-mortar batch requires one operator'
            )

        op = ops['operator_sets'][0]
        if 'right_state_proj' not in op:
            raise ValueError('Missing distributed common-state projection')

        f0 = faces[0]
        local = MortarSide(
            f0.local_side.etype, f0.local_side.face_topology,
            tuple(f.local_side.fidxs[0] for f in faces),
            tuple(f.local_side.eidxs[0] for f in faces),
            f0.local_side.elekey
        )
        if any(f.local_side.runtime_key != local.runtime_key for f in faces):
            raise ValueError('Inconsistent distributed p-mortar local group')

        self.ndims = next(iter(elemap.values())).ndims
        self.nvars = next(iter(elemap.values())).nvars
        if self.ndims != 3:
            raise ValueError('Distributed quad p-mortars require 3D')

        nb = self.ninters
        onfpts = ops['nleftfpts']
        nnfpts = ops['nrightfpts'][0]
        nmpts = ops['nmpts']
        lnfpts = onfpts if owner else nnfpts

        owner_rank = plan.ownership.owner_rank
        nonowner_rank = next(
            r for r in plan.ownership.participant_ranks
            if r != owner_rank
        )
        beta = cfg.getfloat('solver-interfaces', 'ldg-beta')
        self.effective_beta = mpi_ldg_beta(
            beta, owner_rank, nonowner_rank
        )
        self.needs_remote_common_state = self.effective_beta != 0.5
        self.needs_remote_gradient = self.effective_beta != 0.5

        uview = mortar_side_view(
            be, elemap, local, '_scal_fpts', (self.nvars,)
        )
        commview = mortar_side_view(
            be, elemap, local, '_comm_fpts', (self.nvars,)
        )
        gradview = mortar_side_point_view(
            be, elemap, local, 'get_vect_fpts_for_mortars',
            (self.ndims, self.nvars)
        )

        state_xchg = be.xchg_matrix(
            (nnfpts, self.nvars, nb), tags={'align'}
        )
        grad_xchg = be.xchg_matrix(
            (self.ndims*nnfpts, self.nvars, nb), tags={'align'}
        )
        self._state_xchg = state_xchg
        self._grad_xchg = grad_xchg

        c = cfg.items_as('constants', float)
        c |= cfg.items_as('solver-interfaces', float)
        c['ldg-beta'] = self.effective_beta
        tplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'rsolver': cfg.get('solver-interfaces', 'riemann-solver'),
            'visc_corr': 'none',
            'shock_capturing': 'none',
            'c': c,
            'p_min': cfg.getfloat(
                'solver-interfaces', 'p-min', 5*be.fpdtype_eps
            ),
        }

        be.pointwise.register(
            'pyfr.solvers.baseadvec.kernels.mortarsidegather'
        )
        be.pointwise.register(
            'pyfr.solvers.baseadvec.kernels.mortarsidescatter'
        )
        be.pointwise.register(
            'pyfr.solvers.navstokes.kernels.mortargradsidegather'
        )

        grad_xviews = self._grad_buffer_views(
            be, grad_xchg, nnfpts, nb
        )

        if owner:
            be.pointwise.register(
                'pyfr.solvers.navstokes.kernels.mortarconupatch'
            )
            be.pointwise.register(
                'pyfr.solvers.navstokes.kernels.mortarfluxpatch'
            )

            native = be.matrix(
                (onfpts, self.nvars, nb), tags={'align'}
            )
            grad_native = be.matrix(
                (self.ndims*onfpts, self.nvars, nb), tags={'align'}
            )
            grad_nviews = self._grad_buffer_views(
                be, grad_native, onfpts, nb
            )
            ul = be.matrix((nmpts, self.nvars, nb), tags={'align'})
            ur = be.matrix((nmpts, self.nvars, nb), tags={'align'})
            work = be.matrix((nmpts, self.nvars, nb), tags={'align'})
            gl = tuple(
                be.matrix((nmpts, self.nvars, nb), tags={'align'})
                for _ in range(self.ndims)
            )
            gr = tuple(
                be.matrix((nmpts, self.nvars, nb), tags={'align'})
                for _ in range(self.ndims)
            )
            mats = {
                name: be.const_matrix(op[name][0], tags={'align'})
                for name in (
                    'left_interp', 'right_interp', 'left_proj',
                    'right_proj', 'right_state_proj'
                )
            }
            normals = np.concatenate([
                f.geometry.scaled_normals[0] for f in faces
            ], axis=2)
            normals = be.const_matrix(normals, tags={'align'})

            self.kernels['state_gather'] = lambda: be.kernel(
                'mortarsidegather',
                tplargs=tplargs | {'nfpts': onfpts},
                dims=[nb], u=uview, b=native
            )
            self.kernels['state_unpack'] = lambda: be.kernel(
                'unpack', state_xchg
            )
            self.kernels['state_interp'] = lambda: be.unordered_meta_kernel([
                be.kernel('mul', mats['left_interp'], native, out=ul),
                be.kernel('mul', mats['right_interp'], state_xchg, out=ur),
            ])
            self.kernels['common_state_eval'] = lambda: be.kernel(
                'mortarconupatch', tplargs=tplargs, dims=[nmpts, nb],
                ul=ul, ur=ur, cu=work
            )
            self.kernels['common_state_project'] = (
                lambda: be.ordered_meta_kernel([
                    be.kernel('mul', mats['left_proj'], work, out=native),
                    be.kernel(
                        'mul', mats['right_state_proj'], work,
                        out=state_xchg
                    ),
                ])
            )
            self.kernels['common_state_owner_scatter'] = lambda: be.kernel(
                'mortarsidescatter',
                tplargs=tplargs | {'nfpts': onfpts},
                dims=[nb], b=native, u=commview
            )
            if self.needs_remote_common_state:
                self.kernels['common_state_pack'] = lambda: be.kernel(
                    'pack', state_xchg
                )

            self.kernels['grad_gather'] = lambda: be.kernel(
                'mortargradsidegather', tplargs=tplargs,
                dims=[onfpts*nb], g=gradview,
                b0=grad_nviews[0], b1=grad_nviews[1],
                b2=grad_nviews[2]
            )
            if self.needs_remote_gradient:
                self.kernels['grad_unpack'] = lambda: be.kernel(
                    'unpack', grad_xchg
                )

            def grad_interp():
                kerns = []
                for d in range(self.ndims):
                    kerns.append(be.kernel(
                        'mul', mats['left_interp'],
                        grad_native.slice(d*onfpts, (d + 1)*onfpts),
                        out=gl[d]
                    ))
                    if self.needs_remote_gradient:
                        kerns.append(be.kernel(
                            'mul', mats['right_interp'],
                            grad_xchg.slice(d*nnfpts, (d + 1)*nnfpts),
                            out=gr[d]
                        ))
                return be.unordered_meta_kernel(kerns)

            self.kernels['grad_interp'] = grad_interp

            def flux_eval():
                kwargs = {f'gl{d}': gl[d] for d in range(self.ndims)}
                kwargs.update(
                    {f'gr{d}': gr[d] for d in range(self.ndims)}
                )
                return be.kernel(
                    'mortarfluxpatch', tplargs=tplargs,
                    dims=[nmpts, nb], ul=ul, ur=ur, nl=normals,
                    fc=work, **kwargs
                )

            self.kernels['flux_eval'] = flux_eval
            self.kernels['flux_project'] = lambda: be.ordered_meta_kernel([
                be.kernel('mul', mats['left_proj'], work, out=native),
                be.kernel(
                    'mul', mats['right_proj'], work,
                    out=state_xchg, alpha=-1.0
                ),
            ])
            self.kernels['owner_flux_scatter'] = lambda: be.kernel(
                'mortarsidescatter',
                tplargs=tplargs | {'nfpts': onfpts},
                dims=[nb], b=native, u=uview
            )
            self.kernels['flux_pack'] = lambda: be.kernel(
                'pack', state_xchg
            )

            owner_bytes = (
                native.nbytes + grad_native.nbytes + ul.nbytes + ur.nbytes
                + work.nbytes + sum(x.nbytes for x in (*gl, *gr))
            )
        else:
            self.kernels['state_gather'] = lambda: be.kernel(
                'mortarsidegather',
                tplargs=tplargs | {'nfpts': lnfpts},
                dims=[nb], u=uview, b=state_xchg
            )
            self.kernels['state_pack'] = lambda: be.kernel(
                'pack', state_xchg
            )
            if self.needs_remote_common_state:
                self.kernels['common_state_unpack'] = lambda: be.kernel(
                    'unpack', state_xchg
                )
                self.kernels['common_state_scatter'] = lambda: be.kernel(
                    'mortarsidescatter',
                    tplargs=tplargs | {'nfpts': lnfpts},
                    dims=[nb], b=state_xchg, u=commview
                )
            else:
                self.kernels['common_state_local_scatter'] = (
                    lambda: be.kernel(
                        'mortarsidescatter',
                        tplargs=tplargs | {'nfpts': lnfpts},
                        dims=[nb], b=state_xchg, u=commview
                    )
                )

            self.kernels['grad_gather'] = lambda: be.kernel(
                'mortargradsidegather', tplargs=tplargs,
                dims=[lnfpts*nb], g=gradview,
                b0=grad_xviews[0], b1=grad_xviews[1],
                b2=grad_xviews[2]
            )
            if self.needs_remote_gradient:
                self.kernels['grad_pack'] = lambda: be.kernel(
                    'pack', grad_xchg
                )
            self.kernels['flux_unpack'] = lambda: be.kernel(
                'unpack', state_xchg
            )
            self.kernels['flux_scatter'] = lambda: be.kernel(
                'mortarsidescatter',
                tplargs=tplargs | {'nfpts': lnfpts},
                dims=[nb], b=state_xchg, u=uview
            )
            owner_bytes = 0

        tags = dict(plan.tags)
        nrank = plan.neighbour_rank
        if owner:
            self.mpireqs['mpi_p_state_recv'] = lambda: state_xchg.recvreq(
                comm, nrank, tags['state']
            )
            if self.needs_remote_common_state:
                self.mpireqs['mpi_p_common_send'] = (
                    lambda: state_xchg.sendreq(
                        comm, nrank, tags['common-state']
                    )
                )
            if self.needs_remote_gradient:
                self.mpireqs['mpi_p_grad_recv'] = lambda: grad_xchg.recvreq(
                    comm, nrank, tags['gradient']
                )
            self.mpireqs['mpi_p_flux_send'] = lambda: state_xchg.sendreq(
                comm, nrank, tags['flux']
            )
        else:
            self.mpireqs['mpi_p_state_send'] = lambda: state_xchg.sendreq(
                comm, nrank, tags['state']
            )
            if self.needs_remote_common_state:
                self.mpireqs['mpi_p_common_recv'] = (
                    lambda: state_xchg.recvreq(
                        comm, nrank, tags['common-state']
                    )
                )
            if self.needs_remote_gradient:
                self.mpireqs['mpi_p_grad_send'] = lambda: grad_xchg.sendreq(
                    comm, nrank, tags['gradient']
                )
            self.mpireqs['mpi_p_flux_recv'] = lambda: state_xchg.recvreq(
                comm, nrank, tags['flux']
            )

        self.buffer_bytes = state_xchg.nbytes + grad_xchg.nbytes + owner_bytes

    def _grad_buffer_views(self, be, buf, nfpts, nb):
        n = nfpts*nb
        rbase = np.tile(np.arange(nfpts, dtype=np.int32), nb)
        cmap = np.repeat(np.arange(nb, dtype=np.int32), nfpts)
        return tuple(
            be.view(
                np.full(n, buf.mid), rbase + d*nfpts, cmap,
                np.ones(n, dtype=np.int32), vshape=(self.nvars,)
            )
            for d in range(self.ndims)
        )


class NavierStokesMortarInters(BaseMortarInters):
    mpi_p_state_projection = True

    @classmethod
    def from_mpi_p_mortars(
        cls, be, faces, face_ops, plans, elemap, cfg, comm
    ):
        by_pair = {
            face.face_pair: (face, ops)
            for face, ops in zip(faces, face_ops)
        }
        if len(by_pair) != len(faces):
            raise ValueError('Duplicate distributed p-mortar face identity')

        batches = []
        for i, plan in enumerate(plans):
            try:
                entries = tuple(by_pair[pair] for pair in plan.face_pairs)
            except KeyError as exc:
                raise ValueError(
                    'Distributed p-mortar batch references an unknown face'
                ) from exc
            batches.append(
                _NavierStokesMPIPMortarBatch(
                    be, plan, entries, elemap, cfg, comm, i
                )
            )

        return tuple(batches)

    @classmethod
    def from_p_mortar(cls, be, mesh, group, elemap, cfg, name):
        self = cls.__new__(cls)
        self.name = name
        self._be = be
        self.cfg = cfg
        self.kernels = {}
        self.mpireqs = {}
        self.ndims = next(iter(elemap.values())).ndims
        self.nvars = next(iter(elemap.values())).nvars
        self.ninters = len(group.left.eidxs)

        impl = cfg.get(
            'solver-interfaces', 'mortar-implementation', 'fused'
        )
        self.mortar_implementation = impl
        if impl != 'staged':
            raise ValueError('Mixed-p mortars require staged execution')

        visc_corr = cfg.get('solver', 'viscosity-correction', 'none')
        if visc_corr != 'none':
            raise ValueError(
                'Mixed-p Navier-Stokes mortars require constant viscosity'
            )

        ops = build_quad_p_operators(
            mesh, elemap, group, cfg, state_projection=True
        )
        self._geometry = ops['geometry']
        self._ownership = local_mortar_ownership()
        self.max_geom_error = ops['max_geom_error']
        self.max_normal_error = ops['max_normal_error']
        self.noperator_sets = ops['nops']
        self.shared_operator_bytes = ops['shared_operator_bytes']
        self.geometry_bytes = self._geometry.backend_nbytes
        self.coarse_etype = group.left.etype
        self.fine_etype = group.right[0].etype
        self.fused_operator_bytes = 0
        self.operator_bytes = self.shared_operator_bytes

        self._init_general_staged(be, elemap, cfg, ops)
        return self

    def __init__(self, be, mesh, mcon, elemap, cfg):
        self.name = mcon.name
        self._be = be
        self.cfg = cfg
        self.kernels = {}
        self.mpireqs = {}
        self.ndims = next(iter(elemap.values())).ndims
        self.nvars = next(iter(elemap.values())).nvars
        self.ninters = len(mcon)

        impl = cfg.get(
            'solver-interfaces', 'mortar-implementation', 'fused'
        )
        self.mortar_implementation = impl

        if mcon.format == 'one-to-many-v1':
            if mcon.template == 'quad-2x2':
                if self.ndims != 3:
                    raise ValueError('Quad mortars require 3D')
                operator_builder = build_quad_quad4_operators
            elif mcon.template == 'line-1x2':
                if self.ndims != 2:
                    raise ValueError('Line mortars require 2D')
                operator_builder = build_line_line2_operators
            else:
                raise ValueError(
                    f'Unsupported general mortar template {mcon.template!r}'
                )
            if impl != 'staged':
                raise ValueError(
                    'General one-to-many mortars require staged execution'
                )
            ops = operator_builder(
                mesh, elemap, mcon, cfg, state_projection=True
            )
            general = True
        elif mcon.format == 'quad-tri-v1':
            if self.ndims != 3:
                raise ValueError('Quad mortars require 3D')
            ops = build_quad_tri_operators(mesh, elemap, mcon, cfg)
            general = False
        else:
            raise ValueError(f'Unsupported mortar format {mcon.format!r}')

        self._geometry = ops['geometry']
        self._ownership = local_mortar_ownership()
        self.max_geom_error = ops['max_geom_error']
        self.max_normal_error = ops['max_normal_error']
        self.noperator_sets = ops['nops']
        self.shared_operator_bytes = ops['shared_operator_bytes']
        self.geometry_bytes = self._geometry.backend_nbytes

        if general:
            group = ops['mortar_group']
            self.coarse_etype = group.left.etype
            self.fine_etype = group.right[0].etype
            self.fused_operator_bytes = 0
            self.operator_bytes = self.shared_operator_bytes
            self._init_general_staged(be, elemap, cfg, ops)
        else:
            self.coarse_etype = ops['coarse'][0]
            self.fine_etype = ops['fine0'][0]
            self.fused_operator_bytes = ops['fused_operator_bytes']
            self.operator_bytes = (
                self.shared_operator_bytes if impl == 'staged'
                else self.fused_operator_bytes
            )

            if impl == 'fused':
                self._init_fused(be, elemap, cfg, ops)
            elif impl == 'staged':
                self._init_staged(be, elemap, cfg, ops)
            else:
                raise ValueError(f'Invalid mortar implementation {impl!r}')

    def _tplargs(self, be, cfg, ops):
        rsolver = cfg.get('solver-interfaces', 'riemann-solver')
        visc_corr = cfg.get('solver', 'viscosity-correction', 'none')
        p_min = cfg.getfloat(
            'solver-interfaces', 'p-min', 5*be.fpdtype_eps
        )
        c = cfg.items_as('constants', float)
        c |= cfg.items_as('solver-interfaces', float)

        return {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'ncfpts': ops['ncfpts'],
            'ntfpts': ops['ntfpts'],
            'nmpts': ops['nmpts'],
            'rsolver': rsolver,
            'visc_corr': visc_corr,
            'shock_capturing': 'none',
            'c': c,
            'p_min': p_min,
        }

    def _side_views(self, be, elemap, side, suffix):
        return mortar_face_view(
            be, elemap, *side, suffix, (self.nvars,)
        )

    def _grad_views(self, be, elemap, side):
        etype = side[0]
        eles = elemap[etype]

        return [
            mortar_face_view(
                be, elemap, *side, '_vect_fpts', (self.nvars,),
                row_offset=d*eles.nfpts
            )
            for d in range(self.ndims)
        ]

    def _init_fused(self, be, elemap, cfg, ops):
        self.nbatches = 1
        self.staged_buffer_bytes = 0
        self.staged_allocated_bytes = 0
        sides = ('coarse', 'fine0', 'fine1')
        for name in sides:
            side = ops[name]
            setattr(
                self, f'u{name}',
                self._side_views(be, elemap, side, '_scal_fpts')
            )
            setattr(
                self, f'comm{name}',
                self._side_views(be, elemap, side, '_comm_fpts')
            )
            setattr(self, f'grad{name}', self._grad_views(
                be, elemap, side
            ))

        opmats = {
            name: be.const_matrix(ops[name], tags={'align'})
            for name in (
                'ic0', 'ic1', 'if0', 'if1', 'pc0', 'pc1',
                'pf0', 'pf1'
            )
        }
        normmats = {
            name: be.const_matrix(ops[name], tags={'align'})
            for name in ('nl0', 'nl1')
        }
        tplargs = self._tplargs(be, cfg, ops)
        be.pointwise.register('pyfr.solvers.navstokes.kernels.mortarconu')
        be.pointwise.register('pyfr.solvers.navstokes.kernels.mortarcflux')

        self.kernels['con_u'] = lambda: be.kernel(
            'mortarconu', tplargs=tplargs, dims=[self.ninters],
            ucoarse=self.ucoarse, ufine0=self.ufine0,
            ufine1=self.ufine1, commcoarse=self.commcoarse,
            commfine0=self.commfine0, commfine1=self.commfine1,
            **opmats
        )

        gradargs = {}
        for d in range(self.ndims):
            gradargs[f'gradcoarse{d}'] = self.gradcoarse[d]
            gradargs[f'gradfine0{d}'] = self.gradfine0[d]
            gradargs[f'gradfine1{d}'] = self.gradfine1[d]

        self.kernels['comm_flux'] = lambda: be.kernel(
            'mortarcflux', tplargs=tplargs, dims=[self.ninters],
            ucoarse=self.ucoarse, ufine0=self.ufine0,
            ufine1=self.ufine1, **gradargs, **opmats, **normmats
        )

    def _init_staged(self, be, elemap, cfg, ops):
        tplargs = self._tplargs(be, cfg, ops)
        ncfpts, ntfpts = ops['ncfpts'], ops['ntfpts']
        nmpts, nvars = ops['nmpts'], self.nvars

        be.pointwise.register(
            'pyfr.solvers.baseadvec.kernels.mortargather'
        )
        be.pointwise.register(
            'pyfr.solvers.baseadvec.kernels.mortarscatter'
        )
        be.pointwise.register(
            'pyfr.solvers.navstokes.kernels.mortarconustaged'
        )
        be.pointwise.register(
            'pyfr.solvers.navstokes.kernels.mortargradgather'
        )
        be.pointwise.register(
            'pyfr.solvers.navstokes.kernels.mortarfluxstaged'
        )

        execution_batches = build_mortar_execution_batches(
            ops, elemap, 'navier-stokes', 'staged',
            self._ownership
        )
        self._execution_batches = execution_batches

        batches = []
        for execution in execution_batches:
            group = list(execution.indices)
            op = ops['operator_sets'][execution.operator_set]
            left = execution.group.left.as_legacy()
            right = tuple(
                side.as_legacy() for side in execution.group.right
            )
            if len(right) != 2:
                raise ValueError(
                    'The V9 staged Navier-Stokes kernel requires two '
                    'mortar patches'
                )

            nb = len(group)
            uviews = {
                'uc': self._side_views(
                    be, elemap, left, '_scal_fpts'
                )
            }
            uviews.update({
                f'uf{i}': self._side_views(
                    be, elemap, side, '_scal_fpts'
                )
                for i, side in enumerate(right)
            })
            commviews = {
                'uc': self._side_views(
                    be, elemap, left, '_comm_fpts'
                )
            }
            commviews.update({
                f'uf{i}': self._side_views(
                    be, elemap, side, '_comm_fpts'
                )
                for i, side in enumerate(right)
            })

            gradviews = {}
            for d, view in enumerate(self._grad_views(
                be, elemap, left
            )):
                gradviews[f'gc{d}'] = view
            for i, side in enumerate(right):
                for d, view in enumerate(self._grad_views(
                    be, elemap, side
                )):
                    gradviews[f'gf{i}{d}'] = view

            # Alias buffers whose live ranges do not overlap.  The face
            # buffers are reused by the mortar-gradient phase, while the
            # common-flux buffers are reused by the face-gradient phase.
            fgext = be.extent()
            fgrp, gmgrp = fgext.alias_group(), fgext.alias_group()
            oext = be.extent()
            ogrp, gfgrp = oext.alias_group(), oext.alias_group()

            bufs = {
                'bc': be.matrix(
                    (ncfpts, nvars, nb), extent=fgrp, tags={'align'}
                )
            }
            bufs.update({
                f'bf{i}': be.matrix(
                    (ntfpts, nvars, nb), extent=fgrp, tags={'align'}
                )
                for i in range(len(right))
            })
            for i in range(len(right)):
                bufs[f'ul{i}'] = be.matrix(
                    (nmpts, nvars, nb), tags={'align'}
                )
                bufs[f'ur{i}'] = be.matrix(
                    (nmpts, nvars, nb), tags={'align'}
                )
            for i in range(len(right)):
                bufs[f'out{i}'] = be.matrix(
                    (nmpts, nvars, nb), extent=ogrp, tags={'align'}
                )

            for d in range(self.ndims):
                bufs[f'bcg{d}'] = be.matrix(
                    (ncfpts, nvars, nb), extent=gfgrp,
                    tags={'align'}
                )
                for i in range(len(right)):
                    bufs[f'bf{i}g{d}'] = be.matrix(
                        (ntfpts, nvars, nb), extent=gfgrp,
                        tags={'align'}
                    )
                for i in range(len(right)):
                    bufs[f'gl{i}{d}'] = be.matrix(
                        (nmpts, nvars, nb), extent=gmgrp,
                        tags={'align'}
                    )
                    bufs[f'gr{i}{d}'] = be.matrix(
                        (nmpts, nvars, nb), extent=gmgrp,
                        tags={'align'}
                    )

            be.commit_extent(fgext)
            be.commit_extent(oext)

            mats = {}
            for prefix in ('ic', 'if', 'pc', 'pf'):
                for i in range(len(right)):
                    name = f'{prefix}{i}'
                    mats[name] = be.const_matrix(op[name], tags={'align'})
            for i in range(len(right)):
                mats[f'nl{i}'] = be.const_matrix(
                    execution.geometry.scaled_normals[i],
                    tags={'align'}
                )

            batches.append({
                'n': nb, 'uviews': uviews, 'commviews': commviews,
                'gradviews': gradviews, 'bufs': bufs, 'mats': mats,
                'extent_bytes': fgext.nbytes + oext.nbytes,
                'execution': execution
            })

        self._staged_batches = batches
        self.nbatches = len(batches)
        self.staged_buffer_bytes = sum(
            matrix.nbytes for batch in batches
            for matrix in batch['bufs'].values()
        )
        self.staged_allocated_bytes = sum(
            batch['extent_bytes'] + sum(
                matrix.nbytes for name, matrix in batch['bufs'].items()
                if name.startswith(('ul', 'ur'))
            )
            for batch in batches
        )
        for stage, method in (
            ('con_u_gather', self._staged_con_u_gather),
            ('con_u_interp', self._staged_state_interp),
            ('con_u_eval', self._staged_con_u_eval),
            ('con_u_project', self._staged_con_u_project),
            ('con_u_scatter', self._staged_con_u_scatter),
            ('comm_flux_gather', self._staged_grad_gather),
            ('comm_flux_interp', self._staged_grad_interp),
            ('comm_flux_eval', self._staged_flux_eval),
            ('comm_flux_project', self._staged_flux_project),
            ('comm_flux_scatter', self._staged_flux_scatter),
        ):
            self.kernels[stage] = lambda method=method: [
                method(batch, tplargs) for batch in batches
            ]

    def _staged_con_u_gather(self, batch, tplargs):
        return self._be.kernel(
            'mortargather', tplargs=tplargs, dims=[batch['n']],
            **batch['uviews'], **{
                name: batch['bufs'][name]
                for name in ('bc', 'bf0', 'bf1')
            }
        )

    def _staged_state_interp(self, batch, tplargs):
        k, m, b = self._be.kernel, batch['mats'], batch['bufs']
        return self._be.unordered_meta_kernel([
            k('mul', m['ic0'], b['bc'], out=b['ul0']),
            k('mul', m['ic1'], b['bc'], out=b['ul1']),
            k('mul', m['if0'], b['bf0'], out=b['ur0']),
            k('mul', m['if1'], b['bf1'], out=b['ur1']),
        ])

    def _staged_con_u_eval(self, batch, tplargs):
        b = batch['bufs']
        return self._be.kernel(
            'mortarconustaged', tplargs=tplargs,
            dims=[tplargs['nmpts'], batch['n']],
            ul0=b['ul0'], ur0=b['ur0'], ul1=b['ul1'], ur1=b['ur1'],
            cu0=b['out0'], cu1=b['out1']
        )

    def _staged_con_u_project(self, batch, tplargs):
        k, m, b = self._be.kernel, batch['mats'], batch['bufs']
        return self._be.unordered_meta_kernel([
            k('mul', m['pc0'], b['out0'], out=b['bc']),
            k('mul', m['pc1'], b['out1'], out=b['bc'], beta=1.0),
            k('mul', m['pf0'], b['out0'], out=b['bf0']),
            k('mul', m['pf1'], b['out1'], out=b['bf1']),
        ])

    def _staged_con_u_scatter(self, batch, tplargs):
        return self._be.kernel(
            'mortarscatter', tplargs=tplargs, dims=[batch['n']],
            **batch['commviews'], **{
                name: batch['bufs'][name]
                for name in ('bc', 'bf0', 'bf1')
            }
        )

    def _staged_grad_gather(self, batch, tplargs):
        b = batch['bufs']
        return self._be.kernel(
            'mortargradgather', tplargs=tplargs, dims=[batch['n']],
            **batch['gradviews'], **{
                f'bc{d}': b[f'bcg{d}'] for d in range(self.ndims)
            }, **{
                f'bf0{d}': b[f'bf0g{d}'] for d in range(self.ndims)
            }, **{
                f'bf1{d}': b[f'bf1g{d}'] for d in range(self.ndims)
            }
        )

    def _staged_grad_interp(self, batch, tplargs):
        k, m, b = self._be.kernel, batch['mats'], batch['bufs']
        kerns = []
        for d in range(self.ndims):
            kerns.extend([
                k('mul', m['ic0'], b[f'bcg{d}'], out=b[f'gl0{d}']),
                k('mul', m['if0'], b[f'bf0g{d}'], out=b[f'gr0{d}']),
                k('mul', m['ic1'], b[f'bcg{d}'], out=b[f'gl1{d}']),
                k('mul', m['if1'], b[f'bf1g{d}'], out=b[f'gr1{d}']),
            ])

        return self._be.unordered_meta_kernel(kerns)

    def _staged_flux_eval(self, batch, tplargs):
        b, m = batch['bufs'], batch['mats']
        gradargs = {}
        for child in range(2):
            for d in range(self.ndims):
                gradargs[f'gl{child}{d}'] = b[f'gl{child}{d}']
                gradargs[f'gr{child}{d}'] = b[f'gr{child}{d}']

        return self._be.kernel(
            'mortarfluxstaged', tplargs=tplargs,
            dims=[tplargs['nmpts'], batch['n']],
            ul0=b['ul0'], ur0=b['ur0'], ul1=b['ul1'], ur1=b['ur1'],
            fc0=b['out0'], fc1=b['out1'],
            nl0=m['nl0'], nl1=m['nl1'], **gradargs
        )

    def _staged_flux_project(self, batch, tplargs):
        k, m, b = self._be.kernel, batch['mats'], batch['bufs']
        return self._be.unordered_meta_kernel([
            k('mul', m['pc0'], b['out0'], out=b['bc']),
            k('mul', m['pc1'], b['out1'], out=b['bc'], beta=1.0),
            k('mul', m['pf0'], b['out0'], out=b['bf0'], alpha=-1.0),
            k('mul', m['pf1'], b['out1'], out=b['bf1'], alpha=-1.0),
        ])

    def _staged_flux_scatter(self, batch, tplargs):
        return self._be.kernel(
            'mortarscatter', tplargs=tplargs, dims=[batch['n']],
            **batch['uviews'], **{
                name: batch['bufs'][name]
                for name in ('bc', 'bf0', 'bf1')
            }
        )

    def _init_general_staged(self, be, elemap, cfg, ops):
        tplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'rsolver': cfg.get('solver-interfaces', 'riemann-solver'),
            'visc_corr': cfg.get('solver', 'viscosity-correction', 'none'),
            'shock_capturing': 'none',
            'c': cfg.items_as('constants', float) |
                 cfg.items_as('solver-interfaces', float),
            'p_min': cfg.getfloat(
                'solver-interfaces', 'p-min', 5*be.fpdtype_eps
            ),
        }

        be.pointwise.register(
            'pyfr.solvers.baseadvec.kernels.mortarsidegather'
        )
        be.pointwise.register(
            'pyfr.solvers.baseadvec.kernels.mortarsidescatter'
        )
        be.pointwise.register(
            'pyfr.solvers.navstokes.kernels.mortarconupatch'
        )
        be.pointwise.register(
            'pyfr.solvers.navstokes.kernels.mortargradsidegather'
        )
        be.pointwise.register(
            'pyfr.solvers.navstokes.kernels.mortarfluxpatch'
        )

        executions = build_mortar_execution_batches(
            ops, elemap, 'navier-stokes', 'staged', self._ownership
        )
        self._execution_batches = executions

        batches = []
        for execution in executions:
            op = ops['operator_sets'][execution.operator_set]
            left = execution.group.left
            right = execution.group.right
            nb = len(execution.indices)
            npatches = len(right)

            if npatches != len(op['left_interp']):
                raise ValueError('Inconsistent mortar patch/operator count')
            if 'right_state_proj' not in op:
                raise ValueError('Missing mortar common-state projection')

            lnfpts = op['left_interp'][0].shape[1]
            rnfpts = tuple(
                matrix.shape[1] for matrix in op['right_interp']
            )
            nmpts = op['left_interp'][0].shape[0]

            uviews = {
                'left': mortar_side_view(
                    be, elemap, left, '_scal_fpts', (self.nvars,)
                ),
                'right': tuple(
                    mortar_side_view(
                        be, elemap, side, '_scal_fpts', (self.nvars,)
                    )
                    for side in right
                ),
            }
            commviews = {
                'left': mortar_side_view(
                    be, elemap, left, '_comm_fpts', (self.nvars,)
                ),
                'right': tuple(
                    mortar_side_view(
                        be, elemap, side, '_comm_fpts', (self.nvars,)
                    )
                    for side in right
                ),
            }
            gradviews = {
                'left': mortar_side_point_view(
                    be, elemap, left, 'get_vect_fpts_for_mortars',
                    (self.ndims, self.nvars)
                ),
                'right': tuple(
                    mortar_side_point_view(
                        be, elemap, side, 'get_vect_fpts_for_mortars',
                        (self.ndims, self.nvars)
                    )
                    for side in right
                ),
            }

            bufs = {
                'left': be.matrix(
                    (lnfpts, self.nvars, nb), tags={'align'}
                ),
                'right': tuple(
                    be.matrix((nfpts, self.nvars, nb), tags={'align'})
                    for nfpts in rnfpts
                ),
                'comm_left': be.matrix(
                    (lnfpts, self.nvars, nb), tags={'align'}
                ),
                'comm_right': tuple(
                    be.matrix((nfpts, self.nvars, nb), tags={'align'})
                    for nfpts in rnfpts
                ),
                'ul': tuple(
                    be.matrix((nmpts, self.nvars, nb), tags={'align'})
                    for _ in right
                ),
                'ur': tuple(
                    be.matrix((nmpts, self.nvars, nb), tags={'align'})
                    for _ in right
                ),
                'work': tuple(
                    be.matrix((nmpts, self.nvars, nb), tags={'align'})
                    for _ in right
                ),
                'grad_left': tuple(
                    be.matrix(
                        (lnfpts, self.nvars, nb), tags={'align'}
                    )
                    for _ in range(self.ndims)
                ),
                'grad_right': tuple(
                    tuple(
                        be.matrix(
                            (nfpts, self.nvars, nb), tags={'align'}
                        )
                        for _ in range(self.ndims)
                    )
                    for nfpts in rnfpts
                ),
                'gl': tuple(
                    tuple(
                        be.matrix(
                            (nmpts, self.nvars, nb), tags={'align'}
                        )
                        for _ in range(self.ndims)
                    )
                    for _ in right
                ),
                'gr': tuple(
                    tuple(
                        be.matrix(
                            (nmpts, self.nvars, nb), tags={'align'}
                        )
                        for _ in range(self.ndims)
                    )
                    for _ in right
                ),
            }
            mats = {
                name: tuple(
                    be.const_matrix(matrix, tags={'align'})
                    for matrix in op[name]
                )
                for name in (
                    'left_interp', 'right_interp', 'left_proj',
                    'right_proj', 'right_state_proj'
                )
            }
            mats['normals'] = tuple(
                be.const_matrix(normals, tags={'align'})
                for normals in execution.geometry.scaled_normals
            )

            batches.append({
                'n': nb, 'npatches': npatches, 'nmpts': nmpts,
                'lnfpts': lnfpts, 'rnfpts': rnfpts, 'uviews': uviews,
                'commviews': commviews, 'gradviews': gradviews,
                'bufs': bufs, 'mats': mats, 'execution': execution,
            })

        self._staged_batches = batches
        self.nbatches = len(batches)
        self.staged_buffer_bytes = sum(
            matrix.nbytes for batch in batches
            for matrix in self._general_matrices(batch['bufs'])
        )
        self.staged_allocated_bytes = self.staged_buffer_bytes

        for stage, method in (
            ('con_u_gather', self._general_con_u_gather),
            ('con_u_interp', self._general_state_interp),
            ('con_u_eval', self._general_con_u_eval),
            ('con_u_project', self._general_con_u_project),
            ('con_u_scatter', self._general_con_u_scatter),
            ('comm_flux_gather', self._general_grad_gather),
            ('comm_flux_interp', self._general_grad_interp),
            ('comm_flux_eval', self._general_flux_eval),
            ('comm_flux_project', self._general_flux_project),
            ('comm_flux_scatter', self._general_flux_scatter),
        ):
            self.kernels[stage] = lambda method=method: [
                method(batch, tplargs) for batch in batches
            ]

    @staticmethod
    def _general_matrices(obj):
        if isinstance(obj, tuple):
            for item in obj:
                yield from NavierStokesMortarInters._general_matrices(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                yield from NavierStokesMortarInters._general_matrices(item)
        else:
            yield obj

    def _general_side_gather(self, view, buf, nfpts, batch, tplargs):
        return self._be.kernel(
            'mortarsidegather',
            tplargs=tplargs | {'nfpts': nfpts}, dims=[batch['n']],
            u=view, b=buf
        )

    def _general_side_scatter(self, buf, view, nfpts, batch, tplargs):
        return self._be.kernel(
            'mortarsidescatter',
            tplargs=tplargs | {'nfpts': nfpts}, dims=[batch['n']],
            b=buf, u=view
        )

    def _general_con_u_gather(self, batch, tplargs):
        kerns = [self._general_side_gather(
            batch['uviews']['left'], batch['bufs']['left'],
            batch['lnfpts'], batch, tplargs
        )]
        kerns.extend(
            self._general_side_gather(view, buf, nfpts, batch, tplargs)
            for view, buf, nfpts in zip(
                batch['uviews']['right'], batch['bufs']['right'],
                batch['rnfpts']
            )
        )
        return self._be.unordered_meta_kernel(kerns)

    def _general_state_interp(self, batch, tplargs):
        be, mats, bufs = self._be, batch['mats'], batch['bufs']
        kerns = []
        for i in range(batch['npatches']):
            kerns.extend((
                be.kernel(
                    'mul', mats['left_interp'][i], bufs['left'],
                    out=bufs['ul'][i]
                ),
                be.kernel(
                    'mul', mats['right_interp'][i], bufs['right'][i],
                    out=bufs['ur'][i]
                ),
            ))
        return be.unordered_meta_kernel(kerns)

    def _general_con_u_eval(self, batch, tplargs):
        be, bufs = self._be, batch['bufs']
        return be.unordered_meta_kernel([
            be.kernel(
                'mortarconupatch', tplargs=tplargs,
                dims=[batch['nmpts'], batch['n']],
                ul=bufs['ul'][i], ur=bufs['ur'][i],
                cu=bufs['work'][i]
            )
            for i in range(batch['npatches'])
        ])

    def _general_con_u_project(self, batch, tplargs):
        be, mats, bufs = self._be, batch['mats'], batch['bufs']
        kerns = []
        for i in range(batch['npatches']):
            kerns.append(be.kernel(
                'mul', mats['left_proj'][i], bufs['work'][i],
                out=bufs['comm_left'], beta=0.0 if i == 0 else 1.0
            ))
            kerns.append(be.kernel(
                'mul', mats['right_state_proj'][i], bufs['work'][i],
                out=bufs['comm_right'][i]
            ))
        return be.ordered_meta_kernel(kerns)

    def _general_con_u_scatter(self, batch, tplargs):
        kerns = [self._general_side_scatter(
            batch['bufs']['comm_left'], batch['commviews']['left'],
            batch['lnfpts'], batch, tplargs
        )]
        kerns.extend(
            self._general_side_scatter(buf, view, nfpts, batch, tplargs)
            for buf, view, nfpts in zip(
                batch['bufs']['comm_right'], batch['commviews']['right'],
                batch['rnfpts']
            )
        )
        return self._be.unordered_meta_kernel(kerns)

    def _general_grad_buffer_views(self, bufs, nfpts, batch):
        n = nfpts*batch['n']
        rmap = np.tile(np.arange(nfpts, dtype=np.int32), batch['n'])
        cmap = np.repeat(np.arange(batch['n'], dtype=np.int32), nfpts)

        return tuple(
            self._be.view(
                np.full(n, buf.mid), rmap, cmap,
                np.ones(n, dtype=np.int32), vshape=(self.nvars,)
            )
            for buf in bufs
        )

    def _general_grad_side_gather(self, view, bufs, nfpts, batch, tplargs):
        bviews = self._general_grad_buffer_views(
            bufs, nfpts, batch
        )
        kwargs = {f'b{d}': bviews[d] for d in range(self.ndims)}
        return self._be.kernel(
            'mortargradsidegather', tplargs=tplargs,
            dims=[nfpts*batch['n']], g=view, **kwargs
        )

    def _general_grad_gather(self, batch, tplargs):
        kerns = [self._general_grad_side_gather(
            batch['gradviews']['left'], batch['bufs']['grad_left'],
            batch['lnfpts'], batch, tplargs
        )]
        kerns.extend(
            self._general_grad_side_gather(
                batch['gradviews']['right'][i],
                batch['bufs']['grad_right'][i], batch['rnfpts'][i],
                batch, tplargs
            )
            for i in range(batch['npatches'])
        )
        return self._be.unordered_meta_kernel(kerns)

    def _general_grad_interp(self, batch, tplargs):
        be, mats, bufs = self._be, batch['mats'], batch['bufs']
        kerns = []
        for i in range(batch['npatches']):
            for d in range(self.ndims):
                kerns.extend((
                    be.kernel(
                        'mul', mats['left_interp'][i],
                        bufs['grad_left'][d], out=bufs['gl'][i][d]
                    ),
                    be.kernel(
                        'mul', mats['right_interp'][i],
                        bufs['grad_right'][i][d], out=bufs['gr'][i][d]
                    ),
                ))
        return be.unordered_meta_kernel(kerns)

    def _general_flux_eval(self, batch, tplargs):
        be, mats, bufs = self._be, batch['mats'], batch['bufs']
        kerns = []
        for i in range(batch['npatches']):
            kwargs = {
                f'gl{d}': bufs['gl'][i][d] for d in range(self.ndims)
            }
            kwargs.update({
                f'gr{d}': bufs['gr'][i][d] for d in range(self.ndims)
            })
            kerns.append(be.kernel(
                'mortarfluxpatch', tplargs=tplargs,
                dims=[batch['nmpts'], batch['n']],
                ul=bufs['ul'][i], ur=bufs['ur'][i],
                nl=mats['normals'][i], fc=bufs['work'][i], **kwargs
            ))
        return be.unordered_meta_kernel(kerns)

    def _general_flux_project(self, batch, tplargs):
        be, mats, bufs = self._be, batch['mats'], batch['bufs']
        kerns = []
        for i in range(batch['npatches']):
            kerns.append(be.kernel(
                'mul', mats['left_proj'][i], bufs['work'][i],
                out=bufs['left'], beta=0.0 if i == 0 else 1.0
            ))
            kerns.append(be.kernel(
                'mul', mats['right_proj'][i], bufs['work'][i],
                out=bufs['right'][i], alpha=-1.0
            ))
        return be.ordered_meta_kernel(kerns)

    def _general_flux_scatter(self, batch, tplargs):
        kerns = [self._general_side_scatter(
            batch['bufs']['left'], batch['uviews']['left'],
            batch['lnfpts'], batch, tplargs
        )]
        kerns.extend(
            self._general_side_scatter(buf, view, nfpts, batch, tplargs)
            for buf, view, nfpts in zip(
                batch['bufs']['right'], batch['uviews']['right'],
                batch['rnfpts']
            )
        )
        return self._be.unordered_meta_kernel(kerns)
