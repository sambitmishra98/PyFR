import numpy as np

from pyfr.mortars import (
    MortarSide, build_line_line2_operators, build_quad_p_operators,
    build_quad_quad4_operators, build_quad_tri_operators,
)
from pyfr.solvers.base.mortars import (
    BaseMortarInters, build_mortar_execution_batches, local_mortar_ownership,
    mortar_face_view, mortar_side_view,
)


class _EulerMPIPMortarBatch:
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
        nb = self.ninters
        onfpts = ops['nleftfpts']
        nnfpts = ops['nrightfpts'][0]
        nmpts = ops['nmpts']
        lnfpts = onfpts if owner else nnfpts

        view = mortar_side_view(
            be, elemap, local, '_scal_fpts', (self.nvars,)
        )
        xchg = be.xchg_matrix(
            (nnfpts, self.nvars, nb), tags={'align'}
        )
        self._xchg = xchg

        tplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'rsolver': cfg.get('solver-interfaces', 'riemann-solver'),
            'c': cfg.items_as('constants', float),
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

        if owner:
            be.pointwise.register(
                'pyfr.solvers.euler.kernels.mortarfluxpatch'
            )
            native = be.matrix(
                (onfpts, self.nvars, nb), tags={'align'}
            )
            ul = be.matrix((nmpts, self.nvars, nb), tags={'align'})
            ur = be.matrix((nmpts, self.nvars, nb), tags={'align'})
            op = ops['operator_sets'][0]
            mats = {
                name: be.const_matrix(op[name][0], tags={'align'})
                for name in (
                    'left_interp', 'right_interp',
                    'left_proj', 'right_proj'
                )
            }
            normals = np.concatenate([
                f.geometry.scaled_normals[0] for f in faces
            ], axis=2)
            normals = be.const_matrix(normals, tags={'align'})

            self.kernels['state_gather'] = lambda: be.kernel(
                'mortarsidegather',
                tplargs=tplargs | {'nfpts': onfpts},
                dims=[nb], u=view, b=native
            )
            self.kernels['state_unpack'] = lambda: be.kernel(
                'unpack', xchg
            )
            self.kernels['state_interp'] = lambda: be.unordered_meta_kernel([
                be.kernel('mul', mats['left_interp'], native, out=ul),
                be.kernel('mul', mats['right_interp'], xchg, out=ur),
            ])
            self.kernels['flux_eval'] = lambda: be.kernel(
                'mortarfluxpatch', tplargs=tplargs, dims=[nmpts, nb],
                ul=ul, ur=ur, nl=normals
            )
            self.kernels['flux_project'] = lambda: be.ordered_meta_kernel([
                be.kernel('mul', mats['left_proj'], ul, out=native),
                be.kernel(
                    'mul', mats['right_proj'], ul, out=xchg, alpha=-1.0
                ),
            ])
            self.kernels['owner_scatter'] = lambda: be.kernel(
                'mortarsidescatter',
                tplargs=tplargs | {'nfpts': onfpts},
                dims=[nb], b=native, u=view
            )
            self.kernels['flux_pack'] = lambda: be.kernel('pack', xchg)
        else:
            self.kernels['state_gather'] = lambda: be.kernel(
                'mortarsidegather',
                tplargs=tplargs | {'nfpts': lnfpts},
                dims=[nb], u=view, b=xchg
            )
            self.kernels['state_pack'] = lambda: be.kernel('pack', xchg)
            self.kernels['flux_unpack'] = lambda: be.kernel('unpack', xchg)
            self.kernels['flux_scatter'] = lambda: be.kernel(
                'mortarsidescatter',
                tplargs=tplargs | {'nfpts': lnfpts},
                dims=[nb], b=xchg, u=view
            )

        tags = dict(plan.tags)
        nrank = plan.neighbour_rank
        if owner:
            self.mpireqs['mpi_p_state_recv'] = lambda: xchg.recvreq(
                comm, nrank, tags['state']
            )
            self.mpireqs['mpi_p_flux_send'] = lambda: xchg.sendreq(
                comm, nrank, tags['flux']
            )
        else:
            self.mpireqs['mpi_p_state_send'] = lambda: xchg.sendreq(
                comm, nrank, tags['state']
            )
            self.mpireqs['mpi_p_flux_recv'] = lambda: xchg.recvreq(
                comm, nrank, tags['flux']
            )

        self.buffer_bytes = xchg.nbytes
        if owner:
            self.buffer_bytes += native.nbytes + ul.nbytes + ur.nbytes



class EulerMortarInters(BaseMortarInters):
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
                _EulerMPIPMortarBatch(
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

        ops = build_quad_p_operators(mesh, elemap, group, cfg)
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
                operator_builder = build_quad_quad4_operators
            elif mcon.template == 'line-1x2':
                operator_builder = build_line_line2_operators
            else:
                raise ValueError(
                    f'Unsupported general mortar template {mcon.template!r}'
                )
            if impl != 'staged':
                raise ValueError(
                    'General one-to-many mortars require staged execution'
                )
            ops = operator_builder(mesh, elemap, mcon, cfg)
            general = True
        elif mcon.format == 'quad-tri-v1':
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
                raise ValueError(
                    f'Invalid mortar implementation {impl!r}'
                )

    def _tplargs(self, be, cfg, ops):
        return {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'ncfpts': ops['ncfpts'],
            'ntfpts': ops['ntfpts'],
            'nmpts': ops['nmpts'],
            'rsolver': cfg.get('solver-interfaces', 'riemann-solver'),
            'c': cfg.items_as('constants', float),
            'p_min': cfg.getfloat(
                'solver-interfaces', 'p-min', 5*be.fpdtype_eps
            ),
        }

    def _init_fused(self, be, elemap, cfg, ops):
        self.nbatches = 1
        self.staged_buffer_bytes = 0
        self.staged_allocated_bytes = 0
        self.uc = mortar_face_view(
            be, elemap, *ops['coarse'], '_scal_fpts', (self.nvars,)
        )
        self.uf0 = mortar_face_view(
            be, elemap, *ops['fine0'], '_scal_fpts', (self.nvars,)
        )
        self.uf1 = mortar_face_view(
            be, elemap, *ops['fine1'], '_scal_fpts', (self.nvars,)
        )

        mats = {
            name: be.const_matrix(ops[name], tags={'align'})
            for name in (
                'ic0', 'ic1', 'if0', 'if1', 'pc0', 'pc1',
                'pf0', 'pf1', 'nl0', 'nl1'
            )
        }
        tplargs = self._tplargs(be, cfg, ops)
        be.pointwise.register('pyfr.solvers.euler.kernels.mortarcflux')
        self.kernels['comm_flux'] = lambda: be.kernel(
            'mortarcflux', tplargs=tplargs, dims=[self.ninters],
            uc=self.uc, uf0=self.uf0, uf1=self.uf1, **mats
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
            'pyfr.solvers.euler.kernels.mortarfluxstaged'
        )

        execution_batches = build_mortar_execution_batches(
            ops, elemap, 'euler', 'staged', self._ownership
        )
        self._execution_batches = execution_batches

        batches = []
        for execution in execution_batches:
            group = np.asarray(execution.indices)
            op = ops['operator_sets'][execution.operator_set]
            left = execution.group.left
            right = execution.group.right
            if len(right) != 2:
                raise ValueError(
                    'The V9 staged Euler kernel requires two mortar patches'
                )

            nb = len(group)
            views = {
                'uc': mortar_side_view(
                    be, elemap, left, '_scal_fpts', (nvars,)
                )
            }
            views.update({
                f'uf{i}': mortar_side_view(
                    be, elemap, side, '_scal_fpts', (nvars,)
                )
                for i, side in enumerate(right)
            })

            bufs = {
                'bc': be.matrix((ncfpts, nvars, nb), tags={'align'})
            }
            bufs.update({
                f'bf{i}': be.matrix(
                    (ntfpts, nvars, nb), tags={'align'}
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
                'n': nb, 'views': views, 'bufs': bufs, 'mats': mats,
                'execution': execution
            })

        self._staged_batches = batches
        self.nbatches = len(batches)
        self.staged_buffer_bytes = sum(
            matrix.nbytes for batch in batches
            for matrix in batch['bufs'].values()
        )
        self.staged_allocated_bytes = self.staged_buffer_bytes
        self.kernels['comm_flux_gather'] = lambda: [
            self._staged_gather(b, tplargs) for b in batches
        ]
        self.kernels['comm_flux_interp'] = lambda: [
            self._staged_interp(b) for b in batches
        ]
        self.kernels['comm_flux_eval'] = lambda: [
            self._staged_eval(b, tplargs) for b in batches
        ]
        self.kernels['comm_flux_project'] = lambda: [
            self._staged_project(b) for b in batches
        ]
        self.kernels['comm_flux_scatter'] = lambda: [
            self._staged_scatter(b, tplargs) for b in batches
        ]

    def _staged_gather(self, batch, tplargs):
        return self._be.kernel(
            'mortargather', tplargs=tplargs, dims=[batch['n']],
            **batch['views'], **{
                name: batch['bufs'][name] for name in ('bc', 'bf0', 'bf1')
            }
        )

    def _staged_interp(self, batch):
        k, m, b = self._be.kernel, batch['mats'], batch['bufs']
        return self._be.unordered_meta_kernel([
            k('mul', m['ic0'], b['bc'], out=b['ul0']),
            k('mul', m['ic1'], b['bc'], out=b['ul1']),
            k('mul', m['if0'], b['bf0'], out=b['ur0']),
            k('mul', m['if1'], b['bf1'], out=b['ur1']),
        ])

    def _staged_eval(self, batch, tplargs):
        b, m = batch['bufs'], batch['mats']
        return self._be.kernel(
            'mortarfluxstaged', tplargs=tplargs,
            dims=[tplargs['nmpts'], batch['n']],
            ul0=b['ul0'], ur0=b['ur0'], ul1=b['ul1'], ur1=b['ur1'],
            nl0=m['nl0'], nl1=m['nl1']
        )

    def _staged_project(self, batch):
        k, m, b = self._be.kernel, batch['mats'], batch['bufs']
        return self._be.unordered_meta_kernel([
            k('mul', m['pc0'], b['ul0'], out=b['bc']),
            k('mul', m['pc1'], b['ul1'], out=b['bc'], beta=1.0),
            k('mul', m['pf0'], b['ul0'], out=b['bf0'], alpha=-1.0),
            k('mul', m['pf1'], b['ul1'], out=b['bf1'], alpha=-1.0),
        ])

    def _staged_scatter(self, batch, tplargs):
        return self._be.kernel(
            'mortarscatter', tplargs=tplargs, dims=[batch['n']],
            **batch['views'], **{
                name: batch['bufs'][name] for name in ('bc', 'bf0', 'bf1')
            }
        )

    def _init_general_staged(self, be, elemap, cfg, ops):
        tplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'rsolver': cfg.get('solver-interfaces', 'riemann-solver'),
            'c': cfg.items_as('constants', float),
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
            'pyfr.solvers.euler.kernels.mortarfluxpatch'
        )

        executions = build_mortar_execution_batches(
            ops, elemap, 'euler', 'staged', self._ownership
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

            lnfpts = op['left_interp'][0].shape[1]
            rnfpts = tuple(
                matrix.shape[1] for matrix in op['right_interp']
            )
            nmpts = op['left_interp'][0].shape[0]

            views = {
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
            bufs = {
                'left': be.matrix(
                    (lnfpts, self.nvars, nb), tags={'align'}
                ),
                'right': tuple(
                    be.matrix(
                        (nfpts, self.nvars, nb), tags={'align'}
                    )
                    for nfpts in rnfpts
                ),
                'ul': tuple(
                    be.matrix(
                        (nmpts, self.nvars, nb), tags={'align'}
                    )
                    for _ in right
                ),
                'ur': tuple(
                    be.matrix(
                        (nmpts, self.nvars, nb), tags={'align'}
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
                    'left_interp', 'right_interp',
                    'left_proj', 'right_proj'
                )
            }
            mats['normals'] = tuple(
                be.const_matrix(normals, tags={'align'})
                for normals in execution.geometry.scaled_normals
            )

            batches.append({
                'n': nb, 'npatches': npatches, 'nmpts': nmpts,
                'lnfpts': lnfpts, 'rnfpts': rnfpts, 'views': views,
                'bufs': bufs, 'mats': mats, 'execution': execution,
            })

        self._staged_batches = batches
        self.nbatches = len(batches)
        self.staged_buffer_bytes = sum(
            matrix.nbytes for batch in batches
            for name in ('left', 'right', 'ul', 'ur')
            for matrix in (
                batch['bufs'][name]
                if isinstance(batch['bufs'][name], tuple)
                else (batch['bufs'][name],)
            )
        )
        self.staged_allocated_bytes = self.staged_buffer_bytes

        self.kernels['comm_flux_gather'] = lambda: [
            self._general_gather(batch, tplargs) for batch in batches
        ]
        self.kernels['comm_flux_interp'] = lambda: [
            self._general_interp(batch) for batch in batches
        ]
        self.kernels['comm_flux_eval'] = lambda: [
            self._general_eval(batch, tplargs) for batch in batches
        ]
        self.kernels['comm_flux_project'] = lambda: [
            self._general_project(batch) for batch in batches
        ]
        self.kernels['comm_flux_scatter'] = lambda: [
            self._general_scatter(batch, tplargs) for batch in batches
        ]

    def _general_gather(self, batch, tplargs):
        be = self._be
        kerns = [
            be.kernel(
                'mortarsidegather',
                tplargs=tplargs | {'nfpts': batch['lnfpts']},
                dims=[batch['n']], u=batch['views']['left'],
                b=batch['bufs']['left']
            )
        ]
        kerns.extend(
            be.kernel(
                'mortarsidegather',
                tplargs=tplargs | {'nfpts': nfpts},
                dims=[batch['n']], u=view, b=buf
            )
            for nfpts, view, buf in zip(
                batch['rnfpts'], batch['views']['right'],
                batch['bufs']['right']
            )
        )
        return be.unordered_meta_kernel(kerns)

    def _general_interp(self, batch):
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

    def _general_eval(self, batch, tplargs):
        be, mats, bufs = self._be, batch['mats'], batch['bufs']
        return be.unordered_meta_kernel([
            be.kernel(
                'mortarfluxpatch', tplargs=tplargs,
                dims=[batch['nmpts'], batch['n']],
                ul=bufs['ul'][i], ur=bufs['ur'][i],
                nl=mats['normals'][i]
            )
            for i in range(batch['npatches'])
        ])

    def _general_project(self, batch):
        be, mats, bufs = self._be, batch['mats'], batch['bufs']
        kerns = []
        for i in range(batch['npatches']):
            kerns.append(be.kernel(
                'mul', mats['left_proj'][i], bufs['ul'][i],
                out=bufs['left'], beta=0.0 if i == 0 else 1.0
            ))
            kerns.append(be.kernel(
                'mul', mats['right_proj'][i], bufs['ul'][i],
                out=bufs['right'][i], alpha=-1.0
            ))
        return be.ordered_meta_kernel(kerns)

    def _general_scatter(self, batch, tplargs):
        be = self._be
        kerns = [
            be.kernel(
                'mortarsidescatter',
                tplargs=tplargs | {'nfpts': batch['lnfpts']},
                dims=[batch['n']], b=batch['bufs']['left'],
                u=batch['views']['left']
            )
        ]
        kerns.extend(
            be.kernel(
                'mortarsidescatter',
                tplargs=tplargs | {'nfpts': nfpts},
                dims=[batch['n']], b=buf, u=view
            )
            for nfpts, view, buf in zip(
                batch['rnfpts'], batch['views']['right'],
                batch['bufs']['right']
            )
        )
        return be.unordered_meta_kernel(kerns)
