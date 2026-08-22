from dataclasses import dataclass

import numpy as np

from pyfr.quadrules import get_quadrule
from pyfr.shapes import (
    HexShape, LineShape, QuadShape, TriShape, proj_l2, proj_pts,
)


def _mass_matrix(basis, qrule):
    interp = basis.nodal_basis_at(qrule.pts)
    return interp.T @ (qrule.wts[:, None]*interp)


def _affine_map(src, dst):
    lhs = np.column_stack((src, np.ones(len(src))))
    coeff = np.linalg.solve(lhs, dst)
    det = abs(np.linalg.det(coeff[:2]))

    def apply(pts):
        return np.column_stack((pts, np.ones(len(pts)))) @ coeff

    return apply, det


def _fit_affine_map(src, dst, tol=1e-12):
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.ndim != 2 or src.shape[1] != 2 or len(src) < 3:
        raise ValueError('Invalid affine-map source points')
    if dst.ndim != 2 or len(dst) != len(src):
        raise ValueError('Invalid affine-map target points')

    lhs = np.column_stack((src, np.ones(len(src))))
    coeff, _, rank, _ = np.linalg.lstsq(lhs, dst, rcond=None)
    if rank != 3:
        raise ValueError('Degenerate affine-map source points')

    fitted = lhs @ coeff
    scale = max(1.0, float(np.max(np.abs(dst), initial=0.0)))
    if np.max(np.abs(fitted - dst), initial=0.0) > tol*scale:
        raise ValueError('Mortar reference map is not affine')

    det = abs(np.linalg.det(coeff[:2, :2]))

    def apply(pts):
        pts = np.asarray(pts, dtype=float)
        return np.column_stack((pts, np.ones(len(pts)))) @ coeff

    return apply, det


def _fit_line_affine_map(src, dst, tol=1e-12):
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.ndim != 2 or src.shape[1] != 1 or len(src) < 2:
        raise ValueError('Invalid line affine-map source points')
    if dst.ndim != 2 or len(dst) != len(src):
        raise ValueError('Invalid line affine-map target points')

    lhs = np.column_stack((src, np.ones(len(src))))
    coeff, _, rank, _ = np.linalg.lstsq(lhs, dst, rcond=None)
    if rank != 2:
        raise ValueError('Degenerate line affine-map source points')

    fitted = lhs @ coeff
    scale = max(1.0, float(np.max(np.abs(dst), initial=0.0)))
    if np.max(np.abs(fitted - dst), initial=0.0) > tol*scale:
        raise ValueError('Line mortar reference map is not affine')

    det = abs(float(coeff[0, 0]))

    def apply(pts):
        pts = np.asarray(pts, dtype=float)
        return np.column_stack((pts, np.ones(len(pts)))) @ coeff

    return apply, det


def _face_corner_nodes(mesh, elemap, etype, eidx, fidx):
    eles = elemap[etype]
    idxs = eles.basis.face_corner_pts_idxs(fidx, eles.nspts)
    return mesh.spts_nodes[etype][eidx, idxs]


def _face_corner_locs(mesh, elemap, etype, eidx, fidx):
    eles = elemap[etype]
    idxs = eles.basis.face_corner_pts_idxs(fidx, eles.nspts)
    return mesh.spts[etype][idxs, eidx]


def _face_sample_maps_from_basis(shape, kind, basis):
    if 'surf-flux' not in shape.antialias:
        npts = len(basis.pts)
        ident = np.eye(npts)
        return ident, ident

    qrule = shape._iqrules[kind]
    to_nodal = proj_l2(qrule, basis)
    from_nodal = basis.nodal_basis_at(qrule.pts)

    return to_nodal, from_nodal


def _face_sample_maps(eles, kind, basis):
    if 'surf-flux' not in eles.antialias:
        npts = len(basis.pts)
        ident = np.eye(npts)
        return ident, ident

    qrule = eles.basis._iqrules[kind]
    to_nodal = proj_l2(qrule, basis)
    from_nodal = basis.nodal_basis_at(qrule.pts)

    return to_nodal, from_nodal


@dataclass(frozen=True, eq=False)
class MortarGeometry:
    authority: str
    mortar_topology: str
    reference_points: np.ndarray
    quadrature_weights: np.ndarray
    physical_points: tuple
    scaled_normals: tuple
    patch_determinants: tuple
    coordinate_errors: tuple
    normal_errors: tuple

    def __post_init__(self):
        patch_data = (
            self.physical_points, self.scaled_normals,
            self.patch_determinants, self.coordinate_errors,
            self.normal_errors,
        )
        npatches = len(self.scaled_normals)
        if (
            not npatches
            or any(len(values) != npatches for values in patch_data)
        ):
            raise ValueError('Inconsistent mortar geometry patch count')
        if len(self.reference_points) != len(self.quadrature_weights):
            raise ValueError('Inconsistent mortar quadrature geometry')

        nmpts = len(self.reference_points)
        for points, normals, det, cerr, nerr in zip(*patch_data):
            if points.shape != normals.shape or points.ndim != 3:
                raise ValueError('Invalid mortar physical geometry shape')
            if points.shape[0] != nmpts:
                raise ValueError('Invalid mortar geometry point count')

            nrecords = points.shape[-1]
            if any(values.shape != (nrecords,) for values in (
                det, cerr, nerr
            )):
                raise ValueError('Invalid mortar geometry record count')

    @property
    def mortar_scaled_normals(self):
        return tuple(
            normals*det[None, None, :]
            for normals, det in zip(
                self.scaled_normals, self.patch_determinants
            )
        )

    @property
    def surface_jacobians(self):
        return tuple(
            np.linalg.norm(normals, axis=1)
            for normals in self.mortar_scaled_normals
        )

    @property
    def integration_weights(self):
        return tuple(
            self.quadrature_weights[:, None]*sjac
            for sjac in self.surface_jacobians
        )

    @property
    def max_coordinate_error(self):
        return float(max(
            (errors.max(initial=0.0) for errors in self.coordinate_errors),
            default=0.0,
        ))

    @property
    def max_normal_error(self):
        return float(max(
            (errors.max(initial=0.0) for errors in self.normal_errors),
            default=0.0,
        ))

    @property
    def backend_nbytes(self):
        return sum(normals.nbytes for normals in self.scaled_normals)

    @property
    def host_nbytes(self):
        arrays = (
            self.reference_points, self.quadrature_weights,
            *self.physical_points, *self.scaled_normals,
            *self.patch_determinants, *self.coordinate_errors,
            *self.normal_errors,
        )
        return sum(array.nbytes for array in arrays)

    def subset(self, indices):
        indices = np.asarray(indices, dtype=np.intp)

        return MortarGeometry(
            self.authority, self.mortar_topology, self.reference_points,
            self.quadrature_weights,
            tuple(points[..., indices] for points in self.physical_points),
            tuple(normals[..., indices] for normals in self.scaled_normals),
            tuple(det[indices] for det in self.patch_determinants),
            tuple(errors[indices] for errors in self.coordinate_errors),
            tuple(errors[indices] for errors in self.normal_errors),
        )


@dataclass(frozen=True)
class MortarSide:
    etype: str
    face_topology: str
    fidxs: tuple
    eidxs: tuple
    elekey: object = None

    @property
    def runtime_key(self):
        return self.elekey if self.elekey is not None else self.etype

    def as_legacy(self):
        return (
            self.etype,
            np.asarray(self.fidxs, dtype=np.int32),
            np.asarray(self.eidxs, dtype=np.int64),
        )


@dataclass(frozen=True)
class MortarGroup:
    left: MortarSide
    right: tuple


@dataclass(frozen=True)
class MortarTraceKey:
    etype: str
    face_topology: str
    solution_order: int
    fidx: int


@dataclass(frozen=True)
class MortarReferenceMap:
    source_topology: str
    target: tuple

    def __post_init__(self):
        ncorner = {'line': 2, 'tri': 3, 'quad': 4}.get(
            self.source_topology
        )
        if ncorner is None or len(self.target) != ncorner:
            raise ValueError('Invalid mortar reference-map topology')

        target = np.asarray(self.target, dtype=float)
        ndims = 1 if self.source_topology == 'line' else 2
        if target.shape != (ncorner, ndims):
            raise ValueError('Invalid mortar reference-map target')

        if self.source_topology == 'line':
            src = LineShape.std_ele(1)
            _, det = _fit_line_affine_map(src, target)
        else:
            src = (
                TriShape.std_ele(1) if self.source_topology == 'tri'
                else QuadShape.std_ele(1)
            )
            _, det = _fit_affine_map(src, target)
        if det <= 0:
            raise ValueError('Degenerate mortar reference map')

    def affine(self):
        target = np.asarray(self.target, dtype=float)
        if self.source_topology == 'line':
            return _fit_line_affine_map(LineShape.std_ele(1), target)

        src = (
            TriShape.std_ele(1) if self.source_topology == 'tri'
            else QuadShape.std_ele(1)
        )
        return _fit_affine_map(src, target)

    @property
    def determinant(self):
        return self.affine()[1]

    def apply(self, pts):
        return self.affine()[0](pts)


@dataclass(frozen=True, order=True)
class MPIMortarFaceKey:
    etype: str
    global_eidx: int
    fidx: int


@dataclass(frozen=True)
class RemoteMortarSide:
    etype: str
    face_topology: str
    order: int
    geometry_order: int
    fidx: int
    global_eidx: int


@dataclass(frozen=True, eq=False)
class DistributedPMortarFace:
    neighbour_rank: int
    local_side: MortarSide
    remote_side: RemoteMortarSide
    local_key: MPIMortarFaceKey
    remote_key: MPIMortarFaceKey
    local_geometry_order: int
    owner_rank: int
    reference_map: MortarReferenceMap
    geometry: MortarGeometry

    @property
    def face_pair(self):
        return tuple(sorted((self.local_key, self.remote_key)))

    @property
    def local_is_owner(self):
        return self.local_key == self.face_pair[0]

    @property
    def owner_key(self):
        return self.face_pair[0]

    @property
    def nonowner_key(self):
        return self.face_pair[1]

    @property
    def owner_order(self):
        return (
            self.local_side.elekey.order if self.local_is_owner
            else self.remote_side.order
        )

    @property
    def nonowner_order(self):
        return (
            self.remote_side.order if self.local_is_owner
            else self.local_side.elekey.order
        )

    @property
    def owner_geometry_order(self):
        return (
            self.local_geometry_order if self.local_is_owner
            else self.remote_side.geometry_order
        )

    @property
    def nonowner_geometry_order(self):
        return (
            self.remote_side.geometry_order if self.local_is_owner
            else self.local_geometry_order
        )


@dataclass(frozen=True)
class MortarQuadratureKey:
    topology: str
    rule: str = None
    npts: int = None
    qdeg: int = None


@dataclass(frozen=True)
class MortarPatch:
    right: MortarTraceKey
    left_map: MortarReferenceMap


@dataclass(frozen=True)
class MortarOperatorKey:
    left: MortarTraceKey
    patches: tuple
    mortar_topology: str
    mortar_order: int
    quadrature: MortarQuadratureKey
    trace_sampling: tuple


def _mortar_side_ele(elemap, side):
    return elemap[side.runtime_key]


def _runtime_face_corner_locs(eles, eidx, fidx):
    kind, proj, _ = eles.basis.faces[fidx]
    if kind != 'quad':
        raise ValueError('V10C2A p-mortars require quadrilateral faces')

    qcorners = QuadShape.std_ele(1)
    return eles.ploc_at_np(proj_pts(proj, qcorners))[:, :, eidx]


def _quad_full_face_reference_map(left, right, tol=1e-12):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.shape[0] != 4:
        raise ValueError('Invalid p-mortar quad corner geometry')

    scale = max(1.0, float(np.max(np.abs(left), initial=0.0)),
                float(np.max(np.abs(right), initial=0.0)))
    dist = np.max(np.abs(right[:, None, :] - left[None, :, :]), axis=2)
    match = np.argmin(dist, axis=1)
    errors = dist[np.arange(4), match]

    if len(set(map(int, match))) != 4 or np.max(errors) > tol*scale:
        raise ValueError('Mixed-p interface face corners are not coincident')

    qcorners = QuadShape.std_ele(1)
    rmap = MortarReferenceMap(
        'quad', tuple(map(tuple, qcorners[match]))
    )
    if not np.isclose(rmap.determinant, 1.0, rtol=0, atol=40*tol):
        raise ValueError('Mixed-p full-face reference map must have unit area')

    return rmap


def _quad_p_operator_key(etypes, fidxs, index, rmap, orders,
                         trace_sampling):
    lorder, rorder = map(int, orders)
    morder = max(lorder, rorder)
    left = MortarTraceKey(etypes[0], 'quad', lorder, fidxs[0][index])
    patch = MortarPatch(
        MortarTraceKey(etypes[1], 'quad', rorder, fidxs[1][index]),
        rmap,
    )
    quadrature = MortarQuadratureKey(
        'quad', rule='gauss-legendre', npts=(morder + 2)**2
    )
    return MortarOperatorKey(
        left, (patch,), 'quad', morder, quadrature, trace_sampling
    )


def build_quad_p_reference_operators(
    cfg, etypes, fidxs, orders, maps, *, state_projection=False
):
    etypes = tuple(etypes)
    fidxs = tuple(tuple(map(int, fs)) for fs in fidxs)
    orders = tuple(map(int, orders))
    maps = tuple(maps)

    if etypes != ('hex', 'hex'):
        raise ValueError('V10C2A supports Hex/quad p-mortars only')
    if orders[0] == orders[1] or abs(orders[0] - orders[1]) != 1:
        raise ValueError('V10C2A requires adjacent unequal solution orders')
    if len(fidxs) != 2 or len(fidxs[0]) != len(fidxs[1]):
        raise ValueError('Inconsistent p-mortar face identity')
    if len(maps) != len(fidxs[0]):
        raise ValueError('Inconsistent p-mortar reference maps')

    shapes = tuple(HexShape(None, cfg, order=p) for p in orders)
    bases = tuple(shape.facebases['quad'] for shape in shapes)
    sample_maps = tuple(
        _face_sample_maps_from_basis(shape, 'quad', basis)
        for shape, basis in zip(shapes, bases)
    )
    trace_sampling = tuple(
        'surf-flux' if 'surf-flux' in shape.antialias else 'nodal'
        for shape in shapes
    )

    nfacefpts = []
    for fs, shape in zip(fidxs, shapes):
        if not fs:
            raise ValueError('Empty p-mortar face set')
        npts = shape.nfacefpts[fs[0]]
        if any(shape.nfacefpts[fidx] != npts for fidx in fs):
            raise ValueError('Inconsistent p-mortar face point count')
        nfacefpts.append(npts)

    morder = max(orders)
    mqrule = get_quadrule(
        'quad', rule='gauss-legendre', npts=(morder + 2)**2
    )
    mass_qrule = get_quadrule(
        'quad', rule='gauss-legendre', npts=(morder + 2)**2
    )
    masses = tuple(_mass_matrix(basis, mass_qrule) for basis in bases)
    keys = [
        _quad_p_operator_key(
            etypes, fidxs, i, rmap, orders, trace_sampling
        )
        for i, rmap in enumerate(maps)
    ]

    unique_keys, opidx = _unique_signatures(keys)
    operator_sets = []
    for key in unique_keys:
        rmap = key.patches[0].left_map
        lqpts = rmap.apply(mqrule.pts)
        lint = bases[0].nodal_basis_at(lqpts)
        rint = bases[1].nodal_basis_at(mqrule.pts)
        weights = mqrule.wts*rmap.determinant
        lto, lfrom = sample_maps[0]
        rto, rfrom = sample_maps[1]

        opset = {
            'left_interp': (lint @ lto,),
            'right_interp': (rint @ rto,),
            'left_proj': (lfrom @ np.linalg.solve(
                masses[0], lint.T*weights
            ),),
            'right_proj': (rfrom @ np.linalg.solve(
                masses[1], rint.T*weights
            ),),
        }
        if state_projection:
            opset['right_state_proj'] = (
                rfrom @ np.linalg.solve(masses[1], rint.T*mqrule.wts),
            )
        operator_sets.append(opset)

    operator_groups = [
        np.flatnonzero(opidx == oi) for oi in range(len(unique_keys))
    ]
    shared_opbytes = sum(
        matrix.nbytes
        for opset in operator_sets
        for values in opset.values()
        for matrix in values
    )

    return {
        'operator_keys': unique_keys,
        'operator_sets': operator_sets,
        'operator_groups': operator_groups,
        'nleftfpts': nfacefpts[0],
        'nrightfpts': (nfacefpts[1],),
        'nmpts': len(mqrule.pts),
        'nops': len(unique_keys),
        'shared_operator_bytes': shared_opbytes,
        'trace_sampling': trace_sampling,
        'reference_points': np.asarray(mqrule.pts),
        'quadrature_weights': np.asarray(mqrule.wts),
    }


def _quad_face_geometry_at(eles, eidx, fidx, qpts):
    kind, proj, norm = eles.basis.faces[fidx]
    if kind != 'quad':
        raise ValueError('V10C3A p-mortars require quadrilateral faces')

    vpts = proj_pts(proj, qpts)
    ploc = eles.ploc_at_np(vpts)[:, :, eidx]
    normals = np.broadcast_to(norm, (len(qpts), eles.ndims))
    pnorm = eles.pnorm_at(vpts, normals)[:, eidx]
    return ploc, pnorm


def build_distributed_quad_p_operators(
    face, cfg, *, state_projection=False
):
    if face.owner_geometry_order != face.nonowner_geometry_order:
        raise ValueError('V10C3A requires equal geometry order')

    if face.local_is_owner:
        etypes = (face.local_side.etype, face.remote_side.etype)
        fidxs = ((face.local_side.fidxs[0],), (face.remote_side.fidx,))
    else:
        etypes = (face.remote_side.etype, face.local_side.etype)
        fidxs = ((face.remote_side.fidx,), (face.local_side.fidxs[0],))

    refops = build_quad_p_reference_operators(
        cfg, etypes, fidxs, (face.owner_order, face.nonowner_order),
        (face.reference_map,), state_projection=state_projection
    )
    return {
        'distributed_face': face,
        **refops,
        'geometry': face.geometry,
        'max_geom_error': face.geometry.max_coordinate_error,
        'max_normal_error': face.geometry.max_normal_error,
        'patch_determinants': (
            np.asarray([face.reference_map.determinant]),
        ),
    }


def build_quad_p_operators(
    mesh, elemap, group, cfg, *, state_projection=False
):
    geom_tol = cfg.getfloat('solver-interfaces', 'mortar-geom-tol', 1e-10)

    if len(group.right) != 1:
        raise ValueError('V10C2A p-mortars require one right participant')
    sides = (group.left, group.right[0])
    if any(side.etype != 'hex' or side.face_topology != 'quad'
           for side in sides):
        raise ValueError('V10C2A supports Hex/quad p-mortars only')

    eles = tuple(_mortar_side_ele(elemap, side) for side in sides)
    orders = tuple(ele.basis.order for ele in eles)
    if orders[0] == orders[1] or abs(orders[0] - orders[1]) != 1:
        raise ValueError('V10C2A requires adjacent unequal solution orders')
    if eles[0].basis.nsptsord != eles[1].basis.nsptsord:
        raise ValueError('V10C2A requires equal geometry order')

    maps = []
    affine_errors = []
    for i in range(len(group.left.eidxs)):
        lcorners = _runtime_face_corner_locs(
            eles[0], group.left.eidxs[i], group.left.fidxs[i]
        )
        rcorners = _runtime_face_corner_locs(
            eles[1], group.right[0].eidxs[i], group.right[0].fidxs[i]
        )
        rmap = _quad_full_face_reference_map(lcorners, rcorners, geom_tol)
        maps.append(rmap)

        laff = _surface_affine_map(QuadShape.std_ele(1), lcorners, geom_tol)
        raff = _surface_affine_map(QuadShape.std_ele(1), rcorners, geom_tol)
        morder = max(orders)
        mqrule = get_quadrule(
            'quad', rule='gauss-legendre', npts=(morder + 2)**2
        )
        lqpts = rmap.apply(mqrule.pts)
        lploc, _ = _quad_face_geometry_at(
            eles[0], group.left.eidxs[i], group.left.fidxs[i], lqpts
        )
        rploc, _ = _quad_face_geometry_at(
            eles[1], group.right[0].eidxs[i],
            group.right[0].fidxs[i], mqrule.pts
        )
        affine_errors.append(max(
            float(np.max(np.abs(lploc - laff(lqpts)))),
            float(np.max(np.abs(rploc - raff(mqrule.pts)))),
        ))

    if max(affine_errors, default=0.0) > geom_tol:
        raise ValueError('V10C2A p-mortars require affine face geometry')

    fidxs = (group.left.fidxs, group.right[0].fidxs)
    refops = build_quad_p_reference_operators(
        cfg, (sides[0].etype, sides[1].etype), fidxs, orders, maps,
        state_projection=state_projection
    )
    mqpts = refops['reference_points']
    mqwts = refops['quadrature_weights']

    physical_points = []
    normals = []
    geom_errors = []
    normal_errors = []
    determinants = []
    for i, rmap in enumerate(maps):
        lqpts = rmap.apply(mqpts)
        lploc, lpnorm = _quad_face_geometry_at(
            eles[0], group.left.eidxs[i], group.left.fidxs[i], lqpts
        )
        rploc, rpnorm = _quad_face_geometry_at(
            eles[1], group.right[0].eidxs[i],
            group.right[0].fidxs[i], mqpts
        )
        physical_points.append(lploc)
        normals.append(lpnorm)
        geom_errors.append(np.max(np.abs(lploc - rploc)))
        normal_errors.append(np.max(np.abs(rpnorm + lpnorm)))
        determinants.append(rmap.determinant)

    def stack(values):
        return np.stack(values, axis=-1)

    geometry = MortarGeometry(
        'left', 'quad', mqpts, mqwts,
        (stack(physical_points),), (stack(normals),),
        (np.asarray(determinants),), (np.asarray(geom_errors),),
        (np.asarray(normal_errors),),
    )
    if geometry.max_coordinate_error > geom_tol:
        raise ValueError(
            f'P-mortar geometry mismatch {geometry.max_coordinate_error:.3e} '
            f'exceeds tolerance {geom_tol:.3e}'
        )
    if geometry.max_normal_error > 10*geom_tol:
        raise ValueError(
            f'P-mortar normal mismatch {geometry.max_normal_error:.3e} '
            f'exceeds tolerance {10*geom_tol:.3e}'
        )

    return {
        'mortar_group': group,
        **refops,
        'geometry': geometry,
        'max_geom_error': geometry.max_coordinate_error,
        'max_normal_error': geometry.max_normal_error,
        'patch_determinants': (np.asarray(determinants),),
    }
