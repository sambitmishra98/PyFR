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

def _side_info(records, cidxmap, side, face_topology, child=None):
    if side == 'coarse':
        cidxs = records['coarse_cidx']
        eidxs = records['coarse_eidx']
    else:
        cidxs = records['fine_cidx'][:, child]
        eidxs = records['fine_eidx'][:, child]

    info = [cidxmap[int(cidx)] for cidx in cidxs]
    etypes = [etype for etype, _ in info]
    fidxs = tuple(int(fidx) for _, fidx in info)
    eidxs = tuple(int(eidx) for eidx in eidxs)

    if len(set(etypes)) != 1:
        raise ValueError('A mortar batch must have one element type per side')

    return MortarSide(etypes[0], face_topology, fidxs, eidxs)


def _decode_quad_tri_mortar(mesh, mcon):
    records = mcon.records
    left = _side_info(records, mesh.cidxmap, 'coarse', 'quad')
    right = tuple(
        _side_info(records, mesh.cidxmap, 'fine', 'tri', child)
        for child in range(2)
    )

    if any(side.etype != 'tet' for side in right):
        raise ValueError(
            'Quad-triangle mortars require two tetrahedral fine elements'
        )

    return MortarGroup(left, right)


def _general_side_info(records, cidxmap, side, face_topology, child=None):
    cfield = f'{side}_cidx'
    efield = f'{side}_eidx'
    if child is None:
        cidxs = records[cfield]
        eidxs = records[efield]
    else:
        cidxs = records[cfield][:, child]
        eidxs = records[efield][:, child]

    info = [cidxmap[int(cidx)] for cidx in cidxs]
    etypes = [etype for etype, _ in info]
    fidxs = tuple(int(fidx) for _, fidx in info)
    eidxs = tuple(int(eidx) for eidx in eidxs)

    if len(set(etypes)) != 1:
        raise ValueError('A mortar batch must have one element type per side')

    return MortarSide(etypes[0], face_topology, fidxs, eidxs)


def _decode_general_mortar(mesh, mcon):
    if mcon.format != 'one-to-many-v1':
        raise ValueError(f'Unsupported general mortar format {mcon.format!r}')
    if mcon.template == 'quad-2x2':
        face_topology = 'quad'
        nright = 4
    elif mcon.template == 'line-1x2':
        face_topology = 'line'
        nright = 2
    else:
        raise ValueError(
            f'Unsupported general mortar template {mcon.template!r}'
        )
    if mcon.nright != nright:
        if mcon.template == 'quad-2x2':
            raise ValueError(
                'quad-2x2 mortars require four right participants'
            )
        raise ValueError('line-1x2 mortars require two right participants')

    records = mcon.records
    left = _general_side_info(
        records, mesh.cidxmap, 'left', face_topology
    )
    right = tuple(
        _general_side_info(
            records, mesh.cidxmap, 'right', face_topology, child
        )
        for child in range(mcon.nright)
    )
    return MortarGroup(left, right)


def _surface_affine_map(src, dst, tol):
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    lhs = np.column_stack((src, np.ones(len(src))))
    coeff, _, rank, _ = np.linalg.lstsq(lhs, dst, rcond=None)
    if rank != 3:
        raise ValueError('Degenerate mortar face geometry')

    fitted = lhs @ coeff
    scale = max(1.0, float(np.max(np.abs(dst), initial=0.0)))
    if np.max(np.abs(fitted - dst), initial=0.0) > tol*scale:
        raise ValueError('V10B quad mortar requires affine face geometry')

    def apply(pts):
        pts = np.asarray(pts, dtype=float)
        return np.column_stack((pts, np.ones(len(pts)))) @ coeff

    return apply


def _quad_2x2_reference_maps(mesh, elemap, group, index, tol):
    lnodes = _face_corner_locs(
        mesh, elemap, group.left.etype, group.left.eidxs[index],
        group.left.fidxs[index]
    )

    grid = np.array([
        (u, v) for v in (-1.0, 0.0, 1.0)
        for u in (-1.0, 0.0, 1.0)
    ])
    left_ele = elemap[group.left.etype]
    if hasattr(left_ele, 'ploc_at_np') and hasattr(left_ele.basis, 'faces'):
        grid_phys = _quad_face_geometry_at(
            left_ele, group.left.eidxs[index], group.left.fidxs[index], grid
        )[0]
    else:
        grid_phys = _surface_affine_map(QuadShape.std_ele(1), lnodes, tol)(
            grid
        )
    quadrants = (
        frozenset({(-1.0, -1.0), (0.0, -1.0),
                   (-1.0, 0.0), (0.0, 0.0)}),
        frozenset({(0.0, -1.0), (1.0, -1.0),
                   (0.0, 0.0), (1.0, 0.0)}),
        frozenset({(-1.0, 0.0), (0.0, 0.0),
                   (-1.0, 1.0), (0.0, 1.0)}),
        frozenset({(0.0, 0.0), (1.0, 0.0),
                   (0.0, 1.0), (1.0, 1.0)}),
    )

    scale = max(1.0, float(np.max(np.abs(lnodes), initial=0.0)))
    maps = []
    used = set()
    for slot, side in enumerate(group.right):
        rnodes = _face_corner_locs(
            mesh, elemap, side.etype, side.eidxs[index], side.fidxs[index]
        )
        target = []
        matched = set()
        for node in rnodes:
            errors = np.max(np.abs(grid_phys - node), axis=1)
            match = np.flatnonzero(errors <= tol*scale)
            if len(match) != 1:
                raise ValueError(
                    'Fine quad mortar corner does not match coarse 2x2 grid'
                )
            gi = int(match[0])
            if gi in matched:
                raise ValueError('Duplicate fine quad mortar corner')
            matched.add(gi)
            target.append(tuple(float(v) for v in grid[gi]))

        if frozenset(target) != quadrants[slot]:
            raise ValueError('Fine quad mortar slot does not match template')
        if frozenset(target) in used:
            raise ValueError('Duplicate quad-2x2 mortar patch')
        used.add(frozenset(target))

        rmap = MortarReferenceMap('quad', tuple(target))
        if not np.isclose(rmap.determinant, 0.25, rtol=0, atol=20*tol):
            raise ValueError('Invalid quad-2x2 mortar patch determinant')
        maps.append(rmap)

    if used != set(quadrants):
        raise ValueError('Incomplete quad-2x2 mortar patch coverage')
    if not np.isclose(
        sum(rmap.determinant for rmap in maps), 1.0,
        rtol=0, atol=40*tol
    ):
        raise ValueError('Invalid quad-2x2 mortar patch coverage')

    return tuple(maps)


def _quad_quad4_operator_key(group, order, index, maps, trace_sampling):
    left = MortarTraceKey(
        group.left.etype, 'quad', order, group.left.fidxs[index]
    )
    patches = tuple(
        MortarPatch(
            MortarTraceKey(side.etype, 'quad', order, side.fidxs[index]),
            rmap
        )
        for side, rmap in zip(group.right, maps)
    )
    quadrature = MortarQuadratureKey(
        'quad', rule='gauss-legendre', npts=(order + 2)**2
    )
    return MortarOperatorKey(
        left, patches, 'quad', order, quadrature, trace_sampling
    )


def _line_1x2_reference_maps(mesh, elemap, group, index, tol):
    if len(group.right) != 2:
        raise ValueError('line-1x2 mortars require two right participants')

    lc = LineShape.std_ele(1)
    lnodes = _face_corner_locs(
        mesh, elemap, group.left.etype, group.left.eidxs[index],
        group.left.fidxs[index]
    )
    lmap, _ = _fit_line_affine_map(lc, lnodes, tol)

    grid = np.array([(-1.0,), (0.0,), (1.0,)])
    grid_phys = lmap(grid)
    halves = (frozenset((-1.0, 0.0)), frozenset((0.0, 1.0)))

    scale = max(1.0, float(np.max(np.abs(lnodes), initial=0.0)))
    maps = [None, None]
    slots = [None, None]
    for slot, side in enumerate(group.right):
        rnodes = _face_corner_locs(
            mesh, elemap, side.etype, side.eidxs[index], side.fidxs[index]
        )
        target = []
        matched = set()
        for node in rnodes:
            errors = np.max(np.abs(grid_phys - node), axis=1)
            match = np.flatnonzero(errors <= tol*scale)
            if len(match) != 1:
                raise ValueError(
                    'Fine line mortar corner does not match coarse 1x2 grid'
                )
            gi = int(match[0])
            if gi in matched:
                raise ValueError('Duplicate fine line mortar corner')
            matched.add(gi)
            target.append(float(grid[gi, 0]))

        try:
            patch = halves.index(frozenset(target))
        except ValueError:
            raise ValueError(
                'Fine line mortar patch does not match the 1x2 template'
            ) from None
        if maps[patch] is not None:
            raise ValueError('Duplicate line-1x2 mortar patch')

        rmap = MortarReferenceMap(
            'line', tuple((value,) for value in target)
        )
        if not np.isclose(rmap.determinant, 0.5, rtol=0, atol=20*tol):
            raise ValueError('Invalid line-1x2 mortar patch determinant')
        maps[patch] = rmap
        slots[patch] = slot

    if any(rmap is None for rmap in maps):
        raise ValueError('Incomplete line-1x2 mortar patch coverage')
    if not np.isclose(
        sum(rmap.determinant for rmap in maps), 1.0,
        rtol=0, atol=40*tol
    ):
        raise ValueError('Invalid line-1x2 mortar patch coverage')

    return tuple(maps), tuple(slots)


def _canonical_line_1x2_group(group, slots_per_record):
    if len(slots_per_record) != len(group.left.eidxs):
        raise ValueError('Inconsistent line-1x2 mortar record count')
    if not slots_per_record:
        raise ValueError('Empty line-1x2 mortar set')

    right = []
    for patch in range(2):
        sides = tuple(
            group.right[slots[patch]] for slots in slots_per_record
        )
        etypes = {side.etype for side in sides}
        if len(etypes) != 1:
            raise ValueError('Inconsistent line-1x2 canonical side type')
        if any(side.face_topology != 'line' for side in sides):
            raise ValueError('Invalid line-1x2 canonical face topology')

        right.append(MortarSide(
            etypes.pop(), 'line',
            tuple(side.fidxs[index] for index, side in enumerate(sides)),
            tuple(side.eidxs[index] for index, side in enumerate(sides)),
        ))

    return MortarGroup(group.left, tuple(right))


def _line_line2_operator_key(group, order, index, maps, trace_sampling):
    left = MortarTraceKey(
        group.left.etype, 'line', order, group.left.fidxs[index]
    )
    patches = tuple(
        MortarPatch(
            MortarTraceKey(side.etype, 'line', order, side.fidxs[index]),
            rmap
        )
        for side, rmap in zip(group.right, maps)
    )
    quadrature = MortarQuadratureKey(
        'line', rule='gauss-legendre', npts=order + 2
    )
    return MortarOperatorKey(
        left, patches, 'line', order, quadrature, trace_sampling
    )


def _quad_tri_operator_key(
    group, order, index, qnodes, fnodes, trace_sampling=('nodal', 'nodal')
):
    left = MortarTraceKey(
        group.left.etype, group.left.face_topology, order,
        group.left.fidxs[index]
    )
    qcorners = QuadShape.std_ele(1)
    patches = tuple(
        MortarPatch(
            MortarTraceKey(
                side.etype, side.face_topology, order, side.fidxs[index]
            ),
            MortarReferenceMap(
                'tri', tuple(
                    map(tuple, qcorners[list(_mortar_target(qnodes, nodes))])
                )
            ),
        )
        for side, nodes in zip(group.right, fnodes)
    )

    sampling = (trace_sampling[0],) + (trace_sampling[1],)*len(patches)
    quadrature = MortarQuadratureKey('tri', qdeg=2*order + 2)
    return MortarOperatorKey(
        left, patches, 'tri', order, quadrature, sampling
    )


def _legacy_quad_tri_signature(key):
    qcorners = QuadShape.std_ele(1)
    signature = [key.left.etype, key.left.fidx]
    for patch in key.patches:
        target = []
        for point in patch.left_map.target:
            match = np.flatnonzero(np.all(qcorners == point, axis=1))
            if len(match) != 1:
                raise ValueError('Invalid legacy quad-triangle reference map')
            target.append(int(match[0]))
        signature.extend((
            patch.right.etype, patch.right.fidx, tuple(target)
        ))

    return tuple(signature)


def _mortar_target(qnodes, tnodes):
    target = []
    for node in tnodes:
        match = np.flatnonzero(qnodes == node)
        if len(match) != 1:
            raise ValueError('Fine mortar corners do not tile coarse face')
        target.append(int(match[0]))

    return tuple(target)


def _unique_signatures(signatures):
    unique = []
    sigmap = {}
    indices = np.empty(len(signatures), dtype=np.int32)

    for i, sig in enumerate(signatures):
        try:
            indices[i] = sigmap[sig]
        except KeyError:
            indices[i] = sigmap[sig] = len(unique)
            unique.append(sig)

    return unique, indices


def _stack_by_index(mats, indices):
    return np.stack([mats[i] for i in indices], axis=-1)


def build_quad_tri_operators(mesh, elemap, mcon, cfg):
    records = mcon.records
    order = cfg.getint('solver', 'order')
    geom_tol = cfg.getfloat('solver-interfaces', 'mortar-geom-tol', 1e-10)

    group = _decode_quad_tri_mortar(mesh, mcon)
    cetype, cfidxs, ceidxs = group.left.as_legacy()
    f0etype, f0fidxs, f0eidxs = group.right[0].as_legacy()
    f1etype, f1fidxs, f1eidxs = group.right[1].as_legacy()

    coarse = elemap[cetype]
    fine = elemap[f0etype]

    cbasis = coarse.basis.facebases['quad']
    fbasis = fine.basis.facebases['tri']
    ctonodal, cfromnodal = _face_sample_maps(
        coarse, 'quad', cbasis
    )
    ftonodal, ffromnodal = _face_sample_maps(
        fine, 'tri', fbasis
    )
    trace_sampling = (
        'surf-flux' if 'surf-flux' in coarse.antialias else 'nodal',
        'surf-flux' if 'surf-flux' in fine.antialias else 'nodal',
    )
    ncfpts = coarse.nfacefpts[cfidxs[0]]
    ntfpts = fine.nfacefpts[f0fidxs[0]]

    if any(coarse.nfacefpts[f] != ncfpts for f in cfidxs):
        raise ValueError('Inconsistent coarse mortar face point count')
    if any(fine.nfacefpts[f] != ntfpts for f in np.r_[f0fidxs, f1fidxs]):
        raise ValueError('Inconsistent fine mortar face point count')

    mqrule = get_quadrule('tri', qdeg=2*order + 2)
    cqrule = get_quadrule(
        'quad', rule='gauss-legendre', npts=(order + 2)**2
    )
    fqrule = get_quadrule('tri', qdeg=2*order + 2)

    cmass = _mass_matrix(cbasis, cqrule)
    fmass = _mass_matrix(fbasis, fqrule)
    finterp = fbasis.nodal_basis_at(mqrule.pts)


    signatures = []
    for mi in range(len(records)):
        qnodes = _face_corner_nodes(
            mesh, elemap, cetype, ceidxs[mi], cfidxs[mi]
        )
        f0nodes = _face_corner_nodes(
            mesh, elemap, f0etype, f0eidxs[mi], f0fidxs[mi]
        )
        f1nodes = _face_corner_nodes(
            mesh, elemap, f1etype, f1eidxs[mi], f1fidxs[mi]
        )

        signatures.append(_quad_tri_operator_key(
            group, order, mi, qnodes, (f0nodes, f1nodes), trace_sampling
        ))

    unique_keys, opidx = _unique_signatures(signatures)
    unique_sigs = [_legacy_quad_tri_signature(k) for k in unique_keys]
    nops = len(unique_keys)

    ic = [[], []]
    pc = [[], []]
    pf = [[], []]
    qpts = [[], []]
    dets = [[], []]
    tcorners = TriShape.std_ele(1)

    for key in unique_keys:
        for child, patch in enumerate(key.patches):
            target = np.asarray(patch.left_map.target)
            amap, det = _affine_map(tcorners, target)
            child_qpts = amap(mqrule.pts)
            cint = cbasis.nodal_basis_at(child_qpts)
            weights = mqrule.wts*det

            ic[child].append(cint @ ctonodal)
            pc[child].append(cfromnodal @ np.linalg.solve(
                cmass, cint.T*weights
            ))
            pf[child].append(ffromnodal @ np.linalg.solve(
                fmass, finterp.T*weights
            ))
            qpts[child].append(child_qpts)
            dets[child].append(det)

    physical_points = [[], []]
    normals = [[], []]
    geom_errors = [[], []]
    normal_errors = [[], []]
    determinants = [[], []]

    for mi, oi in enumerate(opidx):
        for child, (fidxs, eidxs) in enumerate(
            ((f0fidxs, f0eidxs), (f1fidxs, f1eidxs))
        ):
            child_qpts = qpts[child][oi]
            det = dets[child][oi]
            determinants[child].append(det)

            ckind, cproj, cnorm = coarse.basis.faces[cfidxs[mi]]
            fkind, fproj, fnorm = fine.basis.faces[fidxs[mi]]
            if ckind != 'quad' or fkind != 'tri':
                raise ValueError('Invalid mortar face types')

            cvpts = proj_pts(cproj, child_qpts)
            fvpts = proj_pts(fproj, mqrule.pts)
            cploc = coarse.ploc_at_np(cvpts)[:, :, ceidxs[mi]]
            fploc = fine.ploc_at_np(fvpts)[:, :, eidxs[mi]]
            physical_points[child].append(cploc)
            geom_errors[child].append(np.max(np.abs(cploc - fploc)))

            cn = np.broadcast_to(cnorm, (len(child_qpts), coarse.ndims))
            fn = np.broadcast_to(fnorm, (len(child_qpts), fine.ndims))
            cpnorm = coarse.pnorm_at(cvpts, cn)[:, ceidxs[mi]]
            fpnorm = fine.pnorm_at(fvpts, fn)[:, eidxs[mi]]
            normals[child].append(cpnorm)
            normal_errors[child].append(
                np.max(np.abs(fpnorm + det*cpnorm))
            )

    def stack(mats):
        return np.stack(mats, axis=-1)

    geometry = MortarGeometry(
        'left', 'tri', np.asarray(mqrule.pts), np.asarray(mqrule.wts),
        tuple(stack(points) for points in physical_points),
        tuple(stack(values) for values in normals),
        tuple(np.asarray(values) for values in determinants),
        tuple(np.asarray(values) for values in geom_errors),
        tuple(np.asarray(values) for values in normal_errors),
    )
    max_geom_error = geometry.max_coordinate_error
    max_normal_error = geometry.max_normal_error

    if max_geom_error > geom_tol:
        raise ValueError(
            f'Mortar geometry mismatch {max_geom_error:.3e} exceeds '
            f'tolerance {geom_tol:.3e}'
        )
    if max_normal_error > 10*geom_tol:
        raise ValueError(
            f'Mortar normal mismatch {max_normal_error:.3e} exceeds '
            f'tolerance {10*geom_tol:.3e}; the two face metric spaces '
            'are not compatible at this solver order. Curved '
            'quad-triangle mortars require solution order >= 2 with '
            'the current metric formulation'
        )

    ifmat = finterp @ ftonodal
    shared_opbytes = sum(
        arr.nbytes
        for arr in (
            *ic[0], *ic[1], *pc[0], *pc[1], *pf[0], *pf[1],
            ifmat,
        )
    )

    operator_sets = []
    operator_groups = []
    for oi in range(nops):
        operator_sets.append({
            'ic0': ic[0][oi], 'ic1': ic[1][oi],
            'if0': ifmat, 'if1': ifmat,
            'pc0': pc[0][oi], 'pc1': pc[1][oi],
            'pf0': pf[0][oi], 'pf1': pf[1][oi],
        })
        operator_groups.append(np.flatnonzero(opidx == oi))

    per_interface = {
        'ic0': _stack_by_index(ic[0], opidx),
        'ic1': _stack_by_index(ic[1], opidx),
        'if0': np.repeat(ifmat[..., None], len(records), axis=-1),
        'if1': np.repeat(ifmat[..., None], len(records), axis=-1),
        'pc0': _stack_by_index(pc[0], opidx),
        'pc1': _stack_by_index(pc[1], opidx),
        'pf0': _stack_by_index(pf[0], opidx),
        'pf1': _stack_by_index(pf[1], opidx),
    }
    fused_opbytes = sum(m.nbytes for m in per_interface.values())

    return {
        'coarse': (cetype, cfidxs, ceidxs),
        'fine0': (f0etype, f0fidxs, f0eidxs),
        'fine1': (f1etype, f1fidxs, f1eidxs),
        'ncfpts': ncfpts,
        'ntfpts': ntfpts,
        'nmpts': len(mqrule.pts),
        'nops': nops,
        'operator_signatures': unique_sigs,
        'operator_keys': unique_keys,
        'mortar_group': group,
        'geometry': geometry,
        'operator_sets': operator_sets,
        'operator_groups': operator_groups,
        'shared_operator_bytes': shared_opbytes,
        'fused_operator_bytes': fused_opbytes,
        **per_interface,
        'nl0': geometry.scaled_normals[0],
        'nl1': geometry.scaled_normals[1],
        'max_geom_error': max_geom_error,
        'max_normal_error': max_normal_error,
        'patch_determinants': determinants,
    }


def _build_mortar_operator_sets(
    group, bases, sample_maps, masses, right_interp, mqrule, unique_keys,
    state_projection
):
    operator_sets = []
    for key in unique_keys:
        left_interp = []
        left_proj = []
        right_ops = []
        for pi, patch in enumerate(key.patches):
            qpts = patch.left_map.apply(mqrule.pts)
            det = patch.left_map.determinant
            lint = bases[0].nodal_basis_at(qpts)
            weights = mqrule.wts*det

            lto, lfrom = sample_maps[0]
            rto, rfrom = sample_maps[pi + 1]
            rint = right_interp[pi]

            left_interp.append(lint @ lto)
            left_proj.append(lfrom @ np.linalg.solve(
                masses[0], lint.T*weights
            ))
            right_ops.append((
                rint @ rto,
                rfrom @ np.linalg.solve(masses[pi + 1], rint.T*weights),
            ))

        opset = {
            'left_interp': tuple(left_interp),
            'right_interp': tuple(op[0] for op in right_ops),
            'left_proj': tuple(left_proj),
            'right_proj': tuple(op[1] for op in right_ops),
        }
        if state_projection:
            opset['right_state_proj'] = tuple(
                sample_maps[pi + 1][1] @ np.linalg.solve(
                    masses[pi + 1], right_interp[pi].T*mqrule.wts
                )
                for pi in range(len(group.right))
            )
        operator_sets.append(opset)

    return operator_sets


def _build_mortar_geometry(
    group, eles, maps_per_record, mqrule, topology, geom_tol
):
    physical_points = [[] for _ in group.right]
    normals = [[] for _ in group.right]
    geom_errors = [[] for _ in group.right]
    normal_errors = [[] for _ in group.right]
    determinants = [[] for _ in group.right]

    left_ele = eles[0]
    for mi, maps in enumerate(maps_per_record):
        lfidx = group.left.fidxs[mi]
        leidx = group.left.eidxs[mi]
        lkind, lproj, lnorm = left_ele.basis.faces[lfidx]
        if lkind != topology:
            raise ValueError('Invalid left mortar face type')

        for pi, (side, rmap) in enumerate(zip(group.right, maps)):
            right_ele = eles[pi + 1]
            rfidx = side.fidxs[mi]
            reidx = side.eidxs[mi]
            rkind, rproj, rnorm = right_ele.basis.faces[rfidx]
            if rkind != topology:
                raise ValueError('Invalid right mortar face type')

            lqpts = rmap.apply(mqrule.pts)
            det = rmap.determinant
            determinants[pi].append(det)

            lvpts = proj_pts(lproj, lqpts)
            rvpts = proj_pts(rproj, mqrule.pts)
            lploc = left_ele.ploc_at_np(lvpts)[:, :, leidx]
            rploc = right_ele.ploc_at_np(rvpts)[:, :, reidx]
            physical_points[pi].append(lploc)
            geom_errors[pi].append(np.max(np.abs(lploc - rploc)))

            ln = np.broadcast_to(lnorm, (len(lqpts), left_ele.ndims))
            rn = np.broadcast_to(rnorm, (len(lqpts), right_ele.ndims))
            lpnorm = left_ele.pnorm_at(lvpts, ln)[:, leidx]
            rpnorm = right_ele.pnorm_at(rvpts, rn)[:, reidx]
            normals[pi].append(lpnorm)
            normal_errors[pi].append(
                np.max(np.abs(rpnorm + det*lpnorm))
            )

    def stack(values):
        return np.stack(values, axis=-1)

    geometry = MortarGeometry(
        'left', topology, np.asarray(mqrule.pts), np.asarray(mqrule.wts),
        tuple(stack(values) for values in physical_points),
        tuple(stack(values) for values in normals),
        tuple(np.asarray(values) for values in determinants),
        tuple(np.asarray(values) for values in geom_errors),
        tuple(np.asarray(values) for values in normal_errors),
    )
    if geometry.max_coordinate_error > geom_tol:
        raise ValueError(
            f'Mortar geometry mismatch {geometry.max_coordinate_error:.3e} '
            f'exceeds tolerance {geom_tol:.3e}'
        )
    if geometry.max_normal_error > 10*geom_tol:
        raise ValueError(
            f'Mortar normal mismatch {geometry.max_normal_error:.3e} '
            f'exceeds tolerance {10*geom_tol:.3e}'
        )

    return geometry, determinants


def build_quad_quad4_operators(
    mesh, elemap, mcon, cfg, *, state_projection=False
):
    order = cfg.getint('solver', 'order')
    geom_tol = cfg.getfloat('solver-interfaces', 'mortar-geom-tol', 1e-10)

    group = _decode_general_mortar(mesh, mcon)
    sides = (group.left, *group.right)
    if any(side.face_topology != 'quad' for side in sides):
        raise ValueError('quad-2x2 mortars require quadrilateral faces')

    eles = tuple(elemap[side.etype] for side in sides)
    bases = tuple(ele.basis.facebases['quad'] for ele in eles)
    sample_maps = tuple(
        _face_sample_maps(ele, 'quad', basis)
        for ele, basis in zip(eles, bases)
    )
    trace_sampling = tuple(
        'surf-flux' if 'surf-flux' in ele.antialias else 'nodal'
        for ele in eles
    )

    nfacefpts = []
    for side, ele in zip(sides, eles):
        npts = ele.nfacefpts[side.fidxs[0]]
        if any(ele.nfacefpts[fidx] != npts for fidx in side.fidxs):
            raise ValueError('Inconsistent mortar face point count')
        nfacefpts.append(npts)

    mqrule = get_quadrule(
        'quad', rule='gauss-legendre', npts=(order + 2)**2
    )
    mass_qrule = get_quadrule(
        'quad', rule='gauss-legendre', npts=(order + 2)**2
    )
    masses = tuple(_mass_matrix(basis, mass_qrule) for basis in bases)
    right_interp = tuple(
        basis.nodal_basis_at(mqrule.pts) for basis in bases[1:]
    )

    maps_per_record = []
    keys = []
    for mi in range(len(mcon.records)):
        maps = _quad_2x2_reference_maps(
            mesh, elemap, group, mi, geom_tol
        )
        maps_per_record.append(maps)
        keys.append(_quad_quad4_operator_key(
            group, order, mi, maps, trace_sampling
        ))

    unique_keys, opidx = _unique_signatures(keys)
    nops = len(unique_keys)

    operator_sets = _build_mortar_operator_sets(
        group, bases, sample_maps, masses, right_interp, mqrule,
        unique_keys, state_projection
    )

    geometry, determinants = _build_mortar_geometry(
        group, eles, maps_per_record, mqrule, 'quad', geom_tol
    )

    operator_groups = [
        np.flatnonzero(opidx == oi) for oi in range(nops)
    ]
    shared_opbytes = sum(
        matrix.nbytes
        for opset in operator_sets
        for values in opset.values()
        for matrix in values
    )

    return {
        'mortar_group': group,
        'operator_keys': unique_keys,
        'operator_sets': operator_sets,
        'operator_groups': operator_groups,
        'geometry': geometry,
        'nleftfpts': nfacefpts[0],
        'nrightfpts': tuple(nfacefpts[1:]),
        'nmpts': len(mqrule.pts),
        'nops': nops,
        'shared_operator_bytes': shared_opbytes,
        'max_geom_error': geometry.max_coordinate_error,
        'max_normal_error': geometry.max_normal_error,
        'patch_determinants': tuple(
            np.asarray(values) for values in determinants
        ),
    }


def build_line_line2_operators(
    mesh, elemap, mcon, cfg, *, state_projection=False
):
    order = cfg.getint('solver', 'order')
    geom_tol = cfg.getfloat('solver-interfaces', 'mortar-geom-tol', 1e-10)

    if mcon.template != 'line-1x2':
        raise ValueError('build_line_line2_operators requires line-1x2')
    raw_group = _decode_general_mortar(mesh, mcon)
    maps_per_record = []
    slots_per_record = []
    for mi in range(len(mcon.records)):
        maps, slots = _line_1x2_reference_maps(
            mesh, elemap, raw_group, mi, geom_tol
        )
        maps_per_record.append(maps)
        slots_per_record.append(slots)

    group = _canonical_line_1x2_group(raw_group, slots_per_record)
    sides = (group.left, *group.right)
    if any(side.face_topology != 'line' for side in sides):
        raise ValueError('line-1x2 mortars require Line faces')
    if group.left.etype not in {'tri', 'quad'} or any(
        side.etype != 'quad' for side in group.right
    ):
        raise ValueError(
            'line-1x2 mortars require a Tri/Quad coarse Line and two Quad '
            'fine Lines'
        )

    eles = tuple(elemap[side.etype] for side in sides)
    bases = tuple(ele.basis.facebases['line'] for ele in eles)
    sample_maps = tuple(
        _face_sample_maps(ele, 'line', basis)
        for ele, basis in zip(eles, bases)
    )
    trace_sampling = tuple(
        'surf-flux' if 'surf-flux' in ele.antialias else 'nodal'
        for ele in eles
    )

    nfacefpts = []
    for side, ele in zip(sides, eles):
        npts = ele.nfacefpts[side.fidxs[0]]
        if any(ele.nfacefpts[fidx] != npts for fidx in side.fidxs):
            raise ValueError('Inconsistent mortar face point count')
        nfacefpts.append(npts)

    mqrule = get_quadrule(
        'line', rule='gauss-legendre', npts=order + 2
    )
    mass_qrule = get_quadrule(
        'line', rule='gauss-legendre', npts=order + 2
    )
    masses = tuple(_mass_matrix(basis, mass_qrule) for basis in bases)
    right_interp = tuple(
        basis.nodal_basis_at(mqrule.pts) for basis in bases[1:]
    )
    keys = [
        _line_line2_operator_key(
            group, order, mi, maps, trace_sampling
        )
        for mi, maps in enumerate(maps_per_record)
    ]

    unique_keys, opidx = _unique_signatures(keys)
    operator_sets = _build_mortar_operator_sets(
        group, bases, sample_maps, masses, right_interp, mqrule,
        unique_keys, state_projection
    )

    geometry, determinants = _build_mortar_geometry(
        group, eles, maps_per_record, mqrule, 'line', geom_tol
    )

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
        'mortar_group': group,
        'operator_keys': unique_keys,
        'operator_sets': operator_sets,
        'operator_groups': operator_groups,
        'geometry': geometry,
        'nleftfpts': nfacefpts[0],
        'nrightfpts': tuple(nfacefpts[1:]),
        'nmpts': len(mqrule.pts),
        'nops': len(unique_keys),
        'shared_operator_bytes': shared_opbytes,
        'max_geom_error': geometry.max_coordinate_error,
        'max_normal_error': geometry.max_normal_error,
        'patch_determinants': tuple(
            np.asarray(values) for values in determinants
        ),
    }
