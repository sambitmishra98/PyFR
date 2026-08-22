import numpy as np
import pytest

from pyfr.mortars import (
    MortarGeometry, MortarGroup, MortarOperatorKey, MortarPatch,
    MortarQuadratureKey, MortarReferenceMap, MortarSide, MortarTraceKey,
    _affine_map, _decode_general_mortar, _decode_quad_tri_mortar,
    _legacy_quad_tri_signature, _line_1x2_reference_maps, _mass_matrix,
    _quad_2x2_reference_maps, _quad_tri_operator_key, _stack_by_index,
    _unique_signatures,
)
from pyfr.polys import get_polybasis
from pyfr.quadrules import get_quadrule
from pyfr.shapes import LineShape, QuadShape, TriShape


@pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
def test_quad_to_two_triangle_projection_conservation(order):
    cq = get_quadrule(
        'quad', rule='gauss-legendre', npts=(order + 1)**2
    )
    fq = get_quadrule(
        'tri', rule='williams-shunn',
        npts=TriShape.npts_from_order(order)
    )
    mq = get_quadrule('tri', qdeg=2*order + 2)

    cbasis = get_polybasis('quad', order, cq.pts)
    fbasis = get_polybasis('tri', order, fq.pts)
    cmass = _mass_matrix(cbasis, get_quadrule(
        'quad', rule='gauss-legendre', npts=(order + 2)**2
    ))
    fmass = _mass_matrix(fbasis, mq)
    finterp = fbasis.nodal_basis_at(mq.pts)

    qcorners = QuadShape.std_ele(1)
    tcorners = TriShape.std_ele(1)
    targets = [
        qcorners[[0, 2, 1]],
        qcorners[[2, 3, 1]],
    ]

    coarse = np.zeros(len(cq.pts))
    fine = []
    fluxes = []
    for child, target in enumerate(targets):
        amap, det = _affine_map(tcorners, target)
        assert np.allclose(amap(tcorners), target)
        assert np.isclose(det, 1.0)

        qpts = amap(mq.pts)
        cint = cbasis.nodal_basis_at(qpts)
        weights = mq.wts*det
        pc = np.linalg.solve(cmass, cint.T*weights)
        pf = np.linalg.solve(fmass, finterp.T*weights)

        x, y = qpts.T
        flux = 1.0 + 0.2*x - 0.3*y + 0.1*x*y
        fluxes.append(flux)
        coarse += pc @ flux
        fine.append(-(pf @ flux))

    onec = np.ones(len(cq.pts))
    onef = np.ones(len(fq.pts))
    coarse_integral = onec @ cmass @ coarse
    fine_integral = sum(onef @ fmass @ values for values in fine)

    assert abs(coarse_integral + fine_integral) < 5e-14

    # A constant common flux must project to +1 on the coarse side and
    # -1 on each child side.
    coarse = np.zeros(len(cq.pts))
    fine = []
    for target in targets:
        amap, det = _affine_map(tcorners, target)
        cint = cbasis.nodal_basis_at(amap(mq.pts))
        weights = mq.wts*det
        coarse += np.linalg.solve(cmass, cint.T*weights) @ np.ones(len(mq.pts))
        fine.append(-np.linalg.solve(
            fmass, finterp.T*weights
        ) @ np.ones(len(mq.pts)))

    assert np.allclose(coarse, 1.0, atol=2e-14)
    assert all(np.allclose(values, -1.0, atol=2e-14) for values in fine)


@pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
def test_quad_to_two_triangle_common_trace_reproduction(order):
    cq = get_quadrule(
        'quad', rule='gauss-legendre', npts=(order + 1)**2
    )
    fq = get_quadrule(
        'tri', rule='williams-shunn',
        npts=TriShape.npts_from_order(order)
    )
    mq = get_quadrule('tri', qdeg=2*order + 2)

    cbasis = get_polybasis('quad', order, cq.pts)
    fbasis = get_polybasis('tri', order, fq.pts)
    cmass = _mass_matrix(cbasis, get_quadrule(
        'quad', rule='gauss-legendre', npts=(order + 2)**2
    ))
    fmass = _mass_matrix(fbasis, mq)
    finterp = fbasis.nodal_basis_at(mq.pts)

    qcorners = QuadShape.std_ele(1)
    tcorners = TriShape.std_ele(1)
    targets = [qcorners[[0, 2, 1]], qcorners[[2, 3, 1]]]

    degree = min(order, 2)

    def trace(points):
        x, y = points.T
        value = 1.0 + 0.2*x - 0.3*y
        if degree == 2:
            value += 0.1*x*y + 0.05*x*x
        return value

    coarse = np.zeros(len(cq.pts))
    fine = []
    for target in targets:
        amap, det = _affine_map(tcorners, target)
        qpts = amap(mq.pts)
        cint = cbasis.nodal_basis_at(qpts)
        weights = mq.wts*det
        pc = np.linalg.solve(cmass, cint.T*weights)
        pf = np.linalg.solve(fmass, finterp.T*weights)

        common = trace(qpts)
        coarse += pc @ common
        fine.append(pf @ common)

    assert np.allclose(coarse, trace(cq.pts), atol=5e-13)
    for target, values in zip(targets, fine):
        amap, _ = _affine_map(tcorners, target)
        assert np.allclose(values, trace(amap(fq.pts)), atol=5e-13)


def test_mortar_surface_flux_sample_maps():
    from types import SimpleNamespace

    from pyfr.mortars import _face_sample_maps
    from pyfr.polys import get_polybasis

    for kind, npts in [('quad', 16), ('tri', 12)]:
        order = 3
        qrule = get_quadrule(
            kind,
            rule='gauss-legendre' if kind == 'quad' else None,
            npts=npts if kind == 'quad' else None,
            qdeg=None if kind == 'quad' else 2*(order + 2),
        )
        std = get_quadrule(
            kind,
            rule='gauss-legendre' if kind == 'quad' else 'williams-shunn',
            npts=(order + 1)**2 if kind == 'quad' else
                  (order + 1)*(order + 2) // 2,
        )
        basis = get_polybasis(kind, order, std.pts)
        eles = SimpleNamespace(
            antialias={'surf-flux'},
            basis=SimpleNamespace(_iqrules={kind: qrule}),
        )
        to_nodal, from_nodal = _face_sample_maps(eles, kind, basis)

        assert np.allclose(to_nodal @ from_nodal, np.eye(len(std.pts)))

        coeff = np.arange(1, len(std.pts) + 1, dtype=float)
        sampled = from_nodal @ coeff
        assert np.allclose(to_nodal @ sampled, coeff)


def test_mortar_nodal_sample_maps_are_identity():
    from types import SimpleNamespace

    from pyfr.mortars import _face_sample_maps
    from pyfr.polys import get_polybasis

    order = 3
    rule = get_quadrule('quad', 'gauss-legendre', (order + 1)**2)
    basis = get_polybasis('quad', order, rule.pts)
    eles = SimpleNamespace(antialias=set())
    to_nodal, from_nodal = _face_sample_maps(eles, 'quad', basis)

    assert np.array_equal(to_nodal, np.eye(len(rule.pts)))
    assert np.array_equal(from_nodal, np.eye(len(rule.pts)))


def test_legacy_quad_tri_mortar_adapter():
    from types import SimpleNamespace

    dtype = [
        ('coarse_cidx', 'i4'), ('coarse_eidx', 'i8'),
        ('fine_cidx', 'i4', (2,)), ('fine_eidx', 'i8', (2,)),
    ]
    records = np.zeros(2, dtype=dtype)
    records['coarse_cidx'] = [0, 0]
    records['coarse_eidx'] = [4, 5]
    records['fine_cidx'] = [[1, 2], [1, 2]]
    records['fine_eidx'] = [[7, 9], [8, 10]]

    mesh = SimpleNamespace(cidxmap={
        0: ('hex', 1), 1: ('tet', 2), 2: ('tet', 3),
    })
    mcon = SimpleNamespace(records=records)
    group = _decode_quad_tri_mortar(mesh, mcon)

    assert group.left == MortarSide('hex', 'quad', (1, 1), (4, 5))
    assert group.right == (
        MortarSide('tet', 'tri', (2, 2), (7, 8)),
        MortarSide('tet', 'tri', (3, 3), (9, 10)),
    )

    etype, fidxs, eidxs = group.left.as_legacy()
    assert etype == 'hex'
    assert fidxs.dtype == np.int32
    assert eidxs.dtype == np.int64
    assert np.array_equal(fidxs, [1, 1])
    assert np.array_equal(eidxs, [4, 5])


def test_quad_tri_operator_key_preserves_legacy_signature():
    group = MortarGroup(
        MortarSide('hex', 'quad', (1, 1), (4, 5)),
        (
            MortarSide('tet', 'tri', (2, 2), (7, 8)),
            MortarSide('tet', 'tri', (3, 3), (9, 10)),
        ),
    )
    qnodes = np.array([10, 20, 30, 40])
    fnodes = (
        np.array([10, 30, 20]),
        np.array([30, 40, 20]),
    )

    key0 = _quad_tri_operator_key(group, 3, 0, qnodes, fnodes)
    key1 = _quad_tri_operator_key(group, 3, 1, qnodes, fnodes)

    assert key0 == key1
    assert key0 != _quad_tri_operator_key(group, 4, 0, qnodes, fnodes)
    assert key0 != _quad_tri_operator_key(
        group, 3, 0, qnodes, fnodes, ('surf-flux', 'surf-flux')
    )
    qcorners = QuadShape.std_ele(1)
    assert key0 == MortarOperatorKey(
        MortarTraceKey('hex', 'quad', 3, 1),
        (
            MortarPatch(
                MortarTraceKey('tet', 'tri', 3, 2),
                MortarReferenceMap(
                    'tri', tuple(map(tuple, qcorners[[0, 2, 1]]))
                ),
            ),
            MortarPatch(
                MortarTraceKey('tet', 'tri', 3, 3),
                MortarReferenceMap(
                    'tri', tuple(map(tuple, qcorners[[2, 3, 1]]))
                ),
            ),
        ),
        'tri', 3, MortarQuadratureKey('tri', qdeg=8),
        ('nodal', 'nodal', 'nodal'),
    )
    assert _legacy_quad_tri_signature(key0) == (
        'hex', 1,
        'tet', 2, (0, 2, 1),
        'tet', 3, (2, 3, 1),
    )


def test_mortar_operator_signature_metadata():
    signatures = [('hex', 0), ('hex', 1), ('hex', 0), ('hex', 1)]
    unique, indices = _unique_signatures(signatures)

    assert unique == [('hex', 0), ('hex', 1)]
    assert np.array_equal(indices, [0, 1, 0, 1])

    mats = [np.full((2, 3), 10.0), np.full((2, 3), 20.0)]
    expanded = _stack_by_index(mats, indices)

    assert expanded.shape == (2, 3, 4)
    assert np.array_equal(expanded[..., 0], mats[0])
    assert np.array_equal(expanded[..., 1], mats[1])
    assert np.array_equal(expanded[..., 2], mats[0])
    assert np.array_equal(expanded[..., 3], mats[1])


def test_signature_batched_mortar_transforms_match_expanded():
    rng = np.random.default_rng(271828)
    ncfpts, ntfpts, nmpts, nvars, ninters = 4, 3, 5, 2, 6
    opidx = np.array([0, 1, 0, 1, 1, 0])

    ops = []
    for _ in range(2):
        ops.append({
            'ic0': rng.normal(size=(nmpts, ncfpts)),
            'ic1': rng.normal(size=(nmpts, ncfpts)),
            'if0': rng.normal(size=(nmpts, ntfpts)),
            'if1': rng.normal(size=(nmpts, ntfpts)),
            'pc0': rng.normal(size=(ncfpts, nmpts)),
            'pc1': rng.normal(size=(ncfpts, nmpts)),
            'pf0': rng.normal(size=(ntfpts, nmpts)),
            'pf1': rng.normal(size=(ntfpts, nmpts)),
        })

    uc = rng.normal(size=(ncfpts, nvars, ninters))
    uf0 = rng.normal(size=(ntfpts, nvars, ninters))
    uf1 = rng.normal(size=(ntfpts, nvars, ninters))

    expanded = {
        name: _stack_by_index([op[name] for op in ops], opidx)
        for name in ops[0]
    }
    fused = [np.zeros_like(uc), np.zeros_like(uf0), np.zeros_like(uf1)]
    for i in range(ninters):
        ul0 = expanded['ic0'][..., i] @ uc[..., i]
        ul1 = expanded['ic1'][..., i] @ uc[..., i]
        ur0 = expanded['if0'][..., i] @ uf0[..., i]
        ur1 = expanded['if1'][..., i] @ uf1[..., i]
        flux0, flux1 = ul0 - ur0, ul1 - ur1

        fused[0][..., i] = (
            expanded['pc0'][..., i] @ flux0
            + expanded['pc1'][..., i] @ flux1
        )
        fused[1][..., i] = -expanded['pf0'][..., i] @ flux0
        fused[2][..., i] = -expanded['pf1'][..., i] @ flux1

    staged = [np.zeros_like(uc), np.zeros_like(uf0), np.zeros_like(uf1)]
    for oi, op in enumerate(ops):
        group = np.flatnonzero(opidx == oi)
        ul0 = np.einsum('qj,jvg->qvg', op['ic0'], uc[..., group])
        ul1 = np.einsum('qj,jvg->qvg', op['ic1'], uc[..., group])
        ur0 = np.einsum('qj,jvg->qvg', op['if0'], uf0[..., group])
        ur1 = np.einsum('qj,jvg->qvg', op['if1'], uf1[..., group])
        flux0, flux1 = ul0 - ur0, ul1 - ur1

        staged[0][..., group] = (
            np.einsum('jq,qvg->jvg', op['pc0'], flux0)
            + np.einsum('jq,qvg->jvg', op['pc1'], flux1)
        )
        staged[1][..., group] = -np.einsum(
            'jq,qvg->jvg', op['pf0'], flux0
        )
        staged[2][..., group] = -np.einsum(
            'jq,qvg->jvg', op['pf1'], flux1
        )

    for actual, expected in zip(staged, fused):
        assert np.allclose(actual, expected, rtol=0, atol=2e-14)




def _quad_d4_transforms(points):
    x, y = points.T
    return [
        np.column_stack((x, y)),
        np.column_stack((-y, x)),
        np.column_stack((-x, -y)),
        np.column_stack((y, -x)),
        np.column_stack((-x, y)),
        np.column_stack((x, -y)),
        np.column_stack((y, x)),
        np.column_stack((-y, -x)),
    ]


def test_quad_mortar_reference_map_all_d4_orientations():
    corners = QuadShape.std_ele(1)
    for oriented in _quad_d4_transforms(corners):
        target = 0.5*oriented - 0.5
        rmap = MortarReferenceMap('quad', tuple(map(tuple, target)))
        assert np.isclose(rmap.determinant, 0.25, rtol=0, atol=1e-14)
        assert np.allclose(rmap.apply(corners), target, atol=2e-15)

    bad = ((-1, -1), (0, -1), (0, 0), (-1, 0))
    with pytest.raises(ValueError, match='not affine'):
        MortarReferenceMap('quad', (bad[0], bad[2], bad[1], bad[3]))


def test_general_quad_2x2_mortar_adapter():
    from types import SimpleNamespace

    dtype = [
        ('left_cidx', 'i4'), ('left_eidx', 'i8'),
        ('right_cidx', 'i4', (4,)), ('right_eidx', 'i8', (4,)),
    ]
    records = np.zeros(2, dtype=dtype)
    records['left_cidx'] = 0
    records['left_eidx'] = [4, 5]
    records['right_cidx'] = 1
    records['right_eidx'] = [[7, 8, 9, 10], [11, 12, 13, 14]]

    mesh = SimpleNamespace(cidxmap={0: ('hex', 1), 1: ('hex', 3)})
    mcon = SimpleNamespace(
        records=records, format='one-to-many-v1', template='quad-2x2',
        nright=4,
    )
    group = _decode_general_mortar(mesh, mcon)

    assert group.left == MortarSide('hex', 'quad', (1, 1), (4, 5))
    assert len(group.right) == 4
    for child, side in enumerate(group.right):
        assert side == MortarSide(
            'hex', 'quad', (3, 3), (7 + child, 11 + child)
        )


def test_quad_2x2_reference_map_inference():
    from types import SimpleNamespace

    class Basis:
        @staticmethod
        def face_corner_pts_idxs(fidx, nspts):
            return np.arange(4)

    def phys(points):
        u, v = points.T
        return np.column_stack((np.zeros(len(points)), u, v))

    corners = QuadShape.std_ele(1)
    targets = [
        ((-1, -1), (0, -1), (-1, 0), (0, 0)),
        ((0, -1), (1, -1), (0, 0), (1, 0)),
        ((-1, 0), (0, 0), (-1, 1), (0, 1)),
        ((0, 0), (1, 0), (0, 1), (1, 1)),
    ]
    locs = [phys(corners)]
    locs.extend(phys(np.asarray(target)) for target in targets)
    spts = np.swapaxes(np.asarray(locs), 0, 1)
    node_ids = np.arange(100, 120, dtype=np.int64).reshape(5, 4)
    mesh = SimpleNamespace(
        spts={'hex': spts}, spts_nodes={'hex': node_ids.copy()}
    )
    elemap = {'hex': SimpleNamespace(basis=Basis(), nspts=4)}
    group = MortarGroup(
        MortarSide('hex', 'quad', (0,), (0,)),
        tuple(MortarSide('hex', 'quad', (0,), (i,)) for i in range(1, 5)),
    )

    maps = _quad_2x2_reference_maps(mesh, elemap, group, 0, 1e-12)
    assert len(maps) == 4
    assert np.allclose([rmap.determinant for rmap in maps], 0.25)
    for rmap, target in zip(maps, targets):
        assert set(rmap.target) == set(target)

    mesh.spts_nodes['hex'] = node_ids[:, ::-1] + 1000
    remapped = _quad_2x2_reference_maps(mesh, elemap, group, 0, 1e-12)
    assert remapped == maps


def test_line_mortar_reference_map_both_orientations():
    corners = LineShape.std_ele(1)
    targets = (
        ((-1.0,), (0.0,)), ((0.0,), (-1.0,)),
        ((0.0,), (1.0,)), ((1.0,), (0.0,)),
    )

    for target in targets:
        rmap = MortarReferenceMap('line', target)
        assert np.isclose(rmap.determinant, 0.5, rtol=0, atol=1e-14)
        assert np.allclose(rmap.apply(corners), target, atol=2e-15)


def test_general_line_1x2_mortar_adapter():
    from types import SimpleNamespace

    dtype = [
        ('left_cidx', 'i4'), ('left_eidx', 'i8'),
        ('right_cidx', 'i4', (2,)), ('right_eidx', 'i8', (2,)),
    ]
    records = np.zeros(2, dtype=dtype)
    records['left_cidx'] = 0
    records['left_eidx'] = [4, 5]
    records['right_cidx'] = 1
    records['right_eidx'] = [[7, 8], [9, 10]]

    mesh = SimpleNamespace(cidxmap={0: ('quad', 1), 1: ('quad', 3)})
    mcon = SimpleNamespace(
        records=records, format='one-to-many-v1', template='line-1x2',
        nright=2,
    )
    group = _decode_general_mortar(mesh, mcon)

    assert group.left == MortarSide('quad', 'line', (1, 1), (4, 5))
    assert group.right == (
        MortarSide('quad', 'line', (3, 3), (7, 9)),
        MortarSide('quad', 'line', (3, 3), (8, 10)),
    )


@pytest.mark.parametrize('reverse_edge', [False, True])
def test_line_1x2_reference_map_inference_is_physical_and_canonical(
    reverse_edge
):
    from types import SimpleNamespace

    class Basis:
        @staticmethod
        def face_corner_pts_idxs(fidx, nspts):
            return np.arange(2)

    def phys(points):
        s = points[:, 0]
        return np.column_stack((np.zeros(len(points)),
                                -s if reverse_edge else s))

    corners = LineShape.std_ele(1)
    low = ((-1.0,), (0.0,))
    high_reversed = ((1.0,), (0.0,))
    locs = [phys(corners), phys(np.asarray(high_reversed)),
            phys(np.asarray(low))]
    spts = np.swapaxes(np.asarray(locs), 0, 1)
    node_ids = np.arange(100, 106, dtype=np.int64).reshape(3, 2)
    mesh = SimpleNamespace(
        spts={'quad': spts}, spts_nodes={'quad': node_ids.copy()}
    )
    elemap = {'quad': SimpleNamespace(basis=Basis(), nspts=2)}
    group = MortarGroup(
        MortarSide('quad', 'line', (0,), (0,)),
        (
            MortarSide('quad', 'line', (0,), (1,)),
            MortarSide('quad', 'line', (0,), (2,)),
        ),
    )

    maps, slots = _line_1x2_reference_maps(mesh, elemap, group, 0, 1e-12)
    assert slots == (1, 0)
    assert maps[0].target == low
    assert maps[1].target == high_reversed
    assert np.allclose([rmap.determinant for rmap in maps], 0.5)

    mesh.spts_nodes['quad'] = node_ids[:, ::-1] + 1000
    remapped, reslots = _line_1x2_reference_maps(
        mesh, elemap, group, 0, 1e-12
    )
    assert remapped == maps
    assert reslots == slots


@pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
def test_line_to_two_line_projection_conservation(order):
    q = get_quadrule('line', rule='gauss-legendre', npts=order + 1)
    mq = get_quadrule('line', rule='gauss-legendre', npts=order + 2)
    basis = get_polybasis('line', order, q.pts)
    mass = _mass_matrix(basis, mq)
    interp = basis.nodal_basis_at(mq.pts)
    maps = (
        MortarReferenceMap('line', ((-1.0,), (0.0,))),
        MortarReferenceMap('line', ((0.0,), (1.0,))),
    )

    left = np.zeros(len(q.pts))
    right = []
    for rmap in maps:
        lpts = rmap.apply(mq.pts)
        lint = basis.nodal_basis_at(lpts)
        weights = mq.wts*rmap.determinant
        lp = np.linalg.solve(mass, lint.T*weights)
        rp = np.linalg.solve(mass, interp.T*weights)

        s = lpts[:, 0]
        flux = 1.0 + 0.2*s - 0.1*s*s
        left += lp @ flux
        right.append(-(rp @ flux))

    one = np.ones(len(q.pts))
    balance = one @ mass @ left
    balance += sum(one @ mass @ values for values in right)
    assert abs(balance) < 5e-14

    left = np.zeros(len(q.pts))
    right = []
    for rmap in maps:
        lint = basis.nodal_basis_at(rmap.apply(mq.pts))
        weights = mq.wts*rmap.determinant
        left += np.linalg.solve(mass, lint.T*weights) @ np.ones(len(mq.pts))
        right.append(-np.linalg.solve(mass, interp.T*weights)
                     @ np.ones(len(mq.pts)))

    assert np.allclose(left, 1.0, atol=3e-14)
    assert all(np.allclose(values, -0.5, atol=3e-14) for values in right)


@pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
def test_line_to_two_line_common_trace_reproduction(order):
    q = get_quadrule('line', rule='gauss-legendre', npts=order + 1)
    mq = get_quadrule('line', rule='gauss-legendre', npts=order + 2)
    basis = get_polybasis('line', order, q.pts)
    mass = _mass_matrix(basis, mq)
    interp = basis.nodal_basis_at(mq.pts)
    maps = (
        MortarReferenceMap('line', ((-1.0,), (0.0,))),
        MortarReferenceMap('line', ((0.0,), (1.0,))),
    )

    def trace(points):
        s = np.asarray(points).reshape(-1)
        values = 1.0 + 0.2*s
        if order > 1:
            values -= 0.1*s*s
        return values

    left = np.zeros(len(q.pts))
    for rmap in maps:
        lpts = rmap.apply(mq.pts)
        lint = basis.nodal_basis_at(lpts)
        weights = mq.wts*rmap.determinant
        lp = np.linalg.solve(mass, lint.T*weights)
        rp = np.linalg.solve(mass, interp.T*weights)
        common = trace(lpts)

        left += lp @ common
        assert np.allclose(
            rp @ common, 0.5*trace(rmap.apply(q.pts)), atol=5e-13
        )

    assert np.allclose(left, trace(q.pts), atol=5e-13)


@pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
def test_quad_to_four_quad_projection_conservation(order):
    q = get_quadrule(
        'quad', rule='gauss-legendre', npts=(order + 1)**2
    )
    mq = get_quadrule(
        'quad', rule='gauss-legendre', npts=(order + 2)**2
    )
    basis = get_polybasis('quad', order, q.pts)
    mass = _mass_matrix(basis, mq)
    rinterp = basis.nodal_basis_at(mq.pts)

    targets = [
        ((-1, -1), (0, -1), (-1, 0), (0, 0)),
        ((0, -1), (1, -1), (0, 0), (1, 0)),
        ((-1, 0), (0, 0), (-1, 1), (0, 1)),
        ((0, 0), (1, 0), (0, 1), (1, 1)),
    ]
    maps = [MortarReferenceMap('quad', target) for target in targets]

    left = np.zeros(len(q.pts))
    right = []
    for rmap in maps:
        lpts = rmap.apply(mq.pts)
        lint = basis.nodal_basis_at(lpts)
        weights = mq.wts*rmap.determinant
        pl = np.linalg.solve(mass, lint.T*weights)
        pr = np.linalg.solve(mass, rinterp.T*weights)

        x, y = lpts.T
        flux = 1 + 0.2*x - 0.3*y + 0.1*x*y
        left += pl @ flux
        right.append(-(pr @ flux))

    one = np.ones(len(q.pts))
    balance = one @ mass @ left
    balance += sum(one @ mass @ values for values in right)
    assert abs(balance) < 8e-14

    left = np.zeros(len(q.pts))
    right = []
    for rmap in maps:
        lint = basis.nodal_basis_at(rmap.apply(mq.pts))
        weights = mq.wts*rmap.determinant
        left += np.linalg.solve(mass, lint.T*weights) @ np.ones(len(mq.pts))
        right.append(-np.linalg.solve(
            mass, rinterp.T*weights
        ) @ np.ones(len(mq.pts)))

    assert np.allclose(left, 1.0, atol=4e-14)
    assert all(np.allclose(values, -0.25, atol=4e-14) for values in right)


@pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
def test_quad_to_four_quad_state_projection(order):
    q = get_quadrule(
        'quad', rule='gauss-legendre', npts=(order + 1)**2
    )
    mq = get_quadrule(
        'quad', rule='gauss-legendre', npts=(order + 2)**2
    )
    basis = get_polybasis('quad', order, q.pts)
    mass = _mass_matrix(basis, mq)
    rinterp = basis.nodal_basis_at(mq.pts)
    targets = [
        ((-1, -1), (0, -1), (-1, 0), (0, 0)),
        ((0, -1), (1, -1), (0, 0), (1, 0)),
        ((-1, 0), (0, 0), (-1, 1), (0, 1)),
        ((0, 0), (1, 0), (0, 1), (1, 1)),
    ]

    left = np.zeros(len(q.pts))
    for target in targets:
        rmap = MortarReferenceMap('quad', target)
        lint = basis.nodal_basis_at(rmap.apply(mq.pts))
        left += np.linalg.solve(
            mass, lint.T*(mq.wts*rmap.determinant)
        ) @ np.ones(len(mq.pts))

        right_state = np.linalg.solve(
            mass, rinterp.T*mq.wts
        ) @ np.ones(len(mq.pts))
        right_flux = np.linalg.solve(
            mass, rinterp.T*(mq.wts*rmap.determinant)
        ) @ np.ones(len(mq.pts))
        assert np.allclose(right_state, 1.0, atol=4e-14)
        assert np.allclose(right_flux, 0.25, atol=4e-14)

    assert np.allclose(left, 1.0, atol=4e-14)


@pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
def test_quad_to_four_quad_common_trace_reproduction(order):
    q = get_quadrule(
        'quad', rule='gauss-legendre', npts=(order + 1)**2
    )
    mq = get_quadrule(
        'quad', rule='gauss-legendre', npts=(order + 2)**2
    )
    basis = get_polybasis('quad', order, q.pts)
    mass = _mass_matrix(basis, mq)
    rinterp = basis.nodal_basis_at(mq.pts)
    targets = [
        ((-1, -1), (0, -1), (-1, 0), (0, 0)),
        ((0, -1), (1, -1), (0, 0), (1, 0)),
        ((-1, 0), (0, 0), (-1, 1), (0, 1)),
        ((0, 0), (1, 0), (0, 1), (1, 1)),
    ]

    degree = min(order, 2)
    def trace(points):
        x, y = points.T
        values = 1 + 0.2*x - 0.3*y
        if degree == 2:
            values += 0.1*x*y + 0.05*x*x
        return values

    left = np.zeros(len(q.pts))
    for target in targets:
        rmap = MortarReferenceMap('quad', target)
        lpts = rmap.apply(mq.pts)
        lint = basis.nodal_basis_at(lpts)
        weights = mq.wts*rmap.determinant
        left += np.linalg.solve(mass, lint.T*weights) @ trace(lpts)

        right = np.linalg.solve(
            mass, rinterp.T*weights
        ) @ trace(lpts)
        assert np.allclose(right, 0.25*trace(rmap.apply(q.pts)), atol=8e-13)

    assert np.allclose(left, trace(q.pts), atol=8e-13)


def test_mortar_stats_record():
    from pyfr.solvers.base.mortars import BaseMortarInters

    mortar = BaseMortarInters()
    values = {
        'name': 'mortar-0', 'ninters': 8,
        'coarse_etype': 'hex', 'fine_etype': 'tet',
        'mortar_implementation': 'staged', 'noperator_sets': 2,
        'nbatches': 2, 'operator_bytes': 1024,
        'shared_operator_bytes': 1024, 'fused_operator_bytes': 8192,
        'geometry_bytes': 512, 'staged_buffer_bytes': 4096,
        'staged_allocated_bytes': 3072, 'max_geom_error': 1e-13,
        'max_normal_error': 2e-13,
    }
    for name, value in values.items():
        setattr(mortar, name, value)

    stats = mortar.stats
    assert stats['implementation'] == 'staged'
    assert stats['operator-sets'] == 2
    assert stats['staged-allocated-bytes'] == 3072
    assert stats['max-normal-error'] == 2e-13


def test_mortar_geometry_resource_contract():
    qpts = np.array([[-0.5, -0.5], [0.5, 0.5]])
    qwts = np.array([1.0, 2.0])
    points = np.zeros((2, 3, 2))
    normals = np.zeros((2, 3, 2))
    normals[:, 0, 0] = [2.0, 3.0]
    normals[:, 0, 1] = [4.0, 5.0]
    det = np.array([0.5, 0.25])
    cerr = np.array([1e-14, 2e-14])
    nerr = np.array([3e-14, 4e-14])

    geometry = MortarGeometry(
        'left', 'tri', qpts, qwts, (points,), (normals,), (det,),
        (cerr,), (nerr,),
    )

    assert geometry.authority == 'left'
    assert np.array_equal(geometry.surface_jacobians[0], [
        [1.0, 1.0], [1.5, 1.25]
    ])
    assert np.array_equal(geometry.integration_weights[0], [
        [1.0, 1.0], [3.0, 2.5]
    ])
    assert geometry.max_coordinate_error == 2e-14
    assert geometry.max_normal_error == 4e-14
    assert geometry.backend_nbytes == normals.nbytes

    subset = geometry.subset((1,))
    assert subset.physical_points[0].shape == (2, 3, 1)
    assert np.array_equal(subset.patch_determinants[0], [0.25])
    assert subset.max_coordinate_error == 2e-14


def test_local_mortar_ownership_contract():
    from pyfr.solvers.base.mortars import MortarOwnership

    ownership = MortarOwnership.local(3)
    assert ownership.kind == 'local'
    assert ownership.owner_rank == 3
    assert ownership.participant_ranks == (3,)

    with pytest.raises(ValueError, match='one participant rank'):
        MortarOwnership('local', 3, (3, 4))
    distributed = MortarOwnership.distributed(3, (4, 3))
    assert distributed.kind == 'distributed'
    assert distributed.owner_rank == 3
    assert distributed.participant_ranks == (3, 4)

    with pytest.raises(ValueError, match='two distinct'):
        MortarOwnership.distributed(3, (3, 3))
    with pytest.raises(ValueError, match='including the owner'):
        MortarOwnership.distributed(3, (4, 5))


def test_mortar_execution_batches_are_homogeneous():
    from types import SimpleNamespace

    from pyfr.solvers.base.mortars import (
        MortarExecutionSignature, MortarOwnership,
        build_mortar_execution_batches,
    )

    group = MortarGroup(
        MortarSide('hex', 'quad', (1, 1, 1, 1), (4, 5, 6, 7)),
        (
            MortarSide('tet', 'tri', (2, 2, 2, 2), (8, 9, 10, 11)),
            MortarSide('tet', 'tri', (3, 3, 3, 3), (12, 13, 14, 15)),
        ),
    )
    keys = (
        MortarOperatorKey(
            MortarTraceKey('hex', 'quad', 3, 1),
            (
                MortarPatch(
                    MortarTraceKey('tet', 'tri', 3, 2), (0, 2, 1)
                ),
                MortarPatch(
                    MortarTraceKey('tet', 'tri', 3, 3), (2, 3, 1)
                ),
            ),
            'tri', 3, 8, ('nodal', 'nodal'),
        ),
        MortarOperatorKey(
            MortarTraceKey('hex', 'quad', 3, 1),
            (
                MortarPatch(
                    MortarTraceKey('tet', 'tri', 3, 2), (0, 1, 3)
                ),
                MortarPatch(
                    MortarTraceKey('tet', 'tri', 3, 3), (0, 3, 2)
                ),
            ),
            'tri', 3, 8, ('nodal', 'nodal'),
        ),
    )
    geometry = MortarGeometry(
        'left', 'tri', np.zeros((2, 2)), np.ones(2),
        (np.zeros((2, 3, 4)), np.zeros((2, 3, 4))),
        (np.ones((2, 3, 4)), np.ones((2, 3, 4))),
        (np.ones(4), np.ones(4)),
        (np.zeros(4), np.zeros(4)),
        (np.zeros(4), np.zeros(4)),
    )
    ops = {
        'mortar_group': group,
        'operator_keys': keys,
        'operator_groups': (np.array([0, 2]), np.array([1, 3])),
        'geometry': geometry,
    }
    elemap = {
        'hex': SimpleNamespace(
            basis=SimpleNamespace(nsptsord=2),
            antialias={'flux'},
        ),
        'tet': SimpleNamespace(
            basis=SimpleNamespace(nsptsord=3),
            antialias={'flux', 'surf-flux'},
        ),
    }

    ownership = MortarOwnership.local(7)
    batches = build_mortar_execution_batches(
        ops, elemap, 'euler', 'staged', ownership
    )

    assert len(batches) == 2
    assert batches[0].signature == MortarExecutionSignature(
        keys[0], (2, 3, 3),
        (('flux',), ('flux', 'surf-flux'), ('flux', 'surf-flux')),
        'euler', 'local', 'staged'
    )
    assert batches[0].indices == (0, 2)
    assert batches[0].operator_set == 0
    assert batches[0].ownership == ownership
    assert batches[0].geometry.authority == 'left'
    assert batches[0].geometry.physical_points[0].shape == (2, 3, 2)
    assert batches[0].group.left.eidxs == (4, 6)
    assert batches[0].group.right[0].eidxs == (8, 10)
    assert batches[0].group.right[1].eidxs == (12, 14)
    assert batches[1].indices == (1, 3)
    assert batches[1].operator_set == 1

    ns_batches = build_mortar_execution_batches(
        ops, elemap, 'navier-stokes', 'staged', ownership
    )
    assert ns_batches[0].signature != batches[0].signature
    assert ns_batches[0].signature.equation == 'navier-stokes'


def _p_mortar_cfg(anti_alias='none'):
    from pyfr.inifile import Inifile

    return Inifile(f'''\
[solver]
order = 2
anti-alias = {anti_alias}

[solver-elements-hex]
soln-pts = gauss-legendre

[solver-interfaces]
riemann-solver = rusanov
mortar-geom-tol = 1e-10

[solver-interfaces-quad]
flux-pts = gauss-legendre
''')


def _line_mortar_cfg(anti_alias='none'):
    from pyfr.inifile import Inifile

    return Inifile(f'''\
[solver]
order = 3
anti-alias = {anti_alias}

[solver-elements-quad]
soln-pts = gauss-legendre

[solver-interfaces]
riemann-solver = rusanov
mortar-geom-tol = 1e-10

[solver-interfaces-line]
flux-pts = gauss-legendre
''')


def _line_1x2_ops(
    reverse_edge, anti_alias='none', *, state_projection=False
):
    from types import SimpleNamespace

    from pyfr.mortars import build_line_line2_operators
    from pyfr.shapes import QuadShape
    from pyfr.solvers.euler.elements import EulerElements

    cfg = _line_mortar_cfg(anti_alias)
    std = np.asarray(QuadShape.std_ele(1), dtype=float)

    def cell(xr, yr):
        points = std.copy()
        if reverse_edge:
            points[:, 0] = (1 - points[:, 0])*(xr[1] - xr[0])/2 + xr[0]
            points[:, 1] = (1 - points[:, 1])*(yr[1] - yr[0])/2 + yr[0]
        else:
            points[:, 0] = (points[:, 0] + 1)*(xr[1] - xr[0])/2 + xr[0]
            points[:, 1] = (points[:, 1] + 1)*(yr[1] - yr[0])/2 + yr[0]
        return points

    spts = np.stack((
        cell((-1.0, 0.0), (-1.0, 1.0)),
        cell((0.0, 1.0), (-1.0, 0.0)),
        cell((0.0, 1.0), (0.0, 1.0)),
    ), axis=1)
    elements = EulerElements(QuadShape, spts, cfg, order=3)
    left_fidx, right_fidx = (3, 1) if reverse_edge else (1, 3)
    right_eidx = (1, 2) if reverse_edge else (2, 1)
    dtype = [
        ('left_cidx', 'i4'), ('left_eidx', 'i8'),
        ('right_cidx', 'i4', (2,)), ('right_eidx', 'i8', (2,)),
    ]
    records = np.zeros(1, dtype=dtype)
    records['left_cidx'] = 0
    records['right_cidx'] = 1
    records['right_eidx'] = right_eidx
    mcon = SimpleNamespace(
        records=records, format='one-to-many-v1', template='line-1x2',
        nright=2,
    )
    mesh = SimpleNamespace(
        cidxmap={0: ('quad', left_fidx), 1: ('quad', right_fidx)},
        spts={'quad': spts},
    )

    ops = build_line_line2_operators(
        mesh, {'quad': elements}, mcon, cfg,
        state_projection=state_projection
    )
    return ops, elements


@pytest.mark.parametrize('reverse_edge', [False, True])
@pytest.mark.parametrize('anti_alias', ['none', 'surf-flux'])
def test_line_line2_operators_are_canonical_and_conservative(
    reverse_edge, anti_alias
):
    from pyfr.mortars import _face_sample_maps

    ops, elements = _line_1x2_ops(reverse_edge, anti_alias)
    group = ops['mortar_group']
    key = ops['operator_keys'][0]
    mats = ops['operator_sets'][0]
    line = get_quadrule('line', 'gauss-legendre', 5)
    basis = elements.basis.facebases['line']
    sample = (elements.basis._iqrules['line'].pts
              if anti_alias == 'surf-flux' else basis.pts)

    expected_eidx = (2, 1) if reverse_edge else (1, 2)
    assert tuple(side.eidxs[0] for side in group.right) == expected_eidx
    assert key.mortar_topology == 'line'
    assert key.quadrature == MortarQuadratureKey(
        'line', rule='gauss-legendre', npts=5
    )
    assert ops['nmpts'] == 5
    assert ops['nleftfpts'] == elements.nfacefpts[group.left.fidxs[0]]
    assert ops['nrightfpts'] == tuple(
        elements.nfacefpts[side.fidxs[0]] for side in group.right
    )
    assert np.allclose(ops['patch_determinants'][0], 0.5)
    assert np.allclose(ops['patch_determinants'][1], 0.5)
    assert ops['max_geom_error'] < 1e-12
    assert ops['max_normal_error'] < 1e-12

    def trace(points):
        s = np.asarray(points).reshape(-1)
        return 1.0 + 0.2*s - 0.1*s*s

    left = np.zeros(len(sample))
    right = []
    for patch, li, ri, lp, rp in zip(
        key.patches, mats['left_interp'], mats['right_interp'],
        mats['left_proj'], mats['right_proj'],
    ):
        common = trace(patch.left_map.apply(line.pts))
        rvals = trace(patch.left_map.apply(sample))
        assert np.allclose(li @ trace(sample), common, atol=5e-13)
        assert np.allclose(ri @ rvals, common, atol=5e-13)

        left += lp @ common
        right.append(rp @ common)
        assert np.allclose(
            rp @ common, 0.5*rvals, atol=5e-13
        )

    assert np.allclose(left, trace(sample), atol=5e-13)

    mass = _mass_matrix(basis, line)
    to_nodal, _ = _face_sample_maps(elements, 'line', basis)
    left_integral = np.ones(len(basis.pts)) @ mass @ (to_nodal @ left)
    right_integral = sum(
        np.ones(len(basis.pts)) @ mass @ (to_nodal @ values)
        for values in right
    )
    assert abs(left_integral - right_integral) < 5e-13


@pytest.mark.parametrize('reverse_edge', [False, True])
@pytest.mark.parametrize('anti_alias', ['none', 'surf-flux'])
def test_line_line2_state_projection_is_not_flux_scaled(
    reverse_edge, anti_alias
):
    ops, elements = _line_1x2_ops(
        reverse_edge, anti_alias, state_projection=True
    )
    mats = ops['operator_sets'][0]
    key = ops['operator_keys'][0]
    mq = get_quadrule('line', 'gauss-legendre', 5).pts
    basis = elements.basis.facebases['line']
    sample = (elements.basis._iqrules['line'].pts
              if anti_alias == 'surf-flux' else basis.pts)

    def trace(points):
        s = np.asarray(points).reshape(-1)
        return 1.0 + 0.2*s - 0.1*s*s

    assert 'right_state_proj' in mats
    for patch, state_proj, flux_proj in zip(
        key.patches, mats['right_state_proj'], mats['right_proj']
    ):
        common = trace(patch.left_map.apply(mq))
        expected = trace(patch.left_map.apply(sample))

        assert np.allclose(state_proj @ common, expected, atol=5e-13)
        assert np.allclose(flux_proj @ common, 0.5*expected, atol=5e-13)
        assert np.allclose(state_proj, 2*flux_proj, atol=5e-14)


def _affine_hex_elements(order, xr, cfg):
    from pyfr.shapes import HexShape
    from pyfr.solvers.euler.elements import EulerElements

    spts = np.asarray(HexShape.std_ele(1), dtype=float)
    spts[:, 0] = (spts[:, 0] + 1)*(xr[1] - xr[0])/2 + xr[0]
    return EulerElements(HexShape, spts[:, None, :], cfg, order=order)


def _p_mortar_ops(left_order=2, right_order=3, anti_alias='none'):
    from pyfr.mortars import build_quad_p_operators
    from pyfr.solvers.base.groups import ElementGroupKey

    cfg = _p_mortar_cfg(anti_alias)
    lkey = ElementGroupKey('hex', left_order)
    rkey = ElementGroupKey('hex', right_order)
    elemap = {
        lkey: _affine_hex_elements(left_order, (-1.0, 0.0), cfg),
        rkey: _affine_hex_elements(right_order, (0.0, 1.0), cfg),
    }
    group = MortarGroup(
        MortarSide('hex', 'quad', (2,), (0,), lkey),
        (MortarSide('hex', 'quad', (4,), (0,), rkey),),
    )
    return build_quad_p_operators(None, elemap, group, cfg), elemap


def test_quad_full_face_reference_map_all_d4():
    from pyfr.mortars import _quad_full_face_reference_map

    q = np.asarray(QuadShape.std_ele(1), dtype=float)
    left = np.column_stack((np.zeros(4), q))
    transforms = (
        lambda x, y: (x, y),
        lambda x, y: (-y, x),
        lambda x, y: (-x, -y),
        lambda x, y: (y, -x),
        lambda x, y: (-x, y),
        lambda x, y: (x, -y),
        lambda x, y: (y, x),
        lambda x, y: (-y, -x),
    )

    targets = set()
    for transform in transforms:
        tq = np.asarray([transform(*xy) for xy in q])
        right = np.column_stack((np.zeros(4), tq))
        rmap = _quad_full_face_reference_map(left, right)
        assert np.isclose(rmap.determinant, 1.0)
        assert np.allclose(rmap.apply(q), np.asarray(rmap.target))
        targets.add(rmap.target)

    assert len(targets) == 8


@pytest.mark.parametrize('orders', [(2, 3), (3, 2)])
@pytest.mark.parametrize('anti_alias', ['none', 'surf-flux'])
def test_quad_p_operator_shapes_constants_and_q2_trace(orders, anti_alias):
    ops, elemap = _p_mortar_ops(*orders, anti_alias=anti_alias)
    key = ops['operator_keys'][0]
    mats = ops['operator_sets'][0]
    left = elemap[ops['mortar_group'].left.elekey]
    right = elemap[ops['mortar_group'].right[0].elekey]

    assert key.left.solution_order == orders[0]
    assert key.patches[0].right.solution_order == orders[1]
    assert key.mortar_order == 3
    assert key.quadrature.npts == 25
    assert ops['nmpts'] == 25
    assert ops['nleftfpts'] == left.nfacefpts[2]
    assert ops['nrightfpts'] == (right.nfacefpts[4],)
    assert ops['max_geom_error'] < 1e-13
    assert ops['max_normal_error'] < 1e-13
    assert np.allclose(ops['patch_determinants'][0], 1.0)

    li = mats['left_interp'][0]
    ri = mats['right_interp'][0]
    lp = mats['left_proj'][0]
    rp = mats['right_proj'][0]
    common_one = np.ones(ops['nmpts'])

    assert np.allclose(li @ np.ones(li.shape[1]), common_one, atol=1e-13)
    assert np.allclose(ri @ np.ones(ri.shape[1]), common_one, atol=1e-13)
    assert np.allclose(lp @ common_one, 1.0, atol=1e-13)
    assert np.allclose(rp @ common_one, 1.0, atol=1e-13)

    rmap = key.patches[0].left_map
    mq = get_quadrule('quad', 'gauss-legendre', 25).pts
    lb = left.basis.facebases['quad']
    rb = right.basis.facebases['quad']

    def q2(pts):
        x, y = pts.T
        return 1 + 0.2*x - 0.3*y + 0.1*x*y + 0.05*x*x - 0.04*y*y

    lsample = (left.basis._iqrules['quad'].pts if anti_alias == 'surf-flux'
               else lb.pts)
    rsample = (right.basis._iqrules['quad'].pts if anti_alias == 'surf-flux'
               else rb.pts)
    lvals = q2(lsample)
    rvals = q2(rmap.apply(rsample))
    expected = q2(rmap.apply(mq))
    assert np.allclose(li @ lvals, expected, atol=5e-13)
    assert np.allclose(ri @ rvals, expected, atol=5e-13)
    assert np.allclose(lp @ expected, lvals, atol=5e-13)
    assert np.allclose(rp @ expected, rvals, atol=5e-13)


@pytest.mark.parametrize('orders', [(2, 3), (3, 2)])
def test_quad_p_integrated_reference_flux_conservation(orders):
    ops, elemap = _p_mortar_ops(*orders)
    mats = ops['operator_sets'][0]
    left = elemap[ops['mortar_group'].left.elekey]
    right = elemap[ops['mortar_group'].right[0].elekey]
    key = ops['operator_keys'][0]
    mq = get_quadrule('quad', 'gauss-legendre', 25)

    lb = left.basis.facebases['quad']
    rb = right.basis.facebases['quad']
    massq = get_quadrule('quad', 'gauss-legendre', 25)
    lm = _mass_matrix(lb, massq)
    rm = _mass_matrix(rb, massq)

    x, y = key.patches[0].left_map.apply(mq.pts).T
    flux = 1 + 0.2*x - 0.3*y + 0.1*x*y + 0.05*x*x
    lf = mats['left_proj'][0] @ flux
    rf = -(mats['right_proj'][0] @ flux)

    lint = np.ones(len(lb.pts)) @ lm @ lf
    rint = np.ones(len(rb.pts)) @ rm @ rf
    assert abs(lint + rint) < 1e-13


def test_quad_p_state_and_flux_projection_remain_distinct_objects():
    from pyfr.mortars import build_quad_p_operators

    ops, elemap = _p_mortar_ops(2, 3)
    group = ops['mortar_group']
    cfg = _p_mortar_cfg()
    sops = build_quad_p_operators(
        None, elemap, group, cfg, state_projection=True
    )['operator_sets'][0]

    assert 'right_state_proj' in sops
    assert sops['right_state_proj'][0] is not sops['right_proj'][0]
    assert np.allclose(sops['right_state_proj'][0], sops['right_proj'][0])



def _oriented_right_hex(order, mode, cfg):
    from pyfr.shapes import HexShape, proj_pts
    from pyfr.solvers.euler.elements import EulerElements

    transforms = (
        lambda x, y, z: (x, y, z),
        lambda x, y, z: (x, z, -y),
        lambda x, y, z: (x, -y, -z),
        lambda x, y, z: (x, -z, y),
        lambda x, y, z: (-x, y, -z),
        lambda x, y, z: (-x, -y, z),
        lambda x, y, z: (-x, z, y),
        lambda x, y, z: (-x, -z, -y),
    )
    std = np.asarray(HexShape.std_ele(1), dtype=float)
    spts = np.asarray([transforms[mode](*p) for p in std])
    spts[:, 0] = (spts[:, 0] + 1)/2
    eles = EulerElements(HexShape, spts[:, None, :], cfg, order=order)

    qcorners = np.asarray(QuadShape.std_ele(1), dtype=float)
    matches = []
    for fidx, (kind, proj, _) in enumerate(eles.basis.faces):
        if kind != 'quad':
            continue
        locs = eles.ploc_at_np(proj_pts(proj, qcorners))[:, :, 0]
        if np.max(np.abs(locs[:, 0])) < 1e-12:
            matches.append(fidx)

    assert len(matches) == 1
    return eles, matches[0]


@pytest.mark.parametrize('orders', [(2, 3), (3, 2)])
def test_quad_p_operator_all_eight_d4_real_geometry(orders):
    from pyfr.mortars import build_quad_p_operators
    from pyfr.solvers.base.groups import ElementGroupKey

    cfg = _p_mortar_cfg()
    lkey = ElementGroupKey('hex', orders[0])
    rkey = ElementGroupKey('hex', orders[1])
    left = _affine_hex_elements(orders[0], (-1.0, 0.0), cfg)
    targets = set()

    for mode in range(8):
        right, rfidx = _oriented_right_hex(orders[1], mode, cfg)
        group = MortarGroup(
            MortarSide('hex', 'quad', (2,), (0,), lkey),
            (MortarSide('hex', 'quad', (rfidx,), (0,), rkey),),
        )
        ops = build_quad_p_operators(
            None, {lkey: left, rkey: right}, group, cfg
        )
        key = ops['operator_keys'][0]

        assert key.left.solution_order == orders[0]
        assert key.patches[0].right.solution_order == orders[1]
        assert np.isclose(key.patches[0].left_map.determinant, 1.0)
        assert ops['max_geom_error'] < 5e-15
        assert ops['max_normal_error'] < 5e-15
        targets.add(key.patches[0].left_map.target)

    assert len(targets) == 8


def test_p_mortar_execution_batch_uses_runtime_element_keys():
    from types import SimpleNamespace

    from pyfr.solvers.base.groups import ElementGroupKey
    from pyfr.solvers.base.mortars import (
        build_mortar_execution_batches, MortarOwnership,
    )

    lkey = ElementGroupKey('hex', 2)
    rkey = ElementGroupKey('hex', 3)
    group = MortarGroup(
        MortarSide('hex', 'quad', (2,), (0,), lkey),
        (MortarSide('hex', 'quad', (4,), (0,), rkey),),
    )
    key = MortarOperatorKey(
        MortarTraceKey('hex', 'quad', 2, 2),
        (MortarPatch(
            MortarTraceKey('hex', 'quad', 3, 4),
            MortarReferenceMap('quad', tuple(map(tuple, QuadShape.std_ele(1)))),
        ),),
        'quad', 3,
        MortarQuadratureKey('quad', rule='gauss-legendre', npts=25),
        ('nodal', 'nodal'),
    )
    q = get_quadrule('quad', 'gauss-legendre', 25)
    zeros = np.zeros((25, 3, 1))
    geometry = MortarGeometry(
        'left', 'quad', q.pts, q.wts, (zeros,), (zeros,),
        (np.ones(1),), (np.zeros(1),), (np.zeros(1),),
    )
    ops = {
        'mortar_group': group,
        'operator_keys': (key,),
        'operator_groups': (np.array([0]),),
        'geometry': geometry,
    }
    elemap = {
        lkey: SimpleNamespace(
            basis=SimpleNamespace(nsptsord=1), antialias=set()
        ),
        rkey: SimpleNamespace(
            basis=SimpleNamespace(nsptsord=1), antialias=set()
        ),
    }

    batches = build_mortar_execution_batches(
        ops, elemap, 'euler', 'staged', MortarOwnership.local(0)
    )
    assert len(batches) == 1
    assert batches[0].group.left.elekey == lkey
    assert batches[0].group.right[0].elekey == rkey
    assert batches[0].signature.geometry_orders == (1, 1)


@pytest.mark.parametrize('orders', [(2, 3), (3, 2)])
@pytest.mark.parametrize('anti_alias', ['none', 'surf-flux'])
@pytest.mark.parametrize('mode', range(8))
def test_distributed_quad_p_reference_matches_local_c2(
    orders, anti_alias, mode
):
    from pyfr.mortars import (
        DistributedPMortarFace, MPIMortarFaceKey, RemoteMortarSide,
        build_distributed_quad_p_operators, build_quad_p_operators,
    )
    from pyfr.solvers.base.groups import ElementGroupKey

    cfg = _p_mortar_cfg(anti_alias)
    lkey = ElementGroupKey('hex', orders[0])
    rkey = ElementGroupKey('hex', orders[1])
    left = _affine_hex_elements(orders[0], (-1.0, 0.0), cfg)
    right, rfidx = _oriented_right_hex(orders[1], mode, cfg)
    group = MortarGroup(
        MortarSide('hex', 'quad', (2,), (0,), lkey),
        (MortarSide('hex', 'quad', (rfidx,), (0,), rkey),),
    )
    local = build_quad_p_operators(
        None, {lkey: left, rkey: right}, group, cfg,
        state_projection=True
    )
    rmap = local['operator_keys'][0].patches[0].left_map
    face = DistributedPMortarFace(
        1, group.left,
        RemoteMortarSide(
            'hex', 'quad', orders[1], right.basis.nsptsord, rfidx, 11
        ),
        MPIMortarFaceKey('hex', 10, 2),
        MPIMortarFaceKey('hex', 11, rfidx),
        left.basis.nsptsord, 0, rmap, local['geometry']
    )
    distributed = build_distributed_quad_p_operators(
        face, cfg, state_projection=True
    )

    assert distributed['operator_keys'] == local['operator_keys']
    assert distributed['nleftfpts'] == local['nleftfpts']
    assert distributed['nrightfpts'] == local['nrightfpts']
    assert distributed['nmpts'] == local['nmpts']
    assert distributed['trace_sampling'] == local['trace_sampling']
    assert distributed['max_geom_error'] == local['max_geom_error']
    assert distributed['max_normal_error'] == local['max_normal_error']

    for dset, lset in zip(
        distributed['operator_sets'], local['operator_sets']
    ):
        assert set(dset) == set(lset)
        for name in dset:
            for dm, lm in zip(dset[name], lset[name]):
                assert np.max(np.abs(dm - lm), initial=0.0) < 5e-14

    mats = distributed['operator_sets'][0]
    li = mats['left_interp'][0]
    ri = mats['right_interp'][0]
    lp = mats['left_proj'][0]
    rp = mats['right_proj'][0]
    common_one = np.ones(distributed['nmpts'])

    assert np.allclose(li @ np.ones(li.shape[1]), common_one, atol=5e-14)
    assert np.allclose(ri @ np.ones(ri.shape[1]), common_one, atol=5e-14)
    assert np.allclose(lp @ common_one, 1.0, atol=5e-14)
    assert np.allclose(rp @ common_one, 1.0, atol=5e-14)

    mq = get_quadrule('quad', 'gauss-legendre', distributed['nmpts'])
    lb = left.basis.facebases['quad']
    rb = right.basis.facebases['quad']

    def q2(pts):
        x, y = pts.T
        return 1 + 0.2*x - 0.3*y + 0.1*x*y + 0.05*x*x - 0.04*y*y

    lsample = (left.basis._iqrules['quad'].pts
               if anti_alias == 'surf-flux' else lb.pts)
    rsample = (right.basis._iqrules['quad'].pts
               if anti_alias == 'surf-flux' else rb.pts)
    lvals = q2(lsample)
    rvals = q2(rmap.apply(rsample))
    expected = q2(rmap.apply(mq.pts))

    assert np.allclose(li @ lvals, expected, atol=2e-13)
    assert np.allclose(ri @ rvals, expected, atol=2e-13)
    assert np.allclose(lp @ expected, lvals, atol=2e-13)
    assert np.allclose(rp @ expected, rvals, atol=2e-13)

    if anti_alias == 'none':
        lm = _mass_matrix(lb, mq)
        rm = _mass_matrix(rb, mq)
        flux = expected
        lf = lp @ flux
        rf = -(rp @ flux)
        lint = np.ones(len(lb.pts)) @ lm @ lf
        rint = np.ones(len(rb.pts)) @ rm @ rf
        assert abs(lint + rint) < 5e-14


def test_distributed_quad_p_rejects_unequal_geometry_order():
    from dataclasses import replace

    from pyfr.mortars import (
        DistributedPMortarFace, MPIMortarFaceKey, RemoteMortarSide,
        build_distributed_quad_p_operators,
    )
    from pyfr.solvers.base.groups import ElementGroupKey

    cfg = _p_mortar_cfg()
    lkey = ElementGroupKey('hex', 2)
    rkey = ElementGroupKey('hex', 3)
    ops, elemap = _p_mortar_ops(2, 3)
    group = ops['mortar_group']
    left, right = elemap[lkey], elemap[rkey]
    rmap = ops['operator_keys'][0].patches[0].left_map
    face = DistributedPMortarFace(
        1, group.left,
        RemoteMortarSide(
            'hex', 'quad', 3, right.basis.nsptsord, 4, 11
        ),
        MPIMortarFaceKey('hex', 10, 2),
        MPIMortarFaceKey('hex', 11, 4),
        left.basis.nsptsord, 0, rmap, ops['geometry']
    )
    face = replace(face, local_geometry_order=left.basis.nsptsord + 1)

    with pytest.raises(ValueError, match='equal geometry order'):
        build_distributed_quad_p_operators(face, cfg)


def test_mpi_p_mortar_batch_plan_is_deterministic():
    from dataclasses import replace

    from pyfr.mortars import (
        DistributedPMortarFace, MPIMortarFaceKey, RemoteMortarSide,
        build_distributed_quad_p_operators,
    )
    from pyfr.solvers.base.groups import ElementGroupKey
    from pyfr.solvers.base.mortars import build_mpi_p_mortar_batch_plans

    cfg = _p_mortar_cfg()
    lkey = ElementGroupKey('hex', 2)
    rkey = ElementGroupKey('hex', 3)
    ops, elemap = _p_mortar_ops(2, 3)
    group = ops['mortar_group']
    rmap = ops['operator_keys'][0].patches[0].left_map
    left, right = elemap[lkey], elemap[rkey]
    face = DistributedPMortarFace(
        1, group.left,
        RemoteMortarSide('hex', 'quad', 3, right.basis.nsptsord, 4, 11),
        MPIMortarFaceKey('hex', 10, 2),
        MPIMortarFaceKey('hex', 11, 4),
        left.basis.nsptsord, 0, rmap, ops['geometry']
    )
    face2 = replace(
        face,
        local_key=MPIMortarFaceKey('hex', 20, 2),
        remote_key=MPIMortarFaceKey('hex', 21, 4),
    )
    dops = build_distributed_quad_p_operators(face, cfg)
    dops2 = build_distributed_quad_p_operators(face2, cfg)

    plans = build_mpi_p_mortar_batch_plans(
        [(face2, dops2), (face, dops)]
    )
    assert len(plans) == 1
    plan = plans[0]
    assert plan.neighbour_rank == 1
    assert plan.ownership.owner_rank == 0
    assert plan.face_pairs == (face.face_pair, face2.face_pair)
    assert plan.tags == (
        ('state', 0), ('common-state', 1), ('gradient', 2), ('flux', 3)
    )

def test_mpi_p_mortar_batch_plan_splits_execution_signatures():
    from dataclasses import replace

    from pyfr.mortars import (
        DistributedPMortarFace, MPIMortarFaceKey, RemoteMortarSide,
        build_distributed_quad_p_operators,
    )
    from pyfr.solvers.base.groups import ElementGroupKey
    from pyfr.solvers.base.mortars import build_mpi_p_mortar_batch_plans

    cfg = _p_mortar_cfg()
    lkey = ElementGroupKey('hex', 2)
    rkey = ElementGroupKey('hex', 3)
    ops, elemap = _p_mortar_ops(2, 3)
    group = ops['mortar_group']
    rmap = ops['operator_keys'][0].patches[0].left_map
    left, right = elemap[lkey], elemap[rkey]
    face = DistributedPMortarFace(
        1, group.left,
        RemoteMortarSide('hex', 'quad', 3, right.basis.nsptsord, 4, 11),
        MPIMortarFaceKey('hex', 10, 2),
        MPIMortarFaceKey('hex', 11, 4),
        left.basis.nsptsord, 0, rmap, ops['geometry']
    )
    other = replace(
        face,
        remote_side=replace(face.remote_side, fidx=3, global_eidx=21),
        local_key=MPIMortarFaceKey('hex', 20, 2),
        remote_key=MPIMortarFaceKey('hex', 21, 3),
    )
    fops = build_distributed_quad_p_operators(face, cfg)
    oops = build_distributed_quad_p_operators(other, cfg)

    plans = build_mpi_p_mortar_batch_plans([(other, oops), (face, fops)])

    assert len(plans) == 2
    assert plans[0].signature != plans[1].signature
    assert plans[0].tags == (
        ('state', 0), ('common-state', 1), ('gradient', 2), ('flux', 3)
    )
    assert plans[1].tags == (
        ('state', 4), ('common-state', 5), ('gradient', 6), ('flux', 7)
    )
