import itertools as it

import numpy as np
import pytest

from pyfr.amr import (
    HexRefineTransferKey, QuadRefineTransferKey,
    apply_hex_refine_transfer, apply_quad_refine_transfer,
    build_hex_refine_transfer, build_quad_refine_transfer,
    hex_child_to_parent, quad_child_to_parent,
)
from pyfr.inifile import Inifile
from pyfr.quadrules import get_quadrule
from pyfr.shapes import HexShape, QuadShape


def _basis(order, soln_pts='gauss-legendre', anti_alias='none'):
    cfg = Inifile(f'''\
[solver]
order = {order}
anti-alias = {anti_alias}

[solver-elements-hex]
soln-pts = {soln_pts}
''')
    return HexShape(None, cfg)


def _quad_basis(order, soln_pts='gauss-legendre', anti_alias='none'):
    cfg = Inifile(f'''\
[solver]
order = {order}
anti-alias = {anti_alias}

[solver-elements-quad]
soln-pts = {soln_pts}
''')
    return QuadShape(None, cfg)


def _monomial(pts, powers):
    return np.prod(pts**np.asarray(powers), axis=1)


def _scaled_tol(*arrays, factor=4096):
    scale = max(1.0, *(float(np.max(np.abs(a), initial=0.0))
                       for a in arrays))
    return factor*np.finfo(float).eps*scale


def _integral_weights(basis):
    qrule = get_quadrule(
        'hex', rule='gauss-legendre', npts=(basis.order + 2)**3
    )
    interp = basis.ubasis.nodal_basis_at(qrule.pts)
    return qrule.wts @ interp


def _quad_integral_weights(basis):
    qrule = get_quadrule(
        'quad', rule='gauss-legendre', npts=(basis.order + 2)**2
    )
    interp = basis.ubasis.nodal_basis_at(qrule.pts)
    return qrule.wts @ interp


def test_quad_child_to_parent_matches_quadrants():
    corners = QuadShape.std_ele(1)

    for quadrant in range(4):
        mapped = quad_child_to_parent(quadrant, corners)
        bits = (quadrant & 1, (quadrant >> 1) & 1)

        for d, bit in enumerate(bits):
            expect = (0.0, 1.0) if bit else (-1.0, 0.0)
            assert np.allclose(
                (mapped[:, d].min(), mapped[:, d].max()),
                expect, rtol=0, atol=0,
            )


def test_quad_child_to_parent_rejects_invalid_inputs():
    with pytest.raises(ValueError, match='quadrant'):
        quad_child_to_parent(4, np.zeros((1, 2)))
    with pytest.raises(ValueError, match='quadrant'):
        quad_child_to_parent(1.0, np.zeros((1, 2)))
    with pytest.raises(ValueError, match='shape'):
        quad_child_to_parent(0, np.zeros((2,)))


def test_quad_refine_transfer_key_distinguishes_solution_points():
    gl = build_quad_refine_transfer(_quad_basis(3, 'gauss-legendre'))
    gll = build_quad_refine_transfer(
        _quad_basis(3, 'gauss-legendre-lobatto')
    )

    assert gl.nupts == gll.nupts
    assert gl.key != gll.key
    assert gl.key.template == 'quadtree-2x2-v1'

    cache = {gl.key: gl}
    repeat_key = build_quad_refine_transfer(_quad_basis(3)).key
    assert cache[repeat_key] is gl
    assert gll.key not in cache


def test_quad_refine_transfer_is_surface_aa_independent():
    plain = build_quad_refine_transfer(_quad_basis(3))
    aa = build_quad_refine_transfer(
        _quad_basis(3, anti_alias='flux,surf-flux')
    )

    assert aa.key == plain.key
    assert np.array_equal(aa.interp, plain.interp)


@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize(
    'soln_pts', ['gauss-legendre', 'gauss-legendre-lobatto']
)
def test_quad_refine_preserves_constants(order, soln_pts):
    basis = _quad_basis(order, soln_pts)
    transfer = build_quad_refine_transfer(basis)

    assert transfer.key == QuadRefineTransferKey('quad', order, soln_pts)
    assert transfer.order == order
    assert not transfer.interp.flags.writeable
    assert np.allclose(
        transfer.interp @ np.ones(basis.nupts), 1,
        rtol=0, atol=1024*np.finfo(float).eps,
    )


@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize(
    'soln_pts', ['gauss-legendre', 'gauss-legendre-lobatto']
)
def test_quad_refine_reproduces_complete_tensor_polynomial_space(
    order, soln_pts
):
    basis = _quad_basis(order, soln_pts)
    transfer = build_quad_refine_transfer(basis)

    for powers in it.product(range(order + 1), repeat=2):
        parent = _monomial(basis.upts, powers)
        for quadrant, op in enumerate(transfer.interp):
            child_pts = quad_child_to_parent(quadrant, basis.upts)
            expect = _monomial(child_pts, powers)
            actual = op @ parent
            assert np.allclose(
                actual, expect, rtol=0, atol=_scaled_tol(actual, expect)
            )


@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize(
    'soln_pts', ['gauss-legendre', 'gauss-legendre-lobatto']
)
def test_quad_refine_random_representable_polynomial(order, soln_pts):
    basis = _quad_basis(order, soln_pts)
    transfer = build_quad_refine_transfer(basis)
    rng = np.random.default_rng(654321 + order)
    powers = list(it.product(range(order + 1), repeat=2))
    coeffs = rng.standard_normal(len(powers))

    def poly(pts):
        return sum(c*_monomial(pts, p) for c, p in zip(coeffs, powers))

    parent = poly(basis.upts)
    for quadrant, op in enumerate(transfer.interp):
        expect = poly(quad_child_to_parent(quadrant, basis.upts))
        actual = op @ parent
        assert np.allclose(
            actual, expect, rtol=0,
            atol=_scaled_tol(actual, expect, factor=8192),
        )


@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize(
    'soln_pts', ['gauss-legendre', 'gauss-legendre-lobatto']
)
def test_quad_refine_reference_integral_is_conservative(order, soln_pts):
    basis = _quad_basis(order, soln_pts)
    transfer = build_quad_refine_transfer(basis)
    iwts = _quad_integral_weights(basis)

    children = sum(iwts @ op for op in transfer.interp)/4
    assert np.allclose(
        children, iwts, rtol=0, atol=_scaled_tol(children, iwts)
    )


@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize(
    'soln_pts', ['gauss-legendre', 'gauss-legendre-lobatto']
)
def test_quad_refine_affine_physical_integral_is_conservative(
    order, soln_pts
):
    basis = _quad_basis(order, soln_pts)
    transfer = build_quad_refine_transfer(basis)
    iwts = _quad_integral_weights(basis)
    rng = np.random.default_rng(13579 + order)
    parent = rng.standard_normal((basis.nupts, 3, 4))
    child = apply_quad_refine_transfer(transfer, parent)

    parent_jac = 0.731
    pint = parent_jac*np.einsum('i,ivn->vn', iwts, parent)
    cint = parent_jac*np.einsum('i,qivn->vn', iwts, child)/4
    assert np.allclose(
        cint, pint, rtol=0, atol=_scaled_tol(cint, pint, factor=8192)
    )


@pytest.mark.parametrize('order', [2, 3, 4])
@pytest.mark.parametrize(
    'soln_pts', ['gauss-legendre', 'gauss-legendre-lobatto']
)
def test_quad_refine_conserves_batched_compressible_state(order, soln_pts):
    basis = _quad_basis(order, soln_pts)
    transfer = build_quad_refine_transfer(basis)
    x, y = basis.upts.T

    rho = 1.2 + 0.02*x + 0.01*y
    u = 0.15 + 0.01*y
    v = -0.04 - 0.005*x
    p = 1.0 + 0.01*x - 0.005*y
    gamma = 1.4

    state = np.stack([
        rho, rho*u, rho*v,
        p/(gamma - 1) + 0.5*rho*(u*u + v*v),
    ], axis=1)
    parent = np.stack([state, 1.01*state], axis=2)
    child = apply_quad_refine_transfer(transfer, parent)

    assert child.shape == (4, basis.nupts, 4, 2)
    assert np.isfinite(child).all()

    iwts = _quad_integral_weights(basis)
    pint = np.einsum('i,ivn->vn', iwts, parent)
    cint = np.einsum('i,qivn->vn', iwts, child)/4
    assert np.allclose(
        cint, pint, rtol=0, atol=_scaled_tol(cint, pint, factor=8192)
    )

    crho, crhou, crhov, cE = np.moveaxis(child, 2, 0)
    cp = (gamma - 1)*(
        cE - 0.5*(crhou*crhou + crhov*crhov)/crho
    )
    assert np.all(crho > 0)
    assert np.all(cp > 0)


def test_quad_refine_batched_layout_matches_operator_application():
    basis = _quad_basis(3)
    transfer = build_quad_refine_transfer(basis)
    rng = np.random.default_rng(271828)
    parent = rng.standard_normal((basis.nupts, 4, 4))

    child = apply_quad_refine_transfer(transfer, parent)
    for quadrant, op in enumerate(transfer.interp):
        for pidx in range(parent.shape[2]):
            expect = op @ parent[:, :, pidx]
            actual = child[quadrant, :, :, pidx]
            assert np.allclose(
                actual, expect, rtol=0, atol=_scaled_tol(actual, expect)
            )


def test_quad_refine_batched_result_is_deterministic():
    basis = _quad_basis(3)
    transfer = build_quad_refine_transfer(basis)
    rng = np.random.default_rng(161803)
    parent = rng.standard_normal((basis.nupts, 4, 7))

    a = apply_quad_refine_transfer(transfer, parent)
    b = apply_quad_refine_transfer(transfer, parent)
    assert np.array_equal(a, b)


def test_quad_refine_rejects_nonquad_and_bad_bank_shape():
    with pytest.raises(ValueError, match='Quad basis'):
        build_quad_refine_transfer(_basis(2))

    basis = _quad_basis(2)
    transfer = build_quad_refine_transfer(basis)
    with pytest.raises(ValueError, match='shape'):
        apply_quad_refine_transfer(transfer, np.zeros((basis.nupts, 4)))
    with pytest.raises(ValueError, match='point count'):
        apply_quad_refine_transfer(
            transfer, np.zeros((basis.nupts + 1, 4, 1))
        )


def test_quad_refine_transfer_reuses_immutable_operator_by_key():
    a = build_quad_refine_transfer(_quad_basis(3))
    b = build_quad_refine_transfer(_quad_basis(3))
    aa = build_quad_refine_transfer(
        _quad_basis(3, anti_alias='flux,surf-flux')
    )
    gll = build_quad_refine_transfer(
        _quad_basis(3, 'gauss-legendre-lobatto')
    )

    assert b is a
    assert aa is a
    assert gll is not a
    assert not a.interp.flags.writeable


@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize(
    'soln_pts', ['gauss-legendre', 'gauss-legendre-lobatto']
)
def test_quad_refine_depth_two_preserves_modes_and_integral(order, soln_pts):
    basis = _quad_basis(order, soln_pts)
    transfer = build_quad_refine_transfer(basis)
    iwts = _quad_integral_weights(basis)
    rng = np.random.default_rng(424242 + order)
    powers = list(it.product(range(order + 1), repeat=2))
    coeffs = rng.standard_normal(len(powers))

    def poly(pts):
        return sum(c*_monomial(pts, p) for c, p in zip(coeffs, powers))

    parent = poly(basis.upts)[:, None, None]
    level1 = apply_quad_refine_transfer(transfer, parent)
    level2 = np.stack([
        apply_quad_refine_transfer(transfer, level1[q])
        for q in range(4)
    ])

    for q1 in range(4):
        for q2 in range(4):
            final_pts = quad_child_to_parent(
                q1, quad_child_to_parent(q2, basis.upts)
            )
            actual = level2[q1, q2, :, 0, 0]
            expect = poly(final_pts)
            assert np.allclose(
                actual, expect, rtol=0,
                atol=_scaled_tol(actual, expect, factor=16384),
            )

    pint = iwts @ parent[:, 0, 0]
    cint = np.einsum('i,abivn->vn', iwts, level2)[0, 0]/16
    assert np.allclose(
        cint, pint, rtol=0, atol=_scaled_tol(cint, pint, factor=16384)
    )


def test_hex_child_to_parent_matches_d2_octants():
    corners = HexShape.std_ele(1)

    for octant in range(8):
        mapped = hex_child_to_parent(octant, corners)
        bits = (
            octant & 1,
            (octant >> 1) & 1,
            (octant >> 2) & 1,
        )

        for d, bit in enumerate(bits):
            expect = (0.0, 1.0) if bit else (-1.0, 0.0)
            assert np.allclose((mapped[:, d].min(), mapped[:, d].max()),
                               expect, rtol=0, atol=0)


def test_hex_child_to_parent_rejects_invalid_inputs():
    with pytest.raises(ValueError, match='octant'):
        hex_child_to_parent(8, np.zeros((1, 3)))
    with pytest.raises(ValueError, match='octant'):
        hex_child_to_parent(1.0, np.zeros((1, 3)))
    with pytest.raises(ValueError, match='shape'):
        hex_child_to_parent(0, np.zeros((3,)))


def test_hex_refine_transfer_key_distinguishes_solution_points():
    gl = build_hex_refine_transfer(_basis(3, 'gauss-legendre'))
    gll = build_hex_refine_transfer(_basis(3, 'gauss-legendre-lobatto'))

    assert gl.nupts == gll.nupts
    assert gl.key != gll.key
    assert gl.key.template == 'octree-2x2x2-v1'

    cache = {gl.key: gl}
    repeat_key = build_hex_refine_transfer(_basis(3)).key
    assert cache[repeat_key] is gl
    assert gll.key not in cache


def test_hex_refine_transfer_is_surface_aa_independent():
    plain = build_hex_refine_transfer(_basis(3))
    aa = build_hex_refine_transfer(_basis(3, anti_alias='flux,surf-flux'))

    assert aa.key == plain.key
    assert np.array_equal(aa.interp, plain.interp)


@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize(
    'soln_pts', ['gauss-legendre', 'gauss-legendre-lobatto']
)
def test_hex_refine_preserves_constants(order, soln_pts):
    basis = _basis(order, soln_pts)
    transfer = build_hex_refine_transfer(basis)

    assert transfer.key == HexRefineTransferKey('hex', order, soln_pts)
    assert transfer.order == order
    assert not transfer.interp.flags.writeable
    assert np.allclose(
        transfer.interp @ np.ones(basis.nupts), 1,
        rtol=0, atol=1024*np.finfo(float).eps,
    )


@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize(
    'soln_pts', ['gauss-legendre', 'gauss-legendre-lobatto']
)
def test_hex_refine_reproduces_complete_tensor_polynomial_space(
    order, soln_pts
):
    basis = _basis(order, soln_pts)
    transfer = build_hex_refine_transfer(basis)

    for powers in it.product(range(order + 1), repeat=3):
        parent = _monomial(basis.upts, powers)
        for octant, op in enumerate(transfer.interp):
            child_pts = hex_child_to_parent(octant, basis.upts)
            expect = _monomial(child_pts, powers)
            actual = op @ parent
            assert np.allclose(
                actual, expect, rtol=0, atol=_scaled_tol(actual, expect)
            )


@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize(
    'soln_pts', ['gauss-legendre', 'gauss-legendre-lobatto']
)
def test_hex_refine_random_representable_polynomial(order, soln_pts):
    basis = _basis(order, soln_pts)
    transfer = build_hex_refine_transfer(basis)
    rng = np.random.default_rng(123456 + order)
    powers = list(it.product(range(order + 1), repeat=3))
    coeffs = rng.standard_normal(len(powers))

    def poly(pts):
        return sum(c*_monomial(pts, p) for c, p in zip(coeffs, powers))

    parent = poly(basis.upts)
    for octant, op in enumerate(transfer.interp):
        expect = poly(hex_child_to_parent(octant, basis.upts))
        actual = op @ parent
        assert np.allclose(
            actual, expect, rtol=0,
            atol=_scaled_tol(actual, expect, factor=8192),
        )


@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize(
    'soln_pts', ['gauss-legendre', 'gauss-legendre-lobatto']
)
def test_hex_refine_reference_integral_is_conservative(order, soln_pts):
    basis = _basis(order, soln_pts)
    transfer = build_hex_refine_transfer(basis)
    iwts = _integral_weights(basis)

    children = sum(iwts @ op for op in transfer.interp)/8
    assert np.allclose(
        children, iwts, rtol=0, atol=_scaled_tol(children, iwts)
    )


@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize(
    'soln_pts', ['gauss-legendre', 'gauss-legendre-lobatto']
)
def test_hex_refine_affine_physical_integral_is_conservative(order, soln_pts):
    basis = _basis(order, soln_pts)
    transfer = build_hex_refine_transfer(basis)
    iwts = _integral_weights(basis)
    rng = np.random.default_rng(24680 + order)
    parent = rng.standard_normal((basis.nupts, 3, 4))
    child = apply_hex_refine_transfer(transfer, parent)

    # Any positive affine parent Jacobian scales all children by J/8.
    parent_jac = 0.731
    pint = parent_jac*np.einsum('i,ivn->vn', iwts, parent)
    cint = parent_jac*np.einsum('i,oivn->vn', iwts, child)/8
    assert np.allclose(
        cint, pint, rtol=0, atol=_scaled_tol(cint, pint, factor=8192)
    )


@pytest.mark.parametrize('order', [2, 3, 4])
@pytest.mark.parametrize(
    'soln_pts', ['gauss-legendre', 'gauss-legendre-lobatto']
)
def test_hex_refine_conserves_batched_compressible_state(order, soln_pts):
    basis = _basis(order, soln_pts)
    transfer = build_hex_refine_transfer(basis)
    x, y, z = basis.upts.T

    rho = 1.2 + 0.02*x + 0.01*y - 0.015*z
    u = 0.15 + 0.01*y
    v = -0.04 + 0.01*z
    w = 0.03 - 0.005*x
    p = 1.0 + 0.01*x - 0.005*y
    gamma = 1.4

    state = np.stack([
        rho, rho*u, rho*v, rho*w,
        p/(gamma - 1) + 0.5*rho*(u*u + v*v + w*w),
    ], axis=1)
    parent = np.stack([state, 1.01*state], axis=2)
    child = apply_hex_refine_transfer(transfer, parent)

    assert child.shape == (8, basis.nupts, 5, 2)
    assert np.isfinite(child).all()

    iwts = _integral_weights(basis)
    pint = np.einsum('i,ivn->vn', iwts, parent)
    cint = np.einsum('i,oivn->vn', iwts, child)/8
    assert np.allclose(
        cint, pint, rtol=0, atol=_scaled_tol(cint, pint, factor=8192)
    )

    crho, crhou, crhov, crhow, cE = np.moveaxis(child, 2, 0)
    cp = (gamma - 1)*(
        cE - 0.5*(crhou*crhou + crhov*crhov + crhow*crhow)/crho
    )
    assert np.all(crho > 0)
    assert np.all(cp > 0)


def test_hex_refine_batched_layout_matches_operator_application():
    basis = _basis(3)
    transfer = build_hex_refine_transfer(basis)
    rng = np.random.default_rng(314159)
    parent = rng.standard_normal((basis.nupts, 5, 4))

    child = apply_hex_refine_transfer(transfer, parent)
    for octant, op in enumerate(transfer.interp):
        for pidx in range(parent.shape[2]):
            expect = op @ parent[:, :, pidx]
            assert np.array_equal(child[octant, :, :, pidx], expect)


def test_hex_refine_batched_result_is_deterministic():
    basis = _basis(3)
    transfer = build_hex_refine_transfer(basis)
    rng = np.random.default_rng(98765)
    parent = rng.standard_normal((basis.nupts, 5, 7))

    a = apply_hex_refine_transfer(transfer, parent)
    b = apply_hex_refine_transfer(transfer, parent)
    assert np.array_equal(a, b)


def test_hex_refine_rejects_nonhex_and_bad_bank_shape():
    cfg = Inifile('''\
[solver]
order = 2
[solver-elements-quad]
soln-pts = gauss-legendre
''')
    with pytest.raises(ValueError, match='Hex basis'):
        build_hex_refine_transfer(QuadShape(None, cfg))

    basis = _basis(2)
    transfer = build_hex_refine_transfer(basis)
    with pytest.raises(ValueError, match='shape'):
        apply_hex_refine_transfer(transfer, np.zeros((basis.nupts, 5)))
    with pytest.raises(ValueError, match='point count'):
        apply_hex_refine_transfer(transfer, np.zeros((basis.nupts + 1, 5, 1)))


def test_hex_refine_transfer_reuses_immutable_operator_by_key():
    a = build_hex_refine_transfer(_basis(3))
    b = build_hex_refine_transfer(_basis(3))
    aa = build_hex_refine_transfer(_basis(3, anti_alias='flux,surf-flux'))
    gll = build_hex_refine_transfer(_basis(3, 'gauss-legendre-lobatto'))

    assert b is a
    assert aa is a
    assert gll is not a
    assert not a.interp.flags.writeable
