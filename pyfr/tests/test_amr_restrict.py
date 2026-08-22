import itertools as it

import numpy as np
import pytest

from pyfr.amr import (
    apply_hex_refine_transfer, apply_hex_restrict_transfer,
    build_hex_refine_transfer, build_hex_restrict_transfer,
    hex_restriction_projection_loss,
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


@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize(
    'soln_pts', ['gauss-legendre', 'gauss-legendre-lobatto']
)
def test_hex_restrict_reproduces_identity(order, soln_pts):
    rt = build_hex_restrict_transfer(_basis(order, soln_pts))
    ident = sum(rt.restrict[o] @ rt.refine.interp[o] for o in range(8))
    assert np.allclose(
        ident, np.eye(rt.nupts), rtol=0, atol=_scaled_tol(ident, factor=8192)
    )


@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize(
    'soln_pts', ['gauss-legendre', 'gauss-legendre-lobatto']
)
def test_hex_restrict_preserves_constants(order, soln_pts):
    rt = build_hex_restrict_transfer(_basis(order, soln_pts))
    total = sum(rt.restrict[o] @ np.ones(rt.nupts) for o in range(8))
    assert np.allclose(
        total, 1, rtol=0, atol=_scaled_tol(total, factor=8192)
    )


@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize(
    'soln_pts', ['gauss-legendre', 'gauss-legendre-lobatto']
)
def test_hex_refine_restrict_reproduces_complete_tensor_polynomial_space(
    order, soln_pts
):
    basis = _basis(order, soln_pts)
    rt = build_hex_restrict_transfer(basis)

    for powers in it.product(range(order + 1), repeat=3):
        parent = _monomial(basis.upts, powers)[:, None, None]
        child = apply_hex_refine_transfer(rt.refine, parent)
        recovered = apply_hex_restrict_transfer(rt, child)
        assert np.allclose(
            recovered[:, 0, 0], parent[:, 0, 0], rtol=0,
            atol=_scaled_tol(recovered, parent, factor=8192),
        )


def test_hex_restrict_batched_round_trip_matches_direct_application():
    basis = _basis(3)
    rt = build_hex_restrict_transfer(basis)
    rng = np.random.default_rng(20260810)
    parent = rng.standard_normal((rt.nupts, 5, 6))
    child = apply_hex_refine_transfer(rt.refine, parent)
    recovered = apply_hex_restrict_transfer(rt, child)

    assert np.allclose(
        recovered, parent, rtol=0, atol=_scaled_tol(recovered, parent)
    )

    for n in range(parent.shape[2]):
        direct = sum(rt.restrict[o] @ child[o, :, :, n] for o in range(8))
        assert np.allclose(
            direct, recovered[:, :, n], rtol=0,
            atol=_scaled_tol(direct, recovered),
        )


def test_hex_restrict_arbitrary_child_conserves_physical_integral():
    basis = _basis(3)
    rt = build_hex_restrict_transfer(basis)
    iwts = _integral_weights(basis)
    rng = np.random.default_rng(998877)
    arbitrary = rng.standard_normal((8, rt.nupts, 5, 4))
    projected = apply_hex_restrict_transfer(rt, arbitrary)

    pint = np.einsum('i,ivn->vn', iwts, projected)
    cint = np.einsum('i,oivn->vn', iwts, arbitrary)/8
    assert np.allclose(
        pint, cint, rtol=0, atol=_scaled_tol(pint, cint, factor=8192)
    )


def test_hex_restrict_residual_is_l2_orthogonal_to_parent_space():
    basis = _basis(3)
    rt = build_hex_restrict_transfer(basis)
    rng = np.random.default_rng(998877)
    arbitrary = rng.standard_normal((8, rt.nupts, 5, 4))
    projected = apply_hex_restrict_transfer(rt, arbitrary)

    # For any parent-space test vector v, phi_o = T_o @ v is the octant-o
    # restriction of the corresponding parent function. Orthogonality of
    # the residual to the entire parent Q_p space is equivalent to the
    # vector identity sum_o (1/8) T_o^T M r_o == 0, since R_o is defined
    # as (1/8) M^-1 T_o^T M.
    residual = arbitrary - apply_hex_refine_transfer(rt.refine, projected)
    orth = np.einsum(
        'oji,jk,okvn->ivn', rt.refine.interp, rt.mass, residual,
        optimize=True,
    )/8
    scale = max(1.0, float(np.max(np.abs(arbitrary))))
    assert np.max(np.abs(orth)) < 1e-8*scale


def test_hex_restriction_projection_loss_is_roundoff_zero_for_d3_children():
    basis = _basis(3)
    rt = build_hex_restrict_transfer(basis)
    rng = np.random.default_rng(20260810)
    parent = rng.standard_normal((rt.nupts, 5, 6))
    child = apply_hex_refine_transfer(rt.refine, parent)
    recovered = apply_hex_restrict_transfer(rt, child)

    loss = hex_restriction_projection_loss(rt, child, recovered)
    assert np.max(loss) < 1e-20


def test_hex_restriction_projection_loss_is_positive_for_incompatible_child():
    basis = _basis(3)
    rt = build_hex_restrict_transfer(basis)
    rng = np.random.default_rng(998877)
    arbitrary = rng.standard_normal((8, rt.nupts, 5, 4))
    projected = apply_hex_restrict_transfer(rt, arbitrary)

    loss = hex_restriction_projection_loss(rt, arbitrary, projected)
    assert np.min(loss) > 1e-6


@pytest.mark.parametrize('order', [2, 3, 4])
@pytest.mark.parametrize(
    'soln_pts', ['gauss-legendre', 'gauss-legendre-lobatto']
)
def test_hex_restrict_smooth_compressible_round_trip_positive_eos(
    order, soln_pts
):
    basis = _basis(order, soln_pts)
    rt = build_hex_restrict_transfer(basis)
    x, y, z = basis.upts.T
    gamma = 1.4

    rho = 1.2 + 0.02*x + 0.01*y - 0.015*z
    u = 0.15 + 0.01*y
    v = -0.04 + 0.01*z
    w = 0.03 - 0.005*x
    p = 1.0 + 0.01*x - 0.005*y
    state = np.stack([
        rho, rho*u, rho*v, rho*w,
        p/(gamma - 1) + 0.5*rho*(u*u + v*v + w*w),
    ], axis=1)[:, :, None]

    child = apply_hex_refine_transfer(rt.refine, state)
    back = apply_hex_restrict_transfer(rt, child)

    iwts = _integral_weights(basis)
    pint = np.einsum('i,ivn->vn', iwts, state)
    cint = np.einsum('i,ivn->vn', iwts, back)
    assert np.allclose(
        cint, pint, rtol=0, atol=_scaled_tol(cint, pint, factor=8192)
    )

    rrho = back[:, 0, 0]
    rrhov = back[:, 1:4, 0]
    rE = back[:, 4, 0]
    rp = (gamma - 1)*(rE - 0.5*np.sum(rrhov*rrhov, axis=1)/rrho)
    assert np.all(rrho > 0)
    assert np.all(rp > 0)


def test_hex_restrict_frozen_p2_adversarial_fixture_negative_parent():
    """Positive child nodal density does NOT imply positive restricted
    parent density. Restriction must not silently clip. This mirrors the
    frozen reference-probe counterexample; the exact numeric values differ
    from the standalone probe because PyFR's own basis point ordering
    differs, but the qualitative counterexample is reproduced against the
    real PyFR basis.
    """
    basis = _basis(2, 'gauss-legendre')
    rt = build_hex_restrict_transfer(basis)
    rng = np.random.default_rng(1002)

    found = False
    for _ in range(100):
        child_rho = np.exp(rng.normal(0, 1.5, size=(8, rt.nupts)))
        parent_rho = apply_hex_restrict_transfer(
            rt, child_rho[:, :, None, None]
        )[:, 0, 0]
        assert np.all(child_rho > 0)
        if np.min(parent_rho) < 0:
            found = True
            break

    assert found, 'expected a positive-child/negative-parent counterexample'


def test_hex_restrict_deterministic_and_read_only():
    a = build_hex_restrict_transfer(_basis(3))
    b = build_hex_restrict_transfer(_basis(3))
    assert np.array_equal(a.restrict, b.restrict)
    assert not a.restrict.flags.writeable
    assert not a.mass.flags.writeable


def test_hex_restrict_key_separation():
    gl = build_hex_restrict_transfer(_basis(3, 'gauss-legendre'))
    gll = build_hex_restrict_transfer(_basis(3, 'gauss-legendre-lobatto'))
    assert gl.key != gll.key
    cache = {gl.key: gl}
    assert gll.key not in cache


def test_hex_restrict_surface_aa_independent():
    plain = build_hex_restrict_transfer(_basis(3, anti_alias='none'))
    aa = build_hex_restrict_transfer(
        _basis(3, anti_alias='flux,surf-flux')
    )
    assert plain.key == aa.key
    assert np.array_equal(plain.restrict, aa.restrict)


def test_hex_restrict_rejects_wrong_child_shape():
    rt = build_hex_restrict_transfer(_basis(3))

    with pytest.raises(ValueError, match='8 octants'):
        apply_hex_restrict_transfer(rt, np.zeros((7, rt.nupts, 5, 1)))
    with pytest.raises(ValueError, match='point count'):
        apply_hex_restrict_transfer(rt, np.zeros((8, rt.nupts + 1, 5, 1)))
    with pytest.raises(ValueError, match='shape'):
        apply_hex_restrict_transfer(rt, np.zeros((8, rt.nupts, 5)))


def test_hex_restrict_rejects_nonhex_basis():
    cfg = Inifile('''\
[solver]
order = 2
[solver-elements-quad]
soln-pts = gauss-legendre
''')
    with pytest.raises(ValueError, match='Hex basis'):
        build_hex_restrict_transfer(QuadShape(None, cfg))


def test_hex_restrict_transfer_reuses_immutable_operator_by_key():
    a = build_hex_restrict_transfer(_basis(3))
    b = build_hex_restrict_transfer(_basis(3))
    aa = build_hex_restrict_transfer(
        _basis(3, anti_alias='flux,surf-flux')
    )
    gll = build_hex_restrict_transfer(
        _basis(3, 'gauss-legendre-lobatto')
    )

    assert b is a
    assert aa is a
    assert gll is not a
    assert a.refine is build_hex_refine_transfer(_basis(3))
    assert not a.restrict.flags.writeable
    assert not a.mass.flags.writeable
