from types import SimpleNamespace

import numpy as np
import pytest

from pyfr.inifile import Inifile
from pyfr.shapes import HexShape
from pyfr.solvers.baseadvecdiff.elements import BaseAdvectionDiffusionElements


def _hex_shape(points):
    cfg = Inifile()
    cfg.set('solver', 'order', '2')
    cfg.set('solver', 'anti-alias', 'none')
    cfg.set('solver-elements-hex', 'soln-pts', points)
    cfg.set('solver-interfaces-quad', 'flux-pts', points)
    return HexShape(8, cfg)


def _fake_elements(basis):
    return SimpleNamespace(
        basis=basis,
        nfacefpts=basis.nfacefpts,
        nfpts=basis.nfpts,
        nupts=basis.nupts,
        _vect_fpts=SimpleNamespace(mid=11),
        _grad_upts=SimpleNamespace(mid=17),
    )


@pytest.mark.parametrize('fidx', range(6))
def test_mortar_gradient_map_gl_local_face_order(fidx):
    basis = _hex_shape('gauss-legendre')
    eles = _fake_elements(basis)
    eidxs = np.array([2, 5], dtype=np.int64)

    mid, rmap, cmap, lda = (
        BaseAdvectionDiffusionElements._get_vect_fpts_for_mortars(
            eles, eidxs, fidx
        )
    )
    expected = np.tile(basis.facefpts[fidx], len(eidxs))

    assert np.all(mid == 11)
    assert np.array_equal(rmap, expected)
    assert np.array_equal(cmap, np.repeat(eidxs, len(expected)//len(eidxs)))
    assert np.all(lda == basis.nfpts)


@pytest.mark.parametrize('fidx', range(6))
def test_mortar_gradient_map_gll_fusion_local_face_order(fidx):
    basis = _hex_shape('gauss-legendre-lobatto')
    eles = _fake_elements(basis)
    eidxs = np.array([1, 4], dtype=np.int64)

    mid, rmap, cmap, lda = (
        BaseAdvectionDiffusionElements._get_grad_upts_for_mortars(
            eles, eidxs, fidx
        )
    )
    face = basis.facefpts[fidx]
    expected = np.tile(basis.fpts_map_upts[face], len(eidxs))

    assert basis.fpts_in_upts
    assert np.all(mid == 17)
    assert np.array_equal(rmap, expected)
    assert np.array_equal(cmap, np.repeat(eidxs, len(expected)//len(eidxs)))
    assert np.all(lda == basis.nupts)
