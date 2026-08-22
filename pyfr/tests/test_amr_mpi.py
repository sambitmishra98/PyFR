from io import StringIO

import h5py
import numpy as np
import pytest

from pyfr.amr import encode_hex_leaf_tree
from pyfr.amrmpi import (
    MPIAMRTransactionError, _normalise_destination_parts,
    _close_coarsening, _close_refinements, _coarsened_parents,
    _inherited_mixed_ownership, _physical_hex_volume,
    _produce_local_refined_records,
    _serial_mixed_root_mesh, _serial_root_mesh,
    _typed_ownership_from_partitioning,
    _validate_mixed_mortar_affinity, _validate_mortar_affinity,
)
from pyfr.amrmesh import (
    materialize_native_hex_tree, materialize_native_mixed_hex_tree,
)
from pyfr.amrtransaction import _close_mixed_hex_refinements
from pyfr.amrpolicy import balanced_affinity_destination_parts
from pyfr.inifile import Inifile
from pyfr.partitioners.baseline import BaselinePartitioner
from pyfr.progress import NullProgressSequence
from pyfr.readers.gmsh import GmshReader
from pyfr.shapes import HexShape
from pyfr.amrwriter import write_adapted_mixed_hex_mesh


def _two_hex_gmsh():
    from pyfr.tests.test_amr_transaction import _two_hex_gmsh
    return _two_hex_gmsh()


def _mixed_hex_gmsh():
    from pyfr.tests.test_amr_online_mixed_hex import _mixed_hex_gmsh
    return _mixed_hex_gmsh()


def _cfg():
    return Inifile('''
[solver]
system = euler
order = 2
[solver-elements-hex]
soln-pts = gauss-legendre
''')


def test_serial_root_mesh_matches_native_root(tmp_path):
    path = tmp_path / 'root.pyfrm'
    GmshReader(StringIO(_two_hex_gmsh()), NullProgressSequence()).write(
        str(path), 1e-5
    )
    mesh = _serial_root_mesh(str(path))
    assert mesh.etypes == ['hex']
    assert np.array_equal(mesh.eidxs['hex'], [0, 1])
    assert len(mesh.con[0]) == len(mesh.con[1]) == 1
    assert sum(len(c) for c in mesh.bcon.values()) == 10
    assert not mesh.con_p and not mesh.mcon

    tree = encode_hex_leaf_tree(mesh.uuid, [(0, ()), (1, ())])
    raw = materialize_native_hex_tree(mesh, tree)
    assert tuple(raw.leaf_order) == tuple(tree.leaves())
    assert len(raw.mortars) == 0


def test_serial_mixed_root_and_inherited_ownership(tmp_path):
    path = tmp_path / 'mixed-root.pyfrm'
    GmshReader(StringIO(_mixed_hex_gmsh()), NullProgressSequence()).write(
        str(path), 1e-5
    )
    mesh = _serial_mixed_root_mesh(str(path))
    assert set(mesh.etypes) == {'hex', 'pyr', 'tet'}
    assert {e: len(mesh.eidxs[e]) for e in mesh.etypes} == {
        'hex': 2, 'pyr': 1, 'tet': 1
    }

    old = encode_hex_leaf_tree(mesh.uuid, [(0, ()), (1, ())])
    proposed, _ = _close_mixed_hex_refinements(mesh, old, [(0, ())])
    raw = materialize_native_mixed_hex_tree(mesh, proposed)
    old_parts = {
        'hex': {(0, ()): 1, (1, ()): 1},
        'pyr': {0: 0}, 'tet': {0: 0},
    }
    owner = _inherited_mixed_ownership(
        mesh, old, proposed, old_parts, 2
    )
    _validate_mixed_mortar_affinity(raw, owner)
    assert set(owner['hex']) == {1}
    assert owner['pyr'].tolist() == [0]
    assert owner['tet'].tolist() == [0]



def test_serial_mixed_root_accepts_curved_geometry_flags(tmp_path):
    path = tmp_path / 'mixed-curved-root.pyfrm'
    GmshReader(StringIO(_mixed_hex_gmsh()), NullProgressSequence()).write(
        str(path), 1e-5
    )
    with h5py.File(path, 'r+') as f:
        data = f['eles/hex'][()]
        data['curved'][0] = True
        del f['eles/hex']
        f['eles'].create_dataset('hex', data=data)

    mesh = _serial_mixed_root_mesh(str(path))
    assert mesh.spts_curved['hex'].tolist() == [True, False]

    old = encode_hex_leaf_tree(mesh.uuid, [(0, ()), (1, ())])
    proposed, _ = _close_mixed_hex_refinements(mesh, old, [(0, ())])
    raw = materialize_native_mixed_hex_tree(mesh, proposed)
    assert np.count_nonzero(raw.hex_curved) == 8


def test_physical_hex_volume_integrates_nonaffine_root_map():
    cfg = _cfg()
    basis = HexShape(None, cfg)
    pts = np.asarray(HexShape.std_ele(2), dtype=float)
    coords = np.array(pts, copy=True)
    coords[:, 0] += 0.1*pts[:, 0]*pts[:, 1]
    spts = coords[:, None, :]

    assert np.isclose(_physical_hex_volume(spts, basis), 8.0,
                      rtol=0, atol=1e-13)

def test_mixed_mortar_split_fails_closed(tmp_path):
    path = tmp_path / 'mixed-root.pyfrm'
    GmshReader(StringIO(_mixed_hex_gmsh()), NullProgressSequence()).write(
        str(path), 1e-5
    )
    mesh = _serial_mixed_root_mesh(str(path))
    old = encode_hex_leaf_tree(mesh.uuid, [(0, ()), (1, ())])
    proposed, _ = _close_mixed_hex_refinements(mesh, old, [(1, ())])
    raw = materialize_native_mixed_hex_tree(mesh, proposed)
    old_parts = {
        'hex': {(0, ()): 1, (1, ()): 1},
        'pyr': {0: 0}, 'tet': {0: 0},
    }
    owner = _inherited_mixed_ownership(
        mesh, old, proposed, old_parts, 2
    )
    with pytest.raises(MPIAMRTransactionError, match='splits mortar'):
        _validate_mixed_mortar_affinity(raw, owner)




def test_mixed_baseline_repartition_is_deterministic_and_mortar_local(tmp_path):
    path = tmp_path / 'mixed-root.pyfrm'
    GmshReader(StringIO(_mixed_hex_gmsh()), NullProgressSequence()).write(
        str(path), 1e-5
    )
    mesh = _serial_mixed_root_mesh(str(path))
    old = encode_hex_leaf_tree(mesh.uuid, [(0, ()), (1, ())])
    proposed, _ = _close_mixed_hex_refinements(mesh, old, [(0, ())])
    raw = materialize_native_mixed_hex_tree(mesh, proposed)
    adapted = tmp_path / 'adapted.pyfrm'
    write_adapted_mixed_hex_mesh(raw, str(adapted))

    with h5py.File(adapted, 'r+') as f:
        part = BaselinePartitioner([1, 1], elewts='balanced')
        pinfo = part.partition(f)
        again = part.partition(f)
        owner = _typed_ownership_from_partitioning(f, pinfo, 2)
        owner_again = _typed_ownership_from_partitioning(f, again, 2)

    assert all(np.array_equal(owner[e], owner_again[e]) for e in owner)
    _validate_mixed_mortar_affinity(raw, owner)

    # Every active Hex descended from the refined root remains one atomic
    # ownership family.  Partition labels themselves are intentionally not
    # part of the contract.
    hex_rank = int(owner['hex'][0])
    assert np.all(owner['hex'] == hex_rank)
    assert {int(r) for parts in owner.values() for r in parts} == {0, 1}


def test_destination_parts_and_mortar_affinity(tmp_path):
    path = tmp_path / 'root.pyfrm'
    GmshReader(StringIO(_two_hex_gmsh()), NullProgressSequence()).write(
        str(path), 1e-5
    )
    mesh = _serial_root_mesh(str(path))
    tree = encode_hex_leaf_tree(
        mesh.uuid, [(0, (o,)) for o in range(8)] + [(1, ())]
    )
    raw = materialize_native_hex_tree(mesh, tree)

    parts = {leaf: 0 for leaf in tree.leaves()}
    parts[(1, ())] = 1
    _, vparts = _normalise_destination_parts(parts, tree, 2)
    with pytest.raises(MPIAMRTransactionError, match='splits mortar'):
        _validate_mortar_affinity(raw, vparts)

    parts = {leaf: 0 for leaf in tree.leaves()}
    with pytest.raises(MPIAMRTransactionError, match='every MPI rank'):
        _normalise_destination_parts(parts, tree, 2)



def test_balanced_affinity_policy_is_deterministic_and_mortar_atomic(tmp_path):
    path = tmp_path / 'root.pyfrm'
    GmshReader(StringIO(_two_hex_gmsh()), NullProgressSequence()).write(
        str(path), 1e-5
    )
    mesh = _serial_root_mesh(str(path))
    old = encode_hex_leaf_tree(mesh.uuid, [(0, ()), (1, ())])
    proposed = encode_hex_leaf_tree(
        mesh.uuid, [(0, (o,)) for o in range(8)] + [(1, ())]
    )
    raw = materialize_native_hex_tree(mesh, proposed)
    old_parts = {(0, ()): 0, (1, ()): 1}

    parts = balanced_affinity_destination_parts(
        old, proposed, raw, old_parts, 2
    )
    again = balanced_affinity_destination_parts(
        old, proposed, raw, old_parts, 2
    )

    assert parts == again
    assert [sum(r == i for r in parts.values()) for i in range(2)] == [5, 4]
    leaves = tuple(proposed.leaves())
    for mortar in raw.mortars:
        eidxs = (mortar.left_eidx, *mortar.right_eidx)
        assert len({parts[leaves[int(i)]] for i in eidxs}) == 1
    assert any(parts[leaf] != old_parts.get(leaf, 0)
               for leaf in proposed.leaves())

def test_balanced_affinity_policy_rejects_incomplete_old_ownership(tmp_path):
    path = tmp_path / 'root.pyfrm'
    GmshReader(StringIO(_two_hex_gmsh()), NullProgressSequence()).write(
        str(path), 1e-5
    )
    mesh = _serial_root_mesh(str(path))
    old = encode_hex_leaf_tree(mesh.uuid, [(0, ()), (1, ())])
    proposed = encode_hex_leaf_tree(
        mesh.uuid, [(0, (o,)) for o in range(8)] + [(1, ())]
    )
    raw = materialize_native_hex_tree(mesh, proposed)

    with pytest.raises(ValueError, match='cover the old tree exactly'):
        balanced_affinity_destination_parts(
            old, proposed, raw, {(0, ()): 0}, 2
        )


def test_local_refine_records_are_keyed_by_proposed_leaf_ordinal():
    cfg = _cfg()
    basis = HexShape(None, cfg)
    old = encode_hex_leaf_tree('root', [(0, ()), (1, ())])
    proposed = encode_hex_leaf_tree(
        'root', [(0, (o,)) for o in range(8)] + [(1, ())]
    )
    state = np.empty((basis.nupts, 3, 1))
    state[:, 0, 0] = 1
    state[:, 1, 0] = basis.upts[:, 0]
    state[:, 2, 0] = basis.upts[:, 1]

    class System:
        pass
    system = System(); system.cfg = cfg
    ords, values = _produce_local_refined_records(
        system, old, proposed, state, {(0, ()): 0}
    )
    assert np.array_equal(ords, np.arange(8))
    assert values.shape == (8, basis.nupts, 3)
    assert np.isfinite(values).all()


def test_global_closure_splits_coarse_neighbour(tmp_path):
    path = tmp_path / 'root.pyfrm'
    GmshReader(StringIO(_two_hex_gmsh()), NullProgressSequence()).write(
        str(path), 1e-5
    )
    mesh = _serial_root_mesh(str(path))
    old = encode_hex_leaf_tree(
        mesh.uuid, [(0, (o,)) for o in range(8)] + [(1, ())]
    )
    proposed = _close_refinements(mesh, old, [(0, (1,))])
    leaves = set(proposed.leaves())

    assert proposed.nleaves == 23
    assert (1, ()) not in leaves
    assert {(1, (o,)) for o in range(8)} <= leaves
    assert {(0, (1, o)) for o in range(8)} <= leaves


def test_global_coarsening_reuses_d4_legality(tmp_path):
    path = tmp_path / 'root.pyfrm'
    GmshReader(StringIO(_two_hex_gmsh()), NullProgressSequence()).write(
        str(path), 1e-5
    )
    mesh = _serial_root_mesh(str(path))
    level1 = encode_hex_leaf_tree(
        mesh.uuid, [(0, (o,)) for o in range(8)] + [(1, ())]
    )
    old = _close_refinements(mesh, level1, [(0, (1,))])
    marks = {(0, (1, o)) for o in range(8)}
    proposed = _close_coarsening(mesh, old, marks)
    leaves = set(proposed.leaves())

    assert proposed.nleaves == 16
    assert (0, (1,)) in leaves
    assert not (marks & leaves)
    assert {(1, (o,)) for o in range(8)} <= leaves
    assert _coarsened_parents(old, proposed) == ((0, (1,)),)


def test_global_coarsening_accepts_multiple_complete_families(tmp_path):
    path = tmp_path / 'root.pyfrm'
    GmshReader(StringIO(_two_hex_gmsh()), NullProgressSequence()).write(
        str(path), 1e-5
    )
    mesh = _serial_root_mesh(str(path))
    old = encode_hex_leaf_tree(
        mesh.uuid,
        [(0, (o,)) for o in range(8)] +
        [(1, (o,)) for o in range(8)],
    )
    marks = set(old.leaves())
    proposed = _close_coarsening(mesh, old, marks)

    assert tuple(proposed.leaves()) == ((0, ()), (1, ()))
    assert _coarsened_parents(old, proposed) == ((0, ()), (1, ()))


def test_global_coarsening_rejects_incomplete_batch(tmp_path):
    path = tmp_path / 'root.pyfrm'
    GmshReader(StringIO(_two_hex_gmsh()), NullProgressSequence()).write(
        str(path), 1e-5
    )
    mesh = _serial_root_mesh(str(path))
    old = encode_hex_leaf_tree(
        mesh.uuid,
        [(0, (o,)) for o in range(8)] +
        [(1, (o,)) for o in range(8)],
    )
    marks = (
        {(0, (o,)) for o in range(8)} |
        {(1, (o,)) for o in range(7)} |
        {(1, ())}
    )

    with pytest.raises(MPIAMRTransactionError):
        _close_coarsening(mesh, old, marks)


def test_global_coarsening_rejects_21_violation(tmp_path):
    path = tmp_path / 'root.pyfrm'
    GmshReader(StringIO(_two_hex_gmsh()), NullProgressSequence()).write(
        str(path), 1e-5
    )
    mesh = _serial_root_mesh(str(path))
    level1 = encode_hex_leaf_tree(
        mesh.uuid, [(0, (o,)) for o in range(8)] + [(1, ())]
    )
    old = _close_refinements(mesh, level1, [(0, (1,))])
    marks = {(1, (o,)) for o in range(8)}

    with pytest.raises(MPIAMRTransactionError, match='legal sibling'):
        _close_coarsening(mesh, old, marks)
