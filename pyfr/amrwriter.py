"""V10 D5C - adapted native mesh persistence.

Writes an :class:`~pyfr.amrmesh.AdaptedRawMesh` (D5B2's proposed raw
mesh) as a standard native ``.pyfrm`` file, by reusing the accepted
:class:`~pyfr.readers.base.BaseReader` write path rather than inventing
a second HDF5 dialect. Element colouring is computed exactly once here
via the existing accepted
:meth:`~pyfr.readers.base.NodalMeshAssembler.compute_element_colouring`
- children inherit root tags, but never root colour (colour is a
property of the *adapted* adjacency graph, not of the pre-adaptation
element).

V10D5C2 addition: after the standard physical mesh is written, append
the D5A canonical leaf codec (``amr/version``, ``amr/root-mesh-uuid``,
``amr/leaves/root-eidx``, ``amr/leaves/path-offsets``,
``amr/leaves/path-data``) so the file alone - without the in-memory
``AdaptedRawMesh``/``HexLeafTree`` that produced it - carries persistent
octree ancestry. The arrays written are read directly off the
:class:`~pyfr.amr.HexLeafTree` re-derived from ``raw.leaf_order`` via
the existing accepted :func:`~pyfr.amr.encode_hex_leaf_tree`, not a
second encoding.

OFFLINE-2D 0054 adds a direct Quad sibling. Quad files use the same leaf-array
schema plus ``amr/template = quadtree-2x2-v1`` so NativeReader can distinguish
them from legacy accepted Hex ancestry, whose on-disk representation remains
unchanged and intentionally carries no template field.
"""
from uuid import UUID

import h5py
import numpy as np

from pyfr._version import __version__
from pyfr.amr import (
    AMR_LEAF_TREE_VERSION, encode_hex_leaf_tree, encode_quad_leaf_tree,
)
from pyfr.readers.base import BaseReader, MortarData, NodalMeshAssembler
from pyfr.readers.native import Connectivity, Mesh, MortarConnectivity
from pyfr.readers.shared_nodes import SharedNodesFinder
from pyfr.util import digest


class AdaptedHexReader(BaseReader):
    """A :class:`BaseReader` whose raw mesh is a pre-built
    :class:`~pyfr.amrmesh.AdaptedRawMesh` rather than a file on disk.
    Exists solely so :meth:`BaseReader.write` (UUID computation, default
    partitioning, HDF5 layout) is reused unchanged."""

    name = 'amr-adapted'

    def __init__(self, raw, progress=None):
        from pyfr.progress import NullProgressSequence
        super().__init__(progress or NullProgressSequence())
        self._raw = raw

    def _to_raw_mesh(self, lintol):
        raw = self._raw

        # Compact node ids into a dense 0..N-1 index space, in
        # deterministic (sorted-id) order.
        pos = {int(nid): i for i, nid in enumerate(raw.node_ids)}
        nodes = np.asarray(raw.node_locs, dtype=float)

        n = len(raw.leaf_order)
        codec = list(raw.codec)
        codec_index = {c: i for i, c in enumerate(codec)}

        has_mortars = len(raw.mortars) > 0
        if has_mortars and 'mortar/quad-2x2' not in codec_index:
            codec_index['mortar/quad-2x2'] = len(codec)
            codec.append('mortar/quad-2x2')
        mortar_cidx = codec_index.get('mortar/quad-2x2')

        fdtype = [('cidx', np.int16), ('off', np.int64)]
        edtype = [('nodes', np.int64, 8), ('curved', bool),
                  ('faces', fdtype, 6), ('colour', np.uint8),
                  ('tags', np.uint64)]

        einfo = np.zeros(n, dtype=edtype)
        for i, row in enumerate(raw.hex_nodes):
            einfo['nodes'][i] = [pos[int(nid)] for nid in row]
        einfo['curved'] = False
        einfo['tags'] = raw.leaf_tags

        cidx = raw.hex_faces_cidx.copy()
        off = raw.hex_faces_off.copy()
        # Resolve D5B2's mortar placeholder (-1) to the real codec index;
        # off == -2 already unambiguously marks a mortar face, cidx here
        # is a self-consistent placeholder D5B2 has no other use for.
        if mortar_cidx is not None:
            cidx[off == -2] = mortar_cidx

        einfo['faces']['cidx'] = cidx
        einfo['faces']['off'] = off

        eles = {'hex': einfo}

        # Compute element colouring fresh from the ADAPTED adjacency
        # graph - never copied from the pre-adaptation root.
        NodalMeshAssembler.compute_element_colouring(eles, codec)

        periodic = {}
        mortars = {}
        if raw.mortars:
            gdtype = [
                ('left_cidx', np.int16), ('left_eidx', np.int64),
                ('right_cidx', np.int16, 4), ('right_eidx', np.int64, 4),
            ]
            records = np.zeros(len(raw.mortars), dtype=gdtype)
            for i, m in enumerate(raw.mortars):
                records[i]['left_cidx'] = codec_index[
                    f'eles/hex/face/{m.left_fidx}'
                ]
                records[i]['left_eidx'] = m.left_eidx
                records[i]['right_cidx'] = [
                    codec_index[f'eles/hex/face/{f}'] for f in m.right_fidx
                ]
                records[i]['right_eidx'] = list(m.right_eidx)

            mortars['quad-2x2'] = MortarData(
                records, format='one-to-many-v1', template='quad-2x2'
            )

        return nodes, eles, codec, periodic, mortars


class AdaptedQuadReader(BaseReader):
    """A :class:`BaseReader` for a pre-built adapted Quad raw mesh."""

    name = 'amr-adapted-quad'

    def __init__(self, raw, progress=None):
        from pyfr.progress import NullProgressSequence
        super().__init__(progress or NullProgressSequence())
        self._raw = raw

    def _to_raw_mesh(self, lintol):
        raw = self._raw

        # Compact node ids into deterministic dense native indices.
        pos = {int(nid): i for i, nid in enumerate(raw.node_ids)}
        nodes = np.asarray(raw.node_locs, dtype=float)

        n = len(raw.leaf_order)
        codec = list(raw.codec)
        codec_index = {c: i for i, c in enumerate(codec)}

        has_mortars = len(raw.mortars) > 0
        if has_mortars and 'mortar/general-v1' not in codec_index:
            codec_index['mortar/general-v1'] = len(codec)
            codec.append('mortar/general-v1')
        mortar_cidx = codec_index.get('mortar/general-v1')

        fdtype = [('cidx', np.int16), ('off', np.int64)]
        edtype = [
            ('nodes', np.int64, 4), ('curved', bool),
            ('faces', fdtype, 4), ('colour', np.uint8),
            ('tags', np.uint64),
        ]

        einfo = np.zeros(n, dtype=edtype)
        for i, row in enumerate(raw.quad_nodes):
            einfo['nodes'][i] = [pos[int(nid)] for nid in row]
        einfo['curved'] = False
        einfo['tags'] = raw.leaf_tags

        cidx = raw.quad_faces_cidx.copy()
        off = raw.quad_faces_off.copy()
        if mortar_cidx is not None:
            cidx[off == -2] = mortar_cidx

        einfo['faces']['cidx'] = cidx
        einfo['faces']['off'] = off

        eles = {'quad': einfo}
        NodalMeshAssembler.compute_element_colouring(eles, codec)

        mortars = {}
        if raw.mortars:
            gdtype = [
                ('left_cidx', np.int16), ('left_eidx', np.int64),
                ('right_cidx', np.int16, 2),
                ('right_eidx', np.int64, 2),
            ]
            records = np.zeros(len(raw.mortars), dtype=gdtype)
            for i, m in enumerate(raw.mortars):
                records[i]['left_cidx'] = codec_index[
                    f'eles/quad/face/{m.left_fidx}'
                ]
                records[i]['left_eidx'] = m.left_eidx
                records[i]['right_cidx'] = [
                    codec_index[f'eles/quad/face/{f}']
                    for f in m.right_fidx
                ]
                records[i]['right_eidx'] = list(m.right_eidx)

            mortars['line-1x2'] = MortarData(
                records, format='one-to-many-v1', template='line-1x2'
            )

        return nodes, eles, codec, {}, mortars


class AdaptedMixedQuadReader(BaseReader):
    """Reader adapter for mixed Tri+Quad meshes with Quad-only AMR."""

    name = 'amr-adapted-mixed-quad'

    def __init__(self, raw, progress=None):
        from pyfr.progress import NullProgressSequence
        super().__init__(progress or NullProgressSequence())
        self._raw = raw

    def _to_raw_mesh(self, lintol):
        raw = self._raw
        pos = {int(nid): i for i, nid in enumerate(raw.node_ids)}
        nodes = np.asarray(raw.node_locs, dtype=float)
        codec = list(raw.codec)
        codec_index = {c: i for i, c in enumerate(codec)}

        if raw.mortars and 'mortar/general-v1' not in codec_index:
            codec_index['mortar/general-v1'] = len(codec)
            codec.append('mortar/general-v1')
        mortar_cidx = codec_index.get('mortar/general-v1')

        fdtype = [('cidx', np.int16), ('off', np.int64)]

        def einfo(nodes_in, cidx, off, tags, nfaces):
            edtype = [
                ('nodes', np.int64, nodes_in.shape[1]), ('curved', bool),
                ('faces', fdtype, nfaces), ('colour', np.uint8),
                ('tags', np.uint64),
            ]
            out = np.zeros(len(nodes_in), dtype=edtype)
            for i, row in enumerate(nodes_in):
                out['nodes'][i] = [pos[int(nid)] for nid in row]
            out['curved'] = False
            out['tags'] = tags
            fcidx = np.array(cidx, copy=True)
            foff = np.array(off, copy=True)
            if mortar_cidx is not None:
                fcidx[foff == -2] = mortar_cidx
            out['faces']['cidx'] = fcidx
            out['faces']['off'] = foff
            return out

        eles = {
            'quad': einfo(
                raw.quad_nodes, raw.quad_faces_cidx, raw.quad_faces_off,
                raw.leaf_tags, 4
            ),
            'tri': einfo(
                raw.tri_nodes, raw.tri_faces_cidx, raw.tri_faces_off,
                raw.tri_tags, 3
            ),
        }
        NodalMeshAssembler.compute_element_colouring(eles, codec)

        mortars = {}
        if raw.mortars:
            gdtype = [
                ('left_cidx', np.int16), ('left_eidx', np.int64),
                ('right_cidx', np.int16, 2),
                ('right_eidx', np.int64, 2),
            ]
            groups = {}
            for mortar in raw.mortars:
                key = mortar.left_etype, mortar.right_etype
                groups.setdefault(key, []).append(mortar)

            for (left_etype, right_etype), group in sorted(groups.items()):
                if len(set(right_etype)) != 1:
                    raise ValueError(
                        'mixed Line-1x2 fine-side element type is not uniform'
                    )
                name = (
                    f'line-1x2-{left_etype}-to-{right_etype[0]}'
                )
                records = np.zeros(len(group), dtype=gdtype)
                for i, m in enumerate(group):
                    records[i]['left_cidx'] = codec_index[
                        f'eles/{m.left_etype}/face/{m.left_fidx}'
                    ]
                    records[i]['left_eidx'] = m.left_eidx
                    records[i]['right_cidx'] = [
                        codec_index[f'eles/{et}/face/{fi}']
                        for et, fi in zip(m.right_etype, m.right_fidx)
                    ]
                    records[i]['right_eidx'] = list(m.right_eidx)

                mortars[name] = MortarData(
                    records, format='one-to-many-v1', template='line-1x2'
                )

        return nodes, eles, codec, {}, mortars


def build_adapted_quad_mesh(raw, lintol=1e-5, progress=None):
    """Build the one-rank native :class:`Mesh` for an adapted Quad raw mesh.

    This is the in-memory sibling of :func:`write_adapted_quad_mesh`.  It
    deliberately reuses ``AdaptedQuadReader._to_raw_mesh`` so element
    colouring, codec construction, mortar records, UUID calculation, and
    canonical element numbering are identical to the ordinary native write
    path without serialising a live adaptation event through ``.pyfrm``.
    """
    reader = AdaptedQuadReader(raw, progress)
    nodes, eles, codec, periodic, mortars = reader._to_raw_mesh(lintol)
    if periodic:
        raise ValueError('adapted in-memory Quad mesh must be nonperiodic')
    if set(eles) != {'quad'}:
        raise ValueError('adapted in-memory mesh must contain only Quads')

    einfo = eles['quad']
    neles = len(einfo)
    eidxs = np.arange(neles, dtype=np.int64)
    cidxmap = {
        cidx: ('quad', fidx)
        for fidx in range(4)
        for cidx in [codec.index(f'eles/quad/face/{fidx}')]
    }

    # Reproduce NativeReader._flatten_faces/_construct_con exactly for the
    # accepted one-rank pure-Quad scope.  In particular, connectivity is
    # face-major and an internal pair is oriented by packed (cidx, eidx),
    # not by element number alone.
    lcidx, leidx, rcidx, reidx = [], [], [], []
    for fidx, eface in enumerate(einfo['faces'].T):
        own_cidx = codec.index(f'eles/quad/face/{fidx}')
        lcidx.append(np.full(neles, own_cidx, dtype=np.int16))
        leidx.append(eidxs)
        rcidx.append(np.asarray(eface['cidx'], dtype=np.int16))
        reidx.append(np.asarray(eface['off'], dtype=np.int64))

    lcidx = np.concatenate(lcidx)
    leidx = np.concatenate(leidx)
    rcidx = np.concatenate(rcidx)
    reidx = np.concatenate(reidx)

    is_boundary = reidx == -1
    is_mortar = reidx == -2
    is_local = reidx >= 0
    if np.any(reidx < -2):
        raise ValueError('adapted Quad face has invalid offset')

    stride = neles
    lkey = lcidx[is_local].astype(np.int64)*stride + leidx[is_local]
    rkey = rcidx[is_local].astype(np.int64)*stride + reidx[is_local]
    iidxs = np.flatnonzero(is_local)[lkey < rkey]

    def con(cidxs, eidxs):
        return Connectivity(
            np.asarray(cidxs, dtype=np.int16),
            np.asarray(eidxs, dtype=np.int64), cidxmap,
        )

    con_l = con(lcidx[iidxs], leidx[iidxs])
    con_r = con(rcidx[iidxs], reidx[iidxs])
    bcon = {}
    for bccidx in np.unique(rcidx[is_boundary]):
        name = codec[int(bccidx)]
        if not name.startswith('bc/'):
            raise ValueError('adapted Quad boundary has invalid native codec')
        mask = rcidx == bccidx
        bcon[name[3:]] = con(lcidx[mask], leidx[mask])

    marked = set(zip(lcidx[is_mortar].tolist(),
                     leidx[is_mortar].tolist()))

    mcon = {}
    for name, minfo in mortars.items():
        if not isinstance(minfo, MortarData):
            raise ValueError('adapted Quad mortar metadata is invalid')

        records = np.array(minfo.records, copy=True)
        refs = set()
        for rec in records:
            refs.add((int(rec['left_cidx']), int(rec['left_eidx'])))
            refs.update(
                (int(cidx), int(eidx))
                for cidx, eidx in zip(
                    rec['right_cidx'], rec['right_eidx']
                )
            )
        if not refs <= marked:
            raise ValueError('adapted Quad mortar metadata mismatch')

        mcon[name] = MortarConnectivity(
            name, records, cidxmap, format=minfo.format,
            template=minfo.template,
        )

    node_idxs = np.arange(len(nodes), dtype=np.int64)
    node_valency = np.zeros(len(nodes), dtype=np.uint16)
    idx, count = np.unique(einfo['nodes'], return_counts=True)
    node_valency[idx] = count.astype(np.uint16)
    spts = nodes[einfo['nodes']].swapaxes(0, 1)
    tree = encode_quad_leaf_tree(raw.root_mesh_uuid, raw.leaf_order)
    uuid = str(UUID(digest((nodes, eles, codec, periodic, mortars))[:32]))

    mesh = Mesh(
        fname='<online-quad-amr>', raw={}, ndims=nodes.shape[1],
        creator=f'pyfr {__version__}', codec=list(codec), uuid=uuid,
        version=2, etypes=['quad'], eidxs={'quad': eidxs},
        spts={'quad': spts}, spts_nodes={'quad': einfo['nodes']},
        spts_curved={'quad': einfo['curved']},
        colours={'quad': einfo['colour']}, tags={'quad': einfo['tags']},
        con=(con_l, con_r), con_p={}, bcon=bcon, mcon=mcon,
        cidxmap=cidxmap, node_idxs=node_idxs, node_valency=node_valency,
        node_locs=np.asarray(nodes), amr_tree=tree,
    )
    mesh.shared_nodes = SharedNodesFinder(
        eles, node_idxs, node_valency
    ).compute()
    return mesh


def build_adapted_mixed_quad_mesh(raw, lintol=1e-5, progress=None):
    """Build one-rank mixed Tri+Quad mesh in memory without serialisation."""
    reader = AdaptedMixedQuadReader(raw, progress)
    nodes, eles, codec, periodic, mortars = reader._to_raw_mesh(lintol)
    if periodic:
        raise ValueError('adapted mixed Quad mesh must be nonperiodic')
    if set(eles) != {'tri', 'quad'}:
        raise ValueError('adapted mixed mesh must contain Tri+Quad elements')

    etypes = sorted(eles)
    eidxs = {
        etype: np.arange(len(eles[etype]), dtype=np.int64)
        for etype in etypes
    }
    cidxmap = {}
    for cidx, value in enumerate(codec):
        for etype, einfo in eles.items():
            for fidx in range(einfo['faces'].shape[-1]):
                if value == f'eles/{etype}/face/{fidx}':
                    cidxmap[cidx] = etype, fidx

    lcidx, leidx, rcidx, reidx = [], [], [], []
    for etype, einfo in eles.items():
        ne = len(einfo)
        for fidx, eface in enumerate(einfo['faces'].T):
            own = codec.index(f'eles/{etype}/face/{fidx}')
            lcidx.append(np.full(ne, own, dtype=np.int16))
            leidx.append(np.arange(ne, dtype=np.int64))
            rcidx.append(np.asarray(eface['cidx'], dtype=np.int16))
            reidx.append(np.asarray(eface['off'], dtype=np.int64))

    lcidx = np.concatenate(lcidx)
    leidx = np.concatenate(leidx)
    rcidx = np.concatenate(rcidx)
    reidx = np.concatenate(reidx)
    if np.any(reidx < -2):
        raise ValueError('adapted mixed face has invalid offset')

    is_boundary = reidx == -1
    is_mortar = reidx == -2
    is_local = reidx >= 0
    stride = max((len(e) for e in eles.values()), default=0) + 1
    lkey = lcidx[is_local].astype(np.int64)*stride + leidx[is_local]
    rkey = rcidx[is_local].astype(np.int64)*stride + reidx[is_local]
    iidxs = np.flatnonzero(is_local)[lkey < rkey]

    def con(cidxs, idxs):
        return Connectivity(
            np.asarray(cidxs, dtype=np.int16),
            np.asarray(idxs, dtype=np.int64), cidxmap,
        )

    con_l = con(lcidx[iidxs], leidx[iidxs])
    con_r = con(rcidx[iidxs], reidx[iidxs])
    bcon = {}
    for bccidx in np.unique(rcidx[is_boundary]):
        name = codec[int(bccidx)]
        if not name.startswith('bc/'):
            raise ValueError('adapted mixed boundary has invalid codec')
        mask = is_boundary & (rcidx == bccidx)
        bcon[name[3:]] = con(lcidx[mask], leidx[mask])

    marked = set(zip(lcidx[is_mortar].tolist(),
                     leidx[is_mortar].tolist()))
    mcon = {}
    for name, minfo in mortars.items():
        records = np.array(minfo.records, copy=True)
        refs = set()
        for rec in records:
            refs.add((int(rec['left_cidx']), int(rec['left_eidx'])))
            refs.update(
                (int(cidx), int(eidx))
                for cidx, eidx in zip(
                    rec['right_cidx'], rec['right_eidx']
                )
            )
        if not refs <= marked:
            raise ValueError('adapted mixed mortar metadata mismatch')
        mcon[name] = MortarConnectivity(
            name, records, cidxmap, format=minfo.format,
            template=minfo.template,
        )

    node_idxs = np.arange(len(nodes), dtype=np.int64)
    node_valency = np.zeros(len(nodes), dtype=np.uint16)
    for einfo in eles.values():
        idx, count = np.unique(einfo['nodes'], return_counts=True)
        node_valency[idx] += count.astype(np.uint16)

    spts = {
        etype: nodes[einfo['nodes']].swapaxes(0, 1)
        for etype, einfo in eles.items()
    }
    tree = encode_quad_leaf_tree(raw.root_mesh_uuid, raw.leaf_order)
    uuid = str(UUID(digest((nodes, eles, codec, periodic, mortars))[:32]))
    mesh = Mesh(
        fname='<online-mixed-quad-amr>', raw={}, ndims=nodes.shape[1],
        creator=f'pyfr {__version__}', codec=list(codec), uuid=uuid,
        version=2, etypes=etypes, eidxs=eidxs, spts=spts,
        spts_nodes={et: eles[et]['nodes'] for et in etypes},
        spts_curved={et: eles[et]['curved'] for et in etypes},
        colours={et: eles[et]['colour'] for et in etypes},
        tags={et: eles[et]['tags'] for et in etypes},
        con=(con_l, con_r), con_p={}, bcon=bcon, mcon=mcon,
        cidxmap=cidxmap, node_idxs=node_idxs, node_valency=node_valency,
        node_locs=np.asarray(nodes), amr_tree=tree,
    )
    mesh.shared_nodes = SharedNodesFinder(
        eles, node_idxs, node_valency
    ).compute()
    return mesh

def _append_amr_ancestry(raw, fname):
    """Append the persistent octree ancestry codec to an already-written
    adapted ``.pyfrm``. Re-derives the :class:`~pyfr.amr.HexLeafTree` from
    ``raw.leaf_order`` through the single accepted D5A encoder
    (:func:`~pyfr.amr.encode_hex_leaf_tree`) rather than writing the
    arrays out by hand, so the persisted schema is always exactly what
    D5A's own validator will accept back."""
    tree = encode_hex_leaf_tree(raw.root_mesh_uuid, raw.leaf_order)

    with h5py.File(fname, 'a', libver='latest') as f:
        f['amr/version'] = np.int64(AMR_LEAF_TREE_VERSION)
        f['amr/root-mesh-uuid'] = np.array(raw.root_mesh_uuid, dtype='S')
        f['amr/leaves/root-eidx'] = tree.root_eidx
        f['amr/leaves/path-offsets'] = tree.path_offsets
        f['amr/leaves/path-data'] = tree.path_data


def write_adapted_mesh(raw, fname, lintol=1e-5, progress=None):
    """Write ``raw`` (an :class:`~pyfr.amrmesh.AdaptedRawMesh`) to
    ``fname`` as a standard native ``.pyfrm`` file, with the persistent
    octree ancestry codec (``amr/*``) appended. Root physical UUID
    reproduction is not required - per the addendum, the written mesh's
    own freshly computed UUID (from :meth:`BaseReader.write`) is
    authoritative. ``amr/root-mesh-uuid`` is a *separate* record: the
    UUID of the immutable pre-adaptation root the tree's leaves reference
    (``raw.root_mesh_uuid``), not the adapted file's own UUID - these are
    deliberately different identities serving different purposes."""
    AdaptedHexReader(raw, progress).write(fname, lintol)
    _append_amr_ancestry(raw, fname)


def _append_quad_amr_ancestry(raw, fname):
    """Append typed persistent quadtree ancestry to an adapted Quad mesh."""
    tree = encode_quad_leaf_tree(raw.root_mesh_uuid, raw.leaf_order)

    with h5py.File(fname, 'a', libver='latest') as f:
        f['amr/version'] = np.int64(AMR_LEAF_TREE_VERSION)
        f['amr/template'] = np.array('quadtree-2x2-v1', dtype='S')
        f['amr/root-mesh-uuid'] = np.array(raw.root_mesh_uuid, dtype='S')
        f['amr/leaves/root-eidx'] = tree.root_eidx
        f['amr/leaves/path-offsets'] = tree.path_offsets
        f['amr/leaves/path-data'] = tree.path_data


def write_adapted_quad_mesh(raw, fname, lintol=1e-5, progress=None):
    """Write an adapted Quad raw mesh with persistent quadtree ancestry."""
    AdaptedQuadReader(raw, progress).write(fname, lintol)
    _append_quad_amr_ancestry(raw, fname)


def write_adapted_mixed_quad_mesh(raw, fname, lintol=1e-5, progress=None):
    """Write mixed Tri+Quad mesh with persistent Quad-only ancestry."""
    AdaptedMixedQuadReader(raw, progress).write(fname, lintol)
    _append_quad_amr_ancestry(raw, fname)

# ===========================================================================
# V10J - mixed Tet+Pyramid+Hex persistence with Hex-only ancestry
# ===========================================================================

class AdaptedMixedHexReader(BaseReader):
    """Reader adapter for mixed Tet+Pyramid+Hex meshes with Hex-only AMR."""

    name = 'amr-adapted-mixed-hex'

    def __init__(self, raw, progress=None):
        from pyfr.progress import NullProgressSequence
        super().__init__(progress or NullProgressSequence())
        self._raw = raw

    def _to_raw_mesh(self, lintol):
        raw = self._raw
        pos = {int(nid): i for i, nid in enumerate(raw.node_ids)}
        nodes = np.asarray(raw.node_locs, dtype=float)
        codec = list(raw.codec)
        codec_index = {c: i for i, c in enumerate(codec)}

        if raw.mortars and 'mortar/general-v1' not in codec_index:
            codec_index['mortar/general-v1'] = len(codec)
            codec.append('mortar/general-v1')
        mortar_cidx = codec_index.get('mortar/general-v1')

        fdtype = [('cidx', np.int16), ('off', np.int64)]

        def einfo(nodes_in, cidx, off, tags, curved):
            nfaces = cidx.shape[1]
            edtype = [
                ('nodes', np.int64, nodes_in.shape[1]), ('curved', bool),
                ('faces', fdtype, nfaces), ('colour', np.uint8),
                ('tags', np.uint64),
            ]
            out = np.zeros(len(nodes_in), dtype=edtype)
            for i, row in enumerate(nodes_in):
                out['nodes'][i] = [pos[int(nid)] for nid in row]
            out['curved'] = curved
            out['tags'] = tags
            fcidx = np.array(cidx, copy=True)
            foff = np.array(off, copy=True)
            if mortar_cidx is not None:
                fcidx[foff == -2] = mortar_cidx
            out['faces']['cidx'] = fcidx
            out['faces']['off'] = foff
            return out

        eles = {
            etype: einfo(
                raw.fixed_nodes[etype], raw.fixed_faces_cidx[etype],
                raw.fixed_faces_off[etype], raw.fixed_tags[etype],
                raw.fixed_curved[etype]
            )
            for etype in sorted(raw.fixed_nodes)
        }
        eles['hex'] = einfo(
            raw.hex_nodes, raw.hex_faces_cidx, raw.hex_faces_off,
            raw.leaf_tags, raw.hex_curved
        )
        NodalMeshAssembler.compute_element_colouring(eles, codec)

        mortars = {}
        if raw.mortars:
            gdtype = [
                ('left_cidx', np.int16), ('left_eidx', np.int64),
                ('right_cidx', np.int16, 4),
                ('right_eidx', np.int64, 4),
            ]
            groups = {}
            for mortar in raw.mortars:
                key = mortar.left_etype, mortar.right_etype
                groups.setdefault(key, []).append(mortar)

            for (left_etype, right_etype), group in sorted(groups.items()):
                if len(set(right_etype)) != 1:
                    raise ValueError(
                        'mixed quad-2x2 fine-side element type is not uniform'
                    )
                name = f'quad-2x2-{left_etype}-to-{right_etype[0]}'
                records = np.zeros(len(group), dtype=gdtype)
                for i, m in enumerate(group):
                    records[i]['left_cidx'] = codec_index[
                        f'eles/{m.left_etype}/face/{m.left_fidx}'
                    ]
                    records[i]['left_eidx'] = m.left_eidx
                    records[i]['right_cidx'] = [
                        codec_index[f'eles/{et}/face/{fi}']
                        for et, fi in zip(m.right_etype, m.right_fidx)
                    ]
                    records[i]['right_eidx'] = list(m.right_eidx)

                mortars[name] = MortarData(
                    records, format='one-to-many-v1', template='quad-2x2'
                )

        return nodes, eles, codec, {}, mortars


def build_adapted_mixed_hex_mesh(raw, lintol=1e-5, progress=None):
    """Build one-rank mixed Tet+Pyramid+Hex mesh in memory."""
    reader = AdaptedMixedHexReader(raw, progress)
    nodes, eles, codec, periodic, mortars = reader._to_raw_mesh(lintol)
    if periodic:
        raise ValueError('adapted mixed Hex mesh must be nonperiodic')
    if set(eles) != {'hex', 'pyr', 'tet'}:
        raise ValueError(
            'adapted mixed Hex mesh must contain Tet+Pyramid+Hex elements'
        )

    etypes = sorted(eles)
    eidxs = {
        etype: np.arange(len(eles[etype]), dtype=np.int64)
        for etype in etypes
    }
    cidxmap = {}
    for cidx, value in enumerate(codec):
        for etype, einfo in eles.items():
            for fidx in range(einfo['faces'].shape[-1]):
                if value == f'eles/{etype}/face/{fidx}':
                    cidxmap[cidx] = etype, fidx

    lcidx, leidx, rcidx, reidx = [], [], [], []
    for etype, einfo in eles.items():
        ne = len(einfo)
        for fidx, eface in enumerate(einfo['faces'].T):
            own = codec.index(f'eles/{etype}/face/{fidx}')
            lcidx.append(np.full(ne, own, dtype=np.int16))
            leidx.append(np.arange(ne, dtype=np.int64))
            rcidx.append(np.asarray(eface['cidx'], dtype=np.int16))
            reidx.append(np.asarray(eface['off'], dtype=np.int64))

    lcidx = np.concatenate(lcidx)
    leidx = np.concatenate(leidx)
    rcidx = np.concatenate(rcidx)
    reidx = np.concatenate(reidx)
    if np.any(reidx < -2):
        raise ValueError('adapted mixed Hex face has invalid offset')

    is_boundary = reidx == -1
    is_mortar = reidx == -2
    is_local = reidx >= 0
    stride = max((len(e) for e in eles.values()), default=0) + 1
    lkey = lcidx[is_local].astype(np.int64)*stride + leidx[is_local]
    rkey = rcidx[is_local].astype(np.int64)*stride + reidx[is_local]
    iidxs = np.flatnonzero(is_local)[lkey < rkey]

    def con(cidxs, idxs):
        return Connectivity(
            np.asarray(cidxs, dtype=np.int16),
            np.asarray(idxs, dtype=np.int64), cidxmap,
        )

    con_l = con(lcidx[iidxs], leidx[iidxs])
    con_r = con(rcidx[iidxs], reidx[iidxs])
    bcon = {}
    for bccidx in np.unique(rcidx[is_boundary]):
        name = codec[int(bccidx)]
        if not name.startswith('bc/'):
            raise ValueError('adapted mixed Hex boundary has invalid codec')
        mask = is_boundary & (rcidx == bccidx)
        bcon[name[3:]] = con(lcidx[mask], leidx[mask])

    marked = set(zip(lcidx[is_mortar].tolist(), leidx[is_mortar].tolist()))
    mcon = {}
    for name, minfo in mortars.items():
        records = np.array(minfo.records, copy=True)
        refs = set()
        for rec in records:
            refs.add((int(rec['left_cidx']), int(rec['left_eidx'])))
            refs.update(
                (int(cidx), int(eidx))
                for cidx, eidx in zip(
                    rec['right_cidx'], rec['right_eidx']
                )
            )
        if not refs <= marked:
            raise ValueError('adapted mixed Hex mortar metadata mismatch')
        mcon[name] = MortarConnectivity(
            name, records, cidxmap, format=minfo.format,
            template=minfo.template,
        )

    node_idxs = np.arange(len(nodes), dtype=np.int64)
    node_valency = np.zeros(len(nodes), dtype=np.uint16)
    for einfo in eles.values():
        idx, count = np.unique(einfo['nodes'], return_counts=True)
        node_valency[idx] += count.astype(np.uint16)

    spts = {
        etype: nodes[einfo['nodes']].swapaxes(0, 1)
        for etype, einfo in eles.items()
    }
    tree = encode_hex_leaf_tree(raw.root_mesh_uuid, raw.leaf_order)
    uuid = str(UUID(digest((nodes, eles, codec, periodic, mortars))[:32]))
    mesh = Mesh(
        fname='<online-mixed-hex-amr>', raw={}, ndims=nodes.shape[1],
        creator=f'pyfr {__version__}', codec=list(codec), uuid=uuid,
        version=2, etypes=etypes, eidxs=eidxs, spts=spts,
        spts_nodes={et: eles[et]['nodes'] for et in etypes},
        spts_curved={et: eles[et]['curved'] for et in etypes},
        colours={et: eles[et]['colour'] for et in etypes},
        tags={et: eles[et]['tags'] for et in etypes},
        con=(con_l, con_r), con_p={}, bcon=bcon, mcon=mcon,
        cidxmap=cidxmap, node_idxs=node_idxs, node_valency=node_valency,
        node_locs=np.asarray(nodes), amr_tree=tree,
    )
    mesh.shared_nodes = SharedNodesFinder(
        eles, node_idxs, node_valency
    ).compute()
    return mesh


def write_adapted_mixed_hex_mesh(raw, fname, lintol=1e-5, progress=None):
    """Write mixed Tet+Pyramid+Hex mesh with persistent Hex ancestry."""
    AdaptedMixedHexReader(raw, progress).write(fname, lintol)
    _append_amr_ancestry(raw, fname)
