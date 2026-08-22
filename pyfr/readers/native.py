from dataclasses import dataclass, field, replace
import re

import h5py
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as s2u

from pyfr.inifile import Inifile
from pyfr.mpiutil import (Scatterer, SparseScatterer, autofree,
                          get_comm_rank_root)
from pyfr.readers.shared_nodes import SharedNodesFinder
from pyfr.shapes import BaseShape
from pyfr.util import first, subclass_where


@dataclass
class Mesh:
    fname: str
    raw: object

    ndims: int = None
    subset: bool = False
    parent: 'Mesh' = None

    creator: str = None
    codec: list = None
    uuid: str = None
    version: int = None

    etypes: list = field(default_factory=list)
    eidxs: dict = field(default_factory=dict)

    spts: dict = field(default_factory=dict)
    spts_nodes: dict = field(default_factory=dict)
    spts_curved: dict = field(default_factory=dict)
    colours: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)

    con: tuple = field(default_factory=tuple)
    con_p: dict = field(default_factory=dict)
    bcon: dict = field(default_factory=dict)
    mcon: dict = field(default_factory=dict)
    cidxmap: dict = field(default_factory=dict)

    # Shared nodes for C0 continuous fields
    node_idxs: np.ndarray = None
    node_valency: np.ndarray = None
    node_locs: np.ndarray = None
    shared_nodes: object = None

    # V10D5C2: persistent native octree ancestry (pyfr.amr.HexLeafTree),
    # populated only when the file carries an 'amr' group; None for every
    # ordinary non-AMR native mesh, which is otherwise completely
    # unaffected by this field's existence.
    amr_tree: object = None


@dataclass(frozen=True)
class SolutionGroup:
    etype: str
    order: int
    eidxs: np.ndarray


@dataclass
class Solution:
    config: object
    stats: object
    fields: list
    data: dict = field(default_factory=dict)
    grad_data: dict = field(default_factory=dict)
    aux: dict = field(default_factory=dict)
    dtypes: dict = field(default_factory=dict)
    prevcfgs: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)


class Connectivity:
    def __init__(self, cidxs, eidxs, cidxmap):
        self.cidxmap = cidxmap
        self.cidxs = cidxs
        self.eidxs = eidxs
        self._ucidxs = np.unique(cidxs).tolist()

    def __len__(self):
        return len(self.cidxs)

    def items(self):
        for cidx in self._ucidxs:
            etype, fidx = self.cidxmap[cidx]
            yield etype, fidx, self.eidxs[self.cidxs == cidx]

    def foreach(self):
        for cidx in self._ucidxs:
            mask = self.cidxs == cidx
            etype, fidx = self.cidxmap[cidx]
            yield etype, fidx, self.eidxs[mask], np.flatnonzero(mask)

    def map_eles(self, data, dtype=None):
        result = np.empty(len(self), dtype=dtype or first(data.values()).dtype)
        for etype, fidx, eidxs, mask in self.foreach():
            result[mask] = data[etype][eidxs]
        return result


class MortarConnectivity:
    def __init__(
        self, name, records, cidxmap, *, format=None, template=None
    ):
        self.name = name
        self.records = records
        self.cidxmap = cidxmap
        self.format = format or 'quad-tri-v1'
        self.template = template or 'quad-tri'

    def __len__(self):
        return len(self.records)

    @property
    def nright(self):
        fields = self.records.dtype.fields
        if 'right_cidx' in fields:
            shape = fields['right_cidx'][0].shape
            return shape[0] if shape else 1
        if 'fine_cidx' in fields:
            return self.records.dtype.fields['fine_cidx'][0].shape[0]
        raise ValueError('Invalid mortar connectivity record')

    def side(self, name, child=None):
        fields = self.records.dtype.fields
        if name in ('left', 'coarse'):
            cfield = 'left_cidx' if 'left_cidx' in fields else 'coarse_cidx'
            efield = 'left_eidx' if 'left_eidx' in fields else 'coarse_eidx'
            cidx = self.records[cfield]
            eidx = self.records[efield]
        elif name in ('right', 'fine') and child is not None:
            cfield = 'right_cidx' if 'right_cidx' in fields else 'fine_cidx'
            efield = 'right_eidx' if 'right_eidx' in fields else 'fine_eidx'
            if child < 0 or child >= self.nright:
                raise ValueError('Invalid mortar side')
            cidx = self.records[cfield][:, child]
            eidx = self.records[efield][:, child]
        else:
            raise ValueError('Invalid mortar side')

        return Connectivity(cidx, eidx, self.cidxmap)


class NativeReader:
    def __init__(self, fname, pname=None, *, construct_con=True):
        self.f = h5py.File(fname, 'r')
        self.mesh = Mesh(fname=fname, raw=self.f)

        # Read in and transform the various parts of the mesh
        self._read_metadata()
        self._read_amr_tree()
        self._read_partitioning(pname)
        self._read_eles()
        self._validate_amr_tree()
        self._read_nodes()

        if construct_con:
            self._construct_con()

        self._construct_shared_nodes()

    def close(self):
        self.f.close()

    def load_soln(self, sname, prefix=None):
        mesh, soln = self.load_subset_mesh_soln(sname, prefix)

        # Ensure the solution is not subset
        if mesh is not self.mesh:
            raise ValueError('Subset solutions are not supported')

        return soln

    def _read_soln_header(self, f):
        comm, rank, root = get_comm_rank_root()

        if rank == root:
            # Ensure the solution is from the mesh we are using
            uuid = f['mesh-uuid'][()].decode()
            if uuid != self.mesh.uuid:
                raise RuntimeError('Invalid solution for mesh')

            # Ensure solution format is v2
            if f['version'][()] != 2:
                raise RuntimeError('Solution file must be format v2')

            # Read any config and stats records
            cfgs = {fname: f[fname][()].decode()
                    for fname in f if fname.startswith('config')}
            cfgs['stats'] = f['stats'][()].decode()
        else:
            cfgs = None

        # Broadcast and parse
        cfgs = comm.bcast(cfgs, root=root)
        cfgs = {k: Inifile(v) for k, v in cfgs.items()}

        # Extract previous configs
        prevcfgs = {k: v for k, v in cfgs.items() if k.startswith('config-')}

        # Read serialised state (plugins, bcs, intg)
        if rank == root:
            state = {}
            def svisit(name):
                if name.startswith(('plugins/', 'bcs/', 'intg/')):
                    if not isinstance(f[name], h5py.Group):
                        state[name] = f[name][()]
            f.visit(svisit)
        else:
            state = None

        state = comm.bcast(state, root=root)

        return Solution(config=cfgs['config'], stats=cfgs['stats'],
                        fields=None, prevcfgs=prevcfgs, state=state)

    def _soln_fields(self, dtype):
        fields = []
        for g in dtype.names:
            if g in ('grad', 'aux'):
                continue

            prefix = '' if g == 'soln' else f'{g}-'
            fields.extend(f'{prefix}{n}' for n in dtype[g].names)
        return fields

    def _unpack_esoln(self, soln, etype, esoln, dtype):
        dgroups = [g for g in dtype.names if g not in ('grad', 'aux')]
        ne, nd = len(esoln), self.mesh.ndims

        # Unpack all data groups into a single array
        parts = []
        for g in dgroups:
            arr = s2u(esoln[g]).reshape(ne, len(dtype[g].names), -1)
            parts.append(arr.transpose(2, 1, 0))

        soln.data[etype] = np.concatenate(parts, axis=1)

        # Gradient data
        if 'grad' in dtype.names:
            gv = len(dtype['grad'].names)
            g = s2u(esoln['grad']).reshape(ne, gv, nd, -1)
            soln.grad_data[etype] = g.transpose(2, 3, 1, 0)

        # Auxiliary fields
        if 'aux' in dtype.names:
            soln.aux[etype] = {n: esoln['aux'][n] for n in dtype['aux'].names}

    def _soln_dtype_signature(self, dtype):
        return tuple(
            (g, tuple(dtype[g].names or ())) for g in dtype.names
        )

    def _soln_nupts(self, dtype):
        try:
            field = dtype['soln'].names[0]
            shape = dtype['soln'].fields[field][0].shape
        except (KeyError, TypeError, IndexError):
            raise ValueError('Invalid solution data type') from None

        if len(shape) != 1:
            raise ValueError('Invalid solution point field shape')

        return shape[0]

    def _persistent_soln_group_names(self, f, prefix):
        byetype = {}
        group = f.get(prefix)

        if group is None:
            return byetype

        for name, dset in group.items():
            if not isinstance(dset, h5py.Dataset):
                continue
            if m := re.fullmatch(r'p(\d+)-(\w+)', name):
                byetype.setdefault(m[2], []).append(name)

        return byetype

    def _mixed_soln_groups(self, f, prefix):
        groups = {}
        byetype = {}
        group = f.get(prefix)

        if group is None:
            return groups, byetype

        for name, dset in group.items():
            if not isinstance(dset, h5py.Dataset):
                continue
            if not (m := re.fullmatch(r'p(\d+)-(\w+)', name)):
                continue

            order, etype = int(m[1]), m[2]
            if etype not in self.mesh.etypes:
                raise ValueError(f'Unknown solution element type {etype!r}')

            shapecls = subclass_where(BaseShape, name=etype)
            if self._soln_nupts(dset.dtype) != shapecls.npts_from_order(order):
                raise ValueError(
                    f'{name}: persistent order/point-count mismatch'
                )

            epath = f'{prefix}/{name}-idxs'
            if epath in f:
                eidxs = np.asarray(f[epath], dtype=np.int64)
                if len(eidxs) != len(dset):
                    raise ValueError(f'{name}: invalid element index count')
            else:
                neles = len(self.f[f'eles/{etype}'])
                if len(dset) != neles:
                    raise ValueError(f'{name}: missing element index array')
                eidxs = np.arange(neles, dtype=np.int64)

            neles = len(self.f[f'eles/{etype}'])
            if np.any((eidxs < 0) | (eidxs >= neles)):
                raise ValueError(f'{name}: element index out of range')
            if len(np.unique(eidxs)) != len(eidxs):
                raise ValueError(f'{name}: duplicate element indices')

            groups[name] = (etype, order, eidxs, dset.dtype)
            byetype.setdefault(etype, []).append(name)

        for etype, names in byetype.items():
            eidxs = np.concatenate([groups[n][2] for n in names])
            if len(np.unique(eidxs)) != len(eidxs):
                raise ValueError(f'{etype}: overlapping persistent groups')

        return groups, byetype

    def _is_mixed_soln(self, soln, byetype):
        cfgmixed = any(
            s.startswith('solver-order-') for s in soln.config.sections()
        )
        filemixed = any(len(names) > 1 for names in byetype.values())
        return cfgmixed or filemixed

    def _load_mixed_subset_mesh_soln(self, f, soln, prefix, groups, byetype):
        comm, _, _ = get_comm_rank_root()
        subset = {}
        signature = None

        for etype in self.mesh.etypes:
            names = byetype.get(etype, [])
            present = np.concatenate(
                [groups[n][2] for n in names]
            ) if names else np.empty(0, dtype=np.int64)

            local = self.mesh.eidxs.get(etype, np.empty(0, dtype=int))
            ridx = np.flatnonzero(np.isin(local, present))
            if len(ridx) != len(local):
                subset[etype] = ridx

        for name in sorted(groups):
            etype, order, eidxs, dtype = groups[name]
            ek = f'{prefix}/{name}'
            ei = f'{ek}-idxs'
            local = self.mesh.eidxs.get(etype, np.empty(0, dtype=int))

            if ei in f:
                escatter = SparseScatterer(comm, f[ei], local)
                geidxs = local[escatter.ridx]
            else:
                escatter = self.escatter[etype]
                geidxs = local

            sig = self._soln_dtype_signature(dtype)
            if signature is None:
                signature = sig
                soln.fields = self._soln_fields(dtype)
            elif sig != signature or self._soln_fields(dtype) != soln.fields:
                raise ValueError('Incompatible fields across solution groups')

            soln.groups[name] = SolutionGroup(etype, order, geidxs)
            soln.dtypes[name] = dtype

            esoln = escatter(f[ek])
            if escatter.cnt:
                self._unpack_esoln(soln, name, esoln, dtype)

        if subset:
            return self._subset_mesh(subset), soln
        return self.mesh, soln

    def load_subset_mesh_soln(self, sname, prefix=None):
        comm, rank, root = get_comm_rank_root()

        with h5py.File(sname, 'r') as f:
            soln = self._read_soln_header(f)

            # If no prefix has been specified then obtain it from the file
            if prefix is None:
                prefix = soln.stats.get('data', 'prefix')

            byetype = self._persistent_soln_group_names(f, prefix)
            if self._is_mixed_soln(soln, byetype):
                groups, byetype = self._mixed_soln_groups(f, prefix)
                return self._load_mixed_subset_mesh_soln(
                    f, soln, prefix, groups, byetype
                )

            # Obtain the polynomial order
            order = soln.config.getint('solver', 'order')

            # Note if any elements are subset
            subset = {}

            # Read and scatter the solution data
            for etype in self.escatter:
                # If the element is not present, mark it as completely subset
                if (ek := f'{prefix}/p{order}-{etype}') not in f:
                    subset[etype] = []
                    continue
                # If the element is partially subset use a sparse scatterer
                elif (ei := f'{ek}-idxs') in f:
                    try:
                        idxs = self.mesh.eidxs[etype]
                    except KeyError:
                        idxs = np.empty(0, dtype=int)

                    escatter = SparseScatterer(comm, f[ei], idxs)
                    subset[etype] = escatter.ridx
                # Complete element present so reuse the elements scatterer
                else:
                    escatter = self.escatter[etype]

                soln.dtypes[etype] = f[ek].dtype

                # Build field list from the first dataset encountered
                if soln.fields is None:
                    soln.fields = self._soln_fields(f[ek].dtype)

                esoln = escatter(f[ek])
                if escatter.cnt:
                    self._unpack_esoln(soln, etype, esoln, f[ek].dtype)

        # If the solution is subset then subset the mesh, too
        if subset:
            return self._subset_mesh(subset), soln
        else:
            return self.mesh, soln

    def _subset_mesh(self, subset):
        eidxs, spts, spts_nodes, spts_curved = {}, {}, {}, {}

        for etype in self.mesh.spts:
            if etype in subset:
                sidx = subset[etype]
                if len(sidx):
                    eidxs[etype] = self.mesh.eidxs[etype][sidx]
                    spts[etype] = self.mesh.spts[etype][:, sidx]
                    spts_nodes[etype] = self.mesh.spts_nodes[etype][sidx]
                    spts_curved[etype] = self.mesh.spts_curved[etype][sidx]
            else:
                eidxs[etype] = self.mesh.eidxs[etype]
                spts[etype] = self.mesh.spts[etype]
                spts_nodes[etype] = self.mesh.spts_nodes[etype]
                spts_curved[etype] = self.mesh.spts_curved[etype]

        return replace(self.mesh, subset=True, parent=self.mesh,
                       eidxs=eidxs, spts=spts, spts_nodes=spts_nodes,
                       spts_curved=spts_curved, con=None, con_p=None,
                       bcon=None)

    def _read_metadata(self):
        mesh = self.mesh
        comm, rank, root = get_comm_rank_root()

        if rank == root:
            creator = self.f['creator'][()].decode()
            codec = [c.decode() for c in self.f['codec']]
            uuid = self.f['mesh-uuid'][()].decode()
            version = self.f['version'][()]

            meta = (creator, codec, uuid, version)
        else:
            meta = None

        meta = comm.bcast(meta, root=root)
        mesh.creator, mesh.codec, mesh.uuid, mesh.version = meta

    def _read_amr_tree(self):
        # V10D5C2: optional persistent octree ancestry.  Ordinary non-AMR
        # native meshes carry no 'amr' group and mesh.amr_tree stays None;
        # this method changes no other behaviour of the reader.
        comm, rank, root = get_comm_rank_root()

        if rank == root:
            if 'amr' in self.f:
                from pyfr.amr import (
                    hex_leaf_tree_from_arrays, quad_leaf_tree_from_arrays,
                )

                grp = self.f['amr']
                args = (
                    int(grp['version'][()]),
                    grp['root-mesh-uuid'][()].decode(),
                    grp['leaves']['root-eidx'][()],
                    grp['leaves']['path-offsets'][()],
                    grp['leaves']['path-data'][()],
                )

                if 'template' not in grp:
                    # Legacy accepted Hex ancestry has no template field.
                    tree = hex_leaf_tree_from_arrays(*args)
                else:
                    template = grp['template'][()]
                    if isinstance(template, bytes):
                        template = template.decode()
                    if template == 'quadtree-2x2-v1':
                        tree = quad_leaf_tree_from_arrays(*args)
                    else:
                        raise ValueError(
                            f'Unsupported persistent AMR template '
                            f'{template!r}'
                        )
            else:
                tree = None
        else:
            tree = None

        self.mesh.amr_tree = comm.bcast(tree, root=root)

    def _validate_amr_tree(self):
        # A structurally valid leaf tree is not automatically ancestry for
        # THIS adapted mesh. Canonical leaf ordinals are partition-independent
        # global element eidxs, whose union must be exactly 0..N-1. Ordinary
        # non-AMR meshes return above and pay no collective cost.
        tree = self.mesh.amr_tree
        if tree is None:
            return

        from pyfr.amr import HexLeafTree, QuadLeafTree

        if isinstance(tree, HexLeafTree):
            etype, label = 'hex', 'Hex'
        elif isinstance(tree, QuadLeafTree):
            etype, label = 'quad', 'Quad'
        else:
            raise RuntimeError('Unknown persistent AMR leaf-tree type')

        if isinstance(tree, HexLeafTree):
            etypes = set(self.mesh.etypes)
            if etypes not in ({'hex'}, {'hex', 'pyr', 'tet'}):
                raise RuntimeError(
                    'Persistent Hex AMR ancestry requires a pure-Hex or '
                    'Tet+Pyramid+Hex topology; '
                    f'found element types {self.mesh.etypes}'
                )
        if isinstance(tree, QuadLeafTree):
            extra = set(self.mesh.etypes) - {'quad', 'tri'}
            if 'quad' not in self.mesh.etypes or extra:
                raise RuntimeError(
                    'Persistent Quad AMR ancestry requires a pure-Quad or '
                    'Tri+Quad '
                    f'topology; found element types {self.mesh.etypes}'
                )

        neles = len(self.f[f'eles/{etype}'])
        if tree.nleaves != neles:
            raise RuntimeError(
                f'Persistent AMR ancestry leaf count ({tree.nleaves}) does '
                f'not match the adapted mesh {label} element count ({neles})'
            )

        comm, _, _ = get_comm_rank_root()
        geidx = np.asarray(
            self.mesh.eidxs.get(etype, np.empty(0, dtype=np.int64)),
            dtype=np.int64
        )
        if (len(np.unique(geidx)) != len(geidx) or
                np.any((geidx < 0) | (geidx >= tree.nleaves))):
            raise RuntimeError(
                f'Persistent AMR ancestry has invalid local canonical '
                f'{label} ordinals'
            )

        all_geidx = comm.allgather(geidx)
        flat = np.concatenate(all_geidx)
        if (len(flat) != tree.nleaves or
                not np.array_equal(np.sort(flat),
                                   np.arange(tree.nleaves, dtype=np.int64))):
            raise RuntimeError(
                f'Persistent AMR ancestry requires the global adapted '
                f'{label} eidx union to be exactly 0..N-1 with no gaps or '
                'duplicates'
            )

    def _read_with_idxs(self, dset, idxs):
        comm, rank, root = get_comm_rank_root()

        # Construct a Scatterer to read in and distribute the data
        s = Scatterer(comm, idxs)

        return s(dset), s

    def _select_partitioning(self, size, pname=None):
        # If a partitioning has been specified then use it
        if pname:
            pinfo = self.f[f'partitionings/{pname}']
            nparts = len(pinfo['eles'].attrs['regions'])
            if nparts != size:
                raise RuntimeError(f'Partitioning {pname} has {nparts} parts '
                                   f'but running with {size} ranks')
        # Otherwise, try to find one
        else:
            for pname, pinfo in self.f['partitionings'].items():
                nparts = len(pinfo['eles'].attrs['regions'])
                if nparts == size:
                    break
            else:
                raise RuntimeError('Mesh does not have any partitionings with '
                                   f'{size} ranks')

        return pname, pinfo

    def _read_partitioning(self, pname=None):
        comm, rank, root = get_comm_rank_root()
        size = comm.size

        # Have the root rank read in the partitioning metadata
        if rank == root:
            pname, pinfo = self._select_partitioning(size, pname)

            # Read the element region data
            einfo = pinfo['eles'].attrs['regions']

            # Read the neighbours data
            if size > 1:
                ninfo = pinfo['neighbours']
                ninfo = np.split(ninfo[()], ninfo.attrs['regions'][1:-1])
            else:
                ninfo = [[]]
        else:
            pname = einfo = ninfo = None

        # Broadcast this metadata
        ppath = 'partitionings/' + comm.bcast(pname, root=root)
        einfo = comm.scatter(einfo, root=root)
        self.neighbours = comm.scatter(ninfo, root=root)

        # Determine the element types in the mesh
        self.mesh.etypes = etypes = sorted(self.f['eles'])

        # Read our portion of the partitioning table
        peles = self.f[f'{ppath}/eles'][einfo[0]:einfo[-1]]
        peles = np.split(peles, [i - einfo[0] for i in einfo[1:-1]])

        # With this determine the indices associated with each element
        self.mesh.eidxs = {et: pe for et, pe in zip(etypes, peles) if pe.size}

    def _read_eles(self):
        self.eles, self.escatter = eles, escatter = {}, {}

        # Collectively read in and distribute each element array
        for etype in self.mesh.etypes:
            dset = self.f[f'eles/{etype}']
            idxs = self.mesh.eidxs.get(etype, [])
            einfo, escatter[etype] = self._read_with_idxs(dset, idxs)

            # If we have any elements of this type then save the einfo
            if len(idxs):
                eles[etype] = einfo

    def _read_nodes(self):
        enodes = [einfo['nodes'] for einfo in self.eles.values()]

        # Determine the overall set of nodes across all element types
        idxs = np.concatenate([en.ravel() for en in enodes])

        # Note how many dimensions we have
        self.mesh.ndims = self.f['nodes'].dtype['location'].shape[0]

        # Read in these nodes
        nodes = self._read_with_idxs(self.f['nodes'], idxs.ravel())[0]

        # Store unique node indices, valency, and locations for vertices
        unique_idxs, first_occ = np.unique(idxs, return_index=True)
        self.mesh.node_idxs = unique_idxs
        self.mesh.node_valency = nodes['valency'][first_occ]
        self.mesh.node_locs = nodes['location'][first_occ]

        # Determine where each element type is in the nodes array
        eoffs = np.cumsum([en.size for en in enodes])

        # Use this to split the nodes array back up
        locs = np.split(nodes['location'], eoffs[:-1])

        # Reshape and add to the mesh
        for (etype, einfo), n in zip(self.eles.items(), locs):
            spts = n.reshape(*einfo['nodes'].shape, -1).swapaxes(0, 1)

            self.mesh.spts[etype] = spts
            self.mesh.spts_nodes[etype] = einfo['nodes']
            self.mesh.spts_curved[etype] = einfo['curved']
            self.mesh.colours[etype] = einfo['colour']
            self.mesh.tags[etype] = einfo['tags']

    def _parse_codec(self):
        codec = self.mesh.codec
        ncodec = len(codec)

        cidxmap = {}
        cetmap = np.full(ncodec, -1, dtype=np.int16)

        for cidx, c in enumerate(codec):
            if (m := re.match(r'eles/(\w+)/face/(\d+)$', c)):
                cidxmap[cidx] = etype, fidx = m[1], int(m[2])
                cetmap[cidx] = self.mesh.etypes.index(etype)

        self.mesh.cidxmap = cidxmap
        return cidxmap, cetmap

    @staticmethod
    def _pack_pairs(*pairs):
        stride = max(o.max(initial=-1) for _, o in pairs) + 1
        return [c*stride + o for c, o in pairs]

    @staticmethod
    def _pair_finder(lhs, stride):
        # Pack (cidx, idx) pairs into flat keys for binary search
        keys = lhs.cidx.astype(int)*stride + lhs.idx
        sord = np.argsort(keys)

        def find(rec):
            qkeys = rec['cidx'].astype(int)*stride + rec['idx']
            pos = np.searchsorted(keys, qkeys, sorter=sord)
            idx = np.take(sord, pos, mode='clip')
            return idx[keys[idx] == qkeys]

        return find

    def _build_g2l(self):
        g2l = {}

        for etype in self.mesh.etypes:
            if (gi := self.mesh.eidxs.get(etype)) is not None:
                perm = np.argsort(gi)
                g2l[etype] = (gi, gi[perm], perm)

        return g2l

    def _flatten_faces(self, g2l):
        codec = self.mesh.codec
        parts = []
        for etype, einfo in self.eles.items():
            gi = g2l[etype][0]
            for fidx, eface in enumerate(einfo['faces'].T):
                n = len(eface)
                efcidx = codec.index(f'eles/{etype}/face/{fidx}')
                parts.append((np.broadcast_to(np.int16(efcidx), n),
                              np.arange(n), gi, eface['cidx'], eface['off']))

        return map(np.concatenate, zip(*parts))

    def _construct_con(self):
        cidxmap, cetmap = self._parse_codec()
        g2l = self._build_g2l()
        lcidx, leidx, lgidx, rcidx, rgidx = self._flatten_faces(g2l)

        # Global-to-local lookup for rhs element-neighbour faces
        reidx = np.full(len(lcidx), -1)
        for etidx, etype in enumerate(self.mesh.etypes):
            if etype not in g2l:
                continue

            ordgi, perm = g2l[etype][1:]

            # Select rhs faces whose neighbour is this element type
            mask = cetmap[rcidx] == etidx
            offs = rgidx[mask]

            # Map global element numbers to partition local numbers
            pos = np.searchsorted(ordgi, offs)
            pos = np.clip(pos, 0, len(ordgi) - 1)
            reidx[mask] = np.where(ordgi[pos] == offs, perm[pos], -1)

        # Classify interfaces
        is_boundary, is_mortar = rgidx == -1, rgidx == -2
        is_local = reidx >= 0
        is_mpi = ~(is_boundary | is_mortar | is_local)

        # Deduplicate internal interfaces
        lkey, rkey = self._pack_pairs((lcidx[is_local], leidx[is_local]),
                                      (rcidx[is_local], reidx[is_local]))
        iidxs = np.flatnonzero(is_local)[lkey < rkey]

        con = lambda c, e: Connectivity(c, e, cidxmap)
        self.mesh.con = (con(lcidx[iidxs], leidx[iidxs]),
                         con(rcidx[iidxs], reidx[iidxs]))

        # Boundary connectivity
        for bccidx in np.unique(rcidx[is_boundary]):
            name = self.mesh.codec[bccidx][3:]
            bmask = rcidx == bccidx
            self.mesh.bcon[name] = con(lcidx[bmask], leidx[bmask])

        # Mortar connectivity.  The partitioner keeps all participants in
        # each mortar on one rank, so every local record can use the existing
        # local mortar execution path unchanged.
        if 'mortars' in self.f:
            self._construct_mortar_con(g2l, cidxmap, is_mortar, lcidx, leidx)

        # MPI connectivity
        if np.any(is_mpi):
            dt = [('cidx', np.int16), ('idx', int)]
            lhs = np.rec.fromarrays([lcidx[is_mpi], lgidx[is_mpi]], dtype=dt)
            rhs = np.rec.fromarrays([rcidx[is_mpi], rgidx[is_mpi]], dtype=dt)

            # Stride for packing (cidx, idx) into collision-free keys
            stride = max(len(self.f[f'eles/{et}']) for et in self.mesh.etypes)

            self._construct_mpi_con(g2l, cetmap, cidxmap, lhs, rhs, stride)

    def _construct_mortar_con(self, g2l, cidxmap, is_mortar,
                              lcidx, leidx):
        marked = set(zip(lcidx[is_mortar].tolist(),
                         leidx[is_mortar].tolist()))

        def local_eidx(cidx, geidx):
            etype, _ = cidxmap[int(cidx)]
            if etype not in g2l:
                return None

            gids, ordg, perm = g2l[etype]
            pos = np.searchsorted(ordg, geidx)
            if pos >= len(ordg) or ordg[pos] != geidx:
                return None

            return perm[pos]

        def participant_fields(dset):
            fields = dset.dtype.fields
            generic = {
                'left_cidx', 'left_eidx', 'right_cidx', 'right_eidx'
            }
            legacy = {
                'coarse_cidx', 'coarse_eidx', 'fine_cidx', 'fine_eidx'
            }
            if generic <= fields.keys():
                nright = fields['right_cidx'][0].shape[0]
                return (
                    ('left_cidx', 'left_eidx', None),
                    *(('right_cidx', 'right_eidx', i) for i in range(nright)),
                )
            if legacy <= fields.keys():
                nright = fields['fine_cidx'][0].shape[0]
                return (
                    ('coarse_cidx', 'coarse_eidx', None),
                    *(('fine_cidx', 'fine_eidx', i) for i in range(nright)),
                )
            raise ValueError('Unsupported mortar connectivity record')

        for name, dset in self.f['mortars'].items():
            refs = participant_fields(dset)
            local = []
            for source in dset:
                record = source.copy()
                mapped = []
                for cfield, efield, child in refs:
                    cidx = record[cfield]
                    geidx = record[efield]
                    if child is not None:
                        cidx = cidx[child]
                        geidx = geidx[child]
                    mapped.append(local_eidx(cidx, geidx))

                nlocal = sum(eidx is not None for eidx in mapped)
                if not nlocal:
                    continue
                if nlocal != len(mapped):
                    raise RuntimeError(
                        'Mortar participants span multiple partitions'
                    )

                for (cfield, efield, child), eidx in zip(refs, mapped):
                    if child is None:
                        record[efield] = eidx
                    else:
                        record[efield][child] = eidx
                local.append(record)

            if not local:
                continue

            records = np.asarray(local, dtype=dset.dtype)
            refs_marked = set()
            for rec in records:
                for cfield, efield, child in refs:
                    if child is None:
                        pair = (int(rec[cfield]), int(rec[efield]))
                    else:
                        pair = (
                            int(rec[cfield][child]), int(rec[efield][child])
                        )
                    refs_marked.add(pair)
            if not refs_marked <= marked:
                msg = 'Mortar metadata does not match marked faces'
                raise RuntimeError(msg)

            def attr(name, default=None):
                value = dset.attrs.get(name, default)
                if isinstance(value, bytes):
                    value = value.decode()
                return value

            self.mesh.mcon[name] = MortarConnectivity(
                name, records, cidxmap, format=attr('format'),
                template=attr('template')
            )

    def _construct_mpi_con(self, g2l, cetmap, cidxmap, lhs, rhs, stride):
        comm, rank, root = get_comm_rank_root()

        # Create a neighbourhood collective communicator
        ncomm = autofree(comm.Create_dist_graph_adjacent(self.neighbours,
                                                         self.neighbours))

        # Build a lookup to match (cidx, offset) pairs against lhs faces
        find = self._pair_finder(lhs, stride)

        # See which of our neighbours' unpaired faces we have
        matches = [rhs[find(u)] for u in ncomm.neighbor_allgather(rhs)]

        # Distribute this information back to our neighbours
        nmatches = ncomm.neighbor_alltoall(matches)

        etypes = self.mesh.etypes
        for nrank, nmatch in zip(self.neighbours, nmatches):
            # Find which of our lhs faces match this neighbour
            idx = find(nmatch)

            # Both ranks must agree on face ordering; sort by the
            # lower-ranked side so each rank's pairing is consistent
            ref = rhs if rank < nrank else lhs
            idx = idx[np.lexsort((ref.idx[idx], ref.cidx[idx]))]

            # Codec and element type indices for matched faces
            cidxs, etidxs = lhs.cidx[idx], cetmap[lhs.cidx[idx]]

            # Convert global element offsets to partition-local indices
            eidxs = np.empty(len(idx), dtype=int)
            for ti in np.unique(etidxs):
                ordgi, perm = g2l[etypes[ti]][1:]
                mask = etidxs == ti
                pos = np.searchsorted(ordgi, lhs.idx[idx[mask]])
                eidxs[mask] = perm[pos]

            self.mesh.con_p[nrank] = Connectivity(cidxs, eidxs, cidxmap)

    def _construct_shared_nodes(self):
        snf = SharedNodesFinder(self.eles, self.mesh.node_idxs,
                                self.mesh.node_valency)
        self.mesh.shared_nodes = snf.compute()
