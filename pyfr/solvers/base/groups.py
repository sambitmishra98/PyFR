from dataclasses import dataclass

import numpy as np

from pyfr.mortars import (
    DistributedPMortarFace, MPIMortarFaceKey, MortarGeometry, MortarGroup,
    MortarSide, RemoteMortarSide, _quad_face_geometry_at,
    _quad_full_face_reference_map, _runtime_face_corner_locs,
    _surface_affine_map,
)
from pyfr.mpiutil import autofree, get_comm_rank_root
from pyfr.quadrules import get_quadrule
from pyfr.shapes import QuadShape


@dataclass(frozen=True, order=True)
class ElementGroupKey:
    etype: str
    order: int

    @property
    def persistent_name(self):
        return f'p{self.order}-{self.etype}'

    def __str__(self):
        return self.persistent_name


class ElementGroupMap:
    def __init__(self, mesh, cfg):
        self.default_order = cfg.getint('solver', 'order')
        if self.default_order < 1:
            raise ValueError('Mixed-p solution orders must be >= 1')
        self.sections = [s for s in cfg.sections()
                         if s.startswith('solver-order-')]

        if not self.sections:
            raise ValueError('ElementGroupMap requires mixed-p configuration')


        orders = {
            etype: np.full(len(tags), self.default_order, dtype=np.int16)
            for etype, tags in mesh.tags.items()
        }
        assigned = {
            etype: np.zeros(len(tags), dtype=bool)
            for etype, tags in mesh.tags.items()
        }

        tags = [c.removeprefix('tag/') for c in mesh.codec
                if c.startswith('tag/')]

        for sect in self.sections:
            tag = cfg.get(sect, 'tag')
            order = cfg.getint(sect, 'order')
            if order < 1:
                raise ValueError('Mixed-p solution orders must be >= 1')
            if tag not in tags:
                raise ValueError(f'Unknown mixed-p mesh tag {tag!r}')

            tbit = np.uint64(1 << tags.index(tag))
            for etype, etags in mesh.tags.items():
                mask = (etags & tbit) != 0
                if np.any(mask & assigned[etype]):
                    raise ValueError('Mixed-p order override tags overlap')
                orders[etype][mask] = order
                assigned[etype][mask] = True

        comm, _, _ = get_comm_rank_root()
        local_active = sorted({int(p) for ps in orders.values() for p in ps})
        active = sorted({
            p for values in comm.allgather(local_active) for p in values
        })
        if len(active) != 2 or active[1] - active[0] != 1:
            raise ValueError(
                'V10C1A requires exactly two adjacent active solution orders'
            )

        self.orders = orders
        self.active_orders = tuple(active)
        self.global_active_orders = self.active_orders
        self.group_eidxs = {}
        self.group_global_eidxs = {}
        self.base_to_group = {}
        self.base_to_local = {}

        for etype in mesh.etypes:
            if etype not in orders:
                continue

            ps = orders[etype]
            geidxs = np.asarray(mesh.eidxs[etype])
            gmap = np.empty(len(ps), dtype=object)
            lmap = np.empty(len(ps), dtype=np.int64)

            for p in active:
                eidxs = np.flatnonzero(ps == p)
                if not len(eidxs):
                    continue

                key = ElementGroupKey(etype, p)
                self.group_eidxs[key] = eidxs
                self.group_global_eidxs[key] = geidxs[eidxs]
                gmap[eidxs] = key
                lmap[eidxs] = np.arange(len(eidxs))

            self.base_to_group[etype] = gmap
            self.base_to_local[etype] = lmap

        local_keys = [(key.etype, key.order) for key in self.group_eidxs]
        global_keys = {
            pair for values in comm.allgather(local_keys) for pair in values
        }
        self.global_keys = tuple(
            ElementGroupKey(*pair) for pair in sorted(global_keys)
        )

    @property
    def keys(self):
        return tuple(self.group_eidxs)

    def resolve(self, etype, eidx):
        return (self.base_to_group[etype][eidx],
                int(self.base_to_local[etype][eidx]))

    def split_region(self, ridxs):
        regions = {}

        for etype, eidxs in ridxs.items():
            if isinstance(eidxs, slice):
                eidxs = np.arange(len(self.base_to_group[etype]))[eidxs]
            else:
                eidxs = np.asarray(eidxs, dtype=np.int64)

            if not len(eidxs):
                continue

            gkeys = self.base_to_group[etype][eidxs]
            leidxs = self.base_to_local[etype][eidxs]

            for gkey in self.keys:
                if gkey.etype != etype:
                    continue

                mask = np.fromiter(
                    (key == gkey for key in gkeys), bool, len(gkeys)
                )
                if np.any(mask):
                    regions[gkey] = leidxs[mask]

        return regions

    def validate_solution_groups(self, groups):
        expected = {k.persistent_name: k for k in self.global_keys}
        actual = set(groups)

        if actual != set(expected):
            missing = sorted(set(expected) - actual)
            extra = sorted(actual - set(expected))
            raise RuntimeError(
                'Mixed-p restart group mismatch: '
                f'missing={missing}, unexpected={extra}'
            )

        for name, gkey in expected.items():
            group = groups[name]
            if group.etype != gkey.etype or group.order != gkey.order:
                raise RuntimeError(
                    f'Mixed-p restart identity mismatch for {name}'
                )

            expected_eidxs = self.group_global_eidxs.get(
                gkey, np.empty(0, dtype=np.int64)
            )
            if not np.array_equal(group.eidxs, expected_eidxs):
                raise RuntimeError(
                    f'Mixed-p restart element/order map mismatch for {name}'
                )

        return expected

    def remap_connectivity(self, con):
        cidxmap = {}
        cmap = {}
        cidxs = np.empty(len(con), dtype=np.int32)
        eidxs = np.empty(len(con), dtype=np.int64)

        for i, (cidx, eidx) in enumerate(zip(con.cidxs, con.eidxs)):
            etype, fidx = con.cidxmap[int(cidx)]
            gkey, geidx = self.resolve(etype, int(eidx))
            face = gkey, fidx

            if face not in cmap:
                ncidx = cmap[face] = len(cmap)
                cidxmap[ncidx] = face

            cidxs[i] = cmap[face]
            eidxs[i] = geidx

        return con.__class__(cidxs, eidxs, cidxmap)

    @staticmethod
    def _subset_connectivity(con, indices):
        indices = np.asarray(indices, dtype=np.intp)
        return con.__class__(con.cidxs[indices], con.eidxs[indices],
                             con.cidxmap)

    def split_internal(self, lhs, rhs):
        if len(lhs.cidxs) != len(rhs.cidxs):
            raise ValueError('Internal connectivity side lengths differ')

        same = []
        pgroups = {}
        for i, (lcidx, leidx, rcidx, reidx) in enumerate(zip(
            lhs.cidxs, lhs.eidxs, rhs.cidxs, rhs.eidxs
        )):
            letype, lfidx = lhs.cidxmap[int(lcidx)]
            retype, rfidx = rhs.cidxmap[int(rcidx)]
            lkey, lleidx = self.resolve(letype, int(leidx))
            rkey, rleidx = self.resolve(retype, int(reidx))

            if lkey.order == rkey.order:
                same.append(i)
                continue

            if letype != 'hex' or retype != 'hex':
                raise RuntimeError(
                    'V10C2A mixed-p interfaces support Hex/quad only'
                )

            key = lkey, rkey
            data = pgroups.setdefault(key, [[], [], [], []])
            data[0].append(int(lfidx))
            data[1].append(lleidx)
            data[2].append(int(rfidx))
            data[3].append(rleidx)

        slhs = self._subset_connectivity(lhs, same)
        srhs = self._subset_connectivity(rhs, same)
        same_con = (self.remap_connectivity(slhs),
                    self.remap_connectivity(srhs))

        mortar_groups = []
        for (lkey, rkey), (lfidxs, leidxs, rfidxs, reidxs) in pgroups.items():
            mortar_groups.append(MortarGroup(
                MortarSide(
                    lkey.etype, 'quad', tuple(lfidxs), tuple(leidxs),
                    lkey
                ),
                (MortarSide(
                    rkey.etype, 'quad', tuple(rfidxs), tuple(reidxs),
                    rkey
                ),),
            ))

        return same_con, tuple(mortar_groups)

    def _mpi_face_metadata(self, mesh, elemap, con):
        dtype = np.dtype([
            ('etype', np.int16), ('fidx', np.int16),
            ('geidx', np.int64), ('order', np.int16),
            ('gorder', np.int16),
            ('corners', np.float64, (4, mesh.ndims)),
        ])
        data = np.empty(len(con), dtype=dtype)

        for i, (cidx, eidx) in enumerate(zip(con.cidxs, con.eidxs)):
            etype, fidx = con.cidxmap[int(cidx)]
            gkey, leidx = self.resolve(etype, int(eidx))
            ele = elemap[gkey]

            data['etype'][i] = mesh.etypes.index(etype)
            data['fidx'][i] = fidx
            data['geidx'][i] = mesh.eidxs[etype][eidx]
            data['order'][i] = gkey.order
            data['gorder'][i] = ele.basis.nsptsord

            if etype == 'hex':
                data['corners'][i] = _runtime_face_corner_locs(
                    ele, leidx, int(fidx)
                )
            else:
                data['corners'][i] = np.nan

        return data

    def split_mpi(self, mesh, elemap, cfg):
        if not mesh.con_p:
            return {}, ()

        comm, rank, _ = get_comm_rank_root()
        neighbours = sorted(mesh.con_p)
        ncomm = autofree(comm.Create_dist_graph_adjacent(
            neighbours, neighbours
        ))

        local_meta = {
            nrank: self._mpi_face_metadata(mesh, elemap, mesh.con_p[nrank])
            for nrank in neighbours
        }
        remote_values = ncomm.neighbor_alltoall([
            local_meta[nrank] for nrank in neighbours
        ])
        remote_meta = dict(zip(neighbours, remote_values))

        same_con = {}
        pending = {nrank: [] for nrank in neighbours}
        pair_records = []
        geom_tol = cfg.getfloat(
            'solver-interfaces', 'mortar-geom-tol', 1e-10
        )

        for nrank in neighbours:
            con = mesh.con_p[nrank]
            lmeta, rmeta = local_meta[nrank], remote_meta[nrank]
            if len(lmeta) != len(rmeta):
                raise RuntimeError('Mixed-p MPI face pairing count mismatch')

            same = []
            pairs = []
            for i, (lm, rm) in enumerate(zip(lmeta, rmeta)):
                if int(lm['order']) == int(rm['order']):
                    same.append(i)
                    continue

                letype = mesh.etypes[int(lm['etype'])]
                retype = mesh.etypes[int(rm['etype'])]
                if letype != 'hex' or retype != 'hex':
                    raise RuntimeError(
                        'V10C3A mixed-p MPI supports Hex/quad only'
                    )
                if int(lm['gorder']) != int(rm['gorder']):
                    raise RuntimeError(
                        'V10C3A requires equal geometry order'
                    )
                if abs(int(lm['order']) - int(rm['order'])) != 1:
                    raise RuntimeError(
                        'V10C3A requires adjacent unequal solution orders'
                    )

                lkey = MPIMortarFaceKey(
                    letype, int(lm['geidx']), int(lm['fidx'])
                )
                rkey = MPIMortarFaceKey(
                    retype, int(rm['geidx']), int(rm['fidx'])
                )
                pair = tuple(sorted((lkey, rkey)))
                local_is_owner = lkey == pair[0]
                owner_rank = rank if local_is_owner else nrank
                if local_is_owner:
                    owner_corners, nonowner_corners = (
                        lm['corners'], rm['corners']
                    )
                else:
                    owner_corners, nonowner_corners = (
                        rm['corners'], lm['corners']
                    )
                rmap = _quad_full_face_reference_map(
                    owner_corners, nonowner_corners, geom_tol
                )

                etype, fidx = con.cidxmap[int(con.cidxs[i])]
                gkey, leidx = self.resolve(etype, int(con.eidxs[i]))
                local_side = MortarSide(
                    etype, 'quad', (int(fidx),), (leidx,), gkey
                )
                remote_side = RemoteMortarSide(
                    retype, 'quad', int(rm['order']), int(rm['gorder']),
                    int(rm['fidx']), int(rm['geidx'])
                )
                pending[nrank].append({
                    'local_side': local_side,
                    'remote_side': remote_side,
                    'local_key': lkey,
                    'remote_key': rkey,
                    'local_gorder': int(lm['gorder']),
                    'owner_rank': owner_rank,
                    'rmap': rmap,
                    'local_corners': np.array(lm['corners'], copy=True),
                    'remote_corners': np.array(rm['corners'], copy=True),
                })
                pairs.append((pair, owner_rank))

            subset = self._subset_connectivity(con, same)
            same_con[nrank] = self.remap_connectivity(subset)
            pair_records.append(self._pair_identity_records(mesh, pairs))

        remote_pair_records = ncomm.neighbor_alltoall(pair_records)
        for nrank, remote in zip(neighbours, remote_pair_records):
            local = self._pair_identity_records(
                mesh, [
                    (tuple(sorted((p['local_key'], p['remote_key']))),
                     p['owner_rank'])
                    for p in pending[nrank]
                ]
            )
            if not np.array_equal(local, remote):
                raise RuntimeError(
                    'Mixed-p MPI canonical face identity mismatch'
                )

        morder = max(self.global_active_orders)
        mqrule = get_quadrule(
            'quad', rule='gauss-legendre', npts=(morder + 2)**2
        )
        qcorners = QuadShape.std_ele(1)
        local_geom = []
        for nrank in neighbours:
            values = np.empty(
                (len(pending[nrank]), len(mqrule.pts), 2, mesh.ndims),
                dtype=float
            )
            for i, pdata in enumerate(pending[nrank]):
                local_is_owner = (
                    pdata['local_key'] ==
                    min(pdata['local_key'], pdata['remote_key'])
                )
                qpts = (
                    pdata['rmap'].apply(mqrule.pts)
                    if local_is_owner else mqrule.pts
                )
                ele = elemap[pdata['local_side'].runtime_key]
                ploc, pnorm = _quad_face_geometry_at(
                    ele, pdata['local_side'].eidxs[0],
                    pdata['local_side'].fidxs[0], qpts
                )
                values[i, :, 0] = ploc
                values[i, :, 1] = pnorm
            local_geom.append(values)

        remote_geom = ncomm.neighbor_alltoall(local_geom)
        faces = []
        for nidx, (nrank, rvalues) in enumerate(
            zip(neighbours, remote_geom)
        ):
            if len(rvalues) != len(pending[nrank]):
                raise RuntimeError('Mixed-p MPI geometry evidence mismatch')

            for pidx, (pdata, rgeom) in enumerate(
                zip(pending[nrank], rvalues)
            ):
                local_is_owner = (
                    pdata['local_key'] ==
                    min(pdata['local_key'], pdata['remote_key'])
                )
                lgeom = local_geom[nidx][pidx]
                if local_is_owner:
                    owner_geom, nonowner_geom = lgeom, rgeom
                    owner_corners = pdata['local_corners']
                    nonowner_corners = pdata['remote_corners']
                    owner_qpts = pdata['rmap'].apply(mqrule.pts)
                    nonowner_qpts = mqrule.pts
                else:
                    owner_geom, nonowner_geom = rgeom, lgeom
                    owner_corners = pdata['remote_corners']
                    nonowner_corners = pdata['local_corners']
                    owner_qpts = pdata['rmap'].apply(mqrule.pts)
                    nonowner_qpts = mqrule.pts

                oaff = _surface_affine_map(
                    qcorners, owner_corners, geom_tol
                )
                naff = _surface_affine_map(
                    qcorners, nonowner_corners, geom_tol
                )
                affine_error = max(
                    float(np.max(np.abs(
                        owner_geom[:, 0] - oaff(owner_qpts)
                    ))),
                    float(np.max(np.abs(
                        nonowner_geom[:, 0] - naff(nonowner_qpts)
                    ))),
                )
                if affine_error > geom_tol:
                    raise RuntimeError(
                        'V10C3A p-mortars require affine face geometry'
                    )

                geom_error = float(np.max(np.abs(
                    owner_geom[:, 0] - nonowner_geom[:, 0]
                )))
                normal_error = float(np.max(np.abs(
                    owner_geom[:, 1] + nonowner_geom[:, 1]
                )))
                if geom_error > geom_tol:
                    raise RuntimeError(
                        'V10C3A mixed-p MPI geometry mismatch'
                    )
                if normal_error > 10*geom_tol:
                    raise RuntimeError(
                        'V10C3A mixed-p MPI normal mismatch'
                    )

                geometry = MortarGeometry(
                    'owner', 'quad', np.asarray(mqrule.pts),
                    np.asarray(mqrule.wts),
                    (owner_geom[:, 0, :, None],),
                    (owner_geom[:, 1, :, None],),
                    (np.asarray([pdata['rmap'].determinant]),),
                    (np.asarray([geom_error]),),
                    (np.asarray([normal_error]),),
                )
                faces.append(DistributedPMortarFace(
                    nrank, pdata['local_side'], pdata['remote_side'],
                    pdata['local_key'], pdata['remote_key'],
                    pdata['local_gorder'], pdata['owner_rank'],
                    pdata['rmap'], geometry
                ))

        return same_con, tuple(faces)

    @staticmethod
    def _pair_identity_records(mesh, pairs):
        records = np.empty((len(pairs), 7), dtype=np.int64)
        for i, (pair, owner_rank) in enumerate(pairs):
            records[i] = (
                mesh.etypes.index(pair[0].etype), pair[0].global_eidx,
                pair[0].fidx, mesh.etypes.index(pair[1].etype),
                pair[1].global_eidx, pair[1].fidx, owner_rank,
            )
        return records

    def remap_internal(self, lhs, rhs):
        con, pgroups = self.split_internal(lhs, rhs)
        if pgroups:
            raise RuntimeError(
                'Mixed-p internal interface requires V10C2 p-mortar'
            )
        return con
