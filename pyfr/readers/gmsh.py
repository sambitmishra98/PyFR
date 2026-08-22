from collections import defaultdict
from functools import lru_cache
import re

import numpy as np

from pyfr.amr import (
    HexNodeStore, hex_affine_map, hex_child_face_nodes, hex_d4_transform,
    hex_face_axis_side, hex_order_quad2x2_faces, hex_refined_children,
    hex_rect_overlap, hex_root_face_uv, hex_transform_rect,
    hex_tree_cell_index, hex_tree_face_groups, hex_tree_face_pairs,
    hex_tree_leaves,
)
from pyfr.polys import get_polybasis
from pyfr.readers import BaseReader, NodalMeshAssembler
from pyfr.readers.base import _pyr_parallelogram_mask
from pyfr.shapes import PyrShape, TetShape, TriShape


@lru_cache(maxsize=None)
def _tet_jacobian_operator(order):
    spts = TetShape.std_ele(order)
    qpts = TetShape.std_ele(max(8, 2*order))
    basis = get_polybasis('tet', order, spts)

    return basis.jac_nodal_basis_at(qpts)


@lru_cache(maxsize=None)
def _tet_promotion_operator(srcorder, dstorder):
    srcpts = TetShape.std_ele(srcorder)
    dstpts = TetShape.std_ele(dstorder)
    basis = get_polybasis('tet', srcorder, srcpts)

    return basis.nodal_basis_at(dstpts)


def _affine_reference_map(src, dst, points):
    lhs = np.column_stack((src, np.ones(len(src))))
    coeff = np.linalg.solve(lhs, dst)

    points = np.asarray(points)
    return np.column_stack((points, np.ones(len(points)))) @ coeff


@lru_cache(maxsize=None)
def _pyramid_child_operator(porder, torder, childref):
    childref = np.asarray(childref).reshape(4, 3)
    mapped = _affine_reference_map(
        TetShape.std_ele(1), childref, TetShape.std_ele(torder)
    )
    basis = get_polybasis('pyr', porder, PyrShape.std_ele(porder))

    return basis.nodal_basis_at(mapped)


@lru_cache(maxsize=None)
def _pyramid_tri_operator(porder, torder, childref):
    childref = np.asarray(childref).reshape(3, 3)
    mapped = _affine_reference_map(
        TriShape.std_ele(1), childref, TriShape.std_ele(torder)
    )
    basis = get_polybasis('pyr', porder, PyrShape.std_ele(porder))

    return basis.nodal_basis_at(mapped)


def msh_section(mshit, section):
    endln = f'$End{section}\n'
    endix = int(next(mshit))

    for i, l in enumerate(mshit, start=1):
        if l == endln:
            raise ValueError(f'Unexpected end of section ${section}')

        yield l.strip()

        if i == endix:
            break
    else:
        raise ValueError('Unexpected EOF')

    if next(mshit) != endln:
        raise ValueError(f'Expected $End{section}')


class GmshReader(BaseReader):
    # Supported file types and extensions
    name = 'gmsh'
    extn = ['.msh']

    # Gmsh element types to PyFR type (petype) and node counts
    _etype_map = {
        1: ('line', 2), 8: ('line', 3), 26: ('line', 4), 27: ('line', 5),
        2: ('tri', 3), 9: ('tri', 6), 21: ('tri', 10), 23: ('tri', 15),
        3: ('quad', 4), 10: ('quad', 9), 36: ('quad', 16), 37: ('quad', 25),
        4: ('tet', 4), 11: ('tet', 10), 29: ('tet', 20), 30: ('tet', 35),
        5: ('hex', 8), 12: ('hex', 27), 92: ('hex', 64), 93: ('hex', 125),
        6: ('pri', 6), 13: ('pri', 18), 90: ('pri', 40), 91: ('pri', 75),
        7: ('pyr', 5), 14: ('pyr', 14), 118: ('pyr', 30), 119: ('pyr', 55)
    }

    # First-order node numbers associated with each element face
    _petype_fnmap = {
        'tri': {'line': [[0, 1], [1, 2], [2, 0]]},
        'quad': {'line': [[0, 1], [1, 2], [2, 3], [3, 0]]},
        'tet': {'tri': [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]},
        'hex': {'quad': [[0, 1, 2, 3], [0, 1, 4, 5], [1, 2, 5, 6],
                         [2, 3, 6, 7], [0, 3, 4, 7], [4, 5, 6, 7]]},
        'pri': {'quad': [[0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5]],
                'tri': [[0, 1, 2], [3, 4, 5]]},
        'pyr': {'quad': [[0, 1, 2, 3]],
                'tri': [[0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 3, 4]]}
    }

    # Mappings between the node ordering of PyFR and that of Gmsh
    _nodemaps = {
        ('tet', 4): [0, 1, 2, 3],
        ('tet', 10): [0, 4, 1, 6, 5, 2, 7, 9, 8, 3],
        ('tet', 20): [0, 4, 5, 1, 9, 16, 6, 8, 7, 2, 11, 17, 15, 18, 19, 13,
                      10, 14, 12, 3],
        ('tet', 35): [0, 4, 5, 6, 1, 12, 22, 24, 7, 11, 23, 8, 10, 9, 2, 15,
                      25, 26, 21, 28, 34, 32, 30, 33, 18, 14, 27, 20, 29, 31,
                      17, 13, 19, 16, 3],
        ('tet', 56): [0, 4, 5, 6, 7, 1, 15, 28, 33, 30, 8, 14, 31, 32, 9, 13,
                      29, 10, 12, 11, 2, 19, 34, 37, 35, 27, 40, 52, 53, 47,
                      45, 54, 50, 42, 48, 23, 18, 39, 38, 26, 43, 55, 49, 44,
                      51, 22, 17, 36, 25, 41, 46, 21, 16, 24, 20, 3],
        ('tet', 84): [
            0, 4, 5, 6, 7, 8, 1, 18, 34, 42, 41, 36, 9, 17, 37, 43, 40, 10, 16,
            38, 39, 11, 15, 35, 12, 14, 13, 2, 23, 44, 47, 48, 45, 33, 54, 74,
            78, 75, 65, 62, 80, 79, 69, 61, 76, 70, 56, 66, 28, 22, 52, 53, 49,
            32, 57, 81, 83, 68, 63, 82, 73, 60, 71, 27, 21, 51, 50, 31, 58, 77,
            67, 59, 72, 26, 20, 46, 30, 55, 64, 25, 19, 29, 24, 3
        ],
        ('pri', 6): [0, 1, 2, 3, 4, 5],
        ('pri', 18): [0, 6, 1, 7, 9, 2, 8, 15, 10, 16, 17, 11, 3, 12, 4, 13,
                      14, 5],
        ('pri', 40): [0, 6, 7, 1, 8, 24, 12, 9, 13, 2, 10, 26, 27, 14, 30, 38,
                      34, 33, 35, 16, 11, 29, 28, 15, 31, 39, 37, 32, 36, 17,
                      3, 18, 19, 4, 20, 25, 22, 21, 23, 5],
        ('pri', 75): [
            0, 6, 7, 8, 1, 9, 33, 35, 15, 10, 34, 16, 11, 17, 2, 12, 39, 43,
            40, 18, 48, 66, 69, 57, 55, 72, 61, 51, 58, 21, 13, 46, 47, 44,
            19, 52, 68, 71, 64, 56, 74, 65, 54, 62, 22, 14, 42, 45, 41, 20, 49,
            67, 70, 60, 53, 73, 63, 50, 59, 23, 3, 24, 25, 26, 4, 27, 36, 37,
            30, 28, 38, 31, 29, 32, 5
        ],
        ('pri', 126): [
            0, 6, 7, 8, 9, 1, 10, 42, 47, 44, 18, 11, 45, 46, 19, 12, 43, 20,
            13, 21, 2, 14, 54, 58, 59, 55, 22, 70, 102, 114, 106, 86, 81, 122,
            118, 90, 80, 110, 91, 73, 87, 26, 15, 65, 66, 67, 60, 23, 74, 104,
            116, 108, 97, 82, 124, 120, 98, 85, 112, 99, 79, 92, 27, 16, 64,
            69, 68, 61, 24, 75, 105, 117, 109, 96, 83, 125, 121, 101, 84, 113,
            100, 78, 93, 28, 17, 57, 63, 62, 56, 25, 71, 103, 115, 107, 89, 76,
            123, 119, 95, 77, 111, 94, 72, 88, 29, 3, 30, 31, 32, 33, 4, 34,
            48, 51, 49, 38, 35, 53, 52, 39, 36, 50, 40, 37, 41, 5
        ],
        ('pri', 196): [
            0, 6, 7, 8, 9, 10, 1, 11, 51, 59, 58, 53, 21, 12, 54, 60, 57, 22,
            13, 55, 56, 23, 14, 52, 24, 15, 25, 2, 16, 71, 75, 76, 77, 72, 26,
            96, 146, 161, 166, 151, 121, 111, 186, 191, 171, 125, 110, 181,
            176, 126, 109, 156, 127, 99, 122, 31, 17, 86, 87, 91, 88, 78, 27,
            100, 148, 163, 168, 153, 136, 112, 188, 193, 173, 137, 119, 183,
            178, 141, 115, 158, 138, 108, 128, 32, 18, 85, 94, 95, 92, 79, 28,
            101, 149, 164, 169, 154, 135, 116, 189, 194, 174, 144, 120, 184,
            179, 145, 118, 159, 142, 107, 129, 33, 19, 84, 90, 93, 89, 80, 29,
            102, 150, 165, 170, 155, 134, 113, 190, 195, 175, 140, 117, 185,
            180, 143, 114, 160, 139, 106, 130, 34, 20, 74, 83, 82, 81, 73, 30,
            97, 147, 162, 167, 152, 124, 103, 187, 192, 172, 133, 104, 182,
            177, 132, 105, 157, 131, 98, 123, 35, 3, 36, 37, 38, 39, 40, 4, 41,
            61, 64, 65, 62, 46, 42, 69, 70, 66, 47, 43, 68, 67, 48, 44, 63, 49,
            45, 50, 5
        ],
        ('pyr', 5): [0, 1, 3, 2, 4],
        ('pyr', 14): [0, 5, 1, 6, 13, 8, 3, 10, 2, 7, 9, 12, 11, 4],
        ('pyr', 30): [0, 5, 6, 1, 7, 25, 28, 11, 8, 26, 27, 12, 3, 16, 15, 2,
                      9, 21, 13, 22, 29, 23, 19, 24, 17, 10, 14, 20, 18, 4],
        ('pyr', 55): [0, 5, 6, 7, 1, 8, 41, 48, 44, 14, 9, 45, 49, 47, 15, 10,
                      42, 46, 43, 16, 3, 22, 21, 20, 2, 11, 29, 30, 17, 33, 50,
                      51, 35, 32, 53, 52, 36, 26, 39, 38, 23, 12, 31, 18, 34,
                      54, 37, 27, 40, 24, 13, 19, 28, 25, 4],
        ('pyr', 91): [
            0, 5, 6, 7, 8, 1, 9, 61, 72, 71, 64, 17, 10, 65, 73, 76, 70, 18,
            11, 66, 74, 75, 69, 19, 12, 62, 67, 68, 63, 20, 3, 28, 27, 26, 25,
            2, 13, 37, 40, 38, 21, 44, 77, 82, 78, 49, 46, 83, 90, 85, 52, 43,
            80, 87, 79, 50, 33, 56, 58, 55, 29, 14, 42, 41, 22, 47, 84, 86, 54,
            48, 89, 88, 53, 34, 59, 60, 30, 15, 39, 23, 45, 81, 51, 35, 57, 31,
            16, 24, 36, 32, 4
        ],
        ('pyr', 140): [
            0, 5, 6, 7, 8, 9, 1, 10, 85, 100, 99, 98, 88, 20, 11, 89, 101, 108,
            104, 97, 21, 12, 90, 105, 109, 107, 96, 22, 13, 91, 102, 106, 103,
            95, 23, 14, 86, 92, 93, 94, 87, 24, 3, 34, 33, 32, 31, 30, 2, 15,
            45, 48, 49, 46, 25, 56, 110, 115, 116, 111, 65, 59, 117, 135, 138,
            121, 68, 58, 118, 136, 137, 122, 69, 55, 113, 126, 125, 112, 66,
            40, 76, 79, 78, 75, 35, 16, 53, 54, 50, 26, 60, 119, 131, 123, 73,
            64, 132, 139, 133, 74, 63, 129, 134, 127, 70, 41, 80, 84, 83, 36,
            17, 52, 51, 27, 61, 120, 124, 72, 62, 130, 128, 71, 42, 81, 82, 37,
            18, 47, 28, 57, 114, 67, 43, 77, 38, 19, 29, 44, 39, 4
        ],
        ('hex', 8): [0, 1, 3, 2, 4, 5, 7, 6],
        ('hex', 27): [0, 8, 1, 9, 20, 11, 3, 13, 2, 10, 21, 12, 22, 26, 23, 15,
                      24, 14, 4, 16, 5, 17, 25, 18, 7, 19, 6],
        ('hex', 64): [
            0, 8, 9, 1, 10, 32, 35, 14, 11, 33, 34, 15, 3, 19, 18, 2, 12, 36,
            37, 16, 40, 56, 57, 44, 43, 59, 58, 45, 22, 49, 48, 20, 13, 39, 38,
            17, 41, 60, 61, 47, 42, 63, 62, 46, 23, 50, 51, 21, 4, 24, 25, 5,
            26, 52, 53, 28, 27, 55, 54, 29, 7, 31, 30, 6
        ],
        ('hex', 125): [
            0, 8, 9, 10, 1, 11, 44, 51, 47, 17, 12, 48, 52, 50, 18, 13, 45, 49,
            46, 19, 3, 25, 24, 23, 2, 14, 53, 57, 54, 20, 62, 98, 106, 99, 71,
            69, 107, 118, 109, 75, 65, 101, 111, 100, 72, 29, 81, 84, 80, 26,
            15, 60, 61, 58, 21, 66, 108, 119, 110, 78, 70, 120, 124, 121, 79,
            68, 113, 122, 112, 76, 30, 85, 88, 87, 27, 16, 56, 59, 55, 22, 63,
            102, 114, 103, 74, 67, 115, 123, 116, 77, 64, 105, 117, 104, 73,
            31, 82, 86, 83, 28, 4, 32, 33, 34, 5, 35, 89, 93, 90, 38, 36, 96,
            97, 94, 39, 37, 92, 95, 91, 40, 7, 43, 42, 41, 6
        ],
        ('hex', 216): [
            0, 8, 9, 10, 11, 1, 12, 56, 67, 66, 59, 20, 13, 60, 68, 71, 65, 21,
            14, 61, 69, 70, 64, 22, 15, 57, 62, 63, 58, 23, 3, 31, 30, 29, 28,
            2, 16, 72, 76, 77, 73, 24, 88, 152, 160, 161, 153, 104, 99, 162,
            184, 187, 166, 108, 98, 163, 185, 186, 167, 109, 91, 155, 171, 170,
            154, 105, 36, 121, 125, 124, 120, 32, 17, 83, 84, 85, 78, 25, 92,
            164, 188, 189, 168, 115, 100, 192, 208, 209, 196, 116, 103, 195,
            211, 210, 197, 117, 97, 174, 201, 200, 172, 110, 37, 126, 133, 132,
            131, 33, 18, 82, 87, 86, 79, 26, 93, 165, 191, 190, 169, 114, 101,
            193, 212, 213, 199, 119, 102, 194, 215, 214, 198, 118, 96, 175,
            202, 203, 173, 111, 38, 127, 134, 135, 130, 34, 19, 75, 81, 80, 74,
            27, 89, 156, 176, 177, 157, 107, 94, 178, 204, 205, 180, 113, 95,
            179, 207, 206, 181, 112, 90, 159, 183, 182, 158, 106, 39, 122, 128,
            129, 123, 35, 4, 40, 41, 42, 43, 5, 44, 136, 140, 141, 137, 48, 45,
            147, 148, 149, 142, 49, 46, 146, 151, 150, 143, 50, 47, 139, 145,
            144, 138, 51, 7, 55, 54, 53, 52, 6
        ],
        ('tri', 3): [0, 1, 2],
        ('tri', 6): [0, 3, 1, 5, 4, 2],
        ('tri', 10): [0, 3, 4, 1, 8, 9, 5, 7, 6, 2],
        ('tri', 15): [0, 3, 4, 5, 1, 11, 12, 13, 6, 10, 14, 7, 9, 8, 2],
        ('tri', 21): [0, 3, 4, 5, 6, 1, 14, 15, 16, 17, 7, 13, 20, 18, 8, 12,
                      19, 9, 11, 10, 2],
        ('quad', 4): [0, 1, 3, 2],
        ('quad', 9): [0, 4, 1, 7, 8, 5, 3, 6, 2],
        ('quad', 16): [0, 4, 5, 1, 11, 12, 13, 6, 10, 15, 14, 7, 3, 9, 8, 2],
        ('quad', 25): [0, 4, 5, 6, 1, 15, 16, 20, 17, 7, 14, 23, 24, 21, 8, 13,
                       19, 22, 18, 9, 3, 12, 11, 10, 2],
        ('quad', 36): [0, 4, 5, 6, 7, 1, 19, 20, 24, 25, 21, 8, 18, 31, 32, 33,
                       26, 9, 17, 30, 35, 34, 27, 10, 16, 23, 29, 28, 22, 11,
                       3, 15, 14, 13, 12, 2]
    }

    def __init__(self, msh, progress):
        super().__init__(progress)

        if isinstance(msh, str):
            msh = open(msh)

        with progress.start_with_spinner('Reading .msh') as pspinner:
            # Get an iterator over the lines of the mesh
            mshit = iter(msh)

            # Have our spinner flashed every 10,000 lines
            mshit = pspinner.wrap_file_lines(mshit, 10000)

            # Section readers
            sect_map = {
                'MeshFormat': self._read_mesh_format,
                'PhysicalNames': self._read_phys_names,
                'Entities': self._read_entities,
                'Nodes': self._read_nodes,
                'Elements': self._read_eles
            }

            for l in filter(lambda l: l != '\n', mshit):
                # Ensure we have encountered a section
                if not l.startswith('$'):
                    raise ValueError('Expected a mesh section')

                # Strip the '$' and '\n' to get the section name
                sect = l[1:-1]

                # Try to read the section
                try:
                    sect_map[sect](mshit)
                # Else skip over it
                except KeyError:
                    endsect = f'$End{sect}\n'

                    for el in mshit:
                        if el == endsect:
                            break
                    else:
                        raise ValueError(f'Expected $End{sect}')

        # Account for any starting node offsets
        for k, v in self._elenodes.items():
            v -= self._nodeoff


    def _begin_nodepts_append(self):
        self._pending_nodepts = []
        self._pending_nnodes = 0

    def _append_nodepts(self, points):
        points = np.asarray(points, dtype=float)
        pending = getattr(self, '_pending_nodepts', None)
        if pending is None:
            start = len(self._nodepts)
            self._nodepts = np.vstack((self._nodepts, points))
        else:
            start = len(self._nodepts) + self._pending_nnodes
            pending.append(points)
            self._pending_nnodes += len(points)

        return np.arange(start, start + len(points), dtype=np.int64)

    def _commit_nodepts_append(self):
        if self._pending_nodepts:
            self._nodepts = np.vstack(
                (self._nodepts, *self._pending_nodepts)
            )

        del self._pending_nodepts
        del self._pending_nnodes

    @staticmethod
    def _affine_map(src, dst):
        lhs = np.column_stack((src, np.ones(len(src))))
        coeff = np.linalg.solve(lhs, dst)

        def apply(points):
            points = np.asarray(points)
            return np.column_stack((points, np.ones(len(points)))) @ coeff

        return apply

    @staticmethod
    def _generated_tet_storage(order):
        npts = TetShape.npts_from_order(order)
        corners = TetShape.corner_pts_idxs(npts)
        mask = np.ones(npts, dtype=bool)
        mask[corners] = False
        storage = np.concatenate((corners, np.flatnonzero(mask)))

        return storage, np.argsort(storage)

    def _prepare_generated_tets(self, order):
        npts = TetShape.npts_from_order(order)
        etype = -npts
        storage, nodemap = self._generated_tet_storage(order)

        # These are instance-local copies.  The generated element type is an
        # importer-internal representation whose first four entries are its
        # corners and whose remaining entries follow PyFR's canonical order.
        self._etype_map = dict(self._etype_map)
        self._nodemaps = dict(self._nodemaps)
        self._etype_map[etype] = ('tet', npts)
        self._nodemaps['tet', npts] = nodemap.tolist()

        return etype, storage

    def _promote_tet(self, nodes, order, nodemaps, storage):
        nnodes = len(nodes)
        srcorder = TetShape.order_from_npts(nnodes)
        srcpts = TetShape.std_ele(srcorder)
        dstpts = TetShape.std_ele(order)

        srcids = nodes[nodemaps['tet', nnodes]]
        srccoords = self._nodepts[srcids]

        if srcorder == order:
            dstids = srcids.copy()
        else:
            dstcoords = (
                _tet_promotion_operator(srcorder, order) @ srccoords
            )

            dstids = np.empty(len(dstpts), dtype=np.int64)
            srcorners = TetShape.corner_pts_idxs(nnodes)
            dstcorners = TetShape.corner_pts_idxs(len(dstpts))
            dstids[dstcorners] = srcids[srcorners]

            mask = np.ones(len(dstpts), dtype=bool)
            mask[dstcorners] = False
            dstids[mask] = self._append_nodepts(dstcoords[mask])

        return dstids[storage]

    def _promote_existing_tets(self, order, etype, nodemaps, storage):
        promoted = defaultdict(list)
        for key in list(self._elenodes):
            srctype, pents = key
            if self._etype_map[srctype][0] != 'tet':
                continue

            promoted[pents].extend(
                self._promote_tet(row, order, nodemaps, storage)
                for row in self._elenodes.pop(key)
            )

        for pents, rows in promoted.items():
            self._elenodes[etype, pents] = np.asarray(rows, dtype=np.int64)

    def _pyramid_child_tet_coords(self, parent, corners, order):
        pmap = self._nodemaps['pyr', len(parent)]
        pids = parent[pmap]
        pcoords = self._nodepts[pids]
        porder = PyrShape.order_from_npts(len(parent))

        if order < 2*porder:
            raise ValueError(
                f'Pyramid order {porder} requires tetrahedral geometry '
                f'order at least {2*porder}; received {order}'
            )

        pref = PyrShape.std_ele(1)
        gref = pref[np.argsort(self._nodemaps['pyr', 5])]
        refbyid = {int(node): point for node, point in zip(parent[:5], gref)}
        childref = np.asarray([refbyid[int(node)] for node in corners])

        operator = _pyramid_child_operator(
            porder, order, tuple(childref.ravel())
        )
        return operator @ pcoords

    def _store_pyramid_child_tet(self, coords, corners, order, storage):
        tpts = TetShape.std_ele(order)

        ids = np.empty(len(tpts), dtype=np.int64)
        tcorners = TetShape.corner_pts_idxs(len(tpts))
        ids[tcorners] = corners

        mask = np.ones(len(tpts), dtype=bool)
        mask[tcorners] = False
        ids[mask] = self._append_nodepts(coords[mask])

        return ids[storage]

    @staticmethod
    def _tet_geometry_quality(coords, order):
        """Return a sampled scale-independent minimum Jacobian."""
        jacop = _tet_jacobian_operator(order)
        jac = np.einsum('dnp,nq->pdq', jacop, coords)
        mindet = np.linalg.det(jac).min()

        corners = coords[TetShape.corner_pts_idxs(len(coords))]
        scale = max(
            np.linalg.norm(a - b)
            for i, a in enumerate(corners)
            for b in corners[i + 1:]
        )

        return mindet/scale**3

    def _orient_tet(self, corners):
        a, b, c, d = (self._nodepts[n] for n in corners)
        det = np.linalg.det(np.column_stack((b - a, c - a, d - a)))
        if det < 0:
            corners = [corners[0], corners[2], corners[1], corners[3]]
            det = -det
        return corners, det

    @staticmethod
    def _as_pent_tuple(pents):
        return pents if isinstance(pents, tuple) else (pents,)

    def _boundary_quad_lookup(self):
        """Return fixed/periodic boundary quad faces keyed by corner IDs."""
        fixed = set(getattr(self, '_bfacespents', {}).values())
        periodic = {
            pent
            for pair in getattr(self, '_pfacespents', {}).values()
            for pent in pair
        }

        lookup = {}
        for ekey, rows in self._elenodes.items():
            etype, pents = ekey
            if self._etype_map[etype][0] != 'quad':
                continue

            epents = set(self._as_pent_tuple(pents))
            if epents & fixed:
                kind = 'fixed'
            elif epents & periodic:
                kind = 'periodic'
            else:
                continue

            for eidx, row in enumerate(rows):
                fkey = tuple(sorted(int(n) for n in row[:4]))
                if fkey in lookup:
                    raise ValueError(
                        'Duplicate boundary quadrilateral face for pyramid '
                        f'splitting: {fkey}'
                    )
                lookup[fkey] = (kind, ekey, eidx)

        return lookup

    def _boundary_tri_etype(self, pents, order):
        """Select the complete triangle type for a boundary pyramid."""
        try:
            etype = {1: 2, 2: 9, 3: 21, 4: 23}[order]
        except KeyError:
            raise ValueError(
                'Boundary pyramid splitting does not have a Gmsh triangle '
                f'representation for geometry order {order}'
            ) from None

        nnodes = self._etype_map[etype][1]
        if ('tri', nnodes) not in self._nodemaps:
            raise ValueError(
                'Boundary pyramid splitting requires a complete supported '
                f'triangle element; found Gmsh type {etype}'
            )

        return etype

    def _boundary_tri_nodes(self, parent, corners, etype):
        """Build a tagged boundary triangle from one pyramid base half."""
        corners = [int(n) for n in corners]
        nnodes = self._etype_map[etype][1]
        order = TriShape.order_from_npts(nnodes)
        tpts = TriShape.std_ele(order)

        tcoords = self._pyramid_tri_coords(parent, corners, tpts)

        tids = np.empty(nnodes, dtype=np.int64)
        tcorners = TriShape.corner_pts_idxs(nnodes)
        tids[tcorners] = corners
        mask = np.ones(nnodes, dtype=bool)
        mask[tcorners] = False
        tids[mask] = self._append_nodepts(tcoords[mask])

        tmap = self._nodemaps['tri', nnodes]
        return tids[np.argsort(tmap)]

    def _pyramid_tri_coords(self, parent, corners, tpts):
        corners = [int(n) for n in corners]

        pmap = self._nodemaps['pyr', len(parent)]
        pids = parent[pmap]
        pcoords = self._nodepts[pids]
        porder = PyrShape.order_from_npts(len(parent))

        pref = PyrShape.std_ele(1)
        gref = pref[np.argsort(self._nodemaps['pyr', 5])]
        refbyid = {int(node): point for node, point in zip(parent[:5], gref)}
        childref = np.asarray([refbyid[node] for node in corners])
        torder = TriShape.order_from_npts(len(tpts))
        operator = _pyramid_tri_operator(
            porder, torder, tuple(childref.ravel())
        )
        return operator @ pcoords

    def _pyramid_candidates(self, nodes, order):
        v = [int(n) for n in nodes[:5]]
        specs = [
            (0, (v[0], v[2]), [[v[0], v[1], v[2], v[4]],
                                [v[0], v[2], v[3], v[4]]]),
            (1, (v[1], v[3]), [[v[0], v[1], v[3], v[4]],
                                [v[1], v[2], v[3], v[4]]]),
        ]

        candidates = {}
        for diagonal, edge, ccorners in specs:
            oriented, childcoords = [], []
            for corners in ccorners:
                corners, _ = self._orient_tet(corners)
                oriented.append(corners)
                childcoords.append(
                    self._pyramid_child_tet_coords(nodes, corners, order)
                )

            quality = min(
                self._tet_geometry_quality(coords, order)
                for coords in childcoords
            )
            edge = tuple(sorted(edge))
            candidates[edge] = {
                'edge': edge,
                'diagonal': diagonal,
                'corners': oriented,
                'coords': childcoords,
                'quality': quality,
            }

        return candidates

    def _paired_pyramid_geometry_error(self, left, right, edge):
        lcandidate = left['candidates'][edge]
        rcandidate = right['candidates'][edge]
        ltris = {tuple(sorted(c[:3])) for c in lcandidate['corners']}
        rtris = {tuple(sorted(c[:3])) for c in rcandidate['corners']}
        if ltris != rtris:
            raise ValueError('Paired pyramids do not produce matching faces')

        lorder = PyrShape.order_from_npts(len(left['nodes']))
        rorder = PyrShape.order_from_npts(len(right['nodes']))
        tpts = TriShape.std_ele(2*max(lorder, rorder))

        return max(
            np.max(np.abs(
                self._pyramid_tri_coords(left['nodes'], tri, tpts) -
                self._pyramid_tri_coords(right['nodes'], tri, tpts)
            ))
            for tri in ltris
        )

    @staticmethod
    def _point_set_error(left, right):
        distances = np.max(
            np.abs(left[:, None] - right[None, :]), axis=2
        )
        return max(
            distances.min(axis=0).max(),
            distances.min(axis=1).max(),
        )

    def _select_periodic_pyramid_diagonals(self, records):
        periodic_pairs = []
        pending = [
            r for r in records
            if r['kind'] == 'boundary' and r['boundary'][0] == 'periodic'
        ]

        for pname, (lpent, rpent) in getattr(
            self, '_pfacespents', {}
        ).items():
            left = [
                r for r in pending
                if lpent in self._as_pent_tuple(r['boundary'][1][1])
            ]
            right = [
                r for r in pending
                if rpent in self._as_pent_tuple(r['boundary'][1][1])
            ]
            if len(left) != len(right):
                raise ValueError(
                    f'Periodic pyramid boundary {pname} has {len(left)} '
                    f'left faces and {len(right)} right faces'
                )
            if not left:
                continue

            lcent = np.array([
                self._nodepts[list(r['quad'])].mean(axis=0) for r in left
            ])
            rcent = np.array([
                self._nodepts[list(r['quad'])].mean(axis=0) for r in right
            ])
            lfidx, rfidx, _, trans = NodalMeshAssembler._pair_translational(
                lcent, rcent
            )

            for li, ri in zip(lfidx, rfidx):
                lrec, rrec = left[li], right[ri]
                choices = []
                for ledge, lcandidate in lrec['candidates'].items():
                    lpts = self._nodepts[list(ledge)] + trans
                    for redge, rcandidate in rrec['candidates'].items():
                        rpts = self._nodepts[list(redge)]
                        error = min(
                            np.max(np.abs(lpts - rpts)),
                            np.max(np.abs(lpts - rpts[::-1])),
                        )
                        choices.append((
                            error,
                            min(lcandidate['quality'], rcandidate['quality']),
                            ledge,
                            redge,
                        ))

                compatible = [c for c in choices if c[0] <= 1e-10]
                if not compatible:
                    raise ValueError(
                        f'Periodic pyramid boundary {pname} does not expose '
                        'matching base diagonals'
                    )
                _, quality, ledge, redge = max(
                    compatible, key=lambda c: (c[1], c[2], c[3])
                )
                lrec['selected'] = lrec['candidates'][ledge]
                rrec['selected'] = rrec['candidates'][redge]

                lorder = PyrShape.order_from_npts(len(lrec['nodes']))
                rorder = PyrShape.order_from_npts(len(rrec['nodes']))
                tpts = TriShape.std_ele(2*max(lorder, rorder))
                ltris = lrec['selected']['corners']
                rtris = rrec['selected']['corners']
                lcoords = [
                    self._pyramid_tri_coords(lrec['nodes'], t[:3], tpts)
                    + trans for t in ltris
                ]
                rcoords = [
                    self._pyramid_tri_coords(rrec['nodes'], t[:3], tpts)
                    for t in rtris
                ]
                geom_error = 0.0
                for lc in lcoords:
                    errors = [
                        self._point_set_error(lc, rc) for rc in rcoords
                    ]
                    geom_error = max(geom_error, min(errors))
                if geom_error > 1e-10:
                    raise ValueError(
                        f'Periodic pyramid boundary {pname} geometry '
                        f'mismatch {geom_error:.3e} exceeds tolerance '
                        '1.000e-10'
                    )

                periodic_pairs.append({
                    'name': pname,
                    'left_quad': lrec['qkey'],
                    'right_quad': rrec['qkey'],
                    'left_diagonal_edge': ledge,
                    'right_diagonal_edge': redge,
                    'quality': quality,
                    'geometry_error': geom_error,
                    'translation': trans,
                })

        if any('selected' not in record for record in pending):
            raise ValueError('Unmatched periodic pyramid boundary face')

        return periodic_pairs

    @staticmethod
    def _parse_hex_refine_selector(selector):
        selector = selector.strip()
        if selector == 'all':
            return None

        requested = set()
        try:
            for token in selector.split(','):
                parts = token.strip().split('/')
                if not token.strip() or any(not p for p in parts):
                    raise ValueError

                tag = int(parts[0])
                path = tuple(int(p) for p in parts[1:])
                requested.add((tag, path))
        except ValueError:
            raise ValueError(
                'Hex refinement selector must be "all" or comma-separated '
                'Gmsh tags with optional /0../7 octant paths'
            ) from None

        if not requested or any(tag <= 0 for tag, _ in requested):
            raise ValueError('Hex refinement requires positive Gmsh tags')
        if any(octant not in range(8) for _, path in requested
               for octant in path):
            raise ValueError('Hex refinement octants must be in 0..7')

        return frozenset(requested)

    def _volume_face_lookup(self):
        volpents = set(self._volpents.values())
        lookup = defaultdict(list)

        for ekey, rows in self._elenodes.items():
            etype, pents = ekey
            if not set(self._as_pent_tuple(pents)) & volpents:
                continue

            petype, nnodes = self._etype_map[etype]
            fnmap = self._petype_fnmap[petype].get('quad', ())
            for eidx, row in enumerate(rows):
                for fidx, fmap in enumerate(fnmap):
                    qkey = tuple(sorted(int(n) for n in row[fmap]))
                    lookup[qkey].append(
                        (ekey, eidx, fidx, petype, nnodes)
                    )

        return lookup

    # D5B1: the octree mathematics below is extracted to `pyfr.amr` so a
    # native materializer can share it. These stay as thin adapters that
    # convert between Gmsh's own file-order node layout (as read into
    # `_elenodes` rows) and the shared core's canonical PyFR Hex8 order,
    # preserving accepted D1/D2 observable behaviour exactly - see
    # `d5b1/crosscheck.py` evidence in the AMR-M2 build report for the
    # verification this refactor was checked against.

    def _hex_node_store(self):
        return HexNodeStore(
            coords=lambda ids: self._nodepts[np.asarray(ids)],
            allocate=lambda pts: self._append_nodepts(pts),
        )

    def _hex_affine_map(self, nodes, tol=1e-10):
        pids = nodes[self._nodemaps['hex', 8]]
        return pids, hex_affine_map(pids, self._hex_node_store(), tol=tol)

    def _refined_hex_children(self, nodes, node_cache, coord_cache):
        nodemap = self._nodemaps['hex', 8]
        pids = nodes[nodemap]
        children = hex_refined_children(
            pids, self._hex_node_store(), node_cache, coord_cache
        )
        invmap = np.argsort(nodemap)
        return [
            (ix, iy, iz, np.asarray(row)[invmap])
            for ix, iy, iz, row in children
        ]

    def _hex_child_face_nodes(self, children, fidx):
        fmap = self._petype_fnmap['hex']['quad'][fidx]
        return hex_child_face_nodes(children, fidx, fmap)

    def _order_quad_2x2_faces(self, coarse, fidx, fine, tol=1e-10):
        pids = coarse[self._nodemaps['hex', 8]]
        return hex_order_quad2x2_faces(
            pids, fidx, fine, self._hex_node_store(), tol=tol
        )

    @staticmethod
    def _hex_tree_cell_index(path):
        return hex_tree_cell_index(path)

    @staticmethod
    def _hex_face_axis_side(fidx):
        return hex_face_axis_side(fidx)

    def _hex_root_face_uv(self, row, fidx):
        pids = row[self._nodemaps['hex', 8]]
        return hex_root_face_uv(pids, fidx)

    @staticmethod
    def _hex_d4_transform(src, dst):
        return hex_d4_transform(src, dst)

    @staticmethod
    def _hex_transform_rect(rect, transform):
        return hex_transform_rect(rect, transform)

    def _hex_root_topology(self, eligible):
        faces = self._volume_face_lookup()
        boundary = self._boundary_quad_lookup()
        etotag = {ele: tag for tag, ele in eligible.items()}
        rootfaces = {}

        for tag, (ekey, eidx) in eligible.items():
            row = self._elenodes[ekey][eidx]
            for fidx, fmap in enumerate(self._petype_fnmap['hex']['quad']):
                qnodes = tuple(int(n) for n in row[fmap])
                qkey = tuple(sorted(qnodes))
                owners = faces[qkey]

                if len(owners) == 2:
                    other = owners[0]
                    if other[:2] == (ekey, eidx):
                        other = owners[1]

                    if other[3:] == ('hex', 8) and other[:2] in etotag:
                        kind = 'interior'
                        info = (etotag[other[:2]], other[2])
                    else:
                        kind = 'unsupported'
                        info = other
                elif len(owners) == 1:
                    bmatch = boundary.get(qkey)
                    if bmatch is None:
                        kind, info = 'unmatched', None
                    else:
                        kind, bkey, bidx = bmatch
                        info = bkey, bidx
                else:
                    kind, info = 'nonmanifold', None

                rootfaces[tag, fidx] = {
                    'qkey': qkey, 'kind': kind, 'info': info,
                    'transform': (1, 0, 0, 0, 1, 0),
                }

        byqkey = defaultdict(list)
        for side, face in rootfaces.items():
            byqkey[face['qkey']].append(side)

        shared = {}
        for qkey, sides in byqkey.items():
            interior = [
                side for side in sides
                if rootfaces[side]['kind'] == 'interior'
            ]
            if interior and len(interior) != 2:
                for side in sides:
                    rootfaces[side]['kind'] = 'unsupported'
                continue
            if interior:
                shared[qkey] = interior

        for qkey, sides in shared.items():
            owner = min(sides)
            okey, oeidx = eligible[owner[0]]
            ouv = self._hex_root_face_uv(
                self._elenodes[okey][oeidx], owner[1]
            )
            for tag, fidx in sides:
                ekey, eidx = eligible[tag]
                suv = self._hex_root_face_uv(
                    self._elenodes[ekey][eidx], fidx
                )
                rootfaces[tag, fidx]['transform'] = \
                    self._hex_d4_transform(suv, ouv)

        return rootfaces

    def _validate_hex_tree_roots(self, roots, eligible, rootfaces):
        for tag in sorted(roots):
            ekey, eidx = eligible[tag]
            self._hex_affine_map(self._elenodes[ekey][eidx])

            for fidx in range(6):
                face = rootfaces[tag, fidx]
                kind, info = face['kind'], face['info']
                if kind == 'interior':
                    continue
                if kind == 'periodic':
                    raise ValueError(
                        'V10D2 does not refine periodic boundary cells'
                    )
                if kind == 'fixed':
                    bkey, _ = info
                    if self._etype_map[bkey[0]] != ('quad', 4):
                        raise ValueError(
                            'V10D2 requires Quad4 on refined boundaries'
                        )
                    continue
                if kind == 'unsupported':
                    raise ValueError(
                        'V10D2 refined Hex faces require an unrefined '
                        'Hex8 neighbour'
                    )
                if kind == 'unmatched':
                    raise ValueError(
                        'Selected Hex8 has an unmatched exterior face'
                    )
                raise ValueError(
                    'Selected Hex8 has a non-manifold quadrilateral face'
                )

    @staticmethod
    def _hex_tree_leaves(eligible, split):
        return hex_tree_leaves(eligible, split)

    def _hex_tree_face_groups(self, leaves, rootfaces):
        return hex_tree_face_groups(leaves, rootfaces)

    @staticmethod
    def _hex_rect_overlap(a, b):
        return hex_rect_overlap(a, b)

    def _hex_tree_face_pairs(self, groups, rootfaces):
        rootkinds = {
            ('root', face['qkey']): face['kind']
            for face in rootfaces.values()
        }
        yield from hex_tree_face_pairs(groups, rootkinds)

    def _balance_hex_tree(self, split, eligible, rootfaces):
        split = set(split)
        while True:
            leaves = self._hex_tree_leaves(eligible, split)
            groups = self._hex_tree_face_groups(leaves, rootfaces)
            added = set()

            for lface, rface in self._hex_tree_face_pairs(
                groups, rootfaces
            ):
                llevel, rlevel = lface['level'], rface['level']
                if abs(llevel - rlevel) > 1:
                    coarse = lface if llevel < rlevel else rface
                    added.add(coarse['leaf'])

            if not added:
                return split, leaves, groups

            split.update(added)

    def _materialize_hex_tree(self, eligible, split):
        leafrows = {}
        node_cache, coord_cache = {}, {}

        def walk(tag, path, row):
            if (tag, path) not in split:
                leafrows[tag, path] = row
                return

            children = self._refined_hex_children(
                row, node_cache, coord_cache
            )
            self._commit_nodepts_append()
            self._begin_nodepts_append()
            for ix, iy, iz, child in children:
                octant = ix + 2*iy + 4*iz
                walk(tag, path + (octant,), child)

        self._begin_nodepts_append()
        try:
            for tag in sorted(eligible):
                ekey, eidx = eligible[tag]
                row = self._elenodes[ekey][eidx]
                if (tag, ()) in split:
                    walk(tag, (), row)
                else:
                    leafrows[tag, ()] = row
            self._commit_nodepts_append()
        except:
            if hasattr(self, '_pending_nodepts'):
                del self._pending_nodepts
                del self._pending_nnodes
            raise

        return leafrows

    def refine_hexes(self, selector):
        """Recursively refine selected affine Gmsh Hex8 octree cells."""
        if getattr(self, '_hex_refine_applied', False):
            return self._hex_refine_summary
        if getattr(self, '_pyramid_split_applied', False):
            raise ValueError(
                'V10D2 Hex refinement cannot be combined with pyramid '
                'splitting'
            )

        requested = self._parse_hex_refine_selector(selector)
        volpents = set(self._volpents.values())
        eligible = {}
        taginfo = {}
        for ekey, rows in self._elenodes.items():
            etype, pents = ekey
            tags = self._eletags.get(ekey)
            if tags is None:
                continue

            isvol = bool(set(self._as_pent_tuple(pents)) & volpents)
            petype, nnodes = self._etype_map[etype]
            for eidx, tag in enumerate(tags):
                taginfo[int(tag)] = (ekey, eidx, isvol, petype, nnodes)
                if isvol and petype == 'hex' and nnodes == 8:
                    eligible[int(tag)] = (ekey, eidx)

        if requested is None:
            explicit_roots = set(eligible)
            split = {(tag, ()) for tag in explicit_roots}
        else:
            explicit_roots = {tag for tag, _ in requested}
            missing = explicit_roots - taginfo.keys()
            if missing:
                raise ValueError(
                    f'Unknown Gmsh element tags for Hex refinement: '
                    f'{sorted(missing)}'
                )
            invalid = explicit_roots - eligible.keys()
            if invalid:
                raise ValueError(
                    'V10D2 can refine only complete volume Hex8 elements; '
                    f'invalid tags {sorted(invalid)}'
                )

            split = set()
            for tag, path in requested:
                for depth in range(len(path) + 1):
                    split.add((tag, path[:depth]))

        if not split:
            self._hex_refine_mortars = []
            self._hex_refine_applied = True
            self._hex_refine_summary = {
                'selected': 0, 'generated-hexes': 0, 'mortars': 0,
                'boundary-splits': 0, 'conforming-refined-faces': 0,
            }
            return self._hex_refine_summary

        rootfaces = self._hex_root_topology(eligible)
        split, leaves, groups = self._balance_hex_tree(
            split, eligible, rootfaces
        )
        refined_roots = {tag for tag, path in split if not path}
        self._validate_hex_tree_roots(
            refined_roots, eligible, rootfaces
        )
        leafrows = self._materialize_hex_tree(eligible, split)

        mortar_fines = defaultdict(list)
        mortar_faces = {}
        mortar_surfaces = set()
        for lface, rface in self._hex_tree_face_pairs(groups, rootfaces):
            dl = lface['level'] - rface['level']
            if dl == 0:
                lrow = leafrows[lface['leaf']]
                rrow = leafrows[rface['leaf']]
                lfmap = self._petype_fnmap['hex']['quad'][lface['fidx']]
                rfmap = self._petype_fnmap['hex']['quad'][rface['fidx']]
                if tuple(sorted(int(n) for n in lrow[lfmap])) != tuple(
                    sorted(int(n) for n in rrow[rfmap])
                ):
                    raise ValueError(
                        'Same-level Hex tree neighbours are not conforming'
                    )
                continue
            if abs(dl) != 1:
                raise ValueError('Unbalanced Hex tree face escaped closure')

            coarse, fine = (
                (rface, lface) if dl > 0 else (lface, rface)
            )
            ckey = coarse['leaf'], coarse['fidx']
            mortar_faces[ckey] = coarse
            mortar_fines[ckey].append(fine)
            mortar_surfaces.add(coarse['surface'])

        mortars = []
        for ckey in sorted(mortar_fines, key=repr):
            coarse = mortar_faces[ckey]
            fine = mortar_fines[ckey]
            if len(fine) != 4:
                raise ValueError(
                    'Balanced Hex tree mortar does not have four fine faces'
                )

            crow = leafrows[coarse['leaf']]
            faces = []
            for fface in fine:
                frow = leafrows[fface['leaf']]
                fmap = self._petype_fnmap['hex']['quad'][fface['fidx']]
                faces.append(tuple(int(n) for n in frow[fmap]))

            ordered = self._order_quad_2x2_faces(
                crow, coarse['fidx'], faces
            )
            cfmap = self._petype_fnmap['hex']['quad'][coarse['fidx']]
            mortars.append({
                'format': 'one-to-many-v1',
                'template': 'quad-2x2',
                'left': tuple(int(n) for n in crow[cfmap]),
                'right': ordered,
            })

        boundary_remove = defaultdict(set)
        boundary_add = defaultdict(list)
        for tag in sorted(refined_roots):
            for fidx in range(6):
                face = rootfaces[tag, fidx]
                if face['kind'] != 'fixed':
                    continue

                bkey, bidx = face['info']
                surface = ('root', face['qkey'])
                fdescs = sorted(
                    groups[surface][tag],
                    key=lambda d: (d['leaf'], d['fidx'])
                )
                boundary_remove[bkey].add(bidx)
                fmap = self._petype_fnmap['hex']['quad'][fidx]
                boundary_add[bkey].extend(
                    tuple(int(n) for n in leafrows[d['leaf']][fmap])
                    for d in fdescs
                )

        generated_tag = -1
        refined_by_key = defaultdict(list)
        for tag in refined_roots:
            ekey, eidx = eligible[tag]
            refined_by_key[ekey].append((eidx, tag))

        for ekey in sorted(refined_by_key, key=repr):
            rows = self._elenodes[ekey]
            tags = self._eletags[ekey]
            selected = sorted(refined_by_key[ekey])
            keep = np.ones(len(rows), dtype=bool)
            keep[[eidx for eidx, _ in selected]] = False

            newrows, newtags = [], []
            for _, tag in selected:
                paths = sorted(
                    path for ltag, path in leaves if ltag == tag
                )
                for path in paths:
                    newrows.append(leafrows[tag, path])
                    newtags.append(generated_tag)
                    generated_tag -= 1

            self._elenodes[ekey] = np.vstack((rows[keep], newrows))
            self._eletags[ekey] = np.concatenate((tags[keep], newtags))

        for bkey in sorted(boundary_remove, key=repr):
            idxs = boundary_remove[bkey]
            rows = self._elenodes[bkey]
            tags = self._eletags[bkey]
            keep = np.ones(len(rows), dtype=bool)
            keep[list(idxs)] = False
            newrows = np.asarray(boundary_add[bkey], dtype=np.int64)
            newtags = np.arange(
                generated_tag - len(newrows) + 1,
                generated_tag + 1, dtype=np.int64
            )[::-1]
            generated_tag -= len(newrows)
            self._elenodes[bkey] = np.vstack((rows[keep], newrows))
            self._eletags[bkey] = np.concatenate((tags[keep], newtags))

        shared_root_surfaces = set()
        for (tag, _), face in rootfaces.items():
            if face['kind'] != 'interior':
                continue
            other = face['info'][0]
            if tag in refined_roots and other in refined_roots:
                shared_root_surfaces.add(('root', face['qkey']))

        self._hex_refine_mortars = mortars
        self._hex_refine_applied = True
        self._hex_refine_summary = {
            'selected': len(explicit_roots),
            'generated-hexes': sum(
                1 for tag, _ in leaves if tag in refined_roots
            ),
            'mortars': len(mortars),
            'boundary-splits': sum(
                len(v) for v in boundary_remove.values()
            ),
            'conforming-refined-faces': len(
                shared_root_surfaces - mortar_surfaces
            ),
        }
        return self._hex_refine_summary

    def split_pyramids(self, policy='all'):
        """Replace selected pyramids with equivalent tetrahedra."""
        if getattr(self, '_pyramid_split_applied', False):
            return self._pyramid_split_summary

        if policy not in {'all', 'incompatible'}:
            raise ValueError(f'Invalid pyramid splitting policy {policy!r}')

        pkeys = [
            key for key in self._elenodes
            if self._etype_map[key[0]][0] == 'pyr'
        ]
        input_count = sum(len(self._elenodes[key]) for key in pkeys)

        split_rows = {}
        split_masks = {}
        for key in pkeys:
            rows = self._elenodes[key]
            if policy == 'all':
                mask = np.ones(len(rows), dtype=bool)
            else:
                mask = ~_pyr_parallelogram_mask(
                    self._nodepts, rows[:, :5], self._petype_fnmap,
                    self._nodemaps
                )

            if np.any(mask):
                split_rows[key] = rows[mask]
                split_masks[key] = mask

        split_count = sum(len(rows) for rows in split_rows.values())
        retained_count = input_count - split_count
        if not split_count:
            self._pyramid_split_applied = True
            self._pyramid_split_policy = policy
            self._pyramid_mortars = []
            self._pyramid_boundary_splits = []
            self._pyramid_conforming_pairs = []
            self._pyramid_periodic_pairs = []
            self._pyramid_target_tet_order = None
            self._pyramid_split_summary = {
                'policy': policy, 'input': input_count,
                'split': 0, 'retained': retained_count,
                'generated-tets': 0, 'mortars': 0,
                'boundary-splits': 0, 'conforming-pairs': 0,
                'periodic-pairs': 0,
            }
            return self._pyramid_split_summary

        unsupported = [
            key[0] for key in split_rows
            if ('pyr', self._etype_map[key[0]][1]) not in self._nodemaps
        ]
        if unsupported:
            raise ValueError(
                'Pyramid splitting requires complete supported Gmsh '
                f'pyramids; found element types {unsupported}'
            )

        porders = [
            PyrShape.order_from_npts(self._etype_map[key[0]][1])
            for key in split_rows
        ]
        torders = [
            TetShape.order_from_npts(self._etype_map[key[0]][1])
            for key in self._elenodes
            if self._etype_map[key[0]][0] == 'tet'
        ]
        target_order = max([2*max(porders), *torders])

        # Preserve the source Gmsh maps before installing the importer-local
        # generated tetrahedral ordering for the selected target order.
        source_nodemaps = dict(self._nodemaps)
        tetetype, storage = self._prepare_generated_tets(target_order)
        self._begin_nodepts_append()
        self._promote_existing_tets(
            target_order, tetetype, source_nodemaps, storage
        )

        boundary_quads = self._boundary_quad_lookup()
        records = []
        by_quad = defaultdict(list)
        for pkey, rows in split_rows.items():
            pents = pkey[1]
            for nodes in rows:
                quad = tuple(int(n) for n in nodes[:4])
                record = {
                    'pents': pents,
                    'nodes': nodes,
                    'quad': quad,
                    'qkey': tuple(sorted(quad)),
                    'candidates': self._pyramid_candidates(
                        nodes, target_order
                    ),
                }
                records.append(record)
                by_quad[record['qkey']].append(record)

        for pkey, mask in split_masks.items():
            rows = self._elenodes[pkey]
            if np.any(~mask):
                self._elenodes[pkey] = rows[~mask]
            else:
                del self._elenodes[pkey]

        conforming_pairs = []
        for qkey, group in by_quad.items():
            bmatch = boundary_quads.get(qkey)
            if len(group) == 1:
                record = group[0]
                record['kind'] = 'boundary' if bmatch else 'mortar'
                record['boundary'] = bmatch
                if not bmatch or bmatch[0] != 'periodic':
                    record['selected'] = max(
                        record['candidates'].values(),
                        key=lambda c: (c['quality'], c['edge'])
                    )
            elif len(group) == 2:
                if bmatch:
                    raise ValueError(
                        'A boundary quadrilateral is shared by two pyramids: '
                        f'{qkey}'
                    )

                left, right = group
                common = (
                    left['candidates'].keys() & right['candidates'].keys()
                )
                if len(common) != 2:
                    raise ValueError(
                        'Paired pyramids do not expose the same base '
                        f'diagonals: {qkey}'
                    )

                edge = max(
                    common,
                    key=lambda e: (
                        min(left['candidates'][e]['quality'],
                            right['candidates'][e]['quality']),
                        e,
                    )
                )
                geom_error = self._paired_pyramid_geometry_error(
                    left, right, edge
                )
                if geom_error > 1e-10:
                    raise ValueError(
                        'Paired pyramid base geometry mismatch '
                        f'{geom_error:.3e} exceeds tolerance 1.000e-10'
                    )

                for record in group:
                    record['selected'] = record['candidates'][edge]
                    record['kind'] = 'paired'
                    record['boundary'] = None

                conforming_pairs.append({
                    'quad': qkey,
                    'diagonal_edge': edge,
                    'quality': min(
                        record['selected']['quality'] for record in group
                    ),
                    'geometry_error': geom_error,
                })
            else:
                raise ValueError(
                    f'Non-manifold pyramid base shared by {len(group)} '
                    f'elements: {qkey}'
                )

        periodic_pairs = self._select_periodic_pyramid_diagonals(
            records
        )

        consumed_boundary_quads = set()
        boundary_remove = defaultdict(set)
        boundary_add = defaultdict(list)
        boundary_splits = []
        mortars = []
        converted = defaultdict(list)
        for record in records:
            selected = record['selected']
            quality = selected['quality']
            if quality <= 1e-10:
                raise ValueError(
                    'Pyramid splitting produced a nonpositive or '
                    f'near-singular tetrahedral child; quality={quality:.3e}'
                )

            converted[record['pents']].extend(
                self._store_pyramid_child_tet(
                    coords, corners, target_order, storage
                )
                for coords, corners in zip(
                    selected['coords'], selected['corners']
                )
            )

            quad = record['quad']
            qkey = record['qkey']
            tris = tuple(tuple(c[:3]) for c in selected['corners'])
            if record['kind'] == 'boundary':
                if qkey in consumed_boundary_quads:
                    raise ValueError(
                        'Multiple pyramids reference the same boundary '
                        f'quadrilateral face: {qkey}'
                    )

                kind, bkey, bidx = record['boundary']
                porder = PyrShape.order_from_npts(len(record['nodes']))
                trietype = self._boundary_tri_etype(bkey[1], porder)
                tkeyb = (trietype, bkey[1])
                boundary_add[tkeyb].extend(
                    self._boundary_tri_nodes(
                        record['nodes'], tri, trietype
                    )
                    for tri in tris
                )
                boundary_remove[bkey].add(bidx)
                consumed_boundary_quads.add(qkey)
                boundary_splits.append({
                    'quad': quad,
                    'tris': tris,
                    'boundary_pents': bkey[1],
                    'boundary_kind': kind,
                    'diagonal': selected['diagonal'],
                    'quality': quality,
                })
            elif record['kind'] == 'mortar':
                mortars.append({
                    'quad': quad,
                    'tris': tris,
                    'diagonal': selected['diagonal'],
                    'quality': quality,
                })

        for pents, children in converted.items():
            tkey = (tetetype, pents)
            children = np.asarray(children, dtype=np.int64)
            if tkey in self._elenodes:
                self._elenodes[tkey] = np.vstack(
                    (self._elenodes[tkey], children)
                )
            else:
                self._elenodes[tkey] = children

        # Replace each matched boundary quad by two triangles carrying
        # the same physical-entity key.  These are ordinary boundary faces,
        # not mortar interfaces.
        for bkey, idxs in boundary_remove.items():
            rows = self._elenodes[bkey]
            keep = np.ones(len(rows), dtype=bool)
            keep[list(idxs)] = False
            if np.any(keep):
                self._elenodes[bkey] = rows[keep]
            else:
                del self._elenodes[bkey]

        for bkey, rows in boundary_add.items():
            rows = np.asarray(rows, dtype=np.int64)
            if bkey in self._elenodes:
                self._elenodes[bkey] = np.vstack((self._elenodes[bkey], rows))
            else:
                self._elenodes[bkey] = rows

        self._commit_nodepts_append()

        self._pyramid_mortars = mortars
        self._pyramid_boundary_splits = boundary_splits
        self._pyramid_conforming_pairs = conforming_pairs
        self._pyramid_periodic_pairs = periodic_pairs
        self._pyramid_target_tet_order = target_order
        self._pyramid_split_policy = policy
        self._pyramid_split_summary = {
            'policy': policy, 'input': input_count,
            'split': split_count, 'retained': retained_count,
            'generated-tets': 2*split_count, 'mortars': len(mortars),
            'boundary-splits': len(boundary_splits),
            'conforming-pairs': len(conforming_pairs),
            'periodic-pairs': len(periodic_pairs),
        }
        self._pyramid_split_applied = True
        return self._pyramid_split_summary

    def _read_mesh_format(self, mshit):
        ver, ftype, dsize = next(mshit).split()

        if ver == '2.2':
            self._read_nodes_impl = self._read_nodes_impl_v2
            self._read_eles_impl = self._read_eles_impl_v2
        elif ver == '4.1':
            self._read_nodes_impl = self._read_nodes_impl_v41
            self._read_eles_impl = self._read_eles_impl_v41
        else:
            raise ValueError('Invalid mesh version')

        if ftype != '0':
            raise ValueError('Invalid file type')
        if dsize != '8':
            raise ValueError('Invalid data size')

        if next(mshit) != '$EndMeshFormat\n':
            raise ValueError('Expected $EndMeshFormat')

    def _read_phys_names(self, mshit):
        # Physical entities can be divided up into:
        #  - volume elements (one or more named material regions)
        #  - boundary faces
        #  - periodic faces
        self._volpents = {}
        self._bfacespents = {}
        self._pfacespents = defaultdict(list)

        # Seen physical names and IDs
        seen_names = set()
        seen_ids = set()

        # Collect all physical names with their dimensions
        pnames = []
        for l in msh_section(mshit, 'PhysicalNames'):
            m = re.match(r'(\d+) (\d+) "((?:[^"\\]|\\.)*)"$', l)
            if not m:
                raise ValueError('Malformed physical entity')

            dim, pent, name = int(m[1]), int(m[2]), m[3].lower()

            # Ensure we have not seen this name before
            if name in seen_names:
                raise ValueError(f'Duplicate physical name: {name}')

            # Ensure physical entitiy IDs are unique
            if pent in seen_ids:
                raise ValueError(f'Duplicate physical entity ID: {pent}')

            pnames.append((dim, pent, name))
            seen_names.add(name)
            seen_ids.add(pent)

        # Classify by dimension
        voldim = max(dim for dim, _, _ in pnames)

        for dim, pent, name in pnames:
            # Periodic boundary faces
            if name.startswith('periodic'):
                p = re.match(r'periodic[ _-]([a-z0-9]+)[ _-](l|r)$', name)
                if not p:
                    raise ValueError('Invalid periodic boundary condition')

                self._pfacespents[p[1]].append(pent)
            # Volume elements
            elif dim == voldim:
                self._volpents[name] = pent
            # Other boundary faces
            else:
                self._bfacespents[name] = pent

        if not self._volpents:
            raise ValueError('No volume elements in mesh')

        if any(len(pf) != 2 for pf in self._pfacespents.values()):
            raise ValueError('Unpaired periodic boundary in mesh')

    def _read_entities(self, mshit):
        self._tagpents = tagpents = {}

        # Obtain the entity counts
        npts, *ents = (int(i) for i in next(mshit).split())

        # Skip over the point entities
        for _ in range(npts):
            next(mshit)

        # Iterate through the curves, surfaces, and volume entities
        for ndim, nent in enumerate(ents, start=1):
            for _ in range(nent):
                ent = next(mshit).split()
                etag, enphys = int(ent[0]), int(ent[7])

                if enphys == 0:
                    continue
                else:
                    pents = (abs(int(ent[8 + i])) for i in range(enphys))
                    tagpents[ndim, etag] = tuple(sorted(pents))

        if next(mshit) != '$EndEntities\n':
            raise ValueError('Expected $EndEntities')

    def _read_nodes(self, mshit):
        self._read_nodes_impl(mshit)

    def _read_nodes_impl_v2(self, mshit):
        n = int(next(mshit))
        nbuf = np.loadtxt(mshit, dtype='i8,3f8', max_rows=n)

        # Determine the minimum and maximum node numbers
        ixl, ixu = nbuf['f0'].min(), nbuf['f0'].max()

        # Allocate a dense array for the nodes
        self._nodepts = nodepts = np.empty((ixu - ixl + 1, 3))
        nodepts.fill(np.nan)

        nodepts[nbuf['f0'] - ixl] = nbuf['f1']

        # Save the starting node offset
        self._nodeoff = ixl

        if next(mshit) != '$EndNodes\n':
            raise ValueError('Expected $EndNodes')

    def _read_nodes_impl_v41(self, mshit):
        # Entity count, node count, minimum and maximum node numbers
        ne, nn, ixl, ixu = (int(i) for i in next(mshit).split())

        self._nodepts = nodepts = np.empty((ixu - ixl + 1, 3))
        nodepts.fill(np.nan)

        for _ in range(ne):
            nen = int(next(mshit).split()[-1])
            nix = np.loadtxt(mshit, dtype=np.int64, max_rows=nen)
            nodepts[nix - ixl] = np.loadtxt(mshit, max_rows=nen)

        # Save the starting node offset
        self._nodeoff = ixl

        if next(mshit) != '$EndNodes\n':
            raise ValueError('Expected $EndNodes')

    def _read_eles(self, mshit):
        self._read_eles_impl(mshit)

    def _read_eles_impl_v2(self, mshit):
        elenodes = defaultdict(list)
        eletags = defaultdict(list)

        for l in msh_section(mshit, 'Elements'):
            # Extract the raw element data
            elei = [int(i) for i in l.split()]
            enum, etype, entags = elei[:3]
            etags, enodes = elei[3:3 + entags], elei[3 + entags:]

            if etype not in self._etype_map:
                raise ValueError(f'Unsupported element type {etype}')

            # Physical entity type (used for BCs)
            key = etype, (etags[0],)
            elenodes[key].append(enodes)
            eletags[key].append(enum)

        self._elenodes = {k: np.array(v) for k, v in elenodes.items()}
        self._eletags = {k: np.array(v) for k, v in eletags.items()}

    def _read_eles_impl_v41(self, mshit):
        elenodes = defaultdict(list)
        eletags = defaultdict(list)

        # Block and total element count
        nb, ne = (int(i) for i in next(mshit).split()[:2])

        for _ in range(nb):
            edim, etag, etype, ecount = (int(j) for j in next(mshit).split())

            if etype not in self._etype_map:
                raise ValueError(f'Unsupported element type {etype}')

            # Determine the number of nodes associated with each element
            nnodes = self._etype_map[etype][1]

            # Lookup the physical entity type(s)
            epents = self._tagpents[edim, etag]

            # Allocate space for, and read in, these elements
            ebuf = np.loadtxt(
                mshit, dtype=np.int64, max_rows=ecount,
                usecols=range(nnodes + 1), ndmin=2
            )
            key = etype, epents
            eletags[key].append(ebuf[:, 0])
            elenodes[key].append(ebuf[:, 1:])

        if ne != sum(len(vv) for v in elenodes.values() for vv in v):
            raise ValueError('Invalid element count')

        if next(mshit) != '$EndElements\n':
            raise ValueError('Expected $EndElements')

        self._elenodes = {k: np.vstack(v) for k, v in elenodes.items()}
        self._eletags = {k: np.concatenate(v) for k, v in eletags.items()}

    def _merge_vol(self):
        # Map from volume pent ID to bit mask
        pbits = {p: 1 << i
                 for i, (_, p) in enumerate(sorted(self._volpents.items()))}
        volpent = min(self._volpents.values())

        elenodes = {}
        tagruns, vnodes = defaultdict(list), defaultdict(list)

        for (etype, pents), nodes in sorted(self._elenodes.items()):
            # Compute the combined bitmask for volume pents
            mask = sum(pbits[p] for p in pents if p in pbits)

            # Pass through non-volume entries unchanged
            if not mask:
                elenodes[etype, pents[0]] = nodes
                continue

            petype = self._etype_map[etype][0]

            # Collect nodes and record the tag bitmask
            vnodes[etype].append(nodes)
            tagruns[petype].append((mask, len(nodes)))

        # Stack the collected nodes for each volume element type
        for etype, enodes in vnodes.items():
            elenodes[etype, volpent] = np.vstack(enodes)

        return elenodes, volpent, tagruns

    @staticmethod
    def _assign_tags(eles, tagruns):
        # Set per-element tag bits from precomputed bitmasks
        for petype, runs in tagruns.items():
            masks, counts = zip(*runs)
            eles[petype]['tags'] = np.repeat(masks, counts)

    def _to_raw_mesh(self, lintol):
        # Merge volume groups into a single pent, tracking tags
        elenodes, volpent, tagruns = self._merge_vol()

        # Assemble a nodal mesh
        maps = self._etype_map, self._petype_fnmap, self._nodemaps
        mortars = [
            *getattr(self, '_pyramid_mortars', ()),
            *getattr(self, '_hex_refine_mortars', ()),
        ]
        mesh = NodalMeshAssembler(
            self._nodepts, elenodes, volpent, self._bfacespents,
            self._pfacespents, maps, mortars=mortars
        )

        nodepts, eles, codec, periodic, mortars = mesh.get_eles(
            lintol, self.progress
        )

        # Append tag entries to the codec and assign per-element values
        codec.extend(f'tag/{tn}' for tn in sorted(self._volpents))
        self._assign_tags(eles, tagruns)

        return nodepts, eles, codec, periodic, mortars
