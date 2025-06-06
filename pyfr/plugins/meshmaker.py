import argparse
import io
from itertools import product
import os
from pathlib import Path
import subprocess
import sys
from collections import namedtuple

import h5py
import numpy as np
import pandas as pd

from pyfr.partitioners import get_partitioner
from pyfr.plugins.base import BaseCLIPlugin, cli_external
from pyfr.progress import NullProgressSequence
from pyfr.readers.gmsh import GmshReader

# Constants for Gmsh element types
ElementType = namedtuple('ElementType', ['code', 'num_nodes'])
ELEMENT_TYPES = {
     'tri': ElementType(code=2, num_nodes=3),
    'quad': ElementType(code=3, num_nodes=4),
     'tet': ElementType(code=4, num_nodes=4),
     'hex': ElementType(code=5, num_nodes=8),
     'pri': ElementType(code=6, num_nodes=6),
     'pyr': ElementType(code=7, num_nodes=5),
}

class MeshMaker:
    def __init__(self, length = 2* np.pi):
        self.l = length
        self.header =  '''
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
7
3 1 "fluid"
2 2 "periodic_0_l"
2 3 "periodic_1_l"
2 4 "periodic_2_l"
2 5 "periodic_0_r"
2 6 "periodic_1_r"
2 7 "periodic_2_r"
$EndPhysicalNames
'''

    def gmsh_header(self):
        return self.header

    @staticmethod
    def gmsh_nodes(X):
        nodes = '\n'.join(f'{i+1} {" ".join(map(str, x))}' for i, x in enumerate(X))
        return f'$Nodes\n{len(X)}\n{nodes}\n$EndNodes\n'

    @staticmethod
    def grid_index(nx, ny, i, j, k):
        return 1 + i + j*nx + k*nx*ny

    def gmsh_boundaries_tet(self, nx0, nx, ny0, ny, nz0, nz, ele, nele, boundaries = [True, True, True, True, True, True]):
        ind = lambda i, j, k: self.grid_index(nx, ny, i, j, k)
        for yi in range(ny0-1, ny-1):
            for zi in range(nz0-1, nz-1):
                if boundaries[0]:
                    nele += 1 ; n = [ind(nx0-1,  yi+0, zi+0), ind(nx0-1,  yi+1,  zi+0), ind(nx0-1,  yi+0,  zi+1)                       ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 2 2 2 2 {n_str}\n' #   WEST 1: i=0
                    nele += 1 ; n = [ind(nx0-1,  yi+0, zi+1), ind(nx0-1,  yi+1,  zi+0), ind(nx0-1,  yi+1,  zi+1)                       ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 2 2 2 2 {n_str}\n' #   WEST 2: i=0
                if boundaries[1]:
                    nele += 1 ; n = [ind( nx-1,  yi+0, zi+0), ind( nx-1,  yi+0,  zi+1), ind( nx-1,  yi+1,  zi+0)                       ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 2 2 5 5 {n_str}\n' #   EAST 1: i=nx-1
                    nele += 1 ; n = [ind( nx-1,  yi+0, zi+1), ind( nx-1,  yi+1,  zi+1), ind( nx-1,  yi+1,  zi+0)                       ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 2 2 5 5 {n_str}\n' #   EAST 2: i=nx-1

        for xi in range(nx0-1, nx-1):
            for zi in range(nz0-1, nz-1):
                if boundaries[2]:
                    nele += 1 ; n = [ind( xi+0, ny0-1, zi+0), ind( xi+0, ny0-1,  zi+1), ind( xi+1, ny0-1,  zi+0)                       ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 2 2 3 3 {n_str}\n' #  SOUTH 1: j=0
                    nele += 1 ; n = [ind( xi+1, ny0-1, zi+0), ind( xi+0, ny0-1,  zi+1), ind( xi+1, ny0-1,  zi+1)                       ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 2 2 3 3 {n_str}\n' #  SOUTH 2: j=0
                if boundaries[3]:
                    nele += 1 ; n = [ind( xi+0,  ny-1, zi+0), ind( xi+1,  ny-1,  zi+0), ind( xi+0,  ny-1,  zi+1)                       ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 2 2 6 6 {n_str}\n' #  NORTH 1: j=nx-1
                    nele += 1 ; n = [ind( xi+1,  ny-1, zi+0), ind( xi+1,  ny-1,  zi+1), ind( xi+0,  ny-1,  zi+1)                       ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 2 2 6 6 {n_str}\n' #  NORTH 2: j=nx-1

        for xi in range(nx0-1, nx-1):
            for yi in range(ny0-1, ny-1):
                if boundaries[4]:
                    nele += 1 ; n = [ind( xi+0,  yi+0,nz0-1), ind( xi+1,  yi+0, nz0-1), ind( xi+0,  yi+1, nz0-1)                       ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 2 2 4 4 {n_str}\n' # BOTTOM 1: k=0
                    nele += 1 ; n = [ind( xi+1,  yi+0,nz0-1), ind( xi+1,  yi+1, nz0-1), ind( xi+0,  yi+1, nz0-1)                       ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 2 2 4 4 {n_str}\n' # BOTTOM 2: k=0
                if boundaries[5]:
                    nele += 1 ; n = [ind( xi+0,  yi+0, nz-1), ind( xi+0,  yi+1,  nz-1), ind( xi+1,  yi+0,  nz-1)                       ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 2 2 7 7 {n_str}\n' #    TOP 1: k=nx-1
                    nele += 1 ; n = [ind( xi+1,  yi+0, nz-1), ind( xi+0,  yi+1,  nz-1), ind( xi+1,  yi+1,  nz-1)                       ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 2 2 7 7 {n_str}\n' #    TOP 2: k=nx-1
        return ele, nele

    def gmsh_boundaries_pyr(self, nx0, nx, ny0, ny, nz0, nz, ele, nele, boundaries = [True, True, True, True, True, True]):
        ind = lambda i, j, k: self.grid_index(nx, ny, i, j, k)
        for zi in range(nz0-1, nz-1):
            for yi in range(ny0-1, ny-1):
                if boundaries[0]: nele += 1 ; n = [ind(nx0-1,  yi+0,  zi+0), ind(nx0-1,  yi+0,  zi+1), ind(nx0-1,  yi+1,  zi+1), ind(nx0-1,  yi+1,  zi+0)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 3 2 2 2 {n_str}\n' #   WEST: i=0
                if boundaries[1]: nele += 1 ; n = [ind( nx-1,  yi+0,  zi+0), ind( nx-1,  yi+0,  zi+1), ind( nx-1,  yi+1,  zi+1), ind( nx-1,  yi+1,  zi+0)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 3 2 5 5 {n_str}\n' #   EAST: i=nx-1

        for zi in range(nz0-1, nz-1):
            for xi in range(nx0-1, nx-1):
                if boundaries[2]: nele += 1 ; n = [ind( xi+0, ny0-1,  zi+0), ind( xi+1, ny0-1,  zi+0), ind( xi+1, ny0-1,  zi+1), ind( xi+0, ny0-1,  zi+1)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 3 2 3 3 {n_str}\n' #  SOUTH: j=0
                if boundaries[3]: nele += 1 ; n = [ind( xi+0,  ny-1,  zi+0), ind( xi+1,  ny-1,  zi+0), ind( xi+1,  ny-1,  zi+1), ind( xi+0,  ny-1,  zi+1)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 3 2 6 6 {n_str}\n' #  NORTH: j=nx-1

        for yi in range(ny0-1, ny-1):
            for xi in range(nx0-1, nx-1):
                if boundaries[4]: nele += 1 ; n = [ind( xi+0,  yi+0, nz0-1), ind( xi+1,  yi+0, nz0-1), ind( xi+1,  yi+1, nz0-1), ind( xi+0,  yi+1, nz0-1)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 3 2 4 4 {n_str}\n' # BOTTOM: k=0
                if boundaries[5]: nele += 1 ; n = [ind( xi+0,  yi+0,  nz-1), ind( xi+1,  yi+0,  nz-1), ind( xi+1,  yi+1,  nz-1), ind( xi+0,  yi+1,  nz-1)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 3 2 7 7 {n_str}\n' #    TOP: k=nx-1
        return ele, nele

    def gmsh_boundaries_pri(self, nx0, nx, ny0, ny, nz0, nz, ele, nele, boundaries = [True, True, True, True, True, True]):
        ind = lambda i, j, k: self.grid_index(nx, ny, i, j, k)

        for zi in range(nz0-1, nz-1):
            for yi in range(ny0-1, ny-1):
                if boundaries[0]: nele += 1 ; n = [ind(nx0-1,  yi+0,   zi+0), ind(nx0-1,  yi+1,   zi+0), ind(nx0-1,  yi+1,   zi+1), ind(nx0-1,  yi+0,   zi+1)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 3 2 2 2 {n_str}\n' # i=0
                if boundaries[1]: nele += 1 ; n = [ind( nx-1,  yi+0,   zi+0), ind( nx-1,  yi+1,   zi+0), ind( nx-1,  yi+1,   zi+1), ind( nx-1,  yi+0,   zi+1)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 3 2 5 5 {n_str}\n' # i=nx-1

        for zi in range(nz0-1, nz-1):
            for xi in range(nx0-1, nx-1):
                if boundaries[2]: nele += 1 ; n = [ind( xi+0, ny0-1,   zi+0), ind( xi+1, ny0-1,   zi+0), ind( xi+1, ny0-1,   zi+1), ind( xi+0, ny0-1,   zi+1)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 3 2 3 3 {n_str}\n' # j=0
                if boundaries[3]: nele += 1 ; n = [ind( xi+0,  ny-1,   zi+0), ind( xi+1,  ny-1,   zi+0), ind( xi+1,  ny-1,   zi+1), ind( xi+0,  ny-1,   zi+1)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 3 2 6 6 {n_str}\n' # j=nx-1

        for yi in range(ny0-1, ny-1):
            for xi in range(nx0-1, nx-1):
                if boundaries[4]:
                    nele += 1 ; n = [ind( xi+0,  yi+0,  nz0-1), ind( xi+0,  yi+1,  nz0-1), ind( xi+1,  yi+0,  nz0-1)                           ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 2 2 4 4 {n_str}\n' # k=0
                    nele += 1 ; n = [ind( xi+1,  yi+1,  nz0-1), ind( xi+0,  yi+1,  nz0-1), ind( xi+1,  yi+0,  nz0-1)                           ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 2 2 4 4 {n_str}\n' 
                if boundaries[5]:
                    nele += 1 ; n = [ind( xi+0,  yi+0,   nz-1), ind( xi+0,  yi+1,   nz-1), ind( xi+1,  yi+0,   nz-1)                           ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 2 2 7 7 {n_str}\n' # k=nx-1
                    nele += 1 ; n = [ind( xi+1,  yi+1,   nz-1), ind( xi+0,  yi+1,   nz-1), ind( xi+1,  yi+0,   nz-1)                           ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 2 2 7 7 {n_str}\n'
        return ele, nele

    def gmsh_boundaries_hex(self, nx0, nx, ny0, ny, nz0, nz, ele, nele, boundaries = [True, True, True, True, True, True]):
        ind = lambda i, j, k: self.grid_index(nx, ny, i, j, k)

        for zi in range(nz0-1, nz-1):
            for yi in range(ny0-1, ny-1):
                if boundaries[0]: nele += 1 ; n = [ind(nx0-1,    yi,     zi), ind(nx0-1,  yi+1,     zi), ind(nx0-1,  yi+1,   zi+1), ind(nx0-1,    yi,   zi+1)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 3 2 2 2 {n_str}\n' # i=0
                if boundaries[1]: nele += 1 ; n = [ind( nx-1,    yi,     zi), ind( nx-1,  yi+1,     zi), ind( nx-1,  yi+1,   zi+1), ind( nx-1,    yi,   zi+1)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 3 2 5 5 {n_str}\n' # i=nx-1

        for zi in range(nz0-1, nz-1):
            for xi in range(nx0-1, nx-1):
                if boundaries[2]: nele += 1 ; n = [ind( xi  , ny0-1,     zi), ind( xi+1, ny0-1,     zi), ind( xi+1, ny0-1,   zi+1), ind(   xi, ny0-1,   zi+1)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 3 2 3 3 {n_str}\n' # j=0
                if boundaries[3]: nele += 1 ; n = [ind( xi  ,  ny-1,     zi), ind( xi+1,  ny-1,     zi), ind( xi+1,  ny-1,   zi+1), ind(   xi,  ny-1,   zi+1)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 3 2 6 6 {n_str}\n' # j=nx-1

        for yi in range(ny0-1, ny-1):
            for xi in range(nx0-1, nx-1):
                if boundaries[4]: nele += 1 ; n = [ind( xi  ,    yi,  nz0-1), ind( xi  ,  yi+1,  nz0-1), ind( xi+1,  yi+1,  nz0-1), ind( xi+1,    yi,  nz0-1)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 3 2 4 4 {n_str}\n' # k=0
                if boundaries[5]: nele += 1 ; n = [ind( xi  ,    yi,   nz-1), ind( xi  ,  yi+1,   nz-1), ind( xi+1,  yi+1,   nz-1), ind( xi+1,    yi,   nz-1)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 3 2 7 7 {n_str}\n' # k=nx-1

        return ele, nele

    def gmsh_elements_pri(self, nx0, nx, ny0, ny, nz0, nz, ele, nele):
        ind = lambda i, j, k: self.grid_index(nx, ny, i, j, k)

        for k in range(nz0-1, nz - 1):
            for j in range(ny0-1, ny - 1):
                for i in range(nx0-1, nx - 1):
                    nele += 1 ; n = [ ind(i+0, j+0, k+0), ind(i+1, j+0, k+0), ind(i+0, j+1, k+0), ind(i+0, j+0, k+1), ind(i+1, j+0, k+1), ind(i+0, j+1, k+1), ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 6 2 1 1 {n_str} \n'
                    nele += 1 ; n = [ ind(i+1, j+1, k+0), ind(i+0, j+1, k+0), ind(i+1, j+0, k+0), ind(i+1, j+1, k+1), ind(i+0, j+1, k+1), ind(i+1, j+0, k+1), ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 6 2 1 1 {n_str} \n'
        return ele, nele

    def gmsh_elements_pyr(self, nx0, nx, nxL, ny0, ny, nyL, nz0, nz, nzL, ele, nele):
        ind  = lambda i, j, k:               self.grid_index(nx  , ny  , i, j, k)
        mind = lambda i, j, k: nxL*nyL*nzL + self.grid_index(nx-1, ny-1, i, j, k)

        for k in range(nz0-1, nz - 1):
            for j in range(ny0-1, ny - 1):
                for i in range(nx0-1, nx - 1):
                    nele += 1 ; n = [ ind(i+0, j+0, k+0), ind(i+1, j+0, k+0), ind(i+1, j+1, k+0), ind(i+0, j+1, k+0), mind(i, j, k)                           ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 7 2 1 1 {n_str} \n' # Bottom
                    nele += 1 ; n = [ ind(i+0, j+0, k+1), ind(i+0, j+1, k+1), ind(i+1, j+1, k+1), ind(i+1, j+0, k+1), mind(i, j, k)                           ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 7 2 1 1 {n_str} \n' #    Top
                    nele += 1 ; n = [ ind(i+1, j+0, k+0), ind(i+1, j+0, k+1), ind(i+1, j+1, k+1), ind(i+1, j+1, k+0), mind(i, j, k)                           ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 7 2 1 1 {n_str} \n' #   East
                    nele += 1 ; n = [ ind(i+0, j+0, k+0), ind(i+0, j+1, k+0), ind(i+0, j+1, k+1), ind(i+0, j+0, k+1), mind(i, j, k)                           ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 7 2 1 1 {n_str} \n' #   West
                    nele += 1 ; n = [ ind(i+0, j+0, k+0), ind(i+0, j+0, k+1), ind(i+1, j+0, k+1), ind(i+1, j+0, k+0), mind(i, j, k)                           ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 7 2 1 1 {n_str} \n' #  South
                    nele += 1 ; n = [ ind(i+0, j+1, k+0), ind(i+1, j+1, k+0), ind(i+1, j+1, k+1), ind(i+0, j+1, k+1), mind(i, j, k)                           ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 7 2 1 1 {n_str} \n' #  North
        return ele, nele

    def gmsh_elements_tet(self, nx0, nx, nxL, ny0, ny, nyL, nz0, nz, nzL, ele, nele):
        ind  = lambda i, j, k:               self.grid_index(nx  , ny  , i, j, k)
        mind = lambda i, j, k: nxL*nyL*nzL + self.grid_index(nx-1, ny-1, i, j, k)

        for k in range(nz0-1, nz - 1):
            for j in range(ny0-1, ny - 1):
                for i in range(nx0-1, nx - 1):
                    nele += 1 ; n = [ind(i+0, j+0, k+0), ind(i+1, j+0, k+0), ind(i+0, j+1, k+0), mind(i, j, k)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 4 2 1 1 {n_str} \n' # Bottom 1
                    nele += 1 ; n = [ind(i+1, j+0, k+0), ind(i+1, j+1, k+0), ind(i+0, j+1, k+0), mind(i, j, k)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 4 2 1 1 {n_str} \n' # Bottom 2
                    nele += 1 ; n = [ind(i+0, j+0, k+1), ind(i+0, j+1, k+1), ind(i+1, j+0, k+1), mind(i, j, k)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 4 2 1 1 {n_str} \n' #    Top 1
                    nele += 1 ; n = [ind(i+1, j+0, k+1), ind(i+0, j+1, k+1), ind(i+1, j+1, k+1), mind(i, j, k)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 4 2 1 1 {n_str} \n' #    Top 2
                    nele += 1 ; n = [ind(i+1, j+0, k+0), ind(i+1, j+0, k+1), ind(i+1, j+1, k+0), mind(i, j, k)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 4 2 1 1 {n_str} \n' #   East 1
                    nele += 1 ; n = [ind(i+1, j+0, k+1), ind(i+1, j+1, k+1), ind(i+1, j+1, k+0), mind(i, j, k)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 4 2 1 1 {n_str} \n' #   East 2
                    nele += 1 ; n = [ind(i+0, j+0, k+0), ind(i+0, j+1, k+0), ind(i+0, j+0, k+1), mind(i, j, k)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 4 2 1 1 {n_str} \n' #   West 1
                    nele += 1 ; n = [ind(i+0, j+0, k+1), ind(i+0, j+1, k+0), ind(i+0, j+1, k+1), mind(i, j, k)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 4 2 1 1 {n_str} \n' #   West 2
                    nele += 1 ; n = [ind(i+0, j+0, k+0), ind(i+0, j+0, k+1), ind(i+1, j+0, k+0), mind(i, j, k)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 4 2 1 1 {n_str} \n' #  South 1
                    nele += 1 ; n = [ind(i+1, j+0, k+0), ind(i+0, j+0, k+1), ind(i+1, j+0, k+1), mind(i, j, k)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 4 2 1 1 {n_str} \n' #  South 2
                    nele += 1 ; n = [ind(i+0, j+1, k+0), ind(i+1, j+1, k+0), ind(i+0, j+1, k+1), mind(i, j, k)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 4 2 1 1 {n_str} \n' #  North 1
                    nele += 1 ; n = [ind(i+1, j+1, k+0), ind(i+1, j+1, k+1), ind(i+0, j+1, k+1), mind(i, j, k)] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 4 2 1 1 {n_str} \n' #  North 2
        return ele, nele

    def gmsh_elements_hex(self, nx0, nx, ny0, ny, nz0, nz, ele, nele):
        ind = lambda i, j, k: self.grid_index(nx, ny, i, j, k)
        for k in range(nz0-1, nz - 1):
            for j in range(ny0-1, ny - 1):
                for i in range(nx0-1, nx - 1):
                    nele += 1 ; n = [ ind(i+0, j+0, k+0), ind(i+1, j+0, k+0), ind(i+1, j+1, k+0), ind(i+0, j+1, k+0), ind(i+0, j+0, k+1), ind(i+1, j+0, k+1), ind(i+1, j+1, k+1), ind(i+0, j+1, k+1), ] ; n_str = ' '.join('{ni}'.format(ni=ni) for ni in n) ; ele += f'{nele} 5 2 1 1 {n_str} \n'
        return ele, nele

    def make_mesh(self, etype, nvertices):

        nx0 = 1 
        ny0 = 1 
        nz0 = 1 

        nx = nvertices + 1 
        ny = nvertices + 1 
        nz = nvertices + 1 

        nxL = nx
        nyL = ny
        nzL = nz

        Rx = np.linspace(0, self.l, nx)
        Ry = np.linspace(0, self.l, ny)
        Rz = np.linspace(0, self.l, nz)

        dx = self.l / nx
        dy = self.l / ny
        dz = self.l / nz 

        Mx = np.linspace(0.5*dx, self.l - 0.5*dx, nx-1)
        My = np.linspace(0.5*dy, self.l - 0.5*dy, ny-1)
        Mz = np.linspace(0.5*dz, self.l - 0.5*dz, nz-1)
        i = 0

        ele = ''
        nele = 0

        def modify_X(X, mesh, i):
            coords = np.stack([mesh[2], mesh[1], mesh[0]], axis=-1).reshape(-1, 3)
            num_new_points = coords.shape[0]
            X[i:i + num_new_points, :] = coords
            return X, i+num_new_points

        X = np.zeros(((nx-nx0+1)*(ny-ny0+1)*(nz-nz0+1) + (nx-nx0)*(ny-ny0)*(nz-nz0), 3))
        X, i = modify_X(X, np.meshgrid(Rx, Ry, Rz, indexing='ij'), i)
        X, i = modify_X(X, np.meshgrid(Mx, My, Mz, indexing='ij'), i)

        if etype == 'tet': 
            ele, nele = self.gmsh_boundaries_tet(nx0, nx, ny0, ny, nz0, nz, ele, nele)
            ele, nele = self.gmsh_elements_tet(  nx0, nx, nxL, ny0, ny, nyL, nz0, nz, nzL, ele, nele)

        elif etype == 'pyr':
            ele, nele = self.gmsh_boundaries_pyr(nx0, nx, ny0, ny, nz0, nz, ele, nele)
            ele, nele = self.gmsh_elements_pyr(  nx0, nx, nxL, ny0, ny, nyL, nz0, nz, nzL, ele, nele)

        elif etype == 'pri':
            ele, nele = self.gmsh_boundaries_pri(nx0, nx, ny0, ny, nz0, nz, ele, nele)
            ele, nele = self.gmsh_elements_pri(  nx0, nx, ny0, ny, nz0, nz, ele, nele)

        elif etype == 'hex':
            ele, nele = self.gmsh_boundaries_hex(nx0, nx, ny0, ny, nz0, nz, ele, nele)
            ele, nele = self.gmsh_elements_hex(  nx0, nx, ny0, ny, nz0, nz, ele, nele)

        elif etype == 'hexpyr':
            nz2 = nvertices//2 + 1 
            ele, nele = self.gmsh_boundaries_hex(nx0, nx,      ny0, ny,      nz0, nz2,      ele, nele, boundaries = [True, True, True, True, True, False])
            ele, nele = self.gmsh_boundaries_pyr(nx0, nx,      ny0, ny,      nz2, nz ,      ele, nele, boundaries = [True, True, True, True, False, True])
            ele, nele = self.gmsh_elements_hex(  nx0, nx,      ny0, ny,      nz0, nz2,      ele, nele)
            ele, nele = self.gmsh_elements_pyr(  nx0, nx, nxL, ny0, ny, nyL, nz2, nz , nzL, ele, nele)

        elif etype == 'pritet':
            nz2 = nvertices//2 + 1 
            ele, nele = self.gmsh_boundaries_pri(nx0, nx,      ny0, ny,      nz0, nz2,      ele, nele, boundaries = [True, True, True, True, True, False])
            ele, nele = self.gmsh_boundaries_tet(nx0, nx,      ny0, ny,      nz2, nz ,      ele, nele, boundaries = [True, True, True, True, False, True])
            ele, nele = self.gmsh_elements_pri(  nx0, nx,      ny0, ny,      nz0, nz2,      ele, nele)
            ele, nele = self.gmsh_elements_tet(  nx0, nx, nxL, ny0, ny, nyL, nz2, nz , nzL, ele, nele)
        else:
            raise ValueError(f"Mesh type {etype} not recognized")

        return self.gmsh_header() + self.gmsh_nodes(X) + '$Elements\n'+str(nele)+'\n' + ele + '$EndElements\n'

class MeshMakerCLI(BaseCLIPlugin):
    name = 'meshmaker'

    @classmethod
    def add_cli(cls, parser):
        # Add subparsers for the 'meshmaker' command
        sp = parser.add_subparsers()
        mp = sp.add_parser('generate-mesh',
                           help='Generate meshes using MeshMaker')
        mp.set_defaults(process=cls.generate_mesh_cli)

        # Command-line arguments
        mp.add_argument('--options', nargs='*', help='CSV file with mesh parameters', required=False)
        mp.add_argument('--length', type=float, help='Length of the domain (default is 2π)', default=6.28318530718)
        mp.add_argument('--overwrite', action='store_true', help='Overwrite existing mesh files')
        mp.add_argument('-l', '--lintol', type=float, default=1e-5, help='Linearization tolerance')

    @cli_external
    def generate_mesh_cli(self, args):
        # Check if CSV options file is provided
        if args.options:
            self.generate_meshes_from_csv(args)
        else:
            print("Error: '--options' must be specified to provide mesh parameters.")
            sys.exit(1)

    def generate_meshes_from_csv(self, args):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(args.options[0], sep=',', skipinitialspace=True, comment='#')

        # Filter columns that start with 'mesh:'
        df_mesh = df.filter(regex='mesh:')
        if df_mesh.empty:
            print("No 'mesh:' columns found in CSV file.")
            sys.exit(1)

        # Generate file names based on mesh parameters
        df_mesh['file-name'] = df_mesh.apply(
            lambda row: '_'.join(
                [f'{column.split(":")[1]}-{str(value).strip()}' for column, value in row.items() if not pd.isna(value) and column != 'mesh:partitions']) + '.pyfrm',
            axis=1
        )

        # Remove duplicate file names if any
        df_mesh = df_mesh[~df_mesh['file-name'].duplicated(keep='first')]

        # Group by file-name to handle multiple partitions per mesh
        grouped = df_mesh.groupby('file-name')

        # Iterate over each group to generate meshes
        for file_name, group in grouped:
            # Extract mesh parameters (assuming they are the same within the group)
            row = group.iloc[0]
            etype = row.get('mesh:etype')
            order = int(row.get('mesh:order', 0))
            dofs = float(row.get('mesh:dof', 0))
            partitions_list = group['mesh:partitions'].astype(int).tolist()

            # Create args namespace for this mesh
            mesh_args = argparse.Namespace(
                etype=etype,
                nvertices=None,
                dofs=dofs,
                order=order,
                output=file_name,
                partitions=partitions_list,
                length=args.length,
                overwrite=args.overwrite,
                lintol=args.lintol,
            )

            # Generate the mesh
            self.generate_single_mesh(mesh_args)

    def generate_single_mesh(self, args):
        # Create an instance of MeshMaker
        mesh_maker = MeshMaker()
        mesh_maker.l = args.length

        # Generate the Gmsh mesh as a string
        try:
            etype = args.etype
            output_file = args.output

            # Determine nvertices
            if args.nvertices:
                nvertices = args.nvertices
            elif args.dofs:
                # Ensure that order is provided
                if args.order is None:
                    print("Error: 'order' must be specified when using 'dofs'.")
                    sys.exit(1)
                order = args.order
                # Calculate nvertices based on target dofs
                nvertices = self.calculate_nvertices(etype, order, args.dofs)
                print(f"Calculated nvertices: {nvertices} to achieve target DoFs: {args.dofs}")
            else:
                print("Error: Either 'nvertices' or 'dofs' must be specified.")
                sys.exit(1)

            print(f"Generating mesh: etype={etype}, nvertices={nvertices}")

            # Generate the mesh as a Gmsh format string
            msh_str = mesh_maker.make_mesh(etype, nvertices)

            # Write the output file
            if not os.path.isfile(output_file) or args.overwrite:
                ext = Path(output_file).suffix.lower()
                if ext == '.msh':
                    # Write the Gmsh mesh string directly to the file
                    with open(output_file, 'w') as f:
                        f.write(msh_str)
                    print(f"Gmsh mesh file generated: {output_file}")
                elif ext == '.pyfrm':
                    # Create a GmshReader instance using the file-like object
                    reader = GmshReader(io.StringIO(msh_str),
                                        progress=NullProgressSequence())

                    # Write the PyFR mesh file
                    reader.write(output_file, args.lintol)
                    print(f"PyFR mesh file generated: {output_file}")
                else:
                    print(f"Unsupported extension '{ext}'. Use .msh or .pyfrm")
            else:
                print(f"Mesh file already exists: {output_file}")
                print("Use '--overwrite' to regenerate the mesh.")

            # Partition the mesh if requested
            if args.partitions:
                # Ensure output file is a PyFR mesh
                if Path(output_file).suffix.lower() == '.pyfrm':
                    # Add each partition count to the mesh file
                    for partition_count in args.partitions:
                        print(f"Adding partitioning with {partition_count} partitions to mesh '{output_file}'...")
                        partition_name = f'part{partition_count}'
                        # Check if the partitioning already exists unless overwrite is specified
                        existing_partitions = self.get_existing_partitions(output_file)
                        if partition_name in existing_partitions and not args.overwrite:
                            print(f"Partitioning '{partition_name}' already exists in mesh '{output_file}'.")
                            continue
                        # Use PyFR's partitioning functionality
                        subprocess.run(['pyfr', 'partition', 'add', 
                                        '-p', 'metis',
                                        '-e', 'balanced',
                                        output_file,
                                        str(partition_count),
                                        partition_name])
                else:
                    print("Error: Partitions can only be added to PyFR mesh files.")

        except Exception as e:
            print(f'An error occurred while creating the mesh: {e}')
            sys.exit(1)

    def get_existing_partitions(self, mesh_file):
        existing_partitions = []
        try:
            with h5py.File(mesh_file, 'r') as mesh:
                if 'partitionings' in mesh:
                    existing_partitions = list(mesh['partitionings'].keys())
        except Exception:
            pass
        return existing_partitions

    def calculate_dofs_per_element(self, order, element_type):
        p = order
        if   element_type == 'tet':    dofs_per_ele =    (p + 1) * (p + 2) * (p + 3)      // 6                         ; eles_per_subdom = 12
        elif element_type == 'pri':    dofs_per_ele =   ((p + 1) ** 2 * (p + 2))          // 2                         ; eles_per_subdom = 2
        elif element_type == 'hex':    dofs_per_ele =    (p + 1) ** 3                                                  ; eles_per_subdom = 1
        elif element_type == 'pyr':    dofs_per_ele =   ((p + 1) * (p + 2) * (2 * p + 3)) // 6                         ; eles_per_subdom = 6
        elif element_type == 'hexpyr': dofs_per_ele = ((((p + 1) * (p + 2) * (2 * p + 3)) // 6) + ((p + 1) ** 3)) / 2  ; eles_per_subdom = (6 + 1) / 2
        elif element_type == 'pritet': dofs_per_ele = ((((p + 1) ** 2 * (p + 2)) // 2) + ((p + 1) * (p + 2) * (p + 3)) // 6) / 2 ; eles_per_subdom = (2 + 12) / 2
        else: raise ValueError(f"Unknown element type: {element_type}")
        return dofs_per_ele * eles_per_subdom

    def calculate_nvertices(self, etype, order, dofs_aim, fvars=5):
        # Degrees of freedom per element
        dofs_per_subdom = self.calculate_dofs_per_element(order, etype)
        neles_total = dofs_aim / (fvars * dofs_per_subdom)
        nedgeints_calc = neles_total ** (1 / 3)
        nedgeints = int(np.floor(nedgeints_calc))
        nvertices = nedgeints + 1

        if nvertices < 2:
            raise ValueError("Resulting vertices is < 2. Increase DoFs.")

        return nvertices