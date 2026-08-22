from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionSystem
from pyfr.solvers.navstokes.elements import NavierStokesElements
from pyfr.solvers.navstokes.inters import (NavierStokesBaseBCInters,
                                           NavierStokesIntInters,
                                           NavierStokesMPIInters)
from pyfr.solvers.navstokes.mortars import NavierStokesMortarInters


class NavierStokesSystem(BaseAdvectionDiffusionSystem):
    name = 'navier-stokes'
    ef_solver = 'euler'

    elementscls = NavierStokesElements
    intinterscls = NavierStokesIntInters
    mpiinterscls = NavierStokesMPIInters
    mortarinterscls = NavierStokesMortarInters
    bbcinterscls = NavierStokesBaseBCInters
