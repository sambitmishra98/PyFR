from pyfr.solvers.baseadvec import BaseAdvectionSystem
from pyfr.solvers.euler.elements import EulerElements
from pyfr.solvers.euler.inters import (EulerIntInters, EulerMPIInters,
                                       EulerBaseBCInters)
from pyfr.solvers.euler.mortars import EulerMortarInters


class EulerSystem(BaseAdvectionSystem):
    name = 'euler'
    ef_solver = 'euler'

    elementscls = EulerElements
    intinterscls = EulerIntInters
    mpiinterscls = EulerMPIInters
    mortarinterscls = EulerMortarInters
    bbcinterscls = EulerBaseBCInters
