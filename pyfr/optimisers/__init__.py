
from pyfr.optimisers.base import (BaseOptimiser, 
                                  BaseGlobalOptimiser, 
                                  BaseLocalOptimiser)
from pyfr.optimisers.binarystepper import BinaryStepper
from pyfr.optimisers.PIController import PIController
from pyfr.optimisers.zeta import Zeta
from pyfr.optimisers.bayesianoptimiser import BayesianOptimiser
from pyfr.optimisers.botorch import BoTorch

from pyfr.util import subclass_where

def get_optimiser(prefix, name, *args, **kwargs):
    cls = subclass_where(BaseOptimiser, prefix=prefix, name=name)
    return cls(*args, **kwargs)
