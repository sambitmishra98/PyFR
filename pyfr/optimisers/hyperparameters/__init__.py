from pyfr.optimisers.hyperparameters.base import BaseHyperparameter

from pyfr.optimisers.hyperparameters.pmglastlevelsteps import PMGLastLevelSteps
from pyfr.optimisers.hyperparameters.pmggroupedsteps import PMGGroupedSteps
from pyfr.optimisers.hyperparameters.dtaumax import PseudoTimeMax

from pyfr.optimisers.hyperparameters.composite import CompositeHyperparameter


from pyfr.util import subclass_where

def get_hyperparameter(name, *args, **kwargs):
    cls = subclass_where(BaseHyperparameter, name=name)
    return cls(*args, **kwargs)
