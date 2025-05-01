from pyfr.optimisers.observers.base import BaseObserver, BaseObjective

from pyfr.optimisers.observers.computetimedifference import ComputeTimeDifference
from pyfr.optimisers.observers.dtaumax import PseudoTimeMax

from pyfr.util import subclass_where

def get_observer(name, intg, cfgsect):
    cls = subclass_where(BaseObserver, name=name)
    return cls(intg, cfgsect)
