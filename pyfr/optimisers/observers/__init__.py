from pyfr.optimisers.observers.base import BaseObserver

from pyfr.optimisers.observers.rankcomputetime import RankComputeTime
from pyfr.optimisers.observers.dtaumax import PseudoTimeMax

from pyfr.util import subclass_where

def get_observer(name, *args, **kwargs):
    cls = subclass_where(BaseObserver, name=name)
    return cls(*args, **kwargs)
