from pyfr.optimisers.observers.base import BaseObserver

from pyfr.optimisers.observers.rankcomputetime import RankComputeTime
from pyfr.optimisers.observers.rankwaittime import RankWaitTime
from pyfr.optimisers.observers.dtaumax import PseudoTimeMax
from pyfr.optimisers.observers.meshspecs import MeshSpecs
from pyfr.optimisers.observers.intgspecs import IntegratorSpecs

from pyfr.util import subclass_where

def get_observer(name, *args, **kwargs):
    cls = subclass_where(BaseObserver, name=name)
    return cls(*args, **kwargs)
