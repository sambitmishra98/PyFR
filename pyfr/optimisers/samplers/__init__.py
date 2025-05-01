from pyfr.optimisers.samplers.base import BaseSampler
from pyfr.util import subclass_where

from pyfr.optimisers.samplers.samplebyexpression import SampleByExpression
from pyfr.optimisers.samplers.bayesianoptimiser import BayesianOptimiser

def get_sampler(name, intg, cfgsect):
    cls = subclass_where(BaseSampler, name=name)
    return cls(intg, cfgsect)
