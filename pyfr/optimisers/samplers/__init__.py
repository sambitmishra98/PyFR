from pyfr.optimisers.samplers.base import BaseSampler

from pyfr.optimisers.samplers.samplebyexpression import SampleByExpression
from pyfr.optimisers.samplers.bayesianoptimiser import BayesianOptimiser

from pyfr.util import subclass_where

def get_sampler(name, *args, **kwargs):
    cls = subclass_where(BaseSampler, name=name)
    return cls(*args, **kwargs)
