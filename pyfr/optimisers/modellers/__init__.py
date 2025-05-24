from pyfr.optimisers.modellers.base import BaseModeller, EmptyModeller
from pyfr.util import subclass_where

from pyfr.optimisers.modellers.gpmodeller import GPModeller

def get_modeller(name, *args, **kwargs):
    cls = subclass_where(BaseModeller, name=name)
    return cls(*args, **kwargs)
