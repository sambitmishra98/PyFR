import numpy as np
from itertools import accumulate
from pyfr.optimisers.hyperparameters.base import BaseHyperparameter
from pyfr.util import subclass_where


class CompositeHyperparameter(BaseHyperparameter):
    """
    Combine a list of *already-declared* hyper-parameter sections into one
    optimisable vector.  Example INI::

        [hyperparameter-combined]
        components = dtaumax, pmggroupedsteps
        file       = observers/combined_params.csv
    """
    name = 'composite'

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        # List of component section names, keep declared order
        self._comp_sects = [
            s.strip() for s in intg.cfg.getliteral(cfgsect, 'components')
        ]

        # Build concrete objects *once* and cache slice indices
        self._comps = [subclass_where('name', intg.cfg.get(s, 'type'))
                       (intg, s) for s in self._comp_sects]

        lengths = [c.n_hparams for c in self._comps]
        self._breaks = list(accumulate(lengths))
        self._n_hparams = sum(lengths)

    # ------------------------------------------------------------------ attrs
    @property
    def n_hparams(self):
        return self._n_hparams

    @property
    def param(self):
        return np.concatenate([c.param for c in self._comps])

    @property
    def hparam(self):
        return np.concatenate([c.hparam for c in self._comps])

    @hparam.setter
    def hparam(self, y):
        # Split incoming vector and delegate
        starts = [0] + self._breaks[:-1]
        for comp, s, e in zip(self._comps, starts, self._breaks):
            comp.hparam = y[s:e]
        self.config_change = any(c.config_change for c in self._comps)

    # ---------------------------------------------------------------- bounds
    @property
    def bounds(self):
        """Return concatenated soft-bounds if all children define them."""
        if any(c.bounds is None for c in self._comps):
            return None
        return np.concatenate([c.bounds for c in self._comps])
