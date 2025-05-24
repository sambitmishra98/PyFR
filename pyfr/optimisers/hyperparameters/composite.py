import numpy as np
from itertools import accumulate
from pyfr.optimisers.hyperparameters.base import BaseHyperparameter


class CompositeHyperparameter(BaseHyperparameter):
    """
    Combine a list of *already-declared* hyper-parameter sections into one
    optimisable vector.  Example INI::

        [hyperparameter-composite]
        capture-interval  = 20 
        file       = observers/composite_params.csv
        n-hyperparameters =  5
        components = [dtaumax, pmggroupedsteps]
        soft-bounds = [[0,  5], [0,  5], [1, 10], [1, 10], [0.0050, 0.10], ]
        hard-bounds = [[0, 10], [0, 10], [0, 10], [0, 10], [0.0001, 1.00], ]

    """
    name = 'composite'

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        # List of component section names in declared order
        sects = [s for s in intg.cfg.getliteral(cfgsect, 'components')]

        # Build concrete objects *once* and cache slice indices
        self._comps = [hp for hp in intg.hyperparameters if hp.name in sects]

        lengths = [c.n_hparams for c in self._comps]
        self._breaks = list(accumulate(lengths))

    @property
    def param(self):
        return np.concatenate([c.param for c in self._comps])

    @property
    def hparam(self):
        parts = []
        for c in self._comps:
            # Ensure we have a NumPy array
            arr = np.asarray(c.hparam)
            # Flatten any extra dims into 1-D
            parts.append(arr.ravel())
        return np.concatenate(parts)

    @hparam.setter
    def hparam(self, y):
        # Split incoming vector and delegate
        starts = [0] + self._breaks[:-1]
        for comp, s, e in zip(self._comps, starts, self._breaks):
            comp.hparam = y[s:e]
        self.config_change = any(c.config_change for c in self._comps)
