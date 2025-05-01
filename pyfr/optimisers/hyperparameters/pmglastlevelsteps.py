import numpy as np

from pyfr.optimisers.hyperparameters.base import BaseHyperparameter

class PMGLastLevelSteps(BaseHyperparameter):
    name = 'pmglastlevelsteps'

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)
        mgsect = 'solver-dual-time-integrator-multip'

        if mgsect not in intg.cfg.sections():
            raise ValueError(
                f'Multi-P smoothing iterations observation'
                f' requires {mgsect} section.')

    @property
    def param(self):
        return self.intg.pseudointegrator.csteps[-1:]

    @property
    def hparam(self):
        csteps = self.intg.pseudointegrator.csteps[-1:]
        return np.array([csteps], dtype=np.float64)

    @hparam.setter
    def hparam(self, y):
        self.config_change = True
        # Tuple to list
        csteps = list(self.intg.pseudointegrator.csteps)

        # Set the new value back in tuple
        self.intg.pseudointegrator.csteps = tuple(csteps[:-1] + [y])
