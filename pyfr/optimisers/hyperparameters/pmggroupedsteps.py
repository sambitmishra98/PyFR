import numpy as np
from pyfr.optimisers.hyperparameters.base import BaseHyperparameter

class PMGGroupedSteps(BaseHyperparameter):
    name = 'pmggroupedsteps'

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        # Grab the base csteps tuple
        self._csteps_base = tuple(self.intg.pseudointegrator.csteps)

        # Read your groups list
        self._groups = intg.cfg.getliteral(cfgsect, 'groups')
        if len(self._groups) != len(self._csteps_base):
            raise ValueError(f"Expected {len(self._csteps_base)} group labels, got {len(self._groups)}")

        # Only non-negative labels count toward hyperparameters
        valid = {g for g in self._groups if g >= 0}
        if len(valid) != self.n_hparams:
            raise ValueError(
                f"n_hyperparameters={self.n_hparams} does not match "
                f"{len(valid)} non-negative groups"
            )

    @property
    def param(self):
        return self.condense_csteps(self.intg.pseudointegrator.csteps)
    
    @property
    def hparam(self):
        return self.condense_csteps(self.intg.pseudointegrator.csteps)

    @hparam.setter
    def hparam(self, y):
        self.config_change = True
        self.intg.pseudointegrator.csteps = self.expand_ccsteps(y)
        
    def condense_csteps(self, csteps: tuple[float]) -> np.ndarray[float]:
        return self.condense_by_group(csteps, self._groups)
    
    def expand_ccsteps(self, ccsteps: np.ndarray[float]) -> tuple[float, ...]:
        """
        ccsteps: array of length n_hparams
        Returns tuple of length len(self._groups)
        """
        out = []
        for i, g in enumerate(self._groups):
            if g < 0:
                out.append(self._csteps_base[i])
            else:
                out.append(float(ccsteps[g]))
        return tuple(out)
