import numpy as np

class BaseModeller:
    name = None
    
    def __init__(self, intg, cfgsect, suffix=None):
        self.cfgsect = cfgsect
        
        self.suffix = suffix

        # Find out objective and hyperparameters
        observer_name = intg.cfg.get(self.cfgsect, 'observer')
        for obs in intg.observers:
            if obs.name == observer_name and obs.suffix == self.suffix:
                self.observer = obs
                break
        else:
            raise ValueError(f'Set up observer-{observer_name}-{suffix}.')
 
        # Get the hyperparameter
        hparam_name = intg.cfg.get(self.cfgsect, 'hyperparameter')
        for hparam in intg.hyperparameters:
            if hparam.name == hparam_name and hparam.suffix == self.suffix:
                self.hparam = hparam
                break
        else:
            raise ValueError(f'Set up hyperparameter-{hparam_name}-{suffix}.')

        self.n_hparams = self.hparam.n_hparams

        self.interval = intg.cfg.getint(self.cfgsect, 'capture-interval', 0)

        self.config_prepare = False 

    def fit_model(self, x, y, ystd=None):
        pass

    @property
    def config_prepare(self):
        """Return the config prepare status."""
        return self._config_prepare

    @config_prepare.setter
    def config_prepare(self, y):
        """Set the config prepare status."""
        self._config_prepare = y
        self.hparam.config_prepare = self.observer.config_prepare = y

    @property
    def interval(self):
        """Return the interval."""
        return self._interval
    
    @interval.setter
    def interval(self, y):
        """Set the interval."""
        self._interval = y
        self.hparam.interval = self.observer.interval = 0

    @property
    def bounds(self) -> list[tuple[float, float]]:
        init_lo, init_hi = self.hparam.bounds          # both (d,)
        lo, hi = init_lo.copy(), init_hi.copy()

        if hasattr(self, 'X') and len(self.X):
            samp_lo = np.min(self.X, axis=0)
            samp_hi = np.max(self.X, axis=0)
            lo = np.minimum(lo, samp_lo)
            hi = np.maximum(hi, samp_hi)

        return list(zip(lo, hi))                      # [(lo1,hi1), …]


class EmptyModeller(BaseModeller):
    """Empty Modeller."""
    name = 'empty'
