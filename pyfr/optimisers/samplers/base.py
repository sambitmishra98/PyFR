class BaseSampler:
    name = None
    
    def __init__(self, intg, cfgsect, suffix=None):
        self.cfgsect = cfgsect
        self.intg = intg
        
        self.suffix = suffix

        modeller_name = intg.cfg.get(self.cfgsect, 'modeller')

        for modeller in intg.modellers:
            if modeller.name == modeller_name and modeller.suffix == self.suffix:
                self.modeller = modeller
                break
        else:
            raise ValueError(f'Set up modeller-{modeller_name}-{self.suffix}.')

        # Get the list of hyperparameters from the modeller

        self.hparam = self.modeller.hparam

        self.n_hparams = self.hparam.n_hparams

        self.interval = intg.cfg.getint(self.cfgsect, 'capture-interval')

        self.config_prepare = False

    def __call__(self):

        if self.config_prepare:
            self.hparam.hparam_pending_update.append(self.hparam_candidate())

        self.config_prepare = self.intg.nsteps % self.interval == 0

    @property
    def config_prepare(self):
        """Return the config prepare status."""
        return self._config_prepare
    
    @config_prepare.setter
    def config_prepare(self, y):
        self._config_prepare = y
        self.modeller.config_prepare = y

    @property 
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, y):
        self._interval = y
        self.modeller.interval = y
