class BaseSampler:
    name = None
    
    def __init__(self, intg, cfgsect):
        self.cfgsect = cfgsect
        self.intg = intg

        # Find out modeller
        
        modeller_name = intg.cfg.get(self.cfgsect, 'modeller')

        for modeller in intg.modellers:
            if modeller.name == modeller_name:
                self.modeller = modeller
                break
        else:
            raise ValueError(
                f'Modeller {modeller_name} not setup in config file.')

        # Get the list of hyperparameters from the modeller

        self.hparam = self.modeller.hparam

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
