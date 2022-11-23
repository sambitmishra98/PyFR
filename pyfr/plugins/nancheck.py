import numpy as np

from pyfr.plugins.base import BasePlugin


class NaNCheckPlugin(BasePlugin):
    name = 'nancheck'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nsteps = self.cfg.getint(self.cfgsect, 'nsteps')

    def __call__(self, intg):
        if intg.nacptsteps % self.nsteps == 0:
            if any(np.isnan(np.sum(s)) for s in intg.soln):

                if   intg.rewind == True:  print("Expected to rewind now.")
                elif intg.rewind == False: print("Rewind is supposed to be set to true. Something is wrong.")

                if intg.rewind == None:
                    raise RuntimeError(f'NaNs detected at t = {intg.tcurr}')
