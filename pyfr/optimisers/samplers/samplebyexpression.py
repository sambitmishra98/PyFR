from pyfr.optimisers.samplers.base import BaseSampler

class SampleByExpression(BaseSampler):
    name = 'samplebyexpression'

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        self.sampling_expression = intg.cfg.get(self.cfgsect, 'expression-x')

        # Parse and set up lambda function to sample the next step
        self.next_value = eval(f'lambda x: {self.sampling_expression}',)

    def hparam_candidate(self):
        nval = self.next_value(self.hparam.hparam)
        print(f"hyperparameter change: {self.hparam.hparam} -> {nval}")
        return nval
