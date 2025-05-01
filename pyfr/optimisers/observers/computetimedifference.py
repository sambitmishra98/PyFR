from pyfr.optimisers.observers import BaseObjective


class ComputeTimeDifference(BaseObjective):
    name = 'computetimedifference'
    objective = 'minimise'

    @staticmethod
    def observation(intg):
        return sum(intg.system.ctimediff)
