from pyfr.optimisers.observers import BaseObserver

class OneRankComputeTime(BaseObserver):
    name = 'onerankcomputetime'
    objective = 'minimise'

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)
        self.interval = intg.cfg.getint(cfgsect, 'capture-interval', 0)

        comm, rank, root = get_comm_rank_root()

    def _init_observer(self, intg):
        # This is invoked twice by BaseObserver; guard duplicate prints
        if getattr(self, '_hdr_built', False):
            return
        self._hdr_built = True

        # Build CSV header
        cols = ['tprev','tcurr','mean','sem'] 
        self._init_history(len(cols), cols)

    def observation(self, intg):
        cnt = len(intg.ctimediff)
        mean = sum(intg.ctimediff) / cnt / 1e9
        sem = (sum((x/1e9 - mean) ** 2 for x in intg.ctimediff) / cnt) ** 0.5
        intg.ctimediff.clear()

        return [mean, sem]