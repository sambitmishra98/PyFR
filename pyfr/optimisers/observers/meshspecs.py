# pyfr/optimisers/observers/meshspecs.py

from pyfr.optimisers.observers.base import BaseObserver
from pyfr.mpiutil import get_comm_rank_root


class MeshSpecs(BaseObserver):
    name      = 'meshspecs'
    objective = None

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)
        self.interval = intg.cfg.getint(cfgsect, 'capture-interval', 0)

    def _init_observer(self, intg):
        # This is invoked twice by BaseObserver; guard duplicate prints
        if getattr(self, '_hdr_built', False):
            return
        self._hdr_built = True

        specs = intg.mesh_specifications()

        # Build CSV header
        cols = ['tprev', 'tcurr']
        for key, val in specs.items():
            if key.startswith('nelems'):
                for i in range(len(val.split(','))):
                    cols.append(f'{key}-{i}')
            
            elif key.startswith('interface'):
                for i, v in enumerate(val.split(',')):
                    ranki = i // get_comm_rank_root()[0].size
                    rankj = i %  get_comm_rank_root()[0].size
                    cols.append(f'{key}-{ranki}-{rankj}')
            else:
                cols.append(f'{key}')
            
        self._init_history(len(cols), cols)

    # ------------------------------------------------------------
    def observation(self, intg):
        specs = intg.mesh_specifications()
        return [int(v) for spec in specs.values() for v in spec.split(',')]
