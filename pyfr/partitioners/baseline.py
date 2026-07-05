from pyfr.graphutil import GraphPartitioner
from pyfr.partitioners.base import BasePartitioner


class BaselinePartitioner(BasePartitioner):
    name = 'baseline'
    has_part_weights = True
    has_multiple_constraints = True

    int_opts = {
        'seed', 'nrefine', 'ufactor', 'niter', 'nvcycles',
        'coarsen_min_mult', 'coarsen_min_abs', 'coarsen_stop_pct',
        'rb_limit', 'init_nparts_small', 'greedy_restarts_sc',
        'greedy_restarts_mc'
    }
    enum_opts = {}
    dflt_opts = {
        'seed': 2079, 'nrefine': 10, 'ufactor': 10, 'niter': 1,
        'nvcycles': 4, 'coarsen_min_mult': 20, 'coarsen_min_abs': 500,
        'coarsen_stop_pct': 80, 'rb_limit': 64, 'init_nparts_small': 16,
        'greedy_restarts_sc': 4, 'greedy_restarts_mc': 8
    }

    def _partition_graph(self, graph, partwts):
        opts = dict(self.opts)
        opts['ufactor'] /= 1000
        opts['coarsen_stop'] = opts.pop('coarsen_stop_pct') / 100

        return GraphPartitioner(**opts).partition(graph, partwts)
