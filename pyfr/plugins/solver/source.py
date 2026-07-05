import math
import re

from pyfr.plugins.solver.base import BaseSolverPlugin
from pyfr.util import first


class SourcePlugin(BaseSolverPlugin):
    name = 'source'
    systems = '.*'
    dimensions = '2|3'

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        convars = first(intg.system.ele_map.values()).convars

        subs = self.cfg.items('constants')
        subs |= dict(x='ploc[0]', y='ploc[1]', z='ploc[2]')
        subs |= dict(abs='fabs', pi=math.pi)
        subs |= {v: f'u[{i}]' for i, v in enumerate(convars)}

        self.src_exprs = [self.cfg.getexpr(cfgsect, v, subs=subs)
                          for v in convars]

        self.ploc_in_src = any(re.search(r'\bploc\b', ex)
                               for ex in self.src_exprs)
        self.soln_in_src = any(re.search(r'\bu\b', ex)
                               for ex in self.src_exprs)

        self._bind_system(intg)

    def _bind_system(self, intg):
        for etype, eles in intg.system.ele_map.items():
            eles.add_src_macro('pyfr.plugins.solver.kernels.source', 'source',
                               {'src_exprs': self.src_exprs},
                               ploc=self.ploc_in_src, soln=self.soln_in_src)

    def post_rebalance(self, intg, exchangers):
        self._bind_system(intg)
