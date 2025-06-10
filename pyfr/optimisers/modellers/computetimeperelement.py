from __future__ import annotations

from pyfr.optimisers.modellers.base import BaseModeller


class ComputeTimePerElementModeller(BaseModeller):
    name = 'computetimeperelement'

    def initialise_csv_colnames(self):

        d = self.n_hparams
        colnames = (
            ['tstart', 'tcurr'] +
            [f'h{i}'    for i in range(d)] +
            [f'mean{i}' for i in range(d)] +
            [f'std{i}'  for i in range(d)]
        )

        return colnames

    def process_model(self, xrow, yrow):
        n = len(xrow)
        means = [yrow[i]     / xrow[i] for i in range(n)]
        stds  = [yrow[i + n] / xrow[i] for i in range(n)]
        return (list(xrow), means, stds)