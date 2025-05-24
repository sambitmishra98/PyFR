import matplotlib.pyplot as plt

import torch

from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.models import SingleTaskGP
from botorch.models import SaasFullyBayesianSingleTaskGP
from botorch.fit import fit_gpytorch_mll

from botorch.models.transforms import Normalize, Standardize
from botorch.utils.transforms import unnormalize

from pyfr.optimisers.modellers.base import BaseModeller

torch.set_default_dtype(torch.double)

class GPModeller(BaseModeller):
    name = 'gpmodeller'

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        d = self.n_hparams
        self._input_tf = Normalize(d=d)      # maps to [0,1]^d
        self._input_un = unnormalize         # will invert the transform

        self._outcome_tf = Standardize(m=1)  # maps to mean=0, std=1

        self.plot_name = intg.cfg.get(self.cfgsect, 'plot-name', None)

    def __call__(self):
        if not self.config_prepare:
            return

        self.x_np    = self.hparam.stats_hist
        self.y_np    = self.observer.stats_hist[:, 3]
        self.ystd_np = self.observer.stats_hist[:, 2]

        if len(self.x_np) != len(self.y_np):
            raise ValueError(f'Length mismatch between X and Y: '
                             f'{len(self.x_np)} ≠ {len(self.y_np)}')

        # Update model with new data
        self.update_model()

        # Optional 1‑D visualisation
        if self.plot_name and len(self.x_np) >= 2 and self.x_np.shape[1] == 1:
            self._plot_1d_posterior()

    @property
    def best_y(self) -> float:
        """Return the current best (minimum) observed Y value."""
        if not hasattr(self, 'y_np') or len(self.y_np) == 0:
            raise RuntimeError("No observations available for best_y")
        return float(min(self.y_np))

    @property
    def torch_bounds(self) -> torch.Tensor:
        b = torch.tensor(self.bounds, dtype=torch.float64)
        return b.t().contiguous()

    def _prepare_training_data(self):
        """
        Fetches and transforms X, Y, Ystd for GP fitting.

        Returns:
            x_train (Tensor[n,d]),
            y_train (Tensor[n,1]),
            yvar_train (Tensor[n,1]) or None
        """
        # Prepare inputs
        x_train    = torch.from_numpy(self.x_np                 ).to(dtype=torch.float64)
        y_train    = torch.from_numpy(self.y_np   ).unsqueeze(-1).to(dtype=torch.float64)
        ystd_train = torch.from_numpy(self.ystd_np).unsqueeze(-1).to(dtype=torch.float64)
        yvar_train = ystd_train.pow(2)

        return x_train, y_train, yvar_train

    def _build_and_fit_gp(self, x_train, y_train, yvar_train):
        """
        Constructs and optimizes the GP model.
        """
        gp = SingleTaskGP(train_X=x_train,
                          train_Y=y_train,
                          train_Yvar=yvar_train,
                          input_transform=self._input_tf,
                          outcome_transform=self._outcome_tf
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        return gp

    def update_model(self):
        """
        Update fitted GP model based on current history.
        """
        x_train, y_train, yvar_train = self._prepare_training_data()
        self.model = self._build_and_fit_gp(x_train, y_train, yvar_train)
        return self.model


    def _plot_1d_posterior(self):
        """
        Generate and save a 1D posterior plot if applicable.
        """
        # Only valid when d=1
        lo, hi = self.bounds[0]

        x_plot = torch.linspace(lo, hi, 400).unsqueeze(-1)
        post = self.model.posterior(x_plot)
        mean = post.mean.squeeze().cpu().detach().numpy()
        std = post.variance.sqrt().squeeze().cpu().detach().numpy()
        xp = x_plot.squeeze().cpu().detach().numpy()

        plt.fill_between(xp, mean - 2 * std, mean + 2 * std, alpha=0.3)
        plt.plot(xp, mean, label='GP mean')
        plt.scatter(self.x_np, self.y_np, c='k', label='observations')
        plt.xlabel('Hyperparameter')
        plt.ylabel('Objective')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_name, dpi=160)
        plt.close()
