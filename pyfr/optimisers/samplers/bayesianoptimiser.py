import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

from pyfr.optimisers.samplers.base import BaseSampler

class BayesianOptimiser(BaseSampler):
    name = 'bayesianoptimiser'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = None

        self.objective = self.modeller.observer.objective
        if self.objective not in ['minimise',]:
            raise ValueError(f'Objective {self._objective} not supported.') 

        self.hparam = self.modeller.hparam

    def _acqf(self, best_f: torch.Tensor) -> LogExpectedImprovement:
        """
        Build the acquisition function for EI (minimisation).
        """
        gp = self.modeller.model
        return LogExpectedImprovement(model=gp, best_f=best_f, maximize=False)

    def _optimize_acqf(self, acqf: LogExpectedImprovement,
                       bounds: torch.Tensor,
                       num_restarts: int,
                       raw_samples: int) -> torch.Tensor:
        """
        Run BoTorch's optimize_acqf to maximise the acquisition function.
        Returns a (d,) tensor candidate.
        """
        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples
        )
        return candidate.squeeze(0)

    def suggest(self, *, num_restarts: int = 5, raw_samples: int = 20
                ) -> torch.Tensor:
        """
        Suggest the next hyperparameter vector (d,) that maximises EI.
        """
        gp = self.modeller.model
        if gp is None:
            raise RuntimeError("GP model not fitted; cannot suggest.")

        # Bounds for optimisation: shape (2, d)
        bounds = self.modeller.torch_bounds

        # Current best objective value
        best_val = torch.tensor([self.modeller.best_y],
                                dtype=gp.train_targets.dtype,
                                device=gp.train_targets.device)

        acq = self._acqf(best_val)
        return self._optimize_acqf(acq, bounds, num_restarts, raw_samples)

    def hparam_candidate(self):
        """
        Convert torch tensor → NumPy float64 row‑vector so
        `hparam.hparam_pending_update` receives the same shape it produces.
        """
        candidate = self.suggest().cpu().double().numpy()
        print(f"hyperparameter change: {self.hparam.hparam} -> {candidate}")
        return candidate
