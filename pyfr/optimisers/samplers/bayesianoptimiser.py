import numpy as np

import torch
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf

from botorch.acquisition.objective import GenericMCObjective



from pyfr.optimisers.samplers.base import BaseSampler

torch.set_default_dtype(torch.double)

class BayesianOptimiser(BaseSampler):
    name = 'bayesianoptimiser'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = None

        self.objective = self.modeller.observer.objective
        if self.objective not in ['minimise',]:
            raise ValueError(f'Objective {self._objective} not supported.') 

        self.hparam = self.modeller.hparam

    def setup_acqf(self, model):
        
        # 1. Build baseline set of past Xs in normalized space
        X_baseline = self.modeller._input_tf(
            torch.from_numpy(self.modeller.x_np).to(dtype=torch.float64)
        )  # (n, d) in [0,1]^d

        if self.objective == 'minimise':
            minimize_obj = GenericMCObjective(lambda Y, X: -Y[..., 0])
        else:
            minimize_obj = GenericMCObjective(lambda Y, X: Y[..., 0])

        return qLogNoisyExpectedImprovement(model=model, 
                                            X_baseline = X_baseline,
                                            objective = minimize_obj)

    def _optimize_acqf(self, acqf: qLogNoisyExpectedImprovement) -> torch.Tensor:
        """
        Run BoTorch's optimize_acqf to maximise the acquisition function.
        Returns a (d,) tensor candidate.
        """

        # 3. Define bounds in unit cube
        bounds = torch.stack([torch.zeros(self.n_hparams, dtype=torch.double),
                               torch.ones(self.n_hparams, dtype=torch.double)])

        candidate, _ = optimize_acqf(acq_function=acqf, bounds=bounds, 
                                     q=1, num_restarts=5, raw_samples=10
                                    )
        return candidate.squeeze(0)

    def suggest(self, *, num_restarts: int = 5, raw_samples: int = 20) -> torch.Tensor:
        if self.modeller.model is None:
            raise RuntimeError("GP model not fitted; cannot suggest.")

        acqf = self.setup_acqf(self.modeller.model)
        X_next, _ = self._optimize_acqf(acqf)
        x_next = self.modeller._input_un(X_next, bounds=self.modeller.torch_bounds)

        return x_next.cpu().numpy().ravel()

    def hparam_candidate(self):
        """
        Convert torch tensor → NumPy float64 row‑vector so
        `hparam.hparam_pending_update` receives the same shape it produces.
        """

        cand = self.suggest()
        # If it's already a NumPy array, just ensure dtype=float64
        if isinstance(cand, np.ndarray):
            candidate = cand.astype(np.float64)
        else:
            # Otherwise assume it's a Tensor
            candidate = cand.cpu().double().numpy()

        print(f"hyperparameter change: {self.hparam.hparam} -> {candidate}")
        return candidate
