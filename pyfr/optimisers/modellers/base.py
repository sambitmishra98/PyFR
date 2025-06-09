from __future__ import annotations

"""Shared base-classes for optimisation *modellers*.

This module is thin glue:
  • **Flag sharing** is handled by :class:`~pyfr.optimisers.base.FlagSyncMixin`.
  • **Bounds handling** ultimately lives in the *hyper-parameter* object which
    derives from :class:`~pyfr.optimisers.base.BoundsMixin`; we make that
    dependency *explicit* by importing it here even though the modeller only
    proxies the property.
"""

from collections.abc import Sequence
from typing import List

import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.optimisers.base import FlagSyncMixin, BoundsMixin


class BaseModeller(FlagSyncMixin, BoundsMixin):
    """Abstract base-class for all modellers.
    A *modeller* learns mapping *hyper-parameters* --> *observer* ``y``.
    """

    #: Sub-classes **must** override with a short identifier.
    name: str | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, intg, cfgsect: str, suffix: str | None = None):
        # Shared flag namespace via FlagSyncMixin
        super().__init__(intg, suffix)
        self.intg, self.cfgsect, self.suffix = intg, cfgsect, suffix

        # Locate observer / hyper-parameter referenced in the CFG section
        self.observer = self._find_instance(
            intg.observers, self.intg.cfg.get(cfgsect, "observer")
        )
        self.hparam = self._find_instance(
            intg.hyperparameters, self.intg.cfg.get(cfgsect, "hyperparameter")
        )

        self.n_hparams = self.hparam.n_hparams

        # Capture cadence (propagated through FlagSyncMixin)
        self.interval = self.intg.cfg.getint(cfgsect, "capture-interval", 0)

        # Root-rank banner
        comm, rank, root = get_comm_rank_root()
        if rank == root:
            print(
                f"[Modeller:init] '{self.name or '<unnamed>'}' "
                f"obs={self.observer.name}:{self.observer.suffix or ''}  "
                f"hp={self.hparam.name}:{self.hparam.suffix or ''}  "
                f"interval={self.interval}",
                flush=True,
            )

        # Sample cache (grows over time-steps)
        self.X: List[np.ndarray] = []
        self.y: List[float] = []
        self.ystd: List[float] = []

    # ------------------------------------------------------------------
    # Per-step entry point
    # ------------------------------------------------------------------

    def __call__(self):
        if not self.should_capture(self.intg.nsteps):
            return

        x = np.asarray(self.hparam.hparam, dtype=float).ravel()
        y = float(self.observer.observation(self.intg))
        ystd = getattr(self.observer, "current_std", None)

        self.X.append(x)
        self.y.append(y)
        if ystd is not None:
            self.ystd.append(float(ystd))

        self.fit_model(
            np.vstack(self.X),
            np.array(self.y),
            np.array(self.ystd) if self.ystd else None,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _find_instance(self, seq: Sequence, name: str):
        for obj in seq:
            if obj.name == name and obj.suffix == self.suffix:
                return obj
        raise ValueError(
            f"Missing instance '{name}-{self.suffix}' referenced in [{self.cfgsect}]."
        )

    # ------------------------------------------------------------------
    # API for concrete subclasses
    # ------------------------------------------------------------------

    def fit_model(self, x: np.ndarray, y: np.ndarray, ystd: np.ndarray | None = None):
        """Update (or build) the internal surrogate model."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience – forward *bounds* straight from the hyper-parameter.
    # ------------------------------------------------------------------

    @property
    def bounds(self):
        """Return the hyper-parameter bounds (no duplication here)."""
        return self.hparam.bounds


# ----------------------------------------------------------------------
# Trivial implementation for smoke-tests
# ----------------------------------------------------------------------


class EmptyModeller(BaseModeller):
    """A do-nothing modeller useful for plumbing checks."""

    name = "empty"

    def fit_model(self, x, y, ystd=None):
        # Intentionally left blank
        pass
