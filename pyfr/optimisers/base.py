# pyfr/optimisers/base.py
import csv
import pathlib

import numpy as np


def find_instance(seq, name, suffix):
    for obj in seq:
        if obj.name == name and obj.suffix == suffix:
            return obj

        print(f"Checking {obj.name}-{obj.suffix} against {name}-{suffix}")

    raise ValueError(f"Instance '{name}-{suffix}' not found. Existing: {seq}")


class HistoryMixin:
    """
    Collects fixed-width rows and (optionally) dumps new ones to CSV.
    """

    # ------------------------------------------------------------------ #
    # setup
    def _init_history(self, n_cols: int, colnames: list[str] | None = None):
        self._cols_hdr = colnames or [f'col{i}' for i in range(n_cols)]
        self._rows: list[tuple] = []      # permanent
        self._pending: list[tuple] = []   # temp
        self._last_row: tuple | None = None

    def append_row(self, row):
        tup = tuple(row)
        self._rows.append(tup)
        self._pending.append(tup)
        self._last_row = tup


    # helper for integrator -------------------------------------------------
    def last_row_as_dict(self) -> dict:
        """
        Return {header_i: value_i, …} for the most recent row,
        or {} if no row was ever appended.
        """
        if self._last_row is None:
            return {}
        return dict(zip(self._cols_hdr, self._last_row))

    @property
    def history(self):
        return self._rows

    def dump_csv(self, filepath: str | pathlib.Path, *, flush=True):
        if not self._pending:
            return

        path = pathlib.Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()

        with path.open('a', newline='') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(self._cols_hdr)
            w.writerows(self._pending)
            if flush:
                f.flush()

        self._pending.clear()

    def init_csv(self, cfg, cfgsect, header, *, filekey='file', headerkey='header'):
        # Determine the file path
        fname = cfg.get(cfgsect, filekey)

        # Append the '.csv' extension
        if not fname.endswith('.csv'):
            fname += '.csv'

        # Open for appending
        outf = open(fname, 'a')

        # Output a header if required
        if outf.tell() == 0 and cfg.getbool(cfgsect, headerkey, True):
            print(header, file=outf)

        # Return the file
        return outf


class FlagSyncMixin:
    """
    Share three common flags across *all* optimiser helpers which carry
    the same <suffix> inside the same time-integrator:

        * config_prepare   (bool)
        * config_change    (bool)
        * interval         (int, ≥0)

    Usage
    -----
    class MyHelper(HistoryMixin, FlagSyncMixin):
        def __init__(self, intg, cfgsect, suffix=None):
            FlagSyncMixin.__init__(self, intg, suffix)
            ...
    """

    # every Intg gets one private dict for each suffix
    _intg_namespace = '_opt_shared'

    # ..................................................................
    def __init__(self, intg, suffix):
        ns = getattr(intg, self._intg_namespace, {})
        key = suffix or ''                       # None → empty suffix
        self._shared = ns.setdefault(key, {
            'config_prepare': False,
            'config_change' : False,
            'interval'      : 0,
        })
        setattr(intg, self._intg_namespace, ns)

    # ..................................................................
    # three properties transparently forwarded to the shared dict

    @property
    def config_prepare(self):          # noqa: D401
        return self._shared['config_prepare']

    @config_prepare.setter
    def config_prepare(self, value: bool):
        self._shared['config_prepare'] = bool(value)

    # ..................................................................

    @property
    def config_change(self):
        return self._shared['config_change']

    @config_change.setter
    def config_change(self, value: bool):
        self._shared['config_change'] = bool(value)

    # ..................................................................

    @property
    def interval(self):
        return self._shared['interval']

    @interval.setter
    def interval(self, value: int):
        iv = int(value)
        if iv < 0:
            raise ValueError('interval must be ≥ 0')
        self._shared['interval'] = iv

    # ..................................................................
    # small convenience helper (identical to your earlier logic)

    def should_capture(self, nsteps: int) -> bool:
        """
        True when a helper should emit its row this step.
        Call from __call__() of Observer / Hyper-parameter / …
        """
        if self.interval == 0:
            return self.config_prepare
        return nsteps % self.interval == 0


class BoundsMixin:
    """
    Share handling of *soft* and *hard* bounds.

    After `_init_bounds(cfg, cfgsect, d_expected)` you will have  

        self.soft_bounds : ndarray shape (d, 2)
        self.hard_bounds : ndarray shape (d, 2)

    plus a convenience read-only `bounds` property that returns the stacked
    array  [[soft_lo, …], [soft_hi, …], [hard_lo, …], [hard_hi, …]].
    """

    # ..................................................................

    def _init_bounds(self, cfg, cfgsect: str, d_expected: int | None = None):
        """
        Parse  soft-bounds / hard-bounds  from the CFG section.

        Each entry may be written either as  [(lo, hi), …]  or its transpose;
        both wind up in shape (d, 2).
        """
        sb_raw = np.asarray(cfg.getliteral(cfgsect, 'soft-bounds'), dtype=float)
        hb_raw = np.asarray(cfg.getliteral(cfgsect, 'hard-bounds'), dtype=float)

        soft = sb_raw.T if sb_raw.shape[0] == 2 else sb_raw
        hard = hb_raw.T if hb_raw.shape[0] == 2 else hb_raw

        if soft.shape != hard.shape:
            raise ValueError('soft-bounds and hard-bounds shapes differ')

        d, w = soft.shape
        if w != 2:
            raise ValueError('bounds must be (d,2) after transpose')

        if d_expected is not None and d != d_expected:
            raise ValueError(f'expect {d_expected} parameters, got {d}')

        self.soft_bounds = soft.copy()
        self.hard_bounds = hard.copy()

        # stacked view: (4, d)
        self._bounds_view = np.vstack(
            [soft[:, 0], soft[:, 1], hard[:, 0], hard[:, 1]]
        )

    @property
    def bounds(self) -> np.ndarray:
        """Return stacked [[soft_lo],[soft_hi],[hard_lo],[hard_hi]] (shape 4×d)."""
        return self._bounds_view
