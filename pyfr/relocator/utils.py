from __future__ import annotations
# Location: pyfr/relocator/crprint_utils.py

# export FORCE_COLOR=1

import sys
import typing
from termcolor import colored
from pprint import pformat
from pyfr.mpiutil import get_comm_rank_root

from dataclasses import dataclass
from itertools   import chain
from typing      import Any, Dict, List, Sequence

import numpy as np
from mpi4py import MPI
from tabulate import tabulate

if typing.TYPE_CHECKING:
    from pyfr.relocator.metamesh import MetaMesh, SubMesh

# ────────────────────────────────────────────────────────────────────────
# Convenience coloured print helpers
# -----------------------------------------------------------------------

def crprint(rank, *args, sep: str = " ", end: str = "\n", flush: bool = True):
    """Colour‑by‑Rank print.

    * If *rank* is ``-1`` auto‑detect via :pyfunc:`get_comm_rank_root`.
    * Message is printed colourised to stdout **and** appended (plain) to
      ``rank-<rank>.txt`` for post‑mortem inspection.
    """
    comm, this_rank, _ = get_comm_rank_root()
    if rank == -1:
        rank = this_rank

    colour_map = {0: "blue", 1: "green", 2: "red", 3: "magenta"}
    colour = colour_map.get(rank, "white")

    msg_body   = sep.join(str(a) for a in args)
    msg        = f"[Rank {rank}] {msg_body}"

    print(colored(msg, colour), end=end)
    if flush:
        sys.stdout.flush()

    with open(f"rank-{rank}.txt", "a", encoding="utf-8") as f:
        f.write(msg + end)
        if flush:
            f.flush()


def crpprint(rank, obj, flush: bool = True):
    """Pretty‑print an *obj* in colour with rank prefix + tee to file."""
    crprint(rank, pformat(obj), flush=flush)


# ────────────────────────────────────────────────────────────────────────
# Compact statistics container (flat → vector for easy MPI reduce)
# -----------------------------------------------------------------------

@dataclass
class _Stats:
    """One row in the sub‑mesh inventory table (all integers)."""

    elem_flat : List[int]          # [tri, quad, …]
    elm_sum   : int               # total elements (all etypes)
    iface_int : int               # internal faces
    iface_mpi : List[int]         # faces to each rank
    iface_bc  : List[int]         # faces to each BC id
    iface_sum : int               # total faces

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def to_vector(self) -> np.ndarray:
        """Pack into 1‑D NumPy array (int64) ready for MPI reduction."""
        return np.array(
            self.elem_flat +                # element counts per etype
            [self.elm_sum] +                # Σ elements
            [self.iface_int] +              # interior interfaces
            self.iface_mpi +                # MPI interfaces per rank
            self.iface_bc  +                # BC interfaces per id
            [self.iface_sum],               # Σ interfaces
            dtype=np.int64,
        )

    @classmethod
    def from_vector(cls, vec: Sequence[int], netype: int, nrank: int, nbc: int) -> "_Stats":
        """Inverse of :pyfunc:`to_vector` – used on rank‑0 after MPI reduce."""
        n_elem_cols = netype
        i = 0
        elem_flat  = list(vec[i : i + n_elem_cols]);      i += n_elem_cols
        elm_sum    = int(vec[i]);                         i += 1
        iface_int  = int(vec[i]);                         i += 1
        iface_mpi  = list(vec[i : i + nrank]);            i += nrank
        iface_bc   = list(vec[i : i + nbc]);              i += nbc
        iface_sum  = int(vec[i])

        return cls(elem_flat, elm_sum, iface_int, iface_mpi, iface_bc, iface_sum)


# ────────────────────────────────────────────────────────────────────────
# Pretty inventory / sanity‑check tables
# -----------------------------------------------------------------------

class TablePrinter:
    """Generate barrier‑serialised GitHub‑style tables for debug output."""

    def __init__(self, mmesh: "MetaMesh") -> None:
        self.comm: MPI.Comm = mmesh.comm
        self.ranks          = list(range(self.comm.size))
        self.etypes         = sorted(mmesh.etypes)
        self.bcs            = sorted(mmesh.bc_map.values())

        # Build column headers ─ each etype has one column (no curved)
        elem_cols  = list(self.etypes)
        iface_cols = (
            ["interior"] +                      # internal faces
            [f"R{r}" for r in self.ranks] +     # per‑rank MPI
            [f"BC{b}" for b in self.bcs] +      # per‑BC
            ["∀ ints"]                          # Σ interfaces
        )

        self.headers  = ["sub‑mesh"] + elem_cols + ["∀ elems"] + iface_cols
        self.colalign = ("left",) + ("right",) * (len(self.headers) - 1)

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # Helpers on a single sub‑mesh
    # ----------------------------------------------------------------

    def _elem_flat(self, sm: SubMesh) -> List[int]:
        return [sm.N_ei.get(et, 0) for et in self.etypes]

    def _iface_blocks(self, sid: int, sm: SubMesh):
        cnt_int  = 0
        cnt_rank = {r: 0 for r in self.ranks}
        cnt_bc   = {b: 0 for b in self.bcs}

        for et, ten in sm.con.items():
            own_gids = set(sm.eidxs[et])
            
            for lid, gid in enumerate(own_gids):
                for f in range(ten.shape[1]):
                    nrank, nt, neid, nface = ten[lid, f]
                    if nrank == sid and nt >= 0:
                        cnt_int += 1                       # internal
                    elif nrank >= 0:
                        cnt_rank[nrank] += 1               # MPI
                    else:
                        cnt_bc[nface]  += 1                # boundary

        ifΣ = cnt_int + sum(cnt_rank.values()) + sum(cnt_bc.values())
        return cnt_int, [cnt_rank[r] for r in self.ranks], [cnt_bc[b] for b in self.bcs], ifΣ



    def _row_stats(self, sid: int, sm: SubMesh) -> _Stats:
        elem  = self._elem_flat(sm)
        elmΣ  = sm.N_i
        intf, if_r, if_bc, ifΣ = self._iface_blocks(sid, sm)
        return _Stats(elem, elmΣ, intf, if_r, if_bc, ifΣ)

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # Public API
    # ----------------------------------------------------------------

    def print_local_tables(self, mmesh: "MetaMesh", colours: Dict[int, str] | None = None, ) -> None:
        """Print one coloured table per rank (barrier‑serialised)."""
        if colours is None:
            colours = {0: "blue", 1: "green", 2: "red", 3: "magenta"}

        rank = self.comm.rank
        rows: List[List[Any]] = []

        # Local accumulation vector for per‑rank totals
        totals = np.zeros_like(self._row_stats(rank, next(iter(mmesh.smeshes.values()))).to_vector())

        for sid, sm in sorted(mmesh.smeshes.items()):
            s = self._row_stats(sid, sm)
            row_disp = [self._fmt(x) for x in s.to_vector()]
            rows.append([sid] + row_disp)
            totals += s.to_vector()          # keep numeric for Σ

        rows.append(["Σ"] + [self._fmt(x) for x in totals.tolist()])

        block = tabulate(rows, self.headers, tablefmt="github", colalign=self.colalign)

        # Barrier‑serialise so ranks print sequentially (readable order)
        for r in self.ranks:
            self.comm.Barrier()
            if r == rank:
                border = colored("─" * len(block.splitlines()[0]), colours.get(r, "white"))
                for ln in block.splitlines():
                    print(colored(f"[Rank {rank}] {ln}", colours.get(r, "white")))
        self.comm.Barrier()

    # ----------------------------------------------------------------
    def print_global_totals(self, mmesh: "MetaMesh", label: str = "ΣΣΣ") -> None:
        """Reduce across ranks and print a single yellow bold Σ line (rank‑0)."""
        vec_local = np.zeros(len(self.headers) - 1, dtype=np.int64)
        for sid, sm in mmesh.smeshes.items():
            vec_local += self._row_stats(sid, sm).to_vector()

        vec_global = np.zeros_like(vec_local)
        self.comm.Reduce(vec_local, vec_global, op=MPI.SUM, root=0)

        if self.comm.rank == 0:
            disp = [self._fmt(x) for x in vec_global.tolist()]

            table = tabulate([[label] + disp], self.headers, tablefmt="github", colalign=self.colalign)
            indent = " " * len("[Rank 0] ")
            for ln in table.splitlines():
                print(indent + colored(ln, "yellow", attrs=["bold"]))

    @staticmethod
    def _fmt(val: int | str) -> str:
        """Return '·' for 0, else the original value as str."""
        return "·" if val == 0 else str(val)