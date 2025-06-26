from __future__ import annotations
"""pyfr.relocator.plotting  – quick visual diagnostic helpers

Creates two kinds of PNGs right next to your run directory:

* **Per‑rank graph**  (`plot_rank_graph`) – a node for every *destination* slot
  in ``MetaMesh.smeshes`` on *this* rank.  Node label contains:

    rank:{dest}\nquad:{n}\ntri:{m}  (extra etypes auto‑added)

  Edges are undirected and labelled with the number of faces whose two
  neighbouring elements currently sit in *different* sub‑meshes on the
  same MPI rank **(now correctly detected across *all* sub‑meshes).**

* **Global graph** (`plot_global_graph`) – one node per MPI rank.  Node
  label contains ``rank:{r}`` and element counts aggregated over all
  sub‑meshes on that rank.  Edge label = #MPI faces between the two ranks.

Visual tweaks requested by the user:
------------------------------------
* Mild colour palette – ranks are coloured **blue → red → green → gold → plum → sky** (cycling if >6).
* Node circles are **2× larger** for better label fit.
* Edge labels now shown in per‑rank graphs as well.

Dependencies: *networkx* + *matplotlib* + *mpi4py* (assumed present).
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import networkx as nx
from mpi4py import MPI
import math

from pyfr.mpiutil import get_comm_rank_root
from pyfr.relocator.submesh import SubMesh  # type hints only

__all__ = [
    "plot_rank_graph",
    "plot_global_graph",
]

# ---------------------------------------------------------------------------
#  Colour palette (mild, paper‑friendly)
# ---------------------------------------------------------------------------
_PALETTE = [
    "cornflowerblue",  # rank‑0
    "salmon",          # rank‑1
    "mediumseagreen",  # rank‑2
    "gold",            # rank‑3
    "plum",            # rank‑4
    "lightskyblue",    # rank‑5
]

# =============================================================================
#  Helper utilities
# =============================================================================

def _pretty(path: Path) -> str:
    """Return *path* relative to CWD if possible, else absolute."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _etype_counts(sm: SubMesh) -> Dict[str, int]:
    """Return {etype:count} with zeros filtered out."""
    return {et: n for et, n in sm.N_ei.items() if n}


def _counts_to_str(cnt: Dict[str, int]) -> str:
    """Format ``{'quad':17,'tri':5} → 'quad:17\ntri:5'`` ("·" if empty)."""
    return "\n".join(f"{k}:{v}" for k, v in cnt.items()) or "·"


# =============================================================================
#  Per‑rank graph
# =============================================================================

def plot_rank_graph(mmesh, rank: int, out: str | Path | None = None) -> None:
    """Draw *this* rank's internal SubMesh connectivity and save a PNG."""

    if out is None:
        out = Path(f"rank{rank:04d}.png")
    out = Path(out)

    G = nx.Graph()

    # ------------------------------ nodes ----------------------------------
    node_labels: Dict[int, str] = {}
    for dest, sm in mmesh.smeshes.items():
        counts = _etype_counts(sm)
        label = f"rank:{dest}\n" + _counts_to_str(counts)
        size_metric = sum(counts.values()) or 1  # avoid 0‑area nodes
        G.add_node(dest, size=size_metric)
        node_labels[dest] = label

    # ------------------------------ edges ----------------------------------
    meta = mmesh.meta
    # map (etype_code, gid) → slot
    gid_to_slot: Dict[Tuple[int, int], int] = {}
    for slot, sm in mmesh.smeshes.items():
        for et in meta.etypes:
            code = meta.e_to_i[et]
            for gid in sm.eidxs[et]:
                gid_to_slot[(code, int(gid))] = slot

    edge_weights: Dict[Tuple[int, int], int] = defaultdict(int)

    # scan **all** sub‑meshes on this rank so we catch interfaces between them
    for slot, sm in mmesh.smeshes.items():
        for et in meta.etypes:
            con = sm.con[et]
            if con.size == 0:
                continue
            gidxs = sm.eidxs[et]
            code_et = meta.e_to_i[et]

            for row_idx, faces in enumerate(con):
                src_gid = int(gidxs[row_idx])
                src_slot = gid_to_slot[(code_et, src_gid)]

                for owner, ncode, ngid, _ in faces:
                    if ncode < 0:  # boundary face
                        continue
                    dst_slot = gid_to_slot.get((int(ncode), int(ngid)))
                    if dst_slot is None or dst_slot == src_slot:
                        continue
                    a, b = sorted((src_slot, dst_slot))
                    edge_weights[(a, b)] += 1

    for (a, b), w in edge_weights.items():
        G.add_edge(a, b, weight=w)

    # ------------------------------ draw -----------------------------------
    #pos = nx.spring_layout(G, seed=42)

    R = len(G.nodes)
    pos = {n: (math.cos(2*math.pi*n/R), math.sin(2*math.pi*n/R)) for n in G.nodes}


    # 2× larger nodes than before
    sizes = [100 + 20 * G.nodes[n]['size'] for n in G.nodes]
    colors = [_PALETTE[n % len(_PALETTE)] for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=11)

    nx.draw_networkx_edges(G, pos)
    if edge_weights:
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels={k: str(v) for k, v in edge_weights.items()}, font_size=9
        )

    plt.axis("off")
    plt.tight_layout()

    plt.margins(0.15) # add 15 % padding so big circles stay inside

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"[plot] saved per‑rank graph at {_pretty(out)}")


# =============================================================================
#  Global graph
# =============================================================================

def plot_global_graph(mmesh, out: str | Path | None = None) -> None:
    """Collect MPI‑face + element counts across all ranks and draw one graph."""

    comm, rank, _ = get_comm_rank_root()

    if out is None:
        out = Path("ranks_global.png")
    out = Path(out)

    # Gather per‑rank data ---------------------------------------------------
    local_counts = _aggregate_counts(mmesh)
    local_edges = _aggregate_mpi_edges(mmesh)

    gathered_counts: List[Dict[int, Dict[str, int]]] = comm.allgather(local_counts)
    gathered_edges: List[Dict[Tuple[int, int], int]] = comm.allgather(local_edges)

    # Combine on every rank (small dicts) -----------------------------------
    all_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for cdict in gathered_counts:
        for r, edict in cdict.items():
            for et, n in edict.items():
                all_counts[r][et] += n

    all_edges: Dict[Tuple[int, int], int] = defaultdict(int)
    for edict in gathered_edges:
        for key, w in edict.items():
            all_edges[key] += w

    if rank != 0:
        return  # only root writes PNG

    # ------------------------------ build graph -----------------------------
    G = nx.Graph()
    node_labels: Dict[int, str] = {}

    for r in sorted(all_counts):
        cnt = all_counts[r]
        label = f"rank:{r}\n" + _counts_to_str(cnt)
        size_metric = sum(cnt.values()) or 1
        G.add_node(r, size=size_metric)
        node_labels[r] = label

    for (a, b), w in all_edges.items():
        G.add_edge(a, b, weight=w)

    # ------------------------------ draw -----------------------------------
    #pos = nx.kamada_kawai_layout(G)
    R = len(G.nodes)
    pos = {n: (math.cos(2*math.pi*n/R), math.sin(2*math.pi*n/R)) for n in G.nodes}


    sizes = [100 + 20 * G.nodes[n]["size"] for n in G.nodes]
    colors = [_PALETTE[n % len(_PALETTE)] for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12)

    nx.draw_networkx_edges(G, pos)
    if all_edges:
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels={k: str(v) for k, v in all_edges.items()}, font_size=10
        )

    plt.axis("off")
    plt.tight_layout()

    plt.margins(0.15) # add 15 % padding so big circles stay inside
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"[plot] saved global graph at {_pretty(out)}")

# =============================================================================
#  Aggregation helpers
# =============================================================================

def _aggregate_counts(mmesh) -> Dict[int, Dict[str, int]]:
    """Return {rank: {etype:count}} for *this* rank only."""
    total: Dict[str, int] = defaultdict(int)
    for sm in mmesh.smeshes.values():
        for et, n in sm.N_ei.items():
            total[et] += n
    return {mmesh.rank: dict(total)}


def _aggregate_mpi_edges(mmesh) -> Dict[Tuple[int, int], int]:
    """Return {(rankA, rankB): nfaces} for MPI faces owned by this rank."""
    meta = mmesh.meta
    counts: Dict[Tuple[int, int], int] = defaultdict(int)

    local_sm = mmesh.smeshes[mmesh.rank]
    for et in meta.etypes:
        con = local_sm.con[et]
        if con.size == 0:
            continue
        for faces in con:  # shape (Nf,4)
            for owner, ncode, _, _ in faces:
                if owner < 0 or owner == mmesh.rank:
                    continue  # boundary or not MPI
                a, b = sorted((mmesh.rank, int(owner)))
                counts[(a, b)] += 1
    return counts
