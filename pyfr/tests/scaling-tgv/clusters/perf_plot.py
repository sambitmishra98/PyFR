#!/usr/bin/env python3
"""
Plot DOF / s vs global DOFs for tet & hex benchmarks.
The script expects the CSV produced by

    pyfr benchmark postprocess --options \
         'config:solver_order' \
         'stats:observer-onerankcomputetime_mean' \
         'stats:observer-onerankcomputetime_sem' \
         'stats:mesh_gndofs' \
         'stats:mesh_nelems-.*'

It infers the element type *per row* from the only non-NaN
stats:mesh_nelems-* column; if zero or multiple are non-NaN the script
aborts to avoid ambiguity.
"""

import sys
from pathlib import Path
import re
import math

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bench_results.csv")
if not csv_path.exists():
    sys.exit(f"CSV file '{csv_path}' not found")

df = pd.read_csv(csv_path)

# ----------------------------------------------------------------------
# 1. infer element type
nelem_cols = [c for c in df.columns if re.match(r"stats:mesh_nelems-.*", c)]
if not nelem_cols:
    sys.exit("No 'stats:mesh_nelems-*' columns found in CSV")

def _deduce_etype(row):
    non_nan = [c for c in nelem_cols if not math.isnan(row[c])]
    if len(non_nan) != 1:
        raise ValueError(
            f"Row '{row['file-name']}' has {len(non_nan)} non-NaN nelem columns "
            f"(expected exactly 1).")
    return non_nan[0].split("-")[-1]       # 'stats:mesh_nelems-hex' -> 'hex'

df["etype"] = df.apply(_deduce_etype, axis=1)

# ----------------------------------------------------------------------
# map integrator scheme → stage count
stage_map = {
    "euler": 1, "rk1": 1,
    "rk2": 2, "rk3": 3, "rk4": 4,
    "rk45": 5, "rk46": 6
}
df["nstages"] = df["config:solver-time-integrator_scheme"].str.lower().map(stage_map).fillna(1)

# throughput scaled by nstages
df["throughput"]     = df["stats:mesh_gndofs"] * df["nstages"] / df["stats:observer-onerankcomputetime_mean"]
df["throughput_sem"] = df["stats:mesh_gndofs"] * df["nstages"] / df["stats:observer-onerankcomputetime_mean"]**2 \
                     * df["stats:observer-onerankcomputetime_sem"]

# ----------------------------------------------------------------------
# 3. plotting
fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=200, sharey=True)
etypes   = ["tet", "hex"]
markers  = ["o", "s", "D", "^", "v", "P", "X"]
colors   = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for ax, et in zip(axes, etypes):
    sub = df[df["etype"] == et]
    if sub.empty:
        ax.set_visible(False)
        continue

    ax.set_title(et)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    orders = sorted(sub["config:solver_order"].unique())
    for idx, (p, mk, clr) in enumerate(zip(orders, markers, colors)):
        s = (sub[sub["config:solver_order"] == p].sort_values("stats:mesh_gndofs"))
        ax.errorbar(
            s["stats:mesh_gndofs"],
            s["throughput"],
            yerr=s["throughput_sem"],
            fmt=mk,
            markersize=5,
            markerfacecolor=clr,
            markeredgecolor="black",
            linestyle="-",
            linewidth=1,
            capsize=3,
            label=f"p={p}",
        )

    ax.set_xlim(left=1e5)  # set minimum x-axis limit
    ax.set_ylim(bottom=1e8)  # set minimum y-axis limit

    ax.set_xlabel("Global DOFs")
    ax.set_xlim(left=df["stats:mesh_gndofs"].min()*0.8)
    ax.legend(fontsize="small")

axes[0].set_ylabel("Throughput  (DOF s$^{-1}$)")

fig.suptitle("PyFR throughput vs problem size")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("throughput_tet_hex.png")
print("Saved throughput_tet_hex.png")
