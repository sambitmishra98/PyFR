# pyfr/partitioners/construct.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pyfr.mpiutil import comm, rank, root, mpi


# -----------------------------
# Helpers: disps / targets
# -----------------------------
def _equal_targets(nelems_g: int, nparts: int) -> list[int]:
    # stable remainder distribution: lowest ranks get +1
    q, r = divmod(int(nelems_g), int(nparts))
    return [q + (1 if i < r else 0) for i in range(int(nparts))]


def _centroids_from_mesh(mesh, etypes, eidxs) -> dict[str, np.ndarray]:
    """
    Compute centroids for *local* elements only; shape (Ne_local, 3).
    Uses mesh.spts[et] expected as (npts, Ne_global, nd) or similar;
    adjust if your mesh stores it as (npts, Ne_local, nd) already.
    """
    out = {}
    for et in etypes:
        gids = np.asarray(eidxs.get(et, ()), dtype=np.int64)
        if gids.size == 0:
            out[et] = np.zeros((0, 3), dtype=np.float64)
            continue

        spts = mesh.spts.get(et, None)
        if spts is None:
            out[et] = np.zeros((gids.size, 3), dtype=np.float64)
            continue

        spts = np.asarray(spts, dtype=np.float64)

        # ---- IMPORTANT ASSUMPTION ----
        # If spts is global (npts, Ne_global, nd), slice by gids.
        # If spts is already local (npts, Ne_local, nd), gids should be 0..Ne_local-1.
        if spts.ndim != 3:
            raise RuntimeError(f"spts[{et}].ndim={spts.ndim}, expected 3")

        npts, Ne2, nd = spts.shape
        if gids.max(initial=-1) >= Ne2:
            raise RuntimeError(
                f"spts[{et}] looks local but eidxs[{et}] contains global gids "
                f"(max gid={int(gids.max())} >= Ne={Ne2})."
            )

        c = spts[:, gids, :].mean(axis=0)  # (Ne_local, nd)

        if c.shape[1] < 3:
            c3 = np.zeros((c.shape[0], 3), dtype=np.float64)
            c3[:, :c.shape[1]] = c
            c = c3
        elif c.shape[1] > 3:
            c = c[:, :3]

        out[et] = c.astype(np.float64, copy=False)

    return out


# -----------------------------
# Bridge: partitioning -> State -> _MetaMesh
# -----------------------------
def metamesh_from_partitioning(
    *,
    mesh,                 # your _Mesh instance (already loaded for this rank)
    pname: str,           # partitioning name in file
    vparts: np.ndarray,   # global vparts array (length nelems_g)
    State,                # your State class
    MetaMesh,             # your _MetaMesh class
) -> "MetaMesh":
    """
    Build State+MetaMesh from an existing global vparts.

    Critical invariants:
      - eidxs[et] are per-etype gids (dense 0..Ne(etype)-1 globally)
      - con_idx[et] stores global element numbers (eid) or -1 on BC
      - con_mpi[et] stores neighbour owner ranks per face or -1
      - spts_nodes[et] indexed by local gids
    """

    W = comm["world"]
    rnk = int(rank["world"])
    P = int(W.size)

    # Canonical etype ordering: MUST match what you used to form vparts blocks.
    # If you already have mesh.etypes in canonical order, use that.
    etypes = list(getattr(mesh, "etypes", [])) or sorted(mesh.spts_nodes.keys())

    # Global element counts per etype (from mesh global arrays)
    # If your mesh already holds *local* spts_nodes, replace with mesh.ecnts_g.
    ecnts_g = {et: int(np.asarray(mesh.spts_nodes[et]).shape[0]) for et in etypes}

    # Build PyFR-style edisps from ecnts_g (global)
    disp = 0
    edisps = {}
    for et in etypes:
        edisps[et] = int(disp)
        disp += int(ecnts_g[et])

    nelems_g = int(disp)
    vparts = np.asarray(vparts, dtype=np.int64)
    if vparts.size != nelems_g:
        raise ValueError(f"vparts.size={vparts.size} but nelems_g={nelems_g} (edisps/ecnts mismatch)")

    # Local eidxs per etype: gids where vparts in this etype-block equals my rank
    eidxs = {}
    for et in etypes:
        off = edisps[et]
        cnt = ecnts_g[et]
        blk = vparts[off:off + cnt]
        gids = np.nonzero(blk == rnk)[0].astype(np.int64, copy=False)
        eidxs[et] = gids

    # ---- Connectivity extraction (your project-specific “truth”) ----
    # You MUST supply global con_idx per etype from whatever your reader already builds.
    # Common patterns in your codebase:
    #   - mesh.con_idx[et] already global eids (best case)
    #   - mesh.con_p[et] contains rows you can decode into (owner, eid)
    if hasattr(mesh, "con_idx") and isinstance(mesh.con_idx, dict):
        con_idx_g = mesh.con_idx
    else:
        raise RuntimeError(
            "Need global con_idx per etype (neighbour global element numbers). "
            "Expose it as mesh.con_idx[etype] in your mesh loader (same as diffuse path)."
        )

    con_idx = {}
    con_mpi = {}
    spts_nodes = {}

    for et in etypes:
        gids = eidxs[et]
        # slice local rows
        cidx = np.asarray(con_idx_g[et], dtype=np.int64)
        if cidx.ndim == 1:
            # allow (Ne*nfaces,) flattened; reshape to (Ne, -1)
            # only safe if Ne matches
            Ne = int(ecnts_g[et])
            cidx = cidx.reshape(Ne, -1)

        cidx_loc = cidx[gids, :] if gids.size else np.empty((0, cidx.shape[1]), dtype=np.int64)
        con_idx[et] = cidx_loc

        # owner ranks derived from vparts + con_idx
        owners = np.full(cidx_loc.shape, -1, dtype=np.int64)
        m = (cidx_loc >= 0)
        if np.any(m):
            owners[m] = vparts[cidx_loc[m]]
        con_mpi[et] = owners

        nds = np.asarray(mesh.spts_nodes[et])
        if nds.ndim == 1:
            nds = nds.reshape(ecnts_g[et], -1)
        spts_nodes[et] = nds[gids, :] if gids.size else np.empty((0, nds.shape[1]), dtype=nds.dtype)

    # Centroids (optional but strongly recommended for any seed/core growth)
    centroids = _centroids_from_mesh(mesh, etypes, eidxs)

    st_i = State(
        eidxs=eidxs,
        con_idx=con_idx,
        con_mpi=con_mpi,
        spts_nodes=spts_nodes,
        centroids=centroids,
    )
    st_j = st_i.clone()

    mm = MetaMesh(
        mesh_src=mesh,
        etypes=st_i.etypes,
        e2i=getattr(mesh, "e2i", {et: k for k, et in enumerate(st_i.etypes)}),
        bc2id=getattr(mesh, "bc2id", {}),
        i=st_i,
        j=st_j,
    )
    return mm


# -----------------------------
# Seed-core growth: fixed cores
# -----------------------------
def seed_cores_from_seed_eids(mmesh, seed_eids: np.ndarray) -> np.ndarray:
    """
    Build fixed cores (P,3) from a per-rank seed global-eid list.
    Each rank r uses centroid of its seed element if it currently owns it.
    """
    W = comm["world"]
    rnk = int(rank["world"])
    P = int(W.size)

    seed_eids = np.asarray(seed_eids, dtype=np.int64).reshape(-1)
    if seed_eids.size != P:
        raise ValueError(f"seed_eids must have length P={P}")

    seed_eid = int(seed_eids[rnk])

    # Find local flat index of seed_eid
    gids = np.asarray(mmesh.i.eidxs_flat, dtype=np.int64)
    if gids.size == 0:
        core = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    else:
        order = np.argsort(gids, kind="mergesort")
        gs = gids[order]
        pos = np.searchsorted(gs, seed_eid)
        if pos < gs.size and int(gs[pos]) == seed_eid:
            f = int(order[pos])

            # map flat -> (etype,lid) via slices; then index centroids
            core = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            for et, sl in mmesh.i.etype_slices.items():
                if sl.start <= f < sl.stop:
                    lid = f - sl.start
                    c = mmesh.i.centroids.get(et, None)
                    if c is not None and 0 <= lid < c.shape[0]:
                        core = np.asarray(c[lid, :], dtype=np.float64).copy()
                    break
        else:
            core = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    cores = np.asarray(W.allgather(core), dtype=np.float64)
    return cores


def add_inliers_step_fixedcores(
    mmesh,
    *,
    cores: np.ndarray,     # (P,3) fixed
    top: float = 0.02,
    mode: str = "vertices",
    margin: float = 0.0,
    move_spts_nodes: bool = True,
    verbose: bool = True,
) -> int:
    """
    Same idea as your add_inliers_step, but cores are FIXED (seed cores),
    so the dynamics become “graph-constrained Voronoi / region growth”.
    """
    W      = comm["world"]
    rnk    = int(rank["world"])
    root_w = int(root["world"])
    P      = int(W.size)

    st = mmesh.i
    mmesh._reset_j_with_i()

    top = float(top)
    if top <= 0.0:
        mmesh._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
        return 0

    cores = np.asarray(cores, dtype=np.float64)
    if cores.shape != (P, 3):
        mmesh._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
        return 0

    core_self = cores[rnk]
    if not np.all(np.isfinite(core_self)):
        mmesh._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
        return 0

    # Flat centroids + validity
    cflat = np.zeros((st.eidxs_flat.size, 3), dtype=np.float64)
    okc   = np.zeros((st.eidxs_flat.size,), dtype=bool)
    for et, sl in st.etype_slices.items():
        if sl.stop <= sl.start:
            continue
        c = st.centroids.get(et)
        if c is None or c.shape[0] != (sl.stop - sl.start):
            continue
        cflat[sl, :] = c
        okc[sl] = np.isfinite(c).all(axis=1)

    iface_mask, best_dest, best_dist = mmesh._best_dest_by_core_from_deltas(
        mode, cores=cores, cflat=cflat, okc=okc
    )

    mpi_flat = np.nonzero(iface_mask & (best_dest >= 0) & okc)[0].astype(np.int64, copy=False)
    n_mpi = int(mpi_flat.size)
    if n_mpi == 0:
        mmesh._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
        return 0

    c      = cflat[mpi_flat, :]
    d_self = np.linalg.norm(c - core_self[None, :], axis=1)
    d_dest = best_dist[mpi_flat]
    score  = d_self - d_dest  # improvement if moved to dest

    cand = score > float(margin)
    cand_flat = mpi_flat[cand]
    if cand_flat.size == 0:
        mmesh._apply_plan_and_commit({}, move_spts_nodes=move_spts_nodes)
        return 0

    nsel = int(np.ceil(top * int(cand_flat.size)))
    nsel = max(1, min(nsel, int(cand_flat.size)))

    gids  = st.eidxs_flat[cand_flat].astype(np.int64, copy=False)
    sc    = score[cand]
    order = np.lexsort((gids, -sc))
    chosen_flat = cand_flat[order[:nsel]]

    chosen_nbrs = best_dest[chosen_flat].astype(np.int64, copy=False)
    eidxs_diff  = mmesh._build_eidxs_diff_from_flat(chosen_flat, chosen_nbrs, st)

    moved_local = sum(len(g) for per in eidxs_diff.values() for g in per.values())
    mmesh._apply_plan_and_commit(eidxs_diff, move_spts_nodes=move_spts_nodes)

    moved_glob = int(W.allreduce(int(moved_local), op=mpi.SUM))
    if verbose and rnk == root_w:
        print(
            f"[seedgrow] mode={mode} top={top} margin={margin} "
            f"cand={int(cand_flat.size)} nsel={nsel} moved_glob={moved_glob}",
            flush=True,
        )

    return int(moved_local)
