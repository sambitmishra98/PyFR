"""Deterministic ownership policies for online Hex AMR.

D7D introduces ownership *decision* logic only.  The accepted D7 MPI
transaction remains solely responsible for solution migration, native mesh
partitioning/rebuild, validation, COMMIT, and rollback.
"""
from collections import defaultdict

import numpy as np


class AMROwnershipPolicyError(ValueError):
    """An AMR ownership policy cannot produce a supported partition."""


def _balanced_targets(nitems, nranks):
    if nranks <= 0:
        raise AMROwnershipPolicyError('number of ranks must be positive')
    if nitems < nranks:
        raise AMROwnershipPolicyError(
            'AMR policy requires at least one leaf per MPI rank'
        )

    q, r = divmod(int(nitems), int(nranks))
    return np.array([q + (i < r) for i in range(nranks)], dtype=np.int64)


def _leaf_affinity(leaf, old_leaves, old_parts, nranks):
    root, path = leaf
    affinity = np.zeros(nranks, dtype=np.int64)

    # Unchanged leaf or a newly refined descendant: the old active ancestor is
    # the state record from which this proposed leaf is produced.
    ancestors = [
        old for old in old_leaves
        if old[0] == root and path[:len(old[1])] == old[1]
    ]
    if ancestors:
        old = max(ancestors, key=lambda x: len(x[1]))
        affinity[old_parts[old]] = 1
        return affinity

    # Coarsening: all retiring active descendants contribute to the proposed
    # parent, so count their current owners as migration affinity.
    descendants = [
        old for old in old_leaves
        if old[0] == root and old[1][:len(path)] == path
    ]
    if not descendants:
        raise AMROwnershipPolicyError(
            f'proposed leaf {leaf!r} has no relation to the current tree'
        )
    for old in descendants:
        affinity[old_parts[old]] += 1

    return affinity


def _mortar_units(raw):
    nleaves = len(raw.leaf_order)
    parent = np.arange(nleaves, dtype=np.int64)

    def find(i):
        i = int(i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri == rj:
            return
        if ri > rj:
            ri, rj = rj, ri
        parent[rj] = ri

    for mortar in raw.mortars:
        members = (int(mortar.left_eidx),
                   *(int(i) for i in mortar.right_eidx))
        for i in members[1:]:
            union(members[0], i)

    groups = defaultdict(list)
    for i in range(nleaves):
        groups[find(i)].append(i)

    return tuple(
        tuple(v) for _, v in sorted(groups.items(), key=lambda kv: min(kv[1]))
    )


def _unit_adjacency(raw, units):
    owner = {}
    for uidx, members in enumerate(units):
        for i in members:
            owner[int(i)] = uidx

    adj = [defaultdict(int) for _ in units]
    offs = np.asarray(raw.hex_faces_off)
    for i in range(len(raw.leaf_order)):
        ui = owner[i]
        for j in offs[i]:
            j = int(j)
            if j < 0:
                continue
            uj = owner[j]
            if ui != uj:
                adj[ui][uj] += 1

    return tuple(dict(a) for a in adj)


def balanced_affinity_destination_parts(old_tree, proposed_tree, raw,
                                        old_parts, nranks):
    """Choose a deterministic balanced ownership map for a proposed AMR tree.

    The policy is intentionally small and homogeneous-rank specific.  It uses
    two principles from the proven online load-balancing lineage: explicit
    target element counts and ownership/interface affinity.  H-mortar
    participants are contracted into indivisible units before assignment.
    Migration/rebuild remains the responsibility of :mod:`pyfr.amrmpi`.
    """
    if old_tree.root_mesh_uuid != proposed_tree.root_mesh_uuid:
        raise AMROwnershipPolicyError(
            'old and proposed trees do not share a root mesh UUID'
        )
    old_leaves = tuple(old_tree.leaves())
    new_leaves = tuple(proposed_tree.leaves())
    if tuple(raw.leaf_order) != new_leaves:
        raise AMROwnershipPolicyError(
            'materialized leaf order does not match proposed tree'
        )
    if set(old_parts) != set(old_leaves):
        raise AMROwnershipPolicyError(
            'current ownership must cover the old tree exactly'
        )
    if any(not isinstance(r, (int, np.integer)) or not 0 <= int(r) < nranks
           for r in old_parts.values()):
        raise AMROwnershipPolicyError(
            'current ownership contains invalid rank'
        )

    targets = _balanced_targets(len(new_leaves), nranks)
    units = _mortar_units(raw)
    if len(units) < nranks:
        raise AMROwnershipPolicyError(
            'h-mortar affinity leaves fewer ownership units than MPI ranks'
        )
    adjacency = _unit_adjacency(raw, units)

    leaf_aff = [
        _leaf_affinity(leaf, old_leaves, old_parts, nranks)
        for leaf in new_leaves
    ]
    unit_aff = [sum((leaf_aff[i] for i in u),
                    np.zeros(nranks, dtype=np.int64)) for u in units]

    # Large constrained units first; canonical minimum leaf ordinal is the
    # deterministic tie-break.  This is a best-fit assignment, not a graph
    # partitioner, and deliberately avoids importing legacy migration code.
    order = sorted(
        range(len(units)), key=lambda u: (-len(units[u]), units[u][0])
    )
    counts = np.zeros(nranks, dtype=np.int64)
    assignment = {}

    for uidx in order:
        members = units[uidx]
        weight = len(members)
        affinity = unit_aff[uidx]
        atotal = int(affinity.sum())

        best = None
        for rank in range(nranks):
            projected = int(counts[rank]) + weight
            over = max(0, projected - int(targets[rank]))
            deficit = max(0, int(targets[rank]) - int(counts[rank]))
            residual = abs(projected - int(targets[rank]))
            migration = atotal - int(affinity[rank])
            cut = sum(
                nfaces for nbr, nfaces in adjacency[uidx].items()
                if nbr in assignment and assignment[nbr] != rank
            )
            score = (over, -deficit, migration, cut, residual, rank)
            if best is None or score < best[0]:
                best = (score, rank)

        rank = best[1]
        assignment[uidx] = rank
        counts[rank] += weight

    if np.any(counts == 0):
        raise AMROwnershipPolicyError(
            'balanced-affinity-v1 produced an empty MPI rank'
        )

    parts = {}
    for uidx, members in enumerate(units):
        rank = assignment[uidx]
        for i in members:
            parts[new_leaves[i]] = rank

    return parts
