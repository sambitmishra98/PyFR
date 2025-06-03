from pyfr.mpiutil import get_comm_rank_root
from pyfr.readers.native import _Mesh

from pyfr.relocator.submesh import SubMesh


import numpy as np
from pyfr.relocator.utils import crpprint

class MetaMesh:
    """
    One per MPI rank.  Owns:
      • the original local _Mesh                       (self.mesh)
      • one SubMesh *per rank* (self.smeshes[dest])    after exchanges
      • the global constant maps e_to_i / i_to_e / nfaces
    All element motion + communication happens at this layer.
    """

    e_to_i: dict[str, int] = {}
    i_to_e: dict[int, str] = {}
    etype_nfaces_map: dict[str, int] = {}

    def __init__(self, mesh: _Mesh):
        self.comm, self.rank, _ = get_comm_rank_root()

        self.mesh     = mesh
        
        self.etypes   = self._allgather_etypes(mesh)
        self.bc_map   = self._allgather_bc_map(mesh)

        self._init_global_maps()

        self.smeshes: dict[int, SubMesh] = {}
        for dest in range(self.comm.size):
            if dest == self.rank: self.smeshes[dest] = SubMesh.from_native_mesh(mesh, self.etypes, self.bc_map)
            else:                 self.smeshes[dest] = SubMesh.blank(self.etypes)

    # ──────────────────────────────────────────────────────────
    #  helpers – global maps
    # ──────────────────────────────────────────────────────────
    def _init_global_maps(self) -> None:
        """Create limited maps for the etypes *actually present* and push
        them onto SubMesh so every instance sees the same dictionaries."""
        # keep PyFR's canonical n-faces per type but drop unused ones

        # geometry dimension (last axis of any mesh.spts array)
        sample_etype = next(iter(self.mesh.spts))
        self.edim = getattr(self.mesh, 'spts')[sample_etype].shape[-1]
        SubMesh.edim = self.edim            # broadcast to all SubMesh instances

        self.etype_nfaces_map = {
            et: SubMesh.etype_nfaces_map[et] for et in self.etypes
        }

        self.e_to_i = {et: i for i, et in enumerate(self.etypes)}
        self.i_to_e = {i: et for et, i in self.e_to_i.items()}

        self.nv_map = {et: getattr(self.mesh, 'spts_nodes')[et].shape[1] for et in self.etypes}
        SubMesh.nv_map = self.nv_map

        SubMesh.etype_nfaces_map = self.etype_nfaces_map
        SubMesh.e_to_i           = self.e_to_i
        SubMesh.i_to_e           = self.i_to_e

    # ──────────────────────────────────────────────────────────
    #  POLICIES  – self-contained helpers you can swap/extend
    # ──────────────────────────────────────────────────────────
    def _rank_order_for_donors(self, N_i: np.ndarray,
                            nfaces: np.ndarray | None = None,
                            ninters: np.ndarray | None = None,
                            *, mode: str = "largest_surplus") -> list[int]:
        """
        Return a list of ranks **in the order we should consider as donors**.

        Parameters
        ----------
        N_i
            Current element counts per rank   (length R).
        nfaces, ninters
            Optional diagnostics (#neighbours, #interfaces) per rank.
        mode
            - "largest_surplus"  (default)  : sort by descending N_i
            - "largest_faces"              : descending nfaces  (tie -> N_i)
            - "largest_interfaces"         : descending ninters (tie -> N_i)
        """
        R = len(N_i)
        if nfaces is None:   nfaces   = np.zeros(R, int)
        if ninters is None:  ninters  = np.zeros(R, int)

        # build a 2-column key we can lexsort on
        if mode == "largest_surplus":
            key = np.stack([-N_i,           np.arange(R)], axis=0)
        elif mode == "largest_faces":
            key = np.stack([-nfaces,       -N_i],        axis=0)
        elif mode == "largest_interfaces":
            key = np.stack([-ninters,      -N_i],        axis=0)
        else:
            raise ValueError(f"unknown donor ordering mode '{mode}'")

        order = np.lexsort(key)          # idx of sorted ascending rows
        # we want donors first → reverse
        return list(order[::-1])


    def _split_surplus(self, donor: int, receivers: list[int],
                    surplus_d: int, deficits: np.ndarray,
                    *, mode: str = "ratio_deficit") -> dict[int, int]:
        """
        Decide **how many** elements donor gives each receiver.

        Returns
        -------
        give : {recv_rank: n_elements}
        """
        give = {r: 0 for r in receivers}
        if not receivers or surplus_d <= 0:
            return give

        if mode == "equal_share":
            q, r = divmod(surplus_d, len(receivers))
            for rr in receivers:
                give[rr] = q
            # distribute the remainder deterministically (lowest rank first)
            for rr in sorted(receivers)[:r]:
                give[rr] += 1

        elif mode == "ratio_deficit":
            # proportional to current deficits but never exceed them
            demand = deficits[receivers].astype(float)
            total  = demand.sum()
            if total == 0:
                return give

            share  = surplus_d * demand / total
            alloc  = np.floor(share).astype(int)
            remain = surplus_d - alloc.sum()
            # hand out remaining 1-by-1 to those with largest fractional part
            frac   = share - alloc
            for rr in np.argsort(frac)[::-1][:remain]:
                alloc[rr] += 1

            for rcv, n in zip(receivers, alloc):
                give[rcv] = min(n, deficits[rcv])  # safety

        else:
            raise ValueError(f"unknown split_surplus mode '{mode}'")

        # final sanity
        assert sum(give.values()) <= surplus_d
        return give




    # ──────────────────────────────────────────────────────────
    #  small all-gathers
    # ──────────────────────────────────────────────────────────
    def _allgather_etypes(self, mesh: _Mesh) -> list[str]:
        local = set(mesh.etypes)
        return sorted(set.union(*self.comm.allgather(local)))

    def _allgather_bc_map(self, mesh: _Mesh) -> dict[str, int]:
        local = set(mesh.bcon.keys())
        names = sorted(set.union(*self.comm.allgather(local)))
        return {name: i for i, name in enumerate(names)}

    # ──────────────────────────────────────────────────────────
    #  owner-column maintenance
    # ──────────────────────────────────────────────────────────
    def sync_owner_columns(self) -> None:
        """Broadcast the authoritative owner-rank for every (etype,gid) pair
        into *all* SubMesh connectivity tensors (col-0)."""
        owner = {(self.e_to_i[et], int(gid)): r for r, sm in self.smeshes.items() for et in self.etypes for gid in sm.eidxs[et]}
        for sm in self.smeshes.values():
            sm.patch_owner_columns(owner)

    # ──────────────────────────────────────────────────────────
    #  utility
    # ──────────────────────────────────────────────────────────
    def recreate_mesh(self, rank: int) -> _Mesh:
        return self.smeshes[rank].recreate_mesh(self.mesh, rank)

    def integrate_exchange(
        self,
        eidx_by_src: dict[int, dict[str, np.ndarray]],
        ary_by_src : dict[int, dict[str, dict[str, np.ndarray]]],
    ) -> None:
        """
        Merge incoming element IDs + any arrays listed in SubMesh.array_specs.
        After merge, wipe all remote slots so next round starts clean.
        """
        # Ensure local slot exists
        local = self.smeshes.setdefault(self.rank, SubMesh.blank(self.etypes))

        # ---------- merge -------------------------------------------------
        for src in range(self.comm.size):
            if src == self.rank:
                continue

            for et in self.etypes:
                inc_ids = eidx_by_src[src][et]
                if not inc_ids.size:
                    continue

                local.eidxs[et] = (np.concatenate([local.eidxs[et], inc_ids])
                                if local.eidxs[et].size else inc_ids.copy())

                for ary in SubMesh.array_specs:
                    inc_arr = ary_by_src[src][et][ary]
                    dst_arr = local.arrays[ary][et]
                    new_arr = (np.concatenate([dst_arr, inc_arr], axis=0)
                            if dst_arr.size else inc_arr.copy())
                    local.arrays[ary][et] = new_arr

        for dest in range(self.comm.size):
            if dest != self.rank:
                self.smeshes[dest] = SubMesh.blank(self.etypes)



    # ──────────────────────────────────────────────────────────
    #  warm-up: pair-wise redistribution (R(R-1)/2 carveouts)
    # ──────────────────────────────────────────────────────────
    def redistribute_pairwise(self, targets: list[int]) -> None:
        """
        Single global sweep of pair-wise transfers:

        • For every unordered pair (i<j) decide who donates to whom.
        • Move up to the donor's *current* surplus, never exceeding the
          receiver's *current* deficit.
        • Perform one carveout + SubMeshExchanger + integrate per pair.
        """
        comm, rank, _ = get_comm_rank_root()
        R = comm.size
        assert len(targets) == R, "targets list must match communicator"

        # ---------- helpers ---------------------------------------------
        def gather_Ni() -> np.ndarray:
            """Return element counts on every rank (after last integrate)."""
            return np.array(
                comm.allgather(self.smeshes[rank].N_i), dtype=int
            )

        # ---------- initial bookkeeping ---------------------------------
        N_i = gather_Ni()
        if rank == 0:
            print(f"[pairwise] initial element counts {N_i.tolist()}")

        # =============== main double loop ===============================
        from pyfr.relocator.submeshexchanger import SubMeshExchanger
        exchanger = SubMeshExchanger(comm)



        # just before the inner loops start
        if rank == 0:
            order = self._rank_order_for_donors(N_i)
            print(f"[order] donor priority this sweep: {order}")


        for i in range(R):
            for j in range(i + 1, R):
                # keep everything up-to-date
                N_i = gather_Ni()
                surplus_i = N_i[i] - targets[i]
                surplus_j = N_i[j] - targets[j]

                # decide donor / recv ------------------------------------
                if surplus_i <= 0 and surplus_j <= 0:
                    if rank == 0:
                        print(f"[pairwise] ({i},{j}) : both at / below target – skip")
                    continue

                if surplus_i >= surplus_j:
                    donor, recv = i, j
                    surplus_d, deficit_r = surplus_i, targets[j] - N_i[j]
                else:
                    donor, recv = j, i
                    surplus_d, deficit_r = surplus_j, targets[i] - N_i[i]

                if deficit_r <= 0:
                    if rank == 0:
                        print(f"[pairwise] ({donor}->{recv})  receiver full – skip")
                    continue

                ntarget = min(surplus_d, deficit_r)
                if rank == 0:
                    print(f"[pairwise] donor {donor} → recv {recv}   ntarget={ntarget}  "
                          f"(surplus={surplus_d}  deficit={deficit_r})")

                # ---------- carve patch on *donor* rank ------------------
                if rank == donor:
                    patch = self.smeshes[donor].carveout_for_nrank(
                        recv, ntarget=ntarget
                    )
                    # very verbose print
                    crpprint(
                        -1,
                        patch.eidxs,
                        f"[patch] donor={donor} → recv={recv}   eidxs before send",
                    )
                    # place patch in the slot for receiver so exchanger picks it up
                    self.smeshes[recv] = patch

                # make sure non-sending ranks have a blank slot
                if rank != donor:
                    self.smeshes[recv] = self.smeshes.get(recv, SubMesh.blank(self.etypes))

                # ---------- exchange + integrate -------------------------
                eidxs, arrays = exchanger.exchange(self)
                self.integrate_exchange(eidxs, arrays)


                # ---- NEW: update owner columns & restore ordering -------
                self.sync_owner_columns()                                     # <-- ADD
                for r in (donor, recv):                                        # <-- ADD
                    self.smeshes[r].restore_native_order(r)
                
                    # deficits array (positive where we need more)
                    deficits = targets - N_i
                    # all ranks except donor that still need elements
                    needers  = [r for r in range(R) if r != donor and deficits[r] > 0]
                    suggest  = self._split_surplus(donor, needers, surplus_d, deficits)
                    if rank == 0:
                        print(f"[split] donor {donor} proposed split: {suggest}")




                # ---------- post-exchange accounting --------------------
                N_i = gather_Ni()
                if rank == 0:
                    print(f"[pairwise]  after {donor}->{recv}  element counts {N_i.tolist()}")

        if rank == 0:
            print("[pairwise] *** completed one full R(R-1)/2 pass ***")

    def redistribute_sweeps(self, targets: list[int], *,
                            donor_mode: str = "largest_surplus",
                            split_mode: str = "ratio_deficit") -> None:
        """
        Perform **R-1** sweeps.

        sweep k = pick 1 donor (largest surplus by default);
                    split its *current* surplus across every rank
                    that is below target (policy-controlled);
                    build one carve-out per donor→receiver pair;
                    call a single SubMeshExchanger.exchange.

        No convergence checks, no early exit.
        """
        comm, rank, _ = get_comm_rank_root()
        R = comm.size
        assert len(targets) == R

        # helper to broadcast current element counts
        def gather_Ni() -> np.ndarray:
            return np.array(comm.allgather(self.smeshes[rank].N_i), int)

        from pyfr.relocator.submeshexchanger import SubMeshExchanger
        exchanger = SubMeshExchanger(comm)

        # -------------- main loop: exactly R-1 sweeps --------------------
        for sweep in range(1, R):
            N_i      = gather_Ni()
            deficits = targets - N_i
            surplus  = N_i - targets

            # ---- 1) choose the donor for *this* sweep -------------------
            donor_order = self._rank_order_for_donors(
                N_i, mode=donor_mode
            )
            donor = next(d for d in donor_order if surplus[d] > 0)
            surplus_d = int(surplus[donor])          # Python int for clarity

            if rank == 0:
                print(f"\n[sweep {sweep}/{R-1}] "
                    f"donor={donor}  surplus={surplus_d}  "
                    f"counts={N_i.tolist()}  targets={targets}")

            # ---- 2) decide how donor splits surplus ---------------------
            receivers = [r for r in range(R) if r != donor and deficits[r] > 0]
            give_plan = self._split_surplus(
                donor, receivers, surplus_d, deficits,
                mode=split_mode,
            )
            if rank == 0:
                print(f"[sweep {sweep}] donor→recv plan {give_plan}")

            # ---- 3) clear stale outgoing slots --------------------------
            for r in range(R):
                if r != self.rank:
                    self.smeshes[r] = SubMesh.blank(self.etypes)

            # ---- 4) donor builds patches --------------------------------
            if rank == donor:
                for recv, ntarget in give_plan.items():
                    if ntarget == 0:
                        continue
                    patch = self.smeshes[donor].carveout_for_nrank(
                        recv, ntarget=int(ntarget)
                    )
                    self.smeshes[recv] = patch      # exchanger will pick this up

            # ensure non-donor ranks have a blank slot for every recv
            if rank != donor:
                for recv in receivers:
                    if recv not in self.smeshes:
                        self.smeshes[recv] = SubMesh.blank(self.etypes)

            # ---- 5) single MPI exchange for this sweep ------------------
            eidxs, arrays = exchanger.exchange(self)
            self.integrate_exchange(eidxs, arrays)

            # ---- 6) owner-columns + ordering maintenance ----------------
            self.sync_owner_columns()
            for r in range(R):
                self.smeshes[r].restore_native_order(r)

            # ---- 7) diagnostics ----------------------------------------
            N_i = gather_Ni()
            if rank == 0:
                moved = sum(give_plan.values())
                print(f"[sweep {sweep}] moved={moved}  new counts {N_i.tolist()}")

        if rank == 0:
            print(f"[sweeps] completed {R-1} donor→receivers sweeps")



    def _rank_order_for_donors(self, N_i: np.ndarray, *,
                            mode: str = "largest_surplus",
                            nfaces=None, ninters=None) -> list[int]:
        R = len(N_i)
        if nfaces is None:   nfaces   = np.zeros(R, int)
        if ninters is None:  ninters  = np.zeros(R, int)

        if mode == "largest_surplus":
            key = np.stack([-N_i, np.arange(R)], axis=0)
        elif mode == "largest_faces":
            key = np.stack([-nfaces, -N_i], axis=0)
        elif mode == "largest_interfaces":
            key = np.stack([-ninters, -N_i], axis=0)
        else:
            raise ValueError(f"unknown donor ordering mode '{mode}'")

        # cast to plain int so the printout is clean
        return [int(i) for i in np.lexsort(key)[::-1]]

if __name__ == "__main__":
    from pyfr.readers.native import NativeReader
    from pyfr.relocator.utils import TablePrinter
    from pyfr.relocator.submeshexchanger import SubMeshExchanger

    comm, rank, _ = get_comm_rank_root()
    meshf = "/scratch/EFFORTS/LoadBalancer3/PyFR-Test-Cases/couette/squaremesh/square.pyfrm"
    mesh  = NativeReader(meshf, pname="3", construct_con=True).mesh

    mmesh  = MetaMesh(mesh)

    #tp = TablePrinter(mmesh) ; tp.print_global_totals(mmesh) ; tp.print_local_tables(mmesh)
    #crpprint(-1, mmesh.smeshes[mmesh.rank].con, "Local SubMesh before exchange")
    crpprint(-1, mmesh.smeshes[mmesh.rank].eidxs, "Local SubMesh before exchange")

    # simple “donor rank 2 → rank 1” test move
    #if mmesh.rank == 2:
    #    carved = mmesh.smeshes[2].carveout_for_nrank(1, ntarget=5)
    #    if any(len(v) for v in carved.eidxs.values()):     # replaces old N_i
    #        mmesh.smeshes[1] = carved
    #eidxs, con = SubMeshExchanger(comm).exchange(mmesh)
    #mmesh.integrate_exchange(eidxs, con)
    #mmesh.sync_owner_columns()
    #mmesh.smeshes[mmesh.rank].restore_native_order(rank)

    targets = [4, 4, 22]           # example from the chat
    mmesh.redistribute_sweeps(targets)

    # diagnostics
    #tp = TablePrinter(mmesh) ; tp.print_global_totals(mmesh) ; tp.print_local_tables(mmesh)
    crpprint(-1, mmesh.smeshes[mmesh.rank].eidxs, "Local SubMesh after exchange")
