from __future__ import annotations

# location: pyfr/relocator/metamesh.py

import time

import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.readers.native import _Mesh
from pyfr.relocator.submesh import SubMesh, MeshMetadata
from pyfr.relocator.utils import crpprint

from pyfr.relocator.plotting import plot_rank_graph, plot_global_graph


class MetaMesh:
    """
    One per MPI rank.  Owns:
      • the original local _Mesh                       (self.mesh)
      • one SubMesh *per rank* (self.smeshes[dest])    after exchanges
      • the global constant maps e_to_i / i_to_e / nfaces
    All element motion + communication happens at this layer.
    """

    def __init__(self, mesh: _Mesh):
        self.comm, self.rank, _ = get_comm_rank_root()

        self.mesh     = mesh
        
        self.etypes   = self._allgather_etypes(mesh)
        self.bc_map   = self._allgather_bc_map(mesh)

        self.meta = self._build_metadata()

        self.smeshes: dict[int, SubMesh] = {}
        for dest in range(self.comm.size):
            if dest == self.rank: self.smeshes[dest] = SubMesh.from_native_mesh(mesh, self.meta, self.bc_map)
            else:                 self.smeshes[dest] = SubMesh.blank(self.meta)

    # ──────────────────────────────────────────────────────────
    #  helpers – global maps
    # ──────────────────────────────────────────────────────────

    def _build_metadata(self) -> MeshMetadata:
        """Assemble an immutable MeshMeta and keep a reference to it."""
        # geometry dimension (last axis of any mesh.spts array)
        sample_etype = next(iter(self.mesh.spts))
        edim = next(iter(self.mesh.spts.values())).shape[-1]

        etype_nfaces = {
            et: SubMesh.etype_nfaces_map[et] for et in self.etypes
        }
        nv_map = {
            et: getattr(self.mesh, 'spts_nodes')[et].shape[1]
            for et in self.etypes
        }
        e_to_i = {et: i for i, et in enumerate(self.etypes)}
        i_to_e = {i: et for et, i in e_to_i.items()}

        # ONE shared immutable object
        return MeshMetadata(etypes=self.etypes, edim=edim, 
                            etype_nfaces_map=etype_nfaces, nv_map=nv_map, 
                            e_to_i=e_to_i, i_to_e=i_to_e,)

    def _allgather_etypes(self, mesh: _Mesh) -> list[str]:
        """
        Return the *sorted* global list of element types.

        Collective cost: one ``comm.gather`` + one ``comm.bcast``.
        Result is cached on the instance.
        """
        if hasattr(self, "_cached_etypes"):
            return self._cached_etypes          # reuse

        local_set = set(mesh.etypes)
        all_sets = self.comm.gather(local_set, root=0)

        if self.rank == 0:
            union = sorted(set.union(*all_sets))
        else:
            union = None
        union = self.comm.bcast(union, root=0)

        self._cached_etypes = union
        return union


    # ──────────────────────────────────────────────────────────
    def _allgather_bc_map(self, mesh: _Mesh) -> dict[str, int]:
        """
        Return a stable **name → small-int** map for all boundary-condition
        tags present in the global mesh.

        Collective cost: one gather + broadcast.
        """
        if hasattr(self, "_cached_bc_map"):
            return self._cached_bc_map

        local_set = set(mesh.bcon.keys())
        all_sets = self.comm.gather(local_set, root=0)

        if self.rank == 0:
            names = sorted(set.union(*all_sets))
            bc_map = {name: i for i, name in enumerate(names)}
        else:
            bc_map = None
        bc_map = self.comm.bcast(bc_map, root=0)

        self._cached_bc_map = bc_map
        return bc_map


    # ──────────────────────────────────────────────────────────
    #  fast owner-rank rebroadcast
    # ──────────────────────────────────────────────────────────
    def sync_owner_columns(self) -> None:
        """
        For every (etype, gid) that *currently* lives on this rank, broadcast
        the owning rank index into **column-0** of every connectivity tensor
        held in ``self.smeshes`` (all slots, all ranks).

        Notes
        -----
        * Works in-place; no data is returned.
        * Relies on the guarantee that global element-IDs are unique.
        """
        e2i = self.meta.e_to_i

        # -------- build { (code,gid) : owner_rank } --------------------------
        owner_map: dict[tuple[int, int], int] = {}
        for r, sm in self.smeshes.items():
            for et, gids in sm.eidxs.items():
                if gids.size == 0:
                    continue
                code = e2i[et]
                # one vectorised update instead of Python loop per gid
                owner_map.update({(code, int(g)): r for g in gids})

        # -------- patch every slot in-place ----------------------------------
        for sm in self.smeshes.values():
            sm.patch_owner_columns(owner_map)


    # ──────────────────────────────────────────────────────────
    #  utility
    # ──────────────────────────────────────────────────────────
    def recreate_mesh(self, rank: int) -> _Mesh:
        return self.smeshes[rank].recreate_mesh(self.mesh, rank)

    # ──────────────────────────────────────────────────────────
    #  merge received patches; wipe remote slots afterwards
    # ──────────────────────────────────────────────────────────
    def integrate_exchange(
        self,
        eidx_by_src: dict[int, dict[str, np.ndarray]],
        ary_by_src : dict[int, dict[str, dict[str, np.ndarray]]],
    ) -> None:
        """
        Merge incoming element IDs *and* every array listed in
        ``SubMesh.array_specs`` into this rank’s local slot.

        After the merge, all non-local slots are reset to blank so the next
        redistribution round starts with a clean slate.
        """
        local = self.smeshes.setdefault(self.rank, SubMesh.blank(self.meta))

        # ---------- 1) merge element IDs & arrays ----------------------------
        for src, eidict in eidx_by_src.items():
            if src == self.rank:
                continue  # nothing to merge from ourselves

            for et, inc_ids in eidict.items():
                if inc_ids.size == 0:
                    continue

                # ---- element IDs
                local.eidxs[et] = np.concatenate([local.eidxs[et], inc_ids])

                # ---- all registered arrays share axis-0 = element axis
                for ary in SubMesh.array_specs:
                    local.arrays[ary][et] = np.concatenate(
                        [local.arrays[ary][et], ary_by_src[src][et][ary]],
                        axis=0,
                    )

        # ---------- 2) blank every remote slot ------------------------------
        for dest in range(self.comm.size):
            if dest != self.rank:
                self.smeshes[dest] = SubMesh.blank(self.meta)

    @staticmethod
    def build_global_plan(N_i: np.ndarray, targets: np.ndarray) -> dict[int, dict[int, int]]:
        """Return {donor: {recv: n_elem, …}, …} moving all surplus in one sweep."""
        surplus  = N_i - targets                       # positive = donors
        donors   = np.where(surplus > 0)[0]
        recvs    = np.where(surplus < 0)[0]
        plan     = {d: {} for d in donors}

        deficits = {r: -surplus[r] for r in recvs}     # positive numbers
        donors   = sorted(donors, key=lambda d: surplus[d], reverse=True)

        for r in sorted(recvs, key=deficits.get, reverse=True):
            need = deficits[r]
            for d in donors:
                give = min(need, surplus[d])
                if give:
                    plan[d][r] = give
                    surplus[d] -= give
                    need       -= give
                if need == 0:
                    break
            assert need == 0, "should be enough surplus to cover all deficits"
        return plan

    def redistribute(self, targets: list[int]) -> None:

        # perfcounter()
        tstart = time.perf_counter()

        crpprint(-1, self.smeshes[self.rank].N_i, "Local SubMesh before exchange")

        comm, rank, _ = get_comm_rank_root()
        R = comm.size
        assert len(targets) == R

        def gather_Ni():
            return np.array(comm.allgather(self.smeshes[rank].N_i), int)

        from pyfr.relocator.submeshexchanger import SubMeshExchanger
        exchanger = SubMeshExchanger(comm)

        # ─── 1. current counts & global plan ──────────────────────────────
        N_i   = gather_Ni()

        plan  = self.build_global_plan(N_i, np.asarray(targets))
        if rank == 0:
            print("[global-plan]", plan)

        # ─── 2. clear all non-local slots once ────────────────────────────
        for dest in range(R):
            if dest != self.rank:
                self.smeshes[dest] = SubMesh.blank(self.meta)

        # ─── 3. donors build their patches ────────────────────────────────
        for donor, sends in plan.items():
            if rank == donor:
                for recv, want in sends.items():
                    patch, moved = self._safe_carve(donor, recv, want)
                    assert moved == want, \
                        f"donor {donor}→{recv}: asked {want}, carved {moved}"
                    self.smeshes[recv] = patch  # one slot per recv

        # ─── 4. single all-to-all exchange ────────────────────────────────
        eidxs, arrays = exchanger.exchange(self)
        self.integrate_exchange(eidxs, arrays)

        # ─── 5. owner columns + ordering ──────────────────────────────────
        self.sync_owner_columns()
        for r in range(R):
            self.smeshes[r].restore_native_order(r)

        # ─── 6. final diagnostics ─────────────────────────────────────────
        N_i = gather_Ni()
        if rank == 0:
            print("[redistribute] new counts", N_i.tolist())

        crpprint(-1, self.smeshes[self.rank].N_i, "Local SubMesh after exchange")

        print(f"[redistribute] took {time.perf_counter() - tstart:.4f} seconds")

    def _safe_carve(self, donor: int, recv: int, want: int) -> tuple["SubMesh", int]:
        """Carve exactly *want* elements; raise if impossible."""
        patch = self.smeshes[donor].carveout_for_nrank(recv, ntarget=want)
        moved = sum(len(v) for v in patch.eidxs.values())
        if moved != want:
            raise ValueError(f"donor {donor}→{recv}: asked {want}, got {moved}")
        return patch, moved

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
    mmesh.redistribute(targets)

    # diagnostics
    tp = TablePrinter(mmesh) ; tp.print_global_totals(mmesh) ; tp.print_local_tables(mmesh)


    crpprint(-1, mmesh.smeshes[mmesh.rank].eidxs, "Local SubMesh after exchange")
