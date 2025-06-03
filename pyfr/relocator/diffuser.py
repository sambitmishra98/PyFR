from __future__ import annotations

from copy import deepcopy

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.readers.native import _Mesh
from pyfr.relocator.metamesh import MetaMesh

# Fix seed
BASE_TAG = 47
np.random.seed(BASE_TAG)

class LoadRelocator():
    # Initialise as an empty list
    etypes = []

    def __init__(self, base_mesh: _Mesh, *,
                 bmmesh = 'base', cmmesh = 'compute', cnmmesh = 'compute_new'):

        self.mm = MetaMesh()
        self.mm.add_mmesh(bmmesh, MetaMesh.from_mesh(base_mesh), if_base=True)

        # Print mmesh to check if it is added
        print(MetaMesh.from_mesh(base_mesh).nelems, flush=True)

        self.mm.copy_mmesh(bmmesh, cmmesh)
        self.mm.copy_mmesh(cmmesh, cnmmesh)

        self.new_ranks = list(range(mpi.COMM_WORLD.size))

    def firstguess_target_nelems(self, weights: list[float]):
        '''
            Calculate the target elements per rank per rank weights.
            
            Args:
                mesh_nelems (list[int]): Elements per rank in mesh 
                weights (list[float]): Weights per rank.

            Returns:
                list[int]: Target elements per rank.
                
            Scenarios considered:
                1. Weights sum to 1.
                2. Target elements are integers that sum to total.

            Rule of thumb:
                1. Move any extra elements into rank with highest elements.
                - Assumption: Lowest wait-time rank with highest number of 
                              inter-rank interfaces load-balances the best.
        '''

        weights = np.array(weights, dtype=float)
        weights = weights / np.sum(weights, dtype=float)

        t_nelems = self.mm.gnelems*weights

        # Move deficit into rank with highest elements
        t_nelems[np.argmax(t_nelems)] += self.mm.gnelems - sum(t_nelems)

        # Convert to list of integers
        return t_nelems.tolist()

    def diffuse_computation(self, mesh_name, target_nelems, cli = False,
                            move_priority = None):
        '''
            Iteratively diffuse elements until reaching target within each rank.
            This function is specific to 'compute' meta-mesh.
        '''

        self.cli = cli

        if move_priority == None:
            self.move_priority = sorted(range(len(target_nelems)), reverse=True)
        else:
            self.move_priority = move_priority

        # Get MPI comm world for 'compute' mesh
        comm, rank, root = get_comm_rank_root()

        # Create a copy of the meta mesh
        self.mm.move_mmesh(mesh_name+'_new', mesh_name)
        self.mm.copy_mmesh(mesh_name, mesh_name+'_new')

        init_nelems     = self.mm.get_mmesh(mesh_name).nelems
        previous_nelems = np.ones(comm.size, dtype=int)
        curr_nelems  = np.array(comm.allgather(self.mm.get_mmesh(mesh_name).nelems))
        
        # Create a matrix that lets us decide whether or not to move elements across interface between i and j ranks
        self.if_move_along_interface = np.ones((comm.size, comm.size))

        # Create a movement-multiplier for elements movement

        # If ccc is not equal to previous_nelems, and if even one of the targets is 0 and we haven't reached that yet
        for i in range(5):
            if rank == root:
                print(f"Current nelems: {np.round(curr_nelems).astype(int)} "
                      f"\n New movements: {np.round(target_nelems - curr_nelems).astype(int)}"
                      f"\n ABS SUM new movements: {np.sum(np.abs(target_nelems - curr_nelems))}",
                      flush=True)
            if np.array_equal(curr_nelems, previous_nelems):
                break

            if np.any(target_nelems == 0) and np.any(curr_nelems == 0) and np.where(target_nelems == 0)[0][0] == np.where(curr_nelems == 0)[0][0]:
                break

            if np.any(target_nelems == 0) and np.any(curr_nelems != 0):
                target_nelems = self.freeze_ranks_and_update_targets(previous_nelems, curr_nelems, target_nelems)

            previous_nelems = curr_nelems.copy()

            nelems_diff = target_nelems - np.array(comm.allgather(self.mm.get_mmesh(mesh_name+'_new').nelems))

            # If any rank moves < 1% initial t_nelems, set rows/columns in self.if_move_along_interface to 0
            #for i, diff in enumerate(nelems_diff):
            #    #if abs(diff) < self.imbalance_allowance * init_nelems:
            #    if abs(diff) < 0.001 * init_nelems:
            #        self.if_move_along_interface[i, :] = 0
            #        self.if_move_along_interface[:, i] = 0

            move_to_nrank         = self.get_move_to_nrank(nelems_diff[rank], self.mm.get_mmesh(mesh_name+'_new'))
            self.movable_to_nrank = self.get_movable_to_nrank(self.mm.get_mmesh(mesh_name+'_new'))

            # Force-stop movement across some interfaces
            for etype in self.movable_to_nrank:
                self.movable_to_nrank[etype] = np.multiply(self.movable_to_nrank[etype], self.if_move_along_interface)

            # Move elements
            move_elems = self.reloc_interface_elems(move_to_nrank, self.mm.get_mmesh(mesh_name+'_new'))

            preordered_eidxs = self.get_preordered_eidxs(move_elems, self.mm.get_mmesh(mesh_name+'_new'))

            # PARTIAL MOVEMENT HERE
            self.mm.copy_partial_mmesh(mesh_name+'_new', mesh_name+'-temp', preordered_eidxs)

            self.mm.move_partial_mmesh(mesh_name+'-temp', mesh_name+'_new')

            curr_nelems = np.array(comm.allgather(self.mm.get_mmesh(mesh_name+'_new').nelems))

        # Complete movement here
        self.mm.remove_mmesh(mesh_name+'_new')
        self.mm.copy_mmesh(mesh_name, mesh_name+'_new', preordered_eidxs)
        #self.mm.move_mmesh(mesh_name, mesh_name+'_new')
        
        #self.mm.connect_mmeshes(mesh_name+'_new', mesh_name)

        if not cli:
            # CLI only needs eidxs, if-curved and if-mpi.
            self.mm.get_mmesh(mesh_name+'_new').spts        = self.reloc(mesh_name, mesh_name+'_new', self.mm.get_mmesh(mesh_name).spts,        edim=0)

        # Re-create ary
        new_mesh = self.mm.get_mmesh(mesh_name+ '_new').to_mesh()
        
        print(self.mm, flush=True)

        self.new_ranks = [r for r in range(comm.size) if target_nelems[r] > 0]

        if 1 in curr_nelems:
            raise ValueError("Rank with 1 element!")
        
        return new_mesh

    def freeze_ranks_and_update_targets(self, gprev, gcurr, gtarget):
        """
        For each rank, if the current element count did not change from the previous
        count or if its target is 0, then that rank is frozen and its new target is fixed
        (set to gcurr if frozen by no change, or to 0 if target==0). For non-frozen ranks,
        the provisional target remains gtarget. Then, any difference between the total
        current elements (sum(gcurr)) and the provisional sum is redistributed proportionally
        among the non-frozen (nonzero-target) ranks so that the total new targets equal sum(gcurr).
        
        Parameters:
            gprev : array_like
                The previous element counts per rank.
            gcurr : array_like
                The current element counts per rank.
            gtarget : array_like
                The original target element counts per rank.
        
        Returns:
            new_gtarget : ndarray
                The updated targets per rank whose sum equals sum(gcurr).
        """

        # Ensure all inputs are NumPy arrays
        gprev   = np.asarray(gprev)
        gcurr   = np.asarray(gcurr)
        gtarget = np.asarray(gtarget)

        # Create masks:
        freeze_target = (gtarget == 0)            # Freeze any rank whose target is zero.

        # Also freeze any rank that did not change and has more than 2 elements initially.
        freeze_no_change = (gcurr == gprev)

        frozen = freeze_target | freeze_no_change

        # For frozen ranks, we want:
        #   - If forced to zero (target==0), then new target becomes 0.
        #   - Otherwise (no change), set new target equal to the current count.
        # For non-frozen ranks, start with the given target.
        new_gtarget = np.where(freeze_target, 0,
                            np.where(freeze_no_change, gcurr, gtarget))

        # If any rank is currently at 1 element, then add 100 to it and subtract 100 from rank 0
        if np.any(gcurr == 1):
            new_gtarget[0] -= 2
            new_gtarget[np.where(gcurr == 1)[0][0]] += 2

        # Compute the total difference between the current total and the provisional total.
        total_new = np.sum(new_gtarget)
        total_curr = np.sum(gcurr)
        diff = total_curr - total_new

        # Identify non-frozen ranks (those allowed to adjust) and compute the sum of their targets.
        non_frozen_mask = ~frozen
        denom = np.sum(gtarget[non_frozen_mask])
        
        # Only adjust if there is something to distribute.
        if denom > 0:
            # Distribute 'diff' proportionally to the original gtarget values among non-frozen ranks.
            adjustment = diff * (gtarget[non_frozen_mask] / denom)
            new_gtarget[non_frozen_mask] += adjustment

        return new_gtarget

    def reloc(self, src_mesh_name : str, dest_mesh_name: str,
                 edict: dict[str, np.ndarray],*, edim) -> dict[str, np.ndarray]:

        dest_mmesh = self.mm.get_mmesh(dest_mesh_name)

        edict = dest_mmesh.preproc_edict(self.mm.etypes, edict, edim=edim)
        relocated_dict = dest_mmesh.interconnector[src_mesh_name].relocate(edict)
        return dest_mmesh.postproc_edict(relocated_dict, edim=edim)

    def get_move_to_nrank(self, nelems_diff: int, mesh: MetaMesh):
        '''
            Save element movements as a matrix, 
            Elements move from row-rank to column-rank.
        '''

        comm, rank, root = get_comm_rank_root()

        move_to_nrank = np.zeros((comm.size, comm.size))
        for nrank, inters in mesh.ncon_p_nrank_etype.items():
            if nelems_diff != 0:
                if mesh.ncon_p > 0:
                    move_to_nrank[rank, nrank] = -nelems_diff * mesh.ncon_p_nrank[nrank] / mesh.ncon_p
                else:
                    move_to_nrank[rank, nrank] = 0
        move_to_nrank = comm.allreduce(move_to_nrank, op=mpi.SUM)
        move_to_nrank = np.maximum((move_to_nrank - move_to_nrank.T) / 2, 0)

        return move_to_nrank

    def get_movable_to_nrank(self, mesh):
        comm, rank, root = get_comm_rank_root()
        movable_to_nrank = {etype: np.zeros((comm.size, comm.size)) for etype in self.mm.etypes}
        for nrank in mesh.con_p.keys():
            for etype in self.mm.etypes:
                n_available = mesh.ncon_p_nrank_etype[nrank][etype]
                movable_to_nrank[etype][rank, nrank] = n_available
                movable_to_nrank[etype][nrank, rank] = n_available
        return movable_to_nrank

    def reloc_interface_elems(self, move_to_nrank: np.ndarray, mesh: _MetaMesh):
    
        comm, rank, root = get_comm_rank_root()
    
        # Initialise dictionary for the elements to be moved:
        move_elems = {
            int(nrank): {etype: np.empty(0, dtype=np.int32) for etype in self.mm.etypes}
            for nrank in mesh.con_p.keys()
        }
    
        for nrank in mesh.con_p.keys():
            # Compute the total available interfaces (Y_total) across all etypes for this neighbor:
            total_available = sum(self.movable_to_nrank[etype][rank, nrank]
                                  for etype in self.mm.etypes)
            # X: total number of elements to move from rank to nrank.
            X = move_to_nrank[rank][nrank]
            for etype in self.mm.etypes:
                # Available interface elements for this etype (Y_i)
                available = self.movable_to_nrank[etype][rank, nrank]
                if total_available > 0:
                    # Distribute the total moves X proportionally:
                    n_to_move = int(np.ceil(X * available / total_available))
                else:
                    n_to_move = 0
    
                # Ensure we do not try to move more than available:
                n_to_move = min(n_to_move, available)
                
                if n_to_move == 0:
                    move_elemsf = np.array([], dtype=np.int32)
                else:
                    # If we need to move as many as (or more than) available, take them all;
                    # otherwise, choose n_to_move from the available interface elements.
                    if n_to_move >= available:
                        move_elemsf = mesh.gcon_p[nrank][etype]
                    else:
                        move_elemsf = np.random.choice(mesh.gcon_p[nrank][etype],
                                                       size=n_to_move,
                                                       replace=False)
                    move_elemsf = np.array(move_elemsf, dtype=np.int32)
                    move_elemsf = np.sort(move_elemsf)
                    move_elemsf = np.unique(move_elemsf)
                move_elems[nrank][etype] = move_elemsf
    
        # Make sure all ranks have an entry for each rank
        move_elems_f = {nrank: move_elems[nrank] for nrank in sorted(move_elems.keys())}
        for nrank in range(comm.size):
            if nrank not in move_elems_f:
                move_elems_f[nrank] = {etype: np.empty(0, dtype=np.int32)
                                       for etype in self.mm.etypes}
        # Remove duplicates if necessary:
        rem_dups = self.remove_duplicate_movements(move_elems_f, mesh)
        for nrank in range(comm.size):
            if nrank not in rem_dups:
                rem_dups[nrank] = {etype: np.empty(0, dtype=np.int32)
                                   for etype in self.mm.etypes}
        
        # --- New logic: cancel movements that would reduce count below 2 ---
        # For each element type on this rank, compute the total number to be moved out.
        # If the current number minus the total to be moved is less than 2,
        # then delete (i.e. set to empty) all movements for that element type.
        for etype in self.mm.etypes:
            current_count = len(mesh.eidxs[etype])
            total_to_move = sum(len(rem_dups[nrank][etype]) for nrank in rem_dups)
            if current_count - total_to_move < 2:
                # Cancel all movements for this element type.
                for nrank in rem_dups:
                    rem_dups[nrank][etype] = np.empty(0, dtype=np.int32)
        # -------------------------------------------------------------------

        return rem_dups

    def reloc_interface_elems2(self, move_to_nrank: np.ndarray, mesh: _MetaMesh):

        comm, rank, root = get_comm_rank_root()

        # Initialise dictionary for the elements to be moved:
        move_elems = {
            int(nrank): {etype: np.empty(0, dtype=np.int32) for etype in self.mm.etypes} 
            for nrank in mesh.con_p.keys()
            }

        for nrank in mesh.con_p.keys():

            # Compute the total available interfaces (Y_total) across all etypes for this neighbor:
            total_available = sum(self.movable_to_nrank[etype][rank, nrank]
                              for etype in self.mm.etypes)

            # X: total number of elements to move from rank to nrank
            X = move_to_nrank[rank][nrank]
            for etype in self.mm.etypes:
                # Available interface elements for this etype (Y_i)
                available = self.movable_to_nrank[etype][rank, nrank]
                
                if total_available > 0:
                    # Distribute the total moves X proportionally:
                    n_to_move = int(np.ceil(X * available / total_available))
                else:
                    n_to_move = 0                

                # Ensure we do not try to move more than available:
                n_to_move = min(n_to_move, available)
                
                if n_to_move == 0:
                    # No rank interfaces, element movement impossible.
                    move_elemsf = np.array([], dtype=np.int32)
                else:
                    # Number of elements to move for this etype with nrank
                    n_to_move = np.ceil(move_to_nrank[rank][nrank]).astype(int)
                    # n_to_move = np.round(move_to_nrank[rank][nrank] * n_available / mesh.ncon_p).astype(int)

                    # Enforce Nₑ-to-move ≤ Nₑ-available-to-move
                    n_to_move = min(n_to_move, available)

                    if n_to_move in [1, 2]:
                        move_elemsf = mesh.gcon_p[nrank][etype]

                    elif n_to_move == available:
                        # If elements to move > interface elements with nrank
                        # Move all elements 
                        move_elemsf = mesh.gcon_p[nrank][etype]
                    else:
                        # Select any n_to_move elements from interface elements
                        move_elemsf = np.random.choice(mesh.gcon_p[nrank][etype], 
                                                       size=n_to_move,
                                                       replace=False)

                    move_elemsf = [intelem for intelem, move in zip(mesh.gcon_p[nrank][etype], move_elemsf) if move]

                    move_elemsf = np.array(move_elemsf, dtype=np.int32)
                    move_elemsf = np.sort(move_elemsf)
                    move_elemsf = np.unique(move_elemsf)

                move_elems[nrank][etype] = move_elemsf
        
        # Empty if self
        move_elems[rank] = {etype: np.empty(0, dtype=np.int32) for etype in self.mm.etypes}
        move_elems_f = {nrank: move_elems[nrank] for nrank in sorted(move_elems.keys())}

        # Add empty arrays if we are not moving anything to nranks
        for nrank in range(comm.size):
            if nrank not in move_elems_f:
                move_elems_f[nrank] = {etype: np.empty(0, dtype=np.int32) for etype in self.mm.etypes}

        rem_dups = self.remove_duplicate_movements(move_elems_f, mesh)
        
        # Add empty arrays if we are not moving anything to nranks
        for nrank in range(comm.size):
            if nrank not in rem_dups:
                rem_dups[nrank] = {etype: np.empty(0, dtype=np.int32) for etype in self.mm.etypes}
        
        return rem_dups

    def remove_duplicate_movements(self, 
                                   move_elems: dict[int, dict[str, np.ndarray]],
                                   mesh: _Mesh):
        """
        Remove duplicate movements from the move_to_nrank dictionary.
        """

        move_initial = {
            nrank: {
                etype: 
                    len(elems) for etype, elems in etypeelems.items()
                    } 
                        for nrank, etypeelems in move_elems.items()
                        }

        # Step 8: Remove duplicates by iterating over sorted ranks
        for i, nrank in enumerate(self.move_priority):
            for etype in self.mm.etypes:
                for nnrank in self.move_priority[i+1:]:

                    if etype not in move_elems[nnrank]:
                        move_elems[nnrank][etype] = np.empty(0, dtype=np.int32)
                    else:
                        move_elems[nnrank][etype] = np.setdiff1d(
                            move_elems[nnrank][etype], 
                            move_elems[ nrank][etype]
                    )

        move_final = {
                      nrank: {
                              etype: len(elems) 
                              for etype, elems in etypeelems.items()
                             } for nrank, etypeelems in move_elems.items()
                     }

        return move_elems
    
    def get_preordered_eidxs(self, 
                             move_to_nrank: dict[int, dict[str, np.ndarray]],
                             mesh: _MetaMesh) -> dict[str, np.ndarray[int]]:
        """
        Get new element indices on each rank after relocation.
        """

        comm, rank, root = get_comm_rank_root()

        moved_from_nrank0 = MeshInterConnector._send_recv0(self.mm.etypes, move_to_nrank)

        new_mesh_eidxs = deepcopy(mesh.eidxs)

        # preprocs new_mesh_eidxs
        new_mesh_eidxs = mesh.preproc_edict(self.mm.etypes, new_mesh_eidxs)

        for etype in self.mm.etypes:
            elements_to_remove = np.sort(np.concatenate([elements[etype] for elements in move_to_nrank.values()]))
            new_mesh_eidxs[etype] = np.setdiff1d(new_mesh_eidxs[etype], elements_to_remove)

        for etype in self.mm.etypes:
            elements_to_add = np.sort(moved_from_nrank0[etype])
            new_mesh_eidxs[etype] = np.unique(np.concatenate((new_mesh_eidxs[etype], elements_to_add)).astype(np.int32))

        return new_mesh_eidxs

    def get_gcon_p(self, mesh: _Mesh):
        return {
                nrank: {
                etype: np.array([mesh.eidxs[etype][inter[1]] 
                                 for inter in inters if inter[0] == etype ]) 
                       for etype in mesh.etypes 
                    } 
                for nrank, inters in mesh.con_p.items()
               } 

    @staticmethod
    def count_all_eidxs(mesh_eidxs):
        comm, rank, root = get_comm_rank_root()
        total_elements = sum(len(e) for e in mesh_eidxs.values())
        return comm.allreduce(total_elements, op=mpi.SUM)
