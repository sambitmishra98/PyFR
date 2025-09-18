from pyfr.mpiutil import get_comm_rank_root
from pyfr.readers.native import NativeReader

from pyfr.relocator.utils import crpprint

if __name__ == "__main__":
    comm, rank, _ = get_comm_rank_root()
    meshf = "/scratch/EFFORTS/LoadBalancer3/PyFR-Test-Cases/couette/squaremesh/square.pyfrm"
    reader  = NativeReader(meshf, pname="3", construct_con=True)
    
    mesh      = reader.mesh
    mmesh     = reader.mmesh

    mesh = mmesh.to_mesh(mesh) ; crpprint(-1,list(mesh.eidxs.items()))



    mmesh.diffuse(perf_cost_per_rank=[18,10,2])




    # Print eidxs to verify correctness
    mesh_recreated = mmesh.to_mesh(mesh) ; crpprint(-1,list(mesh_recreated.eidxs.items()))
    
    mmesh.smooth_interfaces_once()  # at most 3 from rank1 this pass

    mesh_recreated = mmesh.to_mesh(mesh) ; crpprint(-1,list(mesh_recreated.eidxs.items()))

    mmesh.smooth_interfaces_once()  # at most 3 from rank1 this pass
#
#    #if rank == 0:
#    #    pprint(summary)
#
    # Print eidxs to verify correctness
    mesh_recreated = mmesh.to_mesh(mesh) ; crpprint(-1,list(mesh_recreated.eidxs.items()))
#
#
#    mmesh.diffuse(perf_cost_per_rank=[1.5,1.0,0.5])
#    # Print eidxs to verify correctness
#    mesh_recreated = mmesh.to_mesh(mesh) ; crpprint(-1,list(mesh_recreated.eidxs.items()))
    
