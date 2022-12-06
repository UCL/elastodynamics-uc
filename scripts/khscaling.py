import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, plot, io
from petsc4py.PETSc import ScalarType
from math import pi,log
import math
from meshes import get_mesh_hierarchy, get_mesh_hierarchy_nonconvex,get_mesh_hierarchy_fitted_disc,get_mesh_convex,create_initial_mesh_convex,get_mesh_inclusion,get_mesh_inclusion_square,DrawMeshTikz,get_mesh_bottom_data
from problems import elastic_convex, elastic_nonconvex,mu_const,lam_const,mu_var,lam_var,get_ksquared_const
from dolfinx.cpp.mesh import h as geth
from petsc4py import PETSc
from slepc4py import SLEPc
from elastodynamics_UC import SolveProblem, get_reference_sol, ConvergenceStudy
import matplotlib.pyplot as plt 
import sys
plt.rc('legend',fontsize=14)
plt.rc('axes',titlesize=14)
plt.rc('axes',labelsize=14)
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

def RunProblemConvexOscillatoryKhscaling(gamma_CIP_primal = 0,gamma_CIP_primal_str="gamma-CIP-primal-0"):
    
    orders = [1,2] 
    #orders = [1,2,3] 
    #ratio = 1/2
    ratio = 1
    #kks = [1+2*j for j in range(4)]
    
    kks = [1+j for j in range(10)]
    #kks = [1+j for j in range(8)]
    
    #kks = [1+j for j in range(5)]
    kks_np = np.array(kks)
    
    meshwidths = [ ratio/kk for kk in kks   ] 
    print("meshwidth = ", meshwidths )
    ls_mesh = [ create_initial_mesh_convex(init_h_scale = h_k) for h_k in meshwidths ]
    #ls_mesh = get_mesh_convex(6,init_h_scale=1.0)   
    
    elastic_convex.mu = mu_const
    elastic_convex.lam = lam_const

    #for str_param in ["tuned","naive"]:
    for str_param in ["tuned"]:
        #for add_bc,problem_type,pgamma,palpha,pGLS in zip([True,False],["well-posed","ill-posed"], [ ScalarType(1e-4),ScalarType(1e-4) ], 
        #                                                 [ ScalarType(5e-1),ScalarType(5e-1)],[ ScalarType(1e-4),ScalarType(1e-4)]   ):
        for add_bc,problem_type,pgamma,palpha,pGLS in zip([True,False],["well-posed","ill-posed"], [ ScalarType(1e-5),ScalarType(1e-5) ], 
                                                         [ ScalarType(1e-3),ScalarType(1e-3)],[ ScalarType(1e-5),ScalarType(1e-5)]   ):
        #for add_bc,problem_type,pgamma,palpha,pGLS in zip([False],["ill-posed"], [ ScalarType(1e-5) ], 
        #                                                 [ ScalarType(1e-3)],[ ScalarType(1e-5)]   ):
            print("Considering {0} problem".format(problem_type))
            l2_errors_order = { }
            h1_semi_errors_order = { }
            k_norm_error = { } 
            for order in orders:
                l2_errors = []
                h1_semi_errors = [] 
                k_norm_errors = [] 
                for kk,msh in zip(kks,ls_mesh):
                    print("kk = {0}".format(kk)) 
                    elastic_convex.rho = get_ksquared_const(kk)
                    refsol = get_reference_sol("oscillatory",kk=kk)
                    if str_param == "tuned":
                        #errors = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma/(order*kk)**2,palpha=palpha,add_bc=add_bc,export_VTK=False,pGLS=pGLS/kk**4)  
                        errors = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma/(order)**3.5,palpha=palpha,add_bc=add_bc,export_VTK=False,pGLS=pGLS/order**3.5, 
                                   gamma_CIP_primal = gamma_CIP_primal )             
                    else:
                        errors = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False,pGLS=pGLS)
                    l2_error = errors["L2-error-u-uh-B"]
                    h1_semi_error = errors["H1-semi-error-u-uh-B-absolute"]
                    ndof = errors["ndof"]     
                    l2_errors.append(l2_error)
                    h1_semi_errors.append(h1_semi_error)
                    k_norm_errors.append(  kk*errors["L2-error-u-uh-B-absolute"] + errors["H1-semi-error-u-uh-B-absolute"] ) 
                l2_errors_order[order] = l2_errors
                h1_semi_errors_order[order] = h1_semi_errors
                k_norm_error[order] =  k_norm_errors 

            if MPI.COMM_WORLD.rank == 0:
                name_str = "Convex-Oscillatory-kh-scaling-{0}-{1}-{2}.dat".format(problem_type,str_param, gamma_CIP_primal_str)
                results = [kks_np]
                header_str = "k "
                for order in orders: 
                    #results.append(np.array(h1_semi_errors_order[order],dtype=float))
                    results.append(np.array(k_norm_error[order],dtype=float))
                    header_str += "k-norm-order{0} ".format(order)
                np.savetxt(fname ="../data/{0}".format(name_str),
                           X = np.transpose(results),
                           header = header_str,
                           comments = '')
            

            #if MPI.COMM_WORLD.rank == 0:
            #    name_str = "Convex-Oscillatory-kh-scaling-{0}-{1}.dat".format(problem_type,str_param)
            #    results = [kks_np]
            #    header_str = "k "
            #    for order in orders: 
            #        results.append(np.array(l2_errors_order[order],dtype=float))
            #        header_str += "l2-order{0} ".format(order)
            #        results.append(np.array(h1_semi_errors_order[order],dtype=float))
            #        header_str += "h1-semi-order{0} ".format(order)
            #    np.savetxt(fname ="../data/{0}".format(name_str),
            #               X = np.transpose(results),
            #               header = header_str,
            #               comments = '')

                for order,lstyle in zip(orders,['solid','dashed','dotted']):
                    #plt.loglog(kks_np, l2_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
                    plt.loglog(kks_np,k_norm_error[order],'-x',label="p={0}".format(order),linewidth=3,markersize=8)
                
                    #tmp_str = "$\mathcal{{O}}(k^{0})$".format(order+1)
                    #plt.loglog(kks_np, 1.35*l2_errors_order[order][0]*(kks_np**(order+1))/( kks_np[0]**(order+1)) ,label=tmp_str,linestyle=lstyle,color='gray')

                plt.loglog(kks_np, 1.35*k_norm_error[1][0]*(kks_np)/( kks_np[0]) ,label="$\mathcal{O}(k)$",linestyle=lstyle,color='gray')
                plt.loglog(kks_np, 1.0*k_norm_error[2][0]*(kks_np**2)/( kks_np[0]**2 ) ,label="$\mathcal{O}(k^2)$",linestyle=lstyle,color='gray')
                #plt.loglog(kks_np, 1.0*k_norm_error[3][0]*(kks_np**2)/( kks_np[0]**2 ) ,label="$\mathcal{O}(k^2)$",linestyle=lstyle,color='gray')
                #plt.loglog(kks_np, 1.35*k_norm_error[3][0]*(kks_np)/( kks_np[0]) ,label="$\mathcal{O}(k)$",linestyle=lstyle,color='gray')
                
                #plt.loglog(kks_np, 1.35*l2_errors_order[2][0]*(kks_np)/( kks_np[0]) ,label="$\mathcal{O}(k)$",linestyle=lstyle,color='gray')
                plt.xlabel("$k$")
                plt.ylabel("L2-error")
                plt.legend()
                plt.savefig("L2-error-k-Gaussian.png",transparent=True,dpi=200)
                #plt.title("L2-error")
                plt.show()


if ( len(sys.argv) > 1): 
    if sys.argv[1] == "small":
        RunProblemConvexOscillatoryKhscaling(gamma_CIP_primal = 1e-4,gamma_CIP_primal_str="gamma-CIP-primal-0p0001")
    elif sys.argv[1] == "tiny":
        RunProblemConvexOscillatoryKhscaling(gamma_CIP_primal = 1e-6,gamma_CIP_primal_str="gamma-CIP-primal-0p000001")
    #elif sys.argv[1] == "larger":
    #   RunProblemConvexOscillatoryKhscaling(gamma_CIP_primal = 1e-3,gamma_CIP_primal_str="gamma-CIP-primal-m0p001-fp")
    else: 
        print("Invalid input.")
else:
    RunProblemConvexOscillatoryKhscaling()

