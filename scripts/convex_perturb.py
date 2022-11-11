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

def RunProblemConvexOscillatory(kk,perturb_theta=None,compute_cond=True,div_known=False):
    orders = [1,2,3] 
    ls_mesh = get_mesh_hierarchy(6)
    refsol = get_reference_sol("oscillatory",kk=kk)
    elastic_convex.rho = get_ksquared_const(kk)
    #elastic_convex.rho = kk**2
    #elastic_convex.mu = 1.0
    #elastic_convex.lam = 1.25
    elastic_convex.mu = mu_var
    elastic_convex.lam = lam_var
    
    #tmp_gamma = 3e-4 
    tmp_gamma = 1e-5 
    if div_known:
        tmp_gamma = 1e-3


    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(3e-4)/kk**2,ScalarType(3e-4)/kk**2], [ ScalarType(1e-2),ScalarType(1e-2)]):
    #for add_bc,problem_type,pgamma,palpha in zip([False],["ill-posed"],[ScalarType(tmp_gamma)/kk**2], [ScalarType(1e-2)]):
    
    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(tmp_gamma),ScalarType(tmp_gamma)], [ScalarType(1e-3), ScalarType(1e-3)]): 
    for add_bc,problem_type,pgamma,palpha in zip([False],["ill-posed"],[ScalarType(tmp_gamma)], [ScalarType(1e-3)]):
        if MPI.COMM_WORLD.rank == 0:
            print("Considering {0} problem".format(problem_type))
        for order in orders:
            if div_known:
                if perturb_theta != None:
                    name_str = "Convex-Oscillatory-{0}-k{1}-order{2}-div-known-theta{3}.dat".format(problem_type,kk,order,perturb_theta)
                else:
                    name_str = "Convex-Oscillatory-{0}-k{1}-order{2}-div-known.dat".format(problem_type,kk,order)
            else:
                if perturb_theta != None:
                    name_str = "Convex-Oscillatory-{0}-k{1}-order{2}-theta{3}.dat".format(problem_type,kk,order,perturb_theta)
                else:
                    name_str = "Convex-Oscillatory-{0}-k{1}-order{2}.dat".format(problem_type,kk,order)
            #print(name_str)
            if MPI.COMM_WORLD.rank == 0:
                print("Computing for order = {0}".format(order))
             
            #errors_order = ConvergenceStudy(elastic_convex,ls_mesh[:-order],refsol,order=order,pgamma=pgamma/order**2,palpha=palpha,add_bc=add_bc,export_VTK=False,rhs=None,mu_Ind=None,perturb_theta=None,pGLS=1e-4/kk**4,name_str = name_str,compute_cond=compute_cond,div_known=div_known)
            errors_order = ConvergenceStudy(elastic_convex,ls_mesh[:-order],refsol,order=order,pgamma=pgamma/order**3.5,palpha=palpha,add_bc=add_bc,export_VTK=False,rhs=None,mu_Ind=None,perturb_theta=perturb_theta,pGLS=tmp_gamma/order**3.5,name_str = name_str,compute_cond=compute_cond,div_known=div_known)
            #print(errors_order)
            
            eoc = [ log(errors_order["L2-error-u-uh-B"][i-1]/errors_order["L2-error-u-uh-B"][i])/log(2) for i in range(1,len(errors_order["L2-error-u-uh-B"]))]
            if MPI.COMM_WORLD.rank == 0:
                print("l2-norm eoc = ", eoc)
            ndofs = np.array(errors_order["ndof"]) 
            h_order = order/ndofs**(1/2) 
            
            if compute_cond:
                eoc = [ log(errors_order["cond"][i-1]/errors_order["cond"][i])/log(2) for i in range(1,len(errors_order["cond"]))]
                print("cond eoc = ", eoc)
            
            if MPI.COMM_WORLD.rank == 0:
                if False:
                    plt.loglog(h_order, errors_order["s-norm"] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
                    tmp_str = "$\mathcal{{O}}(h^{0})$".format(order)
                    plt.loglog(h_order, h_order**order,label=tmp_str,linestyle='dashed',color='gray')
                    plt.xlabel("h")
                    plt.legend()
                    #plt.savefig("L2-error-convex-Gaussian-{0}-k{1}.png".format(problem_type,kk),transparent=True,dpi=200)
                    #plt.title("L2-error")
                    plt.show()
                    
                    for error_str in ["Jump-uh-Ph(u)","GLS-uh-Ph(u)-Omega","Tikh-uh-Ph(u)-Omega","L2-error-uh-Ph(u)-omega","s-norm"]:
                        #print(error_str)
                        eoc = [ log(errors_order[error_str][i-1]/errors_order[error_str][i])/log(2) for i in range(1,len(errors_order[error_str]))]
                        print("{0}, eoc = {1}".format(error_str,eoc))
                        error_vals =  errors_order[error_str]
                        plt.loglog(h_order, error_vals ,'-x',label="{0}".format(error_str),linewidth=3,markersize=8)
                    plt.loglog(h_order, h_order**order ,label=tmp_str,linestyle='dashed',color='gray')
                    plt.xlabel("h")
                    plt.legend()
                    plt.show()


if ( len(sys.argv) > 2): 
    kk = int(sys.argv[1])
    print("kk = ", kk)
    compute_cond = False 
    if sys.argv[2] == "True":
        compute_cond = True 
    if ( len(sys.argv) > 3): 
       perturb_theta =  int(sys.argv[3])
       RunProblemConvexOscillatory(kk=kk,compute_cond=False,perturb_theta=perturb_theta)
    else:
        RunProblemConvexOscillatory(kk=kk,compute_cond=compute_cond)
else:
    print("Invalid input.")


