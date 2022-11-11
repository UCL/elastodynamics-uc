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
from elastodynamics_UC import SolveProblem, get_reference_sol
import matplotlib.pyplot as plt 
plt.rc('legend',fontsize=14)
plt.rc('axes',titlesize=14)
plt.rc('axes',labelsize=14)
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

def RunProblemConvexOscillatoryStabSweep(kk,div_known=False,compute_cond=False):
    
    orders = [1,2,3]  
    #ratio = 1/2 
    #meshwidths = [ ratio/kk for kk in kks   ] 
    #print("meshwidth = ", meshwidths )
    #ls_mesh = [ create_initial_mesh_convex(init_h_scale = h_k) for h_k in meshwidths ]
    ls_mesh = get_mesh_convex(4,init_h_scale=1.0)
    msh = ls_mesh[2]
    
    elastic_convex.mu = mu_var
    elastic_convex.lam = lam_var
    elastic_convex.rho = get_ksquared_const(kk)


    refsol = get_reference_sol("oscillatory",kk=kk)
    #msh = ls_mesh[3]
    add_bc = False
    problem_type = "ill-posed"
    #pgamma = ScalarType(5e-3)
    palpha = ScalarType(1e-1)
    #kks = np.linspace(1,20,6)
    #kks = [1+2*j for j in range(3)]
    #pxs = [1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1]
    pxs = [1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1]
    #pxs = [1e-13,1e-11,1e-9,1e-7,1e-5,1e-3,1e-1,1e1]
    #pxs = [1e-12,1e-3,1e0]
    pxs_np = np.array(pxs)

    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(5e-3),ScalarType(5e-3)], [ ScalarType(1e-1),ScalarType(1e-1)] ):
    for param_str in ["gamma-Jump","gamma-GLS","alpha"]:
    #for param_str in ["gamma-GLS","alpha"]:
        l2_errors_order = { }
        s_errors_order = { }
        cond_order = { } 
        for order in orders:
            l2_errors = [] 
            s_errors = [] 
            cond = [] 
            for px in pxs:
                if param_str == "gamma-Jump":
                    pgamma = ScalarType(px)
                    palpha = ScalarType(1e-3)
                    pGLS = ScalarType(1e-12)
                    #pGLS = ScalarType(1e-4/kk**4)
                    #pGLS = ScalarType(px)
                elif param_str == "gamma-GLS":
                    #pgamma = ScalarType(1e-12)
                    pgamma = ScalarType(1e-5/(order)**3.5)
                    palpha = ScalarType(1e-3)
                    pGLS = ScalarType(px)
                elif param_str == "alpha":
                    pgamma = ScalarType(1e-5/(order)**3.5)
                    #pgamma = ScalarType(1e-4/(order*kk)**2)
                    palpha = ScalarType(px)
                    #pGLS = ScalarType(1e-4/(order*kk)**2)
                    pGLS = ScalarType(1e-5/(order)**3.5)
                    #pGLS = ScalarType(1e-5/(order)**3.5)

                errors = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False,pGLS= pGLS,compute_cond=compute_cond ,div_known=div_known)
                l2_error = errors["L2-error-u-uh-B"]
                ndof = errors["ndof"]     
                l2_errors.append(l2_error)
                s_errors.append(errors["s-norm"])
                cond.append(errors["cond"])
                print("ndof = {0}, l2_error = {1},px = {2}".format(ndof,l2_error,px))
            l2_errors_order[order] = l2_errors
            s_errors_order[order] = s_errors 
            cond_order[order] = cond

        if MPI.COMM_WORLD.rank == 0:
            if div_known:
               name_str = "Convex-Oscillatory-StabSweep-{0}-kk{1}-div-known.dat".format(param_str,kk)
            else:
               name_str = "Convex-Oscillatory-StabSweep-{0}-kk{1}.dat".format(param_str,kk)
            results = [pxs_np]
            header_str = "gamma-Jump "
            for order in orders: 
                results.append(np.array(l2_errors_order[order],dtype=float))
                header_str += "l2-order{0} ".format(order)
                results.append(np.array(s_errors_order[order],dtype=float))
                header_str += "s-order{0} ".format(order)
                results.append(np.array(cond_order[order],dtype=float))
                header_str += "cond-order{0} ".format(order)
            np.savetxt(fname ="../data/{0}".format(name_str),
                       X = np.transpose(results),
                       header = header_str,
                       comments = '')

            for order,lstyle in zip([1,2,3],['solid','dashed','dotted']):
                plt.loglog(pxs_np, l2_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)    
            plt.xlabel("stabilisation")
            plt.ylabel("L2-error")
            plt.legend()
            plt.savefig("Convex-Oscillatory-StabSweep-L2error-{0}-Jump-kk{1}.png".format(param_str,kk),transparent=True,dpi=200)
            #plt.title("L2-error")
            #plt.show()
            
            #for order,lstyle in zip([1,2,3],['solid','dashed','dotted']):
            #    plt.loglog(pxs_np, s_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)    
            #plt.xlabel("$\gamma$")
            #plt.ylabel("s-error")
            #plt.legend()
            #plt.savefig("L2-error-stab-gamma.png",transparent=True,dpi=200)
            #plt.title("L2-error")
            #plt.show()


RunProblemConvexOscillatoryStabSweep(kk=6,compute_cond=True)
