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


def RunProblemJump(kk=1,apgamma=1e-1,apalpha=1e-1,mu_plus=1,mu_minus=2): 
    eta = 0.6
    
    #def omega_Ind_eta(x):
    #    
    #    values = np.zeros(x.shape[1],dtype=ScalarType)
        #omega_coords = np.logical_or(  np.logical_and(  x[0] <= 0.1 , x[1] <=eta  ) , 
        #    np.logical_or(  np.logical_and( x[0] >= 0.9 , x[1] <= eta  ) , (x[1] <= 0.25)  )
        #    ) 
    #    omega_coords = np.logical_or(   ( x[0] <= 0.1 )  , 
    #        np.logical_or(   (x[0] >= 0.9 ), (x[1] <= 0.25)  )
    #        ) 
    #    rest_coords = np.invert(omega_coords)
    #    values[omega_coords] = np.full(sum(omega_coords), 1.0)
    #    values[rest_coords] = np.full(sum(rest_coords), 0)
    #   return values

    #elastic_convex.SetSubdomains(omega_Ind=omega_Ind_eta,B_Ind=elastic_convex.B_Ind)
    
    def omega_Ind(x):
        
        values = np.zeros(x.shape[1],dtype=ScalarType)
        omega_coords = np.logical_or(  np.logical_and(  x[0] <= 0.1 , x[1] <=eta  ) , 
            np.logical_or(  np.logical_and( x[0] >= 0.9 , x[1] <= eta  ) , (x[1] <= 0.25)  )
            ) 
        rest_coords = np.invert(omega_coords)
        values[omega_coords] = np.full(sum(omega_coords), 1.0)
        values[rest_coords] = np.full(sum(rest_coords), 0)
        return values

    def B_Ind(x):
        values = np.zeros(x.shape[1],dtype=ScalarType)
        # Create a boolean array indicating which dofs (corresponding to cell centers)
        # that are in each domain
        B_coords = np.logical_and( ( x[0] >= 0.1 ), 
            np.logical_and(   (x[0] <= 0.9 ),
              np.logical_and(   (x[1]>= 0.25),  (x[1]<= 0.95)  )
            )
        ) 
        rest_coords = np.invert(B_coords)
        values[B_coords] = np.full(sum(B_coords), 1.0)
        values[rest_coords] = np.full(sum(rest_coords), 0)
        return values

    elastic_convex.SetSubdomains(omega_Ind=omega_Ind,B_Ind=B_Ind)
    
    orders = [1,2,3] 
    #order = 3
    elastic_convex.rho = get_ksquared_const(kk)
    elastic_convex.lam = lam_const
    ls_mesh = get_mesh_hierarchy_fitted_disc(6,eta=eta)
    
    DrawMeshTikz(msh=ls_mesh[0],name="omega-splitgeom-mujump-lvl0",case_str="splitgeom-omega-mujump") 
    mu_plus = mu_plus
    #mu_plus = 1
    mu_minus = mu_minus
    refsol,rhs = get_reference_sol(type_str="jump",kk=kk,eta=eta,mu_plus=mu_plus,mu_minus=mu_minus,lam=elastic_convex.lam(1))
    def mu_Ind(x):
        values = np.zeros(x.shape[1],dtype=ScalarType)
        upper_coords = x[1] > eta 
        lower_coords = np.invert(upper_coords)
        values[upper_coords] = np.full(sum(upper_coords), mu_plus)
        values[lower_coords] = np.full(sum(lower_coords), mu_minus)
        return values
    def plus_Ind(x):
        values = np.zeros(x.shape[1],dtype=ScalarType)
        upper_coords = x[1] > eta 
        lower_coords = np.invert(upper_coords)
        values[upper_coords] = np.full(sum(upper_coords), 1.0)
        values[lower_coords] = np.full(sum(lower_coords), 0.0)
        return values
    def minus_Ind(x):
        values = np.zeros(x.shape[1],dtype=ScalarType)
        upper_coords = x[1] > eta 
        lower_coords = np.invert(upper_coords)
        values[upper_coords] = np.full(sum(upper_coords), 0.0)
        values[lower_coords] = np.full(sum(lower_coords), 1.0)
        return values
    elastic_convex.SetDiscontinuityIndicators(plus_Ind=plus_Ind,minus_Ind=minus_Ind)

    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(apgamma),ScalarType(apgamma)], [ ScalarType(apgamma),ScalarType(apgamma)] ):
    #for add_bc,problem_type,pgamma,palpha,pGLS in zip([False],["ill-posed"],[ ScalarType(apgamma)/ (kk**2) ] , [ ScalarType(apalpha)], [ ScalarType(1e-2/kk**4) ] ):
    for add_bc,problem_type,pgamma,palpha,pGLS in zip([False],["ill-posed"],[ ScalarType(apgamma) ] , [ ScalarType(apalpha)], [ ScalarType(apgamma) ] ):
        
        print("Considering {0} problem".format(problem_type))
        l2_errors_order = { }
        eoc_order = { }
        h_order = { }
        L2_error_B_plus_order = { } 
        L2_error_B_minus_order = { } 
        for order in orders:
            l2_errors = [ ]
            L2_error_B_plus = [] 
            L2_error_B_minus = [] 
            ndofs = [] 
            for msh in ls_mesh[:-order]:
                errors = SolveProblem(problem=elastic_convex,msh=msh,refsol=refsol,order=order,pgamma=pgamma/order**3.5,palpha=palpha,add_bc=add_bc,export_VTK=True,rhs=rhs,mu_Ind=mu_Ind,pGLS=pGLS/order**3.5)
                l2_error = errors["L2-error-u-uh-B"]
                ndof = errors["ndof"]  
                print("ndof = {0}, L2-error-B = {1}".format(ndof,l2_error))
                L2_error_B_plus.append(errors["L2-error-u-uh-B-plus"] ) 
                L2_error_B_minus.append(errors["L2-error-u-uh-B-minus"] ) 
                l2_errors.append(l2_error)
                ndofs.append(ndof)

            eoc = [ log(l2_errors[i-1]/l2_errors[i])/log(2) for i in range(1,len(l2_errors))]
            print("eoc = ", eoc)
            ndofs = np.array(ndofs) 
            h_mesh = order/ndofs**(1/2)
            idx_start = 2 
            rate_estimate, _ = np.polyfit(np.log(h_mesh)[idx_start:] , np.log(l2_errors)[idx_start:], 1)
    
            for error_type,error_str in zip([l2_errors,L2_error_B_minus, L2_error_B_plus ],["L2-error-u-uh-B","L2-error-u-uh-B-minus","L2-error-u-uh-B-plus"]):
                #print(error_str)
                eoc = [ log(error_type[i-1]/error_type[i])/log(2) for i in range(1,len(error_type ))]
                print("{0}, eoc = {1}".format(error_str,eoc))


            l2_errors_order[order] =  l2_errors 
            L2_error_B_plus_order[order] = L2_error_B_plus
            L2_error_B_minus_order[order] = L2_error_B_minus
            h_order[order] = h_mesh
            eoc_order[order] = round(eoc[-1],2)
            
            if MPI.COMM_WORLD.rank == 0:
                name_str = "jump-mup{0}-mum{1}-{2}-k{3}-order{4}.dat".format(mu_plus,mu_minus,problem_type,kk,order)
                results = [np.array(ndofs,dtype=float),np.array(h_order[order],dtype=float)]
                header_str = "ndof h "
                for error_type,error_str in zip([l2_errors, L2_error_B_minus, L2_error_B_plus ],["L2-error-u-uh-B","L2-error-u-uh-B-minus","L2-error-u-uh-B-plus"]):
                    results.append( np.array(error_type,dtype=float))
                    header_str += "{0} ".format(error_str)
                np.savetxt(fname ="../data/{0}".format(name_str),
                           X = np.transpose(results),
                           header = header_str,
                           comments = '')

        if MPI.COMM_WORLD.rank == 0:
            for order in [1,2,3]: 
                #plt.loglog(h_order[order], l2_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
                plt.loglog(h_order[order], L2_error_B_plus_order[order] ,'-x',label="+,p={0}".format(order),linewidth=3,markersize=8)
                plt.loglog(h_order[order], L2_error_B_minus_order[order] ,linestyle='dashed',label="-,p={0}".format(order),linewidth=3,markersize=8)
                #tmp_str += ",eoc={:.2f}".format(eoc_order[order])
            if problem_type == "well-posed":
                for order,lstyle in zip([1,2,3],['solid','dashed','dotted']): 
                    tmp_str = "$\mathcal{{O}}(h^{0})$".format(order+1)
                    plt.loglog(h_order[order], l2_errors_order[order][0]*(h_order[order]**(order+1))/( h_order[order][0]**(order+1)) ,label=tmp_str,linestyle=lstyle,color='gray')
            if problem_type == "ill-posed":
                for order,lstyle in zip([1,2],['solid','dashed','dotted']):
                    tmp_str = "$\mathcal{{O}}(h^{0})$".format(order)
                    plt.loglog(h_order[order], l2_errors_order[order][0]*(h_order[order]**(order))/( h_order[order][0]**(order)) ,label=tmp_str,linestyle=lstyle,color='gray')
                    #aeoc = eoc_order[order]
                    #pow_a = "{:.2f}".format(order)
                    #tmp_str = "eoc = $".format(pow_a)
                    #plt.loglog(h_order[order], l2_errors_order[order][0]*(h_order[order]**aeoc)/( h_order[order][0]**aeoc) ,label=tmp_str,linestyle=lstyle,color='gray')

            plt.xlabel("~h")
            plt.ylabel("L2-error")
            plt.legend()
            plt.savefig("L2-error-convex-jump-{0}-k{1}.png".format(problem_type,kk),transparent=True,dpi=200)
            #plt.title("L2-error")
            plt.show()



RunProblemJump(kk=4,apgamma=1e-5,apalpha=1e-3,mu_plus=1,mu_minus=2)
RunProblemJump(kk=4,apgamma=1e-5,apalpha=1e-3,mu_plus=2,mu_minus=1)
