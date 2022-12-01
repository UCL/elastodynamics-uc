# Some additional numerical experiments which are not part of the paper
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


def RunProblemPrimalCIPTest(kk,perturb_theta=None,compute_cond=True,div_known=False):
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
    
    #for add_bc,problem_type,pgamma,palpha in zip([False],["ill-posed"],[ScalarType(tmp_gamma)], [ScalarType(1e-3)]):
    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(tmp_gamma),ScalarType(tmp_gamma)], [ScalarType(1e-3), ScalarType(1e-3)]):
    for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(tmp_gamma),ScalarType(tmp_gamma)], [ScalarType(1e-3), ScalarType(1e-3)]):
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

def RunProblemJumpDisk(kk=1,apgamma=1e-1,apalpha=1e-1,mu_plus=1,mu_minus=2): 
   
    a = 1.0 
    #r = ufl.sqrt(x[0]*x[0]+x[1]*x[1])

    def omega_Ind_Disk(x):
     
        values = np.zeros(x.shape[1],dtype=ScalarType)
        omega_coords = np.logical_or(   ( x[0] <= -1.25 )  , 
            np.logical_or(   (x[0] >= 1.25 ), (x[1] <= -1.25)  )
            ) 
        rest_coords = np.invert(omega_coords)
        values[omega_coords] = np.full(sum(omega_coords), 1.0)
        values[rest_coords] = np.full(sum(rest_coords), 0)
        return values

    def B_Ind_Disk(x):
        values = np.zeros(x.shape[1],dtype=ScalarType)
        # Create a boolean array indicating which dofs (corresponding to cell centers)
        # that are in each domain
        rest_coords = np.logical_and( ( x[0] >= -0.5 ), 
            np.logical_and(   (x[0] <= 0.5 ),
              np.logical_and(   (x[1]>= -0.5),  (x[1]<= 0.5)  )
            )
          ) 
        B_coords = np.invert(rest_coords)
        values[B_coords] = np.full(sum(B_coords), 0.0)
        values[rest_coords] = np.full(sum(rest_coords), 1.0)
        return values

    elastic_convex.SetSubdomains(omega_Ind=omega_Ind_Disk,B_Ind=B_Ind_Disk)
    
    def boundary_indicator_centered_square(x):
        return ( np.isclose(x[0], -1.5) | np.isclose(x[0], 1.5) | np.isclose(x[1], -1.5) | np.isclose(x[1], 1.5) )
    elastic_convex.SetBoundaryIndicator(boundary_indicator_centered_square)
    

    orders = [1,2,3] 
    #order = 3
    elastic_convex.rho = get_ksquared_const(kk)
    elastic_convex.lam = lam_const
    
    h_init = 1.25
    n_ref = 6 
    ls_mesh = []
    for i in range(n_ref): 
        ls_mesh.append( get_mesh_inclusion(h_init=h_init/2**i,order=1) )
    #ls_mesh = get_mesh_hierarchy_fitted_disc(6,eta=eta)
    mu_plus = mu_plus
    #mu_plus = 1
    mu_minus = mu_minus
    refsol = get_reference_sol(type_str="jump-disk",kk=kk,mu_plus=mu_plus,mu_minus=mu_minus,lam=elastic_convex.lam(1))
    
    def mu_Ind(x):
        values = np.zeros(x.shape[1],dtype=ScalarType)
        upper_coords = (x[0]*x[0] + x[1]*x[1]) > a**2
        #print("upper_coords = ", upper_coords)
        lower_coords = np.invert(upper_coords)
        values[upper_coords] = np.full(sum(upper_coords), mu_plus)
        values[lower_coords] = np.full(sum(lower_coords), mu_minus)
        return values

    #elastic_convex.SetDiscontinuityIndicators(plus_Ind=plus_Ind,minus_Ind=minus_Ind)

    for add_bc,problem_type,pgamma,palpha,pGLS in zip([True,False],["well-posed","ill-posed"],[ ScalarType(apgamma),ScalarType(apgamma)], [ ScalarType(apalpha), ScalarType(apalpha)],[ ScalarType(apgamma),ScalarType(apgamma)] ):
    #for add_bc,problem_type,pgamma,palpha,pGLS in zip([False],["ill-posed"],[ ScalarType(apgamma)/ (kk**2) ] , [ ScalarType(apalpha)], [ ScalarType(1e-2/kk**4) ] ):
    #for add_bc,problem_type,pgamma,palpha,pGLS in zip([False],["ill-posed"],[ ScalarType(apgamma) ] , [ ScalarType(apalpha)], [ ScalarType(apgamma) ] ):
        
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
                errors = SolveProblem(problem=elastic_convex,msh=msh,refsol=refsol,order=order,pgamma=pgamma/order**3.5,palpha=palpha,add_bc=add_bc,export_VTK=True,rhs=None,mu_Ind=mu_Ind,pGLS=pGLS/order**3.5)
                #errors = SolveProblem(problem=elastic_convex,msh=msh,refsol=refsol,order=order,pgamma=pgamma/order**3.5,palpha=palpha,add_bc=add_bc,export_VTK=True,rhs=None,pGLS=pGLS/order**3.5)
                l2_error = errors["L2-error-u-uh-B"]
                ndof = errors["ndof"]  
                print("ndof = {0}, L2-error-B = {1}".format(ndof,l2_error))
                #L2_error_B_plus.append(errors["L2-error-u-uh-B-plus"] ) 
                #L2_error_B_minus.append(errors["L2-error-u-uh-B-minus"] ) 
                l2_errors.append(l2_error)
                ndofs.append(ndof)

            eoc = [ log(l2_errors[i-1]/l2_errors[i])/log(2) for i in range(1,len(l2_errors))]
            print("eoc = ", eoc)
            ndofs = np.array(ndofs) 
            h_mesh = order/ndofs**(1/2)
            idx_start = 2 
            rate_estimate, _ = np.polyfit(np.log(h_mesh)[idx_start:] , np.log(l2_errors)[idx_start:], 1)
    
            for error_type,error_str in zip([l2_errors],["L2-error-u-uh-B"]):
                #print(error_str)
                eoc = [ log(error_type[i-1]/error_type[i])/log(2) for i in range(1,len(error_type ))]
                print("{0}, eoc = {1}".format(error_str,eoc))


            l2_errors_order[order] =  l2_errors 
            h_order[order] = h_mesh
            eoc_order[order] = round(eoc[-1],2)
            
            if MPI.COMM_WORLD.rank == 0:
                name_str = "jump-disk-mup{0}-mum{1}-{2}-k{3}-order{4}.dat".format(mu_plus,mu_minus,problem_type,kk,order)
                results = [np.array(ndofs,dtype=float),np.array(h_order[order],dtype=float)]
                header_str = "ndof h "
                for error_type,error_str in zip([l2_errors, L2_error_B_minus, L2_error_B_plus ],["L2-error-u-uh-B"]):
                    results.append( np.array(error_type,dtype=float))
                    header_str += "{0} ".format(error_str)
                np.savetxt(fname ="../data/{0}".format(name_str),
                           X = np.transpose(results),
                           header = header_str,
                           comments = '')

        if MPI.COMM_WORLD.rank == 0:
            for order in [1,2,3]: 
                plt.loglog(h_order[order], l2_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
                #plt.loglog(h_order[order], L2_error_B_plus_order[order] ,'-x',label="+,p={0}".format(order),linewidth=3,markersize=8)
                #plt.loglog(h_order[order], L2_error_B_minus_order[order] ,linestyle='dashed',label="-,p={0}".format(order),linewidth=3,markersize=8)
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
            #plt.savefig("L2-error-convex-jump-{0}-k{1}.png".format(problem_type,kk),transparent=True,dpi=200)
            #plt.title("L2-error")
            plt.show()


def RunProblemJumpWavenumber(km=1,kp=1,apgamma=1e-1,apalpha=1e-1,eta=0.6): 

    #def omega_Ind_eta(x):
    #    
    #    values = np.zeros(x.shape[1],dtype=ScalarType)
    #    #omega_coords = np.logical_or(  np.logical_and(  x[0] <= 0.1 , x[1] <=eta  ) , 
    #    #    np.logical_or(  np.logical_and( x[0] >= 0.9 , x[1] <= eta  ) , (x[1] <= 0.25)  )
    #    #    ) 
    #    omega_coords = np.logical_or(   ( x[0] <= 0.1 )  , 
    #        np.logical_or(   (x[0] >= 0.9 ), (x[1] <= 0.25)  )
    #        ) 
    #    rest_coords = np.invert(omega_coords)
    #    values[omega_coords] = np.full(sum(omega_coords), 1.0)
    #    values[rest_coords] = np.full(sum(rest_coords), 0)
    #    return values
    
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
    elastic_convex.lam = lam_const
    elastic_convex.mu = mu_const
    ls_mesh = get_mesh_hierarchy_fitted_disc(6,eta=eta)
    refsol,rhs = get_reference_sol(type_str="jump-wavenumber",eta=0.6,lam=lam_const(1),km=km,kp=kp,mu=mu_const(1))
    
    def k_Ind(x):
        values = np.zeros(x.shape[1],dtype=ScalarType)
        upper_coords = x[1] > eta 
        lower_coords = np.invert(upper_coords)
        values[upper_coords] = np.full(sum(upper_coords), kp**2)
        values[lower_coords] = np.full(sum(lower_coords), km**2)
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

    #for add_bc,problem_type,pgamma,palpha,pGLS in zip([True,False],["well-posed","ill-posed"],[ ScalarType(apgamma),ScalarType(apgamma)],[ScalarType(apalpha),ScalarType(apalpha)],[ ScalarType(apgamma),ScalarType(apgamma)] ): 
    for add_bc,problem_type,pgamma,palpha,pGLS in zip([False],["ill-posed"],[ ScalarType(apgamma)],[ScalarType(apalpha)],[ ScalarType(apgamma)] ):

    #for add_bc,problem_type,pgamma,palpha,pGLS in zip([False],["ill-posed"],[ ScalarType(apgamma)/ (kk**2) ] , [ ScalarType(apalpha)], [ ScalarType(1e-2/kk**4) ] ):
    #for add_bc,problem_type,pgamma,palpha,pGLS in zip([False],["ill-posed"],[ ScalarType(apgamma) ] , [ ScalarType(apalpha)], [ ScalarType(apgamma) ] ):
        
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
                errors = SolveProblem(problem=elastic_convex,msh=msh,refsol=refsol,order=order,pgamma=pgamma/order**3.5,palpha=palpha,add_bc=add_bc,export_VTK=True,rhs=rhs,k_Ind=k_Ind,pGLS=pGLS/order**3.5)
                #errors = SolveProblem(problem=elastic_convex,msh=msh,refsol=refsol,order=order,pgamma=pgamma/order**3.5,palpha=palpha,add_bc=add_bc,export_VTK=True,rhs=None,k_Ind=k_Ind,pGLS=pGLS/order**3.5)
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
                name_str = "jump-eta-kp{0}-km{1}-{2}-order{3}.dat".format(kp,km,problem_type,order)
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
            #plt.savefig("L2-error-convex-jump-{0}-k{1}.png".format(problem_type,kk),transparent=True,dpi=200)
            #plt.title("L2-error")
            plt.show()


def RunProblemNonConvexOscillatoryGradTikhStab(kk,perturb_theta=None):
    
    orders = [1,2,3] 
    #ls_mesh = get_mesh_hierarchy_nonconvex(6)
    ls_mesh = get_mesh_hierarchy_nonconvex(6)
    refsol = get_reference_sol("oscillatory",kk=kk)
    elastic_nonconvex.rho = get_ksquared_const(kk)
    elastic_nonconvex.mu = mu_const
    elastic_nonconvex.lam = lam_const


    #for add_bc,problem_type,pgamma,palpha in zip([False],["ill-posed"],[ScalarType(1e-5)/kk**2], [ ScalarType(1e-2)]):
    for add_bc,problem_type,pgamma,palpha in zip([False],["ill-posed"],[ ScalarType(1e-6)/kk**2], [ ScalarType(1e-2)]):
    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ScalarType(1e-5)/kk**2, ScalarType(1e-5)/kk**2], [ ScalarType(1e-2), ScalarType(1e-2)]):
    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-2),ScalarType(1e-2)], [ ScalarType(1e-2),ScalarType(1e-2)]):
        print("Considering {0} problem".format(problem_type))
        for order in orders:
            name_str = "Non-Convex-Oscillatory-GradTikhStab-{0}-k{1}-order{2}.dat".format(problem_type,kk,order)
            #print(name_str)
            print("Computing for order = {0}".format(order))
            if kk == 1:
                errors_order = ConvergenceStudy(elastic_nonconvex,ls_mesh[:-order],refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False,rhs=None,mu_Ind=None,perturb_theta=None,pGLS=1e-4/kk**4,name_str = name_str,compute_cond=False, GradTikhStab=True)
            else:
                errors_order = ConvergenceStudy(elastic_nonconvex,ls_mesh[:-order],refsol,order=order,pgamma=10**(2+order)*pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False,rhs=None,mu_Ind=None,perturb_theta=None,pGLS=1e-4/kk**4,name_str = name_str,compute_cond=False, GradTikhStab=True)
            #errors_order = ConvergenceStudy(elastic_nonconvex,ls_mesh[:-order],refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False,rhs=None,mu_Ind=None,perturb_theta=None,pGLS=1e-1,name_str = name_str)
            print(errors_order)
            
            eoc = [ log(errors_order["L2-error-u-uh-B"][i-1]/errors_order["L2-error-u-uh-B"][i])/log(2) for i in range(1,len(errors_order["L2-error-u-uh-B"]))]
            print("eoc = ", eoc)
            ndofs = np.array(errors_order["ndof"]) 
            h_order = order/ndofs**(1/2)
            
            if MPI.COMM_WORLD.rank == 0:
                plt.loglog(h_order, errors_order["L2-error-u-uh-B"] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
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


def RunProblemNonConvexOscillatoryStabSweep(kk):
    orders = [1,2,3]  
    #ls_mesh = [ create_initial_mesh_nonconvex(init_h_scale = h_k) for h_k in meshwidths ]
    #if MPI.COMM_WORLD.rank == 0:
    ls_mesh = get_mesh_hierarchy_nonconvex(4,init_h_scale=1.0)
    msh = ls_mesh[2]
    
    elastic_nonconvex.mu = mu_const
    elastic_nonconvex.lam = lam_const
    elastic_nonconvex.rho = get_ksquared_const(kk)

    refsol = get_reference_sol("oscillatory",kk=kk)
    #msh = ls_mesh[3]
    add_bc = False
    problem_type = "ill-posed"
    #pgamma = ScalarType(5e-3)
    palpha = ScalarType(1e-1)
    #kks = np.linspace(1,20,6)
    #kks = [1+2*j for j in range(3)]
    #pxs = [1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1]
    #pxs = [1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1]
    #pxs = [1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1]
    pxs = [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1]
    #pxs = [1e-13,1e-11,1e-9,1e-7,1e-5,1e-3,1e-1,1e1]
    #pxs = [1e-12,1e-3,1e0]

    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(5e-3),ScalarType(5e-3)], [ ScalarType(1e-1),ScalarType(1e-1)] ):
    for param_str in ["gamma-Jump","gamma-GLS","alpha"]:
        print(param_str)
        px_param = pxs
        #if param_str == "alpha":
        #    print("here")
        #    px_param = pxs[7:]
        #    print("px_param =" , px_param)
        pxs_np = np.array(px_param)
        l2_errors_order = { }
        s_errors_order = { }
        cond_order = { }
        for order in orders:
            l2_errors = [] 
            s_errors = [] 
            conds = [] 
            for px in px_param:
                if param_str == "gamma-Jump":
                    pgamma = ScalarType(px)
                    palpha = ScalarType(1e-2)
                    #pGLS = ScalarType(1e-4/kk**4)
                    pGLS = ScalarType(px)
                elif param_str == "gamma-GLS":
                    pgamma = ScalarType(1e-4/(order*kk)**2)
                    palpha = ScalarType(1e-2)
                    pGLS = ScalarType(px)
                elif param_str == "alpha":
                    pgamma = ScalarType(1e-4/(order*kk)**2)
                    palpha = ScalarType(px)
                    pGLS = ScalarType(1e-4/(order*kk)**2)

                errors = SolveProblem(problem = elastic_nonconvex, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False,pGLS= pGLS,compute_cond=True) 
                l2_error = errors["L2-error-u-uh-B"]
                ndof = errors["ndof"]     
                l2_errors.append(l2_error)
                s_errors.append(errors["s-norm"])
                conds.append(errors["cond"])
                if MPI.COMM_WORLD.rank == 0:
                    print("ndof = {0}, l2_error = {1},px = {2}".format(ndof,l2_error,px))
            l2_errors_order[order] = l2_errors
            s_errors_order[order] = s_errors
            cond_order[order] = conds 

        if MPI.COMM_WORLD.rank == 0:
            name_str = "NonConvex-Oscillatory-StabSweep-{0}-kk{1}.dat".format(param_str,kk) 
            results = [pxs_np]
            header_str = "gamma-Jump "
            for order in orders: 
                results.append(np.array(l2_errors_order[order],dtype=float))
                header_str += "l2-order{0} ".format(order)
                results.append(np.array(s_errors_order[order],dtype=float))
                header_str += "s-order{0} ".format(order)
                results.append(np.array(cond_order[order],dtype=float))
                header_str += "cond{0} ".format(order)
            np.savetxt(fname ="../data/{0}".format(name_str),
                       X = np.transpose(results),
                       header = header_str,
                       comments = '')

            for order,lstyle in zip([1,2,3],['solid','dashed','dotted']):
                plt.loglog(pxs_np, l2_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)    
            plt.xlabel("stabilisation")
            plt.ylabel("L2-error")
            plt.legend()
            plt.savefig("NonConvex-Oscillatory-StabSweep-L2error-{0}-Jump-kk{1}.png".format(param_str,kk),transparent=True,dpi=200)
            #plt.title("L2-error")
            plt.show()
            
            for order,lstyle in zip([1,2,3],['solid','dashed','dotted']):
                plt.loglog(pxs_np, s_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)    
            plt.xlabel("$\gamma$")
            plt.ylabel("s-error")
            plt.legend()
            #plt.savefig("L2-error-stab-gamma.png",transparent=True,dpi=200)
            #plt.title("L2-error")
            plt.show()


def RunProblemNonConvexOscillatory(kk,perturb_theta=None,compute_cond=False,div_known=False):
    
    orders = [1,2,3] 
    ls_mesh = get_mesh_hierarchy_nonconvex(6)
    #ls_mesh = get_mesh_hierarchy_nonconvex(7)
    #ls_mesh = get_mesh_hierarchy_nonconvex(5)
    refsol = get_reference_sol("oscillatory",kk=kk)
    elastic_nonconvex.rho = get_ksquared_const(kk)
    elastic_nonconvex.mu = mu_const 
    elastic_nonconvex.lam = lam_const

    tmp_gamma = 1e-5
    #if kk == 4:
    #    tmp_gamma = 1e-3
    for add_bc,problem_type,pgamma,palpha in zip([False],["ill-posed"],[ScalarType(tmp_gamma)/kk**2], [ ScalarType(1e-2)]):
    #for add_bc,problem_type,pgamma,palpha in zip([False],["ill-posed"],[ ScalarType(1e-6)/kk**2], [ ScalarType(1e-2)]):
    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ScalarType(1e-5)/kk**2, ScalarType(1e-5)/kk**2], [ ScalarType(1e-2), ScalarType(1e-2)]):
    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-2),ScalarType(1e-2)], [ ScalarType(1e-2),ScalarType(1e-2)]):
        print("Considering {0} problem".format(problem_type))
        for order in orders:
            if div_known:
                name_str = "Non-Convex-Oscillatory-{0}-k{1}-order{2}-div-known.dat".format(problem_type,kk,order)
            else:
                name_str = "Non-Convex-Oscillatory-{0}-k{1}-order{2}.dat".format(problem_type,kk,order)
            #print(name_str)
            print("Computing for order = {0}".format(order))
            if kk == 1:
                errors_order = ConvergenceStudy(elastic_nonconvex,ls_mesh[:-order],refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False,rhs=None,mu_Ind=None,perturb_theta=None,pGLS=1e-4/kk**4,name_str = name_str,compute_cond=compute_cond,div_known=div_known)
            else:
                errors_order = ConvergenceStudy(elastic_nonconvex,ls_mesh[:-order],refsol,order=order,pgamma=10**(2+order)*pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False,rhs=None,mu_Ind=None,perturb_theta=None,pGLS=1e-4/kk**4,name_str = name_str,compute_cond=compute_cond,div_known=div_known)
            #errors_order = ConvergenceStudy(elastic_nonconvex,ls_mesh[:-order],refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False,rhs=None,mu_Ind=None,perturb_theta=None,pGLS=1e-1,name_str = name_str)
            print(errors_order)
            
            eoc = [ log(errors_order["L2-error-u-uh-B"][i-1]/errors_order["L2-error-u-uh-B"][i])/log(2) for i in range(1,len(errors_order["L2-error-u-uh-B"]))]
            print("L2-error eoc = ", eoc)
            ndofs = np.array(errors_order["ndof"]) 
            h_order = order/ndofs**(1/2)

            if compute_cond:
                eoc = [ log(errors_order["cond"][i-1]/errors_order["cond"][i])/log(2) for i in range(1,len(errors_order["cond"]))]
                print("cond eoc = ", eoc)

            if MPI.COMM_WORLD.rank == 0:
                plt.loglog(h_order, errors_order["L2-error-u-uh-B"] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
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

def RunProblemNonConvexGaussian(kk,perturb_theta=None):
    
    orders = [1,2,3] 
    ls_mesh = get_mesh_hierarchy_nonconvex(5)
    refsol = get_reference_sol("gaussian",kk=kk)
    elastic_nonconvex.rho = get_ksquared_const(kk)
    elastic_nonconvex.mu = mu_const
    elastic_nonconvex.lam = lam_const


    for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5)/kk**2,ScalarType(1e-5)/kk**2], [ ScalarType(1e-2),ScalarType(1e-2)]):
    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-2),ScalarType(1e-2)], [ ScalarType(1e-2),ScalarType(1e-2)]):
        print("Considering {0} problem".format(problem_type))
        for order in orders:
            name_str = "Non-Convex-Gaussian-{0}-k{1}-order{2}.dat".format(problem_type,kk,order)
            #print(name_str)
            print("Computing for order = {0}".format(order))
            errors_order = ConvergenceStudy(elastic_nonconvex,ls_mesh[:-order],refsol,order=order,pgamma=pgamma/order**2,palpha=palpha,add_bc=add_bc,export_VTK=False,rhs=None,mu_Ind=None,perturb_theta=None,pGLS=1e-5/kk**4,name_str = name_str)
            #errors_order = ConvergenceStudy(elastic_nonconvex,ls_mesh[:-order],refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False,rhs=None,mu_Ind=None,perturb_theta=None,pGLS=1e-1,name_str = name_str)
            print(errors_order)
            
            eoc = [ log(errors_order["L2-error-u-uh-B"][i-1]/errors_order["L2-error-u-uh-B"][i])/log(2) for i in range(1,len(errors_order["L2-error-u-uh-B"]))]
            print("eoc = ", eoc)
            ndofs = np.array(errors_order["ndof"]) 
            h_order = order/ndofs**(1/2) 
            
            if MPI.COMM_WORLD.rank == 0:
                plt.loglog(h_order, errors_order["L2-error-u-uh-B"] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
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

def RunProblemConvexGaussian(kk,perturb_theta=None):
    orders = [1,2,3] 
    ls_mesh = get_mesh_hierarchy(5)
    refsol = get_reference_sol("gaussian",kk=kk)
    elastic_convex.rho = get_ksquared_const(kk)
    elastic_convex.mu = mu_const
    elastic_convex.lam = lam_const

    for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(5e-3)/kk**2,ScalarType(5e-3)/kk**2], [ ScalarType(1e-2),ScalarType(1e-2)]):
        print("Considering {0} problem".format(problem_type))
        for order in orders:
            name_str = "Convex-Gaussian-{0}-k{1}-order{2}.dat".format(problem_type,kk,order)
            #print(name_str)
            print("Computing for order = {0}".format(order))
            errors_order = ConvergenceStudy(elastic_convex,ls_mesh[:-order],refsol,order=order,pgamma=pgamma/order**2,palpha=palpha,add_bc=add_bc,export_VTK=False,rhs=None,mu_Ind=None,perturb_theta=None,pGLS=5e-3/kk**4,name_str = name_str)
            print(errors_order)
            
            eoc = [ log(errors_order["s-norm"][i-1]/errors_order["s-norm"][i])/log(2) for i in range(1,len(errors_order["s-norm"]))]
            print("eoc = ", eoc)
            ndofs = np.array(errors_order["ndof"]) 
            h_order = order/ndofs**(1/2) 
            plt.loglog(h_order, errors_order["s-norm"] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
            tmp_str = "$\mathcal{{O}}(h^{0})$".format(order)
    
            if MPI.COMM_WORLD.rank == 0:
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

def RunProblemJumpEtaDataBottom(kk=1,apgamma=1e-5,apalpha=1e-3,mu_plus=1,mu_minus=2): 
    
    eta = 0.6
    
    def omega_Ind(x):     
        values = np.zeros(x.shape[1],dtype=ScalarType)
        omega_coords = x[1] <= 0.25 
        rest_coords = np.invert(omega_coords)
        values[omega_coords] = np.full(sum(omega_coords), 1.0)
        values[rest_coords] = np.full(sum(rest_coords), 0)
        return values

    def B_Ind(x):
        values = np.zeros(x.shape[1],dtype=ScalarType)
        # Create a boolean array indicating which dofs (corresponding to cell centers)
        # that are in each domain
        B_coords = np.logical_and( ( x[0] >= 0.25 ), 
            np.logical_and(   (x[0] <= 0.75 ),
              np.logical_and(   (x[1]>= 0.25),  (x[1]<= 0.9)  )
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
    
    h_init = 1.25
    n_ref = 7
    ls_mesh = []
    for i in range(n_ref): 
        ls_mesh.append( get_mesh_bottom_data(h_init=h_init/2**i,eta=eta) )

    #DrawMeshTikz(msh=ls_mesh[1],name="BottomDataJumpEta",case_str="BottomDataJumpEta") 
    
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

    
    for add_bc,problem_type,pgamma,palpha,pGLS in zip([False],["ill-posed"],[ ScalarType(apgamma) ] , [ ScalarType(apalpha)], [ ScalarType(apgamma) ] ):
    #for add_bc,problem_type,pgamma,palpha,pGLS in zip([True,False],["well-posed","ill-posed"],[ScalarType(apgamma),ScalarType(apgamma)],[ScalarType(apalpha),ScalarType(apalpha)], [ ScalarType(apgamma),ScalarType(apgamma)] ):
        
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
                name_str = "jump-eta-DataBottom-mup{0}-mum{1}-{2}-k{3}-order{4}.dat".format(mu_plus,mu_minus,problem_type,kk,order)
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
            plt.savefig("L2-error-jump-data-bottom{0}-k{1}.png".format(problem_type,kk),transparent=True,dpi=200)
            #plt.title("L2-error")
            plt.show()



def RunProblemJumpSquare(kk=1,apgamma=1e-1,apalpha=1e-1,mu_plus=1,mu_minus=2): 
   
    #a = 1.0 
    #r = ufl.sqrt(x[0]*x[0]+x[1]*x[1])
    #eta = 1.5
    eta = 0.25

    def omega_Ind_Square(x):
     
        values = np.zeros(x.shape[1],dtype=ScalarType)
        omega_coords = np.logical_or(  np.logical_and(  x[0] <= -1.25 , x[1] <=eta  ) , 
            np.logical_or(  np.logical_and( x[0] >= 1.25 , x[1] <= eta  ) , (x[1] <= -1.25)  )
            ) 
        #omega_coords = np.logical_or(   ( x[0] <= -1.25 )  , 
        #    np.logical_or(   (x[0] >= 1.25 ), (x[1] <= -1.25)  )
        #    ) 
        rest_coords = np.invert(omega_coords)
        values[omega_coords] = np.full(sum(omega_coords), 1.0)
        values[rest_coords] = np.full(sum(rest_coords), 0)
        return values

    def B_Ind_Square(x):
        values = np.zeros(x.shape[1],dtype=ScalarType)
        # Create a boolean array indicating which dofs (corresponding to cell centers)
        rest_coords = np.logical_and( ( x[0] >= -1.25 ), 
            np.logical_and(   (x[0] <= 1.25 ),
              np.logical_and(   (x[1]>= -1.25),  (x[1]<= 1.25)  )
            )
          ) 
        # that are in each domain
        #rest_coords = np.logical_and( ( x[0] >= -0.5 ), 
        #    np.logical_and(   (x[0] <= 0.5 ),
        #      np.logical_and(   (x[1]>= -0.5),  (x[1]<= 0.5)  )
        #    )
        #  ) 
        B_coords = np.invert(rest_coords)
        values[B_coords] = np.full(sum(B_coords), 0.0)
        values[rest_coords] = np.full(sum(rest_coords), 1.0)
        return values

    x_L = -0.75
    x_R = 0.75
    y_L = -0.75
    y_R = 0.75
    #y_R = 1.5

    def mu_Ind(x):
        values = np.zeros(x.shape[1],dtype=ScalarType)
        inner_coords = np.logical_and( ( x[0] >= x_L ), 
                np.logical_and(   (x[0] <= x_R ),
                  np.logical_and(   (x[1]>= y_L),  (x[1]<= y_R)  )
                )
              ) 
        outer_coords = np.invert(inner_coords)
        values[inner_coords] = np.full(sum(inner_coords), mu_plus)
        values[outer_coords] = np.full(sum(outer_coords), mu_minus)
        return values

    elastic_convex.SetSubdomains(omega_Ind=omega_Ind_Square,B_Ind=B_Ind_Square)
    
    def boundary_indicator_centered_square(x):
        return ( np.isclose(x[0], -1.5) | np.isclose(x[0], 1.5) | np.isclose(x[1], -1.5) | np.isclose(x[1], 1.5) )
    elastic_convex.SetBoundaryIndicator(boundary_indicator_centered_square)
    
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


    orders = [1,2,3] 
    #order = 3
    elastic_convex.rho = get_ksquared_const(kk)
    elastic_convex.lam = lam_const
    
    h_init = 1.25
    n_ref = 7
    ls_mesh = []
    for i in range(n_ref): 
        ls_mesh.append( get_mesh_inclusion_square(h_init=h_init/2**i,eta=eta) )
    
    #DrawMeshTikz(msh=ls_mesh[1],name="omega-Ind-incl-level1",case_str="inclusion-omega") 
    #DrawMeshTikz(msh=ls_mesh[1],name="B-Ind-incl-level1",case_str="inclusion-B") 

    #ls_mesh = get_mesh_hierarchy_fitted_disc(6,eta=eta)
    mu_plus = mu_plus
    #mu_plus = 1
    mu_minus = mu_minus
    refsol = get_reference_sol(type_str="jump-square",kk=kk,mu_plus=mu_plus,mu_minus=mu_minus,lam=elastic_convex.lam(1))
    #refsol = get_reference_sol("oscillatory",kk=kk)
    #def mu_Ind(x):
    #    values = np.zeros(x.shape[1],dtype=ScalarType)
    #    inner_coords = np.logical_and( ( x[0] >= x_L ), 
    #            np.logical_and(   (x[0] <= x_R ),
    #              np.logical_and(   (x[1]>= y_L),  (x[1]<= y_R)  )
    #            )
    #          ) 
    #    outer_coords = np.invert(inner_coords)
    #    values[inner_coords] = np.full(sum(inner_coords), mu_plus)
    #    values[outer_coords] = np.full(sum(outer_coords), mu_minus)
    #    return values
    
    #elastic_convex.SetDiscontinuityIndicators(plus_Ind=plus_Ind,minus_Ind=minus_Ind)

    #for add_bc,problem_type,pgamma,palpha,pGLS in zip([True,False],["well-posed","ill-posed"],[ ScalarType(apgamma),ScalarType(apgamma)], [ ScalarType(apalpha), ScalarType(apalpha)],[ ScalarType(apgamma),ScalarType(apgamma)] ):     
    for add_bc,problem_type,pgamma,palpha,pGLS in zip([False],["ill-posed"],[ ScalarType(apgamma)], [ ScalarType(apalpha)],[ ScalarType(apgamma)] ):     
    #for add_bc,problem_type,pgamma,palpha,pGLS in zip([False],["ill-posed"],[ ScalarType(apgamma)], [ ScalarType(apalpha)],[ ScalarType(apgamma)] ):
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
                errors = SolveProblem(problem=elastic_convex,msh=msh,refsol=refsol,order=order,pgamma=pgamma/order**3.5,palpha=palpha,add_bc=add_bc,export_VTK=True,rhs=None,mu_Ind=mu_Ind,pGLS=pGLS/order**3.5)
                #errors = SolveProblem(problem=elastic_convex,msh=msh,refsol=refsol,order=order,pgamma=pgamma/order**3.5,palpha=palpha,add_bc=add_bc,export_VTK=True,rhs=None,mu_Ind=None,pGLS=pGLS/order**3.5)
                #errors = SolveProblem(problem=elastic_convex,msh=msh,refsol=refsol,order=order,pgamma=pgamma/order**3.5,palpha=palpha,add_bc=add_bc,export_VTK=True,rhs=None,pGLS=pGLS/order**3.5)
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
    
            #for error_type,error_str in zip([l2_errors],["L2-error-u-uh-B"]):
            #    #print(error_str)
            #    eoc = [ log(error_type[i-1]/error_type[i])/log(2) for i in range(1,len(error_type ))]
            #    print("{0}, eoc = {1}".format(error_str,eoc))

            for error_type,error_str in zip([l2_errors,L2_error_B_minus, L2_error_B_plus ],["L2-error-u-uh-B","L2-error-u-uh-B-minus","L2-error-u-uh-B-plus"]):
            #for error_type,error_str in zip([l2_errors],["L2-error-u-uh-B"]):
                #print(error_str)
                eoc = [ log(error_type[i-1]/error_type[i])/log(2) for i in range(1,len(error_type ))]
                print("{0}, eoc = {1}".format(error_str,eoc))

            l2_errors_order[order] =  l2_errors 
            h_order[order] = h_mesh
            eoc_order[order] = round(eoc[-1],2)
            
            if MPI.COMM_WORLD.rank == 0:
                name_str = "jump-square-mup{0}-mum{1}-{2}-k{3}-order{4}.dat".format(mu_plus,mu_minus,problem_type,kk,order)
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
            if False:
                for order in [1,2,3]: 
                    plt.loglog(h_order[order], l2_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
                    #plt.loglog(h_order[order], L2_error_B_plus_order[order] ,'-x',label="+,p={0}".format(order),linewidth=3,markersize=8)
                    #plt.loglog(h_order[order], L2_error_B_minus_order[order] ,linestyle='dashed',label="-,p={0}".format(order),linewidth=3,markersize=8)
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
                #plt.savefig("L2-error-convex-jump-{0}-k{1}.png".format(problem_type,kk),transparent=True,dpi=200)
                #plt.title("L2-error")
                plt.show()


#RunProblemJumpSquare(kk=4,apgamma=1e-3,apalpha=1e-3,mu_plus=2,mu_minus=1)
#RunProblemJumpSquare(kk=4,apgamma=1e-3,apalpha=1e-3,mu_plus=1,mu_minus=2)

#RunProblemConvexOscillatoryStabSweep(kk=6,compute_cond=True) # Figure 2
#RunProblemConvexOscillatory(kk=6,compute_cond=True) # Figure 3
#RunProblemConvexOscillatory(kk=1,compute_cond=False,perturb_theta=0)
#RunProblemConvexOscillatory(kk=6,compute_cond=False,perturb_theta=0)
#RunProblemConvexOscillatory(kk=6,compute_cond=False,perturb_theta=1)
#RunProblemConvexOscillatory(kk=6,compute_cond=False,perturb_theta=2)
#RunProblemConvexOscillatoryKhscaling()
#RunProblemJump(kk=6,apgamma=1e-5,apalpha=1e-3,mu_plus=2,mu_minus=1)
#RunProblemJump(kk=6,apgamma=1e-5,apalpha=1e-3,mu_plus=2,mu_minus=1)
#RunProblemSplitGeom(kk=1,apgamma=1e-3,apalpha=1e-5,compute_cond=False,div_known=True)
#RunProblemSplitGeom(kk=1,apgamma=1e-3,apalpha=1e-5,compute_cond=False,div_known=False)
#RunProblemConvexOscillatory(kk=6,compute_cond=True) 

#RunProblemJump(kk=4,apgamma=1e-5,apalpha=1e-3,mu_plus=1,mu_minus=2)
#RunProblemJump(kk=8,apgamma=1e-5,apalpha=1e-3,mu_plus=2,mu_minus=1)

#RunProblemJumpEtaDataBottom(kk=4,apgamma=1e-4,apalpha=1e-3,mu_i=2,mu_e=1)
#RunProblemJumpInclDataBottom(kk=1,apgamma=1e-5,apalpha=1e-3,mu_i=.5,mu_e=1)


# additional runs, debugging
#RunProblemConvexOscillatory(kk=6,compute_cond=False,perturb_theta=0)
#RunProblemJump(kk=4,apgamma=1e-5,apalpha=1e-3,mu_plus=1,mu_minus=2)

#RunProblemJump(kk=8,apgamma=1e-5,apalpha=1e-3,mu_plus=2,mu_minus=1)


#RunProblemSplitGeom(kk=1,apgamma=1e-3,apalpha=1e-5,compute_cond=False,div_known=False)

#RunProblemJumpWavenumber(km=2,kp=6,apgamma=1e-5,apalpha=1e-3,eta=0.6)
#RunProblemJumpWavenumber(km=6,kp=2,apgamma=1e-5,apalpha=1e-3,eta=0.6)

#RunProblemJumpSquare(kk=1,apgamma=1e-1,apalpha=1e-1,mu_plus=1,mu_minus=2) 

#RunProblemJumpSquare(kk=4,apgamma=1e-3,apalpha=1e-3,mu_plus=2,mu_minus=1)
#RunProblemJumpSquare(kk=4,apgamma=1e-3,apalpha=1e-3,mu_plus=1,mu_minus=2)

#RunProblemPrimalCIPTest(kk=6,compute_cond=False) 
#RunProblemConvexOscillatoryKhscaling()
#RunProblemSplitGeom(kk=1,apgamma=5e-2,apalpha=1e-3,compute_cond=False,div_known=False)
#RunProblemSplitGeom(kk=4,apgamma=5e-2,apalpha=1e-3,compute_cond=False,div_known=True)

#RunProblemSplitGeom(kk=1,apgamma=1e-3,apalpha=1e-5,compute_cond=False,div_known=False)
#RunProblemSplitGeom(kk=4,apgamma=1e-3,apalpha=1e-5,compute_cond=False,div_known=False)
#RunProblemSplitGeom(kk=6,apgamma=1e-2,apalpha=1e-5,compute_cond=False,div_known=False)
#RunProblemSplitGeom(kk=1,apgamma=1e-2,apalpha=1e-5,compute_cond=False,div_known=False)

#RunProblemConvexOscillatoryKhscaling(gamma_CIP_primal = 1e-4,gamma_CIP_primal_str="gamma-CIP-primal-0p0001")
#RunProblemConvexOscillatoryKhscaling(gamma_CIP_primal = 1e-3,gamma_CIP_primal_str="gamma-CIP-primal-0p001")
#RunProblemConvexOscillatoryKhscaling(gamma_CIP_primal = 1e-6,gamma_CIP_primal_str="gamma-CIP-primal-0p000001")
#RunProblemConvexOscillatoryKhscaling(gamma_CIP_primal = 1e-5,gamma_CIP_primal_str="gamma-CIP-primal-0p00001")
#RunProblemConvexOscillatoryKhscaling(gamma_CIP_primal = -1e-3,gamma_CIP_primal_str="gamma-CIP-primal-m0p001")

#RunProblemConvexOscillatoryKhscaling(gamma_CIP_primal = -5e-2,gamma_CIP_primal_str="gamma-CIP-primal-m0p05")
#RunProblemConvexOscillatoryKhscaling(gamma_CIP_primal = 5e-2,gamma_CIP_primal_str="gamma-CIP-primal-0p05")
#RunProblemConvexOscillatoryKhscaling(gamma_CIP_primal = 1e-1,gamma_CIP_primal_str="gamma-CIP-primal-0p1")

#RunProblemJumpEtaDataBottom(kk=4,apgamma=1e-4,apalpha=1e-3,mu_plus=2,mu_minus=1)

#RunProblemJumpInclDataBottom(kk=1,apgamma=1e-5,apa)


#RunProblemConvexOscillatory(kk=6,compute_cond=False)

