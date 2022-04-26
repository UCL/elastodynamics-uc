'''
Unique continuation for elastodynamics. 
'''

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, plot, io
from petsc4py.PETSc import ScalarType
from math import pi,log
from meshes import get_mesh_hierarchy, get_mesh_hierarchy_nonconvex,get_mesh_hierarchy_fitted_disc
from problems import elastic_convex, elastic_nonconvex

#kk = 10 # wavenumber
#order = 2

def epsilon(u):
    return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

def get_reference_sol(type_str,kk=1,eta=0.6,mu_plus=2,mu_minus=1,lam=1.25,nn=5):

    if type_str == "oscillatory":
        def oscillatory_sol(x):
            return ufl.as_vector([ ufl.sin(kk*pi*x[0]) * ufl.sin(kk*pi*x[1]),
                            ufl.sin(kk*pi*x[0]) * ufl.sin(kk*pi*x[1])] ) 
        return oscillatory_sol
    elif type_str =="gaussian": 
        sigma_x = ScalarType(0.01)
        sigma_y = ScalarType(0.1)
        x_s = ScalarType(0.5)
        y_s = ScalarType(1.0)
        def gaussian_sol(x):
            return ufl.as_vector([ ufl.exp( -( (x[0] - x_s) ** 2 / (2*sigma_x)  + (x[1] - y_s) ** 2 / (2*sigma_y)  ) ),
                                   ufl.exp( -( (x[0] - x_s) ** 2 / (2*sigma_x)  + (x[1] - y_s) ** 2 / (2*sigma_y)  ) )   ])
        return gaussian_sol
    elif type_str == "jump":
        b1 = 0.0
        c1 = (0.5*kk*pi/eta)*(mu_plus-mu_minus)/mu_plus 
        a1 = 1.0 - c1*eta**2
        b2 = 1.0
        c2 = -0.5/eta 
        a2 = 1 - b2*eta - c2*eta**2
        def jump_sol(x):
            u1_plus = (a1+b1*x[1]+c1*x[1]**2)*ufl.sin(kk*pi*x[0])
            u1_minus = ufl.sin(kk*pi*x[0])*ufl.cos(kk*pi*(x[1]-eta))
            u2_plus = (a2+b2*x[1]+c2*x[1]**2)*ufl.cos(kk*pi*x[0])
            u2_minus = ufl.cos(kk*pi*x[0])*ufl.cos(kk*pi*(x[1]-eta))
            u1 = ufl.conditional( ufl.gt(x[1]-eta,0), u1_plus,u1_minus) 
            u2 = ufl.conditional( ufl.gt(x[1]-eta,0), u2_plus,u2_minus) 
            return ufl.as_vector([u1, u2])
        def jump_f(x):
            u1_plus = (a1+b1*x[1]+c1*x[1]**2)*ufl.sin(kk*pi*x[0])
            u1_minus = ufl.sin(kk*pi*x[0])*ufl.cos(kk*pi*(x[1]-eta))
            u2_plus = (a2+b2*x[1]+c2*x[1]**2)*ufl.cos(kk*pi*x[0])
            u2_minus = ufl.cos(kk*pi*x[0])*ufl.cos(kk*pi*(x[1]-eta))
            f1_minus = (2*mu_minus + lam) * (kk*pi)**2 * u1_minus + mu_minus * (kk*pi)**2*u1_minus - (lam+mu_minus)*(kk*pi)**2*ufl.sin(kk*pi*x[0])*ufl.sin(kk*pi*(x[1]-eta)) - kk**2*u1_minus
            f2_minus = (2*mu_minus + lam) * (kk*pi)**2 * u2_minus + mu_minus * (kk*pi)**2*u2_minus + (lam+mu_minus)*(kk*pi)**2*ufl.cos(kk*pi*x[0])*ufl.sin(kk*pi*(x[1]-eta)) - kk**2*u2_minus
            f1_plus = (2*mu_plus+lam)*(kk*pi)**2*u1_plus - mu_plus *2*c1*ufl.sin(kk*pi*x[0]) + (lam+mu_plus) * (kk*pi) * ufl.sin(kk*pi*x[0])*(b2+2*c2*x[1]) - kk**2*u1_plus 
            f2_plus = -(2*mu_plus+lam)*2*c2*ufl.cos(kk*pi*x[0]) + mu_plus*(kk*pi)**2*u2_plus - (lam+mu_plus) * (kk*pi) * (b1+2*c1*x[1])*ufl.cos(kk*pi*x[0]) - kk**2*u2_plus
            f1 = ufl.conditional( ufl.gt(x[1]-eta,0), f1_plus,f1_minus) 
            f2 = ufl.conditional( ufl.gt(x[1]-eta,0), f2_plus,f2_minus) 
            return ufl.as_vector([f1, f2])
        return jump_sol,jump_f
    if type_str == "Hadamard":
        def Hadamard_sol(x):
            return ufl.as_vector([ ufl.sin(kk*pi*x[0]) * ufl.sinh( ufl.sqrt(nn**2-kk**2)*x[1] )/ ufl.sqrt(nn**2-kk**2) ,
                            ufl.sin(kk*pi*x[0]) *  ufl.sinh( ufl.sqrt(nn**2-kk**2)*x[1] )/ ufl.sqrt(nn**2-kk**2)  ]  ) 
        return Hadamard_sol

    else:
        print("I do not know this solution.")
        return None 


def SolveProblem(problem,msh,refsol,order=1,pgamma=1e-5,palpha=1e-5,add_bc=False,export_VTK=False,rhs=None,mu_Ind=None,perturb_order=None): 
    
    h = ufl.CellDiameter(msh)
    n = ufl.FacetNormal(msh)
    x = ufl.SpatialCoordinate(msh)
    ue = refsol(x)
    
    fe  = ufl.VectorElement('CG', msh.ufl_cell(), order)
    mel = ufl.MixedElement([fe,fe])
    VW = fem.FunctionSpace(msh,mel)
    ndof = VW.sub(0).dofmap.index_map.size_global * VW.sub(0).dofmap.index_map_bs 
    u,z = ufl.TrialFunctions(VW)
    v,w = ufl.TestFunctions(VW)

    Q_ind = fem.FunctionSpace(msh, ("DG", 0)) 
    omega_ind = fem.Function(Q_ind)
    B_ind = fem.Function(Q_ind)
    B_ind.interpolate(problem.B_Ind)
    omega_ind.interpolate(problem.omega_Ind)
    if mu_Ind: 
        mu = fem.Function(Q_ind)
        mu.interpolate(mu_Ind)
        problem.mu = mu 
    if problem.plus_Ind:
        plus_ind = fem.Function(Q_ind)
        plus_ind.interpolate(problem.plus_Ind)
        minus_ind = fem.Function(Q_ind)
        minus_ind.interpolate(problem.minus_Ind)

    def sigma(u):
        return 2*problem.mu*epsilon(u) + problem.lam * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) 
    Lu = lambda u : -ufl.nabla_div(sigma(u)) - problem.rho *u
    if rhs:
        f = rhs(x)
    else:
        f = Lu(ue)

    fdim = msh.topology.dim - 1
    V0, submap = VW.sub(0).collapse()
    W0, submap = VW.sub(1).collapse()
    # determine boundary DOFs
    boundary_dofs0 = fem.locate_dofs_geometrical((VW.sub(0),V0), problem.boundary_indicator )
    boundary_dofs1 = fem.locate_dofs_geometrical((VW.sub(1),W0), problem.boundary_indicator )
    
    u_D = fem.Function(V0)
    u_D.x.array[:] = 0.0
    bc = fem.dirichletbc(u_D, boundary_dofs1, VW.sub(1))
    bcs = [bc] 
    if add_bc:
        ue_h = fem.Function(V0)
        u_expr = fem.Expression(ue, V0.element.interpolation_points)
        ue_h.interpolate(u_expr)
        bc0 = fem.dirichletbc(ue_h, boundary_dofs0, VW.sub(0))
        bcs.append(bc0)
        if export_VTK:
            with io.XDMFFile(msh.comm, "u-jump.xdmf", "w") as xdmf:
                ue_h.name ="ujump"
                xdmf.write_mesh(msh)
                xdmf.write_function(ue_h)

    a = omega_ind * ufl.inner(u,v) * ufl.dx
    a += pgamma*0.5*(h('+')+h('-'))*ufl.inner(ufl.jump(sigma(u),n),ufl.jump(sigma(v),n))*ufl.dS
    a += pgamma * h**2 * ufl.inner(Lu(u),Lu(v)) * ufl.dx 
    a += palpha * h**(2*order) * ufl.inner(u,v) * ufl.dx
    a += ufl.inner(sigma(v),epsilon(z)) * ufl.dx - problem.rho * ufl.inner(v,z) * ufl.dx
    a -= ufl.inner( ufl.nabla_grad(z), ufl.nabla_grad(w)) * ufl.dx 
    a += ufl.inner(sigma(u),epsilon(w)) * ufl.dx - problem.rho * ufl.inner(u,w) * ufl.dx

    if perturb_order!=None:
        delta_u = fem.Function(V0)
        delta_u.x.array[:] = np.random.rand(len(delta_u.x.array) )[:]
        L2_delta_u = fem.form( omega_ind*ufl.inner(delta_u, delta_u) * ufl.dx)
        L2_delta_u_local = fem.assemble_scalar(L2_delta_u) 
        L2_norm_delta_u = np.sqrt(msh.comm.allreduce( L2_delta_u_local , op=MPI.SUM)) 
        delta_u.x.array[:] *= 1/L2_norm_delta_u
        #print("delta_u.x.array[:] =",delta_u.x.array[:])   
        delta_f = fem.Function(V0)
        delta_f.x.array[:] = np.random.rand(len(delta_f.x.array) )[:]
        L2_delta_f = fem.form( ufl.inner(delta_f, delta_f) * ufl.dx)
        L2_delta_f_local = fem.assemble_scalar(L2_delta_f) 
        L2_norm_delta_f = np.sqrt(msh.comm.allreduce( L2_delta_f_local , op=MPI.SUM)) 
        delta_f.x.array[:] *= 1/L2_norm_delta_f
        L = ufl.inner(f+h**(perturb_order)*delta_f, w) * ufl.dx + omega_ind * ufl.inner(ue+h**(perturb_order)*delta_u,v) * ufl.dx  + pgamma * h**2 * ufl.inner(f,Lu(v)) * ufl.dx 
    else:
        L = ufl.inner(f, w) * ufl.dx + omega_ind * ufl.inner(ue,v) * ufl.dx  + pgamma * h**2 * ufl.inner(f,Lu(v)) * ufl.dx 

    prob = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}) 
    sol = prob.solve()
    uh,zh = sol.split() 

    if problem.plus_Ind:
        #print("Measuring subdomain error separately.")
        L2_error_upper =  fem.form(  plus_ind * (B_ind)*ufl.inner(uh - ue, uh - ue) * ufl.dx)
        L2_norm_upper = fem.form( plus_ind * (B_ind)*ufl.inner(ue, ue) * ufl.dx)
        L2_error_lower =  fem.form(  minus_ind * (B_ind)*ufl.inner(uh - ue, uh - ue) * ufl.dx)
        L2_norm_lower = fem.form( minus_ind * (B_ind)*ufl.inner(ue, ue) * ufl.dx)
        L2_local_upper = fem.assemble_scalar(L2_norm_upper)
        L2_local_lower = fem.assemble_scalar(L2_norm_lower)
        error_local_upper = fem.assemble_scalar(L2_error_upper)
        error_local_lower = fem.assemble_scalar(L2_error_lower)
        error_L2 =  np.sqrt( msh.comm.allreduce(error_local_upper, op=MPI.SUM) + msh.comm.allreduce(error_local_lower, op=MPI.SUM) ) / np.sqrt( msh.comm.allreduce(L2_local_upper, op=MPI.SUM)+ msh.comm.allreduce(L2_local_lower, op=MPI.SUM)  ) 
        print("ndof = {0}, error_L2 = {1} ".format(ndof,error_L2)) 
    else:
        L2_error = fem.form( B_ind*ufl.inner(uh - ue, uh - ue) * ufl.dx)
        error_local = fem.assemble_scalar(L2_error)
        L2_norm = fem.form( B_ind*ufl.inner(ue, ue) * ufl.dx)
        
        #diff = ufl.nabla_grad(ue-uh)
        #ue_grad = ufl.nabla_grad(ue) 
        #L2_error = fem.form( B_ind*ufl.inner( diff, diff) * ufl.dx)
        #error_local = fem.assemble_scalar(L2_error)
        #L2_norm = fem.form( B_ind*ufl.inner(ue_grad, ue_grad) * ufl.dx)
        
        L2_local = fem.assemble_scalar(L2_norm)
        error_L2 = np.sqrt(msh.comm.allreduce(error_local, op=MPI.SUM)) / np.sqrt(msh.comm.allreduce(L2_local, op=MPI.SUM)) 
        print("ndof = {0}, error_L2 = {1}".format(ndof,error_L2))

    
    if export_VTK:
        with io.XDMFFile(msh.comm, "deformation.xdmf", "w") as xdmf:
            uh.name ="uh"
            zh.name ="zh"
            xdmf.write_mesh(msh)
            uh.name = "Deformation"
            xdmf.write_function(uh)

    return error_L2,ndof 

import matplotlib.pyplot as plt 
plt.rc('legend',fontsize=14)
plt.rc('axes',titlesize=14)
plt.rc('axes',labelsize=14)
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)


def RunProblemConvexGaussian(kk,perturb_theta=None):
    orders = [1,2,3] 
    ls_mesh = get_mesh_hierarchy(5)
    refsol = get_reference_sol("gaussian",kk=kk)
    elastic_convex.rho = kk**2
    elastic_convex.mu = 1.0
    elastic_convex.lam = 1.25

    for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5),ScalarType(5e-3)], [ ScalarType(1e-3),ScalarType(1e-1)] ):
    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5),ScalarType(5e-1)], [ ScalarType(5e-3),ScalarType(5e-1)] ):
        
        print("Considering {0} problem".format(problem_type))
        l2_errors_order = { }
        eoc_order = { }
        h_order = { }
        for order in orders:
            l2_errors = [ ]
            ndofs = [] 
            for msh in ls_mesh[:-order]:
                if perturb_theta!=None:
                    l2_error, ndof = SolveProblem(problem=elastic_convex,msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,perturb_order=order+perturb_theta)
                else:
                    l2_error, ndof = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False)
                l2_errors.append(l2_error)
                ndofs.append(ndof)

            eoc = [ log(l2_errors[i-1]/l2_errors[i])/log(2) for i in range(1,len(l2_errors))]
            print("eoc = ", eoc)
            ndofs = np.array(ndofs) 
            h_mesh = order/ndofs**(1/2)
            idx_start = 2 
            rate_estimate, _ = np.polyfit(np.log(h_mesh)[idx_start:] , np.log(l2_errors)[idx_start:], 1)
        
            l2_errors_order[order] =  l2_errors 
            h_order[order] = h_mesh
            eoc_order[order] = round(eoc[-1],2)
        
        for order in [1,2,3]: 
            plt.loglog(h_order[order], l2_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
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

        plt.xlabel("h")
        plt.ylabel("L2-error")
        plt.legend()
        plt.savefig("L2-error-convex-Gaussian-{0}-k{1}.png".format(problem_type,kk),transparent=True,dpi=200)
        #plt.title("L2-error")
        plt.show()

def RunProblemConvexOscillatory(kk):
    orders = [1,2,3] 
    ls_mesh = get_mesh_hierarchy(6,init_h_scale=1.0)
    refsol = get_reference_sol("oscillatory",kk=kk)
    elastic_convex.rho = kk**2
    elastic_convex.mu = 1.0
    elastic_convex.lam = 1.25

    for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5),ScalarType(5e-2)], [ ScalarType(1e-3),ScalarType(1e-5)] ):
    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5),ScalarType(5e-2)], [ ScalarType(1e-3),ScalarType(1e+5)] ):
        
        print("Considering {0} problem".format(problem_type))
        l2_errors_order = { }
        eoc_order = { }
        h_order = { }
        for order in orders:
            l2_errors = [ ]
            ndofs = [] 
            for msh in ls_mesh[:-order]:
                l2_error, ndof = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False)
                l2_errors.append(l2_error)
                ndofs.append(ndof)

            eoc = [ log(l2_errors[i-1]/l2_errors[i])/log(2) for i in range(1,len(l2_errors))]
            print("eoc = ", eoc)
            ndofs = np.array(ndofs) 
            h_mesh = order/ndofs**(1/2)
            idx_start = 2 
            rate_estimate, _ = np.polyfit(np.log(h_mesh)[idx_start:] , np.log(l2_errors)[idx_start:], 1)
        
            l2_errors_order[order] =  l2_errors 
            h_order[order] = h_mesh
            eoc_order[order] = round(eoc[-1],2)
        
        for order in [1,2,3]: 
            plt.loglog(h_order[order], l2_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
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
        plt.savefig("L2-error-convex-oscillatory-{0}-k{1}.png".format(problem_type,kk),transparent=True,dpi=200)
        #plt.title("L2-error")
        plt.show()

def RunProblemConvexOscillatoryKscaling():
    orders = [1,2,3] 
    ls_mesh = get_mesh_hierarchy(6,init_h_scale=1.0)
    
    elastic_convex.mu = 1.0
    elastic_convex.lam = 1.25

    msh = ls_mesh[3]
    add_bc = True
    problem_type = "well-posed"
    pgamma = ScalarType(1e-5)
    palpha = ScalarType(1e-3)
    #kks = np.linspace(1,20,6)
    #kks = [1+2*j for j in range(2)]
    kks = [1+2*j for j in range(8)]
    kks_np = np.array(kks)

    l2_errors_order = { }
    for order in orders:
        l2_errors = [] 
        for kk in kks:
            print("kk = {0}".format(kk)) 
            elastic_convex.rho = kk**2
            refsol = get_reference_sol("oscillatory",kk=kk)
            l2_error, ndof = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False)
            l2_errors.append(l2_error)
        l2_errors_order[order] = l2_errors


    for order,lstyle in zip([1,2,3],['solid','dashed','dotted']):
        plt.loglog(kks_np, l2_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
        tmp_str = "$\mathcal{{O}}(k^{0})$".format(order+1)
        plt.loglog(kks_np, 1.35*l2_errors_order[order][0]*(kks_np**(order+1))/( kks_np[0]**(order+1)) ,label=tmp_str,linestyle=lstyle,color='gray')
    
    plt.xlabel("$k$")
    plt.ylabel("L2-error")
    plt.legend()
    plt.savefig("L2-error-k.png",transparent=True,dpi=200)
    #plt.title("L2-error")
    plt.show()

def RunProblemConvexGaussianKscaling():
    orders = [1,2,3] 
    ls_mesh = get_mesh_hierarchy(6,init_h_scale=1.0)
    
    elastic_convex.mu = 1.0
    elastic_convex.lam = 1.25

    msh = ls_mesh[3]
    add_bc = True
    problem_type = "well-posed"
    pgamma = ScalarType(5e-3)
    palpha = ScalarType(1e-1)
    #kks = np.linspace(1,20,6)
    #kks = [1+2*j for j in range(3)]
    kks = [1+2*j for j in range(8)]
    kks_np = np.array(kks)

    l2_errors_order = { }
    for order in orders:
        l2_errors = [] 
        for kk in kks:
            print("kk = {0}".format(kk)) 
            elastic_convex.rho = kk**2
            refsol = get_reference_sol("gaussian",kk=kk)
            l2_error, ndof = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False)
            l2_errors.append(l2_error)
        l2_errors_order[order] = l2_errors


    for order,lstyle in zip([1,2,3],['solid','dashed','dotted']):
        plt.loglog(kks_np, l2_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
        #tmp_str = "$\mathcal{{O}}(k^{0})$".format(order+1)
        #plt.loglog(kks_np, 1.35*l2_errors_order[order][0]*(kks_np**(order+1))/( kks_np[0]**(order+1)) ,label=tmp_str,linestyle=lstyle,color='gray')

    plt.loglog(kks_np, 1.35*l2_errors_order[3][0]*(kks_np)/( kks_np[0]) ,label="$\mathcal{O}(k)$",linestyle=lstyle,color='gray')
    plt.xlabel("$k$")
    plt.ylabel("L2-error")
    plt.legend()
    plt.savefig("L2-error-k-Gaussian.png",transparent=True,dpi=200)
    #plt.title("L2-error")
    plt.show()


def RunProblemNonConvexOscillatory(kk,apgamma=1e-5,apalpha=1e5):
    orders = [1,2,3] 
    ls_mesh = get_mesh_hierarchy_nonconvex(6)
    refsol = get_reference_sol("oscillatory",kk=kk)
    elastic_nonconvex.rho = kk**2
    elastic_nonconvex.mu = 1.0
    elastic_nonconvex.lam = 1.25

    for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(apgamma),ScalarType(apgamma)], [ ScalarType(apalpha),ScalarType(apalpha)] ):
        
        print("Considering {0} problem".format(problem_type))
        l2_errors_order = { }
        eoc_order = { }
        h_order = { }
        for order in orders:
            l2_errors = [ ]
            ndofs = [] 
            for msh in ls_mesh[:-order]:
                l2_error, ndof = SolveProblem(problem = elastic_nonconvex, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False)
                l2_errors.append(l2_error)
                ndofs.append(ndof)

            eoc = [ log(l2_errors[i-1]/l2_errors[i])/log(2) for i in range(1,len(l2_errors))]
            print("eoc = ", eoc)
            ndofs = np.array(ndofs) 
            h_mesh = order/ndofs**(1/2)
            idx_start = 2 
            rate_estimate, _ = np.polyfit(np.log(h_mesh)[idx_start:] , np.log(l2_errors)[idx_start:], 1)
        
            l2_errors_order[order] =  l2_errors 
            h_order[order] = h_mesh
            eoc_order[order] = round(eoc[-1],2)
        
        for order in [1,2,3]: 
            plt.loglog(h_order[order], l2_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
            #tmp_str += ",eoc={:.2f}".format(eoc_order[order])
        if problem_type == "well-posed":
            for order,lstyle in zip([1,2,3],['solid','dashed','dotted']): 
                tmp_str = "$\mathcal{{O}}(h^{0})$".format(order+1)
                plt.loglog(h_order[order], l2_errors_order[order][0]*(h_order[order]**(order+1))/( h_order[order][0]**(order+1)) ,label=tmp_str,linestyle=lstyle,color='gray')
        if problem_type == "ill-posed":
            for order,lstyle in zip([1],['solid','dashed','dotted']):
                tmp_str = "$\mathcal{{O}}(h^{0})$".format(order)
                plt.loglog(h_order[order], l2_errors_order[order+2][0]*(h_order[order]**(order))/( h_order[order+2][0]**(order)) ,label=tmp_str,linestyle=lstyle,color='gray')
                #aeoc = eoc_order[order]
                #pow_a = "{:.2f}".format(order)
                #tmp_str = "eoc = $".format(pow_a)
                #plt.loglog(h_order[order], l2_errors_order[order][0]*(h_order[order]**aeoc)/( h_order[order][0]**aeoc) ,label=tmp_str,linestyle=lstyle,color='gray')

        plt.xlabel("~h")
        plt.ylabel("L2-error")
        plt.legend()
        plt.savefig("L2-error-nonconvex-oscillatory-{0}-k{1}.png".format(problem_type,kk),transparent=True,dpi=200)
        #plt.title("L2-error")
        plt.show()

def RunProblemNonConvexGaussian(kk,apgamma=1e-5,apalpha=1e5,perturb_theta=None):
    orders = [1,2,3] 
    ls_mesh = get_mesh_hierarchy_nonconvex(6)
    refsol = get_reference_sol("gaussian",kk=kk)
    elastic_nonconvex.rho = kk**2
    elastic_nonconvex.mu = 1.0
    elastic_nonconvex.lam = 1.25

    for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(apgamma),ScalarType(apgamma)], [ ScalarType(apalpha),ScalarType(apalpha)] ):
        
        print("Considering {0} problem".format(problem_type))
        l2_errors_order = { }
        eoc_order = { }
        h_order = { }
        for order in orders:
            l2_errors = [ ]
            ndofs = [] 
            for msh in ls_mesh[:-order]:
                if perturb_theta != None:
                    l2_error, ndof = SolveProblem(problem=elastic_nonconvex,msh=msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,perturb_order=order+perturb_theta)
                else:
                    l2_error, ndof = SolveProblem(problem = elastic_nonconvex, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False)
                l2_errors.append(l2_error)
                ndofs.append(ndof)

            eoc = [ log(l2_errors[i-1]/l2_errors[i])/log(2) for i in range(1,len(l2_errors))]
            print("eoc = ", eoc)
            ndofs = np.array(ndofs) 
            h_mesh = order/ndofs**(1/2)
            idx_start = 2 
            rate_estimate, _ = np.polyfit(np.log(h_mesh)[idx_start:] , np.log(l2_errors)[idx_start:], 1)
        
            l2_errors_order[order] =  l2_errors 
            h_order[order] = h_mesh
            eoc_order[order] = round(eoc[-1],2)
        
        for order in [1,2,3]: 
            plt.loglog(h_order[order], l2_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
            #tmp_str += ",eoc={:.2f}".format(eoc_order[order])
        if problem_type == "well-posed":
            for order,lstyle in zip([1,2,3],['solid','dashed','dotted']): 
                tmp_str = "$\mathcal{{O}}(h^{0})$".format(order+1)
                plt.loglog(h_order[order], l2_errors_order[order][0]*(h_order[order]**(order+1))/( h_order[order][0]**(order+1)) ,label=tmp_str,linestyle=lstyle,color='gray')
        if problem_type == "ill-posed":
            for order,lstyle in zip([1],['solid','dashed','dotted']):
                tmp_str = "$\mathcal{{O}}(h^{0})$".format(order)
                plt.loglog(h_order[order], l2_errors_order[order+2][0]*(h_order[order]**(order))/( h_order[order+2][0]**(order)) ,label=tmp_str,linestyle=lstyle,color='gray')
                #aeoc = eoc_order[order]
                #pow_a = "{:.2f}".format(order)
                #tmp_str = "eoc = $".format(pow_a)
                #plt.loglog(h_order[order], l2_errors_order[order][0]*(h_order[order]**aeoc)/( h_order[order][0]**aeoc) ,label=tmp_str,linestyle=lstyle,color='gray')

        plt.xlabel("~h")
        plt.ylabel("L2-error")
        plt.legend()
        plt.savefig("L2-error-nonconvex-Gaussian-{0}-k{1}.png".format(problem_type,kk),transparent=True,dpi=200)
        #plt.title("L2-error")
        plt.show()


def RunProblemJump(kk=1,apgamma=1e-1,apalpha=1e-1):

    orders = [1,2,3] 
    #order = 3
    elastic_convex.rho = kk**2
    elastic_convex.lam = 1.25
    eta = 0.6
    ls_mesh = get_mesh_hierarchy_fitted_disc(6,eta=eta)
    #mu_plus = 2
    mu_plus = 8
    mu_minus = 1 
    refsol,rhs = get_reference_sol(type_str="jump",kk=kk,eta=eta,mu_plus=mu_plus,mu_minus=mu_minus,lam=elastic_convex.lam)
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

    for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(apgamma),ScalarType(apgamma)], [ ScalarType(apgamma),ScalarType(apgamma)] ):
        
        print("Considering {0} problem".format(problem_type))
        l2_errors_order = { }
        eoc_order = { }
        h_order = { }
        for order in orders:
            l2_errors = [ ]
            ndofs = [] 
            for msh in ls_mesh[:-order]:
                l2_error, ndof = SolveProblem(problem=elastic_convex,msh=msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=True,rhs=rhs,mu_Ind=mu_Ind)
                l2_errors.append(l2_error)
                ndofs.append(ndof)

            eoc = [ log(l2_errors[i-1]/l2_errors[i])/log(2) for i in range(1,len(l2_errors))]
            print("eoc = ", eoc)
            ndofs = np.array(ndofs) 
            h_mesh = order/ndofs**(1/2)
            idx_start = 2 
            rate_estimate, _ = np.polyfit(np.log(h_mesh)[idx_start:] , np.log(l2_errors)[idx_start:], 1)
        
            l2_errors_order[order] =  l2_errors 
            h_order[order] = h_mesh
            eoc_order[order] = round(eoc[-1],2)
        
        for order in [1,2,3]: 
            plt.loglog(h_order[order], l2_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
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

    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5),ScalarType(5e-3)], [ ScalarType(1e-3),ScalarType(1e-1)] ):
    #    print("Considering {0} problem".format(problem_type))
    #    l2_errors = [ ]
    #    ndofs = [] 
    #    for msh in ls_mesh:
    #        l2_error, ndof = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False,rhs=rhs,mu_Ind = mu_Ind)
    #        l2_errors.append(l2_error)
    #        ndofs.append(ndof)
    #
    #    eoc = [ log(l2_errors[i-1]/l2_errors[i])/log(2) for i in range(1,len(l2_errors))]
    #    print("eoc = ", eoc)
    #    ndofs = np.array(ndofs) 
    #    h_mesh = order/ndofs**(1/2)
    #    idx_start = 2 
    #    rate_estimate, _ = np.polyfit(np.log(h_mesh)[idx_start:] , np.log(l2_errors)[idx_start:], 1)
    #    print("(Relative) L2-error rate estimate {:.2f}".format(rate_estimate))
    #    plt.loglog(h_mesh, l2_errors,'-x',label="L2-error")
    #    plt.loglog(h_mesh,(l2_errors[0]/h_mesh[0])*h_mesh,label="linear",linestyle='dashed',color='gray')
    #    plt.loglog(h_mesh,(l2_errors[0]/h_mesh[0]**rate_estimate)*h_mesh**rate_estimate,label="eoc={:.2f}".format(rate_estimate),linestyle='solid',color='gray')
    #    plt.xlabel("~h")
    #    plt.ylabel("L2-error")
    #    plt.legend()
    #    plt.savefig("L2-error-elastodynamics-P{0}-{1}.png".format(order,problem_type),transparent=True,dpi=200)
    #    plt.show()


def RunProblemConvexHadamard(kk,nn=5):
    orders = [1,2,3] 
    ls_mesh = get_mesh_hierarchy(6)
    refsol = get_reference_sol("Hadamard",kk=kk,nn=nn)
    elastic_convex.rho = kk**2
    elastic_convex.mu = 1.0
    elastic_convex.lam = 1.25

    for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5),ScalarType(5e-2)], [ ScalarType(1e-3),ScalarType(1e-5)] ):
    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5),ScalarType(5e-2)], [ ScalarType(1e-3),ScalarType(1e+5)] ):
        
        print("Considering {0} problem".format(problem_type))
        l2_errors_order = { }
        eoc_order = { }
        h_order = { }
        for order in orders:
            l2_errors = [ ]
            ndofs = [] 
            for msh in ls_mesh[:-order]:
                l2_error, ndof = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False)
                l2_errors.append(l2_error)
                ndofs.append(ndof)

            eoc = [ log(l2_errors[i-1]/l2_errors[i])/log(2) for i in range(1,len(l2_errors))]
            print("eoc = ", eoc)
            ndofs = np.array(ndofs) 
            h_mesh = order/ndofs**(1/2)
            idx_start = 2 
            rate_estimate, _ = np.polyfit(np.log(h_mesh)[idx_start:] , np.log(l2_errors)[idx_start:], 1)
        
            l2_errors_order[order] =  l2_errors 
            h_order[order] = h_mesh
            eoc_order[order] = round(eoc[-1],2)
        
        for order in [1,2,3]: 
            plt.loglog(h_order[order], l2_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
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
        plt.savefig("L2-error-convex-oscillatory-{0}-k{1}.png".format(problem_type,kk),transparent=True,dpi=200)
        #plt.title("L2-error")
        plt.show()

# Runs for draft
RunProblemConvexGaussian(kk=1)
#RunProblemConvexGaussian(kk=6)
#RunProblemConvexOscillatory(kk=1)
#RunProblemConvexOscillatory(kk=6)



#RunProblemNonConvexOscillatory(kk=1,apgamma=1e-2,apalpha=1e0)
#RunProblemNonConvexOscillatory(kk=4,apgamma=1e-2,apalpha=1e0)
#RunProblemNonConvexOscillatory(kk=6,apgamma=1e-2,apalpha=1e0)

#RunProblemNonConvexOscillatory(kk=4,apgamma=1e-4,apalpha=1e0)
#RunProblemNonConvexOscillatory(kk=4,apgamma=5e-2,apalpha=1e0)
#RunProblemNonConvexOscillatory(kk=6,apgamma=1e-1,apalpha=1e3)
#RunProblemConvexGaussianKscaling()
#RunProblemConvexOscillatoryKscaling()

#RunProblemNonConvexGaussian(kk=1,apgamma=1e-4,apalpha=1e0)
#RunProblemNonConvexGaussian(kk=4,apgamma=1e-4,apalpha=1e0)
#RunProblemNonConvexGaussian(kk=6,apgamma=1e-4,apalpha=1e0)

#RunProblemJump(kk=8,apgamma=1e-2,apalpha=1e-0)
#RunProblemJump(kk=1,apgamma=1e-3,apalpha=1e-0)
#RunProblemJump(kk=8,apgamma=5e-2,apalpha=1e-0)

#RunProblemConvexOscillatory(kk=10)

#RunProblemConvexHadamard(kk=1,nn=5)
#RunProblemConvexHadamard(kk=10,nn=11)

#RunProblemConvexGaussian(kk=1)
#RunProblemConvexGaussian(kk=6,perturb_theta=-2)
#RunProblemNonConvexGaussian(kk=6,apgamma=1e-4,apalpha=1e0,perturb_theta=-2)

#RunProblemConvexOscillatory(kk=1,perturb=True)

'''
def RunProblemConvexOscillatory(kk):
    order = 2 
    ls_mesh = get_mesh_hierarchy(3)
    #ls_mesh = get_mesh_hierarchy_nonconvex(4)
    refsol = get_reference_sol("oscillatory",kk=kk)
    #refsol = get_reference_sol("gaussian",kk=kk)
    #elastic_nonconvex.rho = kk**2
    elastic_convex.rho = kk**2
    elastic_convex.mu = 1.0
    elastic_convex.lam = 1.25

    for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5),ScalarType(5e-2)], [ ScalarType(1e-3),ScalarType(1e-1)] ):
    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5),ScalarType(5e-1)], [ ScalarType(5e-3),ScalarType(5e-1)] ):
        print("Considering {0} problem".format(problem_type))
        l2_errors = [ ]
        ndofs = [] 
        for msh in ls_mesh:
            #l2_error, ndof = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False,rhs=rhs,mu_Ind = mu_Ind)
            l2_error, ndof = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False)
            l2_errors.append(l2_error)
            ndofs.append(ndof)

        eoc = [ log(l2_errors[i-1]/l2_errors[i])/log(2) for i in range(1,len(l2_errors))]
        print("eoc = ", eoc)
        ndofs = np.array(ndofs) 
        h_mesh = order/ndofs**(1/2)
        idx_start = 2 
        rate_estimate, _ = np.polyfit(np.log(h_mesh)[idx_start:] , np.log(l2_errors)[idx_start:], 1)
        print("(Relative) L2-error rate estimate {:.2f}".format(rate_estimate))
        plt.loglog(h_mesh, l2_errors,'-x',label="L2-error")
        plt.loglog(h_mesh,(l2_errors[0]/h_mesh[0])*h_mesh,label="linear",linestyle='dashed',color='gray')
        #plt.loglog( h_mesh,(l2_errors[0]/h_mesh[0]**2)*h_mesh**2,label="quadratic",linestyle="dotted",color='gray')
        plt.loglog(h_mesh,(l2_errors[0]/h_mesh[0]**rate_estimate)*h_mesh**rate_estimate,label="eoc={:.2f}".format(rate_estimate),linestyle='solid',color='gray')
        plt.xlabel("~h")
        plt.ylabel("L2-error")
        plt.legend()
        plt.savefig("L2-error-elastodynamics-P{0}-{1}.png".format(order,problem_type),transparent=True,dpi=200)
        plt.show()
'''
