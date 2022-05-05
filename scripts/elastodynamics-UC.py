'''
Unique continuation for elastodynamics. 
'''

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, plot, io
from petsc4py.PETSc import ScalarType
from math import pi,log
from meshes import get_mesh_hierarchy, get_mesh_hierarchy_nonconvex,get_mesh_hierarchy_fitted_disc,get_mesh_convex,create_initial_mesh_convex
from problems import elastic_convex, elastic_nonconvex

#ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag, metadata=metadata)
#dS = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag, metadata=metadata)

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


def SolveProblem(problem,msh,refsol,order=1,pgamma=1e-5,palpha=1e-5,add_bc=False,export_VTK=False,rhs=None,mu_Ind=None,perturb_order=None,pGLS=None): 

    error_dict = {"L2-error-u-uh-B": None,
                  "H1-semi-error-u-uh-B": None,
                  "H1-semi-error-u-uh-B-absolute": None,
                  "H1-semi-norm-zh-Omega": None,
                  "Jump-uh-Ph(u)": None,
                  "GLS-uh-Ph(u)-Omega": None,
                  "Tikh-uh-Ph(u)-Omega": None,
                  "L2-error-uh-Ph(u)-omega": None,
                  "s-norm": None,
                  "L2-error-u-uh-B-plus": None,
                  "L2-error-u-uh-B-minus": None,
                  "ndof": None,
                  "hmax": None,
                 }

    if pGLS == None:
        pGLS = pgamma 
    #metadata = {"quadrature_degree": 10}
    #dx = ufl.Measure("dx", domain=msh, metadata=metadata)
    dx = ufl.dx

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
    V0, V0_to_VW = VW.sub(0).collapse()
    W0, submap = VW.sub(1).collapse()
    # determine boundary DOFs
    boundary_dofs0 = fem.locate_dofs_geometrical((VW.sub(0),V0), problem.boundary_indicator )
    boundary_dofs1 = fem.locate_dofs_geometrical((VW.sub(1),W0), problem.boundary_indicator )
    
    u_D = fem.Function(V0)
    u_D.x.array[:] = 0.0
    bc = fem.dirichletbc(u_D, boundary_dofs1, VW.sub(1))
    bcs = [bc] 
    ue_h = fem.Function(V0)
    u_expr = fem.Expression(ue, V0.element.interpolation_points)
    ue_h.interpolate(u_expr)
    #print(" ue_h.x.array[:] = ", ue_h.x.array[:]  )
    Phu = ue_h
    if add_bc:
        bc0 = fem.dirichletbc(ue_h, boundary_dofs0, VW.sub(0))
        bcs.append(bc0)
        if export_VTK:
            with io.XDMFFile(msh.comm, "u-jump.xdmf", "w") as xdmf:
                ue_h.name ="ujump"
                xdmf.write_mesh(msh)
                xdmf.write_function(ue_h)
    
    #px = pgamma*1000
    #px = pgamma
    a = omega_ind * ufl.inner(u,v) * dx
    #a +=  pgamma*order**2*0.5*(h('+')+h('-'))*ufl.inner(ufl.jump(sigma(u),n),ufl.jump(sigma(v),n))*ufl.dS
    a +=  pgamma*0.5*(h('+')+h('-'))*ufl.inner(ufl.jump(sigma(u),n),ufl.jump(sigma(v),n))*ufl.dS
    #a += pgamma * h**2 * ufl.inner(Lu(u),Lu(v)) * ufl.dx 
    a += pGLS * h**2 * ufl.inner(Lu(u),Lu(v)) * dx 
    a += palpha * h**(2*order) * ufl.inner(u,v) * dx
    #a += palpha * h**(2*order) * ufl.inner(ufl.nabla_grad(u),ufl.nabla_grad(v)) * ufl.dx
    a += ufl.inner(sigma(v),epsilon(z)) * ufl.dx - problem.rho * ufl.inner(v,z) * dx
    a -= ufl.inner( ufl.nabla_grad(z), ufl.nabla_grad(w)) * dx 
    a += ufl.inner(sigma(u),epsilon(w)) * ufl.dx - problem.rho * ufl.inner(u,w) * dx

    if perturb_order!=None:
        delta_u = fem.Function(V0)
        delta_u.x.array[:] = np.random.rand(len(delta_u.x.array) )[:]
        L2_delta_u = fem.form( omega_ind*ufl.inner(delta_u, delta_u) * dx)
        L2_delta_u_local = fem.assemble_scalar(L2_delta_u) 
        L2_norm_delta_u = np.sqrt(msh.comm.allreduce( L2_delta_u_local , op=MPI.SUM)) 
        delta_u.x.array[:] *= 1/L2_norm_delta_u
        #print("delta_u.x.array[:] =",delta_u.x.array[:])   
        delta_f = fem.Function(V0)
        delta_f.x.array[:] = np.random.rand(len(delta_f.x.array) )[:]
        L2_delta_f = fem.form( ufl.inner(delta_f, delta_f) * dx)
        L2_delta_f_local = fem.assemble_scalar(L2_delta_f) 
        L2_norm_delta_f = np.sqrt(msh.comm.allreduce( L2_delta_f_local , op=MPI.SUM)) 
        delta_f.x.array[:] *= 1/L2_norm_delta_f
        L = ufl.inner(f+h**(perturb_order)*delta_f, w) * dx + omega_ind * ufl.inner(ue+h**(perturb_order)*delta_u,v) * dx  + pGLS * h**2 * ufl.inner(f,Lu(v)) * dx 
    else:
        #L = ufl.inner(f, w) * ufl.dx + omega_ind * ufl.inner(ue,v) * ufl.dx  + pgamma * h**2 * ufl.inner(f,Lu(v)) * ufl.dx 
        L = ufl.inner(f, w) * dx + omega_ind * ufl.inner(ue,v) * dx  + pGLS * h**2 * ufl.inner(f,Lu(v)) * dx 

    prob = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu","pc_factor_mat_solver_type":"superlu" }) 
    sol = prob.solve()
    uh,zh = sol.split() 
    
    if export_VTK:
        uex = fem.Function(V0)
        u_expr = fem.Expression(ue, V0.element.interpolation_points)
        uex.interpolate(u_expr)
        udiff = fem.Function(V0)
        udiff.x.array[:] = np.abs(uh.x.array[V0_to_VW] - uex.x.array)
        with io.XDMFFile(msh.comm, "deformation.xdmf", "w") as xdmf:
            uh.name ="uh"
            zh.name ="zh"
            xdmf.write_mesh(msh)
            uh.name = "Deformation"
            #xdmf.write_function(uh)
            xdmf.write_function(udiff)


    # Postprocessing: measure errors #
    
    # workaround for 4.,5. and 6. as we cannot use uh and Phu in form together (dof map)
    uh_V0 = fem.Function(V0)
    uh_V0.x.array[:] = uh.x.array[V0_to_VW] 
    
    # 1. L2-error-u-uh-B 
    L2_error_B = fem.form( B_ind*ufl.inner(uh - ue, uh - ue) * dx)
    error_local = fem.assemble_scalar(L2_error_B)
    L2_norm = fem.form( B_ind*ufl.inner(ue, ue) * dx) 
    L2_local = fem.assemble_scalar(L2_norm)
    error_L2 = np.sqrt(msh.comm.allreduce(error_local, op=MPI.SUM)) / np.sqrt(msh.comm.allreduce(L2_local, op=MPI.SUM)) 
    error_dict["L2-error-u-uh-B"] = error_L2 

    # 2. H1-semi-error-u-uh-B
    diff = ufl.nabla_grad(ue-uh)
    ue_grad = ufl.nabla_grad(ue)
    error_local_H1 = fem.form( B_ind*ufl.inner( diff, diff) * ufl.dx)
    H1_semi_norm_local = fem.form( B_ind*ufl.inner(ue_grad, ue_grad) * dx) 
    H1_semi_error = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error_local_H1), op=MPI.SUM))  
    error_dict["H1-semi-error-u-uh-B-absolute"] = H1_semi_error
    error_dict["H1-semi-error-u-uh-B"] = H1_semi_error/ np.sqrt(msh.comm.allreduce(fem.assemble_scalar(H1_semi_norm_local), op=MPI.SUM)) 

    # 3. H1-semi-norm-zh-Omega
    zh_grad = ufl.nabla_grad(zh)
    H1_semi_norm_zh_local = fem.form( ufl.inner(zh_grad, zh_grad) * dx) 
    H1_semi_norm_zh = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(H1_semi_norm_zh_local), op=MPI.SUM))
    error_dict["H1-semi-norm-zh-Omega"] = H1_semi_norm_zh 
    #print("ndof = {0}, error_L2 = {1}".format(ndof,error_L2))

    # 4. Jump-uh-Ph(u) 
    jump_error_local = fem.form( pgamma*0.5*(h('+')+h('-'))*ufl.inner(ufl.jump(sigma(uh_V0-Phu),n),ufl.jump(sigma(uh_V0-Phu),n))*ufl.dS )
    jump_error = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(jump_error_local), op=MPI.SUM))
    error_dict["Jump-uh-Ph(u)"] = jump_error 

    # 5. GLS-uh-Ph(u)-Omega
    #GLS_error_local = fem.form(pGLS *  ufl.inner( Lu(uh)-Lu(Phu), Lu(uh)-Lu(Phu) ) * dx)
    GLS_error_local = fem.form(pGLS * h**2 *  ufl.inner( Lu(uh_V0)-Lu(Phu), Lu(uh_V0) -Lu(Phu) ) * dx)
    #GLS_error_local = fem.form(pGLS *  ufl.inner( Lu(Phu)-Lu(ue), Lu(Phu)-Lu(ue) ) * dx)
    #GLS_error_local = fem.form(pGLS *  ufl.inner( Lu(uh)-Lu(ue), Lu(uh)-Lu(ue) ) * dx)
    GLS_error = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(GLS_error_local), op=MPI.SUM))
    error_dict["GLS-uh-Ph(u)-Omega"] = GLS_error 

    # 6. Tikh-uh-Ph(u)-Omega 
    Tikh_error_local = fem.form( palpha * h**(2*order) * ufl.inner(uh_V0-Phu,uh_V0-Phu) * dx )
    Tikh_error = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(Tikh_error_local), op=MPI.SUM))
    error_dict["Tikh-uh-Ph(u)-Omega"] = Tikh_error 

    # 7. L2-error-uh-Ph(u)-omega
    L2_error_omega_local = fem.form( omega_ind * ufl.inner(uh_V0 - Phu, uh_V0 - Phu) * dx ) 
    L2_error_omega = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(L2_error_omega_local), op=MPI.SUM))
    error_dict["L2-error-uh-Ph(u)-omega"] = L2_error_omega 

    # 8. s-norm 
    error_dict["s-norm"] = np.sqrt( error_dict["Jump-uh-Ph(u)"]**2 + error_dict["GLS-uh-Ph(u)-Omega"]**2 
                                    + error_dict["Tikh-uh-Ph(u)-Omega"]**2 + error_dict["L2-error-uh-Ph(u)-omega"]**2 
                                    + error_dict["H1-semi-norm-zh-Omega"]**2 )
    # 9. ndofs
    error_dict["ndof"] = ndof
    
    # 10. h (proportional)
    error_dict["hmax"] = order/np.array(ndof)**(1/2) 
    error_dict["ndof"] = 2*np.array(error_dict["ndof"])

    if problem.plus_Ind:
        #print("Measuring subdomain error separately.")
        L2_error_upper =  fem.form(  plus_ind * (B_ind)*ufl.inner(uh - ue, uh - ue) * dx)
        L2_norm_upper = fem.form( plus_ind * (B_ind)*ufl.inner(ue, ue) * dx)
        L2_error_lower =  fem.form(  minus_ind * (B_ind)*ufl.inner(uh - ue, uh - ue) * dx)
        L2_norm_lower = fem.form( minus_ind * (B_ind)*ufl.inner(ue, ue) * dx)
        L2_local_upper = fem.assemble_scalar(L2_norm_upper)
        L2_local_lower = fem.assemble_scalar(L2_norm_lower)
        error_local_upper = fem.assemble_scalar(L2_error_upper)
        error_local_lower = fem.assemble_scalar(L2_error_lower)
        error_rel_upper = np.sqrt( msh.comm.allreduce(error_local_upper, op=MPI.SUM)) / np.sqrt( msh.comm.allreduce(L2_local_upper, op=MPI.SUM) )  
        error_rel_lower = np.sqrt(  msh.comm.allreduce(error_local_lower, op=MPI.SUM) ) / np.sqrt( msh.comm.allreduce(L2_local_lower, op=MPI.SUM)    ) 
        error_L2 =  np.sqrt( msh.comm.allreduce(error_local_upper, op=MPI.SUM) + msh.comm.allreduce(error_local_lower, op=MPI.SUM) ) / np.sqrt( msh.comm.allreduce(L2_local_upper, op=MPI.SUM)+ msh.comm.allreduce(L2_local_lower, op=MPI.SUM)  ) 
        #print("ndof = {0}, error_upper = {1}, error_lower={2} ".format(ndof, error_rel_upper, error_rel_lower   )) 
        error_dict["L2-error-u-uh-B-plus"] = error_rel_upper  
        error_dict["L2-error-u-uh-B-minus"] = error_rel_lower 
        error_dict["L2-error-u-uh-B"] = error_L2 
    
    return error_dict
    #return  H1_semi_error,ndof 
    #return unnormalized_L2,ndof 
    #return  reliable_L2,ndof 
    #return error_L2,ndof 

import matplotlib.pyplot as plt 
plt.rc('legend',fontsize=14)
plt.rc('axes',titlesize=14)
plt.rc('axes',labelsize=14)
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)


def ConvergenceStudy(problem,ls_mesh,refsol,order=1,pgamma=1e-5,palpha=1e-5,add_bc=False,export_VTK=False,rhs=None,mu_Ind=None,perturb_theta=None,pGLS=None,name_str="dummy.txt"):
    errors = { "L2-error-u-uh-B": [],
               "H1-semi-error-u-uh-B": [],
               "H1-semi-error-u-uh-B-absolute": [],
               "H1-semi-norm-zh-Omega": [],
               "Jump-uh-Ph(u)": [],
               "GLS-uh-Ph(u)-Omega": [],
               "Tikh-uh-Ph(u)-Omega": [],
               "s-norm": [],
               "L2-error-uh-Ph(u)-omega": [],
               "L2-error-u-uh-B-plus": [],
               "L2-error-u-uh-B-minus": [],
               "ndof": [],
               "hmax":[],
               "reflvl":[]
             }

    for msh in ls_mesh:
        if perturb_theta!=None:
            errors_msh = SolveProblem(problem = problem, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,perturb_order=order+perturb_theta,pGLS=pGLS)
        else:
            errors_msh = SolveProblem(problem = problem, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=order==2,pGLS=pGLS)
        print("ndof = {0}, L2-error-u-uh-B = {1}".format(errors_msh["ndof"],errors_msh["L2-error-u-uh-B"])) 
        for error_type in errors_msh:
            errors[error_type].append(errors_msh[error_type])
    
    # 11 reflvl 
    errors["reflvl"] = range(len(errors["hmax"])) 
    
    #results = [np.array(errors_msh["ndof"]]
    #header_str = "ndofs "
    results = []
    header_str = ""
    for error_type in errors:
        if errors[error_type][0] != None: 
            print(errors[error_type]) 
            results.append(np.array(errors[error_type],dtype=float))
            header_str += "{0} ".format(error_type)
    np.savetxt(fname ="../data/{0}".format(name_str),
               X = np.transpose(results),
               header = header_str,
               comments = '')

    return errors 

def RunProblemConvexGaussian(kk,perturb_theta=None):
    orders = [1,2,3] 
    ls_mesh = get_mesh_hierarchy(5)
    refsol = get_reference_sol("gaussian",kk=kk)
    elastic_convex.rho = kk**2
    elastic_convex.mu = 1.0
    elastic_convex.lam = 1.25

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

def RunProblemConvexOscillatory(kk,perturb_theta=None):
    orders = [1,2,3] 
    ls_mesh = get_mesh_hierarchy(5)
    refsol = get_reference_sol("oscillatory",kk=kk)
    elastic_convex.rho = kk**2
    elastic_convex.mu = 1.0
    elastic_convex.lam = 1.25

    for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(3e-4)/kk**2,ScalarType(3e-4)/kk**2], [ ScalarType(1e-2),ScalarType(1e-2)]):
        print("Considering {0} problem".format(problem_type))
        for order in orders:
            name_str = "Convex-Oscillatory-{0}-k{1}-order{2}.dat".format(problem_type,kk,order)
            #print(name_str)
            print("Computing for order = {0}".format(order))
            errors_order = ConvergenceStudy(elastic_convex,ls_mesh[:-order],refsol,order=order,pgamma=pgamma/order**2,palpha=palpha,add_bc=add_bc,export_VTK=False,rhs=None,mu_Ind=None,perturb_theta=None,pGLS=1e-4/kk**4,name_str = name_str)
            print(errors_order)
            
            eoc = [ log(errors_order["L2-error-u-uh-B"][i-1]/errors_order["L2-error-u-uh-B"][i])/log(2) for i in range(1,len(errors_order["L2-error-u-uh-B"]))]
            print("l2-norm eoc = ", eoc)
            ndofs = np.array(errors_order["ndof"]) 
            h_order = order/ndofs**(1/2) 
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

def RunProblemNonConvexGaussian(kk,perturb_theta=None):
    
    orders = [1,2,3] 
    ls_mesh = get_mesh_hierarchy_nonconvex(5)
    refsol = get_reference_sol("gaussian",kk=kk)
    elastic_nonconvex.rho = kk**2
    elastic_nonconvex.mu = 1.0
    elastic_nonconvex.lam = 1.25


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

def RunProblemNonConvexOscillatory(kk,perturb_theta=None):
    
    orders = [1,2,3] 
    ls_mesh = get_mesh_hierarchy_nonconvex(5)
    refsol = get_reference_sol("oscillatory",kk=kk)
    elastic_nonconvex.rho = kk**2
    elastic_nonconvex.mu = 1.0
    elastic_nonconvex.lam = 1.25


    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5)/kk**2,ScalarType(1e-5)/kk**2], [ ScalarType(1e-2),ScalarType(1e-2)]):
    for add_bc,problem_type,pgamma,palpha in zip([False],["ill-posed"],[ScalarType(1e-5)/kk**2], [ ScalarType(1e-2)]):
    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-2),ScalarType(1e-2)], [ ScalarType(1e-2),ScalarType(1e-2)]):
        print("Considering {0} problem".format(problem_type))
        for order in orders:
            name_str = "Non-Convex-Oscillatory-{0}-k{1}-order{2}.dat".format(problem_type,kk,order)
            #print(name_str)
            print("Computing for order = {0}".format(order))
            errors_order = ConvergenceStudy(elastic_nonconvex,ls_mesh[:-order],refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False,rhs=None,mu_Ind=None,perturb_theta=None,pGLS=1e-4/kk**4,name_str = name_str)
            #errors_order = ConvergenceStudy(elastic_nonconvex,ls_mesh[:-order],refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False,rhs=None,mu_Ind=None,perturb_theta=None,pGLS=1e-1,name_str = name_str)
            print(errors_order)
            
            eoc = [ log(errors_order["L2-error-u-uh-B"][i-1]/errors_order["L2-error-u-uh-B"][i])/log(2) for i in range(1,len(errors_order["L2-error-u-uh-B"]))]
            print("eoc = ", eoc)
            ndofs = np.array(errors_order["ndof"]) 
            h_order = order/ndofs**(1/2) 
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


def RunProblemJump(kk=1,apgamma=1e-1,apalpha=1e-1): 
    eta = 0.6
    
    def omega_Ind_eta(x):
        
        values = np.zeros(x.shape[1],dtype=ScalarType)
        omega_coords = np.logical_or(  np.logical_and(  x[0] <= 0.1 , x[1] <=eta  ) , 
            np.logical_or(  np.logical_and( x[0] >= 0.9 , x[1] <= eta  ) , (x[1] <= 0.25)  )
            ) 
        #omega_coords = np.logical_or(   ( x[0] <= 0.1 )  , 
        #    np.logical_or(   (x[0] >= 0.9 ), (x[1] <= 0.25)  )
        #    ) 
        rest_coords = np.invert(omega_coords)
        values[omega_coords] = np.full(sum(omega_coords), 1.0)
        values[rest_coords] = np.full(sum(rest_coords), 0)
        return values

    elastic_convex.SetSubdomains(omega_Ind=omega_Ind_eta,B_Ind=elastic_convex.B_Ind)
    
    orders = [1,2,3] 
    #order = 3
    elastic_convex.rho = kk**2
    elastic_convex.lam = 1.25
    ls_mesh = get_mesh_hierarchy_fitted_disc(5,eta=eta)
    #mu_plus = 2
    mu_plus = 1
    mu_minus = 2
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

    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(apgamma),ScalarType(apgamma)], [ ScalarType(apgamma),ScalarType(apgamma)] ):
    for add_bc,problem_type,pgamma,palpha,pGLS in zip([False],["ill-posed"],[ ScalarType(apgamma)/ (kk**2) ] , [ ScalarType(apalpha)], [ ScalarType(1e-2/kk**4) ] ):
        
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
                errors = SolveProblem(problem=elastic_convex,msh=msh,refsol=refsol,order=order,pgamma=pgamma/order**2,palpha=palpha,add_bc=add_bc,export_VTK=order==3,rhs=rhs,mu_Ind=mu_Ind,pGLS=pGLS)
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
    
            for error_type,error_str in zip([ L2_error_B_minus, L2_error_B_plus ],["L2-error-u-uh-B-minus","L2-error-u-uh-B-plus"]):
                #print(error_str)
                eoc = [ log(error_type[i-1]/error_type[i])/log(2) for i in range(1,len(error_type ))]
                print("{0}, eoc = {1}".format(error_str,eoc))


            l2_errors_order[order] =  l2_errors 
            L2_error_B_plus_order[order] = L2_error_B_plus
            L2_error_B_minus_order[order] = L2_error_B_minus
            h_order[order] = h_mesh
            eoc_order[order] = round(eoc[-1],2)
            
            name_str = "jump-mup{0}-mum{1}-{2}-k{3}-order{4}.dat".format(mu_plus,mu_minus,problem_type,kk,order)
            results = [np.array(ndofs,dtype=float),np.array(h_order[order],dtype=float)]
            header_str = "ndof h "
            for error_type,error_str in zip([ L2_error_B_minus, L2_error_B_plus ],["L2-error-u-uh-B-minus","L2-error-u-uh-B-plus"]):
                results.append( np.array(error_type,dtype=float))
                header_str += "{0} ".format(error_str)
            np.savetxt(fname ="../data/{0}".format(name_str),
                       X = np.transpose(results),
                       header = header_str,
                       comments = '')

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

#####################
####################

def RunProblemConvexOscillatoryKhscaling():
    
    orders = [1,2] 
    ratio = 1/2
    #kks = [1+2*j for j in range(4)]
    kks = [1+j for j in range(7)]
    #kks = [1+j for j in range(4)]
    #kks = [1+j for j in range(9)]
    kks_np = np.array(kks)
    
    meshwidths = [ ratio/kk for kk in kks   ] 
    print("meshwidth = ", meshwidths )
    ls_mesh = [ create_initial_mesh_convex(init_h_scale = h_k) for h_k in meshwidths ]
    #ls_mesh = get_mesh_convex(6,init_h_scale=1.0)   
    
    elastic_convex.mu = 1.0
    elastic_convex.lam = 1.25

    for str_param in ["tuned","naive"]:
        for add_bc,problem_type,pgamma,palpha,pGLS in zip([True,False],["well-posed","ill-posed"], [ ScalarType(1e-4),ScalarType(1e-4) ], 
                                                         [ ScalarType(5e-1),ScalarType(5e-1)],[ ScalarType(1e-4),ScalarType(1e-4)]   ):
            print("Considering {0} problem".format(problem_type))
            l2_errors_order = { }
            h1_semi_errors_order = { }
            for order in orders:
                l2_errors = []
                h1_semi_errors = [] 
                for kk,msh in zip(kks,ls_mesh):
                    print("kk = {0}".format(kk)) 
                    elastic_convex.rho = kk**2
                    refsol = get_reference_sol("oscillatory",kk=kk)
                    if str_param == "tuned":
                        errors = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma/(order*kk)**2,palpha=palpha,add_bc=add_bc,export_VTK=False,pGLS=pGLS/kk**4)             
                    else:
                        errors = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False,pGLS=pGLS)
                    l2_error = errors["L2-error-u-uh-B"]
                    h1_semi_error = errors["H1-semi-error-u-uh-B-absolute"]
                    ndof = errors["ndof"]     
                    l2_errors.append(l2_error)
                    h1_semi_errors.append(h1_semi_error)
                l2_errors_order[order] = l2_errors
                h1_semi_errors_order[order] = h1_semi_errors

            name_str = "Convex-Oscillatory-kh-scaling-{0}-{1}.dat".format(problem_type,str_param)
            results = [kks_np]
            header_str = "k "
            for order in orders: 
                results.append(np.array(l2_errors_order[order],dtype=float))
                header_str += "l2-order{0} ".format(order)
                results.append(np.array(h1_semi_errors_order[order],dtype=float))
                header_str += "h1-semi-order{0} ".format(order)
            np.savetxt(fname ="../data/{0}".format(name_str),
                       X = np.transpose(results),
                       header = header_str,
                       comments = '')

            for order,lstyle in zip(orders,['solid','dashed','dotted']):
                plt.loglog(kks_np, l2_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
                #tmp_str = "$\mathcal{{O}}(k^{0})$".format(order+1)
                #plt.loglog(kks_np, 1.35*l2_errors_order[order][0]*(kks_np**(order+1))/( kks_np[0]**(order+1)) ,label=tmp_str,linestyle=lstyle,color='gray')

            plt.loglog(kks_np, 1.35*l2_errors_order[1][0]*(kks_np)/( kks_np[0]) ,label="$\mathcal{O}(k)$",linestyle=lstyle,color='gray')
            #plt.loglog(kks_np, 1.35*l2_errors_order[2][0]*(kks_np)/( kks_np[0]) ,label="$\mathcal{O}(k)$",linestyle=lstyle,color='gray')
            plt.xlabel("$k$")
            plt.ylabel("L2-error")
            plt.legend()
            plt.savefig("L2-error-k-Gaussian.png",transparent=True,dpi=200)
            #plt.title("L2-error")
            plt.show()


def RunProblemConvexOscillatoryStabSweep(kk):
    
    orders = [1,2,3]  
    #ratio = 1/2 
    #meshwidths = [ ratio/kk for kk in kks   ] 
    #print("meshwidth = ", meshwidths )
    #ls_mesh = [ create_initial_mesh_convex(init_h_scale = h_k) for h_k in meshwidths ]
    ls_mesh = get_mesh_convex(4,init_h_scale=1.0)
    msh = ls_mesh[2]
    
    elastic_convex.mu = 1.0
    elastic_convex.lam = 1.25
    elastic_convex.rho = kk**2

    refsol = get_reference_sol("oscillatory",kk=kk)
    #msh = ls_mesh[3]
    add_bc = False
    problem_type = "ill-posed"
    #pgamma = ScalarType(5e-3)
    palpha = ScalarType(1e-1)
    #kks = np.linspace(1,20,6)
    #kks = [1+2*j for j in range(3)]
    pxs = [1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1]
    #pxs = [1e-13,1e-11,1e-9,1e-7,1e-5,1e-3,1e-1,1e1]
    #pxs = [1e-12,1e-3,1e0]
    pxs_np = np.array(pxs)

    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(5e-3),ScalarType(5e-3)], [ ScalarType(1e-1),ScalarType(1e-1)] ):
    for param_str in ["gamma-Jump","gamma-GLS","alpha"]:
        l2_errors_order = { }
        s_errors_order = { }
        for order in orders:
            l2_errors = [] 
            s_errors = [] 
            for px in pxs:
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

                errors = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False,pGLS= pGLS)
                l2_error = errors["L2-error-u-uh-B"]
                ndof = errors["ndof"]     
                l2_errors.append(l2_error)
                s_errors.append(errors["s-norm"])
                print("ndof = {0}, l2_error = {1},px = {2}".format(ndof,l2_error,px))
            l2_errors_order[order] = l2_errors
            s_errors_order[order] = s_errors 

        name_str = "Convex-Oscillatory-StabSweep-{0}-kk{1}.dat".format(param_str,kk) 
        results = [pxs_np]
        header_str = "gamma-Jump "
        for order in orders: 
            results.append(np.array(l2_errors_order[order],dtype=float))
            header_str += "l2-order{0} ".format(order)
            results.append(np.array(s_errors_order[order],dtype=float))
            header_str += "s-order{0} ".format(order)
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
        plt.show()
        
        for order,lstyle in zip([1,2,3],['solid','dashed','dotted']):
            plt.loglog(pxs_np, s_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)    
        plt.xlabel("$\gamma$")
        plt.ylabel("s-error")
        plt.legend()
        #plt.savefig("L2-error-stab-gamma.png",transparent=True,dpi=200)
        #plt.title("L2-error")
        plt.show()

def RunProblemNonConvexOscillatoryStabSweep(kk):
    
    orders = [1,2,3]  
    #ls_mesh = [ create_initial_mesh_nonconvex(init_h_scale = h_k) for h_k in meshwidths ]
    ls_mesh = get_mesh_hierarchy_nonconvex(4,init_h_scale=1.0)
    msh = ls_mesh[2]
    
    elastic_nonconvex.mu = 1.0
    elastic_nonconvex.lam = 1.25
    elastic_nonconvex.rho = kk**2

    refsol = get_reference_sol("oscillatory",kk=kk)
    #msh = ls_mesh[3]
    add_bc = False
    problem_type = "ill-posed"
    #pgamma = ScalarType(5e-3)
    palpha = ScalarType(1e-1)
    #kks = np.linspace(1,20,6)
    #kks = [1+2*j for j in range(3)]
    pxs = [1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1]
    #pxs = [1e-13,1e-11,1e-9,1e-7,1e-5,1e-3,1e-1,1e1]
    #pxs = [1e-12,1e-3,1e0]
    pxs_np = np.array(pxs)

    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(5e-3),ScalarType(5e-3)], [ ScalarType(1e-1),ScalarType(1e-1)] ):
    for param_str in ["gamma-Jump","gamma-GLS","alpha"]:
        l2_errors_order = { }
        s_errors_order = { }
        for order in orders:
            l2_errors = [] 
            s_errors = [] 
            for px in pxs:
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

                errors = SolveProblem(problem = elastic_nonconvex, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False,pGLS= pGLS)
                l2_error = errors["L2-error-u-uh-B"]
                ndof = errors["ndof"]     
                l2_errors.append(l2_error)
                s_errors.append(errors["s-norm"])
                print("ndof = {0}, l2_error = {1},px = {2}".format(ndof,l2_error,px))
            l2_errors_order[order] = l2_errors
            s_errors_order[order] = s_errors 

        name_str = "NonConvex-Oscillatory-StabSweep-{0}-kk{1}.dat".format(param_str,kk) 
        results = [pxs_np]
        header_str = "gamma-Jump "
        for order in orders: 
            results.append(np.array(l2_errors_order[order],dtype=float))
            header_str += "l2-order{0} ".format(order)
            results.append(np.array(s_errors_order[order],dtype=float))
            header_str += "s-order{0} ".format(order)
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



# pgamma = 1e-5/kk**2 , pGLS = 1e-4/kk**4
 
# Runs for draft
#RunProblemConvexOscillatory(kk=1)
#RunProblemConvexOscillatory(kk=6)
#RunProblemConvexGaussian(kk=6,perturb_theta=None)
#RunProblemNonConvexOscillatory(kk=1,perturb_theta=None)

#RunProblemConvexOscillatoryStabSweep(kk=1)
#RunProblemConvexOscillatoryStabSweep(kk=6)
#RunProblemConvexOscillatoryKhscaling()
RunProblemNonConvexOscillatoryStabSweep(kk=1)


#RunProblemJump(kk=6,apgamma=1e-3,apalpha=1e-0)


# old stuff
'''
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
                    l2_error, ndof = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=order==2)
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


def RunProblemNonConvexOscillatory(kk,apgamma=1e-5,apalpha=1e5):
    orders = [1,2,3] 
    ls_mesh = get_mesh_hierarchy_nonconvex(5)
    refsol = get_reference_sol("oscillatory",kk=kk)
    elastic_nonconvex.rho = kk**2
    elastic_nonconvex.mu = 1.0
    elastic_nonconvex.lam = 1.25

    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(apgamma),ScalarType(apgamma/kk**2)], [ ScalarType(apalpha),ScalarType(apalpha)] ):
    for add_bc,problem_type,pgamma,palpha in zip([False],["ill-posed"],[ScalarType(apgamma/kk**2)], [ ScalarType(apalpha)] ):
        
        print("Considering {0} problem".format(problem_type))
        l2_errors_order = { }
        eoc_order = { }
        h_order = { }
        for order in orders:
            l2_errors = [ ]
            ndofs = [] 
            for msh in ls_mesh[:-order]:
                l2_error, ndof = SolveProblem(problem = elastic_nonconvex, msh = msh,refsol=refsol,order=order,pgamma=pgamma/order**2,palpha=palpha,add_bc=add_bc,export_VTK=False,pGLS=1e-5/kk**4)
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

def RunProblemConvexOscillatory(kk):
    orders = [1,2,3] 
    ls_mesh = get_mesh_hierarchy(5,init_h_scale=1.0)
    #ls_mesh = get_mesh_convex(6,init_h_scale=1.0)   
    refsol = get_reference_sol("oscillatory",kk=kk)
    elastic_convex.rho = kk**2
    elastic_convex.mu = 1.0
    elastic_convex.lam = 1.25

    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5),ScalarType(5e-2)], [ ScalarType(1e-3),ScalarType(1e-5)] ):
    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5),ScalarType(5e-3)], [ ScalarType(1e-3),ScalarType(1e-0)] ):
    
    #for add_bc,problem_type,pgamma,palpha,pGLS in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5/kk**2),ScalarType(1e-5/kk**2) ], 
    #                                                 [ ScalarType(5e-1),ScalarType(5e-1)],[ ScalarType(1e-4/kk**4),ScalarType(1e-4/kk**4)] ):
    for add_bc,problem_type,pgamma,palpha,pGLS in zip([False],["ill-posed"],[ ScalarType(3e-4/kk**2) ], 
                                                     [ScalarType(1e-2)],[ScalarType(1e-4/kk**4)] ):
    
    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5),ScalarType(5e-2)], [ ScalarType(1e-3),ScalarType(1e+5)] ):
        
        print("Considering {0} problem".format(problem_type))
        l2_errors_order = { }
        eoc_order = { }
        h_order = { }
        for order in orders:
            l2_errors = [ ]
            ndofs = [] 
            for msh in ls_mesh[:-order]:
                l2_error, ndof = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma/order**2,palpha=palpha,add_bc=add_bc,export_VTK=order==2,pGLS=pGLS)
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

def RunProblemNonConvexOscillatoryStabSweep(kk):
    
    orders = [1,2]  
    #orders = [1]  
    #ratio = 1/2 
    #meshwidths = [ ratio/kk for kk in kks   ] 
    #print("meshwidth = ", meshwidths )
    #ls_mesh = [ create_initial_mesh_convex(init_h_scale = h_k) for h_k in meshwidths ]
    ls_mesh = get_mesh_nonconvex(5,init_h_scale=1.0)
    msh = ls_mesh[3]
    
    elastic_nonconvex.mu = 1.0
    elastic_nonconvex.lam = 1.25
    elastic_nonconvex.rho = kk**2

    refsol = get_reference_sol("oscillatory",kk=kk)
    #msh = ls_mesh[3]
    add_bc = False
    problem_type = "ill-posed"
    #pgamma = ScalarType(5e-3)
    palpha = ScalarType(1e-1)
    #kks = np.linspace(1,20,6)
    #kks = [1+2*j for j in range(3)]
    #pgammas = [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3]
    pxs = [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1]
    #pgammas = [1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4]
    #pgammas = [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
    pxs_np = np.array(pgammas)
    #l2_error, ndof = SolveProblem(problem = elastic_nonconvex, msh = msh,refsol=refsol,order=order,pgamma=ScalarType(1e-4/(order*kk)**2),palpha=ScalarType(1e-1),add_bc=add_bc,export_VTK=False,pGLS=ScalarType(1e-5/kk**4 ))

    #for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(5e-3),ScalarType(5e-3)], [ ScalarType(1e-1),ScalarType(1e-1)] ):
    for px in pgammas:
        l2_errors_order = { }
        for order in orders:
            l2_errors = [] 
            l2_error, ndof = SolveProblem(problem = elastic_nonconvex, msh = msh,refsol=refsol,order=order,pgamma=ScalarType(pgamma),palpha=ScalarType(1e-1),add_bc=add_bc,export_VTK=False,pGLS=ScalarType(1e-5/kk**4 ))
            l2_errors.append(l2_error)
                # 1e-5/(order*kk)**2
            l2_errors_order[order] = l2_errors

        for order,lstyle in zip(orders,['solid','dashed','dotted']):
            plt.loglog(pxs_np, l2_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
            
            #tmp_str = "$\mathcal{{O}}(k^{0})$".format(order+1)
            #plt.loglog(kks_np, 1.35*l2_errors_order[order][0]*(kks_np**(order+1))/( kks_np[0]**(order+1)) ,label=tmp_str,linestyle=lstyle,color='gray')
        #plt.loglog(kks_np, 1.35*l2_errors_order[2][0]*(kks_np)/( kks_np[0]) ,label="$\mathcal{O}(k)$",linestyle=lstyle,color='gray')
        plt.xlabel("$\gamma$")
        plt.ylabel("L2-error")
        plt.legend()
        plt.savefig("L2-error-stab-gamma.png",transparent=True,dpi=200)
        #plt.title("L2-error")
        plt.show()

#RunProblemConvexGaussian(kk=1)
#RunProblemConvexGaussian(kk=6)

#RunProblemConvexOscillatory(kk=1)
#RunProblemConvexOscillatory(kk=4)

#RunProblemConvexOscillatoryKhscaling()

#RunProblemConvexOscillatoryStabSweep(kk=6)

RunProblemNonConvexOscillatoryStabSweep(kk=6)

#RunProblemNonConvexOscillatory(kk=4,apgamma=1e-4,apalpha=1e-2)
#RunProblemNonConvexOscillatory(kk=4,apgamma=1e-5,apalpha=1e-2)

#RunProblemNonConvexOscillatory(kk=4,apgamma=1e-4,apalpha=1e0)
#RunProblemNonConvexOscillatory(kk=4,apgamma=1e-7,apalpha=1e-2)
#RunProblemNonConvexOscillatory(kk=6,apgamma=1e-1,apalpha=1e3)
#RunProblemConvexGaussianKscaling()
#RunProblemConvexOscillatoryKscaling()

#RunProblemNonConvexGaussian(kk=1,apgamma=1e-4,apalpha=1e0)
#RunProblemNonConvexGaussian(kk=4,apgamma=1e-4,apalpha=1e0)
#RunProblemNonConvexGaussian(kk=6,apgamma=1e-4,apalpha=1e0)

#RunProblemJump(kk=8,apgamma=1e-2,apalpha=1e-0)
#RunProblemJump(kk=1,apgamma=1e-3,apalpha=1e-0)
#RunProblemJump(kk=8,apgamma=5e-2,apalpha=1e-0)
#RunProblemJump(kk=6,apgamma=1e-3,apalpha=1e-0)
#RunProblemJump(kk=1,apgamma=1e-3,apalpha=1e-0)
#RunProblemJump(kk=8,apgamma=5e-3,apalpha=1e+0)

#RunProblemConvexOscillatory(kk=10)

#RunProblemConvexHadamard(kk=1,nn=5)
#RunProblemConvexHadamard(kk=10,nn=11)

#RunProblemConvexGaussian(kk=1)
#RunProblemConvexGaussian(kk=6,perturb_theta=-2)
#RunProblemNonConvexGaussian(kk=6,apgamma=1e-4,apalpha=1e0,perturb_theta=-2)

#RunProblemConvexOscillatory(kk=1,perturb=True)



'''
