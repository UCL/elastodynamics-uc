'''
Unique continuation for elastodynamics. 
'''

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, plot, io
from petsc4py.PETSc import ScalarType
from math import pi,log
from meshes import get_mesh_hierarchy, get_mesh_hierarchy_nonconvex
from problems import elastic_convex, elastic_nonconvex

kk = 10 # wavenumber
order = 2

def epsilon(u):
    return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

def get_reference_sol(type_str,kk=1):

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
    else:
        print("I do not know this solution.")
        return None 


def SolveProblem(problem,msh,refsol,order=1,pgamma=1e-5,palpha=1e-5,add_bc=False,export_VTK=False): 
    
    h = ufl.CellDiameter(msh)
    n = ufl.FacetNormal(msh)
    x = ufl.SpatialCoordinate(msh)
    ue = refsol(x)

    def sigma(u):
        return 2*problem.mu*epsilon(u) + problem.lam * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) 
    Lu = lambda u : -ufl.nabla_div(sigma(u)) - problem.rho *u
    f = Lu(ue)

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
    
    fdim = msh.topology.dim - 1
    V0, submap = VW.sub(0).collapse()
    W0, submap = VW.sub(1).collapse()
    # determine boundary DOFs
    boundary_dofs0 = fem.locate_dofs_geometrical((VW.sub(0),V0), problem.boundary_indicator )
    boundary_dofs1 = fem.locate_dofs_geometrical((VW.sub(1),W0), problem.boundary_indicator )
    
    u_D = np.array([0,0], dtype=ScalarType)
    bc = fem.dirichletbc( np.array([0,0], dtype=ScalarType), boundary_dofs1[0], VW.sub(1))
    bcs = [bc] 
    if add_bc:
        ue_h = fem.Function(V0)
        u_expr = fem.Expression(ue, V0.element.interpolation_points)
        ue_h.interpolate(u_expr)
        bc0 = fem.dirichletbc(ue_h, boundary_dofs0, VW.sub(0))
        bcs.append(bc0)

    a = omega_ind * ufl.inner(u,v) * ufl.dx
    a += pgamma*0.5*(h('+')+h('-'))*ufl.inner(ufl.jump(sigma(u),n),ufl.jump(sigma(v),n))*ufl.dS
    a += pgamma * h**2 * ufl.inner(Lu(u),Lu(v)) * ufl.dx 
    a += palpha * h**(2*order) * ufl.inner(u,v) * ufl.dx
    a += ufl.inner(sigma(v),epsilon(z)) * ufl.dx - problem.rho * ufl.inner(v,z) * ufl.dx
    a -= ufl.inner( ufl.nabla_grad(z), ufl.nabla_grad(w)) * ufl.dx 
    a += ufl.inner(sigma(u),epsilon(w)) * ufl.dx - problem.rho * ufl.inner(u,w) * ufl.dx

    L = ufl.inner(f, w) * ufl.dx + omega_ind * ufl.inner(ue,v) * ufl.dx  + pgamma * h**2 * ufl.inner(f,Lu(v)) * ufl.dx 

    prob = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}) 
    sol = prob.solve()
    uh,zh = sol.split() 

    L2_error = fem.form( B_ind*ufl.inner(uh - ue, uh - ue) * ufl.dx)
    error_local = fem.assemble_scalar(L2_error)
    L2_norm = fem.form( B_ind*ufl.inner(ue, ue) * ufl.dx)
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

ls_mesh = get_mesh_hierarchy(3)
#ls_mesh = get_mesh_hierarchy_nonconvex(4)
#refsol = get_reference_sol("oscillatory",kk=kk)
refsol = get_reference_sol("gaussian",kk=kk)
#elastic_nonconvex.rho = kk**2
elastic_convex.rho = kk**2

for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5),ScalarType(5e-3)], [ ScalarType(1e-3),ScalarType(1e-1)] ):
#for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5),ScalarType(5e-1)], [ ScalarType(5e-3),ScalarType(5e-1)] ):
    print("Considering {0} problem".format(problem_type))
    l2_errors = [ ]
    ndofs = [] 
    for msh in ls_mesh:
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
    plt.xlabel("h")
    plt.ylabel("L2-error")
    plt.legend()
    plt.savefig("L2-error-elastodynamics-P{0}-{1}.png".format(order,problem_type),transparent=True,dpi=200)
    plt.show()

