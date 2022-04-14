'''
Unique continuation for elastodynamics. 
'''

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem, plot, io
from math import pi,log
from meshes import get_mesh_hierarchy, get_mesh_hierarchy_nonconvex

mu = 1 # Lame parameter
rho = 1
lam = 1.25 # Lame parameter
kk = 1 # wavenumber
#palpha = ScalarType(1e-5) # Stab. parameter for Tikhonov 

order = 2

# defining subdomains 
#tol = 1e-14
#def omega_Ind(x):
#    return  ~( ( x[0] > (0.1 - tol) ) & ( x[0] < (0.9+tol) ) & ( x[1]>(0.25-tol) ) & ( x[1]<(1+tol) ) ) 
#def B_Ind(x):
#    return ~( ( x[0] > (0.1 - tol) ) & ( x[0] < (0.9+tol) ) & ( x[1]>(0.95-tol) ) & ( x[1]<(1+tol) ) ) 

tol = 1e-13
x_l = 0.1-tol
x_r = 0.9+tol
y_b = 0.25-tol
y_t = 1.0+tol

## convex case 

def omega_Ind(x):
    
    values = np.zeros(x.shape[1],dtype=ScalarType)
    omega_coords = np.logical_or( ( x[0] <= 0.1 ), 
        np.logical_or(   (x[0] >= 0.9 ), (x[1] <= 0.25)  )
        ) 
    rest_coords = np.invert(omega_coords)
    values[omega_coords] = np.full(sum(omega_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    return values

def B_Ind(x):
    values = np.zeros(x.shape[1],dtype=ScalarType)
    # Create a boolean array indicating which dofs (corresponding to cell centers)
    # that are in each domain
    rest_coords = np.logical_and( ( x[0] >= 0.1 ), 
        np.logical_and(   (x[0] <= 0.9 ),
          np.logical_and(   (x[1]>= 0.95),  (x[1]<= 1+tol)  )
        )
      ) 
    B_coords = np.invert(rest_coords)
    values[B_coords] = np.full(sum(B_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    return values

## non convex-case 
'''
def omega_Ind(x):
    
    values = np.zeros(x.shape[1],dtype=ScalarType)
    omega_coords = np.logical_and( ( x[0] >= 0.25 ), 
                     np.logical_and(   (x[0] <= 0.75 ),
                       np.logical_and(   (x[1]>= 0.05),  (x[1]<= 0.5)  )
                     )
                   ) 
    rest_coords = np.invert(omega_coords)
    values[omega_coords] = np.full(sum(omega_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    return values

def B_Ind(x):
    values = np.zeros(x.shape[1],dtype=ScalarType)
    # Create a boolean array indicating which dofs (corresponding to cell centers)
    # that are in each domain
    B_coords = np.logical_and( ( x[0] >= 0.125 ), 
                 np.logical_and(   (x[0] <= 0.875 ),
                   np.logical_and(   (x[1]>= 0.05),  (x[1]<= 0.95)  )
                 )
               ) 
    rest_coords = np.invert(B_coords)
    values[B_coords] = np.full(sum(B_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    return values
'''

def epsilon(u):
    return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

Lu = lambda u : -ufl.nabla_div(2*mu*epsilon(u)) - ufl.nabla_grad(lam*ufl.nabla_div(u)) - kk**2*u

def get_reference_sol(msh,type_str):

    x = ufl.SpatialCoordinate(msh)
    if type_str == "oscillatory":
        ue = ufl.as_vector([ ufl.sin(kk*pi*x[0]) * ufl.sin(kk*pi*x[1]),
                             ufl.sin(kk*pi*x[0]) * ufl.sin(kk*pi*x[1])] )  # analytical solution 
        f = Lu(ue)
        #pgamma = ScalarType(5e-2)
        pgamma = ScalarType(1e-3)
        palpha = ScalarType(1e-3)  
        return ue,f,pgamma,palpha
    elif type_str =="gaussian": 
        sigma_x = ScalarType(0.01)
        sigma_y = ScalarType(0.1)
        x_s = ScalarType(0.5)
        y_s = ScalarType(1.0)
        ue = ufl.as_vector([ ufl.exp( -( (x[0] - x_s) ** 2 / (2*sigma_x)  + (x[1] - y_s) ** 2 / (2*sigma_y)  ) ),
                              ufl.exp( -( (x[0] - x_s) ** 2 / (2*sigma_x)  + (x[1] - y_s) ** 2 / (2*sigma_y)  ) )   ])
        f = Lu(ue)
        pgamma = ScalarType(1e-4) 
        palpha = ScalarType(1e-5) 
        return ue,f,pgamma,palpha
    else:
        print("I do not know this solution.")
        return None 


def SolveProblem(msh,order=1,add_bc=False,export_VTK=False,type_str="oscillatory",pgamma=None,palpha=None): 
    #msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
    #                            points=((0.0, 0.0), (1.0, 1.0)), n=(N_mesh, N_mesh),
    #                            cell_type=mesh.CellType.triangle,)

    #help(msh)
    ue,f,pgamma_,palpha_ = get_reference_sol(msh,type_str)
    if not pgamma:
        pgamma = pgamma_
    if not palpha:
        palpha = palpha_
    
    h = ufl.CellDiameter(msh)
    #h = 2.0*ufl.Circumradius(msh)
    x = ufl.SpatialCoordinate(msh)

    fe  = ufl.VectorElement('CG', msh.ufl_cell(), order)
    mel = ufl.MixedElement([fe,fe])
    VW = fem.FunctionSpace(msh,mel)
    ndof = VW.sub(0).dofmap.index_map.size_global * VW.sub(0).dofmap.index_map_bs 
    u,z = ufl.TrialFunctions(VW)
    v,w = ufl.TestFunctions(VW)

    Q_ind = fem.FunctionSpace(msh, ("DG", 0))
    
    omega_ind = fem.Function(Q_ind)
    B_ind = fem.Function(Q_ind)
    B_ind.interpolate(B_Ind)
    omega_ind.interpolate(omega_Ind)
    #omega_ind = fem.Function(Q_ind)
    #cells_omega = mesh.locate_entities(msh, msh.topology.dim, omega_Ind)
    #omega_ind.x.array[:] = 0.0
    #omega_ind.x.array[cells_omega] = np.full(len(cells_omega), 1)

    #only_B_ind = fem.Function(Q_ind)
    #cells_only_B = mesh.locate_entities(msh, msh.topology.dim, only_B_Ind)
    #only_B_ind.x.array[:] = 0.0
    #only_B_ind.x.array[cells_only_B] = np.full(len(cells_only_B), 1)

    #omega_ind = ufl.conditional(ufl.Not(ufl.And(ufl.And(x[0] >= 0.1, x[0] <= 0.9),ufl.And(x[1] >= 0.25, x[1] <= 1))   ), 1, 0) 
    #B_ind = ufl.conditional( ufl.Not(ufl.And(ufl.And(x[0] >= 0.1, x[0] <= 0.9), ufl.And(x[1] >= 0.95, x[1] <= 1))) , 1, 0 ) 
    #B_ind = ufl.conditional( (ufl.And(ufl.And(x[0] >= 0.1, x[0] <= 0.9), ufl.And(x[1] >= 0.25, x[1] <= 0.9))) , 1, 0 ) 
    
    def boundary_indicator(x):
        return ( np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0) )

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, boundary_indicator)

    u_D = np.array([0,0], dtype=ScalarType)
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(VW.sub(1), fdim, boundary_facets), VW.sub(1))
    bcs = [bc] 
    if add_bc:  # adding homogeneous Dirichlet bc to make problem well-posed
        bc0 = fem.dirichletbc(u_D, fem.locate_dofs_topological(VW.sub(0), fdim, boundary_facets), VW.sub(0))
        bcs.append(bc0)


    a = omega_ind * ufl.inner(u,v) * ufl.dx
    a += pgamma*0.5*(h('+')+h('-'))*ufl.inner(ufl.jump(ufl.nabla_grad(u)),ufl.jump(ufl.nabla_grad(v)))*ufl.dS
    a += pgamma * h**2 * ufl.inner(Lu(u),Lu(v)) * ufl.dx 
    a += palpha * h**(2*order) * ufl.inner(u,v) * ufl.dx
    a += 2 * mu * ufl.inner(epsilon(v), epsilon(z)) * ufl.dx + lam * ufl.inner( ufl.nabla_div(v),ufl.nabla_div(z)) * ufl.dx - kk**2 * ufl.inner(v,z) * ufl.dx
    a -= ufl.inner( ufl.nabla_grad(z), ufl.nabla_grad(w)) * ufl.dx 
    a += 2 * mu * ufl.inner(epsilon(u), epsilon(w)) * ufl.dx + lam * ufl.inner( ufl.nabla_div(u),ufl.nabla_div(w)) * ufl.dx - kk**2 * ufl.inner(u,w) * ufl.dx

    L = ufl.inner(f, w) * ufl.dx + omega_ind * ufl.inner(ue,v) * ufl.dx  + pgamma * h**2 * ufl.inner(f,Lu(v)) * ufl.dx 

    problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}) 
    sol = problem.solve()
    uh,zh = sol.split() 

    #L2_error = fem.form( (omega_ind+only_B_ind)*ufl.inner(uh - ue, uh - ue) * ufl.dx)
    #L2_error = fem.form( (only_B_ind)*ufl.inner(uh - ue, uh - ue) * ufl.dx)
    L2_error = fem.form( B_ind*ufl.inner(uh - ue, uh - ue) * ufl.dx)
    #L2_error = fem.form( (omega_ind+only_B_ind)*ufl.inner(uh - ue, uh - ue) * ufl.dx)
    error_local = fem.assemble_scalar(L2_error)
    #L2_norm = fem.form( (omega_ind + only_B_ind)*ufl.inner(ue, ue) * ufl.dx)
    #L2_norm = fem.form( (only_B_ind)*ufl.inner(ue, ue) * ufl.dx)
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

#import numpy as np
import matplotlib.pyplot as plt 

#ls_mesh = get_mesh_hierarchy(3)
ls_mesh = get_mesh_hierarchy_nonconvex(4)

#for add_bc,problem_type,pgamma in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5),ScalarType(1e-1)]):
#for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5),ScalarType(1e1)], [ ScalarType(1e-3),ScalarType(1e-1)] ):
for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5),ScalarType(5e-1)], [ ScalarType(1e-3),ScalarType(1e-1)] ):
#for add_bc,problem_type,pgamma,palpha in zip([True,False],["well-posed","ill-posed"],[ ScalarType(1e-5),ScalarType(1e-5)], [ ScalarType(1e-3),ScalarType(1e-5)] ):
    print("Considering {0} problem".format(problem_type))
    #Ns = np.linspace(N_start,220,10,dtype=np.int64) 
    l2_errors = [ ]
    ndofs = [] 
    #h_mesh = np.array(1/Ns,dtype=float)
    #for N_mesh in Ns: 
    for msh in ls_mesh:
        l2_error, ndof = SolveProblem(msh=msh,order=order,add_bc=add_bc,pgamma=pgamma,palpha=palpha)
        #l2_errors.append(SolveProblem(N_mesh=N_mesh,order=1,add_bc=False,type_str="gaussian"))
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

#def omega_Ind(x):
#    tmp = ( x[0] < (0.1+tol) ) | ( x[0] > (0.9-tol) ) | ( x[1] < ( 0.25+tol ) ) 
#    return tmp

#def only_B_Ind(x):
#    return  (x[0] > (0.1-tol) ) &  (x[0] < (0.9+tol) ) & (x[1] > (0.25-tol) ) &  (x[1] < (0.95+tol) ) 

#def omega_Ind(x):
#    tmp = ( ( x[0] > (0.25-tol) ) & ( x[0] < (0.75+tol) ) & ( x[1] > ( 0.05-tol ) ) & ( x[1] < ( 0.5+tol ) ) ) 
#    return tmp

#def only_B_Ind(x):
#    tmp = ( ( x[0] > (0.125-tol) ) & ( x[0] < (0.875+tol) ) & ( x[1] > ( 0.05-tol ) ) & ( x[1] < ( 0.95+tol ) ) ) 
# return tmp
