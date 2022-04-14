'''
This script implements a modified version of the method presented in Chapter 4 of the PhD thesis: 
   Unique continuation problems and stabilised finite element methods by Mihai Nechita, UCL 2020.
'''

import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, div,inner, jump, FacetNormal, dot, dS, Circumradius, triangle, FiniteElement, MixedElement, CellDiameter
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from math import log,pi
from meshes import get_mesh_hierarchy, get_mesh_hierarchy_nonconvex
from dolfinx.io import XDMFFile

# problem data. Todo: put into class
kk = ScalarType(1)
k2 = kk**2

order = 1

tol = 1e-13
x_l = 0.1-tol
x_r = 0.9+tol
y_b = 0.25-tol
y_t = 1.0+tol


## convex case 
'''
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
'''

## convex-case 
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

sigma_x = ScalarType(0.01)
sigma_y = ScalarType(0.1)
x_s = ScalarType(0.5)
y_s = ScalarType(1.0)

def SolveProblem(msh,order=1,export_VTK=False,sol_which=1,add_bc=False): 

    #msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
    #                        points=((0.0, 0.0), (1.0, 1.0)), n=(N_mesh, N_mesh),
    #                        #points=((-0.1, -0.1), (1.1, 1.1)), n=(N_mesh, N_mesh),
    #                        cell_type=mesh.CellType.triangle,)
    
    n = FacetNormal(msh)
    h = CellDiameter(msh)
    #h = 2.0*Circumradius(msh)
    
    x = ufl.SpatialCoordinate(msh)

    Lu = lambda u : -div(grad(u)) - k2*u 
    
    if sol_which == 1:
        #pgamma = ScalarType(1e-5)
        pgamma = ScalarType(1e-4)
        ue = ufl.exp( -( (x[0] - x_s) ** 2 / (2*sigma_x)  + (x[1] - y_s) ** 2 / (2*sigma_y)  ) )
        f =   -ue * ( (x[0]-x_s)**2/ sigma_x**2  + (x[1]-y_s)**2 / sigma_y**2 - 1/sigma_x - 1/sigma_y + k2 )
        palpha = ScalarType(1e-5)
    else: 
        #pgamma = ScalarType(5e-3)
        pgamma = ScalarType(5e-3)
        #pgamma = ScalarType(1e-4)
        #pgamma = ScalarType(1e-1)
        ue = ufl.sin(kk*pi*x[0]) * ufl.sin(kk*pi*x[1])
        #f = Lu(ue)
        palpha = ScalarType(5e-3)
        #palpha = ScalarType(1e-2)
        f =   -ue *  (-2*(kk*pi)**2 + k2) 

     
    p_cg_order = FiniteElement('CG', msh.ufl_cell(), order)
    #p_cg_order_dual = FiniteElement('CG', msh.ufl_cell(), order)
    mel = MixedElement([p_cg_order,p_cg_order])
    VW = fem.FunctionSpace(msh,mel)
    #VW = fem.FunctionSpace(msh, p_cg_order_primal * p_cg_order_dual )
    ndof = VW.sub(0).dofmap.index_map.size_global * VW.sub(0).dofmap.index_map_bs 

    facets = mesh.locate_entities_boundary(msh, dim=1,
                                           marker=lambda x: ( np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0) ))
    dofs = fem.locate_dofs_topological(V=VW.sub(1), entity_dim=1, entities=facets)
    bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=VW.sub(1))
    bcs = [bc]
    if add_bc:
        dofs0 = fem.locate_dofs_topological(V=VW.sub(0), entity_dim=1, entities=facets)
        bc0 = fem.dirichletbc(value=ScalarType(0), dofs=dofs0,V=VW.sub(0))
        bcs.append(bc0)

    u,z = ufl.TrialFunctions(VW)
    v,w = ufl.TestFunctions(VW)

    
    Q_ind = fem.FunctionSpace(msh, ("DG", 0))
    
    omega_ind = fem.Function(Q_ind)
    B_ind = fem.Function(Q_ind)
    B_ind.interpolate(B_Ind)
    omega_ind.interpolate(omega_Ind)

    #cells_omega = mesh.locate_entities(msh, msh.topology.dim, omega_Ind)
    #omega_ind.x.array[:] = 0.0
    #omega_ind.x.array[cells_omega] = np.full(len(cells_omega), 1)

    #cells_B = mesh.locate_entities(msh, msh.topology.dim, B_Ind)
    #B_ind.x.array[:] = 0.0
    #B_ind.x.array[cells_B] = np.full(len(cells_B), 1)

    #omega_ind = ufl.conditional(ufl.Not(ufl.And(ufl.And(x[0] >= 0.1, x[0] <= 0.9),ufl.And(x[1] >= 0.25, x[1] <= 1))   ), 1, 0) 
    #B_ind = ufl.conditional( ufl.Not(ufl.And(ufl.And(x[0] >= 0.1, x[0] <= 0.9), ufl.And(x[1] >= 0.95, x[1] <= 1))) , 1, 0 ) 
    #B_ind = ufl.conditional( ufl.Not(ufl.And(ufl.And(x[0] >= 0.1, x[0] <= 0.9), ufl.And(x[1] >= 0.9, x[1] <= 1))) , 1, 0 ) 
    
    #px = 1e+1
    #input("") 
    a = omega_ind * u * v * dx
    #a += pgamma*0.5*(h('+')+h('-'))*dot(jump(grad(u)),jump(grad(v)))*dS
    a += pgamma*0.5*(h('+')+h('-'))*dot(jump(grad(u),n),jump(grad(v),n))*dS
    a += pgamma * h**2 * Lu(u) * Lu(v) * dx 
    #a += pgamma * h**2 * Lu(u) * Lu(v) * dx 
    a += ( inner(grad(v), grad(z)) - k2*v*z) * dx 
    a -= inner(grad(z), grad(w)) * dx 
    a += ( inner(grad(u), grad(w)) - k2*u*w) * dx 
    a += palpha * h**(2*order) * u * v * dx
    
    #a += 1e-8 * u * v * dx
    #a += 1e-8 * w * z * dx

    L = inner(f, w) * dx + omega_ind * ue  * v * dx  + pgamma * h**2 * inner(f,Lu(v)) * dx 

    #problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu","pc_factor_mat_solver_type":"mumps"})
    sol = problem.solve()
    #print("sol = ",  sol.x.array)
    #print(" len( sol.x.array ) = ", len( sol.x.array ) )  
    #print("") 
    
    uh,zh = sol.split() 
    #print("uh = ",  uh.x.array)
    #print(" len( uh.x.array ) = ", len( uh.x.array ) )  
    #print("") 
    
    #print("zh = ",  zh.x.array)
    #print(" len( zh.x.array ) = ", len( zh.x.array ) )  
    #print("") 

    uh.name ="uh"
    zh.name ="zh"

    #uex = fem.Function(V)
    #u_exact = lambda x: np.exp( -( (x[0] - x_s) ** 2 / (2*sigma_x)  + (x[1] - y_s) ** 2 / (2*sigma_y)  ) )
    #uex.interpolate(u_exact)
    
    #L2_error = fem.form( (omega_ind+B_ind)*inner(uh - ue, uh - ue) * dx)
    #L2_norm = fem.form((omega_ind+B_ind)*inner(ue, ue) * dx)
    
    L2_error = fem.form( (B_ind)*inner(uh - ue, uh - ue) * dx)
    L2_norm = fem.form((B_ind)*inner(ue, ue) * dx)
    
    error_local = fem.assemble_scalar(L2_error)
    L2_local = fem.assemble_scalar(L2_norm)
    error_L2 = np.sqrt(msh.comm.allreduce(error_local, op=MPI.SUM)) / np.sqrt(msh.comm.allreduce(L2_local, op=MPI.SUM)) 
    print("ndof = {0}, error_L2 = {1} ".format(ndof,error_L2)) 

    if export_VTK:
        #V = VW.sub(0).collapse()[0]
        #V = VW.sub(0).collapse()
        #p_cg_order = FiniteElement('CG', msh.ufl_cell(), order)
        #V = fem.FunctionSpace(msh,p_cg_order)
        V, V_to_VW = VW.sub(0).collapse()
        #ndof_V = V.dofmap.index_map.size_global * V.dofmap.index_map_bs 
        #print("ndof_V = ", ndof_V)
        #print(" zh.x.array =  ", zh.x.array )   
        #V = fem.FunctionSpace(msh, ("CG", order))
        uex = fem.Function( V )
        u_expr = fem.Expression(ue, V.element.interpolation_points)
        uex.interpolate(u_expr)
        #uex.interpolate(ue )
        udiff = fem.Function(V)
        #print(uh.x.array) 
        #print(uex.x.array) 
        #print(V_to_VW)
        #input("")
        udiff.x.array[:] = np.abs(uh.x.array[V_to_VW] - uex.x.array)
        #udiff.x.array[:] = uh.x.array[V_to_VW]
        with XDMFFile(msh.comm, "helmholtz-UC.xdmf", "w") as file:
            file.write_mesh(msh)
            file.write_function(udiff)
        #with io.XDMFFile(msh.comm, "helmholtz-UC-uh.xdmf", "w") as file:
            #file.write_mesh(msh)
            #file.write_function(uh)
            #file.write_function( uh.sub(0).collapse() )
            #file.write_function( udiff )
    
    return error_L2,ndof

print("Investigating asymptotic regime")
l2_errors = [ ]
#Ns = np.linspace(128,256,9,dtype=np.int64)
#Ns = np.linspace(8,180,9,dtype=np.int64)
#Ns = np.linspace(30,120,9,dtype=np.int64)
#for N_mesh in Ns:  

import matplotlib.pyplot as plt 
#ls_mesh = get_mesh_hierarchy(6)
#ls_mesh = get_mesh_hierarchy(6)
ls_mesh = get_mesh_hierarchy_nonconvex(7)

l2_errors_order = { }
eoc_order = {  }
h_order = {  }
for order in [1,2,3]:
#for order in [1]:
    print("order = {0}".format(order))
    ndofs = [] 
    l2_errors = [ ]
    for msh in ls_mesh[:-order]:
        #l2_error, ndof = SolveProblem(msh=msh,order=order)
        l2_error, ndof = SolveProblem(msh=msh,order=order, export_VTK=True,sol_which=2,add_bc=False)
        #l2_errors.append(SolveProblem(N_mesh=N_mesh,order=1, export_VTK=False,sol_which=2,add_bc=False))
        #l2_errors.append(SolveProblem(msh=mesh,order=1, export_VTK=False,sol_which=2,add_bc=True))
        l2_errors.append(l2_error)
        ndofs.append(ndof)
    eoc = [ log(l2_errors[i-1]/l2_errors[i])/log(2) for i in range(1,len(l2_errors))]
    print("eoc = ", eoc)
    ndofs = np.array(ndofs) 
    h_mesh = order/ndofs**(1/2)

    l2_errors_order[order] =  l2_errors 
    h_order[order] = h_mesh
    eoc_order[order] = eoc[-1]

    #hh = 1/Ns
    #rate_estimate, _ = np.polyfit(np.log(hh), np.log(l2_errors), 1)
    #print("(Relative) L2-error rate estimate = ", rate_estimate)
    
    #plt.loglog(h_mesh, l2_errors,'-x',label="p={0}".format(order))
    #tmp_str = "p={0}".format(order)
    #tmp_str += ",eoc={:.2f}".format(eoc[-1])
    #plt.loglog(h_mesh,h_mesh**eoc[-1],label=tmp_str,linestyle='solid',color='gray')

    #plt.loglog(h_mesh,h_mesh,label="linear",linestyle='dashed',color='gray')
    #plt.loglog(h_mesh,h_mesh**2,label="quadratic",linestyle="dotted",color='gray')
    #plt.loglog(h_mesh,h_mesh**eoc[-1],label="p={0}".format(order) + "eoc={:.2f}".format(eoc[-1]),linestyle='solid',color='gray')

for order in [1,2,3]: 
    plt.loglog(h_order[order], l2_errors_order[order] ,'-x',label="p={0}".format(order))
    tmp_str = "p={0}".format(order)
    tmp_str += ",eoc={:.2f}".format(eoc_order[order])
    plt.loglog(h_order[order],h_order[order]**eoc_order[order],label=tmp_str,linestyle='solid',color='gray')
print("hello")
plt.xlabel("h")
plt.ylabel("L2-error")
plt.legend()
#plt.title("L2-error")
plt.show()

#print("\n Investigating pre-asymptotic regime")
#l2_errors = [ ]
#Ns = [2**(3+j) for j in range(6)]
#for N_mesh in Ns:  
#    l2_errors.append(SolveProblem(N_mesh=N_mesh,order=1))
#eoc = [ log(l2_errors[i-1]/l2_errors[i])/log(2) for i in range(1,len(l2_errors))]
#print("eoc = ", eoc)
#import matplotlib.pyplot as plt 
#plt.loglog(1/np.array(Ns), l2_errors,'-x')
#plt.xlabel("h")
#plt.ylabel("L2-error")
#plt.show()

'''
def omega_Ind(x):
    tmp = ( x[0] < (0.1+tol) ) | ( x[0] > (0.9-tol) ) | ( x[1] < ( 0.25+tol ) ) 
    return tmp

def B_Ind(x):
    return  (x[0] > (0.1-tol) ) &  (x[0] < (0.9+tol) ) & (x[1] > (0.25-tol) ) &  (x[1] < (0.95+tol) ) 

def omega_Ind(x):
    tmp = ( ( x[0] > (0.25-tol) ) & ( x[0] < (0.75+tol) ) & ( x[1] > ( 0.05-tol ) ) & ( x[1] < ( 0.5+tol ) ) ) 
    return tmp

def B_Ind(x):
    tmp = ( ( x[0] > (0.125-tol) ) & ( x[0] < (0.875+tol) ) & ( x[1] > ( 0.05-tol ) ) & ( x[1] < ( 0.95+tol ) ) ) 
    return tmp
'''
