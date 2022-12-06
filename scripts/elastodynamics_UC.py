'''
Unique continuation for elastodynamics. 
'''
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
#def gamma_analytical(t,p=1):
#    if p ==1:
#        return t**2*(np.cos( ) 
np.random.seed(123)

def get_h_max(mesh):
    mesh.topology.create_connectivity(2, 0)
    triangles = mesh.topology.connectivity(2, 0).array.reshape((-1, 3))
    return geth(mesh, 2, triangles).max()

def epsilon(u):
    return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

def condest(bilinear_form,bcs,msh,target_shift=1e-13):
    A = fem.petsc.assemble_matrix(fem.form(bilinear_form), bcs)
    A.assemble()
   
    Ainv = PETSc.KSP().create(msh.comm)
    Ainv.setOperators(A)
    Ainv.setType(PETSc.KSP.Type.PREONLY)
    Ainv.getPC().setType(PETSc.PC.Type.LU)

    eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
    eigensolver.setOperators(A)
    eigensolver.setDimensions(nev=1)

    eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    eigensolver.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    eigensolver.solve()
    eigen_max = eigensolver.getEigenvalue(0)
    
    shift = eigensolver.getST()
    shift.setType(SLEPc.ST.Type.SINVERT)

    eigensolver.setST(shift)
    eigensolver.setOperators(A)
    #eigensolver.setTarget(1e-10)
    eigensolver.setTarget(target_shift)
    eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
    eigensolver.setType('krylovschur')
    eigensolver.setFromOptions()
    eigensolver.setUp()
    eigensolver.solve()

    evs = eigensolver.getConverged()
    eigen_min = eigensolver.getEigenvalue(0)
    cond_est = abs(eigen_max)/abs(eigen_min)
    #print("cond = " , cond_est)
    return cond_est

def get_reference_sol(type_str,kk=1,eta=0.6,mu_plus=2,mu_minus=1,lam=1.25,nn=5,km=1,kp=1,mu=1,x_L = -0.75,x_R = 0.75,y_L = -0.75,y_R = 0.75
        ):

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
    elif type_str == "jump-wavenumber":
        kbar = 0.5*(km+kp)
        Ap_1 = (math.cos(kp*pi*eta)/kp)*(km + kp*math.tan(kp*pi*eta))
        #Ap_1 = (math.cos(kp*pi*eta)/kp)*(6 + kp*math.tan(kp*pi*eta))
        Ap_2 = (math.cos(kp*pi*eta)/kp)*(km + kp*math.tan(kp*pi*eta))
        Bp_1 = (1-Ap_1*math.sin(kp*pi*eta))/math.cos(kp*pi*eta) 
        Bp_2 = (1-Ap_2*math.sin(kp*pi*eta))/math.cos(kp*pi*eta)
        
        #u1_plus = ufl.sin(kbar*pi*x[0])*(Ap_1*ufl.sin(kp*pi*x[1]) + Bp_1*ufl.cos(kp*pi*x[1]))
        #u1_minus = ufl.cos(kbar*pi*x[0])*(Ap_2*ufl.sin(kp*pi*x[1]) + Bp_2*ufl.cos(kp*pi*x[1]))
        #u2_plus = ufl.sin(kbar*pi*x[0])*(ufl.sin(km*pi*(x[1]-eta))  + ufl.cos(km*pi*(x[1]-eta)))
        #u2_minus = ufl.cos(kbar*pi*x[0])*(ufl.sin(km*pi*(x[1]-eta))  + ufl.cos(km*pi*(x[1]-eta)))
        def jump_k_sol(x):
            u1_plus = ufl.sin(kbar*pi*x[0])*(Ap_1*ufl.sin(kp*pi*x[1]) + Bp_1*ufl.cos(kp*pi*x[1]))
            u1_minus = ufl.sin(kbar*pi*x[0])*(ufl.sin(km*pi*(x[1]-eta))  + ufl.cos(km*pi*(x[1]-eta)))
            u2_plus = ufl.cos(kbar*pi*x[0])*(Ap_2*ufl.sin(kp*pi*x[1]) + Bp_2*ufl.cos(kp*pi*x[1]))
            u2_minus = ufl.cos(kbar*pi*x[0])*(ufl.sin(km*pi*(x[1]-eta))  + ufl.cos(km*pi*(x[1]-eta)))
            u1 = ufl.conditional( ufl.gt(x[1]-eta,0), u1_plus,u1_minus) 
            u2 = ufl.conditional( ufl.gt(x[1]-eta,0), u2_plus,u2_minus) 
            return ufl.as_vector([u1, u2])
        def jump_k_f(x):
            u1_plus = ufl.sin(kbar*pi*x[0])*(Ap_1*ufl.sin(kp*pi*x[1]) + Bp_1*ufl.cos(kp*pi*x[1]))
            u1_minus = ufl.sin(kbar*pi*x[0])*(ufl.sin(km*pi*(x[1]-eta))  + ufl.cos(km*pi*(x[1]-eta)))
            u2_plus = ufl.cos(kbar*pi*x[0])*(Ap_2*ufl.sin(kp*pi*x[1]) + Bp_2*ufl.cos(kp*pi*x[1]))
            u2_minus = ufl.cos(kbar*pi*x[0])*(ufl.sin(km*pi*(x[1]-eta))  + ufl.cos(km*pi*(x[1]-eta)))
            
            f1_minus = u1_minus*((2*mu+lam)*(kbar*pi)**2 + mu*(km*pi)**2 - km**2) + (lam+mu)*km*kbar*(pi**2)*ufl.sin(kbar*pi*x[0])*(ufl.cos(km*pi*(x[1]-eta))- ufl.sin(km*pi*(x[1]-eta)))   
            f2_minus = u2_minus*((2*mu+lam)*(km*pi)**2 + mu*(kbar*pi)**2 - km**2) + (lam+mu)*km*kbar*(pi**2)*ufl.cos(kbar*pi*x[0])*(-ufl.cos(km*pi*(x[1]-eta))+ ufl.sin(km*pi*(x[1]-eta)))  
            f1_plus = u1_plus*((2*mu+lam)*(kbar*pi)**2 + mu*(kp*pi)**2 - kp**2) + (lam+mu)*kp*kbar*(pi**2)*ufl.sin(kbar*pi*x[0])*(Ap_2*ufl.cos(kp*pi*x[1])- Bp_2*ufl.sin(kp*pi*x[1]))    
            f2_plus = u2_plus*( (2*mu+lam)*(kp*pi)**2+mu*(kbar*pi)**2-kp**2)  + (lam+mu)*kp*kbar*(pi**2)*ufl.cos(kbar*pi*x[0])*(-Ap_1*ufl.cos(kp*pi*x[1]) + Bp_1*ufl.sin(kp*pi*x[1]))  
            f1 = ufl.conditional( ufl.gt(x[1]-eta,0), f1_plus,f1_minus) 
            f2 = ufl.conditional( ufl.gt(x[1]-eta,0), f2_plus,f2_minus) 

            #f1_minus = ufl.sin(kbar*pi*x[0])*( ufl.sin(km*pi*(x[1]-eta))*( (2*mu+lam)*(kbar*pi)**2-(lam+mu)*kbar*pi*km*pi-km**2+(km*pi)**2*mu ) 
            #                                + ufl.cos(km*pi*(x[1]-eta))*( (2*mu+lam)*(kbar*pi)**2 + mu*(km*pi)**2 + (lam+mu)*km*pi*kbar*pi - km**2) )
            #f2_minus = ufl.cos(kbar*pi*x[0])*( ufl.sin(km*pi*(x[1]-eta))*( (2*mu+lam)*(km*pi)**2+(lam+mu)*kbar*pi*km*pi*-km**2+(kbar*pi)**2*mu ) 
            #                                + ufl.cos(km*pi*(x[1]-eta))*( (2*mu+lam)*(km*pi)**2 + mu*(kbar*pi)**2 - (lam+mu)*km*pi*kbar*pi - km**2 ) )
            #f1_plus = ufl.sin(kbar*pi*x[0])*( ufl.sin(kp*pi*x[1])*( (2*mu+lam)*(kbar*pi)**2*Ap_1-(lam+mu)*kbar*pi*kp*pi*Bp_2-kp**2*Ap_1+(kp*pi)**2*mu*Ap_1 ) 
            #                                + ufl.cos(kp*pi*x[1])*( (2*mu+lam)*(kbar*pi)**2*Bp_1 + mu*(kp*pi)**2*Bp_1 + (lam+mu)*kp*pi*kbar*pi*Ap_2 - kp**2*Bp_1 ) )
            #f2_plus = ufl.cos(kbar*pi*x[0])*( ufl.sin(kp*pi*x[1])*( (2*mu+lam)*(kp*pi)**2*Ap_2+(lam+mu)*kbar*pi*kp*pi*Bp_1-kp**2*Ap_2+(kbar*pi)**2*mu*Ap_2 ) 
            #                                + ufl.cos(kp*pi*x[1])*( (2*mu+lam)*(kp*pi)**2*Bp_2 + mu*(kbar*pi)**2*Bp_2 - (lam+mu)*kp*pi*kbar*pi*Ap_1 - kp**2*Bp_2) )
            #f1 = ufl.conditional( ufl.gt(x[1]-eta,0), f1_plus,f1_minus) 
            #f2 = ufl.conditional( ufl.gt(x[1]-eta,0), f2_plus,f2_minus) 
            return ufl.as_vector([f1, f2])
        return jump_k_sol,jump_k_f
    
    elif type_str == "jump-disk":
        def jump_disk_sol(x):
            a = 1
            r = ufl.sqrt(x[0]*x[0]+x[1]*x[1] + 1e-10)
            #r = (x[0]*x[0]+x[1]*x[1])
            #r = x[0]
            gamma_r = (r-a)*(r-a)
            Ap_1 = 1.0
            Bp_1 = 1.0 
            Ap_2 = 1.0 
            Bp_2 = 1.0
            Am_1 = 1.0
            Bm_1 = 1.0 
            Am_2 = 1.0 
            Bm_2 = 1.0
            u1_plus = gamma_r*( Ap_1*ufl.sin(kk*pi*r) + Bp_1*ufl.cos(kk*pi*r) )   
            u1_minus = gamma_r*( Am_1*ufl.sin(kk*pi*r) + Bm_1*ufl.cos(kk*pi*r) ) 
            u2_plus =gamma_r*( Ap_2*ufl.sin(kk*pi*r) + Bp_2*ufl.cos(kk*pi*r) ) 
            u2_minus = gamma_r*( Am_2*ufl.sin(kk*pi*r) + Bm_2*ufl.cos(kk*pi*r) ) 
            u1 = ufl.conditional( ufl.gt((x[0]*x[0]+x[1]*x[1])**2-a**2,0), u1_plus,u1_minus) 
            u2 = ufl.conditional( ufl.gt((x[0]*x[0]+x[1]*x[1])**2-a**2,0), u2_plus,u2_minus) 
            return ufl.as_vector([u1, u2])
            #return ufl.as_vector([u1_plus, u2_plus])
            #return ufl.as_vector([x[0] , x[1]])
        return jump_disk_sol

    elif type_str == "jump-square":
        #x_L = -0.75
        #x_R = 0.75
        #y_L = -0.75
        #y_R = 0.75
        def jump_square_sol(x):
            square_indicator = ufl.And(
                ufl.And(x[0] >= x_L, x[0] <= x_R), 
                ufl.And(x[1] >= y_L, x[1] <= y_R) ) 
            phi_square = (x[0]-x_L)*(x[0]-x_L)*(x[0]-x_R)*(x[0]-x_R)*(x[1]-y_L)*(x[1]-y_L)*(x[1]-y_R)*(x[1]-y_R)
            u1_plus = phi_square *  ufl.sin(kk*pi*x[0]) * ufl.sin(kk*pi*x[1])
            u1_minus = phi_square *  ufl.cos(kk*pi*x[0]) * ufl.sin(kk*pi*x[1])
            u2_plus = phi_square *  ufl.sin(kk*pi*x[0]) * ufl.cos(kk*pi*x[1])
            u2_minus = phi_square *  ufl.cos(kk*pi*x[0]) * ufl.cos(kk*pi*x[1])
            u1 = ufl.conditional( square_indicator , u1_minus , u1_plus) 
            u2 = ufl.conditional( square_indicator , u2_minus , u2_plus) 
            return ufl.as_vector([u1, u2])
        return jump_square_sol
    
    if type_str == "Hadamard":
        def Hadamard_sol(x):
            return ufl.as_vector([ ufl.sin(kk*pi*x[0]) * ufl.sinh( ufl.sqrt(nn**2-kk**2)*x[1] )/ ufl.sqrt(nn**2-kk**2) ,
                            ufl.sin(kk*pi*x[0]) *  ufl.sinh( ufl.sqrt(nn**2-kk**2)*x[1] )/ ufl.sqrt(nn**2-kk**2)  ]  ) 
        return Hadamard_sol

    else:
        print("I do not know this solution.")
        return None 


def SolveProblem(problem,msh,refsol,order=1,pgamma=1e-5,palpha=1e-5,add_bc=False,export_VTK=False,rhs=None,mu_Ind=None,k_Ind=None,perturb_order=None,pGLS=None,compute_cond=False,GradTikhStab=False,div_known=False,gamma_CIP_primal = 0): 

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
                  "cond": None
                 }

    #gamma_CIP_primal = 1e-1
    #gamma_CIP_primal = 1e-18

    if pGLS == None:
        pGLS = pgamma

    dx = ufl.dx
    if mu_Ind: 
        metadata = {"quadrature_degree": order+4}
        dx = ufl.Measure("dx", domain=msh, metadata=metadata)
    
    h = ufl.CellDiameter(msh)
    n = ufl.FacetNormal(msh)
    x = ufl.SpatialCoordinate(msh)
    ue = refsol(x)
    
    mu = problem.mu(x)
    lam = problem.lam(x)
    rho = problem.rho(x) 

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
    if True: 
        with io.XDMFFile(msh.comm, "omega-P0-order{0}-ndof{1}.xdmf".format(order,ndof), "w") as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(omega_ind)
        with io.XDMFFile(msh.comm, "B-P0-order{0}-ndof{1}.xdmf".format(order,ndof), "w") as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(B_ind)

    if mu_Ind: 
        #metadata = {"quadrature_degree": order+4}
        #dx = ufl.Measure("dx", domain=msh, metadata=metadata)
        mu = fem.Function(Q_ind)
        mu.interpolate(mu_Ind)
        with io.XDMFFile(msh.comm, "mu-jump-order{0}-ndof{1}.xdmf".format(order,ndof), "w") as xdmf:
            #mu.name ="mu-jump"
            xdmf.write_mesh(msh)
            xdmf.write_function(mu)
        #problem.mu = mu 
    if k_Ind: 
        rho = fem.Function(Q_ind)
        rho.interpolate(k_Ind)
        with io.XDMFFile(msh.comm, "k-squared-jump.xdmf", "w") as xdmf:
            #rho.name ="k-squared-jump"
            xdmf.write_mesh(msh)
            xdmf.write_function(rho)
    
    if problem.plus_Ind:
        plus_ind = fem.Function(Q_ind)
        plus_ind.interpolate(problem.plus_Ind)
        minus_ind = fem.Function(Q_ind)
        minus_ind.interpolate(problem.minus_Ind)
        with io.XDMFFile(msh.comm, "B-minus.xdmf", "w") as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(minus_ind)
        with io.XDMFFile(msh.comm, "B-plus.xdmf", "w") as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(plus_ind)

    #print(" u.geometric_dimension() = ", u.geometric_dimension())
    def sigma(u):
        return 2*mu*epsilon(u) + lam * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) 
    Lu = lambda u : -ufl.nabla_div(sigma(u)) - rho *u
    if rhs:
        f = rhs(x)
    else:
        f = Lu(ue)
    q_div = ufl.nabla_div(ue) 

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
    u_expr = fem.Expression(ue, V0.element.interpolation_points())
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
    a =  pgamma*0.5*(h('+')+h('-'))*ufl.inner(ufl.jump(sigma(u),n),ufl.jump(sigma(v),n))*ufl.dS
    if div_known: 
        a += omega_ind * ufl.inner(u,v) * dx
        #a += omega_ind * ufl.inner(u[0],v[0]) * dx
        a += ufl.inner(ufl.nabla_div(u),ufl.nabla_div(v))*dx
    else:
        a += omega_ind * ufl.inner(u,v) * dx
        #a += ufl.inner(omega_ind*u,v) * dx
    #a +=  pgamma*order**2*0.5*(h('+')+h('-'))*ufl.inner(ufl.jump(sigma(u),n),ufl.jump(sigma(v),n))*ufl.dS
    #a += pgamma * h**2 * ufl.inner(Lu(u),Lu(v)) * ufl.dx 
    a += pGLS * h**2 * ufl.inner(Lu(u),Lu(v)) * dx 
    if GradTikhStab:
        a += palpha * h**(2*order) * ufl.inner(ufl.nabla_grad(u),ufl.nabla_grad(v)) * ufl.dx
    else:
        a += palpha * h**(2*order) * ufl.inner(u,v) * dx
    a += ufl.inner(sigma(v),epsilon(z)) * ufl.dx - rho * ufl.inner(v,z) * dx
    a -= ufl.inner( ufl.nabla_grad(z), ufl.nabla_grad(w)) * dx 
    a += ufl.inner(sigma(u),epsilon(w)) * ufl.dx - rho * ufl.inner(u,w) * dx
    if abs(gamma_CIP_primal) > 1e-14:
        hbar = 0.5*(h('+')+h('-'))
        if order >= 1:
            a += gamma_CIP_primal * hbar * ufl.inner(ufl.jump(sigma(u),n),ufl.jump(sigma(w),n))*ufl.dS 
            a += gamma_CIP_primal * hbar * ufl.inner(ufl.jump(sigma(v),n),ufl.jump(sigma(z),n))*ufl.dS 
            #a += gamma_CIP_primal * hbar * ufl.inner(ufl.jump(ufl.grad(u),n),ufl.jump(ufl.grad(w),n))*ufl.dS 
            #a += gamma_CIP_primal * hbar * ufl.inner(ufl.jump(ufl.grad(u)),ufl.jump(ufl.grad(w)))*ufl.dS 
        if order >= 2:
            a += abs(gamma_CIP_primal) * hbar**3 * ufl.inner(ufl.jump( ufl.grad(sigma(u)),n),ufl.jump(ufl.grad(sigma(v)),n))*ufl.dS 
            a += gamma_CIP_primal * hbar**3 * ufl.inner(ufl.jump( ufl.grad(sigma(u)),n),ufl.jump(ufl.grad(sigma(w)),n))*ufl.dS 
            a += gamma_CIP_primal * hbar**3 * ufl.inner(ufl.jump( ufl.grad(sigma(v)),n),ufl.jump(ufl.grad(sigma(z)),n))*ufl.dS 
            #
            #a += abs(gamma_CIP_primal) * hbar**3 * ufl.inner(ufl.jump(ufl.grad(sigma(u))),ufl.jump(ufl.grad(sigma(v))))*ufl.dS 
            #a += gamma_CIP_primal * hbar**3 * ufl.inner(ufl.jump( ufl.grad(sigma(u))),ufl.jump(ufl.grad(sigma(w))))*ufl.dS 
            #a += gamma_CIP_primal * hbar**3 * ufl.inner(ufl.jump( ufl.grad(sigma(v))),ufl.jump(ufl.grad(sigma(z))))*ufl.dS 

        #if order == 3:
            #a += gamma_CIP_primal * 1e-2 * hbar**5 * ufl.inner(ufl.jump(  ufl.grad(ufl.grad(sigma(u))),n),ufl.jump(ufl.grad(ufl.grad(sigma(w))),n))*ufl.dS 
            #a += gamma_CIP_primal * hbar**5 * ufl.inner(ufl.jump( ufl.grad(ufl.grad(ufl.grad(u))),n),ufl.jump(ufl.grad(ufl.grad(ufl.grad(w))),n))*ufl.dS 
            #a += gamma_CIP_primal * 1e2 * hbar**5 * ufl.inner(ufl.jump( ufl.grad(ufl.grad(ufl.grad(u)))),ufl.jump(ufl.grad(ufl.grad(ufl.grad(w)))))*ufl.dS 

    
    if compute_cond:  
        cond_est =  condest(a,bcs,msh,target_shift=5e-13)
        print("cond = " , cond_est)
        error_dict["cond"] = cond_est


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
        L = ufl.inner(f+h**(perturb_order)*delta_f, w) * dx  + pGLS * h**2 * ufl.inner(f,Lu(v)) * dx 
        if div_known: 
            #L += ufl.inner(q_div,ufl.nabla_div(v))*dx + omega_ind * ufl.inner(ue[0]+h**(perturb_order)*delta_u[0],v[0]) * dx 
            L +=  ufl.inner(q_div,ufl.nabla_div(v))*dx + omega_ind * ufl.inner(ue+h**(perturb_order)*delta_u,v) * dx 
        else:
            L += omega_ind * ufl.inner(ue+h**(perturb_order)*delta_u,v) * dx 

    else:
        #L = ufl.inner(f, w) * ufl.dx + omega_ind * ufl.inner(ue,v) * ufl.dx  + pgamma * h**2 * ufl.inner(f,Lu(v)) * ufl.dx 
        L = ufl.inner(f, w) * dx + pGLS * h**2 * ufl.inner(f,Lu(v)) * dx 
        if div_known: 
            #L += ufl.inner(q_div,ufl.nabla_div(v))*dx + omega_ind * ufl.inner(ue[0],v[0]) * dx 
            L +=  ufl.inner(q_div,ufl.nabla_div(v))*dx + omega_ind * ufl.inner(ue,v) * dx 
        else:
            L += omega_ind * ufl.inner(ue,v) * dx 
            #L += ufl.inner(omega_ind*ue,v) * dx 

    prob = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu","pc_factor_mat_solver_type":"mumps" }) 
    sol = prob.solve()
    uh,zh = sol.split() 
    
    if export_VTK:
        uex = fem.Function(V0)
        u_expr = fem.Expression(ue, V0.element.interpolation_points())
        uex.interpolate(u_expr)
        udiff = fem.Function(V0)
        udiff.x.array[:] = np.abs(uh.x.array[V0_to_VW] - uex.x.array)
        uh.name ="uh"
        #zh.name ="zh"
        with io.XDMFFile(msh.comm, "diff-order{0}-ndof{1}.xdmf".format(order,ndof), "w") as xdmf:
            xdmf.write_mesh(msh)
            #xdmf.write_function(uh)
            xdmf.write_function(udiff)
        with io.XDMFFile(msh.comm, "uh-sol.xdmf", "w") as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(uh)
        with io.XDMFFile(msh.comm, "u-exact-sol.xdmf", "w") as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(uex)


    # Postprocessing: measure errors #
    
    # workaround for 4.,5. and 6. as we cannot use uh and Phu in form together (dof map)
    uh_V0 = fem.Function(V0)
    uh_V0.x.array[:] = uh.x.array[V0_to_VW] 
    
    # 1. L2-error-u-uh-B 
    L2_error_B = fem.form( B_ind*ufl.inner(uh - ue, uh - ue) * dx)
    error_local = fem.assemble_scalar(L2_error_B)
    L2_norm = fem.form( B_ind*ufl.inner(ue, ue) * dx) 
    L2_local = fem.assemble_scalar(L2_norm)
    error_L2_abs = np.sqrt(msh.comm.allreduce(error_local, op=MPI.SUM)) 
    error_dict["L2-error-u-uh-B-absolute"] = error_L2_abs
    error_L2 = error_L2_abs / np.sqrt(msh.comm.allreduce(L2_local, op=MPI.SUM)) 
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
    #Tikh_error_local = fem.form( palpha * h**(2*order) * ufl.inner(ufl.nabla_grad(uh_V0-Phu),ufl.nabla_grad(uh_V0-Phu)) * dx )
    Tikh_error = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(Tikh_error_local), op=MPI.SUM))
    error_dict["Tikh-uh-Ph(u)-Omega"] = Tikh_error 

    # 7. L2-error-uh-Ph(u)-omega
    L2_error_omega_local = fem.form( omega_ind * ufl.inner(uh_V0 - Phu, uh_V0 - Phu) * dx ) 
    L2_error_omega = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(L2_error_omega_local), op=MPI.SUM))
    error_dict["L2-error-uh-Ph(u)-omega"] = L2_error_omega 

    # 8. s-norm 
    #if MPI.COMM_WORLD.rank == 0:
    error_dict["s-norm"] = np.sqrt( error_dict["Jump-uh-Ph(u)"]**2 + error_dict["GLS-uh-Ph(u)-Omega"]**2 
                                    + error_dict["Tikh-uh-Ph(u)-Omega"]**2 + error_dict["L2-error-uh-Ph(u)-omega"]**2 
                                    + error_dict["H1-semi-norm-zh-Omega"]**2 )
    # 9. ndofs
    error_dict["ndof"] = ndof
    
    # 10. h 
    #error_dict["hmax"] = order/np.array(ndof)**(1/2) 
    error_dict["hmax"] = get_h_max(msh)
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


def ConvergenceStudy(problem,ls_mesh,refsol,order=1,pgamma=1e-5,palpha=1e-5,add_bc=False,export_VTK=False,rhs=None,mu_Ind=None,perturb_theta=None,pGLS=None,name_str="dummy.txt",compute_cond=False,GradTikhStab=False,div_known=False):
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
               "reflvl":[],
               "cond":[]
             }

    for msh in ls_mesh:
        if perturb_theta!=None:
            errors_msh = SolveProblem(problem = problem, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,perturb_order=order-perturb_theta,pGLS=pGLS,compute_cond=compute_cond,GradTikhStab=GradTikhStab,div_known=div_known)
        else:
            errors_msh = SolveProblem(problem = problem, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=order==2,pGLS=pGLS,compute_cond=compute_cond,GradTikhStab=GradTikhStab,div_known=div_known)
        if MPI.COMM_WORLD.rank == 0:
            print("ndof = {0}, L2-error-u-uh-B = {1}".format(errors_msh["ndof"],errors_msh["L2-error-u-uh-B"])) 
        for error_type in errors_msh:
            if error_type in errors.keys():
                errors[error_type].append(errors_msh[error_type])
    
    # 11 reflvl 
    errors["reflvl"] = range(len(errors["hmax"])) 
    
    #results = [np.array(errors_msh["ndof"]]
    #header_str = "ndofs "
    
    if MPI.COMM_WORLD.rank == 0:
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






