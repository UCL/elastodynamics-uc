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
    print("cond = " , cond_est)
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


def SolveProblem(problem,msh,refsol,order=1,pgamma=1e-5,palpha=1e-5,add_bc=False,export_VTK=False,rhs=None,mu_Ind=None,k_Ind=None,perturb_order=None,pGLS=None,compute_cond=False,GradTikhStab=False,div_known=False,gamma_CIP_primal = 1e-18): 

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
            a += gamma_CIP_primal * hbar**3 * ufl.inner(ufl.jump( ufl.grad(sigma(u)),n),ufl.jump(ufl.grad(sigma(w)),n))*ufl.dS 
            a += gamma_CIP_primal * hbar**3 * ufl.inner(ufl.jump( ufl.grad(sigma(v)),n),ufl.jump(ufl.grad(sigma(z)),n))*ufl.dS 
            #a += gamma_CIP_primal * hbar**3 * ufl.inner(ufl.jump(ufl.grad(ufl.grad(u)),n),ufl.jump(ufl.grad(ufl.grad(w)),n))*ufl.dS 
            #a += gamma_CIP_primal * hbar**3 * ufl.inner(ufl.jump(ufl.grad(ufl.grad(u))),ufl.jump(ufl.grad(ufl.grad(w))))*ufl.dS 
            #a += gamma_CIP_primal * hbar * ufl.inner(ufl.jump(ufl.grad(u),n),ufl.jump(ufl.grad(w),n))*ufl.dS 
        #if order == 3:
            #a += gamma_CIP_primal * 1e-2 * hbar**5 * ufl.inner(ufl.jump(  ufl.grad(ufl.grad(sigma(u))),n),ufl.jump(ufl.grad(ufl.grad(sigma(w))),n))*ufl.dS 
            #a += gamma_CIP_primal * hbar**5 * ufl.inner(ufl.jump( ufl.grad(ufl.grad(ufl.grad(u))),n),ufl.jump(ufl.grad(ufl.grad(ufl.grad(w))),n))*ufl.dS 
            #a += gamma_CIP_primal * 1e2 * hbar**5 * ufl.inner(ufl.jump( ufl.grad(ufl.grad(ufl.grad(u)))),ufl.jump(ufl.grad(ufl.grad(ufl.grad(w)))))*ufl.dS 

    
    if compute_cond:  
        cond_est =  condest(a,bcs,msh,target_shift=1e-13)
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
        u_expr = fem.Expression(ue, V0.element.interpolation_points)
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
    
    #for add_bc,problem_type,pgamma,palpha in zip([False],["ill-posed"],[ScalarType(tmp_gamma)], [ScalarType(1e-3)]):
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

def RunProblemConvexOscillatoryKhscaling(gamma_CIP_primal = 0,gamma_CIP_primal_str="gamma-CIP-primal-0"):
    
    orders = [1,2] 
    #orders = [1,2,3] 
    #ratio = 1/2
    ratio = 1
    #kks = [1+2*j for j in range(4)]
    
    kks = [1+j for j in range(10)]
    #kks = [1+j for j in range(8)]
    
    #kks = [1+j for j in range(5)]
    kks_np = np.array(kks)
    
    meshwidths = [ ratio/kk for kk in kks   ] 
    print("meshwidth = ", meshwidths )
    ls_mesh = [ create_initial_mesh_convex(init_h_scale = h_k) for h_k in meshwidths ]
    #ls_mesh = get_mesh_convex(6,init_h_scale=1.0)   
    
    elastic_convex.mu = mu_const
    elastic_convex.lam = lam_const

    #for str_param in ["tuned","naive"]:
    for str_param in ["tuned"]:
        #for add_bc,problem_type,pgamma,palpha,pGLS in zip([True,False],["well-posed","ill-posed"], [ ScalarType(1e-4),ScalarType(1e-4) ], 
        #                                                 [ ScalarType(5e-1),ScalarType(5e-1)],[ ScalarType(1e-4),ScalarType(1e-4)]   ):
        for add_bc,problem_type,pgamma,palpha,pGLS in zip([True,False],["well-posed","ill-posed"], [ ScalarType(1e-5),ScalarType(1e-5) ], 
                                                         [ ScalarType(1e-3),ScalarType(1e-3)],[ ScalarType(1e-5),ScalarType(1e-5)]   ):
        #for add_bc,problem_type,pgamma,palpha,pGLS in zip([False],["ill-posed"], [ ScalarType(1e-5) ], 
        #                                                 [ ScalarType(1e-3)],[ ScalarType(1e-5)]   ):
            print("Considering {0} problem".format(problem_type))
            l2_errors_order = { }
            h1_semi_errors_order = { }
            k_norm_error = { } 
            for order in orders:
                l2_errors = []
                h1_semi_errors = [] 
                k_norm_errors = [] 
                for kk,msh in zip(kks,ls_mesh):
                    print("kk = {0}".format(kk)) 
                    elastic_convex.rho = get_ksquared_const(kk)
                    refsol = get_reference_sol("oscillatory",kk=kk)
                    if str_param == "tuned":
                        #errors = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma/(order*kk)**2,palpha=palpha,add_bc=add_bc,export_VTK=False,pGLS=pGLS/kk**4)  
                        errors = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma/(order)**3.5,palpha=palpha,add_bc=add_bc,export_VTK=False,pGLS=pGLS/order**3.5, 
                                   gamma_CIP_primal = gamma_CIP_primal )             
                    else:
                        errors = SolveProblem(problem = elastic_convex, msh = msh,refsol=refsol,order=order,pgamma=pgamma,palpha=palpha,add_bc=add_bc,export_VTK=False,pGLS=pGLS)
                    l2_error = errors["L2-error-u-uh-B"]
                    h1_semi_error = errors["H1-semi-error-u-uh-B-absolute"]
                    ndof = errors["ndof"]     
                    l2_errors.append(l2_error)
                    h1_semi_errors.append(h1_semi_error)
                    k_norm_errors.append(  kk*errors["L2-error-u-uh-B-absolute"] + errors["H1-semi-error-u-uh-B-absolute"] ) 
                l2_errors_order[order] = l2_errors
                h1_semi_errors_order[order] = h1_semi_errors
                k_norm_error[order] =  k_norm_errors 

            if MPI.COMM_WORLD.rank == 0:
                name_str = "Convex-Oscillatory-kh-scaling-{0}-{1}-{2}.dat".format(problem_type,str_param, gamma_CIP_primal_str)
                results = [kks_np]
                header_str = "k "
                for order in orders: 
                    #results.append(np.array(h1_semi_errors_order[order],dtype=float))
                    results.append(np.array(k_norm_error[order],dtype=float))
                    header_str += "k-norm-order{0} ".format(order)
                np.savetxt(fname ="../data/{0}".format(name_str),
                           X = np.transpose(results),
                           header = header_str,
                           comments = '')
            

            #if MPI.COMM_WORLD.rank == 0:
            #    name_str = "Convex-Oscillatory-kh-scaling-{0}-{1}.dat".format(problem_type,str_param)
            #    results = [kks_np]
            #    header_str = "k "
            #    for order in orders: 
            #        results.append(np.array(l2_errors_order[order],dtype=float))
            #        header_str += "l2-order{0} ".format(order)
            #        results.append(np.array(h1_semi_errors_order[order],dtype=float))
            #        header_str += "h1-semi-order{0} ".format(order)
            #    np.savetxt(fname ="../data/{0}".format(name_str),
            #               X = np.transpose(results),
            #               header = header_str,
            #               comments = '')

                for order,lstyle in zip(orders,['solid','dashed','dotted']):
                    #plt.loglog(kks_np, l2_errors_order[order] ,'-x',label="p={0}".format(order),linewidth=3,markersize=8)
                    plt.loglog(kks_np,k_norm_error[order],'-x',label="p={0}".format(order),linewidth=3,markersize=8)
                
                    #tmp_str = "$\mathcal{{O}}(k^{0})$".format(order+1)
                    #plt.loglog(kks_np, 1.35*l2_errors_order[order][0]*(kks_np**(order+1))/( kks_np[0]**(order+1)) ,label=tmp_str,linestyle=lstyle,color='gray')

                plt.loglog(kks_np, 1.35*k_norm_error[1][0]*(kks_np)/( kks_np[0]) ,label="$\mathcal{O}(k)$",linestyle=lstyle,color='gray')
                plt.loglog(kks_np, 1.0*k_norm_error[2][0]*(kks_np**2)/( kks_np[0]**2 ) ,label="$\mathcal{O}(k^2)$",linestyle=lstyle,color='gray')
                #plt.loglog(kks_np, 1.0*k_norm_error[3][0]*(kks_np**2)/( kks_np[0]**2 ) ,label="$\mathcal{O}(k^2)$",linestyle=lstyle,color='gray')
                #plt.loglog(kks_np, 1.35*k_norm_error[3][0]*(kks_np)/( kks_np[0]) ,label="$\mathcal{O}(k)$",linestyle=lstyle,color='gray')
                
                #plt.loglog(kks_np, 1.35*l2_errors_order[2][0]*(kks_np)/( kks_np[0]) ,label="$\mathcal{O}(k)$",linestyle=lstyle,color='gray')
                plt.xlabel("$k$")
                plt.ylabel("L2-error")
                plt.legend()
                plt.savefig("L2-error-k-Gaussian.png",transparent=True,dpi=200)
                #plt.title("L2-error")
                plt.show()


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


def RunProblemSplitGeom(kk=1,apgamma=1e-1,apalpha=1e-1,compute_cond=True,div_known=False): 
    eta = 0.6
    
    #def omega_Ind_eta(x):
        
    #    values = np.zeros(x.shape[1],dtype=ScalarType)
    #    omega_coords = np.logical_or(  np.logical_and(  x[0] <= 0.1 , x[1] <=eta  ) , 
    #        np.logical_or(  np.logical_and( x[0] >= 0.9 , x[1] <= eta  ) , (x[1] <= 0.25)  )
    #        ) 
    #    #omega_coords = np.logical_or(   ( x[0] <= 0.1 )  , 
    #    #    np.logical_or(   (x[0] >= 0.9 ), (x[1] <= 0.25)  )
    #    #    ) 
    #    rest_coords = np.invert(omega_coords)
    #    values[omega_coords] = np.full(sum(omega_coords), 1.0)
    #    values[rest_coords] = np.full(sum(rest_coords), 0)
    #    return values

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
    elastic_convex.mu = mu_const
    elastic_convex.lam = lam_const
    ls_mesh = get_mesh_hierarchy_fitted_disc(6,eta=eta)
    #mu_plus = 1
    #mu_plus = 1
    #mu_minus = 2
    #refsol,rhs = get_reference_sol(type_str="jump",kk=kk,eta=eta,mu_plus=mu_plus,mu_minus=mu_minus,lam=elastic_convex.lam)
    refsol = get_reference_sol("oscillatory",kk=kk)
    
    #def mu_Ind(x):
    #    values = np.zeros(x.shape[1],dtype=ScalarType)
    #    upper_coords = x[1] > eta 
    #    lower_coords = np.invert(upper_coords)
    #    values[upper_coords] = np.full(sum(upper_coords), mu_plus)
    #    values[lower_coords] = np.full(sum(lower_coords), mu_minus)
    #    return values
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

    
    #for add_bc,problem_type,pgamma,palpha,pGLS in zip([True,False],["well-posed","ill-posed"],[ ScalarType(apgamma)/ (kk**2), ScalarType(apgamma)/ (kk**2) ],[ ScalarType(apalpha), ScalarType(apalpha)],  [ ScalarType(1e-2/kk**4), ScalarType(1e-2/kk**4) ] ):   
    #for add_bc,problem_type,pgamma,palpha,pGLS in zip([False],["ill-posed"],[ScalarType(apgamma)/ (kk**2) ],[ ScalarType(apalpha)],  [ ScalarType(1e-2/kk**4) ] ):
    for add_bc,problem_type,pgamma,palpha,pGLS in zip([False],["ill-posed"],[ScalarType(apgamma) ],[ ScalarType(apalpha)],  [ ScalarType(apgamma) ] ):

        print("Considering {0} problem".format(problem_type))
        l2_errors_order = { }
        eoc_order = { }
        h_order = { }
        L2_error_B_plus_order = { } 
        L2_error_B_minus_order = { } 
        s_errors_order = { }
        cond_order = { } 
        for order in orders:
            l2_errors = [ ]
            L2_error_B_plus = [] 
            L2_error_B_minus = [] 
            s_errors = []
            cond = [] 
            ndofs = [] 
            for msh in ls_mesh[:-order]:
                #errors = SolveProblem(problem=elastic_convex,msh=msh,refsol=refsol,order=order,pgamma=pgamma/order**2,palpha=palpha,add_bc=add_bc,export_VTK=order==3,pGLS=pGLS,compute_cond=compute_cond)
                errors = SolveProblem(problem=elastic_convex,msh=msh,refsol=refsol,order=order,pgamma=pgamma/order**3.5,palpha=palpha,add_bc=add_bc,export_VTK=True,pGLS=pGLS/order**3.5,compute_cond=compute_cond,div_known= div_known)
                l2_error = errors["L2-error-u-uh-B"]
                ndof = errors["ndof"]  
                print("ndof = {0}, L2-error-B = {1}".format(ndof,l2_error))
                L2_error_B_plus.append(errors["L2-error-u-uh-B-plus"] ) 
                L2_error_B_minus.append(errors["L2-error-u-uh-B-minus"] ) 
                l2_errors.append(l2_error)
                s_errors.append(errors["s-norm"])
                ndofs.append(ndof)
                cond.append(errors["cond"]) 

            eoc = [ log(l2_errors[i-1]/l2_errors[i])/log(2) for i in range(1,len(l2_errors))]
            print("eoc = ", eoc)
            ndofs = np.array(ndofs) 
            h_mesh = order/ndofs**(1/2)
            idx_start = 2 
            rate_estimate, _ = np.polyfit(np.log(h_mesh)[idx_start:] , np.log(l2_errors)[idx_start:], 1)
    
            for error_type,error_str in zip([ L2_error_B_minus, L2_error_B_plus,s_errors],["L2-error-u-uh-B-minus","L2-error-u-uh-B-plus","s-norm"]):
                #print(error_str)
                eoc = [ log(error_type[i-1]/error_type[i])/log(2) for i in range(1,len(error_type ))]
                print("{0}, eoc = {1}".format(error_str,eoc))


            l2_errors_order[order] =  l2_errors 
            L2_error_B_plus_order[order] = L2_error_B_plus
            L2_error_B_minus_order[order] = L2_error_B_minus
            h_order[order] = h_mesh
            eoc_order[order] = round(eoc[-1],2)
            s_errors_order[order] = s_errors 
            
            if MPI.COMM_WORLD.rank == 0:
                if div_known: 
                    name_str = "SplitGeom-{0}-k{1}-order{2}-divknown.dat".format(problem_type,kk,order)
                else:
                    name_str = "SplitGeom-{0}-k{1}-order{2}.dat".format(problem_type,kk,order)
                results = [np.array(ndofs,dtype=float),np.array(h_order[order],dtype=float)]
                header_str = "ndof h "
                if cond[0] != None:
                    for error_type,error_str in zip([ L2_error_B_minus, L2_error_B_plus,s_errors,cond],["L2-error-u-uh-B-minus","L2-error-u-uh-B-plus","s-norm","cond"]):
                        results.append( np.array(error_type,dtype=float))
                        header_str += "{0} ".format(error_str)
                else:
                    for error_type,error_str in zip([ L2_error_B_minus, L2_error_B_plus,s_errors],["L2-error-u-uh-B-minus","L2-error-u-uh-B-plus","s-norm"]):
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
            plt.savefig("L2-error-SplitGeom-jump-{0}-k{1}.png".format(problem_type,kk),transparent=True,dpi=200)
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

def RunProblemJumpInclDataBottom(kk=1,apgamma=1e-5,apalpha=1e-3,mu_plus=1,mu_minus=2): 
    
    eta = 0.6
    
    x_L = 0.25
    x_R = 0.75
    y_L = 0.25
    y_R = 0.9
    def omega_Ind(x):     
        values = np.zeros(x.shape[1],dtype=ScalarType)
        omega_coords = x[1] <= y_L 
        rest_coords = np.invert(omega_coords)
        values[omega_coords] = np.full(sum(omega_coords), 1.0)
        values[rest_coords] = np.full(sum(rest_coords), 0)
        return values

    def B_Ind(x):
        values = np.zeros(x.shape[1],dtype=ScalarType)
        # Create a boolean array indicating which dofs (corresponding to cell centers)
        # that are in each domain
        B_coords = np.logical_and( ( x[0] >= x_L ), 
            np.logical_and(   (x[0] <= x_R ),
              np.logical_and(   (x[1]>= y_L),  (x[1]<= y_R)  )
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

    DrawMeshTikz(msh=ls_mesh[1],name="BottomDataJumpIncl",case_str="BottomDataJumpIncl") 
    
    mu_plus = mu_plus
    #mu_plus = 1
    mu_minus = mu_minus

    
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

    #refsol = get_reference_sol("oscillatory",kk=kk)
    refsol = get_reference_sol(type_str="jump-square",kk=kk,mu_plus=mu_plus,mu_minus=mu_minus,lam=elastic_convex.lam(1),x_L=x_L,x_R=x_R,y_L=y_L,y_R=y_R)
    #refsol,rhs = get_reference_sol(type_str="jump",kk=kk,eta=eta,mu_plus=mu_plus,mu_minus=mu_minus,lam=elastic_convex.lam(1))
    #def mu_Ind(x):
    #    values = np.zeros(x.shape[1],dtype=ScalarType)
    #    upper_coords = x[1] > eta 
    #    lower_coords = np.invert(upper_coords)
    #    values[upper_coords] = np.full(sum(upper_coords), mu_plus)
    #    values[lower_coords] = np.full(sum(lower_coords), mu_minus)
    #    return values
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
                errors = SolveProblem(problem=elastic_convex,msh=msh,refsol=refsol,order=order,pgamma=pgamma/order**3.5,palpha=palpha,add_bc=add_bc,export_VTK=True,mu_Ind=mu_Ind,pGLS=pGLS/order**3.5)
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
                name_str = "jump-incl-DataBottom-mup{0}-mum{1}-{2}-k{3}-order{4}.dat".format(mu_plus,mu_minus,problem_type,kk,order)
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
            plt.savefig("L2-error-jump-incl-data-bottom{0}-k{1}.png".format(problem_type,kk),transparent=True,dpi=200)
            #plt.title("L2-error")
            plt.show()

# pgamma = 1e-5/kk**2 , pGLS = 1e-4/kk**4 
# Runs for draft
# 
# RunProblemConvexOscillatoryStabSweep(kk=6,compute_cond=True) # Figure 2
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

#RunProblemConvexOscillatoryKhscaling(gamma_CIP_primal = 1e-4,gamma_CIP_primal_str="gamma-CIP-primal-0p0001")
#RunProblemConvexOscillatoryKhscaling(gamma_CIP_primal = -5e-2,gamma_CIP_primal_str="gamma-CIP-primal-m0p05")
#RunProblemConvexOscillatoryKhscaling(gamma_CIP_primal = 5e-2,gamma_CIP_primal_str="gamma-CIP-primal-0p05")
#RunProblemConvexOscillatoryKhscaling(gamma_CIP_primal = 1e-1,gamma_CIP_primal_str="gamma-CIP-primal-0p1")

#RunProblemJumpEtaDataBottom(kk=4,apgamma=1e-4,apalpha=1e-3,mu_plus=2,mu_minus=1)
#RunProblemJumpInclDataBottom(kk=1,apgamma=1e-5,apalpha=1e-3,mu_plus=.5,mu_minus=1)
