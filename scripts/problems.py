import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem, plot, io
from math import pi,log
from meshes import get_mesh_hierarchy, get_mesh_hierarchy_nonconvex

class elastodynamics(): 
    def __init__(self,lam,mu,rho):
        self.lam = lam 
        self.mu = mu 
        self.rho = rho 
        #self.omega_Ind = None
        #self.B_Ind = None 
    def SetSubdomains(self,omega_Ind,B_Ind):
        self.omega_Ind = omega_Ind 
        self.B_Ind = B_Ind 
    def SetBoundaryIndicator(self,bnd_Ind): 
        self.boundary_indicator = bnd_Ind

def omega_Ind_convex(x):
    
    values = np.zeros(x.shape[1],dtype=ScalarType)
    omega_coords = np.logical_or( ( x[0] <= 0.1 ), 
        np.logical_or(   (x[0] >= 0.9 ), (x[1] <= 0.25)  )
        ) 
    rest_coords = np.invert(omega_coords)
    values[omega_coords] = np.full(sum(omega_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    return values

def B_Ind_convex(x):
    values = np.zeros(x.shape[1],dtype=ScalarType)
    # Create a boolean array indicating which dofs (corresponding to cell centers)
    # that are in each domain
    rest_coords = np.logical_and( ( x[0] >= 0.1 ), 
        np.logical_and(   (x[0] <= 0.9 ),
          np.logical_and(   (x[1]>= 0.95),  (x[1]<= 1)  )
        )
      ) 
    B_coords = np.invert(rest_coords)
    values[B_coords] = np.full(sum(B_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    return values

def omega_Ind_nonconvex(x):
    
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

def B_Ind_nonconvex(x):
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

kk = 1
elastic_convex = elastodynamics(lam=1.25,mu=1,rho=kk**2)
elastic_convex.SetSubdomains(omega_Ind=omega_Ind_convex,B_Ind=B_Ind_convex)
def boundary_indicator_unit_square(x):
    return ( np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0) )
elastic_convex.SetBoundaryIndicator(boundary_indicator_unit_square)

kk = 1
elastic_nonconvex = elastodynamics(lam=1.25,mu=1,rho=kk**2)
elastic_nonconvex.SetSubdomains(omega_Ind=omega_Ind_nonconvex,B_Ind=B_Ind_nonconvex)
elastic_nonconvex.SetBoundaryIndicator(boundary_indicator_unit_square)


