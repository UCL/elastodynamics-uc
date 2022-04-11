import gmsh
import numpy as np

from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, assemble_scalar,
                         form, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square, locate_entities, refine, compute_incident_entities
from dolfinx.plot import create_vtk_mesh

from ufl import (SpatialCoordinate, TestFunction, TrialFunction,
                 dx, grad, inner,And,Not,conditional)

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

def get_mesh_hierarchy(n_ref): 

    gmsh.initialize()
    proc = MPI.COMM_WORLD.rank
    top_marker = 2
    bottom_marker = 1
    bnd_marker = 1
    omega_marker = 1
    Bwithoutomega_marker = 2
    rest_marker = 3 
    if proc == 0:
        # We create one rectangle for each subdomain

        r1 = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1,tag=1)
        r2 = gmsh.model.occ.addRectangle(0.1, 0.25, 0, 0.8, 0.75,tag=2)
        r3 = gmsh.model.occ.cut( [(2,r1)], [(2,r2)],tag=3)

        print("r3 = ", r3)
        #gmsh.model.occ.addRectangle(0.1, 0.1, 0, 0.8, 0.9,tag=4)
        gmsh.model.occ.addRectangle(0.1, 0.25, 0, 0.8, 0.7,tag=4)

        # We fuse the two rectangles and keep the interface between them
        gmsh.model.occ.fragment([(2,3)],[(2,4)])

        tmp = gmsh.model.occ.addRectangle(0.1, 0.95, 0, 0.8, 0.05,tag=6)
        #print("tmp = ", tmp)
        gmsh.model.occ.fragment([(2,3)],[(2,4),(2,tmp)] )

        gmsh.model.occ.synchronize()

        #for surface in gmsh.model.getEntities(dim=2):
        #    gmsh.model.addPhysicalGroup(2, [surface[1]], 1)

        # Mark the top (2) and bottom (1) rectangle
        #top, bottom = None, None
        print(len(gmsh.model.getEntities(dim=2)))
        its = 0 
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            print(com)
            if np.allclose(com, [0.5,0.6, 0]):
                gmsh.model.addPhysicalGroup(2, [surface[1]], Bwithoutomega_marker)
            elif np.allclose(com, [0.5,0.975, 0]):
                gmsh.model.addPhysicalGroup(2, [surface[1]], rest_marker)
            else:
                gmsh.model.addPhysicalGroup(2, [surface[1]], omega_marker)
        #    if np.allclose(com, [0.5,0.25, 0]):
        #        bottom = surface[1]
        #    else:
        #        top = surface[1]
        #gmsh.model.addPhysicalGroup(2, [bottom], bottom_marker)
        #gmsh.model.addPhysicalGroup(2, [top], top_marker)
        # Tag the left boundary
        bnd_square = []
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if np.isclose(com[0], 0) or np.isclose(com[0], 1) or np.isclose(com[1], 0) or  np.isclose(com[1], 1): 
                bnd_square.append(line[1])
        gmsh.model.addPhysicalGroup(1, bnd_square, bnd_marker)
        gmsh.model.mesh.generate(2)
        gmsh.write("mesh.msh")
        gmsh.finalize()

    import meshio
    def create_mesh(mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points[:,:2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
        return out_mesh

    if proc == 0:
        # Read in mesh
        msh = meshio.read("mesh.msh")

        # Create and save one file for the mesh, and one file for the facets
        triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
        line_mesh = create_mesh(msh, "line", prune_z=True)
        meshio.write("mesh.xdmf", triangle_mesh)
        meshio.write("mt.xdmf", line_mesh)

    #n_ref = 2 
    #for i in range(n_ref): 

    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(mesh, name="Grid")
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
    with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
        ft = xdmf.read_meshtags(mesh, name="Grid")

    mesh_hierarchy = [] 
    mesh_hierarchy.append(mesh) 

    def refine_all(x):
        return x[0] >= 0
    for i in range(n_ref):
        mesh.topology.create_entities(1)
        cells = locate_entities(mesh, mesh.topology.dim, refine_all)
        #print(cells)
        if proc == 0:
            edges = compute_incident_entities(mesh, cells, 2, 1)
            print(edges)
            mesh = refine(mesh, edges, redistribute=True)
            mesh_hierarchy.append(mesh) 
    return mesh_hierarchy

'''
ls_mesh = get_mesh_hierarchy(6)
tol = 1e-13
x_l = 0.1-tol
x_r = 0.9+tol
y_b = 0.25-tol
y_t = 1.0+tol
def omega_Ind(x):
    #tmp =  Not(And(And(x[0] >= 0.1, x[0] <= 0.9),And(x[1] >= 0.25, x[1] <= 1))   )
    #tmp =  ~( ( x[0] > (0.1 - tol) ) & ( x[0] < (0.9+tol) ) & ( x[1]>(0.25-tol) ) & ( x[1]<(1.0+tol) ) ) 
    #tmp =  ~( ( x[0] > x_l ) & ( x[0] < x_r ) & ( x[1]>y_b ) & ( x[1]< y_t ) ) 
    tmp = ( x[0] < (0.1+tol) ) | ( x[0] > (0.9-tol) ) | ( x[1] < ( 0.25+tol ) ) 
    #tmp =  ( ( x[1] >= (0.5) )  ) 
    return tmp

def B_Ind(x):
    #return ~( ( x[0] > (0.1 - tol) ) & ( x[0] < (0.9+tol) ) & ( x[1]>(0.95-tol) ) & ( x[1]<(1+tol) ) ) 
    return  (x[0] > (0.1-tol) ) &  (x[0] < (0.9+tol) ) & (x[1] > (0.25-tol) ) &  (x[1] < (0.95+tol) ) 

for idx,mesh in enumerate(ls_mesh):

    Q_ind = FunctionSpace(mesh, ("DG", 0))

    omega_ind = Function(Q_ind)
    cells_omega = locate_entities(mesh, mesh.topology.dim, omega_Ind)
    omega_ind.x.array[:] = 0.0
    omega_ind.x.array[cells_omega] = np.full(len(cells_omega), 1)

    B_ind = Function(Q_ind)
    cells_B = locate_entities(mesh, mesh.topology.dim, B_Ind)
    B_ind.x.array[:] = 0.0
    B_ind.x.array[cells_B] = np.full(len(cells_B), 1)

    with XDMFFile(mesh.comm, "omega-ind-reflvl{0}.xdmf".format(idx), "w") as file:
        file.write_mesh(mesh)
        file.write_function(omega_ind)

    with XDMFFile(mesh.comm, "B-ind-reflvl{0}.xdmf".format(idx), "w") as file:
        file.write_mesh(mesh)
        file.write_function(B_ind)
'''
#V = FunctionSpace(mesh, ("CG", 1))
#u_bc = Function(V)
#left_facets = ft.indices[ft.values==left_marker]
#left_dofs = locate_dofs_topological(V, mesh.topology.dim-1, left_facets)
#bcs = [dirichletbc(ScalarType(1), left_dofs, V)]



