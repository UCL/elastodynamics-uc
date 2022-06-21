import gmsh
import numpy as np

from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, assemble_scalar,
                         form, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square, locate_entities, refine, compute_incident_entities, GhostMode
from dolfinx.plot import create_vtk_mesh

from ufl import (SpatialCoordinate, TestFunction, TrialFunction,
                 dx, grad, inner,And,Not,conditional)

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from math import pi
from dolfinx.io import XDMFFile, ufl_mesh_from_gmsh,extract_gmsh_geometry,extract_gmsh_topology_and_markers,cell_perm_gmsh
from dolfinx.cpp.io import perm_gmsh
from dolfinx.mesh import CellType, create_mesh
from dolfinx.cpp import mesh as cmesh
#help(CellType)

GM = GhostMode.shared_facet
#GM = GhostMode.none
eta = 0.6

def DrawMeshTikz(msh,name,case_str="dummy"): 
    
    ddx = 10
    file = open("../plots/{0}.tex".format(name),"w+")
    file.write("\\documentclass{standalone} \n")
    file.write("\\usepackage{xr} \n")
    file.write("\\usepackage{tikz} \n")
    file.write("\\usepackage{xcolor} \n")
    file.write("\\usepackage{} \n")   
    file.write("\\usetikzlibrary{shapes,arrows,shadows,snakes,calendar,matrix,spy,backgrounds,folding,calc,positioning,patterns} \n")
    file.write("\\begin{document} \n")
    file.write("\\begin{tikzpicture}[scale = 1.0] \n")

    mmap = msh.topology.index_map(msh.topology.dim)
    num_cells = mmap.size_local + mmap.num_ghosts
    for i in range(num_cells):
        # Get indices of cell vertices
        vertex_global_indices = msh.topology.connectivity(msh.topology.dim, 0).links(i)
        #print("vertex_global_indices = ", vertex_global_indices) 
        coords = []
        for idx in vertex_global_indices: 
            #print("idx = {0}, vertex coord = {1}".format(idx,msh.geometry.x[idx])) 
            coords.append(msh.geometry.x[idx])
        file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=1] {1} -- {2} -- {3} -- cycle; \n".format("white", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
        
        if case_str == "convex-Mihai-omega":
            def is_in_dom(coord):
                if coord[0] <= 0.1:
                    return True 
                elif coord[0] >= 0.9:
                    return True 
                elif coord[1] <= 0.25:
                    return True
                else:
                    return False 
            el_in_domain = np.all(np.array([is_in_dom(coord) for coord in coords ]))
            #print("el_in_domain = ", el_in_domain)  
            if el_in_domain:
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.5] {1} -- {2} -- {3} -- cycle; \n".format("cyan", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
            else:
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.2] {1} -- {2} -- {3} -- cycle; \n".format("gray", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
            file.write("\\node (RL) at ({0},{1}) [fill=white,fill opacity=1.0,inner sep = 2.5pt] {{ \\resizebox{{ .125\\linewidth}}{{!}}{{  \\textcolor{{black}}{{$\omega$}}  }} }}; \n".format(0.5*ddx,0.125*ddx))
      
        
        if case_str == "convex-Mihai-B": 
            tol = 1e-5
            def is_in_dom(coord):
                if ( coord[0] >= (0.1+tol) and coord[0] <= (0.9-tol) and coord[1] >= (0.95+tol) and coord[1] <= (1.0-tol) ): 
                    return False 
                else:
                    return True 
            el_in_domain = np.all(np.array([is_in_dom(coord) for coord in coords ]))
            #print("el_in_domain = ", el_in_domain)  
            if el_in_domain:
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.5] {1} -- {2} -- {3} -- cycle; \n".format("cyan", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
            else:
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.2] {1} -- {2} -- {3} -- cycle; \n".format("gray", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
            file.write("\\node (RL) at ({0},{1}) [fill=white,fill opacity=1.0,inner sep = 2.5pt] {{ \\resizebox{{ .125\\linewidth}}{{!}}{{  \\textcolor{{black}}{{$B$}}  }} }}; \n".format(0.5*ddx,0.45*ddx))
            
        if case_str == "splitgeom-omega":
            def is_in_dom(coord):
                if (coord[0] <= 0.1 and coord[1] <= eta):
                    return True 
                elif (coord[0] >= 0.9 and coord[1] <= eta):
                    return True 
                elif (coord[1] <= 0.25):
                    return True
                else:
                    return False 
            el_in_domain = np.all(np.array([is_in_dom(coord) for coord in coords ]))
            #print("el_in_domain = ", el_in_domain)  
            if el_in_domain:
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.5] {1} -- {2} -- {3} -- cycle; \n".format("cyan", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
            else:
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.2] {1} -- {2} -- {3} -- cycle; \n".format("gray", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
            file.write("\\node (RL) at ({0},{1}) [fill=white,fill opacity=1.0,inner sep = 2.5pt] {{ \\resizebox{{ .125\\linewidth}}{{!}}{{  \\textcolor{{black}}{{$\omega$}}  }} }}; \n".format(0.5*ddx,0.125*ddx))

        if case_str == "splitgeom-B":
            tol = 1e-5

            def is_in_Bplus(coord):
                if (coord[0] >= 0.1 and coord[0] <= 0.9 and coord[1] >= eta and coord[1] <= 0.95): 
                    return True
                else:
                    return False 
                #if ( coord[0] >= (0.1+tol) and coord[0] <= (0.9-tol) and coord[1] >= (0.95+tol)  ): 
                #    return False 
                #else:
                #    if coord[1] >= eta:
                #        return True
                #    else:
                #        return False
            def is_in_Bminus(coord):
                if ( coord[1] <= eta ):
                    return True
                else:
                    return False 
                #if ( coord[0] >= (0.1+tol) and coord[0] <= (0.9-tol) and coord[1] >= (0.95+tol)  ): 
                #    return False 
                #else:
                #    if coord[1] >= (eta+tol):
                #        return False
                #    else:
                #        return True
            el_in_Bplus = np.all(np.array([is_in_Bplus(coord) for coord in coords ]))
            el_in_Bminus = np.all(np.array([is_in_Bminus(coord) for coord in coords ]))
            
            if el_in_Bplus:
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.5] {1} -- {2} -- {3} -- cycle; \n".format("red", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
            elif el_in_Bminus: 
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.5] {1} -- {2} -- {3} -- cycle; \n".format("cyan", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
            else:
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.2] {1} -- {2} -- {3} -- cycle; \n".format("gray", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
            file.write("\\node (Rp) at ({0},{1}) [fill=white,fill opacity=1.0,inner sep = 2.5pt] {{ \\resizebox{{ .125\\linewidth}}{{!}}{{  \\textcolor{{black}}{{$B_+$}}  }} }}; \n".format(0.5*ddx,0.775*ddx))
            file.write("\\node (Rp) at ({0},{1}) [fill=white,fill opacity=1.0,inner sep = 2.5pt] {{ \\resizebox{{ .125\\linewidth}}{{!}}{{  \\textcolor{{black}}{{$B_-$}}  }} }}; \n".format(0.5*ddx,0.425*ddx))

        if case_str == "inclusion-omega":
            eta = 0.25
            def is_in_dom(coord):
                if (coord[0] <= -1.25 and coord[1] <= eta+1e-4):
                    return True 
                elif (coord[0] >= 1.25 and coord[1] <= eta+1e-4):
                    return True 
                elif (coord[1] <= -1.25):
                    return True
                else:
                    return False 
            el_in_domain = np.all(np.array([is_in_dom(coord) for coord in coords ]))
    
            def is_in_incl(coord):
                if (coord[0] >= -0.75 and coord[0] <= 0.75 and coord[1] >= -0.75 and coord[1] <= 0.75    ):
                    return True 
                else:
                    return False 
            el_in_incl = np.all(np.array([is_in_incl(coord) for coord in coords ]))


            #print("el_in_domain = ", el_in_domain)  
            if el_in_domain:
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.5] {1} -- {2} -- {3} -- cycle; \n".format("cyan", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
            elif el_in_incl:
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.5] {1} -- {2} -- {3} -- cycle; \n".format("orange", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
            else:
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.2] {1} -- {2} -- {3} -- cycle; \n".format("gray", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
            file.write("\\node (RL) at ({0},{1}) [fill=white,fill opacity=1.0,inner sep = 2.5pt] {{ \\resizebox{{ .25\\linewidth}}{{!}}{{  \\textcolor{{black}}{{$\omega$}}  }} }}; \n".format(0.0*ddx,-1.375*ddx))

            file.write("\\node (RR) at ({0},{1}) [fill=white,fill opacity=1.0,inner sep = 2.5pt] {{ \\resizebox{{ .25\\linewidth}}{{!}}{{  \\textcolor{{black}}{{$\mu_+$}}  }} }}; \n".format(0.0*ddx,0.0*ddx))
            
        if case_str == "inclusion-B":
            tol = 1e-5
            eta = 0.25
            def is_in_Bplus(coord):
                if (coord[0] >= -1.25 and coord[0] <= 1.25 and coord[1] >= eta and coord[1] <= 1.25): 
                    return True
                else:
                    return False 
            
            def is_in_Bminus(coord):
                if (coord[0] >= -1.25 and coord[0] <= 1.25 and coord[1] >= -1.25 and coord[1] <= eta): 
                    return True
                else:
                    return False 
            el_in_Bplus = np.all(np.array([is_in_Bplus(coord) for coord in coords ]))
            el_in_Bminus = np.all(np.array([is_in_Bminus(coord) for coord in coords ]))
            
            if el_in_Bplus:
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.5] {1} -- {2} -- {3} -- cycle; \n".format("red", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
            elif el_in_Bminus: 
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.5] {1} -- {2} -- {3} -- cycle; \n".format("cyan", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
            else:
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.2] {1} -- {2} -- {3} -- cycle; \n".format("gray", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
            file.write("\\node (Rp) at ({0},{1}) [fill=white,fill opacity=1.0,inner sep = 2.5pt] {{ \\resizebox{{ .25\\linewidth}}{{!}}{{  \\textcolor{{black}}{{$B_+$}}  }} }}; \n".format(0.0*ddx,0.75*ddx))
            file.write("\\node (Rp) at ({0},{1}) [fill=white,fill opacity=1.0,inner sep = 2.5pt] {{ \\resizebox{{ .25\\linewidth}}{{!}}{{  \\textcolor{{black}}{{$B_-$}}  }} }}; \n".format(0.0*ddx,-0.75*ddx))

        if case_str == "BottomDataJumpEta":
            eta = 0.6 
            def is_in_omega(coord):
                if (coord[1] <= 0.25+1e-4):
                    return True 
                else:
                    return False 
            el_in_omega = np.all(np.array([is_in_omega(coord) for coord in coords ])) 
            def is_in_Bplus(coord):
                if (coord[0] >= 0.25 and coord[0] <= 0.75 and coord[1] >= eta and coord[1] <= 0.9): 
                    return True
                else:
                    return False 
            def is_in_Bminus(coord):    
                if (coord[0] >= 0.25 and coord[0] <= 0.75 and coord[1] >= 0.25 and coord[1] <= eta): 
                    return True
                else:
                    return False 
            el_in_Bplus = np.all(np.array([is_in_Bplus(coord) for coord in coords ]))
            el_in_Bminus = np.all(np.array([is_in_Bminus(coord) for coord in coords ]))

            if el_in_omega:
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.5] {1} -- {2} -- {3} -- cycle; \n".format("cyan", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
            elif el_in_Bplus:
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.5] {1} -- {2} -- {3} -- cycle; \n".format("red", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
            elif el_in_Bminus: 
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.5] {1} -- {2} -- {3} -- cycle; \n".format("green!70!blue", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))
            else:
                file.write("\\draw[line width=0.01mm,draw =black, fill={0},fill opacity=0.0] {1} -- {2} -- {3} -- cycle; \n".format("gray", (ddx*coords[0][0],ddx*coords[0][1]) , (ddx*coords[1][0],ddx*coords[1][1]) , (ddx*coords[2][0],ddx*coords[2][1])  ))


            file.write("\\node (Rp) at ({0},{1}) [fill=white,fill opacity=1.0,inner sep = 2.5pt] {{ \\resizebox{{ .15\\linewidth}}{{!}}{{  \\textcolor{{black}}{{$B_+$}}  }} }}; \n".format(0.5*ddx,0.7*ddx))
            file.write("\\node (Rp) at ({0},{1}) [fill=white,fill opacity=1.0,inner sep = 2.5pt] {{ \\resizebox{{ .15\\linewidth}}{{!}}{{  \\textcolor{{black}}{{$B_-$}}  }} }}; \n".format(0.5*ddx,0.45*ddx))
            file.write("\\draw[line width=0.4mm,draw =black,dashed] {1} -- {2}; \n".format("cyan", (ddx*0.0,ddx*eta) , (ddx*1.0,ddx*eta)  ))    
            file.write("\\node (RL) at ({0},{1}) [fill=white,fill opacity=1.0,inner sep = 2.5pt] {{ \\resizebox{{ .15\\linewidth}}{{!}}{{  \\textcolor{{black}}{{$\omega$}}  }} }}; \n".format(0.5*ddx,0.1*ddx))
            file.write("\\node (Rp) at ({0},{1}) [fill=white,fill opacity=1.0,inner sep = 2.5pt] {{ \\resizebox{{ .15\\linewidth}}{{!}}{{  \\textcolor{{black}}{{$\\mu_-$}}  }} }}; \n".format(0.1*ddx,0.45*ddx))
            file.write("\\node (Rp) at ({0},{1}) [fill=white,fill opacity=1.0,inner sep = 2.5pt] {{ \\resizebox{{ .15\\linewidth}}{{!}}{{  \\textcolor{{black}}{{$\\mu_+$}}  }} }}; \n".format(0.1*ddx,0.75*ddx))

    file.write("\\end{tikzpicture} \n") 
    file.write("\\end{document} \n")           
    file.close()



def create_initial_mesh_convex(init_h_scale=1.0):

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeFactor", init_h_scale)
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
        mesh = xdmf.read_mesh(name="Grid", ghost_mode = GM) 
        ct = xdmf.read_meshtags(mesh, name="Grid")
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
    with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
        ft = xdmf.read_meshtags(mesh, name="Grid")
    return mesh

def get_mesh_convex(n_ref,init_h_scale=1.0):
    meshes = [] 
    for j in range(n_ref):
        h_scale_j = init_h_scale/(2**j)
        meshes.append(create_initial_mesh_convex(init_h_scale=h_scale_j))
    return meshes 


def get_mesh_hierarchy(n_ref,init_h_scale=1.0): 

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeFactor", init_h_scale)
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
        #help(xdmf.read_mesh)
        print("ghost_mode = GhostMode.shared_facet")
        mesh = xdmf.read_mesh(name="Grid",ghost_mode = GM)
        #help(xdmf.read_mesh)
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
        edges = compute_incident_entities(mesh, cells, 2, 1)
        #print(edges)
        #help(refine)
        mesh = refine(mesh, edges, redistribute=True)
        mesh_hierarchy.append(mesh) 
    return mesh_hierarchy


'''
ls_mesh = get_mesh_hierarchy(2)

DrawMeshTikz(msh=ls_mesh[0],name="omega-Ind-convex-level0",case_str="convex-Mihai-omega") 
DrawMeshTikz(msh=ls_mesh[1],name="B-Ind-convex-level1",case_str="convex-Mihai-B") 

tol = 1e-12
x_l = 0.1-tol
x_r = 0.9+tol
y_b = 0.25-tol
y_t = 1.0+tol

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
          np.logical_and(   (x[1]>= 0.95),  (x[1]<= 1)  )
        )
      ) 
    B_coords = np.invert(rest_coords)
    values[B_coords] = np.full(sum(B_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    return values


for idx,mesh in enumerate(ls_mesh):

    Q_ind = FunctionSpace(mesh, ("DG", 0))
    B_ind  = Function(Q_ind)
    omega_ind = Function(Q_ind)
    B_ind.interpolate(B_Ind)
    omega_ind.interpolate(omega_Ind)

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



def get_mesh_hierarchy_nonconvex(n_ref,init_h_scale=1.0): 

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeFactor", init_h_scale )
    proc = MPI.COMM_WORLD.rank
    top_marker = 2
    bottom_marker = 1
    bnd_marker = 1
    omega_marker = 1
    Bwithoutomega_marker = 2
    rest_marker = 3 
    if proc == 0:
        # We create one rectangle for each subdomain

        #r1 = gmsh.model.occ.addRectangle(0.25, 0, 0, 0.5, 0.5,tag=1)
        #r2 = gmsh.model.occ.addRectangle(0.125, 0.0, 0, 0.75, 0.95,tag=2)
        r1 = gmsh.model.occ.addRectangle(0.25, 0.05, 0, 0.5, 0.45,tag=1)
        r2 = gmsh.model.occ.addRectangle(0.125, 0.05, 0, 0.75, 0.9,tag=2)
        
        r3 = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1,tag=3)
        gmsh.model.occ.fragment([(2,3)],[(2,1),(2,2)])

        gmsh.model.occ.synchronize()
        its = 0 
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            print(com)
            if np.allclose(com, [0.5,0.275, 0]):
                gmsh.model.addPhysicalGroup(2, [surface[1]], omega_marker)
            elif np.allclose(com, [0.5, 0.5, 0]):
                gmsh.model.addPhysicalGroup(2, [surface[1]], Bwithoutomega_marker )
            else:
                gmsh.model.addPhysicalGroup(2, [surface[1]], rest_marker)

            #else:
            #    gmsh.model.addPhysicalGroup(2, [surface[1]], omega_marker)
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

    #return 0

    
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
        #return msh

    #n_ref = 2 
    #for i in range(n_ref): 
    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid", ghost_mode=GM)
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
        edges = compute_incident_entities(mesh, cells, 2, 1)
        print(edges)
        mesh = refine(mesh, edges, redistribute=True)
        mesh_hierarchy.append(mesh) 
    
    return mesh_hierarchy
    
'''
ls_mesh = get_mesh_hierarchy_nonconvex(5)


for idx,mesh in enumerate(ls_mesh):
    with XDMFFile(mesh.comm, "mesh-nonconvex-reflvl{0}.xdmf".format(idx), "w") as file:
        file.write_mesh(mesh)

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

for idx,mesh in enumerate(ls_mesh):

    Q_ind = FunctionSpace(mesh, ("DG", 0))
    omega_ind = Function(Q_ind)
    B_ind = Function(Q_ind)
    B_ind.interpolate(B_Ind)
    omega_ind.interpolate(omega_Ind)
    
    #cells_omega = locate_entities(mesh, mesh.topology.dim, omega_Ind)
    #omega_ind.x.array[:] = 0.0
    #omega_ind.x.array[cells_omega] = np.full(len(cells_omega), 1)

    #cells_B = locate_entities(mesh, mesh.topology.dim, B_Ind)
    #B_ind.x.array[:] = 0.0
    #B_ind.x.array[cells_B] = np.full(len(cells_B), 1)

    with XDMFFile(mesh.comm, "omega-ind-reflvl{0}.xdmf".format(idx), "w") as file:
        file.write_mesh(mesh)
        file.write_function(omega_ind)

    with XDMFFile(mesh.comm, "B-ind-reflvl{0}.xdmf".format(idx), "w") as file:
        file.write_mesh(mesh)
        file.write_function(B_ind)
'''

def get_mesh_hierarchy_fitted_disc(n_ref,eta,h_init=1.25): 

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeFactor", h_init)
    proc = MPI.COMM_WORLD.rank
    bnd_marker = 1
    lower_omega_marker = 1
    side_left_marker = 2 
    side_right_marker = 3
    middle_bottom_marker = 4
    middle_top_marker = 5
    rest_marker = 6

    y_eta = eta-0.25
    y_inc = 0.95-y_eta
    if proc == 0:
        # We create one rectangle for each subdomain

        r1 = gmsh.model.occ.addRectangle(0, 0, 0, 1, eta,tag=1)
        r2 = gmsh.model.occ.addRectangle(0.1, 0.25, 0, 0.8,y_eta,tag=2)
        r3 = gmsh.model.occ.cut( [(2,r1)], [(2,r2)],tag=3)
        B_minus = gmsh.model.occ.addRectangle(0.1, 0.25, 0, 0.8, y_eta,tag=4) 
        B_plus = gmsh.model.occ.addRectangle(0.1, y_eta, 0, 0.8, (0.95-y_eta),tag=5) 
        top_remainder = gmsh.model.occ.addRectangle(0.0, y_eta, 0, 1.0, (1.0-y_eta),tag=6) 

        gmsh.model.occ.fragment([(2,3)],[(2,B_minus),(2,B_plus),(2,top_remainder)])

        #middle_top = gmsh.model.occ.addRectangle(0.1, y_eta, 0, 0.8, y_inc,tag=5)
        #side_left = gmsh.model.occ.addRectangle(0.0, y_eta, 0, 0.1, (1.0-y_eta),tag=6)
        #side_right = gmsh.model.occ.addRectangle(0.9, y_eta, 0, 0.1, (1.0-y_eta),tag=7)
        #top_remainder = gmsh.model.occ.addRectangle(0.1, 0.95, 0, 0.8, 0.05,tag=8)
        #gmsh.model.occ.fragment([(2,3)],[(2,side_left),(2,side_right),(2,middle_bottom),(2,middle_top),(2,top_remainder)])



        # We fuse the two rectangles and keep the interface between them
        #gmsh.model.occ.fragment([(2,3)],[(2,4)])

        #print("tmp = ", tmp)
        #gmsh.model.occ.fragment([(2,3)],[(2,4),(2,tmp)] )

        gmsh.model.occ.synchronize()

        #for surface in gmsh.model.getEntities(dim=2):
        #    gmsh.model.addPhysicalGroup(2, [surface[1]], 1)

        # Mark the top (2) and bottom (1) rectangle
        #top, bottom = None, None
        print(len(gmsh.model.getEntities(dim=2)))
        
        its = 1 
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            print(com)
            gmsh.model.addPhysicalGroup(2, [surface[1]], its )
            its +=1
        
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
        mesh = xdmf.read_mesh(name="Grid",ghost_mode=GM)
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
        edges = compute_incident_entities(mesh, cells, 2, 1)
        print(edges)
        mesh = refine(mesh, edges, redistribute=True)
        mesh_hierarchy.append(mesh) 
    return mesh_hierarchy

'''
ls_mesh = get_mesh_hierarchy_fitted_disc(4,eta=0.6) 
tol = 1e-12
DrawMeshTikz(msh=ls_mesh[0],name="omega-Ind-Split-level0",case_str="splitgeom-omega") 
DrawMeshTikz(msh=ls_mesh[1],name="B-Ind-Split-level1",case_str="splitgeom-B") 

def omega_Ind(x):
    
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

for idx,mesh in enumerate(ls_mesh):

    Q_ind = FunctionSpace(mesh, ("DG", 0))
    B_ind  = Function(Q_ind)
    omega_ind = Function(Q_ind)
    B_ind.interpolate(B_Ind)
    omega_ind.interpolate(omega_Ind)

    with XDMFFile(mesh.comm, "omega-ind-reflvl{0}.xdmf".format(idx), "w") as file:
        file.write_mesh(mesh)
        file.write_function(omega_ind)

    with XDMFFile(mesh.comm, "B-ind-reflvl{0}.xdmf".format(idx), "w") as file:
        file.write_mesh(mesh)
        file.write_function(B_ind)
'''























def get_mesh_inclusion(h_init=1.25,order=2): 
    gdim = 2
    cell_type  = CellType.triangle
    gmsh.initialize()
    gmsh.model.add("inclm")
    gmsh.option.setNumber("Mesh.MeshSizeFactor", h_init)
    proc = MPI.COMM_WORLD.rank
    bnd_marker = 1
    lower_omega_marker = 1
    side_left_marker = 2 
    side_right_marker = 3
    middle_bottom_marker = 4
    middle_top_marker = 5
    rest_marker = 6

    #y_eta = eta-0.25
    #y_inc = 0.95-y_eta
    #if proc == 0:
        # We create one rectangle for each subdomain

    if MPI.COMM_WORLD.rank == 0:
        r1 = gmsh.model.occ.addRectangle(-1.5, -1.5, 0, 3,3,tag=1)
        r2 = gmsh.model.occ.addRectangle(-1.25, -1.25, 0, 2.5,2.75,tag=2)
        r3 = gmsh.model.occ.cut( [(2,r1)], [(2,r2)],tag=3)

        target_dom = gmsh.model.occ.addRectangle(-0.5, -0.5, 0, 1.0,1.0)
        Bdisk = gmsh.model.occ.addDisk(0.0, 0.0, 0.0,1.0,1.0,tag=2)
        gmsh.model.occ.synchronize()
        remainder = gmsh.model.occ.addRectangle(-1.25, -1.25, 0, 2.5,2.75)
        #gmsh.model.occ.fragment([(2,r1)],[(2,Bdisk) ])
        gmsh.model.occ.fragment([(2,3)],[(2,target_dom),(2,Bdisk),(2,remainder) ])
        
        gmsh.model.occ.synchronize()

        print(len(gmsh.model.getEntities(dim=2)))
        
        its = 1 
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            print(com)
            gmsh.model.addPhysicalGroup(2, [surface[1]], its )
            its +=1
        
        # Tag the left boundary
        bnd_square = []
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if np.isclose(com[0], 0) or np.isclose(com[0], 1) or np.isclose(com[1], 0) or  np.isclose(com[1], 1): 
                bnd_square.append(line[1])
        gmsh.model.addPhysicalGroup(1, bnd_square, bnd_marker)
        gmsh.model.mesh.generate(2)
        #gmsh.model.mesh.setOrder(order)
        gmsh.write("mesh.msh")
    #gmsh.finalize()


    if cell_type == CellType.quadrilateral:
        gmsh.model.mesh.recombine()
    gmsh.model.mesh.setOrder(order)
    idx, points, _ = gmsh.model.mesh.getNodes()
    #points = points[:,:2]
    ls_points_2D = [] 
    for i in range(len(points)):
        if (i+1) % 3 != 0:
            ls_points_2D.append(points[i])
    ls_points_2D = np.array(ls_points_2D)
    #points = points.reshape(-1, 3)
    points =  ls_points_2D.reshape(-1, 2)
    #print("points =", points)
    idx -= 1
    srt = np.argsort(idx)
    assert np.all(idx[srt] == np.arange(len(idx)))
    x = points[srt]

    element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)
    name, dim, order, num_nodes, local_coords, num_first_order_nodes = gmsh.model.mesh.getElementProperties(
        element_types[0])

    cells = node_tags[0].reshape(-1, num_nodes) - 1
    if cell_type == CellType.triangle:
        gmsh_cell_id = gmsh.model.mesh.getElementType("triangle", order)
    elif cell_type == CellType.quadrilateral:
        gmsh_cell_id = gmsh.model.mesh.getElementType("quadrangle", order)
    gmsh.finalize()

    cells = cells[:, perm_gmsh(cell_type, cells.shape[1])]
    #print("cells.shape[1] =, ",cells.shape[1])
    msh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh_from_gmsh(gmsh_cell_id, x.shape[1]))
    #msh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh_from_gmsh(gmsh_cell_id,2))
    #print("x.shape[1] =", x.shape[1])

    with XDMFFile(msh.comm, "mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(msh)
    return msh

'''
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

mesh = get_mesh_inclusion(h_init=1.25) 
Q_ind = FunctionSpace(mesh, ("DG", 0))
B_ind  = Function(Q_ind)
omega_ind = Function(Q_ind)
B_ind.interpolate(B_Ind_Disk)
omega_ind.interpolate(omega_Ind_Disk)

idx = 99 
with XDMFFile(mesh.comm, "omega-ind-reflvl{0}.xdmf".format(idx), "w") as file:
    file.write_mesh(mesh)
    file.write_function(omega_ind)

with XDMFFile(mesh.comm, "B-ind-reflvl{0}.xdmf".format(idx), "w") as file:
    file.write_mesh(mesh)
    file.write_function(B_ind)

import ufl
x = ufl.SpatialCoordinate(mesh)
a = 1.0 
mu_plus = 1 
mu_minus = 2 
def mu_Ind(x):
    values = np.zeros(x.shape[1],dtype=ScalarType)
    upper_coords = (x[0]*x[0] + x[1]*x[1]) > a**2
    #print("upper_coords = ", upper_coords)
    lower_coords = np.invert(upper_coords)
    values[upper_coords] = np.full(sum(upper_coords), mu_plus)
    values[lower_coords] = np.full(sum(lower_coords), mu_minus)
    return values

mu = Function(Q_ind)
mu.interpolate(mu_Ind)
with XDMFFile(mesh.comm, "mu-jump-disk-idx{0}.xdmf".format(idx), "w") as file:
    file.write_mesh(mesh)
    file.write_function(mu)
'''




def get_mesh_inclusion_square(h_init=1.25,x_L=-1.25,x_R=1.25,y_L=-1.25,y_R=1.25,eta=1.25): 
    gdim = 2
    cell_type  = CellType.triangle
    gmsh.initialize()
    gmsh.model.add("inclm")
    gmsh.option.setNumber("Mesh.MeshSizeFactor", h_init)
    proc = MPI.COMM_WORLD.rank
    bnd_marker = 1
    lower_omega_marker = 1
    side_left_marker = 2 
    side_right_marker = 3
    middle_bottom_marker = 4
    middle_top_marker = 5
    rest_marker = 6


    if MPI.COMM_WORLD.rank == 0:
        r1 = gmsh.model.occ.addRectangle(-1.5, -1.5, 0, 3,1.5+eta,tag=1)
        r2 = gmsh.model.occ.addRectangle(-1.25, -1.25, 0, 2.5,1.15+eta,tag=2)
        r3 = gmsh.model.occ.cut( [(2,r1)], [(2,r2)],tag=3)

        #mu_inner =  gmsh.model.occ.addRectangle(-0.75, -0.75, 0, 1.5,2.25)
        if eta < 1.5:
            target_dom_minus = gmsh.model.occ.addRectangle(x_L, y_L, 0, x_R-x_L ,eta-y_L)
            target_dom_plus = gmsh.model.occ.addRectangle(x_L, eta, 0, x_R-x_L ,y_R-eta)
            mu_inner =  gmsh.model.occ.addRectangle(-0.75, -0.75, 0, 1.5,1.5)
            #remainder_mid_1 = gmsh.model.occ.addRectangle(-1.25, -1.25, 0,x_L+1.25,1.25+eta)
            #remainder_mid_2 = gmsh.model.occ.addRectangle(x_R, -1.25, 0, 1.25-x_R,1.25+eta)
            remainder_top = gmsh.model.occ.addRectangle(-1.5, eta, 0, 3.0, 1.5-eta)
            gmsh.model.occ.synchronize()
            #gmsh.model.occ.fragment([(2,3)],[(2,target_dom_minus),(2,target_dom_plus),(2,mu_inner),(2,remainder_mid_1),(2,remainder_mid_2),(2,remainder_top)])
            #gmsh.model.occ.fragment([(2,3)],[(2,target_dom_minus),(2,target_dom_plus),(2,mu_inner),(2,remainder_top)])
            #r4 = gmsh.model.occ.fragment([(2,3)],[(2,target_dom_minus),(2,mu_inner), (2,remainder_top),  ])
            r4 = gmsh.model.occ.fragment([(2,3)],[(2,target_dom_minus),(2,mu_inner),  (2,target_dom_plus),(2,remainder_top)   ])
            
            gmsh.model.occ.synchronize()
            #r5 = gmsh.model.occ.addRectangle(-1.5, eta, 0, 3.0, 1.5-eta)
            #r6 = gmsh.model.occ.addRectangle( x_L, eta, 0, x_R-x_L, 1.25-eta)
            #r7 = gmsh.model.occ.cut( [(2,r5)], [(2,r6)])
            #gmsh.model.occ.synchronize()
            #r8 = gmsh.model.occ.addRectangle( x_L, eta, 0, x_R-x_L, 1.25-eta)
            #r9 = gmsh.model.occ.fragment([(2,r4)],[(2,r7),(2,r8)])


        else:
            target_dom = gmsh.model.occ.addRectangle(x_L, y_L, 0, x_R-x_L ,y_R-y_L)
            mu_inner =  gmsh.model.occ.addRectangle(-0.75, -0.75, 0, 1.5,1.5)
            remainder = gmsh.model.occ.addRectangle(-1.25, -1.25, 0, 2.5,2.75)
            gmsh.model.occ.synchronize()
            gmsh.model.occ.fragment([(2,3)],[(2,target_dom),(2,mu_inner),(2,remainder) ])
        
        gmsh.model.occ.synchronize()

        print(len(gmsh.model.getEntities(dim=2)))
        
        its = 1 
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            print(com)
            gmsh.model.addPhysicalGroup(2, [surface[1]], its )
            its +=1
        
        # Tag the left boundary
        bnd_square = []
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if np.isclose(com[0], 0) or np.isclose(com[0], 1) or np.isclose(com[1], 0) or  np.isclose(com[1], 1): 
                bnd_square.append(line[1])
        gmsh.model.addPhysicalGroup(1, bnd_square, bnd_marker)
        gmsh.model.mesh.generate(2)
        #gmsh.model.mesh.setOrder(order)
        gmsh.write("mesh.msh")
    #gmsh.finalize()


    if cell_type == CellType.quadrilateral:
        gmsh.model.mesh.recombine()
    order = 1
    gmsh.model.mesh.setOrder(order)
    idx, points, _ = gmsh.model.mesh.getNodes()
    #points = points[:,:2]
    ls_points_2D = [] 
    for i in range(len(points)):
        if (i+1) % 3 != 0:
            ls_points_2D.append(points[i])
    ls_points_2D = np.array(ls_points_2D)
    #points = points.reshape(-1, 3)
    points =  ls_points_2D.reshape(-1, 2)
    #print("points =", points)
    idx -= 1
    srt = np.argsort(idx)
    assert np.all(idx[srt] == np.arange(len(idx)))
    x = points[srt]

    element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)
    name, dim, order, num_nodes, local_coords, num_first_order_nodes = gmsh.model.mesh.getElementProperties(
        element_types[0])

    cells = node_tags[0].reshape(-1, num_nodes) - 1
    if cell_type == CellType.triangle:
        gmsh_cell_id = gmsh.model.mesh.getElementType("triangle", order)
    elif cell_type == CellType.quadrilateral:
        gmsh_cell_id = gmsh.model.mesh.getElementType("quadrangle", order)
    gmsh.finalize()

    cells = cells[:, perm_gmsh(cell_type, cells.shape[1])]
    #print("cells.shape[1] =, ",cells.shape[1])
    msh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh_from_gmsh(gmsh_cell_id, x.shape[1]))
    #msh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh_from_gmsh(gmsh_cell_id,2))
    #print("x.shape[1] =", x.shape[1])

    with XDMFFile(msh.comm, "mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(msh)
    return msh
'''
x_L = -1.0
x_R = 1.0
y_L = -1.0
y_R = 1.0
mu_plus = 2.0 
mu_minus = 1.0

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

idx = 25
mesh = get_mesh_inclusion_square(h_init=1.25) 
Q_ind = FunctionSpace(mesh, ("DG", 0))
mu = Function(Q_ind)
mu.interpolate(mu_Ind)
with XDMFFile(mesh.comm, "mu-jump-disk-idx{0}.xdmf".format(idx), "w") as file:
    file.write_mesh(mesh)
    file.write_function(mu)
'''



def get_mesh_bottom_data(h_init=1.25,eta=0.6): 
    gdim = 2
    cell_type  = CellType.triangle
    gmsh.initialize()
    gmsh.model.add("inclm")
    gmsh.option.setNumber("Mesh.MeshSizeFactor", h_init)
    proc = MPI.COMM_WORLD.rank

    if MPI.COMM_WORLD.rank == 0:

        #Omega_all = gmsh.model.occ.addRectangle(0.0, 0.0, 0, 1.0,1.0,tag=8)
        omega = gmsh.model.occ.addRectangle(0.0, 0.0, 0, 1.0,0.25,tag=1)
        
        middle_left = gmsh.model.occ.addRectangle(0, 0.25, 0, 0.25,eta-0.25,tag=2)
        middle_right = gmsh.model.occ.addRectangle(0.75, 0.25, 0, 0.25,eta-0.25,tag=3)
        B_minus = gmsh.model.occ.addRectangle(0.25, 0.25, 0, 0.5,eta-0.25,tag=4)
        
        B_plus = gmsh.model.occ.addRectangle(0.25, eta, 0, 0.5,0.9-eta,tag=5)
        top_left = gmsh.model.occ.addRectangle(0, eta, 0, 0.25,1.0-eta,tag=6)
        top_middle = gmsh.model.occ.addRectangle(0.25, eta, 0, 0.5,1.0-eta,tag=7)
        top_right = gmsh.model.occ.addRectangle(0.75, eta, 0, 0.25,1.0-eta,tag=8)


        gmsh.model.occ.synchronize()
        gmsh.model.occ.fragment([(2,omega)],[(2,B_minus),(2,middle_left), (2,middle_right),(2,B_plus),(2,top_left),(2,top_middle),(2,top_right)]) 
        #gmsh.model.occ.fragment([(2,7)],[(2,B_minus),(2,middle_left),(2,middle_right),(2,B_plus),(2,omega) ])
        
        gmsh.model.occ.synchronize()

        print(len(gmsh.model.getEntities(dim=2)))
        
        its = 1 
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            print(com)
            gmsh.model.addPhysicalGroup(2, [surface[1]], its )
            its +=1
        
        # Tag the left boundary
        #bnd_square = []
        #for line in gmsh.model.getEntities(dim=1):
        #    com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
        #    if np.isclose(com[0], 0) or np.isclose(com[0], 1) or np.isclose(com[1], 0) or  np.isclose(com[1], 1): 
        #        bnd_square.append(line[1])
        #gmsh.model.addPhysicalGroup(1, bnd_square, bnd_marker)
        gmsh.model.mesh.generate(2)
        #gmsh.model.mesh.setOrder(order)
        gmsh.write("mesh.msh")
    #gmsh.finalize()


    if cell_type == CellType.quadrilateral:
        gmsh.model.mesh.recombine()
    order = 1
    gmsh.model.mesh.setOrder(order)
    idx, points, _ = gmsh.model.mesh.getNodes()
    #points = points[:,:2]
    ls_points_2D = [] 
    for i in range(len(points)):
        if (i+1) % 3 != 0:
            ls_points_2D.append(points[i])
    ls_points_2D = np.array(ls_points_2D)
    #points = points.reshape(-1, 3)
    points =  ls_points_2D.reshape(-1, 2)
    #print("points =", points)
    idx -= 1
    srt = np.argsort(idx)
    assert np.all(idx[srt] == np.arange(len(idx)))
    x = points[srt]

    element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)
    name, dim, order, num_nodes, local_coords, num_first_order_nodes = gmsh.model.mesh.getElementProperties(
        element_types[0])

    cells = node_tags[0].reshape(-1, num_nodes) - 1
    if cell_type == CellType.triangle:
        gmsh_cell_id = gmsh.model.mesh.getElementType("triangle", order)
    elif cell_type == CellType.quadrilateral:
        gmsh_cell_id = gmsh.model.mesh.getElementType("quadrangle", order)
    gmsh.finalize()

    cells = cells[:, perm_gmsh(cell_type, cells.shape[1])]
    #print("cells.shape[1] =, ",cells.shape[1])
    msh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh_from_gmsh(gmsh_cell_id, x.shape[1]))
    #msh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh_from_gmsh(gmsh_cell_id,2))
    #print("x.shape[1] =", x.shape[1])

    with XDMFFile(msh.comm, "mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(msh)
    return msh

#mesh = get_mesh_bottom_data(h_init=1.25,eta=0.6) 
#with XDMFFile(mesh.comm, "mesh.xdmf".format(idx), "w") as file:
#    file.write_mesh(mesh)
'''
eta = 0.6
h_init = 1.25
n_ref = 5
ls_mesh = []
for i in range(n_ref): 
    ls_mesh.append( get_mesh_bottom_data(h_init=h_init/2**i,eta=eta) )

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

for idx,mesh in enumerate(ls_mesh):

    Q_ind = FunctionSpace(mesh, ("DG", 0))
    B_ind  = Function(Q_ind)
    omega_ind = Function(Q_ind)
    B_ind.interpolate(B_Ind)
    omega_ind.interpolate(omega_Ind)

    with XDMFFile(mesh.comm, "omega-ind-reflvl{0}.xdmf".format(idx), "w") as file:
        file.write_mesh(mesh)
        file.write_function(omega_ind)

    with XDMFFile(mesh.comm, "B-ind-reflvl{0}.xdmf".format(idx), "w") as file:
        file.write_mesh(mesh)
        file.write_function(B_ind)
'''
