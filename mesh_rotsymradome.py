# Create mesh for a rotationally symmetric radome. Also a similar
# geometry with a sphere for verification.
# 
# Daniel Sjöberg, 2023-06-19

import numpy as np
import dolfinx
from mpi4py import MPI
import gmsh
import sys
import pyvista

# Set up design frequency and wavelength
from scipy.constants import c as c0
f0 = 10e9                            # Design frequency
lambda0 = c0/f0                      # Design wavelength

# Use variables tdim and fdim to clarify when addressing a dimension
tdim = 2                             # Dimension of triangles/tetraedra
fdim = tdim - 1                      # Dimension of facets

class PerfectlyMatchedLayer():
    def __init__(self, d=None, radius=None, rho=None, zt=None, zb=None, n=3, cylindrical=False):
        self.d = d                     # Thickness of PML
        self.radius = radius           # Radius of spherical PML
        self.rho = rho                 # Cylindrical radius of cylindrical PML
        self.zt = zt                   # Top z value of cylindrical PML
        self.zb = zb                   # Bottom z value of cylindrical PML
        self.cylindrical = cylindrical # Whether to use cylindrical PML
        
class MeshData():
    def __init__(
            self, mesh=None, subdomains=None, boundaries=None,
            subdomain_markers={'freespace': -1,
                               'material': -1,
                               'pml': -1,
                               'pml_material_overlap': -1},
            boundary_markers={'pec': -1,
                              'antenna': -1,
                              'farfield': -1,
                              'pml': -1,
                              'axis': -1},
            PML=None,
            comm=MPI.COMM_WORLD,
            model_rank=0
    ):
        self.mesh = mesh                           # Mesh
        self.subdomains = subdomains               # Tagged subdomains
        self.boundaries = boundaries               # Tagged boundaries
        self.subdomain_markers = subdomain_markers # Dictionary of subdomain markers
        self.boundary_markers = boundary_markers   # Dictionary of boundary markers
        self.PML = PML                             # PML data
        self.comm = comm                           # MPI communicator
        self.model_rank=model_rank                 # Model rank
        

def CheckGhostFarfieldFacets(comm, model_rank, mesh, boundaries, farfield_surface_marker):
    ff_facets_local = boundaries.find(farfield_surface_marker)
    local_to_global = mesh.topology.index_map(fdim).local_to_global
    ff_facets_global = local_to_global(ff_facets_local)
    ff_facets_local_all = comm.gather(ff_facets_local, root=model_rank)
    ff_facets_global_all = comm.gather(ff_facets_global, root=model_rank)
    if comm.rank == model_rank:
        ghost_ff_facets = []
        for rank_i in range(comm.size-1):
            for idx_i, fff in enumerate(ff_facets_global_all[rank_i]):
                for rank_j in range(rank_i+1, comm.size):
                    if fff in ff_facets_global_all[rank_j]:
                        idx_j = np.where(ff_facets_global_all[rank_j]==fff)[0][0]
                        ghost_ff_facets.append((rank_i, rank_j, fff, ff_facets_local_all[rank_i][idx_i], ff_facets_local_all[rank_j][idx_j]))
    else:
        ghost_ff_facets = None
    ghost_ff_facets = comm.bcast(ghost_ff_facets, root=model_rank)
#    for rank_i, rank_j, gfff in ghost_ff_facets:
#        if comm.rank == rank_j:
#            if gfff in ff_facets:
#                foo = ff_facets.tolist()
#                foo.remove(gfff)
#                ff_facets = np.array(foo)
#    ff_boundary = dolfinx.mesh.meshtags(mesh, fdim, ff_facets, farfield_surface_marker)
    if len(ghost_ff_facets) > 0:
        if comm.rank == model_rank:
            print('Ghost farfield facets detected, farfield results may be inaccurate. Consider making small changes in location of far field surface, mesh size, or use different number of processors.')
            print(f'ghost_ff_facets = {ghost_ff_facets}')
            sys.stdout.flush()
    return(ghost_ff_facets)

def PlotMeshPartition(comm, model_rank, mesh, ghost_ff_facets, boundaries, farfield_surface_marker):
    V = dolfinx.fem.FunctionSpace(mesh, ('CG', 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: np.ones(x.shape[1])*comm.rank)
    mesh.topology.create_connectivity(fdim, 0)
    for rank_i, rank_j, gfff_global, gfff_local_i, gfff_local_j in ghost_ff_facets:
        if comm.rank == rank_i or comm.rank == rank_j:
#            fmap = mesh.topology.index_map(fdim)
#            local_to_global = fmap.local_to_global(range(fmap.size_local + fmap.num_ghosts))
#            gfff_local = np.where(local_to_global == gfff_global)[0][0]
            facets_to_nodes = mesh.topology.connectivity(fdim, 0)
            if comm.rank == rank_i:
                node_indices = facets_to_nodes.links(gfff_local_i)
            else:
                node_indices = facets_to_nodes.links(gfff_local_j)
            dof_indices = []
            nmap = mesh.topology.index_map(0)
            size_local = nmap.size_local
            for idx in node_indices:
                if idx >= size_local:
                    dof_indices.append(size_local + np.where(V.dofmap.index_map.ghosts == nmap.ghosts[idx - size_local])[0][0])
                else:
                    dof_indices.append(idx)
#            print(f'Rank {comm.rank}: {gfff_local}')
#            print(f'Rank {comm.rank}: {dof_indices}')
            u.x.array[dof_indices] = -1
    cells, cell_types, x = dolfinx.plot.create_vtk_mesh(mesh, tdim)
    grid = pyvista.UnstructuredGrid(cells, cell_types, x)
    grid["rank"] = np.real(u.x.array)
    grids = comm.gather(grid, root=model_rank)
    if comm.rank == model_rank:
        plotter = pyvista.Plotter()
        for g in grids:
            plotter.add_mesh(g, show_edges=True)
        plotter.view_xy()
        plotter.add_axes()
        plotter.show()

def partitioner(comm, n, m, topo):
    dests = []
    offsets = [0]
    for i in range(topo.num_nodes):
        dests.append(comm.rank)
        offsets.append(len(dests))
    dests = np.array(dests, dtype=np.int32)
    offsets = np.array(offsets, dtype=np.int32)
    return(dolfinx.cpp.graph.AdjacencyList_int32(dests))
    
def CreateMeshOgive(
        comm=MPI.COMM_WORLD,     # MPI communicator
        model_rank=0,            # Rank of modelling process
        w=10*lambda0,            # Width of antenna
        t=lambda0/10,            # Thickness of antenna
        d0=lambda0/2,            # Distance from antenna to radome
        d=lambda0/(2*np.real(np.sqrt(3*(1-0.01j)))), # Radome thickness
        alpha=2,                 # Ogive shape factor
        H=5*lambda0,             # Height of straight part of radome
        Htransition=lambda0,     # Length of transition region from fuselage
        PMLcylindrical=False,    # Use a spherical PML or not
        PMLpenetrate=False,      # Let the radome penetrate the bottom PML
        Antenna=False,           # Include an antenna or not
        AntennaMetalBase=False,  # Finish the structure with metal into PML
        h=0.1*lambda0,           # Typical mesh size
        hfine=None,              # Typical refined mesh size
        verbosity=1,             # Verbosity of gmsh
        visualize=False,         # Whether to call gmsh to visualize mesh
        filename='ogivemesh.msh' # Name of file to save mesh in
): 
    """Create the mesh using gmsh."""

    # Compute a few shape parameters
    Rb = w/2 + d0 + d                           # Outer radome radius
    Lb = alpha*Rb                               # Outer radome height
    rhob = (Rb**2 + Lb**2)/(2*Rb)               #
    Ra = Rb - d                                 # Inner radome radius
    rhoa = rhob - d                             #
    La = np.sqrt(rhoa**2 - (rhob - Rb)**2)      # Inner radome height
    radius_domain = Lb + lambda0                # Radius of computational domain
    radius_pml = radius_domain + 0.5*lambda0    # Outer radius of PML
    radius_farfield = radius_domain - lambda0/2 # Radius at which to compute farfield
    FF_d = lambda0                              # Distance from farfield to PML boundary
    PML = PerfectlyMatchedLayer()               # Create PML data structure
    PML.cylindrical = PMLcylindrical            # What kind of PML to make
    if AntennaMetalBase:                        # Ensure that penetrated, cylindrical PML is used when specifying antenna with metal base
        PML.cylindrical = True
        PMLpenetrate = True
    PML.d = lambda0/2                           # Cylindrical PML thickness
    PML.zt = Lb + lambda0 + PML.d               # PML boundary at top
    if PMLpenetrate:                            # PML boundary at bottom
        PML.zb = -H
    else:
        PML.zb = -H - lambda0 - PML.d
    PML.rho = Rb + lambda0 + PML.d              # PML boundary in cylindrical radius
    PML.radius = radius_pml                     # Spherical PML boundary
    if hfine == None:                           # If no refinement asked for
        hfine = h
        
    gmsh.initialize()
    if comm.rank == model_rank:
        # Typical mesh size
        gmsh.option.setNumber('General.Verbosity', verbosity)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hfine)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

        # Outer radome curve
        nPoints = 10
        zPoints = np.linspace(0, Lb, nPoints)
        xPoints = np.sqrt(rhob**2 - zPoints**2) + Rb - rhob
        radome_outer_points = []
        for n in range(0, nPoints):
            tag = gmsh.model.occ.addPoint(xPoints[n], zPoints[n], 0)
            radome_outer_points.append(tag)
        radome_outer_curve = gmsh.model.occ.addSpline(radome_outer_points)
    
        # Inner radome curve
        nPoints = 10
        zPoints = np.linspace(0, La, nPoints)
        xPoints = np.sqrt(rhoa**2 - zPoints**2) + Ra - rhoa
        radome_inner_points = []
        for n in range(0, nPoints):
            tag = gmsh.model.occ.addPoint(xPoints[n], zPoints[n], 0)
            radome_inner_points.append(tag)
        radome_inner_curve = gmsh.model.occ.addSpline(radome_inner_points)

        # Straight part of radome
        radome_bottom_inner = gmsh.model.occ.addPoint(Ra, -H, 0)
        radome_bottom_outer = gmsh.model.occ.addPoint(Rb, -H, 0)
#        radome_mid_inner = gmsh.model.occ.addPoint(Ra, -t, 0)
#        radome_mid_outer = gmsh.model.occ.addPoint(Rb, -t, 0)
        radome_transition_inner = gmsh.model.occ.addPoint(Ra, -Htransition, 0)
        radome_transition_outer = gmsh.model.occ.addPoint(Rb, -Htransition, 0)
        
        # Join radome curves
        radome_join1 = gmsh.model.occ.addLine(radome_outer_points[-1], radome_inner_points[-1])
#        radome_join2a = gmsh.model.occ.addLine(radome_inner_points[0], radome_mid_inner)
#        radome_join2b = gmsh.model.occ.addLine(radome_mid_inner, radome_transition_inner)
        radome_join2b = gmsh.model.occ.addLine(radome_inner_points[0], radome_transition_inner)
        radome_join2c = gmsh.model.occ.addLine(radome_transition_inner, radome_bottom_inner)
        radome_join3 = gmsh.model.occ.addLine(radome_bottom_inner, radome_bottom_outer)
        radome_join4a = gmsh.model.occ.addLine(radome_bottom_outer, radome_transition_outer)
#        radome_join4b = gmsh.model.occ.addLine(radome_transition_outer, radome_mid_outer)
#        radome_join4c = gmsh.model.occ.addLine(radome_mid_outer, radome_outer_points[0])
        radome_join4b = gmsh.model.occ.addLine(radome_transition_outer, radome_outer_points[0])
        transition_join1 = gmsh.model.occ.addLine(radome_outer_points[0], radome_inner_points[0])
        transition_join2 = gmsh.model.occ.addLine(radome_transition_inner, radome_transition_outer)
        radome_loop = gmsh.model.occ.addCurveLoop([radome_outer_curve, radome_join1, radome_inner_curve, transition_join1])
#        transition_loop = gmsh.model.occ.addCurveLoop([radome_join2a, radome_join2b, transition_join2, radome_join4b, radome_join4c, transition_join1])
        transition_loop = gmsh.model.occ.addCurveLoop([radome_join2b, transition_join2, radome_join4b, transition_join1])
        CFRP_loop = gmsh.model.occ.addCurveLoop([radome_join2c, radome_join3, radome_join4a, transition_join2])

        # Create domains for radome, transition, and CFRP
        radome_domain = gmsh.model.occ.addPlaneSurface([radome_loop])
        transition_domain = gmsh.model.occ.addPlaneSurface([transition_loop])
        CFRP_domain = gmsh.model.occ.addPlaneSurface([CFRP_loop])

        # Create antenna domain, seems crucial to reuse points defined in radome
        if AntennaMetalBase: # Antenna surface connected to a metal fuselage
            antenna_point1 = gmsh.model.occ.addPoint(0, 0, 0)
            antenna_point2 = gmsh.model.occ.addPoint(w/2, 0, 0)
#            antenna_point3 = gmsh.model.occ.addPoint(w/2, -t, 0)
            antenna_point3 = gmsh.model.occ.addPoint(w/2, -Htransition, 0)
#            antenna_point4 = gmsh.model.occ.addPoint(Rb, -t, 0)
#            antenna_point5 = gmsh.model.occ.addPoint(Rb, -H, 0)
#            antenna_point4a = radome_mid_inner
#            antenna_point4b = radome_mid_outer
            antenna_point4a = radome_transition_inner
            antenna_point4b = radome_transition_outer
            antenna_point5 = radome_bottom_outer
            antenna_point6 = gmsh.model.occ.addPoint(0, -H, 0)
            
            antenna_curve = gmsh.model.occ.addLine(antenna_point1, antenna_point2)
            pec_curve1 = gmsh.model.occ.addLine(antenna_point2, antenna_point3)
            pec_curve2 = gmsh.model.occ.addLine(antenna_point3, antenna_point4a)
#            pec_curve3 = gmsh.model.occ.addLine(antenna_point4a, antenna_point4b)
            pec_curve3 = transition_join2
            pec_curve4 = gmsh.model.occ.addLine(antenna_point4b, antenna_point5)
            pec_curve = [pec_curve1, pec_curve2, pec_curve3, pec_curve4]
            antenna_join1 = gmsh.model.occ.addLine(antenna_point5, antenna_point6)
            antenna_join2 = gmsh.model.occ.addLine(antenna_point6, antenna_point1)
            antenna_join = [antenna_join1, antenna_join2]
            antenna_loop = gmsh.model.occ.addCurveLoop([antenna_curve] + pec_curve + antenna_join)
            antenna_domain = gmsh.model.occ.addPlaneSurface([antenna_loop])

            # Define functions to identify antenna and PEC regions geometrically
            def AntennaSurface(CoM):
                return(np.allclose(CoM, [w/4, 0, 0]))
            def PECSurface(CoM):
                return np.allclose(CoM, [w/2, -Htransition/2, 0]) or np.allclose(CoM, [(w/2 + Ra)/2, -Htransition, 0]) or np.allclose(CoM, [(Ra + Rb)/2, -Htransition, 0]) or np.allclose(CoM, [Rb, (-Htransition - H + PML.d)/2, 0]) or np.allclose(CoM, [Rb, -H + PML.d/2, 0])
            
        elif Antenna: # Just a metal strip
            antenna_point1 = gmsh.model.occ.addPoint(0, 0, 0)
            antenna_point2 = gmsh.model.occ.addPoint(w/2, 0, 0)
            antenna_point3 = gmsh.model.occ.addPoint(w/2, -t, 0)
            antenna_point4 = gmsh.model.occ.addPoint(0, -t, 0)

            antenna_curve = gmsh.model.occ.addLine(antenna_point1, antenna_point2)
            pec_curve1 = gmsh.model.occ.addLine(antenna_point2, antenna_point3)
            pec_curve2 = gmsh.model.occ.addLine(antenna_point3, antenna_point4)
            pec_curve = [pec_curve1, pec_curve2]
            antenna_join = gmsh.model.occ.addLine(antenna_point4, antenna_point1)
            antenna_loop = gmsh.model.occ.addCurveLoop([antenna_curve] + pec_curve + [antenna_join])
            antenna_domain = gmsh.model.occ.addPlaneSurface([antenna_loop])

            # Define functions to identify antenna and PEC regions geometrically
            def AntennaSurface(CoM):
                return(np.allclose(CoM, [w/4, 0, 0]))
            def PECSurface(CoM):
                return(np.allclose(CoM, [w/2, -t/2, 0]) or np.allclose(CoM, [w/4, -t, 0]))

        gmsh.model.occ.synchronize()
        
        # Create free space and PML domains.
        # There is one free space domain on each side of the farfield boundary.
        if not PML.cylindrical: # Spherical PML
            inner_point1 = gmsh.model.occ.addPoint(0, -radius_farfield, 0)
            inner_point2 = gmsh.model.occ.addPoint(0, radius_farfield, 0)
            outer_point1 = gmsh.model.occ.addPoint(0, -radius_domain, 0)
            outer_point2 = gmsh.model.occ.addPoint(0, radius_domain, 0)
            pml_point1 = gmsh.model.occ.addPoint(0, -radius_pml, 0)
            pml_point2 = gmsh.model.occ.addPoint(0, radius_pml, 0)
            
            farfield_boundary = gmsh.model.occ.addCircle(0, 0, 0, radius_farfield, -1, -np.pi/2, np.pi/2)
            domain_boundary = gmsh.model.occ.addCircle(0, 0, 0, radius_domain, -1, -np.pi/2, np.pi/2)
            pml_boundary = gmsh.model.occ.addCircle(0, 0, 0, radius_pml, -1, -np.pi/2, np.pi/2)
            outer_join1 = gmsh.model.occ.addLine(inner_point1, outer_point1)
            outer_join2 = gmsh.model.occ.addLine(inner_point2, outer_point2)
            inner_join = gmsh.model.occ.addLine(inner_point1, inner_point2)
            pml_join1 = gmsh.model.occ.addLine(pml_point1, outer_point1)
            pml_join2 = gmsh.model.occ.addLine(pml_point2, outer_point2)

            outer_loop = gmsh.model.occ.addCurveLoop([farfield_boundary, outer_join2, domain_boundary, outer_join1])
            inner_loop = gmsh.model.occ.addCurveLoop([farfield_boundary, inner_join])
            pml_loop = gmsh.model.occ.addCurveLoop([pml_boundary, pml_join2, domain_boundary, pml_join1])
            
            outer_domain = gmsh.model.occ.addPlaneSurface([outer_loop])
            inner_domain = gmsh.model.occ.addPlaneSurface([inner_loop])
            pml_domain = gmsh.model.occ.addPlaneSurface([pml_loop])

            # To conform with the formulation in the cylindrical PML
            pml_boundary = [pml_boundary]
            farfield_boundary = [farfield_boundary]
            def PMLSurface(CoM):
                return(np.allclose(CoM[0], 2*PML.radius/np.pi))
            CoM1 = gmsh.model.occ.getCenterOfMass(fdim, farfield_boundary[0])
            def FarfieldSurface(CoM):
                return np.allclose(CoM, CoM1) 
            
        else: # Cylindrical PML
            if not PMLpenetrate: # Far field boundary parallel to PML everywhere
                inner_point1 = gmsh.model.occ.addPoint(0, PML.zb+FF_d, 0)
                inner_point2 = gmsh.model.occ.addPoint(0, PML.zt-FF_d, 0)
                inner_point3 = gmsh.model.occ.addPoint(PML.rho-FF_d, PML.zt-FF_d, 0)
                inner_point4 = gmsh.model.occ.addPoint(PML.rho-FF_d, PML.zb+FF_d, 0)
                outer_point1 = gmsh.model.occ.addPoint(0, PML.zb+PML.d, 0)
                outer_point2 = gmsh.model.occ.addPoint(0, PML.zt-PML.d, 0)
                outer_point3 = gmsh.model.occ.addPoint(PML.rho-PML.d, PML.zt-PML.d, 0)
                outer_point4 = gmsh.model.occ.addPoint(PML.rho-PML.d, PML.zb+PML.d, 0)
                pml_point1 = gmsh.model.occ.addPoint(0, PML.zb, 0)
                pml_point2 = gmsh.model.occ.addPoint(0, PML.zt, 0)
                pml_point3 = gmsh.model.occ.addPoint(PML.rho, PML.zt, 0)
                pml_point4 = gmsh.model.occ.addPoint(PML.rho, PML.zb, 0)

                farfield_line1 = gmsh.model.occ.addLine(inner_point2, inner_point3)
                farfield_line2 = gmsh.model.occ.addLine(inner_point3, inner_point4)
                farfield_line3 = gmsh.model.occ.addLine(inner_point4, inner_point1)
                farfield_boundary = [farfield_line1, farfield_line2, farfield_line3]
                domain_line1 = gmsh.model.occ.addLine(outer_point2, outer_point3)
                domain_line2 = gmsh.model.occ.addLine(outer_point3, outer_point4)
                domain_line3 = gmsh.model.occ.addLine(outer_point4, outer_point1)
                domain_boundary = [domain_line1, domain_line2, domain_line3]
                pml_line1 = gmsh.model.occ.addLine(pml_point2, pml_point3)
                pml_line2 = gmsh.model.occ.addLine(pml_point3, pml_point4)
                pml_line3 = gmsh.model.occ.addLine(pml_point4, pml_point1)
                pml_boundary = [pml_line1, pml_line2, pml_line3]

                outer_join1 = gmsh.model.occ.addLine(inner_point1, outer_point1)
                outer_join2 = gmsh.model.occ.addLine(inner_point2, outer_point2)
                inner_join = gmsh.model.occ.addLine(inner_point1, inner_point2)
                pml_join1 = gmsh.model.occ.addLine(pml_point1, outer_point1)
                pml_join2 = gmsh.model.occ.addLine(pml_point2, outer_point2)

                outer_loop = gmsh.model.occ.addCurveLoop(farfield_boundary + [outer_join2] + domain_boundary + [outer_join1])
                inner_loop = gmsh.model.occ.addCurveLoop(farfield_boundary + [inner_join])
                pml_loop = gmsh.model.occ.addCurveLoop(pml_boundary + [pml_join2] + domain_boundary + [pml_join1])

                outer_domain = gmsh.model.occ.addPlaneSurface([outer_loop])
                inner_domain = gmsh.model.occ.addPlaneSurface([inner_loop])
                pml_domain = gmsh.model.occ.addPlaneSurface([pml_loop])

                CoM1 = gmsh.model.occ.getCenterOfMass(fdim, farfield_line1)
                CoM2 = gmsh.model.occ.getCenterOfMass(fdim, farfield_line2)
                CoM3 = gmsh.model.occ.getCenterOfMass(fdim, farfield_line3)
                def FarfieldSurface(CoM):
                    return np.allclose(CoM, CoM1) or np.allclose(CoM, CoM2) or np.allclose(CoM, CoM3)
                
            else: # The radome penetrates into the lower PML, no FF there
                # inner_point1 is the same as outer_point1
                # inner_point1 = gmsh.model.occ.addPoint(0, PML.zb+PML.d, 0)
                inner_point2 = gmsh.model.occ.addPoint(0, PML.zt-FF_d, 0)
                inner_point3 = gmsh.model.occ.addPoint(PML.rho-FF_d, PML.zt-FF_d, 0)
                inner_point4 = gmsh.model.occ.addPoint(PML.rho-FF_d, PML.zb+PML.d, 0)
                outer_point1 = gmsh.model.occ.addPoint(0, PML.zb+PML.d, 0)
                outer_point2 = gmsh.model.occ.addPoint(0, PML.zt-PML.d, 0)
                outer_point3 = gmsh.model.occ.addPoint(PML.rho-PML.d, PML.zt-PML.d, 0)
                outer_point4 = gmsh.model.occ.addPoint(PML.rho-PML.d, PML.zb+PML.d, 0)
                pml_point1 = gmsh.model.occ.addPoint(0, PML.zb, 0)
                pml_point2 = gmsh.model.occ.addPoint(0, PML.zt, 0)
                pml_point3 = gmsh.model.occ.addPoint(PML.rho, PML.zt, 0)
                pml_point4 = gmsh.model.occ.addPoint(PML.rho, PML.zb, 0)

                farfield_line1 = gmsh.model.occ.addLine(inner_point2, inner_point3)
                farfield_line2 = gmsh.model.occ.addLine(inner_point3, inner_point4)
                farfield_boundary = [farfield_line1, farfield_line2]
                domain_line1 = gmsh.model.occ.addLine(outer_point2, outer_point3)
                domain_line2 = gmsh.model.occ.addLine(outer_point3, outer_point4)
                domain_boundary = [domain_line2, domain_line1] # Needed in reverse order when constructing loops
                pml_line1 = gmsh.model.occ.addLine(pml_point2, pml_point3)
                pml_line2 = gmsh.model.occ.addLine(pml_point3, pml_point4)
                pml_line3 = gmsh.model.occ.addLine(pml_point4, pml_point1)
                pml_boundary = [pml_line1, pml_line2, pml_line3]
                outer_join1 = gmsh.model.occ.addLine(inner_point4, outer_point4)
                outer_join2 = gmsh.model.occ.addLine(inner_point2, outer_point2)
                inner_join1 = gmsh.model.occ.addLine(inner_point4, outer_point1)
                inner_join2 = gmsh.model.occ.addLine(outer_point1, inner_point2)
                pml_join1 = gmsh.model.occ.addLine(pml_point1, outer_point1)
                pml_join2 = gmsh.model.occ.addLine(pml_point2, outer_point2)

                outer_loop = gmsh.model.occ.addCurveLoop(farfield_boundary + [outer_join1] + domain_boundary + [outer_join2])
                inner_loop = gmsh.model.occ.addCurveLoop(farfield_boundary + [inner_join1, inner_join2])
                pml_loop = gmsh.model.occ.addCurveLoop(pml_boundary + [pml_join1, inner_join1, outer_join1] + domain_boundary + [pml_join2])

                outer_domain = gmsh.model.occ.addPlaneSurface([outer_loop])
                inner_domain = gmsh.model.occ.addPlaneSurface([inner_loop])
                pml_domain = gmsh.model.occ.addPlaneSurface([pml_loop])

                CoM1 = gmsh.model.occ.getCenterOfMass(fdim, farfield_line1)
                CoM2 = gmsh.model.occ.getCenterOfMass(fdim, farfield_line2)
                def FarfieldSurface(CoM):
                    return np.allclose(CoM, CoM1) or np.allclose(CoM, CoM2)

            def PMLSurface(CoM):
                return(np.allclose(CoM[0], PML.rho) or np.allclose(CoM[1], PML.zt) or np.allclose(CoM[1], PML.zb))
            
#        gmsh.model.occ.synchronize() # Seems to cause trouble with gmsh 4.12.2, according to experiments 240703

        # Fragment radome domain from inner domain and PML
        if PMLpenetrate: # Take several fragmentations into account
            if False: # Original code, remove later
                foo = gmsh.model.occ.fragment([(tdim, inner_domain), (tdim, pml_domain)], [(tdim, radome_domain), (tdim, transition_domain), (tdim, CFRP_domain)])
                radome_domain_dimtags = foo[1][-3]
                transition_domain_dimtags =foo[1][-2]
                CFRP_domain_dimtags = foo[1][-1]
                tmp_dimtags = radome_domain_dimtags + transition_domain_dimtags + CFRP_domain_dimtags
                # Create lists of dimtags not containing the tool (the
                # radome), since we cannot set multiple tags on the cells
                inner_domain_dimtags = [x for x in foo[1][0] if x not in tmp_dimtags]
                outer_domain_dimtags = [(tdim, outer_domain)]
                pml_domain_dimtags = [x for x in foo[1][1] if x not in tmp_dimtags]
                pml_CFRP_overlap_dimtags = [x for x in foo[1][1] if x in tmp_dimtags]
                CFRP_domain_dimtags = [x for x in CFRP_domain_dimtags if x not in pml_CFRP_overlap_dimtags]
#                farfield_boundary_dimtags = [(fdim, x) for x in farfield_boundary]
            else:
                outDimTags, outDimTagsMap= gmsh.model.occ.fragment([(tdim, inner_domain), (tdim, pml_domain)], [(tdim, radome_domain), (tdim, transition_domain), (tdim, CFRP_domain)])
                radome_domain_dimtags = outDimTagsMap[-3]
                transition_domain_dimtags = outDimTagsMap[-2]
                CFRP_domain_dimtags = outDimTagsMap[-1]
                tmp_dimtags = radome_domain_dimtags + transition_domain_dimtags + CFRP_domain_dimtags
                # Create lists of dimtags not containing the tool (the
                # radome), since we cannot set multiple tags on the cells
                inner_domain_dimtags = [x for x in outDimTagsMap[0] if x not in tmp_dimtags]
                outer_domain_dimtags = [(tdim, outer_domain)]
                pml_domain_dimtags = [x for x in outDimTagsMap[1] if x not in tmp_dimtags]
                pml_CFRP_overlap_dimtags = [x for x in outDimTagsMap[1] if x in tmp_dimtags]
                CFRP_domain_dimtags = [x for x in CFRP_domain_dimtags if x not in pml_CFRP_overlap_dimtags]
#                farfield_boundary_dimtags = [(fdim, x) for x in farfield_boundary]
        else: # The radome is completely enclosed in the inner domain
            if False: # Original code, remove later
                foo = gmsh.model.occ.fragment([(tdim, inner_domain)], [(tdim, radome_domain)])
                radome_domain_dimtags = foo[1][-1]
                # Create lists of dimtags not containing the tool (the radome)
                inner_domain_dimtags = [x for x in foo[1][0] if x not in radome_domain_dimtags]
                outer_domain_dimtags = [(tdim, outer_domain)]
                pml_domain_dimtags = [(tdim, pml_domain)]
#                farfield_boundary_dimtags = [(fdim, x) for x in farfield_boundary]
                pml_radome_overlap_dimtags = [(tdim, -1)]
            else:
                outDimTags, outDimTagsMap = gmsh.model.occ.fragment([(tdim, inner_domain)], [(tdim, radome_domain), (tdim, transition_domain), (tdim, CFRP_domain)])
                radome_domain_dimtags = outDimTagsMap[-3]
                transition_domain_dimtags = outDimTagsMap[-2]
                CFRP_domain_dimtags = outDimTagsMap[-1]
                tmp_dimtags = radome_domain_dimtags + transition_domain_dimtags + CFRP_domain_dimtags
                # Create lists of dimtags not containing the tool (the radome)
                inner_domain_dimtags = [x for x in outDimTagsMap[0] if x not in tmp_dimtags]
                outer_domain_dimtags = [(tdim, outer_domain)]
                pml_domain_dimtags = [(tdim, pml_domain)]
#                farfield_boundary_dimtags = [(fdim, x) for x in farfield_boundary]
                pml_CFRP_overlap_dimtags = [(tdim, -1)]
        gmsh.model.occ.synchronize()

        # Cut away the antenna region from the inner domain
        if AntennaMetalBase: # Antenna + base also intersects radome and PML
            if False:
                inner_domain_dimtags, _ = gmsh.model.occ.cut(inner_domain_dimtags, [(tdim, antenna_domain)], removeTool=False)
                transition_domain_dimtags, _ = gmsh.model.occ.cut(transition_domain_dimtags, [(tdim, antenna_domain)], removeTool=False)
#                CFRP_domain_dimtags, _ = gmsh.model.occ.cut(CFRP_domain_dimtags, [(tdim, antenna_domain)], removeTool=False)
                pml_domain_dimtags, _ = gmsh.model.occ.cut(pml_domain_dimtags, [(tdim, antenna_domain)], removeTool=False)
#                pml_CFRP_overlap_dimtags, _ = gmsh.model.occ.cut(pml_CFRP_overlap_dimtags, [(tdim, antenna_domain)], removeTool=False)
                gmsh.model.occ.remove(CFRP_domain_dimtags, recursive=True)
                gmsh.model.occ.remove(pml_CFRP_overlap_dimtags, recursive=True)
                gmsh.model.occ.remove([(tdim, antenna_domain)], recursive=True)
            elif True:
                outDimTags, outDimTagsMap = gmsh.model.occ.fragment(inner_domain_dimtags, [(tdim, antenna_domain)], removeTool=False)
                inner_domain_dimtags = [(tdim, x[1]) for x in outDimTags if x not in outDimTagsMap[-1]]
                gmsh.model.occ.remove([(tdim, x[1]) for x in outDimTagsMap[2]], recursive=True)
                outDimTags, outDimTagsMap = gmsh.model.occ.fragment(pml_domain_dimtags, [(tdim, antenna_domain)], removeTool=True)
                pml_domain_dimtags = [(tdim, x[1]) for x in outDimTags if x not in outDimTagsMap[-1]]
                gmsh.model.occ.remove([(tdim, x[1]) for x in outDimTagsMap[-1]], recursive=True)
                gmsh.model.occ.remove(CFRP_domain_dimtags, recursive=True)
                gmsh.model.occ.remove(pml_CFRP_overlap_dimtags, recursive=True)
            else:
                print(inner_domain_dimtags, transition_domain_dimtags, CFRP_domain_dimtags, pml_CFRP_overlap_dimtags, pml_domain_dimtags)
                outDimTags, outDimTagsMap = gmsh.model.occ.fragment(inner_domain_dimtags + transition_domain_dimtags + CFRP_domain_dimtags + pml_CFRP_overlap_dimtags + pml_domain_dimtags, [(tdim, antenna_domain)])
                print(outDimTags, outDimTagsMap)
                inner_domain_dimtags = [(tdim, x[1]) for x in outDimTagsMap[0] + outDimTagsMap[1] if x not in outDimTagsMap[-1]]
                transition_domain_dimtags = [(tdim, x[1]) for x in outDimTagsMap[2] if x not in outDimTagsMap[-1]]
                CFRP_domain_dimtags = [(tdim, x[1]) for x in outDimTagsMap[3] if x not in outDimTagsMap[-1]]
                pml_CFRP_overlap_dimtags = [(tdim, x[1]) for x in outDimTagsMap[4] if x not in outDimTagsMap[-1]]
                pml_domain_dimtags = [(tdim, x[1]) for x in outDimTagsMap[5] + outDimTagsMap[6] if x not in outDimTagsMap[-1]]
                gmsh.model.occ.remove(outDimTagsMap[-1], recursive=True)
        elif Antenna: # Antenna completely inside inner domain
            inner_domain_dimtags, _ = gmsh.model.occ.cut(inner_domain_dimtags, [(tdim, antenna_domain)], removeTool=False)
            gmsh.model.occ.remove([(tdim, antenna_domain)], recursive=True)
        gmsh.model.occ.synchronize()

        # Create physical groups for domains
        freespace_marker = gmsh.model.addPhysicalGroup(tdim, [x[1] for x in inner_domain_dimtags+outer_domain_dimtags])
        material_marker = gmsh.model.addPhysicalGroup(tdim, [x[1] for x in radome_domain_dimtags])
        transition_marker = gmsh.model.addPhysicalGroup(tdim, [x[1] for x in transition_domain_dimtags])
        CFRP_marker = gmsh.model.addPhysicalGroup(tdim, [x[1] for x in CFRP_domain_dimtags])
        pml_marker = gmsh.model.addPhysicalGroup(tdim, [x[1] for x in pml_domain_dimtags])
        if PMLpenetrate and not AntennaMetalBase:
            pml_CFRP_overlap_marker = gmsh.model.addPhysicalGroup(tdim, [x[1] for x in pml_CFRP_overlap_dimtags])
        else:
            pml_CFRP_overlap_marker = -1
        subdomain_markers = {'freespace': freespace_marker,
                             'material': material_marker,
                             'transition': transition_marker,
                             'CFRP': CFRP_marker,
                             'pml': pml_marker,
                             'pml_CFRP_overlap': pml_CFRP_overlap_marker}

        # Create physical groups for antenna surfaces and far field
        # boundary. Based on geometry functions rather than gmsh
        # entities except for far field (after many tries). 
        axis_lines = []
        pec_surface = []
        antenna_surface = []
        pml_surface = []
        farfield_surface = []
        for boundary in gmsh.model.occ.getEntities(dim=fdim):
            CoM = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            if Antenna or AntennaMetalBase:
                if PECSurface(CoM):
                    pec_surface.append(boundary[1])
                elif AntennaSurface(CoM):
                    antenna_surface.append(boundary[1])
            if PMLSurface(CoM):
                pml_surface.append(boundary[1])
            elif FarfieldSurface(CoM):
                farfield_surface.append(boundary[1])
            elif np.allclose(CoM[0], 0):
                # Need to be careful not adding old lines remaining after
                # cutting out antenna shape, they do not seem to disappear
                if not np.allclose(CoM, [0, 0, 0]) and not np.allclose(CoM, [0, -t/2, 0]): 
                    axis_lines.append(boundary[1])
        if Antenna or AntennaMetalBase:
            pec_surface_marker = gmsh.model.addPhysicalGroup(fdim, pec_surface)
            antenna_surface_marker = gmsh.model.addPhysicalGroup(fdim, antenna_surface)
        else:
            pec_surface_marker = -1
            antenna_surface_marker = -1
#        farfield_surface_marker = gmsh.model.addPhysicalGroup(fdim, [x[1] for x in farfield_boundary_dimtags])
        farfield_surface_marker = gmsh.model.addPhysicalGroup(fdim, farfield_surface)
        pml_surface_marker = gmsh.model.addPhysicalGroup(fdim, pml_surface)
        axis_marker = gmsh.model.addPhysicalGroup(fdim, axis_lines)
#        gmsh.model.occ.synchronize()
        boundary_markers = {'pec': pec_surface_marker,
                            'antenna': antenna_surface_marker,
                            'farfield': farfield_surface_marker,
                            'pml': pml_surface_marker,
                            'axis': axis_marker}

        # Refine mesh in CFRP and transition region
        tol = 1e-6
        if AntennaMetalBase:
            points_dimtags = gmsh.model.getEntitiesInBoundingBox(Ra-tol, -Htransition-tol, 0, Rb+tol, 0+tol, 0, dim=0)
        else:
            points_dimtags = gmsh.model.getEntitiesInBoundingBox(Ra-tol, -H-tol, 0, Rb+tol, 0+tol, 0, dim=0)
        
        
#        if PMLpenetrate:
#            if not AntennaMetalBase:
#                surfs = [x[1] for x in pml_CFRP_overlap_dimtags + CFRP_domain_dimtags + transition_domain_dimtags]
#            else:
#                surfs = [x[1] for x in transition_domain_dimtags]
#        else:
#            surfs = [x[1] for x in CFRP_domain_dimtags + transition_domain_dimtags]
#        curves = []
#        for s in surfs:
#            curves += gmsh.model.getAdjacencies(2, s)[1].tolist()
#        points = []
#        for c in curves:
#            points += gmsh.model.getAdjacencies(1, c)[1].tolist()
#        tol = 1e-6
#        points_dimtags = gmsh.model.getEntitiesInBoundingBox(Ra-tol, -Htransition-tol, 0, Rb+tol, 0+tol, 0, dim=0)
#        CoMs = [gmsh.model.occ.getCenterOfMass(0, p) for p in points]
#        for p in gmsh.model.occ.getEntities(dim=0):
#            if gmsh.model.occ.getCenterOfMass(0, p) in CoMs:
#                if not p in points:
#                    points += [p]
#                    print('hej')
#        points_dimtags = [(0, x) for x in points]
#        print(surfs, curves, points, points_dimtags)
#        gmsh.model.occ.mesh.setSize(points_dimtags, hfine)
#        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(points_dimtags, hfine)
        
        # Generate mesh
        gmsh.model.mesh.generate(tdim)
        gmsh.model.mesh.removeDuplicateNodes() # Some nodes seem to be duplicated
#        gmsh.model.mesh.removeDuplicateElements()
        
        # Save mesh for retrieval of function spaces later on
        gmsh.write(filename)

        if visualize:
            gmsh.fltk.run()
    else: # Some data generated by the meshing that is needed on all ranks
        subdomain_markers = None
        boundary_markers = None
    subdomain_markers = comm.bcast(subdomain_markers, root=model_rank)
    boundary_markers = comm.bcast(boundary_markers, root=model_rank)

    mesh, subdomains, boundaries = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, comm=comm, rank=model_rank, gdim=tdim)
    gmsh.finalize()

    meshdata = MeshData(mesh=mesh, subdomains=subdomains, boundaries=boundaries, subdomain_markers=subdomain_markers, boundary_markers=boundary_markers, PML=PML, comm=comm, model_rank=model_rank)

    return(meshdata)

def CreateMeshSphere(
        comm=MPI.COMM_WORLD,        # MPI communicator
        model_rank=0,               # Rank of modelling process
        radius_sphere=1.0,          # Radius of scattering sphere
        radius_farfield=1.5,        # Radius of farfield surface
        radius_domain=2.0,          # Radius of computational domain
        radius_pml=2.5,             # Outer radius of PML
        pec=False,                  # Whether to have PEC sphere or not
        PMLcylindrical=False,       # To have a PML in cylindrical coordinates
        h=0.1,                      # Typical mesh size
        verbosity=1,                # Verbosity of gmsh
        visualize=False,            # Whether to visualize the mesh
        filename='spheremesh.msh'   # Name of file to save mesh in
):
    """Create the mesh using gmsh."""

    # Set up PML data
    PML = PerfectlyMatchedLayer()
    PML.cylindrical = PMLcylindrical
    PML.d = radius_pml - radius_domain
    PML.zt = radius_pml
    PML.zb = -radius_pml
    PML.rho = radius_pml
    PML.radius = radius_pml
    FF_d = radius_pml - radius_farfield
    
    gmsh.initialize()
    if comm.rank == model_rank:
        # Typical mesh size
        gmsh.option.setNumber('General.Verbosity', verbosity)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

        # Create sphere boundary and domain (if applicable)
        sphere_boundary = gmsh.model.occ.addCircle(0, 0, 0, radius_sphere, -1, -np.pi/2, np.pi/2)
        sphere_point1 = gmsh.model.occ.addPoint(0, -radius_sphere, 0)
        sphere_point2 = gmsh.model.occ.addPoint(0, radius_sphere, 0)
        if not pec:
            sphere_join = gmsh.model.occ.addLine(sphere_point1, sphere_point2)
            sphere_loop = gmsh.model.occ.addCurveLoop([sphere_boundary, sphere_join])
            sphere_domain = gmsh.model.occ.addPlaneSurface([sphere_loop])

        # Create free space domain (one on each side of the farfield boundary)
        if not PMLcylindrical: # Spherical PML
            inner_point1 = gmsh.model.occ.addPoint(0, -radius_farfield, 0)
            inner_point2 = gmsh.model.occ.addPoint(0, radius_farfield, 0)
            outer_point1 = gmsh.model.occ.addPoint(0, -radius_domain, 0)
            outer_point2 = gmsh.model.occ.addPoint(0, radius_domain, 0)
            farfield_boundary = gmsh.model.occ.addCircle(0, 0, 0, radius_farfield, -1, -np.pi/2, np.pi/2)
            domain_boundary = gmsh.model.occ.addCircle(0, 0, 0, radius_domain, -1, -np.pi/2, np.pi/2)
        else: # Cylindrical PML
            inner_point1 = gmsh.model.occ.addPoint(0, PML.zb+FF_d, 0)
            inner_point2 = gmsh.model.occ.addPoint(0, PML.zt-FF_d, 0)
            inner_point3 = gmsh.model.occ.addPoint(PML.rho-FF_d, PML.zt-FF_d, 0)
            inner_point4 = gmsh.model.occ.addPoint(PML.rho-FF_d, PML.zb+FF_d, 0)
            outer_point1 = gmsh.model.occ.addPoint(0, PML.zb+PML.d, 0)
            outer_point2 = gmsh.model.occ.addPoint(0, PML.zt-PML.d, 0)
            outer_point3 = gmsh.model.occ.addPoint(PML.rho-PML.d, PML.zt-PML.d, 0)
            outer_point4 = gmsh.model.occ.addPoint(PML.rho-PML.d, PML.zb+PML.d, 0)

            farfield_boundary = gmsh.model.occ.addBSpline([inner_point2, inner_point3, inner_point4, inner_point1], degree=1)
            domain_boundary = gmsh.model.occ.addBSpline([outer_point2, outer_point3, outer_point4, outer_point1], degree=1)

        outer_join1 = gmsh.model.occ.addLine(inner_point1, outer_point1)
        outer_join2 = gmsh.model.occ.addLine(inner_point2, outer_point2)
        inner_join1 = gmsh.model.occ.addLine(inner_point1, sphere_point1)
        inner_join2 = gmsh.model.occ.addLine(inner_point2, sphere_point2)

        outer_loop = gmsh.model.occ.addCurveLoop([farfield_boundary, outer_join2, domain_boundary, outer_join1])
        inner_loop = gmsh.model.occ.addCurveLoop([farfield_boundary, inner_join1, sphere_boundary, inner_join2])

        outer_domain = gmsh.model.occ.addPlaneSurface([outer_loop])
        inner_domain = gmsh.model.occ.addPlaneSurface([inner_loop])
            
        # Create PML domain
        if not PMLcylindrical: # Spherical PML
            pml_point1 = gmsh.model.occ.addPoint(0, -radius_pml, 0)
            pml_point2 = gmsh.model.occ.addPoint(0, radius_pml, 0)
            pml_boundary = gmsh.model.occ.addCircle(0, 0, 0, radius_pml, -1, -np.pi/2, np.pi/2)
        else: # Cylindrical PML
            pml_point1 = gmsh.model.occ.addPoint(0, PML.zb, 0)
            pml_point2 = gmsh.model.occ.addPoint(0, PML.zt, 0)
            pml_point3 = gmsh.model.occ.addPoint(PML.rho, PML.zt, 0)
            pml_point4 = gmsh.model.occ.addPoint(PML.rho, PML.zb, 0)
            pml_boundary = gmsh.model.occ.addBSpline([pml_point2, pml_point3, pml_point4, pml_point1], degree=1)
            
        pml_join1 = gmsh.model.occ.addLine(pml_point1, outer_point1)
        pml_join2 = gmsh.model.occ.addLine(pml_point2, outer_point2)
        pml_loop = gmsh.model.occ.addCurveLoop([pml_boundary, pml_join2, domain_boundary, pml_join1])
        pml_domain = gmsh.model.occ.addPlaneSurface([pml_loop])
            
        gmsh.model.occ.synchronize()

        # Create physical groups for domains
        if pec:
            sphere_marker = -1
        else:
            sphere_marker = gmsh.model.addPhysicalGroup(tdim, [sphere_domain])
        material_marker = sphere_marker
        freespace_marker = gmsh.model.addPhysicalGroup(tdim, [inner_domain, outer_domain])
        pml_marker = gmsh.model.addPhysicalGroup(tdim, [pml_domain])

        subdomain_markers = {'freespace': freespace_marker,
                             'material': material_marker,
                             'pml': pml_marker,
                             'pml_material_overlap': -1}

        # Create physical groups for sphere surface and far field boundary
        sphere_surface_marker = gmsh.model.addPhysicalGroup(fdim, [sphere_boundary])
        if pec:
            pec_surface_marker = sphere_surface_marker
        else:
            pec_surface_marker = -1
        antenna_surface_marker = -1 # No antenna surface in this mesh
        farfield_surface_marker = gmsh.model.addPhysicalGroup(fdim, [farfield_boundary])
        pml_surface_marker = gmsh.model.addPhysicalGroup(fdim, [pml_boundary])
        if pec:
            axis_marker = gmsh.model.addPhysicalGroup(fdim, [pml_join1, pml_join2, outer_join1, outer_join2, inner_join1, inner_join2])
        else:
            axis_marker = gmsh.model.addPhysicalGroup(fdim, [pml_join1, pml_join2, outer_join1, outer_join2, inner_join1, inner_join2, sphere_join])

        gmsh.model.occ.synchronize()
        boundary_markers = {'pec': pec_surface_marker,
                            'antenna': antenna_surface_marker,
                            'farfield': farfield_surface_marker,
                            'pml': pml_surface_marker,
                            'axis': axis_marker}
    
        # Generate mesh
        gmsh.model.mesh.generate(tdim)
        gmsh.model.mesh.removeDuplicateNodes() # Some nodes seem to be duplicated

        # Save mesh for retrieval of function spaces later on
        gmsh.write(filename)

        if visualize:
            gmsh.fltk.run()
    else: # Some data generated by the meshing that is needed on all ranks
        subdomain_markers = None
        boundary_markers = None
    subdomain_markers = comm.bcast(subdomain_markers, root=model_rank)
    boundary_markers = comm.bcast(boundary_markers, root=model_rank)

    mesh, subdomains, boundaries = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, comm=comm, rank=model_rank, gdim=tdim)
    gmsh.finalize()

    meshdata = MeshData(mesh=mesh, subdomains=subdomains, boundaries=boundaries, subdomain_markers=subdomain_markers, boundary_markers=boundary_markers, PML=PML, comm=comm, model_rank=model_rank)
    return(meshdata)

if __name__ == '__main__':
    # Create and visualize the mesh if run from the prompt
    if True:
        meshdata = CreateMeshOgive(visualize=True, h=0.1*lambda0, PMLcylindrical=True, PMLpenetrate=False, Antenna=True, AntennaMetalBase=True, t=lambda0/4*8, Htransition=1*lambda0, hfine=0.01*lambda0)
    else:
        pec = True
        meshdata = CreateMeshSphere(pec=pec, visualize=True, PMLcylindrical=True, h=0.1)

    if False:
        # Visualize mesh and dofs as handled by dolfinx
        import pyvista as pv

        mesh = meshdata.mesh
        subdomains = meshdata.subdomains
        boundaries = meshdata.boundaries
        
        freespace_marker = meshdata.subdomain_markers['freespace']
        material_marker = meshdata.subdomain_markers['material']
        pml_marker = meshdata.subdomain_markers['pml']
        pml_material_overlap_marker = meshdata.subdomain_markers['pml_material_overlap']
        
        pec_surface_marker = meshdata.boundary_markers['pec']
        antenna_surface_marker = meshdata.boundary_markers['antenna']
        farfield_surface_marker = meshdata.boundary_markers['farfield']
        pml_surface_marker = meshdata.boundary_markers['pml']
        axis_marker = meshdata.boundary_markers['axis']
        
        W_DG = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
        chi = dolfinx.fem.Function(W_DG)
        chi.x.array[:] = 0.0
        if True: # Indicate regions
            freespace_cells = subdomains.find(freespace_marker)
            freespace_dofs = dolfinx.fem.locate_dofs_topological(W_DG, entity_dim=tdim, entities=freespace_cells)
            if material_marker >= 0:
                material_cells = subdomains.find(material_marker)
                material_dofs = dolfinx.fem.locate_dofs_topological(W_DG, entity_dim=tdim, entities=material_cells)
                chi.x.array[material_dofs] = 2.0
            if pml_material_overlap_marker >= 0:
                pml_material_overlap_cells = subdomains.find(pml_material_overlap_marker)
                pml_material_overlap_dofs = dolfinx.fem.locate_dofs_topological(W_DG, entity_dim=tdim, entities=pml_material_overlap_cells)
                chi.x.array[pml_material_overlap_dofs] = 4.0
            pml_cells = subdomains.find(pml_marker)
            pml_dofs = dolfinx.fem.locate_dofs_topological(W_DG, entity_dim=tdim, entities=pml_cells)
            chi.x.array[freespace_dofs] = 1.0
            chi.x.array[pml_dofs] = 3.0
        if True: # Indicate boundaries
            if pec_surface_marker >= 0:
                pec_surface_cells = boundaries.find(pec_surface_marker)
                pec_surface_dofs = dolfinx.fem.locate_dofs_topological(W_DG, entity_dim=fdim, entities=pec_surface_cells)
                chi.x.array[pec_surface_dofs] = 10.0
            if antenna_surface_marker >= 0:
                antenna_surface_cells = boundaries.find(antenna_surface_marker)
                antenna_surface_dofs = dolfinx.fem.locate_dofs_topological(W_DG, entity_dim=fdim, entities=antenna_surface_cells)
                chi.x.array[antenna_surface_dofs] = 20.0
            farfield_surface_cells = boundaries.find(farfield_surface_marker)
            farfield_surface_dofs = dolfinx.fem.locate_dofs_topological(W_DG, entity_dim=fdim, entities=farfield_surface_cells)
            pml_surface_cells = boundaries.find(pml_surface_marker)
            pml_surface_dofs = dolfinx.fem.locate_dofs_topological(W_DG, entity_dim=fdim, entities=pml_surface_cells)
            axis_cells = boundaries.find(axis_marker)
            axis_dofs = dolfinx.fem.locate_dofs_topological(W_DG, entity_dim=fdim, entities=axis_cells)
            chi.x.array[farfield_surface_dofs] = 30.0
            chi.x.array[pml_surface_dofs] = 40.0
            chi.x.array[axis_dofs] = 50.0
        cells, cell_types, x = dolfinx.plot.create_vtk_mesh(mesh, tdim)
        grid = pv.UnstructuredGrid(cells, cell_types, x)
        grid["chi"] = np.real(chi.x.array)
        plotter = pv.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        plotter.view_xy()
        plotter.add_axes()
        plotter.show()

