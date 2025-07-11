# Compute scattering for a rotationally symmetric structure. Exciting
# field is a plane wave and/or antenna field, with arbitrary
# polarization and propagation direction.
#
# The code makes use of MPI to enable parallel processing. However,
# you may need to execute "export OMP_NUM_THREADS=1" when running with
# something like "mpirun -n 4 python testmpi.py", to prevent PETSc
# from starting too many threads, see for instance
# https://fenicsproject.discourse.group/t/running-in-parallel-slower-than-serial/1661
# 
# Daniel Sjöberg, 2024-07-07

from mpi4py import MPI
import numpy as np
import dolfinx, ufl, basix
import dolfinx.fem.petsc
from petsc4py import PETSc
from matplotlib import pyplot as plt
import mesh_rotsymradome
import sphericalvectorwaves as svw
import functools
import pyvista as pv
import pyvistaqt as pvqt
from scipy.special import jv, jvp
import os, sys
import time

# Set up physical constants
from scipy.constants import c as c0
mu0 = 4*np.pi*1e-7
eps0 = 1/c0**2/mu0
eta0 = np.sqrt(mu0/eps0)
TimeConvention = 'j'

tdim = 2                             # Dimension of triangles/tetraedra
fdim = tdim - 1                      # Dimension of facets


########################################################################
#
# Some topics for future development
#
# Possibly save terms in Fm proportional to exp(n*j*phi), where n =
# -2, -1, 0, 1, 2. This would facilitate computing 3D farfield
# patterns from a minimum of data.
#
# Project far field on spherical vector waves?
#
# Parallelization: how to save solutions (renumbering on different
# processes). Perhaps create a global system on model_rank.
# Gather solutions in parallel: https://fenicsproject.discourse.group/t/gather-solutions-in-parallel-in-fenicsx/5907/5
#
########################################################################

        
class RotSymProblem():
    """Class to hold definitions and functions for simulating scattering or transmission of electromagnetic waves for a rotationally symmetric structure."""
    def __init__(self,
                 meshdata,            # Mesh and metadata
                 f0=10e9,             # Frequency of the problem
                 epsr_bkg=1,          # Permittivity of the background medium
                 mur_bkg=1,           # Permeability of the background medium
                 material_epsr=1+0j,  # Permittivity of scatterer material
                 material_mur=1+0j,   # Permeability of scatterer material
                 hull_epsr=1+0j,      # Permittivity of hull (fuselage)
                 hull_mur=1+0j,       # Permeability of hull (fuselage)
                 transition_volumefraction_hull=None,
                 degree=3,            # Degree of finite elements
                 comm=MPI.COMM_WORLD, # MPI communicator
                 model_rank=0,        # Model rank for saving, plotting etc
                 ghost_ff_facets=None, 
                 ):
        """Initialize the problem."""
        self.lambda0 = c0/f0                      # Vacuum wavelength
        self.k0 = 2*np.pi*f0/c0                   # Vacuum wavenumber
        self.epsr_bkg = epsr_bkg
        self.mur_bkg = mur_bkg
        self.n_bkg = np.sqrt(epsr_bkg*mur_bkg)    # Background refractive index
        self.etar_bkg = np.sqrt(mur_bkg/epsr_bkg) # Background relative wave impedance
        self.material_epsr = material_epsr
        self.material_mur = material_mur
        self.hull_epsr = hull_epsr
        self.hull_mur = hull_mur
        if transition_volumefraction_hull == None:
            self.transition_volumefraction_hull = lambda x: np.zeros(x.shape[1])
        else:
            self.transition_volumefraction_hull = transition_volumefraction_hull
        self.degree = degree
        self.ghost_ff_facets = ghost_ff_facets

        # Set up mesh information
        self.mesh = meshdata.mesh
        self.subdomains = meshdata.subdomains
        self.boundaries = meshdata.boundaries
        self.PML = meshdata.PML

        self.freespace_marker = meshdata.subdomain_markers['freespace']
        self.material_marker = meshdata.subdomain_markers['material']
        self.transition_marker = meshdata.subdomain_markers['transition']
        self.hull_marker = meshdata.subdomain_markers['hull']
        self.pml_marker = meshdata.subdomain_markers['pml']
        self.pml_hull_overlap_marker = meshdata.subdomain_markers['pml_hull_overlap']

        self.pec_surface_marker = meshdata.boundary_markers['pec']
        self.antenna_surface_marker = meshdata.boundary_markers['antenna']
        self.farfield_surface_marker = meshdata.boundary_markers['farfield']
        self.pml_surface_marker = meshdata.boundary_markers['pml']
        self.axis_marker = meshdata.boundary_markers['axis']

        self.comm = comm
        self.model_rank = model_rank
        max_r_local = np.sqrt(self.mesh.geometry.x[:,0]**2 + self.mesh.geometry.x[:,1]**2).max()
        max_r_locals = self.comm.gather(max_r_local, root=model_rank)
        max_rho_local = self.mesh.geometry.x[:,0].max()
        max_rho_locals = self.comm.gather(max_rho_local, root=model_rank)
        if self.comm.rank == model_rank:
            max_r = np.max(max_r_locals)
            max_rho = np.max(max_rho_locals)
        else:
            max_r = None
            max_rho = None
        max_r = self.comm.bcast(max_r, root=model_rank)
        max_rho = self.comm.bcast(max_rho, root=model_rank)
        self.max_r = max_r
        self.max_rho = max_rho
        
        # Initialize function spaces, boundary conditions, PML
        self.InitializeFEM()
        self.InitializeFarfieldCells()
        self.InitializeMaterial()
        self.InitializePML()

    def InitializeFEM(self):
        """Set up FEM function spaces and boundary conditions."""
        curl_element = basix.ufl.element('N1curl', self.mesh.basix_cell(), self.degree)
        lagrange_element = basix.ufl.element('CG', self.mesh.basix_cell(), self.degree)
        mixed_element = basix.ufl.mixed_element([curl_element, lagrange_element])
        self.Vspace = dolfinx.fem.functionspace(self.mesh, mixed_element)
        self.V0space, self.V0_dofs = self.Vspace.sub(0).collapse()
        self.V1space, self.V1_dofs = self.Vspace.sub(1).collapse()

        # Set up boundary conditions (relies on using edge elements,
        # be careful with mixed element). Maybe axis conditions can be
        # skipped???
        E0zero = dolfinx.fem.Function(self.V0space)
        E1zero = dolfinx.fem.Function(self.V1space)
        E0zero.interpolate(lambda x: np.zeros((2, x.shape[1]), dtype=complex))
        E1zero.interpolate(lambda x: np.zeros(x.shape[1], dtype=complex))
        axis_dofs0 = dolfinx.fem.locate_dofs_topological(
            (self.Vspace.sub(0), self.V0space), entity_dim=fdim, entities=self.boundaries.find(self.axis_marker))
        axis_dofs1 = dolfinx.fem.locate_dofs_topological(
            (self.Vspace.sub(1), self.V1space), entity_dim=fdim, entities=self.boundaries.find(self.axis_marker))
        self.bc0_axis = dolfinx.fem.dirichletbc(E0zero, axis_dofs0, self.Vspace.sub(0))
        self.bc1_axis = dolfinx.fem.dirichletbc(E1zero, axis_dofs1, self.Vspace.sub(1))

        # Create measures for subdomains and surfaces
        self.dx = ufl.Measure('dx', domain=self.mesh, subdomain_data=self.subdomains, metadata={'quadrature_degree': 5})
        if self.material_marker >= 0:
            if self.transition_marker >= 0 and self.hull_marker >= 0:
                self.dx_dom = self.dx((self.freespace_marker, self.material_marker, self.transition_marker, self.hull_marker))
            else:
                self.dx_dom = self.dx((self.freespace_marker, self.material_marker))
        else:
            self.dx_dom = self.dx(self.freespace_marker)
        if self.pml_hull_overlap_marker >= 0:
            self.dx_pml = self.dx((self.pml_marker, self.pml_hull_overlap_marker))
        else:
            self.dx_pml = self.dx(self.pml_marker)
        self.ds = ufl.Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
        self.ds_axis = self.ds(self.axis_marker)
        if self.antenna_surface_marker >= 0:
            self.ds_antenna = self.ds(self.antenna_surface_marker)
        if self.pec_surface_marker >= 0:
            self.ds_pec = self.ds(self.pec_surface_marker)

        self.dS = ufl.Measure('dS', domain=self.mesh, subdomain_data=self.boundaries)
        self.dS_farfield = self.dS(self.farfield_surface_marker)

        # Compute connectivities
        self.mesh.topology.create_connectivity(tdim, tdim)
        
    def InitializeFarfieldCells(self):
        """Determine the cells adjacent to the farfield curve."""
        cells = []
        ff_facets = self.boundaries.find(self.farfield_surface_marker)
        self.mesh.topology.create_connectivity(fdim, tdim)
        facets_to_cells = self.mesh.topology.connectivity(fdim, tdim)
        for facet in ff_facets:
            for cell in facets_to_cells.links(facet):
                if cell not in cells:
                    cells.append(cell)
        cells.sort()
        self.farfield_cells = np.array(cells)
        
    def InitializeMaterial(self):
        """Set up material parameters."""
        Wspace = dolfinx.fem.functionspace(self.mesh, ("DG", 0))
        # Set up transition region
        self.epsr_transition = dolfinx.fem.Function(Wspace)
        self.mur_transition = dolfinx.fem.Function(Wspace)
        self.epsr_transition.interpolate(lambda x: self.transition_volumefraction_hull(x)*self.hull_epsr + (1 - self.transition_volumefraction_hull(x))*self.material_epsr)
        self.mur_transition.interpolate(lambda x: self.transition_volumefraction_hull(x)*self.hull_mur + (1 - self.transition_volumefraction_hull(x))*self.material_mur)
        # Set up homogeneous material regions
        self.epsr = dolfinx.fem.Function(Wspace)
        self.mur = dolfinx.fem.Function(Wspace)
        self.epsr.x.array[:] = self.epsr_bkg
        self.mur.x.array[:] = self.mur_bkg
        material_cells = self.subdomains.find(self.material_marker)
        material_dofs = dolfinx.fem.locate_dofs_topological(Wspace, entity_dim=tdim, entities=material_cells)
        transition_cells = self.subdomains.find(self.transition_marker)
        transition_dofs = dolfinx.fem.locate_dofs_topological(Wspace, entity_dim=tdim, entities=transition_cells)
        hull_cells = self.subdomains.find(self.hull_marker)
        hull_dofs = dolfinx.fem.locate_dofs_topological(Wspace, entity_dim=tdim, entities=hull_cells)
        self.epsr.x.array[material_dofs] = self.material_epsr
        self.mur.x.array[material_dofs] = self.material_mur
        self.epsr.x.array[transition_dofs] = self.epsr_transition.x.array[transition_dofs]
        self.mur.x.array[transition_dofs] = self.mur_transition.x.array[transition_dofs]
        self.epsr.x.array[hull_dofs] = self.hull_epsr
        self.mur.x.array[hull_dofs] = self.hull_mur
        if self.pml_hull_overlap_marker >= 0:
            hull_cells = self.subdomains.find(self.pml_hull_overlap_marker)
            CFRP_dofs = dolfinx.fem.locate_dofs_topological(Wspace, entity_dim=tdim, entities=hull_cells)
            self.epsr.x.array[hull_dofs] = self.hull_epsr
            self.mur.x.array[hull_dofs] = self.hull_mur
            
        
    def InitializePML(self):
        """Set up PML layer."""
        def pml_stretch(y, x, k0, x_dom=0, x_pml=1, n=3, R0=1e-10):
            return(y*(1 - 1j*(n + 1)*np.log(1/R0)/(2*k0*np.abs(x_pml - x_dom))*((x - x_dom)/(x_pml - x_dom))**n))

        def pml_epsr_murinv(pml_coords, rho):
            J = ufl.grad(pml_coords)

            # Transform the 2x2 Jacobian into a 3x3 matrix.
            J = ufl.as_matrix(((J[0, 0], J[0, 1], 0),
                               (J[1, 0], J[1, 1], 0),
                               (0, 0, pml_coords[0] / rho)))

            A = ufl.inv(J)
            epsr_pml = ufl.det(J) * A * self.epsr * ufl.transpose(A)
            mur_pml = ufl.det(J) * A * self.mur * ufl.transpose(A)
            murinv_pml = ufl.inv(mur_pml)
            return(epsr_pml, murinv_pml)

        rho, z = ufl.SpatialCoordinate(self.mesh)
        if not self.PML.cylindrical: # Spherical PML
            r = ufl.sqrt(rho**2 + z**2)
            rho_stretched = pml_stretch(rho, r, self.k0, x_dom=self.PML.radius-self.PML.d, x_pml=self.PML.radius)
            z_stretched = pml_stretch(z, r, self.k0, x_dom=self.PML.radius-self.PML.d, x_pml=self.PML.radius)
            rho_pml = ufl.conditional(ufl.ge(abs(r), self.PML.radius-self.PML.d), rho_stretched, rho)
            z_pml = ufl.conditional(ufl.ge(abs(r), self.PML.radius-self.PML.d), z_stretched, z)
        else:
            rho_stretched = pml_stretch(rho, rho, self.k0, x_dom=self.PML.rho-self.PML.d, x_pml=self.PML.rho)
            zt_stretched = pml_stretch(z, z, self.k0, x_dom=self.PML.zt-self.PML.d, x_pml=self.PML.zt)
            zb_stretched = pml_stretch(z, z, self.k0, x_dom=self.PML.zb+self.PML.d, x_pml=self.PML.zb)
            rho_pml = ufl.conditional(ufl.ge(rho, self.PML.rho-self.PML.d), rho_stretched, rho)
            z_pml = ufl.conditional(ufl.ge(z, self.PML.zt-self.PML.d), zt_stretched, ufl.conditional(ufl.le(z, self.PML.zb+self.PML.d), zb_stretched, z))
        pml_coords = ufl.as_vector((rho_pml, z_pml))
        self.epsr_pml, self.murinv_pml = pml_epsr_murinv(pml_coords, rho)
        
    def curl_axis(self, a, m: int, rho):
        curl_r = -a[2].dx(1) - 1j * m / rho * a[1]
        curl_z = a[2] / rho + a[2].dx(0) + 1j * m / rho * a[0]
        curl_p = a[0].dx(1) - a[1].dx(0)
        return ufl.as_vector((curl_r, curl_z, curl_p))

    def Excitation(self, E0theta_inc=0, E0phi_inc=0, theta_inc=0, phi_inc=0, E0theta_ant=0, E0phi_ant=0, theta_ant=0, phi_ant=0, antenna_taper=None):
        """Set excitation variables."""
        self.E0theta_inc = E0theta_inc
        self.E0phi_inc = E0phi_inc
        self.theta_inc = theta_inc
        self.phi_inc = phi_inc
        self.E0theta_ant = E0theta_ant
        self.E0phi_ant = E0phi_ant
        self.theta_ant = theta_ant
        self.phi_ant = phi_ant
        if antenna_taper == None:
            self.antenna_taper = lambda x: np.ones(x.shape[1], dtype=complex)
        else:
            self.antenna_taper = antenna_taper

    def planewave_rz(self, theta, phi, E0theta, E0phi, k, m, x):
        rho = x[0]
        z = x[1]
        krs = k*rho*np.sin(theta)
        exp_kzc = np.exp(-1j*k*z*np.cos(theta))
        jv_m = jv(m, krs)
        jv_mp1 = jv(m+1, krs)
        jv_mm1 = jv(m-1, krs)
        Er = E0theta*exp_kzc*1/2*((-1j)**(m+1)*jv_mp1 + (-1j)**(m-1)*jv_mm1)*np.exp(1j*m*phi)*np.cos(theta) \
            + E0phi*exp_kzc*1/2j*((-1j)**(m+1)*jv_mp1 - (-1j)**(m-1)*jv_mm1)*np.exp(1j*m*phi)
        Ez = E0theta*exp_kzc*(-1j)**m*jv_m*np.exp(1j*m*phi)*(-np.sin(theta))
        return((Er, Ez))

    def planewave_phi(self, theta, phi, E0theta, E0phi, k, m, x):
        rho = x[0]
        z = x[1]
        krs = k*rho*np.sin(theta)
        exp_kzc = np.exp(-1j*k*z*np.cos(theta))
        jv_m = jv(m, krs)
        jv_mp1 = jv(m+1, krs)
        jv_mm1 = jv(m-1, krs)
        Ephi = E0theta*exp_kzc*1/2j*((-1j)**(m+1)*jv_mp1 - (-1j)**(m-1)*jv_mm1)*np.exp(1j*m*phi)*(-np.cos(theta)) \
            + E0phi*exp_kzc*1/2*((-1j)**(m+1)*jv_mp1 + (-1j)**(m-1)*jv_mm1)*np.exp(1j*m*phi)
        return(Ephi)

    def antenna_rz(self, theta, phi, E0theta, E0phi, k, m, x):
        Er, Ez = self.planewave_rz(theta, phi, E0theta, E0phi, k, m, x)
        Er = Er*self.antenna_taper(x)
        Ez = Ez*self.antenna_taper(x)
        return((Er, Ez))

    def antenna_phi(self, theta, phi, E0theta, E0phi, k, m, x):
        Ephi = self.planewave_phi(theta, phi, E0theta, E0phi, k, m, x)
        Ephi = Ephi*self.antenna_taper(x)
        return(Ephi)
        
    class FarField():
        """Class to hold the computation of far field of the mth mode."""
        def __init__(self, parent):
            mesh = parent.mesh
            degree = parent.degree
            Vspace = parent.Vspace
            ScalarSpace = dolfinx.fem.functionspace(mesh, ('CG', degree))
            k0 = parent.k0
            self.k = parent.k0*parent.n_bkg
            mur_bkg = parent.mur_bkg
            etar_bkg = parent.etar_bkg
            dS_farfield = parent.dS_farfield
            self.farfield_cells = parent.farfield_cells
                
            self.Em = dolfinx.fem.Function(Vspace)
            self.m = dolfinx.fem.Constant(mesh, 0j)
            self.jv_m = dolfinx.fem.Function(ScalarSpace)
            self.jv_mp1 = dolfinx.fem.Function(ScalarSpace)
            self.jv_mm1 = dolfinx.fem.Function(ScalarSpace)
            self.exp_kzc = dolfinx.fem.Function(ScalarSpace)
            self.Am = dolfinx.fem.Function(ScalarSpace)
            self.Sm = dolfinx.fem.Function(ScalarSpace)
            self.Cm = dolfinx.fem.Function(ScalarSpace)
            self.prefactor = dolfinx.fem.Constant(mesh, 0j)
            self.sinphi = dolfinx.fem.Constant(mesh, 0j)
            self.cosphi = dolfinx.fem.Constant(mesh, 0j)
            self.costheta_sinphi = dolfinx.fem.Constant(mesh, 0j)
            self.costheta_cosphi = dolfinx.fem.Constant(mesh, 0j)
            self.sintheta = dolfinx.fem.Constant(mesh, 0j)
            rho, z = ufl.SpatialCoordinate(mesh)
            n = ufl.FacetNormal(mesh)
            nr = n[0]('+')
            nz = n[1]('+')
            signfactor = ufl.sign(nr*rho + nz*z) # Enforce outward pointing normal
            Er = self.Em.sub(0)[0]('+')
            Ez = self.Em.sub(0)[1]('+')
            Ephi = self.Em.sub(1)('+')
            # The H-field here is really eta0*H in SI units (i.e.,
            # same as E). Hence, etar_bkg is used in the weak
            # formulation, not eta0*etar_bkg.
            Hr, Hz, Hphi = -1/(1j*k0*mur_bkg)*parent.curl_axis(self.Em, self.m, rho)
            Hr = Hr('+')
            Hz = Hz('+')
            Hphi = Hphi('+')

            self.F_theta = signfactor*self.prefactor * \
                (-(self.Cm*nz*Ephi - self.Sm*(nr*Ez - nz*Er))*self.sinphi \
                 + (self.Sm*nz*Ephi + self.Cm*(nr*Ez - nz*Er))*self.cosphi \
                 - etar_bkg*(self.Cm*nz*Hphi - self.Sm*(nr*Hz - nz*Hr))*self.costheta_cosphi \
                 - etar_bkg*(self.Sm*nz*Hphi + self.Cm*(nr*Hz - nz*Hr))*self.costheta_sinphi \
                 + etar_bkg*self.Am*nr*Hphi*self.sintheta)*self.exp_kzc*rho*dS_farfield

            self.F_phi = signfactor*self.prefactor * \
                (-(self.Cm*nz*Ephi - self.Sm*(nr*Ez - nz*Er))*self.costheta_cosphi \
                 - (self.Sm*nz*Ephi + self.Cm*(nr*Ez - nz*Er))*self.costheta_sinphi \
                 + self.Am*nr*Ephi*self.sintheta \
                 + etar_bkg*(self.Cm*nz*Hphi - self.Sm*(nr*Hz - nz*Hr))*self.sinphi \
                 - etar_bkg*(self.Sm*nz*Hphi + self.Cm*(nr*Hz - nz*Hr))*self.cosphi)*self.exp_kzc*rho*dS_farfield
        
        def eval(self, theta, phi, m, Em):
            """Evaluate the far field in direction (theta, phi)."""
            k = self.k
            self.Em.x.array[:] = Em.x.array[:]
            self.m.value = m
            self.jv_m.interpolate(lambda x: jv(m, k*x[0]*np.sin(theta)), self.farfield_cells)
            self.jv_mp1.interpolate(lambda x: jv(m+1, k*x[0]*np.sin(theta)), self.farfield_cells)
            self.jv_mm1.interpolate(lambda x: jv(m-1, k*x[0]*np.sin(theta)), self.farfield_cells)
            self.exp_kzc.interpolate(lambda x: np.exp(1j*k*x[1]*np.cos(theta)), self.farfield_cells)
            self.Am.x.array[:] = self.jv_m.x.array[:]
            self.Sm.x.array[:] = 1/2*(np.exp(1j*phi)*self.jv_mm1.x.array[:] + np.exp(-1j*phi)*self.jv_mp1.x.array[:])
            self.Cm.x.array[:] = 1j/2*(np.exp(1j*phi)*self.jv_mm1.x.array[:] - np.exp(-1j*phi)*self.jv_mp1.x.array[:])
            self.prefactor.value = 1j*k/2*(1j**m)*np.exp(-1j*m*phi)
            self.sinphi.value = np.sin(phi)
            self.cosphi.value = np.cos(phi)
            self.costheta_sinphi.value = np.cos(theta)*np.sin(phi)
            self.costheta_cosphi.value = np.cos(theta)*np.cos(phi)
            self.sintheta.value = np.sin(theta)
            F_theta = dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(self.F_theta))
            F_phi = dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(self.F_phi))
            return(F_theta, F_phi)

    class SphericalVectorWaves():
        """Class to hold the computation of spherical vector waves expansion coefficients representation of the scattered field for mode m."""
        def __init__(self, parent, N):
            mesh = parent.mesh
            degree = parent.degree
            Vspace = parent.Vspace
            k0 = parent.k0
            self.k = parent.k0*parent.n_bkg
            mur_bkg = parent.mur_bkg
            self.etar_bkg = parent.etar_bkg
            dS_farfield = parent.dS_farfield
            self.farfield_cells = parent.farfield_cells

            self.J = 2*N*(N + 2)
            self.Q = np.zeros(self.J, dtype=complex)
                
            self.Em = dolfinx.fem.Function(Vspace)
            self.m = dolfinx.fem.Constant(mesh, 0j)
            self.Esmn1 = dolfinx.fem.Function(Vspace)
            self.Hsmn1 = dolfinx.fem.Function(Vspace)
            rho, z = ufl.SpatialCoordinate(mesh)
            n = ufl.FacetNormal(mesh)
            nr = n[0]
            nz = n[1]
            signfactor = ufl.sign(nr*rho + nz*z) # Enforce outward pointing normal
            nuv = ufl.as_vector([nr, nz, 0]) # Create 3D normal vector
            
            # The H-field here is really eta0*H in SI units (i.e.,
            # same units as E). Hence, etar_bkg is used in the weak
            # formulation, not eta0*etar_bkg. Note the expression for
            # H, curl_axis, and the reciprocity integral assumes time convention
            # 'i'.
            def curl_axis(a, m: int, rho):
                curl_r = -a[2].dx(1) + 1j * m / rho * a[1]
                curl_z = a[2] / rho + a[2].dx(0) - 1j * m / rho * a[0]
                curl_p = a[0].dx(1) - a[1].dx(0)
                return ufl.as_vector((curl_r, curl_z, curl_p))
            Hr, Hz, Hphi = 1/(1j*k0*mur_bkg)*curl_axis(self.Em, self.m, rho)
            self.Hm = ufl.as_vector([Hr, Hz, Hphi])

            self.reciprocity_integral = 2*np.pi*signfactor('+')/np.sqrt(eta0)*(-1j*self.k/np.sqrt(self.etar_bkg)*ufl.dot(ufl.cross(self.Em('+'), self.Hsmn1('+')), nuv('+')) - self.k*np.sqrt(self.etar_bkg)*ufl.dot(ufl.cross(self.Esmn1('+'), self.Hm('+')), nuv('+')))*rho*dS_farfield
        
        def eval_coefficients(self, m, Em):
            """Evaluate the spherical vector wave expansion coefficients for mode m."""
            k = self.k
            if TimeConvention == 'j':
                self.Em.x.array[:] = np.conjugate(Em.x.array[:])
            else:
                self.Em.x.array[:] = Em.x.array
            self.m.value = m
            def Fsmnc_rz(s, m, n, c, x):
                r = np.sqrt(x[0]**2 + x[1]**2)
                kr = k*r
                theta = np.arccos(x[1]/r)
                phi = 0
                Fr, Ftheta, Fphi = svw.Fsmnc(s, m, n, c, kr, theta, phi)
                Frho = Fr*np.sin(theta) + Ftheta*np.cos(theta)
                Fz = Fr*np.cos(theta) - Ftheta*np.sin(theta)
                return Frho, Fz
            def Fsmnc_phi(s, m, n, c, x):
                r = np.sqrt(x[0]**2 + x[1]**2)
                kr = k*r
                theta = np.arccos(x[1]/r)
                phi = 0
                Fr, Ftheta, Fphi = svw.Fsmnc(s, m, n, c, kr, theta, phi)
                return Fphi
            for j in range(self.J):
                s, mm, n = svw.smn_from_j(j)
                if mm == m:
                    Esmn1_rz = functools.partial(Fsmnc_rz, s, -m, n, 1)
                    Esmn1_phi = functools.partial(Fsmnc_phi, s, -m, n, 1)
                    Hsmn1_rz = functools.partial(Fsmnc_rz, 3 - s, -m, n, 1)
                    Hsmn1_phi = functools.partial(Fsmnc_phi, 3 - s, -m, n, 1)
                    self.Esmn1.sub(0).interpolate(Esmn1_rz, self.farfield_cells)
                    self.Esmn1.sub(1).interpolate(Esmn1_phi, self.farfield_cells)
                    self.Hsmn1.sub(0).interpolate(Hsmn1_rz, self.farfield_cells)
                    self.Hsmn1.sub(1).interpolate(Hsmn1_phi, self.farfield_cells)
                    self.Q[j] = dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(self.reciprocity_integral))/(-1)**(np.abs(m + 1))
                    if np.isnan(self.Q[j]):
                        print(f'Rank {comm.rank}: NaN detected in Q computation, s, m, n = {s}, {m}, {n}')
                        sys.stdout.flush()
                        self.Q[j] = 0
#                if TimeConvention == 'j':
#                    self.Q = np.conjugate(self.Q)
                    
        def FarFieldCut(self, theta=np.linspace(-np.pi, np.pi, 361), phi=0):
            """Compute a farfield cut."""
            phivec = np.ones(len(theta))*phi
            phivec[theta<0] = np.mod(phivec[theta<0] + np.pi, 2*np.pi)
            thetavec = np.abs(theta)
            F_theta = np.zeros(len(theta), dtype=complex)
            F_phi = np.zeros(len(theta), dtype=complex)
            for j in range(self.J):
                s, m, n = svw.smn_from_j(j)
                K_theta, K_phi = svw.Ksmn(s, m, n, thetavec, phivec)
#                if TimeConvention == 'j':
#                    K_theta = np.conjugate(K_theta)
#                    K_phi = np.conjugate(K_phi)
                idx_theta = np.isnan(self.Q[j]*K_theta)^True
                idx_phi = np.isnan(self.Q[j]*K_phi)^True
                if sum(idx_theta^True) > 0 or sum(idx_phi^True) > 0:
                    print(f'Rank {comm.rank}: NaN detected in farfield, s, m, n = {s}, {m}, {n}')
                    sys.stdout.flush()
                F_theta[idx_theta] += np.sqrt(self.etar_bkg*eta0)*self.Q[j]*K_theta[idx_theta]/np.sqrt(4*np.pi)
                F_phi[idx_phi] += np.sqrt(self.etar_bkg*eta0)*self.Q[j]*K_phi[idx_phi]/np.sqrt(4*np.pi)
            return thetavec, F_theta, F_phi

    def PlotField(self, E, Erefl=None, animate=False, filename='tmp.mp4', clim='sym'):
        """Plot the field in the domain."""
        PlotSpace = dolfinx.fem.functionspace(self.mesh, ('CG', 1))

        def CreateGrid(E, reflect=False):
            E_plot = dolfinx.fem.Function(PlotSpace)
            E_plot.interpolate(dolfinx.fem.Expression(E, PlotSpace.element.interpolation_points()))
            E_plot_array = E_plot.x.array
            cells, cell_types, x = dolfinx.plot.vtk_mesh(self.mesh, tdim)
            E_grid = pv.UnstructuredGrid(cells, cell_types, x)
            if reflect == True:
                E_grid = E_grid.reflect((1, 0, 0), point=(0, 0, 0))
            E_grid["plotfunc"] = np.real(E_plot_array)
            return(E_grid, E_plot_array)
        
        def sym_clim(E, Erefl=0):
            E_max = np.max(np.abs(E))
            Erefl_max = np.max(np.abs(Erefl))
            E_max = np.max([E_max, Erefl_max])
            # Find max over all ranks
            E_maxs = self.comm.gather(E_max, root=self.model_rank)
            if self.comm.rank == self.model_rank:
                E_max = np.max(E_maxs)
            else:
                E_max = None
            E_max = self.comm.bcast(E_max, root=self.model_rank)
            clim = [-E_max, E_max]
            return(clim)

        Er_grid, Er_array = CreateGrid(E[0])
        Ez_grid, Ez_array = CreateGrid(E[1])
        Ephi_grid, Ephi_array = CreateGrid(E[2])
        Er_grids = self.comm.gather(Er_grid, root=self.model_rank)
        Ez_grids = self.comm.gather(Ez_grid, root=self.model_rank)
        Ephi_grids = self.comm.gather(Ephi_grid, root=self.model_rank)

        if not Erefl == None: # Note minus sign in r and phi components
            Erefl_r_grid, Erefl_r_array = CreateGrid(-Erefl[0], reflect=True)
            Erefl_z_grid, Erefl_z_array = CreateGrid(Erefl[1], reflect=True)
            Erefl_phi_grid, Erefl_phi_array = CreateGrid(-Erefl[2], reflect=True)
            Erefl_r_grids = self.comm.gather(Erefl_r_grid, root=self.model_rank)
            Erefl_z_grids = self.comm.gather(Erefl_z_grid, root=self.model_rank)
            Erefl_phi_grids = self.comm.gather(Erefl_phi_grid, root=self.model_rank)

        if clim == 'sym':
            if not Erefl == None:
                clim_Er = sym_clim(Er_array, Erefl_r_array)
                clim_Ez = sym_clim(Ez_array, Erefl_z_array)
                clim_Ephi = sym_clim(Ephi_array, Erefl_phi_array)
            else:
                clim_Er = sym_clim(Er_array)
                clim_Ez = sym_clim(Ez_array)
                clim_Ephi = sym_clim(Ephi_array)
        else:
            clim_Er = clim
            clim_Ez = clim
            clim_Ephi = clim

        if self.material_marker >= 0: # Indicate material region
            V = dolfinx.fem.functionspace(self.mesh, ('DG', 0))
            u = dolfinx.fem.Function(V)
            material_cells = np.concatenate((self.subdomains.find(self.material_marker), self.subdomains.find(self.transition_marker), self.subdomains.find(self.hull_marker)))
            material_dofs = dolfinx.fem.locate_dofs_topological(V, entity_dim=tdim, entities=material_cells)
            u.x.array[:] = 0
            u.x.array[material_dofs] = 0.25 # Used as opacity value later
            cells, cell_types, x = dolfinx.plot.vtk_mesh(self.mesh, tdim)
            mat_grid = pv.UnstructuredGrid(cells, cell_types, x)
            mat_grid["u"] = np.real(u.x.array)
            mat_grids = self.comm.gather(mat_grid, root=self.model_rank)
            if not Erefl == None:
                mat_refl_grid = mat_grid.reflect((1, 0, 0), point=(0, 0, 0))
                mat_refl_grids = self.comm.gather(mat_refl_grid, root=self.model_rank)
            
        # Gather the data arrays for animation
        Er_arrays = self.comm.gather(Er_array, root=self.model_rank)
        Ez_arrays = self.comm.gather(Ez_array, root=self.model_rank)
        Ephi_arrays = self.comm.gather(Ephi_array, root=self.model_rank)
        if not Erefl == None:
            Erefl_r_arrays = self.comm.gather(Erefl_r_array, root=self.model_rank)
            Erefl_z_arrays = self.comm.gather(Erefl_z_array, root=self.model_rank)
            Erefl_phi_arrays = self.comm.gather(Erefl_phi_array, root=self.model_rank)

        if self.comm.rank == self.model_rank:
            if animate == True:
#                plotter = pvqt.BackgroundPlotter(shape=(1,3), auto_update=True)
                plotter = pvqt.BackgroundPlotter(shape=(1,3), window_size=(3840,2160), auto_update=True)
            else:
                plotter = pv.Plotter(shape=(1,3))

            def AddGrids(E_grids, Erefl_grids=None, title='', clim=None):
                for g in E_grids:
                    plotter.add_mesh(g, show_edges=False, show_scalar_bar=True, scalar_bar_args={'title': title, 'title_font_size': 12, 'label_font_size': 12}, clim=clim, cmap='bwr')
                if self.material_marker >= 0:
                    for g in mat_grids:
                        plotter.add_mesh(g, show_edges=False, scalars='u', opacity='u', cmap='binary', show_scalar_bar=False)
                if not Erefl_grids == None:
                    for g in Erefl_grids:
                        plotter.add_mesh(g, show_edges=False, show_scalar_bar=True, scalar_bar_args={'title': title, 'title_font_size': 12, 'label_font_size': 12}, clim=clim, cmap='bwr')
                    if self.material_marker >= 0:
                        for g in mat_refl_grids:
                            plotter.add_mesh(g, show_edges=False, scalars='u', opacity='u', cmap='binary', show_scalar_bar=False)
                plotter.view_xy()
                plotter.add_text(title, font_size=18)
                plotter.add_axes()

            plotter.subplot(0, 0)
            if not Erefl == None:
                AddGrids(Er_grids, Erefl_r_grids, title='E_rho', clim=clim_Er)
            else:
                AddGrids(Er_grids, title='E_rho', clim=clim_Er)

            plotter.subplot(0, 1)
            if not Erefl == None:
                AddGrids(Ez_grids, Erefl_z_grids, title='E_z', clim=clim_Ez)
            else:
                AddGrids(Ez_grids, title='E_z', clim=clim_Ez)
            
            plotter.subplot(0, 2)
            if not Erefl == None:
                AddGrids(Ephi_grids, Erefl_phi_grids, title='E_phi', clim=clim_Ephi)
            else:
                AddGrids(Ephi_grids, title='E_phi', clim=clim_Ephi)

            if animate:
                Nphase = 120
                phasevec = np.linspace(0, 2*np.pi, Nphase)
                plotter.open_movie(filename)
                for phase in phasevec:
                    for n in range(self.comm.size):
                        Er_grids[n]["plotfunc"] = np.real(Er_arrays[n]*np.exp(1j*phase))
                        Ez_grids[n]["plotfunc"] = np.real(Ez_arrays[n]*np.exp(1j*phase))
                        Ephi_grids[n]["plotfunc"] = np.real(Ephi_arrays[n]*np.exp(1j*phase))
                        if not Erefl == None:
                            Erefl_r_grids[n]["plotfunc"] = np.real(Erefl_r_arrays[n]*np.exp(1j*phase))
                            Erefl_z_grids[n]["plotfunc"] = np.real(Erefl_z_arrays[n]*np.exp(1j*phase))
                            Erefl_phi_grids[n]["plotfunc"] = np.real(Erefl_phi_arrays[n]*np.exp(1j*phase))
                    plotter.app.processEvents()
                    plotter.write_frame()
                plotter.close()
            else:
                plotter.show()

    def wiscombe(self, x):
        """The criterion from Wiscombe (1980) on the number of terms in the Mie series, x=ka."""
        if x < 0.02:
            N = 1
        elif x <= 8:
            N = x + 4*x**(1/3) + 1
        elif x < 4200:
            N = x + 4.05*x**(1/3) + 2
        elif x <= 20000:
            N = x + 4*x**(1/3) + 2
        else:
            N = -1
        return(int(N) + 1)
            
    def ComputeSolutionsAndPostprocess(self, mvec=None, phi_out=0, phi_ff=0, Nangles=361, ComputeSVW=False):
        N = self.wiscombe(self.k0*self.max_r)
        print(f'Rank {self.comm.rank}: N = {N}')
        if mvec == None:
            mvec = np.arange(-N, N+1, dtype=int)
        # Set up output variables
        fzero0 = lambda x: np.zeros((2, x.shape[1]), dtype=complex)
        fzero1 = lambda x: np.zeros(x.shape[1], dtype=complex)
        Esca = dolfinx.fem.Function(self.Vspace)
        Etot = dolfinx.fem.Function(self.Vspace)
        Esca_refl = dolfinx.fem.Function(self.Vspace)
        Etot_refl = dolfinx.fem.Function(self.Vspace)
        Esca.sub(0).interpolate(fzero0)
        Esca.sub(1).interpolate(fzero1)
        Etot.sub(0).interpolate(fzero0)
        Etot.sub(1).interpolate(fzero1)
        Esca_refl.sub(0).interpolate(fzero0)
        Esca_refl.sub(1).interpolate(fzero1)
        Etot_refl.sub(0).interpolate(fzero0)
        Etot_refl.sub(1).interpolate(fzero1)

        scattering_angles = np.linspace(-np.pi, np.pi, Nangles)
        angles = np.vstack((scattering_angles, phi_ff*np.ones(Nangles))).transpose()
        idx = scattering_angles < 0
        angles[idx,0] = np.abs(angles[idx,0])
        angles[idx,1] = phi_ff + np.pi
        Fm = self.FarField(self)

        if ComputeSVW:
            SVW = self.SphericalVectorWaves(self, N)
        
        # Set up computation 
        rho, z = ufl.SpatialCoordinate(self.mesh)
        Ea_m = dolfinx.fem.Function(self.Vspace)
        Eb_m = dolfinx.fem.Function(self.Vspace)
        Ea_m.sub(0).interpolate(fzero0)
        Ea_m.sub(1).interpolate(fzero1)
        Eb_m.sub(0).interpolate(fzero0)
        Eb_m.sub(1).interpolate(fzero1)
        Es_m = ufl.TrialFunction(self.Vspace)
        v_m = ufl.TestFunction(self.Vspace)
        m_idx = dolfinx.fem.Constant(self.mesh, 1j)
        curl_Es_m = self.curl_axis(Es_m, m_idx, rho)
        curl_v_m = self.curl_axis(v_m, m_idx, rho)
        F = - ufl.inner(1/self.mur*curl_Es_m, curl_v_m)*rho*self.dx_dom \
            + ufl.inner(self.epsr*(self.k0**2)*Es_m, v_m)*rho*self.dx_dom \
            + ufl.inner((self.epsr - 1/self.mur*self.mur_bkg*self.epsr_bkg)*(self.k0**2)*Eb_m, v_m)*rho*self.dx_dom \
            - ufl.inner(self.murinv_pml*curl_Es_m, curl_v_m)*rho*self.dx_pml \
            + ufl.inner(self.epsr_pml*(self.k0**2)*Es_m, v_m)*rho*self.dx_pml

        # Weak imposition of nx(Es+Eb)=0 at PEC boundaries, and
        # nx(Es+Eb-Ea)=0 at antenna
        h = 2*ufl.Circumradius(self.mesh)
        alpha_bc = 10
        n0 = ufl.FacetNormal(self.mesh)
        n = ufl.as_vector((n0[0], n0[1], 0))
        if self.pec_surface_marker >= 0:
            F_pec = alpha_bc/h*ufl.inner(ufl.cross(Es_m + Eb_m, n), ufl.cross(v_m, n))*self.ds_pec
            F = F + F_pec
        if self.antenna_surface_marker >= 0:
            F_ant = alpha_bc/h*ufl.inner(ufl.cross(Es_m + Eb_m - Ea_m, n), ufl.cross(v_m, n))*self.ds_antenna
            F = F + F_ant

        a, L = ufl.lhs(F), ufl.rhs(F)

        # Set solver options
        if True: # Direct solver
            petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
        elif False: # Iterative solver, not well configured yet
            petsc_options = {"ksp_type": "gmres", "ksp_rtol": 1e-6, "ksp_atol": 1e-10, "ksp_max_it": 1000, "pc_type": "none"}
        else: # Experiment with solver options
            petsc_options = {"ksp_type": PETSc.KSP.Type.LGMRES, "pc_type": PETSc.PC.Type.SOR, "ksp_rtol": 1e-6, "ksp_atol": 1e-10, "ksp_max_it": 10000, "ksp_inner_prec_side": PETSc.PC.Side.RIGHT}
        
        # Define problem
        problem = dolfinx.fem.petsc.LinearProblem(
            a, L, bcs=[], petsc_options=petsc_options
        )
        
        farfield_amplitudes = np.zeros((2, Nangles), dtype=complex)
        logfilename = f'mylog_rank{self.comm.rank}.txt'
        with open(logfilename, 'w') as f:
            f.write(f'{time.asctime()}: Start of simulation at rank {self.comm.rank}\n')
        for m in mvec:
            print(f'Rank {self.comm.rank}: m = {m}')
            sys.stdout.flush()

            # Compute antenna field
            f_rz = functools.partial(self.antenna_rz, self.theta_ant, self.phi_ant, self.E0theta_ant, self.E0phi_ant, self.k0*self.n_bkg, m)
            f_phi = functools.partial(self.antenna_phi, self.theta_ant, self.phi_ant, self.E0theta_ant, self.E0phi_ant, self.k0*self.n_bkg, m)
            Ea_m.sub(0).interpolate(f_rz)
            Ea_m.sub(1).interpolate(f_phi)

            # Compute incident field
            f_rz = functools.partial(self.planewave_rz, self.theta_inc, self.phi_inc, self.E0theta_inc, self.E0phi_inc, self.k0*self.n_bkg, m)
            f_phi = functools.partial(self.planewave_phi, self.theta_inc, self.phi_inc, self.E0theta_inc, self.E0phi_inc, self.k0*self.n_bkg, m)
            Eb_m.sub(0).interpolate(f_rz)
            Eb_m.sub(1).interpolate(f_phi)
            
            problem.bcs = []
            m_idx.value = m
            with dolfinx.common.Timer() as t:
                Es_m_h = problem.solve()
                print(f'Rank {self.comm.rank} solve time: {t.elapsed()[0]}')
                with open(logfilename, 'a') as f:
                    f.write(f'{time.asctime()}: At rank {self.comm.rank}: m = {m}, solve time: {t.elapsed()[0]}\n')
                sys.stdout.flush()

            Esca.x.array[:] += Es_m_h.x.array[:]*np.exp(-1j*m*phi_out)
            Etot.x.array[:] += (Es_m_h.x.array[:] + Eb_m.x.array[:])*np.exp(-1j*m*phi_out)
            Esca_refl.x.array[:] += Es_m_h.x.array[:]*np.exp(-1j*m*(phi_out + np.pi))
            Etot_refl.x.array[:] += (Es_m_h.x.array[:] + Eb_m.x.array[:])*np.exp(-1j*m*(phi_out + np.pi))
            with dolfinx.common.Timer() as t:
                for n in range(0, Nangles):
                    theta_ff, phi_ff = angles[n]
                    ff_theta, ff_phi = Fm.eval(theta_ff, phi_ff, m, Es_m_h)
                    farfield_amplitudes[0, n] += ff_theta
                    farfield_amplitudes[1, n] += ff_phi
                print(f'Rank {self.comm.rank} farfield time: {t.elapsed()[0]}')
                with open(logfilename, 'a') as f:
                    f.write(f'{time.asctime()}: At rank {self.comm.rank}: m = {m}, farfield time: {t.elapsed()[0]}\n')
                sys.stdout.flush()

            if ComputeSVW:
                with dolfinx.common.Timer() as t:
                    SVW.eval_coefficients(m, Es_m_h)
                    print(f'Rank {self.comm.rank} SVW time: {t.elapsed()[0]}')
                    sys.stdout.flush()
            else:
                SVW = None
                
        return(mvec, Esca, Esca_refl, Etot, Etot_refl, scattering_angles, farfield_amplitudes, SVW)

if __name__ == '__main__':
    # The code below illustrates how to set up a simulation
    
    comm = MPI.COMM_WORLD
    model_rank = 0

    f0 = 10e9
    lambda0 = c0/f0
    E0theta_inc = 1
    E0phi_inc = 0
    theta_inc = np.pi
    phi_inc = 0
    E0theta_ant = 0
    E0phi_ant = 0
    theta_ant = 0
    phi_ant = 0
    Nangles = 361
    h = 0.1*lambda0
    PMLcylindrical = True
    ComputeSVW = False
    
    sphere_case = False
    if sphere_case: 
        radius_sphere = 0.5*lambda0
        radius_farfield = radius_sphere + 0.5*lambda0
        radius_domain = radius_farfield + 0.5*lambda0
        radius_pml = radius_domain + 0.5*lambda0
        material_epsr = 2
        pec = False
        antenna_taper = None
        
        meshdata = mesh_rotsymradome.CreateMeshSphere(comm=comm, model_rank=model_rank, radius_sphere=radius_sphere, radius_farfield=radius_farfield, radius_domain=radius_domain, radius_pml=radius_pml, h=h, pec=pec, visualize=False, PMLcylindrical=PMLcylindrical)
        p = RotSymProblem(meshdata, material_epsr=material_epsr, f0=f0)
    else:
        material_epsr = 3*(1 - 0.01j)
        hull_epsr = 100 - 72j
        hfine = h/10
        PMLpenetrate = False
        AntennaMetalBase = False
        Antenna = True
        w = 10*lambda0
        t = lambda0/10
        Htransition = lambda0
        transition_volumefraction_hull = lambda x: -x[1]/Htransition
        transition_volumefraction_hull = lambda x: 0*np.ones(x.shape[1])
        antenna_taper = lambda x: np.cos(x[0]/(w/2)*np.pi/2)
        
        meshdata = mesh_rotsymradome.CreateMeshOgive(comm=comm, model_rank=model_rank, d=lambda0/(2*np.real(np.sqrt(material_epsr))), h=h, hfine=hfine, w=w, t=t, PMLcylindrical=PMLcylindrical, PMLpenetrate=PMLpenetrate, Antenna=Antenna, AntennaMetalBase=AntennaMetalBase, visualize=False)

        p = RotSymProblem(meshdata, material_epsr=material_epsr, hull_epsr=hull_epsr, f0=f0, transition_volumefraction_hull=transition_volumefraction_hull)

    if True: # Test for ghost farfield facets and plot mesh partition
        ghost_ff_facets = mesh_rotsymradome.CheckGhostFarfieldFacets(comm, model_rank, meshdata.mesh, meshdata.boundaries, meshdata.boundary_markers['farfield'])
        if len(ghost_ff_facets) > 0:
            print('Ghost farfield facets detected, exiting.')
            mesh_rotsymradome.PlotMeshPartition(comm, model_rank, meshdata.mesh, ghost_ff_facets, meshdata.boundaries, meshdata.boundary_markers['farfield'])
            exit()

    p.Excitation(E0theta_inc=E0theta_inc, E0phi_inc=E0phi_inc,
                 theta_inc=theta_inc, phi_inc=phi_inc,
                 E0theta_ant=E0theta_ant, E0phi_ant=E0phi_ant,
                 theta_ant=theta_ant, phi_ant=phi_ant,
                 antenna_taper=antenna_taper)
    mvec, Esca, Esca_refl, Etot, Etot_refl, scattering_angles, farfield_amplitudes, SphericalVectorWaves = p.ComputeSolutionsAndPostprocess(mvec=[-1, 1], Nangles=Nangles, ComputeSVW=ComputeSVW)

    if ComputeSVW:
        with dolfinx.common.Timer() as t:
            Qs = comm.gather(SphericalVectorWaves.Q, root=model_rank)
            if comm.rank == model_rank:
                Q = sum(Qs)
            else:
                Q = None
            Q = comm.bcast(Q, root=model_rank)
            SphericalVectorWaves.Q = Q
            if comm.rank == model_rank:
                _, F_theta, F_phi = SphericalVectorWaves.FarFieldCut(theta=scattering_angles, phi=0)
            print(f'Rank {comm.rank} SVW farfield time: {t.elapsed()[0]}')
            sys.stdout.flush()
    
    # Collect far field results from all ranks
    farfields = comm.gather(farfield_amplitudes, root=model_rank)
    if comm.rank == model_rank:
        ff = sum(farfields)
    else:
        ff = None
    ff = comm.bcast(ff, root=model_rank)
    
    if comm.rank == model_rank: # Save some data
        with open('farfield.dat', 'w') as f:
            print('# Scattering angle (deg), Re(ff[theta]), Im(ff[theta]), Re(ff[phi]), Im(ff[phi])', file=f)
            for n in range(len(scattering_angles)):
                print(f'{scattering_angles[n]*180/np.pi}, {np.real(ff[0,n])}, {np.imag(ff[0,n])}, {np.real(ff[1,n])}, {np.imag(ff[1,n])}', file=f)
        if comm.size == 1: # Saving to xdmf only works in single process now
            # First interpolate the results to a standard Lagrange vector space
            Velement = basix.ufl.element('CG', meshdata.mesh.basix_cell(), 1, shape=(3,))
            Vspace = dolfinx.fem.functionspace(meshdata.mesh, Velement)
            u = dolfinx.fem.Function(Vspace)
            with dolfinx.io.XDMFFile(MPI.COMM_SELF, f'fields.xdmf', 'w') as f:
                f.write_mesh(meshdata.mesh)
                for n, E in enumerate([Esca, Esca_refl, Etot, Etot_refl]):
                    u.interpolate(dolfinx.fem.Expression(ufl.as_vector((E[0], E[1], E[2])), Vspace.element.interpolation_points()))
                    f.write_function(u, n)

    if True: # Visualization of near field
        p.PlotField(Esca, Esca_refl, animate=True, filename='E_sca.mp4', clim=[-2, 2])
        p.PlotField(Etot, Etot_refl, animate=True, filename='E_tot.mp4', clim=[-2, 2])

    if True: # Plot far fields
        # Plot the results
        if comm.rank == model_rank:
            if sphere_case:
                # Compute reference results
                import miepython
                x = 2*np.pi*radius_sphere/lambda0
                if pec:
                    m = -1
                else:
                    m = np.sqrt(material_epsr)
                sigma_par = miepython.i_par(m, x, np.cos(scattering_angles - theta_inc), norm='qsca')*np.pi*radius_sphere**2
                sigma_per = miepython.i_per(m, x, np.cos(scattering_angles - theta_inc), norm='qsca')*np.pi*radius_sphere**2

                plt.figure()
                plt.plot(scattering_angles*180/np.pi, np.abs(ff[0,:])**2, label='|F_theta|^2')
                plt.plot(scattering_angles*180/np.pi, np.abs(ff[1,:])**2, label='|F_phi|^2')
                plt.plot(scattering_angles*180/np.pi, sigma_par, '--', label='dSCS_par')
                plt.plot(scattering_angles*180/np.pi, sigma_per, '--', label='dSCS_per')
                if ComputeSVW:
                    plt.plot(scattering_angles*180/np.pi, np.abs(F_theta)**2, '-', label='|F_theta|^2, SVW')
                    plt.plot(scattering_angles*180/np.pi, np.abs(F_phi)**2, '-', label='|F_phi|^2, SVW')
                plt.xlabel('Theta (degrees)')
                plt.ylabel('Differential SCS (m^2)')
                plt.legend(loc='best')
                plt.show()
            else:
                plt.figure()
                plt.plot(scattering_angles*180/np.pi, 10*np.log10(np.abs(ff[0,:])**2), '-', label='|F_theta|^2')
                plt.plot(scattering_angles*180/np.pi, 10*np.log10(np.abs(ff[1,:])**2), '-', label='|F_phi|^2')
                if ComputeSVW:
                    plt.plot(scattering_angles*180/np.pi, 10*np.log10(np.abs(F_theta)**2), '--', label='|F_theta|^2 SVW')
                    plt.plot(scattering_angles*180/np.pi, 10*np.log10(np.abs(F_phi)**2), '--', label='|F_phi|^2 SVW')
                plt.xlabel('Theta (degrees)')
                plt.ylabel('Differential SCS (dBsm)')
                plt.legend(loc='best')
                plt.show()

