# Script to do a radome simulation.
#
# Daniel Sjöberg, 2023-07-20

import numpy as np
import dolfinx, ufl
from mpi4py import MPI
import rotsymsca
import mesh_rotsymradome
from scipy.constants import c as c0
from matplotlib import pyplot as plt

def compute_radome(pol='theta', antenna_mode=True, theta=np.pi/4, full_computation=True, comm=MPI.COMM_WORLD, model_rank=0):
    f0 = 10e9
    lambda0 = c0/f0
    theta_inc = np.pi - theta
    phi_inc = np.pi
    theta_ant = theta
    phi_ant = 0
    if pol == 'theta':
        if antenna_mode:
            E0theta_inc = 0
            E0phi_inc = 0
            E0theta_ant = 1
            E0phi_ant = 0
        else:
            E0theta_inc = 1
            E0phi_inc = 0
            E0theta_ant = 0
            E0phi_ant = 0
    else:
        if antenna_mode:
            E0theta_inc = 0
            E0phi_inc = 0
            E0theta_ant = 0
            E0phi_ant = 1
        else:
            E0theta_inc = 0
            E0phi_inc = 1
            E0theta_ant = 0
            E0phi_ant = 0
    material_epsr = 3*(1 - 0.01j)
    w = 10*lambda0
    antenna_taper = lambda x: np.cos(x[0]/(w/2)*np.pi/2)
    PMLcylindrical = True
    PMLpenetrate = True
    MetalBase = True
    t = lambda0
    Nangles = 361
    
    meshdata = mesh_rotsymradome.CreateMeshOgive(d=lambda0/(2*np.real(np.sqrt(material_epsr))), h=0.1*lambda0, PMLcylindrical=PMLcylindrical, PMLpenetrate=PMLpenetrate, MetalBase=MetalBase, t=t, comm=comm, model_rank=model_rank)

    p = rotsymsca.RotSymProblem(meshdata, material_epsr=material_epsr)
    p.Excitation(E0theta_inc=E0theta_inc, E0phi_inc=E0phi_inc,
                 theta_inc=theta_inc, phi_inc=phi_inc,
                 E0theta_ant=E0theta_ant, E0phi_ant=E0phi_ant,
                 theta_ant=theta_ant, phi_ant=phi_ant,
                 antenna_taper=antenna_taper)
    if full_computation:
        mvec, Esca, Esca_refl, Etot, Etot_refl, scattering_angles, farfield_amplitudes = p.ComputeSolutionsAndPostprocess(Nangles=Nangles)
    else:
        mvec, Esca, Esca_refl, Etot, Etot_refl, scattering_angles, farfield_amplitudes = p.ComputeSolutionsAndPostprocess(mvec=[-1, 1], Nangles=Nangles)

    # Collect far field results from all ranks
    farfields = comm.gather(farfield_amplitudes, root=model_rank)
    if comm.rank == model_rank:
        ff = sum(farfields)
    else:
        ff = None

    # Save data if on main process
    if comm.rank == model_rank:
        if antenna_mode:
            farfieldfilename = f'farfield_ant_{pol}.txt'
        else:
            farfieldfilename = f'farfield_sca_{pol}.txt'
        with open(farfieldfilename, 'w') as f:
            print('# Scattering angle (deg), Re(ff[theta]), Im(ff[theta]), Re(ff[phi]), Im(ff[phi])', file=f)
            for n in range(len(scattering_angles)):
                print(f'{scattering_angles[n]*180/np.pi}, {np.real(ff[0,n])}, {np.imag(ff[0,n])}, {np.real(ff[1,n])}, {np.imag(ff[1,n])}', file=f)

        if comm.size == 1:
            # Save near field data, only works on single process right now
            # First interpolate the results to a standard Lagrange vector space
            Velement = ufl.VectorElement('CG', meshdata.mesh.ufl_cell(), 1, dim=3)
            Vspace = dolfinx.fem.FunctionSpace(meshdata.mesh, Velement)
            u = dolfinx.fem.Function(Vspace)
            if antenna_mode:
                filename = f'fields_ant_{pol}_.xdmf'
                Elist = [Esca, Esca_refl]
            else:
                filename = f'fields_sca_{pol}_.xdmf'
                Elist = [Esca, Esca_refl, Etot, Etot_refl]
            with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, 'w') as f:
                f.write_mesh(meshdata.mesh)
                for n, E in enumerate(Elist):
                    u.interpolate(dolfinx.fem.Expression(ufl.as_vector((E[0], E[1], E[2])), Vspace.element.interpolation_points()))
                    f.write_function(u, n)

    # Make animation of near fields
    if antenna_mode:
        p.PlotField(Esca, Esca_refl, animate=True, filename=f'nearfield_ant_{pol}.mp4', clim=[-2, 2])
    else:
        p.PlotField(Esca, Esca_refl, animate=True, filename=f'nearfield_sca_{pol}.mp4', clim=[-2, 2])
        p.PlotField(Etot, Etot_refl, animate=True, filename=f'nearfield_tot_{pol}.mp4', clim=[-2, 2])
    return()

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    model_rank = 0
    theta = np.pi/4
    full_computation = True
    compute_radome(pol='theta', antenna_mode=True, theta=theta, full_computation=full_computation, comm=comm, model_rank=model_rank)
    compute_radome(pol='phi', antenna_mode=True, theta=theta, full_computation=full_computation, comm=comm, model_rank=model_rank)
    compute_radome(pol='theta', antenna_mode=False, theta=theta, full_computation=full_computation, comm=comm, model_rank=model_rank)
    compute_radome(pol='phi', antenna_mode=False, theta=theta, full_computation=full_computation, comm=comm, model_rank=model_rank)
