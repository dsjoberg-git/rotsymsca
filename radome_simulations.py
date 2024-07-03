# Script to do a sequence of radome simulations.
# Need to organize filenames!
#
# Daniel Sjöberg, 2023-10-12

import numpy as np
import dolfinx, ufl
from mpi4py import MPI
import rotsymsca
import mesh_rotsymradome
from scipy.constants import c as c0
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

def compute_radome(pol='theta', antenna_mode=True, theta=np.pi/4, full_computation=True, comm=MPI.COMM_WORLD, model_rank=0, hfactor=0.1, wfactor=10, air=False, material_epsr=3-0.01j, CFRP_epsr=100-72j, Htransitionfactor=1):
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
    h = hfactor*lambda0
    w = wfactor*lambda0
    antenna_taper = lambda x: np.cos(x[0]/(w/2)*np.pi/2)
    PMLcylindrical = True
    PMLpenetrate = True
    MetalBase = False
    t = 0.1*lambda0
    Nangles = 360*4 + 1

    epsr1 = material_epsr                 # Permittivity of radome
    epsr2 = CFRP_epsr                     # Permittivity of CFRP
#    f0 = 10e9                             # Frequency
#    lambda0 = c0/f0                       # Wavelength
    d = lambda0/2/np.real(np.sqrt(epsr1)) # Slab thickness
    v = np.linspace(0, 1, 1000)           # Volume fraction of CFRP
    epsr = (1 - v)*epsr1 + v*epsr2        # Simple mixing formula
    n = np.sqrt(epsr)
    r0 = (1 - n)/(1 + n)
    p0 = np.exp(-1j*2*np.pi/lambda0*n*d)
    r = r0*(1 - p0**2)/(1 - r0**2*p0**2)
    v_of_r = interp1d(np.abs(r)/np.abs(r).max(), v, fill_value='extrapolate')
    Htransition = lambda0*Htransitionfactor
    transition_volumefraction_CFRP = lambda x: v_of_r(-x[1]/Htransition)

    meshdata = mesh_rotsymradome.CreateMeshOgive(d=d, h=h, w=w, PMLcylindrical=PMLcylindrical, PMLpenetrate=PMLpenetrate, MetalBase=MetalBase, t=t, comm=comm, model_rank=model_rank, Htransition=Htransition)

    if not air:
        p = rotsymsca.RotSymProblem(meshdata, material_epsr=material_epsr, CFRP_epsr=CFRP_epsr, transition_volumefraction_CFRP=transition_volumefraction_CFRP)
    else:
        p = rotsymsca.RotSymProblem(meshdata, material_epsr=1.0, CFRP_epsr=1.0, transition_volumefraction_CFRP=transition_volumefraction_CFRP)
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

    # Set up a base filename
    basefilename = f'_radome_sca_{pol}_h{hfactor}_w{wfactor}_Ht{Htransitionfactor}'
            
    # Save data if on main process
    if comm.rank == model_rank:
        farfieldfilename = 'farfield' + basefilename + '.txt'
        with open(farfieldfilename, 'w') as f:
            print('# Scattering angle (deg), Re(ff[theta]), Im(ff[theta]), Re(ff[phi]), Im(ff[phi])', file=f)
            for n in range(len(scattering_angles)):
                print(f'{scattering_angles[n]*180/np.pi}, {np.real(ff[0,n])}, {np.imag(ff[0,n])}, {np.real(ff[1,n])}, {np.imag(ff[1,n])}', file=f)

    # Make animation of near fields
    nearfieldfilename = 'nearfield' + basefilename + '.mp4'
    nearfieldfilename_sca = 'nearfield' + basefilename + '_sca.mp4'
    nearfieldfilename_tot = 'nearfield' + basefilename + '_tot.mp4'
    if antenna_mode:
        p.PlotField(Esca, Esca_refl, animate=True, filename=nearfieldfilename, clim=[-2, 2])
    else:
        p.PlotField(Esca, Esca_refl, animate=True, filename=nearfieldfilename_sca, clim=[-2, 2])
        p.PlotField(Etot, Etot_refl, animate=True, filename=nearfieldfilename_tot, clim=[-2, 2])
    return()

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    model_rank = 0
    theta = np.pi/4
    full_computation = True
    hfactor0 = 0.1
    wfactor0 = 10
    epsr_radome = 3 - 0.01j
    epsr_CFRP = 100 - 72j

    if True: # Parameter sweep
        for air in [False]:#[False, True]:
            for pol in ['phi']:#['theta', 'phi']:
                for antenna_mode in [False]:#[True, False]:
                    for Htransitionfactor in [0.11, 4]:#[0.11, 1, 2, 4]:
                        compute_radome(pol=pol, antenna_mode=antenna_mode, theta=theta, full_computation=full_computation, comm=comm, model_rank=model_rank, hfactor=hfactor0, wfactor=wfactor0, air=air, Htransitionfactor=Htransitionfactor)
