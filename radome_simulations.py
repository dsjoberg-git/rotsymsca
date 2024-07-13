# Script to do a sequence of radome simulations.
#
# Daniel Sjöberg, 2024-07-13

import numpy as np
import dolfinx, ufl
from mpi4py import MPI
import rotsymsca
import mesh_rotsymradome
from scipy.constants import c as c0
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

def compute_radome(pol='theta', antenna_mode=True, theta=np.pi/4, full_computation=True, comm=MPI.COMM_WORLD, model_rank=0, hfactor=0.1, hfinefactor=0.1, wfactor=10, air=False, material_epsr=3*(1-0.01j), hull_epsr=100-72j, Htransitionfactor=1, Hfactor=5, ComputeSVW=False, basefilename=''):
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
    hfine = hfinefactor*lambda0
    w = wfactor*lambda0
    H = Hfactor*lambda0
    antenna_taper = lambda x: np.cos(x[0]/(w/2)*np.pi/2)
    PMLcylindrical = True
    PMLpenetrate = False
    AntennaMetalBase = False
    Antenna = True
    t = 0.1*lambda0
    Nangles = 360*4 + 1

    epsr1 = material_epsr                 # Permittivity of radome
    epsr2 = hull_epsr                     # Permittivity of CFRP
    d = lambda0/2/np.real(np.sqrt(epsr1)) # Slab thickness
    v = np.linspace(0, 1, 1000)           # Volume fraction of CFRP
    epsr = (1 - v)*epsr1 + v*epsr2        # Simple mixing formula
    n = np.sqrt(epsr)
    r0 = (1 - n)/(1 + n)
    p0 = np.exp(-1j*2*np.pi/lambda0*n*d)
    r = r0*(1 - p0**2)/(1 - r0**2*p0**2)
    v_of_r = interp1d(np.abs(r)/np.abs(r).max(), v, fill_value='extrapolate')
    Htransition = lambda0*Htransitionfactor
    transition_volumefraction_hull = lambda x: v_of_r(-x[1]/Htransition)

    meshdata = mesh_rotsymradome.CreateMeshOgive(d=d, h=h, hfine=hfine, w=w, PMLcylindrical=PMLcylindrical, PMLpenetrate=PMLpenetrate, AntennaMetalBase=AntennaMetalBase, Antenna=Antenna, t=t, comm=comm, model_rank=model_rank, Htransition=Htransition, H=H, visualize=False)
    
    if not air:
        p = rotsymsca.RotSymProblem(meshdata, material_epsr=material_epsr, hull_epsr=hull_epsr, transition_volumefraction_hull=transition_volumefraction_hull)
    else:
        p = rotsymsca.RotSymProblem(meshdata, material_epsr=1.0, hull_epsr=1.0, transition_volumefraction_hull=transition_volumefraction_hull)
    p.Excitation(E0theta_inc=E0theta_inc, E0phi_inc=E0phi_inc,
                 theta_inc=theta_inc, phi_inc=phi_inc,
                 E0theta_ant=E0theta_ant, E0phi_ant=E0phi_ant,
                 theta_ant=theta_ant, phi_ant=phi_ant,
                 antenna_taper=antenna_taper)
    if True: # Test for ghost farfield facets and plot mesh partition
        ghost_ff_facets = mesh_rotsymradome.CheckGhostFarfieldFacets(comm, model_rank, meshdata.mesh, meshdata.boundaries, meshdata.boundary_markers['farfield'])
        if len(ghost_ff_facets) > 0:
            print('Ghost farfield facets detected, basefilename={basefilename}.')
            if comm.rank == model_rank:
                with open(basefilename + '_ghostfff.txt', 'w') as f:
                    print(f'Ghost facets {ghost_ff_facets}', file=f)

    if full_computation:
        mvec, Esca, Esca_refl, Etot, Etot_refl, scattering_angles, farfield_amplitudes, SVW = p.ComputeSolutionsAndPostprocess(Nangles=Nangles, ComputeSVW=ComputeSVW)
    else:
        mvec, Esca, Esca_refl, Etot, Etot_refl, scattering_angles, farfield_amplitudes, SVW = p.ComputeSolutionsAndPostprocess(mvec=[-1, 1], Nangles=Nangles, ComputeSVW=ComputeSVW)

    # Collect far field results from all ranks
    farfields = comm.gather(farfield_amplitudes, root=model_rank)
    if comm.rank == model_rank:
        ff = sum(farfields)
    else:
        ff = None
    ff = comm.bcast(ff, root=model_rank)

    # Save data if on main process
    if comm.rank == model_rank:
        farfieldfilename = basefilename + '_farfield.txt'
        with open(farfieldfilename, 'w') as f:
            print('# Scattering angle (deg), Re(ff[theta]), Im(ff[theta]), Re(ff[phi]), Im(ff[phi])', file=f)
            for n in range(len(scattering_angles)):
                print(f'{scattering_angles[n]*180/np.pi}, {np.real(ff[0,n])}, {np.imag(ff[0,n])}, {np.real(ff[1,n])}, {np.imag(ff[1,n])}', file=f)

    # Make animation of near fields
    nearfieldfilename = basefilename + '_nearfield.mp4'
    nearfieldfilename_sca = basefilename + '_nearfield_sca.mp4'
    nearfieldfilename_tot = basefilename + '_nearfield_tot.mp4'
    if antenna_mode:
        p.PlotField(Esca, Esca_refl, animate=True, filename=nearfieldfilename, clim=[-2, 2])
    else:
        p.PlotField(Esca, Esca_refl, animate=True, filename=nearfieldfilename_sca, clim=[-2, 2])
        p.PlotField(Etot, Etot_refl, animate=True, filename=nearfieldfilename_tot, clim=[-2, 2])
    return()

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    model_rank = 0
    theta0 = np.pi/4
    full_computation = False
    hfactor0 = 0.1
    wfactor0 = 10
    Htransitionfactor0 = 1
    epsr_radome = 3
    epsr_hull = epsr_radome

    if True: # Parameter sweep
        for wfactor in [10, 20]:
            for air in [False, True]:
                for pol in ['theta', 'phi']:
                    for antenna_mode in [True, False]:
                        for theta_degrees in [0]:#[0, 10, 20, 30, 40, 50]
                            Hfactor = wfactor/2
                            basefilename = f'data/radome_w{wfactor}_air{air}_pol{pol}_antenna{antenna_mode}_theta{theta_degrees}'
                            compute_radome(pol=pol, antenna_mode=antenna_mode, theta=theta_degrees*np.pi/180, full_computation=full_computation, comm=comm, model_rank=model_rank, hfactor=hfactor0, hfinefactor=hfactor0, wfactor=wfactor, air=air, Htransitionfactor=Htransitionfactor0, Hfactor=Hfactor, basefilename=basefilename)

