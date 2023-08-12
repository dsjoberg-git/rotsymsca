# Script to run verification cases for rotsymsca.py
#
# Daniel Sjöberg, 2023-07-19

import numpy as np
import rotsymsca
import mesh_rotsymradome
from mpi4py import MPI
from scipy.constants import c as c0
from matplotlib import pyplot as plt
import miepython

def compute_sphere(a, f0, epsr, pol, h, full_computation=False, comm=MPI.COMM_WORLD, model_rank=0):
    lambda0 = c0/f0
    if pol == 'theta':
        E0theta_inc = 1
        E0phi_inc = 0
    else:
        E0theta_inc = 0
        E0phi_inc = 1
    theta_inc = 0
    phi_inc = 0
    
    radius_sphere = a
    radius_farfield = radius_sphere + 0.5*lambda0
    radius_domain = radius_farfield + 0.5*lambda0
    radius_pml = radius_domain + 0.5*lambda0
    if np.real(epsr) < 0:
        pec = True
    else:
        pec = False
        material_epsr = epsr
        
    meshdata = mesh_rotsymradome.CreateMeshSphere(radius_sphere=radius_sphere, radius_farfield=radius_farfield, radius_domain=radius_domain, radius_pml=radius_pml, h=h, pec=pec, visualize=False, comm=comm, model_rank=model_rank)

    p = rotsymsca.RotSymProblem(meshdata, f0=f0, material_epsr=epsr)
    p.Excitation(E0theta_inc=E0theta_inc, E0phi_inc=E0phi_inc,
                 theta_inc=theta_inc, phi_inc=phi_inc)
    if full_computation:
        mvec, Esca, Esca_refl, Etot, Etot_refl, scattering_angles, farfield_amplitudes = p.ComputeSolutionsAndPostprocess()
    else:
        mvec, Esca, Esca_refl, Etot, Etot_refl, scattering_angles, farfield_amplitudes = p.ComputeSolutionsAndPostprocess(mvec=[-1, 1])

    return(scattering_angles, farfield_amplitudes)

def error_norms(sol, ref_sol):
    absnorm = np.abs(sol - ref_sol)
    relnorm = absnorm/np.abs(ref_sol)
    rms_abs = np.sqrt(np.sum(absnorm**2)/len(ref_sol))
    rms_rel = np.sqrt(np.sum(relnorm**2)/len(ref_sol))
    max_abs = np.max(absnorm)
    max_rel = np.max(relnorm)
    return(rms_abs, rms_rel, max_abs, max_rel)
    
def verification(comm=MPI.COMM_WORLD, model_rank=0):
    f0 = 10e9
    lambda0 = c0/f0
    a = 0.5*lambda0
    epsr = -1#3*(1 - 0.1j)
    pol = 'theta'
    h_vec = [0.4*lambda0, 0.2*lambda0, 0.1*lambda0, 0.05*lambda0, 0.025*lambda0]
    ff_vec = []
    for h in h_vec:
        print(f'h/lambda = {h/lambda0}')
        scattering_angles, farfield_amplitudes = compute_sphere(a, f0, epsr, pol, h, full_computation=False, comm=comm, model_rank=model_rank)
        if pol == 'theta':
            ff_vec.append(farfield_amplitudes[0])
        else:
            ff_vec.append(farfield_amplitudes[1])

    # Collect far field results from all ranks
    ff_vecs = comm.gather(ff_vec, root=model_rank)
    if comm.rank == model_rank:
        ff_vec = np.sum(np.array(ff_vecs), axis=0)
    else:
        ff_vec = None

    if comm.rank == model_rank:
        if np.real(epsr) < 0:
            m = -1
        else:
            m = np.sqrt(epsr)
        x = 2*np.pi*a/lambda0
        if pol == 'theta':
            ref_solution = miepython.i_par(m, x, np.cos(scattering_angles), norm='qsca')*np.pi*a**2
        else:
            ref_solution = miepython.i_per(m, x, np.cos(scattering_angles), norm='qsca')*np.pi*a**2
        rms_abs_vec = []
        rms_rel_vec = []
        max_abs_vec = []
        max_rel_vec = []
        for n, h in enumerate(h_vec):
            rms_abs, rms_rel, max_abs, max_rel = error_norms(np.abs(ff_vec[n])**2, ref_solution)
            rms_abs_vec.append(rms_abs)
            rms_rel_vec.append(rms_rel)
            max_abs_vec.append(max_abs)
            max_rel_vec.append(max_rel)

        plt.figure()
        plt.plot(scattering_angles*180/np.pi, np.abs(ff_vec[-1])**2, label='rotsymsca')
        plt.plot(scattering_angles*180/np.pi, ref_solution, '--', label='miepython')
        plt.legend(loc='best')
        plt.xlabel('theta (degrees)')
 
    
        plt.figure()
        plt.loglog(lambda0/np.array(h_vec), np.array(max_rel_vec), marker='s', label='max rel error')
        plt.loglog(lambda0/np.array(h_vec), np.array(rms_rel_vec), marker='o', label='rms rel error')
        plt.xlabel(r'$\lambda/h$', fontsize=16)
        plt.grid(visible=True, which='both')
        plt.legend(loc='best')
        plt.show()

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    model_rank = 0
    verification(comm=comm, model_rank=model_rank)
    
