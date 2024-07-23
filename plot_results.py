# Plot the results of the radome simulations.
#
# Daniel Sjöberg, 2024-07-13

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert

plt.rc('lines', linewidth=2)
plt.rc('font', size=12)
plt.rc('text', usetex=True)

def ReadData(filename):
    data = np.genfromtxt(filename, delimiter=',')
    theta = data[:,0]
    ff_theta = 20*np.log10(np.abs(data[:,1] + 1j*data[:,2]))
    ff_phi = 20*np.log10(np.abs(data[:,3] + 1j*data[:,4]))
    ff = 10*np.log10(np.abs(data[:,1] + 1j*data[:,2])**2 + np.abs(data[:,3] + 1j*data[:,4])**2)
    return(theta, ff_theta, ff_phi, ff)

for theta_degrees in [0, 10, 20, 30, 40, 50]: 
    for wfactor in [10]:#[10, 20]: 
        results_scatter = []
        results_antenna = []
        for air in [False, True]:
            for pol in ['theta', 'phi']:
                for antenna_mode in [True, False]:
                    filename = f'data/radome_w{wfactor}_air{air}_pol{pol}_antenna{antenna_mode}_theta{theta_degrees}_farfield.txt'
                    label = fr'pol = $\{pol}$, radome = {air^True}'
                    if antenna_mode:
                        results_antenna.append((filename, label))
                    else:
                        results_scatter.append((filename, label))

        # Make plot for scattering case
        fig1, ax1 = plt.subplots()
        ax1.set_xticks([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90])
        ax1.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        ax1.grid(True)
        for filename, label in results_scatter:
            thetaplot, ff_theta, ff_phi, ff = ReadData(filename)
            ax1.plot(thetaplot, ff + 10*np.log10(4*np.pi), label=label)
        ax1.set_xlim(-90, 90)
        ax1.legend(loc='best')
        ax1.set_xlabel('theta (degrees)')
        ax1.set_ylabel('Bistatic cross section (dBsm)')
        ax1.set_title(fr'Scattering case, $w = {wfactor}\lambda_0$, $\theta = {theta_degrees}^\circ$')
        plt.savefig(f'results_figures/scattering_w{wfactor}_theta{theta_degrees}.pdf')
        plt.close(fig1)

        # Make plot for transmitting antenna case
        fig2, ax2 = plt.subplots()
        ax2.set_xticks([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90])
        ax2.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        ax2.grid(True)
        for filename, label in results_antenna:
            thetaplot, ff_theta, ff_phi, ff = ReadData(filename)
            ax2.plot(thetaplot, ff - ff.max(), label=label)
        ax2.set_xlim(-90, 90)
        ax2.legend(loc='best')
        ax2.set_xlabel('theta (degrees)')
        ax2.set_ylabel('Normalized gain (dB)')
        ax2.set_title(fr'Transmitting case, $w = {wfactor}\lambda_0$, $\theta = {theta_degrees}^\circ$')
        plt.savefig(f'results_figures/transmitting_w{wfactor}_theta{theta_degrees}.pdf')
        plt.close(fig2)
    

