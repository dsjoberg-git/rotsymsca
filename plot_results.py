# Plot the results of the radome simulations.
#
# Daniel Sjöberg, 2023-10-12

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert

plt.rc('lines', linewidth=2)
plt.rc('font', size=12)

results = [('farfield_radome_sca_phi_h0.1_w10_Ht0.11.txt', '0'),
           ('farfield_radome_sca_phi_h0.1_w10_Ht1.txt', '1'),
           ('farfield_radome_sca_phi_h0.1_w10_Ht2.txt', '2'),
           ('farfield_radome_sca_phi_h0.1_w10_Ht4.txt', '4')]
results = [('farfield_radome_sca_phi_h0.1_w20_Ht0.11.txt', 'untreated'),
           ('farfield_radome_sca_phi_h0.1_w20_Ht1.txt', 'treated1'),
           ('farfield_radome_sca_phi_h0.1_w20_Ht2.txt', 'treated2'),
           ('farfield_radome_sca_phi_h0.1_w20_Ht4.txt', 'treated')]

def ReadData(filename):
    data = np.genfromtxt(filename, delimiter=',')
    theta = data[:,0]
    ff_theta = 20*np.log10(np.abs(data[:,1] + 1j*data[:,2]))
    ff_phi = 20*np.log10(np.abs(data[:,3] + 1j*data[:,4]))
    ff = 10*np.log10(np.abs(data[:,1] + 1j*data[:,2])**2 + np.abs(data[:,3] + 1j*data[:,4])**2)
    return(theta, ff_theta, ff_phi, ff)

def envelope(data):
    """Compute the envelope of a data set. Does not really work here."""
    x = hilbert(data - np.mean(data))
    return(abs(x))
           
fig, ax = plt.subplots()
ax.set_xticks([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90])
ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
ax.grid(True)
for filename, label in results:
    theta, ff_theta, ff_phi, ff = ReadData(filename)
    ax.plot(theta, ff + 10*np.log10(4*np.pi), label=label)
#    ax.plot(theta, envelope(ff), '--')
#plt.ylim(-80, 0)
plt.xlim(-90, 90)
plt.ylim(-35, -5)
plt.legend(loc='best')
plt.xlabel('theta (degrees)')
plt.ylabel('Bistatic cross section (dBsm)')
plt.show()
