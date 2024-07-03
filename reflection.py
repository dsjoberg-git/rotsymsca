# Compute the reflection coefficient of a plastic slab as the amount
# of carbon fiber is increased.
#
# Daniel Sjöberg, 2023-10-11

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import c as c0
from scipy.interpolate import interp1d

epsr1 = 3 - 0.01j                     # Permittivity of radome
epsr2 = 100 - 72j                     # Permittivity of CFRP
f0 = 10e9                             # Frequency
lambda0 = c0/f0                       # Wavelength
d = lambda0/2/np.real(np.sqrt(epsr1)) # Slab thickness
v = np.linspace(0, 1, 1000)           # Volume fraction of CFRP

epsr = (1 - v)*epsr1 + v*epsr2        # Simple mixing formula
n = np.sqrt(epsr)
r0 = (1 - n)/(1 + n)
p0 = np.exp(-1j*2*np.pi/lambda0*n*d)
r = r0*(1 - p0**2)/(1 - r0**2*p0**2)
t = p0*(1 - r0**2)/(1 - r0**2*p0**2)
r_pec = (r0 - p0**2)/(1 - r0*p0**2)

v_of_r = interp1d(np.abs(r)/np.abs(r).max(), v, fill_value='extrapolate')
rvec = np.linspace(0, 1, 100)

plt.plot(v, np.abs(r))
plt.plot(v, np.abs(r_pec))
plt.plot(rvec, v_of_r(rvec))

plt.figure()
plt.plot(np.real(r), np.imag(r))
plt.plot(np.real(r_pec), np.imag(r_pec))
phi = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(phi), np.sin(phi))
plt.grid(True)
plt.gca().set_aspect('equal')
plt.show()
