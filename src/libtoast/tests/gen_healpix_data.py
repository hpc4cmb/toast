# This generates a small dataset with healpy
# that can be inserted into the unit tests.
# To update test data, do:
#
# %> python gen_healpix_data.py > data_healpix.cpp
#

import numpy as np
import healpy as hp


nside = 16384

angperring = 10

nring = 7

thetainc = 0.98 * (np.pi / nring)

phiinc = 0.98 * (2.0 * np.pi / angperring)

theta = np.zeros(nring * angperring)
phi = np.zeros_like(theta)

n = 0

for t in range(nring):
    for p in range(angperring):
        theta[n] = t * thetainc
        phi[n] = p * phiinc
        n += 1

pixring = hp.ang2pix(nside, theta, phi)
pixnest = hp.ring2nest(nside, pixring)

ringtheta, ringphi = hp.pix2ang(nside, pixring)

print("    int64_t nside = {};".format(nside))
print("")

print("    int64_t ntest = {};".format(n))
print("")

print("    double theta[{}] = {{".format(n))
for i in range(n):
    print("        {},".format(theta[i]))
print("    };")
print("")

print("    double phi[{}] = {{".format(n))
for i in range(n):
    print("        {},".format(phi[i]))
print("    };")
print("")

print("    int64_t pixring[{}] = {{".format(n))
for i in range(n):
    print("        {},".format(pixring[i]))
print("    };")
print("")

print("    int64_t pixnest[{}] = {{".format(n))
for i in range(n):
    print("        {},".format(pixnest[i]))
print("    };")
print("")

print("    double ringtheta[{}] = {{".format(n))
for i in range(n):
    print("        {},".format(ringtheta[i]))
print("    };")
print("")

print("    double ringphi[{}] = {{".format(n))
for i in range(n):
    print("        {},".format(ringphi[i]))
print("    };")
print("")
