
import sys
import os

import matplotlib.pyplot as plt

import numpy as np
import healpy as hp

madam_out = sys.argv[1]

# hits

file = os.path.join(madam_out, 'madam_hmap.fits')
data = hp.read_map(file, nest=True)

outfile = "{}.png".format(file)
hp.mollview(data, xsize=1600, nest=True)
plt.savefig(outfile)
plt.close()

# binned map

file = os.path.join(madam_out, 'madam_bmap.fits')
data = hp.read_map(file, nest=True)

outfile = "{}.png".format(file)
hp.mollview(data, xsize=1600, nest=True, remove_mono=True)
plt.savefig(outfile)
plt.close()

# destriped map

file = os.path.join(madam_out, 'madam_map.fits')
data = hp.read_map(file, nest=True)

outfile = "{}.png".format(file)
hp.mollview(data, xsize=1600, nest=True, remove_mono=True)
plt.savefig(outfile)
plt.close()

