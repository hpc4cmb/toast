"""This script is used to convert MERRA-2 FITS files into 32bit HDF5."""

import os
import sys

import fitsio
import h5py
import numpy as np

infile = sys.argv[1]
outfile = sys.argv[2]

fits = fitsio.FITS(infile)

if os.path.exists(outfile):
    os.remove(outfile)
hf = h5py.File(outfile, "w")

for mn in range(12):
    head = fits[mn + 1].read_header()
    dat = fits[mn + 1].read()

    grp = hf.create_group("month_{:02d}".format(mn))
    meta = grp.attrs
    meta.create("PROBSTRT", head["PROBSTRT"])
    meta.create("PROBSTOP", head["PROBSTOP"])
    meta.create("PROBSTEP", head["PROBSTEP"])
    meta.create("NSTEP", head["NSTEP"])
    meta.create("SOURCE", head["SOURCE"])

    for datname in dat.dtype.names:
        ds = grp.create_dataset(
            datname,
            data=dat[datname].astype(np.float32),
        )

hf.flush()
hf.close()

fits.close()
del fits
