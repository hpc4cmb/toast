#!/usr/bin/env python3
"""
Some of our dependencies use astropy.io.data to do on-demand caching of
data files.  This is undesireable for large jobs.  This script makes the same
calls as the example scripts to attempt to trigger all the same data downloads
ahead of time.
"""

import numpy as np
import healpy as hp

nsides = [32]
cur = 32
while cur < 2048:
    cur *= 2
    nsides.append(cur)

for nside in nsides:
    print(
        "Trigger download of healpy pixel weights for nside {}".format(nside),
        flush=True
    )
    npix = 12 * nside**2
    components = 1
    data = np.zeros((components, npix), dtype=np.float32)
    out = hp.map2alm(data, use_pixel_weights=True, lmax=10)
