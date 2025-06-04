# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
import numpy.testing as nt
from astropy import units as u

from toast.ops import PixelsWCS

from .. import pixels_io_wcs as io
from ..pixels import PixelData, PixelDistribution
from .helpers import create_outdir
from .mpi import MPITestCase

try:
    from pixell import enmap

    available_pixell = True
except ModuleNotFoundError:
    available_pixell = False


class PixelTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        self.wcs, self.wcs_shape = PixelsWCS.create_wcs(
            coord="EQU",
            proj="CAR",
            bounds_deg=[-50, 50, -20, 20],  # lon_min, lon_max, lat_min, lat_max
            res_deg=[1, 1],
        )
        self.npix = np.prod(self.wcs_shape)
        self.nsub = [1, 10, 100]
        self.types = [
            np.float64,
            np.float32,
            np.int64,
            np.uint64,
            np.int32,
            np.uint32,
            np.int16,
            np.uint16,
            np.int8,
            np.uint8,
        ]
        self.datatypes = [np.float64, np.float32, np.int64, np.int32]

    def tearDown(self):
        pass

    def _make_pixdist(self, nsub, comm):
        valid_submaps = np.arange(0, nsub, 2, dtype=np.int32)
        # Make up some local submaps for each process
        local_submaps = None
        if comm is None:
            local_submaps = valid_submaps
        else:
            local_submaps = np.unique(
                np.floor_divide(
                    np.random.randint(0, 2 * nsub, size=(nsub // 2), dtype=np.int32), 2
                )
            )
        dist = PixelDistribution(
            n_pix=self.npix, n_submap=nsub, local_submaps=local_submaps, comm=comm
        )
        dist.wcs = self.wcs
        dist.wcs_shape = self.wcs_shape
        return dist

    def _make_pixdata(self, dist, dtype, nnz):
        units = u.dimensionless_unscaled
        if dtype == np.float64 or dtype == np.float32:
            units = u.K
        pdata = PixelData(dist, dtype, n_value=nnz, units=units)
        gl = list()
        for sm in pdata.distribution.local_submaps:
            for px in range(dist.n_pix_submap):
                if sm * dist.n_pix_submap + px < dist.n_pix:
                    gl.append(sm * dist.n_pix_submap + px)
        gl = np.array(gl, dtype=np.int64)
        subm, subpx = dist.global_pixel_to_submap(gl)
        ploc = dist.global_pixel_to_local(gl)
        ploc[:] *= 2
        pdata.raw[ploc] = 1
        for z in range(1, nnz):
            ploc[:] += 1
            pdata.raw[ploc] = 1
        return pdata

    def _test_io(self, suffix):
        np.random.seed(0)
        if self.comm is not None:
            np.random.seed(self.comm.rank)
        for nsb in self.nsub:
            dist = self._make_pixdist(nsb, self.comm)
            for tp in self.datatypes:
                pdata = self._make_pixdata(dist, tp, 2)
                pdata = PixelData(dist, tp, n_value=6)
                wcsfile = os.path.join(
                    self.outdir,
                    "data_sub{}_type-{}.{}".format(nsb, np.dtype(tp).char, suffix),
                )
                io.write_wcs_parallel(pdata, wcsfile)
                check = PixelData(dist, tp, n_value=6)
                io.read_wcs_parallel(check, wcsfile)
                nt.assert_equal(pdata.data, check.data)

                if not available_pixell:
                    # No serial tests without pixell
                    continue

                if self.comm is None or self.comm.size == 1:
                    # Write out the data serially and compare
                    fdata = list()
                    for col in range(pdata.n_value):
                        fdata.append(np.zeros(pdata.distribution.n_pix))
                    for lc, sm in enumerate(pdata.distribution.local_submaps):
                        global_offset = sm * pdata.distribution.n_pix_submap
                        n_copy = pdata.distribution.n_pix_submap
                        if global_offset + n_copy > pdata.distribution.n_pix:
                            n_copy = pdata.distribution.n_pix - global_offset
                        for col in range(pdata.n_value):
                            fdata[col][global_offset : global_offset + n_copy] = (
                                pdata.data[lc, 0:n_copy, col]
                            )
                    for col in range(pdata.n_value):
                        fdata[col] = fdata[col].reshape(self.wcs_shape)
                    serialfile = os.path.join(
                        self.outdir,
                        "serial_sub{}_type-{}.{}".format(
                            nsb, np.dtype(tp).char, suffix
                        ),
                    )
                    emap = enmap.zeros((pdata.n_value,) + self.wcs_shape, wcs=self.wcs)
                    emap[:] = fdata
                    enmap.write_map(serialfile, emap, fmt=suffix)
                    loaded = enmap.read_map(serialfile, fmt=suffix)
                    for lc, sm in enumerate(pdata.distribution.local_submaps):
                        global_offset = sm * pdata.distribution.n_pix_submap
                        n_check = pdata.distribution.n_pix_submap
                        if global_offset + n_check > pdata.distribution.n_pix:
                            n_check = pdata.distribution.n_pix - global_offset
                        for col in range(pdata.n_value):
                            nt.assert_equal(
                                loaded[col].ravel()[
                                    global_offset : global_offset + n_check
                                ],
                                pdata.data[lc, 0:n_check, col],
                            )
                    # Compare to file written with our own function
                    loaded = enmap.read_map(wcsfile, fmt=suffix)
                    for lc, sm in enumerate(pdata.distribution.local_submaps):
                        global_offset = sm * pdata.distribution.n_pix_submap
                        n_check = pdata.distribution.n_pix_submap
                        if global_offset + n_check > pdata.distribution.n_pix:
                            n_check = pdata.distribution.n_pix - global_offset
                        for col in range(pdata.n_value):
                            nt.assert_equal(
                                loaded[col].ravel()[
                                    global_offset : global_offset + n_check
                                ],
                                pdata.data[lc, 0:n_check, col],
                            )

    def test_io_fits(self):
        self._test_io("fits")

    def test_io_hdf5(self):
        self._test_io("hdf")
