# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import pixels_io_healpix as io
from ..pixels import PixelData, PixelDistribution
from ._helpers import create_outdir
from .mpi import MPITestCase


class PixelTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        self.nsides = [8, 32]
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
        self.fitstypes = [np.float64, np.float32, np.int64, np.int32]
        # self.nsides = [2]
        # self.nsub = [8]
        # self.types = [np.int32]

    def tearDown(self):
        pass

    def _make_pixdist(self, nside, nsub, comm):
        npix = 12 * nside**2
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
            n_pix=npix, n_submap=nsub, local_submaps=local_submaps, comm=comm
        )
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

    def test_data(self):
        np.random.seed(0)
        if self.comm is not None:
            np.random.seed(self.comm.rank)
        for nside in self.nsides:
            for nsb in self.nsub:
                dist = self._make_pixdist(nside, nsb, self.comm)
                for tp in self.types:
                    pdata = self._make_pixdata(dist, tp, 2)
                    pdata = PixelData(dist, tp, n_value=2)

                    other = PixelData(dist, tp, n_value=2)
                    other.raw[:] = pdata.raw[:]

                    # if self.comm.rank == 0:
                    #     print("----- start orig -----")
                    # for p in range(self.comm.size):
                    #     if p == self.comm.rank:
                    #         print("proc {}:".format(p))
                    #         for lc, sm in enumerate(pdata.distribution.local_submaps):
                    #             print("submap {} = ".format(sm))
                    #             print(pdata.data[lc])
                    #         print("", flush=True)
                    #     self.comm.barrier()
                    #
                    # if self.comm.rank == 0:
                    #     print("----- start other -----")
                    # for p in range(self.comm.size):
                    #     if p == self.comm.rank:
                    #         print("proc {}:".format(p))
                    #         for lc, sm in enumerate(other.distribution.local_submaps):
                    #             print("submap {} = ".format(sm))
                    #             print(other.data[lc])
                    #         print("", flush=True)
                    #     self.comm.barrier()

                    pdata.sync_allreduce()

                    # if self.comm.rank == 0:
                    #     print("----- allreduce orig -----")
                    # for p in range(self.comm.size):
                    #     if p == self.comm.rank:
                    #         print("proc {}:".format(p))
                    #         for lc, sm in enumerate(pdata.distribution.local_submaps):
                    #             print("submap {} = ".format(sm))
                    #             print(pdata.data[lc])
                    #         print("", flush=True)
                    #     self.comm.barrier()

                    other.sync_alltoallv()

                    # if self.comm.rank == 0:
                    #     print("----- alltoallv other -----")
                    # for p in range(self.comm.size):
                    #     if p == self.comm.rank:
                    #         print("proc {}:".format(p))
                    #         for lc, sm in enumerate(other.distribution.local_submaps):
                    #             print("submap {} = ".format(sm))
                    #             print(other.data[lc])
                    #         print("", flush=True)
                    #     self.comm.barrier()

                    nt.assert_equal(pdata.data, other.data)

    def test_io_fits(self):
        np.random.seed(0)
        if self.comm is not None:
            np.random.seed(self.comm.rank)
        for nside in self.nsides:
            for nsb in self.nsub:
                dist = self._make_pixdist(nside, nsb, self.comm)
                for tp in self.fitstypes:
                    pdata = self._make_pixdata(dist, tp, 2)
                    pdata = PixelData(dist, tp, n_value=6)
                    fitsfile = os.path.join(
                        self.outdir,
                        "data_N{}_sub{}_type-{}.fits".format(
                            nside, nsb, np.dtype(tp).char
                        ),
                    )
                    io.write_healpix_fits(pdata, fitsfile, nest=True)
                    check = PixelData(dist, tp, n_value=6)
                    io.read_healpix_fits(check, fitsfile, nest=True)
                    nt.assert_equal(pdata.data, check.data)
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
                                fdata[col][
                                    global_offset : global_offset + n_copy
                                ] = pdata.data[lc, 0:n_copy, col]
                        serialfile = os.path.join(
                            self.outdir,
                            "serial_N{}_sub{}_type-{}.fits".format(
                                nside, nsb, np.dtype(tp).char
                            ),
                        )
                        hp.write_map(
                            serialfile,
                            fdata,
                            dtype=fdata[0].dtype,
                            fits_IDL=False,
                            nest=True,
                            overwrite=True,
                        )
                        loaded = hp.read_map(serialfile, nest=True, field=None)
                        for lc, sm in enumerate(pdata.distribution.local_submaps):
                            global_offset = sm * pdata.distribution.n_pix_submap
                            n_check = pdata.distribution.n_pix_submap
                            if global_offset + n_check > pdata.distribution.n_pix:
                                n_check = pdata.distribution.n_pix - global_offset
                            for col in range(pdata.n_value):
                                nt.assert_equal(
                                    loaded[col][
                                        global_offset : global_offset + n_check
                                    ],
                                    pdata.data[lc, 0:n_check, col],
                                )
                        # Compare to file written with our own function
                        loaded = hp.read_map(fitsfile, nest=True, field=None)
                        for lc, sm in enumerate(pdata.distribution.local_submaps):
                            global_offset = sm * pdata.distribution.n_pix_submap
                            n_check = pdata.distribution.n_pix_submap
                            if global_offset + n_check > pdata.distribution.n_pix:
                                n_check = pdata.distribution.n_pix - global_offset
                            for col in range(pdata.n_value):
                                nt.assert_equal(
                                    loaded[col][
                                        global_offset : global_offset + n_check
                                    ],
                                    pdata.data[lc, 0:n_check, col],
                                )

    def test_io_hdf5(self):
        np.random.seed(0)
        if self.comm is not None:
            np.random.seed(self.comm.rank)
        for nside in self.nsides:
            for nsb in self.nsub:
                dist = self._make_pixdist(nside, nsb, self.comm)
                for tp in self.fitstypes:
                    pdata = self._make_pixdata(dist, tp, 2)
                    pdata = PixelData(dist, tp, n_value=6)
                    hdf5file = os.path.join(
                        self.outdir,
                        "data_N{}_sub{}_type-{}.h5".format(
                            nside, nsb, np.dtype(tp).char
                        ),
                    )
                    io.write_healpix_hdf5(pdata, hdf5file, nest=True)
                    check = PixelData(dist, tp, n_value=6)
                    io.read_healpix_hdf5(check, hdf5file, nest=True)
                    nt.assert_equal(pdata.data, check.data)
                    if self.comm is None or self.comm.rank == 0:
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
                                fdata[col][
                                    global_offset : global_offset + n_copy
                                ] = pdata.data[lc, 0:n_copy, col]
                        serialfile = os.path.join(
                            self.outdir,
                            "serial_N{}_sub{}_type-{}.h5".format(
                                nside, nsb, np.dtype(tp).char
                            ),
                        )
                        io.write_healpix(serialfile, fdata, nest=True, overwrite=True)
                        loaded = io.read_healpix(
                            serialfile, nest=True, field=None, verbose=False
                        )
                        for lc, sm in enumerate(pdata.distribution.local_submaps):
                            global_offset = sm * pdata.distribution.n_pix_submap
                            n_check = pdata.distribution.n_pix_submap
                            if global_offset + n_check > pdata.distribution.n_pix:
                                n_check = pdata.distribution.n_pix - global_offset
                            for col in range(pdata.n_value):
                                nt.assert_equal(
                                    loaded[col][
                                        global_offset : global_offset + n_check
                                    ],
                                    pdata.data[lc, 0:n_check, col],
                                )
                        # Compare to file written with our parallel function
                        loaded = io.read_healpix(
                            hdf5file, nest=True, field=None, verbose=False
                        )
                        for lc, sm in enumerate(pdata.distribution.local_submaps):
                            global_offset = sm * pdata.distribution.n_pix_submap
                            n_check = pdata.distribution.n_pix_submap
                            if global_offset + n_check > pdata.distribution.n_pix:
                                n_check = pdata.distribution.n_pix - global_offset
                            for col in range(pdata.n_value):
                                nt.assert_equal(
                                    loaded[col][
                                        global_offset : global_offset + n_check
                                    ],
                                    pdata.data[lc, 0:n_check, col],
                                )
