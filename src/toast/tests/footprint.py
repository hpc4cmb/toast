# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import astropy.io.fits as af
import healpy as hp
import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops
from ..footprint import footprint_distribution
from .helpers import create_outdir
from .mpi import MPITestCase


class FootprintTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        self.wcs_proj_dims = (1000, 500)
        self.nside = 128
        self.nside_submap = 16

    def tearDown(self):
        pass

    def _create_wcs_coverage(self, outfile):
        res_deg = (0.01, 0.01)
        dims = self.wcs_proj_dims
        center_deg = (130.0, -30.0)
        wcs, wcs_shape = ops.PixelsWCS.create_wcs(
            coord="EQU",
            proj="CAR",
            center_deg=center_deg,
            bounds_deg=None,
            res_deg=res_deg,
            dims=dims,
        )
        if self.comm is None or self.comm.rank == 0:
            pixdata = np.ones((1, wcs_shape[1], wcs_shape[0]), dtype=np.float32)
            header = wcs.to_header()
            hdu = af.PrimaryHDU(data=pixdata, header=header)
            hdu.writeto(outfile)
        return wcs, wcs_shape

    def _create_healpix_coverage(self, nside, nside_submap, outfile, is_submap=False):
        n_submap = 12 * nside_submap**2
        hit_submaps = None
        if self.comm is None or self.comm.rank == 0:
            # Randomly select some submaps
            subvals = [True, False]
            hit_submaps = np.random.choice(subvals, size=(n_submap,)).astype(bool)
        if self.comm is not None:
            hit_submaps = self.comm.bcast(hit_submaps, root=0)
        if self.comm is None or self.comm.rank == 0:
            sub_pixels = np.zeros(n_submap, dtype=np.int32)
            sub_pixels[hit_submaps] = 1
            if is_submap:
                # Write it out and we are done
                hp.write_map(outfile, sub_pixels, nest=True)
            else:
                # Compute the full-resolution map and write that
                pixels = hp.ud_grade(
                    sub_pixels, nside, order_in="NEST", order_out="NEST"
                )
                hp.write_map(outfile, pixels, nest=True)
        return hit_submaps

    def test_wcs(self):
        footfile = os.path.join(self.outdir, "wcs_footprint.fits")
        wcs, wcs_shape = self._create_wcs_coverage(footfile)
        dist = footprint_distribution(wcs_coverage_file=footfile, comm=self.comm)

        # Check that the distribution has expected properties
        n_pix = np.prod(wcs_shape)
        self.assertTrue(dist.n_submap == 1)
        self.assertTrue(n_pix == dist.n_pix)
        self.assertTrue(n_pix == dist.n_pix_submap)
        self.assertTrue(dist.local_submaps[0] == 0)

    def test_healpix(self):
        n_submap = 12 * self.nside_submap**2
        n_pix = 12 * self.nside**2
        n_pix_submap = n_pix // n_submap

        # Create a distribution from healpix footprint file
        footfile = os.path.join(self.outdir, "healpix_footprint.fits")
        hit_submaps = self._create_healpix_coverage(
            self.nside, self.nside_submap, footfile, is_submap=False
        )
        dist = footprint_distribution(
            healpix_coverage_file=footfile,
            healpix_nside_submap=self.nside_submap,
            comm=self.comm,
        )
        self.assertTrue(dist.n_submap == n_submap)
        self.assertTrue(dist.n_pix == n_pix)
        self.assertTrue(dist.n_pix_submap == n_pix_submap)
        check_submaps = np.arange(n_submap, dtype=np.int64)[hit_submaps]
        self.assertTrue(np.array_equal(dist.local_submaps, check_submaps))

        # Create a distribution from healpix submap footprint file
        footfile = os.path.join(self.outdir, "healpix_submap_footprint.fits")
        hit_submaps = self._create_healpix_coverage(
            self.nside, self.nside_submap, footfile, is_submap=True
        )
        dist = footprint_distribution(
            healpix_submap_file=footfile, healpix_nside=self.nside, comm=self.comm
        )
        self.assertTrue(dist.n_submap == n_submap)
        self.assertTrue(dist.n_pix == n_pix)
        self.assertTrue(dist.n_pix_submap == n_pix_submap)
        check_submaps = np.arange(n_submap, dtype=np.int64)[hit_submaps]
        self.assertTrue(np.array_equal(dist.local_submaps, check_submaps))

        # Now check manual creation of a full-sky healpix footprint
        dist = footprint_distribution(
            healpix_nside=self.nside,
            healpix_nside_submap=self.nside_submap,
            comm=self.comm,
        )
        self.assertTrue(dist.n_submap == n_submap)
        self.assertTrue(dist.n_pix == n_pix)
        self.assertTrue(dist.n_pix_submap == n_pix_submap)
        check_submaps = np.arange(n_submap, dtype=np.int64)
        self.assertTrue(np.array_equal(dist.local_submaps, check_submaps))
