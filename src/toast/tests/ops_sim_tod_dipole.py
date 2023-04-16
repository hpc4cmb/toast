# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
from astropy import units as u

from .. import ops as ops
from .. import qarray as qa
from ..dipole import dipole
from ..pixels_io_healpix import write_healpix_fits
from ..vis import set_matplotlib_backend
from ._helpers import close_data, create_healpix_ring_satellite, create_outdir
from .mpi import MPITestCase


class SimDipoleTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        self.nside = 64

        # Dipole parameters
        self.solar_speed = 369.0
        gal_theta = np.deg2rad(90.0 - 48.05)
        gal_phi = np.deg2rad(264.31)
        z = self.solar_speed * np.cos(gal_theta)
        x = self.solar_speed * np.sin(gal_theta) * np.cos(gal_phi)
        y = self.solar_speed * np.sin(gal_theta) * np.sin(gal_phi)
        self.solar_vel = np.array([x, y, z])
        self.solar_quat = qa.from_vectors(np.array([0.0, 0.0, 1.0]), self.solar_vel)

        self.dip_check = 0.00335673

        self.dip_max_pix = hp.ang2pix(self.nside, gal_theta, gal_phi, nest=False)
        self.dip_min_pix = hp.ang2pix(
            self.nside, (np.pi - gal_theta), (np.pi + gal_phi), nest=False
        )

    def test_dipole_func(self):
        # Verify that we get the right magnitude if we are pointed at the
        # velocity maximum.
        dtod = dipole(self.solar_quat.reshape((1, 4)), solar=self.solar_vel)
        np.testing.assert_almost_equal(dtod, self.dip_check * np.ones_like(dtod))
        return

    def test_dipole_func_total(self):
        # Verify that we get the right magnitude if we are pointed at the
        # velocity maximum.
        quat = np.array(
            [
                [0.5213338, 0.47771442, -0.5213338, 0.47771442],
                [0.52143458, 0.47770023, -0.52123302, 0.4777286],
                [0.52153535, 0.47768602, -0.52113222, 0.47774277],
                [0.52163611, 0.4776718, -0.52103142, 0.47775692],
                [0.52173686, 0.47765757, -0.52093061, 0.47777106],
            ]
        )
        v_sat = np.array(
            [
                [1.82378638e-15, 2.97846918e01, 0.00000000e00],
                [-1.48252084e-07, 2.97846918e01, 0.00000000e00],
                [-2.96504176e-07, 2.97846918e01, 0.00000000e00],
                [-4.44756262e-07, 2.97846918e01, 0.00000000e00],
                [-5.93008348e-07, 2.97846918e01, 0.00000000e00],
            ]
        )
        v_sol = np.array([-25.7213418, -244.31203375, 275.33805175])

        dtod = dipole(quat, vel=v_sat, solar=v_sol, cmb=2.725 * u.Kelvin)
        # computed with github.com/zonca/dipole
        expected = np.array(
            [0.00196249, 0.00196203, 0.00196157, 0.00196111, 0.00196065]
        )
        np.testing.assert_allclose(dtod, expected, rtol=1e-5)
        return

    def test_sim(self):
        # Create a fake scan strategy that hits every pixel once.
        data = create_healpix_ring_satellite(self.comm, nside=self.nside)

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        # make a simple pointing matrix
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=self.nside,
            nest=False,
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeights(
            mode="I",
            detector_pointing=detpointing,
        )

        # Generate timestreams
        sim_dipole = ops.SimDipole(mode="solar", coord="G")
        sim_dipole.exec(data)

        # Build the covariance and hits
        cov_and_hits = ops.CovarianceAndHits(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            rcond_threshold=1.0e-2,
            sync_type="alltoallv",
        )
        cov_and_hits.apply(data)

        # Clean up temporary objects
        ops.Delete(detdata=[detpointing.quats, pixels.pixels, weights.weights]).apply(
            data
        )

        # Set up binned map

        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data=sim_dipole.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner.apply(data)

        toast_hit_path = os.path.join(self.outdir, "toast_hits.fits")
        toast_bin_path = os.path.join(self.outdir, "toast_bin.fits")
        write_healpix_fits(data[binner.binned], toast_bin_path, nest=False)
        write_healpix_fits(data[cov_and_hits.hits], toast_hit_path, nest=False)

        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        if rank == 0:
            import matplotlib.pyplot as plt

            mapfile = os.path.join(self.outdir, "toast_hits.fits")
            mdata = hp.read_map(mapfile, nest=False)

            outfile = "{}.png".format(mapfile)
            hp.mollview(mdata, xsize=1600, nest=False)
            plt.savefig(outfile)
            plt.close()

            mapfile = os.path.join(self.outdir, "toast_bin.fits")
            mdata = hp.read_map(mapfile, field=0, nest=False)

            outfile = "{}.png".format(mapfile)
            hp.mollview(mdata, xsize=1600, nest=False)
            plt.savefig(outfile)
            plt.close()

            # verify that the extrema are in the correct location
            # and have the correct value.

            minmap = np.min(mdata)
            maxmap = np.max(mdata)
            minloc = np.argmin(mdata)
            maxloc = np.argmax(mdata)

            np.testing.assert_almost_equal(maxmap, self.dip_check, decimal=5)
            np.testing.assert_almost_equal(minmap, -self.dip_check, decimal=5)
            np.testing.assert_equal(minloc, self.dip_min_pix)
            np.testing.assert_equal(maxloc, self.dip_max_pix)

        close_data(data)
