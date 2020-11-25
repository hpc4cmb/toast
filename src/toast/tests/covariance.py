# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt

from astropy import units as u

import healpy as hp

from .mpi import MPITestCase

from .. import future_ops as ops

from ..pixels import PixelDistribution, PixelData

from ..covariance import covariance_invert, covariance_multiply, covariance_apply

from ._helpers import create_outdir, create_satellite_data


class CovarianceTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def create_invnpp(self):
        """Helper function to build a realistic inverse pixel covariance."""

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        pointing = ops.PointingHealpix(
            nside=64, mode="IQU", hwp_angle="hwp_angle", create_dist="pixel_dist"
        )
        pointing.apply(data)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise timestreams

        sim_noise = ops.SimNoise(noise_model="noise_model", out="noise")
        sim_noise.apply(data)

        # Build an inverse covariance

        build_invnpp = ops.BuildInverseCovariance(
            pixel_dist="pixel_dist", noise_model="noise_model"
        )
        invnpp = build_invnpp.apply(data)

        del data
        return invnpp

    def print_cov(self, mat):
        for p in range(mat.distribution.n_local_submap * mat.distribution.n_pix_submap):
            if mat.raw[p * mat.n_value] == 0:
                continue
            msg = "local pixel {}:".format(p)
            for nv in range(mat.n_value):
                msg += " {}".format(mat.raw[p * mat.n_value + nv])
            print(msg)

    def test_invert(self):
        threshold = 1.0e-8

        invnpp = self.create_invnpp()

        check = invnpp.duplicate()

        rcond = PixelData(invnpp.distribution, np.float64, n_value=1)

        # Invert twice, using a different communication algorithm each way.
        covariance_invert(invnpp, threshold, rcond=rcond, use_alltoallv=True)
        covariance_invert(invnpp, threshold, use_alltoallv=False)

        for sm in range(invnpp.distribution.n_local_submap):
            good = np.where(rcond.data[sm] > threshold)[0]
            nt.assert_almost_equal(invnpp.data[sm, good, :], check.data[sm, good, :])

    def test_multiply(self):
        threshold = 1.0e-15

        # Build an inverse
        invnpp = self.create_invnpp()
        # print("invnpp:")
        # self.print_cov(invnpp)

        # Get the covariance
        npp = invnpp.duplicate()
        covariance_invert(npp, threshold, use_alltoallv=True)
        # print("npp:")
        # self.print_cov(npp)

        # Multiply the two
        covariance_multiply(npp, invnpp, use_alltoallv=True)
        # print("identity:")
        # self.print_cov(npp)

        for sm in range(npp.distribution.n_local_submap):
            for spix in range(npp.distribution.n_pix_submap):
                if npp.data[sm, spix, 0] == 0:
                    continue
                nt.assert_almost_equal(npp.data[sm, spix, 0], 1.0)
                nt.assert_almost_equal(npp.data[sm, spix, 1], 0.0)
                nt.assert_almost_equal(npp.data[sm, spix, 2], 0.0)
                nt.assert_almost_equal(npp.data[sm, spix, 3], 1.0)
                nt.assert_almost_equal(npp.data[sm, spix, 4], 0.0)
                nt.assert_almost_equal(npp.data[sm, spix, 5], 1.0)

    def test_apply(self):
        threshold = 1.0e-15

        # Build an inverse
        invnpp = self.create_invnpp()

        # Get the covariance
        npp = invnpp.duplicate()
        covariance_invert(npp, threshold, use_alltoallv=True)

        # Random signal
        sig = PixelData(npp.distribution, np.float64, n_value=3)
        sig.raw[:] = np.random.normal(size=len(sig.raw))

        check = sig.duplicate()

        # Apply inverse and then covariance and check that we recover the original.
        covariance_apply(invnpp, sig)
        covariance_apply(npp, sig)

        for sm in range(npp.distribution.n_local_submap):
            for spix in range(npp.distribution.n_pix_submap):
                if npp.data[sm, spix, 0] == 0:
                    continue
                nt.assert_almost_equal(sig.data[sm, spix], check.data[sm, spix])
