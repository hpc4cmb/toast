# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops as ops
from ..covariance import covariance_apply, covariance_invert, covariance_multiply
from ..mpi import MPI
from ..pixels import PixelData, PixelDistribution
from ._helpers import close_data, create_outdir, create_satellite_data
from .mpi import MPITestCase


class CovarianceTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def create_invnpp(self, stype):
        """Helper function to build a realistic inverse pixel covariance."""
        np.random.seed(123456)

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside_submap=16,
            nside=64,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle="hwp_angle",
            detector_pointing=detpointing,
        )
        weights.apply(data)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise timestreams

        sim_noise = ops.SimNoise(noise_model="noise_model", out="noise")
        sim_noise.apply(data)

        # Build an inverse covariance

        build_invnpp = ops.BuildInverseCovariance(
            pixel_dist="pixel_dist", noise_model="noise_model", sync_type=stype
        )
        build_invnpp.apply(data)
        invnpp = data[build_invnpp.inverse_covariance]

        close_data(data)
        return invnpp

    def print_cov(self, mat):
        comm = mat.distribution.comm
        myrank = 0
        nproc = 1
        if comm is not None:
            myrank = comm.rank
            nproc = comm.size
        for rank in range(nproc):
            if rank == myrank:
                for p in range(
                    mat.distribution.n_local_submap * mat.distribution.n_pix_submap
                ):
                    if mat.raw[p * mat.n_value] == 0:
                        continue
                    msg = "local pixel {}:".format(p)
                    for nv in range(mat.n_value):
                        msg += " {}".format(mat.raw[p * mat.n_value + nv])
                    print(msg)

    def test_invert(self):
        threshold = 1.0e-6

        for stype in ["allreduce", "alltoallv"]:
            invnpp = self.create_invnpp(stype)
            comm = invnpp.distribution.comm
            myrank = 0
            nproc = 1
            if comm is not None:
                myrank = comm.rank
                nproc = comm.size

            check = invnpp.duplicate()

            rcond = PixelData(invnpp.distribution, np.float64, n_value=1)

            # Invert twice
            covariance_invert(
                invnpp, threshold, rcond=rcond, use_alltoallv=(stype == "alltoallv")
            )
            covariance_invert(invnpp, threshold, use_alltoallv=(stype == "alltoallv"))

            failed = False
            for sm in range(invnpp.distribution.n_local_submap):
                good = np.where(rcond.data[sm] > threshold)[0]
                if not np.allclose(invnpp.data[sm, good, :], check.data[sm, good, :]):
                    failed = True
            if comm is not None:
                failed = comm.allreduce(failed, op=MPI.LOR)
            self.assertFalse(failed)

    def test_multiply(self):
        threshold = 1.0e-6

        for stype in ["allreduce", "alltoallv"]:
            # Build an inverse
            invnpp = self.create_invnpp(stype)
            comm = invnpp.distribution.comm
            myrank = 0
            nproc = 1
            if comm is not None:
                myrank = comm.rank
                nproc = comm.size

            rcond = PixelData(invnpp.distribution, np.float64, n_value=1)

            # Get the covariance
            npp = invnpp.duplicate()
            covariance_invert(
                npp, threshold, rcond=rcond, use_alltoallv=(stype == "alltoallv")
            )

            # Multiply the two
            covariance_multiply(npp, invnpp, use_alltoallv=(stype == "alltoallv"))

            failed = False
            for sm in range(invnpp.distribution.n_local_submap):
                for spix in range(npp.distribution.n_pix_submap):
                    if npp.data[sm, spix, 0] == 0:
                        continue
                    for elem, chk in zip(range(6), [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]):
                        if not np.allclose(
                            npp.data[sm, spix, elem], chk, rtol=1e-5, atol=1e-8
                        ):
                            failed = True
                            print(f"{npp.data[sm, spix, elem]} differs from {chk}")
            if comm is not None:
                failed = comm.allreduce(failed, op=MPI.LOR)
            self.assertFalse(failed)

    def test_apply(self):
        threshold = 1.0e-6

        for stype in ["allreduce", "alltoallv"]:
            # Build an inverse
            invnpp = self.create_invnpp(stype)
            comm = invnpp.distribution.comm
            myrank = 0
            nproc = 1
            if comm is not None:
                myrank = comm.rank
                nproc = comm.size

            # Get the covariance
            npp = invnpp.duplicate()
            covariance_invert(npp, threshold, use_alltoallv=(stype == "alltoallv"))

            # Random signal
            sig = PixelData(npp.distribution, np.float64, n_value=3)
            sig.raw[:] = np.random.normal(size=len(sig.raw))
            sig.sync_allreduce()

            check = sig.duplicate()

            # Apply inverse and then covariance and check that we recover the original.
            covariance_apply(invnpp, sig, use_alltoallv=(stype == "alltoallv"))
            covariance_apply(npp, sig, use_alltoallv=(stype == "alltoallv"))

            failed = False
            for sm in range(npp.distribution.n_local_submap):
                for spix in range(npp.distribution.n_pix_submap):
                    if npp.data[sm, spix, 0] == 0:
                        continue
                    if not np.allclose(
                        sig.data[sm, spix],
                        check.data[sm, spix],
                        rtol=1e-5,
                        atol=1e-8,
                    ):
                        failed = True
            if comm is not None:
                failed = comm.allreduce(failed, op=MPI.LOR)
            self.assertFalse(failed)

            # manually check the multiplication
            sig = check.duplicate()
            prod = check.duplicate()

            covariance_apply(invnpp, sig, use_alltoallv=(stype == "alltoallv"))

            # dbg_file = os.path.join(self.outdir, "sig_{}_{}.txt".format(myrank, stype))
            # with open(dbg_file, "w") as f:
            #     for sm in range(sig.distribution.n_local_submap):
            #         for spix in range(sig.distribution.n_pix_submap):
            #             line = "{:02d} {:02d} {:05d} :".format(
            #                 sm, sig.distribution.local_submaps[sm], spix
            #             )
            #             for elem in sig.data[sm, spix]:
            #                 line += " {:0.3e}".format(elem)
            #             line += "\n"
            #             f.write(line)

            temp = np.empty(3)
            for sm in range(invnpp.distribution.n_local_submap):
                for spix in range(invnpp.distribution.n_pix_submap):
                    # if invnpp.data[sm, spix, 0] == 0:
                    #     continue
                    temp[0] = 0
                    temp[0] += invnpp.data[sm, spix, 0] * prod.data[sm, spix, 0]
                    temp[0] += invnpp.data[sm, spix, 1] * prod.data[sm, spix, 1]
                    temp[0] += invnpp.data[sm, spix, 2] * prod.data[sm, spix, 2]
                    temp[1] = 0
                    temp[1] += invnpp.data[sm, spix, 1] * prod.data[sm, spix, 0]
                    temp[1] += invnpp.data[sm, spix, 3] * prod.data[sm, spix, 1]
                    temp[1] += invnpp.data[sm, spix, 4] * prod.data[sm, spix, 2]
                    temp[2] = 0
                    temp[2] += invnpp.data[sm, spix, 2] * prod.data[sm, spix, 0]
                    temp[2] += invnpp.data[sm, spix, 4] * prod.data[sm, spix, 1]
                    temp[2] += invnpp.data[sm, spix, 5] * prod.data[sm, spix, 2]
                    prod.data[sm, spix, :] = temp

            failed = False

            for sm in range(invnpp.distribution.n_local_submap):
                for spix in range(invnpp.distribution.n_pix_submap):
                    if not np.allclose(
                        sig.data[sm, spix],
                        prod.data[sm, spix],
                        rtol=1e-5,
                        atol=1e-8,
                    ):
                        failed = True
            if comm is not None:
                failed = comm.allreduce(failed, op=MPI.LOR)
            self.assertFalse(failed)
