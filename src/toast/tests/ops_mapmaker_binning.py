# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt

from astropy import units as u

import healpy as hp

from .mpi import MPITestCase

from ..noise import Noise

from ..vis import set_matplotlib_backend

from .. import ops as ops

from ..pixels import PixelDistribution, PixelData

from ..pixels_io import write_healpix_fits

from ._helpers import create_outdir, create_satellite_data


class MapmakerBinningTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)

    def test_binned(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise
        sim_noise = ops.SimNoise(noise_model="noise_model", out="noise")
        sim_noise.apply(data)

        # Pointing operator
        pointing = ops.PointingHealpix(nside=64, mode="IQU", hwp_angle="hwp_angle")

        # Build the covariance and hits
        cov_and_hits = ops.CovarianceAndHits(
            pixel_dist="pixel_dist", pointing=pointing, noise_model="noise_model"
        )
        cov_and_hits.apply(data)

        # Set up binned map

        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data="noise",
            pointing=pointing,
            noise_model="noise_model",
        )
        binner.apply(data)

        binmap = data[binner.binned]

        # # Manual check
        #
        # check_invnpp = PixelData(data["pixel_dist"], np.float64, n_value=6)
        # check_invnpp_corr = PixelData(data["pixel_dist"], np.float64, n_value=6)
        #
        # for ob in data.obs:
        #     noise = ob["noise_model"]
        #     noise_corr = ob["noise_model_corr"]
        #
        #     for det in ob.local_detectors:
        #         detweight = noise.detector_weight(det)
        #         detweight_corr = noise_corr.detector_weight(det)
        #
        #         wt = ob.detdata["weights"][det]
        #         local_sm, local_pix = data["pixel_dist"].global_pixel_to_submap(
        #             ob.detdata["pixels"][det]
        #         )
        #         for i in range(ob.n_local_samples):
        #             if local_pix[i] < 0:
        #                 continue
        #             off = 0
        #             for j in range(3):
        #                 for k in range(j, 3):
        #                     check_invnpp.data[local_sm[i], local_pix[i], off] += (
        #                         detweight * wt[i, j] * wt[i, k]
        #                     )
        #                     check_invnpp_corr.data[local_sm[i], local_pix[i], off] += (
        #                         detweight_corr * wt[i, j] * wt[i, k]
        #                     )
        #                     off += 1
        #
        # check_invnpp.sync_allreduce()
        # check_invnpp_corr.sync_allreduce()
        #
        # for sm in range(invnpp.distribution.n_local_submap):
        #     for px in range(invnpp.distribution.n_pix_submap):
        #         if invnpp.data[sm, px, 0] != 0:
        #             nt.assert_almost_equal(
        #                 invnpp.data[sm, px], check_invnpp.data[sm, px]
        #             )
        #         if invnpp_corr.data[sm, px, 0] != 0:
        #             nt.assert_almost_equal(
        #                 invnpp_corr.data[sm, px], check_invnpp_corr.data[sm, px]
        #             )
        del data
        return

    def test_compare_madam(self):
        if not ops.Madam.available:
            print("libmadam not available, skipping binned map comparison")
            return

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise
        sim_noise = ops.SimNoise(noise_model="noise_model", out="noise")
        sim_noise.apply(data)

        # Pointing operator
        pointing = ops.PointingHealpix(
            nside=64, mode="IQU", nest=True, hwp_angle="hwp_angle"
        )

        # Build the covariance and hits
        cov_and_hits = ops.CovarianceAndHits(
            pixel_dist="pixel_dist", pointing=pointing, noise_model="noise_model"
        )
        cov_and_hits.apply(data)

        # Set up binned map

        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data="noise",
            pointing=pointing,
            noise_model="noise_model",
        )
        binner.apply(data)

        # Write binned map to disk so we can load the whole thing on one process.

        toast_hit_path = os.path.join(self.outdir, "toast_hits.fits")
        toast_bin_path = os.path.join(self.outdir, "toast_bin.fits")
        toast_cov_path = os.path.join(self.outdir, "toast_cov.fits")
        write_healpix_fits(data[binner.binned], toast_bin_path, nest=True)
        write_healpix_fits(data[cov_and_hits.hits], toast_hit_path, nest=True)
        write_healpix_fits(data[cov_and_hits.covariance], toast_cov_path, nest=True)

        # Now run Madam on the same data and compare

        sample_rate = data.obs[0]["noise_model"].rate(data.obs[0].local_detectors[0])

        pars = {}
        pars["kfirst"] = "F"
        pars["iter_max"] = 10
        pars["base_first"] = 1.0
        pars["fsample"] = sample_rate
        pars["nside_map"] = pointing.nside
        pars["nside_cross"] = pointing.nside
        pars["nside_submap"] = min(8, pointing.nside)
        pars["write_map"] = "F"
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["kfilter"] = "F"
        pars["path_output"] = self.outdir
        pars["info"] = 0

        madam = ops.Madam(
            params=pars,
            det_data="noise",
            pixels=pointing.pixels,
            weights=pointing.weights,
            pixels_nested=pointing.nest,
            noise_model="noise_model",
        )

        # Generate persistent pointing
        pointing.apply(data)

        # Run Madam
        madam.apply(data)

        madam_hit_path = os.path.join(self.outdir, "madam_hmap.fits")
        madam_bin_path = os.path.join(self.outdir, "madam_bmap.fits")

        if data.comm.world_rank == 0:
            set_matplotlib_backend()
            import matplotlib.pyplot as plt

            # Compare hit maps

            toast_hits = hp.read_map(toast_hit_path, field=None, nest=True)
            madam_hits = hp.read_map(madam_hit_path, field=None, nest=True)
            diff_hits = toast_hits - madam_hits

            outfile = os.path.join(self.outdir, "madam_hits.png")
            hp.mollview(madam_hits, xsize=1600, nest=True)
            plt.savefig(outfile)
            plt.close()
            outfile = os.path.join(self.outdir, "toast_hits.png")
            hp.mollview(toast_hits, xsize=1600, nest=True)
            plt.savefig(outfile)
            plt.close()
            outfile = os.path.join(self.outdir, "diff_hits.png")
            hp.mollview(diff_hits, xsize=1600, nest=True)
            plt.savefig(outfile)
            plt.close()

            # Compare binned maps

            toast_bin = hp.read_map(toast_bin_path, field=None, nest=True)
            madam_bin = hp.read_map(madam_bin_path, field=None, nest=True)
            # Set madam unhit pixels to zero
            for stokes, ststr in zip(range(3), ["I", "Q", "U"]):
                mask = hp.mask_bad(madam_bin[stokes])
                madam_bin[stokes][mask] = 0.0
                diff_map = toast_bin[stokes] - madam_bin[stokes]
                print("diff map {} has rms {}".format(ststr, np.std(diff_map)))
                outfile = os.path.join(self.outdir, "madam_bin_{}.png".format(ststr))
                hp.mollview(madam_bin[stokes], xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()
                outfile = os.path.join(self.outdir, "toast_bin_{}.png".format(ststr))
                hp.mollview(toast_bin[stokes], xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()
                outfile = os.path.join(self.outdir, "diff_bin_{}.png".format(ststr))
                hp.mollview(diff_map, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                nt.assert_almost_equal(toast_bin[stokes], madam_bin[stokes], decimal=6)

        del data
        return
