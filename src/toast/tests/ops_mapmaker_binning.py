# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops as ops
from ..covariance import covariance_apply
from ..mpi import MPI
from ..noise import Noise
from ..observation import default_values as defaults
from ..pixels import PixelData, PixelDistribution
from ..pixels_io_healpix import write_healpix_fits
from ..vis import set_matplotlib_backend
from ._helpers import close_data, create_outdir, create_satellite_data
from .mpi import MPITestCase


class MapmakerBinningTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_binned(self):
        np.random.seed(123456)

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise
        sim_noise = ops.SimNoise(noise_model="noise_model", det_data="noise")
        sim_noise.apply(data)

        # Pointing operator
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )

        binmap = dict()
        for stype in ["allreduce", "alltoallv"]:
            # Build the covariance and hits
            cov_and_hits = ops.CovarianceAndHits(
                pixel_dist="pixel_dist",
                pixel_pointing=pixels,
                stokes_weights=weights,
                noise_model="noise_model",
                covariance="cov_{}".format(stype),
                hits="hits_{}".format(stype),
                rcond="rcond_{}".format(stype),
                sync_type=stype,
            )
            cov_and_hits.apply(data)

            # Set up binned map

            binner = ops.BinMap(
                pixel_dist="pixel_dist",
                covariance=cov_and_hits.covariance,
                binned="binned_{}".format(stype),
                det_data=sim_noise.det_data,
                pixel_pointing=pixels,
                stokes_weights=weights,
                noise_model=default_model.noise_model,
                sync_type=stype,
            )
            binner.apply(data)

            binmap[stype] = data[binner.binned]

            toast_hit_path = os.path.join(
                self.outdir, "toast_hits_{}.fits".format(stype)
            )
            toast_bin_path = os.path.join(
                self.outdir, "toast_bin_{}.fits".format(stype)
            )
            toast_cov_path = os.path.join(
                self.outdir, "toast_cov_{}.fits".format(stype)
            )
            write_healpix_fits(data[binner.binned], toast_bin_path, nest=True)
            write_healpix_fits(data[cov_and_hits.hits], toast_hit_path, nest=True)
            write_healpix_fits(data[cov_and_hits.covariance], toast_cov_path, nest=True)

        # Manual check

        pixels.apply(data)
        weights.apply(data)

        noise_weight = ops.BuildNoiseWeighted(
            pixel_dist="pixel_dist",
            noise_model=default_model.noise_model,
            pixels=pixels.pixels,
            weights=weights.weights,
            det_data=sim_noise.det_data,
            zmap="zmap",
            sync_type="allreduce",
        )
        noise_weight.apply(data)

        covariance_apply(
            data[cov_and_hits.covariance], data["zmap"], use_alltoallv=False
        )

        for stype in ["allreduce", "alltoallv"]:
            bmap = binmap[stype]
            comm = bmap.distribution.comm
            failed = False
            for sm in range(bmap.distribution.n_local_submap):
                for px in range(bmap.distribution.n_pix_submap):
                    if not np.allclose(bmap.data[sm, px], data["zmap"].data[sm, px]):
                        failed = True
            if comm is not None:
                failed = comm.allreduce(failed, op=MPI.LOR)
            self.assertFalse(failed)

        close_data(data)

    def test_compare_madam(self):
        if not ops.madam.available():
            print("libmadam not available, skipping binned map comparison")
            return

        np.random.seed(123456)
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise
        sim_noise = ops.SimNoise(noise_model="noise_model", det_data=defaults.det_data)
        sim_noise.apply(data)

        # Pointing operator
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            nest=True,
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )

        # Build the covariance and hits
        cov_and_hits = ops.CovarianceAndHits(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            rcond_threshold=1.0e-6,
            sync_type="alltoallv",
        )
        cov_and_hits.apply(data)

        # Set up binned map

        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data=sim_noise.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
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
        pars["nside_map"] = pixels.nside
        pars["nside_cross"] = pixels.nside
        pars["nside_submap"] = min(8, pixels.nside)
        pars["pixlim_cross"] = 1.0e-6
        pars["pixlim_map"] = 1.0e-6
        pars["write_map"] = "F"
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["kfilter"] = "F"
        pars["path_output"] = self.outdir

        madam = ops.Madam(
            params=pars,
            det_data=defaults.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model="noise_model",
        )

        # Generate persistent pointing
        pixels.apply(data)
        weights.apply(data)

        # Run Madam
        madam.apply(data)

        madam_hit_path = os.path.join(self.outdir, "madam_hmap.fits")
        madam_bin_path = os.path.join(self.outdir, "madam_bmap.fits")

        fail = False

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

                if not np.allclose(toast_bin[stokes], madam_bin[stokes]):
                    fail = True

        if data.comm.comm_world is not None:
            fail = data.comm.comm_world.bcast(fail, root=0)

        self.assertFalse(fail)

        close_data(data)
