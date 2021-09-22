# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
import numpy.testing as nt
import scipy.sparse
from astropy import units as u
from astropy.table import Column

from .. import ops as ops
from ..noise import Noise
from ..observation import default_names as obs_names
from ..pixels import PixelData, PixelDistribution
from ..pixels_io import write_healpix_fits
from ..vis import set_matplotlib_backend
from ._helpers import create_fake_sky, create_ground_data, create_outdir, fake_flags
from .mpi import MPITestCase


class FilterBinTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.nside = 128

    def test_filterbin(self):

        # Create a fake ground data set for testing
        data = create_ground_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=self.nside,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
            # view="scanning",
        )
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=obs_names.hwp_angle,
            detector_pointing=detpointing,
        )

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise from this model
        sim_noise = ops.SimNoise(noise_model="noise_model", out=obs_names.det_data)
        sim_noise.apply(data)

        # Make fake flags
        # fake_flags(data)

        binning = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance="covariance",
            det_data=sim_noise.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="allreduce",
            shared_flags=obs_names.shared_flags,
            shared_flag_mask=1,
            det_flags=obs_names.det_flags,
            det_flags_mask=255,
        )

        filterbin = ops.FilterBin(
            name="filterbin",
            det_data=obs_names.det_data,
            det_flags=obs_names.det_flags,
            det_flags_mask=255,
            shared_flags=obs_names.shared_flags,
            shared_flag_mask=1,
            binning=binning,
            ground_filter_order=5,
            split_ground_template=True,
            poly_filter_order=2,
            output_dir=self.outdir,
        )
        filterbin.apply(data)

        # Confirm that the filtered map has less noise than the unfiltered map

        if data.comm.world_rank == 0:
            fname_binned = os.path.join(
                self.outdir, f"{filterbin.name}_unfiltered_map.fits"
            )
            fname_filtered = os.path.join(
                self.outdir, f"{filterbin.name}_filtered_map.fits"
            )

            binned = hp.read_map(fname_binned, None)
            filtered = hp.read_map(fname_filtered, None)

            good = binned != 0
            rms1 = np.std(binned[good])
            rms2 = np.std(filtered[good])

            assert rms2 < 0.9 * rms1

        return

    def test_filterbin_obsmatrix(self):

        # Create a fake ground data set for testing
        data = create_ground_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=self.nside,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
            # view="scanning",
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=obs_names.hwp_angle,
            detector_pointing=detpointing,
        )
        weights.apply(data)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Make fake flags
        # fake_flags(data)

        if data.comm.world_rank == 0:
            lmax = 3 * self.nside
            cls = np.ones(4 * (lmax + 1)).reshape(4, -1)
            fwhm = np.radians(10)
            input_map = hp.synfast(cls, self.nside, lmax=lmax, fwhm=fwhm, verbose=False)
            if pixels.nest:
                input_map = hp.reorder(input_map, r2n=True)
            input_map_file = os.path.join(self.outdir, "input_map.fits")
            hp.write_map(input_map_file, input_map, nest=pixels.nest)

        # Scan map into timestreams
        scan_hpix = ops.ScanHealpix(
            file=input_map_file,
            det_data=obs_names.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
        )
        scan_hpix.apply(data)

        # Configure and apply the filterbin operator
        binning = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance="covariance",
            det_data=obs_names.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="allreduce",
            shared_flags=obs_names.shared_flags,
            shared_flag_mask=1,
            det_flags=obs_names.det_flags,
            det_flags_mask=255,
        )

        filterbin = ops.FilterBin(
            name="filterbin",
            det_data=obs_names.det_data,
            det_flags=obs_names.det_flags,
            det_flags_mask=255,
            shared_flags=obs_names.shared_flags,
            shared_flag_mask=1,
            binning=binning,
            ground_filter_order=5,
            split_ground_template=True,
            poly_filter_order=2,
            output_dir=self.outdir,
            write_obs_matrix=True,
        )
        filterbin.apply(data)

        if data.comm.world_rank == 0:
            import matplotlib.pyplot as plt

            rot = [42, -42]
            reso = 4
            fig = plt.figure(figsize=[18, 12])
            cmap = "coolwarm"
            nest = pixels.nest

            rootname = os.path.join(self.outdir, f"{filterbin.name}_obs_matrix")
            fname_matrix = ops.combine_observation_matrix(rootname)

            obs_matrix = scipy.sparse.load_npz(fname_matrix)

            input_map = hp.read_map(input_map_file, None, nest=nest)

            fname_filtered = os.path.join(
                self.outdir, f"{filterbin.name}_filtered_map.fits"
            )
            filtered = hp.read_map(fname_filtered, None, nest=nest)

            test_map = obs_matrix.dot(input_map.ravel()).reshape([3, -1])

            good = filtered[0] != 0

            nrow, ncol = 2, 2
            args = {"rot": rot, "reso": reso, "cmap": cmap, "nest": nest}

            diffmap = test_map - filtered
            diffmap[filtered == 0] = hp.UNSEEN
            filtered[filtered == 0] = hp.UNSEEN
            test_map[test_map == 0] = hp.UNSEEN
            hp.gnomview(filtered[0], sub=[nrow, ncol, 1], title="Filtered map", **args)
            hp.gnomview(
                test_map[0], sub=[nrow, ncol, 2], title="Input x obs.matrix", **args
            )
            hp.gnomview(input_map[0], sub=[nrow, ncol, 3], title="Input map", **args)
            hp.gnomview(diffmap[0], sub=[nrow, ncol, 4], title="Difference", **args)
            fname = os.path.join(self.outdir, "obs_matrix_test.png")
            fig.savefig(fname)

            for i in range(3):
                rms1 = np.std(filtered[i][good])
                rms2 = np.std((filtered - test_map)[i][good])
                assert rms2 < 1e-6 * rms1

        return
