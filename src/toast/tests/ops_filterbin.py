# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
import numpy.testing as nt
from astropy import units as u
from astropy.table import Column

from .. import ops as ops
from ..noise import Noise
from ..observation import default_names as obs_names
from ..pixels import PixelData, PixelDistribution
from ..vis import set_matplotlib_backend
from ._helpers import create_ground_data, create_outdir, fake_flags
from .mpi import MPITestCase


class FilterBinTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_filterbin(self):

        # Create a fake ground data set for testing
        data = create_ground_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
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
            nside=64,
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
            write_obs_matrix=True,
        )
        filterbin.apply(data)

        return
