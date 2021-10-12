# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import healpy as hp
from .mpi import MPITestCase

from ..vis import set_matplotlib_backend

from .. import ops as ops
from .. import rng

from ..observation import default_values as defaults

from ..pixels import PixelDistribution, PixelData

from ..pixels_io import write_healpix_fits

from ..covariance import covariance_apply
from ._helpers import (
    create_outdir,
    create_satellite_data,
    create_satellite_data_big,
    create_fake_sky,
)


class SimGainTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_linear_drift(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(
            self.comm,
        )

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)
        # Simulate noise using this model
        key = "my_signal"
        sim_noise = ops.SimNoise(det_data=key)
        sim_noise.apply(data)
        old_data = []
        for obs in data.obs:
            old = {}
            for det in obs.local_detectors:
                ref = obs.detdata[key][det]
                old[det] = (ref).copy()
            old_data.append(old)

        drifter = ops.GainDrifter(det_data=key, drift_mode="linear_drift")
        drifter.apply(data)
        for obs, old in zip(data.obs, old_data):
            telescope = obs.telescope.uid
            focalplane = obs.telescope.focalplane
            obsindx = obs.uid

            key1 = (
                drifter.realization * 4294967296 + telescope * 65536 + drifter.component
            )
            counter2 = 0
            for det in obs.local_detectors:
                detindx = focalplane[det]["uid"]
                key2 = obsindx
                counter1 = detindx
                rngdata = rng.random(
                    1,
                    sampler="gaussian",
                    key=(key1, key2),
                    counter=(counter1, counter2),
                )
                gf2 = 1 + rngdata[0] * drifter.sigma_drift
                gf1 = (obs.detdata[key][det] / old[det])[-1]
                # assert whether the two values gf2 and gf1  are the same
                # within 1sigma of the distribution
                np.testing.assert_almost_equal(
                    gf1, gf2, decimal=np.log10(drifter.sigma_drift) - 1
                )

    def test_thermal_drift(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data_big(self.comm, pixel_per_process=7)
        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)
        # make a simple pointing matrix
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=16,
            nest=False,
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeights(
            mode="I",
            detector_pointing=detpointing,
        )
        # Generate timestreams
        key = defaults.det_data
        sim_dipole = ops.SimDipole(det_data=key, mode="solar", coord="G")
        sim_dipole.apply(data)

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

        binner1 = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data=key,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner1.apply(data)
        map1_path = os.path.join(self.outdir, "toast_bin1_drift.fits")
        write_healpix_fits(data[binner1.binned], map1_path, nest=False)

        # inject gain drift  w/ common mode

        drifter = ops.GainDrifter(
            det_data=key,
            drift_mode="thermal_drift",
            detector_mismatch=0.7,
        )
        drifter.apply(data)

        binner1.apply(data)
        map2_path = os.path.join(self.outdir, "toast_bin2_drift.fits")
        write_healpix_fits(data[binner1.binned], map2_path, nest=False)

        if data.comm.world_rank == 0:
            # import pdb; pdb.set_trace()
            oldmap = hp.read_map(map1_path, field=None, nest=False)
            newmap = hp.read_map(map2_path, field=None, nest=False)
            # import pylab as pl
            # hp.mollview((oldmap-newmap)/oldmap ) ;pl.show()
            mask = oldmap != 0
            np.testing.assert_almost_equal(
                oldmap[mask], newmap[mask], decimal=np.log10(drifter.sigma_drift)
            )
            rel_res = (oldmap[mask] - newmap[mask]) / oldmap[mask]
            dT = (drifter.thermal_fluctuation_amplitude * drifter.sigma_drift).to(
                drifter.focalplane_Tbath.unit
            ) / drifter.focalplane_Tbath

            assert np.log10(rel_res.std()) <= np.log10(dT)

    def test_slow_drift(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data_big(self.comm, pixel_per_process=7)
        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)
        # make a simple pointing matrix
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=16,
            nest=False,
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeights(
            mode="I",
            detector_pointing=detpointing,
        )
        # Generate timestreams
        key = defaults.det_data
        sim_dipole = ops.SimDipole(det_data=key, mode="solar", coord="G")
        sim_dipole.apply(data)

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

        binner1 = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data=key,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner1.apply(data)
        map1_path = os.path.join(self.outdir, "toast_bin1_drift.fits")
        write_healpix_fits(data[binner1.binned], map1_path, nest=False)

        # inject gain drift  w/ common mode

        drifter = ops.GainDrifter(
            det_data=key,
            drift_mode="slow_drift",
        )
        drifter.apply(data)

        binner1.apply(data)
        map2_path = os.path.join(self.outdir, "toast_bin2_drift.fits")
        write_healpix_fits(data[binner1.binned], map2_path, nest=False)
        if data.comm.world_rank == 0:
            oldmap = hp.read_map(map1_path, field=None, nest=False)
            newmap = hp.read_map(map2_path, field=None, nest=False)
            mask = oldmap != 0
            np.testing.assert_almost_equal(
                oldmap[mask], newmap[mask], decimal=np.log10(drifter.sigma_drift)
            )
            rel_res = (oldmap[mask] - newmap[mask]) / oldmap[mask]
            assert np.log10(rel_res.std()) <= np.log10(drifter.sigma_drift)

    def test_slow_drift_commonmode(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data_big(self.comm, pixel_per_process=7)
        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)
        # make a simple pointing matrix
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=16,
            nest=False,
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeights(
            mode="I",
            detector_pointing=detpointing,
        )
        # Generate timestreams
        key = defaults.det_data
        sim_dipole = ops.SimDipole(det_data=key, mode="solar", coord="G")
        sim_dipole.apply(data)

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

        binner1 = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data=key,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner1.apply(data)
        map1_path = os.path.join(self.outdir, "toast_bin1_drift.fits")
        write_healpix_fits(data[binner1.binned], map1_path, nest=False)

        # inject gain drift  w/ common mode

        drifter = ops.GainDrifter(
            det_data=key, drift_mode="slow_drift", detector_mismatch=0.0
        )
        drifter.apply(data)

        binner1.apply(data)
        map2_path = os.path.join(self.outdir, "toast_bin2_drift.fits")
        write_healpix_fits(data[binner1.binned], map2_path, nest=False)

        if data.comm.world_rank == 0:
            oldmap = hp.read_map(map1_path, field=None, nest=False)
            newmap = hp.read_map(map2_path, field=None, nest=False)
            mask = oldmap != 0
            np.testing.assert_almost_equal(
                oldmap[mask], newmap[mask], decimal=np.log10(drifter.sigma_drift)
            )
            rel_res = (oldmap[mask] - newmap[mask]) / oldmap[mask]
            assert np.log10(rel_res.std()) <= np.log10(drifter.sigma_drift)

    def test_responsivity_function(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data_big(self.comm)
        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)
        # Generate timestreams
        key = defaults.det_data
        sim_dipole = ops.SimDipole(det_data=key, mode="solar", coord="G")
        sim_dipole.apply(data)

        # inject gain drift
        responsivity = lambda x: -2 * x ** 3 + 5 * x ** 2 - 4 * x + 3
        drifter = ops.GainDrifter(
            det_data=key,
            drift_mode="thermal_drift",
            detector_mismatch=0.7,
            sigma_drift=1e-6,
            responsivity_function=responsivity,
        )
        drifter.apply(data)
