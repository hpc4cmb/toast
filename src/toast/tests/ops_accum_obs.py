# Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
from astropy import units as u
import healpy as hp

from .. import ops as ops
from ..footprint import footprint_distribution
from ..mpi import MPI
from ..observation import default_values as defaults
from .helpers import (
    close_data,
    create_ground_data,
    create_outdir,
    create_fake_healpix_map,
)
from .mpi import MPITestCase


class AccumObsTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        np.random.seed(123456)

    def create_observations(self):
        """Generate observations with boresight pointing and no det data."""
        # Slightly slower than 1 Hz
        hwp_rpm = 59.0

        sample_rate = 60 * u.Hz

        # Create a fake ground observation set for testing.
        # Detector data is not created.
        data = create_ground_data(
            self.comm,
            sample_rate=sample_rate,
            hwp_rpm=hwp_rpm,
            no_det_data=True,
        )
        return data

    def create_input_sky(self, data, pixel_dist, map_key, test_dir):
        """Create the input sky map."""
        if map_key in data:
            msg = f"Generated map '{map_key}' already exists in data"
            raise RuntimeError(msg)
        npix = data[pixel_dist].n_pix
        nside = hp.npix2nside(npix)
        lmax = 3 * nside
        fwhm = 10 * u.arcmin
        data[map_key] = create_fake_healpix_map(
            os.path.join(test_dir, "input_sky.fits"),
            data[pixel_dist],
            fwhm=fwhm,
            lmax=lmax,
        )

    def create_sim_pipeline(self, data, map_key):
        # Create an uncorrelated noise model from focalplane detector properties.
        # We do this outside of the pipeline since it does not generate any
        # timestreams.
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        npix = data[map_key].distribution.n_pix
        nside = hp.npix2nside(npix)

        # Now build up the operators to use in the simulation pipeline
        operators = list()

        # Generic pointing matrix
        detpointing = ops.PointingDetectorSimple(
            shared_flag_mask=0,
            quats="sim_quats",
        )
        pixels = ops.PixelsHealpix(
            nside=nside,
            detector_pointing=detpointing,
            pixels="sim_pixels",
        )
        operators.append(pixels)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
            weights="sim_weights",
        )
        operators.append(weights)

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data=defaults.det_data,
            pixels=pixels.pixels,
            weights=weights.weights,
            map_key=map_key,
        )
        operators.append(scanner)

        # Simulate noise from the default model
        sim_noise = ops.SimNoise(
            noise_model=default_model.noise_model, out=defaults.det_data
        )
        operators.append(sim_noise)

        # The pipeline to run on each obs
        pipe = ops.Pipeline(
            detector_sets=["ALL"],
            operators=operators,
        )
        return pipe

    def add_sim_loader(self, data, pipe, pixel_dist, test_dir):
        """Go through all observations and add a loader."""

        # Add to the observations
        for ob in data.obs:
            ob.loader = ops.PipelineLoader(pipeline=pipe)

    def test_accum_sim(self):
        testdir = os.path.join(self.outdir, "accum_sim")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir)

        data = self.create_observations()

        # Generate a full-sky footprint for testing
        nside = 256
        pixel_dist = "pixel_dist"
        data[pixel_dist] = footprint_distribution(
            healpix_nside=nside,
            healpix_nside_submap=16,
        )

        # Create a
        map_key = "input_sky"

        # First simulate the sky map and store it in the data
        self.create_input_sky(data, pixel_dist, map_key, testdir)

        sim_pipe = self.create_sim_pipeline(data, map_key)

        self.add_sim_loader(data, sim_pipe, pixel_dist, testdir)

        # Operator to accumulate our map-domain products with the loader
        detpointing = ops.PointingDetectorSimple(
            shared_flag_mask=0,
            quats="quats",
        )
        pixels = ops.PixelsHealpix(
            nside=nside,
            detector_pointing=detpointing,
            pixels="pixels",
        )
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
            weights="weights",
        )
        accum_obs = ops.AccumulateObservation(
            cache_dir=os.path.join(testdir, "cache"),
            pixel_dist=pixel_dist,
            inverse_covariance="invcov",
            hits="hits",
            zmap="zmap",
            rcond="rcond",
            covariance="cov",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model="noise_model",
            obs_pointing=True,
        )
        accum_obs.load_apply(data)

        # Run the steps manually

        # Clear the loaders
        _ = data.save_loaders()

        # Delete and re-simulate the data
        ops.Delete(detdata=[defaults.det_data]).apply(data)
        sim_pipe.apply(data)

        # Run all the operators with full pointing
        pixels.apply(data)
        weights.apply(data)

        # Hit map operator
        build_hits = ops.BuildHitMap(
            pixel_dist=pixel_dist,
            hits="check_hits",
            view=pixels.view,
            pixels=pixels.pixels,
        )
        build_hits.apply(data)

        # Inverse covariance.
        build_invcov = ops.BuildInverseCovariance(
            pixel_dist=pixel_dist,
            inverse_covariance="check_invcov",
            view=pixels.view,
            pixels=pixels.pixels,
            weights=weights.weights,
            noise_model="noise_model",
        )
        build_invcov.apply(data)

        # Noise weighted map
        build_zmap = ops.BuildNoiseWeighted(
            pixel_dist=pixel_dist,
            zmap="check_zmap",
            view=pixels.view,
            pixels=pixels.pixels,
            weights=weights.weights,
            noise_model="noise_model",
        )
        build_zmap.apply(data)

        # Verify that the accumulated products agree
        for prod in ["hits", "invcov", "zmap"]:
            original = data[prod]
            check = data[f"check_{prod}"]
            failed = False

            if original.distribution != check.distribution:
                failed = True

            if not failed:
                dist = check.distribution
                for sm in range(dist.n_local_submap):
                    for px in range(dist.n_pix_submap):
                        if check.data[sm, px, 0] != 0:
                            if not np.allclose(
                                check.data[sm, px],
                                original.data[sm, px],
                            ):
                                failed = True
            if data.comm.comm_world is not None:
                failed = data.comm.comm_world.allreduce(failed, op=MPI.LOR)
            self.assertFalse(failed)

        close_data(data)

    def test_accum_cache(self):
        testdir = os.path.join(self.outdir, "accum_cache")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir)

        data = self.create_observations()

        # Generate a full-sky footprint for testing
        nside = 256
        pixel_dist = "pixel_dist"
        data[pixel_dist] = footprint_distribution(
            healpix_nside=nside,
            healpix_nside_submap=16,
        )

        # Create a
        map_key = "input_sky"

        # First simulate the sky map and store it in the data
        self.create_input_sky(data, pixel_dist, map_key, testdir)

        sim_pipe = self.create_sim_pipeline(data, map_key)

        self.add_sim_loader(data, sim_pipe, pixel_dist, testdir)

        # Operator to accumulate our map-domain products with the loader
        detpointing = ops.PointingDetectorSimple(
            shared_flag_mask=0,
            quats="quats",
        )
        pixels = ops.PixelsHealpix(
            nside=nside,
            detector_pointing=detpointing,
            pixels="pixels",
        )
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
            weights="weights",
        )
        accum_obs = ops.AccumulateObservation(
            cache_dir=os.path.join(testdir, "cache"),
            cache_detdata=True,
            pixel_dist=pixel_dist,
            inverse_covariance="invcov",
            hits="hits",
            zmap="zmap",
            rcond="rcond",
            covariance="cov",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model="noise_model",
            obs_pointing=True,
        )

        # This will run the sim pipeline to produce data each observation,
        # then cache the TOD and insert a loader for future calls.
        accum_obs.load_apply(data)

        # Run the steps manually.  This will trigger the loading of detector
        # data from the cache.

        # Run all the operators with full pointing.  We pack these into a pipeline
        # so that we can use load_apply and trigger the loading once.
        check_pipe_ops = list()
        check_pipe_ops.append(pixels)
        check_pipe_ops.append(weights)

        # Hit map operator
        build_hits = ops.BuildHitMap(
            pixel_dist=pixel_dist,
            hits="check_hits",
            view=pixels.view,
            pixels=pixels.pixels,
        )
        check_pipe_ops.append(build_hits)

        # Inverse covariance.
        build_invcov = ops.BuildInverseCovariance(
            pixel_dist=pixel_dist,
            inverse_covariance="check_invcov",
            view=pixels.view,
            pixels=pixels.pixels,
            weights=weights.weights,
            noise_model="noise_model",
        )
        check_pipe_ops.append(build_invcov)

        # Noise weighted map
        build_zmap = ops.BuildNoiseWeighted(
            pixel_dist=pixel_dist,
            zmap="check_zmap",
            view=pixels.view,
            pixels=pixels.pixels,
            weights=weights.weights,
            noise_model="noise_model",
        )
        check_pipe_ops.append(build_zmap)

        check_pipe = ops.Pipeline(operators=check_pipe_ops)
        check_pipe.load_apply(data)

        # Verify that the accumulated products agree
        for prod in ["hits", "invcov", "zmap"]:
            original = data[prod]
            check = data[f"check_{prod}"]
            failed = False

            if original.distribution != check.distribution:
                failed = True

            if not failed:
                dist = check.distribution
                for sm in range(dist.n_local_submap):
                    for px in range(dist.n_pix_submap):
                        if check.data[sm, px, 0] != 0:
                            if not np.allclose(
                                check.data[sm, px],
                                original.data[sm, px],
                                rtol=1e-5,
                                atol=1e-5,
                            ):
                                msg = f"{prod} FAIL [{sm},{px}]: "
                                msg += f"{original.data[sm,px]} "
                                msg += f"!= {check.data[sm,px]}"
                                print(msg, flush=True)
                                failed = True
            if data.comm.comm_world is not None:
                failed = data.comm.comm_world.allreduce(failed, op=MPI.LOR)
            self.assertFalse(failed)

        close_data(data)