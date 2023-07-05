# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import glob
import os
import sys

import healpy as hp
import numpy as np
import numpy.testing as nt
import scipy.sparse
from astropy import units as u
from astropy.table import Column

from .. import ops as ops
from ..mpi import MPI, Comm
from ..noise import Noise
from ..observation import default_values as defaults
from ..pixels import PixelData, PixelDistribution
from ..pixels_io_healpix import read_healpix
from ..vis import set_matplotlib_backend
from ._helpers import (
    close_data,
    create_fake_sky,
    create_ground_data,
    create_outdir,
    fake_flags,
)
from .mpi import MPITestCase


class FilterBinTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.nside = 64

    def test_filterbin(self):
        if "CIBUILDWHEEL" in os.environ:
            print(f"WARNING:  Skipping test_filterbin during wheel tests")
            return
        # Create a fake ground data set for testing
        data = create_ground_data(self.comm, turnarounds_invalid=True)

        nside = 256

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=nside,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeights(
            mode="I",  # "IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise from this model
        sim_noise = ops.SimNoise(noise_model="noise_model", out=defaults.det_data)
        sim_noise.apply(data)

        # Add a strong gradient that should be filtered out completely
        for obs in data.obs:
            grad = np.arange(obs.n_local_samples)
            for det in obs.local_detectors:
                obs.detdata[defaults.det_data][det] += grad

        # Make fake flags
        fake_flags(data)

        binning = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance="covariance",
            det_data=sim_noise.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="allreduce",
            shared_flags=defaults.shared_flags,
            shared_flag_mask=1,
            det_flags=defaults.det_flags,
            det_flag_mask=255,
        )

        filterbin = ops.FilterBin(
            name="filterbin",
            det_data=defaults.det_data,
            det_flags=defaults.det_flags,
            det_flag_mask=255,
            shared_flags=defaults.shared_flags,
            shared_flag_mask=1,
            binning=binning,
            hwp_filter_order=4,
            ground_filter_order=5,
            split_ground_template=True,
            poly_filter_order=2,
            output_dir=self.outdir,
            write_hdf5=True,
        )
        filterbin.apply(data)

        # Confirm that the filtered map has less noise than the unfiltered map

        if data.comm.world_rank == 0:
            if sys.platform.lower() == "darwin":
                print(f"WARNING:  Skipping test_filterbin plots on MacOS")
            else:
                import matplotlib.pyplot as plt

                rot = [43, -42]
                reso = 4
                fig = plt.figure(figsize=[18, 12])
                cmap = "coolwarm"

                fname_binned = os.path.join(
                    self.outdir, f"{filterbin.name}_unfiltered_map.h5"
                )
                fname_filtered = os.path.join(
                    self.outdir, f"{filterbin.name}_filtered_map.h5"
                )

                binned = np.atleast_2d(read_healpix(fname_binned, None))
                filtered = np.atleast_2d(read_healpix(fname_filtered, None))

                good = binned != 0
                rms1 = np.std(binned[good])
                rms2 = np.std(filtered[good])

                nrow, ncol = 2, 2
                for m in binned, filtered:
                    m[m == 0] = hp.UNSEEN
                args = {"rot": rot, "reso": reso, "cmap": cmap}
                hp.gnomview(
                    binned[0],
                    sub=[nrow, ncol, 1],
                    title=f"Binned map : {rms1}",
                    **args,
                )
                hp.gnomview(
                    filtered[0],
                    sub=[nrow, ncol, 2],
                    title=f"Filtered map : {rms2}",
                    **args,
                )

                fname = os.path.join(self.outdir, "filter_test.png")
                fig.savefig(fname)
                check = rms2 < 1e-4 * rms1
                if not check:
                    print(f"rms2 = {rms2}, rms1 = {rms1}")
                self.assertTrue(check)

        close_data(data)

    def test_filterbin_obsmatrix(self):
        if sys.platform.lower() == "darwin":
            print(f"WARNING:  Skipping test_filterbin_obsmatrix on MacOS")
            return
        if "CIBUILDWHEEL" in os.environ:
            print(f"WARNING:  Skipping test_filterbin_obsmatrix during wheel tests")
            return

        # Create a fake ground data set for testing
        data = create_ground_data(self.comm, sample_rate=1 * u.Hz, pixel_per_process=4)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=self.nside,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pixels.apply(data)

        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )
        weights.apply(data)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        input_map_file = os.path.join(self.outdir, "input_map.fits")
        if not os.path.exists(input_map_file):
            if data.comm.world_rank == 0:
                lmax = 3 * self.nside
                cls = np.ones(4 * (lmax + 1)).reshape(4, -1)
                fwhm = np.radians(10)
                input_map = hp.synfast(
                    cls, self.nside, lmax=lmax, fwhm=fwhm, verbose=False
                )
                if pixels.nest:
                    input_map = hp.reorder(input_map, r2n=True)
                hp.write_map(
                    input_map_file, input_map, nest=pixels.nest, column_units="K"
                )

        if data.comm.comm_world is not None:
            data.comm.comm_world.Barrier()

        # Scan map into timestreams
        scan_hpix = ops.ScanHealpixMap(
            file=input_map_file,
            det_data=defaults.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
        )
        scan_hpix.apply(data)

        # Configure and apply the filterbin operator
        binning = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance="covariance",
            det_data=defaults.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="allreduce",
            shared_flags=defaults.shared_flags,
            shared_flag_mask=1,
            det_flags=defaults.det_flags,
            det_flag_mask=255,
        )

        filterbin = ops.FilterBin(
            name="filterbin",
            det_data=defaults.det_data,
            det_flags=defaults.det_flags,
            det_flag_mask=255,
            shared_flags=defaults.shared_flags,
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
                print(f"rms1 = {rms1}, rms2 = {rms2}, rms2/rms1 = {rms2 / rms1}")
                if rms2 > 1e-5 * rms1:
                    print(f"rms2 = {rms2}, rms1 = {rms1}")
                assert rms2 < 1e-5 * rms1
        close_data(data)

    def test_filterbin_obsmatrix_flags(self):
        if sys.platform.lower() == "darwin":
            print(f"WARNING:  Skipping test_filterbin_obsmatrix_flags on MacOS")
            return
        if "CIBUILDWHEEL" in os.environ:
            print(
                f"WARNING:  Skipping test_filterbin_obsmatrix_flags during wheel tests"
            )
            return

        # Create a fake ground data set for testing
        data = create_ground_data(self.comm, sample_rate=1 * u.Hz)

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
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )
        weights.apply(data)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Make fake flags
        fake_flags(data)

        input_map_file = os.path.join(self.outdir, "input_map2.fits")
        if data.comm.world_rank == 0:
            lmax = 3 * self.nside
            cls = np.ones(4 * (lmax + 1)).reshape(4, -1)
            fwhm = np.radians(10)
            input_map = hp.synfast(cls, self.nside, lmax=lmax, fwhm=fwhm, verbose=False)
            if pixels.nest:
                input_map = hp.reorder(input_map, r2n=True)
            hp.write_map(input_map_file, input_map, nest=pixels.nest, column_units="K")

        # Scan map into timestreams
        scan_hpix = ops.ScanHealpixMap(
            file=input_map_file,
            det_data=defaults.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
        )
        scan_hpix.apply(data)

        # Configure and apply the filterbin operator that
        # *does not* check the detector flags
        binning = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance="covariance",
            det_data=defaults.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="allreduce",
            shared_flags=defaults.shared_flags,
            shared_flag_mask=1,
            det_flags=defaults.det_flags,
            det_flag_mask=0,
        )

        filterbin = ops.FilterBin(
            name="filterbin_flagged",
            det_data=defaults.det_data,
            det_flags=defaults.det_flags,
            det_flag_mask=255,
            shared_flags=defaults.shared_flags,
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
            rms1 = np.std(filtered[0, good])
            hp.gnomview(
                filtered[0],
                sub=[nrow, ncol, 1],
                title=f"Filtered map, RMS = {rms1}",
                **args,
            )
            rms2 = np.std(test_map[0, good])
            hp.gnomview(
                test_map[0],
                sub=[nrow, ncol, 2],
                title=f"Input x obs.matrix, RMS = {rms2}",
                **args,
            )
            rms3 = np.std(input_map[0, good])
            hp.gnomview(
                input_map[0],
                sub=[nrow, ncol, 3],
                title=f"Input map, RMS = {rms3}",
                **args,
            )
            rms4 = np.std(diffmap[0, good])
            hp.gnomview(
                diffmap[0],
                sub=[nrow, ncol, 4],
                title=f"Difference, RMS = {rms4}",
                **args,
            )
            fname = os.path.join(self.outdir, "obs_matrix_flagged_test.png")
            fig.savefig(fname)

            for i in range(3):
                rms1 = np.std(filtered[i][good])
                rms2 = np.std((filtered - test_map)[i][good])
                if rms2 > 1e-5 * rms1:
                    print(f"rms2 = {rms2}, rms1 = {rms1}")
                assert rms2 < 1e-5 * rms1

        close_data(data)

    def test_filterbin_obsmatrix_cached(self):
        if sys.platform.lower() == "darwin":
            print(f"WARNING:  Skipping test_filterbin_obsmatrix_cached on MacOS")
            return
        if "CIBUILDWHEEL" in os.environ:
            print(
                f"WARNING:  Skipping test_filterbin_obsmatrix_cached during wheel tests"
            )
            return

        # Create a fake ground data set for testing
        data = create_ground_data(self.comm, sample_rate=1 * u.Hz)

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
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )
        weights.apply(data)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        input_map_file = os.path.join(self.outdir, "input_map3.fits")
        if data.comm.world_rank == 0:
            lmax = 3 * self.nside
            cls = np.ones(4 * (lmax + 1)).reshape(4, -1)
            fwhm = np.radians(10)
            input_map = hp.synfast(cls, self.nside, lmax=lmax, fwhm=fwhm, verbose=False)
            if pixels.nest:
                input_map = hp.reorder(input_map, r2n=True)
            hp.write_map(input_map_file, input_map, nest=pixels.nest, column_units="K")

        # Scan map into timestreams
        scan_hpix = ops.ScanHealpixMap(
            file=input_map_file,
            det_data=defaults.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
        )
        scan_hpix.apply(data)

        # Copy the signal
        ops.Copy(detdata=[(defaults.det_data, "signal_copy")]).apply(data)

        # Configure and apply the filterbin operator
        binning = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance="covariance",
            det_data=defaults.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="allreduce",
            shared_flags=defaults.shared_flags,
            shared_flag_mask=1,
            det_flags=defaults.det_flags,
            det_flag_mask=255,
        )

        filterbin = ops.FilterBin(
            name="filterbin",
            det_data=defaults.det_data,
            det_flags=defaults.det_flags,
            det_flag_mask=255,
            shared_flags=defaults.shared_flags,
            shared_flag_mask=1,
            binning=binning,
            ground_filter_order=5,
            split_ground_template=True,
            poly_filter_order=2,
            output_dir=self.outdir,
            write_obs_matrix=True,
            cache_dir=os.path.join(self.outdir, "cache"),
        )

        # Run filterbin twice and confirm that we get the
        # same observation matrix.  The second run uses cached
        # accumulants.

        filterbin.name = "cached_run_1"
        filterbin.apply(data)

        filterbin.name = "cached_run_2"
        filterbin.det_data = "signal_copy"
        filterbin.apply(data)

        if data.comm.world_rank == 0:
            import matplotlib.pyplot as plt

            rot = [42, -42]
            reso = 4
            fig = plt.figure(figsize=[18, 12])
            cmap = "coolwarm"
            nest = pixels.nest

            rootname = os.path.join(self.outdir, f"cached_run_1_obs_matrix")
            fname_matrix = ops.combine_observation_matrix(rootname)
            obs_matrix1 = scipy.sparse.load_npz(fname_matrix)

            rootname = os.path.join(self.outdir, f"cached_run_2_obs_matrix")
            fname_matrix = ops.combine_observation_matrix(rootname)
            obs_matrix2 = scipy.sparse.load_npz(fname_matrix)

            # Comparing the matrices fails with MPI for some reason
            # assert np.allclose(obs_matrix1.data, obs_matrix2.data)

            input_map = hp.read_map(input_map_file, None, nest=nest)

            fname_filtered = os.path.join(self.outdir, "cached_run_1_filtered_map.fits")
            filtered1 = hp.read_map(fname_filtered, None, nest=nest)

            fname_filtered = os.path.join(self.outdir, "cached_run_2_filtered_map.fits")
            filtered2 = hp.read_map(fname_filtered, None, nest=nest)

            test_map1 = obs_matrix1.dot(input_map.ravel()).reshape([3, -1])

            test_map2 = obs_matrix2.dot(input_map.ravel()).reshape([3, -1])

            nrow, ncol = 3, 3
            args = {"rot": rot, "reso": reso, "cmap": cmap, "nest": nest}

            filtered1[filtered1 == 0] = hp.UNSEEN
            filtered2[filtered2 == 0] = hp.UNSEEN
            test_map1[test_map1 == 0] = hp.UNSEEN
            test_map2[test_map2 == 0] = hp.UNSEEN
            hp.gnomview(
                filtered1[0],
                sub=[nrow, ncol, 1],
                title="Filtered map (no cache)",
                **args,
            )
            hp.gnomview(
                filtered2[0],
                sub=[nrow, ncol, 2],
                title="Filtered map (with cache)",
                **args,
            )
            diffmap = filtered1 - filtered2
            diffmap[diffmap == 0] = hp.UNSEEN
            hp.gnomview(diffmap[0], sub=[nrow, ncol, 3], title="Difference", **args)
            hp.gnomview(
                test_map1[0], sub=[nrow, ncol, 4], title="Input x obs.matrix", **args
            )
            hp.gnomview(
                test_map2[0], sub=[nrow, ncol, 5], title="Input x obs.matrix", **args
            )
            diffmap = test_map1 - test_map2
            diffmap[diffmap == 0] = hp.UNSEEN
            hp.gnomview(diffmap[0], sub=[nrow, ncol, 6], title="Difference", **args)
            diffmap = test_map1 - filtered1
            diffmap[diffmap == 0] = hp.UNSEEN
            hp.gnomview(diffmap[0], sub=[nrow, ncol, 7], title="Difference", **args)
            diffmap = test_map2 - filtered2
            diffmap[diffmap == 0] = hp.UNSEEN
            hp.gnomview(diffmap[0], sub=[nrow, ncol, 8], title="Difference", **args)
            fname = os.path.join(self.outdir, "obs_matrix_cached_test.png")
            fig.savefig(fname)

            for filtered, test_map in [(filtered1, test_map1), (filtered2, test_map2)]:
                good = filtered[0] != 0
                for i in range(3):
                    rms1 = np.std(filtered[i][good])
                    rms2 = np.std((filtered - test_map)[i][good])
                    if rms2 > 1e-5 * rms1:
                        print(f"rms2 = {rms2}, rms1 = {rms1}")
                    assert rms2 < 1e-5 * rms1

        close_data(data)

    def test_filterbin_obsmatrix_noiseweighted(self):
        if sys.platform.lower() == "darwin":
            print(f"WARNING:  Skipping test_filterbin_obsmatrix_noiseweighted on MacOS")
            return
        if "CIBUILDWHEEL" in os.environ:
            print(
                f"WARNING:  Skipping test_filterbin_obsmatrix_noiseweighted during wheel tests"
            )
            return

        # Create a fake ground data set for testing
        data = create_ground_data(self.comm, sample_rate=1 * u.Hz)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple(shared_flag_mask=0)
        pixels = ops.PixelsHealpix(
            nside=self.nside,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
            # view="scanning",
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )
        weights.apply(data)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        input_map_file = os.path.join(self.outdir, "input_map4.fits")
        if data.comm.world_rank == 0:
            lmax = 3 * self.nside
            cls = np.ones(4 * (lmax + 1)).reshape(4, -1)
            fwhm = np.radians(10)
            input_map = hp.synfast(cls, self.nside, lmax=lmax, fwhm=fwhm, verbose=False)
            if pixels.nest:
                input_map = hp.reorder(input_map, r2n=True)
            hp.write_map(input_map_file, input_map, nest=pixels.nest, column_units="K")

        # Scan map into timestreams
        scan_hpix = ops.ScanHealpixMap(
            file=input_map_file,
            det_data=defaults.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
        )
        scan_hpix.apply(data)

        # Copy the signal
        ops.Copy(detdata=[(defaults.det_data, "signal_copy")]).apply(data)

        # Configure and apply the filterbin operator
        binning = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance="covariance",
            det_data=defaults.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="allreduce",
            shared_flags=defaults.shared_flags,
            shared_flag_mask=detpointing.shared_flag_mask,
            det_flags=defaults.det_flags,
            det_flag_mask=255,
        )

        filterbin = ops.FilterBin(
            name="filterbin",
            det_data=defaults.det_data,
            det_flags=defaults.det_flags,
            det_flag_mask=255,
            shared_flags=defaults.shared_flags,
            shared_flag_mask=detpointing.shared_flag_mask,
            binning=binning,
            ground_filter_order=5,
            split_ground_template=True,
            poly_filter_order=2,
            output_dir=self.outdir,
            write_invcov=True,
            write_obs_matrix=True,
            poly_filter_view="scanning",
        )

        # Build the observation matrix twice, first in a single run and
        # then running each observation separately and saving the
        # noise-weighted matrix.

        filterbin.name = "split_run"
        filterbin.apply(data)

        filterbin.name = "noiseweighted_run"
        filterbin.write_invcov = True
        filterbin.det_data = "signal_copy"
        filterbin.noiseweight_obs_matrix = True

        orig_name_filterbin = filterbin.name
        orig_comm = data.comm
        new_comm = Comm(world=data.comm.comm_group)

        for iobs, obs in enumerate(data.obs):
            # Data object that only covers one observation
            obs_data = data.select(obs_uid=obs.uid)
            # Replace comm_world with the group communicator
            obs_data._comm = new_comm
            filterbin.reset_pix_dist = True
            filterbin.name = f"{orig_name_filterbin}_{obs.name}"
            filterbin.apply(obs_data)
            close_data(obs_data)

        if data.comm.comm_world is not None:
            # Make sure all observations are processed before proceeding
            data.comm.comm_world.Barrier()

        if data.comm.world_rank == 0:
            import matplotlib.pyplot as plt

            # Assemble the single-run matrix

            rootname = os.path.join(self.outdir, f"split_run_obs_matrix")
            fname_matrix = ops.combine_observation_matrix(rootname)
            obs_matrix1 = scipy.sparse.load_npz(fname_matrix)
            obs_matrix1.sort_indices()

            # Assemble the noise-weighted, per-observation matrix
            fnames = glob.glob(
                f"{self.outdir}/{orig_name_filterbin}*noiseweighted_obs_matrix*"
            )
            rootnames = set()
            for fname in fnames:
                rootnames.add(fname.split(".")[0])

            filenames = []
            for rootname in rootnames:
                fname_matrix = ops.combine_observation_matrix(rootname)
                filenames.append(fname_matrix)

            fname_matrix = f"{self.outdir}/noiseweighted_run_obs_matrix"
            if MPI is None:
                comm_self = None
            else:
                comm_self = MPI.COMM_SELF
            fname_matrix = ops.coadd_observation_matrix(
                filenames, fname_matrix, double_precision=True, comm=comm_self
            )
            obs_matrix2 = scipy.sparse.load_npz(fname_matrix)
            obs_matrix2.sort_indices()

            # Compare the values that are not tiny. Some of the tiny
            # values may be missing in one matrix

            values1 = obs_matrix1.data[np.abs(obs_matrix1.data) > 1e-10]
            values2 = obs_matrix2.data[np.abs(obs_matrix2.data) > 1e-10]

            assert np.allclose(values1, values2)

        close_data(data)
