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

from ..observation import default_values as defaults

from .. import templates

from ..pixels import PixelDistribution, PixelData

from ..pixels_io import write_healpix_fits

from ._helpers import create_outdir, create_satellite_data, create_fake_sky


class MapmakerTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)
        # We want to hold the number of observations fixed, so that we can compare
        # results across different concurrencies.
        self.total_obs = 8
        self.obs_per_group = self.total_obs
        if self.comm is not None and self.comm.size >= 2:
            self.obs_per_group = self.total_obs // 2

    def test_offset(self):
        # Create a fake satellite data set for testing

        data = create_satellite_data(
            self.comm, obs_per_group=self.obs_per_group, obs_time=10.0 * u.minute
        )

        # Create some sky signal timestreams.
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
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

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, "pixel_dist", "fake_map")

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data=defaults.det_data,
            pixels=pixels.pixels,
            weights=weights.weights,
            map_key="fake_map",
        )
        scanner.apply(data)

        # Now clear the pointing and reset things for use with the mapmaking test later
        delete_pointing = ops.Delete(
            detdata=[detpointing.quats, pixels.pixels, weights.weights]
        )
        delete_pointing.apply(data)
        pixels.create_dist = None

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise and accumulate to signal
        sim_noise = ops.SimNoise(
            noise_model=default_model.noise_model, det_data=defaults.det_data
        )
        sim_noise.apply(data)

        # Set up binning operator for solving
        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
        )

        # Set up template matrix with just an offset template.

        # Use 1/10 of an observation as the baseline length.  Make it not evenly
        # divisible in order to test handling of the final amplitude.
        ob_time = (
            data.obs[0].shared[defaults.times][-1]
            - data.obs[0].shared[defaults.times][0]
        )
        step_seconds = float(int(ob_time / 10.0))
        tmpl = templates.Offset(
            times=defaults.times,
            det_flags=None,
            noise_model=default_model.noise_model,
            step_time=step_seconds * u.second,
        )

        tmatrix = ops.TemplateMatrix(templates=[tmpl])

        # Map maker
        mapper = ops.MapMaker(
            name="test1",
            det_data=defaults.det_data,
            binning=binner,
            template_matrix=tmatrix,
            write_hits=False,
            write_map=False,
            write_cov=False,
            write_rcond=False,
            keep_solver_products=True,
            keep_final_products=True,
        )

        # Make the map
        mapper.apply(data)

        # Check that we can also run in full-memory mode
        pixels.apply(data)
        weights.apply(data)
        binner.full_pointing = True
        mapper.name = "test2"
        mapper.apply(data)

        del data
        return

    def test_compare_madam_noprior(self):
        if not ops.madam.available():
            print("libmadam not available, skipping destriping comparison")
            return

        testdir = os.path.join(self.outdir, "compare_madam_noprior")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir)

        # Create a fake satellite data set for testing
        data = create_satellite_data(
            self.comm, obs_per_group=self.obs_per_group, obs_time=10.0 * u.minute
        )

        # Create some sky signal timestreams.
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=16,
            nest=True,
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

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, "pixel_dist", "fake_map")

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data=defaults.det_data,
            pixels=pixels.pixels,
            weights=weights.weights,
            map_key="fake_map",
        )
        scanner.apply(data)

        # Now clear the pointing and reset things for use with the mapmaking test later
        delete_pointing = ops.Delete(detdata=[pixels.pixels, weights.weights])
        delete_pointing.apply(data)
        pixels.create_dist = None

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise and accumulate to signal
        sim_noise = ops.SimNoise(
            noise_model=default_model.noise_model, det_data=defaults.det_data
        )
        sim_noise.apply(data)

        # Set up binning operator for solving
        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
        )

        # Set up template matrix with just an offset template.

        # Use 1/10 of an observation as the baseline length.  Make it not evenly
        # divisible in order to test handling of the final amplitude.
        ob_time = (
            data.obs[0].shared[defaults.times][-1]
            - data.obs[0].shared[defaults.times][0]
        )
        # step_seconds = float(int(ob_time / 10.0))
        step_seconds = 5.0
        tmpl = templates.Offset(
            times=defaults.times,
            det_flags=None,
            noise_model=default_model.noise_model,
            step_time=step_seconds * u.second,
        )

        tmatrix = ops.TemplateMatrix(templates=[tmpl])

        # Map maker
        mapper = ops.MapMaker(
            name="toastmap",
            det_data=defaults.det_data,
            binning=binner,
            template_matrix=tmatrix,
            solve_rcond_threshold=1.0e-6,
            map_rcond_threshold=1.0e-6,
            iter_max=10,
            write_hits=False,
            write_map=False,
            write_cov=False,
            write_rcond=False,
            keep_final_products=True,
        )

        # Make the map
        mapper.apply(data)

        # Write baselines to a file
        toast_amp_path = os.path.join(
            testdir, f"toast_baselines_{data.comm.world_rank}.txt"
        )
        np.savetxt(toast_amp_path, data[f"toastmap_solve_amplitudes"]["Offset"].local)

        # Outputs
        toast_hits = "toastmap_hits"
        toast_map = "toastmap_map"

        # Write map to disk so we can load the whole thing on one process.

        toast_hit_path = os.path.join(testdir, "toast_hits.fits")
        toast_map_path = os.path.join(testdir, "toast_map.fits")
        write_healpix_fits(data[toast_map], toast_map_path, nest=True)
        write_healpix_fits(data[toast_hits], toast_hit_path, nest=True)

        # Now run Madam on the same data and compare

        sample_rate = data.obs[0]["noise_model"].rate(data.obs[0].local_detectors[0])

        pars = {}
        pars["kfirst"] = "T"
        pars["iter_max"] = 10
        pars["base_first"] = step_seconds
        pars["fsample"] = sample_rate
        pars["nside_map"] = pixels.nside
        pars["nside_cross"] = pixels.nside
        pars["nside_submap"] = min(8, pixels.nside)
        pars["good_baseline_fraction"] = tmpl.good_fraction
        pars["pixlim_cross"] = 1.0e-6
        pars["pixlim_map"] = 1.0e-6
        pars["write_map"] = "T"
        pars["write_binmap"] = "F"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["write_base"] = "T"
        pars["kfilter"] = "F"
        pars["path_output"] = testdir

        madam = ops.Madam(
            params=pars,
            det_data=defaults.det_data,
            det_flags=None,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model="noise_model",
        )

        # Generate persistent pointing
        pixels.apply(data)
        weights.apply(data)

        # Run Madam
        madam.apply(data)

        madam_hit_path = os.path.join(testdir, "madam_hmap.fits")
        madam_map_path = os.path.join(testdir, "madam_map.fits")

        fail = False

        if data.comm.world_rank == 0:
            set_matplotlib_backend()
            import matplotlib.pyplot as plt

            # Compare hit maps

            toast_hits = hp.read_map(toast_hit_path, field=None, nest=True)
            madam_hits = hp.read_map(madam_hit_path, field=None, nest=True)
            diff_hits = toast_hits - madam_hits

            outfile = os.path.join(testdir, "madam_hits.png")
            hp.mollview(madam_hits, xsize=1600, nest=True)
            plt.savefig(outfile)
            plt.close()
            outfile = os.path.join(testdir, "toast_hits.png")
            hp.mollview(toast_hits, xsize=1600, nest=True)
            plt.savefig(outfile)
            plt.close()
            outfile = os.path.join(testdir, "diff_hits.png")
            hp.mollview(diff_hits, xsize=1600, nest=True)
            plt.savefig(outfile)
            plt.close()

            # Compare maps

            toast_map = hp.read_map(toast_map_path, field=None, nest=True)
            madam_map = hp.read_map(madam_map_path, field=None, nest=True)
            # Set madam unhit pixels to zero
            for stokes, ststr in zip(range(3), ["I", "Q", "U"]):
                mask = hp.mask_bad(madam_map[stokes])
                madam_map[stokes][mask] = 0.0
                diff_map = toast_map[stokes] - madam_map[stokes]
                print("diff map {} has rms {}".format(ststr, np.std(diff_map)))
                outfile = os.path.join(testdir, "madam_map_{}.png".format(ststr))
                hp.mollview(madam_map[stokes], xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()
                outfile = os.path.join(testdir, "toast_map_{}.png".format(ststr))
                hp.mollview(toast_map[stokes], xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()
                outfile = os.path.join(testdir, "diff_map_{}.png".format(ststr))
                hp.mollview(diff_map, xsize=1600, nest=True)
                plt.savefig(outfile)
                plt.close()

                if not np.allclose(toast_map[stokes], madam_map[stokes], rtol=0.01):
                    print(f"FAIL: max {ststr} diff = {np.max(diff_map)}")
                    fail = True

        if data.comm.comm_world is not None:
            fail = data.comm.comm_world.bcast(fail, root=0)

        self.assertFalse(fail)

        del data
        return

    # def test_compare_madam_diagpre(self):
    #     if not ops.madam.available():
    #         print("libmadam not available, skipping comparison with noise prior")
    #         return

    #     testdir = os.path.join(self.outdir, "compare_madam_diagpre")
    #     if self.comm is None or self.comm.rank == 0:
    #         os.makedirs(testdir)

    #     # Create a fake satellite data set for testing
    #     data = create_satellite_data(
    #         self.comm, obs_per_group=self.obs_per_group, obs_time=10.0 * u.minute
    #     )

    #     # Create some sky signal timestreams.
    #     detpointing = ops.PointingDetectorSimple()
    #     pixels = ops.PixelsHealpix(
    #         nside=16,
    #         nest=True,
    #         create_dist="pixel_dist",
    #         detector_pointing=detpointing,
    #     )
    #     pixels.apply(data)
    #     weights = ops.StokesWeights(
    #         mode="IQU",
    #         hwp_angle=defaults.hwp_angle,
    #         detector_pointing=detpointing,
    #     )
    #     weights.apply(data)

    #     # Create fake polarized sky pixel values locally
    #     create_fake_sky(data, "pixel_dist", "fake_map")

    #     # Scan map into timestreams
    #     scanner = ops.ScanMap(
    #         det_data=defaults.det_data,
    #         pixels=pixels.pixels,
    #         weights=weights.weights,
    #         map_key="fake_map",
    #     )
    #     scanner.apply(data)

    #     # Now clear the pointing and reset things for use with the mapmaking test later
    #     delete_pointing = ops.Delete(detdata=[pixels.pixels, weights.weights])
    #     delete_pointing.apply(data)
    #     pixels.create_dist = None

    #     # Create an uncorrelated noise model from focalplane detector properties
    #     default_model = ops.DefaultNoiseModel(noise_model="noise_model")
    #     default_model.apply(data)

    #     # Simulate noise and accumulate to signal
    #     sim_noise = ops.SimNoise(
    #         noise_model=default_model.noise_model, det_data=defaults.det_data
    #     )
    #     sim_noise.apply(data)

    #     # Set up binning operator for solving
    #     binner = ops.BinMap(
    #         pixel_dist="pixel_dist",
    #         pixel_pointing=pixels,
    #         stokes_weights=weights,
    #         noise_model=default_model.noise_model,
    #     )

    #     # Set up template matrix with just an offset template.

    #     # Use 1/10 of an observation as the baseline length.  Make it not evenly
    #     # divisible in order to test handling of the final amplitude.
    #     ob_time = (
    #         data.obs[0].shared[defaults.times][-1]
    #         - data.obs[0].shared[defaults.times][0]
    #     )
    #     # step_seconds = float(int(ob_time / 10.0))
    #     step_seconds = 5.0
    #     tmpl = templates.Offset(
    #         times=defaults.times,
    #         det_flags=None,
    #         noise_model=default_model.noise_model,
    #         step_time=step_seconds * u.second,
    #         use_noise_prior=True,
    #         precond_width=1,
    #     )

    #     tmatrix = ops.TemplateMatrix(templates=[tmpl])

    #     # Map maker
    #     mapper = ops.MapMaker(
    #         name="toastmap",
    #         det_data=defaults.det_data,
    #         binning=binner,
    #         template_matrix=tmatrix,
    #         solve_rcond_threshold=1.0e-4,
    #         map_rcond_threshold=1.0e-4,
    #         iter_max=50,
    #         write_hits=False,
    #         write_map=False,
    #         write_cov=False,
    #         write_rcond=False,
    #         keep_final_products=True,
    #     )

    #     # Make the map
    #     mapper.apply(data)

    #     # Outputs
    #     toast_hits = "toastmap_hits"
    #     toast_map = "toastmap_map"

    #     # Write map to disk so we can load the whole thing on one process.

    #     toast_hit_path = os.path.join(testdir, "toast_hits.fits")
    #     toast_map_path = os.path.join(testdir, "toast_map.fits")
    #     write_healpix_fits(data[toast_map], toast_map_path, nest=True)
    #     write_healpix_fits(data[toast_hits], toast_hit_path, nest=True)

    #     # Now run Madam on the same data and compare

    #     sample_rate = data.obs[0]["noise_model"].rate(data.obs[0].local_detectors[0])

    #     pars = {}
    #     pars["kfirst"] = "T"
    #     pars["basis_order"] = 0
    #     pars["iter_max"] = 50
    #     pars["base_first"] = step_seconds
    #     pars["fsample"] = sample_rate
    #     pars["nside_map"] = pixels.nside
    #     pars["nside_cross"] = pixels.nside
    #     pars["nside_submap"] = min(8, pixels.nside)
    #     pars["good_baseline_fraction"] = tmpl.good_fraction
    #     pars["pixlim_cross"] = 1.0e-4
    #     pars["pixlim_map"] = 1.0e-4
    #     pars["write_map"] = "T"
    #     pars["write_binmap"] = "F"
    #     pars["write_matrix"] = "F"
    #     pars["write_wcov"] = "F"
    #     pars["write_hits"] = "T"
    #     pars["write_base"] = "T"
    #     pars["kfilter"] = "T"
    #     pars["precond_width_min"] = 1
    #     pars["precond_width_max"] = 1
    #     pars["use_cgprecond"] = "F"
    #     pars["use_fprecond"] = "T"
    #     pars["path_output"] = testdir

    #     madam = ops.Madam(
    #         params=pars,
    #         det_data=defaults.det_data,
    #         det_flags=None,
    #         pixel_pointing=pixels,
    #         stokes_weights=weights,
    #         noise_model="noise_model",
    #     )

    #     # Generate persistent pointing
    #     pixels.apply(data)
    #     weights.apply(data)

    #     # Run Madam
    #     madam.apply(data)

    #     madam_hit_path = os.path.join(testdir, "madam_hmap.fits")
    #     madam_map_path = os.path.join(testdir, "madam_map.fits")

    #     fail = False

    #     if data.comm.world_rank == 0:
    #         set_matplotlib_backend()
    #         import matplotlib.pyplot as plt

    #         # Compare hit maps

    #         toast_hits = hp.read_map(toast_hit_path, field=None, nest=True)
    #         madam_hits = hp.read_map(madam_hit_path, field=None, nest=True)
    #         diff_hits = toast_hits - madam_hits

    #         outfile = os.path.join(testdir, "madam_hits.png")
    #         hp.mollview(madam_hits, xsize=1600, nest=True)
    #         plt.savefig(outfile)
    #         plt.close()
    #         outfile = os.path.join(testdir, "toast_hits.png")
    #         hp.mollview(toast_hits, xsize=1600, nest=True)
    #         plt.savefig(outfile)
    #         plt.close()
    #         outfile = os.path.join(testdir, "diff_hits.png")
    #         hp.mollview(diff_hits, xsize=1600, nest=True)
    #         plt.savefig(outfile)
    #         plt.close()

    #         # Compare maps

    #         toast_map = hp.read_map(toast_map_path, field=None, nest=True)
    #         madam_map = hp.read_map(madam_map_path, field=None, nest=True)
    #         # Set madam unhit pixels to zero
    #         for stokes, ststr in zip(range(3), ["I", "Q", "U"]):
    #             mask = hp.mask_bad(madam_map[stokes])
    #             madam_map[stokes][mask] = 0.0
    #             diff_map = toast_map[stokes] - madam_map[stokes]
    #             print("diff map {} has rms {}".format(ststr, np.std(diff_map)))
    #             outfile = os.path.join(testdir, "madam_map_{}.png".format(ststr))
    #             hp.mollview(madam_map[stokes], xsize=1600, nest=True)
    #             plt.savefig(outfile)
    #             plt.close()
    #             outfile = os.path.join(testdir, "toast_map_{}.png".format(ststr))
    #             hp.mollview(toast_map[stokes], xsize=1600, nest=True)
    #             plt.savefig(outfile)
    #             plt.close()
    #             outfile = os.path.join(testdir, "diff_map_{}.png".format(ststr))
    #             hp.mollview(diff_map, xsize=1600, nest=True)
    #             plt.savefig(outfile)
    #             plt.close()

    #             if not np.allclose(
    #                 toast_map[stokes], madam_map[stokes], atol=0.1, rtol=0.05
    #             ):
    #                 fail = True

    #     if data.comm.comm_world is not None:
    #         fail = data.comm.comm_world.bcast(fail, root=0)

    #     self.assertFalse(fail)

    #     del data
    #     return

    # def test_compare_madam_bandpre(self):
    #     if not ops.madam.available():
    #         print(
    #             "libmadam not available, skipping comparison with banded preconditioner"
    #         )
    #         return

    #     testdir = os.path.join(self.outdir, "compare_madam_bandpre")
    #     if self.comm is None or self.comm.rank == 0:
    #         os.makedirs(testdir)

    #     # Create a fake satellite data set for testing
    #     data = create_satellite_data(
    #         self.comm, obs_per_group=self.obs_per_group, obs_time=10.0 * u.minute
    #     )

    #     # Create some sky signal timestreams.
    #     detpointing = ops.PointingDetectorSimple()
    #     pixels = ops.PixelsHealpix(
    #         nside=16,
    #         nest=True,
    #         create_dist="pixel_dist",
    #         detector_pointing=detpointing,
    #     )
    #     pixels.apply(data)
    #     weights = ops.StokesWeights(
    #         mode="IQU",
    #         hwp_angle=defaults.hwp_angle,
    #         detector_pointing=detpointing,
    #     )
    #     weights.apply(data)

    #     # Create fake polarized sky pixel values locally
    #     create_fake_sky(data, "pixel_dist", "fake_map")

    #     # Scan map into timestreams
    #     scanner = ops.ScanMap(
    #         det_data=defaults.det_data,
    #         pixels=pixels.pixels,
    #         weights=weights.weights,
    #         map_key="fake_map",
    #     )
    #     scanner.apply(data)

    #     # Now clear the pointing and reset things for use with the mapmaking test later
    #     delete_pointing = ops.Delete(detdata=[pixels.pixels, weights.weights])
    #     delete_pointing.apply(data)
    #     pixels.create_dist = None

    #     # Create an uncorrelated noise model from focalplane detector properties
    #     default_model = ops.DefaultNoiseModel(noise_model="noise_model")
    #     default_model.apply(data)

    #     # Simulate noise and accumulate to signal
    #     sim_noise = ops.SimNoise(
    #         noise_model=default_model.noise_model, det_data=defaults.det_data
    #     )
    #     sim_noise.apply(data)

    #     # Set up binning operator for solving
    #     binner = ops.BinMap(
    #         pixel_dist="pixel_dist",
    #         pixel_pointing=pixels,
    #         stokes_weights=weights,
    #         noise_model=default_model.noise_model,
    #     )

    #     # Set up template matrix with just an offset template.

    #     # Use 1/10 of an observation as the baseline length.  Make it not evenly
    #     # divisible in order to test handling of the final amplitude.
    #     ob_time = (
    #         data.obs[0].shared[defaults.times][-1]
    #         - data.obs[0].shared[defaults.times][0]
    #     )
    #     # step_seconds = float(int(ob_time / 10.0))
    #     step_seconds = 5.0
    #     tmpl = templates.Offset(
    #         times=defaults.times,
    #         det_flags=None,
    #         noise_model=default_model.noise_model,
    #         step_time=step_seconds * u.second,
    #         use_noise_prior=True,
    #         precond_width=10,
    #     )

    #     tmatrix = ops.TemplateMatrix(templates=[tmpl])

    #     # Map maker
    #     mapper = ops.MapMaker(
    #         name="toastmap",
    #         det_data=defaults.det_data,
    #         binning=binner,
    #         template_matrix=tmatrix,
    #         solve_rcond_threshold=1.0e-4,
    #         map_rcond_threshold=1.0e-4,
    #         iter_max=50,
    #         write_hits=False,
    #         write_map=False,
    #         write_cov=False,
    #         write_rcond=False,
    #         keep_final_products=True,
    #     )

    #     # Make the map
    #     mapper.apply(data)

    #     # Outputs
    #     toast_hits = "toastmap_hits"
    #     toast_map = "toastmap_map"

    #     # Write map to disk so we can load the whole thing on one process.

    #     toast_hit_path = os.path.join(testdir, "toast_hits.fits")
    #     toast_map_path = os.path.join(testdir, "toast_map.fits")
    #     write_healpix_fits(data[toast_map], toast_map_path, nest=True)
    #     write_healpix_fits(data[toast_hits], toast_hit_path, nest=True)

    #     # Now run Madam on the same data and compare

    #     sample_rate = data.obs[0]["noise_model"].rate(data.obs[0].local_detectors[0])

    #     pars = {}
    #     pars["kfirst"] = "T"
    #     pars["basis_order"] = 0
    #     pars["iter_max"] = 50
    #     pars["base_first"] = step_seconds
    #     pars["fsample"] = sample_rate
    #     pars["nside_map"] = pixels.nside
    #     pars["nside_cross"] = pixels.nside
    #     pars["nside_submap"] = min(8, pixels.nside)
    #     pars["good_baseline_fraction"] = tmpl.good_fraction
    #     pars["pixlim_cross"] = 1.0e-4
    #     pars["pixlim_map"] = 1.0e-4
    #     pars["write_map"] = "T"
    #     pars["write_binmap"] = "F"
    #     pars["write_matrix"] = "F"
    #     pars["write_wcov"] = "F"
    #     pars["write_hits"] = "T"
    #     pars["write_base"] = "T"
    #     pars["kfilter"] = "T"
    #     pars["precond_width_min"] = 10
    #     pars["precond_width_max"] = 10
    #     pars["use_cgprecond"] = "T"
    #     pars["use_fprecond"] = "F"
    #     pars["path_output"] = testdir

    #     madam = ops.Madam(
    #         params=pars,
    #         det_data=defaults.det_data,
    #         det_flags=None,
    #         pixel_pointing=pixels,
    #         stokes_weights=weights,
    #         noise_model="noise_model",
    #     )

    #     # Generate persistent pointing
    #     pixels.apply(data)
    #     weights.apply(data)

    #     # Run Madam
    #     madam.apply(data)

    #     madam_hit_path = os.path.join(testdir, "madam_hmap.fits")
    #     madam_map_path = os.path.join(testdir, "madam_map.fits")

    #     fail = False

    #     if data.comm.world_rank == 0:
    #         set_matplotlib_backend()
    #         import matplotlib.pyplot as plt

    #         # Compare hit maps

    #         toast_hits = hp.read_map(toast_hit_path, field=None, nest=True)
    #         madam_hits = hp.read_map(madam_hit_path, field=None, nest=True)
    #         diff_hits = toast_hits - madam_hits

    #         outfile = os.path.join(testdir, "madam_hits.png")
    #         hp.mollview(madam_hits, xsize=1600, nest=True)
    #         plt.savefig(outfile)
    #         plt.close()
    #         outfile = os.path.join(testdir, "toast_hits.png")
    #         hp.mollview(toast_hits, xsize=1600, nest=True)
    #         plt.savefig(outfile)
    #         plt.close()
    #         outfile = os.path.join(testdir, "diff_hits.png")
    #         hp.mollview(diff_hits, xsize=1600, nest=True)
    #         plt.savefig(outfile)
    #         plt.close()

    #         # Compare maps

    #         toast_map = hp.read_map(toast_map_path, field=None, nest=True)
    #         madam_map = hp.read_map(madam_map_path, field=None, nest=True)
    #         # Set madam unhit pixels to zero
    #         for stokes, ststr in zip(range(3), ["I", "Q", "U"]):
    #             mask = hp.mask_bad(madam_map[stokes])
    #             madam_map[stokes][mask] = 0.0
    #             diff_map = toast_map[stokes] - madam_map[stokes]
    #             print("diff map {} has rms {}".format(ststr, np.std(diff_map)))
    #             outfile = os.path.join(testdir, "madam_map_{}.png".format(ststr))
    #             hp.mollview(madam_map[stokes], xsize=1600, nest=True)
    #             plt.savefig(outfile)
    #             plt.close()
    #             outfile = os.path.join(testdir, "toast_map_{}.png".format(ststr))
    #             hp.mollview(toast_map[stokes], xsize=1600, nest=True)
    #             plt.savefig(outfile)
    #             plt.close()
    #             outfile = os.path.join(testdir, "diff_map_{}.png".format(ststr))
    #             hp.mollview(diff_map, xsize=1600, nest=True)
    #             plt.savefig(outfile)
    #             plt.close()

    #             if not np.allclose(
    #                 toast_map[stokes], madam_map[stokes], atol=0.1, rtol=0.05
    #             ):
    #                 fail = True

    #     if data.comm.comm_world is not None:
    #         fail = data.comm.comm_world.bcast(fail, root=0)

    #     self.assertFalse(fail)

    #     del data
    #     return
