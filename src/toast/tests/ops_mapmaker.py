# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import sys

import healpy as hp
import numpy as np
from astropy import units as u

from .. import ops as ops
from .. import templates
from ..accelerator import accel_enabled
from ..observation import default_values as defaults
from ..pixels import PixelData, PixelDistribution
from ..pixels_io_healpix import write_healpix_fits
from ..vis import set_matplotlib_backend
from ._helpers import close_data, create_fake_sky, create_outdir, create_satellite_data
from .mpi import MPITestCase


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
        if sys.platform.lower() == "darwin":
            print(f"WARNING:  Skipping test_offset on MacOS")
            return

        testdir = os.path.join(self.outdir, "offset")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir)

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
            solve_rcond_threshold=1.0e-1,
            map_rcond_threshold=1.0e-1,
            write_hits=False,
            write_map=True,
            write_cov=False,
            write_rcond=False,
            keep_solver_products=False,
            keep_final_products=False,
            output_dir=testdir,
        )

        # Make the map
        mapper.apply(data)

        # Check that we can also run in full-memory mode
        tmatrix.reset()
        mapper.reset_pix_dist = True
        pixels.apply(data)
        weights.apply(data)

        use_accel = None
        if accel_enabled() and (
            pixels.supports_accel()
            and weights.supports_accel()
            and mapper.supports_accel()
        ):
            use_accel = True
            data.accel_create(pixels.requires())
            data.accel_create(weights.requires())
            data.accel_create(mapper.requires())
            data.accel_update_device(pixels.requires())
            data.accel_update_device(weights.requires())
            data.accel_update_device(mapper.requires())

        binner.full_pointing = True
        mapper.name = "test2"
        mapper.apply(data, use_accel=use_accel)

        close_data(data)

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
        delete_pointing = ops.Delete(
            detdata=[pixels.pixels, weights.weights, detpointing.quats]
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
        # step_seconds = float(int(ob_time / 10.0))
        step_seconds = 5.0
        tmpl = templates.Offset(
            times=defaults.times,
            noise_model=default_model.noise_model,
            step_time=step_seconds * u.second,
        )

        tmatrix = ops.TemplateMatrix(templates=[tmpl])

        ops.Copy(detdata=[(defaults.det_data, "input_signal")]).apply(data)

        # Map maker
        mapper = ops.MapMaker(
            name="toastmap",
            det_data=defaults.det_data,
            binning=binner,
            template_matrix=tmatrix,
            solve_rcond_threshold=1.0e-1,
            map_rcond_threshold=1.0e-1,
            iter_min=30,
            iter_max=100,
            write_hits=True,
            write_map=True,
            write_binmap=False,
            write_noiseweighted_map=False,
            write_cov=False,
            write_rcond=False,
            write_solver_products=True,
            keep_final_products=False,
            output_dir=testdir,
            save_cleaned=True,
            overwrite_cleaned=False,
        )

        # Make the map
        mapper.apply(data)

        toast_hit_path = os.path.join(testdir, f"{mapper.name}_hits.fits")
        toast_map_path = os.path.join(testdir, f"{mapper.name}_map.fits")
        toast_mask_path = os.path.join(testdir, f"{mapper.name}_solve_rcond_mask.fits")

        # Now run Madam on the same data and compare

        sample_rate = data.obs[0]["noise_model"].rate(data.obs[0].local_detectors[0])

        pars = {}
        pars["kfirst"] = "T"
        pars["iter_min"] = 30
        pars["iter_max"] = 100
        pars["base_first"] = step_seconds
        pars["fsample"] = sample_rate
        pars["nside_map"] = pixels.nside
        pars["nside_cross"] = pixels.nside
        pars["nside_submap"] = min(8, pixels.nside)
        pars["good_baseline_fraction"] = tmpl.good_fraction
        pars["pixlim_cross"] = 1.0e-1
        pars["pixmode_cross"] = 2  # Use rcond threshold
        pars["pixlim_map"] = 1.0e-1
        pars["pixmode_map"] = 2  # Use rcond threshold
        pars["write_map"] = "T"
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["write_base"] = "F"
        pars["write_mask"] = "T"
        pars["kfilter"] = "F"
        pars["precond_width_min"] = 1
        pars["precond_width_max"] = 1
        pars["use_cgprecond"] = "F"
        pars["use_fprecond"] = "F"
        pars["info"] = 2
        pars["path_output"] = testdir

        madam = ops.Madam(
            params=pars,
            det_data=defaults.det_data,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model="noise_model",
            det_out="madam_cleaned",
        )

        # Generate persistent pointing
        pixels.apply(data)
        weights.apply(data)

        # Run Madam
        madam.apply(data)

        madam_hit_path = os.path.join(testdir, "madam_hmap.fits")
        madam_map_path = os.path.join(testdir, "madam_map.fits")
        madam_mask_path = os.path.join(testdir, "madam_mask.fits")

        fail = False

        # Compare local destriped TOD on every process

        for ob in data.obs:
            for det in ob.local_detectors:
                input_signal = ob.detdata["input_signal"][det]
                madam_signal = ob.detdata["madam_cleaned"][det]
                toast_signal = ob.detdata["toastmap_cleaned"][det]
                madam_base = input_signal - madam_signal
                toast_base = input_signal - toast_signal
                diff_base = madam_base - toast_base

                if not np.allclose(toast_base, madam_base, rtol=0.01):
                    print(
                        f"FAIL: {det} diff : PtP = {np.ptp(diff_base)}, "
                        f"mean = {np.mean(diff_base)}"
                    )
                    fail = True

                    set_matplotlib_backend()
                    import matplotlib.pyplot as plt

                    dbg_root = os.path.join(
                        testdir, f"base_{ob.name}_{data.comm.world_rank}_{det}"
                    )
                    np.savetxt(f"{dbg_root}_signal.txt", input_signal)
                    np.savetxt(f"{dbg_root}_madam.txt", madam_base)
                    np.savetxt(f"{dbg_root}_toast.txt", toast_base)
                    np.savetxt(f"{dbg_root}_diff.txt", diff_base)

                    fig = plt.figure(figsize=(12, 8), dpi=72)
                    ax = fig.add_subplot(1, 1, 1, aspect="auto")
                    ax.plot(
                        np.arange(len(input_signal)),
                        input_signal,
                        c="black",
                        label="Input",
                    )
                    ax.plot(
                        np.arange(len(madam_base)),
                        madam_base,
                        c="green",
                        label="Madam",
                    )
                    ax.plot(
                        np.arange(len(toast_base)),
                        toast_base,
                        c="red",
                        label="Toast",
                    )
                    ax.legend(loc=1)
                    plt.title("Baseline Comparison")
                    savefile = f"{dbg_root}.pdf"
                    plt.savefig(savefile)
                    plt.close()

        # Compare map-domain products on one process

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

            # Compare masks

            toast_mask = hp.read_map(toast_mask_path, field=None, nest=True)
            madam_mask = hp.read_map(madam_mask_path, field=None, nest=True)

            # Madam uses 1=good, 0=bad, Toast uses 0=good, non-zero=bad:
            tgood = toast_mask == 0
            toast_mask[:] = 0
            toast_mask[tgood] = 1
            diff_mask = toast_mask - madam_mask[0]

            outfile = os.path.join(testdir, "madam_mask.png")
            hp.mollview(madam_mask[0], xsize=1600, nest=True)
            plt.savefig(outfile)
            plt.close()
            outfile = os.path.join(testdir, "toast_mask.png")
            hp.mollview(toast_mask, xsize=1600, nest=True)
            plt.savefig(outfile)
            plt.close()
            outfile = os.path.join(testdir, "diff_mask.png")
            hp.mollview(diff_mask, xsize=1600, nest=True)
            plt.savefig(outfile)
            plt.close()

            # Compare maps

            toast_map = hp.read_map(toast_map_path, field=None, nest=True)
            madam_map = hp.read_map(madam_map_path, field=None, nest=True)
            # Set madam unhit pixels to zero
            for stokes, ststr in zip(range(3), ["I", "Q", "U"]):
                good = madam_map[stokes] != hp.UNSEEN
                diff_map = toast_map[stokes] - madam_map[stokes]

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

                if not np.allclose(
                    toast_map[stokes][good], madam_map[stokes][good], rtol=0.01
                ):
                    print(f"FAIL: max {ststr} diff = {np.max(diff_map[good])}")
                    fail = True

        if data.comm.comm_world is not None:
            fail = data.comm.comm_world.bcast(fail, root=0)

        self.assertFalse(fail)

        close_data(data)

    def test_compare_madam_diagpre(self):
        if not ops.madam.available():
            print("libmadam not available, skipping comparison with noise prior")
            return

        testdir = os.path.join(self.outdir, "compare_madam_diagpre")
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
            noise_model=default_model.noise_model,
            step_time=step_seconds * u.second,
            use_noise_prior=True,
            precond_width=1,
        )

        tmatrix = ops.TemplateMatrix(templates=[tmpl])

        # Map maker
        mapper = ops.MapMaker(
            name="toastmap",
            det_data=defaults.det_data,
            binning=binner,
            template_matrix=tmatrix,
            solve_rcond_threshold=1.0e-4,
            map_rcond_threshold=1.0e-4,
            iter_max=50,
            write_hits=False,
            write_map=False,
            write_cov=False,
            write_rcond=False,
            keep_final_products=True,
        )

        # Make the map
        mapper.apply(data)

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
        pars["basis_order"] = 0
        pars["iter_max"] = 50
        pars["base_first"] = step_seconds
        pars["fsample"] = sample_rate
        pars["nside_map"] = pixels.nside
        pars["nside_cross"] = pixels.nside
        pars["nside_submap"] = min(8, pixels.nside)
        pars["good_baseline_fraction"] = tmpl.good_fraction
        pars["pixlim_cross"] = 1.0e-4
        pars["pixmode_cross"] = 2  # Use rcond threshold
        pars["pixlim_map"] = 1.0e-4
        pars["pixmode_map"] = 2  # Use rcond threshold
        pars["write_map"] = "T"
        pars["write_binmap"] = "F"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["write_base"] = "T"
        pars["kfilter"] = "T"
        pars["precond_width_min"] = 1
        pars["precond_width_max"] = 1
        pars["use_cgprecond"] = "F"
        pars["use_fprecond"] = "T"
        pars["path_output"] = testdir

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

                if not np.allclose(
                    toast_map[stokes], madam_map[stokes], atol=0.1, rtol=0.05
                ):
                    fail = True

        if data.comm.comm_world is not None:
            fail = data.comm.comm_world.bcast(fail, root=0)

        self.assertFalse(fail)

        close_data(data)

    def test_compare_madam_bandpre(self):
        if not ops.madam.available():
            print(
                "libmadam not available, skipping comparison with banded preconditioner"
            )
            return

        testdir = os.path.join(self.outdir, "compare_madam_bandpre")
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
        delete_pointing = ops.Delete(
            detdata=[pixels.pixels, weights.weights, detpointing.quats]
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
        # step_seconds = float(int(ob_time / 10.0))
        step_seconds = 5.0
        tmpl = templates.Offset(
            times=defaults.times,
            noise_model=default_model.noise_model,
            step_time=step_seconds * u.second,
            use_noise_prior=True,
            precond_width=10,
        )

        tmatrix = ops.TemplateMatrix(templates=[tmpl])

        # Map maker
        mapper = ops.MapMaker(
            name="toastmap",
            det_data=defaults.det_data,
            binning=binner,
            template_matrix=tmatrix,
            solve_rcond_threshold=1.0e-4,
            map_rcond_threshold=1.0e-4,
            iter_max=50,
            write_hits=False,
            write_map=False,
            write_cov=False,
            write_rcond=False,
            keep_final_products=True,
        )

        # Make the map
        mapper.apply(data)

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
        pars["basis_order"] = 0
        pars["iter_max"] = 50
        pars["base_first"] = step_seconds
        pars["fsample"] = sample_rate
        pars["nside_map"] = pixels.nside
        pars["nside_cross"] = pixels.nside
        pars["nside_submap"] = min(8, pixels.nside)
        pars["good_baseline_fraction"] = tmpl.good_fraction
        pars["pixlim_cross"] = 1.0e-4
        pars["pixmode_cross"] = 2  # Use rcond threshold
        pars["pixlim_map"] = 1.0e-4
        pars["pixmode_map"] = 2  # Use rcond threshold
        pars["write_map"] = "T"
        pars["write_binmap"] = "F"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["write_base"] = "T"
        pars["kfilter"] = "T"
        pars["precond_width_min"] = 10
        pars["precond_width_max"] = 10
        pars["use_cgprecond"] = "T"
        pars["use_fprecond"] = "F"
        pars["path_output"] = testdir

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
                # print("diff map {} has rms {}".format(ststr, np.std(diff_map)))
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

                if not np.allclose(
                    toast_map[stokes], madam_map[stokes], atol=0.1, rtol=0.05
                ):
                    fail = True

        if data.comm.comm_world is not None:
            fail = data.comm.comm_world.bcast(fail, root=0)

        self.assertFalse(fail)

        close_data(data)
