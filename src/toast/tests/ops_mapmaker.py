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
from ..mpi import MPI
from ..observation import default_values as defaults
from ..timing import GlobalTimers
from ..timing import dump as dump_timers
from ..timing import gather_timers
from ..vis import set_matplotlib_backend, plot_healpix_maps
from .helpers import (
    close_data,
    create_fake_healpix_scanned_tod,
    create_outdir,
    create_satellite_data,
)
from .mpi import MPITestCase


class MapmakerTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        np.random.seed(123456)
        # We want to hold the number of observations fixed, so that we can compare
        # results across different concurrencies.
        self.total_obs = 8
        self.obs_per_group = self.total_obs
        if self.comm is not None and self.comm.size >= 2:
            self.obs_per_group = self.total_obs // 2

    def create_fake_data(self, testdir):
        # Create a fake satellite data set for testing
        data = create_satellite_data(
            self.comm, obs_per_group=self.obs_per_group, obs_time=10.0 * u.minute
        )

        # Create some sky signal timestreams.
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            create_dist="sim_pixel_dist",
            detector_pointing=detpointing,
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )
        weights.apply(data)

        # Create fake polarized sky signal
        skyfile = os.path.join(testdir, "input_map.fits")
        map_key = "fake_map"
        create_fake_healpix_scanned_tod(
            data,
            pixels,
            weights,
            skyfile,
            "sim_pixel_dist",
            map_key=map_key,
            fwhm=30.0 * u.arcmin,
            lmax=3 * pixels.nside,
            I_scale=0.001,
            Q_scale=0.0001,
            U_scale=0.0001,
            det_data=defaults.det_data,
        )

        # Now clear the pointing and reset things for use with the mapmaking test later
        del data["sim_pixel_dist"]
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
        return data

    def compare_madam_outputs(
        self,
        testdir,
        data,
        input="input_signal",
        madam_name="madam",
        toast_name="toastmap",
    ):
        madam_root = os.path.join(testdir, madam_name)
        toast_root = os.path.join(testdir, toast_name)

        madam_cleaned = f"{madam_name}_cleaned"
        toast_cleaned = f"{toast_name}_cleaned"

        madam_hit_path = f"{madam_root}_hmap.fits"
        madam_binned_path = f"{madam_root}_bmap.fits"
        madam_map_path = f"{madam_root}_map.fits"
        madam_mask_path = f"{madam_root}_mask.fits"

        toast_hit_path = f"{toast_root}_hits.fits"
        toast_binned_path = f"{toast_root}_binmap.fits"
        toast_map_path = f"{toast_root}_map.fits"
        toast_mask_path = f"{toast_root}_solve_rcond_mask.fits"
        toast_template_map_path = f"{toast_root}_template_map.fits"

        fail = 0

        # Compare local destriped TOD on every process

        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                input_signal = ob.detdata[input][det]
                madam_signal = ob.detdata[madam_cleaned][det]
                toast_signal = ob.detdata[toast_cleaned][det]
                cleaned_diff = np.absolute(madam_signal - toast_signal)
                madam_base = input_signal - madam_signal
                toast_base = input_signal - toast_signal
                diff_base = madam_base - toast_base
                if not np.allclose(toast_base, madam_base, rtol=0.01, atol=1.0e-6):
                    print(
                        f"FAIL: {ob.name}:{det} diff : PtP = {np.ptp(diff_base)}, "
                        f"mean = {np.mean(diff_base)}"
                    )
                    fail = 1

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

            # First plot everything for visual inspection later

            for hpath, mpaths in [
                (madam_hit_path, [madam_binned_path, madam_map_path]),
                (
                    toast_hit_path,
                    [toast_binned_path, toast_map_path, toast_template_map_path],
                ),
            ]:
                for mpath in mpaths:
                    plot_healpix_maps(
                        hitfile=hpath,
                        mapfile=mpath,
                        range_I=None,
                        range_Q=None,
                        range_U=None,
                        max_hits=None,
                        truth=None,
                        gnomview=True,
                        gnomres=1.5,
                        cmap="viridis",
                        format="pdf",
                    )

            # Now compare in memory

            toast_hits = hp.read_map(toast_hit_path, field=None, nest=True)
            madam_hits = hp.read_map(madam_hit_path, field=None, nest=True)

            toast_mask = hp.read_map(toast_mask_path, field=None, nest=True)
            madam_mask = hp.read_map(madam_mask_path, field=None, nest=True)

            toast_binned = hp.read_map(toast_binned_path, field=None, nest=True)
            madam_binned = hp.read_map(madam_binned_path, field=None, nest=True)

            toast_map = hp.read_map(toast_map_path, field=None, nest=True)
            madam_map = hp.read_map(madam_map_path, field=None, nest=True)

            toast_template_map = hp.read_map(
                toast_template_map_path, field=None, nest=True
            )
            madam_template_map = madam_binned - madam_map

            good_pix = madam_hits > 0
            bad_pix = np.logical_not(good_pix)

            diff_hits = toast_hits - madam_hits
            diff_hits[bad_pix] = 0

            # Compare masks

            # Madam uses 1=good, 0=bad, Toast uses 0=good, non-zero=bad:
            tgood = toast_mask == 0
            toast_mask[:] = 0
            toast_mask[tgood] = 1
            diff_mask = toast_mask - madam_mask[0]

            n_bad_mask = np.count_nonzero(diff_mask)
            if n_bad_mask > 0:
                print(f"FAIL: mask has {n_bad_mask} pixels that disagree")
                fail = 1

            # Compare maps

            for pix in np.arange(len(madam_hits), dtype=np.int64)[good_pix]:
                line = f"{pix}: bin: {madam_binned[0][pix]:0.3e} {toast_binned[0][pix]:0.3e} "
                line += f" tmpl: {madam_template_map[0][pix]:0.3e} {toast_template_map[0][pix]:0.3e} "
                line += f" destrp: {madam_map[0][pix]:0.3e} {toast_map[0][pix]:0.3e}"
                print(line, flush=True)

            for mname, tmap, mmap in [
                ("destriped", toast_map, madam_map),
                ("binned", toast_binned, madam_binned),
                ("template", toast_template_map, madam_template_map),
            ]:
                # Set madam unhit pixels to zero
                for stokes, ststr in zip(range(3), ["I", "Q", "U"]):
                    diff_map = tmap[stokes] - mmap[stokes]
                    diff_map[bad_pix] = hp.UNSEEN

                    outfile = os.path.join(testdir, f"diff_{mname}_{ststr}.png")
                    hp.mollview(diff_map, xsize=1600, nest=True)
                    plt.savefig(outfile)
                    plt.close()

                    if not np.allclose(
                        tmap[stokes][good_pix], mmap[stokes][good_pix], rtol=0.01
                    ):
                        msg = f"FAIL: {mname} max {ststr} diff = "
                        msg += f"{np.max(np.absolute(diff_map[good_pix]))}"
                        print(msg)
                        fail = 1

        if data.comm.comm_world is not None:
            fail = data.comm.comm_world.allreduce(fail, MPI.SUM)

        self.assertTrue(fail == 0)

    # def test_offset(self):
    #     if sys.platform.lower() == "darwin":
    #         print(f"WARNING:  Skipping test_offset on MacOS")
    #         return

    #     testdir = os.path.join(self.outdir, "offset")
    #     if self.comm is None or self.comm.rank == 0:
    #         os.makedirs(testdir)

    #     # Create a fake satellite data set for testing
    #     data = self.create_fake_data(testdir)

    #     detpointing = ops.PointingDetectorSimple()
    #     pixels = ops.PixelsHealpix(
    #         nside=64,
    #         detector_pointing=detpointing,
    #     )

    #     weights = ops.StokesWeights(
    #         mode="IQU",
    #         hwp_angle=defaults.hwp_angle,
    #         detector_pointing=detpointing,
    #     )

    #     # Set up binning operator for solving
    #     binner = ops.BinMap(
    #         pixel_dist="pixel_dist",
    #         pixel_pointing=pixels,
    #         stokes_weights=weights,
    #         noise_model="noise_model",
    #     )

    #     # Set up template matrix with just an offset template.

    #     # Use 1/10 of an observation as the baseline length.  Make it not evenly
    #     # divisible in order to test handling of the final amplitude.
    #     ob_time = (
    #         data.obs[0].shared[defaults.times][-1]
    #         - data.obs[0].shared[defaults.times][0]
    #     )
    #     step_seconds = float(int(ob_time / 10.0))
    #     tmpl = templates.Offset(
    #         times=defaults.times,
    #         noise_model="noise_model",
    #         step_time=step_seconds * u.second,
    #     )

    #     tmatrix = ops.TemplateMatrix(templates=[tmpl])

    #     # Map maker
    #     mapper = ops.MapMaker(
    #         name="test1",
    #         det_data=defaults.det_data,
    #         binning=binner,
    #         template_matrix=tmatrix,
    #         solve_rcond_threshold=1.0e-1,
    #         map_rcond_threshold=1.0e-1,
    #         write_hits=True,
    #         write_map=True,
    #         write_binmap=True,
    #         write_template_map=True,
    #         write_cov=False,
    #         write_rcond=False,
    #         keep_solver_products=False,
    #         keep_final_products=False,
    #         output_dir=testdir,
    #     )

    #     # Make the map
    #     mapper.apply(data)

    #     # Check that we can also run in full-memory mode
    #     tmatrix.reset()
    #     mapper.reset_pix_dist = True
    #     pixels.apply(data)
    #     weights.apply(data)

    #     use_accel = None
    #     if accel_enabled() and (
    #         pixels.supports_accel()
    #         and weights.supports_accel()
    #         and mapper.supports_accel()
    #     ):
    #         use_accel = True
    #         data.accel_create(pixels.requires())
    #         data.accel_create(weights.requires())
    #         data.accel_create(mapper.requires())
    #         data.accel_update_device(pixels.requires())
    #         data.accel_update_device(weights.requires())
    #         data.accel_update_device(mapper.requires())

    #     binner.full_pointing = True
    #     mapper.name = "test2"
    #     mapper.apply(data, use_accel=use_accel)

    #     close_data(data)

    # def test_offset_cache(self):
    #     if sys.platform.lower() == "darwin":
    #         print(f"WARNING:  Skipping test_offset on MacOS")
    #         return

    #     testdir = os.path.join(self.outdir, "offset_cache")
    #     if self.comm is None or self.comm.rank == 0:
    #         os.makedirs(testdir)

    #     # Create a fake satellite data set for testing
    #     data = self.create_fake_data(testdir)

    #     detpointing = ops.PointingDetectorSimple()
    #     pixels = ops.PixelsHealpix(
    #         nside=64,
    #         detector_pointing=detpointing,
    #     )
    #     weights = ops.StokesWeights(
    #         mode="IQU",
    #         hwp_angle=defaults.hwp_angle,
    #         detector_pointing=detpointing,
    #     )

    #     # Set up binning operator for solving
    #     binner = ops.BinMap(
    #         pixel_dist="pixel_dist",
    #         pixel_pointing=pixels,
    #         stokes_weights=weights,
    #         noise_model="noise_model",
    #     )

    #     # Set up template matrix with just an offset template.

    #     # Use 1/10 of an observation as the baseline length.  Make it not evenly
    #     # divisible in order to test handling of the final amplitude.
    #     ob_time = (
    #         data.obs[0].shared[defaults.times][-1]
    #         - data.obs[0].shared[defaults.times][0]
    #     )
    #     step_seconds = float(int(ob_time / 10.0))
    #     tmpl = templates.Offset(
    #         times=defaults.times,
    #         noise_model="noise_model",
    #         step_time=step_seconds * u.second,
    #     )

    #     tmatrix = ops.TemplateMatrix(templates=[tmpl])

    #     # Map maker
    #     mapper = ops.MapMaker(
    #         name="check",
    #         det_data=defaults.det_data,
    #         binning=binner,
    #         template_matrix=tmatrix,
    #         solve_rcond_threshold=1.0e-1,
    #         map_rcond_threshold=1.0e-1,
    #         write_hits=True,
    #         write_map=True,
    #         write_binmap=True,
    #         write_template_map=True,
    #         write_cov=False,
    #         write_rcond=False,
    #         keep_solver_products=False,
    #         keep_final_products=True,
    #         output_dir=testdir,
    #     )

    #     # Make the map
    #     mapper.apply(data)

    #     # Now enable caching of map-domain products and the signal TOD by
    #     # the binning operator.

    #     binner.cache_dir = os.path.join(testdir, "cache")
    #     binner.cache_detdata = True
    #     mapper.name = "test"

    #     mapper.apply(data)

    #     # Verify that the mapmaker products agree
    #     for prod in ["hits", "map", "binmap"]:
    #         original = data[f"test_{prod}"]
    #         check = data[f"check_{prod}"]
    #         failed = False

    #         if original.distribution != check.distribution:
    #             failed = True

    #         if not failed:
    #             dist = check.distribution
    #             for sm in range(dist.n_local_submap):
    #                 for px in range(dist.n_pix_submap):
    #                     if check.data[sm, px, 0] != 0:
    #                         if not np.allclose(
    #                             check.data[sm, px],
    #                             original.data[sm, px],
    #                             rtol=1e-5,
    #                             atol=1e-5,
    #                         ):
    #                             msg = f"{prod} FAIL [{sm},{px}]: "
    #                             msg += f"{original.data[sm,px]} "
    #                             msg += f"!= {check.data[sm,px]}"
    #                             print(msg, flush=True)
    #                             failed = True
    #         if data.comm.comm_world is not None:
    #             failed = data.comm.comm_world.allreduce(failed, op=MPI.LOR)
    #         self.assertFalse(failed)

    #     close_data(data)

    # def test_compare_madam_noprior(self):
    #     if not ops.madam.available():
    #         print("libmadam not available, skipping destriping comparison")
    #         return

    #     testdir = os.path.join(self.outdir, "compare_madam_noprior")
    #     if self.comm is None or self.comm.rank == 0:
    #         os.makedirs(testdir)

    #     # Create a fake satellite data set for testing
    #     data = self.create_fake_data(testdir)
    #     print("SIGNAL = ", data.obs[0].detdata["signal"][0])

    #     # Create some sky signal timestreams.
    #     detpointing = ops.PointingDetectorSimple()
    #     pixels = ops.PixelsHealpix(
    #         nside=16,
    #         nest=True,
    #         create_dist="pixel_dist",
    #         detector_pointing=detpointing,
    #     )
    #     weights = ops.StokesWeights(
    #         mode="IQU",
    #         hwp_angle=defaults.hwp_angle,
    #         detector_pointing=detpointing,
    #     )

    #     # Set up binning operator for solving
    #     binner = ops.BinMap(
    #         pixel_dist="pixel_dist",
    #         pixel_pointing=pixels,
    #         stokes_weights=weights,
    #         noise_model="noise_model",
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
    #         noise_model="noise_model",
    #         step_time=step_seconds * u.second,
    #     )

    #     tmatrix = ops.TemplateMatrix(templates=[tmpl])

    #     ops.Copy(detdata=[(defaults.det_data, "input_signal")]).apply(data)

    #     gt = GlobalTimers.get()
    #     gt.stop_all()
    #     gt.clear_all()

    #     # Map maker
    #     mapper = ops.MapMaker(
    #         name="toastmap",
    #         det_data=defaults.det_data,
    #         binning=binner,
    #         template_matrix=tmatrix,
    #         solve_rcond_threshold=1.0e-1,
    #         map_rcond_threshold=1.0e-1,
    #         iter_min=30,
    #         iter_max=100,
    #         write_hits=True,
    #         write_map=True,
    #         write_binmap=True,
    #         write_noiseweighted_map=False,
    #         write_cov=False,
    #         write_rcond=False,
    #         write_solver_products=True,
    #         keep_final_products=False,
    #         output_dir=testdir,
    #         save_cleaned=True,
    #         overwrite_cleaned=False,
    #     )

    #     # Make the map
    #     mapper.apply(data)

    #     alltimers = gather_timers(comm=data.comm.comm_world)
    #     if data.comm.world_rank == 0:
    #         out = os.path.join(testdir, "timing")
    #         dump_timers(alltimers, out)

    #     # Now run Madam on the same data and compare

    #     sample_rate = data.obs[0]["noise_model"].rate(data.obs[0].local_detectors[0])

    #     pars = {}
    #     pars["kfirst"] = "T"
    #     pars["iter_min"] = 30
    #     pars["iter_max"] = 100
    #     pars["base_first"] = step_seconds
    #     pars["fsample"] = sample_rate
    #     pars["nside_map"] = pixels.nside
    #     pars["nside_cross"] = pixels.nside
    #     pars["nside_submap"] = min(8, pixels.nside)
    #     pars["good_baseline_fraction"] = tmpl.good_fraction
    #     pars["pixlim_cross"] = 1.0e-1
    #     pars["pixmode_cross"] = 2  # Use rcond threshold
    #     pars["pixlim_map"] = 1.0e-1
    #     pars["pixmode_map"] = 2  # Use rcond threshold
    #     pars["write_map"] = "T"
    #     pars["write_binmap"] = "T"
    #     pars["write_matrix"] = "F"
    #     pars["write_wcov"] = "F"
    #     pars["write_hits"] = "T"
    #     pars["write_base"] = "F"
    #     pars["write_mask"] = "T"
    #     pars["kfilter"] = "F"
    #     pars["precond_width_min"] = 1
    #     pars["precond_width_max"] = 1
    #     pars["use_cgprecond"] = "F"
    #     pars["use_fprecond"] = "F"
    #     pars["info"] = 3
    #     pars["path_output"] = testdir

    #     madam = ops.Madam(
    #         params=pars,
    #         det_data=defaults.det_data,
    #         pixel_pointing=pixels,
    #         stokes_weights=weights,
    #         noise_model="noise_model",
    #         det_out="madam_cleaned",
    #     )

    #     # Generate persistent pointing
    #     pixels.apply(data)
    #     weights.apply(data)

    #     # Run Madam
    #     madam.apply(data)

    #     # Compare outputs
    #     self.compare_madam_outputs(testdir, data)

    #     close_data(data)

    def test_compare_madam_diagpre(self):
        if not ops.madam.available():
            print("libmadam not available, skipping comparison with noise prior")
            return

        testdir = os.path.join(self.outdir, "compare_madam_diagpre")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir)

        # Create a fake satellite data set for testing
        raw_data = create_satellite_data(
            self.comm, obs_per_group=self.obs_per_group, obs_time=10.0 * u.minute
        )
        data = raw_data.select(obs_index=0)
        # Cut all but one detector
        alldets = data.obs[0].local_detectors
        data.obs[0].update_local_detector_flags({x: 1 for x in alldets[1:]})

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

        # Create fake polarized sky signal
        skyfile = os.path.join(testdir, "input_map.fits")
        map_key = "fake_map"
        create_fake_healpix_scanned_tod(
            data,
            pixels,
            weights,
            skyfile,
            "pixel_dist",
            map_key=map_key,
            fwhm=30.0 * u.arcmin,
            lmax=3 * pixels.nside,
            I_scale=0.001,
            Q_scale=0.0001,
            U_scale=0.0001,
            det_data=defaults.det_data,
        )

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
            debug_plots=testdir,
        )

        tmatrix = ops.TemplateMatrix(templates=[tmpl])

        ops.Copy(detdata=[(defaults.det_data, "input_signal")]).apply(data)

        gt = GlobalTimers.get()
        gt.stop_all()
        gt.clear_all()

        # Map maker
        mapper = ops.MapMaker(
            name="toastmap",
            det_data=defaults.det_data,
            binning=binner,
            template_matrix=tmatrix,
            solve_rcond_threshold=1.0e-4,
            map_rcond_threshold=1.0e-4,
            iter_max=50,
            write_hits=True,
            write_map=True,
            write_binmap=True,
            write_noiseweighted_map=False,
            write_cov=False,
            write_rcond=False,
            write_solver_products=True,
            write_solver_amplitudes=True,
            keep_final_products=False,
            output_dir=testdir,
            save_cleaned=True,
            overwrite_cleaned=False,
        )

        # Make the map
        mapper.apply(data)

        alltimers = gather_timers(comm=data.comm.comm_world)
        if data.comm.world_rank == 0:
            out = os.path.join(testdir, "timing")
            dump_timers(alltimers, out)

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
        pars["write_binmap"] = "T"
        pars["write_matrix"] = "F"
        pars["write_wcov"] = "F"
        pars["write_hits"] = "T"
        pars["write_base"] = "F"
        pars["write_mask"] = "T"
        pars["kfilter"] = "T"
        pars["precond_width_min"] = 1
        pars["precond_width_max"] = 1
        pars["use_cgprecond"] = "F"
        pars["use_fprecond"] = "F"
        pars["info"] = 3
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

        # Compare outputs
        self.compare_madam_outputs(testdir, data)

        del data
        close_data(raw_data)

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

    #     # Create fake polarized sky signal
    #     skyfile = os.path.join(testdir, "input_map.fits")
    #     map_key = "fake_map"
    #     create_fake_healpix_scanned_tod(
    #         data,
    #         pixels,
    #         weights,
    #         skyfile,
    #         "pixel_dist",
    #         map_key=map_key,
    #         fwhm=30.0 * u.arcmin,
    #         lmax=3 * pixels.nside,
    #         I_scale=0.001,
    #         Q_scale=0.0001,
    #         U_scale=0.0001,
    #         det_data=defaults.det_data,
    #     )

    #     # Now clear the pointing and reset things for use with the mapmaking test later
    #     delete_pointing = ops.Delete(
    #         detdata=[pixels.pixels, weights.weights, detpointing.quats]
    #     )
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
    #         full_pointing=True,
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
    #         noise_model=default_model.noise_model,
    #         step_time=step_seconds * u.second,
    #         use_noise_prior=True,
    #         precond_width=10,
    #         # debug_plots=testdir,
    #     )

    #     tmatrix = ops.TemplateMatrix(templates=[tmpl])

    #     gt = GlobalTimers.get()
    #     gt.stop_all()
    #     gt.clear_all()

    #     # Map maker
    #     mapper = ops.MapMaker(
    #         name="toastmap",
    #         det_data=defaults.det_data,
    #         binning=binner,
    #         template_matrix=tmatrix,
    #         solve_rcond_threshold=1.0e-4,
    #         map_rcond_threshold=1.0e-4,
    #         iter_max=50,
    #         write_hits=True,
    #         write_map=True,
    #         write_cov=False,
    #         write_rcond=False,
    #         keep_final_products=True,
    #     )

    #     # Make the map
    #     mapper.apply(data)

    #     alltimers = gather_timers(comm=data.comm.comm_world)
    #     if data.comm.world_rank == 0:
    #         out = os.path.join(testdir, "timing")
    #         dump_timers(alltimers, out)

    #     # Outputs
    #     toast_hits = "toastmap_hits"
    #     toast_map = "toastmap_map"

    #     # Write map to disk so we can load the whole thing on one process.

    #     toast_hit_path = os.path.join(testdir, "toast_hits.fits")
    #     toast_map_path = os.path.join(testdir, "toast_map.fits")
    #     data[toast_map].write(toast_map_path)
    #     data[toast_hits].write(toast_hit_path)

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
    #     pars["pixmode_cross"] = 2  # Use rcond threshold
    #     pars["pixlim_map"] = 1.0e-4
    #     pars["pixmode_map"] = 2  # Use rcond threshold
    #     pars["write_map"] = "T"
    #     pars["write_binmap"] = "F"
    #     pars["write_matrix"] = "F"
    #     pars["write_wcov"] = "F"
    #     pars["write_hits"] = "T"
    #     pars["write_base"] = "T"
    #     pars["kfilter"] = "T"
    #     pars["precond_width_min"] = 10
    #     pars["precond_width_max"] = 10
    #     pars["use_cgprecond"] = "F"
    #     pars["use_fprecond"] = "F"
    #     pars["info"] = 3
    #     pars["path_output"] = testdir

    #     madam = ops.Madam(
    #         params=pars,
    #         det_data=defaults.det_data,
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

    #         input_map = hp.read_map(
    #             os.path.join(testdir, "input_map.fits"), field=None, nest=True
    #         )
    #         toast_map = hp.read_map(toast_map_path, field=None, nest=True)
    #         madam_map = hp.read_map(madam_map_path, field=None, nest=True)
    #         # Set madam unhit pixels to zero
    #         for stokes, ststr in zip(range(3), ["I", "Q", "U"]):
    #             mask = hp.mask_bad(madam_map[stokes])
    #             madam_map[stokes][mask] = 0.0
    #             input_map[stokes][mask] = 0.0
    #             diff_map = toast_map[stokes] - madam_map[stokes]
    #             madam_diff_truth = madam_map[stokes] - input_map[stokes]
    #             toast_diff_truth = toast_map[stokes] - input_map[stokes]
    #             # print("diff map {} has rms {}".format(ststr, np.std(diff_map)))
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
    #             outfile = os.path.join(testdir, "madam_diff_truth_{}.png".format(ststr))
    #             hp.mollview(madam_diff_truth, xsize=1600, nest=True)
    #             plt.savefig(outfile)
    #             plt.close()
    #             outfile = os.path.join(testdir, "toast_diff_truth_{}.png".format(ststr))
    #             hp.mollview(toast_diff_truth, xsize=1600, nest=True)
    #             plt.savefig(outfile)
    #             plt.close()

    #             if not np.allclose(
    #                 toast_map[stokes], madam_map[stokes], atol=0.1, rtol=0.05
    #             ):
    #                 fail = True

    #     if data.comm.comm_world is not None:
    #         fail = data.comm.comm_world.bcast(fail, root=0)

    #     self.assertFalse(fail)

    #     close_data(data)
