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

from .. import templates

from ..pixels import PixelDistribution, PixelData

from ..pixels_io import write_healpix_fits

from ._helpers import create_outdir, create_satellite_data, create_fake_sky


def add_flags(data):
    """ Add some flagging """
    shared_flags = "flags"
    det_flags = "flags"
    for obs in data.obs:
        common_flags = obs.shared[shared_flags]
        common_flags[::3] |= 255
        obs.detdata.ensure(det_flags, dtype=np.uint8)
        for idet, det in enumerate(obs.select_local_detectors()):
            flags = obs.detdata[det_flags][idet]
            flags[::2] |= 255
    return shared_flags, det_flags


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
        pointing = ops.PointingHealpix(
            nside=64,
            mode="IQU",
            hwp_angle="hwp_angle",
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pointing.apply(data)

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, "pixel_dist", "fake_map")

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data="signal",
            pixels=pointing.pixels,
            weights=pointing.weights,
            map_key="fake_map",
        )
        scanner.apply(data)

        shared_flags, det_flags = add_flags(data)

        # Now clear the pointing and reset things for use with the mapmaking test later
        delete_pointing = ops.Delete(detdata=[pointing.pixels, pointing.weights])
        delete_pointing.apply(data)
        pointing.create_dist = None

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise and accumulate to signal
        sim_noise = ops.SimNoise(
            noise_model=default_model.noise_model, det_data="signal"
        )
        sim_noise.apply(data)

        # Set up binning operator for solving
        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            shared_flags=shared_flags,
            det_flags=det_flags,
            pointing=pointing,
            noise_model=default_model.noise_model,
        )

        # Set up template matrix with just an offset template.

        # Use 1/10 of an observation as the baseline length.  Make it not evenly
        # divisible in order to test handling of the final amplitude.
        ob_time = data.obs[0].shared["times"][-1] - data.obs[0].shared["times"][0]
        step_seconds = float(int(ob_time / 10.0))
        tmpl = templates.Offset(
            times="times",
            noise_model=default_model.noise_model,
            step_time=step_seconds * u.second,
        )

        tmatrix = ops.TemplateMatrix(templates=[tmpl])

        # Map maker
        mapper = ops.MapMaker(
            name="test1",
            det_data="signal",
            shared_flags=shared_flags,
            det_flags=det_flags,
            binning=binner,
            template_matrix=tmatrix,
        )

        # Make the map
        mapper.apply(data)

        # Check that we can also run in full-memory mode
        pointing.apply(data)
        binner.full_pointing = True
        mapper.name = "test2"
        mapper.apply(data)

        del data
        return

    def test_flagging(self):
        testdir = os.path.join(self.outdir, "compare_flagging")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir)

        # Create a fake satellite data set for testing
        data = create_satellite_data(
            self.comm, obs_per_group=self.obs_per_group, obs_time=10.0 * u.minute
        )

        # Create some sky signal timestreams.
        detpointing = ops.PointingDetectorSimple()
        pointing = ops.PointingHealpix(
            nside=16,
            nest=True,
            mode="IQU",
            hwp_angle="hwp_angle",
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pointing.apply(data)

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, "pixel_dist", "fake_map")

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data="signal",
            pixels=pointing.pixels,
            weights=pointing.weights,
            map_key="fake_map",
        )
        scanner.apply(data)

        # Now clear the pointing and reset things for use with the mapmaking test later
        delete_pointing = ops.Delete(detdata=[pointing.pixels, pointing.weights])
        delete_pointing.apply(data)
        pointing.create_dist = None

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise and accumulate to signal
        sim_noise = ops.SimNoise(
            noise_model=default_model.noise_model, det_data="signal"
        )
        sim_noise.apply(data)

        # Set up binning operator for solving
        unflagged_binner = ops.BinMap(
            pixel_dist="pixel_dist",
            pointing=pointing,
            noise_model=default_model.noise_model,
        )

        # Set up template matrix with just an offset template.

        # Use 1/10 of an observation as the baseline length.  Make it not evenly
        # divisible in order to test handling of the final amplitude.
        ob_time = data.obs[0].shared["times"][-1] - data.obs[0].shared["times"][0]
        # step_seconds = float(int(ob_time / 10.0))
        step_seconds = 5.0
        tmpl = templates.Offset(
            times="times",
            noise_model=default_model.noise_model,
            step_time=step_seconds * u.second,
        )

        tmatrix = ops.TemplateMatrix(templates=[tmpl])

        # Map maker
        unflagged_mapper = ops.MapMaker(
            name="unflagged_toastmap",
            det_data="signal",
            binning=unflagged_binner,
            template_matrix=tmatrix,
            solve_rcond_threshold=1.0e-6,
            map_rcond_threshold=1.0e-6,
            iter_max=10,
        )

        # Make the map
        unflagged_mapper.apply(data)

        # Add flags
        shared_flags, det_flags = add_flags(data)

        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            shared_flags=shared_flags,
            shared_flags_mask=255,
            det_flags=det_flags,
            det_flag_mask=255,
            pointing=pointing,
            noise_model=default_model.noise_model,
        )
        mapper = ops.MapMaker(
            name="flagged_toastmap",
            det_data="signal",
            shared_flags=shared_flags,
            det_flags=det_flags,
            binning=binner,
            template_matrix=tmatrix,
            solve_rcond_threshold=1.0e-6,
            map_rcond_threshold=1.0e-6,
            iter_max=10,
        )

        # Make the map
        mapper.apply(data)

        # Outputs
        unflagged_hits = "unflagged_toastmap_hits"
        flagged_hits = "flagged_toastmap_hits"

        # Write map to disk so we can load the whole thing on one process.

        unflagged_hit_path = os.path.join(testdir, "unflagged_hits.fits")
        flagged_hit_path = os.path.join(testdir, "flagged_hits.fits")
        write_healpix_fits(data[unflagged_hits], unflagged_hit_path, nest=True)
        write_healpix_fits(data[flagged_hits], flagged_hit_path, nest=True)

        fail = False

        if data.comm.world_rank == 0:
            set_matplotlib_backend()
            import matplotlib.pyplot as plt

            # Compare hit maps

            unflagged_hits = hp.read_map(unflagged_hit_path, field=None, nest=True)
            flagged_hits = hp.read_map(flagged_hit_path, field=None, nest=True)
            diff_hits = unflagged_hits - flagged_hits

            outfile = os.path.join(testdir, "unflagged_hits.png")
            hp.mollview(unflagged_hits, xsize=1600, nest=True)
            plt.savefig(outfile)
            plt.close()
            outfile = os.path.join(testdir, "flagged_hits.png")
            hp.mollview(flagged_hits, xsize=1600, nest=True)
            plt.savefig(outfile)
            plt.close()
            outfile = os.path.join(testdir, "diff_hits.png")
            hp.mollview(diff_hits, xsize=1600, nest=True)
            plt.savefig(outfile)
            plt.close()

            if np.sum(unflagged_hits) == np.sum(flagged_hits):
                fail = True

        if data.comm.comm_world is not None:
            fail = data.comm.comm_world.bcast(fail, root=0)

        self.assertFalse(fail)

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
        pointing = ops.PointingHealpix(
            nside=16,
            nest=True,
            mode="IQU",
            hwp_angle="hwp_angle",
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pointing.apply(data)

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, "pixel_dist", "fake_map")

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data="signal",
            pixels=pointing.pixels,
            weights=pointing.weights,
            map_key="fake_map",
        )
        scanner.apply(data)

        shared_flags, det_flags = add_flags(data)

        # Now clear the pointing and reset things for use with the mapmaking test later
        delete_pointing = ops.Delete(detdata=[pointing.pixels, pointing.weights])
        delete_pointing.apply(data)
        pointing.create_dist = None

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise and accumulate to signal
        sim_noise = ops.SimNoise(
            noise_model=default_model.noise_model, det_data="signal"
        )
        sim_noise.apply(data)

        # Set up binning operator for solving
        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            shared_flags=shared_flags,
            det_flags=det_flags,
            pointing=pointing,
            noise_model=default_model.noise_model,
        )

        # Set up template matrix with just an offset template.

        # Use 1/10 of an observation as the baseline length.  Make it not evenly
        # divisible in order to test handling of the final amplitude.
        ob_time = data.obs[0].shared["times"][-1] - data.obs[0].shared["times"][0]
        # step_seconds = float(int(ob_time / 10.0))
        step_seconds = 5.0
        tmpl = templates.Offset(
            times="times",
            noise_model=default_model.noise_model,
            step_time=step_seconds * u.second,
        )

        tmatrix = ops.TemplateMatrix(templates=[tmpl])

        # Map maker
        mapper = ops.MapMaker(
            name="toastmap",
            det_data="signal",
            shared_flags=shared_flags,
            det_flags=det_flags,
            binning=binner,
            template_matrix=tmatrix,
            solve_rcond_threshold=1.0e-6,
            map_rcond_threshold=1.0e-6,
            iter_max=10,
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
        pars["iter_max"] = 10
        pars["base_first"] = step_seconds
        pars["fsample"] = sample_rate
        pars["nside_map"] = pointing.nside
        pars["nside_cross"] = pointing.nside
        pars["nside_submap"] = min(8, pointing.nside)
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
            det_data="signal",
            shared_flags=shared_flags,
            det_flags=det_flags,
            pixels=pointing.pixels,
            weights=pointing.weights,
            pixels_nested=pointing.nest,
            noise_model="noise_model",
        )

        # Generate persistent pointing
        pointing.apply(data)

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
                    fail = True

        if data.comm.comm_world is not None:
            fail = data.comm.comm_world.bcast(fail, root=0)

        self.assertFalse(fail)

        del data
        return

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
        pointing = ops.PointingHealpix(
            nside=16,
            nest=True,
            mode="IQU",
            hwp_angle="hwp_angle",
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pointing.apply(data)

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, "pixel_dist", "fake_map")

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data="signal",
            pixels=pointing.pixels,
            weights=pointing.weights,
            map_key="fake_map",
        )
        scanner.apply(data)

        # Now clear the pointing and reset things for use with the mapmaking test later
        delete_pointing = ops.Delete(detdata=[pointing.pixels, pointing.weights])
        delete_pointing.apply(data)
        pointing.create_dist = None

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise and accumulate to signal
        sim_noise = ops.SimNoise(
            noise_model=default_model.noise_model, det_data="signal"
        )
        sim_noise.apply(data)

        # Set up binning operator for solving
        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            pointing=pointing,
            noise_model=default_model.noise_model,
        )

        # Set up template matrix with just an offset template.

        # Use 1/10 of an observation as the baseline length.  Make it not evenly
        # divisible in order to test handling of the final amplitude.
        ob_time = data.obs[0].shared["times"][-1] - data.obs[0].shared["times"][0]
        # step_seconds = float(int(ob_time / 10.0))
        step_seconds = 5.0
        tmpl = templates.Offset(
            times="times",
            noise_model=default_model.noise_model,
            step_time=step_seconds * u.second,
            use_noise_prior=True,
            precond_width=1,
        )

        tmatrix = ops.TemplateMatrix(templates=[tmpl])

        # Map maker
        mapper = ops.MapMaker(
            name="toastmap",
            det_data="signal",
            binning=binner,
            template_matrix=tmatrix,
            solve_rcond_threshold=1.0e-4,
            map_rcond_threshold=1.0e-4,
            iter_max=50,
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
        pars["nside_map"] = pointing.nside
        pars["nside_cross"] = pointing.nside
        pars["nside_submap"] = min(8, pointing.nside)
        pars["good_baseline_fraction"] = tmpl.good_fraction
        pars["pixlim_cross"] = 1.0e-4
        pars["pixlim_map"] = 1.0e-4
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
            det_data="signal",
            pixels=pointing.pixels,
            weights=pointing.weights,
            pixels_nested=pointing.nest,
            noise_model="noise_model",
        )

        # Generate persistent pointing
        pointing.apply(data)

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

        del data
        return

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
        pointing = ops.PointingHealpix(
            nside=16,
            nest=True,
            mode="IQU",
            hwp_angle="hwp_angle",
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pointing.apply(data)

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, "pixel_dist", "fake_map")

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data="signal",
            pixels=pointing.pixels,
            weights=pointing.weights,
            map_key="fake_map",
        )
        scanner.apply(data)

        # Now clear the pointing and reset things for use with the mapmaking test later
        delete_pointing = ops.Delete(detdata=[pointing.pixels, pointing.weights])
        delete_pointing.apply(data)
        pointing.create_dist = None

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate noise and accumulate to signal
        sim_noise = ops.SimNoise(
            noise_model=default_model.noise_model, det_data="signal"
        )
        sim_noise.apply(data)

        # Set up binning operator for solving
        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            pointing=pointing,
            noise_model=default_model.noise_model,
        )

        # Set up template matrix with just an offset template.

        # Use 1/10 of an observation as the baseline length.  Make it not evenly
        # divisible in order to test handling of the final amplitude.
        ob_time = data.obs[0].shared["times"][-1] - data.obs[0].shared["times"][0]
        # step_seconds = float(int(ob_time / 10.0))
        step_seconds = 5.0
        tmpl = templates.Offset(
            times="times",
            noise_model=default_model.noise_model,
            step_time=step_seconds * u.second,
            use_noise_prior=True,
            precond_width=10,
        )

        tmatrix = ops.TemplateMatrix(templates=[tmpl])

        # Map maker
        mapper = ops.MapMaker(
            name="toastmap",
            det_data="signal",
            binning=binner,
            template_matrix=tmatrix,
            solve_rcond_threshold=1.0e-4,
            map_rcond_threshold=1.0e-4,
            iter_max=50,
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
        pars["nside_map"] = pointing.nside
        pars["nside_cross"] = pointing.nside
        pars["nside_submap"] = min(8, pointing.nside)
        pars["good_baseline_fraction"] = tmpl.good_fraction
        pars["pixlim_cross"] = 1.0e-4
        pars["pixlim_map"] = 1.0e-4
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
            det_data="signal",
            pixels=pointing.pixels,
            weights=pointing.weights,
            pixels_nested=pointing.nest,
            noise_model="noise_model",
        )

        # Generate persistent pointing
        pointing.apply(data)

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

        del data
        return
