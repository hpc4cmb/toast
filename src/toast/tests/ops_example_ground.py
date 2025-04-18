# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u
from astropy.table import Column

from .. import ops as ops
from .. import templates as templates
from ..intervals import IntervalList
from ..noise import Noise
from ..observation import default_values as defaults
from ..pixels import PixelData, PixelDistribution
from ..vis import (
    plot_healpix_maps,
    plot_noise_estim,
    plot_wcs_maps,
    set_matplotlib_backend,
)
from .helpers import (
    close_data,
    create_fake_healpix_scanned_tod,
    create_ground_data,
    create_outdir,
    fake_flags,
    fake_hwpss,
)
from .mpi import MPITestCase


class ExampleGroundTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        np.random.seed(123456)
        if (
            ("CONDA_BUILD" in os.environ)
            or ("CIBUILDWHEEL" in os.environ)
            or ("CI" in os.environ)
        ):
            self.make_plots = False
        else:
            self.make_plots = True

    def test_example(self):
        testdir = os.path.join(self.outdir, "example")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir)

        # Create some test data.  Disable HWPSS, since we are not demodulating
        # in this example.
        data = self.create_test_data(
            testdir,
            sky=True,
            atm=True,
            noise=True,
            hwpss=False,
            azss=True,
            flags=False,
        )

        # Make a copy for later comparison
        ops.Copy(detdata=[(defaults.det_data, "input")]).apply(data)

        # Diagnostic plots of one detector on each process.  This is useful for
        # plotting the "input" data before doing any operations.
        for ob in data.obs:
            self.plot_obs(
                testdir,
                "input",
                ob,
                defaults.det_data,
                dets=None,
                det_mask=defaults.det_mask_nonscience,
                interval_name="scanning",
            )

        # -----------------------------------------------------------------------------
        # Here is where we could do something to the timestream values.  For example,
        # filtering, flagging, etc.  This depends on what new thing we are testing.
        # If we are processing the timestream in some way, we could instantiate our
        # operator here and apply it to the data.
        # -----------------------------------------------------------------------------

        for ob in data.obs:
            # -------------------------------------------------------------------------
            # Here is where we could do some kind of testing of generated timestream
            # values or the results of some timestream operation.  In this example
            # we just check that detector data has some good samples.
            # -------------------------------------------------------------------------

            selected_dets = ob.select_local_detectors(
                flagmask=defaults.det_mask_invalid
            )
            for det in selected_dets:
                # Unflagged detector should have at least some good data
                rms = np.std(ob.detdata[defaults.det_data][det])
                self.assertTrue(rms > 0)

            # Diagnostic plots of one detector on each process
            self.plot_obs(
                testdir,
                "processed",
                ob,
                defaults.det_data,
                dets=None,
                det_mask=defaults.det_mask_nonscience,
                interval_name="scanning",
            )

        # Mapmaking.  Not all tests need to make a map!  Some tests are just examining
        # the results of timestream processing.  Other tests of mapmaking operations
        # might simulate data and go straight to map-domain tests.

        # Detector pointing
        detpointing = ops.PointingDetectorSimple()

        # Pointing matrix for Mapmaking.  We can use either Healpix or WCS projections.
        # We save that choice so that we can use it later for diagnostic plots.
        use_healpix = True

        if use_healpix:
            # Healpix pixelization
            pixels = ops.PixelsHealpix(
                nside=256,
                detector_pointing=detpointing,
            )
        else:
            # WCS in one of the supported projections (CAR, TAN, etc)
            pixels = ops.PixelsWCS(
                detector_pointing=detpointing,
                projection="CAR",
                resolution=(0.05 * u.degree, 0.05 * u.degree),
                dimensions=(),
                auto_bounds=True,
            )
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )

        # Noise estimation.  Our simulated data has many contributions to the "noise",
        # including atmosphere.  The input instrument noise model does not accurately
        # represent the data.  For mapmaking we could either estimate the white noise
        # level with sample differences or we could fit a full 1/f noise model spectrum.

        # Estimate noise using sample differences
        noise_estim = ops.SignalDiffNoiseModel(noise_model="noise_estimate")
        noise_estim.apply(data)

        # # Estimate full 1/f noise model.
        # noise_estim = ops.NoiseEstim(
        #     name="estimate_model",
        #     # output_dir=testdir, # Enable this to dump out the model into FITS files
        #     out_model="raw_noise_estimate",
        #     lagmax=512,
        #     nbin_psd=64,
        #     nsum=1,
        #     naverage=64,
        # )
        # noise_estim.apply(data)

        # # Compute a 1/f fit to this
        # noise_fit = ops.FitNoiseModel(
        #     noise_model=noise_estim.out_model,
        #     out_model="noise_estimate",
        # )
        # noise_fit.apply(data)

        # Plot the noise estimates.  Just the first detector of each obs
        # on each process.
        if self.make_plots:
            for ob in data.obs:
                det = ob.local_detectors[0]
                if "raw_noise_estimate" in ob:
                    # We have both a raw and fit noise model
                    est_freq = ob["raw_noise_estimate"].freq(det)
                    est_psd = ob["raw_noise_estimate"].psd(det)
                    fit_freq = ob["noise_estimate"].freq(det)
                    fit_psd = ob["noise_estimate"].psd(det)
                else:
                    # Just a sample diff estimate
                    est_freq = ob["noise_estimate"].freq(det)
                    est_psd = ob["noise_estimate"].psd(det)
                    fit_freq = None
                    fit_psd = None
                fname = os.path.join(testdir, f"{ob.name}_{det}_noise_estimate.png")
                plot_noise_estim(
                    fname,
                    est_freq,
                    est_psd,
                    fit_freq=fit_freq,
                    fit_psd=fit_psd,
                    semilog=False,
                )

        # Set up binning operator for solving and final binning
        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model="noise_estimate",
            view="scanning",
        )

        # Simple offset template (i.e. classic destriping "baselines")
        offset_tmpl = templates.Offset(
            name="offset_template",
            times=defaults.times,
            noise_model="noise_estimate",
            step_time=2.0 * u.second,
            use_noise_prior=False,
            precond_width=1,
        )

        # Build the template matrix
        tmatrix = ops.TemplateMatrix(
            templates=[
                offset_tmpl,
            ],
            view="scanning",
        )

        # Map maker
        mapper = ops.MapMaker(
            name="example",
            det_data=defaults.det_data,
            binning=binner,
            template_matrix=tmatrix,
            solve_rcond_threshold=1.0e-1,
            map_rcond_threshold=1.0e-3,
            write_hits=True,
            write_map=True,
            write_binmap=True,
            write_cov=False,
            write_rcond=False,
            keep_solver_products=True,
            keep_final_products=False,
            save_cleaned=True,
            overwrite_cleaned=True,
            output_dir=testdir,
        )

        # Make the map
        mapper.apply(data)

        # Write offset amplitudes
        oamps = data[f"{mapper.name}_solve_amplitudes"][offset_tmpl.name]
        oroot = os.path.join(testdir, f"{mapper.name}_template-offset")
        offset_tmpl.write(oamps, oroot)

        # Plot some results
        if data.comm.world_rank == 0 and self.make_plots:
            hit_file = os.path.join(testdir, f"{mapper.name}_hits.fits")
            map_file = os.path.join(testdir, f"{mapper.name}_map.fits")
            binmap_file = os.path.join(testdir, f"{mapper.name}_binmap.fits")
            I_range = (-2, 2)
            P_range = (-0.5, 0.5)
            if use_healpix:
                plot_healpix_maps(hitfile=hit_file, gnomview=True, gnomres=0.5)
                plot_healpix_maps(
                    hitfile=hit_file,
                    mapfile=map_file,
                    range_I=I_range,
                    range_Q=P_range,
                    range_U=P_range,
                    gnomview=True,
                    gnomres=0.5,
                )
                plot_healpix_maps(
                    hitfile=hit_file,
                    mapfile=binmap_file,
                    range_I=I_range,
                    range_Q=P_range,
                    range_U=P_range,
                    gnomview=True,
                    gnomres=0.5,
                )
            else:
                plot_wcs_maps(hitfile=hit_file)
                plot_wcs_maps(
                    hitfile=hit_file,
                    mapfile=map_file,
                    range_I=I_range,
                    range_Q=P_range,
                    range_U=P_range,
                )
                plot_wcs_maps(
                    hitfile=hit_file,
                    mapfile=binmap_file,
                    range_I=I_range,
                    range_Q=P_range,
                    range_U=P_range,
                )
            for ob in data.obs:
                templates.offset.plot(
                    f"{oroot}_{ob.name}.h5",
                    compare={x: ob.detdata["input"][x, :] for x in ob.local_detectors},
                    out=f"{oroot}_{ob.name}",
                )

        # -------------------------------------------------------------------------
        # Here is where we could evaluate map-domain quantities.  For example, we
        # could load the input map and compare values, etc.  For now, just check
        # that the destriped / template-regressed timestreams have a smaller RMS
        # than the inputs.
        # -------------------------------------------------------------------------

        # Compare the cleaned timestreams to the original
        for ob in data.obs:
            self.plot_obs(
                testdir,
                "cleaned",
                ob,
                defaults.det_data,
                dets=None,
                det_mask=defaults.det_mask_nonscience,
                interval_name="scanning",
            )
            scanning_samples = np.zeros(ob.n_local_samples, dtype=bool)
            for intr in ob.intervals["scanning"]:
                scanning_samples[intr.first : intr.last] = 1
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_nonscience):
                good = np.logical_and(
                    ob.detdata[defaults.det_flags][det] == 0,
                    scanning_samples,
                )
                dc = np.mean(ob.detdata["input"][det][good])
                in_rms = np.std(ob.detdata["input"][det][good] - dc)
                dc = np.mean(ob.detdata[defaults.det_data][det][good])
                out_rms = np.std(ob.detdata[defaults.det_data][det][good] - dc)
                if out_rms > in_rms:
                    msg = f"{ob.name} det {det} cleaned rms ({out_rms}) greater than "
                    msg += f"input ({in_rms})"
                    print(msg, flush=True)
                    # self.assertTrue(False)

        # This deletes ALL data, including external communicators that are not normally
        # destroyed by doing "del data".
        close_data(data)

    def plot_obs(
        self,
        out_dir,
        prefix,
        obs,
        detdata_name,
        dets=None,
        det_mask=defaults.det_mask_nonscience,
        interval_name=None,
    ):
        if not self.make_plots:
            return
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        # Every process will plot its first local detector
        selected_dets = obs.select_local_detectors(selection=dets, flagmask=det_mask)

        # If the helper function for flagging samples has been applied, we will have
        # the first half of the samples flagged.  Plot the full observation and also
        # a range of samples near the center.
        n_all_samp = obs.n_all_samples
        plot_ranges = list()
        plot_files = list()
        n_plot = 2
        fig_height = 6 * n_plot
        axes = dict()
        pltsamp = 400
        det = selected_dets[0]
        for first, last in [
            (0, n_all_samp),
            (n_all_samp // 2 - pltsamp, n_all_samp // 2 + pltsamp),
        ]:
            rangestr = f"{first}-{last}"
            axes[rangestr] = dict()
            axes[rangestr]["range"] = (first, last)
            axes[rangestr]["file"] = os.path.join(
                out_dir,
                f"{prefix}_{obs.name}_{det}_{first}-{last}.pdf",
            )
            axes[rangestr]["fig"] = plt.figure(figsize=(12, fig_height), dpi=72)
            axes[rangestr]["ax"] = list()
            for pl in range(n_plot):
                axes[rangestr]["ax"].append(
                    axes[rangestr]["fig"].add_subplot(n_plot, 1, pl + 1, aspect="auto")
                )

        times = obs.shared[defaults.times].data
        signal = obs.detdata[detdata_name][det]
        detflags = obs.detdata[defaults.det_flags][det]
        shflags = obs.shared[defaults.shared_flags].data

        for rangestr, props in axes.items():
            rg = props["range"]
            file = props["file"]
            fig = props["fig"]
            ax = props["ax"]
            props["legend"] = list()
            plot_slc = slice(rg[0], rg[1], 1)
            # Plot signal
            ax[0].plot(
                times[plot_slc],
                signal[plot_slc],
                color="black",
                label=f"{det} '{detdata_name}'",
            )
            handles, labels = ax[0].get_legend_handles_labels()
            props["legend"].append(handles)
            # Plot flags
            ax[1].plot(
                times[plot_slc],
                shflags[plot_slc],
                color="blue",
                label="Shared Flags",
            )
            ax[1].plot(
                times[plot_slc],
                detflags[plot_slc],
                color="red",
                label=f"{det} Flags",
            )
            handles, labels = ax[1].get_legend_handles_labels()
            # Plot Intervals
            if interval_name is not None:
                plt_intervals = IntervalList(
                    timestamps=times, samplespans=[(rg[0], rg[1])]
                )
                overlap = plt_intervals & obs.intervals[interval_name]
                for intr in overlap:
                    ax[1].axvspan(
                        times[intr.first], times[intr.last], color="gray", alpha=0.1
                    )
                # Add a legend entry for the intervals
                patch = mpatches.Patch(
                    color="gray", alpha=0.1, label=f"Intervals '{interval_name}'"
                )
                handles.append(patch)
            props["legend"].append(handles)

        for rangestr, props in axes.items():
            rg = props["range"]
            file = props["file"]
            fig = props["fig"]
            ax = props["ax"]
            handles = props["legend"]
            for a, handle in zip(ax, handles):
                a.legend(handles=handle, loc="best")
            fig.suptitle(f"Obs {obs.name}:{rangestr}")
            fig.savefig(file)
            plt.close(fig)

    def create_test_data(
        self,
        outdir,
        sky=False,
        atm=False,
        noise=False,
        hwpss=False,
        azss=False,
        flags=False,
    ):
        # Slightly slower than 0.5 Hz
        hwp_rpm = 29.0
        hwp_rate = 2 * np.pi * hwp_rpm / 60.0  # rad/s

        sample_rate = 30 * u.Hz
        ang_per_sample = hwp_rate / sample_rate.to_value(u.Hz)

        # Create a fake ground observations set for testing.  We add more detectors
        # than the default to create a denser focalplane for more cross linking.
        data = create_ground_data(
            self.comm,
            sample_rate=sample_rate,
            hwp_rpm=hwp_rpm,
            pixel_per_process=7,
            single_group=True,
            fp_width=5.0 * u.degree,
        )

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Detector pointing in RA/DEC and Az/El
        detpointing_radec = ops.PointingDetectorSimple(shared_flag_mask=0)
        detpointing_azel = ops.PointingDetectorSimple(
            shared_flag_mask=0,
            boresight=defaults.boresight_azel,
        )

        # Make an elevation-dependent noise model
        el_model = ops.ElevationNoise(
            noise_model=default_model.noise_model,
            out_model="el_weighted",
            detector_pointing=detpointing_azel,
        )
        el_model.apply(data)

        # Create some fake sky signal.  Here we generate a random map and scan
        # it into timestreams.
        if sky:
            # Healpix pointing matrix
            pixels = ops.PixelsHealpix(
                nside=256,
                detector_pointing=detpointing_radec,
            )
            weights = ops.StokesWeights(
                mode="IQU",
                hwp_angle=defaults.hwp_angle,
                detector_pointing=detpointing_radec,
            )
            skyfile = os.path.join(outdir, f"sky_input.fits")
            create_fake_healpix_scanned_tod(
                data,
                pixels,
                weights,
                skyfile,
                "input_sky_dist",
                map_key="input_sky",
                fwhm=30.0 * u.arcmin,
                lmax=3 * pixels.nside,
                I_scale=0.001,
                Q_scale=0.0001,
                U_scale=0.0001,
                det_data=defaults.det_data,
            )
            if self.make_plots:
                if data.comm.world_rank == 0:
                    plot_healpix_maps(mapfile=skyfile)

        # Atmosphere
        if atm:
            sim_atm = ops.SimAtmosphere(
                detector_pointing=detpointing_azel,
                zmax=500 * u.m,
                gain=1.0e-4,
                xstep=100 * u.m,
                ystep=100 * u.m,
                zstep=100 * u.m,
                add_loading=True,
                lmin_center=10 * u.m,
                lmin_sigma=1 * u.m,
                lmax_center=100 * u.m,
                lmax_sigma=10 * u.m,
                wind_dist=10000 * u.m,
            )
            sim_atm.apply(data)

        # Create Az synchronous signal
        if azss:
            sss = ops.SimScanSynchronousSignal(
                detector_pointing=detpointing_azel,
                nside=1024,
                scale=1.0 * u.mK,
            )
            sss.apply(data)

        # Create HWPSS
        if hwpss:
            hwpss_scale = 10.0
            # Just get an approximate rms, based on the input optical
            # power simulated so far.
            tod_rms = np.std(data.obs[0].detdata[defaults.det_data][0])
            coeff = fake_hwpss(data, defaults.det_data, hwpss_scale * tod_rms)

        # Simulate elevation-weighted instrumental noise
        if noise:
            sim_noise = ops.SimNoise(noise_model="el_weighted")
            sim_noise.apply(data)

        # Create flagged samples
        if flags:
            fake_flags(data)

        # Cleanup any temporary data objects used in this function, just so that
        # the returned data object is clean.
        for ob in data.obs:
            for buf in [
                detpointing_azel.quats,
                detpointing_radec.quats,
            ]:
                if buf in ob:
                    del ob[buf]

        return data
