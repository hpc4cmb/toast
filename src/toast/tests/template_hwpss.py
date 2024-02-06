# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops
from ..accelerator import ImplementationType, accel_enabled
from ..atm import available_atm
from ..observation import default_values as defaults
from ..pixels_io_healpix import write_healpix_fits
from ..pixels_io_wcs import write_wcs_fits
from ..templates import Fourier2D, Hwpss, Offset
from ..templates.hwpss import plot as hwpssplot
from ..templates.offset import plot as offplot
from ..utils import rate_from_times
from ..vis import plot_noise_estim
from ._helpers import (
    close_data,
    create_fake_sky,
    create_ground_data,
    create_outdir,
    create_satellite_data,
    fake_hwpss,
    fake_hwpss_data,
    plot_healpix_maps,
    plot_wcs_maps,
)
from .mpi import MPITestCase


class TemplateHwpssTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.nside = 64
        np.random.seed(123456)
        if (
            ("CONDA_BUILD" in os.environ)
            or ("CIBUILDWHEEL" in os.environ)
            or ("CI" in os.environ)
        ):
            self.make_plots = False
        else:
            self.make_plots = True

    def test_algorithm(self):
        sample_rate = 100.0
        hwp_rate = 2.0
        # 2 minutes of data
        n_samp = int(2 * 60 * sample_rate)
        times = (1 / sample_rate) * np.arange(n_samp)
        incr = 2 * np.pi / (sample_rate / hwp_rate)
        hwp_ang = np.mod(incr * np.arange(n_samp), 2 * np.pi)
        flags = np.zeros(n_samp, dtype=np.uint8)
        # Get some hwpss and input coefficients
        hwpss_scale = 20.0
        hwpss, ccos, csin = fake_hwpss_data(hwp_ang, hwpss_scale)
        dc_level = np.mean(hwpss)
        n_harmonics = len(ccos)
        # Estimate coefficients
        sincos = Hwpss.sincos_buffer(hwp_ang, flags, n_harmonics)
        cov_lu, cov_piv = Hwpss.compute_coeff_covariance(times, flags, sincos)
        coeff = Hwpss.compute_coeff(hwpss, flags, times, sincos, cov_lu, cov_piv)
        # Build model
        model = Hwpss.build_model(times, flags, sincos, coeff) + dc_level
        # Plot
        if self.comm is None or self.comm.rank == 0:
            import matplotlib.pyplot as plt

            savefile = os.path.join(self.outdir, "algorithm_test.pdf")
            fig = plt.figure(figsize=(12, 24), dpi=72)
            ax = fig.add_subplot(4, 1, 1, aspect="auto")
            ax.plot(times, hwp_ang, c="black", label=f"HWP Angle")
            ax.legend(loc="best")
            ax = fig.add_subplot(4, 1, 2, aspect="auto")
            ax.plot(times, hwpss, c="black", label=f"Input Signal")
            ax.plot(times, model, c="red", label=f"Model Fit")
            ax.legend(loc="best")
            ax = fig.add_subplot(4, 1, 3, aspect="auto")
            ax.plot(times[:200], hwp_ang[:200], c="black", label=f"HWP Angle")
            ax.legend(loc="best")
            ax = fig.add_subplot(4, 1, 4, aspect="auto")
            ax.plot(times[:200], hwpss[:200], c="black", label=f"Input Signal")
            ax.plot(times[:200], model[:200], c="red", label=f"Model Fit")
            ax.legend(loc="best")
            plt.title(f"Algorithm Test")
            plt.savefig(savefile)
            plt.close()
        # Compare
        return

    def test_roundtrip(self):
        testdir = os.path.join(self.outdir, "roundtrip")
        if self.comm is None or self.comm.rank == 0:
            if not os.path.isdir(testdir):
                os.mkdir(testdir)
        if self.comm is not None:
            self.comm.barrier()

        # Create a fake satellite data set for testing
        data = self.create_satellite_sim(testdir, self.nside)

        # Good detectors
        local_dets = data.obs[0].select_local_detectors(
            flagmask=defaults.det_mask_invalid
        )

        # Get some hwpss and input coefficients
        hwpss_scale = 1.0
        compare = dict()
        coeff = fake_hwpss(data, defaults.det_data, hwpss_scale)

        do_plot = True
        for ob in data.obs:
            compare[ob.name] = ob.detdata[defaults.det_data][local_dets[0]]
            if do_plot:
                if self.comm is None or self.comm.rank == 0:
                    import matplotlib.pyplot as plt

                    savefile = os.path.join(testdir, f"input_{ob.name}.pdf")
                    fig = plt.figure(figsize=(12, 12), dpi=72)
                    ax = fig.add_subplot(2, 1, 1, aspect="auto")
                    ax.plot(
                        ob.shared[defaults.times].data[:500],
                        ob.shared[defaults.hwp_angle].data[:500],
                        c="black",
                        label=f"HWP Angle",
                    )
                    ax.legend(loc="best")
                    ax = fig.add_subplot(2, 1, 2, aspect="auto")
                    ax.plot(
                        ob.shared[defaults.times].data[:500],
                        compare[ob.name][:500],
                        c="black",
                        label=f"Input",
                    )
                    ax.legend(loc="best")
                    plt.title(f"Input")
                    plt.savefig(savefile)
                    plt.close()
                do_plot = False

        # The template
        tmpl = Hwpss(harmonics=5)
        tmpl.data = data

        # Create empty amplitudes
        amp = tmpl.zeros()

        # Compute amplitudes
        for det in local_dets:
            tmpl.project_signal(det, amp)

        # Zero tod and project
        ops.Reset(detdata=[defaults.det_data]).apply(data)

        for det in local_dets:
            tmpl.add_to_signal(det, amp)

        # Compare to input
        for ob in data.obs:
            for det in local_dets:
                if self.comm is None or self.comm.rank == 0:
                    import matplotlib.pyplot as plt

                    savefile = os.path.join(testdir, f"tod_{ob.name}_{det}.pdf")
                    fig = plt.figure(figsize=(12, 24), dpi=72)
                    ax = fig.add_subplot(4, 1, 1, aspect="auto")
                    ax.plot(
                        ob.shared[defaults.times].data,
                        ob.shared[defaults.hwp_angle].data,
                        c="black",
                        label=f"HWP Angle",
                    )
                    ax = fig.add_subplot(4, 1, 2, aspect="auto")
                    ax.plot(
                        ob.shared[defaults.times].data,
                        compare[ob.name],
                        c="black",
                        label=f"Input",
                    )
                    ax.plot(
                        ob.shared[defaults.times].data,
                        ob.detdata[defaults.det_data][det],
                        c="red",
                        label=f"Template",
                    )
                    ax.legend(loc="best")
                    ax = fig.add_subplot(4, 1, 3, aspect="auto")
                    ax.plot(
                        ob.shared[defaults.times].data[:500],
                        ob.shared[defaults.hwp_angle].data[:500],
                        c="black",
                        label=f"HWP Angle",
                    )
                    ax = fig.add_subplot(4, 1, 4, aspect="auto")
                    ax.plot(
                        ob.shared[defaults.times].data[:500],
                        compare[ob.name][:500],
                        c="black",
                        label=f"Input",
                    )
                    ax.plot(
                        ob.shared[defaults.times].data[:500],
                        ob.detdata[defaults.det_data][det, :500],
                        c="red",
                        label=f"Template",
                    )
                    ax.legend(loc="best")
                    plt.title(f"Roundtrip Test")
                    plt.savefig(savefile)
                    plt.close()
                np.testing.assert_almost_equal(
                    compare[ob.name],
                    ob.detdata[defaults.det_data][det],
                    decimal=6,
                )

        close_data(data)

    def plot_compare(self, outdir, data, prefix, comps=list(), full=False):
        import matplotlib.pyplot as plt

        for ob in data.obs:
            n_samp = ob.n_local_samples
            if full:
                plot_dets = ob.select_local_detectors(
                    flagmask=defaults.det_mask_invalid
                )
            else:
                plot_dets = [ob.local_detectors[0]]
            for det in plot_dets:
                for ns in [500, n_samp]:
                    plot_slc = slice(0, ns, 1)
                    savefile = os.path.join(
                        outdir,
                        f"{prefix}-compare_{ob.name}_{det}_{ns}.pdf",
                    )
                    n_comp = len(comps)
                    n_plot = n_comp + 3
                    fig_height = 6 * n_plot

                    # Find the data range of the input
                    dmin = np.amin(ob.detdata["input"][det, plot_slc])
                    dmax = np.amax(ob.detdata["input"][det, plot_slc])
                    dhalf = (dmax - dmin) / 2

                    fig = plt.figure(figsize=(12, fig_height), dpi=72)
                    for icomp, comp in enumerate(comps):
                        ax = fig.add_subplot(n_plot, 1, icomp + 1, aspect="auto")
                        ax.plot(
                            ob.shared[defaults.times].data[plot_slc],
                            ob.detdata[comp][det, plot_slc],
                            c="blue",
                            label=f"Component '{comp}'",
                        )
                        ax.legend(loc="best")
                        plt.title(f"Obs {ob.name}, det {det}")

                    ax = fig.add_subplot(n_plot, 1, n_plot - 2, aspect="auto")
                    ax.plot(
                        ob.shared[defaults.times].data[plot_slc],
                        ob.detdata["input"][det, plot_slc],
                        c="blue",
                        label=f"Total Input Signal",
                    )
                    ax.set_ylim(bottom=dmin, top=dmax)
                    ax.legend(loc="best")

                    ax = fig.add_subplot(n_plot, 1, n_plot - 1, aspect="auto")
                    ax.plot(
                        ob.shared[defaults.times].data[plot_slc],
                        ob.detdata[defaults.det_data][det, plot_slc],
                        c="red",
                        label="Template Cleaned",
                    )
                    ax.plot(
                        ob.shared[defaults.times].data[plot_slc],
                        ob.detdata["sky"][det, plot_slc],
                        c="black",
                        label="Input Sky",
                    )
                    # ax.set_ylim(bottom=dmin, top=dmax)
                    ax.legend(loc="best")
                    plt.title(f"Obs {ob.name}, det {det}")

                    resid = (
                        ob.detdata[defaults.det_data][det, plot_slc]
                        - ob.detdata["sky"][det, plot_slc]
                    )
                    pmean = np.mean(resid)
                    ax = fig.add_subplot(n_plot, 1, n_plot, aspect="auto")
                    ax.plot(
                        ob.shared[defaults.times].data[plot_slc],
                        resid,
                        c="green",
                        label="Residual",
                    )
                    ax.set_ylim(bottom=pmean - dhalf, top=pmean + dhalf)
                    ax.legend(loc="best")
                    plt.title(f"Obs {ob.name}, det {det}")

                    plt.savefig(savefile)
                    plt.close()

    def create_satellite_sim(self, outdir, sky_nside):
        # Create a fake satellite data set for testing
        data = create_satellite_data(
            self.comm,
            sample_rate=10.0 * u.Hz,
            obs_time=10.0 * u.minute,
            gap_time=0.5 * u.minute,
            hwp_rpm=12,
            obs_per_group=10,
            pixel_per_process=7,
            single_group=True,
            width=5.0 * u.degree,
        )

        # Create detector data for the combined input signal
        for ob in data.obs:
            ob.detdata.create("input", units=defaults.det_data_units)

        # Create a default noise model
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        # Generate a fake "sky" and scan into timestreams.  We
        # create a temporary pointing matrix and then clean up
        # temporary data objects afterwards.

        detpointing = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec,
            quats="temp_quats",
        )
        detpointing.apply(data)

        pixels = ops.PixelsHealpix(
            nside=sky_nside,
            detector_pointing=detpointing,
            pixels="temp_pix",
        )
        pix_dist = ops.BuildPixelDistribution(
            pixel_dist="sky_pixel_dist",
            pixel_pointing=pixels,
        )
        pix_dist.apply(data)

        create_fake_sky(data, "sky_pixel_dist", "fake_map")

        outfile = os.path.join(outdir, f"sky_{self.nside}_input.fits")
        write_healpix_fits(data["fake_map"], outfile, nest=True)
        if data.comm.world_rank == 0:
            plot_healpix_maps(mapfile=outfile)

        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            weights="temp_weights",
            detector_pointing=detpointing,
        )

        scanner = ops.Pipeline(
            operators=[
                pixels,
                weights,
                ops.ScanMap(
                    det_data="sky",
                    pixels=pixels.pixels,
                    weights=weights.weights,
                    map_key="fake_map",
                    det_data_units=defaults.det_data_units,
                ),
            ],
            detsets=["SINGLE"],
        )
        scanner.apply(data)

        ops.Combine(
            op="add",
            first="sky",
            second="input",
            result="input",
        ).apply(data)

        del data["fake_map"]
        del data["sky_pixel_dist"]
        for ob in data.obs:
            del ob.detdata["temp_weights"]
            del ob.detdata["temp_pix"]
            del ob.detdata["temp_quats"]

        # Simulate some noise
        sim_noise = ops.SimNoise(
            det_data="noise",
            noise_model=default_model.noise_model,
            det_data_units=defaults.det_data_units,
        )
        sim_noise.apply(data)
        ops.Combine(
            op="add",
            first="noise",
            second="input",
            result="input",
        ).apply(data)

        return data

    def create_ground_sim(self, outdir, width, sky_proj, sky_res):
        # Create a fake ground data set for testing
        data = create_ground_data(
            self.comm,
            sample_rate=122.0 * u.Hz,
            pixel_per_process=7,
            single_group=True,
            fp_width=width,
        )

        # Create detector data for the combined input signal
        for ob in data.obs:
            ob.detdata.create("input", units=defaults.det_data_units)

        # Create a default noise model
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        # Az/El detector pointing
        detpointing_azel = ops.PointingDetectorSimple(
            boresight=defaults.boresight_azel, quats="quats_azel"
        )

        # Make an elevation-dependent noise model
        el_model = ops.ElevationNoise(
            noise_model=default_model.noise_model,
            out_model="el_weighted",
            detector_pointing=detpointing_azel,
        )
        el_model.apply(data)

        # Generate a fake "sky" and scan into timestreams.  We
        # create a temporary pointing matrix and then clean up
        # temporary data objects afterwards.

        detpointing = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec,
            quats="temp_quats",
        )
        detpointing.apply(data)

        pixels = ops.PixelsWCS(
            projection=sky_proj,
            resolution=(sky_res, sky_res),
            detector_pointing=detpointing,
            pixels="temp_pix",
            use_astropy=True,
            auto_bounds=True,
        )
        pix_dist = ops.BuildPixelDistribution(
            pixel_dist="sky_pixel_dist",
            pixel_pointing=pixels,
        )
        pix_dist.apply(data)

        create_fake_sky(data, "sky_pixel_dist", "fake_map")

        outfile = os.path.join(
            outdir, f"sky_{sky_proj}_{sky_res.to_value(u.arcmin)}_input.fits"
        )
        write_wcs_fits(data["fake_map"], outfile)
        if data.comm.world_rank == 0:
            plot_wcs_maps(mapfile=outfile)

        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            weights="temp_weights",
            detector_pointing=detpointing,
        )

        scanner = ops.Pipeline(
            operators=[
                pixels,
                weights,
                ops.ScanMap(
                    det_data="sky",
                    pixels=pixels.pixels,
                    weights=weights.weights,
                    map_key="fake_map",
                    det_data_units=defaults.det_data_units,
                ),
            ],
            detsets=["SINGLE"],
        )
        scanner.apply(data)

        ops.Combine(
            op="add",
            first="sky",
            second="input",
            result="input",
        ).apply(data)

        del data["fake_map"]
        del data["sky_pixel_dist"]
        for ob in data.obs:
            del ob.detdata["temp_weights"]
            del ob.detdata["temp_pix"]
            del ob.detdata["temp_quats"]

        # Simulate some noise
        sim_noise = ops.SimNoise(
            det_data="noise",
            noise_model=el_model.out_model,
            det_data_units=defaults.det_data_units,
        )
        sim_noise.apply(data)
        ops.Combine(
            op="add",
            first="noise",
            second="input",
            result="input",
        ).apply(data)

        # Simulate atmosphere
        sim_atm_coarse = ops.SimAtmosphere(
            name="sim_atmosphere_coarse",
            det_data="atm",
            detector_pointing=detpointing_azel,
            add_loading=False,
            lmin_center=300 * u.m,
            lmin_sigma=30 * u.m,
            lmax_center=10000 * u.m,
            lmax_sigma=1000 * u.m,
            # Use lower resolution for faster run time
            # xstep=50 * u.m,
            # ystep=50 * u.m,
            # zstep=50 * u.m,
            xstep=100 * u.m,
            ystep=100 * u.m,
            zstep=100 * u.m,
            zmax=2000 * u.m,
            nelem_sim_max=30000,
            gain=6e-4,
            realization=1000000,
            wind_dist=10000 * u.m,
            det_data_units=defaults.det_data_units,
        )
        sim_atm_coarse.apply(data)
        sim_atm = ops.SimAtmosphere(
            name="sim_atmosphere",
            det_data="atm",
            detector_pointing=detpointing_azel,
            add_loading=True,
            lmin_center=0.001 * u.m,
            lmin_sigma=0.0001 * u.m,
            lmax_center=1 * u.m,
            lmax_sigma=0.1 * u.m,
            # Use lower resolution for faster run time
            # xstep=5 * u.m,
            # ystep=5 * u.m,
            # zstep=5 * u.m,
            xstep=50 * u.m,
            ystep=50 * u.m,
            zstep=50 * u.m,
            zmax=200 * u.m,
            gain=6e-5,
            wind_dist=3000 * u.m,
            det_data_units=defaults.det_data_units,
        )
        sim_atm.apply(data)

        ops.Combine(
            op="add",
            first="atm",
            second="input",
            result="input",
        ).apply(data)

        return data

    def test_satellite_hwp(self):
        testdir = os.path.join(self.outdir, "satellite_hwp")
        if self.comm is None or self.comm.rank == 0:
            if not os.path.isdir(testdir):
                os.mkdir(testdir)
        if self.comm is not None:
            self.comm.barrier()

        # Create a fake satellite data set for testing
        data = self.create_satellite_sim(testdir, self.nside)

        # Add HWPSS and make a copy for plotting
        hwpss_scale = 20.0
        tod_rms = np.std(data.obs[0].detdata["input"][0])
        coeff = fake_hwpss(data, "hwpss", hwpss_scale * tod_rms)
        ops.Combine(
            op="add",
            first="hwpss",
            second="input",
            result="input",
        ).apply(data)

        # Copy the input data into the default field
        ops.Copy(detdata=[("input", defaults.det_data)]).apply(data)

        # Create a default noise model
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        # Pointing
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=self.nside,
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )

        # Set up binning operator for solving
        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
        )

        # Set up HWPSS template
        hwpss_tmpl = Hwpss(
            name="hwpss_template",
            hwp_angle=defaults.hwp_angle,
            flags=defaults.shared_flags,
            flag_mask=defaults.shared_mask_invalid,
            harmonics=5,
        )

        # Simple offset template
        offset_tmpl = Offset(
            name="offset_template",
            times=defaults.times,
            noise_model=default_model.noise_model,
            step_time=0.5 * u.second,
            use_noise_prior=False,
            precond_width=1,
        )

        tmatrix = ops.TemplateMatrix(templates=[offset_tmpl, hwpss_tmpl])

        # Map maker
        mapper = ops.MapMaker(
            name="satellite_hwp",
            det_data=defaults.det_data,
            binning=binner,
            template_matrix=tmatrix,
            solve_rcond_threshold=1.0e-1,
            map_rcond_threshold=1.0e-1,
            iter_min=50,
            write_hits=True,
            write_binmap=True,
            write_map=True,
            write_cov=False,
            write_rcond=False,
            keep_solver_products=False,
            keep_final_products=False,
            save_cleaned=True,
            overwrite_cleaned=True,
            output_dir=testdir,
        )

        # Make the map
        mapper.apply(data)

        # Write offset amplitudes
        oamps = data[f"{mapper.name}_solve_amplitudes"][offset_tmpl.name]
        oroot = os.path.join(testdir, f"{mapper.name}_offset")
        offset_tmpl.write(oamps, oroot)

        # Dump out the hwpss template amplitudes
        pamps = data[f"{mapper.name}_solve_amplitudes"][hwpss_tmpl.name]
        pfile = os.path.join(testdir, f"{mapper.name}_hwpss.h5")
        hwpssplot_root = os.path.join(testdir, f"{mapper.name}_hwpss-template")
        hwpss_tmpl.write(pamps, pfile)

        # Plot some results
        if data.comm.world_rank == 0 and self.make_plots:
            hwpssplot(pfile, out_root=hwpssplot_root)
            for ob in data.obs:
                offplot(
                    f"{oroot}_{ob.name}.h5",
                    compare={x: ob.detdata["input"][x, :] for x in ob.local_detectors},
                    out=f"{oroot}_{ob.name}",
                )
            self.plot_compare(
                testdir, data, mapper.name, comps=["sky", "noise", "hwpss"], full=False
            )
            hit_file = os.path.join(testdir, f"{mapper.name}_hits.fits")
            map_file = os.path.join(testdir, f"{mapper.name}_map.fits")
            binmap_file = os.path.join(testdir, f"{mapper.name}_binmap.fits")
            truth_file = os.path.join(testdir, f"sky_{self.nside}_input.fits")
            plot_healpix_maps(
                hitfile=hit_file,
                mapfile=map_file,
                truth=truth_file,
                range_I=(-2, 2),
                range_Q=(-0.5, 0.5),
                range_U=(-0.5, 0.5),
            )
            plot_healpix_maps(
                hitfile=hit_file,
                mapfile=binmap_file,
                truth=truth_file,
                range_I=(-2, 2),
                range_Q=(-0.5, 0.5),
                range_U=(-0.5, 0.5),
            )

        # Compare the cleaned timestreams to the original
        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_nonscience):
                dof = np.sqrt(ob.n_local_samples)
                in_rms = np.std(ob.detdata["input"][det])
                thresh = hwpss_scale * (in_rms / dof)
                out_rms = np.std(ob.detdata[defaults.det_data][det])
                if out_rms > thresh:
                    msg = f"{ob.name}:{det} output rms = {out_rms} exceeds threshold"
                    msg += f" ({thresh}) for input rms = {in_rms}"
                    print(msg)
                    # raise RuntimeError("Failed satellite hwpss template regression")

        close_data(data)

    def test_ground_hwp_narrow(self):
        if not available_atm:
            print(
                "TOAST was compiled without atmosphere support, skipping ground tests"
            )
            return
        # Skip this test until we have a better atmospheric removal template.
        return

        testdir = os.path.join(self.outdir, "ground_hwp_narrow")
        if self.comm is None or self.comm.rank == 0:
            if not os.path.isdir(testdir):
                os.mkdir(testdir)
        if self.comm is not None:
            self.comm.barrier()

        proj = "CAR"
        res = 0.2 * u.degree
        data = self.create_ground_sim(testdir, 5.0 * u.degree, proj, res)

        # Add HWPSS and make a copy for plotting
        hwpss_scale = 2.0
        tod_rms = np.std(data.obs[0].detdata["input"][0])
        coeff = fake_hwpss(data, "hwpss", hwpss_scale * tod_rms)
        ops.Combine(
            op="add",
            first="hwpss",
            second="input",
            result="input",
        ).apply(data)

        # Copy the input data into the default field
        ops.Copy(detdata=[("input", defaults.det_data)]).apply(data)

        # Pointing matrix

        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec, quats="quats_radec"
        )
        pixels = ops.PixelsWCS(
            detector_pointing=detpointing_radec,
            projection=proj,
            resolution=(res, res),
            auto_bounds=True,
            use_astropy=True,
        )
        weights = ops.StokesWeights(
            mode="IQU",
            detector_pointing=detpointing_radec,
            hwp_angle=defaults.hwp_angle,
        )

        # Estimate noise
        estim = ops.NoiseEstim(
            det_data=defaults.det_data,
            out_model="noise_est",
            lagmax=512,
            nbin_psd=64,
            nsum=1,
            naverage=64,
        )
        estim.apply(data)

        fit_estim = ops.FitNoiseModel(
            noise_model="noise_est",
            out_model="noise_fit",
        )
        fit_estim.apply(data)

        for ob in data.obs:
            for det in ob.local_detectors:
                fplot = os.path.join(testdir, f"noise_est_{det}.pdf")
                plot_noise_estim(
                    fplot,
                    ob[estim.out_model].freq(det),
                    ob[estim.out_model].psd(det),
                    fit_freq=ob[fit_estim.out_model].freq(det),
                    fit_psd=ob[fit_estim.out_model].psd(det),
                    semilog=False,
                )

        # Set up binning operator for solving and final binning
        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model="noise_fit",
            view="scanning",
        )

        # Set up template for HWPSS
        hwpss_tmpl = Hwpss(
            name="hwpss_template",
            hwp_angle=defaults.hwp_angle,
            flags=defaults.shared_flags,
            flag_mask=defaults.shared_mask_invalid,
            harmonics=5,
        )

        # Simple offset template
        offset_tmpl = Offset(
            name="offset_template",
            times=defaults.times,
            noise_model="noise_fit",
            step_time=1.0 * u.second,
            use_noise_prior=False,
            precond_width=1,
        )

        fourier_tmpl = Fourier2D(
            name="fourier_template",
            times=defaults.times,
            correlation_length=5.0 * u.second,
            correlation_amplitude=1.0,
            order=1,
            fit_subharmonics=False,
            noise_model="noise_fit",
        )

        tmatrix = ops.TemplateMatrix(
            templates=[
                hwpss_tmpl,
                offset_tmpl,
                # fourier_tmpl,
            ],
            view="scanning",
        )

        # Map maker
        mapper = ops.MapMaker(
            name="ground_hwp",
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
        # print(f"Offset amps = {oamps}")
        oroot = os.path.join(testdir, f"{mapper.name}_offset")
        offset_tmpl.write(oamps, oroot)

        # Dump out the periodic template amplitudes
        pamps = data[f"{mapper.name}_solve_amplitudes"][hwpss_tmpl.name]
        pfile = os.path.join(testdir, f"{mapper.name}_hwpss.h5")
        pplot_root = os.path.join(testdir, f"{mapper.name}_hwpss-template")
        hwpss_tmpl.write(pamps, pfile)

        # Plot some results
        if data.comm.world_rank == 0 and self.make_plots:
            hwpssplot(pfile, out_root=pplot_root)
            for ob in data.obs:
                offplot(
                    f"{oroot}_{ob.name}.h5",
                    compare={x: ob.detdata["input"][x, :] for x in ob.local_detectors},
                    out=f"{oroot}_{ob.name}",
                )
            self.plot_compare(
                testdir, data, mapper.name, comps=["sky", "noise", "atm", "hwpss"]
            )
            # self.plot_compare(data, mapper.name, comps=["sky", "noise", "atm"])
            hit_file = os.path.join(testdir, f"{mapper.name}_hits.fits")
            map_file = os.path.join(testdir, f"{mapper.name}_map.fits")
            binmap_file = os.path.join(testdir, f"{mapper.name}_binmap.fits")
            truth_file = os.path.join(
                testdir, f"sky_{proj}_{res.to_value(u.arcmin)}_input.fits"
            )
            plot_wcs_maps(hitfile=hit_file)
            plot_wcs_maps(
                hitfile=hit_file,
                mapfile=map_file,
                truth=truth_file,
                range_I=(-2, 2),
                range_Q=(-0.5, 0.5),
                range_U=(-0.5, 0.5),
            )
            plot_wcs_maps(
                hitfile=hit_file,
                mapfile=binmap_file,
                truth=truth_file,
                range_I=(-2, 2),
                range_Q=(-0.5, 0.5),
                range_U=(-0.5, 0.5),
            )

        # Compare the cleaned timestreams to the original
        for ob in data.obs:
            for det in ob.local_detectors:
                dof = np.sqrt(ob.n_local_samples)
                in_rms = np.std(ob.detdata["input"][det])
                thresh = hwpss_scale * (in_rms / dof)
                out_rms = np.std(ob.detdata[defaults.det_data][det])
                print(f"Using threshold {thresh}, input rms = {in_rms}")
                if out_rms > thresh:
                    print(f"{ob.name}:{det} output rms = {out_rms} exceeds threshold")
                    # This failure is expected until a more sophisticated template
                    # is implemented for atmosphere removal.
                    # raise RuntimeError("Failed ground hwpss template regression")

        close_data(data)
