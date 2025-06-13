# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
from astropy import units as u

from .. import ops as ops
from .. import qarray as qa
from ..instrument_coords import quat_to_xieta
from ..instrument_sim import plot_focalplane
from ..observation import default_values as defaults
from ..pixels import PixelData
from .helpers import (
    close_data,
    create_healpix_ring_satellite,
    create_outdir,
    create_satellite_data,
)
from .mpi import MPITestCase


class StokesWeightsTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        self.nside = 64

    def create_data(self, I_sky, Q_sky, U_sky, hwp_fixed, hwp_step):
        # Create a telescope with 2 boresight pixels per process, so that
        # each process can compare 4 detector orientations.  The boresight
        # pointing will be centered on each pixel exactly once.
        data = create_healpix_ring_satellite(
            self.comm, pix_per_process=2, nside=self.nside
        )

        # Set the HWP angle either to a fixed value in focalplane coordinates,
        # or rotating with the desired steps.
        hwp_name = None
        if hwp_fixed is not None or hwp_step is not None:
            hwp_name = defaults.hwp_angle
            for ob in data.obs:
                hang = None
                if ob.comm_col_rank == 0:
                    if hwp_fixed is not None:
                        hang = hwp_fixed * np.ones_like(ob.shared[defaults.times].data)
                    else:
                        hang = hwp_step * np.arange(ob.n_local_samples)
                ob.shared[defaults.hwp_angle].set(hang, offset=(0,), fromrank=0)

        # Add a detector calibration dictionary
        for ob in data.obs:
            detcal = dict()
            for det in ob.local_detectors:
                detcal[det] = 0.5
            ob["det_cal"] = detcal

        # Pointing operators
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=self.nside,
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=hwp_name,
            detector_pointing=detpointing,
            fp_gamma="gamma",
            cal="det_cal",
            IAU=False,
        )

        # Build the pixel distribution
        build_dist = ops.BuildPixelDistribution(
            pixel_pointing=pixels,
        )
        build_dist.apply(data)
        ops.Delete(detdata=[pixels.pixels]).apply(data)

        # Create a fake sky with fixed I/Q/U values at all pixels.  Just
        # one submap on all processes.
        dist = data["pixel_dist"]
        pix_data = PixelData(dist, np.float64, n_value=3, units=u.K)
        for sm in range(dist.n_local_submap):
            pix_data.data[sm, :, 0] = I_sky
            pix_data.data[sm, :, 1] = Q_sky
            pix_data.data[sm, :, 2] = U_sky
        data["sky"] = pix_data

        # Map scanning
        scan_map = ops.ScanMap(
            pixels=pixels.pixels,
            weights=weights.weights,
            map_key="sky",
        )

        # Scan map into timestream
        pipe = ops.Pipeline(operators=[pixels, weights, scan_map])
        pipe.apply(data)
        ops.Delete(detdata=[pixels.pixels, weights.weights]).apply(data)

        return data

    def plot_static_hwp(
        self,
        dets,
        focalplane=None,
        width=None,
        height=None,
        outfile=None,
        hwpang=None,
        det_values=None,
        det_expected=None,
        sub_title=None,
    ):
        import matplotlib.pyplot as plt

        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        if width is None:
            width = 2.0 * u.degree
        if height is None:
            height = 2.0 * u.degree

        width_deg = width.to_value(u.degree)
        height_deg = height.to_value(u.degree)

        xfigsize = int(width_deg) + 1
        yfigsize = int(height_deg) + 1
        figdpi = 100

        # Compute the font size to use for detector labels
        fontpix = 0.1 * figdpi
        fontpt = int(0.75 * fontpix)

        fig = plt.figure(figsize=(xfigsize, yfigsize), dpi=figdpi)
        ax = fig.add_subplot(1, 1, 1)

        half_width = 0.6 * width_deg
        half_height = 0.6 * height_deg
        ax.set_xlabel(r"Boresight $\xi$ Degrees", fontsize="medium")
        ax.set_ylabel(r"Boresight $\eta$ Degrees", fontsize="medium")
        ax.set_xlim([-half_width, half_width])
        ax.set_ylim([-half_height, half_height])

        xaxis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        yaxis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        zaxis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # Draw the Q/U directions
        stokes_len = 1.1 * np.sqrt(half_width**2 + half_height**2)
        stokes_width = 2.0

        def _plot_stokes_axis(sx, sy, stext, scolor, sstyle):
            ax.plot(
                sx,
                sy,
                color=scolor,
                linestyle=sstyle,
                linewidth=stokes_width,
                zorder=1,
            )
            ax.text(
                sx[-1],
                sy[-1] + 0.05 * stokes_len,
                stext,
                color=scolor,
                fontsize=fontpt,
                horizontalalignment="center",
                verticalalignment="center",
                bbox=dict(fc="w", ec="none", pad=1, alpha=0.0),
                zorder=1,
            )

        # +Q
        _plot_stokes_axis(
            np.array([0.0, 0.0]),
            np.array([-0.5 * stokes_len, 0.5 * stokes_len]),
            "+Q",
            "pink",
            "solid",
        )
        # -Q
        _plot_stokes_axis(
            np.array([-0.5 * stokes_len, 0.5 * stokes_len]),
            np.array([0.0, 0.0]),
            "-Q",
            "pink",
            "dashed",
        )
        # +U
        ustokes = 0.5 * stokes_len / np.sqrt(2)
        _plot_stokes_axis(
            np.array([-ustokes, ustokes]),
            np.array([-ustokes, ustokes]),
            "+U",
            "lightblue",
            "solid",
        )
        # -U
        _plot_stokes_axis(
            np.array([-ustokes, ustokes]),
            np.array([ustokes, -ustokes]),
            "-U",
            "lightblue",
            "dashed",
        )

        # Plot HWP
        if hwpang is not None:
            hwp_len = 0.9 * np.sqrt(half_width**2 + half_height**2)
            hx = 0.5 * hwp_len * np.sin(hwpang)
            hy = 0.5 * hwp_len * np.cos(hwpang)
            ax.arrow(
                hx,
                hy,
                -2 * hx,
                -2 * hy,
                width=0.01 * hwp_len,
                head_width=0.03 * hwp_len,
                head_length=0.03 * hwp_len,
                fc="purple",
                ec="purple",
                linestyle="dashed",
                length_includes_head=True,
                zorder=2,
            )
            ax.text(
                -hx - 0.05 * hwp_len,
                -hy - 0.1 * hwp_len,
                "HWP",
                color="purple",
                fontsize=fontpt,
                horizontalalignment="center",
                verticalalignment="center",
                bbox=dict(fc="w", ec="none", pad=1, alpha=0.0),
                zorder=2,
            )

        for idet, d in enumerate(dets):
            quat = focalplane[d]["quat"]
            fwhm = focalplane[d]["fwhm"].to_value(u.arcmin)

            # radius in degrees
            detradius = 0.5 * 5.0 / 60.0
            if fwhm is not None:
                detradius = 0.5 * fwhm / 60.0

            xi, eta, gamma = quat_to_xieta(quat)
            xpos = xi * 180.0 / np.pi
            ypos = eta * 180.0 / np.pi
            # Polang is plotted relative to visualization x/y coords
            polang = 1.5 * np.pi - gamma
            plot_gamma = polang

            circ = plt.Circle(
                (xpos, ypos), radius=detradius, fc="none", ec="k", zorder=3
            )
            ax.add_artist(circ)

            ascale = 1.5

            xtail = xpos - ascale * detradius * np.cos(polang)
            ytail = ypos - ascale * detradius * np.sin(polang)
            dx = ascale * 2.0 * detradius * np.cos(polang)
            dy = ascale * 2.0 * detradius * np.sin(polang)

            detcolor = colors[idet]

            xsgn = 1.0
            if dx < 0.0:
                xsgn = -1.0
            labeloff = 0.05 * xsgn * fontpix * len(d) / figdpi
            axstr = d
            if det_values is not None:
                axstr = f"{d}\n{det_values[idet]:0.1f}"
                axstr += f" (=? {det_expected[idet]:0.1f})"
            ax.text(
                (xtail + 1.3 * dx + labeloff),
                (ytail + 1.2 * dy),
                axstr,
                color="k",
                fontsize=fontpt,
                horizontalalignment="center",
                verticalalignment="center",
                bbox=dict(fc="w", ec="none", pad=1, alpha=0.0),
                zorder=4,
            )

            ax.arrow(
                xtail,
                ytail,
                dx,
                dy,
                width=0.1 * detradius,
                head_width=0.3 * detradius,
                head_length=0.3 * detradius,
                fc=detcolor,
                ec=detcolor,
                length_includes_head=True,
                zorder=4,
            )

        # Draw a "mini" coordinate axes for reference
        xmini = -0.8 * half_width
        ymini = -0.8 * half_height
        xlen = 0.1 * half_width
        ylen = 0.1 * half_height
        mini_width = 0.005 * half_width
        mini_head_width = 3 * mini_width
        mini_head_len = 3 * mini_width

        aprops = [
            (xlen, 0, "-", r"$\xi$"),
            (0, ylen, "-", r"$\eta$"),
            (-xlen, 0, "--", "Y"),
            (0, -ylen, "--", "X"),
        ]

        for ap in aprops:
            lx = xmini + 1.5 * ap[0]
            ly = ymini + 1.5 * ap[1]
            lw = figdpi / 200.0
            ax.arrow(
                xmini,
                ymini,
                ap[0],
                ap[1],
                width=mini_width,
                head_width=mini_head_width,
                head_length=mini_head_len,
                fc="k",
                ec="k",
                linestyle=ap[2],
                linewidth=lw,
                length_includes_head=True,
            )
            ax.text(
                lx,
                ly,
                ap[3],
                color="k",
                fontsize=int(figdpi / 10),
                horizontalalignment="center",
                verticalalignment="center",
            )

        if sub_title is not None:
            ax.set_title(sub_title, fontsize=8)
        st = "Detector Response on Sky From Observer"
        fig.suptitle(st)
        plt.savefig(outfile, dpi=figdpi, bbox_inches="tight", format="pdf")
        plt.close()

    def test_hwp_static(self):
        stokes_cases = [
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ]
        hwp_static = [None, 0.0, np.pi / 4, np.pi / 2]

        for I_sky, Q_sky, U_sky in stokes_cases:
            for hang in hwp_static:
                # Create the dataset for this configuration
                data = self.create_data(I_sky, Q_sky, U_sky, hang, None)

                # Compute the expected values for all timestream samples
                # (COSMO convention) for the 4 detectors.  The first local
                # pixel has a detector aligned with the meridian and the
                # other is orthogonal.  The second local pixel is rotated
                # 45 degrees.  The focalplane coordinate frame has its X-axis
                # pointing South along the meridian.
                det_alpha = np.array([0.0, np.pi / 2, np.pi / 4, 3 * np.pi / 4])

                if hang is None:
                    # No HWP
                    expected = 0.5 * (
                        I_sky
                        + Q_sky * np.cos(2 * det_alpha)
                        + U_sky * np.sin(2 * det_alpha)
                    )
                else:
                    expected = 0.5 * (
                        I_sky
                        + Q_sky * np.cos(2 * (det_alpha - 2 * hang))
                        - U_sky * np.sin(2 * (det_alpha - 2 * hang))
                    )

                hstr = "none"
                if hang is not None:
                    hstr = f"{np.degrees(hang):0.1f}"

                for ob in data.obs:
                    fp = ob.telescope.focalplane

                    dbg = f"Stokes {ob.name}:\n"
                    dbg += f"  sky ({I_sky}, {Q_sky}, {U_sky}), "
                    if hang is not None:
                        hdeg = hang * 180.0 / np.pi
                        dbg += f"hwp = {hdeg:0.1f}\n"
                    else:
                        dbg += f"hwp = {hang}\n"
                    failed = list()
                    det_vals = list()
                    for idet, det in enumerate(ob.local_detectors):
                        gamma = fp[det]["gamma"].to_value(u.deg)
                        ddata = ob.detdata[defaults.det_data][det]
                        comp = expected[idet] * np.ones_like(ddata)
                        dbg += f"  det {idet}, gamma={gamma:0.1f} "
                        dbg += f"({expected[idet]:0.3e}) = "
                        dbg += f"[{ddata[0]:0.3e} ... {ddata[-1]:0.3e}]\n"
                        det_vals.append(np.median(ddata[:]))
                        if not np.allclose(ddata, comp):
                            failed.append(idet)

                    # Plot the focalplane once just to check det orientations
                    if ob.comm.group_rank == 0:
                        file_stokes = f"I{int(I_sky)}-Q{int(Q_sky)}-U{int(U_sky)}"
                        out = os.path.join(
                            self.outdir,
                            f"hwp-static_{hstr}_{ob.name}_{file_stokes}.pdf",
                        )
                        fp = data.obs[0].telescope.focalplane
                        pcol = {}
                        for idet, d in enumerate(fp.detectors):
                            if idet // 2 == 0:
                                pcol[d] = "r"
                            else:
                                pcol[d] = "b"
                        subtitle = f"Obs {ob.name}, Sky: "
                        subtitle += f"I={I_sky:0.1f} "
                        subtitle += f"Q={Q_sky:0.1f} "
                        subtitle += f"U={U_sky:0.1f} "
                        self.plot_static_hwp(
                            ob.local_detectors,
                            focalplane=fp,
                            width=3 * u.degree,
                            height=3 * u.degree,
                            outfile=out,
                            det_values=det_vals,
                            det_expected=expected,
                            sub_title=subtitle,
                            hwpang=hang,
                        )

                    if len(failed) > 0:
                        print(dbg)
                        print(f"detectors {failed} failed", flush=True)
                        self.assertTrue(False)

                close_data(data)

    def test_hwp_stepped(self):
        stokes_cases = [
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ]
        hwp_step = np.pi / (12 * self.nside**2)

        done_plot = False
        for I_sky, Q_sky, U_sky in stokes_cases:
            # Create the dataset for this configuration
            data = self.create_data(I_sky, Q_sky, U_sky, None, hwp_step)

            # Plot the focalplane once just to check det orientations
            if not done_plot:
                out = os.path.join(self.outdir, "boresight_fp_stepped.pdf")
                fp = data.obs[0].telescope.focalplane
                pcol = {}
                for idet, d in enumerate(fp.detectors):
                    if idet // 2 == 0:
                        pcol[d] = "r"
                    else:
                        pcol[d] = "b"
                self.plot_static_hwp(
                    data.obs[0].local_detectors,
                    focalplane=fp,
                    width=3 * u.degree,
                    height=3 * u.degree,
                    outfile=out,
                )
                done_plot = True

            # Compute the expected values for all timestream samples
            # (COSMO convention) for the 4 detectors.  The first local
            # pixel has a detector aligned with the meridian and the
            # other is orthogonal.  The second local pixel is rotated
            # 45 degrees.  The focalplane coordinate frame has its X-axis
            # pointing South along the meridian.
            det_alpha = np.array([0.0, np.pi / 2, np.pi / 4, 3 * np.pi / 4])

            for ob in data.obs:
                if ob.comm.group_rank == 0:
                    import matplotlib.pyplot as plt

                    # Make timestream plots of the detector response
                    file_stokes = f"I{int(I_sky)}-Q{int(Q_sky)}-U{int(U_sky)}"
                    title_stokes = (
                        f"Constant Sky Values: I={I_sky}, Q={Q_sky}, U={U_sky}"
                    )
                    fig = plt.figure(figsize=(8, 6), dpi=75)
                    ax = fig.add_subplot(1, 1, 1, aspect="auto")
                    for idet, det in enumerate(ob.local_detectors):
                        gamma = np.degrees(det_alpha[idet])
                        ax.plot(
                            np.degrees(ob.shared[defaults.hwp_angle].data),
                            ob.detdata[defaults.det_data][det],
                            label=f"Det {det}, gamma={gamma:0.1f}",
                        )
                    ax.set_xlabel("HWP Angle (degrees)")
                    ax.set_ylabel("Response")
                    ax.legend(loc=1)
                    plt.title(f"Observation {ob.name} {title_stokes}")
                    savefile = os.path.join(
                        self.outdir,
                        f"hwp_{ob.name}_{file_stokes}.pdf",
                    )
                    plt.savefig(savefile)
                    plt.close()

            for ob in data.obs:
                fp = ob.telescope.focalplane
                hang = ob.shared[defaults.hwp_angle].data
                failed = list()
                dbg = "HWP check:"
                for idet, det in enumerate(ob.local_detectors):
                    gamma = fp[det]["gamma"].to_value(u.radian)
                    ddata = ob.detdata[defaults.det_data][det]

                    # Expected response.  Because we have aligned the fake focalplane
                    # frame to the coordinate system, alpha == gamma_detector
                    alpha = gamma
                    comp = 0.5 * (
                        I_sky
                        + Q_sky * np.cos(2.0 * (2.0 * (gamma - hang) - alpha))
                        - U_sky * np.sin(2.0 * (2.0 * (gamma - hang) - alpha))
                    )
                    dbg += f"\n{det}({idet}):\n"
                    dbg += f"  gamma = alpha = {gamma} ({np.degrees(gamma)} deg)\n"
                    dbg += f"  hwp angle = {hang}\n"
                    dbg += f"  expected = {comp}\n"
                    dbg += f"  actual = {ddata}"
                    if not np.allclose(ddata, comp):
                        failed.append(idet)
                if len(failed) > 0:
                    print(dbg)
                    print(f"detectors {failed} failed", flush=True)
                    self.assertTrue(False)

            close_data(data)

    def test_QU(self):
        # Create a fake satellite data set for testing

        data = create_satellite_data(self.comm)

        # Create Stokes weights in IQU and QU modes

        detpointing = ops.PointingDetectorSimple()

        weights_iqu = ops.StokesWeights(
            mode="IQU",
            weights="weights_iqu",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )
        weights_iqu.apply(data)

        weights_qu = ops.StokesWeights(
            mode="QU",
            weights="weights_qu",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )
        weights_qu.apply(data)

        # Compare the QU part of the weights. It must match

        for ob in data.obs:
            qu = ob.detdata["weights_qu"].data
            iqu = ob.detdata["weights_iqu"].data
            np.testing.assert_array_equal(qu, iqu[:, :, 1:])

        close_data(data)
        if self.comm is not None:
            self.comm.barrier()
