# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
from astropy import units as u

from .. import ops as ops
from .. import qarray as qa
from ..instrument_sim import plot_focalplane
from ..observation import default_values as defaults
from ..pixels import PixelData
from ._helpers import close_data, create_healpix_ring_satellite, create_outdir
from .mpi import MPITestCase


class SimStokesWeightsTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.nside = 64

    def create_data(self, I_sky, Q_sky, U_sky, hwp):
        # Create a telescope with 2 boresight pixels per process, so that
        # each process can compare 4 detector orientations.  The boresight
        # pointing will be centered on each pixel exactly once.
        data = create_healpix_ring_satellite(
            self.comm, pix_per_process=2, nside=self.nside
        )

        # Set the HWP angle to a fixed value in focalplane coordinates
        hwp_name = None
        if hwp is not None:
            hwp_name = defaults.hwp_angle
            for ob in data.obs:
                hang = None
                if ob.comm_col_rank == 0:
                    hang = hwp * np.ones_like(ob.shared[defaults.times].data)
                ob.shared[defaults.hwp_angle].set(hang, offset=(0,), fromrank=0)

        # Expand detector pointing
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=self.nside,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=hwp_name,
            detector_pointing=detpointing,
            fp_gamma="gamma",
            IAU=False,
        )
        pipe = ops.Pipeline(operators=[pixels, weights])
        pipe.apply(data)
        # print(data.obs[0].detdata[weights.weights])

        # Create a fake sky with fixed I/Q/U values at all pixels.  Just
        # one submap on all processes.
        dist = data["pixel_dist"]
        pix_data = PixelData(dist, np.float64, n_value=3, units=u.K)
        for sm in range(dist.n_local_submap):
            pix_data.data[sm, :, 0] = I_sky
            pix_data.data[sm, :, 1] = Q_sky
            pix_data.data[sm, :, 2] = U_sky
        data["sky"] = pix_data

        # Scan this sky into the timestream
        scan_map = ops.ScanMap(
            pixels=pixels.pixels,
            weights=weights.weights,
            map_key="sky",
        )
        scan_map.apply(data)

        return data

    def test_sim(self):
        stokes_cases = [
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        hwp_static = [None, 0.0, np.pi / 4, np.pi / 2]

        done_plot = False
        for I_sky, Q_sky, U_sky in stokes_cases:
            for hang in hwp_static:
                # Create the dataset for this configuration
                data = self.create_data(I_sky, Q_sky, U_sky, hang)

                # Plot the focalplane once just to check det orientations
                if not done_plot:
                    out = os.path.join(self.outdir, "boresight_fp.pdf")
                    fp = data.obs[0].telescope.focalplane
                    pcol = {}
                    for idet, d in enumerate(fp.detectors):
                        if idet // 2 == 0:
                            pcol[d] = "r"
                        else:
                            pcol[d] = "b"
                    plot_focalplane(
                        focalplane=fp,
                        width=3 * u.degree,
                        height=3 * u.degree,
                        outfile=out,
                        show_labels=True,
                        face_color=None,
                        pol_color=pcol,
                        xieta=False,
                    )
                    done_plot = True

                # Compute the expected values for all timestream samples (COSMO convention)
                # for the 4 detectors.  The first local pixel has a detector aligned with
                # the meridian and the other is orthogonal.  The second local pixel is
                # rotated 45 degrees.  The focalplane coordinate frame has its X-axis
                # pointing South along the meridian.
                det_alpha = np.array([0.0, np.pi / 2, np.pi / 4, 3 * np.pi / 4])

                if hang is None:
                    # No HWP
                    expected = (
                        I_sky
                        + Q_sky * np.cos(2 * det_alpha)
                        - U_sky * np.sin(2 * det_alpha)
                    )
                else:
                    expected = (
                        I_sky
                        + Q_sky * np.cos(2 * (det_alpha - 2 * hang))
                        + U_sky * np.sin(2 * (det_alpha - 2 * hang))
                    )

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
                    for idet, det in enumerate(ob.local_detectors):
                        gamma = fp[det]["gamma"].to_value(u.deg)
                        ddata = ob.detdata[defaults.det_data][det]
                        comp = expected[idet] * np.ones_like(ddata)
                        dbg += f"  det {idet}, gamma={gamma:0.1f} "
                        dbg += f"({expected[idet]:0.3e}) = "
                        dbg += f"[{ddata[0]:0.3e} ... {ddata[-1]:0.3e}]\n"
                        if not np.allclose(ddata, comp):
                            failed.append(idet)
                    if len(failed) > 0:
                        print(dbg)
                        print(f"detectors {failed} failed", flush=True)
                        self.assertTrue(False)

                close_data(data)
