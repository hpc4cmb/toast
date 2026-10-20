# Copyright (c) 2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
from astropy import units as u

from .. import ops as ops
from .. import qarray as qa
from ..observation import default_values as defaults
from ..vis import set_matplotlib_backend
from .helpers import (
    close_data,
    create_fake_healpix_scanned_tod,
    create_ground_data,
    create_outdir,
    create_overdistributed_data,
)
from .mpi import MPITestCase


class PairDiffTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def _test_pairdiff(self, weight_mode, data):
        nside = 256

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Pointing operators

        detpointing_azel = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec,
            shared_flag_mask=0,
        )
        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec,
            shared_flag_mask=0,
        )

        pixels = ops.PixelsHealpix(
            nside=nside,
            detector_pointing=detpointing_radec,
        )
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing_radec,
        )

        sky_file = os.path.join(self.outdir, f"fake_sky_{weight_mode}.fits")
        map_key = "fake_map"
        create_fake_healpix_scanned_tod(
            data,
            pixels,
            weights,
            sky_file,
            "pixel_dist",
            map_key=map_key,
            fwhm=1.0 * u.deg,
            lmax=3 * nside,
            I_scale=0.001,
            Q_scale=0.0001,
            U_scale=0.0001,
            det_data=defaults.det_data,
        )

        # Bin signal without differencing

        binner = ops.BinMap(
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
        )

        mapper = ops.MapMaker(
            name=f"undiff_{weight_mode}",
            det_data=defaults.det_data,
            binning=binner,
            template_matrix=None,
            write_hits=True,
            write_map=True,
            write_cov=True,
            write_invcov=True,
            write_rcond=True,
            keep_final_products=True,
            output_dir=self.outdir,
            map_rcond_threshold=1e-2,
        )
        mapper.apply(data)

        # Write timestreams for comparison

        if data.comm.world_rank == 0:
            oname = data.obs[0].name
            valid_dets = data.obs[0].select_local_detectors(
                flagmask=defaults.det_mask_invalid
            )
            for dname in valid_dets:
                tod_input = os.path.join(
                    self.outdir,
                    f"tod_{oname}_{dname}{weight_mode}_in.np",
                )
                data.obs[0].detdata[defaults.det_data][dname].tofile(tod_input)

        # Pairdifference

        pairdiff = ops.PairDifference(fpkeys=["pixel"], mode=weight_mode)
        pairdiff_data = pairdiff.apply(data)

        # Map again

        pairdiff_weights = ops.StokesWeightsPairDiff(
            detector_pointing=detpointing_radec,
            mode=weight_mode,
            hwp_angle=defaults.hwp_angle,
        )

        mapper.name = f"pairdiff_{weight_mode}"
        binner.stokes_weights = pairdiff_weights
        mapper.apply(pairdiff_data)

        if data.comm.world_rank == 0:
            set_matplotlib_backend()
            import matplotlib.pyplot as plt

            # Pick the first local pair of detectors
            local_dets = pairdiff_data.obs[0].select_local_detectors(
                flagmask=defaults.det_mask_invalid
            )
            first_det = local_dets[0]
            if first_det.startswith("pairsum"):
                pair = first_det.replace("pairsum_", "")
            elif first_det.startswith("pairdiff"):
                pair = first_det.replace("pairdiff_", "")
            else:
                raise RuntimeError(f"Cannot parse '{first_det}'")
            det1, det2 = pair.split("&")

            # Plot undifferenced timestreams for comparison
            oname = data.obs[0].name
            tod_in_file1 = os.path.join(
                self.outdir,
                f"tod_{oname}_{det1}{weight_mode}_in.np",
            )
            tod_in_file2 = os.path.join(
                self.outdir,
                f"tod_{oname}_{det2}{weight_mode}_in.np",
            )
            tod_in1 = np.fromfile(tod_in_file1)
            tod_in2 = np.fromfile(tod_in_file2)
            tod_plot = os.path.join(self.outdir, f"tod_{oname}_{pair}{weight_mode}.pdf")

            slc = slice(0, 50)

            fig = plt.figure(figsize=(12, 12), dpi=72)
            ax = fig.add_subplot(2, 1, 1, aspect="auto")
            ax.plot(
                data.obs[0].shared[defaults.times].data[slc],
                tod_in1[slc],
                c="black",
                label=det1,
            )
            ax.plot(
                data.obs[0].shared[defaults.times].data[slc],
                tod_in2[slc],
                c="orange",
                label=det2,
            )
            ax.legend(loc="best")
            ax = fig.add_subplot(2, 1, 2, aspect="auto")
            if "I" in weight_mode:
                ax.plot(
                    pairdiff_data.obs[0].shared[defaults.times].data[slc],
                    pairdiff_data.obs[0].detdata[defaults.det_data][
                        f"pairsum_{pair}", slc
                    ],
                    c="red",
                    label=f"{det1} + {det2}",
                )
            if "QU" in weight_mode:
                ax.plot(
                    pairdiff_data.obs[0].shared[defaults.times].data[slc],
                    pairdiff_data.obs[0].detdata[defaults.det_data][
                        f"pairdiff_{pair}", slc
                    ],
                    c="blue",
                    label=f"{det1} - {det2}",
                )
            ax.legend(loc="best")
            plt.title("Pairdifferencing")
            plt.savefig(tod_plot)
            plt.close()

            fname_undiff = os.path.join(
                self.outdir, f"undiff_{weight_mode}_map.fits"
            )
            fname_pairdiff = os.path.join(
                self.outdir, f"pairdiff_{weight_mode}_map.fits"
            )
            fname_hits = os.path.join(
                self.outdir, f"pairdiff_{weight_mode}_hits.fits"
            )

            map_undiff = hp.read_map(fname_undiff, None)
            map_pairdiff = np.atleast_2d(hp.read_map(fname_pairdiff, None))
            hits = hp.read_map(fname_hits)
            map_input = np.atleast_2d(hp.read_map(sky_file, None))

            # Develop a comparison mask that excludes poorly observed pixels
            sorted_hits = np.sort(hits[hits != 0])
            nhit = len(sorted_hits)
            hit_min = sorted_hits[int(0.1 * nhit)]  # worst 10% of hit pixels
            rms_mask = hits > hit_min

            fig = plt.figure(figsize=[18, 12])
            nrow, ncol = 3, 3
            rot = [42, -42]
            reso = 1
            xsize = 800

            amp = 1e-5
            for i, m in enumerate(map_undiff):
                # Undiff map is full IQU
                value = map_input[i]
                good = m != 0
                rms = np.sqrt(np.mean((m - value)[rms_mask] ** 2))
                m[m == 0] = hp.UNSEEN
                stokes = "IQU"[i]
                hp.gnomview(
                    m,
                    sub=[nrow, ncol, 1 + i],
                    reso=reso,
                    xsize=xsize,
                    rot=rot,
                    title=f"Undiff {stokes} : rms = {rms}",
                    min=np.amin(value[good]) - amp,
                    max=np.amax(value[good]) + amp,
                    cmap="coolwarm",
                )

            all_good = True
            for stokes, m in zip(weight_mode, map_pairdiff):
                # Pairdiff map only has the prescribed components
                i = "IQU".index(stokes)
                value = map_input[i]
                good = m != 0
                rms0 = np.sqrt(np.mean(value[rms_mask] ** 2))
                rms = np.sqrt(np.mean(m[rms_mask] ** 2))
                rms1 = np.sqrt(np.mean((m - value)[rms_mask] ** 2))
                m[m == 0] = hp.UNSEEN
                hp.gnomview(
                    m,
                    sub=[nrow, ncol, ncol + i + 1],
                    reso=reso,
                    xsize=xsize,
                    rot=rot,
                    title=f"Pairdiff {stokes} : rms = {rms / rms0:.3f} x rms(in)",
                    min=np.amin(value[good]) - amp,
                    max=np.amax(value[good]) + amp,
                    cmap="coolwarm",
                )
                resid = m - value
                resid[m == 0] = hp.UNSEEN
                hp.gnomview(
                    resid,
                    sub=[nrow, ncol, 2 * ncol + i + 1],
                    reso=reso,
                    xsize=xsize,
                    rot=rot,
                    title=f"Residual {stokes} : resid rms = {rms1 / rms0:.3f} x rms(in)",
                    min=np.amin(value[good]) - amp,
                    max=np.amax(value[good]) + amp,
                    cmap="coolwarm",
                )
                if rms1 / rms0 > 0.1:
                    print(
                        f"input - pairdiff map RMS = {rms1}, (input RMS = {rms0})",
                        flush=True,
                    )
                    all_good = False

            outfile = os.path.join(
                self.outdir, f"map_comparison.{weight_mode}.png"
            )
            fig.savefig(outfile)
            self.assertTrue(all_good)

        if self.comm is not None:
            self.comm.barrier()
        close_data(pairdiff_data)
        close_data(data)

    def test_pairdiff_IQU(self):
        data = create_ground_data(self.comm, flagged_obs=False)
        self._test_pairdiff(weight_mode="IQU", data=data)

    def test_pairdiff_QU(self):
        data = create_ground_data(self.comm, flagged_obs=False)
        self._test_pairdiff(weight_mode="QU", data=data)

    def test_pairdiff_I(self):
        data = create_ground_data(self.comm, flagged_obs=False)
        self._test_pairdiff(weight_mode="I", data=data)
