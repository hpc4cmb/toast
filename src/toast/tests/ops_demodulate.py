# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
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
)
from .mpi import MPITestCase


class DemodulateTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def _test_demodulate(self, weight_mode):
        nside = 128

        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Pointing operator

        detpointing = ops.PointingDetectorSimple(shared_flag_mask=0)
        pixels = ops.PixelsHealpix(
            nside=nside,
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
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
            fwhm=10.0 * u.deg,
            lmax=3 * nside,
            I_scale=0.001,
            Q_scale=0.0001,
            U_scale=0.0001,
            det_data=defaults.det_data,
        )

        # Bin signal without demodulation

        binner = ops.BinMap(
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
        )

        mapper = ops.MapMaker(
            name=f"modulated_{weight_mode}",
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

        # Write one timestream for comparison
        if data.comm.world_rank == 0:
            oname = data.obs[0].name
            valid_dets = data.obs[0].select_local_detectors(
                flagmask=defaults.det_mask_invalid
            )
            dname = valid_dets[0]
            tod_input = os.path.join(
                self.outdir,
                f"tod_{oname}_{dname}_in.np",
            )
            data.obs[0].detdata[defaults.det_data][dname].tofile(tod_input)

        # Demodulate

        demod_weights_in = ops.StokesWeights(
            weights="demod_weights_in",
            mode=weight_mode,
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )

        downsample = 3
        demod = ops.Demodulate(
            stokes_weights=demod_weights_in,
            nskip=downsample,
            purge=False,
            mode=weight_mode,
        )
        demod_data = demod.apply(data)

        # Map again
        demod_weights = ops.StokesWeightsDemod(mode=weight_mode)

        mapper.name = f"demodulated_{weight_mode}"
        binner.stokes_weights = demod_weights
        mapper.apply(demod_data)

        if data.comm.world_rank == 0:
            set_matplotlib_backend()
            import matplotlib.pyplot as plt

            # Plot demodulated timestreams for comparison
            oname = data.obs[0].name
            valid_dets = data.obs[0].select_local_detectors(
                flagmask=defaults.det_mask_invalid
            )
            dname = valid_dets[0]
            tod_in_file = os.path.join(
                self.outdir,
                f"tod_{oname}_{dname}_in.np",
            )
            tod_in = np.fromfile(tod_in_file)
            tod_plot = os.path.join(self.outdir, f"tod_{oname}_{dname}.pdf")

            slc_in = slice(0, 50)
            slc_demod = slice(0, 50 // downsample)

            fig = plt.figure(figsize=(12, 12), dpi=72)
            ax = fig.add_subplot(2, 1, 1, aspect="auto")
            ax.plot(
                data.obs[0].shared[defaults.times].data[slc_in],
                tod_in[slc_in],
                c="black",
                label=f"Original Signal",
            )
            ax.plot(
                data.obs[0].shared[defaults.times].data[slc_in],
                data.obs[0].shared[defaults.hwp_angle].data[slc_in],
                c="purple",
                label=f"HWP Angle",
            )
            ax.legend(loc="best")
            ax = fig.add_subplot(2, 1, 2, aspect="auto")
            if "I" in weight_mode:
                ax.plot(
                    demod_data.obs[0].shared[defaults.times].data[slc_demod],
                    demod_data.obs[0].detdata[defaults.det_data][
                        f"demod0_{dname}", slc_demod
                    ],
                    c="red",
                    label=f"Demod0",
                )
            if "QU" in weight_mode:
                ax.plot(
                    demod_data.obs[0].shared[defaults.times].data[slc_demod],
                    demod_data.obs[0].detdata[defaults.det_data][
                        f"demod4r_{dname}", slc_demod
                    ],
                    c="blue",
                    label=f"Demod4r",
                )
                ax.plot(
                    demod_data.obs[0].shared[defaults.times].data[slc_demod],
                    demod_data.obs[0].detdata[defaults.det_data][
                        f"demod4i_{dname}", slc_demod
                    ],
                    c="green",
                    label=f"Demod4i",
                )
            ax.legend(loc="best")
            plt.title(f"Demodulation")
            plt.savefig(tod_plot)
            plt.close()

            fname_mod = os.path.join(self.outdir, f"modulated_{weight_mode}_map.fits")
            fname_demod = os.path.join(
                self.outdir, f"demodulated_{weight_mode}_map.fits"
            )

            map_mod = hp.read_map(fname_mod, None)
            map_demod = np.atleast_2d(hp.read_map(fname_demod, None))
            map_input = np.atleast_2d(hp.read_map(sky_file, None))

            fig = plt.figure(figsize=[18, 12])
            nrow, ncol = 2, 3
            rot = [42, -42]
            reso = 5

            amp = 1e-5
            for i, m in enumerate(map_mod):
                # Modulated map is full IQU
                value = map_input[i]
                good = m != 0
                rms = np.sqrt(np.mean((m[good] - value[good]) ** 2))
                m[m == 0] = hp.UNSEEN
                stokes = "IQU"[i]
                hp.gnomview(
                    m,
                    sub=[nrow, ncol, 1 + i],
                    reso=reso,
                    rot=rot,
                    title=f"Modulated {stokes} : rms = {rms}",
                    min=np.amin(value[good]) - amp,
                    max=np.amax(value[good]) + amp,
                    cmap="coolwarm",
                )

            all_good = True
            for stokes, m in zip(weight_mode, map_demod):
                # Demodulated map only has the prescribed components
                i = "IQU".index(stokes)
                value = map_input[i]
                good = m != 0
                rms0 = np.sqrt(np.mean(value[good] ** 2))
                rms = np.sqrt(np.mean(m[good] ** 2))
                rms1 = np.sqrt(np.mean((m[good] - value[good]) ** 2))
                m[m == 0] = hp.UNSEEN
                hp.gnomview(
                    m,
                    sub=[nrow, ncol, 4 + i],
                    reso=reso,
                    rot=rot,
                    title=f"Demodulated {stokes} : rms = {rms}",
                    min=np.amin(value[good]) - amp,
                    max=np.amax(value[good]) + amp,
                    cmap="coolwarm",
                )
                if rms1 / rms0 > 0.5:
                    print(
                        f"input - demodulated map RMS = {rms1}, (input RMS = {rms0})",
                        flush=True,
                    )
                    all_good = False

            outfile = os.path.join(self.outdir, f"map_comparison.{weight_mode}.png")
            fig.savefig(outfile)
            self.assertTrue(all_good)

        if self.comm is not None:
            self.comm.barrier()
        close_data(demod_data)
        close_data(data)

    def test_demodulate_IQU(self):
        self._test_demodulate(weight_mode="IQU")

    def test_demodulate_QU(self):
        self._test_demodulate(weight_mode="QU")

    def test_demodulate_I(self):
        self._test_demodulate(weight_mode="I")
