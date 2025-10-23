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
    create_overdistributed_data,
)
from .mpi import MPITestCase


class DemodulateTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def test_change_frame(self):
        # There are frame changes in all demodulation tests but this one
        # is run with higher resolution input map to minimize T->P leakage

        data = create_ground_data(self.comm)
        nside = 1024

        # Create an uncorrelated noise model from focalplane detector properties

        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Pointing operators

        detpointing_radial = ops.PointingDetectorFP(
            quats="quats_radial",
        )
        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec,
            shared_flag_mask=0,
            quats="quats_radec",
        )

        pixels = ops.PixelsHealpix(
            nside=nside,
            detector_pointing=detpointing_radec,
        )
        weights_radial = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing_radial,
            weights="weights_radial",
        )

        weights_radec = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing_radec,
            weights="weights_radec",
        )

        sky_file = os.path.join(self.outdir, f"fake_sky_coord_change.fits")
        map_key = "fake_map"
        create_fake_healpix_scanned_tod(
            data,
            pixels,
            weights_radec,
            sky_file,
            "pixel_dist",
            map_key=map_key,
            fwhm=10.0 * u.deg,
            lmax=1 * nside,
            I_scale=0.001,
            Q_scale=0.0001,
            U_scale=0.0001,
            det_data=defaults.det_data,
        )

        # Demodulate

        downsample = 3
        demod_radial = ops.Demodulate(
            stokes_weights=weights_radial,
            nskip=downsample,
            purge=False,
        )
        demod_data_radial = demod_radial.apply(data)

        demod_radec = ops.Demodulate(
            stokes_weights=weights_radec,
            nskip=downsample,
        )
        demod_data_radec = demod_radec.apply(data)

        # Get the Stokes weights

        demod_weights_radial = ops.StokesWeightsDemod(
            detector_pointing_in=detpointing_radial,
            detector_pointing_out=detpointing_radec,
        )
        demod_weights_radial.apply(demod_data_radial)

        demod_weights_radec = ops.StokesWeightsDemod()
        demod_weights_radec.apply(demod_data_radec)

        # Check the weights
        for ob_radial, ob_radec in zip(demod_data_radial.obs, demod_data_radec.obs):
            for qdet in ob_radial.local_detectors:
                if not qdet.startswith("demod4r"):
                    continue
                udet = qdet.replace("demod4r", "demod4i")
                # Q/U in the horizontal system
                qsig_radial = ob_radial.detdata[defaults.det_data][qdet]
                usig_radial = ob_radial.detdata[defaults.det_data][udet]
                # Q/U in RA/Dec
                qsig_radec = ob_radec.detdata[defaults.det_data][qdet]
                usig_radec = ob_radec.detdata[defaults.det_data][udet]
                # Rotate horizontal Q/U to RA/Dec
                qweights_radial = ob_radial.detdata[defaults.weights][qdet].T
                uweights_radial = ob_radial.detdata[defaults.weights][udet].T
                qsig_rot = (
                    qsig_radial * qweights_radial[1] + usig_radial * uweights_radial[1]
                )
                usig_rot = (
                    qsig_radial * qweights_radial[2] + usig_radial * uweights_radial[2]
                )

                ind = slice(100, -100)  # Cut ends due to potential ringing
                rms_q = np.std(qsig_radec[ind])
                rms_q_rot = np.std(qsig_rot[ind])
                rms_q_resid = np.std((qsig_radec - qsig_rot)[ind])
                rms_u = np.std(usig_radec[ind])
                rms_u_rot = np.std(usig_rot[ind])
                rms_u_resid = np.std((usig_radec - usig_rot)[ind])

                """
                if rms_q_resid > 1e-2 * rms_q or rms_u_resid > 1e-2 * rms_u:
                    rank = data.comm.world_rank
                    print(f"RMS(Q) = {rms_q}, RMS(resid) = {rms_q_resid / rms_q} x RMS(Q)")
                    print(f"RMS(U) = {rms_u}, RMS(resid) = {rms_u_resid / rms_u} x RMS(U)")
                    import matplotlib.pyplot as plt
                    fname_plot = f"error.{rank}.png"
                    nrow, ncol = 1, 2
                    fig = plt.figure(figsize=[ncol * 6, nrow * 4])
                    ax = fig.add_subplot(nrow, ncol, 1)
                    ax.plot(qsig_radec, label=f"Q RA/Dec, rms={rms_q}")
                    ax.plot(qsig_radial, label="Q Qr/Ur")
                    ax.plot(qsig_rot, label=f"Q Qr/Ur->Ra/Dec, rms={rms_q_rot}")
                    ax.plot(qsig_rot - qsig_radec, label=f"Diff, rms={rms_q_resid}")
                    ax.legend(loc="best")
                    ax = fig.add_subplot(nrow, ncol, 2)
                    ax.plot(usig_radec, label=f"U RA/Dec, rms={rms_u}")
                    ax.plot(usig_radial, label="U Qr/Ur")
                    ax.plot(usig_rot, label=f"U Qr/Ur->Ra/Dec, rms={rms_u_rot}")
                    ax.plot(usig_rot - usig_radec, label=f"Diff, rms={rms_u_resid}")
                    ax.legend(loc="best")
                    fig.savefig(fname_plot)
                """

                assert rms_q_resid < 1e-2 * rms_q
                assert rms_u_resid < 1e-2 * rms_u

        if self.comm is not None:
            self.comm.barrier()
        close_data(demod_data_radial)
        close_data(demod_data_radec)
        close_data(data)

    def _test_demodulate(self, weight_mode, data, suffix=""):
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

        sky_file = os.path.join(self.outdir, f"fake_sky_{weight_mode}{suffix}.fits")
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

        # Bin signal without demodulation

        binner = ops.BinMap(
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
        )

        mapper = ops.MapMaker(
            name=f"modulated_{weight_mode}{suffix}",
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
                f"tod_{oname}_{dname}{suffix}_in.np",
            )
            data.obs[0].detdata[defaults.det_data][dname].tofile(tod_input)

        # Demodulate

        demod_weights_in = ops.StokesWeights(
            weights="demod_weights_in",
            mode=weight_mode,
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing_azel,
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

        demod_weights = ops.StokesWeightsDemod(
            detector_pointing_in=detpointing_azel,
            detector_pointing_out=detpointing_radec,
            mode=weight_mode,
        )

        mapper.name = f"demodulated_{weight_mode}{suffix}"
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
                f"tod_{oname}_{dname}{suffix}_in.np",
            )
            tod_in = np.fromfile(tod_in_file)
            tod_plot = os.path.join(self.outdir, f"tod_{oname}_{dname}{suffix}.pdf")

            slc_in = slice(0, 50)
            slc_demod = slice(0, 50 // downsample)

            fig = plt.figure(figsize=(12, 12), dpi=72)
            ax = fig.add_subplot(2, 1, 1, aspect="auto")
            ax.plot(
                data.obs[0].shared[defaults.times].data[slc_in],
                tod_in[slc_in],
                c="black",
                label="Original Signal",
            )
            ax.plot(
                data.obs[0].shared[defaults.times].data[slc_in],
                data.obs[0].shared[defaults.hwp_angle].data[slc_in],
                c="purple",
                label="HWP Angle",
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
                    label="Demod0",
                )
            if "QU" in weight_mode:
                ax.plot(
                    demod_data.obs[0].shared[defaults.times].data[slc_demod],
                    demod_data.obs[0].detdata[defaults.det_data][
                        f"demod4r_{dname}", slc_demod
                    ],
                    c="blue",
                    label="Demod4r",
                )
                ax.plot(
                    demod_data.obs[0].shared[defaults.times].data[slc_demod],
                    demod_data.obs[0].detdata[defaults.det_data][
                        f"demod4i_{dname}", slc_demod
                    ],
                    c="green",
                    label="Demod4i",
                )
            ax.legend(loc="best")
            plt.title("Demodulation")
            plt.savefig(tod_plot)
            plt.close()

            fname_mod = os.path.join(
                self.outdir, f"modulated_{weight_mode}{suffix}_map.fits"
            )
            fname_demod = os.path.join(
                self.outdir, f"demodulated_{weight_mode}{suffix}_map.fits"
            )
            fname_hits = os.path.join(
                self.outdir, f"demodulated_{weight_mode}{suffix}_hits.fits"
            )

            map_mod = hp.read_map(fname_mod, None)
            map_demod = np.atleast_2d(hp.read_map(fname_demod, None))
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
            for i, m in enumerate(map_mod):
                # Modulated map is full IQU
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
                    title=f"Demodulated {stokes} : rms = {rms / rms0:.3f} x rms(in)",
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
                        f"input - demodulated map RMS = {rms1}, (input RMS = {rms0})",
                        flush=True,
                    )
                    all_good = False

            outfile = os.path.join(
                self.outdir, f"map_comparison.{weight_mode}{suffix}.png"
            )
            fig.savefig(outfile)
            self.assertTrue(all_good)

        if self.comm is not None:
            self.comm.barrier()
        close_data(demod_data)
        close_data(data)

    def test_demodulate_IQU(self):
        data = create_ground_data(self.comm)
        self._test_demodulate(weight_mode="IQU", data=data)

    def test_demodulate_QU(self):
        data = create_ground_data(self.comm)
        self._test_demodulate(weight_mode="QU", data=data)

    def test_demodulate_I(self):
        data = create_ground_data(self.comm)
        self._test_demodulate(weight_mode="I", data=data)

    def test_demodulate_IQU_overdist(self):
        data = create_overdistributed_data(self.comm, single_group=True)
        self._test_demodulate(weight_mode="IQU", data=data, suffix="-overdist")

    def test_demodulate_IQU_detcuts(self):
        data = create_ground_data(self.comm, single_group=True)
        # Flag the second half of the pixels.  Due to the relatively low
        # resolution of the input sky, there is some T->P leakage in the
        # test that blows up when one of the two detectors in a pixel is
        # flagged.
        for ob in data.obs:
            det_flags = dict()
            total_dets = len(ob.local_detectors)
            for idet, det in enumerate(ob.local_detectors):
                if idet >= (total_dets // 4) * 2:
                    det_flags[det] = defaults.det_mask_invalid
            ob.update_local_detector_flags(det_flags)
        self._test_demodulate(weight_mode="IQU", data=data, suffix="-detcuts")

    def test_demodulate_leakage(self):
        data = create_ground_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Pointing operators

        detpointing_azel = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec,
            shared_flag_mask=0,
        )

        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing_azel,
        )

        # Insert a pure intensity signal into the data object
        for ob in data.obs:
            ob.detdata.ensure(defaults.det_data, detectors=ob.local_detectors)
            trend = np.arange(ob.n_local_samples)
            for det in ob.local_detectors:
                ob.detdata[defaults.det_data][det][:] = trend

        # Demodulate

        demod_weights_in = ops.StokesWeights(
            weights="demod_weights_in",
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing_azel,
        )

        demod = ops.Demodulate(
            stokes_weights=demod_weights_in,
            nskip=1,
            purge=False,
            mode="IQU",
        )
        demod_data = demod.apply(data)

        # Confirm that there is no signal in the demodulated Q/U TOD

        for ob, demod_ob in zip(data.obs, demod_data.obs):
            t = ob.shared[defaults.times].data
            for det in ob.select_local_detectors(flagmask=demod.det_mask):
                idet = f"demod0_{det}"
                qdet = f"demod4r_{det}"
                udet = f"demod4i_{det}"
                sig = ob.detdata[defaults.det_data][det]
                isig = demod_ob.detdata[defaults.det_data][idet]
                qsig = demod_ob.detdata[defaults.det_data][qdet]
                usig = demod_ob.detdata[defaults.det_data][udet]

                good = (demod_ob.shared[defaults.shared_flags].data & 1) == 0

                rms = np.std(sig[good])
                irms = np.std(isig[good])
                qrms = np.std(qsig[good]) / irms
                urms = np.std(usig[good]) / irms

                limit = 1e-6
                if qrms > limit or urms > limit:
                    set_matplotlib_backend()
                    import matplotlib.pyplot as plt

                    nrow, ncol = 2, 1
                    fig = plt.figure(figsize=[6 * ncol, 4 * nrow])
                    ax = fig.add_subplot(nrow, ncol, 1)
                    ax.set_title("I -> I")
                    ax.plot(t[good], sig[good], label=f"raw, rms = {rms}")
                    ax.plot(t[good], isig[good], label=f"demod I, rms = {irms}")
                    ax.legend(loc="best")
                    ax = fig.add_subplot(nrow, ncol, 2)
                    ax.set_title("I -> P leakage")
                    ax.plot(t[good], qsig[good], label=f"demod Q, rms = {qrms} x I")
                    ax.plot(t[good], usig[good], label=f"demod U, rms = {urms} x I")
                    ax.legend(loc="best")
                    fname_plot = os.path.join(self.outdir, "i2p_leakage.pdf")
                    plt.savefig(fname_plot)
                    assert False
