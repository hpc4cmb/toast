# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
from astropy import units as u

from .. import ops as ops
from .. import qarray as qa
from ..observation import default_values as defaults
from ..pixels_io_healpix import write_healpix_fits
from ..vis import set_matplotlib_backend
from ._helpers import (
    close_data,
    create_fake_beam_alm,
    create_fake_sky_alm,
    create_outdir,
    create_satellite_data,
)
from .mpi import MPITestCase


class SimConviqtTest(MPITestCase):
    def setUp(self):
        np.random.seed(777)
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        self.nside = 32
        self.lmax = 128
        self.fwhm_sky = 10 * u.degree
        self.fwhm_beam = 15 * u.degree
        self.mmax = self.lmax
        self.fname_sky = os.path.join(self.outdir, "sky_alm.fits")
        self.fname_beam = os.path.join(self.outdir, "beam_alm.fits")

        self.rank = 0
        if self.comm is not None:
            self.rank = self.comm.rank

        if self.rank == 0:
            # Synthetic sky and beam (a_lm expansions)
            self.slm = create_fake_sky_alm(self.lmax, self.fwhm_sky)

            hp.write_alm(self.fname_sky, self.slm, lmax=self.lmax, overwrite=True)

            # Inputs for the TEB convolution
            # FIXME : the TEB conviqt operator should do this match rather than leave it for the user
            hp.write_alm(
                self.fname_sky.replace(".fits", "_components_T.fits"),
                self.slm[0],
                lmax=self.lmax,
                overwrite=True,
            )
            slm = self.slm.copy()
            slm[0] = 0
            hp.write_alm(
                self.fname_sky.replace(".fits", "_components_EB.fits"),
                slm,
                lmax=self.lmax,
                overwrite=True,
            )
            slm[1] = self.slm[2]
            slm[2] = -self.slm[1]
            hp.write_alm(
                self.fname_sky.replace(".fits", "_components_BE.fits"),
                slm,
                lmax=self.lmax,
                overwrite=True,
            )

            self.blm = create_fake_beam_alm(
                self.lmax,
                self.mmax,
                fwhm_x=self.fwhm_beam,
                fwhm_y=self.fwhm_beam,
                normalize_beam=True,
            )

            self.blm_bottom = create_fake_beam_alm(
                self.lmax,
                self.mmax,
                fwhm_x=self.fwhm_beam,
                fwhm_y=self.fwhm_beam,
                normalize_beam=True,
                detB_beam=True,
            )

            hp.write_alm(
                self.fname_beam,
                self.blm,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            hp.write_alm(
                self.fname_beam.replace(".fits", "_bottom.fits"),
                self.blm_bottom,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            blm_I, blm_Q, blm_U = create_fake_beam_alm(
                self.lmax,
                self.mmax,
                fwhm_x=self.fwhm_beam,
                fwhm_y=self.fwhm_beam,
                separate_IQU=True,
                normalize_beam=True,
            )
            blm_Ibot, blm_Qbot, blm_Ubot = create_fake_beam_alm(
                self.lmax,
                self.mmax,
                fwhm_x=self.fwhm_beam,
                fwhm_y=self.fwhm_beam,
                separate_IQU=True,
                detB_beam=True,
                normalize_beam=True,
            )
            hp.write_alm(
                self.fname_beam.replace(".fits", "_I000.fits"),
                blm_I,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            hp.write_alm(
                self.fname_beam.replace(".fits", "_0I00.fits"),
                blm_Q,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            hp.write_alm(
                self.fname_beam.replace(".fits", "_00I0.fits"),
                blm_U,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            hp.write_alm(
                self.fname_beam.replace(".fits", "_bottom_I000.fits"),
                blm_Ibot,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            hp.write_alm(
                self.fname_beam.replace(".fits", "_bottom_0I00.fits"),
                blm_Qbot,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            hp.write_alm(
                self.fname_beam.replace(".fits", "_bottom_00I0.fits"),
                blm_Ubot,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )

            # we explicitly store temperature and polarization beams
            # FIXME : Another place where the operator should handle the decomposition.
            blm_T, blm_P = create_fake_beam_alm(
                self.lmax,
                self.mmax,
                fwhm_x=self.fwhm_beam,
                fwhm_y=self.fwhm_beam,
                separate_TP=True,
                detB_beam=False,
                normalize_beam=True,
            )
            blm_Tbot, blm_Pbot = create_fake_beam_alm(
                self.lmax,
                self.mmax,
                fwhm_x=self.fwhm_beam,
                fwhm_y=self.fwhm_beam,
                separate_TP=True,
                detB_beam=True,
                normalize_beam=True,
            )
            hp.write_alm(
                self.fname_beam.replace(".fits", "_components_T.fits"),
                blm_T,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            hp.write_alm(
                self.fname_beam.replace(".fits", "_bottom_components_T.fits"),
                blm_Tbot,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            hp.write_alm(
                self.fname_beam.replace(".fits", "_components_P.fits"),
                blm_P,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            hp.write_alm(
                self.fname_beam.replace(".fits", "_bottom_components_P.fits"),
                blm_Pbot,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )

        if self.comm is not None:
            self.comm.barrier()

        return

    def make_beam_file_dict(self, data):
        """
        We make sure that data observed in each A/B detector within a
        detector pair will have the right beam associated to it.  In particular,
        Q and U beams of B detector must have a flipped sign wrt the A one.
        """

        fname_top = self.fname_beam
        fname_bottom = fname_top.replace(".fits", "_bottom.fits")

        beam_file_dict = {}
        for det in data.obs[0].all_detectors:
            try:
                a_index = det.index("A")
            except ValueError:
                a_index = -1
            try:
                b_index = det.index("B")
            except ValueError:
                b_index = -1
            if a_index > b_index:
                beam_file_dict[det] = fname_top
            elif b_index > a_index:
                beam_file_dict[det] = fname_bottom
            else:
                msg = f"Cannot determine detector type: {det}"
                raise RuntimeError(msg)
        return beam_file_dict

    def test_sim_conviqt(self):
        if not ops.conviqt.available():
            print("libconviqt not available, skipping tests")
            return

        data = create_satellite_data(
            self.comm, obs_time=120 * u.min, pixel_per_process=2
        )
        beam_file_dict = self.make_beam_file_dict(data)

        # Generate timestreams

        detpointing = ops.PointingDetectorSimple()

        key = defaults.det_data
        sim_conviqt = ops.SimConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file_dict=beam_file_dict,
            dxx=False,
            det_data=key,
            pol=True,
            normalize_beam=False,  # beams are already produced normalized
            fwhm=self.fwhm_sky,
        )

        sim_conviqt.apply(data)

        # Bin a map to study

        pixels = ops.PixelsHealpix(
            nside=self.nside,
            nest=False,
            detector_pointing=detpointing,
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=None,
            detector_pointing=detpointing,
        )
        weights.apply(data)

        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        cov_and_hits = ops.CovarianceAndHits(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            rcond_threshold=1.0e-6,
            sync_type="alltoallv",
        )
        cov_and_hits.apply(data)

        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data=key,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner.apply(data)

        # Study the map on the root process

        toast_bin_path = os.path.join(self.outdir, "toast_bin.fits")
        write_healpix_fits(data[binner.binned], toast_bin_path, nest=pixels.nest)

        toast_hits_path = os.path.join(self.outdir, "toast_hits.fits")
        write_healpix_fits(data[cov_and_hits.hits], toast_hits_path, nest=pixels.nest)

        fail = False

        if self.rank == 0:
            hitsfile = os.path.join(self.outdir, "toast_hits.fits")

            hdata = hp.read_map(hitsfile)

            footprint = np.ma.masked_not_equal(hdata, 0.0).mask

            mapfile = os.path.join(self.outdir, "toast_bin.fits")
            mdata = hp.read_map(mapfile, field=range(3))

            deconv = 1 / hp.gauss_beam(
                self.fwhm_sky.to_value(u.radian),
                lmax=self.lmax,
                pol=False,
            )

            smoothed = hp.alm2map(
                [hp.almxfl(self.slm[ii], deconv) for ii in range(3)],
                self.nside,
                lmax=self.lmax,
                fwhm=self.fwhm_beam.to_value(u.radian),
                verbose=False,
                pixwin=False,
            )
            smoothed[:, ~footprint] = 0
            cl_out = hp.anafast(mdata, lmax=self.lmax)
            cl_smoothed = hp.anafast(smoothed, lmax=self.lmax)

            np.testing.assert_almost_equal(cl_smoothed[0], cl_out[0], decimal=2)

        close_data(data)

    def test_sim_weighted_conviqt(self):
        if not ops.conviqt.available():
            print("libconviqt not available, skipping tests")
            return

        data = create_satellite_data(
            self.comm, obs_time=120 * u.min, pixel_per_process=2
        )
        beam_file_dict = self.make_beam_file_dict(data)

        # Generate timestreams

        detpointing = ops.PointingDetectorSimple()

        key1 = "conviqt"
        sim_conviqt = ops.SimConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file_dict=beam_file_dict,
            dxx=False,
            det_data=key1,
            pol=True,
            normalize_beam=False,
            fwhm=self.fwhm_sky,
        )

        sim_conviqt.apply(data)

        key2 = "wconviqt"

        sim_wconviqt = ops.SimWeightedConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file_dict=beam_file_dict,
            dxx=False,
            det_data=key2,
            pol=True,
            normalize_beam=False,
            fwhm=self.fwhm_sky,
        )

        sim_wconviqt.apply(data)
        # Bin a map to study

        pixels = ops.PixelsHealpix(
            nside=self.nside,
            nest=False,
            detector_pointing=detpointing,
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=None,
            detector_pointing=detpointing,
        )
        weights.apply(data)

        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        cov_and_hits = ops.CovarianceAndHits(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            rcond_threshold=1.0e-6,
            sync_type="alltoallv",
        )
        cov_and_hits.apply(data)

        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )

        binner.det_data = key1
        binner.binned = "binned1"
        binner.apply(data)
        binner.det_data = key2
        binner.binned = "binned2"
        binner.apply(data)

        # Study the map on the root process

        toast_bin_path = os.path.join(self.outdir, f"toast_bin.{key1}.fits")
        write_healpix_fits(data["binned1"], toast_bin_path, nest=pixels.nest)
        toast_bin_path = os.path.join(self.outdir, f"toast_bin.{key2}.fits")
        write_healpix_fits(data["binned2"], toast_bin_path, nest=pixels.nest)

        toast_hits_path = os.path.join(self.outdir, "toast_hits.fits")
        write_healpix_fits(data[cov_and_hits.hits], toast_hits_path, nest=pixels.nest)

        fail = False
        if self.rank == 0:
            mapfile = os.path.join(self.outdir, f"toast_bin.{key1}.fits")
            mdata = hp.read_map(mapfile, field=range(3))
            mapfile = os.path.join(self.outdir, f"toast_bin.{key1}.fits")
            mdataw = hp.read_map(mapfile, field=range(3))

            cl_out = hp.anafast(mdata, lmax=self.lmax)
            cl_outw = hp.anafast(mdataw, lmax=self.lmax)

            np.testing.assert_almost_equal(cl_out, cl_outw, decimal=2)

        close_data(data)

    def test_sim_TEB_conviqt(self):
        if not ops.conviqt.available():
            print("libconviqt not available, skipping tests")
            return

        data = create_satellite_data(
            self.comm,
            obs_time=120 * u.min,
            pixel_per_process=2,
            hwp_rpm=None,
        )

        beam_file_dict = self.make_beam_file_dict(data)

        # Generate timestreams

        detpointing = ops.PointingDetectorSimple()

        key1 = "conviqt0"
        sim_conviqt = ops.SimConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file_dict=beam_file_dict,
            dxx=False,
            det_data=key1,
            pol=True,
            normalize_beam=False,
            fwhm=self.fwhm_sky,
        )

        sim_conviqt.apply(data)

        key2 = "tebconviqt"

        sim_wconviqt = ops.SimTEBConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file_dict=beam_file_dict,
            dxx=False,
            det_data=key2,
            pol=True,
            normalize_beam=False,
            fwhm=self.fwhm_sky,
        )

        sim_wconviqt.apply(data)
        # Bin a map to study

        pixels = ops.PixelsHealpix(
            nside=self.nside,
            nest=False,
            detector_pointing=detpointing,
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=None,
            detector_pointing=detpointing,
        )
        weights.apply(data)

        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        cov_and_hits = ops.CovarianceAndHits(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            rcond_threshold=1.0e-6,
            sync_type="alltoallv",
        )
        cov_and_hits.apply(data)

        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )

        binner.det_data = key1
        binner.binned = "binned1"
        binner.apply(data)
        binner.det_data = key2
        binner.binned = "binned2"
        binner.apply(data)

        # Study the map on the root process

        toast_bin_path = os.path.join(self.outdir, f"toast_bin.{key1}.fits")
        write_healpix_fits(data["binned1"], toast_bin_path, nest=pixels.nest)
        toast_bin_path = os.path.join(self.outdir, f"toast_bin.{key2}.fits")
        write_healpix_fits(data["binned2"], toast_bin_path, nest=pixels.nest)

        toast_hits_path = os.path.join(self.outdir, "toast_hits.fits")
        write_healpix_fits(data[cov_and_hits.hits], toast_hits_path, nest=pixels.nest)

        fail = False
        if self.rank == 0:
            import matplotlib.pyplot as plt

            mapfile = os.path.join(self.outdir, f"toast_bin.{key1}.fits")
            mdata = hp.read_map(mapfile, field=range(3))
            mapfile = os.path.join(self.outdir, f"toast_bin.{key2}.fits")
            mdataw = hp.read_map(mapfile, field=range(3))

            bl_in = hp.gauss_beam(
                self.fwhm_sky.to_value(u.radian), lmax=self.lmax, pol=True
            ).T
            bl_out = hp.gauss_beam(
                self.fwhm_beam.to_value(u.radian), lmax=self.lmax, pol=True
            ).T
            slm = self.slm.copy()
            bl_in[bl_in == 0] = 1
            bl_out[bl_out == 0] = 1
            for i in range(3):
                slm[i] = hp.almxfl(slm[i], bl_out[i] / bl_in[i])
            reference = hp.alm2map(slm, self.nside, lmax=self.lmax, pixwin=True)

            for m in mdata, mdataw:
                m[m == 0] = hp.UNSEEN
            reference[m == hp.UNSEEN] = hp.UNSEEN

            args = {"rot": [156, 11], "xsize": 1200, "reso": 3, "cmap": "bwr"}
            nrow, ncol = 3, 3
            fig = plt.figure(figsize=[18, 12])
            for col, stokes in enumerate("IQU"):
                hp.gnomview(
                    mdata[col],
                    **args,
                    sub=[nrow, ncol, 1 + col],
                    title=f"{stokes} SimConviqt",
                )
                hp.gnomview(
                    mdataw[col],
                    **args,
                    sub=[nrow, ncol, 4 + col],
                    title=f"{stokes} SimTEBConviqt",
                )
                hp.gnomview(
                    reference[col],
                    **args,
                    sub=[nrow, ncol, 7 + col],
                    title=f"{stokes} Reference",
                )
            fname = os.path.join(self.outdir, "map_comparison_TEB.png")
            fig.savefig(fname)

            cl_out = hp.anafast(mdata, lmax=self.lmax)
            cl_outw = hp.anafast(mdataw, lmax=self.lmax)

            np.testing.assert_almost_equal(cl_out, cl_outw, decimal=2)

        close_data(data)

    def test_sim_hwp(self):
        if not ops.conviqt.available():
            print("libconviqt not available, skipping tests")
            return

        data_wo_hwp = create_satellite_data(
            self.comm,
            obs_time=120 * u.min,
            pixel_per_process=2,
            hwp_rpm=None,
        )
        data_w_hwp = create_satellite_data(
            self.comm,
            obs_time=120 * u.min,
            pixel_per_process=2,
            hwp_rpm=100,
        )
        beam_file_dict_wo_hwp = self.make_beam_file_dict(data_wo_hwp)
        beam_file_dict_w_hwp = self.make_beam_file_dict(data_w_hwp)

        # Generate timestreams

        detpointing = ops.PointingDetectorSimple()

        key_wo_hwp = "conviqt_wo_hwp"
        sim_wo_hwp_conviqt = ops.SimConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file_dict=beam_file_dict_wo_hwp,
            dxx=False,
            det_data=key_wo_hwp,
            pol=True,
            normalize_beam=False,
            fwhm=self.fwhm_sky,
            hwp_angle=None,
        )

        sim_wo_hwp_conviqt.apply(data_wo_hwp)

        key_w_hwp = "tebconviqt_w_hwp"

        sim_w_hwp_conviqt = ops.SimTEBConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file_dict=beam_file_dict_w_hwp,
            dxx=False,
            det_data=key_w_hwp,
            pol=True,
            normalize_beam=False,
            fwhm=self.fwhm_sky,
            hwp_angle="hwp_angle",
        )

        sim_w_hwp_conviqt.apply(data_w_hwp)
        # Bin a map to study

        pixels = ops.PixelsHealpix(
            nside=self.nside,
            nest=False,
            detector_pointing=detpointing,
        )
        pixels.apply(data_w_hwp)
        pixels.apply(data_wo_hwp)

        weights_wo_hwp = ops.StokesWeights(
            mode="IQU",
            hwp_angle=None,
            detector_pointing=detpointing,
        )
        weights_wo_hwp.apply(data_wo_hwp)

        default_model = ops.DefaultNoiseModel()
        default_model.apply(data_w_hwp)
        default_model.apply(data_wo_hwp)

        cov_and_hits_wo_hwp = ops.CovarianceAndHits(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights_wo_hwp,
            noise_model=default_model.noise_model,
            rcond_threshold=1.0e-6,
            sync_type="alltoallv",
        )
        cov_and_hits_wo_hwp.apply(data_wo_hwp)

        binner_wo_hwp = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits_wo_hwp.covariance,
            det_flags=None,
            pixel_pointing=pixels,
            stokes_weights=weights_wo_hwp,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )

        binner_wo_hwp.det_data = key_wo_hwp
        binner_wo_hwp.binned = "binned_wo_hwp"
        binner_wo_hwp.apply(data_wo_hwp)

        weights_w_hwp = ops.StokesWeights(
            mode="IQU",
            hwp_angle="hwp_angle",
            detector_pointing=detpointing,
        )
        weights_w_hwp.apply(data_w_hwp)

        cov_and_hits_w_hwp = ops.CovarianceAndHits(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights_w_hwp,
            noise_model=default_model.noise_model,
            rcond_threshold=1.0e-6,
            sync_type="alltoallv",
        )
        cov_and_hits_w_hwp.apply(data_w_hwp)

        binner_w_hwp = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits_w_hwp.covariance,
            det_flags=None,
            pixel_pointing=pixels,
            stokes_weights=weights_w_hwp,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner_w_hwp.det_data = key_w_hwp
        binner_w_hwp.binned = "binned_w_hwp"
        binner_w_hwp.apply(data_w_hwp)

        # Study the map on the root process

        toast_bin_path = os.path.join(self.outdir, f"toast_bin.{key_wo_hwp}.fits")
        write_healpix_fits(
            data_wo_hwp["binned_wo_hwp"], toast_bin_path, nest=pixels.nest
        )
        toast_bin_path = os.path.join(self.outdir, f"toast_bin.{key_w_hwp}.fits")
        write_healpix_fits(data_w_hwp["binned_w_hwp"], toast_bin_path, nest=pixels.nest)

        toast_hits_path = os.path.join(self.outdir, "toast_hits_wo_hwp.fits")
        write_healpix_fits(
            data_wo_hwp[cov_and_hits_wo_hwp.hits], toast_hits_path, nest=pixels.nest
        )
        toast_hits_path = os.path.join(self.outdir, "toast_hits_w_hwp.fits")
        write_healpix_fits(
            data_w_hwp[cov_and_hits_w_hwp.hits], toast_hits_path, nest=pixels.nest
        )
        fail = False
        if self.rank == 0:
            import matplotlib.pyplot as plt

            bl_in = hp.gauss_beam(
                self.fwhm_sky.to_value(u.radian), lmax=self.lmax, pol=True
            ).T
            bl_out = hp.gauss_beam(
                self.fwhm_beam.to_value(u.radian), lmax=self.lmax, pol=True
            ).T
            slm = self.slm.copy()
            bl_in[bl_in == 0] = 1
            bl_out[bl_out == 0] = 1
            for i in range(3):
                slm[i] = hp.almxfl(slm[i], bl_out[i] / bl_in[i])
            reference = hp.alm2map(slm, self.nside, lmax=self.lmax, pixwin=True)

            mapfile = os.path.join(self.outdir, f"toast_bin.{key_wo_hwp}.fits")
            mdata_wo_hwp = hp.read_map(mapfile, field=range(3))
            mapfile = os.path.join(self.outdir, f"toast_bin.{key_w_hwp}.fits")
            mdata_w_hwp = hp.read_map(mapfile, field=range(3))
            diff = mdata_w_hwp - mdata_wo_hwp
            for m in mdata_wo_hwp, mdata_w_hwp, diff:
                m[m == 0] = hp.UNSEEN
            reference[m == hp.UNSEEN] = hp.UNSEEN

            args = {"rot": [156, 11], "xsize": 1200, "reso": 3, "cmap": "bwr"}
            nrow, ncol = 3, 3
            fig = plt.figure(figsize=[18, 12])
            for col, stokes in enumerate("IQU"):
                hp.gnomview(
                    mdata_wo_hwp[col],
                    **args,
                    sub=[nrow, ncol, 1 + col],
                    title=f"{stokes} w/o HWP",
                )
                hp.gnomview(
                    mdata_w_hwp[col],
                    **args,
                    sub=[nrow, ncol, 4 + col],
                    title=f"{stokes} w/ HWP",
                )
                hp.gnomview(
                    reference[col],
                    **args,
                    sub=[nrow, ncol, 7 + col],
                    title=f"{stokes} reference",
                )
            fname = os.path.join(self.outdir, "map_comparison_w_wo_hwp.png")
            fig.savefig(fname)

            cl_reference = hp.anafast(reference, lmax=self.lmax)
            cl_out_wo_hwp = hp.anafast(mdata_wo_hwp, lmax=self.lmax)
            cl_out_w_hwp = hp.anafast(mdata_w_hwp, lmax=self.lmax)

            nrow, ncol = 2, 3
            fig = plt.figure(figsize=[18, 12])
            for col, spec in enumerate(["TT", "EE", "BB"]):
                ax = fig.add_subplot(nrow, ncol, 1 + col)
                ax.set_title(f"{spec}")
                ax.semilogx(cl_reference[col], color="black", label="reference")
                ax.semilogx(cl_out_wo_hwp[col], color="tab:blue", label="w/o HWP")
                ax.semilogx(cl_out_w_hwp[col], color="tab:orange", label="w/ HWP")
                ax = fig.add_subplot(nrow, ncol, 1 + ncol + col)
                ax.set_title(f"{spec}")
                ax.semilogx(
                    cl_reference[col] / cl_out_wo_hwp[col],
                    color="tab:blue",
                    label="ref / w/o HWP",
                )
                ax.semilogx(
                    cl_reference[col] / cl_out_w_hwp[col],
                    color="tab:orange",
                    label="ref / w/ HWP",
                )
            ax.legend(loc="best")
            fname = os.path.join(self.outdir, "cl_comparison_w_wo_hwp.png")
            fig.savefig(fname)

            np.testing.assert_almost_equal(cl_out_wo_hwp, cl_out_w_hwp, decimal=2)

        close_data(data_w_hwp)
        close_data(data_wo_hwp)

    def test_sim_hwp_precomputed(self):
        # Test using precomputed TEB components
        if not ops.conviqt.available():
            print("libconviqt not available, skipping tests")
            return

        data = create_satellite_data(
            self.comm,
            obs_time=120 * u.min,
            pixel_per_process=2,
            hwp_rpm=100,
        )
        data_precomputed = create_satellite_data(
            self.comm,
            obs_time=120 * u.min,
            pixel_per_process=2,
            hwp_rpm=100,
        )
        beam_file_dict = self.make_beam_file_dict(data)
        beam_file_dict_precomputed = self.make_beam_file_dict(data_precomputed)
        for key, path in beam_file_dict_precomputed.items():
            beam_file_dict_precomputed[key] = path.replace(".fits", "_components.fits")

        # Generate timestreams

        detpointing = ops.PointingDetectorSimple()

        key = "conviqt"
        sim_conviqt = ops.SimTEBConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file_dict=beam_file_dict,
            dxx=False,
            det_data=key,
            pol=True,
            normalize_beam=False,
            fwhm=self.fwhm_sky,
            hwp_angle="hwp_angle",
        )

        sim_conviqt.apply(data)

        key_precomputed = "conviqt_precomputed"

        sim_conviqt_precomputed = ops.SimTEBConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky.replace(".fits", "_components.fits"),
            beam_file_dict=beam_file_dict_precomputed,
            dxx=False,
            det_data=key_precomputed,
            pol=True,
            normalize_beam=False,
            fwhm=self.fwhm_sky,
            hwp_angle="hwp_angle",
        )

        sim_conviqt_precomputed.apply(data_precomputed)

        # Compare the simulated TOD. It should be identical.

        for obs1, obs2 in zip(data.obs, data_precomputed.obs):
            for det in obs1.local_detectors:
                sig1 = obs1.detdata[key][det]
                sig2 = obs2.detdata[key_precomputed][det]

                np.testing.assert_almost_equal(sig1, sig2)

        close_data(data)
        close_data(data_precomputed)
