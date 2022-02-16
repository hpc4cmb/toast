# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from astropy import units as u

import healpy as hp

from .mpi import MPITestCase

from ..vis import set_matplotlib_backend

from .. import qarray as qa

from .. import ops as ops

from ..observation import default_values as defaults

from ..pixels_io import write_healpix_fits

from ._helpers import (
    create_outdir,
    create_healpix_ring_satellite,
    create_satellite_data,
    create_fake_sky_alm,
    create_fake_beam_alm,
)


class SimConviqtTest(MPITestCase):
    def setUp(self):

        np.random.seed(777)
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        self.nside = 64
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

            # we explicitly store 3 separate beams for the T, E and B sky alm.
            blm_T = np.zeros_like(self.blm)
            blm_T[0] = self.blm[0].copy()
            hp.write_alm(
                self.fname_beam.replace(".fits", "_T.fits"),
                blm_T,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            # in order to evaluate
            # Q + iU ~  Sum[(b^E + ib^B)(a^E + ia^B)] , this implies
            # beamE = [0, blmE, -blmB]

            blm_E = np.zeros_like(self.blm)
            blm_E[1] = self.blm[1]
            blm_E[2] = -self.blm[2]
            hp.write_alm(
                self.fname_beam.replace(".fits", "_E.fits"),
                blm_E,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            # beamB = [0, blmB, blmE]
            blm_B = np.zeros_like(self.blm)
            blm_B[1] = self.blm[2]
            blm_B[2] = self.blm[1]
            hp.write_alm(
                self.fname_beam.replace(".fits", "_B.fits"),
                blm_B,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )

        if self.comm is not None:
            self.comm.barrier()

        return

    def make_beam_file_dict(self, data):
        """
        We make sure that data observed  in each A/B detector within a
        detector pair will have the right beam associated to it. In particular,
        Q and U beams of B detector must have a flipped sign wrt  the A one.
        """

        fname2 = self.fname_beam.replace(".fits", "_bottom.fits")

        self.beam_file_dict = {}
        for det in data.obs[0].local_detectors:
            if det[-1] == "A":
                self.beam_file_dict[det] = self.fname_beam
            else:
                self.beam_file_dict[det] = fname2

        return

    def test_sim_conviqt(self):
        if not ops.conviqt.available():
            print("libconviqt not available, skipping tests")
            return

        # Create a fake scan strategy that hits every pixel once.
        #        data = create_healpix_ring_satellite(self.comm, nside=self.nside)
        data = create_satellite_data(
            self.comm, obs_time=120 * u.min, pixel_per_process=2
        )
        self.make_beam_file_dict(data)

        # Generate timestreams

        detpointing = ops.PointingDetectorSimple()

        key = defaults.det_data
        sim_conviqt = ops.SimConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file_dict=self.beam_file_dict,
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
            det_flags=None,
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

        return

    def test_sim_weighted_conviqt(self):
        if not ops.conviqt.available():
            print("libconviqt not available, skipping tests")
            return

        # Create a fake scan strategy that hits every pixel once.
        #        data = create_healpix_ring_satellite(self.comm, nside=self.nside)
        data = create_satellite_data(
            self.comm, obs_time=120 * u.min, pixel_per_process=2
        )
        self.make_beam_file_dict(data)

        # Generate timestreams

        detpointing = ops.PointingDetectorSimple()

        key1 = "conviqt"
        sim_conviqt = ops.SimConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file_dict=self.beam_file_dict,
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
            beam_file_dict=self.beam_file_dict,
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

        binner1 = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data=key1,
            det_flags=None,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner1.apply(data)
        binner2 = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data=key2,
            det_flags=None,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner2.apply(data)
        # Study the map on the root process

        toast_bin_path = os.path.join(self.outdir, "toast_bin.conviqt.fits")
        write_healpix_fits(data[binner1.binned], toast_bin_path, nest=pixels.nest)
        toast_bin_path = os.path.join(self.outdir, "toast_bin.wconviqt.fits")
        write_healpix_fits(data[binner2.binned], toast_bin_path, nest=pixels.nest)

        toast_hits_path = os.path.join(self.outdir, "toast_hits.fits")
        write_healpix_fits(data[cov_and_hits.hits], toast_hits_path, nest=pixels.nest)

        fail = False
        if self.rank == 0:
            mapfile = os.path.join(self.outdir, "toast_bin.conviqt.fits")
            mdata = hp.read_map(mapfile, field=range(3))
            mapfile = os.path.join(self.outdir, "toast_bin.wconviqt.fits")
            mdataw = hp.read_map(mapfile, field=range(3))

            cl_out = hp.anafast(mdata, lmax=self.lmax)
            cl_outw = hp.anafast(mdataw, lmax=self.lmax)

            np.testing.assert_almost_equal(cl_out, cl_outw, decimal=2)

        return

    def test_sim_TEB_conviqt(self):
        if not ops.conviqt.available():
            print("libconviqt not available, skipping tests")
            return

        # Create a fake scan strategy that hits every pixel once.
        #        data = create_healpix_ring_satellite(self.comm, nside=self.nside)
        data = create_satellite_data(
            self.comm, obs_time=120 * u.min, pixel_per_process=2
        )
        self.make_beam_file_dict(data)

        # Generate timestreams

        detpointing = ops.PointingDetectorSimple()

        key1 = "conviqt0"
        sim_conviqt = ops.SimConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file_dict=self.beam_file_dict,
            dxx=False,
            det_data=key1,
            pol=True,
            normalize_beam=False,
            fwhm=self.fwhm_sky,
        )

        sim_conviqt.apply(data)
        ## For TEB convolution there's no need to differentiate between beams for detA and detB
        beam_file_dict_teb = {}
        for det in data.obs[0].local_detectors:
            beam_file_dict_teb[det] = self.fname_beam

        key2 = "tebconviqt"

        sim_wconviqt = ops.SimTEBConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file_dict=beam_file_dict_teb,
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

        binner1 = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data=key1,
            det_flags=None,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner1.apply(data)
        binner2 = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data=key2,
            det_flags=None,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner2.apply(data)
        # Study the map on the root process

        toast_bin_path = os.path.join(self.outdir, f"toast_bin.{key1}.fits")
        write_healpix_fits(data[binner1.binned], toast_bin_path, nest=pixels.nest)
        toast_bin_path = os.path.join(self.outdir, f"toast_bin.{key2}.fits")
        write_healpix_fits(data[binner2.binned], toast_bin_path, nest=pixels.nest)

        toast_hits_path = os.path.join(self.outdir, "toast_hits.fits")
        write_healpix_fits(data[cov_and_hits.hits], toast_hits_path, nest=pixels.nest)

        fail = False
        if self.rank == 0:
            mapfile = os.path.join(self.outdir, f"toast_bin.{key1}.fits")
            mdata = hp.read_map(mapfile, field=range(3))
            mapfile = os.path.join(self.outdir, f"toast_bin.{key2}.fits")
            mdataw = hp.read_map(mapfile, field=range(3))

            cl_out = hp.anafast(mdata, lmax=self.lmax)
            cl_outw = hp.anafast(mdataw, lmax=self.lmax)

            np.testing.assert_almost_equal(cl_out, cl_outw, decimal=2)

        return

    """
    def test_sim_hwp(self):
        if not ops.conviqt.available():
            print("libconviqt not available, skipping tests")
            return
        # Create a fake scan strategy that hits every pixel once.
        data = create_satellite_data(self.comm)
        # make a simple pointing matrix
        detpointing = ops.PointingDetectorSimple()
        # Generate timestreams
        sim_conviqt = ops.SimWeightedConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file=self.fname_beam,
            dxx=False,
            hwp_angle="hwp_angle",
        )
        sim_conviqt.exec(data)
        return
    """
