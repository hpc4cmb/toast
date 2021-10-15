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
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        self.nside = 64
        self.lmax = 128
        self.fwhm_sky = 10 * u.degree
        self.fwhm_beam = 30 * u.degree
        self.mmax = self.lmax
        self.fname_sky = os.path.join(self.outdir, "sky_alm.fits")
        self.fname_beam = os.path.join(self.outdir, "beam_alm.fits")

        self.rank = 0
        if self.comm is not None:
            self.rank = self.comm.rank

        if self.rank == 0:
            # Synthetic sky and beam (a_lm expansions)
            self.slm = create_fake_sky_alm(self.lmax, self.fwhm_sky)
            self.slm[1:] = 0  # No polarization
            hp.write_alm(self.fname_sky, self.slm, lmax=self.lmax, overwrite=True)

            self.blm = create_fake_beam_alm(
                self.lmax,
                self.mmax,
                fwhm_x=self.fwhm_beam,
                fwhm_y=self.fwhm_beam,
            )
            hp.write_alm(
                self.fname_beam,
                self.blm,
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

        if self.comm is not None:
            self.comm.barrier()

        return

    def test_sim(self):
        if not ops.conviqt.available():
            print("libconviqt not available, skipping tests")
            return

        # Create a fake scan strategy that hits every pixel once.
        data = create_healpix_ring_satellite(self.comm, nside=self.nside)

        # Generate timestreams

        detpointing = ops.PointingDetectorSimple()

        key = defaults.det_data
        sim_conviqt = ops.SimConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file=self.fname_beam,
            dxx=False,
            det_data=key,
            normalize_beam=True,
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
            mode="I",
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
            set_matplotlib_backend()

            import matplotlib.pyplot as plt

            hitsfile = os.path.join(self.outdir, "toast_hits.fits")
            hdata = hp.read_map(hitsfile)

            hp.mollview(hdata, xsize=1600)
            plt.savefig(os.path.join(self.outdir, "toast_hits.png"))
            plt.close()

            mapfile = os.path.join(self.outdir, "toast_bin.fits")
            mdata = hp.read_map(mapfile)

            hp.mollview(mdata, xsize=1600)
            plt.savefig(os.path.join(self.outdir, "toast_bin.png"))
            plt.close()

            deconv = 1 / hp.gauss_beam(
                self.fwhm_sky.to_value(u.radian),
                lmax=self.lmax,
                pol=False,
            )

            smoothed = hp.alm2map(
                hp.almxfl(self.slm[0], deconv),
                self.nside,
                lmax=self.lmax,
                fwhm=self.fwhm_beam.to_value(u.radian),
                verbose=False,
                pixwin=False,
            )

            cl_out = hp.anafast(mdata, lmax=self.lmax)
            cl_smoothed = hp.anafast(smoothed, lmax=self.lmax)

            cl_in = hp.alm2cl(self.slm[0], lmax=self.lmax)
            blsq = hp.alm2cl(self.blm[0], lmax=self.lmax, mmax=self.mmax)
            blsq /= blsq[1]

            gauss_blsq = hp.gauss_beam(
                self.fwhm_beam.to_value(u.radian),
                lmax=self.lmax,
                pol=False,
            )

            sky = hp.alm2map(self.slm[0], self.nside, lmax=self.lmax, verbose=False)
            beam = hp.alm2map(
                self.blm[0],
                self.nside,
                lmax=self.lmax,
                mmax=self.mmax,
                verbose=False,
            )

            fig = plt.figure(figsize=[12, 8])
            nrow, ncol = 2, 3
            hp.mollview(sky, title="input sky", sub=[nrow, ncol, 1])
            hp.mollview(mdata, title="output sky", sub=[nrow, ncol, 2])
            hp.mollview(smoothed, title="smoothed sky", sub=[nrow, ncol, 3])
            hp.mollview(beam, title="beam", sub=[nrow, ncol, 4], rot=[0, 90])

            ell = np.arange(self.lmax + 1)
            ax = fig.add_subplot(nrow, ncol, 5)
            ax.plot(ell[1:], cl_in[1:], label="input")
            ax.plot(ell[1:], cl_smoothed[1:], label="smoothed")
            ax.plot(ell[1:], blsq[1:], label="beam")
            ax.plot(ell[1:], gauss_blsq[1:], label="gauss beam")
            ax.plot(ell[1:], 1 / deconv[1:] ** 2, label="1 / deconv")
            ax.plot(
                ell[1:],
                cl_in[1:] * blsq[1:] * deconv[1:] ** 2,
                label="input x beam x deconv",
            )
            ax.plot(ell[1:], cl_out[1:], label="output")
            ax.legend(loc="best")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim([1e-20, 1e1])

            # For some reason, matplotlib hangs with multiple tasks,
            # even if only one writes.  Uncomment these lines when debugging.
            #
            # outfile = os.path.join(self.outdir, "cl_comparison.png")
            # plt.savefig(outfile)
            # plt.close()

            compare = blsq > 1e-5
            ref = cl_in[compare] * blsq[compare] * deconv[compare] ** 2
            norm = np.mean(cl_out[compare] / ref)
            if not np.allclose(
                norm * ref,
                cl_out[compare],
                rtol=1e-5,
                atol=1e-5,
            ):
                fail = True

        if self.comm is not None:
            fail = self.comm.bcast(fail, root=0)

        self.assertFalse(fail)

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
