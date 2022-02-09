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



def create_fake_beam_alm(
    lmax=128,
    mmax=10,
    fwhm_x=10 * u.degree,
    fwhm_y=10 * u.degree,
    pol=True,
    separate_IQU=False,
):

    # pick an nside >= lmax to be sure that the a_lm will be fairly accurate
    nside = 2
    while nside < lmax:

        nside *= 2
    npix = 12 * nside ** 2
    pix = np.arange(npix)
    x,y,z  = hp.pix2vec(nside, pix, nest=False)  
    sigma_z = fwhm_x.to_value(u.radian) / np.sqrt(8 * np.log(2))
    sigma_y = fwhm_y.to_value(u.radian) / np.sqrt(8 * np.log(2))
    beam = np.exp(-((z ** 2 /2/sigma_z**2 + y ** 2 /2/ sigma_y**2)))
    beam[x < 0] = 0
    tmp_beam_map = np.zeros([3, npix ])
    tmp_beam_map[0] = beam
    tmp_beam_map[1] = beam
    bl, blm = hp.anafast(tmp_beam_map, lmax=lmax, iter=0, alm=True,pol=True)
    hp.rotate_alm(blm, psi=0, theta=-np.pi/2, phi=0)
    if pol and separate_IQU:

        beam_map= hp.alm2map(blm, nside=nside) 
        empty = np.zeros_like(beam_map[0])
        
        beam_map_I = np.vstack([beam_map[0], empty, empty])
        beam_map_Q = np.vstack([empty, beam_map[0], empty])
        beam_map_U = np.vstack([empty, empty, beam_map[0]])

        try:
            a_lm = [
                hp.map2alm(beam_map_I, lmax=lmax, mmax=mmax, verbose=False),
                hp.map2alm(beam_map_Q, lmax=lmax, mmax=mmax, verbose=False),
                hp.map2alm(beam_map_U, lmax=lmax, mmax=mmax, verbose=False),
            ]
        except TypeError:
            # older healpy which does not have verbose keyword
            a_lm = [
                hp.map2alm(beam_map_I, lmax=lmax, mmax=mmax),
                hp.map2alm(beam_map_Q, lmax=lmax, mmax=mmax),
                hp.map2alm(beam_map_U, lmax=lmax, mmax=mmax),
            ]
        return a_lm 
    else:
 

        return blm

class SimConviqtTest(MPITestCase):
    def setUp(self):
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
            #self.slm[1:] = 0  # No polarization
            hp.write_alm(self.fname_sky, self.slm, lmax=self.lmax, overwrite=True)

            self.blm = create_fake_beam_alm(
                self.lmax,
                self.mmax,
                fwhm_x=self.fwhm_beam,
                fwhm_y=self.fwhm_beam,
            )
            #self.blm[1] = self.blm[0] 
            #self.blm[2] = np.zeros_like(self.blm[0] ) 
            
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
            #beamB = [0, blmB, blmE] 
            blm_B = np.zeros_like(self.blm)
            blm_B[1] =  self.blm[2] 
            blm_B[2] =  self.blm[1]  
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

    def test_sim(self):
        if not ops.conviqt.available():
            print("libconviqt not available, skipping tests")
            return

        # Create a fake scan strategy that hits every pixel once.
        #        data = create_healpix_ring_satellite(self.comm, nside=self.nside)
        data = create_satellite_data(self.comm , obs_time=120*u.min, pixel_per_process=2 )

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
            pol=True ,
            normalize_beam=True  ,
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
            hwp_angle= None ,
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

        if self.rank == 10:
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
            outfile = os.path.join(self.outdir, "cl_comparison.png")
            plt.savefig(outfile)
            plt.close()

            compare = blsq > 1e-5
            ref = cl_in[compare] * blsq[compare] * deconv[compare] ** 2
            norm = np.mean(cl_out[compare] / ref)
            print(norm * ref,cl_out[compare] )
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
     
    def test_TEBconvolution(self):
        if not ops.conviqt.available():
            print("libconviqt not available, skipping tests")
            return

        # Create a fake scan strategy that hits every pixel once.
        data = create_healpix_ring_satellite(self.comm, nside=self.nside)

        # Generate timestreams

        detpointing = ops.PointingDetectorSimple()

       

        conviqt_key = "conviqt_tod"
        sim_conviqt = ops.SimConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file=self.fname_beam ,
            dxx=False,
            det_data=conviqt_key,
            #normalize_beam=True,
            fwhm=self.fwhm_sky,
        )
        sim_conviqt.exec(data)
        
        teb_conviqt_key  = "tebconviqt_tod"
        sim_teb = ops.SimTEBConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file=self.fname_beam ,
            dxx=False,
            det_data=teb_conviqt_key,
            #normalize_beam=True,
            fwhm=self.fwhm_sky,
        )
        sim_teb.exec(data)
        # Bin both signals into maps

        pixels = ops.PixelsHealpix(
            nside=self.nside,
            nest=False,
            detector_pointing=detpointing,
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
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
            det_data=teb_conviqt_key,
            det_flags=None,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner.apply(data)
        path_teb = os.path.join(self.outdir, "toast_bin.tebconviqt.fits")
        
        write_healpix_fits(data[binner.binned], path_teb , nest=False)
        
        binner2 = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data=conviqt_key,
            det_flags=None,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner2.apply(data)
        path_conviqt = os.path.join(self.outdir, "toast_bin.conviqt.fits")
        write_healpix_fits(data[binner2.binned], path_conviqt, nest=False)
        
        print(data[binner2.binned].data.shape) 
        np.testing.assert_almost_equal(data[binner2.binned].data ,    data[binner.binned].data, decimal=6)
        
####################         
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        fail = False

        if rank == 0:
            import matplotlib.pyplot as plt

            sky = hp.alm2map(self.slm , self.nside, lmax=self.lmax, verbose=False)
            beam = hp.alm2map(
                self.blm  ,
                self.nside,
                lmax=self.lmax,
                mmax=self.mmax,
                verbose=False,
            )

            #map_teb = hp.read_map(path_teb , field=range(3) )
            map_conviqt = hp.read_map(path_conviqt   )

            # For some reason, matplotlib hangs with multiple tasks,
            # even if only one writes.
            if self.comm is None or self.comm.size == 1:
                for i, pol in enumerate( 'I'):
                    fig = plt.figure(figsize=[12, 8])
                    nrow, ncol = 2, 2
                    hp.mollview(sky[i], title="input sky", sub=[nrow, ncol, 1])
                    hp.mollview(beam[i], title="beam", sub=[nrow, ncol, 2], rot=[0, 90])
                    #amp = np.amax(map_conviqt[i])/4
                    hp.mollview(
                        map_teb[i],
                        min=-amp,
                        max=amp,
                        title="TEB conviqt",
                        sub=[nrow, ncol, 3],
                    ) 
                    hp.mollview(
                        map_conviqt,
                        #min=-amp,
                        #max=amp,
                        title="conviqt",
                        sub=[nrow, ncol, 4],
                    )
                    outfile = os.path.join(self.outdir, f"map_comparison{pol}.png")
                    fig.savefig(outfile)
            for obs in data.obs:
                for det in obs.local_detectors:
                    tod_teb = obs.detdata[teb_conviqt_key][det]
                    tod_conviqt = obs.detdata[conviqt_key][det]
                    if not np.allclose(
                        tod_teb,
                        tod_conviqt,
                        rtol=1e-3,
                        atol=1e-3,
                    ):
                        import matplotlib.pyplot as plt
                        import pdb

                        pdb.set_trace()
                        fail = True
                        break
                if fail:
                    break
        if data.comm.comm_world is not None:
            fail = data.comm.comm_world.bcast(fail, root=0)

        self.assertFalse(fail)

        return
    
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