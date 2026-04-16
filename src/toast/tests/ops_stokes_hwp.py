# Copyright (c) 2026-2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
from astropy import units as u

from .. import ops as ops
from ..observation import default_values as defaults
from ..pixels_io_healpix import read_healpix
from .helpers import (
    close_data,
    create_outdir,
    create_ground_data,
    create_fake_healpix_scanned_tod,
)
from .mpi import MPITestCase


class StokesWeightsHWPTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        self.nside = 64

    def create_test_data(self, testdir):
        # Slightly slower than 1 Hz
        hwp_rpm = 59.0
        sample_rate = 60 * u.Hz

        # Create a fake ground observations set for testing
        data = create_ground_data(
            self.comm, pixel_per_process=7, sample_rate=sample_rate, hwp_rpm=hwp_rpm
        )

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Generic pointing matrix for sampling from the map
        detpointing = ops.PointingDetectorSimple(shared_flag_mask=0)
        pixels = ops.PixelsHealpix(
            nside=self.nside,
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )

        # Create some fake sky tod
        skyfile = os.path.join(testdir, "input_sky.fits")
        map_key = "input_sky"
        create_fake_healpix_scanned_tod(
            data,
            pixels,
            weights,
            skyfile,
            "input_sky_dist",
            map_key=map_key,
            fwhm=30.0 * u.arcmin,
            lmax=3 * pixels.nside,
            I_scale=0.001,
            Q_scale=0.0001,
            U_scale=0.0001,
            det_data=defaults.det_data,
        )

        # Simulate noise from this model and save the result for comparison
        sim_noise = ops.SimNoise(noise_model="noise_model", out=defaults.det_data)
        sim_noise.apply(data)

        # Clean up simulated pointing used for the map scanning
        ops.Delete(detdata=[pixels.pixels, weights.weights]).apply(data)
        return data

    def plot_results(self, file_root, nnz):
        import matplotlib.pyplot as plt

        hit_file = f"{file_root}_hits.fits"
        rcond_file = f"{file_root}_rcond.fits"
        map_file = f"{file_root}_map.fits"
        invcov_file = f"{file_root}_invcov.fits"

        # The standard plotting tools in toast do not know how to handle
        # the types of maps we are making, so we plot these manually.

        def lonlat_range(nside, pix):
            lon, lat = hp.pix2ang(
                nside,
                pix,
                nest=True,
                lonlat=True,
            )
            sortlon = np.sort(lon)
            safelon = np.unwrap(sortlon, period=360.0)
            maxlon = np.amax(safelon)
            minlon = np.amin(safelon)
            maxlat = np.amax(lat)
            minlat = np.amin(lat)

            loncenter = (maxlon + minlon) / 2
            latcenter = (maxlat + minlat) / 2
            lonhalf = 1.2 * ((maxlon - minlon) / 2)
            lathalf = 1.2 * ((maxlat - minlat) / 2)
            lonspan = [loncenter - lonhalf, loncenter + lonhalf]
            latspan = [latcenter - lathalf, latcenter + lathalf]

            prot = (loncenter, latcenter, 0.0)
            return prot, lonspan, latspan

        hitdata = read_healpix(hit_file, field=None, nest=True)
        maxhits = np.amax(hitdata)
        npix = len(hitdata)
        nside = hp.npix2nside(npix)

        goodhits = hitdata > 0
        badhits = np.logical_not(goodhits)
        goodindx = np.arange(npix, dtype=np.int32)[goodhits]
        rot, lonspan, latspan = lonlat_range(nside, goodindx)

        xsize = 1600
        gnomres = 1.2 * (latspan[1] - latspan[0]) / xsize
        gnomres *= 60
        gnomres *= 1.2
        grat_res = int((gnomres / 60.0) * (xsize / 10))

        out_file = f"{hit_file}.png"
        hp.gnomview(
            map=hitdata,
            rot=rot,
            xsize=xsize,
            reso=gnomres,
            nest=True,
            cmap="bwr",
            min=0,
            max=maxhits,
            title="Hit Map",
        )
        hp.graticule(dpar=grat_res, dmer=grat_res)
        plt.savefig(out_file, format="png")
        plt.close()

        rcond = read_healpix(rcond_file, field=None, nest=True)
        rcond[badhits] = np.nan
        out_file = f"{rcond_file}.png"
        hp.gnomview(
            map=rcond,
            rot=rot,
            xsize=xsize,
            reso=gnomres,
            nest=True,
            cmap="bwr",
            title="Inverse Condition Number",
        )
        hp.graticule(dpar=grat_res, dmer=grat_res)
        plt.savefig(out_file, format="png")
        plt.close()

        for imap in range(nnz):
            mdata = read_healpix(map_file, field=imap, nest=True)
            mdata[badhits] = np.nan
            out_file = f"{map_file}_{imap}.png"
            hp.gnomview(
                map=mdata,
                rot=rot,
                xsize=xsize,
                reso=gnomres,
                nest=True,
                cmap="bwr",
                title=f"Map Component {imap}",
            )
            hp.graticule(dpar=grat_res, dmer=grat_res)
            plt.savefig(out_file, format="png")
            plt.close()

        # Make a panel plot of N_pp'^-1 blocks
        invcov = read_healpix(invcov_file, field=None, nest=True)
        print(f"invcov shape = {invcov.shape}", flush=True)
        print(f"There are {len(goodindx)} hit pixels", flush=True)
        nnz = int((np.sqrt(1 + 8 * invcov.shape[0]) - 1) // 2)
        n_plot = len(goodindx)

        # Extract the per-pixel matrices
        pix_data = np.zeros((n_plot, nnz, nnz))
        elem = 0
        for row in range(nnz):
            for col in range(row, nnz):
                ivdata = invcov[elem]
                for ipix, idx in enumerate(goodindx):
                    pix_data[ipix, row, col] = ivdata[idx]
                    pix_data[ipix, col, row] = ivdata[idx]
                elem += 1
        del invcov

        # Giant panel plot...
        n_col = 5
        n_row = n_plot // n_col
        if n_row * n_col != n_plot:
            n_row += 1
        panel_inches = 4
        fig, axes = plt.subplots(
            nrows=n_row,
            ncols=n_col,
            figsize=(n_col * panel_inches, n_row * panel_inches),
            dpi=100,
        )
        iplot = 0
        for irow in range(n_row):
            for icol in range(n_col):
                if iplot >= n_plot:
                    axes[irow, icol].set_visible(False)
                    continue
                ax = axes[irow, icol]
                cond = np.linalg.cond(pix_data[iplot])
                im = ax.imshow(
                    pix_data[iplot], interpolation=None, cmap="bwr", vmin=0, vmax=1000.0
                )
                ax.set_title(f"Pix {goodindx[iplot]}, Cond = {cond:0.2e}")
                iplot += 1
        # Colorbar
        fig.colorbar(im, ax=axes.ravel().tolist(), aspect=50)
        out_file = f"{invcov_file}.png"
        plt.savefig(out_file, format="png")
        plt.close()

    def test_nominal(self):
        testdir = os.path.join(self.outdir, "nominal")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir, exist_ok=True)

        data = self.create_test_data(testdir)
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Pointing model
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=self.nside,
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeightsHWP(
            mode="nominal",
            detector_pointing=detpointing,
        )

        # Binned mapmaking
        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model="noise_model",
        )

        mapper = ops.MapMaker(
            name="mapmaker",
            det_data=defaults.det_data,
            binning=binner,
            map_rcond_threshold=1.0e-15,
            write_hits=True,
            write_map=True,
            write_noiseweighted_map=True,
            write_invcov=True,
            write_rcond=True,
            output_dir=testdir,
        )

        # Make the map
        mapper.apply(data)

        close_data(data)

        if rank == 0:
            file_root = os.path.join(testdir, mapper.name)
            self.plot_results(file_root, 9)

    def test_mueller_ideal(self):
        testdir = os.path.join(self.outdir, "mueller_ideal")
        if self.comm is None or self.comm.rank == 0:
            os.makedirs(testdir, exist_ok=True)

        data = self.create_test_data(testdir)
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Pointing model
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=self.nside,
            detector_pointing=detpointing,
        )

        # Define both the usual stokes weights and use our HWP specific
        # one with an ideal Mueller matrix.  This should give the same result.
        weights_iqu = ops.StokesWeights(
            mode="IQU",
            detector_pointing=detpointing,
            hwp_angle=defaults.hwp_angle,
        )

        weights_hwp = ops.StokesWeightsHWP(
            mode="mueller",
            detector_pointing=detpointing,
            mueller=[
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, -1],
            ],
        )

        # Binned mapmaking
        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights_hwp,
            noise_model="noise_model",
        )

        mapper = ops.MapMaker(
            name="mapmaker_hwp",
            det_data=defaults.det_data,
            binning=binner,
            map_rcond_threshold=1.0e-15,
            write_hits=True,
            write_map=True,
            write_noiseweighted_map=True,
            write_invcov=True,
            write_rcond=True,
            output_dir=testdir,
            reset_pix_dist=True,
        )

        # Make the map
        mapper.apply(data)

        # Change the Stokes weights operator and re-run
        binner.stokes_weights = weights_iqu
        mapper.name = "mapmaker_iqu"
        mapper.apply(data)

        close_data(data)

        if rank == 0:
            iqu_root = os.path.join(testdir, "mapmaker_iqu")
            self.plot_results(iqu_root, 3)
            hwp_root = os.path.join(testdir, "mapmaker_hwp")
            self.plot_results(hwp_root, 3)

            # Verify that the maps agree
            hiqu = read_healpix(f"{iqu_root}_hits.fits", field=0, nest=True)
            good = hiqu > 0
            for imap in range(3):
                miqu = read_healpix(f"{iqu_root}_map.fits", field=imap, nest=True)
                mhwp = read_healpix(f"{hwp_root}_map.fits", field=imap, nest=True)
                self.assertTrue(np.allclose(mhwp[good], miqu[good]))
