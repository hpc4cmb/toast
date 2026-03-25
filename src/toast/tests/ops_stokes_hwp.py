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
        self.nside = 128

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

    def test_nominal(self):
        data = self.create_test_data(self.outdir)
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
            output_dir=self.outdir,
        )

        # Make the map
        mapper.apply(data)

        close_data(data)

        if rank == 0:
            import matplotlib.pyplot as plt

            hit_file = os.path.join(self.outdir, "mapmaker_hits.fits")
            rcond_file = os.path.join(self.outdir, "mapmaker_rcond.fits")
            map_file = os.path.join(self.outdir, "mapmaker_map.fits")

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

            for imap in range(15):
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
                    title="Map Component {imap}",
                )
                hp.graticule(dpar=grat_res, dmer=grat_res)
                plt.savefig(out_file, format="png")
                plt.close()
