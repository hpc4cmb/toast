# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
from astropy import units as u

from .. import ops as ops
from .. import qarray as qa
from ..observation import default_values as defaults
from ..pixels import PixelData
from ..pixels_io_healpix import write_healpix_fits
from ..vis import set_matplotlib_backend
from ._helpers import close_data, create_ground_data, create_outdir
from .mpi import MPITestCase


class DemodulateTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_demodulate(self):
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

        # Build the pixel distribution
        build_dist = ops.BuildPixelDistribution(pixel_pointing=pixels)
        build_dist.apply(data)

        # Create a fake sky with only intensity and Q-polarization

        map_key = "fake_map"
        dist_key = build_dist.pixel_dist
        dist = data[dist_key]
        pix_data = PixelData(dist, np.float64, n_value=3, units=u.K)
        off = 0
        map_values = [10, -1, 2]
        for submap in range(dist.n_submap):
            if submap in dist.local_submaps:
                pix_data.data[off, :, 0] = map_values[0]
                pix_data.data[off, :, 1] = map_values[1]
                pix_data.data[off, :, 2] = map_values[2]
                off += 1
        data[map_key] = pix_data

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data=defaults.det_data,
            pixels=pixels.pixels,
            weights=weights.weights,
            map_key=map_key,
        )
        scan_pipe = ops.Pipeline(
            detector_sets=["SINGLE"],
            operators=[
                pixels,
                weights,
                scanner,
            ],
        )
        scan_pipe.apply(data)

        # Simulate noise
        # sim_noise = ops.SimNoise(noise_model=default_model.noise_model)
        # sim_noise.apply(data)

        # Bin signal without demodulation

        binner = ops.BinMap(
            pixel_dist=dist_key,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
        )

        mapper = ops.MapMaker(
            name="modulated",
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

        # Demodulate

        demod = ops.Demodulate(stokes_weights=weights, purge=True)
        # demod.purge = False
        demod_data = demod.apply(data)

        # ops.Delete(detdata=[defaults.weights]).apply(demod_data)

        # Map again

        default_model.apply(demod_data)

        demod_weights = ops.StokesWeightsDemod()

        mapper.name = "demodulated"
        binner.stokes_weights = demod_weights
        mapper.apply(demod_data)

        if data.comm.world_rank == 0:
            set_matplotlib_backend()
            import matplotlib.pyplot as plt

            fname_mod = os.path.join(self.outdir, "modulated_map.fits")
            fname_demod = os.path.join(self.outdir, "demodulated_map.fits")

            map_mod = hp.read_map(fname_mod, None)
            map_demod = hp.read_map(fname_demod, None)

            fig = plt.figure(figsize=[18, 12])
            nrow, ncol = 2, 3
            rot = [42, -42]
            reso = 5

            for i, m in enumerate(map_mod):
                value = map_values[i]
                good = m != 0
                rms = np.sqrt(np.mean((m[good] - value) ** 2))
                m[m == 0] = hp.UNSEEN
                stokes = "IQU"[i]
                hp.gnomview(
                    m,
                    sub=[nrow, ncol, 1 + i],
                    reso=reso,
                    rot=rot,
                    title=f"Modulated {stokes} : rms = {rms}",
                    cmap="coolwarm",
                )

            for i, m in enumerate(map_demod):
                value = map_values[i]
                good = m != 0
                rms = np.sqrt(np.mean((m[good] - value) ** 2))
                m[m == 0] = hp.UNSEEN
                stokes = "IQU"[i]
                amp = 0.0001
                hp.gnomview(
                    m,
                    sub=[nrow, ncol, 4 + i],
                    reso=reso,
                    rot=rot,
                    title=f"Demodulated {stokes} : rms = {rms}",
                    min=value - amp,
                    max=value + amp,
                    cmap="coolwarm",
                )
                if rms > 1.0e-3:
                    print(
                        f"WARNING:  demodulated map RMS = {rms}, which is larger than 1e-3",
                        flush=True,
                    )
                    # self.assertTrue(False)

            outfile = os.path.join(self.outdir, "map_comparison.png")
            fig.savefig(outfile)

        if self.comm is not None:
            self.comm.barrier()
        close_data(demod_data)
        close_data(data)
