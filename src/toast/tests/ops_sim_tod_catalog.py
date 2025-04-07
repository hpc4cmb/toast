# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
import tomlkit
from astropy import units as u

from .. import ops as ops
from ..observation import default_values as defaults
from ..vis import set_matplotlib_backend
from .helpers import (
    create_ground_data,
    create_outdir,
)
from .mpi import MPITestCase


class SimCatalogTest(MPITestCase):
    def setUp(self):
        np.random.seed(777)
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

        self.rank = 0
        if self.comm is not None:
            self.rank = self.comm.rank

        self.catalog_file = os.path.join(self.outdir, "catalog.txt")
        if self.rank == 0:
            # Write a point source catalog to simulate from
            catalog = {}

            # Example static source
            catalog["static_source"] = {
                "ra_deg": 41,
                "dec_deg": -41,
                "freqs_ghz": [1.0, 1000.0],
                "flux_density_Jy": [10.0, 1.0],
                "pol_frac": 0.1,
                "pol_angle_deg": 0,
            }

            # Example variable source
            # (the operator will not extrapolate)
            catalog["variable_source"] = {
                "ra_deg": 41,
                "dec_deg": -43,
                "freqs_ghz": [1.0, 1000.0],
                "flux_density_Jy": [
                    [10.0, 1.0],
                    [30.0, 10.0],
                    [10.0, 1.0],
                ],
                "times_mjd": [58800.0, 58850.0, 58900.0],
                "pol_frac": [0.05, 0.15, 0.05],
                "pol_angle_deg": [45, 45, 45],
            }

            # Example transient source
            # (the operator will not extrapolate outside times_mjd)
            catalog["transient_source"] = {
                "ra_deg": 43,
                "dec_deg": -43,
                "freqs_ghz": [1.0, 1000.0],
                "flux_density_Jy": [
                    [10.0, 1.0],
                    [30.0, 10.0],
                ],
                "times_mjd": [
                    58849.0,
                    58850.0,
                ],
            }

            with open(self.catalog_file, "w") as f:
                f.write(tomlkit.dumps(catalog))

        if self.comm is not None:
            self.comm.barrier()

        return

    def test_sim_catalog(self):
        # Create a fake ground data set for testing.  It targets a small patch at
        # RA = [40, 44], Dec = [-44, -40]
        data = create_ground_data(self.comm, turnarounds_invalid=True, hwp_rpm=0)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Create some detector pointing matrices

        nside = 1024

        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=nside,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        weights = ops.StokesWeights(
            mode="IQU",
            detector_pointing=detpointing,
            hwp_angle=defaults.hwp_angle,
        )

        # Simulate point sources

        sim_catalog = ops.SimCatalog(
            catalog_file=self.catalog_file,
            detector_pointing=detpointing,
        )
        sim_catalog.apply(data)

        # Map the signal to check

        binner = ops.BinMap(
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
        )

        mapper = ops.MapMaker(
            name="sources",
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

        if data.comm.world_rank == 0:
            set_matplotlib_backend()
            import matplotlib.pyplot as plt

            fname_map = os.path.join(self.outdir, "sources_map.fits")
            m = hp.read_map(fname_map, None)

            nrow, ncol = 1, 3
            fig = plt.figure(figsize=[6 * ncol, 4 * nrow])
            for i, mm in enumerate(np.atleast_2d(m)):
                hp.gnomview(
                    mm, sub=[nrow, ncol, 1 + i], rot=[42, -42], xsize=800, reso=0.5
                )

            outfile = os.path.join(self.outdir, "map.png")
            fig.savefig(outfile)
