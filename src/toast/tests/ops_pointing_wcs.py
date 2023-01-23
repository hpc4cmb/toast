# Copyright (c) 2021-2022 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import astropy.io.fits as af
import numpy as np
from astropy import units as u
from astropy.wcs import WCS

from .. import ops as ops
from .. import qarray as qa
from .. import templates
from ..data import Data
from ..observation import Observation
from ..observation import default_values as defaults
from ..pixels_io_wcs import write_wcs_fits
from ..vis import set_matplotlib_backend
from ._helpers import (
    close_data,
    create_boresight_telescope,
    create_comm,
    create_fake_sky,
    create_ground_data,
    create_outdir,
    plot_wcs_maps,
)
from .mpi import MPITestCase


class PointingWCSTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        # For debugging, change this to True
        self.write_extra = False

    def check_hits(self, prefix, pixels):
        wcs = pixels.wcs

        toastcomm = create_comm(self.comm)
        data = Data(toastcomm)
        tele = create_boresight_telescope(
            toastcomm.group_size,
            sample_rate=1.0 * u.Hz,
        )

        # Make some fake boresight pointing
        npix_ra = pixels.pix_ra
        npix_dec = pixels.pix_dec
        px = list()
        for ra in range(npix_ra):
            px.extend(
                np.column_stack(
                    [
                        ra * np.ones(npix_dec),
                        np.arange(npix_dec),
                    ]
                ).tolist()
            )
        px = np.array(px, dtype=np.float64)
        coord = wcs.wcs_pix2world(px, 0)
        checkpx = wcs.wcs_world2pix(coord, 0)
        coord *= np.pi / 180.0
        phi = np.array(coord[:, 0], dtype=np.float64)
        half_pi = np.pi / 2
        theta = np.array(half_pi - coord[:, 1], dtype=np.float64)
        bore = qa.from_iso_angles(theta, phi, np.zeros_like(theta))

        nsamp = npix_ra * npix_dec
        data.obs.append(Observation(toastcomm, tele, n_samples=nsamp))
        data.obs[0].shared.create_column(
            defaults.boresight_radec, (nsamp, 4), dtype=np.float64
        )
        data.obs[0].shared.create_column(
            defaults.shared_flags, (nsamp,), dtype=np.uint8
        )
        if toastcomm.group_rank == 0:
            data.obs[0].shared[defaults.boresight_radec].set(bore)
        else:
            data.obs[0].shared[defaults.boresight_radec].set(None)

        pixels.apply(data)

        # Hitmap

        build_hits = ops.BuildHitMap(
            pixel_dist=pixels.create_dist,
            pixels=pixels.pixels,
            det_flags=None,
        )
        build_hits.apply(data)

        if self.write_extra:
            outfile = os.path.join(self.outdir, f"{prefix}.fits")
            write_wcs_fits(data[build_hits.hits], outfile)

            if toastcomm.world_rank == 0:
                set_matplotlib_backend()

                import matplotlib.pyplot as plt

                hdu = af.open(outfile)[0]
                wcs = WCS(hdu.header)

                fig = plt.figure(figsize=(8, 8), dpi=100)
                ax = fig.add_subplot(projection=wcs, slices=("x", "y", 0))
                # plt.imshow(hdu.data, vmin=-2.e-5, vmax=2.e-4, origin='lower')
                im = ax.imshow(
                    np.transpose(hdu.data[0, :, :]), vmin=0, vmax=4, cmap="jet"
                )
                ax.grid(color="white", ls="solid")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                plt.colorbar(im, orientation="vertical")
                fig.savefig(os.path.join(self.outdir, f"{prefix}.pdf"), format="pdf")

        np.testing.assert_array_equal(
            data[build_hits.hits].data,
            data.comm.ngroups
            * len(data.obs[0].all_detectors)
            * np.ones_like(data[build_hits.hits].data),
        )
        close_data(data)

    def test_projections(self):
        centers = list()
        for lon in [130.0, 180.0, 230.0]:
            for lat in [-40.0, 0.0, 40.0]:
                centers.append((lon * u.degree, lat * u.degree))

        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec
        )

        for proj in ["CAR", "TAN", "CEA", "MER", "ZEA"]:
            for center in centers:
                pixels = ops.PixelsWCS(
                    projection=proj,
                    detector_pointing=detpointing_radec,
                    create_dist="dist",
                    use_astropy=True,
                    center=center,
                    dimensions=(710, 350),
                    resolution=(0.1 * u.degree, 0.1 * u.degree),
                )
                self.check_hits(
                    f"hits_{proj}_{center[0].value}_{center[1].value}", pixels
                )
                if self.comm is not None:
                    self.comm.barrier()

    def test_mapmaking(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Test several projections
        resolution = 0.1 * u.degree

        # for proj in ["CAR", "TAN", "CEA", "MER", "ZEA"]:
        for proj in ["CAR"]:
            # Create fake observing of a small patch
            data = create_ground_data(self.comm)

            # Simple detector pointing
            detpointing_radec = ops.PointingDetectorSimple(
                boresight=defaults.boresight_radec,
            )

            # Stokes weights
            weights = ops.StokesWeights(
                mode="IQU",
                hwp_angle=defaults.hwp_angle,
                detector_pointing=detpointing_radec,
            )

            # Pixelization
            pixels = ops.PixelsWCS(
                detector_pointing=detpointing_radec,
                projection=proj,
                # resolution=(0.5 * u.degree, 0.5 * u.degree),
                # auto_bounds=True,
                use_astropy=True,
            )

            pix_dist = ops.BuildPixelDistribution(
                pixel_dist="pixel_dist",
                pixel_pointing=pixels,
            )
            pix_dist.apply(data)

            # Create fake polarized sky pixel values locally
            create_fake_sky(data, "pixel_dist", "fake_map")

            if self.write_extra:
                # Write it out
                outfile = os.path.join(self.outdir, f"mapmaking_{proj}_input.fits")
                write_wcs_fits(data["fake_map"], outfile)
                if rank == 0:
                    plot_wcs_maps(mapfile=outfile)
            if data.comm.comm_world is not None:
                data.comm.comm_world.barrier()

            # Scan map into timestreams
            scanner = ops.Pipeline(
                operators=[
                    pixels,
                    weights,
                    ops.ScanMap(
                        det_data=defaults.det_data,
                        pixels=pixels.pixels,
                        weights=weights.weights,
                        map_key="fake_map",
                    ),
                ],
                detsets=["SINGLE"],
            )
            scanner.apply(data)

            # Create an uncorrelated noise model from focalplane detector properties
            default_model = ops.DefaultNoiseModel(noise_model="noise_model")
            default_model.apply(data)

            # Simulate noise and accumulate to signal
            sim_noise = ops.SimNoise(
                noise_model=default_model.noise_model, det_data=defaults.det_data
            )
            sim_noise.apply(data)

            # Set up binning operator for solving
            binner = ops.BinMap(
                pixel_dist="pixel_dist",
                pixel_pointing=pixels,
                stokes_weights=weights,
                noise_model=default_model.noise_model,
            )

            # Set up template matrix with just an offset template.

            # Use 1/10 of an observation as the baseline length.  Make it not evenly
            # divisible in order to test handling of the final amplitude.
            ob_time = (
                data.obs[0].shared[defaults.times][-1]
                - data.obs[0].shared[defaults.times][0]
            )
            step_seconds = float(int(ob_time / 10.0))
            tmpl = templates.Offset(
                times=defaults.times,
                noise_model=default_model.noise_model,
                step_time=step_seconds * u.second,
            )
            tmatrix = ops.TemplateMatrix(templates=[tmpl])

            # Map maker
            mapper = ops.MapMaker(
                name=f"test_{proj}",
                det_data=defaults.det_data,
                binning=binner,
                template_matrix=tmatrix,
                solve_rcond_threshold=1.0e-2,
                map_rcond_threshold=1.0e-2,
                write_hits=False,
                write_map=False,
                write_cov=False,
                write_rcond=False,
                output_dir=self.outdir,
                keep_solver_products=True,
                keep_final_products=True,
            )

            if data.comm.comm_world is not None:
                data.comm.comm_world.barrier()
            mapper.apply(data)

            if self.write_extra:
                # Write outputs manually
                for prod in ["hits", "map"]:
                    outfile = os.path.join(self.outdir, f"mapmaking_{proj}_{prod}.fits")
                    write_wcs_fits(data[f"{mapper.name}_{prod}"], outfile)

                if rank == 0:
                    outfile = os.path.join(self.outdir, f"mapmaking_{proj}_hits.fits")
                    plot_wcs_maps(hitfile=outfile)

                    outfile = os.path.join(self.outdir, f"mapmaking_{proj}_map.fits")
                    plot_wcs_maps(mapfile=outfile)

            close_data(data)

    def fake_source(self, mission_start, ra_start, dec_start, times, deg_per_hour=1.0):
        deg_sec = deg_per_hour / 3600.0
        t_start = float(times[0])
        first_ra = ra_start + (t_start - mission_start) * deg_sec
        first_dec = dec_start + (t_start - mission_start) * deg_sec
        incr = (times - t_start) * deg_sec
        return first_ra + incr, first_dec + incr

    def create_source_data(
        self, data, proj, res, signal_name, deg_per_hour=1.0, dbg_dir=None
    ):
        detpointing = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec,
            quats="temp_quats",
        )
        detpointing.apply(data)

        # Normal autoscaled projection
        pixels = ops.PixelsWCS(
            projection=proj,
            resolution=(res, res),
            detector_pointing=detpointing,
            pixels="temp_pix",
            use_astropy=True,
            auto_bounds=True,
        )

        pix_dist = ops.BuildPixelDistribution(
            pixel_dist="source_pixel_dist",
            pixel_pointing=pixels,
        )
        pix_dist.apply(data)

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, "source_pixel_dist", "fake_map")

        if self.write_extra:
            # Write it out
            outfile = os.path.join(self.outdir, f"source_{proj}_input.fits")
            write_wcs_fits(data["fake_map"], outfile)
            if data.comm.world_rank == 0:
                plot_wcs_maps(mapfile=outfile)

        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            weights="temp_weights",
            detector_pointing=detpointing,
        )
        weights.apply(data)

        # Scan map into timestreams
        scanner = ops.Pipeline(
            operators=[
                pixels,
                weights,
                ops.ScanMap(
                    det_data=signal_name,
                    pixels=pixels.pixels,
                    weights=weights.weights,
                    map_key="fake_map",
                ),
            ],
            detsets=["SINGLE"],
        )
        scanner.apply(data)

        # Use this overall projection window to determine our source
        # movement.  The source starts at the center of the projection.
        px = np.array(
            [
                [
                    int(0.6 * pixels.pix_ra),
                    int(0.2 * pixels.pix_dec),
                ],
            ],
            dtype=np.float64,
        )
        source_start = pixels.wcs.wcs_pix2world(px, 0)

        # Create the fake ephemeris data and accumulate to signal.
        for ob in data.obs:
            n_samp = ob.n_local_samples
            times = np.array(ob.shared[defaults.times].data)

            source_ra, source_dec = self.fake_source(
                data.obs[0].shared["times"][0],
                source_start[0][0],
                source_start[0][1],
                times,
                deg_per_hour=deg_per_hour,
            )
            source_coord = np.column_stack([source_ra, source_dec])

            # Create a shared data object with the fake source location
            ob.shared.create_column("source", (n_samp, 2), dtype=np.float64)
            if ob.comm.group_rank == 0:
                ob.shared["source"].set(source_coord)
            else:
                ob.shared["source"].set(None)

            zaxis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            sphi = ob.shared["source"].data[:, 0] * np.pi / 180.0
            stheta = (90.0 - ob.shared["source"].data[:, 1]) * np.pi / 180.0
            spos = qa.from_iso_angles(stheta, sphi, np.zeros_like(stheta))
            sdir = qa.rotate(spos, zaxis)
            for det in ob.local_detectors:
                fwhm = 2 * ob.telescope.focalplane[det]["fwhm"].to_value(u.arcmin)
                coeff = 1.0 / (fwhm * np.sqrt(2 * np.pi))
                pre = -0.5 * (1.0 / fwhm) ** 2
                quats = ob.detdata[detpointing.quats][det]
                dir = qa.rotate(quats, zaxis)
                sdist = np.arccos([np.dot(x, y) for x, y in zip(sdir, dir)])
                sdist_arc = sdist * 180.0 * 60.0 / np.pi
                seen = sdist_arc < 10
                seen_samp = np.arange(len(sdist), dtype=np.int32)
                amp = 50.0 * coeff * np.exp(pre * np.square(sdist_arc))
                ob.detdata[signal_name][det, :] += amp[:]

            if dbg_dir is not None and ob.comm.group_rank == 0:
                set_matplotlib_backend()

                import matplotlib.pyplot as plt

                fig = plt.figure(figsize=(8, 8), dpi=100)
                ax = fig.add_subplot()
                ax.scatter(
                    ob.shared["source"].data[:, 0],
                    ob.shared["source"].data[:, 1],
                )
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                plt.savefig(
                    os.path.join(dbg_dir, f"source_coord_{proj}_{ob.name}.pdf"),
                    format="pdf",
                )
                plt.close()

        default_model = ops.DefaultNoiseModel(noise_model="source_noise_model")
        default_model.apply(data)

        # Simulate noise and accumulate to signal
        sim_noise = ops.SimNoise(
            noise_model=default_model.noise_model, det_data=signal_name
        )
        sim_noise.apply(data)

        if dbg_dir is not None:
            # Make a binned map of the signal in the non-source-centered projection
            binner = ops.BinMap(
                pixel_dist="source_pixel_dist",
                pixel_pointing=pixels,
                stokes_weights=weights,
                noise_model=default_model.noise_model,
            )
            mapper = ops.MapMaker(
                name=f"source_{proj}_notrack",
                det_data=signal_name,
                solve_rcond_threshold=1.0e-2,
                map_rcond_threshold=1.0e-2,
                iter_max=10,
                binning=binner,
                template_matrix=None,
                output_dir=dbg_dir,
                write_hits=True,
                write_map=True,
                write_binmap=False,
            )
            mapper.apply(data)
            if data.comm.world_rank == 0:
                outfile = os.path.join(self.outdir, f"source_{proj}_notrack_hits.fits")
                plot_wcs_maps(hitfile=outfile)
                outfile = os.path.join(self.outdir, f"source_{proj}_notrack_map.fits")
                plot_wcs_maps(mapfile=outfile)

        # Cleanup our temp objects
        ops.Delete(
            detdata=[detpointing.quats, pixels.pixels, weights.weights],
            meta=["source_pixel_dist", "source_noise_model"],
        ).apply(data)

    def test_source_map(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Test several projections
        resolution = 0.5 * u.degree

        for proj in ["CAR", "TAN"]:
            # Create fake observing of a small patch
            data = create_ground_data(self.comm, pixel_per_process=10)

            # Create source motion and simulated detector data.
            dbgdir = None
            if proj == "CAR" and self.write_extra:
                dbgdir = self.outdir
            self.create_source_data(
                data, proj, resolution, defaults.det_data, dbg_dir=dbgdir
            )

            # Simple detector pointing
            detpointing_radec = ops.PointingDetectorSimple(
                boresight=defaults.boresight_radec,
            )

            # Stokes weights
            weights = ops.StokesWeights(
                mode="IQU",
                hwp_angle=defaults.hwp_angle,
                detector_pointing=detpointing_radec,
            )

            # Source-centered pointing
            pixels = ops.PixelsWCS(
                projection=proj,
                resolution=(resolution, resolution),
                center_offset="source",
                detector_pointing=detpointing_radec,
                use_astropy=True,
                auto_bounds=True,
            )

            pix_dist = ops.BuildPixelDistribution(
                pixel_dist="pixel_dist",
                pixel_pointing=pixels,
            )
            pix_dist.apply(data)

            default_model = ops.DefaultNoiseModel(noise_model="noise_model")
            default_model.apply(data)

            # Set up binning operator for solving
            binner = ops.BinMap(
                pixel_dist="pixel_dist",
                pixel_pointing=pixels,
                stokes_weights=weights,
                noise_model=default_model.noise_model,
            )

            # Set up template matrix with just an offset template.

            # Use 1/10 of an observation as the baseline length.  Make it not evenly
            # divisible in order to test handling of the final amplitude.
            ob_time = (
                data.obs[0].shared[defaults.times][-1]
                - data.obs[0].shared[defaults.times][0]
            )
            step_seconds = float(int(ob_time / 10.0))
            tmpl = templates.Offset(
                times=defaults.times,
                det_flags=None,
                noise_model=default_model.noise_model,
                step_time=step_seconds * u.second,
            )
            tmatrix = ops.TemplateMatrix(templates=[tmpl])

            # Map maker
            mapper = ops.MapMaker(
                name=f"source_{proj}",
                det_data=defaults.det_data,
                solve_rcond_threshold=1.0e-2,
                map_rcond_threshold=1.0e-2,
                iter_max=10,
                binning=binner,
                template_matrix=tmatrix,
                output_dir=self.outdir,
                write_hits=True,
                write_map=True,
                write_binmap=True,
            )
            mapper.apply(data)

            if rank == 0:
                outfile = os.path.join(self.outdir, f"source_{proj}_hits.fits")
                plot_wcs_maps(hitfile=outfile)
                outfile = os.path.join(self.outdir, f"source_{proj}_map.fits")
                plot_wcs_maps(mapfile=outfile)
                outfile = os.path.join(self.outdir, f"source_{proj}_binmap.fits")
                plot_wcs_maps(mapfile=outfile)

            close_data(data)
