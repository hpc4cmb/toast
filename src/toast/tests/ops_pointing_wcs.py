# Copyright (c) 2021-2025 by the parties listed in the AUTHORS file.
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
from ..pixels_io_wcs import write_wcs
from ..vis import plot_wcs_maps, set_matplotlib_backend
from .helpers import (
    close_data,
    create_boresight_telescope,
    create_comm,
    create_fake_wcs_scanned_tod,
    create_ground_data,
    create_outdir,
)
from .mpi import MPITestCase


class PointingWCSTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        # Uncomment high resolution versions for prettier plots
        # (tests take more time)
        self.proj_dims = (100, 50)
        self.reso = 1.0 * u.degree
        self.fsample = 1.0 * u.Hz
        # self.proj_dims = (1000, 500)
        # self.reso = 0.02 * u.degree
        # self.fsample = 60.0 * u.Hz
        # For debugging, change this to True
        self.write_extra = False

    def create_boresight_pointing(self, pixels):
        # Given a fixed (not auto) wcs spec, simulate boresight pointing
        # that points at each pixel exactly once
        if pixels.auto_bounds:
            raise RuntimeError("Cannot use with auto bounds")
        pixels.set_wcs()
        wcs = pixels.wcs
        nrow, ncol = pixels.wcs_shape

        toastcomm = create_comm(self.comm)
        data = Data(toastcomm)
        tele = create_boresight_telescope(
            toastcomm.group_size,
            sample_rate=1.0 * u.Hz,
        )

        px = list()
        for row in range(nrow):
            for col in range(ncol):
                px.append([col, row])
        coord_deg = wcs.wcs_pix2world(np.array(px, dtype=np.float64), 0)
        coord = np.radians(coord_deg)

        phi = np.array(coord[:, 0], dtype=np.float64)
        half_pi = np.pi / 2
        theta = np.array(half_pi - coord[:, 1], dtype=np.float64)
        bore = qa.from_iso_angles(theta, phi, np.zeros_like(theta))

        nsamp = nrow * ncol
        ob = Observation(toastcomm, tele, n_samples=nsamp)
        ob.shared.create_column(defaults.boresight_radec, (nsamp, 4), dtype=np.float64)
        ob.shared.create_column(defaults.shared_flags, (nsamp,), dtype=np.uint8)
        if toastcomm.group_rank == 0:
            ob.shared[defaults.boresight_radec].set(bore)
        else:
            ob.shared[defaults.boresight_radec].set(None)
        data.obs.append(ob)
        return data

    def check_hits(self, prefix, pixels, data):
        # Clear any existing pointing
        for ob in data.obs:
            if pixels.pixels in ob.detdata:
                del ob.detdata[pixels.pixels]
            if pixels.detector_pointing.quats in ob.detdata:
                del ob.detdata[pixels.detector_pointing.quats]

        # Pixel distribution
        build_dist = ops.BuildPixelDistribution(
            pixel_pointing=pixels,
        )
        if build_dist.pixel_dist in data:
            del data[build_dist.pixel_dist]
        build_dist.apply(data)

        # Expand pointing
        pixels.apply(data)

        # Hitmap
        build_hits = ops.BuildHitMap(
            pixel_dist=build_dist.pixel_dist,
            pixels=pixels.pixels,
            det_flags=None,
        )
        if build_hits.hits in data:
            del data[build_hits.hits]
        build_hits.apply(data)

        if self.write_extra:
            outfile = os.path.join(self.outdir, f"{prefix}.fits")
            write_wcs(data[build_hits.hits], outfile)
            if data.comm.world_rank == 0:
                plot_wcs_maps(hitfile=outfile)

        flat_hits = data[build_hits.hits].data.flatten()
        nonzero = flat_hits != 0
        hits_per_pixel = data.comm.ngroups * len(data.obs[0].all_detectors)
        expected = np.zeros_like(flat_hits)
        expected[nonzero] = hits_per_pixel
        np.testing.assert_array_equal(flat_hits, expected)

    def test_wcs(self):
        # Test basic creation of WCS projections and plotting
        res_deg = (0.01, 0.01)
        dims = self.proj_dims
        center_deg = (130.0, -30.0)
        bounds_deg = (120.0, 140.0, -35.0, -25.0)
        for proj in ["CAR", "TAN", "CEA", "MER", "ZEA", "SFL"]:
            wcs, wcs_shape = ops.PixelsWCS.create_wcs(
                coord="EQU",
                proj=proj,
                center_deg=None,
                bounds_deg=bounds_deg,
                res_deg=res_deg,
                dims=None,
            )
            if self.comm is None or self.comm.rank == 0:
                pixdata = np.ones((1, wcs_shape[1], wcs_shape[0]), dtype=np.float32)
                header = wcs.to_header()
                hdu = af.PrimaryHDU(data=pixdata, header=header)
                outfile = os.path.join(self.outdir, f"test_wcs_{proj}_bounds.fits")
                hdu.writeto(outfile)
                plot_wcs_maps(hitfile=outfile)
        for proj in ["CAR", "TAN", "CEA", "MER", "ZEA", "SFL"]:
            wcs, wcs_shape = ops.PixelsWCS.create_wcs(
                coord="EQU",
                proj=proj,
                center_deg=center_deg,
                bounds_deg=None,
                res_deg=res_deg,
                dims=dims,
            )
            if self.comm is None or self.comm.rank == 0:
                pixdata = np.ones((1, wcs_shape[1], wcs_shape[0]), dtype=np.float32)
                header = wcs.to_header()
                hdu = af.PrimaryHDU(data=pixdata, header=header)
                outfile = os.path.join(self.outdir, f"test_wcs_{proj}_center.fits")
                hdu.writeto(outfile)
                plot_wcs_maps(hitfile=outfile)

    def test_projections(self):
        centers = list()
        for lon in [130.0, 180.0]:
            for lat in [-40.0, 0.0]:
                centers.append((lon * u.degree, lat * u.degree))

        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec
        )

        # For each projection and center, run once and then change resolution
        # test autoscaling.

        for proj in ["CAR", "TAN", "CEA", "MER", "ZEA", "SFL"]:
            for center in centers:
                pixels = ops.PixelsWCS(
                    projection=proj,
                    detector_pointing=detpointing_radec,
                    use_astropy=True,
                )
                # Verify that we can change the projection traits in various ways.
                # First use non-auto_bounds to create one boresight pointing per
                # pixel.
                pixels.auto_bounds = False
                pixels.center = center
                pixels.bounds = ()
                pixels.resolution = (0.02 * u.degree, 0.02 * u.degree)
                pixels.dimensions = self.proj_dims

                data = self.create_boresight_pointing(pixels)
                self.check_hits(
                    f"hits_{proj}_0.02_{center[0].value}_{center[1].value}",
                    pixels,
                    data,
                )

                self.assertFalse(pixels.auto_bounds)
                self.assertTrue(pixels.center == center)
                self.assertTrue(
                    pixels.resolution == (0.02 * u.degree, 0.02 * u.degree)
                )
                self.assertTrue(pixels.dimensions == self.proj_dims)
                self.assertTrue(pixels.dimensions[0] == pixels.wcs_shape[1])
                self.assertTrue(pixels.dimensions[1] == pixels.wcs_shape[0])

                # Note, increasing resolution will leave some pixels
                # un-hit, but the check_hits() helper function will
                # only check pixels with >0 hits. Don't use exactly
                # double resolution or our synthetic pointing only hits
                # pixel boundaries and rounding errors cause sporadic
                # unit test failures.
                pixels.resolution = (0.012 * u.degree, 0.012 * u.degree)
                pixels.center = ()
                pixels.dimensions = ()
                pixels.auto_bounds = True

                self.check_hits(
                    f"hits_{proj}_0.01_{center[0].value}_{center[1].value}_auto",
                    pixels,
                    data,
                )

                self.assertTrue(
                    pixels.resolution == (0.012 * u.degree, 0.012 * u.degree)
                )
                self.assertTrue(pixels.auto_bounds)
                close_data(data)
                if self.comm is not None:
                    self.comm.barrier()

    def test_mapmaking(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Test several projections
        resolution = self.reso

        for proj in ["CAR"]:
            # Create fake observing of a small patch
            data = create_ground_data(self.comm, sample_rate=self.fsample)

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
                resolution=(0.02 * u.degree, 0.02 * u.degree),
                dimensions=(),
                auto_bounds=True,
                use_astropy=True,
            )

            pix_dist = ops.BuildPixelDistribution(
                pixel_dist="pixel_dist",
                pixel_pointing=pixels,
            )
            pix_dist.apply(data)

            # Create fake polarized sky signal
            skyfile = os.path.join(self.outdir, f"mapmaking_{proj}_input.fits")
            map_key = "fake_map"
            create_fake_wcs_scanned_tod(
                data,
                pixels,
                weights,
                skyfile,
                "pixel_dist",
                map_key=map_key,
                fwhm=30.0 * u.arcmin,
                I_scale=0.001,
                Q_scale=0.0001,
                U_scale=0.0001,
                det_data=defaults.det_data,
            )

            if self.write_extra:
                if rank == 0:
                    plot_wcs_maps(mapfile=skyfile)
            if data.comm.comm_world is not None:
                data.comm.comm_world.barrier()

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
                full_pointing=True,
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
                name=f"mapmaking_{proj}",
                det_data=defaults.det_data,
                binning=binner,
                template_matrix=tmatrix,
                solve_rcond_threshold=1.0e-2,
                map_rcond_threshold=1.0e-2,
                write_hits=True,
                write_map=True,
                write_cov=False,
                write_rcond=False,
                output_dir=self.outdir,
                keep_solver_products=False,
                keep_final_products=False,
            )

            if data.comm.comm_world is not None:
                data.comm.comm_world.barrier()
            mapper.apply(data)

            if rank == 0:
                hitfile = os.path.join(self.outdir, f"mapmaking_{proj}_hits.fits")
                mapfile = os.path.join(self.outdir, f"mapmaking_{proj}_map.fits")
                plot_wcs_maps(hitfile=hitfile, mapfile=mapfile)

            close_data(data)

    def fake_source(self, mission_start, ra_start, dec_start, times, deg_per_hour=1.0):
        deg_sec = deg_per_hour / 3600.0
        t_start = float(times[0])
        first_ra = ra_start + (t_start - mission_start) * deg_sec
        first_dec = dec_start + (t_start - mission_start) * deg_sec
        incr = (times - t_start) * deg_sec
        return first_ra + incr, first_dec + incr

    def fake_drone(self, mission_start, az_target, el_target, times, deg_amplitude=1.0):
        # Just simulate moving in a circle around the target location
        t_start = float(times[0])
        t_off = t_start - mission_start
        n_samp = len(times)
        ang = (2 * np.pi / n_samp) * np.arange(n_samp)
        az = az_target + deg_amplitude * np.cos(ang)
        el = el_target + deg_amplitude * np.sin(ang)
        return az, el

    def create_source_data(
        self,
        data,
        proj,
        res,
        signal_name,
        deg_per_hour=1.0,
        azel=False,
        deg_amplitude=1.0,
        dbg_dir=None,
    ):
        if azel:
            detpointing = ops.PointingDetectorSimple(
                boresight=defaults.boresight_azel,
                quats="temp_quats",
            )
            detpointing.apply(data)
        else:
            detpointing = ops.PointingDetectorSimple(
                boresight=defaults.boresight_radec,
                quats="temp_quats",
            )
            detpointing.apply(data)

        # Normal autoscaled projection
        pixels = ops.PixelsWCS(
            projection=proj,
            resolution=(res, res),
            dimensions=(),
            detector_pointing=detpointing,
            pixels="temp_pix",
            use_astropy=True,
            auto_bounds=True,
        )

        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            weights="temp_weights",
            detector_pointing=detpointing,
        )

        pix_dist = ops.BuildPixelDistribution(
            pixel_dist="source_pixel_dist",
            pixel_pointing=pixels,
        )
        pix_dist.apply(data)

        # Create fake polarized sky signal
        skyfile = os.path.join(self.outdir, f"source_{proj}_input.fits")
        map_key = "fake_map"
        create_fake_wcs_scanned_tod(
            data,
            pixels,
            weights,
            skyfile,
            "source_pixel_dist",
            map_key=map_key,
            fwhm=30.0 * u.arcmin,
            I_scale=0.001,
            Q_scale=0.0001,
            U_scale=0.0001,
            det_data=defaults.det_data,
        )

        if self.write_extra:
            if data.comm.world_rank == 0:
                plot_wcs_maps(mapfile=skyfile)

        nrow, ncol = pixels.wcs_shape
        if azel:
            # Simulating a drone near the center
            px = np.array(
                [[int(0.5 * ncol), int(0.5 * nrow)]],
                dtype=np.float64,
            )
        else:
            # Use this overall projection window to determine our source
            # movement.  The source starts at the center of the projection.
            px = np.array(
                [[int(0.6 * ncol), int(0.2 * nrow)]],
                dtype=np.float64,
            )
        source_start = pixels.wcs.wcs_pix2world(px, 0)

        # Create the fake ephemeris data and accumulate to signal.
        for ob in data.obs:
            n_samp = ob.n_local_samples
            times = np.array(ob.shared[defaults.times].data)
            if azel:
                source_lon, source_lat = self.fake_drone(
                    data.obs[0].shared["times"][0],
                    source_start[0][0],
                    source_start[0][1],
                    times,
                    deg_amplitude=deg_amplitude,
                )
            else:
                source_lon, source_lat = self.fake_source(
                    data.obs[0].shared["times"][0],
                    source_start[0][0],
                    source_start[0][1],
                    times,
                    deg_per_hour=deg_per_hour,
                )
            source_coord = np.column_stack([source_lon, source_lat])

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
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                fwhm = 2 * ob.telescope.focalplane[det]["fwhm"].to_value(u.arcmin)
                coeff = 1.0 / (fwhm * np.sqrt(2 * np.pi))
                pre = -0.5 * (1.0 / fwhm) ** 2
                quats = ob.detdata[detpointing.quats][det]
                dir = qa.rotate(quats, zaxis)
                sdist = np.arccos([np.dot(x, y) for x, y in zip(sdir, dir)])
                sdist_arc = sdist * 180.0 * 60.0 / np.pi
                seen = sdist_arc < 10
                seen_samp = np.arange(len(sdist), dtype=np.int32)
                amp = 10.0 * coeff * np.exp(pre * np.square(sdist_arc))
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
                hitfile = os.path.join(self.outdir, f"source_{proj}_notrack_hits.fits")
                mapfile = os.path.join(self.outdir, f"source_{proj}_notrack_map.fits")
                plot_wcs_maps(hitfile=hitfile, mapfile=mapfile)

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
        resolution = self.reso

        for proj in ["TAN"]:
            # Create fake observing of a small patch
            data = create_ground_data(
                self.comm, sample_rate=self.fsample, pixel_per_process=10
            )

            # Create source motion and simulated detector data.
            dbgdir = None
            if self.write_extra:
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
                dimensions=(),
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
                full_pointing=True,
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
                binning=binner,
                template_matrix=tmatrix,
                output_dir=self.outdir,
                write_hits=True,
                write_map=True,
                write_binmap=True,
            )
            mapper.apply(data)

            if rank == 0:
                hitfile = os.path.join(self.outdir, f"source_{proj}_hits.fits")
                mapfile = os.path.join(self.outdir, f"source_{proj}_map.fits")
                binmapfile = os.path.join(self.outdir, f"source_{proj}_binmap.fits")
                plot_wcs_maps(hitfile=hitfile, mapfile=mapfile)
                plot_wcs_maps(hitfile=hitfile, mapfile=binmapfile)

            close_data(data)

    def test_drone_map(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Test several projections
        resolution = self.reso

        for proj in ["SFL"]:
            # Create fake observing of a small patch
            data = create_ground_data(
                self.comm, sample_rate=self.fsample, pixel_per_process=10
            )

            # We are going to hack the boresight pointing so that the RA/DEC simulated
            # pointing is treated as Az/El.  This means that the scan pattern will not
            # be realistic, but at least should cover the source
            for obs in data.obs:
                if obs.comm_col_rank == 0:
                    obs.shared["boresight_azel"].data[:, :] = obs.shared[
                        "boresight_radec"
                    ].data[:, :]

            # Create source motion and simulated detector data.
            dbgdir = None
            if self.write_extra:
                dbgdir = self.outdir
            self.create_source_data(
                data, proj, resolution, defaults.det_data, azel=True, dbg_dir=dbgdir
            )

            # Simple detector pointing
            detpointing_azel = ops.PointingDetectorSimple(
                boresight=defaults.boresight_azel,
            )

            # Stokes weights
            weights = ops.StokesWeights(
                mode="IQU",
                hwp_angle=defaults.hwp_angle,
                detector_pointing=detpointing_azel,
            )

            # Source-centered pointing
            pixels = ops.PixelsWCS(
                coord_frame="AZEL",
                projection=proj,
                resolution=(resolution, resolution),
                dimensions=(),
                center_offset="source",
                detector_pointing=detpointing_azel,
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
                full_pointing=True,
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
                name=f"drone_{proj}",
                det_data=defaults.det_data,
                solve_rcond_threshold=1.0e-2,
                map_rcond_threshold=1.0e-2,
                binning=binner,
                template_matrix=tmatrix,
                output_dir=self.outdir,
                write_hits=True,
                write_map=True,
                write_binmap=True,
            )
            mapper.apply(data)

            if rank == 0:
                hitfile = os.path.join(self.outdir, f"drone_{proj}_hits.fits")
                mapfile = os.path.join(self.outdir, f"drone_{proj}_map.fits")
                binmapfile = os.path.join(self.outdir, f"drone_{proj}_binmap.fits")
                plot_wcs_maps(hitfile=hitfile, mapfile=mapfile, is_azimuth=True)
                plot_wcs_maps(hitfile=hitfile, mapfile=binmapfile, is_azimuth=True)

            close_data(data)
