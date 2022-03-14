# Copyright (c) 2021-2022 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
from astropy import units as u
import astropy.io.fits as af
from astropy.wcs import WCS

from .. import ops as ops
from .. import templates
from .. import qarray as qa
from ..intervals import Interval, IntervalList
from ..observation import default_values as defaults
from ..observation import Observation
from ..data import Data
from ..pixels_io_wcs import write_wcs_fits
from ..vis import set_matplotlib_backend
from ._helpers import (
    create_outdir,
    create_ground_data,
    create_comm,
    create_space_telescope,
    create_fake_sky,
)
from .mpi import MPITestCase


class PointingWCSTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def check_hits(self, prefix, pixels):
        wcs = pixels.wcs

        toastcomm = create_comm(self.comm)
        data = Data(toastcomm)
        tele = create_space_telescope(
            toastcomm.group_size,
            sample_rate=1.0 * u.Hz,
            pixel_per_process=1,
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
        coord *= np.pi / 180.0
        phi = np.array(coord[:, 0], dtype=np.float64)
        half_pi = np.pi / 2
        theta = np.array(half_pi - coord[:, 1], dtype=np.float64)
        bore = qa.from_position(theta, phi)

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
            pixel_dist="dist",
            pixels=pixels.pixels,
            det_flags=None,
        )
        build_hits.apply(data)

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
            im = ax.imshow(np.transpose(hdu.data[0, :, :]), vmin=0, vmax=4, cmap="jet")
            ax.grid(color="white", ls="solid")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.colorbar(im, orientation="vertical")
            fig.savefig(os.path.join(self.outdir, f"{prefix}.pdf"), format="pdf")

        np.testing.assert_array_equal(
            data[build_hits.hits].data,
            len(data.obs[0].all_detectors) * np.ones_like(data[build_hits.hits].data),
        )
        del data

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

    def plot_maps(self, hitfile=None, mapfile=None):
        figsize = (8, 8)
        figdpi = 100

        def plot_single(wcs, hdata, hindx, vmin, vmax, out):
            set_matplotlib_backend()

            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=figsize, dpi=figdpi)
            ax = fig.add_subplot(projection=wcs, slices=("x", "y", hindx))
            im = ax.imshow(np.transpose(hdu.data[hindx, :, :]), cmap="jet")
            ax.grid(color="white", ls="solid")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.colorbar(im, orientation="vertical")
            plt.savefig(out, format="pdf")
            plt.close()

        def map_range(hdata):
            minval = np.amin(hdata)
            maxval = np.amax(hdata)
            margin = 0.1 * (maxval - minval)
            if margin == 0:
                margin = -1
            minval += margin
            maxval -= margin
            return minval, maxval

        if hitfile is not None:
            hdu = af.open(hitfile)[0]
            wcs = WCS(hdu.header)
            maxhits = np.amax(hdu.data[0, :, :])
            plot_single(wcs, hdu, 0, 0, maxhits, f"{hitfile}.pdf")

        if mapfile is not None:
            hdu = af.open(mapfile)[0]
            wcs = WCS(hdu.header)

            mmin, mmax = map_range(hdu.data[0, :, :])
            plot_single(wcs, hdu, 0, mmin, mmax, f"{mapfile}_I.pdf")
            try:
                mmin, mmax = map_range(hdu.data[1, :, :])
                plot_single(wcs, hdu, 1, mmin, mmax, f"{mapfile}_Q.pdf")
            except:
                pass

            try:
                mmin, mmax = map_range(hdu.data[2, :, :])
                plot_single(wcs, hdu, 2, mmin, mmax, f"{mapfile}_U.pdf")
            except:
                pass

    def test_mapmaking(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

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
            use_astropy=True,
        )

        # Operator to reset pointing
        delete_pointing = ops.Delete(
            detdata=[detpointing_radec.quats, pixels.pixels, weights.weights]
        )

        # Test several projections

        for proj in ["CAR", "TAN", "CEA", "MER", "ZEA"]:
            delete_pointing.apply(data)
            if "pixel_dist" in data:
                del data["pixel_dist"]

            # Compute pixels and weights
            pixels.projection = proj
            pixels.resolution = (0.05 * u.degree, 0.05 * u.degree)
            pixels.auto_bounds = True
            pixels.create_dist = "pixel_dist"
            pixels.apply(data)

            weights.apply(data)

            # Create fake polarized sky pixel values locally
            create_fake_sky(data, "pixel_dist", "fake_map")

            # Write it out
            outfile = os.path.join(self.outdir, f"mapmaking_{proj}_input.fits")
            write_wcs_fits(data["fake_map"], outfile)

            if rank == 0:
                self.plot_maps(mapfile=outfile)

            # Scan map into timestreams
            scanner = ops.ScanMap(
                det_data=defaults.det_data,
                pixels=pixels.pixels,
                weights=weights.weights,
                map_key="fake_map",
            )
            scanner.apply(data)

            # Now clear the pointing and reset things for use with the mapmaking
            # test later
            delete_pointing = ops.Delete(
                detdata=[detpointing_radec.quats, pixels.pixels, weights.weights]
            )
            delete_pointing.apply(data)
            pixels.create_dist = None

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
                det_flags=None,
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
                write_hits=False,
                write_map=False,
                write_cov=False,
                write_rcond=False,
                keep_solver_products=True,
                keep_final_products=True,
            )
            mapper.apply(data)

            # Write outputs manually
            for prod in ["hits", "map"]:
                outfile = os.path.join(self.outdir, f"mapmaking_{proj}_{prod}.fits")
                write_wcs_fits(data[f"{mapper.name}_{prod}"], outfile)

            if rank == 0:
                outfile = os.path.join(self.outdir, f"mapmaking_{proj}_hits.fits")
                self.plot_maps(hitfile=outfile)

                outfile = os.path.join(self.outdir, f"mapmaking_{proj}_map.fits")
                self.plot_maps(mapfile=outfile)

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
        # Create fake source position

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
        pixels.create_dist = "source_pixel_dist"
        pixels.apply(data)
        pixels.create_dist = None

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
            spos = qa.from_position(stheta, sphi)
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

        weights = ops.StokesWeights(
            mode="I",
            hwp_angle=None,
            weights="temp_weights",
            detector_pointing=detpointing,
        )
        weights.apply(data)

        default_model = ops.DefaultNoiseModel(noise_model="source_noise_model")
        default_model.apply(data)

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
                det_data=defaults.det_data,
                det_flags=None,
                shared_flags=None,
                solve_rcond_threshold=1e-3,
                map_rcond_threshold=1e-3,
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
                self.plot_maps(hitfile=outfile)
                outfile = os.path.join(self.outdir, f"source_{proj}_notrack_map.fits")
                self.plot_maps(mapfile=outfile)

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
        resolution = 0.1 * u.degree

        for proj in ["CAR", "TAN"]:
            # Create fake observing of a small patch
            data = create_ground_data(self.comm, pixel_per_process=10)

            # Create source motion
            dbgdir = None
            if proj == "CAR":
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

            pixels.create_dist = "pixel_dist"
            pixels.apply(data)
            pixels.create_dist = None

            weights.apply(data)

            # Create fake polarized sky pixel values locally
            create_fake_sky(data, "pixel_dist", "fake_map")

            # Write it out
            outfile = os.path.join(self.outdir, f"source_{proj}_input.fits")
            write_wcs_fits(data["fake_map"], outfile)
            if rank == 0:
                self.plot_maps(mapfile=outfile)

            # Scan map into timestreams
            scanner = ops.ScanMap(
                det_data=defaults.det_data,
                pixels=pixels.pixels,
                weights=weights.weights,
                map_key="fake_map",
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
                det_flags=None,
                noise_model=default_model.noise_model,
                step_time=step_seconds * u.second,
            )
            tmatrix = ops.TemplateMatrix(templates=[tmpl])

            # Map maker
            mapper = ops.MapMaker(
                name=f"source_{proj}",
                det_data=defaults.det_data,
                det_flags=None,
                shared_flags=None,
                solve_rcond_threshold=1e-3,
                map_rcond_threshold=1e-3,
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
                self.plot_maps(hitfile=outfile)
                outfile = os.path.join(self.outdir, f"source_{proj}_map.fits")
                self.plot_maps(mapfile=outfile)
                outfile = os.path.join(self.outdir, f"source_{proj}_binmap.fits")
                self.plot_maps(mapfile=outfile)

            del data
