# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops as ops
from ..accelerator import accel_enabled
from ..mpi import MPI
from ..noise import Noise
from ..observation import default_values as defaults
from ..pixels import PixelData, PixelDistribution
from ._helpers import close_data, create_outdir, create_satellite_data
from .mpi import MPITestCase


class MapmakerUtilsTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)
        self.mix_coeff = np.random.uniform(low=-1.0, high=1.0, size=1000)

    def create_corr_noise(self, dets, nse):
        corr_freqs = {"noise_{}".format(i): nse.freq(x) for i, x in enumerate(dets)}
        corr_psds = {"noise_{}".format(i): nse.psd(x) for i, x in enumerate(dets)}
        corr_indices = {"noise_{}".format(i): 100 + i for i, x in enumerate(dets)}
        corr_mix = dict()
        for i, x in enumerate(dets):
            dmix = self.mix_coeff[: len(dets)]
            corr_mix[x] = {"noise_{}".format(y): dmix[y] for y in range(len(dets))}
        return Noise(
            detectors=dets,
            freqs=corr_freqs,
            psds=corr_psds,
            mixmatrix=corr_mix,
            indices=corr_indices,
        )

    def test_hits(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pixels.apply(data)

        hits = dict()
        for stype in ["allreduce", "alltoallv"]:
            build_hits = ops.BuildHitMap(
                pixel_dist="pixel_dist", hits="hits_{}".format(stype), sync_type=stype
            )
            build_hits.apply(data)
            hits[stype] = data[build_hits.hits]

        # Manual check
        check_hits = PixelData(data["pixel_dist"], np.int64, n_value=1)
        for ob in data.obs:
            for det in ob.local_detectors:
                local_sm, local_pix = data["pixel_dist"].global_pixel_to_submap(
                    ob.detdata["pixels"][det]
                )
                for i in range(ob.n_local_samples):
                    if local_pix[i] >= 0:
                        check_hits.data[local_sm[i], local_pix[i], 0] += 1
        check_hits.sync_allreduce()

        for stype in ["allreduce", "alltoallv"]:
            hmap = hits[stype]
            comm = hmap.distribution.comm
            failed = not np.all(np.equal(hmap.data, check_hits.data))
            if comm is not None:
                failed = comm.allreduce(failed, op=MPI.LOR)
            self.assertFalse(failed)

        close_data(data)

    def test_inv_cov(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )
        weights.apply(data)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Construct a correlated analytic noise model for the detectors for each
        # observation.
        for ob in data.obs:
            nse = ob[default_model.noise_model]
            ob["noise_model_corr"] = self.create_corr_noise(ob.local_detectors, nse)

        # Simulate noise using both models

        sim_noise = ops.SimNoise(noise_model="noise_model", det_data=defaults.det_data)
        sim_noise.apply(data)

        sim_noise_corr = ops.SimNoise(
            noise_model="noise_model_corr", det_data="noise_corr"
        )
        sim_noise_corr.apply(data)

        # Build an inverse covariance from both

        invnpp = dict()
        invnpp_corr = dict()
        for stype in ["allreduce", "alltoallv"]:
            build_invnpp = ops.BuildInverseCovariance(
                pixel_dist="pixel_dist",
                noise_model="noise_model",
                inverse_covariance="invnpp_{}".format(stype),
                sync_type=stype,
            )
            build_invnpp.apply(data)
            invnpp[stype] = data[build_invnpp.inverse_covariance]

            build_invnpp_corr = ops.BuildInverseCovariance(
                pixel_dist="pixel_dist",
                noise_model="noise_model_corr",
                inverse_covariance="invnpp_{}_corr".format(stype),
                sync_type=stype,
            )
            build_invnpp_corr.apply(data)
            invnpp_corr[stype] = data[build_invnpp_corr.inverse_covariance]

        # Manual check

        invnpp_units = 1.0 / (u.K**2)
        check_invnpp = PixelData(
            data["pixel_dist"], np.float64, n_value=6, units=invnpp_units
        )
        check_invnpp_corr = PixelData(
            data["pixel_dist"], np.float64, n_value=6, units=invnpp_units
        )

        for ob in data.obs:
            noise = ob["noise_model"]
            noise_corr = ob["noise_model_corr"]

            for det in ob.local_detectors:
                detweight = noise.detector_weight(det).to_value(invnpp_units)
                detweight_corr = noise_corr.detector_weight(det).to_value(invnpp_units)

                wt = ob.detdata[defaults.weights][det]
                local_sm, local_pix = data["pixel_dist"].global_pixel_to_submap(
                    ob.detdata[defaults.pixels][det]
                )
                for i in range(ob.n_local_samples):
                    if local_pix[i] < 0:
                        continue
                    off = 0
                    for j in range(3):
                        for k in range(j, 3):
                            check_invnpp.data[local_sm[i], local_pix[i], off] += (
                                detweight * wt[i, j] * wt[i, k]
                            )
                            check_invnpp_corr.data[local_sm[i], local_pix[i], off] += (
                                detweight_corr * wt[i, j] * wt[i, k]
                            )
                            off += 1

        check_invnpp.sync_allreduce()
        check_invnpp_corr.sync_allreduce()

        for stype in ["allreduce", "alltoallv"]:
            comm = invnpp[stype].distribution.comm

            failed = False

            for sm in range(invnpp[stype].distribution.n_local_submap):
                for px in range(invnpp[stype].distribution.n_pix_submap):
                    if invnpp[stype].data[sm, px, 0] != 0:
                        if not np.allclose(
                            invnpp[stype].data[sm, px],
                            check_invnpp.data[sm, px],
                        ):
                            failed = True
                    if invnpp_corr[stype].data[sm, px, 0] != 0:
                        if not np.allclose(
                            invnpp_corr[stype].data[sm, px],
                            check_invnpp_corr.data[sm, px],
                        ):
                            failed = True
            if comm is not None:
                failed = comm.allreduce(failed, op=MPI.LOR)
            self.assertFalse(failed)
        close_data(data)

    def test_zmap(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create some detector pointing matrices
        detpointing = ops.PointingDetectorSimple()
        pixels = ops.PixelsHealpix(
            nside=64,
            create_dist="pixel_dist",
            detector_pointing=detpointing,
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )
        weights.apply(data)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Construct a correlated analytic noise model for the detectors for each
        # observation.
        for ob in data.obs:
            nse = ob[default_model.noise_model]
            ob["noise_model_corr"] = self.create_corr_noise(ob.local_detectors, nse)

        # Simulate noise using both models

        sim_noise = ops.SimNoise(noise_model="noise_model", det_data="noise")
        sim_noise.apply(data)

        sim_noise_corr = ops.SimNoise(
            noise_model="noise_model_corr", det_data="noise_corr"
        )
        sim_noise_corr.apply(data)

        # Build a noise weighted map from both

        zmap = dict()
        zmap_corr = dict()
        for stype in ["allreduce", "alltoallv"]:
            build_zmap = ops.BuildNoiseWeighted(
                pixel_dist="pixel_dist",
                noise_model="noise_model",
                det_data="noise",
                zmap="zmap_{}".format(stype),
                sync_type=stype,
            )
            build_zmap.apply(data)
            zmap[stype] = data[build_zmap.zmap]

            build_zmap_corr = ops.BuildNoiseWeighted(
                pixel_dist="pixel_dist",
                noise_model="noise_model_corr",
                det_data="noise_corr",
                zmap="zmap_corr_{}".format(stype),
                sync_type=stype,
            )
            build_zmap_corr.apply(data)
            zmap_corr[stype] = data[build_zmap_corr.zmap]

        # Test one case on the accelerator if supported

        build_zmap = ops.BuildNoiseWeighted(
            pixel_dist="pixel_dist",
            noise_model="noise_model",
            det_data="noise",
            zmap="zmap_gpu",
            sync_type="allreduce",
        )

        use_accel = None
        if accel_enabled() and (
            pixels.supports_accel()
            and weights.supports_accel()
            and build_zmap.supports_accel()
        ):
            use_accel = True
            # puts all inputs on device
            data.accel_create(pixels.requires())
            data.accel_create(weights.requires())
            data.accel_create(build_zmap.requires())
            data.accel_update_device(pixels.requires())
            data.accel_update_device(weights.requires())
            data.accel_update_device(build_zmap.requires())
            # runs on device
            build_zmap.apply(data, use_accel=use_accel)
            # insures everything is back from device
            data.accel_update_host(build_zmap.provides())
            data.accel_delete(pixels.requires())
            data.accel_delete(weights.requires())
            data.accel_delete(build_zmap.requires())
            # stores output
            zmap["gpu"] = data[build_zmap.zmap]

        # Manual check

        invnpp_units = 1.0 / (u.K**2)
        zmap_units = 1.0 / u.K

        check_zmap = PixelData(
            data["pixel_dist"], np.float64, n_value=3, units=zmap_units
        )
        check_zmap_corr = PixelData(
            data["pixel_dist"], np.float64, n_value=3, units=zmap_units
        )

        for ob in data.obs:
            noise = ob["noise_model"]
            noise_corr = ob["noise_model_corr"]
            for det in ob.local_detectors:
                wt = ob.detdata[defaults.weights][det]
                local_sm, local_pix = data["pixel_dist"].global_pixel_to_submap(
                    ob.detdata[defaults.pixels][det]
                )

                for i in range(ob.n_local_samples):
                    if local_pix[i] < 0:
                        continue
                    for j in range(3):
                        check_zmap.data[local_sm[i], local_pix[i], j] += (
                            noise.detector_weight(det).to_value(invnpp_units)
                            * ob.detdata["noise"][det, i]
                            * wt[i, j]
                        )
                        check_zmap_corr.data[local_sm[i], local_pix[i], j] += (
                            noise_corr.detector_weight(det).to_value(invnpp_units)
                            * ob.detdata["noise_corr"][det, i]
                            * wt[i, j]
                        )

        check_zmap.sync_allreduce()
        check_zmap_corr.sync_allreduce()

        for stype in ["allreduce", "alltoallv"]:
            comm = zmap[stype].distribution.comm

            failed = False
            for sm in range(zmap[stype].distribution.n_local_submap):
                for px in range(zmap[stype].distribution.n_pix_submap):
                    if zmap[stype].data[sm, px, 0] != 0:
                        if not np.allclose(
                            zmap[stype].data[sm, px],
                            check_zmap.data[sm, px],
                            atol=1.0e-6,
                        ):
                            print(
                                f"zmap({stype})[{sm}, {px}] = {zmap[stype].data[sm, px]}, check = {check_zmap.data[sm, px]}"
                            )
                            failed = True
                    if zmap_corr[stype].data[sm, px, 0] != 0:
                        if not np.allclose(
                            zmap_corr[stype].data[sm, px],
                            check_zmap_corr.data[sm, px],
                            atol=1.0e-6,
                        ):
                            print(
                                f"zmap_corr({stype})[{sm}, {px}] = {zmap_corr[stype].data[sm, px]}, check = {check_zmap_corr.data[sm, px]}"
                            )
                            failed = True

            if comm is not None:
                failed = comm.allreduce(failed, op=MPI.LOR)
            self.assertFalse(failed)

        if "gpu" in zmap:
            comm = zmap["gpu"].distribution.comm

            failed = False
            for sm in range(zmap["gpu"].distribution.n_local_submap):
                for px in range(zmap["gpu"].distribution.n_pix_submap):
                    if zmap["gpu"].data[sm, px, 0] != 0:
                        if not np.allclose(
                            zmap["gpu"].data[sm, px],
                            check_zmap.data[sm, px],
                            atol=1.0e-6,
                        ):
                            print(
                                f"zmap(gpu)[{sm}, {px}] = {zmap['gpu'].data[sm, px]}, check = {check_zmap.data[sm, px]}"
                            )
                            failed = True
            if comm is not None:
                failed = comm.allreduce(failed, op=MPI.LOR)
            self.assertFalse(failed)

        close_data(data)
