# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
from astropy import units as u

from .. import ops as ops
from .. import qarray as qa
from ..atm import available_atm, available_utils
from ..dipole import dipole
from ..observation import default_values as defaults
from ..pixels_io_healpix import write_healpix_fits
from ..vis import set_matplotlib_backend
from ._helpers import close_data, create_ground_data, create_outdir
from .mpi import MPITestCase


class SimAtmTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.nside = 256

    def test_sim(self):
        if not available_atm:
            print("TOAST was compiled without atmosphere support, skipping tests")
            return

        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        # Simple detector pointing
        detpointing_azel = ops.PointingDetectorSimple(
            boresight=defaults.boresight_azel, quats="quats_azel"
        )
        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec, quats="quats_radec"
        )

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        # Make an elevation-dependent noise model
        el_model = ops.ElevationNoise(
            noise_model="noise_model",
            out_model="el_weighted",
            detector_pointing=detpointing_azel,
        )
        el_model.apply(data)

        # Simulate noise and accumulate to signal
        sim_noise = ops.SimNoise(noise_model=el_model.out_model)
        sim_noise.apply(data)

        if rank == 0:
            set_matplotlib_backend()
            import matplotlib.pyplot as plt

            ob = data.obs[0]
            det = ob.local_detectors[0]
            times = np.array(ob.shared[defaults.times])

            fig = plt.figure(figsize=(12, 8), dpi=72)
            ax = fig.add_subplot(1, 1, 1, aspect="auto")
            ax.plot(times, ob.detdata[defaults.det_data][det])
            ax.set_title(f"Detector {det} Noise TOD")
            outfile = os.path.join(self.outdir, f"{det}_noise_tod.pdf")
            plt.savefig(outfile)
            plt.close()

        # Simulate atmosphere signal and accumulate
        sim_atm = ops.SimAtmosphere(
            detector_pointing=detpointing_azel,
        )
        sim_atm.apply(data)

        if rank == 0:
            import matplotlib.pyplot as plt

            ob = data.obs[0]
            det = ob.local_detectors[0]
            times = np.array(ob.shared[defaults.times])

            fig = plt.figure(figsize=(12, 8), dpi=72)
            ax = fig.add_subplot(1, 1, 1, aspect="auto")
            ax.plot(times, ob.detdata[defaults.det_data][det])
            ax.set_title(f"Detector {det} Atmosphere + Noise TOD")
            outfile = os.path.join(self.outdir, f"{det}_atm-noise_tod.pdf")
            plt.savefig(outfile)
            plt.close()

        # Pointing matrix

        pixels = ops.PixelsHealpix(
            nside=self.nside,
            nest=False,
            detector_pointing=detpointing_radec,
        )
        weights = ops.StokesWeights(
            mode="IQU",
            detector_pointing=detpointing_radec,
        )

        # Make a binned map

        binner = ops.BinMap(
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=el_model.out_model,
        )

        mapmaker = ops.MapMaker(
            name="test_atm",
            binning=binner,
            output_dir=self.outdir,
            keep_final_products=True,
            solve_rcond_threshold=1.0e-2,
            map_rcond_threshold=1.0e-2,
            write_noiseweighted_map=True,
        )
        mapmaker.apply(data)

        if rank == 0:
            import matplotlib.pyplot as plt

            mapfile = os.path.join(self.outdir, f"{mapmaker.name}_hits.fits")
            mdata = hp.read_map(mapfile, nest=False, dtype=float)
            mdata[mdata == 0] = hp.UNSEEN

            outfile = "{}.png".format(mapfile)
            hp.gnomview(mdata, xsize=1600, rot=(42.0, -42.0), reso=0.5, nest=False)
            plt.savefig(outfile)
            plt.close()

            mapfile = os.path.join(self.outdir, f"{mapmaker.name}_map.fits")
            mdata = hp.read_map(mapfile, None, nest=False)
            mdata[mdata == 0] = hp.UNSEEN

            outfile = "{}.png".format(mapfile)
            fig = plt.figure(figsize=[8 * 3, 12])
            hp.gnomview(
                mdata[0],
                xsize=1600,
                sub=[1, 3, 1],
                rot=(42.0, -42.0),
                reso=0.5,
                nest=False,
            )
            hp.gnomview(
                mdata[1],
                xsize=1600,
                sub=[1, 3, 2],
                rot=(42.0, -42.0),
                reso=0.5,
                nest=False,
            )
            hp.gnomview(
                mdata[2],
                xsize=1600,
                sub=[1, 3, 3],
                rot=(42.0, -42.0),
                reso=0.5,
                nest=False,
            )
            plt.savefig(outfile)
            plt.close()

            mapfile = os.path.join(
                self.outdir, f"{mapmaker.name}_noiseweighted_map.fits"
            )
            mdata = hp.read_map(mapfile, None, nest=False)
            mdata[mdata == 0] = hp.UNSEEN

            outfile = "{}.png".format(mapfile)
            fig = plt.figure(figsize=[8 * 3, 12])
            hp.gnomview(
                mdata[0],
                xsize=1600,
                sub=[1, 3, 1],
                rot=(42.0, -42.0),
                reso=0.5,
                nest=False,
            )
            hp.gnomview(
                mdata[1],
                xsize=1600,
                sub=[1, 3, 2],
                rot=(42.0, -42.0),
                reso=0.5,
                nest=False,
            )
            hp.gnomview(
                mdata[2],
                xsize=1600,
                sub=[1, 3, 3],
                rot=(42.0, -42.0),
                reso=0.5,
                nest=False,
            )
            plt.savefig(outfile)
            plt.close()

            good = mdata[0] != hp.UNSEEN
            p = np.sqrt(mdata[1] ** 2 + mdata[2] ** 2)
            pfrac = np.median(p[good] / mdata[0][good])
            if pfrac > 0.01:
                raise RuntimeError("Simulated atmosphere is polarized")
        close_data(data)

    def test_sim_interp(self):
        if not available_atm:
            print("TOAST was compiled without atmosphere support, skipping tests")
            return

        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        # Simple detector pointing
        detpointing_azel = ops.PointingDetectorSimple(
            boresight=defaults.boresight_azel, quats="quats_azel"
        )
        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec, quats="quats_radec"
        )

        # Simulate atmosphere signal at full rate
        sim_atm = ops.SimAtmosphere(
            detector_pointing=detpointing_azel,
            det_data="full_signal",
        )
        sim_atm.apply(data)

        # Simulate atmosphere signal at quarter rate
        sim_atm = ops.SimAtmosphere(
            detector_pointing=detpointing_azel,
            sample_rate=data.obs[0].telescope.focalplane.sample_rate / 4,
            det_data="interpolated_signal",
        )
        sim_atm.apply(data)

        # Compare

        if rank == 0:
            for obs in data.obs:
                for det in obs.local_detectors:
                    sig0 = obs.detdata["full_signal"][det]
                    sig1 = obs.detdata["interpolated_signal"][det]
                    assert np.std(sig1 - sig0) / np.std(sig0) < 1e-2

        close_data(data)

    def test_sim_pol(self):
        if not available_atm:
            print("TOAST was compiled without atmosphere support, skipping tests")
            return

        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        # Simple detector pointing
        detpointing_azel = ops.PointingDetectorSimple(
            boresight=defaults.boresight_azel, quats="quats_azel"
        )
        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec, quats="quats_radec"
        )

        # Detector weights
        azel_weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing_azel,
        )

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        # Make an elevation-dependent noise model
        el_model = ops.ElevationNoise(
            noise_model="noise_model",
            out_model="el_weighted",
            detector_pointing=detpointing_azel,
        )
        el_model.apply(data)

        # Simulate noise and accumulate to signal
        sim_noise = ops.SimNoise(noise_model=el_model.out_model)
        sim_noise.apply(data)

        if rank == 0:
            set_matplotlib_backend()
            import matplotlib.pyplot as plt

            ob = data.obs[0]
            det = ob.local_detectors[0]
            times = np.array(ob.shared[defaults.times])

            fig = plt.figure(figsize=(12, 8), dpi=72)
            ax = fig.add_subplot(1, 1, 1, aspect="auto")
            ax.plot(times, ob.detdata[defaults.det_data][det])
            ax.set_title(f"Detector {det} Noise TOD")
            outfile = os.path.join(self.outdir, f"{det}_noise_tod.pdf")
            plt.savefig(outfile)
            plt.close()

        # Simulate atmosphere signal and accumulate
        sim_atm = ops.SimAtmosphere(
            detector_pointing=detpointing_azel,
            detector_weights=azel_weights,
            polarization_fraction=0.2,
            add_loading=False,  # Loading is not polarized
        )
        sim_atm.apply(data)

        if rank == 0:
            import matplotlib.pyplot as plt

            ob = data.obs[0]
            det = ob.local_detectors[0]
            times = np.array(ob.shared[defaults.times])

            fig = plt.figure(figsize=(12, 8), dpi=72)
            ax = fig.add_subplot(1, 1, 1, aspect="auto")
            ax.plot(times, ob.detdata[defaults.det_data][det])
            ax.set_title(f"Detector {det} Atmosphere + Noise TOD")
            outfile = os.path.join(self.outdir, f"{det}_atm-noise_tod.pdf")
            plt.savefig(outfile)
            plt.close()

        # Pointing matrix

        pixels = ops.PixelsHealpix(
            nside=self.nside,
            nest=False,
            detector_pointing=detpointing_radec,
        )
        weights = ops.StokesWeights(
            mode="IQU",
            detector_pointing=detpointing_radec,
        )

        # Make a binned map

        binner = ops.BinMap(
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=el_model.out_model,
        )

        mapmaker = ops.MapMaker(
            name="test_atm_pol",
            binning=binner,
            output_dir=self.outdir,
        )
        mapmaker.apply(data)

        if rank == 0:
            import matplotlib.pyplot as plt

            mapfile = os.path.join(self.outdir, f"{mapmaker.name}_hits.fits")
            mdata = hp.read_map(mapfile, nest=False, dtype=float)
            mdata[mdata == 0] = hp.UNSEEN

            outfile = "{}.png".format(mapfile)
            hp.gnomview(mdata, xsize=1600, rot=(42.0, -42.0), reso=0.5, nest=False)
            plt.savefig(outfile)
            plt.close()

            mapfile = os.path.join(self.outdir, f"{mapmaker.name}_map.fits")
            mdata = hp.read_map(mapfile, None, nest=False)
            mdata[mdata == 0] = hp.UNSEEN

            outfile = "{}.png".format(mapfile)
            fig = plt.figure(figsize=[8 * 3, 12])
            hp.gnomview(
                mdata[0],
                xsize=1600,
                sub=[1, 3, 1],
                rot=(42.0, -42.0),
                reso=0.5,
                nest=False,
            )
            hp.gnomview(
                mdata[1],
                xsize=1600,
                sub=[1, 3, 2],
                rot=(42.0, -42.0),
                reso=0.5,
                nest=False,
            )
            hp.gnomview(
                mdata[2],
                xsize=1600,
                sub=[1, 3, 3],
                rot=(42.0, -42.0),
                reso=0.5,
                nest=False,
            )
            plt.savefig(outfile)
            plt.close()

            good = mdata[0] != hp.UNSEEN
            p = np.sqrt(mdata[1][good] ** 2 + mdata[2][good] ** 2)
            i = np.abs(mdata[0][good])
            pfrac = np.median(p / i)
            if pfrac < 0.01:
                print(f"polarized atmosphere: using {np.count_nonzero(good)} pixels")
                print(
                    f"polarized atmosphere: p_avg = {np.mean(p)}, i_avg = {np.mean(i)}"
                )
                print(f"polarized atmosphere: pfrac = {pfrac}")
                raise RuntimeError("Simulated atmosphere is not polarized")
        close_data(data)

    def test_loading(self):
        if not available_atm:
            print("TOAST was compiled without atmosphere support, skipping tests")
            return

        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Create fake observing of a small patch
        data = create_ground_data(self.comm, el_nod=True)

        # Simple detector pointing
        detpointing_azel = ops.PointingDetectorSimple(
            boresight=defaults.boresight_azel, quats="quats_azel"
        )
        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec, quats="quats_radec"
        )

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        # Simulate atmosphere signal
        sim_atm = ops.SimAtmosphere(
            detector_pointing=detpointing_azel,
            gain=0,
        )
        sim_atm.apply(data)

        if rank == 0:
            import matplotlib.pyplot as plt

            ob = data.obs[0]
            det = ob.local_detectors[0]
            times = np.array(ob.shared[defaults.times])

            fig = plt.figure(figsize=(12, 8), dpi=72)
            ax = fig.add_subplot(1, 1, 1, aspect="auto")
            ax.plot(times, ob.detdata[defaults.det_data][det])
            ax.set_title(f"Detector {det} Atmospheric loading TOD")
            outfile = os.path.join(self.outdir, f"{det}_atm_loading_tod.pdf")
            plt.savefig(outfile)
            plt.close()

        # We simulated atmosphere with zero gain, so only atmospheric
        # loading is included.  Confirm that all detectors are seeing
        # a non-zero signal

        for obs in data.obs:
            for det in obs.local_detectors:
                sig = obs.detdata[defaults.det_data][det]
                assert np.std(sig) != 0

        close_data(data)

    def test_bandpass(self):
        if not available_atm:
            print("TOAST was compiled without atmosphere support, skipping tests")
            return

        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        # Override the simulated bandpass
        for obs in data.obs:
            fp = obs.telescope.focalplane
            fp.detector_data["bandcenter"] = 100 * u.GHz
            fp.detector_data["bandwidth"] = 10 * u.GHz
            fp._get_bandpass()

        # Simple detector pointing
        detpointing_azel = ops.PointingDetectorSimple(
            boresight=defaults.boresight_azel, quats="quats_azel"
        )
        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec, quats="quats_radec"
        )

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        cache_dir = os.path.join(self.outdir, "atm_cache")

        # Simulate atmosphere signal
        sim_atm = ops.SimAtmosphere(
            detector_pointing=detpointing_azel,
            cache_dir=cache_dir,
        )
        sim_atm.apply(data)

        old_rms = {}
        for obs in data.obs:
            old_rms[obs.name] = {}
            for det in obs.local_detectors:
                sig = obs.detdata[defaults.det_data][det]
                old_rms[obs.name][det] = np.std(sig)
                sig[:] = 0

        # Override the simulated bandpass again
        for obs in data.obs:
            fp = obs.telescope.focalplane
            fp.detector_data["bandcenter"] = 150 * u.GHz
            fp.detector_data["bandwidth"] = 15 * u.GHz
            fp._get_bandpass()

        # Simulate atmosphere signal again
        sim_atm.apply(data)

        # Check that the atmospheric fluctuations are stronger at higher frequency
        for obs in data.obs:
            for det in obs.local_detectors:
                new_rms = np.std(obs.detdata[defaults.det_data][det])
                assert new_rms > 1.1 * old_rms[obs.name][det]

        close_data(data)

    def count_mem(self, data, pref):
        n_seg = 0
        n_bytes = 0
        shnames = set()
        mem_count = ops.MemoryCounter()
        mem_count.prefix = pref
        for ob in data.obs:
            for name, info in ob.shared._internal.items():
                shnames.add(name)
                n_seg += 1
                n_bytes += info[0]._n * info[0]._dsize
        # Uncomment for debugging...
        # print(f"{pref} {n_seg} shared memory segments with {n_bytes} bytes: {shnames}")
        # mem_count.apply(data)

    def test_sim_split(self):
        if not available_atm:
            print("TOAST was compiled without atmosphere support, skipping tests")
            return

        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        testdir = os.path.join(self.outdir, "sim_split")
        if rank == 0:
            os.makedirs(testdir)
        if self.comm is not None:
            self.comm.barrier()

        # Set up operators

        # Simple detector pointing
        detpointing_azel = ops.PointingDetectorSimple(
            boresight=defaults.boresight_azel, quats="quats_azel"
        )
        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec, quats="quats_radec"
        )

        # Pointing matrix
        pixels = ops.PixelsHealpix(
            nside=self.nside,
            nest=False,
            detector_pointing=detpointing_radec,
        )
        weights_radec = ops.StokesWeights(
            mode="IQU",
            detector_pointing=detpointing_radec,
        )
        weights_azel = ops.StokesWeights(
            mode="IQU",
            detector_pointing=detpointing_azel,
        )

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()

        # Make an elevation-dependent noise model
        el_model = ops.ElevationNoise(
            noise_model="noise_model",
            out_model="el_weighted",
            detector_pointing=detpointing_azel,
        )

        # Simulate noise and accumulate to signal
        sim_noise = ops.SimNoise(noise_model=el_model.out_model)

        # Simulate atmosphere signal and accumulate
        sim_atm = ops.SimAtmosphere(
            detector_pointing=detpointing_azel,
            detector_weights=weights_azel,
        )

        # Build hit map and covariance
        cov_and_hits = ops.CovarianceAndHits(
            pixel_pointing=pixels,
            stokes_weights=weights_radec,
            noise_model=el_model.out_model,
            inverse_covariance="invcov",
        )

        # HDF5 saving
        saver = ops.SaveHDF5(detdata_float32=True, verify=True)

        # Simulate and process all detectors simultaneously

        ppp = 10

        freq_list = [(100 + 10 * x) * u.GHz for x in range(3)]

        full_data = create_ground_data(
            self.comm,
            freqs=freq_list,
            pixel_per_process=ppp,
        )
        self.count_mem(full_data, "After full sim ground")

        default_model.apply(full_data)
        self.count_mem(full_data, "After full default model")

        el_model.apply(full_data)
        self.count_mem(full_data, "After full el model")

        sim_noise.apply(full_data)
        self.count_mem(full_data, "After full sim noise")

        sim_atm.apply(full_data)
        self.count_mem(full_data, "After full sim atm")

        saver.volume = os.path.join(testdir, "data_full")
        saver.apply(full_data)

        cov_and_hits.apply(full_data)
        self.count_mem(full_data, "After full cov and hits")

        # Simulate data split by frequency and process

        split_data = create_ground_data(
            self.comm, freqs=freq_list, pixel_per_process=ppp, split=True
        )
        self.count_mem(split_data, "After split sim ground")

        default_model.apply(split_data)
        self.count_mem(split_data, "After split default model")

        el_model.apply(split_data)
        self.count_mem(split_data, "After split el model")

        sim_noise.apply(split_data)
        self.count_mem(split_data, "After split sim noise")

        sim_atm.apply(split_data)
        self.count_mem(split_data, "After split sim atm")

        saver.volume = os.path.join(testdir, "data_split")
        saver.apply(split_data)

        cov_and_hits.apply(split_data)
        self.count_mem(split_data, "After split cov and hits")

        # Compare two cases

        # The following code is useful for debugging, but in general
        # there is no guarantee that a process has the same detectors
        # in the monolithic and split cases.
        #
        # for obs in full_data.obs:
        #     # Get the session
        #     ses = obs.session
        #     # Find split observations in this same session
        #     for sobs in split_data.obs:
        #         if sobs.session.name == ses.name:
        #             # Compare detector timestreams.
        #             for det in sobs.local_detectors:
        #                 stod = sobs.detdata[defaults.det_data][det]
        #                 ftod = obs.detdata[defaults.det_data][det]
        #                 if not np.allclose(stod, ftod):
        #                     msg = f"{ses.name} {det}\n"
        #                     msg += f"  full = {ftod}\n"
        #                     msg += f"  split = {stod}\n"
        #                     print(msg)
        #                     self.assertTrue(False)

        if full_data[cov_and_hits.hits] != split_data[cov_and_hits.hits]:
            hfull = full_data[cov_and_hits.hits]
            hsplit = split_data[cov_and_hits.hits]
            hfull_good = hfull.data != 0
            hsplit_good = hsplit.data != 0
            hfull_nz = np.count_nonzero(hfull_good)
            hsplit_nz = np.count_nonzero(hsplit_good)
            hfull_max = np.max(hfull.data[hfull_good])
            hsplit_max = np.max(hsplit.data[hsplit_good])
            msg = f"Proc {full_data.comm.world_rank} full hits: "
            msg += f"{hfull_nz} hit pixels, max = {hfull_max} "
            msg += f"split hits: {hsplit_nz} hit pixels, max = {hsplit_max}"
            print(msg)
            self.assertTrue(False)

        if (
            full_data[cov_and_hits.inverse_covariance]
            != split_data[cov_and_hits.inverse_covariance]
        ):
            ifull = full_data[cov_and_hits.inverse_covariance]
            isplit = split_data[cov_and_hits.inverse_covariance]
            hfull = full_data[cov_and_hits.hits]
            hsplit = split_data[cov_and_hits.hits]
            full_good = hfull.data != 0
            split_good = hsplit.data != 0
            diff = ifull.data[full_good] - isplit.data[split_good]
            msg = f"Proc {full_data.comm.world_rank} inv cov diff: "
            msg += f"{diff}"
            print(msg)
            self.assertTrue(False)

        # Cleanup
        close_data(full_data)
        close_data(split_data)
