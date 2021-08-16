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

from ..pixels_io import write_healpix_fits

from ..dipole import dipole

from ._helpers import create_outdir, create_ground_data


class SimAtmTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.nside = 256

    def test_sim(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        # Simple detector pointing
        detpointing_azel = ops.PointingDetectorSimple(
            boresight="boresight_azel", quats="quats_azel"
        )
        detpointing_radec = ops.PointingDetectorSimple(
            boresight="boresight_radec", quats="quats_radec"
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
            times = np.array(ob.shared["times"])

            fig = plt.figure(figsize=(12, 8), dpi=72)
            ax = fig.add_subplot(1, 1, 1, aspect="auto")
            ax.plot(times, ob.detdata["signal"][det])
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
            times = np.array(ob.shared["times"])

            fig = plt.figure(figsize=(12, 8), dpi=72)
            ax = fig.add_subplot(1, 1, 1, aspect="auto")
            ax.plot(times, ob.detdata["signal"][det])
            ax.set_title(f"Detector {det} Atmosphere + Noise TOD")
            outfile = os.path.join(self.outdir, f"{det}_atm-noise_tod.pdf")
            plt.savefig(outfile)
            plt.close()

        # Pointing matrix
        pointing = ops.PointingHealpix(
            nside=self.nside,
            nest=False,
            mode="IQU",
            detector_pointing=detpointing_radec,
        )

        # Make a binned map

        binner = ops.BinMap(
            pointing=pointing,
            noise_model=el_model.out_model,
        )

        mapmaker = ops.MapMaker(
            name="test_atm",
            binning=binner,
            output_dir=self.outdir,
            keep_final_products=True,
        )
        mapmaker.apply(data)

        if rank == 0:
            import matplotlib.pyplot as plt

            mapfile = os.path.join(self.outdir, f"{mapmaker.name}_hits.fits")
            mdata = hp.read_map(mapfile, nest=False)
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
            p = np.sqrt(mdata[1] ** 2 + mdata[2] ** 2)
            pfrac = np.median(p[good] / mdata[0][good])
            if pfrac > 0.01:
                raise RuntimeError("Simulated atmosphere is polarized")

    def test_sim_pol(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        # Simple detector pointing
        detpointing_azel = ops.PointingDetectorSimple(
            boresight="boresight_azel", quats="quats_azel"
        )
        detpointing_radec = ops.PointingDetectorSimple(
            boresight="boresight_radec", quats="quats_radec"
        )

        # Detector weights
        detweights_azel = ops.PointingHealpix(
            nside=64,
            mode="IQU",
            hwp_angle="hwp_angle",
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
            times = np.array(ob.shared["times"])

            fig = plt.figure(figsize=(12, 8), dpi=72)
            ax = fig.add_subplot(1, 1, 1, aspect="auto")
            ax.plot(times, ob.detdata["signal"][det])
            ax.set_title(f"Detector {det} Noise TOD")
            outfile = os.path.join(self.outdir, f"{det}_noise_tod.pdf")
            plt.savefig(outfile)
            plt.close()

        # Simulate atmosphere signal and accumulate
        sim_atm = ops.SimAtmosphere(
            detector_pointing=detpointing_azel,
            detector_weights=detweights_azel,
            polarization_fraction=0.2,
        )
        sim_atm.apply(data)

        if rank == 0:
            import matplotlib.pyplot as plt

            ob = data.obs[0]
            det = ob.local_detectors[0]
            times = np.array(ob.shared["times"])

            fig = plt.figure(figsize=(12, 8), dpi=72)
            ax = fig.add_subplot(1, 1, 1, aspect="auto")
            ax.plot(times, ob.detdata["signal"][det])
            ax.set_title(f"Detector {det} Atmosphere + Noise TOD")
            outfile = os.path.join(self.outdir, f"{det}_atm-noise_tod.pdf")
            plt.savefig(outfile)
            plt.close()

        # Pointing matrix
        pointing = ops.PointingHealpix(
            nside=self.nside,
            nest=False,
            mode="IQU",
            detector_pointing=detpointing_radec,
        )

        # Make a binned map

        binner = ops.BinMap(
            pointing=pointing,
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
            mdata = hp.read_map(mapfile, nest=False)
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
            p = np.sqrt(mdata[1] ** 2 + mdata[2] ** 2)
            i = np.abs(mdata[0])
            pfrac = np.median(p[good] / i[good])
            if pfrac < 0.01:
                raise RuntimeError("Simulated atmosphere is not polarized")

    def test_loading(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Create fake observing of a small patch
        data = create_ground_data(self.comm, el_nod=True)

        # Simple detector pointing
        detpointing_azel = ops.PointingDetectorSimple(
            boresight="boresight_azel", quats="quats_azel"
        )
        detpointing_radec = ops.PointingDetectorSimple(
            boresight="boresight_radec", quats="quats_radec"
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
            times = np.array(ob.shared["times"])

            fig = plt.figure(figsize=(12, 8), dpi=72)
            ax = fig.add_subplot(1, 1, 1, aspect="auto")
            ax.plot(times, ob.detdata["signal"][det])
            ax.set_title(f"Detector {det} Atmospheric loadingg TOD")
            outfile = os.path.join(self.outdir, f"{det}_atm_loading_tod.pdf")
            plt.savefig(outfile)
            plt.close()

        # We simulated atmosphere with zero gain, so only atmospheric
        # loading is included.  Confirm that all detectors are seeing
        # a non-zero signal

        for obs in data.obs:
            for det in obs.local_detectors:
                sig = obs.detdata["signal"][det]
                assert np.std(sig) != 0

        return
