# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

import os
import shutil

import numpy as np
import numpy.testing as nt

from ..mpi import MPI
from ..tod import AnalyticNoise, OpSimNoise
from ..todmap import (
    TODGround,
    OpPointingHpix,
    OpSimGradient,
    OpSimScan,
    OpSimAtmosphere,
    atm_available_utils,
)

from ..weather import Weather

from ._helpers import (
    create_outdir,
    create_distdata,
    boresight_focalplane,
    create_weather,
)


class OpsSimAtmosphereTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.atm_cache = os.path.join(self.outdir, "atm_cache")

        # Create one observation per group, and each observation will have
        # one detector per process and a single chunk.

        self.data = create_distdata(self.comm, obs_per_group=1)

        # This serial data will exist separately on each process
        if MPI is None:
            self.data_serial = create_distdata(None, obs_per_group=1)
        else:
            self.data_serial = create_distdata(MPI.COMM_SELF, obs_per_group=1)

        self.ndet = self.data.comm.group_size
        self.rate = 10.0

        # Create detectors with white noise
        self.NET = 5.0

        (
            dnames,
            dquat,
            depsilon,
            drate,
            dnet,
            dfmin,
            dfknee,
            dalpha,
        ) = boresight_focalplane(
            self.ndet, samplerate=self.rate, fknee=0.0, net=self.NET
        )

        dnames_serial, dquat_serial, _, _, _, _, _, _ = boresight_focalplane(
            1, samplerate=self.rate, fknee=0.0, net=self.NET
        )

        # Samples per observation
        self.totsamp = 1000

        # Pixelization
        nside = 256
        self.sim_nside = nside
        self.map_nside = nside

        # Scan properties
        self.site_lon = "-67:47:10"
        self.site_lat = "-22:57:30"
        self.site_alt = 5200.0
        self.coord = "C"
        self.azmin = 45
        self.azmax = 55
        self.el = 60
        self.scanrate = 1.0
        self.scan_accel = 3.0
        self.CES_start = None

        # Populate the single observation per group

        tod = TODGround(
            self.data.comm.comm_group,
            dquat,
            self.totsamp,
            detranks=self.data.comm.group_size,
            firsttime=0.0,
            rate=self.rate,
            site_lon=self.site_lon,
            site_lat=self.site_lat,
            site_alt=self.site_alt,
            azmin=self.azmin,
            azmax=self.azmax,
            el=self.el,
            coord=self.coord,
            scanrate=self.scanrate,
            scan_accel=self.scan_accel,
            CES_start=self.CES_start,
        )

        tod_serial = TODGround(
            self.data_serial.comm.comm_group,
            dquat_serial,
            self.totsamp,
            detranks=self.data_serial.comm.group_size,
            firsttime=0.0,
            rate=self.rate,
            site_lon=self.site_lon,
            site_lat=self.site_lat,
            site_alt=self.site_alt,
            azmin=self.azmin,
            azmax=self.azmax,
            el=self.el,
            coord=self.coord,
            scanrate=self.scanrate,
            scan_accel=self.scan_accel,
            CES_start=self.CES_start,
        )

        self.common_flag_mask = tod.TURNAROUND

        common_flags = tod.read_common_flags()

        # Number of flagged samples in each observation.  Only the first row
        # of the process grid needs to contribute, since all process columns
        # have identical common flags.
        nflagged = 0
        if (tod.grid_comm_col is None) or (tod.grid_comm_col.rank == 0):
            nflagged += np.sum((common_flags & self.common_flag_mask) != 0)

        # Number of flagged samples across all observations
        self.nflagged = None
        if self.comm is None:
            self.nflagged = nflagged
        else:
            self.nflagged = self.data.comm.comm_world.allreduce(nflagged)

        wfile = os.path.join(self.outdir, "weather.fits")
        if self.comm is None or self.comm.rank == 0:
            create_weather(wfile)
        if self.comm is not None:
            self.comm.barrier()

        weather = Weather(wfile)
        weather.set(123, 4, 5)

        self.data.obs[0]["tod"] = tod
        self.data.obs[0]["id"] = self.data.comm.group
        self.data.obs[0]["telescope_id"] = 1
        self.data.obs[0]["site"] = "blah"
        self.data.obs[0]["site_id"] = 123
        self.data.obs[0]["weather"] = weather
        self.data.obs[0]["altitude"] = 2000
        self.data.obs[0]["fpradius"] = 1.0

        self.data_serial.obs[0]["tod"] = tod_serial
        self.data_serial.obs[0]["id"] = self.data.comm.group
        self.data_serial.obs[0]["telescope_id"] = 1
        self.data_serial.obs[0]["site"] = "blah"
        self.data_serial.obs[0]["site_id"] = 123
        self.data_serial.obs[0]["weather"] = weather
        self.data_serial.obs[0]["altitude"] = 2000
        self.data_serial.obs[0]["fpradius"] = 1.0

        self.common_params = {
            "realization": 0,
            "component": 123456,
            "lmin_center": 0.01,
            "lmin_sigma": 0.001,
            "lmax_center": 10,
            "lmax_sigma": 10,
            "zatm": 40000.0,
            "zmax": 2000.0,
            "xstep": 100.0,
            "ystep": 100.0,
            "zstep": 100.0,
            "nelem_sim_max": 10000,
            "gain": 1,
            "z0_center": 2000,
            "z0_sigma": 0,
            "apply_flags": True,
            "common_flag_name": None,
            "common_flag_mask": self.common_flag_mask,
            "flag_name": None,
            "flag_mask": 255,
            "report_timing": True,
            "wind_dist": 10000,
            "write_debug": False,
        }
        return

    def test_atm(self):
        if self.comm is None:
            # Cannot perform serial/MPI test
            print("No MPI available, skipping MPI/serial test")
            return

        # Generate an atmosphere sim with no loading or absorption.
        # Verify that serial and MPI results agree

        atm = OpSimAtmosphere(out="atm", cachedir=None, freq=None, **self.common_params)

        atm_utils = None
        if atm_available_utils:
            atm_utils = OpSimAtmosphere(
                out="atm-utils", cachedir=None, freq=150, **self.common_params
            )

        # Do the simulation with the default data distribution and communicator

        atm.exec(self.data)

        if atm_utils is not None:
            atm_utils.exec(self.data)

        # Now do an explicit serial calculation on each process for one detector.

        atm.exec(self.data_serial)
        if atm_utils is not None:
            atm_utils.exec(self.data_serial)

        # Check that the two cases agree on the process which has overlap between them
        tod = self.data.obs[0]["tod"]
        oid = self.data.obs[0]["id"]
        tod_serial = self.data_serial.obs[0]["tod"]
        oid_serial = self.data_serial.obs[0]["id"]
        if oid == oid_serial:
            for d in tod.local_dets:
                if d in tod_serial.local_dets:
                    cname = "atm_{}".format(d)
                    ref = tod.cache.reference(cname)
                    ref_serial = tod_serial.cache.reference(cname)
                    nt.assert_allclose(ref[:], ref_serial[:], rtol=1e-7)
                    if atm_utils is not None:
                        cname = "atm-utils_{}".format(d)
                        ref = tod.cache.reference(cname)
                        ref_serial = tod_serial.cache.reference(cname)
                        nt.assert_allclose(ref[:], ref_serial[:], rtol=1e-7)
        return

    def test_atm_caching(self):
        if self.comm is None or self.comm.rank == 0:
            try:
                shutil.rmtree(self.atm_cache)
            except OSError:
                pass
        if self.comm is not None:
            self.comm.barrier()

        # Generate an atmosphere sim with no loading or absorption.
        # Verify that serial and MPI results agree

        atm = OpSimAtmosphere(
            out="atm", cachedir=self.atm_cache, freq=None, **self.common_params
        )

        # Do the simulation, caching the atmosphere

        atm.exec(self.data)

        # Re-run, this time loading the cached atmosphere

        atm = OpSimAtmosphere(
            out="cached_atm", cachedir=self.atm_cache, freq=None, **self.common_params
        )

        atm.exec(self.data)

        # Check that the two cases agree on the process which has overlap between them
        tod = self.data.obs[0]["tod"]
        for d in tod.local_dets:
            if d in tod.local_dets:
                cname = "atm_{}".format(d)
                ref1 = tod.cache.reference(cname)
                cname = "cached_atm_{}".format(d)
                ref2 = tod.cache.reference(cname)
                nt.assert_allclose(ref1[:], ref2, rtol=1e-7)

        return
