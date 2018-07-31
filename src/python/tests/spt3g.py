# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

import sys
import os
import re
import shutil

import unittest

import numpy as np
import numpy.testing as nt

from ..dist import *

from .. import qarray as qa

from ..tod import Interval, TODCache, TODGround

from ._helpers import (create_outdir, create_distdata, boresight_focalplane,
    uniform_chunks)

# This file will only be imported if SPT3G is already available
from spt3g import core as c3g
from ..tod import spt3g_utils as s3utils
from ..tod import spt3g as s3g


class Spt3gTest(MPITestCase):

    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        self.datawrite = os.path.join(self.outdir, "test_3g")
        self.dataexport = os.path.join(self.outdir, "export_3g")
        self.groundexport = os.path.join(self.outdir, "export_3g_ground")

        # Reset the frame file size for these tests, so that we can test
        # boundary effects.
        self._original_framefile = s3g.TARGET_FRAMEFILE_SIZE

        # Create observations, divided evenly between groups
        self.nobs = 4
        opg = self.nobs
        if self.comm.size >= 2:
            opg = self.nobs // 2
        self.data = create_distdata(self.comm, obs_per_group=opg)

        # Create a set of boresight detectors
        self.ndet = 4
        self.rate = 20.0

        self.dnames, self.dquat, depsilon, drate, dnet, dfmin, dfknee, dalpha \
            = boresight_focalplane(self.ndet, samplerate=self.rate)

        # Properties of each observation

        self.obslen = 90.0
        self.obsgap = 10.0

        self.obstotalsamp = int((self.obslen + self.obsgap)
            * self.rate)
        self.obssamp = int(self.obslen * self.rate)
        self.obsgapsamp = self.obstotalsamp - self.obssamp

        self.obstotal = self.obstotalsamp / self.rate
        self.obslen = self.obssamp / self.rate
        self.obsgap = self.obstotal - self.obsgap

        # Properties of the intervals within an observation
        self.nsub = 12
        self.subtotsamp = self.obssamp // self.nsub
        self.subgapsamp = 0
        self.subsamp = self.subtotsamp - self.subgapsamp

        self.frames = list()
        for i in range(self.nsub):
            self.frames.append(self.subtotsamp)
        if self.obssamp > self.subtotsamp * self.nsub:
            self.frames.append(self.obssamp - self.subtotsamp * self.nsub)

        # Ground scan properties
        self.site_lon = '-67:47:10'
        self.site_lat = '-22:57:30'
        self.site_alt = 5200.
        self.coord = 'C'
        self.azmin=45
        self.azmax=55
        self.el=60
        self.scanrate = 5.0
        self.scan_accel = 100.0
        self.CES_start = None



    def tearDown(self):
        # Reset the frame file size.
        s3g.TARGET_FRAMEFILE_SIZE = self._original_framefile
        return


    def meta_setup(self):
        ret = dict()
        ret["string"] = "blahblahblah"
        ret["double"] = -1.234567890123e9
        ret["int64"] = -100000000000
        return ret


    def meta_verify(self, dct):
        nt.assert_equal(dct["string"], "blahblahblah")
        nt.assert_equal(dct["int64"], -100000000000)
        nt.assert_almost_equal(dct["double"], -1.234567890123e9)
        return


    def create_bore(self, total, local):
        theta_incr = (0.5*np.pi) / (total - 1)
        phi_incr = (2.0*np.pi) / (total - 1)

        theta_start = local[0] * theta_incr
        phi_start = local[0] * phi_incr

        theta_stop = theta_start + (local[1] - 1) * theta_incr
        phi_stop = phi_start + (local[1] - 1) * phi_incr

        theta = np.linspace(theta_start, theta_stop, num=local[1],
            endpoint=True, dtype=np.float64)
        phi = np.linspace(phi_start, phi_stop, num=local[1],
            endpoint=True, dtype=np.float64)
        pa = np.zeros(local[1], dtype=np.float64)

        return qa.from_angles(theta, phi, pa)


    def obs_create(self, comm, name):
        # fake metadata
        props = self.meta_setup()

        tod = s3g.TOD3G(comm, comm.size, detectors=self.dquat,
            samples=self.obssamp, framesizes=self.frames,
            azel=True, meta=props)

        obs = dict()
        obs["name"] = name
        obs["tod"] = tod

        return obs


    def obs_init(self, obs, start, off):
        tod = obs["tod"]
        # Now write the data.  For this test, we simply write the detector
        # index (as a float) to the detector timestream.  We also flag every
        # other sample.  For the boresight pointing, we create a fake spiral
        # pattern.

        detranks, sampranks = tod.grid_size
        rankdet, ranksamp = tod.grid_ranks

        off = tod.local_samples[0]
        n = tod.local_samples[1]

        # We use this for both the common and all the detector
        # flags just to check write/read roundtrip.
        flags = np.zeros(n, dtype=np.uint8)
        flags[::2] = 1

        # Everyone writes their timestamps

        incr = 1.0 / self.rate
        stamps = np.arange(n, dtype=np.float64)
        stamps *= incr
        stamps += start + (off * incr)
        tod.write_times(stamps=stamps)

        # Same with the boresight
        boresight = self.create_bore(tod.total_samples, tod.local_samples)
        tod.write_boresight(data=boresight)

        # Just duplicate the RA/DEC quaternions to AZ/EL.  We are just
        # checking read / write integrity.
        tod.write_boresight_azel(data=boresight)

        # Fake velocity / position data
        posvec = np.zeros((n, 3), dtype=np.float64)
        posvec[:,2] = 1.0
        tod.write_velocity(vel=posvec)
        tod.write_position(pos=posvec)

        # Now the common flags
        tod.write_common_flags(flags=flags)

        # Detector data
        fakedata = np.empty(n, dtype=np.float64)
        for d in tod.local_dets:
            # get unique detector index and convert to float
            indx = float(tod.detindx[d])
            # write data based on this to all local elements
            fakedata[:] = np.arange(n)
            fakedata += off
            fakedata *= indx
            tod.write(detector=d, data=fakedata)
            # write detector flags
            tod.write_flags(detector=d, flags=flags)
        return


    def obs_zero(self, obs, start, off):
        tod = obs["tod"]
        # Write empty data to all fields

        detranks, sampranks = tod.grid_size
        rankdet, ranksamp = tod.grid_ranks

        off = tod.local_samples[0]
        n = tod.local_samples[1]

        # We use this for both the common and all the detector
        # flags just to check write/read roundtrip.
        flags = np.zeros(n, dtype=np.uint8)

        # Everyone writes their timestamps

        stamps = np.zeros(n, dtype=np.float64)
        tod.write_times(stamps=stamps)

        # Same with the boresight
        boresight = np.zeros((n, 4), dtype=np.float64)
        tod.write_boresight(data=boresight)
        tod.write_boresight_azel(data=boresight)

        # Fake velocity / position data
        posvec = np.zeros((n, 3), dtype=np.float64)
        tod.write_velocity(vel=posvec)
        tod.write_position(pos=posvec)

        # Now the common flags
        tod.write_common_flags(flags=flags)

        # Detector data
        fakedata = np.zeros(n, dtype=np.float64)
        for d in tod.local_dets:
            tod.write(detector=d, data=fakedata)
            tod.write_flags(detector=d, flags=flags)
        return


    def obs_verify(self, tod, start, off):
        nlocal = tod.local_samples[1]
        odd = False
        if tod.local_samples[0] % 2 != 0:
            odd = True

        # Verify metadata

        self.meta_verify(tod.meta())

        # Read and compare timestamps

        compstamps = np.arange(nlocal, dtype=np.float64)
        compstamps /= self.rate
        compstamps += start + (tod.local_samples[0] / self.rate)

        stamps = tod.read_times()
        nt.assert_almost_equal(stamps, compstamps)
        del stamps

        # Read and compare boresight

        compbore = self.create_bore(tod.total_samples, tod.local_samples)

        boresight = tod.read_boresight()
        nt.assert_almost_equal(boresight, compbore)
        del boresight

        boresight = tod.read_boresight_azel()
        nt.assert_almost_equal(boresight, compbore)
        del boresight

        # flags.  We use this for both the common and all the detector
        # flags just to check write/read roundtrip.

        compflags = np.zeros(nlocal, dtype=np.uint8)
        if odd:
            compflags[1::2] = 1
        else:
            compflags[::2] = 1

        flags = tod.read_common_flags()
        nt.assert_equal(flags, compflags)
        del flags

        # detector data

        compdata = np.empty(nlocal, dtype=np.float64)
        for d in tod.local_dets:
            # get unique detector index and convert to float
            indx = float(tod.detindx[d])
            # comparison values
            compdata[:] = np.arange(tod.local_samples[1])
            compdata += tod.local_samples[0]
            compdata *= indx
            # read and check
            data = tod.read(detector=d)
            nt.assert_almost_equal(data, compdata)
            del data

            # check detector flags
            flags = tod.read_flags(detector=d)
            nt.assert_equal(flags, compflags)
            del flags

        return


    def data_init(self):
        data = Data(self.data.comm)

        for ob in range(len(self.data.obs)):
            obsname = "obs_{}_{:02d}".format(self.data.comm.group, ob)
            obsstart = ob * self.obstotal
            obsoff = ob * self.obstotalsamp
            obs = self.obs_create(self.data.comm.comm_group, obsname)
            self.obs_init(obs, obsstart, obsoff)
            data.obs.append(obs)

        return data


    def init_ground(self):
        # Create the simulated TODs
        for ob in range(len(self.data.obs)):
            obsname = "obs_{}_{:02d}".format(self.data.comm.group, ob)
            start = ob * self.obstotal
            first = ob * self.obstotalsamp
            tod = TODGround(
                self.data.comm.comm_group,
                self.dquat,
                self.obstotalsamp,
                detranks=self.data.comm.group_size,
                firsttime=start,
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
                CES_start=self.CES_start)
            self.data.obs[ob]["tod"] = tod
        return


    def data_verify(self, path, prefix):
        for ob in range(len(self.data.obs)):
            obsname = "obs_{}_{:02d}".format(self.data.comm.group, ob)
            obsdir = os.path.join(path, obsname)
            obsstart = ob * self.obstotal
            obsoff = ob * self.obstotalsamp
            tod = s3g.TOD3G(self.data.comm.comm_group,
                self.data.comm.comm_group.size, path=obsdir, prefix=prefix)
            self.obs_verify(tod, obsstart, obsoff)
        return


    def test_utils(self):
        s3g.TARGET_FRAMEFILE_SIZE = 200000
        # We want to test the frame operations with a process grid that has
        # multiple ranks in both the sample and detector directions.
        detranks = self.comm.size
        if self.comm.size % 2 == 0:
            detranks = 2

        # Create a simple tod with a cache that we can use for testing.
        tod_in = TODCache(self.comm, self.dnames, self.obssamp,
            detquats=self.dquat, detranks=detranks,
            sampsizes=self.frames)

        tod_out = TODCache(self.comm, self.dnames, self.obssamp,
            detquats=self.dquat, detranks=detranks,
            sampsizes=self.frames)

        # Write some fake data to this TOD
        obs_in = {"tod": tod_in}
        self.obs_init(obs_in, 0.0, 0)

        obs_out = {"tod": tod_out}
        self.obs_zero(obs_out, 0.0, 0)

        # For timestamps, we need to copy the internal cached timestamps
        # (float64) into spt3g timestamps.
        stamps = np.copy(tod_in.cache.reference(tod_in._stamps))
        stamps *= 1.0e9
        istamps = stamps.astype(np.int64)
        tod_in.cache.put("spt3gtime", istamps)

        # Make lists of fields that we are going to write to the spt3g frames.
        # These use the internal names of the cache objects in a TODCache
        # class.  We will test that we can dump this data to frames and
        # restore it.

        common = list()
        common.append( ("spt3gtime", c3g.G3VectorTime, "spt3gtime") )
        common.append( (tod_in._bore, c3g.G3VectorDouble, tod_in._bore) )
        common.append( (tod_in._bore_azel, c3g.G3VectorDouble, tod_in._bore_azel) )
        common.append( (tod_in._pos, c3g.G3VectorDouble, tod_in._pos) )
        common.append( (tod_in._vel, c3g.G3VectorDouble, tod_in._vel) )
        common.append( (tod_in._common, c3g.G3VectorUnsignedChar, tod_in._common) )

        detfields = [ ("{}{}".format(tod_in._pref_detdata, x),
                       "{}{}".format(tod_in._pref_detdata, x))\
                     for x in tod_in.detectors ]

        flagfields = [ ("{}{}".format(tod_in._pref_detflags, x),
                        "{}{}".format(tod_in._pref_detflags, x)) \
                     for x in tod_in.detectors ]

        off = 0
        frames = list()
        for findx, frm in enumerate(self.frames):
            #print("cache to frame {} at {}".format(findx, off), flush=True)
            fdata = s3utils.cache_to_frames(tod_in, findx, 1, [off], [frm],
                common=common, detector_fields=detfields,
                flag_fields=flagfields, units=c3g.G3TimestreamUnits.Tcmb)
            #print("  got ",fdata, flush=True)
            frames.extend(fdata)
            off += frm

        # Restore frames to cache

        off = 0
        for findx, frm in enumerate(self.frames):
            #print("frame to cache {} at {}".format(findx, off), flush=True)
            s3utils.frame_to_cache(tod_out, findx, off, frm,
                frame_data=frames[findx])
            off += frm

        # Compare input to output

        np.testing.assert_almost_equal(tod_in.cache.reference("spt3gtime"),
            tod_out.cache.reference("spt3gtime"))

        np.testing.assert_almost_equal(tod_in.read_position(),
            tod_out.read_position())

        np.testing.assert_almost_equal(tod_in.read_velocity(),
            tod_out.read_velocity())

        np.testing.assert_almost_equal(tod_in.read_boresight(),
            tod_out.read_boresight())

        np.testing.assert_almost_equal(tod_in.read_boresight_azel(),
            tod_out.read_boresight_azel())

        np.testing.assert_equal(tod_in.read_common_flags(),
            tod_out.read_common_flags())

        for det in tod_in.local_dets:
            np.testing.assert_almost_equal(tod_in.read(detector=det),
                tod_out.read(detector=det))
            np.testing.assert_equal(tod_in.read_flags(detector=det),
                tod_out.read_flags(detector=det))

        return


    def test_io(self):
        s3g.TARGET_FRAMEFILE_SIZE = 200000
        origdata = self.data_init()

        if self.comm.rank == 0:
            if os.path.isdir(self.datawrite):
                shutil.rmtree(self.datawrite)
        self.comm.barrier()

        dumper = s3g.Op3GExport(self.datawrite, s3g.TOD3G, use_todchunks=True,
                                export_opts={"prefix" : "test"})
        dumper.exec(origdata)

        #print("{}: Done with export 1".format(self.comm.rank), flush=True)
        self.comm.barrier()

        self.data_verify(self.datawrite, "test")

        #print("{}: Done with verify 1".format(self.comm.rank), flush=True)
        self.comm.barrier()

        loaddata = s3g.load_spt3g(self.data.comm,
                                  self.data.comm.comm_group.size,
                                  self.datawrite, "test",
                                  s3g.obsweight_spt3g,
                                  s3g.TOD3G)

        #print("{}: Done with load".format(self.comm.rank), flush=True)
        self.comm.barrier()

        if self.comm.rank == 0:
            if os.path.isdir(self.dataexport):
                shutil.rmtree(self.dataexport)
        self.comm.barrier()

        exporter = s3g.Op3GExport(self.dataexport, s3g.TOD3G,
                                  use_todchunks=True,
                                  export_opts={"prefix" : "test"})
        exporter.exec(loaddata)

        #print("{}: Done with export 2".format(self.comm.rank), flush=True)
        self.comm.barrier()

        self.data_verify(self.dataexport, "test")

        #print("{}: Done with verify 2".format(self.comm.rank), flush=True)
        self.comm.barrier()

        return


    def test_ground(self):
        s3g.TARGET_FRAMEFILE_SIZE = 400000
        # Create simulated tods in memory
        self.init_ground()

        # Export this
        if self.comm.rank == 0:
            if os.path.isdir(self.groundexport):
                shutil.rmtree(self.groundexport)
        self.comm.barrier()

        dumper = s3g.Op3GExport(self.groundexport, s3g.TOD3G,
                                use_todchunks=True,
                                export_opts={"prefix" : "test"})
        dumper.exec(self.data)

        # Load it back in.

        loaddata = s3g.load_spt3g(self.data.comm,
                                  self.data.comm.comm_group.size,
                                  self.groundexport, "test",
                                  s3g.obsweight_spt3g,
                                  s3g.TOD3G)

        return
