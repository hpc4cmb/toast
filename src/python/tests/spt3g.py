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

from ..tod import Interval

# This file will only be imported if SPT3G is already available
from ..tod import spt3g as s3g


class Spt3gTest(MPITestCase):

    def setUp(self):
        # Reset the frame file size for these tests, so that we can test
        # boundary effects.
        self._original_framefile = s3g.TARGET_FRAMEFILE_SIZE
        s3g.TARGET_FRAMEFILE_SIZE = 100000

        # Note: self.comm is set by the test infrastructure
        self.worldsize = self.comm.size
        self.groupsize = None
        self.ngroup = None
        if (self.worldsize >= 2):
            self.groupsize = int( self.worldsize / 2 )
            self.ngroup = 2
        else:
            self.groupsize = 1
            self.ngroup = 1
        self.toastcomm = Comm(self.comm, groupsize=self.groupsize)

        self.mygroup = self.toastcomm.group

        self.outdir = "toast_test_output"
        if self.comm.rank == 0:
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)

        self.datawrite = os.path.join(self.outdir, "test_3g")
        self.dataexport = os.path.join(self.outdir, "export_3g")
        self.rate = 20.0

        # Properties of the observations
        self.nobs = 8

        dist_obs = distribute_uniform(self.nobs, self.ngroup)
        self.myobs_first = dist_obs[self.mygroup][0]
        self.myobs_n = dist_obs[self.mygroup][1]

        self.obslen = 90.0
        self.obsgap = 10.0

        self.obstotalsamp = int((self.obslen + self.obsgap)
            * self.rate)
        self.obssamp = int(self.obslen * self.rate)
        self.obsgapsamp = self.obstotalsamp - self.obssamp

        self.obstotal = (self.obstotalsamp - 1) / self.rate
        self.obslen = (self.obssamp - 1) / self.rate
        self.obsgap = (self.obsgapsamp - 1) / self.rate

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

        # Detectors
        self.dets = [
            "d100-1a",
            "d100-1b",
            "d145-2a",
            "d145-2b",
            "d220-3a",
            "d220-3b"
        ]

        self.detquats = {}
        for d in self.dets:
            self.detquats[d] = np.array([0,0,0,1], dtype=np.float64)


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


    def obs_create(self, comm, name, path, prefix):
        # fake metadata
        props = self.meta_setup()

        tod = s3g.TOD3G(comm, comm.size, path=path,
            prefix=prefix, detectors=self.detquats,
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


    def data_init(self, path, prefix):
        data = Data(self.toastcomm)

        for ob in range(self.myobs_first, self.myobs_first + self.myobs_n):
            obsname = "obs_{:02d}".format(ob)
            obsstart = ob * self.obstotal
            obsoff = ob * self.obstotalsamp
            obs = self.obs_create(self.toastcomm.comm_group, obsname, path,
                prefix)
            self.obs_init(obs, obsstart, obsoff)
            data.obs.append(obs)

        return data


    def data_verify(self, path, prefix):
        for ob in range(self.myobs_first, self.myobs_first + self.myobs_n):
            obsname = "obs_{:02d}".format(ob)
            obsdir = os.path.join(path, obsname)
            obsstart = ob * self.obstotal
            obsoff = ob * self.obstotalsamp
            tod = s3g.TOD3G(self.toastcomm.comm_group,
                self.toastcomm.comm_group.size, path=obsdir, prefix=prefix)
            self.obs_verify(tod, obsstart, obsoff)
        return


    def test_io(self):
        start = MPI.Wtime()

        origdata = self.data_init(self.datawrite, "test")

        if self.comm.rank == 0:
            if os.path.isdir(self.datawrite):
                shutil.rmtree(self.datawrite)
        self.comm.barrier()

        dumper = s3g.Op3GExport(self.datawrite, "test", s3g.TOD3G,
            use_todchunks=True)
        dumper.exec(origdata)

        self.data_verify(self.datawrite, "test")

        loaddata = s3g.load_spt3g(self.toastcomm,
            self.toastcomm.comm_group.size, self.datawrite, "test", s3g.TOD3G)

        if self.comm.rank == 0:
            if os.path.isdir(self.dataexport):
                shutil.rmtree(self.dataexport)
        self.comm.barrier()

        exporter = s3g.Op3GExport(self.dataexport, "test", s3g.TOD3G,
            use_todchunks=True)
        exporter.exec(loaddata)

        self.data_verify(self.dataexport, "test")

        stop = MPI.Wtime()
        elapsed = stop - start
        return
