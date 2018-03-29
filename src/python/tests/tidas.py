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

# This file will only be imported if TIDAS is already available
import tidas as tds
from tidas.mpi import MPIVolume
from ..tod import tidas as tt


class TidasTest(MPITestCase):

    def setUp(self):
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

        self.outvol = os.path.join(self.outdir, "test_tidas")
        self.export = os.path.join(self.outdir, "export_tidas")
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

        # Group schema
        self.schm = tt.create_tidas_schema(self.dets, tds.DataType.float64,
            "volts")


    def tearDown(self):
        pass


    def meta_setup(self):
        ret = tds.Dictionary()
        ret.put_string("string", "blahblahblah")
        ret.put_float64("double", -1.234567890123e9)
        ret.put_float32("float", -123456.0)
        ret.put_int8("int8", -1)
        ret.put_uint8("uint8", 1)
        ret.put_int16("int16", -10000)
        ret.put_uint16("uint16", 10000)
        ret.put_int32("int32", -1000000000)
        ret.put_uint32("uint32", 1000000000)
        ret.put_int64("int64", -100000000000)
        ret.put_uint64("uint64", 100000000000)
        return ret


    def meta_verify(self, dct):
        nt.assert_equal(dct.get_string("string"), "blahblahblah")
        nt.assert_equal(dct.get_int8("int8"), -1)
        nt.assert_equal(dct.get_uint8("uint8"), 1)
        nt.assert_equal(dct.get_int16("int16"), -10000)
        nt.assert_equal(dct.get_uint16("uint16"), 10000)
        nt.assert_equal(dct.get_int32("int32"), -1000000000)
        nt.assert_equal(dct.get_uint32("uint32"), 1000000000)
        nt.assert_equal(dct.get_int64("int64"), -100000000000)
        nt.assert_equal(dct.get_uint64("uint64"), 100000000000)
        nt.assert_almost_equal(dct.get_float32("float"), -123456.0)
        nt.assert_almost_equal(dct.get_float64("double"), -1.234567890123e9)
        return


    def create_intervals(self, start, first):
        ret = []
        for i in range(self.nsub):
            ifirst = i * self.subtotsamp
            if i == self.nsub - 1:
                ilast = self.obssamp - 1
            else:
                ilast = (i+1) * self.subtotsamp - 1
            istart = float(ifirst) / self.rate
            istop = float(ilast) / self.rate
            ret.append(tds.Intrvl(
                (start + istart),
                (start + istop),
                (first + ifirst),
                (first + ilast)))
        return ret


    def intervals_init(self, start, first):
        return self.create_intervals(start, first)


    def intervals_verify(self, ilist, start, first):
        reg = self.create_intervals(start, first)
        nt.assert_equal(len(ilist), len(reg))
        for i in range(len(ilist)):
            nt.assert_almost_equal(ilist[i].start, reg[i].start)
            nt.assert_equal(ilist[i].first, reg[i].first)
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


    def obs_create(self, vol, parent, name, start, first):
        # fake metadata
        props = self.meta_setup()
        props = tt.encode_tidas_quats(self.detquats, props=props)

        # create intervals
        ilist = self.intervals_init(start, 0)

        # create the observation within the TIDAS volume

        obs = tt.create_tidas_obs(vol, parent, name,
            groups={
                "detectors" : (self.schm, props, self.obssamp)
            },
            intervals={
                "chunks" : (tds.Dictionary(), len(ilist))
            })

        # write the intervals that will be used for data distribution
        it = obs.intervals_get("chunks")
        it.write(ilist)

        return


    def obs_init(self, obscomm, vol, parent, name, start, first):
        # instantiate a TOD for this observation
        tod = tt.TODTidas(obscomm, vol, "{}/{}".format(parent, name),
            detgroup="detectors", distintervals="chunks")

        # Now write the data.  For this test, we simply write the detector
        # index (as a float) to the detector timestream.  We also flag every
        # other sample.  For the boresight pointing, we create a fake spiral
        # pattern.

        detranks, sampranks = tod.grid_size
        rankdet, ranksamp = tod.grid_ranks
        blk = tod.block

        off = tod.local_samples[0]
        n = tod.local_samples[1]

        # We use this for both the common and all the detector
        # flags just to check write/read roundtrip.
        flags = np.zeros(n, dtype=np.uint8)
        flags[::2] = 1

        if rankdet == 0:
            # Processes along the first row of the process grid write
            # their slice of the timestamps.
            incr = 1.0 / self.rate
            stamps = np.arange(n, dtype=np.float64)
            stamps *= incr
            stamps += start + (off * incr)

            for p in range(sampranks):
                if p == ranksamp:
                    tod.write_times(stamps=stamps)
                tod.grid_comm_row.barrier()

            # Same with the boresight
            boresight = self.create_bore(tod.total_samples, tod.local_samples)

            for p in range(sampranks):
                if p == ranksamp:
                    tod.write_boresight(data=boresight)
                tod.grid_comm_row.barrier()

            # Now the common flags
            for p in range(sampranks):
                if p == ranksamp:
                    tod.write_common_flags(flags=flags)
                tod.grid_comm_row.barrier()

        tod.mpicomm.barrier()

        # Detector data- serialize for now
        for p in range(tod.mpicomm.size):
            if p == tod.mpicomm.rank:
                fakedata = np.empty(n, dtype=np.float64)
                for d in tod.local_dets:
                    # get unique detector index and convert to float
                    indx = float(tod.detindx[d])
                    # write this to all local elements
                    fakedata[:] = indx
                    tod.write(detector=d, data=fakedata)
                    # write detector flags
                    tod.write_flags(detector=d, flags=flags)
            tod.mpicomm.barrier()

        del tod

        return


    def obs_verify(self, tod, start, first):
        nlocal = tod.local_samples[1]
        odd = False
        if tod.local_samples[0] % 2 != 0:
            odd = True

        # Read the intervals and compare

        intr = tod.block.intervals_get("chunks")
        ilist = intr.read()
        self.intervals_verify(ilist, start, 0)

        # Verify group properties

        self.meta_verify(tod.group.dictionary())

        # Read and compare timestamps

        compstamps = np.arange(nlocal, dtype=np.float64)
        compstamps /= self.rate
        compstamps += start + (tod.local_samples[0] / self.rate)

        stamps = tod.read_times()

        nt.assert_almost_equal(stamps, compstamps)

        # Read and compare boresight

        compbore = self.create_bore(tod.total_samples, tod.local_samples)
        boresight = tod.read_boresight()

        nt.assert_almost_equal(boresight, compbore)

        # flags.  We use this for both the common and all the detector
        # flags just to check write/read roundtrip.

        compflags = np.zeros(nlocal, dtype=np.uint8)
        if odd:
            compflags[1::2] = 1
        else:
            compflags[::2] = 1

        flags = tod.read_common_flags()

        nt.assert_equal(flags, compflags)

        # detector data

        compdata = np.empty(nlocal, dtype=np.float64)
        for d in tod.local_dets:
            # get unique detector index and convert to float
            indx = float(tod.detindx[d])
            # comparison values
            compdata[:] = indx
            # read and check
            data = tod.read(detector=d)
            nt.assert_almost_equal(data, compdata)
            # check detector flags
            flags = tod.read_flags(detector=d)
            nt.assert_equal(flags, compflags)

        return


    def volume_init(self, path):
        if self.comm.rank == 0:
            if os.path.isdir(path):
                shutil.rmtree(path)
        self.comm.barrier()

        # Create the volume on the world communicator
        vol = MPIVolume(self.toastcomm.comm_world, path, tds.BackendType.hdf5,
            tds.CompressionType.none, dict())

        # Usually for real data we will have a hierarchy of blocks
        # (year, month, season, etc).  For this test we just add generic
        # observations to the root block.

        # One process from each group creates the observations
        for ob in range(self.myobs_first, self.myobs_first + self.myobs_n):
            obsname = "obs_{:02d}".format(ob)
            start = ob * self.obstotal
            first = ob * self.obstotalsamp
            if self.toastcomm.comm_group.rank == 0:
                self.obs_create(vol, "", obsname, start, first)

        vol.meta_sync()

        for ob in range(self.myobs_first, self.myobs_first + self.myobs_n):
            obsname = "obs_{:02d}".format(ob)
            start = ob * self.obstotal
            first = ob * self.obstotalsamp
            self.obs_init(self.toastcomm.comm_group, vol, "", obsname, start,
                first)

        vol.meta_sync()

        del vol

        return


    def volume_verify(self, path):
        vol = MPIVolume(self.toastcomm.comm_world, path, tds.AccessMode.read)
        root = vol.root()
        for ob in range(self.myobs_first, self.myobs_first + self.myobs_n):
            obsname = "obs_{:02d}".format(ob)
            obs = root.block_get(obsname)
            start = ob * self.obstotal
            first = ob * self.obstotalsamp
            tod = tt.TODTidas(self.toastcomm.comm_group, vol,
                "/{}".format(obsname), detgroup="detectors",
                distintervals="chunks")
            self.obs_verify(tod, start, first)
        del vol
        return


    def test_io(self):
        start = MPI.Wtime()

        self.volume_init(self.outvol)
        self.volume_verify(self.outvol)

        stop = MPI.Wtime()
        elapsed = stop - start
        #print("Proc {}:  test took {:.4f} s".format( MPI.COMM_WORLD.rank, elapsed ))


    def test_export(self):
        start = MPI.Wtime()

        self.volume_init(self.outvol)

        distdata = tt.load_tidas(self.toastcomm, self.outvol, mode="w",
            distintervals="chunks")

        if self.comm.rank == 0:
            if os.path.isdir(self.export):
                shutil.rmtree(self.export)
        self.comm.barrier()

        dumper = tt.OpTidasExport(self.export, usedist=True)
        dumper.exec(distdata)

        self.volume_verify(self.export)

        stop = MPI.Wtime()
        elapsed = stop - start

        #print("Proc {}:  test took {:.4f} s".format( MPI.COMM_WORLD.rank, elapsed ))
