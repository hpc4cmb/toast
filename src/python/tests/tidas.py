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

from ..tod import tidas_available
from ..tod import Interval

if tidas_available:
    import tidas as tds
    from tidas.mpi_volume import MPIVolume
    from ..tod import tidas as tt


class TidasTest(MPITestCase):

    def setUp(self):
        self.outdir = "toast_test_output"
        if self.comm.rank == 0:
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)
        self.outvol = os.path.join(self.outdir, "test_tidas")
        self.export = os.path.join(self.outdir, "export_tidas")
        self.rate = 20.0

        # Properties of the observations
        self.nobs = 6
        self.obslen = 3600.0
        self.obsgap = 600.0

        self.obstotalsamp = int(0.5 + (self.obslen + self.obsgap) 
            * self.rate) + 1
        self.obssamp = int(0.5 + self.obslen * self.rate) + 1
        self.obsgapsamp = self.obstotalsamp - self.obssamp

        self.obstotal = (self.obstotalsamp - 1) / self.rate
        self.obslen = (self.obssamp - 1) / self.rate
        self.obsgap = (self.obsgapsamp - 1) / self.rate
        
        # Properties of the intervals within an observation
        self.nsub = 5
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

        # Skip the rest of the setup if we don't have tidas.
        if not tidas_available:
            return

        # Group schema
        self.schm = tt.create_tidas_schema(self.dets, "float64", "volts")


    def tearDown(self):
        pass
        

    def meta_setup(self):
        ret = {}
        ret["string"] = "blahblahblah"
        ret["double"] = -123456789.0123
        ret["float"] = -123456789.0123
        ret["int8"] = -100
        ret["uint8"] = 100
        ret["int16"] = -10000
        ret["uint16"] = 10000
        ret["int32"] = -1000000000
        ret["uint32"] = 1000000000
        ret["int64"] = -100000000000
        ret["uint64"] = 100000000000
        return ret


    def meta_verify(self, dct):
        nt.assert_equal(dct["string"], "blahblahblah")
        nt.assert_equal(dct["int8"], -100)
        nt.assert_equal(dct["uint8"], 100)
        nt.assert_equal(dct["int16"], -10000)
        nt.assert_equal(dct["uint16"], 10000)
        nt.assert_equal(dct["int32"], -1000000000)
        nt.assert_equal(dct["uint32"], 1000000000)
        nt.assert_equal(dct["int64"], -100000000000)
        nt.assert_equal(dct["uint64"], 100000000000)
        nt.assert_almost_equal(dct["float"], -123456789.0123)
        nt.assert_almost_equal(dct["double"], -123456789.0123)
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
            ret.append(Interval(
                start=(start + istart),
                stop=(start + istop), 
                first=(first + ifirst), 
                last=(first + ilast)))
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


    def obs_init(self, vol, parent, name, start, first):

        # fake metadata
        props = self.meta_setup()
        props = tt.encode_tidas_quats(self.detquats, props=props)

        # create intervals
        ilist = self.intervals_init(start, 0)

        # create the observation within the TIDAS volume

        obs = tt.create_tidas_obs(vol, parent, name, 
            groups={
                "detectors" : (self.schm, self.obssamp, props)
            }, 
            intervals={
                "chunks" : (len(ilist), dict())
            })

        # write the intervals that will be used for data distribution
        if vol.comm.rank == 0:
            obs.intervals_get("chunks").write(ilist)

        # instantiate a TOD for this observation

        tod = tt.TODTidas(vol.comm, vol, "{}/{}".format(parent, name), 
            detgroup="detectors", distintervals="chunks")

        # Now write the data.  For this test, we simple write the detector
        # index (as a float) to the detector timestream.  We also flag every
        # other sample.  For the boresight pointing, we create a fake spiral
        # pattern.

        if vol.comm.rank == 0:
            # number of samples
            n = tod.total_samples

            # Write some simple timestamps
            incr = 1.0 / self.rate
            stamps = np.arange(n, dtype=np.float64)
            stamps *= incr
            stamps += start

            tod.write_times(stamps=stamps)

            # boresight
            boresight = self.create_bore(n, (0,n))
            tod.write_boresight(data=boresight)

            # flags.  We use this for both the common and all the detector
            # flags just to check write/read roundtrip.
            flags = np.zeros(n, dtype=np.uint8)
            flags[::2] = 1

            tod.write_common_flags(flags=flags)

            # detector data
            fakedata = np.empty(n, dtype=np.float64)
            for d in tod.detectors:
                # get unique detector index and convert to float
                indx = float(tod.detindx[d])
                # write this to all local elements
                fakedata[:] = indx
                tod.write(detector=d, data=fakedata)
                # write detector flags
                tod.write_det_flags(detector=d, flags=flags)

        # FIXME: consider doing this in parallel once either TIDAS supports
        # that or there is a toast-specific workaround to serialize I/O to a
        # single HDF5 file.
        #
        # # number of local samples
        # nlocal = tod.local_samples[1]

        # # Write some simple timestamps
        # stamps = np.arange(nlocal, dtype=np.float64)
        # stamps /= self.rate
        # stamps += start + (tod.local_samples[0] / self.rate)

        # tod.write_times(stamps=stamps)

        # # boresight
        # boresight = self.create_bore(tod.total_samples, tod.local_samples)
        # tod.write_boresight(boresight)

        # # flags.  We use this for both the common and all the detector
        # # flags just to check write/read roundtrip.
        # flags = np.zeros(nlocal, dtype=np.uint8)
        # flags[::2] = 1

        # tod.write_common_flags(flags=flags)

        # # detector data
        # fakedata = np.empty(nlocal, dtype=np.float64)
        # for d in range(len(self.dets)):
        #     # get unique detector index and convert to float
        #     indx = float(tod.detindx[d])
        #     # write this to all local elements
        #     fakedata[:] = indx
        #     tod.write(detector=self.dets[d], data=fakedata)
        #     # write detector flags
        #     tod.write_det_flags(detector=self.dets[d], flags=flags)

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

        self.meta_verify(tod.group.props)

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
            flags, cflags = tod.read_flags(detector=d)
            nt.assert_equal(flags, compflags)

        return


    def volume_init(self, path):
        if self.comm.rank == 0:
            if os.path.isdir(path):
                shutil.rmtree(path)
        self.comm.barrier()

        with MPIVolume(self.comm, path, backend="hdf5", comp="none") as vol:
            # Usually for real data we will have a hierarchy of blocks 
            # (year, month, season, etc).  For this test we just add generic
            # observations to the root block.
            for ob in range(self.nobs):
                obsname = "obs_{:02d}".format(ob)
                start = ob * self.obstotal
                first = ob * self.obstotalsamp
                self.obs_init(vol, "", obsname, start, first)
        return

 
    def volume_verify(self, path):
        with MPIVolume(self.comm, path) as vol:
            root = vol.root()
            for ob in range(self.nobs):
                obsname = "obs_{:02d}".format(ob)
                obs = root.block_get(obsname)
                start = ob * self.obstotal
                first = ob * self.obstotalsamp
                tod = tt.TODTidas(self.comm, vol, "/{}".format(obsname), 
                    detgroup="detectors", distintervals="chunks")
                self.obs_verify(tod, start, first)
        return


    @unittest.skipIf(not tidas_available, "TIDAS not found")
    def test_io(self):
        start = MPI.Wtime()

        self.volume_init(self.outvol)
        self.volume_verify(self.outvol)

        stop = MPI.Wtime()
        elapsed = stop - start
        #print("Proc {}:  test took {:.4f} s".format( MPI.COMM_WORLD.rank, elapsed ))


    @unittest.skipIf(not tidas_available, "TIDAS not found")
    def test_export(self):
        start = MPI.Wtime()

        self.volume_init(self.outvol)

        worldsize = self.comm.size
        if (worldsize >= 2):
            groupsize = int( worldsize / 2 )
            ngroup = 2
        else:
            groupsize = 1
            ngroup = 1
        toastcomm = Comm(self.comm, groupsize=groupsize)

        distdata = tt.load_tidas(toastcomm, self.outvol, mode="r",
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

