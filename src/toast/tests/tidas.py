# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

import os
import shutil

import numpy as np
import numpy.testing as nt

from .. import qarray as qa

from ..tod import regular_intervals, TODGround

from ._helpers import create_outdir, create_distdata, boresight_focalplane

# This file will only be imported if TIDAS is already available
import tidas as tds
from tidas.mpi import MPIVolume
from ..tod import tidas as tt
from ..tod import tidas_utils as ttutils


class TidasTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        self.outvol = os.path.join(self.outdir, "test_tidas")
        self.export = os.path.join(self.outdir, "export_tidas")
        self.groundexport = os.path.join(self.outdir, "export_tidas_ground")

        # Create observations, divided evenly between groups
        self.nobs = 4
        opg = self.nobs
        if (self.comm is not None) and (self.comm.size >= 2):
            opg = self.nobs // 2
        self.data = create_distdata(self.comm, obs_per_group=opg)

        # Create a set of boresight detectors
        self.ndet = 8
        self.rate = 20.0

        (
            self.dnames,
            self.dquat,
            depsilon,
            drate,
            dnet,
            dfmin,
            dfknee,
            dalpha,
        ) = boresight_focalplane(self.ndet, samplerate=self.rate)

        # Create a set of intervals that sub-divide each observation.

        self.obslen = 90.0
        self.obsgap = 10.0

        self.obstotalsamp = int((self.obslen + self.obsgap) * self.rate)
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

        # Ground scan properties
        self.site_lon = "-67:47:10"
        self.site_lat = "-22:57:30"
        self.site_alt = 5200.0
        self.coord = "C"
        self.azmin = 45
        self.azmax = 55
        self.el = 60
        self.scanrate = 1.0
        self.scan_accel = 0.1
        self.CES_start = None

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
        return ttutils.to_dict(ret)

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
        subtime = self.subsamp / self.rate
        subgap = self.subgapsamp / self.rate

        ret = regular_intervals(self.nsub, start, first, self.rate, subtime, subgap)

        tret = [tds.Intrvl(x.start, x.stop, x.first, x.last) for x in ret]

        return tret

    def intervals_init(self, start, first):
        return self.create_intervals(start, first)

    def intervals_verify(self, ilist, start, first):
        reg = self.create_intervals(start, first)
        inlen = int(len(ilist))
        checklen = int(len(reg))
        self.assertTrue(inlen == checklen)

        for i in range(len(ilist)):
            nt.assert_almost_equal(ilist[i].start, reg[i].start)
            nt.assert_equal(ilist[i].first, reg[i].first)
        return

    def create_bore(self, total, local):
        theta_incr = (0.5 * np.pi) / (total - 1)
        phi_incr = (2.0 * np.pi) / (total - 1)

        theta_start = local[0] * theta_incr
        phi_start = local[0] * phi_incr

        theta_stop = theta_start + (local[1] - 1) * theta_incr
        phi_stop = phi_start + (local[1] - 1) * phi_incr

        theta = np.linspace(
            theta_start, theta_stop, num=local[1], endpoint=True, dtype=np.float64
        )
        phi = np.linspace(
            phi_start, phi_stop, num=local[1], endpoint=True, dtype=np.float64
        )
        pa = np.zeros(local[1], dtype=np.float64)

        return qa.from_angles(theta, phi, pa)

    def obs_create(self, vol, name, obsid, start, first):
        # create the observation within the TIDAS volume
        tob = ttutils.find_obs(vol, "", name)
        obsprops = tds.Dictionary()
        obsprops.put_int64("obs_id", obsid)
        obsprops.put_int64("obs_telescope_id", 1234)
        obsprops.put_int64("obs_site_id", 5678)

        # Create the observation group
        obsgroup = tob.group_add("observation", tds.Group(tds.Schema(), obsprops, 0))
        del obsprops
        del obsgroup

        # create intervals
        ilist = self.intervals_init(start, 0)
        dist = [(y.first - x.first) for x, y in zip(ilist[:-1], ilist[1:])]
        dist.append(self.obstotalsamp - ilist[-1].first)

        intr = tob.intervals_add("chunks", tds.Intervals(tds.Dictionary(), len(ilist)))
        # print("rank {}: write intervals".format(self.comm.rank), flush=True)
        intr.write(ilist)
        del intr
        del tob

        # Create a TOD for this observation.
        tt.TODTidas.create(
            vol,
            "/{}".format(name),
            self.dquat,
            self.obstotalsamp,
            self.meta_setup(),
            True,
        )

        return

    def obs_init(self, obscomm, vol, name, start, first):
        detranks = 1
        if obscomm is not None:
            detranks = obscomm.size
        tod = tt.TODTidas(
            obscomm, detranks, vol, "/{}".format(name), distintervals="chunks"
        )

        # print("rank {}: done create TOD {}".format(self.comm.rank, name), flush=True)

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

        if rankdet == 0:
            # print("rank {}: working on timestamps".format(self.comm.rank), flush=True)
            # Processes along the first row of the process grid write
            # their slice of the timestamps.
            incr = 1.0 / self.rate
            stamps = np.arange(n, dtype=np.float64)
            stamps *= incr
            stamps += start + (off * incr)

            for p in range(sampranks):
                if p == ranksamp:
                    # print("rank {}: write timestamp slice".format(self.comm.rank), flush=True)
                    tod.write_times(stamps=stamps)
                # print("rank {}: grid row barrier".format(self.comm.rank), flush=True)
                if tod.grid_comm_row is not None:
                    tod.grid_comm_row.barrier()

            # print("rank {}: done timestamps".format(self.comm.rank), flush=True)

            # Same with the boresight
            boresight = self.create_bore(tod.total_samples, tod.local_samples)

            for p in range(sampranks):
                if p == ranksamp:
                    # print("rank {}: write boresight".format(self.comm.rank), flush=True)
                    tod.write_boresight(data=boresight)
                # print("rank {}: bore grid row barrier".format(self.comm.rank), flush=True)
                if tod.grid_comm_row is not None:
                    tod.grid_comm_row.barrier()

            # Now the common flags
            for p in range(sampranks):
                if p == ranksamp:
                    # print("rank {}: write common".format(self.comm.rank), flush=True)
                    tod.write_common_flags(flags=flags)
                # print("rank {}: com grid row barrier".format(self.comm.rank), flush=True)
                if tod.grid_comm_row is not None:
                    tod.grid_comm_row.barrier()

        # print("rank {}: hit tod comm barrier".format(self.comm.rank), flush=True)
        if tod.mpicomm is not None:
            tod.mpicomm.barrier()
        # print("rank {}: done tod comm barrier".format(self.comm.rank), flush=True)

        # Detector data- serialize for now
        groupsize = 1
        grouprank = 0
        if tod.mpicomm is not None:
            groupsize = tod.mpicomm.size
            grouprank = tod.mpicomm.rank

        for p in range(groupsize):
            if p == grouprank:
                fakedata = np.empty(n, dtype=np.float64)
                for d in tod.local_dets:
                    # get unique detector index and convert to float
                    indx = float(tod.detindx[d])
                    # write this to all local elements
                    fakedata[:] = np.arange(tod.local_samples[1])
                    fakedata += tod.local_samples[0]
                    fakedata *= indx
                    # print("rank {}: write det {}".format(self.comm.rank, d), flush=True)
                    tod.write(detector=d, data=fakedata)
                    # write detector flags
                    tod.write_flags(detector=d, flags=flags)
            # print("rank {}: write det barrier".format(self.comm.rank), flush=True)
            if tod.mpicomm is not None:
                tod.mpicomm.barrier()

        del tod

        return

    def obs_verify(self, vol, parent, name, tod, start, first, ignore_last):
        nlocal = tod.local_samples[1]
        odd = False
        if tod.local_samples[0] % 2 != 0:
            odd = True

        block = ttutils.find_obs(vol, parent, name)

        # Read the intervals and compare

        intr = block.intervals_get("chunks")
        ilist = intr.read()
        if ignore_last:
            # These intervals included a final one representing the last
            # gap in sampling.  Trim it before comparing.
            ilist = ilist[0:-1]
        self.intervals_verify(ilist, start, 0)

        # Verify group properties

        self.meta_verify(ttutils.from_dict(tod.meta()))

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
            compdata[:] = np.arange(tod.local_samples[1])
            compdata += tod.local_samples[0]
            compdata *= indx
            # read and check
            data = tod.read(detector=d)
            nt.assert_almost_equal(data, compdata)
            # check detector flags
            flags = tod.read_flags(detector=d)
            nt.assert_equal(flags, compflags)

        return

    def volume_init(self, path):
        rank = 0
        grouprank = 0
        if self.comm is not None:
            rank = self.comm.rank
            grouprank = self.data.comm.group_rank
        if rank == 0:
            if os.path.isdir(path):
                shutil.rmtree(path)
        if self.comm is not None:
            self.comm.barrier()

        # print("rank {}: creating volume".format(self.comm.rank), flush=True)
        # Create the volume on the world communicator
        vol = None
        if self.comm is None:
            vol = tds.Volume(
                path, tds.BackendType.hdf5, tds.CompressionType.none, dict()
            )
        else:
            vol = MPIVolume(
                self.data.comm.comm_world,
                path,
                tds.BackendType.hdf5,
                tds.CompressionType.none,
                dict(),
            )
        del vol
        # print("rank {}: creating volume done".format(self.comm.rank), flush=True)
        # Usually for real data we will have a hierarchy of blocks
        # (year, month, season, etc).  For this test we just add generic
        # observations to the root block.

        # For simplicity, just have every group take turns writing their data.
        # this is fine for a small unit test.

        for pgrp in range(self.data.comm.ngroups):
            if pgrp == self.data.comm.group:
                vol = None
                if self.comm is None:
                    vol = tds.Volume(path, tds.AccessMode.write)
                else:
                    vol = MPIVolume(
                        self.data.comm.comm_group, path, tds.AccessMode.write
                    )

                # One process from each group creates the observations
                for ob in range(len(self.data.obs)):
                    obsname = "obs_{}_{:02d}".format(self.data.comm.group, ob)
                    start = ob * self.obstotal
                    first = ob * self.obstotalsamp
                    if grouprank == 0:
                        self.obs_create(vol, obsname, ob, start, first)

                if self.comm is not None:
                    vol.meta_sync()

                for ob in range(len(self.data.obs)):
                    obsname = "obs_{}_{:02d}".format(self.data.comm.group, ob)
                    start = ob * self.obstotal
                    first = ob * self.obstotalsamp
                    # print("rank {}: init obs {}".format(self.comm.rank, obsname), flush=True)
                    self.obs_init(self.data.comm.comm_group, vol, obsname, start, first)

                if self.comm is not None:
                    vol.meta_sync()
                del vol

            if self.comm is not None:
                self.comm.barrier()

        return

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
                CES_start=self.CES_start,
            )
            self.data.obs[ob]["tod"] = tod
        return

    def volume_verify(self, path, detgroup="detectors", ignore_last=False):
        vol = None
        if self.comm is None:
            vol = tds.Volume(path, tds.AccessMode.read)
        else:
            vol = MPIVolume(self.data.comm.comm_world, path, tds.AccessMode.read)
        root = vol.root()

        for ob in range(len(self.data.obs)):
            obsname = "obs_{}_{:02d}".format(self.data.comm.group, ob)
            # print("rank {}: verify obs {}".format(self.comm.rank, obsname), flush=True)
            obs = root.block_get(obsname)
            start = ob * self.obstotal
            first = ob * self.obstotalsamp
            ilist = self.intervals_init(start, 0)
            dist = [(y.first - x.first) for x, y in zip(ilist[:-1], ilist[1:])]
            dist.append(self.obstotalsamp - ilist[-1].first)
            tod = tt.TODTidas(
                self.data.comm.comm_group,
                self.data.comm.group_size,
                vol,
                "/{}".format(obsname),
                distintervals=dist,
                group_dets=detgroup,
            )
            self.obs_verify(vol, "", obsname, tod, start, first, ignore_last)
        del vol
        return

    def test_io(self):
        self.volume_init(self.outvol)
        self.volume_verify(self.outvol)
        return

    def test_export(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank
        self.volume_init(self.outvol)

        distdata = tt.load_tidas(
            self.data.comm,
            self.data.comm.group_size,
            self.outvol,
            "w",
            "detectors",
            tt.TODTidas,
            distintervals="chunks",
        )

        if rank == 0:
            if os.path.isdir(self.export):
                shutil.rmtree(self.export)
        if self.comm is not None:
            self.comm.barrier()

        # NOTE:  the use_todchunks option will take the distribution chunks
        # of the TOD and create a set of intervals in the volume.  These
        # chunks were themselves created from a set of intervals above with
        # gaps.  So there will be final chunk that is the last "gap" period
        # to add up to the total samples.
        dumper = tt.OpTidasExport(
            self.export, tt.TODTidas, backend="hdf5", use_todchunks=True
        )
        dumper.exec(distdata)

        self.volume_verify(self.export, ignore_last=True)

        # print("finished verify 1", flush=True)
        distdata2 = tt.load_tidas(
            self.data.comm,
            self.data.comm.group_size,
            self.export,
            "r",
            "detectors",
            tt.TODTidas,
            group_dets="detectors",
            distintervals="chunks",
        )

        # print("finished load 2", flush=True)

        dumper2 = tt.OpTidasExport(
            self.export,
            tt.TODTidas,
            use_todchunks=True,
            create_opts={"group_dets": "bolos"},
            ctor_opts={"group_dets": "bolos"},
        )
        dumper2.exec(distdata2)

        # print("finished export 2", flush=True)

        self.volume_verify(self.export, detgroup="bolos", ignore_last=True)

        return

    def test_ground(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank
        # Create simulated tods in memory
        self.init_ground()

        # Export this
        if rank == 0:
            if os.path.isdir(self.groundexport):
                shutil.rmtree(self.groundexport)
        if self.comm is not None:
            self.comm.barrier()

        dumper = tt.OpTidasExport(
            self.groundexport, tt.TODTidas, backend="hdf5", use_todchunks=True
        )
        dumper.exec(self.data)

        # Load it back in.
        distdata = tt.load_tidas(
            self.data.comm,
            self.data.comm.group_size,
            self.groundexport,
            "w",
            "detectors",
            tt.TODTidas,
            distintervals="chunks",
            group_dets="detectors",
        )

        return
