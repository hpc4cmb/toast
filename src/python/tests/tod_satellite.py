# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

import sys
import os
import numpy as np

import healpy as hp

from ..dist import Comm

from .. import healpix as hpx

from .. import qarray as qa

from ..tod.sim_tod import (slew_precession_axis, satellite_scanning,
    TODSatellite)


class TODSatelliteTest(MPITestCase):

    def setUp(self):
        self.outdir = "toast_test_output"
        if self.comm.rank == 0:
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)

    def tearDown(self):
        pass


    def test_precession(self):
        start = MPI.Wtime()

        # Test precession axis slew
        slewrate = 1.0 / (24.0*3600.0)
        degday = 360.0 / 365.0

        qprec = slew_precession_axis(nsim=366, firstsamp=0,
            samplerate=slewrate, degday=degday)

        zaxis = np.array([0.0, 0.0, 1.0])

        v = qa.rotate(qprec, zaxis)

        dotprod = v[0][0] * v[-1][0] + v[0][1] * v[-1][1] + v[0][2] * v[-1][2]
        np.testing.assert_almost_equal(dotprod, 1.0)

        stop = MPI.Wtime()
        elapsed = stop - start
        #print("Proc {}:  test took {:.4f} s".format(MPI.COMM_WORLD.rank, elapsed))


    def test_phase(self):
        start = MPI.Wtime()

        # Choose scan parameters so that we return to the origin.
        nobs = 366
        precangle = 35.0
        spinangle = 55.0
        spinperiod = 240
        precperiod = 8640

        # Precession axis slew
        degday = 360.0 / 366.0

        samplerate = 1.0 / 60.0

        daysamps = 24 * 60
        goodsamps = 24 * 60
        #goodsamps = 22 * 60
        badsamps = daysamps - goodsamps

        # hit map
        nside = 32
        pix = hpx.Pixels(nside)
        pdata = np.zeros(12*nside*nside, dtype=np.int32)

        # Only simulate 22 out of 24 hours per observation, in order to test
        # that the starting phase is propagated.  Put the "gap" at the start
        # of each day, so that we can look at the final data point of the
        # final observation.

        zaxis = np.array([0.0, 0.0, 1.0])

        vlast = None

        for ob in range(nobs):
            firstsamp = ob * daysamps + badsamps

            # On the last observation, simulate one extra sample so we get
            # back to the starting point.
            nsim = goodsamps
            if ob == nobs - 1:
                nsim += 1

            qprec = slew_precession_axis(nsim=nsim, firstsamp=firstsamp,
                samplerate=samplerate, degday=degday)

            boresight = satellite_scanning(nsim=nsim, firstsamp=firstsamp,
                samplerate=samplerate, qprec=qprec, spinperiod=spinperiod,
                spinangle=spinangle, precperiod=precperiod, precangle=precangle)

            v = qa.rotate(boresight, zaxis)
            p = pix.vec2ring(v)
            pdata[p] += 1
            vlast = v[-1]
            # print("obs {}:  {} -->".format(ob, v[0]))
            # print("obs {}:      {}".format(ob, v[-6]))
            # print("obs {}:      {}".format(ob, v[-5]))
            # print("obs {}:      {}".format(ob, v[-4]))
            # print("obs {}:      {}".format(ob, v[-3]))
            # print("obs {}:      {}".format(ob, v[-2]))
            # print("obs {}:      {}".format(ob, v[-1]))

        import matplotlib.pyplot as plt
        hitsfile = os.path.join(self.outdir, 'tod_satellite_hits.fits')
        if self.comm.rank == 0:
            if os.path.isfile(hitsfile):
                os.remove(hitsfile)
        self.comm.barrier()
        hp.write_map(hitsfile, pdata, nest=False, dtype=np.int32)

        outfile = "{}.png".format(hitsfile)
        hp.mollview(pdata, xsize=1600, nest=False)
        plt.savefig(outfile)
        plt.close()

        np.testing.assert_almost_equal(vlast[0], 0.0)
        np.testing.assert_almost_equal(vlast[1], -1.0)
        np.testing.assert_almost_equal(vlast[2], 0.0)

        stop = MPI.Wtime()
        elapsed = stop - start
        #print("Proc {}:  test took {:.4f} s".format(MPI.COMM_WORLD.rank, elapsed))


    def test_todclass(self):
        start = MPI.Wtime()

        tcomm = Comm(world=self.comm, groupsize=self.comm.size)

        # Choose scan parameters so that we return to the origin.
        nobs = 366
        precangle = 35.0
        spinangle = 55.0
        spinperiod = 240
        precperiod = 8640

        # Precession axis slew
        degday = 360.0 / 366.0

        samplerate = 1.0 / 60.0

        daysamps = 24 * 60
        goodsamps = 24 * 60
        #goodsamps = 22 * 60
        badsamps = daysamps - goodsamps

        # hit map
        nside = 32
        pix = hpx.Pixels(nside)
        pdata = np.zeros(12*nside*nside, dtype=np.int32)

        # Only simulate 22 out of 24 hours per observation, in order to test
        # that the starting phase is propagated.  Put the "gap" at the start
        # of each day, so that we can look at the final data point of the
        # final observation.

        zaxis = np.array([0.0, 0.0, 1.0])

        vlast = None

        dets = {
            'bore' : np.array([0.0, 0.0, 1.0, 0.0])
        }

        for ob in range(nobs):
            firstsamp = ob * daysamps + badsamps

            # On the last observation, simulate one extra sample so we get
            # back to the starting point.
            nsim = goodsamps
            if ob == nobs - 1:
                nsim += 1

            tod = TODSatellite(
                tcomm.comm_group, dets, nsim,
                firstsamp=firstsamp,
                firsttime=(firstsamp / samplerate),
                rate=samplerate,
                spinperiod=spinperiod,
                spinangle=spinangle,
                precperiod=precperiod,
                precangle=precangle
            )

            qprec = slew_precession_axis(nsim=nsim, firstsamp=firstsamp,
                samplerate=samplerate, degday=degday)

            tod.set_prec_axis(qprec=qprec)

            boresight = tod.read_boresight()

            v = qa.rotate(boresight, zaxis)
            p = pix.vec2ring(v)
            pdata[p] += 1
            vlast = v[-1]
            print("obs {}:  {} -->".format(ob, v[0]))
            print("obs {}:      {}".format(ob, v[-6]))
            print("obs {}:      {}".format(ob, v[-5]))
            print("obs {}:      {}".format(ob, v[-4]))
            print("obs {}:      {}".format(ob, v[-3]))
            print("obs {}:      {}".format(ob, v[-2]))
            print("obs {}:      {}".format(ob, v[-1]))

        import matplotlib.pyplot as plt
        hitsfile = os.path.join(self.outdir, 'tod_satellite_classhits.fits')
        if self.comm.rank == 0:
            if os.path.isfile(hitsfile):
                os.remove(hitsfile)
        self.comm.barrier()
        hp.write_map(hitsfile, pdata, nest=False, dtype=np.int32)

        outfile = "{}.png".format(hitsfile)
        hp.mollview(pdata, xsize=1600, nest=False)
        plt.savefig(outfile)
        plt.close()

        np.testing.assert_almost_equal(vlast[0], 0.0)
        np.testing.assert_almost_equal(vlast[1], -1.0)
        np.testing.assert_almost_equal(vlast[2], 0.0)

        stop = MPI.Wtime()
        elapsed = stop - start
        #print("Proc {}:  test took {:.4f} s".format(MPI.COMM_WORLD.rank, elapsed))
