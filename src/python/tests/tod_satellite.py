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

from ._helpers import (create_outdir, create_distdata, boresight_focalplane,
    uniform_chunks)


class TODSatelliteTest(MPITestCase):

    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        # Create 366 observations, divided evenly between groups
        self.nobs = 366
        opg = self.nobs
        if self.comm.size >= 2:
            opg = self.nobs // 2
        self.data = create_distdata(self.comm, obs_per_group=opg)

        # One boresight detector
        self.ndet = 1
        self.rate = 1.0 / 60.0

        dnames, dquat, depsilon, drate, dnet, dfmin, dfknee, dalpha = \
            boresight_focalplane(self.ndet, samplerate=self.rate)

        # Scan strategy.
        # Choose scan parameters so that we return to the origin.
        precangle = 35.0
        spinangle = 55.0
        spinperiod = 240
        precperiod = 8640

        # Precession axis slew
        degday = 360.0 / self.nobs

        # Sampling

        # Only simulate 22 out of 24 hours per observation, in order to test
        # that the starting phase is propagated.  Put the "gap" at the start
        # of each day, so that we can look at the final data point of the
        # final observation.

        daysamps = 24 * 60
        #goodsamps = 24 * 60
        goodsamps = 22 * 60
        badsamps = daysamps - goodsamps

        # Populate the observations

        for oid, obs in enumerate(self.data.obs):
            firstsamp = oid * daysamps + badsamps

            # On the last observation, simulate one extra sample so we get
            # back to the starting point.
            nsim = goodsamps
            if (oid == len(self.data.obs) - 1) and (self.data.comm.group == 1):
                nsim += 1

            tod = TODSatellite(
                self.data.comm.comm_group,
                dquat,
                nsim,
                detranks=1,
                firstsamp=firstsamp,
                firsttime=(firstsamp / self.rate),
                rate=self.rate,
                spinperiod=spinperiod,
                spinangle=spinangle,
                precperiod=precperiod,
                precangle=precangle
            )

            qprec = np.empty(4 * tod.local_samples[1],
                dtype=np.float64).reshape((-1, 4))

            slew_precession_axis(qprec,
                firstsamp=(firstsamp + tod.local_samples[0]),
                samplerate=self.rate, degday=degday)

            tod.set_prec_axis(qprec=qprec)

            obs["tod"] = tod


    def tearDown(self):
        pass


    def test_precession(self):
        # Test precession axis slew
        slewrate = 1.0 / (24.0*3600.0)
        nobs = 366
        degday = 360.0 / nobs

        # Plus one is to get back to the beginning
        nsim = nobs + 1

        qprec = np.empty(4 * nsim, dtype=np.float64).reshape((-1, 4))

        slew_precession_axis(qprec, firstsamp=0,
            samplerate=slewrate, degday=degday)

        zaxis = np.array([0.0, 0.0, 1.0])

        v = qa.rotate(qprec, zaxis)

        dotprod = v[0][0] * v[-1][0] + v[0][1] * v[-1][1] + v[0][2] * v[-1][2]
        np.testing.assert_almost_equal(dotprod, 1.0)
        return


    def test_phase(self):
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

            qprec = np.empty(4 * nsim, dtype=np.float64).reshape((-1, 4))
            slew_precession_axis(qprec, firstsamp=firstsamp,
                samplerate=samplerate, degday=degday)

            boresight = np.empty(4 * nsim, dtype=np.float64).reshape((-1, 4))
            satellite_scanning(boresight, firstsamp=firstsamp,
                samplerate=samplerate, qprec=qprec, spinperiod=spinperiod,
                spinangle=spinangle, precperiod=precperiod, precangle=precangle)

            v = qa.rotate(boresight, zaxis)
            p = pix.vec2ring(v)
            pdata[p] += 1
            vlast = v[-1]

        if self.comm.rank == 0:
            import matplotlib.pyplot as plt
            hitsfile = os.path.join(self.outdir, 'tod_satellite_hits.fits')
            if self.comm.rank == 0:
                if os.path.isfile(hitsfile):
                    os.remove(hitsfile)
            hp.write_map(hitsfile, pdata, nest=False, dtype=np.int32)

            outfile = "{}.png".format(hitsfile)
            hp.mollview(pdata, xsize=1600, nest=False)
            plt.savefig(outfile)
            plt.close()

        np.testing.assert_almost_equal(vlast[0], 0.0)
        np.testing.assert_almost_equal(vlast[1], -1.0)
        np.testing.assert_almost_equal(vlast[2], 0.0)
        return


    def test_todclass(self):
        # Hit map
        nside = 32
        pix = hpx.Pixels(nside)
        pdata = np.zeros(12*nside*nside, dtype=np.int32)

        # Compute the boresight hitmap

        zaxis = np.array([0.0, 0.0, 1.0])
        vlast = None

        for obs in self.data.obs:
            tod = obs["tod"]
            boresight = tod.read_boresight()
            v = qa.rotate(boresight, zaxis)
            p = pix.vec2ring(v)
            pdata[p] += 1
            vlast = v[-1]

        allpdata = None
        if self.comm.rank == 0:
            allpdata = np.zeros_like(pdata)
        self.comm.Reduce(pdata, allpdata, op=MPI.SUM, root=0)

        if self.comm.rank == 0:
            import matplotlib.pyplot as plt
            hitsfile = os.path.join(self.outdir, 'tod_satellite_classhits.fits')
            if os.path.isfile(hitsfile):
                os.remove(hitsfile)
            hp.write_map(hitsfile, allpdata, nest=False, dtype=np.int32)

            outfile = "{}.png".format(hitsfile)
            hp.mollview(allpdata, xsize=1600, nest=False)
            plt.savefig(outfile)
            plt.close()

        # The last sample on the last process should be back to the origin
        lastpass = True
        lastproc = (self.comm.size - 1)
        if self.comm.rank == lastproc:
            lastpass = np.allclose(vlast[0], 0.0)
            lastpass = np.allclose(vlast[1], -1.0)
            lastpass = np.allclose(vlast[2], 0.0)
        lastpass = self.comm.bcast(lastpass, root=lastproc)
        self.assertTrue(lastpass)

        return
