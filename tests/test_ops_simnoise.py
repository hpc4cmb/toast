# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import sys
import os

if 'TOAST_NO_MPI' in os.environ.keys():
    from toast import fakempi as MPI
else:
    from mpi4py import MPI

import numpy as np
import numpy.testing as nt

import scipy.interpolate as si

from toast.tod.tod import *
from toast.tod.pointing import *
from toast.tod.noise import *
from toast.tod.sim_noise import *
from toast.tod.sim_detdata import *
from toast.tod.sim_tod import *

from toast.mpirunner import MPITestCase


class OpSimNoiseTest(MPITestCase):

    def setUp(self):
        self.outdir = "tests_output"
        if self.comm.rank == 0:
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)

        # Note: self.comm is set by the test infrastructure
        self.worldsize = self.comm.size
        if (self.worldsize >= 2):
            self.groupsize = int( self.worldsize / 2 )
            self.ngroup = 2
        else:
            self.groupsize = 1
            self.ngroup = 1
        self.toastcomm = Comm(world=self.comm, groupsize=self.groupsize)
        self.data = Data(self.toastcomm)

        self.dets = ["f1a", "f1b", "f2a", "f2b"]
        self.fp = {}
        for d in self.dets:
            self.fp[d] = np.array([0.0, 0.0, 1.0, 0.0])

        self.rate = 20.0
        self.fmin = 0.05
        self.fknee = {}
        self.alpha = {}
        self.NET = {}

        self.fknee["f1a"] = 0.1
        self.alpha["f1a"] = 1.0
        self.NET["f1a"] = 10.0

        self.fknee["f1b"] = 0.1
        self.alpha["f1b"] = 1.0
        self.NET["f1b"] = 10.0

        self.fknee["f2a"] = 0.15
        self.alpha["f2a"] = 1.0
        self.NET["f2a"] = 10.0

        self.fknee["f2b"] = 0.15
        self.alpha["f2b"] = 1.0
        self.NET["f2b"] = 10.0

        self.totsamp = 50000

        self.MC = 100

        # in order to make sure that the noise realization is reproducible
        # all all concurrencies, we set the chunksize to something independent
        # of the number of ranks.

        nchunk = 10
        chunksize = int(self.totsamp / nchunk)
        chunks = np.ones(nchunk, dtype=np.int64)
        chunks *= chunksize
        remain = self.totsamp - (nchunk * chunksize)
        for r in range(remain):
            chunks[r] += 1

        self.chunksize = chunksize

        # Construct an empty TOD (no pointing needed)

        self.tod = TODHpixSpiral(mpicomm=self.toastcomm.comm_group, detectors=self.fp, samples=self.totsamp, firsttime=0.0, rate=self.rate, nside=512, sizes=chunks)

        # construct an analytic noise model

        self.nse = AnalyticNoise(rate=self.rate, fmin=self.fmin, detectors=self.dets, fknee=self.fknee, alpha=self.alpha, NET=self.NET)

        ob = {}
        ob['id'] = 'noisetest-{}'.format(self.toastcomm.group)
        ob['tod'] = self.tod
        ob['intervals'] = None
        ob['baselines'] = None
        ob['noise'] = self.nse

        self.data.obs.append(ob)


    def test_sim(self):
        start = MPI.Wtime()

        ob = self.data.obs[0]
        tod = ob['tod']
        nse = ob['noise']

        # verify that the white noise part of the spectrum is normalized correctly

        fsamp = nse.rate
        cutoff = 0.95 * (fsamp / 2.0)
        indx = np.where(nse.freq > cutoff)
        for det in tod.local_dets:
            NET = self.nse.NET(det)
            knee = self.nse.fknee(det)
            avg = np.mean(nse.psd(det)[indx])
            NETsq = NET*NET
            print("det {} NETsq = {}, average white noise level = {}".format(det, NETsq, avg))
            self.assertTrue((np.absolute(avg - NETsq)/NETsq) < 0.02)

        ntod = self.totsamp

        # Reconstruct the fft length that was used when generating the TOD.
        # Then interpolate the PSD to this sampling before integrating.

        oversample = 2 # this matches default in OpSimNoise...

        fftlen = 2
        half = 1
        while fftlen <= (oversample * self.chunksize):
            fftlen *= 2
            half *= 2

        rawfreq = nse.freq
        nyquist = fsamp / 2.0

        df = fsamp / fftlen
        freq = np.linspace(df, df*half, num=half, endpoint=True)

        logfreq = np.log10(freq)
        lograwfreq = np.log10(rawfreq)

        psds = {}
        psdnorm = {}
        todvar = {}

        for det in tod.local_dets:
            todvar[det] = np.zeros(self.MC, dtype=np.float64)

            rawpsd = nse.psd(det)
            lograwpsd = np.log10(rawpsd)

            interp = si.InterpolatedUnivariateSpline(lograwfreq, lograwpsd, k=1, ext=0)
            logpsd = interp(logfreq)
            psds[det] = np.power(10.0, logpsd)

            # Factor of 2 comes from the negative frequency values.
            psdnorm[det] = 2.0 * np.sum(psds[det] * df)
            print("psd[{}] integral = {}".format(det, psdnorm[det]))

        if self.comm.rank == 0:
            np.savetxt(os.path.join(self.outdir,"out_test_simnoise_psd.txt"), np.transpose([freq, psds[self.dets[0]], psds[self.dets[1]], psds[self.dets[2]], psds[self.dets[3]]]), delimiter=' ')

        for r in range(self.MC):

            # generate timestreams

            op = OpSimNoise(stream=0, realization=r)
            op.exec(self.data)

            if r == 0:
                # write timestreams to disk for debugging
                check1 = tod.cache.reference("noise_{}".format(self.dets[0]))
                check2 = tod.cache.reference("noise_{}".format(self.dets[1]))
                check3 = tod.cache.reference("noise_{}".format(self.dets[2]))
                check4 = tod.cache.reference("noise_{}".format(self.dets[3]))
                if self.comm.rank == 0:
                    np.savetxt(os.path.join(self.outdir,"out_test_simnoise_tod.txt"), np.transpose([check1, check2, check3, check4]), delimiter=' ')

                # verify that timestreams with the same PSD *DO NOT* have the same
                # values (this is a crude test that the RNG state is being incremented)

                dif = np.fabs(check1 - check2)
                check = np.mean(dif)
                self.assertTrue(check > (0.01 / np.sqrt(self.totsamp)))

                dif = np.fabs(check3 - check4)
                check = np.mean(dif)
                self.assertTrue(check > (0.01 / np.sqrt(self.totsamp)))

            for det in tod.local_dets:
                # compute the TOD variance
                td = tod.cache.reference("noise_{}".format(det))
                dclevel = np.mean(td)
                variance = np.vdot(td-dclevel, td-dclevel) / ntod
                todvar[det][r] = variance

            tod.cache.clear()


        if self.comm.rank == 0:
            np.savetxt(os.path.join(self.outdir,"out_test_simnoise_tod_var.txt"), np.transpose([todvar[self.dets[0]], todvar[self.dets[1]], todvar[self.dets[2]], todvar[self.dets[3]]]), delimiter=' ')

        # Verify that Parseval's theorem holds- that the variance of the TOD
        # equals the integral of the PSD.  We do this for an ensemble of realizations
        # and compare the TOD variance to the integral of the PSD accounting
        # for the error on the variance due to finite numbers of samples.

        for det in tod.local_dets:
            histcenter = psdnorm[det]
            print("tod[{}] mean variance = {}".format(det, np.mean(todvar[det])))
            sig = np.mean(todvar[det]) * np.sqrt(2.0/(ntod-1))

            histrange = 3.0*sig
            histmin = histcenter - histrange
            histmax = histcenter + histrange
            nbins = 10
            histbins = np.arange(nbins, dtype=np.float64)
            histbins -= 0.5 * nbins
            histbins *= 2.0 * histrange / nbins
            histbins += histcenter
            hist = np.histogram(todvar[det], bins=nbins, range=(histmin, histmax))[0]
            if self.comm.rank == 0:
                np.savetxt(os.path.join(self.outdir,"out_test_simnoise_var_{}.txt".format(det)), np.transpose([histbins, hist]), delimiter=' ')

            over3sig = np.where(np.absolute(todvar[det] - histcenter) > 3.0*sig)[0]
            overfrac = float(len(over3sig)) / self.MC
            print(overfrac)
            self.assertTrue(overfrac < 0.1)

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("simnoise test took {:.3f} s".format(elapsed))

