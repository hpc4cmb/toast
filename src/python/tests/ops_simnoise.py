# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

import sys
import os

import numpy as np
import numpy.testing as nt

import scipy.interpolate as si

from ..tod.tod import *
from ..tod.pointing import *
from ..tod.noise import *
from ..tod.sim_noise import *
from ..tod.sim_detdata import *
from ..tod.sim_tod import *

from .. import rng as rng


class OpSimNoiseTest(MPITestCase):

    def setUp(self):
        self.outdir = "toast_test_output"
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

        self.dets = ["f1a", "f1b", "f2a", "f2b", "white", "high"]
        self.fp = {}
        for d in self.dets:
            self.fp[d] = np.array([0.0, 0.0, 1.0, 0.0])

        self.rate = 20.0

        self.rates = {}
        self.fmin = {}
        self.fknee = {}
        self.alpha = {}
        self.NET = {}

        self.rates["f1a"] = self.rate
        self.fmin["f1a"] = 1.0e-5
        self.fknee["f1a"] = 0.15
        self.alpha["f1a"] = 1.0
        self.NET["f1a"] = 10.0

        self.rates["f1b"] = self.rate
        self.fmin["f1b"] = 1.0e-5
        self.fknee["f1b"] = 0.1
        self.alpha["f1b"] = 1.0
        self.NET["f1b"] = 10.0

        self.rates["f2a"] = self.rate
        self.fmin["f2a"] = 1.0e-5
        self.fknee["f2a"] = 0.05
        self.alpha["f2a"] = 1.0
        self.NET["f2a"] = 10.0

        self.rates["f2b"] = self.rate
        self.fmin["f2b"] = 1.0e-5
        self.fknee["f2b"] = 0.001
        self.alpha["f2b"] = 1.0
        self.NET["f2b"] = 10.0

        self.rates["white"] = self.rate
        self.fmin["white"] = 0.0
        self.fknee["white"] = 0.0
        self.alpha["white"] = 1.0
        self.NET["white"] = 10.0

        self.rates["high"] = self.rate
        self.fmin["high"] = 1.0e-5
        self.fknee["high"] = 40.0
        self.alpha["high"] = 2.0
        self.NET["high"] = 10.0

        self.totsamp = 20000

        self.oversample = 2

        self.MC = 100

        # in order to make sure that the noise realization is reproducible
        # all all concurrencies, we set the chunksize to something independent
        # of the number of ranks.

        nchunk = 4
        chunksize = int(self.totsamp / nchunk)
        chunks = np.ones(nchunk, dtype=np.int64)
        chunks *= chunksize
        remain = self.totsamp - (nchunk * chunksize)
        for r in range(remain):
            chunks[r] += 1

        self.chunksize = chunksize

        # Construct an empty TOD (no pointing needed)

        self.tod = TODHpixSpiral(
            self.toastcomm.comm_group, 
            self.fp, 
            self.totsamp, 
            firsttime=0.0, 
            rate=self.rate, 
            nside=512, 
            sampsizes=chunks)

        # construct an analytic noise model

        self.nse = AnalyticNoise(
            rate=self.rates, 
            fmin=self.fmin, 
            detectors=self.dets, 
            fknee=self.fknee, 
            alpha=self.alpha, 
            NET=self.NET
        )

        ob = {}
        ob['name'] = 'noisetest-{}'.format(self.toastcomm.group)
        ob['id'] = 0
        ob['tod'] = self.tod
        ob['intervals'] = None
        ob['baselines'] = None
        ob['noise'] = self.nse

        self.data.obs.append(ob)


    def test_gauss(self):
        # test that the same samples from different calls are reproducible.

        ob = self.data.obs[0]
        nse = ob['noise']

        det = self.dets[-1]

        psd = np.ones_like(nse.freq(det))
        npsd = len(psd)
        
        detindx = len(self.dets) - 1
        telescope = 5
        realization = 1000
        component = 3
        obsindx = 1234

        key1 = realization * 4294967296 + telescope * 65536 + component
        key2 = obsindx * 4294967296 + detindx 
        counter1 = 0

        compoff = 500
        ncomp = 10

        counter2 = 0
        d1 = rng.random(1000, sampler="gaussian", key=(key1, key2), 
            counter=(counter1, counter2))
        print(d1[compoff:compoff+ncomp])

        counter2 = 100
        d2 = rng.random(1000, sampler="gaussian", key=(key1, key2), 
            counter=(counter1, counter2))
        print(d2[compoff-100:compoff-100+ncomp])

        counter2 = 300
        d3 = rng.random(1000, sampler="gaussian", key=(key1, key2), 
            counter=(counter1, counter2))
        print(d3[compoff-300:compoff-300+ncomp])

        np.testing.assert_array_almost_equal(d1[compoff:compoff+ncomp], d2[compoff-100:compoff-100+ncomp])
        np.testing.assert_array_almost_equal(d1[compoff:compoff+ncomp], d3[compoff-300:compoff-300+ncomp])


    def test_sim(self):
        start = MPI.Wtime()

        ob = self.data.obs[0]
        tod = ob['tod']
        nse = ob['noise']

        # verify that the white noise part of the spectrum is normalized correctly

        for det in tod.local_dets:
            fsamp = nse.rate(det)
            cutoff = 0.95 * (fsamp / 2.0)
            indx = np.where(nse.freq(det) > cutoff)

            NET = nse.NET(det)
            knee = nse.fknee(det)
            avg = np.mean(nse.psd(det)[indx])
            NETsq = NET*NET
            print("det {} NETsq = {}, average white noise level = {}".format(det, NETsq, avg))
            if det != "high":
                self.assertTrue((np.absolute(avg - NETsq)/NETsq) < 0.02)

        if self.comm.rank == 0:
            import matplotlib.pyplot as plt

            for det in tod.local_dets:
                savefile = os.path.join(self.outdir, "out_test_simnoise_rawpsd_{}.txt".format(det))
                np.savetxt(savefile, np.transpose([nse.freq(det), nse.psd(det)]), delimiter=' ')

                fig = plt.figure(figsize=(12,8), dpi=72)
                ax = fig.add_subplot(1, 1, 1, aspect='auto')
                ax.loglog(nse.freq(det), nse.psd(det), marker='o', c="red", label='{}: rate={:0.1f} NET={:0.1f} fknee={:0.4f}, fmin={:0.4f}'.format(det, self.rates[det], self.NET[det], self.fknee[det], self.fmin[det]))
                cur_ylim = ax.get_ylim()
                ax.set_ylim([0.001*(nse.NET(det)**2), 10.0*cur_ylim[1]])
                ax.legend(loc=1)
                plt.title("Simulated PSD from toast.AnalyticNoise")

                savefile = os.path.join(self.outdir, "out_test_simnoise_rawpsd_{}.png".format(det))
                plt.savefig(savefile)
                plt.close()

        ntod = self.totsamp

        # this replicates the calculation in sim_noise_timestream()

        fftlen = 2
        while fftlen <= (self.oversample * self.chunksize):
            fftlen *= 2

        freqs = {}
        psds = {}
        psdnorm = {}
        todvar = {}

        cfftlen = 2
        while cfftlen <= ntod:
            cfftlen *= 2

        #print("fftlen = ", fftlen)
        #print("cfftlen = ", cfftlen)

        checkpsd = {}
        binsamps = cfftlen // 4096
        nbins = binsamps - 1
        bstart = (self.rate / 2) / nbins
        bins = np.linspace(bstart, self.rate / 2, num=(nbins-1), endpoint=True)
        #print("nbins = ",nbins)
        #print(bins)

        checkfreq = np.fft.rfftfreq(cfftlen, d=1/self.rate)
        #print("checkfreq len = ",len(checkfreq))
        #print(checkfreq[:10])
        #print(checkfreq[-10:])
        checkbinmap = np.searchsorted(bins, checkfreq, side='left')
        #print("checkbinmap len = ",len(checkbinmap))
        #print(checkbinmap[:10])
        #print(checkbinmap[-10:])
        bcount = np.bincount(checkbinmap)
        #print("bcount len = ",len(bcount))
        #print(bcount)

        bintruth = {}

        idet = 0
        for det in tod.local_dets:

            df = nse.rate(det) / float(fftlen)

            (temp, freqs[det], psds[det]) = sim_noise_timestream(0, 0, 0, 0, idet, nse.rate(det), 0, self.chunksize, self.oversample, nse.freq(det), nse.psd(det))

            # Factor of 2 comes from the negative frequency values.
            psdnorm[det] = 2.0 * np.sum(psds[det] * df)
            print("psd[{}] integral = {}".format(det, psdnorm[det]))

            todvar[det] = np.zeros(self.MC, dtype=np.float64)
            checkpsd[det] = np.zeros((nbins-1, self.MC), dtype=np.float64)

            idet += 1

        if self.comm.rank == 0:
            import matplotlib.pyplot as plt

            for det in tod.local_dets:
                savefile = os.path.join(self.outdir, "out_test_simnoise_psd_{}.txt".format(det))
                np.savetxt(savefile, np.transpose([freqs[det], psds[det]]), delimiter=' ')

                fig = plt.figure(figsize=(12,8), dpi=72)
                ax = fig.add_subplot(1, 1, 1, aspect='auto')
                ax.loglog(freqs[det], psds[det], marker='+', c="blue", label='{}: rate={:0.1f} NET={:0.1f} fknee={:0.4f}, fmin={:0.4f}'.format(det, self.rates[det], self.NET[det], self.fknee[det], self.fmin[det]))
                cur_ylim = ax.get_ylim()
                ax.set_ylim([0.001*(nse.NET(det)**2), 10.0*cur_ylim[1]])
                ax.legend(loc=1)
                plt.title("Interpolated PSD with High-pass from {:0.1f} second Simulation Interval".format((float(self.totsamp)/self.rate)))

                savefile = os.path.join(self.outdir, "out_test_simnoise_psd_{}.png".format(det))
                plt.savefig(savefile)
                plt.close()

                tmap = np.searchsorted(bins, freqs[det], side='left')
                tcount = np.bincount(tmap)
                tpsd = np.bincount(tmap, weights=psds[det])
                good = (tcount > 0)
                tpsd[good] /= tcount[good]
                bintruth[det] = tpsd

        # Run both the numpy FFT case and the toast FFT case.

        for case in ['npFFT', 'toastFFT']:

            for r in range(self.MC):

                # generate timestreams

                op = OpSimNoise(realization=r, altFFT=(case == 'toastFFT'))
                op.exec(self.data)

                if r == 0:
                    # write timestreams to disk for debugging

                    if self.comm.rank == 0:
                        import matplotlib.pyplot as plt

                        for det in tod.local_dets:

                            check = tod.cache.reference("noise_{}".format(det))

                            savefile = os.path.join(self.outdir, "out_{}_test_simnoise_tod_mc0_{}.txt".format(case, det))
                            np.savetxt(savefile, np.transpose([check]), delimiter=' ')

                            fig = plt.figure(figsize=(12,8), dpi=72)
                            ax = fig.add_subplot(1, 1, 1, aspect='auto')
                            ax.plot(np.arange(len(check)), check, c="black", label='Det {}'.format(det))
                            ax.legend(loc=1)
                            plt.title("First Realization of Simulated TOD from toast.sim_noise_timestream()")

                            savefile = os.path.join(self.outdir, "out_{}_test_simnoise_tod_mc0_{}.png".format(case, det))
                            plt.savefig(savefile)
                            plt.close()

                for det in tod.local_dets:
                    # compute the TOD variance
                    td = tod.cache.reference("noise_{}".format(det))
                    dclevel = np.mean(td)
                    variance = np.vdot(td-dclevel, td-dclevel) / ntod
                    todvar[det][r] = variance

                    # compute the PSD
                    buffer = np.zeros(cfftlen, dtype=np.float64)
                    offset = (cfftlen - len(td)) // 2
                    buffer[offset:offset+len(td)] = td
                    rawpsd = np.fft.rfft(buffer)
                    norm = 1.0 / (self.rate * self.totsamp)
                    rawpsd = norm * np.abs(rawpsd**2)
                    bpsd = np.bincount(checkbinmap, weights=rawpsd)
                    good = (bcount > 0)
                    bpsd[good] /= bcount[good]
                    checkpsd[det][:,r] = bpsd[:]

                tod.cache.clear()

            if self.comm.rank == 0:
                np.savetxt(os.path.join(self.outdir,"out_{}_test_simnoise_tod_var.txt".format(case)), np.transpose([todvar[self.dets[0]], todvar[self.dets[1]], todvar[self.dets[2]], todvar[self.dets[3]], todvar[self.dets[4]]]), delimiter=' ')

            if self.comm.rank == 0:
                import matplotlib.pyplot as plt

                for det in tod.local_dets:
                    savefile = os.path.join(self.outdir, "out_{}_test_simnoise_tod_var_{}.txt".format(case, det))
                    np.savetxt(savefile, np.transpose([todvar[det]]), delimiter=' ')

                    sig = np.mean(todvar[det]) * np.sqrt(2.0/(self.chunksize-1))
                    histrange = 5.0 * sig
                    histmin = psdnorm[det] - histrange
                    histmax = psdnorm[det] + histrange

                    fig = plt.figure(figsize=(12,8), dpi=72)

                    ax = fig.add_subplot(1, 1, 1, aspect='auto')
                    hn, hbins, hpatches = plt.hist(todvar[det], 10, range=(histmin, histmax), facecolor="magenta", alpha=0.75, label="{}:  PSD integral = {:0.1f} expected sigma = {:0.1f}".format(det, psdnorm[det], sig))
                    ax.legend(loc=1)
                    plt.title("Distribution of TOD Variance for {} Realizations".format(self.MC))

                    savefile = os.path.join(self.outdir, "out_{}_test_simnoise_tod_var_{}.png".format(case, det))
                    plt.savefig(savefile)
                    plt.close()

                    meanpsd = np.asarray([ np.mean(checkpsd[det][x,:]) for x in range(nbins-1) ])

                    fig = plt.figure(figsize=(12,8), dpi=72)

                    ax = fig.add_subplot(1, 1, 1, aspect='auto')
                    ax.plot(bins, bintruth[det], c='k', label="Input Truth")
                    ax.plot(bins, meanpsd, c='b', marker="o", label="Mean Binned PSD")
                    ax.scatter(np.repeat(bins, self.MC), checkpsd[det].flatten(), marker='x', color='r', label="Binned PSD")
                    #ax.set_xscale('log')
                    #ax.set_yscale('log')
                    ax.legend(loc=1)
                    plt.title("Detector {} Binned PSDs for {} Realizations".format(det, self.MC))

                    savefile = os.path.join(self.outdir, "out_{}_test_simnoise_binpsd_dist_{}.png".format(case, det))
                    plt.savefig(savefile)
                    plt.close()

                    # The data will likely not be gaussian distributed.  Just check that the mean
                    # is "close enough" to the truth.
                    errest = np.absolute(np.mean((meanpsd - tpsd) / tpsd))
                    print("Det {} avg rel error = {}".format(det, errest))
                    if self.fknee[det] < 0.1:
                        self.assertTrue(errest < 0.1)


            # Verify that Parseval's theorem holds- that the variance of the TOD
            # equals the integral of the PSD.  We do this for an ensemble of realizations
            # and compare the TOD variance to the integral of the PSD accounting
            # for the error on the variance due to finite numbers of samples.

            for det in tod.local_dets:
                sig = np.mean(todvar[det]) * np.sqrt(2.0/(self.chunksize-1))
                over3sig = np.where(np.absolute(todvar[det] - psdnorm[det]) > 3.0*sig)[0]
                overfrac = float(len(over3sig)) / self.MC
                print(det, " : ", overfrac)
                if self.fknee[det] < 0.1:
                    self.assertTrue(overfrac < 0.1)


        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("simnoise test took {:.3f} s".format(elapsed))

