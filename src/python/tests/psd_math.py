# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

import sys
import os

import numpy as np

from ..tod.tod import *
from ..tod.pointing import *
from ..tod.noise import *
from ..tod.sim_noise import *
from ..tod.sim_det_noise import *
from ..tod.sim_tod import *

from ..fod import autocov_psd


class PSDTest(MPITestCase):


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
        self.fknee["f1a"] = 20.00
        self.alpha["f1a"] = 2.0
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
        self.fknee["high"] = 2.0
        self.alpha["high"] = 1.0
        self.NET["high"] = 10.0

        self.totsamp = 100000

        self.oversample = 2

        self.MC = 100

        # in order to make sure that the noise realization is reproducible
        # all all concurrencies, we set the chunksize to something independent
        # of the number of ranks.

        nchunk = 2
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
        ob['baselines'] = None
        ob['noise'] = self.nse

        self.data.obs.append(ob)

        #data
        #self.nsamp = 100000
        self.stationary_period = self.totsamp
        self.lagmax = self.totsamp // 10
        #self.fsample = 4.0
        #self.times = np.arange(self.nsamp) / self.fsample
        #self.sigma = 10.
        #self.signal = np.random.randn(self.nsamp) * self.sigma
        #self.flags = np.zeros(self.nsamp, dtype=np.bool)
        #self.flags[int(self.nsamp/4):int(self.nsamp/2)] = True

    def tearDown(self):
        pass


    def test_autocov_psd(self):
        start = MPI.Wtime()

        ob = self.data.obs[0]
        tod = ob['tod']
        nse = ob['noise']

        ntod = self.totsamp

        r = 0 # noise realization
        op = OpSimNoise(realization=r)
        op.exec(self.data)

        # this replicates the calculation in sim_noise_timestream()

        fftlen = 2
        half = 1
        while fftlen <= (self.oversample * self.chunksize):
            fftlen *= 2
            half *= 2

        freqs = {}
        psds = {}
        psdnorm = {}
        todvar = {}

        for idet, det in enumerate( tod.local_dets ):
            fsamp = nse.rate(det)
            cutoff = 0.95 * (fsamp / 2.0)
            indx = np.where(nse.freq(det) > cutoff)

            NET = nse.NET(det)
            knee = nse.fknee(det)
            avg = np.mean(nse.psd(det)[indx])
            NETsq = NET*NET

            df = nse.rate(det) / float(fftlen)

            #(temp, freqs[det], psds[det]) = sim_noise_timestream(0, 0, 0, 0, idet, nse.rate(det), 0, self.chunksize, self.oversample, nse.freq(det), nse.psd(det))
            temp = sim_noise_timestream(0, 0, 0, 0, idet, nse.rate(det), 0, self.chunksize, self.oversample, nse.freq(det), nse.psd(det))

            if False:
                psdfreq = freqs[det]
                psd = psds[det]

                nn = 2
                while nn < ntod: nn *= 2
                freq = np.fft.rfftfreq( nn, 1/fsamp )
                fnn = freq.size
                psd_interp = np.interp( freq, psdfreq, psd )
                fnoisetod = np.random.randn(fnn) + 1j*np.random.randn(fnn)
                fnoisetod *= np.sqrt(psd_interp * fsamp) * np.sqrt(nn) / np.sqrt(2)
                noisetod = np.fft.irfft( fnoisetod )[:ntod]

                #noisetod[::2] = 1
                #noisetod[1::2] = -1
                #print(noisetod[:100])
            else:
                noisetod = tod.cache.reference("noise_{}".format(det))

            noisetod2 = noisetod.copy()
            for i in range(1,noisetod.size):
                noisetod[i] = .999*( noisetod[i-1] + noisetod2[i] - noisetod2[i-1] )

            autocovs = autocov_psd(np.arange(ntod)/fsamp, noisetod, np.zeros(ntod,dtype=np.bool), self.lagmax, self.stationary_period, fsamp, comm=self.comm)
            #autocovs = autocov_psd(np.arange(ntod)/fsamp, noisetod, np.zeros(ntod,dtype=np.bool), 10, self.stationary_period, fsamp, comm=self.comm)

            if self.comm.rank == 0:
                import matplotlib.pyplot as plt

                nn = 2
                while nn*2 < noisetod.size: nn *= 2
                fnoise = np.abs( np.fft.rfft( noisetod[:nn] ) )**2 / nn / fsamp
                ffreq = np.fft.rfftfreq( nn, 1/fsamp )

                nbin = 300
                fnoisebin, hits = log_bin( fnoise, nbin=nbin )
                ffreqbin, hits = log_bin( ffreq, nbin=nbin )
                fnoisebin = fnoisebin[ hits != 0 ]
                ffreqbin = ffreqbin[ hits != 0 ]

                fig = plt.figure(figsize=(12,8), dpi=72)
                ax = fig.add_subplot(1, 1, 1, aspect='auto')
                for i in range(len(autocovs)):
                    t0, t1, freq, psd = autocovs[i]
                    bfreq, hits = log_bin( freq, nbin=nbin )
                    bpsd, hits = log_bin( psd, nbin=nbin )
                    ax.loglog( freq, psd, '.', color='magenta', label='autocov PSD' )
                    ax.loglog( bfreq, bpsd, '-', color='red', label='autocov PSD (binned)' )
                #ax.loglog( ffreq, fnoise, '.', color='green', label='FFT of the noise' )
                ax.loglog( ffreqbin, fnoisebin, '.', color='green', label='FFT of the noise' )
                #ax.loglog(freqs[det], psds[det], marker='+', c="blue", label='{}: rate={:0.1f} NET={:0.1f} fknee={:0.4f}, fmin={:0.4f}'.format(det, self.rates[det], self.NET[det], self.fknee[det], self.fmin[det]))
                ax.loglog(nse.freq(det), nse.psd(det), '-b', lw=2, label='{}: rate={:0.1f} NET={:0.1f} fknee={:0.4f}, fmin={:0.4f}'.format(det, self.rates[det], self.NET[det], self.fknee[det], self.fmin[det]))
                cur_ylim = ax.get_ylim()
                ax.set_xlim([1e-5, fsamp/2])
                ax.set_ylim([0.001*(nse.NET(det)**2), 10.0*cur_ylim[1]])
                ax.legend(loc=1)
                plt.title("Simulated PSD from toast.AnalyticNoise")

                savefile = os.path.join(self.outdir, "out_test_psd_math_rawpsd_{}.png".format(det))
                plt.savefig(savefile)
                plt.close()

            del noisetod

        """
        autocovs = autocov_psd(self.times, self.signal, self.flags, self.lagmax, self.stationary_period, self.fsample, comm=self.comm)

        for i in range(len(autocovs)):
            t0, t1, freq, psd = autocovs[i]

            n = len(psd)
            mn = np.mean( np.abs( psd ) )
            err = np.std( np.abs( psd ) )

            ref = self.sigma**2 / self.fsample
            if np.abs(mn - ref) > err / np.sqrt(n) * 4.:
                raise RuntimeError('White noise input failed to produce a properly normalized white noise spectrum')
        """
        return


def log_bin( data, nbin=100 ):

    # Take a regularly sampled, ascending vector of values and bin it to
    # logaritmically narrowing bins

    # To get the bin positions, you must call log_bin twice: first with x and then y vectors
    n = len(data)

    ind = np.arange(n)+1

    bins = np.logspace(
        np.log(ind[0]), np.log(ind[-1]), num=nbin+1, endpoint=True, base=np.e
        )
    bins[-1] *= 1.01 # Widen the last bin not to have a bin with one entry

    locs = np.digitize(ind, bins)

    hits = np.zeros( nbin+2, dtype=np.int )
    binned = np.zeros( nbin+2, dtype=data.dtype )

    for i, ibin in enumerate(locs):
        hits[ibin] += 1
        binned[ibin] += data[i]

    ind = hits > 0
    binned[ind] /= hits[ind]

    return binned[ind], hits[ind]


