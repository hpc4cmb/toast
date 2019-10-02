# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from .mpi import MPITestCase

from ..tod import Noise, sim_noise_timestream, AnalyticNoise, OpSimNoise
from ..todmap import TODHpixSpiral

from .. import rng as rng

from ._helpers import (
    create_outdir,
    create_distdata,
    boresight_focalplane,
    uniform_chunks,
)


class OpSimNoiseTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        # Create one observation per group, and each observation will have
        # a fixed number of detectors and one chunk per process.

        # We create two data sets- one for testing uncorrelated noise and
        # one for testing correlated noise.

        self.data = create_distdata(self.comm, obs_per_group=1)
        self.data_corr = create_distdata(self.comm, obs_per_group=1)

        self.ndet = 4
        self.rate = 20.0

        # Create detectors with a range of knee frequencies.
        (
            dnames,
            dquat,
            depsilon,
            drate,
            dnet,
            dfmin,
            dfknee,
            dalpha,
        ) = boresight_focalplane(
            self.ndet,
            samplerate=self.rate,
            net=10.0,
            fmin=1.0e-5,
            fknee=np.linspace(0.0, 0.1, num=self.ndet),
        )

        # Total samples per observation
        self.totsamp = 200000

        # Chunks
        chunks = uniform_chunks(self.totsamp, nchunk=self.data.comm.group_size)

        # Noise sim oversampling
        self.oversample = 2

        # MCs for testing statistics of simulated noise
        self.nmc = 100

        # Populate the observations (one per group)

        tod = TODHpixSpiral(
            self.data.comm.comm_group,
            dquat,
            self.totsamp,
            detranks=1,
            firsttime=0.0,
            rate=self.rate,
            nside=512,
            sampsizes=chunks,
        )

        # Construct an uncorrelated analytic noise model for the detectors

        nse = AnalyticNoise(
            rate=drate,
            fmin=dfmin,
            detectors=dnames,
            fknee=dfknee,
            alpha=dalpha,
            NET=dnet,
        )

        self.data.obs[0]["tod"] = tod
        self.data.obs[0]["noise"] = nse

        # Construct a correlated analytic noise model for the detectors

        corr_freqs = {
            "noise_{}".format(x): nse.freq(dnames[x]) for x in range(self.ndet)
        }

        corr_psds = {"noise_{}".format(x): nse.psd(dnames[x]) for x in range(self.ndet)}

        corr_indices = {"noise_{}".format(x): 100 + x for x in range(self.ndet)}

        corr_mix = dict()
        for x in range(self.ndet):
            dmix = np.random.uniform(low=-1.0, high=1.0, size=self.ndet)
            corr_mix[dnames[x]] = {
                "noise_{}".format(y): dmix[y] for y in range(self.ndet)
            }

        nse_corr = Noise(
            detectors=dnames,
            freqs=corr_freqs,
            psds=corr_psds,
            mixmatrix=corr_mix,
            indices=corr_indices,
        )

        self.data_corr.obs[0]["tod"] = tod
        self.data_corr.obs[0]["noise"] = nse_corr

        return

    def test_gauss(self):
        # Test that the same samples from different calls are reproducible.
        # All processes run this identical test.

        detindx = self.ndet - 1
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
        data1 = rng.random(
            1000, sampler="gaussian", key=(key1, key2), counter=(counter1, counter2)
        )
        # print(data1[compoff:compoff+ncomp])

        counter2 = 100
        data2 = rng.random(
            1000, sampler="gaussian", key=(key1, key2), counter=(counter1, counter2)
        )
        # print(data2[compoff-100:compoff-100+ncomp])

        counter2 = 300
        data3 = rng.random(
            1000, sampler="gaussian", key=(key1, key2), counter=(counter1, counter2)
        )
        # print(data3[compoff-300:compoff-300+ncomp])

        np.testing.assert_array_almost_equal(
            data1[compoff : compoff + ncomp],
            data2[compoff - 100 : compoff - 100 + ncomp],
        )
        np.testing.assert_array_almost_equal(
            data1[compoff : compoff + ncomp],
            data3[compoff - 300 : compoff - 300 + ncomp],
        )
        return

    def test_sim(self):
        # Test the uncorrelated noise generation.

        # Verify that the white noise part of the spectrum is normalized
        # correctly.

        # We have purposely distributed the TOD data so that every process has
        # a single stationary interval for all detectors.

        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        for ob in self.data.obs:
            tod = ob["tod"]
            nse = ob["noise"]

            for det in tod.local_dets:
                fsamp = nse.rate(det)
                cutoff = 0.95 * (fsamp / 2.0)
                indx = np.where(nse.freq(det) > cutoff)

                net = nse.NET(det)
                avg = np.mean(nse.psd(det)[indx])
                netsq = net * net
                # print("det {} NETsq = {}, average white noise level = {}"
                #      "".format(det, netsq, avg))
                self.assertTrue((np.absolute(avg - netsq) / netsq) < 0.02)

            if rank == 0:
                # One process dumps debugging info
                import matplotlib.pyplot as plt

                for det in tod.local_dets:
                    savefile = os.path.join(
                        self.outdir, "out_test_simnoise_rawpsd_{}.txt".format(det)
                    )
                    np.savetxt(
                        savefile,
                        np.transpose([nse.freq(det), nse.psd(det)]),
                        delimiter=" ",
                    )

                    fig = plt.figure(figsize=(12, 8), dpi=72)
                    ax = fig.add_subplot(1, 1, 1, aspect="auto")
                    ax.loglog(
                        nse.freq(det),
                        nse.psd(det),
                        marker="o",
                        c="red",
                        label="{}: rate={:0.1f} NET={:0.1f} fknee={:0.4f}, "
                        "fmin={:0.4f}".format(
                            det,
                            nse.rate(det),
                            nse.NET(det),
                            nse.fknee(det),
                            nse.fmin(det),
                        ),
                    )

                    cur_ylim = ax.get_ylim()
                    ax.set_ylim([0.001 * (nse.NET(det) ** 2), 10.0 * cur_ylim[1]])
                    ax.legend(loc=1)
                    plt.title("Simulated PSD from toast.AnalyticNoise")

                    savefile = os.path.join(
                        self.outdir, "out_test_simnoise_rawpsd_{}.png".format(det)
                    )
                    plt.savefig(savefile)
                    plt.close()

            ntod = tod.local_samples[1]

            # this replicates the calculation in sim_noise_timestream()

            fftlen = 2
            while fftlen <= (self.oversample * ntod):
                fftlen *= 2

            freqs = {}
            psds = {}
            psdnorm = {}
            todvar = {}

            cfftlen = 2
            while cfftlen <= ntod:
                cfftlen *= 2

            # print("fftlen = ", fftlen)
            # print("cfftlen = ", cfftlen)

            checkpsd = {}
            binsamps = cfftlen // 4096
            nbins = binsamps - 1
            bstart = (self.rate / 2) / nbins
            bins = np.linspace(bstart, self.rate / 2, num=(nbins - 1), endpoint=True)
            # print("nbins = ",nbins)
            # print(bins)

            checkfreq = np.fft.rfftfreq(cfftlen, d=1 / self.rate)
            # print("checkfreq len = ",len(checkfreq))
            # print(checkfreq[:10])
            # print(checkfreq[-10:])
            checkbinmap = np.searchsorted(bins, checkfreq, side="left")
            # print("checkbinmap len = ",len(checkbinmap))
            # print(checkbinmap[:10])
            # print(checkbinmap[-10:])
            bcount = np.bincount(checkbinmap)
            # print("bcount len = ",len(bcount))
            # print(bcount)

            bintruth = {}

            idet = 0
            for det in tod.local_dets:

                dfreq = nse.rate(det) / float(fftlen)

                (pytod, freqs[det], psds[det]) = sim_noise_timestream(
                    0,
                    0,
                    0,
                    0,
                    idet,
                    nse.rate(det),
                    0,
                    ntod,
                    self.oversample,
                    nse.freq(det),
                    nse.psd(det),
                    py=True,
                )

                libtod = sim_noise_timestream(
                    0,
                    0,
                    0,
                    0,
                    idet,
                    nse.rate(det),
                    0,
                    ntod,
                    self.oversample,
                    nse.freq(det),
                    nse.psd(det),
                    py=False,
                )

                np.testing.assert_array_almost_equal(pytod, libtod, decimal=2)

                # Factor of 2 comes from the negative frequency values.
                psdnorm[det] = 2.0 * np.sum(psds[det] * dfreq)
                # print("psd[{}] integral = {}".format(det, psdnorm[det]))

                todvar[det] = np.zeros(self.nmc, dtype=np.float64)
                checkpsd[det] = np.zeros((nbins - 1, self.nmc), dtype=np.float64)

                idet += 1

            if rank == 0:
                import matplotlib.pyplot as plt

                for det in tod.local_dets:
                    savefile = os.path.join(
                        self.outdir, "out_test_simnoise_psd_{}.txt".format(det)
                    )
                    np.savetxt(
                        savefile, np.transpose([freqs[det], psds[det]]), delimiter=" "
                    )

                    fig = plt.figure(figsize=(12, 8), dpi=72)
                    ax = fig.add_subplot(1, 1, 1, aspect="auto")
                    ax.loglog(
                        freqs[det],
                        psds[det],
                        marker="+",
                        c="blue",
                        label="{}: rate={:0.1f} NET={:0.1f} fknee={:0.4f}, "
                        "fmin={:0.4f}".format(
                            det,
                            nse.rate(det),
                            nse.NET(det),
                            nse.fknee(det),
                            nse.fmin(det),
                        ),
                    )

                    cur_ylim = ax.get_ylim()
                    ax.set_ylim([0.001 * (nse.NET(det) ** 2), 10.0 * cur_ylim[1]])
                    ax.legend(loc=1)
                    plt.title(
                        "Interpolated PSD with High-pass from {:0.1f} "
                        "second Simulation Interval".format((float(ntod) / self.rate))
                    )

                    savefile = os.path.join(
                        self.outdir, "out_test_simnoise_psd_{}.png".format(det)
                    )
                    plt.savefig(savefile)
                    plt.close()

                    tmap = np.searchsorted(bins, freqs[det], side="left")
                    tcount = np.bincount(tmap)
                    tpsd = np.bincount(tmap, weights=psds[det])
                    good = tcount > 0
                    tpsd[good] /= tcount[good]
                    bintruth[det] = tpsd

            hpy = None
            if rank == 0:
                if "TOAST_TEST_BIGTOD" in os.environ.keys():
                    try:
                        import h5py as hpy
                    except ImportError:
                        # just write the first realization as usual
                        hpy = None

            # if we have the h5py module and a special environment variable is set,
            # then process zero will dump out its full timestream data for more
            # extensive sharing / tests.  Just dump a few detectors to keep the
            # file size reasonable.

            hfile = None
            dset = {}
            if hpy is not None:
                hfile = hpy.File(
                    os.path.join(self.outdir, "out_test_simnoise_tod.hdf5"), "w"
                )
                for det in tod.detectors:
                    dset[det] = hfile.create_dataset(
                        det, (self.nmc, ntod), dtype="float64"
                    )

            # Run both the numpy FFT case and the toast FFT case.

            for realization in range(self.nmc):

                # generate timestreams

                opnoise = OpSimNoise(realization=realization)
                opnoise.exec(self.data)

                if realization == 0:
                    # write timestreams to disk for debugging

                    if rank == 0:
                        import matplotlib.pyplot as plt

                        for det in tod.local_dets:

                            check = tod.cache.reference("noise_{}".format(det))

                            savefile = os.path.join(
                                self.outdir,
                                "out_test_simnoise_tod_mc0_{}.txt" "".format(det),
                            )
                            np.savetxt(savefile, np.transpose([check]), delimiter=" ")

                            fig = plt.figure(figsize=(12, 8), dpi=72)
                            ax = fig.add_subplot(1, 1, 1, aspect="auto")
                            ax.plot(
                                np.arange(len(check)),
                                check,
                                c="black",
                                label="Det {}".format(det),
                            )
                            ax.legend(loc=1)
                            plt.title(
                                "First Realization of Simulated TOD "
                                "from toast.sim_noise_timestream()"
                            )

                            savefile = os.path.join(
                                self.outdir,
                                "out_test_simnoise_tod_mc0_{}.png" "".format(det),
                            )
                            plt.savefig(savefile)
                            plt.close()

                for det in tod.local_dets:
                    # compute the TOD variance
                    ref = tod.cache.reference("noise_{}".format(det))
                    dclevel = np.mean(ref)
                    variance = np.vdot(ref - dclevel, ref - dclevel) / ntod
                    todvar[det][realization] = variance

                    if hfile is not None:
                        if det in dset:
                            dset[det][realization, :] = ref[:]

                    # compute the PSD
                    buffer = np.zeros(cfftlen, dtype=np.float64)
                    offset = (cfftlen - len(ref)) // 2
                    buffer[offset : offset + len(ref)] = ref
                    rawpsd = np.fft.rfft(buffer)
                    norm = 1.0 / (self.rate * ntod)
                    rawpsd = norm * np.abs(rawpsd ** 2)
                    bpsd = np.bincount(checkbinmap, weights=rawpsd)
                    good = bcount > 0
                    bpsd[good] /= bcount[good]
                    checkpsd[det][:, realization] = bpsd[:]

                tod.cache.clear()

            if hfile is not None:
                hfile.close()

            if rank == 0:
                np.savetxt(
                    os.path.join(self.outdir, "out_test_simnoise_tod_var.txt"),
                    np.transpose([todvar[x] for x in tod.local_dets]),
                    delimiter=" ",
                )

            if rank == 0:
                import matplotlib.pyplot as plt

                for det in tod.local_dets:
                    savefile = os.path.join(
                        self.outdir, "out_test_simnoise_tod_var_{}.txt".format(det)
                    )
                    np.savetxt(savefile, np.transpose([todvar[det]]), delimiter=" ")

                    sig = np.mean(todvar[det]) * np.sqrt(2.0 / (ntod - 1))
                    histrange = 5.0 * sig
                    histmin = psdnorm[det] - histrange
                    histmax = psdnorm[det] + histrange

                    fig = plt.figure(figsize=(12, 8), dpi=72)

                    ax = fig.add_subplot(1, 1, 1, aspect="auto")
                    plt.hist(
                        todvar[det],
                        10,
                        range=(histmin, histmax),
                        facecolor="magenta",
                        alpha=0.75,
                        label="{}:  PSD integral = {:0.1f} expected sigma = "
                        "{:0.1f}".format(det, psdnorm[det], sig),
                    )
                    ax.legend(loc=1)
                    plt.title(
                        "Distribution of TOD Variance for {} "
                        "Realizations".format(self.nmc)
                    )

                    savefile = os.path.join(
                        self.outdir, "out_test_simnoise_tod_var_{}.png".format(det)
                    )
                    plt.savefig(savefile)
                    plt.close()

                    meanpsd = np.asarray(
                        [np.mean(checkpsd[det][x, :]) for x in range(nbins - 1)]
                    )

                    fig = plt.figure(figsize=(12, 8), dpi=72)

                    ax = fig.add_subplot(1, 1, 1, aspect="auto")
                    ax.plot(bins, bintruth[det], c="k", label="Input Truth")
                    ax.plot(bins, meanpsd, c="b", marker="o", label="Mean Binned PSD")
                    ax.scatter(
                        np.repeat(bins, self.nmc),
                        checkpsd[det].flatten(),
                        marker="x",
                        color="r",
                        label="Binned PSD",
                    )
                    # ax.set_xscale("log")
                    # ax.set_yscale("log")
                    ax.legend(loc=1)
                    plt.title(
                        "Detector {} Binned PSDs for {} Realizations"
                        "".format(det, self.nmc)
                    )

                    savefile = os.path.join(
                        self.outdir, "out_test_simnoise_binpsd_dist_{}.png".format(det)
                    )
                    plt.savefig(savefile)
                    plt.close()

                    # The data will likely not be gaussian distributed.
                    # Just check that the mean is "close enough" to the truth.
                    errest = np.absolute(np.mean((meanpsd - tpsd) / tpsd))
                    # print("Det {} avg rel error = {}".format(det, errest), flush=True)
                    if nse.fknee(det) < 0.1:
                        self.assertTrue(errest < 0.1)

            # Verify that Parseval's theorem holds- that the variance of
            # the TOD equals the integral of the PSD.  We do this for an
            # ensemble of realizations
            #
            # and compare the TOD variance to the integral of the PSD
            # accounting for the error on the variance due to finite
            # numbers of samples.
            #

            for det in tod.local_dets:
                sig = np.mean(todvar[det]) * np.sqrt(2.0 / (ntod - 1))
                over3sig = np.where(
                    np.absolute(todvar[det] - psdnorm[det]) > 3.0 * sig
                )[0]
                overfrac = float(len(over3sig)) / self.nmc
                # print(det, " : ", overfrac, flush=True)
                if nse.fknee(det) < 0.01:
                    self.assertTrue(overfrac < 0.1)
        return

    def test_sim_correlated(self):
        # Test the correlated noise generation.
        opnoise = OpSimNoise(realization=0)
        opnoise.exec(self.data_corr)

        total = None

        for ob in self.data.obs:
            tod = ob["tod"]
            for det in tod.local_dets:
                # compute the TOD variance
                ref = tod.cache.reference("noise_{}".format(det))
                self.assertTrue(np.std(ref) > 0)
                if total is None:
                    total = ref.copy()
                else:
                    total[:] += ref
                del ref

        # np.testing.assert_almost_equal(np.std(total), 0)
        return
