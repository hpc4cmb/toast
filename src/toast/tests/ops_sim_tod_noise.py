# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import sys

import numpy as np
from astropy import units as u

from .. import ops as ops
from .. import rng as rng
from ..noise import Noise
from ..ops.sim_tod_noise import sim_noise_timestream
from ..vis import set_matplotlib_backend
from ._helpers import close_data, create_outdir, create_satellite_data
from .mpi import MPITestCase


class SimNoiseTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.oversample = 2
        self.nmc = 100

    def test_gauss(self):
        # Test that the same samples from different calls are reproducible.
        # All processes run this identical test.

        detindx = 99
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

    def test_sim_once(self):
        # Test the uncorrelated noise generation.
        # Verify that the white noise part of the spectrum is normalized
        # correctly.

        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # This is a simulation with the same focalplane for every obs...
        sample_rate = data.obs[0].telescope.focalplane.sample_rate

        # Create a noise model from focalplane detector properties
        noise_model = ops.DefaultNoiseModel()
        noise_model.apply(data)

        # Simulate data in parallel
        sim_noise = ops.SimNoise()
        sim_noise.serial = False
        sim_noise.det_data = "noise_batch"
        sim_noise.apply(data)

        # Simulate noise serially and compare results
        sim_noise.serial = True
        sim_noise.det_data = "noise"
        sim_noise.apply(data)

        for ob in data.obs:
            for det in ob.local_detectors:
                np.testing.assert_array_almost_equal(
                    ob.detdata["noise_batch"][det], ob.detdata["noise"][det], decimal=6
                )

        wrank = data.comm.world_rank
        grank = data.comm.group_rank

        for ob in data.obs:
            nse = ob[noise_model.noise_model]
            for det in ob.local_detectors:
                # Verify that the white noise level of the PSD is correctly normalized.
                # Only check the high frequency part of the spectrum to avoid 1/f.
                fsamp = nse.rate(det).to_value(u.Hz)
                cutoff = 0.95 * (fsamp / 2.0)
                indx = np.where(nse.freq(det).to_value(u.Hz) > cutoff)
                net = nse.NET(det)
                avg = np.mean(nse.psd(det)[indx])
                netsq = net * net
                self.assertTrue((np.absolute(avg - netsq) / netsq) < 0.02)

            if wrank == 0:
                set_matplotlib_backend()
                import matplotlib.pyplot as plt

                # Just one process dumps out local noise model for debugging

                for det in ob.local_detectors:
                    savefile = os.path.join(
                        self.outdir, "out_{}_rawpsd_{}.txt".format(ob.name, det)
                    )
                    np.savetxt(
                        savefile,
                        np.transpose([nse.freq(det), nse.psd(det)]),
                        delimiter=" ",
                    )

                    fig = plt.figure(figsize=(12, 8), dpi=72)
                    ax = fig.add_subplot(1, 1, 1, aspect="auto")
                    ax.loglog(
                        nse.freq(det).to_value(u.Hz),
                        nse.psd(det).to_value(u.K**2 * u.second),
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
                    ax.set_ylim(
                        [
                            0.001
                            * (nse.NET(det).to_value(u.K * np.sqrt(1 * u.second)) ** 2),
                            10.0 * cur_ylim[1],
                        ]
                    )
                    ax.legend(loc=1)
                    plt.title("Simulated PSD from toast.AnalyticNoise")

                    savefile = os.path.join(
                        self.outdir, "out_{}_rawpsd_{}.pdf".format(ob.name, det)
                    )

                    plt.savefig(savefile)
                    plt.close()

            # Now generate noise timestreams in python and compare to the results of
            # running the operator.

            freqs = dict()
            psds = dict()

            fftlen = 2
            while fftlen <= (self.oversample * ob.n_local_samples):
                fftlen *= 2

            for idet, det in enumerate(ob.local_detectors):
                dfreq = nse.rate(det).to_value(u.Hz) / float(fftlen)
                (pytod, freqs[det], psds[det]) = sim_noise_timestream(
                    realization=sim_noise.realization,
                    telescope=ob.telescope.uid,
                    component=sim_noise.component,
                    sindx=ob.session.uid,
                    detindx=nse.index(det),
                    rate=nse.rate(det).to_value(u.Hz),
                    firstsamp=ob.local_index_offset,
                    samples=ob.n_local_samples,
                    oversample=self.oversample,
                    freq=nse.freq(det).to_value(u.Hz),
                    psd=nse.psd(det).to_value(u.K**2 * u.second),
                    py=True,
                )
                np.testing.assert_array_almost_equal(
                    pytod.array(), ob.detdata[sim_noise.det_data][det], decimal=2
                )
                pytod.clear()

            if wrank == 0:
                # One process dumps out interpolated PSD for debugging
                import matplotlib.pyplot as plt

                for det in ob.local_detectors:
                    savefile = os.path.join(
                        self.outdir, "out_{}_interppsd_{}.txt".format(ob.name, det)
                    )
                    np.savetxt(
                        savefile,
                        np.transpose([freqs[det], psds[det]]),
                        delimiter=" ",
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
                    ax.set_ylim(
                        [
                            0.001
                            * (nse.NET(det).to_value(u.K * np.sqrt(1 * u.second)) ** 2),
                            10.0 * cur_ylim[1],
                        ]
                    )
                    ax.legend(loc=1)
                    plt.title(
                        "Interpolated PSD with High-pass from {:0.1f} "
                        "second Simulation Interval".format(
                            (float(ob.n_local_samples) / sample_rate)
                        )
                    )

                    savefile = os.path.join(
                        self.outdir, "out_{}_interppsd_{}.pdf".format(ob.name, det)
                    )
                    plt.savefig(savefile)
                    plt.close()

        close_data(data)

    def test_sim_mc(self):
        # Create a fake satellite data set for testing.  We explicitly generate
        # only one observation per group.
        data = create_satellite_data(
            self.comm,
            obs_per_group=1,
            sample_rate=100.0 * u.Hz,
            obs_time=10.0 * u.minute,
        )

        # This is a simulation with the same focalplane for every obs...
        sample_rate = data.obs[0].telescope.focalplane.sample_rate.to_value(u.Hz)

        # Create a noise model from focalplane detector properties
        noise_model = ops.DefaultNoiseModel()
        noise_model.apply(data)

        wrank = data.comm.world_rank

        # First we make one pass through the data and examine the noise model.
        # We interpolate the PSD using pure python code and compute some normalization
        # factors and also bin the true PSD to the final binning we will use for the
        # PSDs made from the timestreams.

        todvar = dict()
        ntod_var = None
        psd_norm = dict()
        checkpsd = dict()
        freqs = dict()
        psds = dict()

        cfftlen = 2
        while cfftlen <= data.obs[0].n_local_samples:
            cfftlen *= 2
        binsamps = cfftlen // 2048
        nbins = binsamps - 1
        bstart = (sample_rate / 2) / nbins
        bins = np.linspace(bstart, sample_rate / 2, num=(nbins - 1), endpoint=True)
        checkfreq = np.fft.rfftfreq(cfftlen, d=(1 / sample_rate))
        checkbinmap = np.searchsorted(bins, checkfreq, side="left")
        bcount = np.bincount(checkbinmap)
        bintruth = dict()
        tpsd = None
        good = None

        for ob in data.obs[:1]:
            ntod_var = ob.n_local_samples
            nse = ob[noise_model.noise_model]
            fftlen = 2
            while fftlen <= (self.oversample * ob.n_local_samples):
                fftlen *= 2
            for idet, det in enumerate(ob.local_detectors):
                dfreq = nse.rate(det).to_value(u.Hz) / float(fftlen)
                (pytod, freqs[det], psds[det]) = sim_noise_timestream(
                    realization=0,
                    telescope=ob.telescope.uid,
                    component=0,
                    sindx=ob.session.uid,
                    detindx=idet,
                    rate=nse.rate(det).to_value(u.Hz),
                    firstsamp=ob.local_index_offset,
                    samples=ob.n_local_samples,
                    oversample=self.oversample,
                    freq=nse.freq(det).to_value(u.Hz),
                    psd=nse.psd(det).to_value(u.K**2 * u.second),
                    py=True,
                )
                pytod.clear()

                # Factor of 2 comes from the negative frequency values.
                psd_norm[det] = 2.0 * np.sum(psds[det] * dfreq)

                # Allocate buffers for MC loop
                todvar[det] = np.zeros(self.nmc, dtype=np.float64)
                checkpsd[det] = np.zeros((nbins - 1, self.nmc), dtype=np.float64)

                # Bin the true high-resolution PSD.
                tmap = np.searchsorted(bins, freqs[det], side="left")
                tcount = np.bincount(tmap)
                tpsd = np.bincount(tmap, weights=psds[det])
                good = tcount > 0
                tpsd[good] /= tcount[good]
                bintruth[det] = tpsd

        # Perform noise realizations and accumulation statistics.

        for realization in range(self.nmc):
            # Simulate noise using the model, with a different realization each time
            sim_noise = ops.SimNoise(realization=realization)

            # Clear any previously generated data
            for ob in data.obs:
                if sim_noise.det_data in ob.detdata:
                    ob.detdata[sim_noise.det_data][:] = 0.0

            sim_noise.apply(data)

            if realization == 0:
                # write timestreams to disk for debugging
                if wrank == 0:
                    import matplotlib.pyplot as plt

                    for ob in data.obs:
                        for det in ob.local_detectors:
                            check = ob.detdata[sim_noise.det_data][det]

                            savefile = os.path.join(
                                self.outdir,
                                "out_{}_tod-mc0_{}.txt" "".format(ob.name, det),
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
                                "Observation {}, First Realization of {}".format(
                                    ob.name, det
                                )
                            )

                            savefile = os.path.join(
                                self.outdir,
                                "out_{}_tod-mc0_{}.pdf" "".format(ob.name, det),
                            )
                            plt.savefig(savefile)
                            plt.close()

            for ob in data.obs[:1]:
                for det in ob.local_detectors:
                    # compute the TOD variance
                    tod = ob.detdata[sim_noise.det_data][det]
                    dclevel = np.mean(tod)
                    variance = np.vdot(tod - dclevel, tod - dclevel) / len(tod)
                    todvar[det][realization] = variance

                    # compute the PSD
                    buffer = np.zeros(cfftlen, dtype=np.float64)
                    offset = (cfftlen - len(tod)) // 2
                    buffer[offset : offset + len(tod)] = tod
                    rawpsd = np.fft.rfft(buffer)
                    norm = 1.0 / (sample_rate * ob.n_local_samples)
                    rawpsd = norm * np.abs(rawpsd**2)
                    bpsd = np.bincount(checkbinmap, weights=rawpsd)
                    good = bcount > 0
                    bpsd[good] /= bcount[good]
                    checkpsd[det][:, realization] = bpsd[:]

        lds = sorted(todvar.keys())

        if wrank == 0:
            import matplotlib.pyplot as plt

            np.savetxt(
                os.path.join(self.outdir, "out_tod_variance.txt"),
                np.transpose([todvar[x] for x in lds]),
                delimiter=" ",
            )

            for det in lds:
                sig = np.mean(todvar[det]) * np.sqrt(2.0 / (ntod_var - 1))
                histrange = 5.0 * sig
                histmin = psd_norm[det] - histrange
                histmax = psd_norm[det] + histrange

                fig = plt.figure(figsize=(12, 8), dpi=72)

                ax = fig.add_subplot(1, 1, 1, aspect="auto")
                plt.hist(
                    todvar[det],
                    10,
                    range=(histmin, histmax),
                    facecolor="magenta",
                    alpha=0.75,
                    label="{}:  PSD integral = {:0.1f} expected sigma = "
                    "{:0.1f}".format(det, psd_norm[det], sig),
                )
                ax.legend(loc=1)
                plt.title(
                    "Detector {} Distribution of TOD Variance for {} "
                    "Realizations".format(det, self.nmc)
                )

                savefile = os.path.join(
                    self.outdir, "out_tod-variance_{}.pdf".format(det)
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
                    self.outdir, "out_psd-histogram_{}.pdf".format(det)
                )
                plt.savefig(savefile)
                plt.close()

                # The data will likely not be gaussian distributed.
                # Just check that the mean is "close enough" to the truth.
                errest = np.absolute(np.mean((meanpsd - tpsd) / tpsd))
                # print("Det {} avg rel error = {}".format(det, errest), flush=True)
                if nse.fknee(det).to_value(u.Hz) < 0.1:
                    self.assertTrue(errest < 0.1)

        # Verify that Parseval's theorem holds- that the variance of the TOD equals the
        # integral of the PSD.  We do this for an ensemble of realizations and compare
        # the TOD variance to the integral of the PSD accounting for the error on the
        # variance due to finite numbers of samples.

        ntod = data.obs[0].n_local_samples
        for det in lds:
            sig = np.mean(todvar[det]) * np.sqrt(2.0 / (ntod - 1))
            over3sig = np.where(np.absolute(todvar[det] - psd_norm[det]) > 3.0 * sig)[0]
            overfrac = float(len(over3sig)) / self.nmc
            # print(det, " : ", overfrac, flush=True)
            if nse.fknee(det).to_value(u.Hz) < 0.1:
                self.assertTrue(overfrac < 0.1)

        close_data(data)

    def test_sim_correlated(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        noise_model = ops.DefaultNoiseModel()
        noise_model.apply(data)

        # Construct a correlated analytic noise model for the detectors for each
        # observation.
        for ob in data.obs:
            nse = ob[noise_model.noise_model]
            corr_freqs = {
                "noise_{}".format(i): nse.freq(x)
                for i, x in enumerate(ob.local_detectors)
            }
            corr_psds = {
                "noise_{}".format(i): nse.psd(x)
                for i, x in enumerate(ob.local_detectors)
            }
            corr_indices = {
                "noise_{}".format(i): 100 + i for i, x in enumerate(ob.local_detectors)
            }
            corr_mix = dict()
            for i, x in enumerate(ob.local_detectors):
                dmix = np.random.uniform(
                    low=-1.0, high=1.0, size=len(ob.local_detectors)
                )
                corr_mix[x] = {
                    "noise_{}".format(y): dmix[y]
                    for y in range(len(ob.local_detectors))
                }
            ob["noise_model_corr"] = Noise(
                detectors=ob.local_detectors,
                freqs=corr_freqs,
                psds=corr_psds,
                mixmatrix=corr_mix,
                indices=corr_indices,
            )

        # Simulate noise using this model
        sim_noise = ops.SimNoise(noise_model="noise_model_corr")
        sim_noise.apply(data)

        total = None

        for ob in data.obs:
            for det in ob.local_detectors:
                # compute the TOD variance
                tod = ob.detdata[sim_noise.det_data][det]
                self.assertTrue(np.std(tod) > 0)
                if total is None:
                    total = np.zeros(ob.n_local_samples, dtype=np.float64)
                total[:] += tod

        # np.testing.assert_almost_equal(np.std(total), 0)
        close_data(data)
