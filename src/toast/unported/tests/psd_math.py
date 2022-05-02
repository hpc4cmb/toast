# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from ..fod import autocov_psd
from ..tod import AnalyticNoise, OpSimNoise
from ..todmap import TODHpixSpiral
from ._helpers import (
    boresight_focalplane,
    create_distdata,
    create_outdir,
    uniform_chunks,
)
from .mpi import MPITestCase

# FIXME:  This seems like a useful generic function- maybe move it into
# toast.fod.psd_math?


def log_bin(data, nbin=100):

    # Take a regularly sampled, ascending vector of values and bin it to
    # logaritmically narrowing bins

    # To get the bin positions, you must call log_bin twice: first with x and then
    # y vectors
    n = len(data)

    ind = np.arange(n) + 1

    bins = np.logspace(
        np.log(ind[0]), np.log(ind[-1]), num=nbin + 1, endpoint=True, base=np.e
    )
    bins[-1] *= 1.01  # Widen the last bin not to have a bin with one entry

    locs = np.digitize(ind, bins)

    hits = np.zeros(nbin + 2, dtype=np.int)
    binned = np.zeros(nbin + 2, dtype=data.dtype)

    for i, ibin in enumerate(locs):
        hits[ibin] += 1
        binned[ibin] += data[i]

    ind = hits > 0
    binned[ind] /= hits[ind]

    return binned[ind], hits[ind]


class PSDTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        # Create one observation per group, and each observation will have
        # one detector per process and a single chunk.  Data within an
        # observation is distributed by detector.

        self.data = create_distdata(self.comm, obs_per_group=1)

        self.ndet = 4
        self.rate = 20.0

        # Create detectors with default properties
        (
            dnames,
            dquat,
            depsilon,
            drate,
            dnet,
            dfmin,
            dfknee,
            dalpha,
        ) = boresight_focalplane(self.ndet)

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

        # Total samples in one observation
        self.totsamp = 10000

        # Chunks - one per process
        chunks = uniform_chunks(self.totsamp, nchunk=self.data.comm.group_size)

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

    def tearDown(self):
        pass

    def test_autocov_psd(self):
        # Simulate noise TODs
        op = OpSimNoise()
        op.exec(self.data)

        # Compute the noise covariance
        for ob in self.data.obs:
            tod = ob["tod"]
            nse = ob["noise"]
            oid = ob["id"]

            ntod = tod.local_samples[1]
            lagmax = ntod // 10

            for det in tod.local_dets:
                noisetod = tod.cache.reference("noise_{}".format(det))

                autocovs = autocov_psd(
                    np.arange(ntod) / self.rate,
                    noisetod,
                    np.zeros(ntod, dtype=np.bool),
                    lagmax,
                    ntod,
                    self.rate,
                    comm=tod.mpicomm,
                )

                if (tod.mpicomm is None) or (tod.mpicomm.rank == 0):
                    # Plot the results for the single stationary interval
                    # assigned to one process.
                    import matplotlib.pyplot as plt

                    nn = 2
                    while nn * 2 < ntod:
                        nn *= 2

                    fnoise = np.abs(np.fft.rfft(noisetod[:nn])) ** 2 / nn / self.rate

                    ffreq = np.fft.rfftfreq(nn, 1.0 / self.rate)

                    nbin = 300
                    fnoisebin, hits = log_bin(fnoise, nbin=nbin)
                    ffreqbin, hits = log_bin(ffreq, nbin=nbin)

                    fnoisebin = fnoisebin[hits != 0]
                    ffreqbin = ffreqbin[hits != 0]

                    fig = plt.figure(figsize=(12, 8), dpi=72)
                    ax = fig.add_subplot(1, 1, 1, aspect="auto")

                    for i in range(len(autocovs)):
                        t0, t1, freq, psd = autocovs[i]
                        bfreq, hits = log_bin(freq, nbin=nbin)
                        bpsd, hits = log_bin(psd, nbin=nbin)
                        ax.loglog(freq, psd, ".", color="magenta", label="autocov PSD")
                        ax.loglog(
                            bfreq, bpsd, "-", color="red", label="autocov PSD (binned)"
                        )

                    ax.loglog(
                        ffreqbin,
                        fnoisebin,
                        ".",
                        color="green",
                        label="FFT of the noise",
                    )

                    ax.loglog(
                        nse.freq(det),
                        nse.psd(det),
                        "-b",
                        lw=2,
                        label="{}: rate={:0.1f} NET={:0.1f} fknee={:0.4f},"
                        " fmin={:0.4f}".format(
                            det,
                            nse.rate(det),
                            nse.NET(det),
                            nse.fknee(det),
                            nse.fmin(det),
                        ),
                    )

                    cur_ylim = ax.get_ylim()
                    ax.set_xlim([1e-5, self.rate / 2])
                    ax.set_ylim([0.001 * (nse.NET(det) ** 2), 10.0 * cur_ylim[1]])
                    ax.legend(loc=1)
                    plt.title("Simulated PSD from toast.AnalyticNoise")

                    savefile = os.path.join(
                        self.outdir,
                        "out_test_psd_math_rawpsd_{}_{}.png".format(oid, det),
                    )
                    plt.savefig(savefile)
                    plt.close()

                del noisetod

        return
