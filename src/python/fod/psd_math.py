# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from .. import timing as timing
from ..ctoast import fod_autosums, fod_crosssums
from ..mpi import MPI


def autocov_psd(times, signal, flags, lagmax, stationary_period, fsample,
                comm=None):
    """
    Compute the sample autocovariance function and Fourier transform it
    for a power spectral density. The resulting power spectral densities
    are distributed across the communicator as tuples of
    (start_time, stop_time, bin_frequency, bin_value)

    Args:
        times (float):  Signal time stamps.
        signal (float):  Regularly sampled signal vector.
        flags (float):  Signal quality flags.
        lagmax (int):  Largest sample separation to evaluate.
        stationary_period (float):  Length of a stationary interval in
            units of the times vector.
        fsample (float):  The sampling frequency in Hz
    """
    return crosscov_psd(times, signal, None, flags, lagmax, stationary_period,
                        fsample, comm)


def crosscov_psd(times, signal1, signal2, flags, lagmax, stationary_period,
                 fsample, comm=None):
    """
    Compute the sample (cross)covariance function and Fourier transform it
    for a power spectral density. The resulting power spectral densities
    are distributed across the communicator as tuples of
    (start_time, stop_time, bin_frequency, bin_value)

    Args:
        times (float):  Signal time stamps.
        signal1 (float):  Regularly sampled signal vector.
        signal2 (float):  Regularly sampled signal vector or None.
        flags (float):  Signal quality flags.
        lagmax (int):  Largest sample separation to evaluate.
        stationary_period (float):  Length of a stationary interval in
            units of the times vector.
        fsample (float):  The sampling frequency in Hz
    """
    autotimer = timing.auto_timer()
    if comm is None:
        rank = 0
        ntask = 1
        comm = MPI.COMM_SELF
    else:
        rank = comm.rank
        ntask = comm.size

    time_start = comm.bcast(times[0], root=0)
    time_stop = comm.bcast(times[-1], root=ntask - 1)

    nreal = np.int(np.ceil((time_stop - time_start) / stationary_period))

    # Communicate lagmax samples from the beginning of the array
    # backwards in the MPI communicator

    nsamp = signal1.size

    if lagmax > nsamp:
        raise RuntimeError(
            'crosscov_psd: Communicating TOD beyond nearest neighbors is not '
            'implemented. Reduce lagmax or the size of the MPI communicator.')

    if rank != ntask - 1:
        nextend = lagmax
    else:
        nextend = 0

    extended_signal1 = np.zeros(nsamp + nextend, dtype=np.float64)
    if signal2 is not None:
        extended_signal2 = np.zeros(nsamp + nextend, dtype=np.float64)
    extended_flags = np.zeros(nsamp + nextend, dtype=np.bool)
    extended_times = np.zeros(nsamp + nextend, dtype=times.dtype)

    extended_signal1[:nsamp] = signal1
    if signal2 is not None:
        extended_signal2[:nsamp] = signal2
    extended_flags[:nsamp] = flags
    extended_times[:nsamp] = times

    for evenodd in range(2):
        if rank % 2 == evenodd % 2:
            # Send
            if rank == 0:
                continue
            comm.send(signal1[:lagmax], dest=rank - 1, tag=0)
            if signal2 is not None:
                comm.send(signal1[:lagmax], dest=rank - 1, tag=3)
            comm.send(flags[:lagmax], dest=rank - 1, tag=1)
            comm.send(times[:lagmax], dest=rank - 1, tag=2)
        else:
            # Receive
            if rank == ntask - 1:
                continue
            extended_signal1[-lagmax:] = comm.recv(source=rank + 1, tag=0)
            if signal2 is not None:
                extended_signal1[-lagmax:] = comm.recv(source=rank + 1, tag=3)
            extended_flags[-lagmax:] = comm.recv(source=rank + 1, tag=1)
            extended_times[-lagmax:] = comm.recv(source=rank + 1, tag=2)

    realization = ((extended_times - time_start)
                   / stationary_period).astype(np.int64)

    # Set flagged elements to zero

    extended_signal1[extended_flags != 0] = 0
    if signal2 is not None:
        extended_signal2[extended_flags != 0] = 0

    covs = {}

    for ireal in range(realization[0], realization[-1] + 1):

        # Evaluate the covariance

        realflg = realization == ireal

        good = extended_flags[realflg] == 0
        ngood = np.sum(good)
        if ngood == 0:
            continue

        sig1 = extended_signal1[realflg].copy()
        sig1[good] -= np.mean(sig1[good])
        if signal2 is None:
            (cov, cov_hits) = fod_autosums(sig1, good.astype(np.int8), lagmax)
        else:
            sig2 = extended_signal2[realflg].copy()
            sig2[good] -= np.mean(sig2[good])
            (cov, cov_hits) = fod_crosssums(sig1, sig2, good.astype(np.int8),
                                            lagmax)

        covs[ireal] = (cov_hits, cov)

    # Collect the estimated covariance functions

    my_covs = {}
    nreal_task = np.int(np.ceil(nreal / ntask))

    for ireal in range(nreal):

        owner = ireal // nreal_task

        if ireal in covs:
            cov_hits, cov = covs[ireal]
        else:
            cov_hits = np.zeros(lagmax, dtype=np.int64)
            cov = np.zeros(lagmax, dtype=np.float64)

        cov_hits_total = np.zeros(lagmax, dtype=np.int64)
        cov_total = np.zeros(lagmax, dtype=np.float64)

        comm.Reduce(cov_hits, cov_hits_total, op=MPI.SUM, root=owner)
        comm.Reduce(cov, cov_total, op=MPI.SUM, root=owner)

        if rank == owner:
            my_covs[ireal] = (cov_hits_total, cov_total)

    # Now process the ones this task owns

    my_psds = []

    for ireal in my_covs.keys():

        cov_hits, cov = my_covs[ireal]

        good = cov_hits != 0
        cov[good] /= cov_hits[good]

        # Interpolate any empty bins

        if not np.all(good) and np.any(good):
            bad = cov_hits == 0
            # The last bins should be left empty
            i = cov.size - 1
            while cov_hits[i] == 0:
                cov[i] = 0
                bad[i] = False
                i -= 1
            nbad = np.sum(bad)
            if nbad > 0:
                good = np.logical_not(bad)
                lag = np.arange(lagmax)
                cov[bad] = np.interp(lag[bad], lag[good], cov[good])

        # Fourier transform for the PSD.  We symmetrize the sample
        # autocovariance so that the FFT is real-valued.  Notice that
        # we are not forcing the PSD to be positive:  each bin is a
        # noisy estimate of the true PSD.

        cov = np.hstack([cov, cov[:0:-1]])

        psd = np.fft.rfft(cov).real
        psdfreq = np.fft.rfftfreq(len(cov), d=1 / fsample)

        # Set the white noise PSD normalization to sigma**2 / fsample
        psd /= fsample

        tstart = time_start + ireal * stationary_period
        tstop = min(tstart + stationary_period, time_stop)

        my_psds.append((tstart, tstop, psdfreq, psd))

    return my_psds
