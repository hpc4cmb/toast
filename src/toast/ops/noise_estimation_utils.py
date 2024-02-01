# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
from scipy.signal import fftconvolve

from .._libtoast import fod_autosums, fod_crosssums
from ..mpi import MPI
from ..timing import function_timer


@function_timer
def flagged_running_average(
    signal, flag, wkernel, return_flags=False, downsample=False
):
    """Compute a running average considering only the unflagged samples.

    Args:
        signal (float)
        flag (bool)
        wkernel (int):  Running average width
        return_flags (bool):  If true, also return flags which are
            a subset of the input flags.
        downsample (bool):  If True, return a downsampled version of the
            filtered timestream.

    Returns:
        (array or tuple):  The filtered signal and optionally the flags.

    """
    if len(signal) != len(flag):
        raise Exception("Signal and flag lengths do not match.")

    bad = flag != 0
    masked_signal = signal.copy()
    masked_signal[bad] = 0

    good = np.ones(len(signal), dtype=np.float64)
    good[bad] = 0

    kernel = np.ones(wkernel, dtype=np.float64)

    filtered_signal = fftconvolve(masked_signal, kernel, mode="same")
    filtered_hits = fftconvolve(good, kernel, mode="same")

    hit = filtered_hits > 0.1
    nothit = np.logical_not(hit)

    filtered_signal[hit] /= filtered_hits[hit]
    filtered_signal[nothit] = 0

    if return_flags or downsample:
        filtered_flags = np.zeros_like(flag)
        filtered_flags[nothit] = True

    if downsample:
        good = filtered_flags == 0
        if return_flags:
            filtered_flags[good][::wkernel]
        filtered_signal[good][::wkernel]

    if return_flags:
        return filtered_signal, filtered_flags
    else:
        return filtered_signal


@function_timer
def highpass_flagged_signal(sig, good, naverage):
    """Highpass-filter the signal to remove sub harmonic modes.

    Args:
        sig (array):  The signal.
        good (array):  Sample flags (zero == *BAD*)
        naverage (int):  The number of samples to average.

    Returns:
        (array):  The processed array.

    """
    ngood = np.sum(good)
    if ngood == 0:
        # No valid samples
        return np.zeros_like(sig)
    """
    De-trending disabled as unnecessary. It also changes results at
    different concurrencies.
    # First fit and remove a linear trend.  Loss of power from this
    # filter is assumed negligible in the frequency bins of interest
    templates = np.vstack([np.ones(ngood), np.arange(good.size)[good]])
    invcov = np.dot(templates, templates.T)
    cov = np.linalg.inv(invcov)
    proj = np.dot(templates, sig[good])
    coeff = np.dot(cov, proj)
    sig[good] -= coeff[0] + coeff[1] * templates[1]
    """
    # Then prewhiten the data.  This filter will be corrected in the
    # PSD estimates.
    trend = flagged_running_average(sig, good == 0, naverage)
    return sig - trend


@function_timer
def communicate_overlap(times, signal1, signal2, flags, lagmax, naverage, comm, group):
    """Send and receive TOD to have margin for filtering and lag"""

    if comm is None:
        rank = 0
        ntask = 1
    else:
        rank = comm.rank
        ntask = comm.size

    # Communicate naverage + lagmax samples between processes so that
    # running averages do not change with distributed data.

    nsamp = signal1.size

    half_average = naverage // 2 + 1
    nextend_backward = half_average
    nextend_forward = half_average + lagmax
    if rank == 0:
        nextend_backward = 0
    if rank == ntask - 1:
        nextend_forward = 0
    nextend = nextend_backward + nextend_forward

    if lagmax + half_average > nsamp and comm is not None and comm.size > 1:
        msg = (
            f"communicate_overlap: lagmax + half_average = {lagmax+half_average} and "
            f"nsample = {nsamp}.  Communicating TOD beyond nearest neighbors is not "
            f"implemented. Reduce lagmax, the size of the averaging window, or the "
            f"size of the MPI communicator."
        )
        raise RuntimeError(msg)

    extended_signal1 = np.zeros(nsamp + nextend, dtype=np.float64)
    if signal2 is None:
        extended_signal2 = None
    else:
        extended_signal2 = np.zeros(nsamp + nextend, dtype=np.float64)
    extended_flags = np.zeros(nsamp + nextend, dtype=bool)
    extended_times = np.zeros(nsamp + nextend, dtype=times.dtype)

    ind = slice(nextend_backward, nextend_backward + nsamp)
    extended_signal1[ind] = signal1
    if signal2 is not None:
        extended_signal2[ind] = signal2
    extended_flags[ind] = flags
    extended_times[ind] = times

    if comm is not None:
        for evenodd in range(2):
            if rank % 2 == evenodd % 2:
                # Tag offset is based on the sending rank
                tag = 8 * (comm.rank + group * comm.size)
                # Send to rank - 1
                if rank != 0:
                    nsend = lagmax + half_average
                    comm.send(signal1[:nsend], dest=rank - 1, tag=tag + 0)
                    if signal2 is not None:
                        comm.send(signal2[:nsend], dest=rank - 1, tag=tag + 1)
                    comm.send(flags[:nsend], dest=rank - 1, tag=tag + 2)
                    comm.send(times[:nsend], dest=rank - 1, tag=tag + 3)
                # Send to rank + 1
                if rank != ntask - 1:
                    nsend = half_average
                    comm.send(signal1[-nsend:], dest=rank + 1, tag=tag + 4)
                    if signal2 is not None:
                        comm.send(signal2[-nsend:], dest=rank + 1, tag=tag + 5)
                    comm.send(flags[-nsend:], dest=rank + 1, tag=tag + 6)
                    comm.send(times[-nsend:], dest=rank + 1, tag=tag + 7)
            else:
                # Receive from rank + 1
                if rank != ntask - 1:
                    tag = 8 * ((comm.rank + 1) + group * comm.size)
                    nrecv = lagmax + half_average
                    extended_signal1[-nrecv:] = comm.recv(source=rank + 1, tag=tag + 0)
                    if signal2 is not None:
                        extended_signal2[-nrecv:] = comm.recv(
                            source=rank + 1, tag=tag + 1
                        )
                    extended_flags[-nrecv:] = comm.recv(source=rank + 1, tag=tag + 2)
                    extended_times[-nrecv:] = comm.recv(source=rank + 1, tag=tag + 3)
                # Receive from rank - 1
                if rank != 0:
                    tag = 8 * ((comm.rank - 1) + group * comm.size)
                    nrecv = half_average
                    extended_signal1[:nrecv] = comm.recv(source=rank - 1, tag=tag + 4)
                    if signal2 is not None:
                        extended_signal2[:nrecv] = comm.recv(
                            source=rank - 1, tag=tag + 5
                        )
                    extended_flags[:nrecv] = comm.recv(source=rank - 1, tag=tag + 6)
                    extended_times[:nrecv] = comm.recv(source=rank - 1, tag=tag + 7)
            comm.barrier()

    return extended_times, extended_flags, extended_signal1, extended_signal2


@function_timer
def autocov_psd(
    times,
    extended_times,
    global_intervals,
    extended_signal,
    extended_flags,
    lagmax,
    naverage,
    stationary_period,
    fsample,
    comm=None,
    return_cov=False,
):
    """Compute the sample autocovariance.

    Compute the sample autocovariance function and Fourier transform it
    for a power spectral density. The resulting power spectral densities
    are distributed across the communicator as tuples of
    (start_time, stop_time, bin_frequency, bin_value)

    Args:
        times (float):  Signal time stamps.
        global_intervals (list):  Time stamp ranges over which to
            perform independent analysis
        signal (float):  Regularly sampled signal vector.
        flags (float):  Signal quality flags.
        lagmax (int):  Largest sample separation to evaluate.
        naverage (int):  Length of running average in highpass filter
        stationary_period (float):  Length of a stationary interval in
            units of the times vector.
        fsample (float):  The sampling frequency in Hz
        comm (MPI.Comm):  The MPI communicator or None.
        return_cov (bool): Return also the covariance function

    Returns:
        (list):  List of local tuples of (start_time, stop_time, bin_frequency,
            bin_value)

    """
    return crosscov_psd(
        times,
        extended_times,
        global_intervals,
        extended_signal,
        None,
        extended_flags,
        lagmax,
        naverage,
        stationary_period,
        fsample,
        comm,
        return_cov,
    )


@function_timer
def crosscov_psd(
    times,
    extended_times,
    global_intervals,
    extended_signal1,
    extended_signal2,
    extended_flags,
    lagmax,
    naverage,
    stationary_period,
    fsample,
    comm=None,
    return_cov=False,
    symmetric=False,
):
    """Compute the sample (cross)covariance.

    Compute the sample (cross)covariance function and Fourier transform it
    for a power spectral density. The resulting power spectral densities
    are distributed across the process grid row communicator as tuples of
    (start_time, stop_time, bin_frequency, bin_value)

    Args:
        times (float):  Signal time stamps.
        global_intervals (list):  Time stamp ranges over which to
            perform independent analysis
        signal1 (float):  Regularly sampled signal vector.
        signal2 (float):  Regularly sampled signal vector or None.
        flags (float):  Signal quality flags.
        lagmax (int):  Largest sample separation to evaluate.
        naverage (int):  Length of running average in highpass filter
        stationary_period (float):  Length of a stationary interval in
            units of the times vector.
        fsample (float):  The sampling frequency in Hz
        comm (MPI.Comm):  The MPI communicator or None.
        return_cov (bool):  Return also the covariance function
        symmetric (bool):  Treat positive and negative lags as equivalent

    Returns:
        (list):  List of local tuples of (start_time, stop_time, bin_frequency,
            bin_value)

    """
    if comm is None:
        rank = 0
        ntask = 1
        time_start = extended_times[0]
        time_stop = extended_times[-1]
    else:
        rank = comm.rank
        ntask = comm.size
        time_start = comm.bcast(extended_times[0], root=0)
        time_stop = comm.bcast(extended_times[-1], root=ntask - 1)

    nreal = int(np.ceil((time_stop - time_start) / stationary_period))
    realization = ((extended_times - time_start) / stationary_period).astype(np.int64)

    # Set flagged elements to zero

    extended_signal1[extended_flags != 0] = 0
    if extended_signal2 is not None:
        extended_signal2[extended_flags != 0] = 0

    covs = {}

    # Loop over local realizations
    for ireal in range(realization[0], realization[-1] + 1):
        # Evaluate the covariance
        realflg = realization == ireal
        realtimes = extended_times[realflg]
        realgood = extended_flags[realflg] == 0
        realsig1 = extended_signal1[realflg].copy()
        if extended_signal2 is not None:
            realsig2 = extended_signal2[realflg].copy()
        cov_hits = np.zeros(lagmax, dtype=np.int64)
        cov = np.zeros(lagmax, dtype=np.float64)
        # Process all global intervals that overlap this realization
        for start_time, stop_time in global_intervals:
            if start_time is not None and (
                start_time > times[-1] or start_time > realtimes[-1]
            ):
                # This interval is owned by another process
                continue
            if stop_time is not None and stop_time < realtimes[0]:
                # no overlap
                continue
            # Avoid double-counting sample pairs
            if stop_time is None:
                all_sums = rank == ntask - 1
            else:
                all_sums = stop_time < realtimes[-1]
            # Find the correct range of samples
            if start_time is None or stop_time is None:
                ind = slice(realsig1.size)
            else:
                istart, istop = np.searchsorted(realtimes, [start_time, stop_time])
                ind = slice(istart, istop)
            good = realgood[ind].astype(np.uint8)
            ngood = np.sum(good)
            if ngood == 0:
                continue
            if extended_signal2 is None:
                fod_autosums(realsig1[ind], good, lagmax, cov, cov_hits, all_sums)
            else:
                fod_crosssums(
                    realsig1[ind],
                    realsig2[ind],
                    good,
                    lagmax,
                    cov,
                    cov_hits,
                    all_sums,
                    symmetric,
                )
        covs[ireal] = (cov_hits, cov)

    # Collect the estimated covariance functions

    my_covs = {}
    nreal_task = int(np.ceil(nreal / ntask))

    if comm is None:
        for ireal in range(nreal):
            if ireal in covs:
                cov_hits, cov = covs[ireal]
            else:
                cov_hits = np.zeros(lagmax, dtype=np.int64)
                cov = np.zeros(lagmax, dtype=np.float64)
            my_covs[ireal] = (cov_hits, cov)
    else:
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
    my_cov = []

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

        # w = np.roll(hamming(cov.size), -lagmax)
        # cov *= w

        psd = np.fft.rfft(cov).real
        psdfreq = np.fft.rfftfreq(len(cov), d=1 / fsample)

        # Post process the PSD estimate:
        #  1) Deconvolve the prewhitening (highpass) filter
        arg = 2 * np.pi * np.abs(psdfreq) * naverage / fsample
        tf = np.ones(lagmax)
        ind = arg != 0
        tf[ind] -= np.sin(arg[ind]) / arg[ind]
        psd[ind] /= tf[ind] ** 2
        #  2) Apply the Hann window to reduce unnecessary noise
        psd = np.convolve(psd, [0.25, 0.5, 0.25], mode="same")

        # Transfrom the corrected PSD back to get an unbiased
        # covariance function
        smooth_cov = np.fft.irfft(psd)
        my_cov.append((cov_hits, smooth_cov[:lagmax]))

        # Set the white noise PSD normalization to sigma**2 / fsample
        psd /= fsample

        tstart = time_start + ireal * stationary_period
        tstop = min(tstart + stationary_period, time_stop)

        my_psds.append((tstart, tstop, psdfreq, psd))

    if return_cov:
        return my_psds, my_cov
    else:
        return my_psds


@function_timer
def smooth_with_hits(hits, cov, wbin):
    """Smooth the covariance function.

    Smooth the covariance function, taking into account the number of hits in each bin.

    Args:
        hits (array_like):  The number of hits for each sample lag.
        cov (array_like):  The time domain covariance.
        wbin (int):  The number of samples per smoothing bin.

    Returns:
        (tuple): The (smoothed hits, smoothed covariance).

    """

    kernel = np.ones(wbin)
    smooth_hits = fftconvolve(hits, kernel, mode="same")
    smooth_cov = fftconvolve(cov * hits, kernel, mode="same")
    good = smooth_hits > 0
    smooth_cov[good] /= smooth_hits[good]

    return smooth_hits, smooth_cov
