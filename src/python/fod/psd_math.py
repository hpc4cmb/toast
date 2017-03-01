# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI

import os

import numpy as np

from ..ctoast import fod_autosums


def autocov_psd(times, signal, flags, lagmax, stationary_period, fsample, comm=None):
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
        stationary_period (float):  Length of a stationary interval in units of the times vector. 
        fsample (float):  The sampling frequency in Hz
    """

    if comm is None:
        rank = 0
        ntask = 1
        comm = MPI.COMM_SELF
    else:
        rank = comm.rank
        ntask = comm.size

    time_start = comm.bcast( times[0], root=0 )
    time_stop = comm.bcast( times[-1], root=ntask-1 )

    nreal = np.int(np.ceil((time_stop - time_start) / stationary_period))

    # Communicate lagmax samples from the beginning of the array backwards in the MPI communicator

    nsamp = signal.size

    if lagmax > nsamp:
        raise RuntimeError('autocov_psd: Communicating TOD beyond nearest neighbors is not implemented. Reduce lagmax or the size of the MPI communicator.')

    if rank != ntask - 1:
        nextend = lagmax
    else:
        nextend = 0

    extended_signal = np.zeros(nsamp+nextend, dtype=np.float64)
    extended_flags = np.zeros(nsamp+nextend, dtype=np.bool)
    extended_times = np.zeros(nsamp+nextend, dtype=times.dtype)

    extended_signal[:nsamp] = signal
    extended_flags[:nsamp] = flags
    extended_times[:nsamp] = times

    for evenodd in range(2):
        if rank%2 == evenodd%2:
            # Send
            if rank == 0: continue
            comm.send(signal[:lagmax], dest=rank-1, tag=0)
            comm.send(flags[:lagmax], dest=rank-1, tag=1)
            comm.send(times[:lagmax], dest=rank-1, tag=2)
        else:
            # Receive
            if rank == ntask-1: continue
            extended_signal[-lagmax:] = comm.recv(source=rank+1, tag=0)
            extended_flags[-lagmax:] = comm.recv(source=rank+1, tag=1)
            extended_times[-lagmax:] = comm.recv(source=rank+1, tag=2)

    realization = ((extended_times - time_start) / stationary_period).astype(np.int64)

    # Set flagged elements to zero

    extended_signal[extended_flags!=0] = 0

    autocovs = {}

    for ireal in range(realization[0], realization[-1]+1):

        # Evaluate the autocovariance

        realflg = realization == ireal

        sig = extended_signal[realflg].copy()
        flg = extended_flags[realflg]

        sig -= np.mean( sig )

        good = np.zeros(len(flg), dtype=np.int8)
        good[flg == 0] = 1
        ngood = np.sum( good )

        if ngood == 0:
            continue

        (autocov, autocov_hits) = fod_autosums(sig, good, lagmax)

        autocovs[ireal] = (autocov_hits, autocov)

    # Collect the estimated autocovariance functions

    my_autocovs = {}
    nreal_task = np.int(np.ceil(nreal/ntask))

    for ireal in range(nreal):
        
        owner = ireal // nreal_task

        if ireal in autocovs:
            autocov_hits, autocov = autocovs[ireal]
        else:
            autocov_hits = np.zeros(lagmax, dtype=np.int64)
            autocov = np.zeros(lagmax, dtype=np.float64)

        autocov_hits_total = np.zeros(lagmax, dtype=np.int64)
        autocov_total = np.zeros(lagmax, dtype=np.float64)

        comm.Reduce(autocov_hits, autocov_hits_total, op=MPI.SUM, root=owner)
        comm.Reduce(autocov, autocov_total, op=MPI.SUM, root=owner)

        if rank == owner:
            my_autocovs[ireal] = (autocov_hits_total, autocov_total)

    # Now process the ones this task owns

    my_psds = []

    for ireal in my_autocovs.keys():
        
        autocov_hits, autocov = my_autocovs[ireal]

        good = autocov_hits != 0
        autocov[good] /= autocov_hits[good]

        # Interpolate any empty bins

        if np.any(autocov_hits==0) and np.any(autocov_hits!=0):
            bad = autocov_hits == 0
            good = np.logical_not(bad)
            lag = np.arange(lagmax)
            autocov[bad] = np.interp(lag[bad], lag[good], autocov[good])

        # Fourier transform for the PSD

        autocov = np.hstack( [autocov, autocov[:0:-1]] )

        psd = np.abs(np.fft.rfft( autocov ))
        psdfreq = np.fft.rfftfreq( len(autocov), d=1/fsample )
        
        # Set the white noise PSD normalization to sigma**2 / fsample
        psd /= fsample

        tstart = time_start + ireal*stationary_period
        tstop = min(tstart + stationary_period, time_stop)

        my_psds.append( (tstart, tstop, psdfreq, psd) )

    return my_psds
