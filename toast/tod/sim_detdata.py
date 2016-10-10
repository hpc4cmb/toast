# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import unittest

import numpy as np

import scipy.fftpack as sft
import scipy.interpolate as si
import scipy.sparse as sp

import healpy as hp

from . import qarray as qa

from .tod import TOD

from .noise import Noise

from ..operator import Operator

from .. import rng as rng


def sim_noise_timestream(realization, stream, rate, samples, oversample, freq, psd):
    
    fftlen = 2
    while fftlen <= (oversample * samples):
        fftlen *= 2
    half = int(fftlen / 2)
    norm = rate * float(half)
    df = rate / fftlen

    interp_freq = np.linspace(df, df*half, num=half, endpoint=True)
    loginterp_freq = np.log10(interp_freq)

    # ignore zero frequency

    trimzero = False
    if freq[0] <= 0.0:
        trimzero = True

    if trimzero:
        rawfreq = freq[1:]
    else:
        rawfreq = freq
    lograwfreq = np.log10(rawfreq)

    if trimzero:
        rawpsd = psd[1:]
    else:
        rawpsd = psd
    lograwpsd = np.log10(rawpsd)

    # Ensure that the input frequency range includes all the frequencies
    # we need.  Otherwise the extrapolation is not well defined.

    if (rawfreq[0] > interp_freq[0]):
        raise RuntimeError("input PSD does not go to low enough frequency to allow for interpolation")
    if (rawfreq[-1] < interp_freq[-1]):
        raise RuntimeError("input PSD does not go to high enough frequency to allow for interpolation")

    # interpolate

    interp = si.InterpolatedUnivariateSpline(lograwfreq, lograwpsd, k=1, ext=2)

    loginterp_psd = interp(loginterp_freq)

    interp_psd = np.power(10.0, loginterp_psd)

    # High-pass filter the PSD to not contain power below
    # fmin to limit the correlation length.  Also cut
    # Nyquist.

    fmin = rate / float(samples)
    interp_psd[(interp_freq < fmin)] = 0.0
    interp_psd[-1] = 0.0

    # gaussian Re/Im randoms

    fdata = rng.random(fftlen, sampler="gaussian", key=(realization, stream), counter=(0,0))

    # scale by PSD

    scale = np.sqrt(interp_psd * norm)

    fdata[0] *= np.sqrt(2.0) * scale[0]
    fdata[1:half] *= scale[0:-1]
    fdata[half] *= np.sqrt(2.0) * scale[-1]
    fdata[half+1:] *= scale[-2::-1]

    # inverse FFT

    tdata = sft.irfft(fdata)

    # return the timestream and interpolated PSD for debugging.

    return (tdata[0:samples], interp_freq, interp_psd)



class OpSimNoise(Operator):
    """
    Operator which generates noise timestreams.

    This passes through each observation and every process generates data
    for its assigned samples.

    Args:
        out (str): accumulate data to the cache with name <out>_<detector>.
            If the named cache objects do not exist, then they are created.
        stream (int): random stream offset.  This should be the same for
            all processes within a group, and should be offset between
            groups in such a way to ensure different streams between every
            detector, in every chunk, of every TOD, across every observation.
        realization (int): if simulating multiple realizations, the realization
            index.  This is used in combination with the stream when calling
            the RNG.
    """

    def __init__(self, out='noise', stream=None, realization=0):
        
        # We call the parent class constructor, which currently does nothing
        super().__init__()

        if stream is None:
            raise RuntimeError("you must specify the random stream index")

        self._out = out
        self._oversample = 2
        self._rngstream = stream
        self._realization = realization


    @property
    def timedist(self):
        return self._timedist


    def exec(self, data):
        """
        Generate noise timestreams.

        This iterates over all observations and detectors.  For each
        locally stored piece of the data, we query which chunks of the
        original data distribution we have.  A "stream" index is
        computed using the observation number, the detector number, and
        the absolute chunk index.  For each chunk assigned to this process,
        generate a noise realization.  The PSD that is valid for the
        current chunk (obtained from the Noise object for each observation)
        is used when generating the timestream.

        Args:
            data (toast.Data): The distributed data.
        """
        comm = data.comm
        rngobs = 0

        for obs in data.obs:
            tod = obs['tod']
            nse = obs['noise']
            if tod.local_chunks is None:
                raise RuntimeError('noise simulation for uniform distributed samples not implemented')

            # for purposes of incrementing the random stream, find
            # the number of detectors
            alldets = tod.detectors
            ndet = len(alldets)

            # compute effective sample rate

            times = tod.read_times(local_start=0, n=tod.local_samples[1])
            dt = np.mean(times[1:-1] - times[0:-2])
            rate = 1.0 / dt

            # eventually we'll redistribute, to allow long correlations...

            # iterate over each chunk (stationary interval)

            tod_offset = 0

            for curchunk in range(tod.local_chunks[1]):
                abschunk = tod.local_chunks[0] + curchunk
                chksamp = tod.total_chunks[abschunk]

                idet = 0
                for det in tod.local_dets:

                    detstream = self._rngstream + rngobs + (abschunk * ndet) + idet

                    (nsedata, freq, psd) = sim_noise_timestream(self._realization, detstream, rate, chksamp, self._oversample, nse.freq(det), nse.psd(det))

                    # write to cache

                    cachename = "{}_{}".format(self._out, det)
                    if not tod.cache.exists(cachename):
                        tod.cache.create(cachename, np.float64, (tod.local_samples[1],))
                    ref = tod.cache.reference(cachename)[tod_offset:tod_offset+chksamp]
                    ref[:] += nsedata

                    idet += 1

                tod_offset += chksamp

            # increment the observation rng stream offset
            # by the number of chunks times number of detectors
            rngobs += (ndet * len(tod.total_chunks))

        return


class OpSimGradient(Operator):
    """
    Generate a fake sky signal as a gradient between the poles.

    This passes through each observation and creates a fake signal timestream
    based on the cartesian Z coordinate of the HEALPix pixel containing the
    detector pointing.

    Args:
        out (str): accumulate data to the cache with name <out>_<detector>.
            If the named cache objects do not exist, then they are created.
        nside (int): the HEALPix NSIDE value to use.
        min (float): the minimum value to use at the South Pole.
        max (float): the maximum value to use at the North Pole.
        nest (bool): whether to use NESTED ordering.
    """

    def __init__(self, out='grad', nside=512, min=-100.0, max=100.0, nest=False):
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        self._nside = nside
        self._out = out
        self._min = min
        self._max = max
        self._nest = nest


    def exec(self, data):
        """
        Create the gradient timestreams.

        This pixelizes each detector's pointing and then assigns a 
        timestream value based on the cartesian Z coordinate of the pixel
        center.

        Args:
            data (toast.Data): The distributed data.
        """
        comm = data.comm

        zaxis = np.array([0,0,1], dtype=np.float64)
        nullquat = np.array([0,0,0,1], dtype=np.float64)

        range = self._max - self._min

        for obs in data.obs:
            tod = obs['tod']
            base = obs['baselines']
            nse = obs['noise']
            intrvl = obs['intervals']

            for det in tod.local_dets:
                pdata = np.copy(tod.read_pntg(detector=det, local_start=0, n=tod.local_samples[1]))
                flags, common = tod.read_flags(detector=det, local_start=0, n=tod.local_samples[1])
                totflags = np.copy(flags)
                totflags |= common

                pdata[totflags != 0,:] = nullquat

                dir = qa.rotate(pdata, zaxis)
                pixels = hp.vec2pix(self._nside, dir[:,0], dir[:,1], dir[:,2], nest=self._nest)
                x, y, z = hp.pix2vec(self._nside, pixels, nest=self._nest)
                z += 1.0
                z *= 0.5
                z *= range
                z += self._min
                z[totflags != 0] = 0.0

                cachename = "{}_{}".format(self._out, det)
                if not tod.cache.exists(cachename):
                    tod.cache.create(cachename, np.float64, (tod.local_samples[1],))
                ref = tod.cache.reference(cachename)
                ref[:] += z

        return

    def sigmap(self):
        """
        (array): Return the underlying signal map (full map on all processes).
        """
        range = self._max - self._min
        pix = np.arange(0, 12*self._nside*self._nside, dtype=np.int64)
        x, y, z = hp.pix2vec(self._nside, pix, nest=self._nest)
        z += 1.0
        z *= 0.5
        z *= range
        z += self._min
        return z


class OpSimScan(Operator):
    """
    Operator which generates sky signal by scanning from a map.

    The signal to use should already be in a distributed pixel structure,
    and local pointing should already exist.

    Args:
        distmap (DistPixels): the distributed map domain data.
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.
        weights (str): the name of the cache object (<weights>_<detector>)
            containing the pointing weights to use.
        out (str): accumulate data to the cache with name <out>_<detector>.
            If the named cache objects do not exist, then they are created.
    """
    def __init__(self, distmap=None, pixels='pixels', weights='weights', out='scan'):
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        self._map = distmap
        self._pixels = pixels
        self._weights = weights
        self._out = out


    def exec(self, data):
        """
        Create the timestreams by scanning from the map.

        This loops over all observations and detectors and uses the pointing
        matrix to project the distributed map into a timestream.

        Args:
            data (toast.Data): The distributed data.
        """
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world
        # the communicator within the group
        cgroup = comm.comm_group
        # the communicator with all processes with
        # the same rank within their group
        crank = comm.comm_rank

        for obs in data.obs:
            tod = obs['tod']

            for det in tod.local_dets:

                # get the pixels and weights from the cache

                pixelsname = "{}_{}".format(self._pixels, det)
                weightsname = "{}_{}".format(self._weights, det)
                pixels = tod.cache.reference(pixelsname)
                weights = tod.cache.reference(weightsname)

                nnz = weights.shape[1]

                sm, lpix = self._map.global_to_local(pixels)

                f = ( np.dot(weights[x], self._map.data[sm[x], lpix[x]]) if (lpix[x] >= 0) else 0 for x in range(tod.local_samples[1]) )
                maptod = np.fromiter(f, np.float64, count=tod.local_samples[1])

                cachename = "{}_{}".format(self._out, det)
                if not tod.cache.exists(cachename):
                    tod.cache.create(cachename, np.float64, (tod.local_samples[1],))
                ref = tod.cache.reference(cachename)
                ref[:] += maptod

        return



