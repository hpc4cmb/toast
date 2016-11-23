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


def sim_noise_timestream(realization, telescope, component, obsindx, detindx,
                         rate, firstsamp, samples, oversample, freq, psd):
    """
    Generate a noise timestream, given a starting RNG state.

    Use the RNG parameters to generate unit-variance Gaussian samples
    and then modify the Fourier domain amplitudes to match the desired
    PSD.

    The RNG (Threefry2x64 from Random123) takes a "key" and a "counter"
    which each consist of two unsigned 64bit integers.  These four
    numbers together uniquely identify a single sample.  We construct
    those four numbers in the following way:

    key1 = realization * 2^32 + telescope * 2^16 + component
    key2 = obsindx * 2^32 + detindx 
    counter1 = currently unused (0)
    counter2 = sample in stream

    counter2 is incremented internally by the RNG function as it calls
    the underlying Random123 library for each sample.

    Args:
        realization (int): the Monte Carlo realization.
        telescope (int): a unique index assigned to a telescope.
        component (int): a number representing the type of timestream
            we are generating (detector noise, common mode noise,
            atmosphere, etc).
        obsindx (int): the global index of this observation.
        detindx (int): the global index of this detector.
        rate (float): the sample rate.
        firstsamp (int): the start sample in the stream.
        samples (int): the number of samples to generate.
        oversample (int): the factor by which to expand the FFT length
            beyond the number of samples.
        freq (array): the frequency points of the PSD.
        psd (array): the PSD values.

    Returns (tuple):
        the timestream array, the interpolated PSD frequencies, and
            the interpolated PSD values.
    """
    
    fftlen = 2
    while fftlen <= (oversample * samples):
        fftlen *= 2
    npsd = fftlen // 2 + 1
    norm = rate * float(npsd - 1)

    interp_freq = np.fft.rfftfreq(fftlen, 1/rate)
    if interp_freq.size != npsd:
        raise RuntimeError("interpolated PSD frequencies do not have expected "
                           "length")

    # Ensure that the input frequency range includes all the frequencies
    # we need.  Otherwise the extrapolation is not well defined.

    if np.amin(freq) < 0.0:
        raise RuntimeError("input PSD frequencies should be >= zero")

    if np.amin(psd) < 0.0:
        raise RuntimeError("input PSD values should be >= zero")

    increment = rate / fftlen

    if freq[0] > increment:
        raise RuntimeError("input PSD does not go to low enough frequency to "
                           "allow for interpolation")

    nyquist = rate / 2
    if np.abs((freq[-1]-nyquist)/nyquist) > .01:
        raise RuntimeError("last frequency element does not match Nyquist "
                           "frequency for given sample rate")

    # Perform a logarithmic interpolation.  In order to avoid zero values, we 
    # shift the PSD by a fixed amount in frequency and amplitude.

    psdshift = 0.01 * np.amin(psd[(psd > 0.0)])
    freqshift = increment

    loginterp_freq = np.log10(interp_freq + freqshift)
    logfreq = np.log10(freq + freqshift)
    logpsd = np.log10(psd + psdshift)

    interp = si.interp1d(logfreq, logpsd, kind='linear',
                         fill_value='extrapolate')
    
    loginterp_psd = interp(loginterp_freq)
    interp_psd = np.power(10.0, loginterp_psd) - psdshift

    # Zero out DC value

    interp_psd[0] = 0.0

    # gaussian Re/Im randoms, packed into a complex valued array

    key1 = realization * 4294967296 + telescope * 65536 + component
    key2 = obsindx * 4294967296 + detindx 
    counter1 = 0
    counter2 = firstsamp * oversample

    rngdata = rng.random(2*npsd, sampler="gaussian", key=(key1, key2), 
        counter=(counter1, counter2))
    fdata = rngdata[:npsd] + 1j * rngdata[npsd:]

    # set the Nyquist frequency imaginary part to zero

    fdata[-1] = fdata[-1].real + 0.0j

    # scale by PSD

    scale = np.sqrt(interp_psd * norm)
    fdata *= scale

    # inverse FFT

    tdata = np.fft.irfft(fdata)

    # subtract the DC level- for just the samples that we are returning

    offset = (fftlen - samples) // 2

    DC = np.mean(tdata[offset:offset+samples])
    tdata[offset:offset+samples] -= DC

    # return the timestream and interpolated PSD for debugging.
    
    return (tdata[offset:offset+samples], interp_freq, interp_psd)


class OpSimNoise(Operator):
    """
    Operator which generates noise timestreams.

    This passes through each observation and every process generates data
    for its assigned samples.

    Args:
        out (str): accumulate data to the cache with name <out>_<detector>.
            If the named cache objects do not exist, then they are created.
        realization (int): if simulating multiple realizations, the realization
            index.
        component (int): the component index to use for this noise simulation.
    """

    def __init__(self, out='noise', realization=0, component=0):
        
        # We call the parent class constructor, which currently does nothing
        super().__init__()

        self._out = out
        self._oversample = 2
        self._realization = realization
        self._component = component

    @property
    def timedist(self):
        return self._timedist

    def exec(self, data):
        """
        Generate noise timestreams.

        This iterates over all observations and detectors and generates
        the noise timestreams based on the noise object for the current
        observation.

        Args:
            data (toast.Data): The distributed data.
        """
        comm = data.comm

        for obs in data.obs:
            obsindx = 0
            if 'id' in obs:
                obsindx = obs['id']
            else:
                print("Warning: observation ID is not set, using zero!")

            telescope = 0
            if 'telescope' in obs:
                telescope = obs['telescope']

            tod = obs['tod']
            nse = obs['noise']
            if tod.local_chunks is None:
                raise RuntimeError('noise simulation for uniform distributed '
                                   'samples not implemented')

            # compute effective sample rate

            times = tod.read_times(local_start=0, n=tod.local_samples[1])
            dt = np.mean(times[1:-1] - times[0:-2])
            rate = 1.0 / dt

            # eventually we'll redistribute, to allow long correlations...

            # Iterate over each chunk.

            tod_first = tod.local_samples[0]
            chunk_first = tod_first

            for curchunk in range(tod.local_chunks[1]):
                abschunk = tod.local_chunks[0] + curchunk
                chunk_samp = tod.total_chunks[abschunk]
                local_offset = chunk_first - tod_first

                idet = 0
                for det in tod.local_dets:

                    detindx = tod.detindx[det]

                    (nsedata, freq, psd) = sim_noise_timestream(
                        self._realization, telescope, self._component, obsindx,
                        detindx, rate, chunk_first, chunk_samp,
                        self._oversample, nse.freq(det), nse.psd(det))

                    # write to cache

                    cachename = "{}_{}".format(self._out, det)
                    if not tod.cache.exists(cachename):
                        tod.cache.create(cachename, np.float64,
                                         (tod.local_samples[1],))
                    
                    ref = tod.cache.reference(cachename)[
                        local_offset:local_offset+chunk_samp]
                    ref[:] += nsedata

                    idet += 1

                chunk_first += chunk_samp

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
                pdata = np.copy(tod.read_pntg(detector=det, local_start=0,
                                              n=tod.local_samples[1]))
                flags, common = tod.read_flags(detector=det, local_start=0,
                                               n=tod.local_samples[1])
                totflags = np.copy(flags)
                totflags |= common

                pdata[totflags != 0,:] = nullquat

                dir = qa.rotate(pdata, zaxis)
                pixels = hp.vec2pix(self._nside, dir[:,0], dir[:,1], dir[:,2],
                                    nest=self._nest)
                x, y, z = hp.pix2vec(self._nside, pixels, nest=self._nest)
                z += 1.0
                z *= 0.5
                z *= range
                z += self._min
                z[totflags != 0] = 0.0

                cachename = "{}_{}".format(self._out, det)
                if not tod.cache.exists(cachename):
                    tod.cache.create(cachename, np.float64,
                                     (tod.local_samples[1],))
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
    def __init__(self, distmap=None, pixels='pixels', weights='weights',
                 out='scan'):
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

                f = (np.dot(weights[x], self._map.data[sm[x], lpix[x]])
                     if (lpix[x] >= 0) else 0
                     for x in range(tod.local_samples[1]))
                maptod = np.fromiter(f, np.float64, count=tod.local_samples[1])

                cachename = "{}_{}".format(self._out, det)
                if not tod.cache.exists(cachename):
                    tod.cache.create(cachename, np.float64,
                                     (tod.local_samples[1],))
                ref = tod.cache.reference(cachename)
                ref[:] += maptod

        return
