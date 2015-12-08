# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

import scipy.fftpack as sft
import scipy.interpolate as si

import healpy as hp

import quaternionarray as qa

from ..dist import distribute_det_samples

from .tod import TOD

from ..operator import Operator


class TODFake(TOD):
    """
    Provide a simple generator of fake detector pointing.

    Detector focalplane offsets are specified as a dictionary of 4-element
    ndarrays.  The boresight pointing is a simple looping over HealPix 
    ring ordered pixel centers.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the data is distributed.
        detectors (dictionary): each key is the detector name, and each value
                  is a quaternion tuple.
        samples (int): maximum allowed samples.
        firsttime (float): starting time of data.
        rate (float): sample rate in Hz.
    """

    def __init__(self, mpicomm=MPI.COMM_WORLD, detectors=None, samples=0, firsttime=0.0, rate=100.0, nside=512):
        if detectors is None:
            self._fp = {TOD.DEFAULT_FLAVOR : np.array([0.0, 0.0, 1.0, 0.0])}
        else:
            self._fp = detectors

        self._detlist = sorted(list(self._fp.keys()))
        
        super().__init__(mpicomm=mpicomm, timedist=True, detectors=self._detlist, flavors=None, samples=samples)

        self._firsttime = firsttime
        self._rate = rate
        self._nside = nside
        self._npix = 12 * self._nside * self._nside


    def _get(self, detector, flavor, start, n):
        # This class just returns data streams of zeros
        return ( np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.uint8) )


    def _put(self, detector, flavor, start, data, flags):
        raise RuntimeError('cannot write data to simulated data streams')
        return


    def _get_times(self, start, n):
        start_abs = self.local_offset + start
        start_time = self._firsttime + float(start_abs) / self._rate
        stop_time = start_time + float(n) / self._rate
        stamps = np.linspace(start_time, stop_time, num=n, endpoint=False, dtype=np.float64)
        return stamps


    def _put_times(self, start, stamps):
        raise RuntimeError('cannot write timestamps to simulated data streams')
        return


    def _get_pntg(self, detector, start, n):
        # compute the absolute sample offset
        start_abs = self.local_offset + start

        detquat = np.asarray(self._fp[detector])

        # pixel offset
        start_pix = int(start_abs % self._npix)
        pixels = np.linspace(start_pix, start_pix + n, num=n, endpoint=False)
        pixels = np.mod(pixels, self._npix*np.ones(n, dtype=np.int64)).astype(np.int64)

        # the result of this is normalized
        x, y, z = hp.pix2vec(self._nside, pixels, nest=False)

        # z axis is obviously normalized
        zaxis = np.array([0,0,1], dtype=np.float64)
        ztiled = np.tile(zaxis, x.shape[0]).reshape(-1,3)

        # ... so dir is already normalized
        dir = np.ravel(np.column_stack((x, y, z))).reshape(-1,3)

        # get the rotation axis
        v = np.cross(ztiled, dir)
        v = v / np.sqrt(np.sum(v * v, axis=1)).reshape(-1,1)

        # this is the vector-wise dot product
        zdot = np.sum(ztiled * dir, axis=1).reshape(-1,1)
        ang = 0.5 * np.arccos(zdot)

        # angle element
        s = np.cos(ang)

        # axis
        v *= np.sin(ang)

        # build the un-normalized quaternion
        boresight = np.concatenate((v, s), axis=1)

        boresight = qa.norm(boresight)

        # boredir = qa.rotate(boresight, zaxis)
        # boredir = boredir / np.sum(boredir * boredir, axis=1).reshape(-1,1)

        # check = hp.vec2pix(self._nside, boredir[:,0], boredir[:,1], boredir[:,2], nest=False)
        # if not np.array_equal(pixels, check):
        #     print(list(enumerate(zip(dir,boredir))))
        #     print(pixels)
        #     print(check)
        #     raise RuntimeError('FAIL on TODFake')

        flags = np.zeros(n, dtype=np.uint8)
        data = qa.mult(boresight, detquat).flatten()

        return (data, flags)


    def _put_pntg(self, detector, start, data, flags):
        raise RuntimeError('cannot write data to simulated pointing')
        return


class OpSimGradient(Operator):
    """
    Operator which generates fake sky signal as a gradient between the poles
    and accumulates this.

    This passes through each observation and ...

    Args:
        
    """

    def __init__(self, nside=512, flavor=None, min=-100.0, max=100.0):
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        self._nside = nside
        self._flavor = flavor
        self._min = min
        self._max = max

    def exec(self, data):
        comm = data.comm

        zaxis = np.array([0,0,1], dtype=np.float64)
        range = self._max - self._min

        for obs in data.obs:
            tod = obs['tod']
            base = obs['baselines']
            nse = obs['noise']
            intrvl = obs['intervals']

            for det in tod.local_dets:
                pdata, pflags = tod.read_pntg(detector=det, local_start=0, n=tod.local_samples)
                dir = qa.rotate(pdata.reshape(-1, 4), zaxis)
                pixels = hp.vec2pix(self._nside, dir[:,0], dir[:,1], dir[:,2], nest=False)
                x, y, z = hp.pix2vec(self._nside, pixels, nest=False)
                z += 1.0
                z *= 0.5
                z *= range
                z += self._min
                data, flags = tod.read(detector=det, flavor=self._flavor, local_start=0, n=tod.local_samples)
                data += z
                tod.write(detector=det, flavor=self._flavor, local_start=0, data=data, flags=flags)

        return



class OpSimNoise(Operator):
    """
    Operator which generates noise timestreams and accumulates that data
    to a particular timestream flavor.

    This passes through each observation and every process generates data
    for its assigned samples.

    Args:
        flavor (str): the TOD flavor to accumulate the noise data to.
        stream (int): random stream for first detector.
        accum (bool): should output be accumulated or overwritten.
    """

    def __init__(self, flavor=None, stream=None, accum=False):
        
        # We call the parent class constructor, which currently does nothing
        super().__init__()

        if stream is None:
            raise RuntimeError("you must specify the random stream index")

        if flavor is None:
            self._flavor = TOD.DEFAULT_FLAVOR
        self._flavor = flavor

        self._oversample = 2
        self._accum = accum

        self._rngstream = stream


    @property
    def timedist(self):
        return self._timedist


    def exec(self, data):
        comm = data.comm

        for obs in data.obs:
            tod = obs['tod']
            nse = obs['noise']

            # compute effective sample rate

            times = tod.read_times(local_start=0, n=tod.local_samples)
            dt = np.mean(times[1:-1] - times[0:-2])
            rate = 1.0 / dt

            # eventually we'll redistribute, to allow long correlations...
            
            nsedata = np.zeros(tod.local_samples)

            fftlen = 2
            while fftlen <= (self._oversample * tod.local_samples):
                fftlen *= 2
            half = int(fftlen / 2)
            norm = rate * half;
            df = rate / fftlen

            freq = np.linspace(df, df*half, num=half, endpoint=True)
            logfreq = np.log10(freq)

            tempdata = np.zeros(tod.local_samples, dtype=np.float64)
            tempflags = np.zeros(tod.local_samples, dtype=np.uint8)
            tdata = np.zeros(fftlen, dtype=np.float64)
            fdata = np.zeros(fftlen, dtype=np.float64)

            # ignore zero frequency
            trimzero = False
            if nse.freq[0] == 0:
                trimzero = True

            if trimzero:
                rawfreq = nse.freq[1:]
            else:
                rawfreq = nse.freq
            lograwfreq = np.log10(rawfreq)

            idet = 0
            for det in tod.local_dets:

                # interpolate the psd

                if trimzero:
                    rawpsd = nse.psd(det)[1:]
                else:
                    rawpsd = nse.psd(det)

                #np.savetxt("out_simnoise_rawpsd.txt", np.transpose(np.vstack((rawfreq, rawpsd))))

                lograwpsd = np.log10(rawpsd)

                interp = si.InterpolatedUnivariateSpline(lograwfreq, lograwpsd, k=1, ext=0)

                logpsd = interp(logfreq)

                psd = np.power(10.0, logpsd)

                # High-pass filter the PSD to not contain power below
                # fmin to limit the correlation length.  Also cut
                # Nyquist.
                fmin = rate / float(tod.local_samples)
                psd[freq < fmin] = 0.0
                psd[-1] = 0.0

                #np.savetxt("out_simnoise_psd.txt", np.transpose(np.vstack((freq, psd))))

                # gaussian Re/Im randoms

                # FIXME: Setting the seed like this does NOT guarantee uncorrelated
                # results from the generator.  This is just a place holder until
                # the streamed rng is implemented.

                np.random.seed(self._rngstream + idet)

                fdata = np.zeros(fftlen)
                fdata = np.random.normal(loc=0.0, scale=1.0, size=fftlen)
                #np.savetxt("out_simnoise_fdata_uni.txt", fdata, delimiter='\n')

                # scale by PSD

                scale = np.sqrt(psd * norm)

                fdata[0] *= 0.0
                fdata[1:half] *= scale[0:-1]
                fdata[half] *= np.sqrt(2 * scale[-1])
                fdata[half+1:] *= scale[-2::-1]

                #np.savetxt("out_simnoise_fdata.txt", fdata, delimiter='\n')

                # make the noise TOD

                tdata = sft.irfft(fdata)
                #np.savetxt("out_simnoise_tdata.txt", tdata, delimiter='\n')

                if self._accum:
                    tempdata, tempflags = tod.read(detector=det, flavor=self._flavor, local_start=0, n=tod.local_samples)
                    tempdata += tdata[0:tod.local_samples]
                else:
                    tempdata = tdata[0:tod.local_samples]

                tod.write(detector=det, flavor=self._flavor, local_start=0, data=tempdata, flags=tempflags)

                idet += 1

        return


