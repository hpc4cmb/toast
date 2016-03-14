# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

import scipy.fftpack as sft
import scipy.interpolate as si
import scipy.sparse as sp

import astropy.io.fits as af

import healpy as hp

import quaternionarray as qa

from .tod import TOD

from .noise import Noise

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
        start_abs = self.local_samples[0] + start
        start_time = self._firsttime + float(start_abs) / self._rate
        stop_time = start_time + float(n) / self._rate
        stamps = np.linspace(start_time, stop_time, num=n, endpoint=False, dtype=np.float64)
        return stamps


    def _put_times(self, start, stamps):
        raise RuntimeError('cannot write timestamps to simulated data streams')
        return


    def _get_pntg(self, detector, start, n):
        # compute the absolute sample offset
        start_abs = self.local_samples[0] + start

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
                pdata, pflags = tod.read_pntg(detector=det, local_start=0, n=tod.local_samples[1])
                dir = qa.rotate(pdata.reshape(-1, 4), zaxis)
                pixels = hp.vec2pix(self._nside, dir[:,0], dir[:,1], dir[:,2], nest=False)
                x, y, z = hp.pix2vec(self._nside, pixels, nest=False)
                z += 1.0
                z *= 0.5
                z *= range
                z += self._min
                data, flags = tod.read(detector=det, flavor=self._flavor, local_start=0, n=tod.local_samples[1])
                data += z
                tod.write(detector=det, flavor=self._flavor, local_start=0, data=data, flags=flags)

        return


# class OpSimScan(Operator):
#     """
#     Operator which generates sky signal by scanning from a map.


#     Args:
        
#     """

#     def __init__(self, mapfile=None, local=None, pname=None, flavor=None, accum=False):
#         # We call the parent class constructor, which currently does nothing
#         super().__init__()
#         self._mapfile = mapfile
#         self._local = local
#         self._pname = pname
#         self._flavor = flavor
#         self._accum = accum
#         self._bufsize = 2000000


#     def exec(self, data):
#         comm = data.comm
#         # the global communicator
#         cworld = comm.comm_world
#         # the communicator within the group
#         cgroup = comm.comm_group
#         # the communicator with all processes with
#         # the same rank within their group
#         crank = comm.comm_rank

#         # open the file and read maps in chunks.  broadcast
#         # to local maps.

#         mapdata = None
#         mapnnz = 0
#         map_npix = 0
#         if cworld.rank == 0:
#             mapdata = hp.read_map(self._mapfile)
#             mapnnz = mapdata.shape[0]
#             map_npix = mapdata.shape[1]

#         mapnnz = cworld.bcast(mapnnz, root=0)
#         map_npix = cworld.bcast(map_npix, root=0)

#         local_npix = len(self._local)
#         local_pixels = np.array(self._local, dtype=np.int64)
#         local_map = sp.csr_matrix((np.ones(local_npix, dtype=np.float64), (np.zeros(local_npix, dtype=np.int64), local_pixels), shape=(1,map_npix))


# local_map = sp.csr_matrix((np.ones(5*2, dtype=np.float64), (np.tile(np.array([2, 6, 9, 15, 17]), 2), np.array([0,1,0,1,0,1,0,1,0,1]))), shape=(20,2) )


# local = set([2, 6, 9, 15, 17])

# In [4]: pixels = np.array([2, 6, 9, 15, 17], dtype=np.int64)

# In [5]: weights = np.array([2.0, 2.0, 6.0, 6.0, 9.0, 9.0, 15.0, 15.0, 17.0, 17.0])

# In [6]: nnz = 2


# np.array([ np.dot(local_map[pixels,:].toarray()[i], weights.reshape(-1,2)[i]) for i in range(len(pixels)) ])



#         off = 0
#         nbcast = self._bufsize
#         buf_data = np.zeros(nbcast, dtype=np.float64)

#         while off < map_npix:
#             if off + nbcast > map_npix:
#                 nbcast = map_npix - off
#             if cworld.rank == 0:
#                 buf_data = mapdata[off:off+nbcast]
#             cworld.Bcast(buf_data, root=0)
#             buf_mask = np.where((local_pixels > off) and (local_pixels < off+nbcast))
#             buf_elem = local_pixels[buf_mask]
#             rel_elem = buf_elem - off

#             local_map[buf_elem] = buf_data[rel_elem]

#             off += nbcast





#         for obs in data.obs:
#             tod = obs['tod']

#             for det in tod.local_dets:
#                 nnz = tod.pmat_nnz(self, name=pname, detector=det)

#                 pixels, weights = tod.read_pmat(name=self._pname, detector=det, local_start=tod.local_offset, n=tod.local_samples)
                

#         return



class SimNoise(Noise):
    """
    Class representing an analytic noise model.

    This generates an analytic PSD for a set of detectors, given
    input values for the knee frequency, NET, exponent, sample rate,
    minimum frequency, etc.

    Args:
        rate (float): sample rate in Hertz.
        fmin (float): minimum frequency for high pass
        detectors (list): list of detectors.
        fknee (array like): list of knee frequencies.
        alpha (array like): list of alpha exponents (positive, not negative!).
        NET (array like): list of detector NETs.
    """

    def __init__(self, rate=None, fmin=None, detectors=None, fknee=None, alpha=None, NET=None):
        if rate is None:
            raise RuntimeError("you must specify the sample rate")
        if fmin is None:
            raise RuntimeError("you must specify the frequency for high pass")
        if detectors is None:
            raise RuntimeError("you must specify the detector list")
        if fknee is None:
            raise RuntimeError("you must specify the knee frequency list")
        if alpha is None:
            raise RuntimeError("you must specify the exponent list")
        if NET is None:
            raise RuntimeError("you must specify the NET")

        self._rate = rate
        self._fmin = fmin
        self._detectors = detectors
        
        self._fknee = {}
        for f in enumerate(fknee):
            self._fknee[detectors[f[0]]] = f[1]

        self._alpha = {}
        for a in enumerate(alpha):
            if a < 0.0:
                raise RuntimeError("alpha exponents should be positive in this formalism")
            self._alpha[detectors[f[0]]] = f[1]

        self._NET = {}
        for n in enumerate(NET):
            self._NET[detectors[f[0]]] = f[1]

        # for purposes of determining the common frequency sampling
        # points, use the lowest knee frequency.
        lowknee = np.min(fknee)

        tempfreq = []
        cur = self._fmin
        while cur < 10.0 * lowknee:
            tempfreq.append(cur)
            cur *= 2.0
        nyquist = self._rate / 2.0
        df = (nyquist - cur) / 10.0
        tempfreq.extend([ (cur+x*df) for x in range(11) ])
        freq = np.array(tempfreq, dtype=np.float64)

        psds = {}

        for d in self._detectors:
            ktemp = np.power(self._knee[d], self._alpha[d])
            mtemp = np.power(self._fmin, self._alpha[d])
            temp = np.power(self._freq, self._alpha[d])
            psd[d] = (temp + ktemp) / (temp + mtemp)
            psd[d] *= (self._NET[d] * self._NET[d])

        # call the parent class constructor to store the psds
        super().__init__(detectors=detectors, freq=freq, psds=psds)

    @property
    def fmin(self):
        return self._fmin

    def fknee(self, det):
        return self._fknee[det]

    def alpha(self, det):
        return self._alpha[det]

    def NET(self, det):
        return self._NET[det]


class OpSimNoise(Operator):
    """
    Operator which generates noise timestreams and accumulates that data
    to a particular timestream flavor.

    This passes through each observation and every process generates data
    for its assigned samples.

    Args:
        flavor (str): the TOD flavor to accumulate the noise data to.
        stream (int): random stream offset.  This should be the same for
            all processes within a group, and should be offset between
            groups in such a way to ensure different streams between every
            detector, in every chunk, of every TOD, across every observation.
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
        rngobs = 0

        for obs in data.obs:
            tod = obs['tod']
            nse = obs['noise']
            if tod.local_chunks is None:
                raise RuntimeError('noise simulation for uniform distributed samples not implemented')

            # for purposes of incrementing the random seed, find
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
            print(tod.local_chunks)

            for curchunk in range(tod.local_chunks[1]):
                abschunk = tod.local_chunks[0] + curchunk
                print("abschunk = ", abschunk)
                chksamp = tod.total_chunks[abschunk]
                nsedata = np.zeros(chksamp, dtype=np.float64)

                fftlen = 2
                while fftlen <= (self._oversample * chksamp):
                    fftlen *= 2
                half = int(fftlen / 2)
                norm = rate * half;
                df = rate / fftlen

                freq = np.linspace(df, df*half, num=half, endpoint=True)
                logfreq = np.log10(freq)

                tempdata = np.zeros(chksamp, dtype=np.float64)
                tempflags = np.zeros(chksamp, dtype=np.uint8)
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
                    fmin = rate / float(chksamp)
                    psd[freq < fmin] = 0.0
                    psd[-1] = 0.0

                    #np.savetxt("out_simnoise_psd.txt", np.transpose(np.vstack((freq, psd))))

                    # gaussian Re/Im randoms

                    # FIXME: Setting the seed like this does NOT guarantee uncorrelated
                    # results from the generator.  This is just a place holder until
                    # the streamed rng is implemented.

                    seed = self._rngstream + rngobs + (abschunk * ndet) + idet
                    np.random.seed(seed)

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
                        tempdata, tempflags = tod.read(detector=det, flavor=self._flavor, local_start=tod_offset, n=chksamp)
                        tempdata += tdata[0:chksamp]
                    else:
                        tempdata = tdata[0:chksamp]

                    tod.write(detector=det, flavor=self._flavor, local_start=tod_offset, data=tempdata, flags=tempflags)

                    idet += 1

                tod_offset += chksamp

            # increment the observation rng stream offset
            # by the number of chunks times number of detectors
            rngobs += (ndet * len(tod.total_chunks))

        return


