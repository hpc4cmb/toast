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




