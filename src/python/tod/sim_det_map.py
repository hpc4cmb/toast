# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import numpy as np
import healpy as hp
import libsharp

from .. import qarray as qa
from .tod import TOD
from ..op import Operator
from ..ctoast import sim_map_scan_map


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

    def __init__(self, out='grad', nside=512, min=-100.0, max=100.0, nest=False,
                 flag_mask=255, common_flag_mask=255):
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        self._nside = nside
        self._out = out
        self._min = min
        self._max = max
        self._nest = nest
        self._flag_mask = flag_mask
        self._common_flag_mask = common_flag_mask

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
                totflags = flags & self._flag_mask
                totflags |= (common & self._common_flag_mask)

                del flags
                del common

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
                del ref
                #print('Grad timestream:', ref, np.sum(ref!=0),' non-zeros', flush=True) # DEBUG

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

                nsamp, nnz = weights.shape

                sm, lpix = self._map.global_to_local(pixels)

                #f = (np.dot(weights[x], self._map.data[sm[x], lpix[x]])
                #     if (lpix[x] >= 0) else 0
                #     for x in range(tod.local_samples[1]))
                #maptod = np.fromiter(f, np.float64, count=tod.local_samples[1])
                maptod = np.zeros(nsamp)
                sim_map_scan_map(sm, weights, lpix, self._map.data, maptod)

                cachename = "{}_{}".format(self._out, det)
                if not tod.cache.exists(cachename):
                    tod.cache.create(cachename, np.float64,
                                     (tod.local_samples[1],))
                ref = tod.cache.reference(cachename)
                ref[:] += maptod
                
                del ref
                del pixels
                del weights

        return


class OpSmooth(Operator):
    """
    Operator that smooths a distributed map with a beam

    Apply a beam window function to a distributed map, it transforms
    a I or IQU distributed map to a_lm with libsharp, multiplies by beam(ell)
    and transforms back to pixel space.

    The beam can be specified either as a Full Width at Half Maximum of a gaussian beam
    in degrees or with a custom beam factor as a function of ell to be multiplied to the a_lm

    Parameters
    ----------
        comm: mpi4py communicator
            MPI communicator object
        signal_map : str
            the name of the cache object (<signal_map>_<detector>) containing the input map.
        lmax : int
            maximum ell value of the spherical harmonics transform
        grid : libsharp healpix_grid
            libsharp healpix grid object
        fwhm_deg : float
            Full Width Half Max in degrees of the gaussian beam
        beam : 2D np.ndarray
            1 column (I only) or 3 columns (different beam for polarization)
            beam as a function of ell
    """
    def __init__(self, comm=None, signal_map="signal_map",
                lmax=None, grid=None, fwhm_deg=None, beam=None, out="smoothed_signal_map"):
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        self.comm = comm
        self.signal_map = signal_map
        self.lmax = lmax
        self.out = out
        self.grid = grid

        # distribute alms
        local_m_indices = np.arange(self.comm.rank, lmax + 1, self.comm.size, dtype=np.int32)

        self.order = libsharp.packed_real_order(lmax, ms=local_m_indices)

        if (fwhm_deg is not None) and (beam is not None):
            raise Exception("OpSmooth error, specify either fwhm_deg or beam, not both")

        if (fwhm_deg is None) and (beam is None):
            raise Exception("OpSmooth error, specify fwhm_deg or beam")

        if fwhm_deg is not None:
            self.beam = hp.gauss_beam(fwhm=np.radians(fwhm_deg), lmax=lmax, pol=True)
        else:
            self.beam = beam

    def exec(self, data):
        """
        Create the timestreams...

        This loops over all observations and detectors and uses the pointing
        matrix to ...

        Args:
            data (toast.Data): The distributed data.
        """

        has_pol = len(data[self.signal_map]) > 1

        alm_sharp_I = libsharp.analysis(self.grid, self.order,
                                        np.ascontiguousarray(data[self.signal_map][0].reshape((1, 1, -1))),
                                        spin=0, comm=self.comm)
        self.order.almxfl(alm_sharp_I, np.ascontiguousarray(self.beam[:, 0:1]))
        data[self.out] = libsharp.synthesis(self.grid, self.order, alm_sharp_I, spin=0, comm=self.comm)[0]
        print(data[self.out].shape)

        if has_pol:
            alm_sharp_P = libsharp.analysis(self.grid, self.order,
                                            np.ascontiguousarray(data[self.signal_map][1:3].reshape((1, 2, -1))),
                                            spin=2, comm=self.comm)

            self.order.almxfl(alm_sharp_P, np.ascontiguousarray(self.beam[:, (1, 2)]))

            signal_map_P = libsharp.synthesis(self.grid, self.order, alm_sharp_P, spin=2, comm=self.comm)[0]
            data[self.out] = np.vstack((
                        data[self.out],
                        signal_map_P))
