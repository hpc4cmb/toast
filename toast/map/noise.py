# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import unittest

import numpy as np

from ..dist import Comm, Data
from ..operator import Operator
from .pixels import DistPixels

from ._noise import (_accumulate_diagonal, _eigendecompose_covariance, 
    _multiply_covariance, _apply_covariance)



class OpAccumDiag(Operator):
    """
    Operator which accumulates the diagonal covariance and noise weighted map.

    This operator requires that the local pointing matrix has already been
    computed.  Each process has local pieces of the map products.  This
    operator can optionally accumulate any combination of the hit map,
    the white noise weighted map, and the diagonal inverse pixel covariance.

    NOTE:  The input DistPixels objects (noise weighted map, hit map, and
    inverse covariance) are purposefully NOT set to zero at the start.  This
    allows accumulating multiple instances of the distributed data.  This might
    be needed (for example) if all desired timestream data does not fit into
    memory.  You should manually clear the pixel domain objects before
    accumulation if desired.

    Args:
        zmap (DistPixels):  (optional) the noise weighted map to accumulate.
        hits (DistPixels):  (optional) the hits to accumulate.
        invnpp (DistPixels):  (optional) the diagonal covariance matrix.
        detweights (dictionary): individual noise weights to use for each
            detector.
        name (str): the name of the cache object (<name>_<detector>) to
            use for the detector timestream.  If None, use the TOD.
        flag_name (str): the name of the cache object (<flag_name>_<detector>) to
            use for the detector flags.  If None, use the TOD.
        flag_mask (int): the integer bit mask (0-255) that should be 
            used with the detector flags in a bitwise AND.
        common_flag_name (str): the name of the cache object 
            (<common_flag_name>_<detector>) to use for the common flags.  
            If None, use the TOD.
        common_flag_mask (int): the integer bit mask (0-255) that should be 
            used with the common flags in a bitwise AND.
        apply_flags (bool): whether to apply flags to the pixel numbers.
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.
        weights (str): the name of the cache object (<weights>_<detector>)
            containing the pointing weights to use.
    """

    def __init__(self, zmap=None, hits=None, invnpp=None, detweights=None, name=None, flag_name=None, 
                flag_mask=255, common_flag_name=None, common_flag_mask=255, pixels='pixels', 
                weights='weights', apply_flags=True):
        
        self._flag_name = flag_name
        self._flag_mask = flag_mask
        self._common_flag_name = common_flag_name
        self._common_flag_mask = common_flag_mask
        self._apply_flags = apply_flags

        self._name = name
        self._pixels = pixels
        self._weights = weights
        self._detweights = detweights

        self._zmap = zmap
        self._hits = hits
        self._invnpp = invnpp

        # We call the parent class constructor, which currently does nothing
        super().__init__()


    def exec(self, data):
        """
        Iterate over all observations and detectors and accumulate.
        
        Args:
            data (toast.Data): The distributed data.
        """
        # the two-level pytoast communicator
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world
        # the communicator within the group
        cgroup = comm.comm_group
        # the communicator with all processes with
        # the same rank within their group
        crank = comm.comm_rank

        # since we could get any combination of hits, invnpp, and zmap,
        # we pick one of them to use for the mapping between global and
        # local pixels.

        globloc = None

        # set up some helper variables to pass to cython code

        do_hits = 0
        chits = np.zeros((1,1,1), dtype=np.int64)
        if self._hits is not None:
            do_hits = 1
            chits = self._hits.data
            if globloc is None:
                globloc = self._hits
        
        do_zmap = 0
        czmap = np.zeros((1,1,1), dtype=np.float64)
        if self._zmap is not None:
            do_zmap = 1
            czmap = self._zmap.data
            if globloc is None:
                globloc = self._zmap
        
        do_invnpp = 0
        cinvnpp = np.zeros((1,1,1), dtype=np.float64)
        if self._invnpp is not None:
            do_invnpp = 1
            cinvnpp = self._invnpp.data
            if globloc is None:
                globloc = self._invnpp

        for obs in data.obs:
            tod = obs['tod']

            # FIXME:  put this into cython and thread over detectors.

            for det in tod.local_dets:

                # get the pixels and weights from the cache

                pixelsname = "{}_{}".format(self._pixels, det)
                weightsname = "{}_{}".format(self._weights, det)
                pixels = tod.cache.reference(pixelsname)
                weights = tod.cache.reference(weightsname)

                cachename = None
                if self._name is not None:
                    cachename = "{}_{}".format(self._name, det)
                    signal = tod.cache.reference(cachename)
                else:
                    signal = tod.read(detector=det)

                # get flags

                if self._apply_flags:
                    if self._flag_name is not None:
                        cacheflagname = "{}_{}".format(self._flag_name, det)
                        detflags = tod.cache.reference(cacheflagname)
                        flags = (detflags & self._flag_mask) != 0
                        if self._common_flag_name is not None:
                            commonflags = tod.cache.reference(self._common_flag_name)
                            flags[(commonflags & self._common_flag_mask) != 0] = True
                    else:
                        detflags, commonflags = tod.read_flags(detector=det)
                        flags = np.logical_or((detflags & self._flag_mask) != 0, (commonflags & self._common_flag_mask) != 0)

                    pixels = pixels.copy() # Don't change the cached pixel numbers
                    pixels[flags] = -1

                # local pointing

                sm, lpix = globloc.global_to_local(pixels)
                
                detweight = 1.0
                
                if self._detweights is not None:
                    if det not in self._detweights.keys():
                        raise RuntimeError("no detector weights found for {}".format(det))
                    detweight = self._detweights[det]
                    if detweight == 0:
                        continue

                _accumulate_diagonal(czmap, do_zmap, chits, do_hits, cinvnpp, do_invnpp, signal, sm, lpix, weights, detweight)

        return


def covariance_invert(npp, threshold, rcond=None):
    """
    Invert the local piece of a diagonal noise covariance.

    This does an in-place inversion of the covariance.  The threshold is
    applied to the condition number of each block of the matrix.  Pixels
    failing the cut are set to zero.

    Args:
        npp (3D array): The data member of a distributed covariance.
        threshold (float): The condition number threshold to apply.
        rcond (3D array): (Optional) the inverse condition number map to fill.
    """
    if len(npp.shape) != 3:
        raise RuntimeError("distributed covariance matrix must have dimensions (number of submaps, pixels per submap, nnz*(nnz+1)/2)")
    if rcond is None:
        _eigendecompose_covariance(npp, np.zeros((1,1,1), dtype=np.float64), threshold, 1, 0)
    else:
        _eigendecompose_covariance(npp, rcond, threshold, 1, 1)
    return


def covariance_multiply(npp1, npp2):
    """
    Multiply two local piece of diagonal noise covariance.

    This does an in-place multiplication of the covariance.
    The results are returned in place of the first argument, npp1.

    Args:
        npp1 (3D array): The data member of a distributed covariance.
        npp2 (3D array): The data member of a distributed covariance.
    """
    if len(npp1.shape) != 3 or len(npp2.shape) != 3:
        raise RuntimeError("distributed covariance matrix must have dimensions (number of submaps, pixels per submap, nnz*(nnz+1)/2)")
    if np.any( npp1.shape != npp2.shape ):
        raise RuntimeError("Dimensions of the distributed matrices must agree but {} != {}".format(npp1.shape, npp2.shape))
    _multiply_covariance(npp1, npp2)
    return


def covariance_apply(npp, m):
    """
    Multiply two local piece of diagonal noise covariance.

    This does an in-place multiplication of the covariance.
    The results are returned in place of the first argument, npp1.

    Args:
        npp (3D array): The data member of a distributed covariance.
        m (3D array): The data member of a distributed map.
    """
    if len(npp.shape) != 3 or len(m.shape) != 3:
        raise RuntimeError("distributed covariance matrix must have dimensions (number of submaps, pixels per submap, nnz*(nnz+1)/2)")
    _apply_covariance(npp, m)
    return


def covariance_rcond(npp):
    """
    Return the local piece of the inverse condition number map.

    This computes the local piece of the inverse condition number map
    from the supplied covariance matrix data.

    Args:
        npp (3D array): The data member of a distributed covariance.

    Returns:
        rcond (3D array): The local data piece of the distributed
            inverse condition number map.
    """
    if len(npp.shape) != 3:
        raise RuntimeError("distributed covariance matrix must have dimensions (number of submaps, pixels per submap, nnz*(nnz+1)/2)")
    rcond = np.zeros((npp.shape[0], npp.shape[1], 1), dtype=np.float64)
    threshold = np.finfo(np.float64).eps
    _eigendecompose_covariance(npp, rcond, threshold, 0, 1)
    return rcond
