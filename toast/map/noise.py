# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import unittest

import numpy as np

from ..dist import Comm, Data
from ..operator import Operator
from .pixels import DistPixels

from ._noise import (_accumulate_inverse_covariance,
    _invert_covariance, _cond_covariance, _multiply_covariance)



class OpInvCovariance(Operator):
    """
    Operator which computes the diagonal inverse pixel covariance.

    This operator requires that the local pointing matrix has already been
    computed.  Each process has a local piece of the inverse covariance.

    Args:
        invnpp (DistPixels):  The matrix to accumulate.
        hits (DistPixels):  (optional) the hits to accumulate.
        detweights (dictionary): individual noise weights to use for each
            detector.
        flag_name (str): the name of the cache object (<flag_name>_<detector>) to
            use for the detector flags.  If None, use the TOD.
        flag_mask (int): the integer bit mask (0-255) that should be 
            used with the detector flags in a bitwise AND.
        common_flag_name (str): the name of the cache object 
            (<common_flag_name>_<detector>) to use for the common flags.  
            If None, use the TOD.
        common_flag_mask (int): the integer bit mask (0-255) that should be 
            used with the common flags in a bitwise AND.
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.
        weights (str): the name of the cache object (<weights>_<detector>)
            containing the pointing weights to use.
    """

    def __init__(self, invnpp=None, hits=None, detweights=None, flag_name=None, 
                flag_mask=255, common_flag_name=None, common_flag_mask=255, 
                pixels='pixels', weights='weights'):
        
        self._pixels = pixels
        self._weights = weights
        self._detweights = detweights

        if invnpp is None:
            raise RuntimeError("you must specify the invnpp to accumulate")
        self._invnpp = invnpp

        self._hits = hits

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

        for obs in data.obs:
            tod = obs['tod']
            for det in tod.local_dets:

                # get the pixels and weights from the cache

                pixelsname = "{}_{}".format(self._pixels, det)
                weightsname = "{}_{}".format(self._weights, det)
                pixels = tod.cache.reference(pixelsname)
                weights = tod.cache.reference(weightsname)

                sm, lpix = self._invnpp.global_to_local(pixels)
                
                detweight = 1.0
                
                if self._detweights is not None:
                    if det not in self._detweights.keys():
                        raise RuntimeError("no detector weights found for {}".format(det))
                    detweight = self._detweights[det]
                    if detweight == 0:
                        continue

                if self._hits is not None:
                    _accumulate_inverse_covariance(self._invnpp.data, sm, lpix, weights, detweight, self._hits.data)
                else:
                    fakehits = np.zeros((1,1,1), dtype=np.int64)
                    _accumulate_inverse_covariance(self._invnpp.data, sm, lpix, weights, detweight, fakehits)
        return


class OpNoiseWeighted(Operator):
    """
    Operator which computes the noise-weighted map.

    This operator requires that the local pointing matrix has already been
    computed.  Each process has a local piece of the noise weighted map.

    Args:
        zmap (DistPixels):  The map to accumulate.
        hits (DistPixels):  (optional) the hits to accumulate.
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
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.
        weights (str): the name of the cache object (<weights>_<detector>)
            containing the pointing weights to use.
    """

    def __init__(self, zmap=None, hits=None, detweights=None, name=None, flag_name=None, 
                flag_mask=255, common_flag_name=None, common_flag_mask=255, pixels='pixels', 
                weights='weights'):
        
        self._name = name
        self._pixels = pixels
        self._weights = weights
        self._detweights = detweights

        if zmap is None:
            raise RuntimeError("you must specify the map to accumulate")
        self._zmap = zmap

        self._hits = hits

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

        for obs in data.obs:
            tod = obs['tod']
            for det in tod.local_dets:

                # get the pixels and weights from the cache

                pixelsname = "{}_{}".format(self._pixels, det)
                weightsname = "{}_{}".format(self._weights, det)
                pixels = tod.cache.reference(pixelsname)
                weights = tod.cache.reference(weightsname)

                sm, lpix = self._zmap.global_to_local(pixels)
                
                detweight = 1.0
                
                if self._detweights is not None:
                    if det not in self._detweights.keys():
                        raise RuntimeError("no detector weights found for {}".format(det))
                    detweight = self._detweights[det]
                    if detweight == 0:
                        continue

                if self._hits is not None:
                    _accumulate_inverse_covariance(self._invnpp.data, sm, lpix, weights, detweight, self._hits.data)
                else:
                    fakehits = np.zeros((1,1,1), dtype=np.int64)
                    _accumulate_inverse_covariance(self._invnpp.data, sm, lpix, weights, detweight, fakehits)
        return


def covariance_invert(npp, threshold):
    """
    Invert the local piece of a diagonal noise covariance.

    This does an in-place inversion of the covariance.  The threshold is
    applied to the condition number of each block of the matrix.  Pixels
    failing the cut are set to zero.

    Args:
        npp (3D array): The data member of a distributed covariance.
        threshold (float): The condition number threshold to apply.
    """
    if len(npp.shape) != 3:
        raise RuntimeError("distributed covariance matrix must have dimensions (number of submaps, pixels per submap, nnz*(nnz+1)/2)")
    _invert_covariance(npp, threshold)
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
    _cond_covariance(npp, rcond)
    return rcond
