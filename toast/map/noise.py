# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

from ..dist import Comm, Data
from ..operator import Operator
from .pixels import DistPixels

from ._helpers import (_accumulate_inverse_covariance,
    _invert_covariance, _cond_covariance)



class OpInvCovariance(Operator):
    """
    Operator which computes the diagonal inverse pixel covariance.

    This operator requires that the local pointing matrix has already been
    computed.  Each process has a local piece of the inverse covariance.

    Args:
        pmat (string):  Name of the pointing matrix to use.
        invnpp (DistPixels):  The matrix to accumulate.
        hits (DistPixels):  (optional) the hits to accumulate.
    """

    def __init__(self, detweights=None, pmat=None, invnpp=None, hits=None):
        
        self._pmat = pmat
        self._detweights = detweights

        if invnpp is None:
            raise RuntimeError("you must specify the invnpp to accumulate")
        self._invnpp = invnpp

        self._hits = hits

        self._nnz = invnpp.nnz
        self._nelem = np.floor_divide((self._nnz * (self._nnz + 1)), 2)

        # We call the parent class constructor, which currently does nothing
        super().__init__()


    def exec(self, data):
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
                pixels, weights = tod.read_pmat(name=self._pmat, detector=det, local_start=0, n=tod.local_samples[1])
                sm, lpix = self._invnpp.global_to_local(pixels)
                wt = weights.reshape(pixels.shape[0], -1)
                detweight = 1.0
                if self._detweights is not None:
                    if det not in self._detweights.keys():
                        raise RuntimeError("no detector weights found for {}".format(det))
                    detweight = self._detweights[det]
                if self._hits is not None:
                    _accumulate_inverse_covariance(self._invnpp.data, sm, lpix, wt, detweight, self._hits.data)
                else:
                    fakehits = np.zeros((1,1,1), dtype=np.int64)
                    _accumulate_inverse_covariance(self._invnpp.data, sm, lpix, wt, detweight, fakehits)
        return


def covariance_invert(npp, threshold):
    if len(npp.shape) != 3:
        raise RuntimeError("distributed covariance matrix must have dimensions (number of submaps, pixels per submap, nnz*(nnz+1)/2)")
    _invert_covariance(npp, threshold)
    return


def covariance_rcond(npp):
    if len(npp.shape) != 3:
        raise RuntimeError("distributed covariance matrix must have dimensions (number of submaps, pixels per submap, nnz*(nnz+1)/2)")
    rcond = np.zeros((npp.shape[0], npp.shape[1], 1), dtype=np.float64)
    _cond_covariance(npp, rcond)
    return rcond

    