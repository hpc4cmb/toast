# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

import healpy as hp

import quaternionarray as qa

import toast


class OpPointingPlanck(toast.Operator):
    """
    Operator which generates healpix pointing

    Args:
        nside (int): NSIDE resolution for Healpix maps.

    """

    def __init__(self, nside=1024, mode='I', detweights=None):
        self._nside = nside
        self._mode = mode
        self._detweights = detweights

        # We call the parent class constructor, which currently does nothing
        super().__init__()


    @property
    def nside(self):
        return self._nside


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

        xaxis = np.array([1,0,0], dtype=np.float64)
        yaxis = np.array([0,1,0], dtype=np.float64)
        zaxis = np.array([0,0,1], dtype=np.float64)
        nullquat = np.array([0,0,0,1], dtype=np.float64)

        # FIXME: use detweights or noise information

        for obs in data.obs:
            tod = obs['tod']
            for det in tod.local_dets:
                pdata, pflags = tod.read_pntg(detector=det, local_start=0, n=tod.local_samples)

                pdata = np.where((np.repeat(pflags, 4) == 0), pdata, np.tile(nullquat, tod.local_samples))

                dir = qa.rotate(pdata.reshape(-1, 4), np.tile(zaxis, tod.local_samples).reshape(-1,3))
                pixels = hp.vec2pix(self._nside, dir[:,0], dir[:,1], dir[:,2], nest=True)
                pixels = np.where((pflags == 0), pixels, np.repeat(-1, pixels.shape[0]))

                # FIXME: get epsilon
                epsilon = 1.0

                oneplus = 0.5 * (1.0 + epsilon)
                oneminus = 0.5 * (1.0 - epsilon)

                dweight = 1.0
                if self._detweights is not None:
                    dweight = self._detweights[det]

                if self._mode == 'I':
                    weights = np.ones(tod.local_samples, dtype=np.float64)
                    weights *= dweight
                    tod.write_pmat(detector=det, local_start=0, pixels=pixels, weights=weights)
                elif self._mode == 'IQU':
                    orient = qa.rotate(pdata.reshape(-1, 4), np.tile(xaxis, tod.local_samples).reshape(-1,3))
                    y = orient[:,0] * dir[:,1] - orient[:,1] * dir[:,0]
                    x = orient[:,0] * (-dir[:,2] * dir[:,0]) + orient[:,1] * (-dir[:,2] * dir[:,1]) + orient[:,2] * (dir[:,0] * dir[:,0] + dir[:,1] * dir[:,1])
                        
                    detang = np.arctan2(y, x)
                    cang = np.cos(detang)
                    sang = np.sin(detang)
                    
                    Ival = np.zeros_like(cang)
                    Ival[:] = dweight * oneplus
                    Qval = dweight * oneminus * cang
                    Uval = dweight * oneminus * sang

                    weights = np.ravel(np.column_stack((Ival, Qval, Uval)))
                    tod.write_pmat(detector=det, local_start=0, pixels=pixels, weights=weights)
                else:
                    raise RuntimeError("invalid mode for Planck Pointing")
                    
        return


