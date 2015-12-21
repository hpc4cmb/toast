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

    def __init__(self, nside=1024, mode='I', detweights=None, RIMO=None, highmem=False):
        self._nside = nside
        self._mode = mode
        self._detweights = detweights
        self._highmem = highmem
        
        if RIMO is None:
            raise ValueError('You must specify which RIMO to use')

        self.RIMO = RIMO # The Reduced Instrument Model contains the necessary detector parameters

        # We call the parent class constructor, which currently does nothing
        super().__init__()


    @property
    def rimo(self):
        return self.RIMO

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

        for obs in data.obs:
            tod = obs['tod']

            for det in tod.local_dets:
                pdata, pflags = tod.read_pntg(detector=det, local_start=0, n=tod.local_samples)

                pdata = pdata.reshape(-1,4).copy()
                pdata[ pflags != 0 ] = nullquat
                vec_dir = qa.rotate( pdata, zaxis ).T.copy()
                
                pixels = hp.vec2pix(self._nside, *vec_dir, nest=True)
                pixels[ pflags != 0 ] = -1

                epsilon = self.RIMO[ det ].epsilon
                eta = (1 - epsilon) / (1 + epsilon)

                dweight = 1.0
                if self._detweights is not None:
                    dweight = self._detweights[det]

                if self._mode == 'I':
                    weights = np.ones(tod.local_samples, dtype=np.float64)
                    weights *= dweight
                    tod.write_pmat(detector=det, local_start=0, pixels=pixels, weights=weights)
                elif self._mode == 'IQU':
                    vec_orient = qa.rotate( pdata, xaxis ).T.copy()

                    ypa = vec_orient[0]*vec_dir[1] - vec_orient[1]*vec_dir[0]
                    #xpa = -vec_orient[0]*vec_dir[2]*vec_dir[0] - vec_orient[1]*vec_dir[2]*vec_dir[1] + vec_orient[2]*(vec_dir[0]**2 + vec_dir[1]**2)
                    xpa = -vec_dir[2]*(vec_orient[0]*vec_dir[0] - vec_orient[1]*vec_dir[1]) + vec_orient[2]*(vec_dir[0]**2 + vec_dir[1]**2)

                    psi = np.arctan2( ypa, xpa )

                    Ival = dweight * np.ones( tod.local_samples )
                    Qval = dweight * eta * np.cos( 2 * psi )
                    Uval = dweight * eta * np.sin( 2 * psi )

                    weights = np.ravel(np.column_stack((Ival, Qval, Uval)))

                    tod.write_pmat(detector=det, local_start=0, pixels=pixels, weights=weights)
                else:
                    raise RuntimeError("invalid mode for Planck Pointing")

                if not self._highmem:
                    tod.clear_pntg(detector=det)
                    
        return


