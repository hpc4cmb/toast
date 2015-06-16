# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

from ..dist import distribute_det_samples

from ..tod.pointing import Pointing


class PointingPlanckEFF(object):
    """
    Provide pointing for Planck Exchange File Format data.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the data is distributed.
        timedist (bool): if True, the data is distributed by time, otherwise by
                  detector.
        detectors (list): list of names to use for the detectors. Must match the names in the FITS HDUs.
        ringdb: Path to an SQLite database defining ring boundaries.
        effdir: directory containing the exchange files
        obt_range: data span in TAI seconds, overrides ring_range
        ring_range: data span in pointing periods, overrides od_range
        od_range: data span in operational days
    """

    def __init__(self, mpicomm=MPI.COMM_WORLD, timedist=True, detectors=None, fn_ringdb=None, effdir=None, obt_range=None, ring_range=None, od_range=None, freq=None):
        
        if detectors is None:
            raise ValueError('you must specify a list of detectors')

        if fn_ringdb is None:
            raise ValueError('You must provide a path to the ring database')

        if effdir is None:
            raise ValueError('You must provide a path to the exchange files')
        
        if freq is None:
            raise ValueError('You must set specify the frequency to run on')

        if obt_range is None and ring_range is None and od_range is None:
            raise ValueError('Cannot initialize EFF streams without one of obt_range, ring_range or od_range')

        # We call the parent class constructor to set the MPI communicator and
        # distribution type, but we do NOT pass the detector list, as this 
        # would allocate memory for the data buffer of the base class.
        super().__init__(mpicomm=mpicomm, timedist=timedist, detectors=None, flavors=None, samples=0)



    def _get(self, detector, start, n):
        data = np.zeros(n, dtype=np.float64)
        flags = np.zeros(n, dtype=np.uint8)
        return (data, flags)


    def _put(self, detector, start, data, flags):
        return


