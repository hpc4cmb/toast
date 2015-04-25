# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest


class Obs(object):
    """
    Class which represents a single observation.

    Each observation contains one instance (or None) of each 
    type of class: Pointing, Streams, Baselines, and Noise.

    An Obs class is basically just a container of these data
    classes.

    Args:
        mpicomm (mpi4py.MPI.Comm): The MPI communicator sharing this observation
        streams (toast.tod.Streams): An instance of the Streams class
        pointing (toast.tod.Pointing): An instance of the Pointing class
        baselines (toast.tod.Baselines): An instance of the Baselines class
        noise (toast.tod.Noise): An instance of the Noise class
    """

    def __init__(self, mpicomm=MPI.COMM_WORLD, streams=None, pointing=None, baselines=None, noise=None ):

        self.mpicomm = mpicomm
        self.streams = streams
        self.pointing = pointing
        self.baselines = baselines
        self.noise = noise


