# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest


class Obs(object):
    """
    Class which represents a single observation.
    Each observation contains one instance of each type of class:
    Pointing, Streams, Baselines, and Noise.
    """

    def __init__(self, mpicomm=MPI.COMM_WORLD, streams=None, pointing=None, baselines=None, noise=None ):
        """
        Construct an Obs object given instances of the different
        data classes (or None).

        An Obs class is basically just a container of these data
        classes.

        Args:
            mpicomm: The MPI communicator sharing this observation
            streams: An instance of the tod.streams.Streams class
            pointing: An instance of the tod.pointing.Pointing class
            baselines: An instance of the tod.baselines.Baselines class
            noise: An instance of the fod.noise.Noise class

        Returns:
            Nothing

        Raises:
            Nothing
        """
        self.mpicomm = mpicomm
        self.streams = streams
        self.pointing = pointing
        self.baselines = baselines
        self.noise = noise


