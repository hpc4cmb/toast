# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest


class Comm(object):
    """
    Class which represents a two-level hierarchy of MPI communicators.
    """

    def __init__(self, world=MPI.COMM_WORLD, groupsize=0):
        """
        Construct a Comm object given an existing MPI communicator.

        A Comm object splits the full set of processes into groups
        of size "group".  If group_size does not divide evenly
        into the size of the given communicator, then those processes
        remain idle.

        A Comm object stores three MPI communicators:  The "world"
        communicator given here, which contains all processes to
        consider, a "group" communicator (one per group)

        Args:
            world: the MPI communicator containing all processes.
            group: the size of each process group.

        Returns:
            Nothing

        Raises:
            RuntimeError: if the world communicator is not defined
            or the group size is larger than the communicator.
        """

        self.comm_world = world
        if( (groupsize <= 0) or (groupsize > self.comm_world.size) ):
            groupsize = self.comm_world.size
 
        self.group_size = groupsize

        if self.group_size > self.comm_world.size:
            raise RuntimeError('requested group size ({}) is larger than world communicator ({})'.format(self.group_size, self.comm_world.size))

        self.ngroup = int(self.comm_world.size / self.group_size)
        self.group = int(self.comm_world.rank / self.group_size)

        self.group_rank = self.comm_world.rank % self.group_size

        if self.group >= self.ngroup:
            self.group = MPI.UNDEFINED
            self.group_rank = MPI.UNDEFINED

        self.comm_group = self.comm_world.Split(self.group, self.group_rank)
        self.comm_rank = self.comm_world.Split(self.group_rank, self.group)


class Dist(object):
    """
    Class which represents distributed data
    """

    def __init__(self, comm=Comm()):
        """
        Construct a Dist object given a toast Comm.

        A Dist object contains a list of observations assigned to
        each process group in the Comm.

        Args:
            comm: the toast Comm class for distributing the data.

        Returns:
            Nothing

        Raises:
            Nothing
        """

        self.comm = comm
        self.obs = []

