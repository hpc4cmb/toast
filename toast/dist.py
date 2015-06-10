# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI
from collections import namedtuple

import unittest


class Comm(object):
    """
    Class which represents a two-level hierarchy of MPI communicators.

    A Comm object splits the full set of processes into groups
    of size "group".  If group_size does not divide evenly
    into the size of the given communicator, then those processes
    remain idle.

    A Comm object stores three MPI communicators:  The "world"
    communicator given here, which contains all processes to
    consider, a "group" communicator (one per group)

    Args:
        world (mpi4py.MPI.Comm): the MPI communicator containing all processes.
        group (int): the size of each process group.
    """

    def __init__(self, world=MPI.COMM_WORLD, groupsize=0):

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


def distribute_discrete(sizes, groups):
    totalsize = sum(sizes)
    target = float(totalsize) / float(groups)
    



def distribute_uniform(totalsize, groups):
    """
    Uniformly distribute items between groups.

    Given some number of items and some number of groups,
    distribute the items between groups in the most Uniform
    way possible.

    Args:
        totalsize (int): The total number of items.
        groups (int): The number of groups.

    Returns:
        list of tuples: there is one tuple per group.  The 
        first element of the tuple is the first item 
        assigned to the group, and the second element is 
        the number of items assigned to the group. 
    """
    ret = []
    for i in range(groups):
        myn = totalsize // groups
        off = 0
        leftover = totalsize % groups
        if ( i < leftover ):
            myn = myn + 1
            off = i * myn
        else:
            off = ((myn + 1) * leftover) + (myn * (i - leftover))
        ret.append( (off, myn) )
    return ret


def distribute_det_samples(mpicomm, timedist, detectors, samples):
    dist_dets = detectors
    dist_samples = (0, samples)
    if timedist:
        dist_all = distribute_uniform(samples, mpicomm.size)
        dist_samples = dist_all[mpicomm.rank]
    else:
        dist_detsindx = distribute_uniform(len(detectors), mpicomm.size)[mpicomm.rank]
        dist_dets = detectors[dist_detsindx[0]:dist_detsindx[0]+dist_detsindx[1]]
    return (dist_dets, dist_samples)


Obs = namedtuple('Obs', ['mpicomm', 'streams', 'pointing', 'baselines', 'noise'])

class Dist(object):
    """
    Class which represents distributed data

    A Dist object contains a list of observations assigned to
    each process group in the Comm.

    Args:
        comm (toast.Comm): the toast Comm class for distributing the data.
    """

    def __init__(self, comm=Comm()):

        self.comm = comm
        self.obs = []

