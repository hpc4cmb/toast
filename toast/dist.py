# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np


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

        self._wcomm = world
        self._wrank = self._wcomm.rank
        self._wsize = self._wcomm.size

        self._gsize = groupsize

        if( (self._gsize < 0) or (self._gsize > self._wsize) ):
            raise ValueError("Invalid groupsize ({}).  Should be between {} and {}.".format(groupsize, 0, self._wsize))

        if self._gsize == 0:
            self._gsize = self._wsize

        self._ngroups = int(self._wsize / self._gsize)
        self._group = int(self._wrank / self._gsize)

        self._grank = self._wrank % self._gsize

        if self._group >= self._ngroups:
            self._group = MPI.UNDEFINED
            self._grank = MPI.UNDEFINED

        self._gcomm = self._wcomm.Split(self._group, self._grank)
        self._rcomm = self._wcomm.Split(self._grank, self._group)

    @property
    def world_size(self):
        """
        The size of the world communicator.
        """
        return self._wsize

    @property
    def world_rank(self):
        """
        The rank of this process in the world communicator.
        """
        return self._wrank

    @property
    def ngroups(self):
        """
        The number of process groups.
        """
        return self._ngroups

    @property
    def group(self):
        """
        The group containing this process.
        """
        return self._group

    @property
    def group_size(self):
        """
        The size of the group containing this process.
        """
        return self._gsize

    @property
    def group_rank(self):
        """
        The rank of this process in the group communicator.
        """
        return self._grank

    @property
    def comm_world(self):
        """
        The world communicator.
        """
        return self._wcomm

    @property
    def comm_group(self):
        """
        The communicator shared by processes within this group.
        """
        return self._gcomm

    @property
    def comm_rank(self):
        """
        The communicator shared by processes with the same group_rank.
        """
        return self._rcomm


# This is effectively the "Painter's Partition Problem".

def distribute_required_groups(A, max_per_group):
    ngroup = 1
    total = 0
    for i in range(A.shape[0]):
        total += A[i]
        if total > max_per_group:
            total = A[i]
            ngroup += 1
    return ngroup

def distribute_partition(A, k):
    low = np.max(A)
    high = np.sum(A)
    while low < high:
        mid = low + int((high - low) / 2)
        required = distribute_required_groups(A, mid)
        if required <= k:
            high = mid
        else:
            low = mid + 1
    return low

def distribute_discrete(sizes, groups, pow=1.0):
    """
    Distribute indivisible blocks of items between groups.

    Given some contiguous blocks of items which cannot be 
    subdivided, distribute these blocks to the specified
    number of groups in a way which minimizes the maximum
    total items given to any group.  Optionally weight the
    blocks by a power of their size when computing the
    distribution.

    Args:
        sizes (list): The sizes of the indivisible blocks.
        groups (int): The number of groups.
        pow (float): The power to use for weighting

    Returns:
        A list of tuples.  There is one tuple per group.  
        The first element of the tuple is the first item 
        assigned to the group, and the second element is 
        the number of items assigned to the group.
    """
    chunks = np.array(sizes, dtype=np.int64)
    weights = np.power(chunks.astype(np.float64), pow)
    max_per_proc = float(distribute_partition(weights.astype(np.int64), groups))

    target = np.sum(weights) / groups

    dist = []

    off = 0
    curweight = 0.0
    proc = 0
    for cur in range(0, weights.shape[0]):
        if curweight + weights[cur] > max_per_proc:
            dist.append( (off, cur-off) )
            over = curweight - target
            curweight = weights[cur] + over
            off = cur
            proc += 1
        else:
            curweight += weights[cur]

    dist.append( (off, weights.shape[0]-off) )

    if len(dist) != groups:
        raise RuntimeError("Number of distributed groups different than number requested")

    return dist


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


def distribute_samples(mpicomm, timedist, detectors, samples, sizes=None):
    dist_dets = detectors
    dist_samples = None
    dist_sizes = None

    if timedist:
        if sizes is not None:
            dist_sizes = distribute_discrete(sizes, mpicomm.size)
            dist_samples = []
            off = 0
            for ds in dist_sizes:
                cursamp = np.sum(sizes[ds[0]:ds[0]+ds[1]])
                dist_samples.append( (off, cursamp) )
                off += cursamp
        else:
            dist_samples = distribute_uniform(samples, mpicomm.size)
    else:
        dist_detsindx = distribute_uniform(len(detectors), mpicomm.size)[mpicomm.rank]
        dist_dets = detectors[dist_detsindx[0]:dist_detsindx[0]+dist_detsindx[1]]
        dist_samples = [ (0, samples) for x in range(mpicomm.size) ]

    return (dist_dets, dist_samples, dist_sizes)


class Data(object):
    """
    Class which represents distributed data

    A Data object contains a list of observations assigned to
    each process group in the Comm.

    Args:
        comm (toast.Comm): the toast Comm class for distributing the data.
    """

    def __init__(self, comm=Comm()):

        self._comm = comm
        self.obs = []


    @property
    def comm(self):
        """
        The toast.Comm over which the data is distributed.
        """
        return self._comm


    def clear(self):
        """
        Clear the dictionary of all observations, so that they can be garbage collected.
        """
        for ob in self.obs:
            ob.clear()
        return


    def info(self, handle):
        """
        Print information about the distributed data to the
        specified file handle.  Only the rank 0 process writes.
        """

        # Each process group gathers their output

        groupstr = ""
        procstr = ""

        gcomm = self._comm.comm_group
        wcomm = self._comm.comm_world
        rcomm = self._comm.comm_rank

        if wcomm.rank == 0:
            handle.write("Data distributed over {} processes in {} groups\n".format(self._comm.world_size, self._comm.ngroups))

        for ob in self.obs:
            id = ob['id']
            tod = ob['tod']
            base = ob['baselines']
            nse = ob['noise']
            intrvl = ob['intervals']

            if gcomm.rank == 0:
                groupstr = "observation {}:\n".format(id)
                groupstr = "{}  {} total samples, {} detectors\n".format(groupstr, tod.total_samples, len(tod.detectors))
                if intrvl is not None:
                    groupstr = "{}  {} intervals:\n".format(groupstr, len(intrvl))
                    for it in intrvl:
                        groupstr = "{}    {} --> {} ({} --> {})\n".format(groupstr, it.first, it.last, it.start, it.stop)

            # rank zero of the group will print general information,
            # and each process will get its statistics.

            nsamp = tod.local_samples[1]
            dets = tod.local_dets

            procstr = "  proc {}\n".format(gcomm.rank)
            my_chunks = 1
            if tod.local_chunks is not None:
                my_chunks = tod.local_chunks[1]
            procstr = "{}    sample range {} --> {} in {} chunks:\n".format(procstr, tod.local_samples[0], (tod.local_samples[0] + nsamp - 1), my_chunks)
            
            if tod.local_chunks is not None:
                chkoff = tod.local_samples[0]
                for chk in range(tod.local_chunks[1]):
                    abschk = tod.local_chunks[0] + chk
                    chkstart = chkoff
                    chkstop = chkstart + tod.total_chunks[abschk] - 1
                    procstr = "{}      {} --> {}\n".format(procstr, chkstart, chkstop)
                    chkoff += tod.total_chunks[abschk]

            if nsamp > 0:
    
                stamps = tod.read_times(local_start=0, n=nsamp)

                procstr = "{}    timestamps {} --> {}\n".format(procstr, stamps[0], stamps[-1])

                for dt in dets:
                    procstr = "{}    det {}:\n".format(procstr, dt)

                    pdata, pflags = tod.read_pntg(detector=dt, local_start=0, n=nsamp)

                    procstr = "{}      pntg [{:.3e} {:.3e} {:.3e} {:.3e}] ({}) --> [{:.3e} {:.3e} {:.3e} {:.3e}] ({})\n".format(procstr, pdata[0], pdata[1], pdata[2], pdata[3], pflags[0], pdata[-4], pdata[-3], pdata[-2], pdata[-1], pflags[-1])
                    good = np.where(pflags == 0)[0]
                    procstr = "{}      {} good pointings\n".format(procstr, len(good))

                    for flv in tod.flavors:
                        data, flags = tod.read(detector=dt, flavor=flv, local_start=0, n=nsamp)
                        procstr = "{}      flavor {}:  {:.3e} ({}) --> {:.3e} ({})\n".format(procstr, flv, data[0], flags[0], data[-1], flags[-1])
                        good = np.where(flags == 0)[0]
                        procstr = "{}        {} good samples\n".format(procstr, len(good))
                        min = np.min(data[good])
                        max = np.max(data[good])
                        mean = np.mean(data[good])
                        rms = np.std(data[good])
                        procstr = "{}        min = {:.4e}, max = {:.4e}, mean = {:.4e}, rms = {:.4e}\n".format(procstr, min, max, mean, rms)

                    for name in tod.pointings:
                        pixels, weights = tod.read_pmat(name=name, detector=dt, local_start=0, n=nsamp)
                        nnz = int(len(weights) / len(pixels))
                        procstr = "{}      pmat {}:\n".format(procstr, name)
                        procstr = "{}        {} : ".format(procstr, pixels[0])
                        for i in range(nnz):
                            procstr = "{} {:.3e}".format(procstr, weights[i])
                        procstr = "{} -->\n".format(procstr)
                        procstr = "{}        {} : ".format(procstr, pixels[-1])
                        for i in range(nnz):
                            procstr = "{} {:.3e}".format(procstr, weights[-(nnz-i)])
                        procstr = "{}\n".format(procstr)
                        procstr = "{}        {} good elements\n".format(procstr, len(np.where(pixels >= 0)[0]))

            recvstr = ""
            if gcomm.rank == 0:
                groupstr = "{}{}".format(groupstr, procstr)
            for p in range(1, gcomm.size):
                if gcomm.rank == 0:
                    recvstr = gcomm.recv(source=p, tag=p)
                    groupstr = "{}{}".format(groupstr, recvstr)
                elif p == gcomm.rank:
                    gcomm.send(procstr, dest=0, tag=p)
                gcomm.barrier()

        # the world rank 0 process collects output from all groups and
        # writes to the handle

        recvgrp = ""
        if wcomm.rank == 0:
            handle.write(groupstr)
        for g in range(1, self._comm.ngroups):
            if wcomm.rank == 0:
                recvgrp = rcomm.recv(source=g, tag=g)
                handle.write(recvgrp)
            elif g == self._comm.group:
                if gcomm.rank == 0:
                    rcomm.send(groupstr, dest=0, tag=g)
            wcomm.barrier()

        return

