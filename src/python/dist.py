# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPI

import os

import numpy as np
import numpy.testing as nt


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
        if self._ngroups*self._gsize != self._wsize:
            raise RuntimeError('Requested group size ')
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


def distribute_discrete(sizes, groups, pow=1.0, breaks=None):
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
        breaks (list): List of hard breaks in the data distribution.

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

    all_breaks = None
    if breaks is not None:
        # Check that the problem makes sense
        all_breaks = np.unique(breaks)
        all_breaks = all_breaks[all_breaks > 0]
        all_breaks = all_breaks[all_breaks < chunks.size]
        all_breaks = np.sort(all_breaks)
        if all_breaks.size + 1 > groups:
            raise RuntimeError(
                'Cannot divide {} chunks to {} groups with {} breaks.'.format(
                    chunks.size, groups, all_breaks.size))

    at_break = False
    for cur in range(0, weights.shape[0]):
        if curweight + weights[cur] > max_per_proc or at_break:
            dist.append( (off, cur-off) )
            over = curweight - target
            curweight = weights[cur] + over
            off = cur
        else:
            curweight += weights[cur]
        if all_breaks is not None:
            at_break = False
            if cur+1 in all_breaks:
                at_break = True

    dist.append( (off, weights.shape[0]-off) )

    if len(dist) != groups:
        raise RuntimeError("Number of distributed groups different than number requested")

    return dist


def distribute_uniform(totalsize, groups, breaks=None):
    """
    Uniformly distribute items between groups.

    Given some number of items and some number of groups,
    distribute the items between groups in the most Uniform
    way possible.

    Args:
        totalsize (int): The total number of items.
        groups (int): The number of groups.
        breaks (list): List of hard breaks in the data distribution.

    Returns:
        list of tuples: there is one tuple per group.  The
        first element of the tuple is the first item
        assigned to the group, and the second element is
        the number of items assigned to the group.
    """
    if breaks is not None:
        all_breaks = np.unique(breaks)
        all_breaks = all_breaks[all_breaks > 0]
        all_breaks = all_breaks[all_breaks < totalsize]
        all_breaks = np.sort(all_breaks)
        if len(all_breaks) > groups-1:
            raise RuntimeError(
                'Cannot distribute {} chunks with {} breaks over {} groups'
                ''.format(totalsize, len(all_breaks), groups))
        groupcounts = []
        groupsizes = []
        offset = 0
        groupsleft = groups
        totalleft = totalsize
        for brk in all_breaks:
            length = brk - offset
            groupcount = int(np.round(groupsleft*length/totalleft))
            groupcount = max(1, groupcount)
            groupcount = min(groupcount, groupsleft)
            groupcounts.append(groupcount)
            groupsizes.append(length)
            groupsleft -= groupcount
            totalleft -= length
            offset = brk
        groupcounts.append(groupsleft)
        groupsizes.append(totalleft)
    else:
        groupcounts = [groups]
        groupsizes = [totalsize]

    dist = []
    offset = 0
    for groupsize, groupcount in zip(groupsizes, groupcounts):
        for i in range(groupcount):
            myn = groupsize // groupcount
            off = 0
            leftover = groupsize % groupcount
            if ( i < leftover ):
                myn = myn + 1
                off = i * myn
            else:
                off = ((myn + 1) * leftover) + (myn * (i - leftover))
            dist.append((offset+off, myn))
        offset += groupsize

    return dist


def distribute_samples(mpicomm, detectors, samples, detranks=1, detbreaks=None, sampsizes=None, sampbreaks=None):
    """
    Distribute data by detector and sample.

    Given a list of detectors and some number of samples, distribute
    the data in a load balanced way.  Optionally account for constraints
    on this distribution.  The samples may be grouped by indivisible
    chunks, and there may be forced breaks in the distribution in both
    the detector and chunk directions.

                            samples -->
                      +--------------+--------------
                    / | sampsize[0]  | sampsize[1] ...
        detrank = 0   +--------------+--------------
                    \ | sampsize[0]  | sampsize[1] ...
                      +--------------+--------------
                    / | sampsize[0]  | sampsize[1] ...
        detrank = 1   +--------------+--------------
                    \ | sampsize[0]  | sampsize[1] ...
                      +--------------+--------------
                      | ...

    Args:
        mpicomm (mpi4py.MPI.Comm):  the MPI communicator over which the
            data is distributed.
        detectors (list):  The list of detector names.
        samples (int):  The total number of samples.
        detranks (int):  The dimension of the process grid in the detector
            direction.  The MPI communicator size must be evenly divisible
            by this number.
        detbreaks (list):  Optional list of hard breaks in the detector
            distribution.
        sampsizes (list):  Optional list of sample chunk sizes which
            cannot be split.
        sampbreaks (list):  Optional list of hard breaks in the sample
            distribution.

    Returns:
        tuple of lists: the 3 lists returned contain information about
        the detector distribution, the sample distribution, and the chunk
        distribution.  The first list has one entry for each detrank and
        contains the list of detectors for that row of the process grid.
        The second list contains tuples of (first sample, N samples) for
        each column of the process grid.  The third list contains tuples
        of (first chunk, N chunks) for each column of the process grid.

    """
    if mpicomm.size % detranks != 0:
        raise RuntimeError("The number of detranks ({}) does not divide evenly into the communicator size ({})".format(detranks, mpicomm.size))

    # Compute the other dimension of the process grid.

    sampranks = mpicomm.size // detranks

    # Distribute detectors uniformly, but respecting forced breaks in the
    # grouping specified by the calling code.

    dist_detsindx = distribute_uniform(len(detectors), detranks, breaks=detbreaks)
    dist_dets = [ detectors[d[0]:d[0]+d[1]] for d in dist_detsindx ]

    # Distribute samples using both the chunking and the forced breaks

    if sampsizes is not None:
        dist_sizes = distribute_discrete(
            sampsizes, sampranks, breaks=sampbreaks)
        dist_samples = []
        off = 0
        for ds in dist_sizes:
            cursamp = np.sum(sampsizes[ds[0]:ds[0]+ds[1]])
            dist_samples.append( (off, cursamp) )
            off += cursamp
    else:
        dist_samples = distribute_uniform(
            samples, sampranks, breaks=sampbreaks)
        dist_sizes = [ (x, 1) for x in range(sampranks) ]

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
        """
        The list of observations.
        """


    @property
    def comm(self):
        """
        The toast.Comm over which the data is distributed.
        """
        return self._comm


    def clear(self):
        """
        Clear the dictionary of all observations, so that they can be
        garbage collected.
        """
        for ob in self.obs:
            ob.clear()
        return


    def info(self, handle, flag_mask=255, common_flag_mask=255,
             intervals='intervals'):
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
            handle.write("Data distributed over {} processes in {} groups\n"
                         "".format(self._comm.world_size, self._comm.ngroups))

        for ob in self.obs:
            id = ob['id']
            tod = ob['tod']
            base = ob['baselines']
            nse = ob['noise']
            intrvl = ob[intervals]

            if gcomm.rank == 0:
                groupstr = "observation {}:\n".format(id)
                groupstr = "{}  {} total samples, {} detectors\n".format(
                    groupstr, tod.total_samples, len(tod.detectors))
                if intrvl is not None:
                    groupstr = "{}  {} intervals:\n".format(groupstr,
                                                            len(intrvl))
                    for it in intrvl:
                        groupstr = "{}    {} --> {} ({} --> {})\n".format(
                            groupstr, it.first, it.last, it.start, it.stop)

            # rank zero of the group will print general information,
            # and each process will get its statistics.

            offset, nsamp = tod.local_samples
            dets = tod.local_dets

            procstr = "  proc {}\n".format(gcomm.rank)
            my_chunks = 1
            if tod.local_chunks is not None:
                my_chunks = tod.local_chunks[1]
            procstr = "{}    sample range {} --> {} in {} chunks:\n".format(
                procstr, offset, (offset + nsamp - 1), my_chunks)

            if tod.local_chunks is not None:
                chkoff = tod.local_samples[0]
                for chk in range(tod.local_chunks[1]):
                    abschk = tod.local_chunks[0] + chk
                    chkstart = chkoff
                    chkstop = chkstart + tod.total_chunks[abschk] - 1
                    procstr = "{}      {} --> {}\n".format(procstr, chkstart,
                                                           chkstop)
                    chkoff += tod.total_chunks[abschk]

            if nsamp > 0:

                stamps = tod.local_times()

                procstr = "{}    timestamps {} --> {}\n".format(
                    procstr, stamps[0], stamps[-1])

                common = tod.local_common_flags()
                for dt in dets:
                    procstr = "{}    det {}:\n".format(procstr, dt)

                    pdata = tod.local_pointing(dt)

                    procstr = "{}      pntg [{:.3e} {:.3e} {:.3e} {:.3e}] " \
                              "--> [{:.3e} {:.3e} {:.3e} {:.3e}]\n".format(
                                  procstr, pdata[0, 0], pdata[0, 1], pdata[0, 2],
                                  pdata[0, 3], pdata[-1, 0], pdata[-1, 1],
                                  pdata[-1, 2], pdata[-1, 3])

                    data = tod.local_signal(dt)
                    flags = tod.local_flags(dt)
                    procstr = "{}      {:.3e} ({}) --> {:.3e} ({})\n".format(
                        procstr, data[0], flags[0], data[-1], flags[-1])
                    good = np.where(((flags & flag_mask) |
                                     (common & common_flag_mask)) == 0)[0]
                    procstr = "{}        {} good samples\n".format(procstr,
                                                                   len(good))
                    try:
                        min = np.min(data[good])
                        max = np.max(data[good])
                        mean = np.mean(data[good])
                        rms = np.std(data[good])
                        procstr = "{}        min = {:.4e}, max = {:.4e}, " \
                                  "mean = {:.4e}, rms = {:.4e}\n".format(
                                      procstr, min, max, mean, rms)
                    except:
                        procstr = "{}        min = N/A, max = N/A, " \
                                  "mean = N/A, rms = N/A\n".format(procstr)

                for cname in tod.cache.keys():
                    procstr = "{}    cache {}:\n".format(procstr, cname)
                    ref = tod.cache.reference(cname)
                    min = np.min(ref)
                    max = np.max(ref)
                    mean = np.mean(ref)
                    rms = np.std(ref)
                    procstr = "{}        min = {:.4e}, max = {:.4e}, " \
                              "mean = {:.4e}, rms = {:.4e}\n".format(
                                  procstr, min, max, mean, rms)

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


    def split(self, key):
        """
        Split the Data object.

        Split the Data object based on the value of `key` in the
        observation dictionary.

        Args:
            key(str) :  Observation key to use.

        Returns:
            List of 2-tuples of the form (value, data)
        """

        # Build a superset of all values

        values = set()
        for obs in self.obs:
            if key not in obs:
                raise RuntimeError('Cannot split data by "{}". Key is not '
                                   'defined for all observations.'.format(key))
            values.add(obs[key])
        all_values = self._comm.comm_world.allgather(values)
        for vals in all_values:
            values = values.union(vals)

        # Split the data

        datasplit = []
        for value in values:
            new_data = Data(comm=self._comm)
            for obs in self.obs:
                if obs[key] == value:
                    new_data.obs.append(obs)
            datasplit.append((value, new_data))

        return datasplit
