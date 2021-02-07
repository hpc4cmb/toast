# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections.abc import MutableMapping

import numpy as np

from .mpi import Comm


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
    """Distribute indivisible blocks of items between groups.

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
                "Cannot divide {} chunks to {} groups with {} breaks.".format(
                    chunks.size, groups, all_breaks.size
                )
            )

    at_break = False
    for cur in range(0, weights.shape[0]):
        if curweight + weights[cur] > max_per_proc or at_break:
            dist.append((off, cur - off))
            over = curweight - target
            curweight = weights[cur] + over
            off = cur
        else:
            curweight += weights[cur]
        if all_breaks is not None:
            at_break = False
            if cur + 1 in all_breaks:
                at_break = True

    dist.append((off, weights.shape[0] - off))

    if len(dist) != groups:
        raise RuntimeError(
            "Number of distributed groups different than " "number requested"
        )
    return dist


def distribute_uniform(totalsize, groups, breaks=None):
    """Uniformly distribute items between groups.

    Given some number of items and some number of groups,
    distribute the items between groups in the most Uniform
    way possible.

    Args:
        totalsize (int): The total number of items.
        groups (int): The number of groups.
        breaks (list): List of hard breaks in the data distribution.

    Returns:
        (list): there is one tuple per group.  The first element of the tuple
            is the first item assigned to the group, and the second element is
            the number of items assigned to the group.

    """
    if breaks is not None:
        all_breaks = np.unique(breaks)
        all_breaks = all_breaks[all_breaks > 0]
        all_breaks = all_breaks[all_breaks < totalsize]
        all_breaks = np.sort(all_breaks)
        if len(all_breaks) > groups - 1:
            raise RuntimeError(
                "Cannot distribute {} chunks with {} breaks over {} groups"
                "".format(totalsize, len(all_breaks), groups)
            )
        groupcounts = []
        groupsizes = []
        offset = 0
        groupsleft = groups
        totalleft = totalsize
        for brk in all_breaks:
            length = brk - offset
            groupcount = int(np.round(groupsleft * length / totalleft))
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
            if i < leftover:
                myn = myn + 1
                off = i * myn
            else:
                off = ((myn + 1) * leftover) + (myn * (i - leftover))
            dist.append((offset + off, myn))
        offset += groupsize
    return dist


def distribute_samples(
    mpicomm, detectors, samples, detranks=1, detsets=None, sampsets=None
):
    """Distribute data by detector and sample.

    Given a list of detectors and some number of samples, distribute the data in a load
    balanced way.  Optionally account for constraints on this distribution.  Both the
    detectors and the samples may be arranged into "sets" that must not be split
    between processes.

                            samples -->
                      +--------------+--------------
                    / | sampset[0]  | sampset[1] ...
        detrank = 0   +--------------+--------------
                    \ | sampset[0]  | sampset[1] ...
                      +--------------+--------------
                    / | sampset[0]  | sampset[1] ...
        detrank = 1   +--------------+--------------
                    \ | sampset[0]  | sampset[1] ...
                      +--------------+--------------
                      | ...

    Args:
        mpicomm (mpi4py.MPI.Comm):  the MPI communicator over which the data is
            distributed.  If None, then all data is assigned to a single process.
        detectors (list):  The list of detector names.
        samples (int):  The total number of samples.
        detranks (int):  The dimension of the process grid in the detector direction.
            The MPI communicator size must be evenly divisible by this number.
        detsets (list):  Optional list of lists of detectors that must not be split up
            between process rows.
        sampsets (list):  Optional list of lists of sample chunks that must not be
            split up between process columns

    Returns:
        (tuple):  4 lists are returned, and each has an entry for every process.  The
            first list entries are the (first det, n_det) for each process.  The
            second list is the (first det set, n_detset) for each process.  The third
            list is the (first sample, n_sample) for each process and the last list
            is the (first sample set, n_sample_set) for each process.

    """
    nproc = 1
    if mpicomm is not None:
        nproc = mpicomm.size

    if nproc % detranks != 0:
        raise RuntimeError(
            "The number of detranks ({}) does not divide evenly "
            "into the number of processes ({})".format(detranks, nproc)
        )

    # Compute the other dimension of the process grid.
    sampranks = nproc // detranks

    # Distribute detectors either by set or uniformly.

    dist_dets = None
    dist_detsets = None

    if detsets is None:
        # Uniform distribution
        dist_detsindx = distribute_uniform(len(detectors), detranks)
        dist_dets = [detectors[d[0] : d[0] + d[1]] for d in dist_detsindx]
    else:
        # Distribute by det set
        detsizes = [len(x) for x in detsets]
        dist_detsets = distribute_discrete(detsizes, detranks)
        dist_dets = list()
        for set_off, n_set in dist_detsets:
            cur = list()
            for ds in range(n_set):
                cur.extend(detsets[set_off + ds])
            dist_dets.append(cur)

    # Distribute samples either uniformly or by set.

    dist_samples = None
    dist_chunks = None

    if sampsets is None:
        dist_samples = distribute_uniform(samples, sampranks)
        dist_chunks = None
    else:
        sampsetsizes = [np.sum(x) for x in sampsets]
        dist_sampsets = distribute_discrete(sampsetsizes, sampranks)
        dist_chunks = list()
        dist_samples = list()
        samp_off = 0
        chunk_off = 0
        for set_off, n_set in dist_sampsets:
            setsamp = 0
            setchunk = 0
            for ds in range(n_set):
                sset = sampsets[set_off + ds]  # One sample set
                setsamp += np.sum(sset)
                setchunk += len(sset)
            dist_chunks.append((chunk_off, setchunk))
            dist_samples.append((samp_off, setsamp))
            samp_off += setsamp
            chunk_off += setchunk

    return (dist_dets, dist_detsets, dist_samples, dist_chunks)
