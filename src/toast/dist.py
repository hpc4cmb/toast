# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections.abc import MutableMapping
from typing import NamedTuple

import numpy as np

from .mpi import Comm
from .timing import function_timer


class DistRange(NamedTuple):
    """The offset and number of elements in a distribution range."""

    offset: int
    n_elem: int


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


@function_timer
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
        (list):  A list of DistRange tuples, one per group.

    """
    chunks = None
    weights = None
    max_per_proc = None
    target = None
    if len(sizes) < groups:
        msg = (
            f"Too many groups: cannot distribute {len(sizes)} blocks between "
            f"{groups} groups."
        )
        raise RuntimeError(msg)
    elif len(sizes) == groups:
        # One chunk per group, trivial solution
        dist = [DistRange(off, 1) for off in range(len(sizes))]
    elif groups == 1:
        # Only one group, with all chunks
        dist = [DistRange(0, len(sizes))]
    else:
        chunks = np.array(sizes, dtype=np.int64)
        weights = np.power(chunks.astype(np.float64), pow)
        max_per_proc = float(distribute_partition(weights.astype(np.int64), groups))
        target = np.sum(weights) / groups

        dist = []

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

        off = 0
        curweight = 0.0
        at_break = False
        for cur in range(0, weights.shape[0]):
            if curweight + weights[cur] > max_per_proc or at_break:
                dist.append(DistRange(off, cur - off))
                over = curweight - target
                curweight = weights[cur] + over
                off = cur
            else:
                curweight += weights[cur]
            if all_breaks is not None:
                at_break = False
                if cur + 1 in all_breaks:
                    at_break = True

        dist.append(DistRange(off, weights.shape[0] - off))

    if len(dist) != groups:
        msg = (
            f"Number of distributed groups ({len(dist)}) different than number "
            f"requested ({groups})."
        )
        msg += f"  sizes={sizes}, groups={groups}, pow={pow}, breaks={breaks}, "
        msg += f"dist={dist}, chunks={chunks}, weights={weights}, "
        msg += f"max_per_proc={max_per_proc}, target={target}"
        raise RuntimeError(msg)

    return dist


@function_timer
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
        (list):  A list of DistRange tuples, one per group.

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
            dist.append(DistRange(offset + off, myn))
        offset += groupsize
    return dist


@function_timer
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
            first list contains a list of detector names for each process.  The second
            list contains a list of detector sets for each process.  The third list
            contains the DistRange for the samples assigned to each process.  The
            fourth list contains the DistRange for the sample sets assigned to each
            process.

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

    dist_dets_col = None
    dist_detsets_col = None

    if detsets is None:
        # Uniform distribution
        dist_detsindx = distribute_uniform(len(detectors), detranks)
        dist_dets_col = [detectors[d[0] : d[0] + d[1]] for d in dist_detsindx]
    else:
        # Distribute by det set
        detsizes = [len(x) for x in detsets]
        dist_detsets_col = distribute_discrete(detsizes, detranks)
        dist_dets_col = list()
        for set_range in dist_detsets_col:
            cur = list()
            for ds in range(set_range.n_elem):
                cur.extend(detsets[set_range.offset + ds])
            dist_dets_col.append(cur)

    # Distribute samples either uniformly or by set.

    dist_samples_row = None
    dist_chunks_row = None

    if sampsets is None:
        dist_samples_row = distribute_uniform(samples, sampranks)
    else:
        sampsetsizes = [np.sum(x) for x in sampsets]
        dist_sampsets_row = distribute_discrete(sampsetsizes, sampranks)
        dist_chunks_row = list()
        dist_samples_row = list()
        samp_off = 0
        chunk_off = 0
        for set_off, n_set in dist_sampsets_row:
            setsamp = 0
            setchunk = 0
            for ds in range(n_set):
                sset = sampsets[set_off + ds]  # One sample set
                setsamp += np.sum(sset)
                setchunk += len(sset)
            dist_chunks_row.append(DistRange(chunk_off, setchunk))
            dist_samples_row.append(DistRange(samp_off, setsamp))
            samp_off += setsamp
            chunk_off += setchunk

    # Replicate the detector distribution across all process columns
    dist_dets = list()
    dist_detsets = None
    if dist_detsets_col is not None:
        dist_detsets = list()
    dist_samples = list()
    dist_chunks = None
    if dist_chunks_row is not None:
        dist_chunks = list()
    for r in range(detranks):
        for c in range(sampranks):
            dist_dets.append(dist_dets_col[r])
            if dist_detsets is not None:
                dist_detsets.append(dist_detsets_col[r])
            dist_samples.append(dist_samples_row[c])
            if dist_chunks is not None:
                dist_chunks.append(dist_chunks_row[c])

    return (dist_dets, dist_detsets, dist_samples, dist_chunks)
