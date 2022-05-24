# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import psutil
from astropy import units as u

from .mpi import MPI
from .utils import Logger


def get_node_mem(world_comm, node_rank):
    """Find the minimum available memory across all nodes.

    Args:
        world_comm (MPI.Comm):  The MPI world communicator (or None).
        node_rank (int):  The rank on this node.

    Returns:
        (int):  The minimum number of bytes of available node memory.

    """
    min_avail = 2**62
    max_avail = 0
    if node_rank == 0:
        vmem = psutil.virtual_memory()._asdict()
        min_avail = vmem["available"]
        max_avail = vmem["available"]
    if world_comm is not None:
        min_avail = world_comm.allreduce(min_avail, op=MPI.MIN)
        max_avail = world_comm.allreduce(max_avail, op=MPI.MAX)
    return int(min_avail), int(max_avail)


def job_size(world_comm):
    log = Logger.get()

    procs_per_node = 1
    node_rank = 0
    rank = 0
    procs = 1

    if world_comm is not None:
        rank = world_comm.rank
        procs = world_comm.size
        node_comm = world_comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
        node_rank = node_comm.rank
        procs_per_node = node_comm.size
        min_per_node = world_comm.allreduce(procs_per_node, op=MPI.MIN)
        max_per_node = world_comm.allreduce(procs_per_node, op=MPI.MAX)
        if min_per_node != max_per_node:
            raise RuntimeError("Nodes have inconsistent numbers of MPI ranks")
        node_comm.Free()
        del node_comm

    n_node = procs // procs_per_node

    # One process on each node gets available RAM and communicates it
    min_avail, max_avail = get_node_mem(world_comm, node_rank)

    return (n_node, procs_per_node, min_avail, max_avail)


def job_group_size(
    world_comm,
    job_args,
    schedule=None,
    obs_times=None,
    focalplane=None,
    num_dets=None,
    sample_rate=None,
    det_copies=2,
    full_pointing=False,
):
    """Given parameters about the job, estimate the best group size.

    Using the provided information, this function determines the distribution of MPI
    processes across nodes and selects the group size to be a whole number of nodes
    with the following criteria:

        - The nodes per group divides evenly into the total number of nodes.

        - The nodes in one group have enough total memory to fit the largest
          observation.

        - If possible, the observations are load balanced (in terms of memory) across
          the groups.

    Args:

    Returns:
        (int):  The best number of processes in a group.

    """
    log = Logger.get()
    gb = 1024**3

    rank = 0
    procs = 1

    if world_comm is not None:
        rank = world_comm.rank
        procs = world_comm.size

    # If the user has manually specifed the group size, just return that.
    if job_args.group_size is not None:
        if rank == 0:
            log.info(
                "Job plan: user-specified group size set to {}".format(
                    job_args.group_size
                )
            )
        return job_args.group_size

    n_det = None
    rate = None
    if focalplane is not None:
        n_det = len(focalplane.detector_data)
        rate = focalplane.sample_rate.to_value(u.Hz)
    else:
        if num_dets is None or sample_rate is None:
            msg = "specify either a focalplane or both the number of detectors and rate"
            log.error(msg)
            raise RuntimeError(msg)
        n_det = num_dets
        rate = sample_rate.to_value(u.Hz)

    obs_len = list()
    if schedule is not None:
        for sc in schedule.scans:
            obs_len.append((sc.stop - sc.start).total_seconds())
    elif obs_times is not None:
        for start, stop in obs_times:
            obs_len.append((stop - start).total_seconds())
    else:
        msg = "specify either a schedule or a list of observation start / stop tuples"
        log.error(msg)
        raise RuntimeError(msg)
    obs_len = list(sorted(obs_len, reverse=True))

    # Get memory available

    n_node, procs_per_node, min_avail, max_avail = job_size(world_comm)

    node_mem = min_avail
    if job_args.node_mem is not None:
        node_mem = job_args.node_mem

    def observation_mem(seconds, nodes):
        m = 0
        # Assume 2 copies of node-shared boresight pointing
        m += 2 * 4 * 8 * rate * seconds * nodes
        # Assume 8 byte floats per det sample
        m += det_copies * n_det * 8 * rate * seconds
        return m

    group_nodes = 1
    n_group = n_node // group_nodes
    group_mem = group_nodes * node_mem
    obs_mem = [observation_mem(x, group_nodes) for x in obs_len]

    if rank == 0:
        log.info(
            "Job has {} nodes each with {} processes ({} total)".format(
                n_node, procs_per_node, procs
            )
        )
        log.info(
            "Job nodes have {:02f}GB / {:02f}GB available memory (min / max)".format(
                min_avail / gb, max_avail / gb
            )
        )
        if job_args.node_mem is not None:
            log.info(
                "Job plan using user-specified node memory of {:02f}GB".format(
                    node_mem / gb
                )
            )
        else:
            log.info(
                "Job plan using minimum node memory of {:02f}GB".format(node_mem / gb)
            )
        log.info(
            "Job observation lengths = {} minutes / {} minutes (min / max)".format(
                int(obs_len[-1] / 60), int(obs_len[0] / 60)
            )
        )

    if rank == 0:
        log.verbose(
            "Job test {} nodes per group, largest obs = {:0.2f}GB".format(
                group_nodes, obs_mem[0] / gb
            )
        )

    while (group_nodes < n_node) and (
        (group_mem < obs_mem[0]) or (n_group > len(obs_len))
    ):
        # The group size cannot fit the largest observation
        # Increase to the next valid value.
        try_group = group_nodes + 1
        while (try_group < n_node) and (n_node % try_group != 0):
            try_group += 1
        group_nodes = try_group
        n_group = n_node // group_nodes
        group_mem = group_nodes * node_mem
        obs_mem = [observation_mem(x, group_nodes) for x in obs_len]
        if rank == 0:
            log.verbose(
                "Job test {} nodes per group, largest obs = {:0.2f}GB".format(
                    group_nodes, obs_mem[0] / gb
                )
            )

    if rank == 0:
        log.info("Job selecting {} nodes per group".format(group_nodes))

    total_mem = np.sum(obs_mem)
    if total_mem > n_node * node_mem:
        msg = "Sum of observation memory use ({:0.2f}GB) greater than job total ({:0.2f}GB)".format(
            total_mem / gb, (n_node * node_mem) / gb
        )
        log.error(msg)
        raise RuntimeError(msg)

    return group_nodes * procs_per_node
