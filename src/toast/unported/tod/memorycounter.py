# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI

from ..operator import Operator

from ..utils import Logger

from ..timing import function_timer


class OpMemoryCounter(Operator):
    """Compute total memory used by TOD objects.

    Operator which loops over the TOD objects and computes the total
    amount of memory allocated.

    Args:
        silent (bool):  Only count and return the memory without
            printing.
        *other_caching_objects:  Additional objects that have a cache
            member and user wants to include in the total counts
            (e.q. DistPixels objects).

    """

    def __init__(self, *other_caching_objects, silent=False):
        self._silent = silent
        self._objects = []
        for obj in other_caching_objects:
            self._objects.append(obj)
        super().__init__()

    @function_timer
    def exec(self, data):
        """
        Count the memory

        Args:
            data (toast.Data): The distributed data.

        """
        log = Logger.get()
        # the two-level pytoast communicator
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world
        # the communicator within the group
        cgroup = comm.comm_group

        tot_task = 0
        for obj in self._objects:
            try:
                tot_task += obj.cache.report(silent=True)
            except:
                pass
            try:
                tot_task += obj._cache.report(silent=True)
            except:
                pass

        for obs in data.obs:
            tod = obs["tod"]
            tot_task += tod.cache.report(silent=True)

        tot_group = tot_task
        tot_world = tot_task
        tot_task_max = tot_task
        tot_group_max = tot_task
        if cworld is not None:
            tot_group = cgroup.allreduce(tot_task, op=MPI.SUM)
            tot_world = cworld.allreduce(tot_task, op=MPI.SUM)
            tot_task_max = cworld.allreduce(tot_task, op=MPI.MAX)
            tot_group_max = cgroup.allreduce(tot_group, op=MPI.MAX)

        if (not self._silent) and (cworld is None or cworld.rank == 0):
            msg = "Memory usage statistics:\n\
                - Max memory (task): {:.2f} GB\n\
                - Max memory (group): {:.2f} GB\n\
                Total memory: {:.2f} GB\n\
                ".format(
                (tot_task_max / 2**30),
                (tot_group_max / 2**30),
                (tot_world / 2**30),
            )
            log.info(msg)

        return tot_world
