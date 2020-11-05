# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections.abc import MutableMapping

import numpy as np

from .mpi import Comm


class Data(MutableMapping):
    """Class which represents distributed data

    A Data object contains a list of observations assigned to
    each process group in the Comm.

    Args:
        comm (:class:`toast.Comm`): the toast Comm class for distributing the data.

    """

    def __init__(self, comm=Comm()):
        self._comm = comm
        self.obs = []
        """The list of observations.
        """
        self._internal = dict()

    def __getitem__(self, key):
        return self._internal[key]

    def __delitem__(self, key):
        del self._internal[key]

    def __setitem__(self, key, value):
        self._internal[key] = value

    def __iter__(self):
        return iter(self._internal)

    def __len__(self):
        return len(self._internal)

    def __repr__(self):
        val = "<Data with {} Observations:\n".format(len(self.obs))
        for ob in self.obs:
            val += "{}\n".format(ob)
        val += "Metadata:\n"
        val += "{}".format(self._internal)
        val += "\n>"
        return val

    def __del__(self):
        if hasattr(self, "obs"):
            self.clear()

    @property
    def comm(self):
        """The toast.Comm over which the data is distributed."""
        return self._comm

    def clear(self):
        """Clear the list of observations."""
        for ob in self.obs:
            ob.clear()
        self.obs.clear()
        return

    def info(self, handle=None):
        """Print information about the distributed data.

        Information is written to the specified file handle.  Only the rank 0
        process writes.

        Args:
            handle (descriptor):  file descriptor supporting the write()
                method.  If None, use print().

        Returns:
            None

        """
        # Each process group gathers their output

        groupstr = ""
        procstr = ""

        gcomm = self._comm.comm_group
        wcomm = self._comm.comm_world
        rcomm = self._comm.comm_rank

        if wcomm is None:
            msg = "Data distributed over a single process (no MPI)"
            if handle is None:
                print(msg, flush=True)
            else:
                handle.write(msg)
        else:
            if wcomm.rank == 0:
                msg = "Data distributed over {} processes in {} groups\n".format(
                    self._comm.world_size, self._comm.ngroups
                )
                if handle is None:
                    print(msg, flush=True)
                else:
                    handle.write(msg)

        def _get_optional(k, dt):
            if k in dt:
                return dt[k]
            else:
                return None

        for ob in self.obs:
            if self._comm.group_rank == 0:
                groupstr = "{}{}\n".format(groupstr, str(ob))

        # The world rank 0 process collects output from all groups and
        # writes to the handle

        recvgrp = ""
        if self._comm.world_rank == 0:
            if handle is None:
                print(groupstr, flush=True)
            else:
                handle.write(groupstr)
        if wcomm is not None:
            for g in range(1, self._comm.ngroups):
                if wcomm.rank == 0:
                    recvgrp = rcomm.recv(source=g, tag=g)
                    if handle is None:
                        print(recvgrp, flush=True)
                    else:
                        handle.write(recvgrp)
                elif g == self._comm.group:
                    if gcomm.rank == 0:
                        rcomm.send(groupstr, dest=0, tag=g)
                wcomm.barrier()
        return

    def split(self, key):
        """Split the Data object.

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
                raise RuntimeError(
                    'Cannot split data by "{}". Key is not '
                    "defined for all observations.".format(key)
                )
            values.add(obs[key])
        all_values = None
        if self._comm.comm_world is None:
            all_values = [values]
        else:
            all_values = self._comm.comm_world.allgather(values)
        for vals in all_values:
            values = values.union(vals)

        # Order the values alphabetically.
        values = sorted(list(values))

        # Split the data
        datasplit = []
        for value in values:
            new_data = Data(comm=self._comm)
            for obs in self.obs:
                if obs[key] == value:
                    new_data.obs.append(obs)
            datasplit.append((value, new_data))

        return datasplit
