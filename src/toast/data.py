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

    @property
    def comm(self):
        """The toast.Comm over which the data is distributed.
        """
        return self._comm

    def clear(self):
        """Clear the list of observations.
        """
        for ob in self.obs:
            ob.clear()
        self.obs.clear()
        return

    def info(self, handle=None, flag_mask=255, common_flag_mask=255, intervals=None):
        """Print information about the distributed data.

        Information is written to the specified file handle.  Only the rank 0
        process writes.  Optional flag masks are used when computing the
        number of good samples.

        Args:
            handle (descriptor):  file descriptor supporting the write()
                method.  If None, use print().
            flag_mask (int):  bit mask to use when computing the number of
                good detector samples.
            common_flag_mask (int):  bit mask to use when computing the
                number of good telescope pointings.
            intervals (str):  optional name of an intervals object to print
                from each observation.

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
            id = None
            name = None
            try:
                id = ob.UID
                name = ob.name
            except:
                id = ob["id"]
                name = ob["name"]
            tod = _get_optional("tod", ob)
            intrvl = None
            if intervals is not None:
                _get_optional(intervals, ob)

            if self._comm.group_rank == 0:
                groupstr = "observation {} (UID = {}):\n".format(name, id)
                for ko in sorted(ob.keys()):
                    groupstr = "{}  key {}\n".format(groupstr, ko)
                if tod is not None:
                    groupstr = "{}  {} total samples, {} detectors\n".format(
                        groupstr, tod.total_samples, len(tod.detectors)
                    )
                if intrvl is not None:
                    groupstr = "{}  {} intervals:\n".format(groupstr, len(intrvl))
                    for it in intrvl:
                        groupstr = "{}    {} --> {} ({} --> {})\n".format(
                            groupstr, it.first, it.last, it.start, it.stop
                        )

            # rank zero of the group will print general information,
            # and each process will get its statistics.

            procstr = "  proc {}\n".format(self._comm.group_rank)
            if tod is not None:
                offset, nsamp = tod.local_samples
                dets = tod.local_dets

                my_chunks = 1
                if tod.local_chunks is not None:
                    my_chunks = tod.local_chunks[1]
                procstr = "{}    sample range {} --> {} in {} chunks:\n".format(
                    procstr, offset, (offset + nsamp - 1), my_chunks
                )

                if tod.local_chunks is not None:
                    chkoff = tod.local_samples[0]
                    for chk in range(tod.local_chunks[1]):
                        abschk = tod.local_chunks[0] + chk
                        chkstart = chkoff
                        chkstop = chkstart + tod.total_chunks[abschk] - 1
                        procstr = "{}      {} --> {}\n".format(
                            procstr, chkstart, chkstop
                        )
                        chkoff += tod.total_chunks[abschk]

                if nsamp > 0:
                    stamps = tod.local_times()

                    procstr = "{}    timestamps {} --> {}\n".format(
                        procstr, stamps[0], stamps[-1]
                    )

                    common = tod.local_common_flags()
                    for dt in dets:
                        procstr = "{}    det {}:\n".format(procstr, dt)

                        pdata = tod.local_pointing(dt)

                        procstr = (
                            "{}      pntg [{:.3e} {:.3e} {:.3e} {:.3e}] "
                            "--> [{:.3e} {:.3e} {:.3e} {:.3e}]\n".format(
                                procstr,
                                pdata[0, 0],
                                pdata[0, 1],
                                pdata[0, 2],
                                pdata[0, 3],
                                pdata[-1, 0],
                                pdata[-1, 1],
                                pdata[-1, 2],
                                pdata[-1, 3],
                            )
                        )

                        data = tod.local_signal(dt)
                        flags = tod.local_flags(dt)
                        procstr = "{}      {:.3e} ({}) --> {:.3e} ({})\n".format(
                            procstr, data[0], flags[0], data[-1], flags[-1]
                        )
                        good = np.where(
                            ((flags & flag_mask) | (common & common_flag_mask)) == 0
                        )[0]
                        procstr = "{}        {} good samples\n".format(
                            procstr, len(good)
                        )
                        try:
                            min = np.min(data[good])
                            max = np.max(data[good])
                            mean = np.mean(data[good])
                            rms = np.std(data[good])
                            procstr = (
                                "{}        min = {:.4e}, max = {:.4e},"
                                " mean = {:.4e}, rms = {:.4e}\n".format(
                                    procstr, min, max, mean, rms
                                )
                            )
                        except FloatingPointError:
                            procstr = (
                                "{}        min = N/A, max = N/A, "
                                "mean = N/A, rms = N/A\n".format(procstr)
                            )

                    for cname in tod.cache.keys():
                        procstr = "{}    cache {}:\n".format(procstr, cname)
                        ref = tod.cache.reference(cname)
                        min = np.min(ref)
                        max = np.max(ref)
                        mean = np.mean(ref)
                        rms = np.std(ref)
                        procstr = (
                            "{}        min = {:.4e}, max = {:.4e}, "
                            "mean = {:.4e}, rms = {:.4e}\n".format(
                                procstr, min, max, mean, rms
                            )
                        )

            recvstr = ""
            if self._comm.group_rank == 0:
                groupstr = "{}{}".format(groupstr, procstr)
            if gcomm is not None:
                for p in range(1, self._comm.group_size):
                    if gcomm.rank == 0:
                        recvstr = gcomm.recv(source=p, tag=p)
                        groupstr = "{}{}".format(groupstr, recvstr)
                    elif p == gcomm.rank:
                        gcomm.send(procstr, dest=0, tag=p)
                    gcomm.barrier()

        # the world rank 0 process collects output from all groups and
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
