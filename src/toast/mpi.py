# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import sys
import itertools

import numpy as np

from ._libtoast import Logger

from .pshmem import MPIShared, MPILock

use_mpi = None
MPI = None

if use_mpi is None:
    # See if the user has explicitly disabled MPI.
    if "MPI_DISABLE" in os.environ:
        use_mpi = False
    else:
        # Special handling for running on a NERSC login node.  This is for convenience.
        # The same behavior could be implemented with environment variables set in a
        # shell resource file.
        at_nersc = False
        if "NERSC_HOST" in os.environ:
            at_nersc = True
        in_slurm = False
        if "SLURM_JOB_NAME" in os.environ:
            in_slurm = True
        if (not at_nersc) or in_slurm:
            try:
                import mpi4py.MPI as MPI

                use_mpi = True
            except:
                # There could be many possible exceptions raised...
                from ._libtoast import Logger

                log = Logger.get()
                log.info("mpi4py not found- using serial operations only")
                use_mpi = False


def get_world():
    """Retrieve the default world communicator and its properties.

    If MPI is enabled, this returns MPI.COMM_WORLD and the process rank and number of
    processes.  If MPI is disabled, this returns None for the communicator, zero
    for the rank, and one for the number of processes.

    Returns:
        (tuple):  The communicator, number of processes, and rank.

    """
    rank = 0
    procs = 1
    world = None
    if use_mpi:
        world = MPI.COMM_WORLD
        rank = world.rank
        procs = world.size
    return world, procs, rank


class Comm(object):
    """Class which represents a two-level hierarchy of MPI communicators.

    A Comm object splits the full set of processes into groups of size
    "group".  If group_size does not divide evenly into the size of the given
    communicator, then those processes remain idle.

    A Comm object stores three MPI communicators:  The "world" communicator
    given here, which contains all processes to consider, a "group"
    communicator (one per group), and a "rank" communicator which contains the
    processes with the same group-rank across all groups.

    If MPI is not enabled, then all communicators are set to None.

    Args:
        world (mpi4py.MPI.Comm): the MPI communicator containing all processes.
        group (int): the size of each process group.

    """

    def __init__(self, world=None, groupsize=0):
        log = Logger.get()
        if world is None:
            if use_mpi:
                # Default is COMM_WORLD
                world = MPI.COMM_WORLD
            else:
                # MPI is disabled, leave world as None.
                pass
        else:
            if use_mpi:
                # We were passed a communicator to use. Check that it is
                # actually a communicator, otherwise fall back to COMM_WORLD.
                if not isinstance(world, MPI.Comm):
                    log.warning(
                        "Specified world communicator is not a valid "
                        "mpi4py.MPI.Comm object.  Using COMM_WORLD."
                    )
                    world = MPI.COMM_WORLD
            else:
                log.warning(
                    "World communicator specified even though "
                    "MPI is disabled.  Ignoring this constructor "
                    "argument."
                )
                world = None
            # Special case, MPI available but the user want a serial
            # data object
            if world == MPI.COMM_SELF:
                world = None

        self._wcomm = world
        self._wrank = 0
        self._wsize = 1
        if self._wcomm is not None:
            self._wrank = self._wcomm.rank
            self._wsize = self._wcomm.size

        self._gsize = groupsize

        if (self._gsize < 0) or (self._gsize > self._wsize):
            log.warning(
                "Invalid groupsize ({}).  Should be between {} "
                "and {}.  Using single process group instead.".format(
                    groupsize, 0, self._wsize
                )
            )
            self._gsize = 0

        if self._gsize == 0:
            self._gsize = self._wsize

        self._ngroups = self._wsize // self._gsize

        if self._ngroups * self._gsize != self._wsize:
            msg = (
                "World communicator size ({}) is not evenly divisible "
                "by requested group size ({}).".format(self._wsize, self._gsize)
            )
            log.error(msg)
            raise RuntimeError(msg)

        self._group = self._wrank // self._gsize
        self._grank = self._wrank % self._gsize

        if self._ngroups == 1:
            # We just have one group with all processes.
            self._gcomm = self._wcomm
            if use_mpi:
                self._rcomm = MPI.COMM_SELF
            else:
                self._rcomm = None
        else:
            # We need to split the communicator.  This code is never executed
            # unless MPI is enabled and we have multiple groups.
            self._gcomm = self._wcomm.Split(self._group, self._grank)
            self._rcomm = self._wcomm.Split(self._grank, self._group)

    @property
    def world_size(self):
        """The size of the world communicator."""
        return self._wsize

    @property
    def world_rank(self):
        """The rank of this process in the world communicator."""
        return self._wrank

    @property
    def ngroups(self):
        """The number of process groups."""
        return self._ngroups

    @property
    def group(self):
        """The group containing this process."""
        return self._group

    @property
    def group_size(self):
        """The size of the group containing this process."""
        return self._gsize

    @property
    def group_rank(self):
        """The rank of this process in the group communicator."""
        return self._grank

    @property
    def comm_world(self):
        """The world communicator."""
        return self._wcomm

    @property
    def comm_group(self):
        """The communicator shared by processes within this group."""
        return self._gcomm

    @property
    def comm_rank(self):
        """The communicator shared by processes with the same group_rank."""
        return self._rcomm

    def __repr__(self):
        lines = [
            "  World MPI communicator = {}".format(self._wcomm),
            "  World MPI size = {}".format(self._wsize),
            "  World MPI rank = {}".format(self._wrank),
            "  Group MPI communicator = {}".format(self._gcomm),
            "  Group MPI size = {}".format(self._gsize),
            "  Group MPI rank = {}".format(self._grank),
            "  Rank MPI communicator = {}".format(self._rcomm),
        ]
        return "<toast.Comm\n{}\n>".format("\n".join(lines))
