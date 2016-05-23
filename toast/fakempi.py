# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


# fake MPI functions used for systems without MPI or for cases where
# initializing MPI is not allowed at install time.

import time


ANY_SOURCE = 0
ANY_TAG = 0

OP_NULL = 0
SUM = 0
MAX = 1
MIN = 2


class Status(object):
    def __init__(self):
        self.val = 0


class Op(object):
    def __init__(self):
        self.op = OP_NULL


class Comm(object):
    """
    Class which represents a fake MPI implementation.

    We only implement here the functions that we actually use in
    the source.  The interface is identical to mpi4py.  See that
    documentation for reference.
    """
    def __init__(self):
        self._size = 1
        self._rank = 0
        self._msgs = {}

    @property
    def size(self):
        """
        The size of the communicator.
        """
        return self._size

    @property
    def rank(self):
        """
        The rank within the communicator.
        """
        return self._rank

    
    def Get_size(self):
        """
        The size of the communicator.
        """
        return self._size


    def Get_rank(self):
        """
        The rank within the communicator.
        """
        return self._rank


    def Split(self, color=0, key=0):
        return Comm()


    def Send(self, buf, dest, tag=0):
        """
        Blocking send
        """
        self._msgs[tag] = buf


    def Recv(self, buf, source=ANY_SOURCE, tag=ANY_TAG,
             status=None):
        """
        Blocking receive
        """
        buf = self._msgs[tag]
        return


    def Isend(self, buf, dest, tag=0):
        """
        Nonblocking send
        """
        self.Send(buf, dest, tag)
        return


    def Irecv(self, buf, source=ANY_SOURCE, tag=ANY_TAG):
        """
        Nonblocking receive
        """
        self.Recv(buf, source, tag)
        return


    def Barrier(self):
        """
        Barrier synchronization
        """
        pass


    def Bcast(self, buf, root=0):
        """
        Broadcast a message from one process
        to all other processes in a group
        """
        pass


    def Gather(self, sendbuf, recvbuf, root=0):
        """
        Gather together values from a group of processes
        """
        recvbuf = sendbuf
        return


    def Gatherv(self, sendbuf, recvbuf, root=0):
        """
        Gather Vector, gather data to one process from all other
        processes in a group providing different amount of data and
        displacements at the receiving sides
        """
        recvbuf = sendbuf
        return


    def Scatter(self, sendbuf, recvbuf, root=0):
        """
        Scatter data from one process
        to all other processes in a group
        """
        recvbuf = sendbuf
        return


    def Scatterv(self, sendbuf, recvbuf, root=0):
        """
        Scatter Vector, scatter data from one process to all other
        processes in a group providing different amount of data and
        displacements at the sending side
        """
        recvbuf = sendbuf
        return


    def Allgather(self, sendbuf, recvbuf):
        """
        Gather to All, gather data from all processes and
        distribute it to all other processes in a group
        """
        recvbuf = sendbuf
        return


    def Allgatherv(self, sendbuf, recvbuf):
        """
        Gather to All Vector, gather data from all processes and
        distribute it to all other processes in a group providing
        different amount of data and displacements
        """
        recvbuf = sendbuf
        return


    def Alltoall(self, sendbuf, recvbuf):
        """
        All to All Scatter/Gather, send data from all to all
        processes in a group
        """
        recvbuf = sendbuf
        return


    def Alltoallv(self, sendbuf, recvbuf):
        """
        All to All Scatter/Gather Vector, send data from all to all
        processes in a group providing different amount of data and
        displacements
        """
        recvbuf = sendbuf
        return


    def Reduce(self, sendbuf, recvbuf, op=SUM, root=0):
        """
        Reduce
        """
        recvbuf[:] = sendbuf
        return


    def Allreduce(self, sendbuf, recvbuf, op=SUM):
        """
        All Reduce
        """
        recvbuf[:] = sendbuf
        return


    def Abort(self, errorcode=0):
        """
        Terminate MPI execution environment
        """
        raise RuntimeError("Fake MPI Abort!")


    def send(self, obj, dest, tag=0):
        """Send"""
        self._msgs[tag] = obj
        return


    def recv(self, buf=None, source=ANY_SOURCE, tag=ANY_TAG,
             status=None):
        """Receive"""
        return self._msgs[tag]


    def isend(self, obj, dest, tag=0):
        """Nonblocking send"""
        self._msgs[tag] = obj
        return


    def irecv(self, buf=None, source=ANY_SOURCE, tag=ANY_TAG):
        """Nonblocking receive"""
        return self._msgs[tag]


    def barrier(self):
        """Barrier"""
        pass


    def bcast(self, obj, root=0):
        """Broadcast"""
        return obj


    def gather(self, sendobj, root=0):
        """Gather"""
        return [sendobj]


    def scatter(self, sendobj, root=0):
        """Scatter"""
        return sendobj


    def allgather(self, sendobj):
        """Gather to All"""
        return [sendobj]


    def alltoall(self, sendobj):
        """All to All Scatter/Gather"""
        return sendobj


    def reduce(self, sendobj, op=SUM, root=0):
        """Reduce"""
        return sendobj


    def allreduce(self, sendobj, op=SUM):
        """Reduce to All"""
        return sendobj


COMM_NULL = None

COMM_SELF = Comm()

COMM_WORLD = Comm()


def Wtime():
    return time.time()


def _sizeof(comm=Comm()):
    return 4

