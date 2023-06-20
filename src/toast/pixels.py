# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
from astropy import units as u
from pshmem.utils import mpi_data_type

from ._libtoast import global_to_local as libtoast_global_to_local
from .accelerator import (
    AcceleratorObject,
    accel_data_create,
    accel_data_delete,
    accel_data_present,
    accel_data_reset,
    accel_data_update_device,
    accel_data_update_host,
    accel_enabled,
    use_accel_jax,
    use_accel_omp,
)
from .dist import distribute_uniform
from .mpi import MPI
from .timing import GlobalTimers, Timer, function_timer
from .utils import (
    AlignedF32,
    AlignedF64,
    AlignedI8,
    AlignedI16,
    AlignedI32,
    AlignedI64,
    AlignedU8,
    AlignedU16,
    AlignedU32,
    AlignedU64,
    Logger,
)


class PixelDistribution(AcceleratorObject):
    """Class representing the distribution of submaps.

    This object is used to describe the properties of a pixelization scheme and which
    "submaps" are strored on each process.  The size of the submap can be tuned to
    balance storage (smaller submap size means fewer wasted pixels stored) and ease of
    indexing (larger submap size means faster global-to-local pixel lookups).

    Args:
        n_pix (int): the total number of pixels.
        n_submap (int): the number of submaps to use.
        local_submaps (array): the list of local submaps (integers).
        comm (mpi4py.MPI.Comm): The MPI communicator or None.

    """

    def __init__(self, n_pix=None, n_submap=1000, local_submaps=None, comm=None):
        super().__init__()
        self._n_pix = n_pix
        self._n_submap = n_submap
        if self._n_submap > self._n_pix:
            msg = "Cannot create a PixelDistribution with more submaps ({}) than pixels ({})".format(
                n_submap, n_pix
            )
            raise RuntimeError(msg)
        self._n_pix_submap = self._n_pix // self._n_submap
        if self._n_pix % self._n_submap != 0:
            self._n_pix_submap += 1

        self._local_submaps = local_submaps
        self._comm = comm

        self._glob2loc = None
        self._n_local = 0

        if self._local_submaps is not None and len(self._local_submaps) > 0:
            if np.max(self._local_submaps) > self._n_submap - 1:
                raise RuntimeError("local submap indices out of range")
            self._n_local = len(self._local_submaps)
            self._glob2loc = AlignedI64.zeros(self._n_submap)
            self._glob2loc[:] = -1
            for ilocal_submap, iglobal_submap in enumerate(self._local_submaps):
                self._glob2loc[iglobal_submap] = ilocal_submap

        self._submap_owners = None
        self._owned_submaps = None
        self._alltoallv_info = None
        self._all_hit_submaps = None

    def __eq__(self, other):
        local_eq = True
        if self._n_pix != other._n_pix:
            local_eq = False
        if self._n_submap != other._n_submap:
            local_eq = False
        if self._n_pix_submap != other._n_pix_submap:
            local_eq = False
        if not np.array_equal(self._local_submaps, other._local_submaps):
            local_eq = False
        if self._comm is None and other._comm is not None:
            local_eq = False
        if self._comm is not None and other._comm is None:
            local_eq = False
        if self._comm is not None:
            comp = MPI.Comm.Compare(self._comm, other._comm)
            if comp not in (MPI.IDENT, MPI.CONGRUENT):
                local_eq = False
        return local_eq

    def __ne__(self, other):
        return not self.__eq__(other)

    def clear(self):
        """Delete the underlying memory.

        This will forcibly delete the C-allocated memory and invalidate all python
        references to this object.  DO NOT CALL THIS unless you are sure all references
        are no longer being used and you are about to delete the object.

        """
        if hasattr(self, "_glob2loc"):
            if self._glob2loc is not None:
                self._glob2loc.clear()
                del self._glob2loc

    def __del__(self):
        self.clear()

    @property
    def comm(self):
        """(mpi4py.MPI.Comm): The MPI communicator used (or None)"""
        return self._comm

    @property
    def n_pix(self):
        """(int): The global number of pixels."""
        return self._n_pix

    @property
    def n_pix_submap(self):
        """(int): The number of pixels in each submap."""
        return self._n_pix_submap

    @property
    def n_submap(self):
        """(int): The total number of submaps."""
        return self._n_submap

    @property
    def n_local_submap(self):
        """(int): The number of submaps stored on this process."""
        return self._n_local

    @property
    def local_submaps(self):
        """(array): The list of local submaps or None if process has no data."""
        return self._local_submaps

    @property
    def all_hit_submaps(self):
        """(array): The list of submaps local to atleast one process."""
        if self._all_hit_submaps is None:
            hits = np.zeros(self._n_submap)
            hits[self._local_submaps] += 1
            if self._comm is not None:
                self._comm.Allreduce(MPI.IN_PLACE, hits)
            self._all_hit_submaps = np.argwhere(hits != 0).ravel()
        return self._all_hit_submaps

    @property
    def global_submap_to_local(self):
        """(array): The mapping from global submap to local."""
        return self._glob2loc

    @function_timer
    def global_pixel_to_submap(self, gl):
        """Convert global pixel indices into the local submap and pixel.

        Args:
            gl (array): The global pixel numbers.

        Returns:
            (tuple):  A tuple of arrays containing the local submap index (int) and the
                pixel index local to that submap (int).

        """
        if len(gl) == 0:
            return (np.zeros_like(gl), np.zeros_like(gl))
        if np.max(gl) >= self._n_pix:
            log = Logger.get()
            msg = "Global pixel indices exceed the maximum for the pixelization"
            log.error(msg)
            raise RuntimeError(msg)
        return libtoast_global_to_local(gl, self._n_pix_submap, self._glob2loc)

        # global_sm = np.floor_divide(gl, self._n_pix_submap, dtype=np.int64)
        # submap_pixel = np.mod(gl, self._n_pix_submap, dtype=np.int64)
        # local_sm = np.array([self._glob2loc[x] for x in global_sm], dtype=np.int64)
        # return (local_sm, submap_pixel)

    @function_timer
    def global_pixel_to_local(self, gl):
        """Convert global pixel indices into local pixel indices.

        Args:
            gl (array): The global pixel numbers.

        Returns:
            (array): The local raw (flat packed) buffer index for each pixel.

        """
        if len(gl) == 0:
            return np.zeros_like(gl)
        if np.max(gl) >= self._n_pix:
            log = Logger.get()
            msg = "Global pixel indices exceed the maximum for the pixelization"
            log.error(msg)
            raise RuntimeError(msg)
        local_sm, pixels = libtoast_global_to_local(
            gl, self._n_pix_submap, self._glob2loc
        )
        local_sm *= self._n_pix_submap
        pixels += local_sm
        return pixels

    def __repr__(self):
        val = "<PixelDistribution {} pixels, {} submaps, submap size = {}>".format(
            self._n_pix, self._n_submap, self._n_pix_submap
        )
        return val

    @property
    def submap_owners(self):
        """The owning process for every hit submap.

        This information is used in several other operations, including serializing
        PixelData objects to a single process and also communication needed for
        reducing data globally.
        """
        if self._submap_owners is not None:
            # Already computed
            return self._submap_owners

        self._submap_owners = np.empty(self._n_submap, dtype=np.int32)
        self._submap_owners[:] = -1

        if self._comm is None:
            # Trivial case
            if self._local_submaps is not None and len(self._local_submaps) > 0:
                self._submap_owners[self._local_submaps] = 0
        else:
            # Need to compute it.
            local_hit_submaps = np.zeros(self._n_submap, dtype=np.uint8)
            local_hit_submaps[self._local_submaps] = 1

            hit_submaps = None
            if self._comm.rank == 0:
                hit_submaps = np.zeros(self._n_submap, dtype=np.uint8)

            self._comm.Reduce(local_hit_submaps, hit_submaps, op=MPI.LOR, root=0)
            del local_hit_submaps

            if self._comm.rank == 0:
                total_hit_submaps = np.sum(hit_submaps.astype(np.int32))
                tdist = distribute_uniform(total_hit_submaps, self._comm.size)

                # The target number of submaps per process
                target = [x[1] for x in tdist]

                # Assign the submaps in rank order.  This ensures better load
                # distribution when serializing some operations and also reduces needed
                # memory copies when using Alltoallv.
                proc_offset = 0
                proc = 0
                for sm in range(self._n_submap):
                    if hit_submaps[sm] > 0:
                        self._submap_owners[sm] = proc
                        proc_offset += 1
                        if proc_offset >= target[proc]:
                            proc += 1
                            proc_offset = 0
                del hit_submaps

            self._comm.Bcast(self._submap_owners, root=0)
        return self._submap_owners

    @property
    def owned_submaps(self):
        """The submaps owned by this process."""
        if self._owned_submaps is not None:
            # Already computed
            return self._owned_submaps
        owners = self.submap_owners
        if self._comm is None:
            self._owned_submaps = np.array(
                [x for x, y in enumerate(owners) if y == 0], dtype=np.int32
            )
        else:
            self._owned_submaps = np.array(
                [x for x, y in enumerate(owners) if y == self._comm.rank],
                dtype=np.int32,
            )
        return self._owned_submaps

    @property
    def alltoallv_info(self):
        """Return the offset information for Alltoallv communication.

        This returns a tuple containing:
            - The send displacements for the Alltoallv submap gather
            - The send counts for the Alltoallv submap gather
            - The receive displacements for the Alltoallv submap gather
            - The receive counts for the Alltoallv submap gather
            - The locations in the receive buffer of each submap.

        """
        log = Logger.get()
        if self._alltoallv_info is not None:
            # Already computed
            return self._alltoallv_info

        owners = self.submap_owners
        our_submaps = self.owned_submaps

        send_counts = None
        send_displ = None
        recv_counts = None
        recv_displ = None
        recv_locations = None

        if self._comm is None:
            recv_counts = len(self._local_submaps) * np.ones(1, dtype=np.int32)
            recv_displ = np.zeros(1, dtype=np.int32)
            recv_locations = dict()
            for offset, sm in enumerate(self._local_submaps):
                recv_locations[sm] = np.array([offset], dtype=np.int32)
            send_counts = len(self._local_submaps) * np.ones(1, dtype=np.int32)
            send_displ = np.zeros(1, dtype=np.int32)
        else:
            # Compute the other "contributing" processes that have submaps which we own.
            # Also track the receive buffer offsets for each owned submap.
            send = [list() for x in range(self._comm.size)]
            for sm in self._local_submaps:
                # Tell the owner of this submap that we are a contributor
                send[owners[sm]].append(sm)
            recv = self._comm.alltoall(send)

            recv_counts = np.zeros(self._comm.size, dtype=np.int32)
            recv_displ = np.zeros(self._comm.size, dtype=np.int32)
            recv_locations = dict()

            offset = 0
            for proc, sms in enumerate(recv):
                recv_displ[proc] = offset
                for sm in sms:
                    if sm not in recv_locations:
                        recv_locations[sm] = list()
                    recv_locations[sm].append(offset)
                    recv_counts[proc] += 1
                    offset += 1

            for sm in list(recv_locations.keys()):
                recv_locations[sm] = np.array(recv_locations[sm], dtype=np.int32)

            # Compute the Alltoallv send offsets in terms of submaps
            send_counts = np.zeros(self._comm.size, dtype=np.int32)
            send_displ = np.zeros(self._comm.size, dtype=np.int32)
            offset = 0
            last_offset = 0
            last_own = -1
            for sm in self._local_submaps:
                if last_own != owners[sm]:
                    # Moving on to next owning process...
                    if last_own >= 0:
                        send_displ[last_own] = last_offset
                        last_offset = offset
                send_counts[owners[sm]] += 1
                offset += 1
                last_own = owners[sm]
            if last_own >= 0:
                # Finish up last process
                send_displ[last_own] = last_offset

        self._alltoallv_info = (
            send_counts,
            send_displ,
            recv_counts,
            recv_displ,
            recv_locations,
        )

        msg_rank = 0
        if self._comm is not None:
            msg_rank = self._comm.rank
        msg = f"alltoallv_info[{msg_rank}]:\n"
        msg += f"  send_counts={send_counts} "
        msg += f"send_displ={send_displ}\n"
        msg += f"  recv_counts={recv_counts} "
        msg += f"recv_displ={recv_displ} "
        msg += f"recv_locations={recv_locations}"
        log.verbose(msg)

        return self._alltoallv_info

    def _accel_exists(self):
        return accel_data_present(self._glob2loc, self._accel_name)

    def _accel_create(self):
        self._glob2loc = accel_data_create(self._glob2loc, self._accel_name)

    def _accel_update_device(self):
        self._glob2loc = accel_data_update_device(self._glob2loc, self._accel_name)

    def _accel_update_host(self):
        self._glob2loc = accel_data_update_host(self._glob2loc, self._accel_name)

    def _accel_reset(self):
        accel_data_reset(self._glob2loc, self._accel_name)

    def _accel_delete(self):
        self._glob2loc = accel_data_delete(self._glob2loc, self._accel_name)


class PixelData(AcceleratorObject):
    """Distributed map-domain data.

    The distribution information is stored in a PixelDistribution instance passed to
    the constructor.  Each process has local data stored in one or more "submaps".

    Although multiple processes may have the same submap of data stored locally, only
    one process is considered the "owner".  This ownership is used when serializing the
    data and when doing reductions in certain cases.  Ownership can be set to either
    the lowest rank process which has the submap or to a balanced distribution.

    Args:
        dist (PixelDistribution):  The distribution of submaps.
        dtype (numpy.dtype):  A numpy-compatible dtype for each element of the data.
            The only supported types are 1, 2, 4, and 8 byte signed and unsigned
            integers, 4 and 8 byte floating point numbers, and 4 and 8 byte complex
            numbers.
        n_value (int):  The number of values per pixel.
        units (Unit):  The units of the map data.

    """

    def __init__(self, dist, dtype, n_value=1, units=u.dimensionless_unscaled):
        super().__init__()
        log = Logger.get()

        self._dist = dist
        self._n_value = n_value
        self._units = units

        # construct a new dtype in case the parameter given is shortcut string
        ttype = np.dtype(dtype)

        self.storage_class = None
        if ttype.char == "b":
            self.storage_class = AlignedI8
        elif ttype.char == "B":
            self.storage_class = AlignedU8
        elif ttype.char == "h":
            self.storage_class = AlignedI16
        elif ttype.char == "H":
            self.storage_class = AlignedU16
        elif ttype.char == "i":
            self.storage_class = AlignedI32
        elif ttype.char == "I":
            self.storage_class = AlignedU32
        elif (ttype.char == "q") or (ttype.char == "l"):
            self.storage_class = AlignedI64
        elif (ttype.char == "Q") or (ttype.char == "L"):
            self.storage_class = AlignedU64
        elif ttype.char == "f":
            self.storage_class = AlignedF32
        elif ttype.char == "d":
            self.storage_class = AlignedF64
        elif ttype.char == "F":
            raise NotImplementedError("No support yet for complex numbers")
        elif ttype.char == "D":
            raise NotImplementedError("No support yet for complex numbers")
        else:
            msg = "Unsupported data typecode '{}'".format(ttype.char)
            log.error(msg)
            raise ValueError(msg)
        self._dtype = ttype

        self.mpitype = None
        self.mpibytesize = None
        if self._dist.comm is not None:
            self.mpibytesize, self.mpitype = mpi_data_type(self._dist.comm, self._dtype)

        self._shape = (
            self._dist.n_local_submap,
            self._dist.n_pix_submap,
            self._n_value,
        )
        self._flatshape = (
            self._dist.n_local_submap * self._dist.n_pix_submap * self._n_value
        )
        self._n_submap_value = self._dist.n_pix_submap * self._n_value

        self.raw = self.storage_class.zeros(self._flatshape)
        self.data = self.raw.array().reshape(self._shape)

        # Allreduce quantities
        self._all_comm_submap = None
        self._all_send = None
        self._all_send_raw = None
        self._all_recv = None
        self._all_recv_raw = None

        # Alltoallv quantities
        self._send_counts = None
        self._send_displ = None
        self._recv_counts = None
        self._recv_displ = None
        self._recv_locations = None
        self.receive = None
        self._receive_raw = None
        self.reduce_buf = None
        self._reduce_buf_raw = None

    def clear(self):
        """Delete the underlying memory.

        This will forcibly delete the C-allocated memory and invalidate all python
        references to this object.  DO NOT CALL THIS unless you are sure all references
        are no longer being used and you are about to delete the object.

        """
        if hasattr(self, "data"):
            # we keep the attribute to avoid errors in _accel_exists
            self.data = None
        if hasattr(self, "raw"):
            if self.accel_exists():
                self.accel_delete()
            if self.raw is not None:
                self.raw.clear()
            del self.raw
        if hasattr(self, "receive"):
            del self.receive
            if self._receive_raw is not None:
                self._receive_raw.clear()
            del self._receive_raw
        if hasattr(self, "reduce_buf"):
            del self.reduce_buf
            if self._reduce_buf_raw is not None:
                self._reduce_buf_raw.clear()
            del self._reduce_buf_raw
        if hasattr(self, "_all_send"):
            del self._all_send
            if self._all_send_raw is not None:
                self._all_send_raw.clear()
            del self._all_send_raw
        if hasattr(self, "_all_recv"):
            del self._all_recv
            if self._all_recv_raw is not None:
                self._all_recv_raw.clear()
            del self._all_recv_raw

    def __del__(self):
        self.clear()

    def reset(self):
        """Set memory to zero"""
        self.raw[:] = 0
        if self.accel_exists():
            self.accel_reset()

    @property
    def distribution(self):
        """(PixelDistribution): The distribution information."""
        return self._dist

    @property
    def dtype(self):
        """(numpy.dtype): The data type of the values."""
        return self._dtype

    @property
    def n_value(self):
        """(int): The number of non-zero values per pixel."""
        return self._n_value

    @property
    def units(self):
        """(Unit):  The map data units."""
        return self._units

    def update_units(self, new_units):
        """Update the units associated with the data."""
        self._units = new_units

    def __getitem__(self, key):
        return np.array(self.data[key], dtype=self._dtype, copy=False)

    def __delitem__(self, key):
        raise NotImplementedError("Cannot delete individual memory elements")
        return

    def __setitem__(self, key, value):
        self.data[key] = value

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        val = (
            "<PixelData {} values per pixel, dtype = {}, units= {}, dist = {}>".format(
                self._n_value, self._dtype, self._units, self._dist
            )
        )
        return val

    def __eq__(self, other):
        if self.distribution != other.distribution:
            return False
        if self.dtype.char != other.dtype.char:
            return False
        if self.n_value != other.n_value:
            return False
        if not np.allclose(self.data, other.data):
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def duplicate(self):
        """Create a copy of the data with the same distribution.

        Returns:
            (PixelData):  A duplicate of the instance with copied data but the same
                distribution.

        """
        dup = PixelData(
            self.distribution, self.dtype, n_value=self.n_value, units=self._units
        )
        dup.data[:] = self.data
        return dup

    def comm_nsubmap(self, bytes):
        """Given a buffer size, compute the number of submaps to communicate.

        Args:
            bytes (int):  The number of bytes.

        Returns:
            (int):  The number of submaps in each buffer.

        """
        dbytes = self._dtype.itemsize
        nsub = int(bytes / (dbytes * self._n_submap_value))
        if nsub == 0:
            nsub = 1
        allsub = int(self._dist.n_pix / self._dist.n_pix_submap)
        if nsub > allsub:
            nsub = allsub
        return nsub

    @function_timer
    def setup_allreduce(self, n_submap):
        """Check that allreduce buffers exist and create them if needed."""
        # Allocate persistent send / recv buffers that only change if number of submaps
        # change.
        if self._all_comm_submap is not None:
            # We have already done a reduce...
            if n_submap == self._all_comm_submap:
                # Already allocated with correct size
                return
            else:
                # Delete the old buffers.
                del self._all_send
                self._all_send_raw.clear()
                del self._all_send_raw
                del self._all_recv
                self._all_recv_raw.clear()
                del self._all_recv_raw
        # Allocate with the new size
        self._all_comm_submap = n_submap
        self._all_recv_raw = self.storage_class.zeros(n_submap * self._n_submap_value)
        self._all_recv = self._all_recv_raw.array()
        self._all_send_raw = self.storage_class.zeros(n_submap * self._n_submap_value)
        self._all_send = self._all_send_raw.array()

    @function_timer
    def sync_allreduce(self, comm_bytes=10000000):
        """Perform a buffered allreduce of the data.

        Args:
            comm_bytes (int): The approximate message size to use.

        Returns:
            None.

        """
        if self.accel_in_use():
            msg = f"PixelData {self._accel_name} currently on accelerator"
            msg += " cannot do MPI communication"
            raise RuntimeError(msg)

        if self._dist.comm is None:
            return

        comm_submap = self.comm_nsubmap(comm_bytes)
        self.setup_allreduce(comm_submap)

        dist = self._dist
        nsub = dist.n_submap

        sendview = self._all_send.reshape(
            (comm_submap, dist.n_pix_submap, self._n_value)
        )

        recvview = self._all_recv.reshape(
            (comm_submap, dist.n_pix_submap, self._n_value)
        )

        owners = dist.submap_owners

        submap_off = 0
        ncomm = comm_submap

        gt = GlobalTimers.get()

        while submap_off < nsub:
            if submap_off + ncomm > nsub:
                ncomm = nsub - submap_off
            if np.sum(owners[submap_off : submap_off + ncomm]) != -ncomm:
                # At least one submap has some hits.  Do the allreduce.
                # Otherwise we would skip this buffer to avoid reducing a
                # bunch of zeros.
                for c in range(ncomm):
                    glob = submap_off + c
                    if glob in dist.local_submaps:
                        # copy our data in.
                        loc = dist.global_submap_to_local[glob]
                        sendview[c, :, :] = self.data[loc, :, :]

                gt.start("PixelData.sync_allreduce MPI Allreduce")
                dist.comm.Allreduce(self._all_send, self._all_recv, op=MPI.SUM)
                gt.stop("PixelData.sync_allreduce MPI Allreduce")

                for c in range(ncomm):
                    glob = submap_off + c
                    if glob in dist.local_submaps:
                        # copy the reduced data
                        loc = dist.global_submap_to_local[glob]
                        self.data[loc, :, :] = recvview[c, :, :]

                self._all_send.fill(0)
                self._all_recv.fill(0)

            submap_off += ncomm

        return

    @staticmethod
    def local_reduction(n_submap_value, receive_locations, receive, reduce_buf):
        # Locally reduce owned submaps
        for sm, locs in receive_locations.items():
            reduce_buf[:] = 0
            for lc in locs:
                reduce_buf += receive[lc : lc + n_submap_value]
            for lc in locs:
                receive[lc : lc + n_submap_value] = reduce_buf

    @function_timer
    def setup_alltoallv(self):
        """Check that alltoallv buffers exist and create them if needed."""
        if self._send_counts is None:
            if self.accel_in_use():
                msg = f"PixelData {self._accel_name} currently on accelerator"
                msg += " cannot do MPI communication"
                raise RuntimeError(msg)
            log = Logger.get()
            # Get the parameters in terms of submaps.
            (
                send_counts,
                send_displ,
                recv_counts,
                recv_displ,
                recv_locations,
            ) = self._dist.alltoallv_info

            # Pixel values per submap
            scale = self._n_submap_value

            # Scale these quantites by the submap size and the number of values per
            # pixel.

            self._send_counts = scale * np.array(send_counts, dtype=np.int32)
            self._send_displ = scale * np.array(send_displ, dtype=np.int32)
            self._recv_counts = scale * np.array(recv_counts, dtype=np.int32)
            self._recv_displ = scale * np.array(recv_displ, dtype=np.int32)
            self._recv_locations = dict()
            for sm, locs in recv_locations.items():
                self._recv_locations[sm] = scale * np.array(locs, dtype=np.int32)

            msg_rank = 0
            if self._dist.comm is not None:
                msg_rank = self._dist.comm.rank
            msg = f"setup_alltoallv[{msg_rank}]:\n"
            msg += f"  send_counts={self._send_counts} "
            msg += f"send_displ={self._send_displ}\n"
            msg += f"  recv_counts={self._recv_counts} "
            msg += f"recv_displ={self._recv_displ} "
            msg += f"recv_locations={self._recv_locations}"
            log.verbose(msg)

            # Allocate a persistent single-submap buffer
            self._reduce_buf_raw = self.storage_class.zeros(self._n_submap_value)
            self.reduce_buf = self._reduce_buf_raw.array()

            buf_check_fail = False
            try:
                if self._dist.comm is None:
                    # For this case, point the receive member to the original data.
                    # This will allow codes processing locally owned submaps to work
                    # transparently in the serial case.
                    self.receive = self.data.reshape((-1,))
                else:
                    # Check that our send and receive buffers do not exceed 32bit
                    # indices required by MPI
                    max_int = 2147483647
                    recv_buf_size = self._recv_displ[-1] + self._recv_counts[-1]
                    if recv_buf_size > max_int:
                        msg = "Proc {} Alltoallv receive buffer size exceeds max 32bit integer".format(
                            self._dist.comm.rank
                        )
                        log.error(msg)
                        buf_check_fail = True
                    if len(self.raw) > max_int:
                        msg = "Proc {} Alltoallv send buffer size exceeds max 32bit integer".format(
                            self._dist.comm.rank
                        )
                        log.error(msg)
                        buf_check_fail = True

                    # Allocate a persistent receive buffer
                    msg = f"{msg_rank}:  allocate receive buffer of "
                    msg += f"{recv_buf_size} elements"
                    log.verbose(msg)
                    self._receive_raw = self.storage_class.zeros(recv_buf_size)
                    self.receive = self._receive_raw.array()
            except:
                buf_check_fail = True
            if self._dist.comm is not None:
                buf_check_fail = self._dist.comm.allreduce(buf_check_fail, op=MPI.LOR)
            if buf_check_fail:
                msg = "alltoallv buffer setup failed on one or more processes"
                raise RuntimeError(msg)

    @function_timer
    def forward_alltoallv(self):
        """Communicate submaps into buffers on the owning process.

        On the first call, some initialization is done to compute send and receive
        displacements and counts.  A persistent receive buffer is allocated.  Submap
        data is sent to their owners simultaneously using alltoallv.

        Returns:
            None.

        """
        if self.accel_in_use():
            msg = f"PixelData {self._accel_name} currently on accelerator"
            msg += " cannot do MPI communication"
            raise RuntimeError(msg)

        log = Logger.get()
        gt = GlobalTimers.get()
        self.setup_alltoallv()

        if self._dist.comm is None:
            # No communication needed
            return

        # Gather owned submaps locally
        gt.start("PixelData.forward_alltoallv MPI Alltoallv")
        self._dist.comm.Alltoallv(
            [self.raw, self._send_counts, self._send_displ, self.mpitype],
            [self.receive, self._recv_counts, self._recv_displ, self.mpitype],
        )
        gt.stop("PixelData.forward_alltoallv MPI Alltoallv")
        return

    @function_timer
    def reverse_alltoallv(self):
        """Communicate submaps from the owning process back to all processes.

        Returns:
            None.

        """
        if self.accel_in_use():
            msg = f"PixelData {self._accel_name} currently on accelerator"
            msg += " cannot do MPI communication"
            raise RuntimeError(msg)

        gt = GlobalTimers.get()
        if self._dist.comm is None:
            # No communication needed
            return
        if self._send_counts is None:
            raise RuntimeError(
                "Cannot do reverse alltoallv before buffers have been setup"
            )

        # Scatter result back
        gt.start("PixelData.reverse_alltoallv MPI Alltoallv")
        self._dist.comm.Alltoallv(
            [self.receive, self._recv_counts, self._recv_displ, self.mpitype],
            [self.raw, self._send_counts, self._send_displ, self.mpitype],
        )
        gt.stop("PixelData.reverse_alltoallv MPI Alltoallv")
        return

    @function_timer
    def sync_alltoallv(self, local_func=None):
        """Perform operations on locally owned submaps using Alltoallv communication.

        On the first call, some initialization is done to compute send and receive
        displacements and counts.  A persistent receive buffer is allocated.  Submap
        data is sent to their owners simultaneously using alltoallv.  Each process does
        a local operation on their owned submaps before sending the result back with
        another alltoallv call.

        Args:
            local_func (function):  A function for processing the local submap data.

        Returns:
            None.

        """
        self.forward_alltoallv()

        if local_func is None:
            local_func = self.local_reduction

        # Run operation on locally owned submaps
        local_func(
            self._n_submap_value, self._recv_locations, self.receive, self.reduce_buf
        )

        self.reverse_alltoallv()
        return

    @function_timer
    def stats(self, comm_bytes=10000000):
        """Compute some simple statistics of the pixel data.

        The map should already be consistent across all processes with overlapping
        submaps.

        Args:
            comm_bytes (int): The approximate message size to use.

        Returns:
            (dict):  The computed properties on rank zero, None on other ranks.

        """
        if self.accel_in_use():
            msg = f"PixelData {self._accel_name} currently on accelerator"
            msg += " cannot do MPI communication"
            raise RuntimeError(msg)
        dist = self._dist
        nsub = dist.n_submap

        if dist.comm is None:
            return {
                "sum": [np.sum(self.data[:, :, x]) for x in range(self._n_value)],
                "mean": [np.mean(self.data[:, :, x]) for x in range(self._n_value)],
                "rms": [np.std(self.data[:, :, x]) for x in range(self._n_value)],
            }

        # The lowest rank with a locally-hit submap will contribute to the reduction
        local_hit_submaps = dist.comm.size * np.ones(nsub, dtype=np.int32)
        local_hit_submaps[dist.local_submaps] = dist.comm.rank
        hit_submaps = np.zeros_like(local_hit_submaps)
        dist.comm.Allreduce(local_hit_submaps, hit_submaps, op=MPI.MIN)
        del local_hit_submaps

        comm_submap = self.comm_nsubmap(comm_bytes)

        send_buf = np.zeros(
            comm_submap * dist.n_pix_submap * self._n_value,
            dtype=self._dtype,
        ).reshape((comm_submap, dist.n_pix_submap, self._n_value))

        recv_buf = None
        if dist.comm.rank == 0:
            # Alloc receive buffer
            recv_buf = np.zeros(
                comm_submap * dist.n_pix_submap * self._n_value,
                dtype=self._dtype,
            ).reshape((comm_submap, dist.n_pix_submap, self._n_value))
            # Variables for variance calc
            accum_sum = np.zeros(self._n_value, dtype=np.float64)
            accum_count = np.zeros(self._n_value, dtype=np.int64)
            accum_mean = np.zeros(self._n_value, dtype=np.float64)
            accum_var = np.zeros(self._n_value, dtype=np.float64)

        # Doing a two-pass variance calculation is faster than looping
        # over individual samples in python.

        submap_off = 0
        ncomm = comm_submap

        while submap_off < nsub:
            if submap_off + ncomm > nsub:
                ncomm = nsub - submap_off
            send_buf[:, :, :] = 0
            for sm in range(ncomm):
                abs_sm = submap_off + sm
                if hit_submaps[abs_sm] == dist.comm.rank:
                    # Contribute
                    loc = dist.global_submap_to_local[abs_sm]
                    send_buf[sm, :, :] = self.data[loc, :, :]

            dist.comm.Reduce(send_buf, recv_buf, op=MPI.SUM, root=0)

            if dist.comm.rank == 0:
                for sm in range(ncomm):
                    for v in range(self._n_value):
                        accum_sum[v] += np.sum(recv_buf[sm, :, v])
                        accum_count[v] += dist.n_pix_submap
            dist.comm.barrier()
            submap_off += ncomm

        if dist.comm.rank == 0:
            for v in range(self._n_value):
                accum_mean[v] = accum_sum[v] / accum_count[v]

        submap_off = 0
        ncomm = comm_submap

        while submap_off < nsub:
            if submap_off + ncomm > nsub:
                ncomm = nsub - submap_off
            send_buf[:, :, :] = 0
            for sm in range(ncomm):
                abs_sm = submap_off + sm
                if hit_submaps[abs_sm] == dist.comm.rank:
                    # Contribute
                    loc = dist.global_submap_to_local[abs_sm]
                    send_buf[sm, :, :] = self.data[loc, :, :]

            dist.comm.Reduce(send_buf, recv_buf, op=MPI.SUM, root=0)

            if dist.comm.rank == 0:
                for sm in range(ncomm):
                    for v in range(self._n_value):
                        accum_var[v] += np.sum(
                            (recv_buf[sm, :, v] - accum_mean[v]) ** 2
                        )
            dist.comm.barrier()
            submap_off += ncomm

        if dist.comm.rank == 0:
            return {
                "sum": [float(accum_sum[x]) for x in range(self._n_value)],
                "mean": [float(accum_mean[x]) for x in range(self._n_value)],
                "rms": [
                    np.sqrt(accum_var[x] / (accum_count[x] - 1))
                    for x in range(self._n_value)
                ],
            }
        else:
            return None

    @function_timer
    def broadcast_map(self, fdata, comm_bytes=10000000):
        """Distribute a map located on a single process.

        The root process takes a map in memory and distributes it.   Chunks of submaps
        are broadcast to all processes, and each process copies data to its local
        submaps.

        Args:
            fdata (array): The input data (only significant on process 0).
            comm_bytes (int): The approximate message size to use.

        Returns:
            None

        """
        if self.accel_in_use():
            msg = f"PixelData {self._accel_name} currently on accelerator"
            msg += " cannot do MPI communication"
            raise RuntimeError(msg)
        rank = 0
        if self._dist.comm is not None:
            rank = self._dist.comm.rank
        comm_submap = self.comm_nsubmap(comm_bytes)

        # we make the assumption that FITS binary tables are still stored in
        # blocks of 2880 bytes just like always...
        dbytes = self._dtype.itemsize
        rowbytes = self._n_value * dbytes
        optrows = int(2880 / rowbytes)

        # Get a tuple of all columns in the table.  We choose memmap here so
        # that we can (hopefully) read through all columns in chunks such that
        # we only ever have a couple FITS blocks in memory.
        if rank == 0:
            if self._n_value == 1:
                fdata = (fdata,)

        buf = np.zeros(
            comm_submap * self._dist.n_pix_submap * self._n_value, dtype=self._dtype
        )
        view = buf.reshape((comm_submap, self._dist.n_pix_submap, self._n_value))

        in_off = 0
        out_off = 0
        submap_off = 0

        rows = optrows
        while in_off < self._dist.n_pix:
            if in_off + rows > self._dist.n_pix:
                rows = self._dist.n_pix - in_off
            # is this the last block for this communication?
            islast = False
            copyrows = rows
            if out_off + rows > (comm_submap * self._dist.n_pix_submap):
                copyrows = (comm_submap * self._dist.n_pix_submap) - out_off
                islast = True

            if rank == 0:
                for col in range(self._n_value):
                    coloff = (out_off * self._n_value) + col
                    buf[
                        coloff : coloff + (copyrows * self._n_value) : self._n_value
                    ] = fdata[col][in_off : in_off + copyrows]

            out_off += copyrows
            in_off += copyrows

            if islast:
                if self._dist.comm is not None:
                    self._dist.comm.Bcast(buf, root=0)
                # loop over these submaps, and copy any that we are assigned
                for sm in range(submap_off, submap_off + comm_submap):
                    if sm in self._dist.local_submaps:
                        loc = self._dist.global_submap_to_local[sm]
                        self.data[loc, :, :] = view[sm - submap_off, :, :]
                out_off = 0
                submap_off += comm_submap
                buf.fill(0)
                islast = False

        # flush the remaining buffer

        if out_off > 0:
            if self._dist.comm is not None:
                self._dist.comm.Bcast(buf, root=0)
            # loop over these submaps, and copy any that we are assigned
            for sm in range(submap_off, submap_off + comm_submap):
                if sm in self._dist.local_submaps:
                    loc = self._dist.global_submap_to_local[sm]
                    self.data[loc, :, :] = view[sm - submap_off, :, :]
        return

    def _accel_exists(self):
        if use_accel_omp:
            return accel_data_present(self.raw, self._accel_name)
        elif use_accel_jax:
            return accel_data_present(self.data)
        else:
            return False

    def _accel_create(self, zero_out=False):
        if use_accel_omp:
            self.raw = accel_data_create(self.raw, self._accel_name, zero_out=zero_out)
        elif use_accel_jax:
            self.data = accel_data_create(self.data, zero_out=zero_out)

    def _accel_update_device(self):
        if use_accel_omp:
            self.raw = accel_data_update_device(self.raw, self._accel_name)
        elif use_accel_jax:
            self.data = accel_data_update_device(self.data)

    def _accel_update_host(self):
        if use_accel_omp:
            self.raw = accel_data_update_host(self.raw, self._accel_name)
        elif use_accel_jax:
            self.data = accel_data_update_host(self.data)

    def _accel_reset(self):
        if use_accel_omp:
            accel_data_reset(self.raw, self._accel_name)
        elif use_accel_jax:
            accel_data_reset(self.data)

    def _accel_delete(self):
        if use_accel_omp:
            self.raw = accel_data_delete(self.raw, self._accel_name)
        elif use_accel_jax:
            self.data = accel_data_delete(self.data)
