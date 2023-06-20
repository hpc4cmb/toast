# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections.abc import MutableMapping

import numpy as np

from ..accelerator import (
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
from ..mpi import MPI
from ..utils import (
    AlignedF32,
    AlignedF64,
    AlignedI32,
    AlignedU8,
    Logger,
    dtype_to_aligned,
)

if use_accel_jax:
    import jax
    import jax.numpy as jnp


class Amplitudes(AcceleratorObject):
    """Class for distributed template amplitudes.

    In the general case, template amplitudes exist as sparse, non-unique values across
    all processes.  This object provides methods for describing the local distribution
    of amplitudes and for doing global reductions and dot products.

    There are 4 supported cases:

        1.  If n_global == n_local, then every process has a full copy of the amplitude
            values.

        2.  If n_global != n_local and both local_indices and local_ranges are None,
            then every process has a disjoint set of amplitudes.  The sum of n_local
            across all processes must equal n_global.

        3.  If n_global != n_local and local_ranges is not None, then local_ranges
            specifies the contiguous global slices that are concatenated to form the
            local data.  The sum of the lengths of the slices must equal n_local.

        4.  If n_global != n_local and local_indices is not None, then local_indices
            is an array of the global indices of all the local data.  The length of
            local_indices must equal n_local.  WARNING:  this case is more costly in
            terms of storage and reduction.  Avoid it if possible.

    Because different process groups have different sets of observations, there are
    some types of templates which may only have shared amplitudes within the group
    communicator.  If use_group is True, the group communicator is used instead of the
    world communicator, and n_global is interpreted as the number of amplitudes in the
    group.  This information is needed whenever working with the full set of amplitudes
    (for example when doing I/O).

    Args:
        comm (toast.Comm):  The toast communicator.
        n_global (int):  The number of global values across all processes.
        n_local (int):  The number of values on this process.
        local_indices (array):  If not None, the explicit indices of the local
            amplitudes within the global array.
        local_ranges (list):  If not None, a list of tuples with the (offset, n_amp)
            amplitude ranges stored locally.
        dtype (dtype):  The amplitude dtype.
        use_group (bool):  If True, use the group rather than world communicator.

    """

    def __init__(
        self,
        comm,
        n_global,
        n_local,
        local_indices=None,
        local_ranges=None,
        dtype=np.float64,
        use_group=False,
    ):
        super().__init__()
        # print(
        #     f"Amplitudes({comm.world_rank}, n_global={n_global}, n_local={n_local}, lc_ind={local_indices}, lc_rng={local_ranges}, dt={dtype}, use_group={use_group}"
        # )
        self._comm = comm
        self._n_global = n_global
        self._n_local = n_local
        self._local_indices = local_indices
        self._local_ranges = local_ranges
        self._use_group = use_group
        if use_group:
            self._mpicomm = self._comm.comm_group
        else:
            self._mpicomm = self._comm.comm_world
        self._dtype = np.dtype(dtype)
        self._storage_class, self._itemsize = dtype_to_aligned(dtype)
        self._full = False
        self._global_first = None
        self._global_last = None
        if self._n_global == self._n_local:
            self._full = True
            self._global_first = 0
            self._global_last = self._n_local - 1
        else:
            if (self._local_indices is None) and (self._local_ranges is None):
                rank = 0
                if self._mpicomm is not None:
                    all_n_local = self._mpicomm.gather(self._n_local, root=0)
                    rank = self._mpicomm.rank
                    if rank == 0:
                        all_n_local = np.array(all_n_local, dtype=np.int64)
                        if np.sum(all_n_local) != self._n_global:
                            msg = "Total amplitudes on all processes does "
                            msg += "not equal n_global"
                            raise RuntimeError(msg)
                    all_n_local = self._mpicomm.bcast(all_n_local, root=0)
                else:
                    all_n_local = np.array([self._n_local], dtype=np.int64)
                self._global_first = 0
                for i in range(rank):
                    self._global_first += all_n_local[i]
                self._global_last = self._global_first + self._n_local - 1
            elif self._local_ranges is not None:
                # local data is specified by ranges
                check = 0
                last = 0
                for off, n in self._local_ranges:
                    check += n
                    if off < last:
                        msg = "local_ranges must not overlap and must be sorted"
                        raise RuntimeError(msg)
                    last = off + n
                    if last > self._n_global:
                        msg = "local_ranges extends beyond the number of global amps"
                        raise RuntimeError(msg)
                if check != self._n_local:
                    raise RuntimeError("local_ranges must sum to n_local")
                self._global_first = self._local_ranges[0][0]
                self._global_last = (
                    self._local_ranges[-1][0] + self._local_ranges[-1][1] - 1
                )
            else:
                # local data has explicit global indices
                if len(self._local_indices) != self._n_local:
                    msg = "Length of local_indices must match n_local"
                    raise RuntimeError(msg)
                self._global_first = self._local_indices[0]
                self._global_last = self._local_indices[-1]
        self._raw = self._storage_class.zeros(self._n_local)
        self.local = self._raw.array()

        # Support flagging of template amplitudes.  This can be used to flag some
        # amplitudes if too many timestream samples contributing to the amplitude value
        # are bad.  We will be passing these flags to compiled code, and there
        # is no way easy way to do this using numpy bool and C++ bool.  So we waste
        # a bit of memory and use a whole byte per amplitude.
        self._raw_flags = AlignedU8.zeros(self._n_local)
        self.local_flags = self._raw_flags.array()

    def clear(self):
        """Delete the underlying memory.

        This will forcibly delete the C-allocated memory and invalidate all python
        references to this object.  DO NOT CALL THIS unless you are sure all references
        are no longer being used and you are about to delete the object.

        """
        if self.accel_exists():
            self.accel_delete()
        if hasattr(self, "local"):
            del self.local
            self.local = None
        if hasattr(self, "local_flags"):
            del self.local_flags
            self.local_flags = None
        if hasattr(self, "_raw"):
            if self._raw is not None:
                self._raw.clear()
            del self._raw
            self._raw = None
        if hasattr(self, "_raw_flags"):
            if self._raw_flags is not None:
                self._raw_flags.clear()
            del self._raw_flags
            self._raw_flags = None

    def __del__(self):
        self.clear()

    def __repr__(self):
        val = "<Amplitudes n_global={} n_local={} comm={}\n  {}\n  {}>".format(
            self.n_global, self.n_local, self.comm, self.local, self.local_flags
        )
        return val

    def __eq__(self, value):
        if isinstance(value, Amplitudes):
            return self.local == value.local
        else:
            return self.local == value

    # Arithmetic.  These assume that flagging is consistent between the pairs of
    # Amplitudes (always true when used in the mapmaking) or that the flagged values
    # have been zeroed out.

    def __iadd__(self, other):
        if isinstance(other, Amplitudes):
            self.local[:] += other.local
        else:
            self.local[:] += other
        return self

    def __isub__(self, other):
        if isinstance(other, Amplitudes):
            self.local[:] -= other.local
        else:
            self.local[:] -= other
        return self

    def __imul__(self, other):
        if isinstance(other, Amplitudes):
            self.local[:] *= other.local
        else:
            self.local[:] *= other
        return self

    def __itruediv__(self, other):
        if isinstance(other, Amplitudes):
            self.local[:] /= other.local
        else:
            self.local[:] /= other
        return self

    def __add__(self, other):
        result = self.duplicate()
        result += other
        return result

    def __sub__(self, other):
        result = self.duplicate()
        result -= other
        return result

    def __mul__(self, other):
        result = self.duplicate()
        result *= other
        return result

    def __truediv__(self, other):
        result = self.duplicate()
        result /= other
        return result

    def reset(self):
        """Set all amplitude values to zero."""
        self.local[:] = 0
        if self.accel_exists():
            self._accel_reset_local()

    def reset_flags(self):
        """Set all flag values to zero."""
        self.local_flags[:] = 0
        if self.accel_exists():
            self._accel_reset_local_flags()

    def duplicate(self):
        """Return a copy of the data."""
        ret = Amplitudes(
            self._comm,
            self._n_global,
            self._n_local,
            local_indices=self._local_indices,
            local_ranges=self._local_ranges,
            dtype=self._dtype,
            use_group=self._use_group,
        )
        if self.accel_exists():
            ret.accel_create(self._accel_name)
        restore = False
        if self.accel_in_use():
            # We have no good way to copy between device buffers,
            # so do this on the host.  The duplicate() method is
            # not used inside the solver loop.
            self.accel_update_host()
            restore = True
        ret.local[:] = self.local
        ret.local_flags[:] = self.local_flags
        if restore:
            self.accel_update_device()
            ret.accel_update_device()
        return ret

    @property
    def comm(self):
        """The toast communicator in use."""
        return self._comm

    @property
    def n_global(self):
        """The total number of amplitudes."""
        return self._n_global

    @property
    def n_local(self):
        """The number of locally stored amplitudes."""
        return self._n_local

    @property
    def n_local_flagged(self):
        """The number of local amplitudes that are flagged."""
        return np.count_nonzero(self.local_flags)

    @property
    def local_indices(self):
        """The global indices of the local amplitudes, or None."""
        return self._local_indices

    @property
    def local_ranges(self):
        """The global slices covered by local amplitudes, or None."""
        return self._local_indices

    @property
    def use_group(self):
        """Whether to use the group communicator rather than the global one."""
        return self._use_group

    def sync(self, comm_bytes=10000000):
        """Perform an Allreduce across all processes.

        Args:
            comm_bytes (int):  The maximum number of bytes to communicate in each
                call to Allreduce.

        Returns:
            None

        """
        if self._mpicomm is None:
            # Nothing to do
            return

        if not self._full and (
            self._local_indices is None and self._local_ranges is None
        ):
            # Disjoint set of amplitudes, no communication needed.
            return

        log = Logger.get()

        n_comm = int(comm_bytes / self._itemsize)
        n_total = self._n_global
        if n_comm > n_total:
            n_comm = n_total

        # Create persistent buffers for the reduction

        send_raw = self._storage_class.zeros(n_comm)
        send_buffer = send_raw.array()
        recv_raw = self._storage_class.zeros(n_comm)
        recv_buffer = recv_raw.array()

        # Buffered Allreduce

        # For each buffer, the local indices of relevant data
        local_selected = None

        # For each buffer, the indices of relevant data in the buffer
        buffer_selected = None

        comm_offset = 0
        while comm_offset < n_total:
            if comm_offset + n_comm > n_total:
                n_comm = n_total - comm_offset

            if self._full:
                # Shortcut if we have all global amplitudes locally
                send_buffer[:n_comm] = self.local[comm_offset : comm_offset + n_comm]
                bad = self.local_flags[comm_offset : comm_offset + n_comm] != 0
                send_buffer[:n_comm][bad] = 0
            else:
                # Need to compute our overlap with the global amplitude range.
                send_buffer[:] = 0
                if (self._global_last >= comm_offset) and (
                    self._global_first < comm_offset + n_comm
                ):
                    # We have some overlap
                    if self._local_ranges is not None:
                        sel_start = None
                        n_sel = 0

                        # current local offset of the range
                        range_off = 0

                        # build up the corresponding buffer indices
                        buffer_selected = list()

                        for off, n in self._local_ranges:
                            if off >= comm_offset + n_comm:
                                range_off += n
                                continue
                            if off + n <= comm_offset:
                                range_off += n
                                continue
                            # This range has some overlap...

                            # This is the starting local memory offset of this range:
                            local_off = range_off

                            # Copy offset into the buffer
                            buf_off = 0

                            # The global starting index of the copy
                            start_indx = None

                            if comm_offset > off:
                                local_off += comm_offset - off
                                start_indx = comm_offset
                            else:
                                buf_off = off - comm_offset
                                start_indx = off

                            if sel_start is None:
                                # this is the first range with some overlap
                                sel_start = local_off

                            n_copy = None
                            if comm_offset + n_comm > off + n:
                                n_copy = off + n - start_indx
                            else:
                                n_copy = comm_offset + n_comm - start_indx

                            n_sel += n_copy

                            buffer_selected.append(
                                np.arange(buf_off, buf_off + n_copy, 1, dtype=np.int64)
                            )
                            send_view = send_buffer[buf_off : buf_off + n_copy]
                            send_view[:] = self.local[local_off : local_off + n_copy]
                            send_view[
                                self.local_flags[local_off : local_off + n_copy] != 0
                            ] = 0
                            range_off += n

                        local_selected = slice(sel_start, sel_start + n_sel, 1)
                        buffer_selected = np.concatenate(buffer_selected)

                    elif self._local_indices is not None:
                        local_selected = np.logical_and(
                            np.logical_and(
                                self._local_indices >= comm_offset,
                                self._local_indices < comm_offset + n_comm,
                            ),
                            self.local_flags == 0,
                        )
                        buffer_selected = (
                            self._local_indices[local_selected] - comm_offset
                        )
                        send_buffer[buffer_selected] = self.local[local_selected]
                    else:
                        raise RuntimeError(
                            "should never get here- non-full, disjoint data requires no sync"
                        )

            self._mpicomm.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)

            if self._full:
                # Shortcut if we have all global amplitudes locally
                self.local[comm_offset : comm_offset + n_comm] = recv_buffer[:n_comm]
            else:
                if (self._global_last >= comm_offset) and (
                    self._global_first < comm_offset + n_comm
                ):
                    self.local[local_selected] = recv_buffer[buffer_selected]

            comm_offset += n_comm

        # Cleanup
        del send_buffer
        del recv_buffer
        send_raw.clear()
        recv_raw.clear()
        del send_raw
        del recv_raw

    def dot(self, other, comm_bytes=10000000):
        """Perform a dot product with another Amplitudes object.

        The other instance must have the same data distribution.  The two objects are
        assumed to have already been synchronized, so that any amplitudes that exist
        on multiple processes have the same values.  This further assumes that any
        flagged amplitudes have been set to zero.

        Args:
            other (Amplitudes):  The other instance.
            comm_bytes (int):  The maximum number of bytes to communicate in each
                call to Allreduce.  Only used in the case of explicitly indexed
                amplitudes on each process.

        Result:
            (float):  The dot product.

        """
        if other.n_global != self.n_global:
            raise RuntimeError("Amplitudes must have the same number of values")
        if other.n_local != self.n_local:
            raise RuntimeError("Amplitudes must have the same number of local values")

        if self._mpicomm is None or self._full:
            # Only one process, or every process has the full set of values.
            return np.dot(
                np.where(self.local_flags == 0, self.local, 0),
                np.where(other.local_flags == 0, other.local, 0),
            )

        if (self._local_ranges is None) and (self._local_indices is None):
            # Every process has a unique set of amplitudes.  Reduce the local
            # dot products.
            local_result = np.dot(
                np.where(self.local_flags == 0, self.local, 0),
                np.where(other.local_flags == 0, other.local, 0),
            )
            result = self._mpicomm.allreduce(local_result, op=MPI.SUM)
            return result

        # Each amplitude must only contribute once to the dot product.  Every
        # amplitude will be processed by the lowest-rank process which has
        # that amplitude.  We do this in a buffered way so that we don't need
        # store this amplitude assignment information for the whole data at
        # once.
        n_comm = int(comm_bytes / self._itemsize)
        n_total = self._n_global
        if n_comm > n_total:
            n_comm = n_total

        local_raw = AlignedI32.zeros(n_comm)
        assigned_raw = AlignedI32.zeros(n_comm)
        local = local_raw.array()
        assigned = assigned_raw.array()

        local_result = 0

        # For each buffer, the local indices of relevant data
        local_selected = None

        # For each buffer, the indices of relevant data in the buffer
        buffer_selected = None

        comm_offset = 0
        while comm_offset < n_total:
            if comm_offset + n_comm > n_total:
                n_comm = n_total - comm_offset
            local[:] = self._mpicomm.size

            if (self._global_last >= comm_offset) and (
                self._global_first < comm_offset + n_comm
            ):
                # We have some overlap
                if self._local_ranges is not None:
                    sel_start = None
                    n_sel = 0

                    # current local offset of the range
                    range_off = 0

                    # build up the corresponding buffer indices
                    buffer_selected = list()

                    for off, n in self._local_ranges:
                        if off >= comm_offset + n_comm:
                            range_off += n
                            continue
                        if off + n <= comm_offset:
                            range_off += n
                            continue
                        # This range has some overlap...

                        # This is the starting local memory offset of this range:
                        local_off = range_off

                        # Copy offset into the buffer
                        buf_off = 0

                        # The global starting index of the copy
                        start_indx = None

                        if comm_offset > off:
                            local_off += comm_offset - off
                            start_indx = comm_offset
                        else:
                            buf_off = off - comm_offset
                            start_indx = off

                        if sel_start is None:
                            # this is the first range with some overlap
                            sel_start = local_off

                        n_set = None
                        if comm_offset + n_comm > off + n:
                            n_set = off + n - start_indx
                        else:
                            n_set = comm_offset + n_comm - start_indx

                        n_sel += n_set

                        buffer_selected.append(
                            np.arange(buf_off, buf_off + n_set, 1, dtype=np.int64)
                        )
                        local_view = local[buf_off : buf_off + n_set]
                        local_view[:] = self._mpicomm.rank
                        local_view[
                            self.local_flags[local_off : local_off + n_set] != 0
                        ] = self._mpicomm.size
                        range_off += n

                    local_selected = slice(sel_start, sel_start + n_sel, 1)
                    buffer_selected = np.concatenate(buffer_selected)

                elif self._local_indices is not None:
                    local_selected = np.logical_and(
                        np.logical_and(
                            self._local_indices >= comm_offset,
                            self._local_indices < comm_offset + n_comm,
                        ),
                        self.local_flags == 0,
                    )
                    buffer_selected = self._local_indices[local_selected] - comm_offset
                    local[buffer_selected] = self._mpicomm.rank
                else:
                    raise RuntimeError(
                        "should never get here- non-full, disjoint data requires no sync"
                    )

            self._mpicomm.Allreduce(local, assigned, op=MPI.MIN)

            if (self._global_last >= comm_offset) and (
                self._global_first < comm_offset + n_comm
            ):
                # Compute local dot product of just our assigned, unflagged elements
                local_result += np.dot(
                    np.where(
                        np.logical_and(
                            self.local_flags[local_selected] == 0,
                            assigned[buffer_selected] == self._mpicomm.rank,
                        ),
                        self.local[local_selected],
                        0,
                    ),
                    np.where(
                        np.logical_and(
                            other.local_flags[local_selected] == 0,
                            assigned[buffer_selected] == self._mpicomm.rank,
                        ),
                        other.local[local_selected],
                        0,
                    ),
                )

            comm_offset += n_comm

        result = self._mpicomm.allreduce(local_result, op=MPI.SUM)

        del local
        del assigned
        local_raw.clear()
        assigned_raw.clear()
        del local_raw
        del assigned_raw

        return result

    def _accel_exists(self):
        if use_accel_omp:
            return accel_data_present(
                self._raw, name=self._accel_name
            ) and accel_data_present(self._raw_flags, name=self._accel_name)
        elif use_accel_jax:
            return accel_data_present(self.local) and accel_data_present(
                self.local_flags
            )
        else:
            return False

    def _accel_create(self, zero_out=False):
        if use_accel_omp:
            _ = accel_data_create(self._raw, name=self._accel_name, zero_out=zero_out)
            _ = accel_data_create(
                self._raw_flags, name=self._accel_name, zero_out=zero_out
            )
        elif use_accel_jax:
            self.local = accel_data_create(self.local, zero_out=zero_out)
            self.local_flags = accel_data_create(self.local_flags, zero_out=zero_out)

    def _accel_update_device(self):
        if use_accel_omp:
            _ = accel_data_update_device(self._raw, name=self._accel_name)
            _ = accel_data_update_device(self._raw_flags, name=self._accel_name)
        elif use_accel_jax:
            self.local = accel_data_update_device(self.local)
            self.local_flags = accel_data_update_device(self.local_flags)

    def _accel_update_host(self):
        if use_accel_omp:
            _ = accel_data_update_host(self._raw, name=self._accel_name)
            _ = accel_data_update_host(self._raw_flags, name=self._accel_name)
        elif use_accel_jax:
            self.local = accel_data_update_host(self.local)
            self.local_flags = accel_data_update_host(self.local_flags)

    def _accel_delete(self):
        if use_accel_omp:
            _ = accel_data_delete(self._raw, name=self._accel_name)
            _ = accel_data_delete(self._raw_flags, name=self._accel_name)
        elif use_accel_jax:
            self.local = accel_data_delete(self.local)
            self.local_flags = accel_data_delete(self.local_flags)

    def _accel_reset_local(self):
        # if not self.accel_in_use():
        #     return
        if use_accel_omp:
            accel_data_reset(self._raw, name=self._accel_name)
        elif use_accel_jax:
            accel_data_reset(self.local)

    def _accel_reset_local_flags(self):
        # if not self.accel_in_use():
        #     return
        if use_accel_omp:
            accel_data_reset(self._raw_flags, name=self._accel_name)
        elif use_accel_jax:
            accel_data_reset(self.local_flags)

    def _accel_reset(self):
        self._accel_reset_local()
        self._accel_reset_local_flags()


class AmplitudesMap(MutableMapping, AcceleratorObject):
    """Helper class to provide arithmetic operations on a collection of Amplitudes.

    This simply provides syntactic sugar to reduce duplicated code when working with
    a collection of Amplitudes in the map making.

    """

    def __init__(self):
        super().__init__()
        self._internal = dict()

    # Mapping methods

    def __getitem__(self, key):
        return self._internal[key]

    def __delitem__(self, key):
        del self._internal[key]

    def __setitem__(self, key, value):
        if not isinstance(value, Amplitudes):
            raise RuntimeError(
                "Only Amplitudes objects may be assigned to an AmplitudesMap"
            )
        self._internal[key] = value

    def __iter__(self):
        return iter(self._internal)

    def __len__(self):
        return len(self._internal)

    def __repr__(self):
        val = "<AmplitudesMap"
        for k, v in self._internal.items():
            val += "\n  {} = {}".format(k, v)
        val += "\n>"
        return val

    # Arithmetic.  These operations are done between corresponding Amplitude keys.

    def _check_other(self, other):
        log = Logger.get()
        if sorted(self._internal.keys()) != sorted(other._internal.keys()):
            msg = "Arithmetic between AmplitudesMap objects requires identical keys"
            log.error(msg)
            raise RuntimeError(msg)
        for k, v in self._internal.items():
            if v.n_global != other[k].n_global:
                msg = "Number of global amplitudes not equal for key '{}'".format(k)
                log.error(msg)
                raise RuntimeError(msg)
            if v.n_local != other[k].n_local:
                msg = "Number of local amplitudes not equal for key '{}'".format(k)
                log.error(msg)
                raise RuntimeError(msg)

    def __eq__(self, value):
        if isinstance(value, AmplitudesMap):
            self._check_other(value)
            for k, v in self._internal.items():
                if v != value[k]:
                    return False
            return True
        else:
            for k, v in self._internal.items():
                if v != value:
                    return False
            return True

    def __iadd__(self, other):
        if isinstance(other, AmplitudesMap):
            self._check_other(other)
            for k, v in self._internal.items():
                v += other[k]
        else:
            for k, v in self._internal.items():
                v += other
        return self

    def __isub__(self, other):
        if isinstance(other, AmplitudesMap):
            self._check_other(other)
            for k, v in self._internal.items():
                v -= other[k]
        else:
            for k, v in self._internal.items():
                v -= other
        return self

    def __imul__(self, other):
        if isinstance(other, AmplitudesMap):
            self._check_other(other)
            for k, v in self._internal.items():
                v *= other[k]
        else:
            for k, v in self._internal.items():
                v *= other
        return self

    def __itruediv__(self, other):
        if isinstance(other, AmplitudesMap):
            self._check_other(other)
            for k, v in self._internal.items():
                v /= other[k]
        else:
            for k, v in self._internal.items():
                v /= other
        return self

    def __add__(self, other):
        result = self.duplicate()
        result += other
        return result

    def __sub__(self, other):
        result = self.duplicate()
        result -= other
        return result

    def __mul__(self, other):
        result = self.duplicate()
        result *= other
        return result

    def __truediv__(self, other):
        result = self.duplicate()
        result /= other
        return result

    def reset(self):
        """Set all amplitude values to zero."""
        for k, v in self._internal.items():
            v.reset()

    def reset_flags(self):
        """Set all flag values to zero."""
        for k, v in self._internal.items():
            v.reset_flags()

    def duplicate(self):
        """Return a copy of the data."""
        ret = AmplitudesMap()
        for k, v in self._internal.items():
            ret[k] = v.duplicate()
        return ret

    def dot(self, other):
        """Dot product of all corresponding Amplitudes.

        Args:
            other (AmplitudesMap):  The other instance.

        Result:
            (float):  The dot product.

        """
        log = Logger.get()
        if not isinstance(other, AmplitudesMap):
            msg = "dot product must be with another AmplitudesMap object"
            log.error(msg)
            raise RuntimeError(msg)
        self._check_other(other)
        result = 0.0
        for k, v in self._internal.items():
            result += v.dot(other[k])
        return result

    def accel_used(self, state):
        super().accel_used(state)
        for k, v in self._internal.items():
            v.accel_used(state)

    def _accel_exists(self):
        if not accel_enabled():
            return False
        result = 0
        for k, v in self._internal.items():
            if v.accel_exists():
                result += 1
        if result == 0:
            return False
        elif result != len(self._internal):
            log = Logger.get()
            msg = f"Only some of the Amplitudes exist on device"
            log.error(msg)
            raise RuntimeError(msg)
        return True

    def _accel_create(self, zero_out=False):
        if not accel_enabled():
            return
        for k, v in self._internal.items():
            v.accel_create(f"{self._accel_name}_{k}", zero_out=zero_out)

    def _accel_update_device(self):
        if not accel_enabled():
            return
        for k, v in self._internal.items():
            v.accel_update_device()

    def _accel_update_host(self):
        if not accel_enabled():
            return
        for k, v in self._internal.items():
            v.accel_update_host()

    def _accel_delete(self):
        if not accel_enabled():
            return
        for k, v in self._internal.items():
            v.accel_delete()

    def _accel_reset(self):
        if not accel_enabled():
            return
        for k, v in self._internal.items():
            v.accel_reset()
