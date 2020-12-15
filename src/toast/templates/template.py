# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


from ..utils import (
    Logger,
    AlignedF32,
    AlignedF64,
)

from ..traits import TraitConfig

from ..data import Data


class Template(TraitConfig):
    """Base class for timestream templates.

    A template defines a mapping to / from timestream values to a set of template
    amplitudes.  These amplitudes are usually quantities being solved as part of the
    map-making.  Examples of templates might be destriping baseline offsets,
    azimuthally binned ground pickup, etc.

    The template amplitude data may be distributed in a variety of ways.  For some
    types of templates, every process may have their own unique set of amplitudes based
    on the data that they have locally.  In other cases, every process may have a full
    local copy of all template amplitudes.  There might also be cases where each
    process has a non-unique subset of amplitude values (similar to the way that
    pixel domain quantities are distributed).

    """

    # Note:  The TraitConfig base class defines a "name" attribute.

    data = Instance(
        None,
        klass=Data,
        allow_none=True,
        help="This must be an instance of a Data class (or None)",
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    det_data = Unicode(
        None, allow_none=True, help="Observation detdata key for the timestream data"
    )

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(0, help="Bit mask value for optional flagging")

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional shared flagging")

    @traitlets.validate("data")
    def _check_data(self, proposal):
        dat = proposal["value"]
        if dat is not None:
            if not isinstance(dat, Data):
                raise traitlets.TraitError("data should be a Data instance")
            # Call the instance initialization.
            self.initialize(dat)
        return dat

    @traitlets.observe("data")
    def _initialize(self, change):
        # Derived classes should implement this method to do any set up (like
        # computing the number of amplitudes) whenever the data changes.
        newdata = change["data"]
        raise NotImplementedError("Derived class must implement _initialize()")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _zeros(self):
        raise NotImplementedError("Derived class must implement _zeros()")

    def zeros(self):
        """Return an Amplitudes object filled with zeros.

        This returns an Amplitudes instance with appropriate dimensions for this
        template.  This will raise an exception if called before the `data` trait
        is set.

        Returns:
            (Amplitudes):  Zero amplitudes.

        """
        if self.data is None:
            raise RuntimeError("You must set the data trait before using a template")
        return self._zeros()

    def _add_to_signal(self, detector, amplitudes):
        raise NotImplementedError("Derived class must implement _add_to_signal()")

    def add_to_signal(self, detector, amplitudes):
        """Accumulate the projected amplitudes to a timestream.

        This performs the operation:

        .. math::
            s += F \\cdot a

        Where `s` is the det_data signal, `F` is the template and `a` is the amplitudes.

        Args:
            detector (str):  The detector name.
            amplitudes (Amplitudes):  The Amplitude values for this template.

        Returns:
            None

        """
        if self.data is None:
            raise RuntimeError("You must set the data trait before using a template")
        return self._add_to_signal(detector, amplitudes)

    def _project_signal(self, detector, amplitudes):
        raise NotImplementedError("Derived class must implement _project_signal()")

    def project_signal(self, detector, amplitudes):
        """Project a timestream into template amplitudes.

        This performs:

        .. math::
            a += F^T \\cdot s

        Where `s` is the det_data signal, `F` is the template and `a` is the amplitudes.

        Args:
            detector (str):  The detector name.
            amplitudes (Amplitudes):  The Amplitude values for this template.

        Returns:
            None

        """
        if self.data is None:
            raise RuntimeError("You must set the data trait before using a template")
        self._project_signal(detector, amplitudes)

    def _add_prior(self, amplitudes_in, amplitudes_out):
        # Not all Templates implement the prior
        return

    def add_prior(self, amplitudes_in, amplitudes_out):
        """Apply the inverse amplitude covariance as a prior.

        This performs:

        .. math::
            a' += {C_a}^{-1} \\cdot a

        Args:
            amplitudes_in (Amplitudes):  The input Amplitude values for this template.
            amplitudes_out (Amplitudes):  The input Amplitude values for this template.

        Returns:
            None

        """
        if self.data is None:
            raise RuntimeError("You must set the data trait before using a template")
        self._add_prior(amplitudes_in, amplitudes_out)

    def _apply_precond(self, amplitudes_in, amplitudes_out):
        raise NotImplementedError("Derived class must implement _apply_precond()")

    def apply_precond(self, amplitudes_in, amplitudes_out):
        """Apply the template preconditioner.

        This performs:

        .. math::
            a' += M^{-1} \\cdot a

        Args:
            amplitudes_in (Amplitudes):  The input Amplitude values for this template.
            amplitudes_out (Amplitudes):  The input Amplitude values for this template.

        Returns:
            None

        """
        if self.data is None:
            raise RuntimeError("You must set the data trait before using a template")
        self._apply_precond(amplitudes_in, amplitudes_out)

    @classmethod
    def get_class_config_path(cls):
        return "/templates/{}".format(cls.__qualname__)

    def get_config_path(self):
        if self.name is None:
            return None
        return "/templates/{}".format(self.name)

    @classmethod
    def get_class_config(cls, input=None):
        """Return a dictionary of the default traits of an Template class.

        This returns a new or appended dictionary.  The class instance properties are
        contained in a dictionary found in result["templates"][cls.name].

        If the specified named location in the input config already exists then an
        exception is raised.

        Args:
            input (dict):  The optional input dictionary to update.

        Returns:
            (dict):  The created or updated dictionary.

        """
        return super().get_class_config(section="templates", input=input)

    def get_config(self, input=None):
        """Return a dictionary of the current traits of a Template *instance*.

        This returns a new or appended dictionary.  The operator instance properties are
        contained in a dictionary found in result["templates"][self.name].

        If the specified named location in the input config already exists then an
        exception is raised.

        Args:
            input (dict):  The optional input dictionary to update.

        Returns:
            (dict):  The created or updated dictionary.

        """
        return super().get_config(section="templates", input=input)

    @classmethod
    def translate(cls, props):
        """Given a config dictionary, modify it to match the current API."""
        # For templates, the derived classes should implement this method as needed
        # and then call super().translate(props) to trigger this method.  Here we strip
        # the 'API' key from the config.
        props = super().translate(props)
        if "API" in props:
            del props["API"]
        return props


class Amplitudes(object):
    """Class for distributed template amplitudes.

    In the general case, template amplitudes exist as sparse, non-unique values across
    all processes.  This object provides methods for describing the local distribution
    of amplitudes and for doing global reductions.

    If n_global == n_local, then every process has a full copy of the amplitude
    values.  If the two arguments are different, then each process has a subset of
    values.  If local_indices is None, then each process has a unique set of values
    and the total number across all processes must sum to n_global.  If local_indices
    is given, then it is the explicit locations of the local values within the global
    set.

    Args:
        comm (mpi4py.MPI.Comm):  The MPI communicator or None.
        n_global (int):  The number of global values across all processes.
        n_local (int):  The number of values on this process.
        local_indices (array):  If not None, the explicit indices of the local
            amplitudes within the global array.
        dtype (dtype):  The amplitude dtype.

    """

    def __init__(self, comm, n_global, n_local, local_indices=None, dtype=np.float64):
        self._comm = comm
        self._n_global = n_global
        self._n_local = n_local
        self._local_indices = local_indices
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
            if self._local_indices is None:
                check = [self._n_local]
                rank = 0
                if self._comm is not None:
                    check = self._comm.allgather(check)
                    rank = self._comm.rank
                if np.sum(check) != self._n_global:
                    msg = "Total amplitudes on all processes does not equal n_global"
                    raise RuntimeError(msg)
                self._global_first = 0
                for i in range(rank):
                    self._global_first += check[i]
                self._global_last = self._global_first + self._n_local - 1
            else:
                if len(self._local_indices) != self._n_local:
                    msg = "Length of local_indices must match n_local"
                    raise RuntimeError(msg)
                self._global_first = self._local_indices[0]
                self._global_last = self._local_indices[-1]
        self._raw = self._storage_class.zeros(self._n_local)
        self.local = self._raw.array()

    def clear(self):
        """Delete the underlying memory.

        This will forcibly delete the C-allocated memory and invalidate all python
        references to this object.  DO NOT CALL THIS unless you are sure all references
        are no longer being used and you are about to delete the object.

        """
        if hasattr(self, "local"):
            del self.local
        if hasattr(self, "_raw"):
            self._raw.clear()
            del self._raw

    def __del__(self):
        self.clear()

    @property
    def comm(self):
        return _comm

    @property
    def n_global(self):
        """The total number of amplitudes."""
        return self._n_global

    @property
    def n_local(self):
        """The number of locally stored amplitudes."""
        return self._n_local

    def _get_global_values(comm_offset, send_buffer):
        n_buf = len(send_buffer)
        if self._full:
            # Shortcut if we have all global amplitudes locally
            send_buffer[:] = self.local[comm_offset : comm_offset + n_buf]
        else:
            # Need to compute our overlap with the global range.
            send_buffer[:] = 0
            if (self._global_last < comm_offset) or (
                self._global_first >= comm_offset + n_buf
            ):
                # No overlap with our local data
                return
            if self._local_indices is None:
                local_off = 0
                buf_off = 0
                if comm_offset > self._global_first:
                    local_off = comm_offset - self._global_first
                else:
                    buf_off = self._global_first - comm_offset
                n_copy = None
                if comm_offset + n_buf > self._global_last:
                    n_copy = self._global_last + 1 - local_off
                else:
                    n_copy = n_buf - buf_off
                send_buffer[buf_off : buf_off + n_copy] = self.local[
                    local_off : local_off + n_copy
                ]
            else:
                # Need to efficiently do the lookup.  Pull existing techniques from
                # old code when we need this.
                raise NotImplementedError("sync of explicitly indexed amplitudes")

    def _set_global_values(comm_offset, recv_buffer):
        n_buf = len(recv_buffer)
        if self._full:
            # Shortcut if we have all global amplitudes locally
            self.local[comm_offset : comm_offset + n_buf] = recv_buffer
        else:
            # Need to compute our overlap with the global range.
            if (self._global_last < comm_offset) or (
                self._global_first >= comm_offset + n_buf
            ):
                # No overlap with our local data
                return
            if self._local_indices is None:
                local_off = 0
                buf_off = 0
                if comm_offset > self._global_first:
                    local_off = comm_offset - self._global_first
                else:
                    buf_off = self._global_first - comm_offset
                n_copy = None
                if comm_offset + n_buf > self._global_last:
                    n_copy = self._global_last + 1 - local_off
                else:
                    n_copy = n_buf - buf_off
                self.local[local_off : local_off + n_copy] = recv_buffer[
                    buf_off : buf_off + n_copy
                ]
            else:
                # Need to efficiently do the lookup.  Pull existing techniques from
                # old code when we need this.
                raise NotImplementedError("sync of explicitly indexed amplitudes")

    def sync(self, comm_bytes=10000000):
        """Perform an Allreduce across all processes.

        If a derived class has only locally unique amplitudes on each process (for
        example, destriping baseline offsets), then they should override this method
        and make it a no-op.

        Args:
            comm_bytes (int):  The maximum number of bytes to communicate in each
                call to Allreduce.

        Returns:
            None

        """
        if self._comm is None or self._local_indices is None:
            # We have either one process or every process has a disjoint set of
            # amplitudes.  Nothing to sync.
            return
        log = Logger.get()

        n_comm = int(comm_bytes / self._itemsize)
        n_total = self._n_global

        # Create persistent buffers for the reduction

        send_raw = self._storage_class.zeros(n_comm)
        send_buffer = send_raw.array()
        recv_raw = self._storage_class.zeros(n_comm)
        recv_buffer = recv_raw.array()

        # Buffered Allreduce

        comm_offset = 0
        while comm_offset < n_total:
            if comm_offset + n_comm > n_total:
                n_comm = n_total - comm_offset
            self._get_global_values(comm_offset, send_buffer)
            self._comm.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
            self._set_global_values(comm_offset, recv_buffer)
            comm_offset += n_comm

        # Cleanup
        del send_buffer
        del recv_buffer
        send_raw.clear()
        recv_raw.clear()
        del send_raw
        del recv_raw

    def dot(self, other):
        """Perform a dot product with another Amplitudes object.

        The other instance must have the same data distribution.  The two objects are
        assumed to have already been synchronized, so that any amplitudes that exist
        on multiple processes have the same values.

        Args:
            other (Amplitudes):  The other instance.

        Result:
            (float):  The dot product.

        """
        if other.n_global != self.n_global:
            raise RuntimeError("Amplitudes must have the same number of values")
        if other.n_local != self.n_local:
            raise RuntimeError("Amplitudes must have the same number of local values")
        local_result = np.dot(self.local, other.local)
        result = None
        if self._comm is None or self._full:
            # Only one process, or every process has the full set of values.
            result = local_result
        else:
            if self._local_indices is None:
                # Every process has a unique set of amplitudes.  Reduce the local
                # dot products.
                result = MPI.allreduce(local_result, op=MPI.SUM)
            else:
                # More complicated, since we need to reduce each amplitude only
                # once.  Implement techniques from other existing code when needed.
                raise NotImplementedError("dot of explicitly indexed amplitudes")
        return result
