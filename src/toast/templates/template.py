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

    det_data = Unicode(
        None, allow_none=True, help="Observation detdata key for the timestream data"
    )

    data = Instance(
        None,
        klass=Data,
        allow_none=True,
        help="This must be an instance of a Data class (or None)",
    )

    @traitlets.validate("data")
    def _check_data(self, proposal):
        dat = proposal["value"]
        if dat is not None:
            if not isinstance(dat, Data):
                raise traitlets.TraitError("data should be a Data instance")
            # Call the instance initialization.
            self.initialize(dat)
        return dat

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, newdata):
        raise NotImplementedError("Derived class must implement _initialize()")

    def initialize(self, newdata):
        """Initialize instance after the data trait has been set.

        Templates use traits to set their properties, which allows them to be
        configured easily with the constructor overrides and enables them to be built
        from config files.  However, the `data` trait may not be set at construction
        time and this trait is likely used to compute the number of template amplitudes
        that will be used and other parameters.  This explicit initialize method is
        called whenever the `data` trait is set.

        """
        self._initialize(newdata)

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

    Args:
        comm (mpi4py.MPI.Comm):  The MPI communicator or None.
        n_global (int):  The number of global values across all

    """

    def __init__(self, comm, n_global, local_indices=None):
        self._comm = comm

    @property
    def comm(self):
        return _comm

    def _n_vales(self):
        raise NotImplementedError("Derived classes must implement _n_values()")

    def n_values(self):
        """Returns the total number of amplitudes."""
        return self._n_values()

    def _n_local(self):
        raise NotImplementedError("Derived classes must implement _n_local()")

    def n_local(self):
        """Returns the number of locally stored amplitudes."""
        return self._n_local()

    def _get_global_values(self, offset, buffer):
        raise NotImplementedError("Derived classes must implement _get_global_values()")

    def get_global_values(self, offset, buffer):
        """For the given range of global values, populate the buffer.

        This function takes the provided buffer for the global sample offset and fills
        it with any local values that fall in that sample range.  Other values should be
        set to zero.  This is used in synchronization / reduction.

        Args:
            offset (int):  The global sample offset.
            buffer (array):  A pre-existing 1D array of amplitudes.

        Returns:
            None

        """
        return self._global_values(offset, buffer)

    def _set_global_values(self, offset, buffer):
        raise NotImplementedError("Derived classes must implement _set_global_values()")

    def set_global_values(self, offset, buffer):
        """For the given range of global values, set local values.

        This function takes the provided buffer for the global sample offset and uses
        it to set any local values that fall in that sample range.  This is used in
        synchronization / reduction.

        Args:
            offset (int):  The global sample offset.
            buffer (array):  A 1D array of amplitudes.

        Returns:
            None

        """
        return self._global_values(offset, buffer)

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
        log = Logger.get()
        dt = np.dtype(self.local.dtype)

        storage_class = None
        if dt.char == "f":
            storage_class = AlignedF32
        elif dt.char == "d":
            storage_class = AlignedF64
        elif dt.char == "F":
            raise NotImplementedError("No support yet for complex numbers")
        elif dt.char == "D":
            raise NotImplementedError("No support yet for complex numbers")
        else:
            msg = "Unsupported data typecode '{}'".format(dt.char)
            log.error(msg)
            raise ValueError(msg)

        item_size = self.local.dtype.itemsize
        n_comm = int(comm_bytes / item_size)
        n_total = self.n_values()

        # Create a persistent buffer for the reduction

        send_raw = storage_class.zeros(n_comm)
        send_buffer = send_raw.array()
        recv_raw = storage_class.zeros(n_comm)
        recv_buffer = recv_raw.array()

        # Buffered Allreduce

        comm_offset = 0
        while comm_offset < n_total:
            if comm_offset + n_comm > n_total:
                n_comm = n_total - comm_offset
            self.get_global_values(comm_offset, send_buffer)
            self._comm.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
            self.set_global_values(comm_offset, recv_buffer)
            comm_offset += n_comm

        # Cleanup

        del send_buffer
        del recv_buffer
        send_raw.clear()
        recv_raw.clear()
        del send_raw
        del recv_raw

    def _local_dot(self, other):
        """Perform a dot product with the local values of another Amplitudes object.

        It is safe to assume that the calling code has verified that the other
        Amplitudes instance has a matching data distribution.

        """
        raise NotImplementedError("Derived classes must implement _local_dot()")

    def dot(self, other):
        """Perform a dot product with another Amplitudes object.

        The other instance must have the same data distribution.

        Args:
            other (Amplitudes):  The other instance.

        Result:
            (float):  The dot product.

        """
        if other.n_values() != self.n_values():
            raise RuntimeError("Amplitudes must have the same number of values")
        if other.n_local() != self.n_local():
            raise RuntimeError("Amplitudes must have the same number of local values")
        local_result = self._local_dot(other)
        result = local_result
        if self._comm is not None:
            result = MPI.allreduce(result, op=MPI.SUM)
        return result
