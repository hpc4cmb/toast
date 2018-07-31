# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI

import os

import numpy as np

from ..dist import distribute_samples
from ..cache import Cache
from .. import timing as timing
from .interval import Interval

class TOD(object):

    TIMESTAMP_NAME = 'timestamps'
    COMMON_FLAG_NAME = 'common_flags'
    VELOCITY_NAME = 'velocity'
    POSITION_NAME = 'position'
    SIGNAL_NAME = 'signal'
    FLAG_NAME = 'flags'
    POINTING_NAME = 'quat'

    """
    Base class for an object that provides detector pointing and
    timestreams for a single observation.

    This class provides high-level functions that are common to all derived
    classes.  It also defines the internal methods that should be overridden
    by all derived classes.  These internal methods throw an exception if they
    are called.  A TOD base class should never be directly instantiated.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the
            data is distributed.
        detectors (list):  The list of detector names.
        samples (int):  The total number of samples.
        detindx (dict): the detector indices for use in simulations.  Default is
            { x[0] : x[1] for x in zip(detectors, range(len(detectors))) }.
        detranks (int):  The dimension of the process grid in the detector
            direction.  The MPI communicator size must be evenly divisible
            by this number.
        detbreaks (list):  Optional list of hard breaks in the detector
            distribution.
        sampsizes (list):  Optional list of sample chunk sizes which
            cannot be split.
        sampbreaks (list):  Optional list of hard breaks in the sample
            distribution.
        meta (dict):  Optional dictionary of metadata properties.

    """
    def __init__(self, mpicomm, detectors, samples, detindx=None, detranks=1,
                 detbreaks=None, sampsizes=None, sampbreaks=None, meta=None):

        self._mpicomm = mpicomm
        self._detranks = detranks

        if mpicomm.size % detranks != 0:
            raise RuntimeError(
                "The number of detranks ({}) does not divide evenly into the "
                "communicator size ({})".format(detranks, mpicomm.size))

        self._sampranks = mpicomm.size // detranks

        self._rank_det = mpicomm.rank // self._sampranks
        self._rank_samp = mpicomm.rank % self._sampranks

        # Split the main communicator into process row and column
        # communicators, since this is useful for gathering data in some
        # operations.

        if self._sampranks == 1:
            self._comm_row = MPI.COMM_SELF
        else:
            self._comm_row = self._mpicomm.Split(self._rank_det,
                                                 self._rank_samp)

        if self._detranks == 1:
            self._comm_col = MPI.COMM_SELF
        else:
            self._comm_col = self._mpicomm.Split(self._rank_samp,
                                                 self._rank_det)

        self._dets = detectors

        self._nsamp = samples

        self._sizes = sampsizes

        self._meta = meta
        if meta is None:
            self._meta = {}

        if detindx is not None:
            for d in self._dets:
                if d not in detindx:
                    raise RuntimeError(
                        "detindx must have a value for every detector")
            self._detindx = detindx
        else:
            self._detindx = { x[0] : x[1] for x in zip(
                detectors, range(len(detectors))) }

        # if sizes is specified, it must be consistent with
        # the total number of samples.
        if self._sizes is not None:
            test = np.sum(self._sizes)
            if samples != test:
                raise RuntimeError(
                    "Sum of sampsizes ({}) does not equal total samples ({})"
                    "".format(test, samples))

        (self._dist_dets, self._dist_samples,
         self._dist_sizes) = distribute_samples(
             self._mpicomm, self._dets, self._nsamp, detranks=self._detranks,
             detbreaks=detbreaks, sampsizes=sampsizes, sampbreaks=sampbreaks)

        if self._sizes is None:
            # in this case, the chunks just come from the uniform distribution.
            self._sizes = [
                self._dist_samples[x][1] for x in range(self._sampranks)]

        if self._mpicomm.rank == 0:
            # check that all processes have some data, otherwise print warning
            for d in range(self._detranks):
                if len(self._dist_dets[d]) == 0:
                    print("WARNING: detector rank {} has no detectors"
                    " assigned.".format(d))
            for r in range(self._sampranks):
                if self._dist_samples[r][1] <= 0:
                    print("WARNING: sample rank {} has no data assigned "
                    "in TOD.".format(r))

        self.cache = Cache()

    def __del__(self):
        self.cache.clear()

    @property
    def detectors(self):
        """
        (list): The total list of detectors.
        """
        return self._dets

    def meta(self):
        """
        (dict): Dictionary of metadata properties.
        """
        return dict(self._meta)

    def detoffset(self):
        """
        Return dictionary of detector quaternions.

        This returns a dictionary with the detector names as the keys and the
        values are 4-element numpy arrays containing the quaternion offset
        from the boresight.

        Args:
            None

        Returns (dict):
            the dictionary of quaternions.
        """
        raise NotImplementedError("Fell through to TOD base class method")
        return None

    @property
    def detindx(self):
        """
        (dict): The detector indices.
        """
        return self._detindx

    @property
    def local_dets(self):
        """
        (list): The detectors assigned to this process.
        """
        return self._dist_dets[self._rank_det]

    @property
    def total_chunks(self):
        """
        (list): the full list of sample chunk sizes that were used in the
            data distribution.
        """
        return self._sizes

    @property
    def dist_chunks(self):
        """
        (list): this is a list of 2-tuples, one for each column of the process
        grid.  Each element of the list is the same as the information returned
        by the "local_chunks" member for a given process column.
        """
        return self._dist_sizes

    @property
    def local_chunks(self):
        """
        (2-tuple): the first element of the tuple is the index of the
        first chunk assigned to this process (i.e. the index in the list
        given by the "total_chunks" member).  The second element of the
        tuple is the number of chunks assigned to this process.
        """
        return self._dist_sizes[self._rank_samp]

    def local_times(self, name=None, **kwargs):
        """ Timestamps covering locally stored data.

        Args:
            name (str):  Optional cache key to use.
        Returns:
            A cache reference to a timestamp vector.  If 'name' is None
            a default name 'timestamps' is used and the vector may be
            constructed and cached using the 'read_times' method.
            If 'name' is given, then the times must already be cached.

        """
        if name is None:
            cachename = self.TIMESTAMP_NAME
            if not self.cache.exists(cachename):
                times = self.read_times(**kwargs)
                self.cache.put(cachename, times)
        else:
            cachename = name
        return self.cache.reference(cachename)

    def local_signal(self, det, name=None, **kwargs):
        """ Locally stored signal.

        Args:
            det (str):  Name of the detector.
            name (str):  Optional cache key to use.
        Returns:
            A cache reference to a signal vector.  If 'name' is None
            a default name 'signal' is used and the vector may be
            constructed and cached using the 'read' method.
            If 'name' is given, then the signal must already be cached.

        """
        if name is None:
            cachename = '{}_{}'.format(self.SIGNAL_NAME, det)
            if not self.cache.exists(cachename):
                signal = self.read(detector=det, **kwargs)
                self.cache.put(cachename, signal)
        else:
            cachename = '{}_{}'.format(name, det)
        return self.cache.reference(cachename)

    def local_pointing(self, det, name=None, **kwargs):
        """ Locally stored pointing.

        Args:
            det (str):  Name of the detector.
            name (str):  Optional cache key to use.
        Returns:
            A cache reference to a pointing array.  If 'name' is None
            a default name 'quat' is used and the array may be
            constructed and cached using the 'read_pntg' method.
            If 'name' is given, then the pointing must already be cached.

        """
        if name is None:
            cachename = '{}_{}'.format(self.POINTING_NAME, det)
            if not self.cache.exists(cachename):
                quats = self.read_pntg(detector=det, **kwargs)
                self.cache.put(cachename, quats)
        else:
            cachename = '{}_{}'.format(name, det)
        return self.cache.reference(cachename)

    def local_position(self, name=None, **kwargs):
        """ Locally stored position.

        Args:
            name (str):  Optional cache key to use.
        Returns:
            A cache reference to a position array.  If 'name' is None
            a default name 'position' is used and the array may be
            constructed and cached using the 'read_position' method.
            If 'name' is given, then the position must already be cached.

        """
        if name is None:
            cachename = self.POSITION_NAME
            if not self.cache.exists(cachename):
                pos = self.read_position(**kwargs)
                self.cache.put(cachename, pos)
        else:
            cachename = name
        return self.cache.reference(cachename)

    def local_velocity(self, name=None, **kwargs):
        """ Locally stored velocity.

        Args:
            name (str):  Optional cache key to use.
        Returns:
            A cache reference to a velocity array.  If 'name' is None
            a default name 'velocity' is used and the array may be
            constructed and cached using the 'read_velocity' method.
            If 'name' is given, then the velocity must already be cached.

        """
        if name is None:
            cachename = self.VELOCITY_NAME
            if not self.cache.exists(cachename):
                vel = self.read_velocity(**kwargs)
                self.cache.put(cachename, vel)
        else:
            cachename = name
        return self.cache.reference(cachename)

    def local_flags(self, det, name=None, **kwargs):
        """ Locally stored flags.

        Args:
            det (str):  Name of the detector.
            name (str):  Optional cache key to use.
        Returns:
            A cache reference to a flag vector.  If 'name' is None
            a default name 'flags' is used and the vector may be
            constructed and cached using the 'read_flags' method.
            If 'name' is given, then the flags must already be cached.

        """
        if name is None:
            cachename = '{}_{}'.format(self.FLAG_NAME, det)
            if not self.cache.exists(cachename):
                flags = self.read_flags(detector=det, **kwargs)
                self.cache.put(cachename, flags)
        else:
            cachename = '{}_{}'.format(name, det)
        return self.cache.reference(cachename)

    def local_common_flags(self, name=None, **kwargs):
        """ Locally stored common flags.

        Args:
            name (str):  Optional cache key to use.
        Returns:
            A cache reference to a common flag vector.  If 'name' is None
            a default name 'common_flags' is used and the vector may be
            constructed and cached using the 'read_common_flags' method.
            If 'name' is given, then the flags must already be cached.

        """
        if name is None:
            cachename = self.COMMON_FLAG_NAME
            if not self.cache.exists(cachename):
                common_flags = self.read_common_flags(**kwargs)
                self.cache.put(cachename, common_flags)
        else:
            cachename = name
        return self.cache.reference(cachename)

    def local_intervals(self, intervals):
        """ Translate observation-wide intervals into local sample indices.
        """
        if intervals is None:
            intervals = [Interval(start=0, stop=0,
                                  first=0, last=self.total_samples-1)]
        offset, nsamp = self.local_samples
        local_intervals = []
        times = self.local_times()
        if len(times) != nsamp:
            raise RuntimeError(
                'Length of cached timestamps does not match local samples. '
                'Cannot produce local intervals.')
        for ival in intervals:
            previous_last = None
            if (ival.last >= offset and
                ival.first < offset + nsamp):
                local_first = max(0, ival.first-offset)
                if previous_last is not None and previous_last >= local_first:
                    raise RuntimeError('Provided intervals overlap')
                local_last = min(nsamp-1, ival.last-offset)
                previous_last = local_last
                local_start = times[local_first]
                local_stop = times[local_last]
                local_intervals.append(
                    Interval(start=local_start, stop=local_stop,
                             first=local_first, last=local_last))
        return local_intervals

    @property
    def total_samples(self):
        """
        (int): the total number of samples in this TOD.
        """
        return self._nsamp

    @property
    def dist_samples(self):
        """
        (list): This is a list of 2-tuples, with one element per column
            of the process grid.  Each tuple is the same information
            returned by the "local_samples" member for the corresponding
            process grid column rank.
        """
        return self._dist_samples

    @property
    def local_samples(self):
        """
        (2-tuple): The first element of the tuple is the first global
            sample assigned to this process.  The second element of
            the tuple is the number of samples assigned to this
            process.
        """
        return self._dist_samples[self._rank_samp]

    @property
    def mpicomm(self):
        """
        (mpi4py.MPI.Comm): the communicator assigned to this TOD.
        """
        return self._mpicomm

    @property
    def grid_size(self):
        """
        (tuple): the dimensions of the process grid in (detector, sample)
            directions.
        """
        return (self._detranks, self._sampranks)

    @property
    def grid_ranks(self):
        """
        (tuple): the ranks of this process in the (detector, sample)
            directions.
        """
        return (self._rank_det, self._rank_samp)

    @property
    def grid_comm_row(self):
        """
        (mpi4py.MPI.Comm): a communicator across all detectors in the same
            row of the process grid.
        """
        return (self._comm_row)

    @property
    def grid_comm_col(self):
        """
        (mpi4py.MPI.Comm): a communicator across all detectors in the same
            column of the process grid.
        """
        return (self._comm_col)

    def _get(self, detector, start, n):
        raise NotImplementedError("Fell through to TOD._get base class method")
        return None

    def _put(self, detector, start, data):
        raise NotImplementedError("Fell through to TOD._put base class method")
        return

    def _get_boresight(self, start, n):
        raise NotImplementedError("Fell through to TOD._get_boresight base "
            "class method")
        return None

    def _put_boresight(self, start, data):
        raise NotImplementedError("Fell through to TOD._put_boresight base "
            "class method")
        return

    def _get_boresight_azel(self, start, n):
        raise NotImplementedError("Fell through to TOD._get_boresight_azel base"
            "class method")
        return None

    def _put_boresight_azel(self, start, data):
        raise NotImplementedError("Fell through to TOD._put_boresight_azel base"
            "class method")
        return

    def _get_pntg(self, detector, start, n):
        raise NotImplementedError(
            "Fell through to TOD._get_pntg base class method")
        return None

    def _put_pntg(self, detector, start, data):
        raise NotImplementedError(
            "Fell through to TOD._put_pntg base class method")
        return

    def _get_flags(self, detector, start, n):
        raise NotImplementedError(
            "Fell through to TOD._get_flags base class method")
        return None

    def _put_flags(self, detector, start, flags):
        raise NotImplementedError(
            "Fell through to TOD._put_flags base class method")
        return

    def _get_common_flags(self, start, n):
        raise NotImplementedError(
            "Fell through to TOD._get_common_flags base class method")
        return None

    def _put_common_flags(self, start, flags):
        raise NotImplementedError(
            "Fell through to TOD._put_common_flags base class method")
        return

    def _get_times(self, start, n):
        raise NotImplementedError(
            "Fell through to TOD._get_times base class method")
        return None

    def _put_times(self, start, stamps):
        raise NotImplementedError(
            "Fell through to TOD._put_times base class method")
        return None

    def _get_position(self, start, n):
        raise NotImplementedError(
            "Fell through to TOD._get_position base class method")
        return None

    def _put_position(self, start, pos):
        raise NotImplementedError(
            "Fell through to TOD._put_position base class method")
        return

    def _get_velocity(self, start, n):
        raise NotImplementedError(
            "Fell through to TOD._get_velocity base class method")
        return None

    def _put_velocity(self, start, vel):
        raise NotImplementedError(
            "Fell through to TOD._put_velocity base class method")
        return

    # Read and write the common timestamps

    def read_times(self, local_start=0, n=0, **kwargs):
        """
        Read timestamps.

        This reads the common set of timestamps that apply to all detectors
        in the TOD.

        Args:
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            n (int): the number of samples to read.  If zero, read to end.

        Returns:
            (array): a numpy array containing the timestamps.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        if n == 0:
            n = self.local_samples[1] - local_start
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                "cannot read times- process has no assigned local samples")
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError("local sample range {} - {} is invalid"
                             "".format(local_start, local_start+n-1))
        return self._get_times(local_start, n, **kwargs)

    def write_times(self, local_start=0, stamps=None, **kwargs):
        """
        Write timestamps.

        This writes the common set of timestamps that apply to all detectors
        in the TOD.

        Args:
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            stamps (array): the array of timestamps to write.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        if stamps is None:
            raise ValueError("you must specify the vector of time stamps")
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                "cannot write times- process has no assigned local samples")
        if (local_start < 0) \
           or (local_start + stamps.shape[0] > self.local_samples[1]):
            raise ValueError(
                "local sample range {} - {} is invalid"
                "".format(local_start, local_start+stamps.shape[0]-1))
        self._put_times(local_start, stamps, **kwargs)
        return

    # Read and write telescope boresight pointing

    def read_boresight(self, local_start=0, n=0, **kwargs):
        """
        Read boresight quaternion pointing.

        This returns the pointing of the boresight in quaternions.

        Args:
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            n (int): the number of samples to read.  If zero, read to end.

        Returns:
            A 2D array of shape (n, 4)
        """
        autotimer = timing.auto_timer(type(self).__name__)
        if n == 0:
            n = self.local_samples[1] - local_start
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                "cannot read boresight- process has no local samples")
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError(
                "local sample range {} - {} is invalid"
                "".format(local_start, local_start+n-1))
        return self._get_boresight(local_start, n, **kwargs)

    def write_boresight(self, local_start=0, data=None, **kwargs):
        """
        Write boresight quaternion pointing.

        This writes the quaternion pointing for the boresight.

        Args:
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            data (array): 2D array of quaternions with shape[1] == 4.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        if len(data.shape) != 2:
            raise ValueError("data should be a 2D array")
        if data.shape[1] != 4:
            raise ValueError("data should have second dimension of size 4")
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                "cannot write boresight- process has no local samples")
        if (local_start < 0) \
           or (local_start + data.shape[0] > self.local_samples[1]):
            raise ValueError("local sample range is invalid")
        self._put_boresight(local_start, data, **kwargs)
        return


    def read_boresight_azel(self, local_start=0, n=0, **kwargs):
        """
        Read boresight Azimuth / Elevation quaternion pointing.

        This returns the pointing of the boresight in the horizontal coordinate
        system, if it exists.

        Args:
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            n (int): the number of samples to read.  If zero, read to end.

        Returns:
            A 2D array of shape (n, 4)

        Raises:
            NotImplementedError: if the telescope is not on the Earth.

        """
        autotimer = timing.auto_timer(type(self).__name__)
        if n == 0:
            n = self.local_samples[1] - local_start
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                "cannot read boresight- process has no local samples")
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError(
                "local sample range {} - {} is invalid"
                "".format(local_start, local_start+n-1))
        return self._get_boresight_azel(local_start, n, **kwargs)


    def write_boresight_azel(self, local_start=0, data=None, **kwargs):
        """Write boresight Azimuth / Elevation quaternion pointing.

        This writes the quaternion pointing for the boresight in the horizontal
        coordinate system, if it exists.

        Args:
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            data (array): 2D array of quaternions with shape[1] == 4.

        Raises:
            RuntimeError or AttributeError : if the telescope is not on
                the Earth.

        """
        autotimer = timing.auto_timer(type(self).__name__)
        if len(data.shape) != 2:
            raise ValueError("data should be a 2D array")
        if data.shape[1] != 4:
            raise ValueError("data should have second dimension of size 4")
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                "cannot write boresight- process has no local samples")
        if (local_start < 0) \
           or (local_start + data.shape[0] > self.local_samples[1]):
            raise ValueError("local sample range is invalid")
        self._put_boresight_azel(local_start, data, **kwargs)
        return


    # Read and write detector data

    def read(self, detector=None, local_start=0, n=0, **kwargs):
        """
        Read detector data.

        This returns the timestream data for a single detector.

        Args:
            detector (str): the name of the detector.
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            n (int): the number of samples to read.  If zero, read to end.

        Returns:
            An array containing the data.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        if detector is None:
            raise ValueError("you must specify the detector")
        if detector not in self.local_dets:
            raise ValueError("detector {} not found".format(detector))
        if n == 0:
            n = self.local_samples[1] - local_start
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                "cannot read- process has no assigned local samples")
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError(
                "local sample range {} - {} is invalid"
                "".format(local_start, local_start+n-1))
        return self._get(detector, local_start, n, **kwargs)

    def write(self, detector=None, local_start=0, data=None, **kwargs):
        """
        Write detector data.

        This writes the detector data.

        Args:
            detector (str): the name of the detector.
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            data (array): the data array.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        if detector is None:
            raise ValueError("you must specify the detector")
        if detector not in self.local_dets:
            raise ValueError("detector {} not found".format(detector))
        if data is None:
            raise ValueError("data array must be specified")
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                "cannot write- process has no assigned local samples")
        if (local_start < 0) \
           or (local_start + data.shape[0] > self.local_samples[1]):
            raise ValueError(
                "local sample range {} - {} is invalid"
                "".format(local_start, local_start+data.shape[0]-1))
        self._put(detector, local_start, data, **kwargs)
        return

    # Read and write detector quaternion pointing

    def read_pntg(self, detector=None, local_start=0, n=0, **kwargs):
        """
        Read detector quaternion pointing.

        This returns the pointing for a single detector in quaternions.

        Args:
            detector (str): the name of the detector.
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            n (int): the number of samples to read.  If zero, read to end.

        Returns:
            A 2D array of shape (n, 4)
        """
        autotimer = timing.auto_timer(type(self).__name__)
        if detector is None:
            raise ValueError("you must specify the detector")
        if detector not in self.local_dets:
            raise ValueError("detector {} not found".format(detector))
        if n == 0:
            n = self.local_samples[1] - local_start
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                "cannot read pntg- process has no assigned local samples")
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError(
                "local sample range {} - {} is invalid"
                "".format(local_start, local_start+n-1))
        return self._get_pntg(detector, local_start, n, **kwargs)

    def write_pntg(self, detector=None, local_start=0, data=None, **kwargs):
        """
        Write detector quaternion pointing.

        This writes the quaternion pointing for a single detector.

        Args:
            detector (str): the name of the detector.
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            data (array): 2D array of quaternions with shape[1] == 4.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        if detector is None:
            raise ValueError("you must specify the detector")
        if detector not in self.local_dets:
            raise ValueError("detector {} not found".format(detector))
        if data is None:
            raise ValueError("data must be specified")
        if len(data.shape) != 2:
            raise ValueError("data should be a 2D array")
        if data.shape[1] != 4:
            raise ValueError("data should have second dimension of size 4")
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                "cannot write pntg- process has no assigned local samples")
        if (local_start < 0) \
           or (local_start + data.shape[0] > self.local_samples[1]):
            raise ValueError("local sample range is invalid")
        self._put_pntg(detector, local_start, data, **kwargs)
        return

    # Read and write detector flags

    def read_flags(self, detector=None, local_start=0, n=0, **kwargs):
        """
        Read detector flags.

        This returns the detector-specific flags.

        Args:
            detector (str): the name of the detector.
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            n (int): the number of samples to read.  If zero, read to end.

        Returns:
            An array containing the detector flags.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        if detector is None:
            raise ValueError("you must specify the detector")
        if detector not in self.local_dets:
            raise ValueError("detector {} not found".format(detector))
        if n == 0:
            n = self.local_samples[1] - local_start
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                "cannot read flags- process has no assigned local samples")
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError(
                "local sample range {} - {} is invalid"
                "".format(local_start, local_start+n-1))
        return self._get_flags(detector, local_start, n, **kwargs)


    def read_common_flags(self, local_start=0, n=0, **kwargs):
        """
        Read common flags.

        This reads the common set of flags that should be applied to all
        detectors.

        Args:
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            n (int): the number of samples to read.  If zero, read to end.

        Returns:
            (array): a numpy array containing the flags.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        if self.local_samples[1] <= 0:
            raise RuntimeError("cannot read common flags- process has no "
                               "assigned local samples")
        if n == 0:
            n = self.local_samples[1] - local_start
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError(
                "local sample range {} - {} is invalid"
                "".format(local_start, local_start+n-1))
        return self._get_common_flags(local_start, n, **kwargs)

    def write_common_flags(self, local_start=0, flags=None, **kwargs):
        """
        Write common flags.

        This writes the common set of flags that should be applied to all
        detectors.

        Args:
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            flags (array): array containing the flags to write.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        if flags is None:
            raise ValueError("flags must be specified")
        if self.local_samples[1] <= 0:
            raise RuntimeError("cannot write common flags- process has no "
                               "assigned local samples")

        if (local_start < 0) \
           or (local_start + flags.shape[0] > self.local_samples[1]):
            raise ValueError(
                "local sample range {} - {} is invalid"
                "".format(local_start, local_start+flags.shape[0]-1))
        self._put_common_flags(local_start, flags, **kwargs)
        return

    def write_flags(self, detector=None, local_start=0, flags=None,
                        **kwargs):
        """
        Write detector flags.

        This writes the detector-specific flags.

        Args:
            detector (str): the name of the detector.
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            flags (array): the detector flags.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        if detector is None:
            raise ValueError("you must specify the detector")
        if detector not in self.local_dets:
            raise ValueError("detector {} not found".format(detector))
        if flags is None:
            raise ValueError("flags must be specified")
        if self.local_samples[1] <= 0:
            raise RuntimeError("cannot write flags- process has no assigned "
                               "local samples")
        if (local_start < 0) \
           or (local_start + flags.shape[0] > self.local_samples[1]):
            raise ValueError(
                "local sample range {} - {} is invalid"
                "".format(local_start, local_start+flags.shape[0]-1))
        self._put_flags(detector, local_start, flags, **kwargs)
        return

    # Read and write telescope position

    def read_position(self, local_start=0, n=0, **kwargs):
        """
        Read telescope position.

        This reads the telescope position in solar system barycenter
        coordinates (in Kilometers).

        Args:
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            n (int): the number of samples to read.  If zero, read to end.

        Returns:
            (array): a 2D numpy array containing the x,y,z coordinates at each
                sample.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        if n == 0:
            n = self.local_samples[1] - local_start
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                "cannot read position- process has no assigned local samples")
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError(
                "local sample range {} - {} is invalid"
                "".format(local_start, local_start+n-1))
        return self._get_position(local_start, n, **kwargs)

    def write_position(self, local_start=0, pos=None, **kwargs):
        """
        Write telescope position.

        This writes the telescope position in solar system barycenter
        coordinates (in Kilometers).

        Args:
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            pos (array): the 2D array of x,y,z coordinates at each sample.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        if pos is None:
            raise ValueError("you must specify the array of coordinates")
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                "cannot write position- process has no assigned local samples")
        if (local_start < 0) \
           or (local_start + pos.shape[0] > self.local_samples[1]):
            raise ValueError(
                "local sample range {} - {} is invalid"
                "".format(local_start, local_start+pos.shape[0]-1))
        self._put_position(local_start, pos, **kwargs)
        return

    # Read and write telescope velocity

    def read_velocity(self, local_start=0, n=0, **kwargs):
        """
        Read telescope velocity.

        This reads the telescope velocity in solar system barycenter
        coordinates (in Kilometers/s).

        Args:
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            n (int): the number of samples to read.  If zero, read to end.

        Returns:
            (array): a 2D numpy array containing the x,y,z velocity components
                at each sample.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        if n == 0:
            n = self.local_samples[1] - local_start
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                "cannot read position- process has no assigned local samples")
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError(
                "local sample range {} - {} is invalid"
                "".format(local_start, local_start+n-1))
        return self._get_velocity(local_start, n, **kwargs)

    def write_velocity(self, local_start=0, vel=None, **kwargs):
        """
        Write telescope velocity.

        This writes the telescope velocity in solar system barycenter
        coordinates (in Kilometers/s).

        Args:
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            vel (array): the 2D array of x,y,z velocity components at each
                sample.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        if vel is None:
            raise ValueError("you must specify the array of velocities.")
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                "cannot write times- process has no assigned local samples")
        if (local_start < 0) \
           or (local_start + vel.shape[0] > self.local_samples[1]):
            raise ValueError(
                "local sample range {} - {} is invalid"
                "".format(local_start, local_start+vel.shape[0]-1))
        self._put_velocity(local_start, vel, **kwargs)
        return


class TODCache(TOD):
    """
    TOD class that uses a memory cache for storage.

    This class simply uses a manually managed Cache object to store time
    ordered data.  You must "write" the data before you can "read" it.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the
            data is distributed.
        detectors (list):  The list of detector names.
        samples (int):  The total number of samples.
        detindx (dict): the detector indices for use in simulations.  Default is
            { x[0] : x[1] for x in zip(detectors, range(len(detectors))) }.
        detquats (dict):  Dictionary of detector quaternions.
        detranks (int):  The dimension of the process grid in the detector
            direction.  The MPI communicator size must be evenly divisible
            by this number.
        detbreaks (list):  Optional list of hard breaks in the detector
            distribution.
        sampsizes (list):  Optional list of sample chunk sizes which
            cannot be split.
        sampbreaks (list):  Optional list of hard breaks in the sample
            distribution.
    """

    def __init__(self, mpicomm, detectors, samples, detindx=None, detquats=None,
                 detranks=1, detbreaks=None, sampsizes=None, sampbreaks=None):

        super().__init__(mpicomm, detectors, samples, detindx=detindx,
            detranks=detranks, detbreaks=detbreaks, sampsizes=sampsizes,
            sampbreaks=sampbreaks)

        self._detquats = detquats
        self._pref_detdata = "toast_tod_detdata_"
        self._pref_detflags = "toast_tod_detflags_"
        self._pref_detpntg = "toast_tod_detpntg_"
        self._bore = "toast_boresight"
        self._bore_azel = "toast_boresight_azel"
        self._common = "toast_tod_common_flags"
        self._stamps = "toast_tod_stamps"
        self._pos = "toast_tod_pos"
        self._vel = "toast_tod_vel"

    def __del__(self):
        self.cache.clear()

    def detoffset(self):
        if self._detquats is None:
            raise NotImplementedError("TODCache does not contain detector "
                "quaternions.")
            return None
        else:
            return self._detquats

    # This class just use a Cache object to store things.

    def _get(self, detector, start, n):
        if detector not in self.local_dets:
            raise ValueError(
                "detector {} not assigned to local process".format(detector))
        cachedata = "{}{}".format(self._pref_detdata, detector)
        if not self.cache.exists(cachedata):
            raise ValueError(
                "detector {} data not yet written".format(detector))
        dataref = self.cache.reference(cachedata)[start:start+n]
        return dataref

    def _put(self, detector, start, data):
        if detector not in self.local_dets:
            raise ValueError(
                "detector {} not assigned to local process".format(detector))
        cachedata = "{}{}".format(self._pref_detdata, detector)

        if not self.cache.exists(cachedata):
            self.cache.create(cachedata, np.float64, (self.local_samples[1],))

        n = data.shape[0]
        refdata = self.cache.reference(cachedata)[start:start+n]

        refdata[:] = data
        return

    def _get_boresight(self, start, n):
        if not self.cache.exists(self._bore):
            raise ValueError("boresight not yet written")
        ref = self.cache.reference(self._bore)[start:start+n,:]
        return ref

    def _put_boresight(self, start, data):
        if not self.cache.exists(self._bore):
            self.cache.create(self._bore, np.float64,
                (self.local_samples[1],4))
        ref = self.cache.reference(self._bore)
        ref[start:(start+data.shape[0]),:] = data
        return

    def _get_boresight_azel(self, start, n):
        if not self.cache.exists(self._bore_azel):
            raise ValueError("boresight not yet written")
        ref = self.cache.reference(self._bore_azel)[start:start+n,:]
        return ref

    def _put_boresight_azel(self, start, data):
        if not self.cache.exists(self._bore_azel):
            self.cache.create(self._bore_azel, np.float64,
                (self.local_samples[1],4))
        ref = self.cache.reference(self._bore_azel)
        ref[start:(start+data.shape[0]),:] = data
        return

    def _get_pntg(self, detector, start, n):
        cachepntg = "{}{}".format(self._pref_detpntg, detector)
        if not self.cache.exists(cachepntg):
            # No detector-specific pointing written.  See if we have
            # boresight pointing and detector quaternions.
            if self.cache.exists(self._bore) and (self._detquats is not None):
                return qa.mult()
            else:
                raise ValueError(
                    "detector {}: pointing data not yet written, and boresight"
                    " and detector quaternions do not exist.".format(detector))
        else:
            return self.cache.reference(cachepntg)[start:start+n,:]

    def _put_pntg(self, detector, start, data):
        if detector not in self.local_dets:
            raise ValueError(
                "detector {} not assigned to local process".format(detector))
        cachepntg = "{}{}".format(self._pref_detpntg, detector)
        if not self.cache.exists(cachepntg):
            self.cache.create(cachepntg, np.float64, (self.local_samples[1],4))
        pntgref = self.cache.reference(cachepntg)[start:(start+data.shape[0]),:]
        pntgref[:] = data
        return

    def _get_flags(self, detector, start, n):
        if detector not in self.local_dets:
            raise ValueError(
                "detector {} not assigned to local process".format(detector))
        cacheflags = "{}{}".format(self._pref_detflags, detector)
        if not self.cache.exists(cacheflags):
            raise ValueError(
                "detector {} flags not yet written".format(detector))
        flagsref = self.cache.reference(cacheflags)[start:start+n]
        return flagsref

    def _put_flags(self, detector, start, flags):
        if detector not in self.local_dets:
            raise ValueError(
                "detector {} not assigned to local process".format(detector))
        cacheflags = "{}{}".format(self._pref_detflags, detector)

        if not self.cache.exists(cacheflags):
            self.cache.create(cacheflags, np.uint8, (self.local_samples[1],))

        n = flags.shape[0]
        refflags = self.cache.reference(cacheflags)[start:start+n]

        refflags[:] = flags
        return

    def _get_common_flags(self, start, n):
        if not self.cache.exists(self._common):
            raise ValueError("common flags not yet written")
        comref = self.cache.reference(self._common)[start:start+n]
        return comref

    def _put_common_flags(self, start, flags):
        if not self.cache.exists(self._common):
            self.cache.create(self._common, np.uint8, (self.local_samples[1],))
        n = flags.shape[0]
        comref = self.cache.reference(self._common)[start:start+n]
        comref[:] = flags
        return

    def _get_times(self, start, n):
        if not self.cache.exists(self._stamps):
            raise ValueError("timestamps not yet written")
        ref = self.cache.reference(self._stamps)[start:start+n]
        return ref

    def _put_times(self, start, stamps):
        if not self.cache.exists(self._stamps):
            self.cache.create(self._stamps, np.float64,
                              (self.local_samples[1],))
        n = stamps.shape[0]
        ref = self.cache.reference(self._stamps)[start:start+n]
        ref[:] = stamps
        return

    def _get_position(self, start, n):
        if not self.cache.exists(self._pos):
            raise ValueError("telescope position not yet written")
        ref = self.cache.reference(self._pos)[start:start+n]
        return ref

    def _put_position(self, start, pos):
        if not self.cache.exists(self._pos):
            self.cache.create(self._pos, np.float64, (self.local_samples[1], 3))
        n = pos.shape[0]
        ref = self.cache.reference(self._pos)[start:start+n, :]
        ref[:, :] = pos
        return

    def _get_velocity(self, start, n):
        if not self.cache.exists(self._vel):
            raise ValueError("telescope velocity not yet written")
        ref = self.cache.reference(self._vel)[start:start+n]
        return ref

    def _put_velocity(self, start, vel):
        if not self.cache.exists(self._vel):
            self.cache.create(self._vel, np.float64, (self.local_samples[1], 3))
        n = vel.shape[0]
        ref = self.cache.reference(self._vel)[start:start+n, :]
        ref[:, :] = vel
        return
