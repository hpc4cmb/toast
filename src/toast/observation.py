# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy
import numbers
import sys
import types
from collections.abc import Mapping, MutableMapping, Sequence

import numpy as np
from astropy import units as u
from pshmem.utils import mpi_data_type

from .accelerator import AcceleratorObject
from .dist import distribute_samples
from .instrument import Session, Telescope
from .intervals import IntervalList, interval_dtype
from .mpi import MPI, comm_equal
from .observation_data import (
    DetDataManager,
    DetectorData,
    IntervalsManager,
    SharedDataManager,
)
from .observation_dist import DistDetSamp, redistribute_data
from .observation_view import DetDataView, SharedView, View, ViewInterface, ViewManager
from .timing import function_timer
from .utils import Logger, name_UID

default_values = None


def set_default_values(values=None):
    """Update default values for common Observation objects.

    Args:
        names (dict):  The dictionary specifying any name overrides.

    Returns:
        None

    """
    global default_values

    defaults = {
        # names
        "times": "times",
        "shared_flags": "flags",
        "det_data": "signal",
        "det_flags": "flags",
        "hwp_angle": "hwp_angle",
        "azimuth": "azimuth",
        "elevation": "elevation",
        "boresight_azel": "boresight_azel",
        "boresight_radec": "boresight_radec",
        "position": "position",
        "velocity": "velocity",
        "pixels": "pixels",
        "weights": "weights",
        "quats": "quats",
        "quats_azel": "quats_azel",
        #
        # flag masks
        #
        "shared_mask_invalid": 1,
        "shared_mask_unstable_scanrate": 2,
        "shared_mask_irregular": 4,
        "det_mask_invalid": 1,
        "det_mask_sso": 1 + 2,
        "det_mask_processing": 2,
        #
        # ground-specific flag masks
        #
        # This marks the turnarounds as "invalid".  To simulate data
        # in the turnarounds, set the turnaround mask in SimGround to
        # just "2".
        "turnaround": 1 + 2,
        "scan_leftright": 8,
        "scan_rightleft": 16,
        "sun_up": 32,
        "sun_close": 64,
        "elnod": 1 + 2 + 4,
        #
        # ground-specific interval names
        #
        "scanning_interval": "scanning",
        "turnaround_interval": "turnaround",
        "throw_leftright_interval": "throw_leftright",
        "throw_rightleft_interval": "throw_rightleft",
        "throw_interval": "throw",
        "scan_leftright_interval": "scan_leftright",
        "scan_rightleft_interval": "scan_rightleft",
        "turn_leftright_interval": "turn_leftright",
        "turn_rightleft_interval": "turn_rightleft",
        "elnod_interval": "elnod",
        "sun_up_interval": "sun_up",
        "sun_close_interval": "sun_close",
        #
        # Units
        #
        "det_data_units": u.Kelvin,
    }

    if values is not None:
        defaults.update(values)

    default_values = types.SimpleNamespace(**defaults)


if default_values is None:
    set_default_values()


class Observation(MutableMapping):
    """Class representing the data for one observation.

    An Observation stores information about data distribution across one or more MPI
    processes and is a container for four types of objects:

        * Local detector data (unique to each process).
        * Shared data that has one common copy for every node spanned by the
          observation.
        * Intervals defining spans of data with some common characteristic.
        * Other arbitrary small metadata.

    Small metadata can be stored directly in the Observation using normal square
    bracket "[]" access to elements (an Observation is a dictionary).  Groups of
    detector data (e.g. "signal", "flags", etc) can be accessed in the separate
    detector data dictionary (the "detdata" attribute).  Shared data can be similarly
    stored in the "shared" attribute.  Lists of intervals are accessed in the
    "intervals" attribute and data views can use any interval list to access subsets
    of detector and shared data.

    **Notes on distributed use with MPI**

    The detector data within an Observation is distributed among the processes in an
    MPI communicator.  The processes in the communicator are arranged in a rectangular
    grid, with each process storing some number of detectors for a piece of time
    covered by the observation.  The most common configuration (and the default) is to
    make this grid the size of the communicator in the "detector direction" and a size
    of one in the "sample direction"::

        MPI           det1  sample(0), sample(1), sample(2), ...., sample(N-1)
        rank 0        det2  sample(0), sample(1), sample(2), ...., sample(N-1)
        ----------------------------------------------------------------------
        MPI           det3  sample(0), sample(1), sample(2), ...., sample(N-1)
        rank 1        det4  sample(0), sample(1), sample(2), ...., sample(N-1)

    So each process has a subset of detectors for the whole span of the observation
    time.  You can override this shape by setting the process_rows to something
    else.  For example, process_rows=1 would result in this::

        MPI rank 0                        |        MPI rank 1
        ----------------------------------+----------------------------
        det1  sample(0), sample(1), ...,  |  ...., sample(N-1)
        det2  sample(0), sample(1), ...,  |  ...., sample(N-1)
        det3  sample(0), sample(1), ...,  |  ...., sample(N-1)
        det4  sample(0), sample(1), ...,  |  ...., sample(N-1)


    Args:
        comm (toast.Comm):  The toast communicator containing information about the
            process group for this observation.
        telescope (Telescope):  An instance of a Telescope object.
        n_samples (int):  The total number of samples for this observation.
        name (str):  (Optional) The observation name.
        uid (int):  (Optional) The Unique ID for this observation.  If not specified,
            the UID will be computed from a hash of the name.
        session (Session):  The observing session that this observation is contained
            in or None.
        detector_sets (list):  (Optional) List of lists containing detector names.
            These discrete detector sets are used to distribute detectors- a detector
            set will always be within a single row of the process grid.  If None,
            every detector is a set of one.
        sample_sets (list):  (Optional) List of lists of chunk sizes (integer numbers of
            samples).  These discrete sample sets are used to distribute sample data.
            A sample set will always be within a single column of the process grid.  If
            None, any distribution break in the sample direction will happen at an
            arbitrary place.  The sum of all chunks must equal the total number of
            samples.
        process_rows (int):  (Optional) The size of the rectangular process grid
            in the detector direction.  This number must evenly divide into the size of
            comm.  If not specified, defaults to the size of the communicator.

    """

    view = ViewInterface()

    @function_timer
    def __init__(
        self,
        comm,
        telescope,
        n_samples,
        name=None,
        uid=None,
        session=None,
        detector_sets=None,
        sample_sets=None,
        process_rows=None,
    ):
        log = Logger.get()
        self._telescope = telescope
        self._name = name
        self._uid = uid
        self._session = session

        if self._uid is None and self._name is not None:
            self._uid = name_UID(self._name)

        if self._session is None:
            if self._name is not None:
                self._session = Session(
                    name=self._name,
                    uid=self._uid,
                    start=None,
                    end=None,
                )
        elif not isinstance(self._session, Session):
            raise RuntimeError("session should be a Session instance or None")

        self.dist = DistDetSamp(
            n_samples,
            self._telescope.focalplane.detectors,
            sample_sets,
            detector_sets,
            comm,
            process_rows,
        )

        # The internal metadata dictionary
        self._internal = dict()

        # Set up the data managers
        self.detdata = DetDataManager(self.dist)
        self.shared = SharedDataManager(self.dist)
        self.intervals = IntervalsManager(self.dist, n_samples)

        # Set up local per-detector cutting
        self._detflags = {x: int(0) for x in self.dist.dets[self.dist.comm.group_rank]}

    # Fully clear the observation

    def clear(self):
        self.view.clear()
        self.intervals.clear()
        self.detdata.clear()
        self.shared.clear()
        self._internal.clear()

    # General properties

    @property
    def telescope(self):
        """
        (Telescope):  The Telescope instance for this observation.
        """
        return self._telescope

    @property
    def name(self):
        """
        (str):  The name of the observation.
        """
        return self._name

    @property
    def uid(self):
        """
        (int):  The Unique ID for this observation.
        """
        return self._uid

    @property
    def session(self):
        """
        (Session):  The Session instance for this observation.
        """
        return self._session

    @property
    def comm(self):
        """
        (toast.Comm):  The overall communicator.
        """
        return self.dist.comm

    # The MPI communicator along the current row of the process grid

    @property
    def comm_row(self):
        """
        (mpi4py.MPI.Comm):  The communicator for processes in the same row (or None).
        """
        return self.dist.comm_row

    @property
    def comm_row_size(self):
        """
        (int): The number of processes in the row communicator.
        """
        return self.dist.comm_row_size

    @property
    def comm_row_rank(self):
        """
        (int): The rank of this process in the row communicator.
        """
        return self.dist.comm_row_rank

    # The MPI communicator along the current column of the process grid

    @property
    def comm_col(self):
        """
        (mpi4py.MPI.Comm):  The communicator for processes in the same column (or None).
        """
        return self.dist.comm_col

    @property
    def comm_col_size(self):
        """
        (int): The number of processes in the column communicator.
        """
        return self.dist.comm_col_size

    @property
    def comm_col_rank(self):
        """
        (int): The rank of this process in the column communicator.
        """
        return self.dist.comm_col_rank

    # Detector distribution

    @property
    def all_detectors(self):
        """
        (list): All detectors stored in this observation.
        """
        return self.dist.detectors

    @property
    def local_detectors(self):
        """
        (list): The detectors assigned to this process.
        """
        return self.dist.dets[self.dist.comm.group_rank]

    @property
    def local_detector_flags(self):
        """(dict): The local per-detector flags"""
        return self._detflags

    def update_local_detector_flags(self, vals):
        """Update the per-detector flagging.

        This does a bitwise OR with the existing flag values.

        Args:
            vals (dict):  The flag values for one or more detectors.

        Returns:
            None

        """
        ldets = set(self.local_detectors)
        for k, v in vals.items():
            if k not in ldets:
                msg = f"Cannot update per-detector flag for '{k}', which is"
                msg += " not a local detector"
                raise RuntimeError(msg)
            self._detflags[k] |= int(v)

    def set_local_detector_flags(self, vals):
        """Set the per-detector flagging.

        This resets the per-detector flags to the specified values.

        Args:
            vals (dict):  The flag values for one or more detectors.

        Returns:
            None

        """
        ldets = set(self.local_detectors)
        for k, v in vals.items():
            if k not in ldets:
                msg = f"Cannot set per-detector flag for '{k}', which is"
                msg += " not a local detector"
                raise RuntimeError(msg)
            self._detflags[k] = int(v)

    def select_local_detectors(
        self,
        selection=None,
        flagmask=(default_values.det_mask_invalid | default_values.det_mask_processing),
    ):
        """Get the local detectors assigned to this process.

        This takes the full list of local detectors and optionally prunes them
        by the specified selection and / or applies per-detector flags with
        the given mask.

        Args:
            selection (list):  Only return detectors in this set.
            flagmask (uint8):  Apply this mask to per-detector flags and only
                include detectors with a result of zero (good).

        Returns:
            (list):  The selected detectors.

        """
        if flagmask is None:
            good = set(self.local_detectors)
        else:
            good = set(
                [
                    x
                    for x in self.local_detectors
                    if (self.local_detector_flags[x] & flagmask) == 0
                ]
            )
        dets = list()
        if selection is None:
            for det in self.local_detectors:
                if det in good:
                    dets.append(det)
        else:
            sel_set = set(selection)
            for det in self.local_detectors:
                if (det in sel_set) and (det in good):
                    dets.append(det)
        return dets

    # Detector set distribution

    @property
    def all_detector_sets(self):
        """
        (list):  The total list of detector sets for this observation.
        """
        return self.dist.detector_sets

    @property
    def local_detector_sets(self):
        """
        (list):  The detector sets assigned to this process (or None).
        """
        if self.dist.detector_sets is None:
            return None
        else:
            ds = list()
            for d in range(self.dist.det_sets[self.dist.comm.group_rank].n_elem):
                off = self.dist.det_sets[self.dist.comm.group_rank].offset
                ds.append(self.dist.detector_sets[off + d])
            return ds

    # Sample distribution

    @property
    def n_all_samples(self):
        """(int): the total number of samples in this observation."""
        return self.dist.samples

    @property
    def local_index_offset(self):
        """
        The first sample on this process, relative to the observation start.
        """
        return self.dist.samps[self.dist.comm.group_rank].offset

    @property
    def n_local_samples(self):
        """
        The number of local samples on this process.
        """
        return self.dist.samps[self.dist.comm.group_rank].n_elem

    # Sample set distribution

    @property
    def all_sample_sets(self):
        """
        (list):  The input full list of sample sets used in data distribution
        """
        return self.dist.sample_sets

    @property
    def local_sample_sets(self):
        """
        (list):  The sample sets assigned to this process (or None).
        """
        if self.dist.sample_sets is None:
            return None
        else:
            ss = list()
            for s in range(self.dist.samp_sets[self.dist.comm.group_rank].n_elem):
                off = self.dist.samp_sets[self.dist.comm.group_rank].offset
                ss.append(self.dist.sample_sets[off + s])
            return ss

    # Mapping methods

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

    def __del__(self):
        if hasattr(self, "detdata"):
            self.detdata.clear()
        if hasattr(self, "shared"):
            self.shared.clear()

    def __repr__(self):
        val = "<Observation"
        val += f"\n  name = '{self.name}'"
        val += f"\n  uid = '{self.uid}'"
        if self.comm.comm_group is None:
            val += "  group has a single process (no MPI)"
        else:
            val += f"  group has {self.comm.group_size} processes"
        val += f"\n  telescope = {self._telescope.__repr__()}"
        val += f"\n  session = {self._session.__repr__()}"
        for k, v in self._internal.items():
            val += f"\n  {k} = {v}"
        val += f"\n  {self.n_all_samples} total samples ({self.n_local_samples} local)"
        val += f"\n  shared:  {self.shared}"
        val += f"\n  detdata:  {self.detdata}"
        val += f"\n  intervals:  {self.intervals}"
        val += "\n>"
        return val

    def __eq__(self, other):
        # Note that testing for equality is quite expensive, since it means testing all
        # metadata and also all detector, shared, and interval data.  This is mainly
        # used for unit tests.
        log = Logger.get()
        fail = 0
        if self.name != other.name:
            fail = 1
            log.verbose(
                f"Proc {self.comm.world_rank}:  Obs names {self.name} != {other.name}"
            )
        if self.uid != other.uid:
            fail = 1
            log.verbose(
                f"Proc {self.comm.world_rank}:  Obs uid {self.uid} != {other.uid}"
            )
        if self.telescope != other.telescope:
            fail = 1
            log.verbose(f"Proc {self.comm.world_rank}:  Obs telescopes not equal")
        if self.session != other.session:
            fail = 1
            log.verbose(f"Proc {self.comm.world_rank}:  Obs sessions not equal")
        if self.dist != other.dist:
            fail = 1
            log.verbose(f"Proc {self.comm.world_rank}:  Obs distributions not equal")
        if set(self._internal.keys()) != set(other._internal.keys()):
            fail = 1
            log.verbose(f"Proc {self.comm.world_rank}:  Obs metadata keys not equal")
        for k, v in self._internal.items():
            if v != other._internal[k]:
                feq = True
                try:
                    feq = np.allclose(v, other._internal[k])
                except Exception:
                    # Not floating point data
                    feq = False
                if not feq:
                    fail = 1
                    log.verbose(
                        f"Proc {self.comm.world_rank}:  Obs metadata[{k}]:  {v} != {other[k]}"
                    )
                    break
        if self.shared != other.shared:
            fail = 1
            log.verbose(f"Proc {self.comm.world_rank}:  Obs shared data not equal")
        if self.detdata != other.detdata:
            fail = 1
            log.verbose(f"Proc {self.comm.world_rank}:  Obs detdata not equal")
        if self.intervals != other.intervals:
            fail = 1
            log.verbose(f"Proc {self.comm.world_rank}:  Obs intervals not equal")
        if self.comm.comm_group is not None:
            fail = self.comm.comm_group.allreduce(fail, op=MPI.SUM)
        return fail == 0

    def __ne__(self, other):
        return not self.__eq__(other)

    def duplicate(
        self, times=None, meta=None, shared=None, detdata=None, intervals=None
    ):
        """Return a copy of the observation and all its data.

        The times field should be the name of the shared field containing timestamps.
        This is used when copying interval lists to the new observation so that these
        objects reference the timestamps within this observation (rather than the old
        one).  If this is not specified and some intervals exist, then an exception is
        raised.

        The meta, shared, detdata, and intervals list specifies which of those objects
        to copy to the new observation.  If these are None, then all objects are
        duplicated.

        Args:
            times (str):  The name of the timestamps shared field.
            meta (list):  List of metadata objects to copy, or None.
            shared (list):  List of shared objects to copy, or None.
            detdata (list):  List of detdata objects to copy, or None.
            intervals (list):  List of intervals objects to copy, or None.

        Returns:
            (Observation):  The new copy of the observation.

        """
        log = Logger.get()
        if times is None and len(self.intervals) > 0:
            msg = "You must specify the times field when duplicating observations "
            msg += "that have some intervals defined."
            log.error(msg)
            raise RuntimeError(msg)
        new_obs = Observation(
            self.dist.comm,
            self.telescope,
            self.n_all_samples,
            name=self.name,
            uid=self.uid,
            session=self.session,
            detector_sets=self.all_detector_sets,
            sample_sets=self.all_sample_sets,
            process_rows=self.dist.process_rows,
        )
        for k, v in self._internal.items():
            if meta is None or k in meta:
                new_obs[k] = copy.deepcopy(v)
        for name, data in self.detdata.items():
            if detdata is None or name in detdata:
                new_obs.detdata[name] = data
        copy_shared = list()
        if times is not None:
            copy_shared.append(times)
        if shared is not None:
            copy_shared.extend(shared)
        for name, data in self.shared.items():
            if shared is None or name in copy_shared:
                # Create the object on the corresponding communicator in the new obs
                new_obs.shared.assign_mpishared(name, data, self.shared.comm_type(name))
        for name, data in self.intervals.items():
            if intervals is None or name in intervals:
                timespans = [(x.start, x.stop) for x in data]
                new_obs.intervals[name] = IntervalList(
                    new_obs.shared[times], timespans=timespans
                )
        return new_obs

    def memory_use(self):
        """Estimate the memory used by shared and detector data.

        This sums the memory used by the shared and detdata attributes and returns the
        total on all processes.  This function is blocking on the observation
        communicator.

        Returns:
            (int):  The number of bytes of memory used by timestream data.

        """
        # Get local memory from detector data
        local_mem = self.detdata.memory_use()

        # If there are many intervals, this could take up non-trivial space.  Add them
        # to the local total
        for iname, it in self.intervals.items():
            if len(it) > 0:
                local_mem += len(it) * interval_dtype.itemsize

        # Sum the aggregate local memory
        total = None
        if self.comm.comm_group is None:
            total = local_mem
        else:
            total = self.comm.comm_group.allreduce(local_mem, op=MPI.SUM)

        # The total shared memory use is already returned on every process by this
        # next function.
        total += self.shared.memory_use()
        return total

    # Redistribution

    @function_timer
    def redistribute(
        self,
        process_rows,
        times=None,
        override_sample_sets=False,
        override_detector_sets=False,
    ):
        """Take the currently allocated observation and redistribute in place.

        This changes the data distribution within the observation.  After
        re-assigning all detectors and samples, the currently allocated shared data
        objects and detector data objects are redistributed using the observation
        communicator.

        Args:
            process_rows (int):  The size of the new process grid in the detector
                direction.  This number must evenly divide into the size of the
                observation communicator.
            times (str):  The shared data field representing the timestamps.  This
                is used to recompute the intervals after redistribution.
            override_sample_sets (False, None or list):  If not False, override
                existing sample set boundaries in the redistributed data.
            override_detector_sets (False, None or list):  If not False, override
                existing detector set boundaries in the redistributed data.

        Returns:
            None

        """
        log = Logger.get()
        if process_rows == self.dist.process_rows:
            # Nothing to do!
            return

        if override_sample_sets == False:
            sample_sets = self.dist.sample_sets
        else:
            sample_sets = override_sample_sets

        if override_detector_sets == False:
            detector_sets = self.dist.detector_sets
        else:
            detector_sets = override_detector_sets

        # Get the total set of per-detector flags
        if self.comm_col_size == 1:
            all_det_flags = self.local_detector_flags
        else:
            pdflags = self.comm_col.gather(self.local_detector_flags, root=0)
            all_det_flags = None
            if self.comm_col_rank == 0:
                all_det_flags = dict()
                for pf in pdflags:
                    all_det_flags.update(pf)
            all_det_flags = self.comm_col.bcast(all_det_flags, root=0)

        # Create the new distribution
        new_dist = DistDetSamp(
            self.dist.samples,
            self._telescope.focalplane.detectors,
            sample_sets,
            detector_sets,
            self.dist.comm,
            process_rows,
        )

        # Do the actual redistribution
        new_shr_manager, new_det_manager, new_intervals_manager = redistribute_data(
            self.dist,
            new_dist,
            self.shared,
            self.detdata,
            self.intervals,
            times=times,
            dbg=self.name,
        )

        # Redistribute any metadata objects that support it.
        for k, v in self._internal.items():
            if hasattr(v, "redistribute"):
                v.redistribute(self.dist, new_dist)

        # Replace our distribution and data managers with the new ones.
        del self.dist
        self.dist = new_dist

        self.shared.clear()
        del self.shared
        self.shared = new_shr_manager

        self.detdata.clear()
        del self.detdata
        self.detdata = new_det_manager

        self.intervals.clear()
        del self.intervals
        self.intervals = new_intervals_manager

        # Restore detector flags for our new local detectors
        self.set_local_detector_flags(
            {x: all_det_flags[x] for x in self.local_detectors}
        )

    # Accelerator use

    def accel_create(self, names):
        """Create a set of data objects on the device.

        This takes a dictionary with the same format as those used by the Operator
        provides() and requires() methods.

        Args:
            names (dict):  Dictionary of lists.

        Returns:
            None

        """
        for key in names["detdata"]:
            self.detdata.accel_create(key)
        for key in names["shared"]:
            self.shared.accel_create(key)
        for key in names["intervals"]:
            self.intervals.accel_create(key)
        for key, val in self._internal.items():
            if isinstance(val, AcceleratorObject):
                if not val.accel_exists():
                    val.accel_create()

    def accel_update_device(self, names):
        """Copy data objects to the device.

        This takes a dictionary with the same format as those used by the Operator
        provides() and requires() methods.

        Args:
            names (dict):  Dictionary of lists.

        Returns:
            None

        """
        for key in names["detdata"]:
            self.detdata.accel_update_device(key)
        for key in names["shared"]:
            self.shared.accel_update_device(key)
        for key in names["intervals"]:
            self.intervals.accel_update_device(key)
        for key, val in self._internal.items():
            if isinstance(val, AcceleratorObject):
                if not val.accel_in_use():
                    val.accel_update_device()

    def accel_update_host(self, names):
        """Copy data objects from the device.

        This takes a dictionary with the same format as those used by the Operator
        provides() and requires() methods.

        Args:
            names (dict):  Dictionary of lists.

        Returns:
            None

        """
        for key in names["detdata"]:
            self.detdata.accel_update_host(key)
        for key in names["shared"]:
            self.shared.accel_update_host(key)
        for key in names["intervals"]:
            self.intervals.accel_update_host(key)
        for key, val in self._internal.items():
            if isinstance(val, AcceleratorObject):
                if val.accel_in_use():
                    val.accel_update_host()

    def accel_clear(self):
        self.detdata.accel_clear()
        self.shared.accel_clear()
        self.intervals.accel_clear()
        for key, val in self._internal.items():
            if isinstance(val, AcceleratorObject):
                if val.accel_exists():
                    val.accel_delete()
