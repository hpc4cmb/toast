# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import re
from collections import OrderedDict
from collections.abc import MutableMapping

import numpy as np

from .accelerator import AcceleratorObject, accel_enabled
from .mpi import Comm
from .utils import Logger


class Data(MutableMapping):
    """Class which represents distributed data

    A Data object contains a list of observations assigned to
    each process group in the Comm.

    Args:
        comm (:class:`toast.Comm`):  The toast Comm class for distributing the data.
        view (bool):  If True, do not explicitly clear observation data on deletion.

    """

    def __init__(self, comm=Comm(), view=False):
        self._comm = comm
        self._view = view
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
        if not self._view:
            self.accel_clear()
            for ob in self.obs:
                ob.clear()
        self.obs.clear()
        if not self._view:
            self._internal.clear()
        return

    def all_local_detectors(self, selection=None):
        """Get the superset of local detectors in all observations.

        This builds up the result from calling `select_local_detectors()` on
        all observations.

        Returns:
            (list):  The list of all local detectors across all observations.

        """
        all_dets = OrderedDict()
        for ob in self.obs:
            dets = ob.select_local_detectors(selection=selection)
            for d in dets:
                if d not in all_dets:
                    all_dets[d] = None
        return list(all_dets.keys())

    def detector_units(self, det_data):
        """Get the detector data units for a given field.

        This verifies that the specified detector data field has the same
        units in all observations where it occurs, and returns that unit.

        Args:
            det_data (str):  The detector data field.

        Returns:
            (Unit):  The unit used across all observations.

        """
        log = Logger.get()
        local_units = None
        for ob in self.obs:
            if det_data not in ob.detdata:
                continue
            ob_units = ob.detdata[det_data].units
            if local_units is None:
                local_units = ob_units
            else:
                if ob_units != local_units:
                    msg = f"obs {ob.name} detdata {det_data} units "
                    msg += f"{ob_units} != {local_units}"
                    log.error(msg)
                    raise RuntimeError(msg)
        if self.comm.comm_world is None:
            det_units = local_units
        else:
            det_units = self.comm.comm_world.gather(local_units, root=0)
            if self.comm.world_rank == 0:
                for dtu in det_units:
                    if dtu != local_units:
                        msg = f"observations have different units "
                        msg += f"{dtu} != {local_units}"
                        log.error(msg)
                        raise RuntimeError(msg)
            # We know that every process is the same now
            det_units = local_units
        return det_units

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
        rcomm = self._comm.comm_group_rank

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

    def split(
        self,
        obs_index=False,
        obs_name=False,
        obs_uid=False,
        obs_session_name=False,
        obs_key=None,
        require_full=False,
    ):
        """Split the Data object.

        Create new Data objects that have views into unique subsets of the observations
        (the observations are not copied).  Only one "criteria" may be used to perform
        this splitting operation.  The observations may be split by index in the
        original list, by name, by UID, by session, or by the value of a specified key.

        The new Data objects are returned in a dictionary whose keys are the value of
        the selection criteria (index, name, uid, or value of the key).  Any observation
        that cannot be placed (because it is missing a name, uid or key) will be ignored
        and not added to any of the returned Data objects.  If the `require_full`
        parameter is set to True, such situations will raise an exception.

        Args:
            obs_index (bool):  If True, split by index in original list of observations.
            obs_name (bool):  If True, split by observation name.
            obs_uid (bool):  If True, split by observation UID.
            obs_session_name (bool):  If True, split by session name.
            obs_key (str):  Split by values of this observation key.
            require_full (bool):  If True, every observation must be placed in the
                output.

        Returns:
            (OrderedDict):  The dictionary of new Data objects.

        """
        log = Logger.get()
        check = (
            int(obs_index)
            + int(obs_name)
            + int(obs_uid)
            + int(obs_session_name)
            + int(obs_key is not None)
        )
        if check == 0 or check > 1:
            raise RuntimeError("You must specify exactly one split criteria")

        datasplit = OrderedDict()

        group_rank = self.comm.group_rank
        group_comm = self.comm.comm_group

        if obs_index:
            # Splitting by (unique) index
            for iob, ob in enumerate(self.obs):
                newdat = Data(comm=self._comm, view=True)
                newdat._internal = self._internal
                newdat.obs.append(ob)
                datasplit[iob] = newdat
        elif obs_name:
            # Splitting by (unique) name
            for iob, ob in enumerate(self.obs):
                if ob.name is None:
                    if require_full:
                        msg = f"require_full is True, but observation {iob} has no name"
                        log.error_rank(msg, comm=group_comm)
                        raise RuntimeError(msg)
                else:
                    newdat = Data(comm=self._comm, view=True)
                    newdat._internal = self._internal
                    newdat.obs.append(ob)
                    datasplit[ob.name] = newdat
        elif obs_uid:
            # Splitting by UID
            for iob, ob in enumerate(self.obs):
                if ob.uid is None:
                    if require_full:
                        msg = f"require_full is True, but observation {iob} has no UID"
                        log.error_rank(msg, comm=group_comm)
                        raise RuntimeError(msg)
                else:
                    newdat = Data(comm=self._comm, view=True)
                    newdat._internal = self._internal
                    newdat.obs.append(ob)
                    datasplit[ob.uid] = newdat
        elif obs_session_name:
            # Splitting by (non-unique) session name
            for iob, ob in enumerate(self.obs):
                if ob.session is None or ob.session.name is None:
                    if require_full:
                        msg = f"require_full is True, but observation {iob} has no session name"
                        log.error_rank(msg, comm=group_comm)
                        raise RuntimeError(msg)
                else:
                    sname = ob.session.name
                    if sname not in datasplit:
                        newdat = Data(comm=self._comm, view=True)
                        newdat._internal = self._internal
                        datasplit[sname] = newdat
                    datasplit[sname].obs.append(ob)
        elif obs_key is not None:
            # Splitting by arbitrary key.  Unlike name / uid which are built it to the
            # observation class, arbitrary keys might be modified in different ways
            # across all processes in a group.  For this reason, we do an additional
            # check for consistent values across the process group.
            for iob, ob in enumerate(self.obs):
                if obs_key not in ob:
                    if require_full:
                        msg = f"require_full is True, but observation {iob} has no key '{obs_key}'"
                        log.error_rank(msg, comm=group_comm)
                        raise RuntimeError(msg)
                else:
                    obs_val = ob[obs_key]
                    # Get the values from all processes in the group
                    group_vals = None
                    if group_comm is None:
                        group_vals = [obs_val]
                    else:
                        group_vals = group_comm.allgather(obs_val)
                    if group_vals.count(group_vals[0]) != len(group_vals):
                        msg = f"observation {iob}, key '{obs_key}' has inconsistent values across processes"
                        log.error_rank(msg, comm=group_comm)
                        raise RuntimeError(msg)
                    if obs_val not in datasplit:
                        newdat = Data(comm=self._comm, view=True)
                        newdat._internal = self._internal
                        datasplit[obs_val] = newdat
                    datasplit[obs_val].obs.append(ob)
        return datasplit

    def select(
        self,
        obs_index=None,
        obs_name=None,
        obs_uid=None,
        obs_session_name=None,
        obs_key=None,
        obs_val=None,
    ):
        """Create a new Data object with a subset of observations.

        The returned Data object just has a view of the original observations (they
        are not copied).

        The list of observations in the new Data object is a logical OR of the
        criteria passed in:
            * Index location in the original list of observations
            * Name of the observation
            * UID of the observation
            * Session of the observation
            * Existence of the specified dictionary key
            * Required value of the specified dictionary key

        Args:
            obs_index (int):  Observation location in the original list.
            obs_name (str):  The observation name or a compiled regular expression
                object to use for matching.
            obs_uid (int):  The observation UID to select.
            obs_session_name (str):  The name of the session.
            obs_key (str):  The observation dictionary key to examine.
            obs_val (str):  The required value of the observation dictionary key or a
                compiled regular expression object to use for matching.

        Returns:
            (Data):  A new Data object with references to the orginal metadata and
                a subset of observations.

        """
        log = Logger.get()
        if obs_val is not None and obs_key is None:
            raise RuntimeError("If you specify obs_val, you must also specify obs_key")

        group_rank = self.comm.group_rank
        group_comm = self.comm.comm_group

        new_data = Data(comm=self._comm, view=True)

        # Use a reference to the original metadata
        new_data._internal = self._internal

        for iob, ob in enumerate(self.obs):
            if obs_index is not None and obs_index == iob:
                new_data.obs.append(ob)
                continue
            if obs_name is not None and ob.name is not None:
                if isinstance(obs_name, re.Pattern):
                    if obs_name.match(ob.name) is not None:
                        new_data.obs.append(ob)
                        continue
                elif obs_name == ob.name:
                    new_data.obs.append(ob)
                    continue
            if obs_uid is not None and ob.uid is not None and obs_uid == ob.uid:
                new_data.obs.append(ob)
                continue
            if (
                obs_session_name is not None
                and ob.session is not None
                and obs_session_name == ob.session.name
            ):
                new_data.obs.append(ob)
                continue
            if obs_key is not None and obs_key in ob:
                # Get the values from all processes in the group and check
                # for consistency.
                group_vals = None
                if group_comm is None:
                    group_vals = [ob[obs_key]]
                else:
                    group_vals = group_comm.allgather(ob[obs_key])
                if group_vals.count(group_vals[0]) != len(group_vals):
                    msg = f"observation {iob}, key '{obs_key}' has inconsistent values across processes"
                    log.error_rank(msg, comm=group_comm)
                    raise RuntimeError(msg)

                if obs_val is None:
                    # We have the key, and are accepting any value
                    new_data.obs.append(ob)
                    continue
                elif isinstance(obs_val, re.Pattern):
                    if obs_val.match(ob[obs_key]) is not None:
                        # Matches our regex
                        new_data.obs.append(ob)
                        continue
                elif obs_val == ob[obs_key]:
                    new_data.obs.append(ob)
                    continue
        return new_data

    # Accelerator use

    def accel_create(self, names):
        """Create a set of data objects on the device.

        This takes a dictionary with the same format as those used by the Operator
        provides() and requires() methods.  If the data already exists on the
        device then no action is taken.

        Args:
            names (dict):  Dictionary of lists.

        Returns:
            None

        """
        log = Logger.get()
        if not accel_enabled():
            log.verbose(f"accel_enabled is False, canceling accel_create.")
            return

        for ob in self.obs:
            for objname, objmgr in [
                ("detdata", ob.detdata),
                ("shared", ob.shared),
                ("intervals", ob.intervals),
            ]:
                for key in names[objname]:
                    if key not in objmgr:
                        msg = f"ob {ob.name} {objname} accel_create '{key}' "
                        msg += f"not present, ignoring"
                        log.verbose(msg)
                        continue
                    if objmgr.accel_exists(key):
                        msg = f"ob {ob.name} {objname}: accel_create '{key}'"
                        msg += f" already exists"
                        log.verbose(msg)
                    else:
                        log.verbose(f"ob {ob.name} {objname}: accel_create '{key}'")
                        objmgr.accel_create(key)

        for key in names["global"]:
            val = self._internal.get(key, None)
            if isinstance(val, AcceleratorObject):
                if not val.accel_exists():
                    log.verbose(f"Data accel_create: '{key}'")
                    val.accel_create(key)
                else:
                    log.verbose(f"Data accel_create: '{key}' already on device")
            else:
                log.verbose(
                    f"Data accel_create: '{key}' ({type(val)}) is not an AcceleratorObject"
                )

    def accel_update_device(self, names):
        """Copy a set of data objects to the device.

        This takes a dictionary with the same format as those used by the Operator
        provides() and requires() methods.

        Args:
            names (dict):  Dictionary of lists.

        Returns:
            None

        """
        if not accel_enabled():
            return
        log = Logger.get()

        for ob in self.obs:
            for objname, objmgr in [
                ("detdata", ob.detdata),
                ("shared", ob.shared),
                ("intervals", ob.intervals),
            ]:
                for key in names[objname]:
                    if key not in objmgr:
                        msg = f"ob {ob.name} {objname} update_device key '{key}'"
                        msg += f" not present, ignoring"
                        log.verbose(msg)
                        continue
                    if not objmgr.accel_exists(key):
                        msg = f"ob {ob.name} {objname} update_device key '{key}'"
                        msg += f" does not exist on accelerator"
                        log.error(msg)
                        raise RuntimeError(msg)
                    if objmgr.accel_in_use(key):
                        msg = f"ob {ob.name} {objname}: skip update_device '{key}'"
                        msg += f" already in use"
                        log.verbose(msg)
                    else:
                        log.verbose(f"ob {ob.name} {objname}: update_device '{key}'")
                        objmgr.accel_update_device(key)

        for key in names["global"]:
            val = self._internal.get(key, None)
            if isinstance(val, AcceleratorObject):
                if val.accel_in_use():
                    msg = f"Skipping update_device for '{key}', "
                    msg += "device data in use"
                    log.verbose(msg)
                else:
                    log.verbose(f"Calling Data update_device for '{key}'")
                    val.accel_update_device()
            else:
                msg = f"Data accel_update_device: '{key}' ({type(val)}) "
                msg += "is not an AcceleratorObject"
                log.verbose(msg)

    def accel_update_host(self, names):
        """Copy a set of data objects to the host.

        This takes a dictionary with the same format as those used by the Operator
        provides() and requires() methods.

        Args:
            names (dict):  Dictionary of lists.

        Returns:
            None

        """
        if not accel_enabled():
            return
        log = Logger.get()

        for ob in self.obs:
            for objname, objmgr in [
                ("detdata", ob.detdata),
                ("shared", ob.shared),
                ("intervals", ob.intervals),
            ]:
                for key in names[objname]:
                    if key not in objmgr:
                        msg = f"ob {ob.name} {objname} update_host key '{key}'"
                        msg += f" not present, ignoring"
                        log.verbose(msg)
                        continue
                    if not objmgr.accel_exists(key):
                        msg = f"ob {ob.name} {objname} update_host key '{key}'"
                        msg += f" does not exist on accelerator, ignoring"
                        log.verbose(msg)
                        continue
                    if not objmgr.accel_in_use(key):
                        msg = f"ob {ob.name} {objname}: skip update_host, '{key}'"
                        msg += f" already on host"
                        log.verbose(msg)
                    else:
                        log.verbose(f"ob {ob.name} {objname}: update_host '{key}'")
                        objmgr.accel_update_host(key)

        for key in names["global"]:
            val = self._internal.get(key, None)
            if isinstance(val, AcceleratorObject):
                if not val.accel_in_use():
                    msg = f"Skipping update_host for '{key}', "
                    msg += "host data already in use"
                    log.verbose(msg)
                else:
                    log.verbose(f"Calling Data update_host for '{key}'")
                    val.accel_update_host()
            else:
                msg = f"Data accel_update_host: '{key}' ({type(val)}) "
                msg += "is not an AcceleratorObject"
                log.verbose(msg)

    def accel_delete(self, names):
        """Delete a specific set of device objects

        This takes a dictionary with the same format as those used by the Operator
        provides() and requires() methods.

        Args:
            names (dict):  Dictionary of lists.

        Returns:
            None

        """
        if not accel_enabled():
            return
        log = Logger.get()

        for ob in self.obs:
            for objname, objmgr in [
                ("detdata", ob.detdata),
                ("shared", ob.shared),
                ("intervals", ob.intervals),
            ]:
                for key in names[objname]:
                    if key not in objmgr:
                        msg = f"ob {ob.name} {objname} accel_delete key '{key}'"
                        msg += f" not present, ignoring"
                        log.verbose(msg)
                        continue
                    if objmgr.accel_exists(key):
                        log.verbose(f"ob {ob.name} {objname}: accel_delete '{key}'")
                        objmgr.accel_delete(key)
                    else:
                        msg = f"ob {ob.name} {objname}: accel_delete '{key}'"
                        msg += f" not present on device"
                        log.verbose(msg)

        for key in names["global"]:
            val = self._internal.get(key, None)
            if isinstance(val, AcceleratorObject):
                if val.accel_exists():
                    log.verbose(f"Calling Data accel_delete for '{key}'")
                    val.accel_delete()
            else:
                msg = f"Data accel_delete: '{key}' ({type(val)}) "
                msg += "is not an AcceleratorObject"
                log.verbose(msg)

    def accel_clear(self):
        """Delete all accelerator data."""
        if not accel_enabled():
            return
        log = Logger.get()

        for ob in self.obs:
            ob.accel_clear()

        for key, val in self._internal.items():
            if isinstance(val, AcceleratorObject):
                if val.accel_exists():
                    val.accel_delete()
            else:
                msg = f"Data accel_clear: '{key}' ({type(val)}) "
                msg += "is not an AcceleratorObject"
                log.verbose(msg)
