# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import datetime
import io
import os
import re

import h5py
import numpy as np
from astropy import units as u

from ..instrument import Focalplane, GroundSite, SpaceSite
from ..intervals import IntervalList
from ..mpi import MPI, Comm
from ..observation import Observation
from ..timing import function_timer
from ..utils import Environment, Logger, import_from_name
from ..weather import SimWeather
from .spt3g_utils import (
    available,
    decompress_timestream,
    from_g3_array_type,
    from_g3_quats,
    from_g3_scalar_type,
    from_g3_time,
    from_g3_unit,
)

if available:
    from spt3g import core as c3g


def check_obs_range(obs, offset, nsamp):
    """Check that the requested sample range is valid."""
    log = Logger.get()
    if offset < 0 or offset >= obs.n_local_samples:
        msg = f"observation {obs.name}:  sample offset {offset} "
        msg += f"is outside the local range (0 - {obs.n_local_samples})"
        log.error(msg)
        raise RuntimeError(msg)

    if offset + nsamp < 0 or offset + nsamp > obs.n_local_samples:
        msg = f"observation {obs.name}:  end of samples "
        msg += f" {offset + nsamp} "
        msg += f"is outside the local range (0 - {obs.n_local_samples})"
        log.error(msg)
        raise RuntimeError(msg)


def import_shared(obs, name, offset, g3obj):
    """Copy a single G3Object into an observation shared field.

    Args:
        obs (Observation):  The parent observation.
        name (str):  The name of the shared object.
        offset (int):  The local data starting offset.
        g3obj (G3Object):  The G3Object to copy.

    Returns:
        None

    """
    if name not in obs.shared:
        raise KeyError(f"Shared object '{name}' does not exist in observation")

    if isinstance(g3obj, c3g.G3VectorTime):
        check_obs_range(obs, offset, len(g3obj))
        obs.shared[name].data[offset : offset + len(g3obj)] = from_g3_time(g3obj)
    elif isinstance(g3obj, c3g.G3VectorQuat):
        check_obs_range(obs, offset, len(g3obj))
        obs.shared[name].data[offset : offset + len(g3obj), :] = from_g3_quats(g3obj)
    else:
        nnz = 1
        for dim in obs.shared[name].shape[1:]:
            nnz *= dim
        # This is a slight cheat- using the internal MPIShared flat-packed
        # data buffer.
        nsamp = len(g3obj) // nnz
        check_obs_range(obs, offset, nsamp)
        obs.shared[name]._flat[offset * nnz : (offset + nsamp) * nnz] = g3obj


def import_detdata(obs, name, offset, g3obj, compression=None):
    """Copy a single G3Object into a detdata field.

    Args:
        obs (Observation):  The parent observation.
        name (str):  The name of the detdata object.
        offset (int):  The local data starting offset.
        g3obj (G3Object):  The G3Object to copy.

    Returns:
        None.

    """
    log = Logger.get()
    if name not in obs.detdata:
        raise KeyError(f"DetectorData object '{name}' does not exist in observation")

    nsamp = len(g3obj[g3obj.keys()[0]])
    check_obs_range(obs, offset, nsamp)

    if isinstance(g3obj, c3g.G3TimestreamMap):
        for d in obs.local_detectors:
            if compression is not None and d in compression:
                temp_ts = decompress_timestream(
                    g3obj[d],
                    compression[d]["gain"],
                    compression[d]["offset"],
                    compression[d]["units"],
                )
                obs.detdata[name][d][offset : offset + nsamp] = temp_ts
            else:
                obs.detdata[name][d][offset : offset + nsamp] = g3obj[d]
    elif isinstance(g3obj, (c3g.G3MapVectorDouble, c3g.G3MapVectorInt)):
        for d in obs.local_detectors:
            obs.detdata[name][d][offset : offset + nsamp] = g3obj[d]
    else:
        msg = f"Import of G3 object '{g3obj}' into a detdata object is not supported"
        log.error(msg)
        raise RuntimeError(msg)


def import_intervals(obs, name, times, g3obj):
    """Copy a G3Object containing intervals into an observation.

    This will import an IntervalsTime or G3VectorTime object and append the specified
    intervals to the observation intervals.

    Args:
        obs (Observation):  The parent observation.
        name (str):  The name of the intervals.
        times (str):  The shared timestamps to use.
        g3obj (G3Object):  The G3 object containing the time ranges.

    Returns:
        None

    """
    log = Logger.get()
    if name not in obs.intervals:
        raise KeyError(f"Intervals object '{name}' does not exist in observation")
    if times not in obs.shared:
        raise KeyError(f"Shared timestamps '{times}' does not exist in observation")

    timespans = list()
    try:
        if isinstance(g3obj, c3g.IntervalsTime):
            for segment in g3obj.segments:
                timespans.append((from_g3_time(segment[0]), from_g3_time(segment[1])))
    except AttributeError:
        # We don't have Intervals classes
        pass
    finally:
        if isinstance(g3obj, c3g.G3VectorTime):
            # These are flat-packed time ranges
            for seg in range(len(g3obj) // 2):
                seg_start = g3obj[2 * seg]
                seg_stop = g3obj[2 * seg + 1]
                timespans.append((from_g3_time(seg_start), from_g3_time(seg_stop)))

    ilist = IntervalList(obs.shared[times], timespans=timespans)
    obs.intervals[name] |= ilist


class import_obs_meta(object):
    """Default class to import Observation, Calibration, and Wiring frames.

    A list of frames is processed and the returned information consists of the metadata
    needed to construct an observation.  By default, arbitrary scalar metadata is
    imported.  Metadata arrays are not imported unless they are specified in the
    `meta_arrays` constructor argument.  By default, noise models are not imported
    unless they are specified in the `noise_models` constructor argument.

    Args:
        meta_arrays (list):  The observation metadata arrays to import.
        noise_models (list):  The noise models to import.

    """

    def __init__(
        self,
        meta_arrays=list(),
        noise_models=list(),
    ):
        self._meta_arrays = meta_arrays
        self._noise_models = noise_models
        self._obs_reserved = set(
            [
                "observation_name",
                "observation_uid",
                "observation_detector_sets",
                "telescope_name",
                "telescope_class",
                "telescope_uid",
                "site_name",
                "site_class",
                "site_uid",
                "site_lat_deg",
                "site_lon_deg",
                "site_alt_m",
                "session_name",
                "session_uid",
                "session_class",
                "session_start",
                "session_end",
                "site_weather_name",
                "site_weather_time",
                "site_weather_max_pwv",
                "site_weather_realization",
            ]
        )

    @function_timer
    def __call__(self, frames):
        """Process metadata frames.

        Args:
            frames (list):  The list of frames.

        Returns:
            (tuple):  The (observation name, observation UID, observation meta
                dictionary, observation det sets, Telescope, list of noise models)

        """
        name = None
        uid = None
        meta = dict()
        site = None
        focalplane = None
        telescope_class = None
        telescope_name = None
        telescope_uid = None
        telescope = None
        noise = list()
        detsets = None
        for frm in frames:
            if frm.type == c3g.G3FrameType.Observation:
                name = from_g3_scalar_type(frm["observation_name"])
                uid = from_g3_scalar_type(frm["observation_uid"])
                detsets = list()
                for dset in frm["observation_detector_sets"]:
                    detsets.append(list(dset))
                site_class = import_from_name(from_g3_scalar_type(frm["site_class"]))
                if "site_lat_deg" in frm:
                    # This is a GroundSite
                    #
                    # FIXME:  Currently we just support simulated weather objects.  A
                    # real experiment would likely overload this class anyway and
                    # construct a custom weather object.
                    weather_name = None
                    weather_realization = None
                    weather_max_pwv = None
                    weather_time = None
                    if "site_weather_name" in frm:
                        weather_name = from_g3_scalar_type(frm["site_weather_name"])
                        weather_realization = from_g3_scalar_type(
                            frm["site_weather_realization"]
                        )
                        weather_max_pwv = from_g3_scalar_type(
                            frm["site_weather_max_pwv"]
                        )
                        weather_time = datetime.datetime.fromtimestamp(
                            from_g3_time(frm["site_weather_time"]),
                            tz=datetime.timezone.utc,
                        )
                    site_uid = from_g3_scalar_type(frm["site_uid"])
                    weather = None
                    if weather_name is not None:
                        weather = SimWeather(
                            name=weather_name,
                            time=weather_time,
                            site_uid=site_uid,
                            realization=weather_realization,
                            max_pwv=weather_max_pwv,
                        )
                    site = site_class(
                        from_g3_scalar_type(frm["site_name"]),
                        from_g3_scalar_type(frm["site_lat_deg"]) * u.degree,
                        from_g3_scalar_type(frm["site_lon_deg"]) * u.degree,
                        from_g3_scalar_type(frm["site_alt_m"]) * u.meter,
                        uid=site_uid,
                        weather=weather,
                    )
                else:
                    # A SpaceSite
                    site = site_class(
                        from_g3_scalar_type(frm["site_name"]),
                        uid=from_g3_scalar_type(frm["site_uid"]),
                    )
                telescope_class = import_from_name(
                    from_g3_scalar_type(frm["telescope_class"])
                )
                telescope_name = from_g3_scalar_type(frm["telescope_name"])
                telescope_uid = from_g3_scalar_type(frm["telescope_uid"])

                session = None
                if "session_name" in frm.keys():
                    # We have session information
                    session_class = import_from_name(
                        from_g3_scalar_type(frm["session_class"])
                    )
                    session_name = from_g3_scalar_type(frm["session_name"])
                    session_uid = from_g3_scalar_type(frm["session_uid"])
                    session_start = None
                    try:
                        session_start = datetime.datetime.fromtimestamp(
                            from_g3_time(frm["session_start"]),
                            tz=datetime.timezone.utc,
                        )
                    except:
                        pass
                    session_end = None
                    try:
                        session_end = datetime.datetime.fromtimestamp(
                            from_g3_time(frm["session_end"]),
                            tz=datetime.timezone.utc,
                        )
                    except:
                        pass
                    session = session_class(
                        session_name,
                        uid=session_uid,
                        start=session_start,
                        end=session_end,
                    )

                meta = dict()
                unit_match = re.compile(r"^.*_units$")
                for f_key, f_val in frm.items():
                    if f_key in self._obs_reserved:
                        continue
                    if unit_match.match(f_key) is not None:
                        # This is a unit value for some other object
                        continue
                    if f_key in self._meta_arrays:
                        # This is an array we are importing
                        dt = from_g3_array_type(f_val)
                        meta[self._meta_arrays[f_key]] = np.array(f_val, dtype=dt)
                    else:
                        try:
                            l = len(f_val)
                            # This is an array
                        except Exception:
                            # This is a scalar (no len defined)
                            unit_key = f"{f_key}_units"
                            aunit_key = f"{f_key}_astropy_units"
                            unit_val = None
                            aunit = None
                            if unit_key in frm:
                                unit_val = frm[unit_key]
                                aunit = u.Unit(str(frm[aunit_key]))
                            try:
                                meta[f_key] = from_g3_scalar_type(f_val, unit_val).to(
                                    aunit
                                )
                            except Exception:
                                # This is not a datatype we can convert
                                pass

            elif frm.type == c3g.G3FrameType.Calibration:
                # Extract the focalplane and noise models
                byte_reader = io.BytesIO(np.array(frm["focalplane"], dtype=np.uint8))
                with h5py.File(byte_reader, "r") as f:
                    focalplane = Focalplane()
                    focalplane.load_hdf5(f)
                del byte_reader

                telescope = telescope_class(
                    telescope_name, uid=telescope_uid, focalplane=focalplane, site=site
                )

                # Make a fake obs for loading noise models
                if MPI is None:
                    fake_comm = None
                else:
                    fake_comm = MPI.COMM_SELF
                fake_obs = Observation(
                    Comm(world=fake_comm),
                    telescope,
                    1,
                    process_rows=1,
                )

                noise = dict()
                for frm_model, obs_model in self._noise_models:
                    if frm_model not in frm:
                        msg = f"Requested noise model '{frm_model}' is not in Calibration frame"
                        raise RuntimeError(msg)
                    noise_class = import_from_name(
                        from_g3_scalar_type(frm[f"{frm_model}_class"])
                    )
                    byte_reader = io.BytesIO(np.array(frm[frm_model], dtype=np.uint8))
                    noise[obs_model] = noise_class()
                    with h5py.File(byte_reader, "r") as f:
                        noise[obs_model].load_hdf5(f, fake_obs)
                    del byte_reader

                del fake_obs

        return name, uid, meta, detsets, telescope, session, noise


class import_obs_data(object):
    """Default class to import Scan frames.

    This provides a default importer of Scan frames into TOAST observation data.
    The timestamps_name tuple defines the frame key for the sampling times and the
    corresponding shared object name.

    Shared objects:  The `shared_names` list of tuples specifies the mapping from Scan
    frame key to TOAST shared key.  The data type will be converted to the most
    appropriate TOAST dtype.  Only sample-wise data is currently supported.  The format
    of each tuple in the list is:

        (frame key, observation shared key)

    DetData objects:  The `det_names` list of tuples specifies the mapping from Scan
    frame values to TOAST detdata objects.  `G3TimestreamMap` objects are assumed to
    have one element per sample.  Any `G3Map*` objects will assumed to be flat-packed
    and will be reshaped so that the leading dimension is number of samples.  The format
    of each tuple in the list is:

        (frame key, observation detdata key)

    Intervals objects:  The `interval_names` list of tuples specifies the mapping from
    Scan frame values to TOAST intervals.  The frame values can be either
    `IntervalsTime` objects filled with the start / stop times of each interval, or
    flat-packed start / stop times in a `G3VectorTime` object.  The format of each
    tuple in the list is:

        (frame key, observation intervals key)

    In addition to this list, an interval list named "frames" is created that describes
    the frame boundaries.

    Args:
        timestamp_names (tuple):  The name of the shared data containing the
            timestamps, and the output frame key to use.
        shared_names (list):  The observation shared objects to import.
        det_names (list):  The observation detdata objects to import.
        interval_names (list):  The observation intervals to import.

    """

    def __init__(
        self,
        timestamp_names=("times", "times"),
        shared_names=list(),
        det_names=list(),
        interval_names=list(),
    ):
        self._timestamp_names = timestamp_names
        self._shared_names = shared_names
        self._det_names = det_names
        self._interval_names = interval_names

    @function_timer
    def __call__(self, obs, frames):
        log = Logger.get()
        # Sanity check that the lengths of the frames correspond the number of local
        # samples.
        frame_total = np.sum([len(x[self._timestamp_names[0]]) for x in frames])
        if frame_total != obs.n_local_samples:
            msg = f"Process {obs.comm.group_rank} has {obs.n_local_samples} local samples, "
            msg += f"but is importing Scan frames with a total length of {frame_total}."
            log.error(msg)
            raise RuntimeError(msg)
        if frame_total == 0:
            return

        # Using the data types from the first frame, create the observation objects that
        # we will populate.

        # Timestamps are required
        frame_times, obs_times = self._timestamp_names
        obs.shared.create_column(
            obs_times,
            (obs.n_local_samples,),
            dtype=np.float64,
        )
        frame_zero_samples = len(frames[0][self._timestamp_names[0]])
        for frame_field, obs_field in self._shared_names:
            dt = None
            nnz = None
            if isinstance(frames[0][frame_field], c3g.G3VectorQuat):
                dt = np.float64
                nnz = 4
            else:
                dt = from_g3_array_type(frames[0][frame_field])
                nnz = len(frames[0][frame_field]) // frame_zero_samples
            sshape = (obs.n_local_samples,)
            if nnz > 1:
                sshape = (obs.n_local_samples, nnz)
            obs.shared.create_column(
                obs_field,
                sshape,
                dtype=dt,
            )

        for frame_field, obs_field in self._det_names:
            det_type_name = f"{frame_field}_dtype"
            dt = None
            if det_type_name in frames[0]:
                dt = np.dtype(str(frames[0][det_type_name]))
            else:
                dt = from_g3_array_type(frames[0][frame_field])
            units = u.dimensionless_unscaled
            if isinstance(frames[0][frame_field], c3g.G3TimestreamMap):
                check_units_name = (
                    f"compress_{frame_field}_{obs.local_detectors[0]}_units"
                )
                # If the compressed units name for the first detector is in the frame,
                # that means that we are (1) using compression and (2) the original
                # timestream had units (not just counts / dimensionless).
                if check_units_name in frames[0]:
                    units = from_g3_unit(frames[0][check_units_name])
                else:
                    units = from_g3_unit(frames[0][frame_field].units)
            nnz = len(frames[0][frame_field]) // frame_zero_samples
            dshape = None
            if nnz > 1:
                dshape = (nnz,)
            obs.detdata.create(obs_field, sample_shape=dshape, dtype=dt, units=units)

        for frame_field, obs_field in self._interval_names:
            obs.intervals.create_col(obs_field, list(), obs.shared[obs_times])

        # Go through each frame and copy the shared and detector data into the
        # observation.

        offset = 0
        for frm in frames:
            # Copy timestamps and shared data.  Because the data is explicitly
            # distributed in the sample direction, we know that there is only one
            # process accessing the data for each time slice
            import_shared(obs, obs_times, offset, frm[frame_times])
            for frame_field, obs_field in self._shared_names:
                import_shared(obs, obs_field, offset, frm[frame_field])

            # Copy detector data
            for frame_field, obs_field in self._det_names:
                comp = None
                # See if we have compression parameters for this object
                comp_root = f"compress_{frame_field}"
                for d in obs.local_detectors:
                    comp_gain_name = f"{comp_root}_{d}_gain"
                    comp_offset_name = f"{comp_root}_{d}_offset"
                    comp_units_name = f"{comp_root}_{d}_units"
                    if comp_offset_name in frm:
                        # This detector is compressed
                        if comp is None:
                            comp = dict()
                        comp[d] = dict()
                        comp[d]["offset"] = float(frm[comp_offset_name])
                        comp[d]["gain"] = float(frm[comp_gain_name])
                        comp[d]["units"] = c3g.G3TimestreamUnits(frm[comp_units_name])
                import_detdata(
                    obs, obs_field, offset, frm[frame_field], compression=comp
                )
            offset += len(frm[frame_times])

        # Now that we have the full set of timestamps in the observation, we
        # can construct our intervals.

        offset = 0
        sample_spans = list()
        for frm in frames:
            nsamp = len(frm[frame_times])
            sample_spans.append((offset, offset + nsamp - 1))
            for frame_field, obs_field in self._interval_names:
                import_intervals(obs, obs_field, obs_times, frm[frame_field])
            offset += nsamp


class import_obs(object):
    """Import class to build a toast Observation from spt3g frames.

    The frames may be already distributed, in which case each process should pass in
    their local list of frames.  The frames on each process may be in random order, and
    will be sorted and redistributed based on times.  If the frames are being imported
    from one process, then `import_rank` should specify the process passing in the
    data.  In this case, the input frames are ignored on all other processes.

    IMPORTANT:  It is assumed that all processes with Scan frames have an identical set
    of Observation and Calibration frames.

    Args:
        comm (toast.Comm):  The toast communicator.
        timestamps (str):  The name of the frame object containing the sample times.
        meta_import (Object):  Callable class that consumes Observation and Calibration
            frames.
        data_import (Object):  Callable class that consumes Scan frames.
        import_rank (int):  If not None, the process with all frames.

    """

    def __init__(
        self,
        comm,
        timestamps="times",
        meta_import=import_obs_meta(),
        data_import=import_obs_data(),
        import_rank=None,
    ):
        self._comm = comm
        self._timestamps = timestamps
        self._import_rank = import_rank
        self._meta_import = meta_import
        self._data_import = data_import

        self._nproc = self._comm.group_size
        self._rank = self._comm.group_rank

    @property
    def import_rank(self):
        return self._import_rank

    @function_timer
    def __call__(self, frames):
        """Copy spt3g frames into a toast Observation.

        This function returns an Observation with the data distributed in time slices.
        If an alternate distribution is desired, the `Observation.redistribute()`
        method should be used on the result.

        Args:
            frames (list):  The list of frames on each process.

        Returns:
            (Observation):  The TOAST observation.

        """
        lead_rank = 0
        participating = True
        if self._import_rank is not None:
            lead_rank = self._import_rank
            if self._rank != self._import_rank:
                participating = False

        # The Observation and Calibration frames should be duplicated on all processes.
        # We import metadata on one process and broadcast.
        obs_telescope = None
        obs_meta = None
        obs_name = None
        obs_uid = None
        obs_session = None
        obs_detsets = None
        obs_noise = None
        if self._rank == lead_rank:
            (
                obs_name,
                obs_uid,
                obs_meta,
                obs_detsets,
                obs_telescope,
                obs_session,
                obs_noise,
            ) = self._meta_import(frames)
        if self._comm.comm_group is not None:
            obs_telescope = self._comm.comm_group.bcast(obs_telescope, root=lead_rank)
            obs_name = self._comm.comm_group.bcast(obs_name, root=lead_rank)
            obs_meta = self._comm.comm_group.bcast(obs_meta, root=lead_rank)
            obs_uid = self._comm.comm_group.bcast(obs_uid, root=lead_rank)
            obs_session = self._comm.comm_group.bcast(obs_session, root=lead_rank)
            obs_detsets = self._comm.comm_group.bcast(obs_detsets, root=lead_rank)
            obs_noise = self._comm.comm_group.bcast(obs_noise, root=lead_rank)

        # Every process with data builds some information about their
        # scan frames, which can then be used to sort and redistribute them
        # if needed.
        local_frames = list()
        if participating:
            for ifrm, frm in enumerate(frames):
                if not frm.type == c3g.G3FrameType.Scan:
                    continue
                local_frames.append(
                    (
                        self._rank,
                        ifrm,
                        from_g3_time(frm[self._timestamps][0]),
                        len(frm[self._timestamps]),
                    )
                )
        all_frames = None
        if self._comm.comm_group is None:
            all_frames = [local_frames]
        else:
            all_frames = self._comm.comm_group.allgather(local_frames)
        sorted_frames = list()
        for proc_frames in all_frames:
            sorted_frames.extend(proc_frames)
        sorted_frames.sort(key=lambda x: x[2])

        # Compute the total samples and the sample sets (i.e. frames) that
        # will be used in the data distribution.
        total_samples = np.sum([x[3] for x in sorted_frames])
        sample_sets = [[x[3]] for x in sorted_frames]

        # Create the observation
        ob = Observation(
            self._comm,
            obs_telescope,
            total_samples,
            name=obs_name,
            uid=obs_uid,
            session=obs_session,
            detector_sets=obs_detsets,
            sample_sets=sample_sets,
            process_rows=1,
        )

        # Add the metadata
        ob.update(obs_meta)

        # Add the noise models
        for model_name, model in obs_noise.items():
            ob[model_name] = model

        # Given the distribution, compute the receiving process and frame location
        # for all of the frames.
        frame_dist = list()
        ifrm = 0
        for proc, (proc_offset, proc_nset) in enumerate(ob.dist.samp_sets):
            for pf in range(proc_nset):
                send_proc, send_loc, frame_start, frame_len = sorted_frames[ifrm]
                frame_dist.append((send_proc, send_loc, proc, proc_offset + pf))
                ifrm += 1

        # We cannot (easily) use alltoallv for lists of arbitrary objects (frames)
        # so instead we loop over destination processes and gather frames to each.
        obs_frames = None
        if ob.comm.comm_group is None:
            # Just sort our local scan frames
            obs_frames = [frames[x[1]] for x in sorted_frames]
        else:
            for receiver in range(ob.comm.group_size):
                send_frames = list()
                for finfo in frame_dist:
                    if finfo[0] == ob.comm.group_rank and finfo[2] == receiver:
                        # We are sending this frame
                        send_frames.append(frames[finfo[1]])
                recv_frames = ob.comm.comm_group.gather(send_frames, root=receiver)
                if ob.comm.group_rank == receiver:
                    obs_frames = [x for proc_frames in recv_frames for x in proc_frames]

        # Now every process copies their frames into the observation
        self._data_import(ob, obs_frames)

        return ob
