# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import io
import os

import h5py
import numpy as np
from astropy import units as u

from ..instrument import GroundSite, SpaceSite
from ..intervals import IntervalList
from ..timing import function_timer
from ..utils import Environment, Logger, object_fullname
from .spt3g_utils import (
    available,
    compress_timestream,
    to_g3_array_type,
    to_g3_map_array_type,
    to_g3_quats,
    to_g3_scalar_type,
    to_g3_time,
    to_g3_unit,
)

if available:
    from spt3g import core as c3g


@function_timer
def export_shared(obs, name, view_name=None, view_index=0, g3t=None):
    """Convert a single shared object to a G3Object.

    If the G3 data type is not specified, a guess will be made at the closest
    appropriate type.

    Args:
        obs (Observation):  The parent observation.
        name (str):  The name of the shared object.
        view_name (str):  If specified, use this view of the shared object.
        view_index (int):  Export this element of the list of data views.
        g3t (G3Object):  The specific G3Object type to use, or None.

    Returns:
        (G3Object):  The resulting G3 object.

    """
    if name not in obs.shared:
        raise KeyError(f"Shared object '{name}' does not exist in observation")
    if g3t is None:
        g3t = to_g3_array_type(obs.shared[name].dtype)

    sview = obs.shared[name].data
    if view_name is not None:
        sview = np.array(obs.view[view_name].shared[name][view_index], copy=False)

    if g3t == c3g.G3VectorTime:
        return to_g3_time(sview)
    elif g3t == c3g.G3VectorQuat:
        return to_g3_quats(sview)
    else:
        return g3t(sview.flatten().tolist())


@function_timer
def export_detdata(
    obs, name, view_name=None, view_index=0, g3t=None, times=None, compress=None
):
    """Convert a single detdata object to a G3Object.

    If the G3 data type is not specified, a guess will be made at the closest
    appropriate type.

    Args:
        obs (Observation):  The parent observation.
        name (str):  The name of the detdata object.
        view_name (str):  If specified, use this view of the detdata object.
        view_index (int):  Export this element of the list of data views.
        g3t (G3Object):  The specific G3Object type to use, or None.
        times (str):  If g3t is G3TimestreamMap, use this shared name for the
            timestamps.
        compress (bool or dict):  If True or a dictionary of compression parameters,
            store the timestreams as FLAC-compressed, 24-bit integers instead of
            uncompressed doubles.  Only used if g3t is a G3Timestream.

    Returns:
        (tuple):  The resulting G3 object, and compression parameters if enabled.

    """
    do_compression = False
    if isinstance(compress, bool) and compress:
        do_compression = True
    if isinstance(compress, dict):
        do_compression = True
    if name not in obs.detdata:
        raise KeyError(f"DetectorData object '{name}' does not exist in observation")
    if g3t == c3g.G3TimestreamMap and times is None:
        raise RuntimeError(
            f"If exporting a G3TimestreamMap, the times name must be specified"
        )
    gunit, scale = to_g3_unit(obs.detdata[name].units)

    g3array_type = None
    if g3t is None:
        dt = obs.detdata[name].dtype.char
        dshape = obs.detdata[name].detector_shape
        if (
            (dt == "f" or dt == "d")
            and len(dshape) == 1
            and dshape[0] == obs.n_local_samples
        ):
            g3t = c3g.G3TimestreamMap
            g3array_type = c3g.G3Timestream
        else:
            g3t = to_g3_map_array_type(obs.detdata[name].dtype)

    if g3array_type is None:
        g3array_type = to_g3_array_type(obs.detdata[name].dtype)

    out = g3t()

    tstart = None
    tstop = None
    compression = None

    if g3t == c3g.G3TimestreamMap:
        if view_name is None:
            tstart = to_g3_time(obs.shared[times][0])
            tstop = to_g3_time(obs.shared[times][-1])
        else:
            tstart = to_g3_time(obs.view[view_name].shared[times][view_index][0])
            tstop = to_g3_time(obs.view[view_name].shared[times][view_index][-1])
        if do_compression:
            compression = dict()

    dview = obs.detdata[name]
    if view_name is not None:
        dview = obs.view[view_name].detdata[name][view_index]

    for d in dview.detectors:
        if g3t == c3g.G3TimestreamMap:
            out[d] = c3g.G3Timestream(scale * dview[d], gunit)
            out[d].start = tstart
            out[d].stop = tstop
            if do_compression:
                out[d], comp_gain, comp_offset = compress_timestream(out[d], compress)
                compression[d] = dict()
                compression[d]["gain"] = comp_gain
                compression[d]["offset"] = comp_offset
        else:
            out[d] = g3array_type(dview[d].flatten().tolist())

    return out, gunit, compression


@function_timer
def export_intervals(obs, name, iframe):
    """Convert the named intervals into a G3 object.

    Args:
        obs (Observation):  The parent observation.
        name (str):  The name of the intervals.
        iframe (IntervalList):  An interval list defined for this frame.

    Returns:
        (G3Object):  The best container available.

    """
    overlap = iframe & obs.intervals[name]

    out = None
    try:
        out = c3g.IntervalsTime(
            [(to_g3_time(x.start), to_g3_time(x.stop)) for x in overlap]
        )
    except Exception:
        # Intervals objects not available
        out = c3g.G3VectorTime(
            [
                elem
                for x in overlap
                for elem in (to_g3_time(x.start), to_g3_time(x.stop))
            ]
        )
    return out


class export_obs_meta(object):
    """Default class to export Observation and Calibration frames.

    This provides a default exporter of TOAST Observation metadata into Observation
    and Calibration frames.

    The telescope and site information will be written to the Observation frame.  The
    focalplane information will be written to the Calibration frame.

    The observation metadata is exported according to the following rules.
    Scalar values are converted to the closest appropriate G3 type and stored in the
    Observation frame.  Additional metadata arrays can be exported by specifying
    a `meta_arrays` list of tuples containing the TOAST observation name and the
    corresponding output Observation frame key name.  The arrays will be converted
    to the most appropriate type.

    Noise models can be exported to the Calibration frame by specifying a list of
    tuples containing the TOAST and frame keys.

    Args:
        meta_arrays (list):  The observation metadata arrays to export.
        noise_models (list):  The noise models to export.

    """

    def __init__(
        self,
        meta_arrays=list(),
        noise_models=list(),
    ):
        self._meta_arrays = meta_arrays
        self._noise_models = noise_models

    @function_timer
    def __call__(self, obs):
        log = Logger.get()
        log.verbose(f"Create observation frame for {obs.name}")
        # Construct observation frame
        ob = c3g.G3Frame(c3g.G3FrameType.Observation)
        ob["observation_name"] = c3g.G3String(obs.name)
        ob["observation_uid"] = c3g.G3Int(obs.uid)
        ob["observation_detector_sets"] = c3g.G3VectorVectorString(
            obs.all_detector_sets
        )
        ob["telescope_name"] = c3g.G3String(obs.telescope.name)
        ob["telescope_class"] = c3g.G3String(object_fullname(obs.telescope.__class__))
        ob["telescope_uid"] = c3g.G3Int(obs.telescope.uid)
        site = obs.telescope.site
        ob["site_name"] = c3g.G3String(site.name)
        ob["site_class"] = c3g.G3String(object_fullname(site.__class__))
        ob["site_uid"] = c3g.G3Int(site.uid)
        if isinstance(site, GroundSite):
            ob["site_lat_deg"] = c3g.G3Double(site.earthloc.lat.to_value(u.degree))
            ob["site_lon_deg"] = c3g.G3Double(site.earthloc.lon.to_value(u.degree))
            ob["site_alt_m"] = c3g.G3Double(site.earthloc.height.to_value(u.meter))
            if site.weather is not None:
                if hasattr(site.weather, "name"):
                    # This is a simulated weather object, dump it.
                    ob["site_weather_name"] = c3g.G3String(site.weather.name)
                    ob["site_weather_realization"] = c3g.G3Int(site.weather.realization)
                    if site.weather.max_pwv is None:
                        ob["site_weather_max_pwv"] = c3g.G3String("NONE")
                    else:
                        ob["site_weather_max_pwv"] = c3g.G3Double(site.weather.max_pwv)
                    ob["site_weather_time"] = to_g3_time(site.weather.time.timestamp())
        session = obs.session
        if session is not None:
            ob["session_name"] = c3g.G3String(session.name)
            ob["session_class"] = c3g.G3String(object_fullname(session.__class__))
            ob["session_uid"] = c3g.G3Int(session.uid)
            if session.start is None:
                ob["session_start"] = c3g.G3String("NONE")
            else:
                ob["session_start"] = to_g3_time(session.start.timestamp())
            if session.end is None:
                ob["session_end"] = c3g.G3String("NONE")
            else:
                ob["session_end"] = to_g3_time(session.end.timestamp())
        m_export = set()
        for m_in, m_out in self._meta_arrays:
            out_type = to_g3_array_type(obs[m_in].dtype)
            ob[m_out] = out_type(obs[m_in])
            m_export.add(m_in)
        for m_key, m_val in obs.items():
            if m_key in m_export:
                # already done
                continue
            try:
                l = len(m_val)
                # This is an array
            except Exception:
                # This is a scalar (no len defined)
                try:
                    ob[m_key], m_unit = to_g3_scalar_type(m_val)
                    if m_unit is not None:
                        ob[f"{m_key}_astropy_units"] = c3g.G3String(f"{m_val.unit}")
                        ob[f"{m_key}_units"] = m_unit
                except Exception:
                    # This is not a datatype we can convert
                    pass

        # Construct calibration frame
        cal = c3g.G3Frame(c3g.G3FrameType.Calibration)

        # Serialize focalplane to HDF5 bytes and write to frame.
        byte_writer = io.BytesIO()
        with h5py.File(byte_writer, "w") as f:
            obs.telescope.focalplane.save_hdf5(f, comm=None, force_serial=True)
        cal["focalplane"] = c3g.G3VectorUnsignedChar(byte_writer.getvalue())
        del byte_writer

        # Serialize noise models
        for m_in, m_out in self._noise_models:
            byte_writer = io.BytesIO()
            with h5py.File(byte_writer, "w") as f:
                obs[m_in].save_hdf5(f, obs)
            cal[m_out] = c3g.G3VectorUnsignedChar(byte_writer.getvalue())
            del byte_writer
            cal[f"{m_out}_class"] = c3g.G3String(object_fullname(obs[m_in].__class__))

        return ob, cal


class export_obs_data(object):
    """Default class to export Scan frames.

    This provides a default exporter of TOAST observation shared, detdata, and
    intervals objects on each process.  The shared timestamps field is required.  An
    exception is raised if it is not specified.

    Shared objects:  The `shared_names` list of tuples specifies the TOAST shared key,
    corresponding Scan frame key, and optionally the G3 datatype to use.  Each process
    will duplicate shared data into their Scan frame stream.  If the G3 datatype is
    None, the closest G3 object type will be chosen.  If the shared object contains
    multiple values per sample, these are reshaped into a flat-packed array.  Only
    sample-wise shared objects are supported at this time (i.e. no other shared objects
    like beams, etc).  One special case:  The `timestamps` field will always be copied
    to each Scan frame as a `G3Timestream`.

    DetData objects:  The `det_names` list of tuples specifies the TOAST detdata key,
    the corresponding Scan frame key, and optionally the G3 datatype to use.  If the G3
    datatype is None, then detdata objects with one value per sample and of type
    float64 or float32 will be converted to a `G3TimestreamMap`.  All other detdata
    objects will be converted to the most appropriate `G3Map*` data type, with all data
    flat-packed.  In this case any processing code will need to interpret this data in
    conjunction with the separate `timestamps` frame key.

    Intervals objects:  The `interval_names` list of tuples specifies the TOAST
    interval name and associated Scan frame key.  We attempt to use an `IntervalsTime`
    object filled with the start / stop times of each interval, but if that is not
    available we flat-pack the start / stop times into a `G3VectorTime` object.

    Args:
        timestamp_names (tuple):  The name of the shared data containing the
            timestamps, and the output frame key to use.
        frame_intervals (str):  The name of the intervals to use for frame boundaries.
            If not specified, the observation sample sets are used.
        shared_names (list):  The observation shared objects to export.
        det_names (list):  The observation detdata objects to export.
        interval_names (list):  The observation intervals to export.
        compress (bool):  If True, attempt to use flac compression for all exported
            G3TimestreamMap objects.

    """

    def __init__(
        self,
        timestamp_names=("times", "times"),
        frame_intervals=None,
        shared_names=list(),
        det_names=list(),
        interval_names=list(),
        compress=False,
    ):
        self._timestamp_names = timestamp_names
        self._frame_intervals = frame_intervals
        self._shared_names = shared_names
        self._det_names = det_names
        self._interval_names = interval_names
        self._compress = compress

    @property
    def frame_intervals(self):
        return self._frame_intervals

    @function_timer
    def __call__(self, obs):
        log = Logger.get()
        frame_intervals = self._frame_intervals
        if frame_intervals is None:
            # We are using the sample set distribution for our frame boundaries.
            frame_intervals = "frames"
            timespans = list()
            offset = 0
            n_frames = 0
            first_set = obs.dist.samp_sets[obs.comm.group_rank].offset
            n_set = obs.dist.samp_sets[obs.comm.group_rank].n_elem
            for sset in range(first_set, first_set + n_set):
                for chunk in obs.dist.sample_sets[sset]:
                    timespans.append(
                        (
                            obs.shared[self._timestamp_names[0]][offset],
                            obs.shared[self._timestamp_names[0]][offset + chunk - 1],
                        )
                    )
                    n_frames += 1
                    offset += chunk
            obs.intervals.create_col(
                frame_intervals, timespans, obs.shared[self._timestamp_names[0]]
            )

        output = list()
        frame_view = obs.view[frame_intervals]
        for ivw, tview in enumerate(frame_view.shared[self._timestamp_names[0]]):
            msg = f"Create scan frame {obs.name}:{ivw} with fields:"
            msg += f"\n  shared:  {self._timestamp_names[1]}"
            nms = ", ".join([y for x, y, z in self._shared_names])
            msg += f", {nms}"
            nms = ", ".join([y for x, y, z in self._det_names])
            msg += f"\n  detdata:  {nms}"
            nms = ", ".join([y for x, y in self._interval_names])
            msg += f"\n  intervals:  {nms}"
            log.verbose(msg)
            # Construct the Scan frame
            frame = c3g.G3Frame(c3g.G3FrameType.Scan)
            # Add timestamps
            frame[self._timestamp_names[1]] = export_shared(
                obs,
                self._timestamp_names[0],
                view_name=frame_intervals,
                view_index=ivw,
                g3t=c3g.G3VectorTime,
            )
            for shr_key, shr_val, shr_type in self._shared_names:
                frame[shr_val] = export_shared(
                    obs,
                    shr_key,
                    view_name=frame_intervals,
                    view_index=ivw,
                    g3t=shr_type,
                )
            for det_key, det_val, det_type in self._det_names:
                frame[det_val], gunits, compression = export_detdata(
                    obs,
                    det_key,
                    view_name=frame_intervals,
                    view_index=ivw,
                    g3t=det_type,
                    times=self._timestamp_names[0],
                    compress=self._compress,
                )
                # Record the original detdata type, so that it can be reconstructed
                # later.
                det_type_name = f"{det_val}_dtype"
                frame[det_type_name] = c3g.G3String(obs.detdata[det_key].dtype.char)
                if compression is not None:
                    # Store per-detector compression parameters in the frame.  Also
                    # store the original units, since these are wiped by the
                    # compression.
                    froot = f"compress_{det_val}"
                    for d in obs.local_detectors:
                        frame[f"{froot}_{d}_gain"] = compression[d]["gain"]
                        frame[f"{froot}_{d}_offset"] = compression[d]["offset"]
                        frame[f"{froot}_{d}_units"] = gunits
            # If we are exporting intervals, create an interval list with a single
            # interval for this frame.  Then use this repeatedly in the intersection
            # calculation.
            if len(self._interval_names) > 0:
                tview = obs.view[frame_intervals].shared[self._timestamp_names[0]][ivw]
                iframe = IntervalList(
                    obs.shared[self._timestamp_names[0]],
                    timespans=[(tview[0], tview[-1])],
                )
                for ivl_key, ivl_val in self._interval_names:
                    frame[ivl_val] = export_intervals(
                        obs,
                        ivl_key,
                        iframe,
                    )
            output.append(frame)
        # Delete our temporary frame interval if we created it
        if self._frame_intervals is None:
            del obs.intervals[frame_intervals]

        return output


class export_obs(object):
    """Export class to emit spt3g frames of data built from an Observation.

    This function can generate spt3g frames on every process or can gather the data
    to one process and emit frames only on that process (if `export_rank` is specified).

    This function will emit Observation and Calibration frames on each process (or one
    process if `export_rank` is set), followed by one or more Scan frames on those
    processes.

    The `meta_export` and `data_export` parameters should specify the callable classes
    that will create the Observation / Calibration and Scan frames.

    By default, each process will export one single frame containing the local data.
    If `frame_intervals` is specified, then these are used to define the frame
    boundaries.

    Args:
        timestamps (str):  The name of the shared data containing the timestamps.
        meta_export (Object):  Callable class that takes an observation and produces
            Observation and Calibration frames.
        data_export (Object):  Callable class that takes an observation and produces
            Scan frames from the data on each process.
        export_rank (int):  If not None, emit all frames on this rank.

    """

    def __init__(
        self,
        timestamps="times",
        meta_export=export_obs_meta(),
        data_export=export_obs_data(),
        export_rank=None,
    ):
        self._timestamps = timestamps
        self._meta_export = meta_export
        self._data_export = data_export
        self._export_rank = export_rank

    @property
    def export_rank(self):
        return self._export_rank

    @function_timer
    def __call__(self, obs):
        """Generate spt3g frames from an Observation.

        Args:
            obs (Observation):  The observation to use.

        Returns:
            (list):  List of local frames.

        """
        # Ensure data is distributed by time.  If the observation
        # exporter defines existing frames to use, override the sample
        # sets to match those.
        redist_sampsets = False
        if self._data_export.frame_intervals is not None:
            # Create sample sets that match these frame boundaries
            if obs.comm_col_rank == 0:
                # First row of process grid gets local chunks
                local_sets = list()
                offset = 0
                for intr in obs.intervals[self._data_export.frame_intervals]:
                    chunk = intr.last - offset + 1
                    local_sets.append([chunk])
                    offset += chunk
                if offset != obs.n_local_samples:
                    local_sets.append([obs.n_local_samples - offset])
                # Gather across the row
                all_sets = [local_sets]
                if obs.comm_row is not None:
                    all_sets = obs.comm_row.gather(local_sets, root=0)
                if obs.comm_row_rank == 0:
                    redist_sampsets = list()
                    for pset in all_sets:
                        redist_sampsets.extend(pset)
            if obs.comm.comm_group is not None:
                redist_sampsets = obs.comm.comm_group.bcast(redist_sampsets, root=0)

        obs.redistribute(
            1,
            times=self._timestamps,
            override_sample_sets=redist_sampsets,
        )

        # Rank within the observation (group) communicator
        obs_rank = obs.comm.group_rank
        obs_nproc = obs.comm.group_size

        output = list()

        # Export metadata
        obs_frame, cal_frame = self._meta_export(obs)
        if self._export_rank is None or self._export_rank == obs_rank:
            # We are returning frames
            output.append(obs_frame)
            output.append(cal_frame)

        # Construct local scan frames for each frame interval
        local_frames = self._data_export(obs)

        # Communicate frames if needed
        if self._export_rank is None or obs_nproc == 1:
            # All processes are returning their local frames, or only one proc.
            output.extend(local_frames)
        else:
            # Gather all frames to one process.
            all_frames = obs.comm.comm_group.gather(
                local_frames, root=self._export_rank
            )
            if obs_rank == self._export_rank:
                for proc_frames in all_frames:
                    output.extend(proc_frames)

        return output
