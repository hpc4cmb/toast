# Copyright (c) 2021-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import ast
import json
import os
import re
from datetime import datetime, timezone

import h5py
import numpy as np
from astropy import units as u
from astropy.table import QTable
import flacarray

from ..instrument import Telescope, Session, Focalplane
from ..observation import Observation
from ..timing import Timer, function_timer
from ..utils import Environment, Logger, import_from_name
from ..weather import SimWeather, Weather
from .hdf_utils import (
    check_dataset_buffer_size,
    hdf5_config,
    hdf5_open,
    load_meta_object,
)

from .observation_hdf_load_v1 import load_hdf5 as load_hdf5_v1
from .observation_hdf_load_v1 import load_instrument as load_instrument_v1


@function_timer
def load_hdf5_shared(obs, hgrp, fields, log_prefix, parallel):
    log = Logger.get()

    timer = Timer()
    timer.start()

    # Get references to the distribution of detectors and samples
    proc_rows = obs.dist.process_rows
    proc_cols = obs.dist.comm.group_size // proc_rows
    dist_samps = obs.dist.samps
    dist_dets = obs.dist.det_indices

    serial_load = False
    if obs.comm.group_size > 1 and not parallel:
        # We are doing a serial load, but we have multiple processes
        # in the group.
        serial_load = True

    field_list = None
    if hgrp is not None:
        field_list = list(hgrp.keys())
    if serial_load and obs.comm.comm_group is not None:
        # Broadcast the field list
        field_list = obs.comm.comm_group.bcast(field_list, root=0)

    for field in field_list:
        if fields is not None and field not in fields:
            continue
        ds = None
        comm_type = None
        full_shape = None
        dtype = None
        if hgrp is not None:
            ds = hgrp[field]
            comm_type = ds.attrs["comm_type"]
            full_shape = ds.shape
            dtype = ds.dtype
        if serial_load:
            comm_type = obs.comm.comm_group.bcast(comm_type, root=0)
            full_shape = obs.comm.comm_group.bcast(full_shape, root=0)
            dtype = obs.comm.comm_group.bcast(dtype, root=0)

        slc = list()
        shape = list()
        if comm_type == "row":
            off = dist_dets[obs.comm.group_rank].offset
            nelem = dist_dets[obs.comm.group_rank].n_elem
            slc.append(slice(off, off + nelem))
            shape.append(nelem)
        elif comm_type == "column":
            off = dist_samps[obs.comm.group_rank].offset
            nelem = dist_samps[obs.comm.group_rank].n_elem
            slc.append(slice(off, off + nelem))
            shape.append(nelem)
        else:
            slc.append(slice(0, full_shape[0]))
            shape.append(full_shape[0])
        if len(full_shape) > 1:
            for dim in full_shape[1:]:
                slc.append(slice(0, dim))
                shape.append(dim)
        slc = tuple(slc)
        shape = tuple(shape)

        obs.shared.create_type(comm_type, field, shape, dtype)
        shcomm = obs.shared[field].comm

        # Load the data on one process of the communicator if loading in parallel.
        # If doing a serial load, the single reading process must communicate the
        # data to the rank zero process on each object comm.
        if (comm_type == "group") or (not serial_load):
            # Load data on the rank zero process and set
            data = None
            if shcomm is None or shcomm.rank == 0:
                msg = f"Shared field {field} ({comm_type})"
                check_dataset_buffer_size(msg, slc, dtype, parallel)
                data = np.array(ds[slc], copy=False).astype(obs.shared[field].dtype)
            obs.shared[field].set(data, fromrank=0)
            del data
        else:
            # More compilicated, since we have data distributed on along a process
            # row or column, but are loading data on one process.  First load full
            # data on the reader.
            full_data = None
            if obs.comm.group_rank == 0:
                full_data = np.array(ds[:], copy=False)

            # Note:  we could use a scatterv here instead of broadcasting the whole
            # thing, if this ever becomes worth the additional book-keeping.
            data = None
            if comm_type == "row" and obs.comm_row_rank == 0:
                # Distribute to the other rank zeros of the process rows
                if obs.comm_col is not None:
                    full_data = obs.comm_col.bcast(full_data, root=0)
                data = np.array(full_data[slc], dtype=obs.shared[field].dtype)
            elif comm_type == "column" and obs.comm_col_rank == 0:
                # Distribute to the other rank zeros of the process columns
                if obs.comm_row is not None:
                    full_data = obs.comm_row.bcast(full_data, root=0)
                data = np.array(full_data[slc], dtype=obs.shared[field].dtype)
            del full_data

            # Now set the data within each row / column
            obs.shared[field].set(data, fromrank=0)
            del data
        del ds

        if obs.comm.comm_group is not None:
            obs.comm.comm_group.barrier()
        log.verbose_rank(
            f"{log_prefix}  Shared finished {field} read in",
            comm=obs.comm.comm_group,
            timer=timer,
        )

    return


@function_timer
def load_hdf5_detdata(obs, hgrp, fields, log_prefix, parallel):
    log = Logger.get()

    timer = Timer()
    timer.start()

    # Get references to the distribution of detectors and samples
    dist_samps = obs.dist.samps
    dist_dets = obs.dist.det_indices

    # Data ranges for this process
    samp_off = dist_samps[obs.comm.group_rank].offset
    samp_nelem = dist_samps[obs.comm.group_rank].n_elem
    det_off = dist_dets[obs.comm.group_rank].offset
    det_nelem = dist_dets[obs.comm.group_rank].n_elem

    serial_load = False
    if obs.comm.group_size > 1 and not parallel:
        # We are doing a serial load, but we have multiple processes
        # in the group.
        serial_load = True

    field_list = None
    if hgrp is not None:
        field_list = list(hgrp.keys())
    if serial_load and obs.comm.comm_group is not None:
        # Broadcast the field list
        field_list = obs.comm.comm_group.bcast(field_list, root=0)

    for field in field_list:
        if fields is not None and field not in fields:
            continue
        ds = None
        units = None
        full_shape = None
        dtype = None
        orig_dtype = None

        compressed = False
        cgrp = None
        if hgrp is not None:
            if isinstance(hgrp[field], h5py.Dataset):
                # This is uncompressed data
                ds = hgrp[field]
                full_shape = ds.shape
                dtype = ds.dtype
                orig_dtype = dtype
                units = u.Unit(str(ds.attrs["units"]))
            else:
                # This must be a group of datasets containing compressed data.
                # FIXME: we should have a flacarray helper function to get array
                # properties without doing this manually.
                cgrp = hgrp[field]
                units = u.Unit(str(cgrp.attrs["units"]))
                orig_dtype = np.dtype(cgrp.attrs["dtype"])
                detector_shape = tuple(ast.literal_eval(cgrp.attrs["detector_shape"]))
                n_channel = int(cgrp.attrs["flac_channels"])
                starts = cgrp["stream_starts"]
                n_det = starts.shape[0]
                full_shape = (n_det,) + detector_shape
                if "stream_offsets" in cgrp:
                    if n_channel == 2:
                        dtype = np.dtype(np.float64)
                    else:
                        dtype = np.dtype(np.float32)
                else:
                    if n_channel == 2:
                        dtype = np.dtype(np.int64)
                    else:
                        dtype = np.dtype(np.int32)
                compressed = True

        if serial_load:
            compressed = obs.comm.comm_group.bcast(compressed, root=0)
            units = obs.comm.comm_group.bcast(units, root=0)
            full_shape = obs.comm.comm_group.bcast(full_shape, root=0)
            dtype = obs.comm.comm_group.bcast(dtype, root=0)
            orig_dtype = obs.comm.comm_group.bcast(orig_dtype, root=0)

        sample_shape = None
        if len(full_shape) > 2:
            sample_shape = full_shape[2:]

        # All processes create their local detector data
        obs.detdata.create(
            field,
            sample_shape=sample_shape,
            dtype=orig_dtype,
            detectors=obs.local_detectors,
            units=units,
        )

        # All processes independently load their data if running in parallel.
        # If loading serially, one process reads and sends blocks of detector
        # data.  We can do this since we previously checked that for serial loads
        # the data is distributed by detector.

        if compressed:
            # Load with flacarray
            mpi_dist = [(x.offset, x.offset + x.n_elem) for x in dist_dets]
            flcdata = (
                flacarray.hdf5.read_array(
                    cgrp,
                    keep=None,
                    stream_slice=None,
                    keep_indices=False,
                    mpi_comm=obs.comm.comm_group,
                    mpi_dist=mpi_dist,
                    use_threads=False,
                )
                .astype(orig_dtype)
                .reshape(obs.detdata[field].shape)
            )
            for idet, det in enumerate(obs.detdata[field].detectors):
                obs.detdata[field][idet] = flcdata[idet]
            del flcdata
        elif serial_load:
            # Uncompressed read and distribute
            for proc, detrange in enumerate(dist_dets):
                first_det = detrange.offset
                end_det = detrange.offset + detrange.n_elem
                n_local_det = detrange.n_elem
                pslice = (slice(0, n_local_det, 1), slice(0, obs.n_all_samples, 1))
                hslice = (
                    slice(first_det, end_det, 1),
                    slice(0, obs.n_all_samples, 1),
                )
                if obs.comm.group_rank == 0:
                    buffer = np.zeros((n_local_det, obs.n_all_samples), dtype=dtype)
                    ds.read_direct(buffer, hslice, pslice)
                    if proc == 0:
                        # Copy data into place
                        obs.detdata[field][:] = buffer
                    else:
                        # Send
                        obs.comm.comm_group.Send(
                            buffer.reshape(-1), dest=proc, tag=proc
                        )
                        del buffer
                elif obs.comm.group_rank == proc:
                    # Receive and store
                    buffer = np.zeros((n_local_det, obs.n_all_samples), dtype=dtype)
                    obs.comm.comm_group.Recv(buffer, source=0, tag=proc)
                    obs.detdata[field][:] = buffer
                    del buffer
        else:
            # Uncompressed read in parallel
            detdata_slice = [slice(0, det_nelem, 1), slice(0, samp_nelem, 1)]
            hf_slice = [
                slice(det_off, det_off + det_nelem, 1),
                slice(samp_off, samp_off + samp_nelem, 1),
            ]
            if len(full_shape) > 2:
                for dim in full_shape[2:]:
                    detdata_slice.append(slice(0, dim))
                    hf_slice.append(slice(0, dim))
            detdata_slice = tuple(detdata_slice)
            hf_slice = tuple(hf_slice)
            msg = f"Detdata field {field} (group rank {obs.comm.group_rank})"
            check_dataset_buffer_size(msg, hf_slice, dtype, parallel)
            ds.read_direct(obs.detdata[field].data, hf_slice, detdata_slice)

        if obs.comm.comm_group is not None:
            obs.comm.comm_group.barrier()
        log.verbose_rank(
            f"{log_prefix}  Detdata finished {field} read in",
            comm=obs.comm.comm_group,
            timer=timer,
        )
        del ds
        del cgrp


@function_timer
def load_hdf5_intervals(obs, hgrp, times, fields, log_prefix, parallel):
    log = Logger.get()

    timer = Timer()
    timer.start()

    serial_load = False
    if obs.comm.group_size > 1 and not parallel:
        # We are doing a serial load, but we have multiple processes
        # in the group.
        serial_load = True

    field_list = None
    if hgrp is not None:
        field_list = list(hgrp.keys())
    if serial_load:
        # Broadcast the field list
        field_list = obs.comm.comm_group.bcast(field_list, root=0)

    if obs.comm.comm_group is not None:
        obs.comm.comm_group.barrier()

    for field in field_list:
        if fields is not None and field not in fields:
            continue
        # The dataset
        ds = None
        global_times = None
        if obs.comm.group_rank == 0:
            ds = hgrp[field]
            global_times = np.transpose(ds[:])

        obs.intervals.create(field, global_times, times, fromrank=0)
        del ds

        if obs.comm.comm_group is not None:
            obs.comm.comm_group.barrier()
        log.verbose_rank(
            f"{log_prefix}  Intervals finished {field} read in",
            comm=obs.comm.comm_group,
            timer=timer,
        )


def load_instrument(parent_group, detectors=None, file_det_sets=None, comm=None):
    """Load instrument information from an HDF5 group."""
    new_detsets = file_det_sets
    toast_version = None
    has_session = False
    if parent_group is not None:
        inst_group = parent_group["instrument"]
        toast_version = int(inst_group.attrs["toast_format_version"])
        if "session_class" in inst_group.attrs:
            has_session = True
    if comm is not None:
        toast_version = comm.bcast(toast_version, root=0)
        has_session = comm.bcast(has_session, root=0)

    if toast_version < 2:
        return load_instrument_v1(
            parent_group, detectors=detectors, file_det_sets=file_det_sets, comm=comm
        )
    if toast_version > 2:
        msg = "load_instrument() found invalid file format "
        msg += f"version {toast_version}"
        raise RuntimeError(msg)

    tele = Telescope.load_hdf5(inst_group, comm=comm)
    if has_session:
        session = Session.load_hdf5(inst_group, comm=comm)
    else:
        session = None

    # If we are selecting only a subset of detectors, make a restricted
    # focalplane now and also modify detsets.

    if detectors is not None:
        raw_focalplane = tele.focalplane

        # Slice focalplane to include only these detectors.  Also modify
        # detector sets to include only these detectors.
        keep = set(detectors)
        fp_rows = [x["name"] in keep for x in raw_focalplane.detector_data]
        fp_data = QTable(raw_focalplane.detector_data[fp_rows])

        focalplane = Focalplane(
            detector_data=fp_data,
            sample_rate=raw_focalplane.sample_rate,
            field_of_view=raw_focalplane.field_of_view,
        )
        new_detsets = list()
        if isinstance(file_det_sets, list):
            # List of lists
            for ds in file_det_sets:
                new_ds = list()
                for d in ds:
                    if d in keep:
                        new_ds.append(d)
                if len(new_ds) > 0:
                    new_detsets.append(new_ds)
        else:
            # Must be a dictionary
            for dskey, ds in file_det_sets.items():
                new_ds = list()
                for d in ds:
                    if d in keep:
                        new_ds.append(d)
                if len(new_ds) > 0:
                    new_detsets.append(new_ds)
        tele.focalplane = focalplane

    if parent_group is not None:
        del inst_group
    return tele, session, new_detsets


@function_timer
def load_hdf5_obs_meta(
    comm,
    hgroup,
    parallel=False,
    log_prefix="",
    meta=None,
    detectors=None,
    process_rows=None,
):
    log = Logger.get()
    rank = comm.group_rank
    nproc = comm.group_size

    timer = Timer()
    timer.start()

    telescope = None
    obs_samples = None
    obs_name = None
    obs_uid = None
    session = None
    obs_det_sets = None
    obs_sample_sets = None
    all_det_flags = None

    if hgroup is not None:
        # Observation properties
        obs_name = str(hgroup.attrs["observation_name"])
        obs_uid = int(hgroup.attrs["observation_uid"])
        obs_dets = json.loads(hgroup.attrs["observation_detectors"])
        file_det_sets = None
        if hgroup.attrs["observation_detector_sets"] != "NONE":
            file_det_sets = json.loads(hgroup.attrs["observation_detector_sets"])
        obs_samples = int(hgroup.attrs["observation_samples"])
        obs_sample_sets = None
        if hgroup.attrs["observation_sample_sets"] != "NONE":
            obs_sample_sets = [
                [int(x) for x in y]
                for y in json.loads(hgroup.attrs["observation_sample_sets"])
            ]

        # Instrument properties
        telescope, session, obs_det_sets = load_instrument(
            hgroup, detectors=detectors, file_det_sets=file_det_sets, comm=None
        )

        # Per detector flags.
        all_det_flags = json.loads(hgroup.attrs["observation_detector_flags"])

    log.debug_rank(
        f"{log_prefix} Loaded instrument properties in",
        comm=comm.comm_group,
        timer=timer,
    )

    # Broadcast the observation properties if needed
    if not parallel and nproc > 1:
        telescope = comm.comm_group.bcast(telescope, root=0)
        obs_samples = comm.comm_group.bcast(obs_samples, root=0)
        obs_name = comm.comm_group.bcast(obs_name, root=0)
        obs_uid = comm.comm_group.bcast(obs_uid, root=0)
        session = comm.comm_group.bcast(session, root=0)
        obs_det_sets = comm.comm_group.bcast(obs_det_sets, root=0)
        obs_sample_sets = comm.comm_group.bcast(obs_sample_sets, root=0)
        all_det_flags = comm.comm_group.bcast(all_det_flags, root=0)

    # Create the observation
    obs = Observation(
        comm,
        telescope,
        obs_samples,
        name=obs_name,
        uid=obs_uid,
        session=session,
        detector_sets=obs_det_sets,
        sample_sets=obs_sample_sets,
        process_rows=process_rows,
    )

    # Set per-detector flags
    local_det_flags = dict()
    for det in obs.local_detectors:
        local_det_flags[det] = all_det_flags[det]
    obs.set_local_detector_flags(local_det_flags)

    # Load observation metadata.  This is complicated because a subset of processes
    # may have the file open, but the object loader may need the whole communicator
    # to load the object.  First we load all simple metadata and record the names
    # of more complicated objects to load.  Then we ensure that all processes have
    # this information.  Then all processes load more complicated objects
    # collectively.

    meta_load = dict()
    attr_load = dict()

    if hgroup is not None:
        meta_group = hgroup["metadata"]
        for obj_name in meta_group.keys():
            if obj_name == "other":
                # Simple python metadata- will process below
                continue
            obj = meta_group[obj_name]
            if meta is not None and obj_name not in meta:
                # The user restricted the list of things to load, and this is
                # not in the list.
                continue
            if isinstance(obj, h5py.Group):
                # This might be an object to restore
                if "class" in obj.attrs:
                    objclass = import_from_name(obj.attrs["class"])
                    test_obj = objclass()
                    if hasattr(test_obj, "load_hdf5"):
                        # Record this in the dictionary of things to load in the
                        # next step.
                        meta_load[obj_name] = test_obj
                    else:
                        msg = f"metadata object group '{obj_name}' has class "
                        msg += f"{obj.attrs['class']}, but instantiated "
                        msg += "object does not have a load_hdf5() method"
                        log.error(msg)
                        raise RuntimeError(msg)
                    continue
            # Warn that we are not loading this object
            msg = f"Found un-loadable metadata object '{obj_name}'.  Skipping."
            log.warn(msg)
        # Now load regular metadata into a python dictionary
        meta_other = meta_group["other"]
        other = load_meta_object(meta_other)
        meta_load.update(other)
        del other
        del meta_other
        del meta_group

        # Now process observation attribute objects
        attr_group = hgroup["attr"]
        for obj_name in attr_group.keys():
            obj = attr_group[obj_name]
            if isinstance(obj, h5py.Group):
                # This might be an object to restore
                if "class" in obj.attrs:
                    objclass = import_from_name(obj.attrs["class"])
                    test_obj = objclass()
                    if hasattr(test_obj, "load_hdf5"):
                        # Record this in the dictionary of things to load in the
                        # next step.
                        attr_load[obj_name] = test_obj
                    else:
                        msg = f"attr object group '{obj_name}' has class "
                        msg += f"{obj.attrs['class']}, but instantiated "
                        msg += "object does not have a load_hdf5() method"
                        log.error(msg)
                        raise RuntimeError(msg)
                    continue
            # Warn that we are not loading this object
            msg = f"Found un-loadable attribute object '{obj_name}'.  Skipping."
            log.warn(msg)
        del attr_group

    # Communicate the partial metadata
    if not parallel and nproc > 1:
        meta_load = comm.comm_group.bcast(meta_load, root=0)
        attr_load = comm.comm_group.bcast(attr_load, root=0)

    # Now load any remaining metadata objects

    meta_group = None
    attr_group = None
    if hgroup is not None:
        meta_group = hgroup["metadata"]
        attr_group = hgroup["attr"]
    for meta_key in list(meta_load.keys()):
        if hasattr(meta_load[meta_key], "load_hdf5"):
            handle = None
            if hgroup is not None:
                handle = meta_group[meta_key]
            meta_load[meta_key].load_hdf5(handle, obs)
            del handle
    del meta_group
    for attr_key in list(attr_load.keys()):
        if hasattr(attr_load[attr_key], "load_hdf5"):
            handle = None
            if hgroup is not None:
                handle = attr_group[attr_key]
            attr_load[attr_key].load_hdf5(handle, obs)
            del handle
    del attr_group

    # Assign the internal observation dictionary
    obs._internal = meta_load

    # Assign all class attributes
    for k, v in attr_load.items():
        setattr(obs, k, v)

    log.debug_rank(
        f"{log_prefix} Finished other metadata in",
        comm=comm.comm_group,
        timer=timer,
    )
    return obs


@function_timer
def load_hdf5(
    path,
    comm,
    process_rows=None,
    meta=None,
    detdata=None,
    shared=None,
    intervals=None,
    detectors=None,
    force_serial=False,
):
    """Load an HDF5 observation.

    By default, all detdata, shared, intervals, and noise models are loaded into
    memory.  A subset of objects may be specified with a list of names
    passed to the corresponding function arguments.

    Args:
        path (str):  The path to the file on disk.
        comm (toast.Comm):  The toast communicator to use.
        process_rows (int):  (Optional) The size of the rectangular process grid
            in the detector direction.  This number must evenly divide into the size of
            comm.  If not specified, defaults to the size of the communicator.
        meta (list):  Only load this list of metadata objects.
        detdata (list):  Only load this list of detdata objects.
        shared (list):  Only load this list of shared objects.
        intervals (list):  Only load this list of intervals objects.
        detectors (list):  Only load this list of detectors from all detector data
            objects.
        force_serial (bool):  If True, do not use HDF5 parallel support,
            even if it is available.

    Returns:
        (Observation):  The constructed observation.

    """
    log = Logger.get()
    env = Environment.get()

    rank = comm.group_rank
    nproc = comm.group_size
    if nproc == 1:
        # Force serial usage in this case, to avoid any MPI overhead
        force_serial = True

    timer = Timer()
    timer.start()
    log_prefix = f"HDF5 load {os.path.basename(path)}: "

    # Open the file and get the root group.
    hf = None
    hfgroup = None

    parallel, _, _ = hdf5_config(comm=comm.comm_group, force_serial=force_serial)
    if (
        (not parallel)
        and (process_rows is not None)
        and (process_rows != comm.group_size)
    ):
        msg = "When loading observations with serial HDF5, process_rows must equal "
        msg += "the group size"
        log.error(msg)
        raise RuntimeError(msg)

    hf = hdf5_open(path, "r", comm=comm.comm_group, force_serial=force_serial)
    hgroup = hf

    log.debug_rank(
        f"{log_prefix}  Opened file {path} in",
        comm=comm.comm_group,
        timer=timer,
    )

    # The rank zero process gets the file format version and communicates to all
    # processes in the group, regardless of whether they are participating in
    # the load.
    file_version = None
    if rank == 0:
        # Data format version check
        file_version = int(hgroup.attrs["toast_format_version"])
    if comm.comm_group is not None:
        file_version = comm.comm_group.bcast(file_version, root=0)

    if file_version < 2:
        # The v1 loader also deals with v0 data.
        del hgroup
        if hf is not None:
            hf.close()
        del hf
        return load_hdf5_v1(
            path,
            comm,
            process_rows=process_rows,
            meta=meta,
            detdata=detdata,
            shared=shared,
            intervals=intervals,
            detectors=detectors,
            force_serial=force_serial,
        )

    if file_version > 2:
        msg = f"HDF5 file '{path}' using unsupported data format {file_version}"
        log.error(msg)
        raise RuntimeError(msg)

    # Load all metadata into an empty Observation
    obs = load_hdf5_obs_meta(
        comm,
        hgroup,
        parallel=parallel,
        log_prefix="",
        meta=meta,
        detectors=detectors,
        process_rows=process_rows,
    )

    # Load shared data

    shared_group = None
    if hgroup is not None:
        shared_group = hgroup["shared"]
    load_hdf5_shared(obs, shared_group, shared, log_prefix, parallel)
    del shared_group
    log.debug_rank(
        f"{log_prefix} Finished shared data in",
        comm=comm.comm_group,
        timer=timer,
    )

    # Load intervals

    intervals_group = None
    intervals_times = None
    if hgroup is not None:
        intervals_group = hgroup["intervals"]
        intervals_times = intervals_group.attrs["times"]
    if not parallel and nproc > 1:
        intervals_times = comm.comm_group.bcast(intervals_times, root=0)
    load_hdf5_intervals(
        obs,
        intervals_group,
        obs.shared[intervals_times],
        intervals,
        log_prefix,
        parallel,
    )
    del intervals_group
    log.debug_rank(
        f"{log_prefix} Finished intervals in",
        comm=comm.comm_group,
        timer=timer,
    )

    # Load detector data

    detdata_group = None
    if hgroup is not None:
        detdata_group = hgroup["detdata"]
    load_hdf5_detdata(obs, detdata_group, detdata, log_prefix, parallel)
    del detdata_group
    log.debug_rank(
        f"{log_prefix} Finished detector data in",
        comm=comm.comm_group,
        timer=timer,
    )

    # Clean up
    del hgroup
    if hf is not None:
        hf.close()
    del hf

    return obs


def load_instrument_file(path, detectors=None, obs_det_sets=None, comm=None):
    """Load instrument information from an HDF5 file.

    This function loads the telescope and session serially on one process.
    It supports including a relative internal path inside the HDF5 file by separating
    the filesystem path from the internal path with a colon.  For example:

    path="/path/to/file.h5:/obs1

    The internal path should be to the *parent* group of the "instrument" group.

    """
    parts = path.split(":")
    if len(parts) == 1:
        file = parts[0]
        internal = "/"
    else:
        file = parts[0]
        internal = parts[1]
    grouptree = internal.split(os.path.sep)
    with h5py.File(file, "r") as hf:
        parent = hf
        for grp in grouptree:
            if grp == "":
                continue
            parent = parent[grp]
        telescope, session, _ = load_instrument(parent)
    return telescope, session
