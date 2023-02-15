# Copyright (c) 2021-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


import numpy as np
from astropy import units as u

from ..mpi import MPI
from ..timing import Timer, function_timer
from ..utils import Logger
from .hdf_utils import check_dataset_buffer_size


@function_timer
def load_hdf5_detdata_v0(obs, hgrp, fields, log_prefix, parallel):
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
        if hgrp is not None:
            ds = hgrp[field]
            full_shape = ds.shape
            dtype = ds.dtype
            units = u.Unit(str(ds.attrs["units"]))
        if serial_load:
            units = obs.comm.comm_group.bcast(units, root=0)
            full_shape = obs.comm.comm_group.bcast(full_shape, root=0)
            dtype = obs.comm.comm_group.bcast(dtype, root=0)

        sample_shape = None
        if len(full_shape) > 2:
            sample_shape = full_shape[2:]

        # All processes create their local detector data
        obs.detdata.create(
            field,
            sample_shape=sample_shape,
            dtype=dtype,
            detectors=obs.local_detectors,
            units=units,
        )

        # All processes independently load their data if running in
        # parallel.  If loading serially, one process reads ands broadcasts.
        # We implement it this way instead of using a scatter, since the
        # data for each process is not contiguous in the dataset.

        if serial_load:
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
