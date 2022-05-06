#!/usr/bin/env python3

# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script loads HDF5 format data and exports to SPT3G format data.
"""

import argparse
import os
import re
import shutil
import sys

import numpy as np
from astropy import units as u
from spt3g import core as c3g

import toast
import toast.ops
from toast import spt3g as t3g
from toast.observation import default_values as defaults
from toast.timing import dump, gather_timers


def parse_arguments():
    """
    Defines and parses the arguments for the script.
    """
    # defines the parameters of the script
    parser = argparse.ArgumentParser(
        description="Convert TOAST HDF5 data to SPT3G format"
    )

    # The operators we want to configure from the command line or a parameter file.
    operators = [
        toast.ops.LoadHDF5(
            name="load_hdf5",
        ),
        toast.ops.SaveSpt3g(name="save_spt3g"),
    ]

    # Parse all of the operator configuration
    config, args, jobargs = toast.parse_config(parser, operators=operators)

    return config, args, jobargs


def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()
    env.enable_function_timers()
    global_timer = toast.timing.GlobalTimers.get()
    global_timer.start("toast_hdf5_to_spt3g (total)")

    config, args, jobargs = parse_arguments()

    # Instantiate our objects that were configured from the command line / files
    job = toast.create_from_config(config)
    job_ops = job.operators

    # Use a group size of one
    comm = toast.Comm(groupsize=1)

    # Create the (initially empty) data
    data = toast.Data(comm=comm)

    # Load the data.
    log.info_rank(
        f"Loading HDF5 data from {job_ops.load_hdf5.volume}", comm=comm.comm_world
    )
    job_ops.load_hdf5.process_rows = 1
    job_ops.load_hdf5.apply(data)

    # Build up the lists of objects to export from the first observation

    noise_models = list()
    meta_arrays = list()
    shared = list()
    detdata = list()
    intervals = list()

    msg = "Exporting observation fields:"

    ob = data.obs[0]
    for k, v in ob.shared.items():
        g3name = f"shared_{k}"
        if re.match(r".*boresight.*", k) is not None:
            # These are quaternions
            msg += f"\n  (shared):    {k} (quaternions)"
            shared.append((k, g3name, c3g.G3VectorQuat))
        elif k == defaults.times:
            # Timestamps are handled separately
            continue
        else:
            msg += f"\n  (shared):    {k}"
            shared.append((k, g3name, None))
    for k, v in ob.detdata.items():
        msg += f"\n  (detdata):   {k}"
        detdata.append((k, k, None))
    for k, v in ob.intervals.items():
        msg += f"\n  (intervals): {k}"
        intervals.append((k, k))
    for k, v in ob.items():
        if isinstance(v, toast.noise.Noise):
            msg += f"\n  (noise):     {k}"
            noise_models.append((k, k))
        elif isinstance(v, np.ndarray) and len(v.shape) > 0:
            if isinstance(v, u.Quantity):
                raise NotImplementedError("Writing array quantities not yet supported")
            msg += f"\n  (meta arr):  {k}"
            meta_arrays.append((k, k))
        else:
            msg += f"\n  (meta):      {k}"

    log.info_rank(msg, comm=comm.comm_world)

    # Export the data

    meta_exporter = t3g.export_obs_meta(
        noise_models=noise_models,
        meta_arrays=meta_arrays,
    )
    data_exporter = t3g.export_obs_data(
        shared_names=shared,
        det_names=detdata,
        interval_names=intervals,
        compress=True,
    )
    exporter = t3g.export_obs(
        meta_export=meta_exporter,
        data_export=data_exporter,
        export_rank=0,
    )

    log.info_rank(
        f"Exporting SPT3G data to {job_ops.save_spt3g.directory}", comm=comm.comm_world
    )
    job_ops.save_spt3g.obs_export = exporter
    job_ops.save_spt3g.purge = True
    job_ops.save_spt3g.apply(data)

    # dumps all the timing information
    global_timer.stop("toast_hdf5_to_spt3g (total)")
    alltimers = gather_timers(comm=comm.comm_world)
    if comm.world_rank == 0:
        out = os.path.join(job_ops.save_spt3g.directory, "timing")
        dump(alltimers, out)


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
