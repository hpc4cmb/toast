#!/usr/bin/env python3

# Copyright (c) 2025-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script runs a TOAST simulation and / or processing pipeline that is specified
primarily with config files.  This parses all config and command line options and
runs an operator (usually a Pipeline) named "main".

In order to support batched use of this workflow, stdout / stderr is redirected to a
log file within the specified output directory.

You can see the automatically generated command line options with:

    toast_run --help

This script contains just comments about what is going on.  For details about all the
options for a specific Operator, see the documentation or use the help() function from
an interactive python session.

"""

import argparse
import datetime
import os
import types

import toast
import toast.config
import toast.ops
import toast.traits
from toast.utils import stdouterr_redirected


def print_job(job):
    def dump(node, indent):
        spacer = " " * indent
        for k, v in vars(node).items():
            if isinstance(v, types.SimpleNamespace):
                # Descend
                new_indent = indent + 2
                print(f"{spacer}{k}:")
                dump(v, new_indent)
            elif isinstance(v, toast.traits.TraitConfig):
                print(f"{spacer}{k}:")
                for trait_name, trait in v.traits().items():
                    print(f"{spacer}  {trait_name} = {trait.get(v)}")
            else:
                print(f"{spacer}{k} = {v}")

    dump(job, 0)


def main(opts=None, comm=None):
    log = toast.utils.Logger.get()
    gt = toast.timing.GlobalTimers.get()
    gt.start("toast_run (total)")
    timer = toast.timing.Timer()
    timer.start()

    # Get optional MPI parameters
    procs = 1
    rank = 0
    if comm is not None:
        procs = comm.size
        rank = comm.rank

    # If the user has not told us to use multiple threads,
    # then just use one.

    if "OMP_NUM_THREADS" in os.environ:
        nthread = os.environ["OMP_NUM_THREADS"]
    else:
        nthread = "???"
        msg = "OMP_NUM_THREADS not set in the environment. "
        msg += "Job may try to use all cores for threads."
        log.warning_rank(msg, comm=comm)

    msg = f"Executing workflow with {procs} MPI tasks, each with "
    msg += f"{nthread} OpenMP threads at {datetime.datetime.now()}"
    log.info_rank(msg, comm=comm)

    # Argument parsing
    parser = argparse.ArgumentParser(description="Toast Pipeline Runner")

    parser.add_argument(
        "--out_dir",
        required=False,
        type=str,
        default="toast_out",
        help="The output directory",
    )
    parser.add_argument(
        "--out_config_name",
        required=False,
        type=str,
        default="run_config.yml",
        help="Dump out config log to this file within `out_dir`",
    )
    parser.add_argument(
        "--out_log_name",
        required=False,
        type=str,
        default=None,
        help="Redirect stdout / stderr to this file within `out_dir`",
    )
    parser.add_argument(
        "--main",
        required=False,
        type=str,
        default="main",
        help="The operator to run as the 'main'",
    )
    parser.add_argument(
        "--dry_run",
        required=False,
        default=False,
        action="store_true",
        help="If True, log the job config and exit",
    )

    # The operators and templates we want to configure from the command line
    # or a parameter file.
    config, otherargs, runargs = toast.config.run_config(parser, opts=opts)

    # Instantiate operators and templates
    job = toast.traits.create_from_config(config)

    # One process makes output directory
    if comm is None or comm.rank == 0:
        os.makedirs(otherargs.out_dir, exist_ok=True)
    if comm is not None:
        comm.barrier()

    # Log the config that was actually used at runtime.
    config_log = os.path.join(otherargs.out_dir, otherargs.out_config_name)
    toast.config.dump_config(config_log, config, comm=comm)

    # Check that the required operator exists
    if not hasattr(job.operators, otherargs.main):
        msg = "The input config files do not specify an operator named "
        msg += f"'{otherargs.main}'"
        log.error_rank(msg, comm=comm)
        raise RuntimeError(msg)

    # Create communicators and empty data container.
    log = toast.utils.Logger.get()
    if runargs.group_size is not None:
        msg = f"Using user-specifed process group size of {runargs.group_size}"
        log.info_rank(msg, comm=comm)
        group_size = runargs.group_size
    else:
        msg = "Using default process group size"
        log.info_rank(msg, comm=comm)
        if comm is None:
            group_size = 1
        else:
            group_size = comm.size

    if otherargs.dry_run:
        print_job(job)
        return

    toast_comm = toast.Comm(world=comm, groupsize=group_size)
    data = toast.Data(comm=toast_comm)

    # Redirect stdout / stderr during the run
    if otherargs.out_log_name is None:
        # Do not redirect
        main = getattr(job.operators, otherargs.main)
        main.apply(data)
    else:
        out_log = os.path.join(otherargs.out_dir, otherargs.out_log_name)
        with stdouterr_redirected(to=out_log, comm=comm, overwrite=False):
            # Run the main pipeline
            main = getattr(job.operators, otherargs.main)
            main.apply(data)

    # Collect optional timing information
    alltimers = toast.timing.gather_timers(comm=comm)
    if data.comm.world_rank == 0:
        out = os.path.join(otherargs.out_dir, "timing")
        toast.timing.dump(alltimers, out)
    log.info_rank("Workflow completed in", comm=comm, timer=timer)

    # Cleanup
    data.clear()
    del data
    toast_comm.close()
    del toast_comm


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main(comm=world)
