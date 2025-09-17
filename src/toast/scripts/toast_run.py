#!/usr/bin/env python3

# Copyright (c) 2025-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script runs a TOAST simulation and / or processing pipeline
that is specified primarily with config files.  This parses all
config and command line options and runs an operator (usually
a Pipeline) named "main".

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


def main(opts=None):
    log = toast.utils.Logger.get()
    gt = toast.timing.GlobalTimers.get()
    gt.start("toast_run (total)")
    timer = toast.timing.Timer()
    timer.start()

    # Get optional MPI parameters
    comm, procs, rank = toast.get_world()

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

    mem = toast.utils.memreport(msg="(whole node)", comm=comm, silent=True)
    log.info_rank(f"Start of the workflow:  {mem}", comm=comm)

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
        "--log_config",
        required=False,
        default=None,
        help="Dump out config log to this file",
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

    # Log the config that was actually used at runtime.
    if otherargs.log_config is not None:
        toast.config.dump_config(otherargs.log_config, config, comm=comm)

    # Check that the required operator exists
    if not hasattr(job.operators, otherargs.main):
        msg = "The input config files do not specify an operator named "
        msg += f"'{otherargs.main}'"
        log.error_rank(msg, comm=comm)
        raise RuntimeError(msg)

    # Create communicators and empty data container.
    log = toast.utils.Logger.get()
    if runargs.group_size is not None:
        msg = "Using user-specifed process group size of {runargs.group_size}"
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

    # Run the main pipeline
    main = getattr(job.operators, otherargs.main)
    main.apply(data)

    # Collect optional timing information
    alltimers = toast.timing.gather_timers(comm=comm)
    if data.comm.world_rank == 0:
        out = os.path.join(otherargs.out_dir, "timing")
        toast.timing.dump(alltimers, out)

    mem = toast.utils.memreport(msg="(whole node)", comm=comm, silent=True)
    log.info_rank(f"End of the workflow:  {mem}", comm=comm)
    log.info_rank("Workflow completed in", comm=comm, timer=timer)

    # Cleanup
    data.clear()
    del data
    toast_comm.close()
    del toast_comm


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
