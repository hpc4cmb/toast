# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Tools for starting up interactive sessions."""

import os
import shutil

from IPython.core.extensions import ExtensionManager

from ..mpi import MPI
from ..utils import Environment, Logger


def start_parallel(
    procs=1, threads=1, nice=True, auto_mpi=False, shell=None, shell_line=None
):
    """Attempt to start up ipyparallel with mpi4py and OpenMP.

    Args:
        procs (int):  The number of processes to use.
        threads (int):  The number of OpenMP threads per process.
        nice (bool):  If True, "nice" the processes to a lower
            priority.

    Returns:
        (int):  The number of processes started.

    """
    log = Logger.get()
    env = Environment.get()

    if procs > 1:
        try:
            import ipyparallel as ipp

            controller_class = ipp.cluster.launcher.MPIControllerLauncher
            engine_class = ipp.cluster.launcher.MPIEngineSetLauncher
            if "SLURM_JOB_ID" in os.environ:
                print("Running in SLURM")
                srun_path = shutil.which("srun")
                if procs != int(os.environ["SLURM_NTASKS"]):
                    print(
                        f"WARNING:  Slurm environment has {os.environ['SLURM_NTASKS']} processes, not {procs}"
                    )
                controller_class.mpi_cmd = [srun_path]
                engine_class.mpi_cmd = [srun_path]
            cluster = ipp.Cluster(
                controller=controller_class,
                engines=engine_class,
                engine_timeout=120,
                n=procs,
            )

            cluster = ipp.Cluster(engines="mpi", n=procs)
            client = cluster.start_and_connect_sync()
            client.block = True
            if nice:
                # Optionally nice the individual processes if running on a
                # shared node.
                import psutil

                psutil.Process().nice(
                    20 if psutil.POSIX else psutil.IDLE_PRIORITY_CLASS
                )
            # Optionally enable automatic use of MPI
            if auto_mpi and shell is not None:
                # Turn on automatic use of MPI
                shell.run_line_magic(shell_line, "autopx")
        except Exception as e:
            import traceback

            tb_str = "".join(traceback.format_exception(e))
            print(e)
            print(tb_str)
            procs = 1
            print("Failed to start ipyparallel cluster, using one process.")
    extmanager = ExtensionManager()
    try:
        _ = extmanager.load_extension("wurlitzer")
    except Exception:
        # Must be running outside IPython shell
        pass

    if procs == 1:
        rank = 0
        comm = None
        os.environ["MPI_DISABLE"] = "1"
        os.environ["DISABLE_MPI"] = "1"
    else:
        comm = MPI.COMM_WORLD
        rank = comm.rank

    max_threads = env.max_threads()
    if threads > max_threads:
        msg = f"Requested threads per process ({threads}) exceeds"
        msg += f" the maximum ({max_threads}).  Using {max_threads} instead."
        log.warning_rank(msg, comm=comm)
        threads = max_threads
    env.set_threads(threads)

    log.info_rank(f"Using {procs} processes with {threads} threads each.", comm=comm)
